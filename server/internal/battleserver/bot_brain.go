package battleserver

// The bot "mind": a small phase state machine (laning -> roam -> group/push) plus a
// per-tick decision pass. Every decision is expressed by calling the SAME ...Locked entry
// points a real client's packets drive (moveToLocked, startAttackLocked,
// startSkillOrderLocked, upgradeSkillLocked, buyItemLocked...), so a bot can never do
// anything a live player couldn't -- see bot.go's doc comment.

// botThinkInterval throttles full decision-making to a human-reaction-ish cadence. The
// shared 200ms world tick still drives movement arrival / swing timers / status expiry for
// a bot exactly as it does for anyone else; this only paces how often the bot RECONSIDERS
// what it's doing.
const botThinkInterval = 0.3

// botPhase is where a bot's overall game plan currently sits. Transitions are decided
// fresh every think tick (botUpdatePhaseLocked), not stored as a one-way ratchet, so a
// grouped team that loses its fight and scatters correctly falls back to laning instead of
// being stuck "regrouping" forever.
type botPhase int

const (
	botPhaseLane  botPhase = iota // solo/duo the assigned lane: farm, trade, hold ground
	botPhaseRoam                  // lane is stable/pushed out: look for a pickoff or gank
	botPhaseGroup                 // enough of the team is up and nearby: teamfight + push
)

// botBrain is one bot's decision state, held in huntInstance.bots keyed by objID.
type botBrain struct {
	c    *conn
	slot int

	// lane is this bot's default lane index into gamedata.DotaMap.Lanes, assigned once
	// at spawn (assignBotLane) so the team's 5 bots spread roughly 2/1/2 instead of
	// piling onto one lane -- see the constant's doc comment.
	lane int

	phase       botPhase
	nextThinkAt float64

	// engageTarget is the enemy hero this bot has committed to fighting this
	// engagement (0 = none). Kept sticky for a few think-ticks (see botCombatTickLocked)
	// so it doesn't flicker between two equally-close targets every 0.3s.
	engageTarget int32

	// retreating latches once a bot decides to disengage on low HP, and clears once it's
	// safe again -- see botShouldRetreatLocked. Latched (not re-evaluated from scratch
	// every tick) so a bot doesn't oscillate exactly at the HP threshold.
	retreating bool
}

// newBotBrain builds a fresh bot mind and assigns its default lane from its ordinal
// position on its own side (0-based: the 1st, 2nd, 3rd... bot/player to take that side).
func newBotBrain(c *conn, slot, sideOrdinal int) *botBrain {
	return &botBrain{c: c, slot: slot, lane: assignBotLane(sideOrdinal), phase: botPhaseLane}
}

// botLanePattern spreads 5 laners 2/1/2 across the three lanes (index 1 = the map's
// centre lane per gamedata.DotaMap's North/Centre/South doc comment) -- a common, simple
// Dota-style split: two flanks carry two heroes each, the middle carries one.
var botLanePattern = []int{0, 2, 1, 0, 2}

func assignBotLane(sideOrdinal int) int {
	return botLanePattern[sideOrdinal%len(botLanePattern)]
}

// botTickLocked is the per-bot entry point, called once per world tick (200ms) from
// runInstanceTicker for every member with a bot brain. Caller holds inst.mu.
func (s *Server) botTickLocked(b *botBrain, now float64) {
	c := b.c
	hs := c.huntState
	if hs == nil || hs.closed || c.inst == nil || c.inst.dota == nil {
		return
	}
	if hs.deadUntil > 0 {
		b.retreating = false
		b.engageTarget = 0
		return // nothing to decide until respawned; the world tick handles the timer itself
	}
	// Stunned: whatever was already in flight (a committed swing/cast) resolves on its
	// own timers exactly like a real player's would; there is nothing NEW to order.
	if hs.st.stunned(now) {
		return
	}
	// Always-on housekeeping, off the full think cadence: a bot spends a banked skill
	// point or an affordable item the instant one is available rather than sitting on
	// it for up to botThinkInterval.
	s.botSpendSkillPointLocked(b)
	s.botBuyItemsLocked(b, now)

	if now < b.nextThinkAt {
		return
	}
	b.nextThinkAt = now + botThinkInterval

	s.botUpdatePhaseLocked(b, now)
	if s.botCombatTickLocked(b, now) {
		return // an enemy hero is being fought/chased/kited -- that IS this tick's order
	}
	switch b.phase {
	case botPhaseGroup:
		s.botGroupTickLocked(b, now)
	case botPhaseRoam:
		s.botRoamTickLocked(b, now)
	default:
		s.botLaneTickLocked(b, now)
	}
}

// botMatchTime returns elapsed match time in seconds.
func botMatchTime(c *conn, now float64) float64 {
	if c.inst == nil || c.inst.dota == nil {
		return 0
	}
	return now - c.inst.dota.startedAt
}

// botLaneEarlyGame is how long a bot stays committed to pure laning (farm/trade, no
// roaming or grouping) before it starts considering the wider map -- long enough to hit a
// few levels and an early item, short enough that a 15-20 minute «Штурм» match still sees
// real teamfights.
const botLaneEarlyGame = 150.0 // 2.5 minutes

// botGroupUpRadius is how close living teammates must be to each other to count as
// "grouped" for the botPhaseGroup transition.
const botGroupUpRadius = 25.0

// botUpdatePhaseLocked re-derives this tick's phase from current world state -- not a
// one-way ratchet, so a team that groups, fights and loses correctly scatters back to
// laning instead of being stuck "regrouping" against a stronger enemy forever.
func (s *Server) botUpdatePhaseLocked(b *botBrain, now float64) {
	c := b.c
	if botMatchTime(c, now) < botLaneEarlyGame {
		b.phase = botPhaseLane
		return
	}
	cx, cy := c.posAtLocked(float32(now))
	grouped := 1 // counts self
	for _, mem := range c.inst.members {
		if mem == c || mem.huntState == nil || mem.huntState.deadUntil > 0 || mem.playerTeam() != c.playerTeam() {
			continue
		}
		mx, my := mem.posAtLocked(float32(now))
		if dist2(cx, cy, mx, my) <= float32(botGroupUpRadius*botGroupUpRadius) {
			grouped++
		}
	}
	switch {
	case grouped >= 3:
		b.phase = botPhaseGroup
	case grouped >= 2:
		// A pair is worth roaming together for a pickoff, not a full teamfight commit.
		b.phase = botPhaseRoam
	default:
		b.phase = botPhaseLane
	}
}

// botLivingAllies returns every living teammate (bot or real) of c, self excluded.
func botLivingAllies(c *conn) []*conn {
	if c.inst == nil {
		return nil
	}
	var out []*conn
	for _, mem := range c.inst.members {
		if mem == c || mem.huntState == nil || mem.huntState.deadUntil > 0 {
			continue
		}
		if mem.playerTeam() == c.playerTeam() {
			out = append(out, mem)
		}
	}
	return out
}

// botLivingEnemyHeroes returns every living, visible enemy hero.
func botLivingEnemyHeroes(c *conn, now float64) []*conn {
	if c.inst == nil {
		return nil
	}
	var out []*conn
	for _, mem := range c.inst.members {
		if mem == c || mem.huntState == nil {
			continue
		}
		if mem.playerTeam() == c.playerTeam() {
			continue
		}
		if mem.huntState.deadUntil == 0 && now >= mem.huntState.invisibleUntil {
			out = append(out, mem)
		}
	}
	return out
}

// botHPFrac/botManaFrac are the bot's own current resource fractions, used throughout the
// decision tree (retreat thresholds, mana-gated ability use).
func botHPFrac(hs *huntState, now float64) float64 {
	if max := hs.maxHPLocked(now); max > 0 {
		return hs.hp / max
	}
	return 1
}

func botManaFrac(hs *huntState, now float64) float64 {
	if max := hs.maxManaLocked(now); max > 0 {
		return hs.mana / max
	}
	return 1
}

// botRetreatHPFrac/botSafeHPFrac hysteresis: a bot starts retreating at the low threshold
// and only re-engages once healed back past the higher one, so it doesn't dart back into
// a fight the instant its HP ticks fractionally above the retreat line.
const (
	botRetreatHPFrac = 0.30
	botSafeHPFrac    = 0.55
)

// botShouldRetreatLocked updates and returns b.retreating.
func (s *Server) botShouldRetreatLocked(b *botBrain, now float64) bool {
	hs := b.c.huntState
	frac := botHPFrac(hs, now)
	switch {
	case b.retreating && frac >= botSafeHPFrac:
		b.retreating = false
	case !b.retreating && frac <= botRetreatHPFrac:
		b.retreating = true
	}
	return b.retreating
}

// botHomeLocked is the bot's own base spawn -- the retreat destination.
func botHomeLocked(c *conn) (float32, float32) {
	side := sideForTeam(c.playerTeam())
	hx, hy := c.inst.dota.sideSpawn(side)
	return float32(hx), float32(hy)
}

// botMoveTowardLocked issues a move order toward (tx,ty), skipping a redundant re-issue
// when the destination hasn't materially changed since the last order (moveToLocked
// restarts the arrival timer/leg walk every call, which is fine once per think tick but
// wasteful and jittery to repeat at an unchanged goal).
func (s *Server) botMoveTowardLocked(b *botBrain, tx, ty float32, now float64) {
	c := b.c
	cx, cy := c.posAtLocked(float32(now))
	if dist2(cx, cy, tx, ty) <= 2*2 {
		return // already there
	}
	c.moveToLocked(s, tx, ty)
}
