package battleserver

import (
	"math"
	"sort"
)

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

	// lane is this bot's default lane index into gamedata.DotaMap.Lanes. It starts from
	// assignBotLane and may be rebalanced when a real teammate never leaves base, so the
	// active roster keeps the intended 2/1/2 spread instead of reserving an empty slot.
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

	// hpHistory is a short ring of (t, hpFrac) samples taken every world tick (200ms) by
	// botRecordHPLocked -- see botCheckHPCrashLocked's doc comment for why this needs
	// finer granularity than the 0.3s think cadence.
	hpHistory [hpHistoryLen]hpSample
	hpHistIdx int
}

// hpSample is one botBrain.hpHistory entry.
type hpSample struct {
	t    float64
	frac float64
}

// hpHistoryLen: ~1.2s of samples at the 200ms world tick, comfortably spans one full
// botThinkInterval (0.3s) window so a burst that unfolds entirely inside a single think
// interval still leaves a trail botCheckHPCrashLocked can see.
const hpHistoryLen = 6

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

const (
	botHumanLaneGrace      = 25.0
	botLaneRebalanceEvery  = 5.0
	botHumanLeftBaseRadius = 10.0
)

// botRebalanceLanesLocked reapplies the team's 2/1/2 pattern after the opening grace
// period, but only reserves pattern slots for real players who have actually left base.
// Once a player has left, they stay active for lane accounting even when they later return
// to heal; this prevents assignments oscillating on every fountain visit.
func (s *Server) botRebalanceLanesLocked(inst *huntInstance, now float64) {
	d := inst.dota
	if d == nil || now < d.nextLaneRebalanceAt {
		return
	}
	d.nextLaneRebalanceAt = now + botLaneRebalanceEvery
	if d.laneActiveHumans == nil {
		d.laneActiveHumans = map[int32]bool{}
	}

	for _, mem := range inst.members {
		if mem.huntState == nil {
			continue
		}
		if _, isBot := inst.bots[mem.objID]; isBot {
			continue
		}
		mx, my := mem.posAtLocked(float32(now))
		hx, hy := botHomeLocked(mem)
		if dist2(mx, my, hx, hy) > botHumanLeftBaseRadius*botHumanLeftBaseRadius {
			d.laneActiveHumans[mem.objID] = true
		}
	}

	for _, team := range []int32{dotaTeamHuman, dotaTeamElf} {
		activeHumans := 0
		for _, mem := range inst.members {
			if mem.huntState == nil || mem.playerTeam() != team {
				continue
			}
			if _, isBot := inst.bots[mem.objID]; isBot {
				continue
			}
			if now-d.startedAt < botHumanLaneGrace || d.laneActiveHumans[mem.objID] {
				activeHumans++
			}
		}

		var bots []*botBrain
		for _, brain := range inst.bots {
			if brain.c != nil && brain.c.playerTeam() == team {
				bots = append(bots, brain)
			}
		}
		sort.Slice(bots, func(i, j int) bool { return bots[i].slot < bots[j].slot })
		for i, brain := range bots {
			brain.lane = assignBotLane(activeHumans + i)
		}
	}
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
	s.botRecordHPLocked(b, now)
	s.botRebalanceLanesLocked(c.inst, now)

	// A burst of damage (a tower+wave combo, or a hero gank) can blow straight through
	// botRetreatHPFrac's flat threshold faster than the next botThinkInterval
	// reassessment would ever run -- measured live: a bot took 90%+ of its max HP in
	// under 1.2s and never once evaluated retreat until it was already dead. This check
	// runs every world tick specifically so the reaction isn't bottlenecked on think
	// cadence. It only starts the recovery state; once latched, the regular think path
	// remains reachable so heals can fire and safe-HP hysteresis can clear it again.
	if s.botCheckHPCrashLocked(b, now) {
		if s.botConsiderHealLocked(b, now) {
			return
		}
		hx, hy := s.botRetreatPointLocked(b, now)
		s.botMoveTowardLocked(b, hx, hy, now)
		return
	}

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

// botRecordHPLocked appends this tick's HP fraction to the bot's short rolling history.
// Called every world tick (200ms) from botTickLocked, NOT gated behind botThinkInterval,
// so botCheckHPCrashLocked has a trail to work from even when an entire burst unfolds
// inside one 0.3s think window.
func (s *Server) botRecordHPLocked(b *botBrain, now float64) {
	b.hpHistIdx = (b.hpHistIdx + 1) % hpHistoryLen
	b.hpHistory[b.hpHistIdx] = hpSample{t: now, frac: botHPFrac(b.c.huntState, now)}
}

// botHPCrashFrac/hpCrashWindow: losing at least this much of max HP within this many
// seconds counts as burst damage worth reacting to immediately, even above
// botRetreatHPFrac's flat threshold -- see botCheckHPCrashLocked.
const (
	botHPCrashFrac = 0.25
	hpCrashWindow  = 1.0
)

// botCheckHPCrashLocked reports whether HP has dropped by at least botHPCrashFrac within
// the last hpCrashWindow seconds: a tower-plus-wave combo or a hero gank that would
// otherwise blow straight through botRetreatHPFrac's flat 30% floor before the next
// botThinkInterval reassessment ever runs (measured live: two bots died with `retreating`
// never latched, or latched only a fraction of a second before death, because the whole
// burst landed inside one think interval). Latches b.retreating exactly like
// botShouldRetreatLocked's own hysteresis, so a crash-triggered retreat still only clears
// once HP recovers past botSafeHPFrac -- the two share one latch, not two competing ones.
func (s *Server) botCheckHPCrashLocked(b *botBrain, now float64) bool {
	if b.retreating {
		return false
	}
	cur := botHPFrac(b.c.huntState, now)
	for _, sample := range b.hpHistory {
		if sample.t == 0 || now-sample.t > hpCrashWindow {
			continue
		}
		if sample.frac-cur >= botHPCrashFrac {
			b.retreating = true
			return true
		}
	}
	return false
}

// botHomeLocked is the bot's own base spawn -- the fallback retreat destination.
func botHomeLocked(c *conn) (float32, float32) {
	side := sideForTeam(c.playerTeam())
	hx, hy := c.inst.dota.sideSpawn(side)
	return float32(hx), float32(hy)
}

// botRetreatPointLocked is the recovery destination. Low HP and burst damage send the bot
// to its fountain, where DOTA regen can lift it above botSafeHPFrac and end the retreat.
func (s *Server) botRetreatPointLocked(b *botBrain, now float64) (float32, float32) {
	return botHomeLocked(b.c)
}

// botDisengagePointLocked is a short tactical step-back after abandoning a chase. It
// stays behind the friendly wave instead of turning every bad engagement into a full
// fountain trip; recovery retreats above remain intentionally separate.
func (s *Server) botDisengagePointLocked(b *botBrain, now float64) (float32, float32) {
	c := b.c
	hx, hy := botHomeLocked(c)
	if b.lane < 0 || b.lane >= len(c.inst.dota.m.Lanes) {
		return hx, hy
	}
	fx, fy, ok := botLaneFrontLocked(c, c.inst.dota.m.Lanes[b.lane])
	if !ok || s.botEnemyStructureDangerLocked(c, fx, fy) {
		return hx, hy
	}
	dx, dy := hx-fx, hy-fy
	length := float32(math.Hypot(float64(dx), float64(dy)))
	if length == 0 {
		return hx, hy
	}
	const stepBack = float32(8)
	return fx + dx/length*stepBack, fy + dy/length*stepBack
}

// botMoveTowardLocked issues a move order toward (tx,ty), skipping a redundant re-issue
// when the destination hasn't materially changed since the last order (moveToLocked
// restarts the arrival timer/leg walk every call, which is fine once per think tick but
// wasteful and jittery to repeat at an unchanged goal).
//
// A bot's move decision goes straight to conn.moveToLocked, NOT the handleMove packet
// handler a real client's click runs through -- so handleMove's own attack-cancelling
// (1d11366: a real player's flee click was being overridden by their own still-armed
// auto-attack/PvP-attack chase re-issuing chaseMoveLocked on its own retry cadence) never
// applied to a bot at all. Reported live: a low-HP bot tried to retreat and its own
// auto-attack kept walking it back into the fight -- the exact same bug, just reached
// through the bot's retreat call instead of a real MOVE_PLAYER packet. This is the single
// choke point every bot movement decision goes through, so cancelling here covers all of
// them, mirroring handleMove's own cancellation.
func (s *Server) botMoveTowardLocked(b *botBrain, tx, ty float32, now float64) {
	c, hs := b.c, b.c.huntState
	// Cancel any live attack/order BEFORE the "already there" short-circuit below: a bot
	// that decides to disengage while standing right on top of its destination (e.g.
	// retreating to base while already near base) must still drop its own attack chase,
	// not just skip the no-op move.
	if hs.attackTarget != 0 {
		s.stopAttackLocked(c, false)
	}
	if hs.pvpTarget != 0 {
		s.stopPvpAttackLocked(c, false)
	}
	s.cancelOrderLocked(c)
	cx, cy := c.posAtLocked(float32(now))
	if dist2(cx, cy, tx, ty) <= 2*2 {
		return // already there
	}
	c.moveToLocked(s, tx, ty)
}
