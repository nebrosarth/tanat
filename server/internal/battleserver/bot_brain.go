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

// botRetreatMode refines the retreating safety gate without changing its zero-value
// behavior. A constructed brain that sets only retreating (as existing recovery tests
// do) therefore still takes the fountain-recovery path.
type botRetreatMode int

const (
	botRetreatModeRecovery botRetreatMode = iota
	botRetreatModeDisengage
)

// botBrain is one bot's decision state, held in huntInstance.bots keyed by objID.
type botBrain struct {
	c    *conn
	slot int

	// aiVersion is the cumulative decision profile selected for this bot's
	// team when the match was created. It is immutable for the whole match so
	// A/B comparisons cannot change mid-game.
	aiVersion    int
	aiVersionSet bool

	// macroAssignment is written by the side orchestrator for AI-10/AI-20. AI-0
	// deliberately leaves it empty and runs the legacy local phase brain; local
	// combat, retreat, recovery, and safety gates remain authoritative in all cases.
	macroAssignment botMacroAssignment

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

	// Attack watchdog state deliberately distinguishes an attack ORDER from a real
	// swing. huntState.attackTarget is the former and can remain non-zero while the
	// timer chain never reaches its in-range commit point. Without this state the lane
	// tick returns early forever and the bot looks AFK while a creep kills it.
	attackTargetSince        float64
	attackLastSwingAt        float64
	attackLastHitAt          float64
	attackPendingHitUntil    float64
	attackLastProgressAt     float64
	attackLastX, attackLastY float32

	// retreating latches once a bot decides to disengage on low HP, and clears once it's
	// safe again -- see botShouldRetreatLocked. Latched (not re-evaluated from scratch
	// every tick) so a bot doesn't oscillate exactly at the HP threshold.
	retreating bool
	// retreatMode is meaningful only while retreating is true. Zero intentionally means
	// recovery for compatibility with brains constructed by older tests/callers.
	retreatMode botRetreatMode
	// retreatHoldUntil bounds tactical disengage hysteresis independently of the recovery
	// HP hysteresis. Recovery never uses this hold.
	retreatHoldUntil float64

	// hpHistory is a short ring of (t, hpFrac) samples taken every world tick (200ms) by
	// botRecordHPLocked -- see botCheckHPCrashLocked's doc comment for why this needs
	// finer granularity than the 0.3s think cadence.
	hpHistory [hpHistoryLen]hpSample
	hpHistIdx int

	// pendingTeleport is the bot-only preparation channel. While non-nil the
	// bot brain must not issue movement, attacks, skills, or lane orders.
	pendingTeleport *botTeleportOrder

	// structureAvoidUntil is a short, independent safety hold after an enemy
	// structure commits a shot at this bot. It deliberately does not use
	// retreating: healthy bots must not become fountain-bound just because a
	// tower fired once.
	structureAvoidUntil  float64
	structureAvoidTarget int32

	// structureDetour keeps one chosen side and the remaining waypoints for a
	// low-HP route around a firing structure. Keeping this on the brain makes
	// repeated think ticks continue the same side instead of re-solving the
	// circle and alternating left/right.
	structureDetourThreat int32
	structureDetourSide   int
	structureDetourGoalX  float32
	structureDetourGoalY  float32
	structureDetour       []botStructureWaypoint

	// structureEscape telemetry is edge-triggered so a committed shot that is
	// observed on several world ticks produces one useful event, not a stream of
	// identical decisions.
	structureEscapeTelemetryThreat int32
	structureEscapeTelemetryReason string
	structureEscapeTelemetryX      float32
	structureEscapeTelemetryY      float32

	// structureAvoidDestination is the single movement order for the current
	// avoidance hold. It is replaced only when the threat changes, the point is
	// reached, or navigation/safety validation no longer accepts it.
	structureAvoidDestinationX     float32
	structureAvoidDestinationY     float32
	structureAvoidDestinationValid bool

	// laneRedeployPoint holds a bot at the exact safe point where a later
	// structure-fallback teleport landed until its own creep front catches up.
	// It only affects the idle lane destination; combat, healing, and retreat
	// decisions return before botLanePoint is consulted.
	laneRedeployPointX     float32
	laneRedeployPointY     float32
	laneRedeployUntil      float64
	laneRedeployPointValid bool

	// Farm director state is battle-local and observational. It records the live
	// lane/target choice so telemetry can explain XP/min outcomes without changing
	// authoritative XP or combat rules.
	farmTarget      int32
	farmTargetScore float64
	farmDecision    string
	farmLane        int
	farmCatchUp     bool
	// farmLaneArrivalPending is set by the orchestrator when a bot that has
	// already farmed is handed a different live lane. It lets the bot use the
	// same ETA-based structure redeploy as the opening staging path instead of
	// walking across the map while that lane's next wave is already dying.
	farmLaneArrivalPending bool
	farmLastHits           int
	farmWaveClears         int
	// farmXPEvents is the number of creep XP grants received by this bot in the
	// current battle. farmLastXPTAt lets the orchestrator prefer the bot whose
	// farm opportunity is oldest without using match-time phases.
	farmXPEvents  int
	farmLastXPTAt float64
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
	version := botAIVersionDefault
	if c != nil && c.inst != nil {
		version = botTeamAIVersionLocked(c.inst, c.playerTeam())
	}
	return &botBrain{c: c, slot: slot, aiVersion: version, aiVersionSet: true, lane: assignBotLane(sideOrdinal), phase: botPhaseLane}
}

// botAttackWatchdogLocked releases an attack order that has stopped making progress.
// Being out of range is not a stall: the normal attack timer is allowed to chase the
// target. The watchdog only acts once the target is in the avatar's effective reach and
// no real swing has been emitted for the expected first-swing window (or for several
// expected follow-up intervals).
func (s *Server) botAttackWatchdogLocked(b *botBrain, now float64) bool {
	if b == nil || b.c == nil || b.c.huntState == nil {
		return false
	}
	hs := b.c.huntState
	if hs.attackTarget == 0 {
		b.attackTargetSince = 0
		b.attackLastSwingAt = 0
		b.attackLastHitAt = 0
		b.attackPendingHitUntil = 0
		return false
	}
	target := hs.mobs[hs.attackTarget]
	if target == nil || target.dead {
		// The timer normally clears a dead target, but its chain can already have
		// been invalidated by a movement/order update. Do not leave the bot in the
		// lane tick's `attackTarget != 0` early return forever: that is the exact
		// AFK-under-creep-damage failure seen in live telemetry.
		reason := "target_missing"
		if target != nil {
			reason = "target_dead"
		}
		s.telemetryRecordBotAttackCancelLocked(b.c, hs.attackTarget, reason, now)
		s.stopAttackLocked(b.c, false)
		b.farmTarget = 0
		b.farmTargetScore = 0
		return true
	}
	if hs.attackActionActive {
		return false
	}
	// The visible swing may already be closed while a projectile is still in
	// flight. Cancelling here would produce exactly the observed pattern: the
	// bot starts an attack, the client shows it, but the creep never receives the
	// committed hit. The hit scheduler clears this window on arrival.
	if b.attackPendingHitUntil > now {
		return false
	}
	if b.attackTargetSince <= 0 {
		b.attackTargetSince = now
	}
	cx, cy := b.c.posAtLocked(float32(now))
	if b.attackLastProgressAt <= 0 {
		b.attackLastProgressAt = now
		b.attackLastX, b.attackLastY = cx, cy
	} else if dist2(cx, cy, b.attackLastX, b.attackLastY) > float32(0.25*0.25) {
		b.attackLastProgressAt = now
		b.attackLastX, b.attackLastY = cx, cy
	}
	interval := s.swingIntervalLocked(hs).Seconds()
	if interval <= 0 {
		interval = 1
	}
	reach := hs.effAttackRangeLocked(now) + hs.av.Radius() + target.mob.Radius()
	if math.Hypot(float64(target.x-cx), float64(target.y-cy)) > reach {
		// An active chase is allowed to remain out of range, but an order with no
		// movement progress is not. If its timer chain died, the lane tick used to
		// return here forever and the bot could be killed by the very creeps it was
		// supposed to answer. Release the stale order so the next decision pass can
		// reselect a live attacker or move to a safe farm point.
		lastProgress := b.attackLastProgressAt
		if b.attackLastSwingAt > lastProgress {
			lastProgress = b.attackLastSwingAt
		}
		chaseGrace := math.Max(interval*3, 2.0)
		if lastProgress > 0 && now-lastProgress >= chaseGrace {
			s.telemetryRecordBotAttackCancelLocked(b.c, target.id, "stalled_chase", now)
			s.stopAttackLocked(b.c, false)
			b.farmTarget = 0
			b.farmTargetScore = 0
			return true
		}
		return false
	}
	lastActivity := b.attackTargetSince
	grace := interval * 1.25
	if b.attackLastSwingAt > 0 {
		lastActivity = b.attackLastSwingAt
		// Once a swing has committed, do not wait for several more full
		// intervals if no hit arrived. Release the target and let the farm
		// director choose a fresh creep/route. A landed hit keeps the normal
		// cadence watchdog, while a missing hit gets a shorter, progress-based
		// recovery window.
		if b.attackLastHitAt < b.attackLastSwingAt {
			grace = math.Max(interval*2.0, 1.2)
		} else {
			grace = interval * 2.5
		}
	}
	if now-lastActivity < grace {
		return false
	}
	reason := "no_swing"
	if b.attackLastSwingAt > 0 && b.attackLastHitAt < b.attackLastSwingAt {
		reason = "no_hit_rearm"
	}
	s.telemetryRecordBotAttackCancelLocked(b.c, target.id, reason, now)
	s.stopAttackLocked(b.c, false)
	// Do not let botFarmTargetLocked immediately return the same stale creep
	// after a failed attack chain. It is still legal to select it again later,
	// but only after the surrounding wave has been reconsidered.
	b.farmTarget = 0
	b.farmTargetScore = 0
	return true
}

func (s *Server) botTrackAttackOrderLocked(c *conn, targetID int32, now float64) {
	if c == nil || c.inst == nil {
		return
	}
	if b := c.inst.bots[c.objID]; b != nil {
		b.attackTargetSince = now
		b.attackLastSwingAt = 0
		b.attackLastHitAt = 0
		b.attackPendingHitUntil = 0
		b.attackLastProgressAt = now
		b.attackLastX, b.attackLastY = c.posAtLocked(float32(now))
	}
}

func (s *Server) botTrackAttackSwingLocked(c *conn, targetID int32, now float64) {
	if c == nil || c.inst == nil {
		return
	}
	if b := c.inst.bots[c.objID]; b != nil {
		b.attackLastSwingAt = now
		b.attackLastHitAt = 0
		interval := s.swingIntervalLocked(c.huntState).Seconds()
		if interval <= 0 {
			interval = 1
		}
		pending := interval / 2
		if c.huntState.hasProjectile {
			cx, cy := c.posAtLocked(float32(now))
			if target := c.huntState.mobs[targetID]; target != nil {
				flight := math.Hypot(float64(target.x-cx), float64(target.y-cy))/24 + 0.1
				pending = c.huntState.av.AttackWindup*interval + flight
			}
		}
		b.attackPendingHitUntil = now + math.Max(pending+0.15, interval/2)
	}
	s.telemetryRecordBotAttackStartLocked(c, targetID, now)
}

func (s *Server) botTrackAttackHitLocked(c *conn, targetID int32, now float64) {
	if c == nil || c.inst == nil || c.huntState == nil || c.huntState.attackTarget != targetID {
		return
	}
	if b := c.inst.bots[c.objID]; b != nil {
		b.attackLastHitAt = now
		b.attackPendingHitUntil = 0
	}
}

func (s *Server) botClearAttackWatchdogLocked(c *conn) {
	if c == nil || c.inst == nil {
		return
	}
	if b := c.inst.bots[c.objID]; b != nil {
		b.attackTargetSince = 0
		b.attackLastSwingAt = 0
		b.attackLastHitAt = 0
		b.attackPendingHitUntil = 0
		b.attackLastProgressAt = 0
		b.attackLastX, b.attackLastY = 0, 0
	}
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
	if b.pendingTeleport != nil {
		if botAnyMobilizationReason(b.macroAssignment.Reason) && b.pendingTeleport.targetKind != "recovery_structure" {
			// A pending ordinary lane teleport was selected before the team entered
			// mobilization. It would land outside the rendezvous and split the group;
			// cancel it before it can complete. The scroll remains consumed by the
			// normal cancellation rule.
			s.cancelBotTeleportLocked(b, "mobilization_rally")
			return
		}
		if reason := s.botUrgentBaseDefenseTeleportReasonLocked(b, b.pendingTeleport, now); reason != "" {
			s.cancelBotTeleportAtLocked(b, reason, now)
			return
		}
	}
	if hs.deadUntil == 0 && !hs.st.stunned(now) {
		if threat := s.botCommittedStructureFocusLocked(c, now); threat != nil &&
			!s.botCanCommitStructureFocusLocked(b, threat, now) {
			if b.pendingTeleport != nil {
				reason := "structure_focus"
				if b.pendingTeleport.targetKind == "recovery_structure" {
					reason = "recovery_origin_pressure"
				}
				s.cancelBotTeleportLocked(b, reason)
			}
			s.botEscapeStructureFocusLocked(b, threat, now)
			return
		}
	}
	if b.pendingTeleport != nil {
		s.botTickTeleportLocked(b, now)
		return
	}
	if hs.deadUntil > 0 {
		botClearRetreatLocked(b)
		b.engageTarget = 0
		return // nothing to decide until respawned; the world tick handles the timer itself
	}
	// Stunned: whatever was already in flight (a committed swing/cast) resolves on its
	// own timers exactly like a real player's would; there is nothing NEW to order.
	if hs.st.stunned(now) {
		return
	}
	// A targeted channel is committed as soon as its cast is accepted, while its
	// channelState is only created by the delayed payload. Treat that pending impact
	// as blocking now, not only after the first pulse, so no bot route can interrupt it.
	if botHasPendingBlockingChannelLocked(hs, now) {
		return
	}
	if now < b.structureAvoidUntil {
		s.botHoldStructureAvoidanceLocked(b, now)
		return
	}
	b.structureAvoidTarget = 0
	b.structureAvoidDestinationValid = false
	// Always-on housekeeping, off the full think cadence: a bot spends a banked skill
	// point or an affordable item the instant one is available rather than sitting on
	// it for up to botThinkInterval.
	s.botSpendSkillPointLocked(b)
	s.botBuyItemsLocked(b, now)
	s.botRecordHPLocked(b, now)
	s.botShouldRetreatLocked(b, now)
	// A farm attack is wave maintenance, not a last-hit chase. Keep it only while
	// the creep remains inside both the real attack reach and the XP envelope;
	// once it moves out, release the order so the safe farm anchor takes over.
	if hs.attackTarget != 0 {
		if target := hs.mobs[hs.attackTarget]; target != nil && !target.structure &&
			target.enemyOf(c.playerTeam()) {
			cx, cy := c.posAtLocked(float32(now))
			distance := math.Hypot(float64(target.x-cx), float64(target.y-cy))
			attackReach := hs.effAttackRangeLocked(now) + hs.av.Radius() + target.mob.Radius()
			if distance > dotaXPShareRadius || distance > attackReach {
				s.stopAttackLocked(c, false)
			}
		}
	}
	if b.retreating && s.botFarmXPShadowTickLocked(b, now) {
		return
	}
	if s.botAttackWatchdogLocked(b, now) {
		// A stale order was released. Re-open the decision pass immediately so the bot
		// can select another live target instead of waiting for the next think slot.
		b.nextThinkAt = 0
	}
	// Do not start a teleport or another order while a targeted/self channel is
	// sustaining. Retreat logic above still has priority and may deliberately break
	// the channel when survival requires it.
	if botHasBlockingChannelLocked(hs, now) || botHasPendingBlockingChannelLocked(hs, now) {
		return
	}
	if s.botMaybeStartTeleportLocked(b, now) {
		return
	}

	// A burst of damage (a tower+wave combo, or a hero gank) can blow straight through
	// botRetreatHPFrac's flat threshold faster than the next botThinkInterval
	// reassessment would ever run -- measured live: a bot took 90%+ of its max HP in
	// under 1.2s and never once evaluated retreat until it was already dead. This check
	// runs every world tick specifically so the reaction isn't bottlenecked on think
	// cadence. It only starts the recovery state; once latched, the regular think path
	// remains reachable so heals can fire and safe-HP hysteresis can clear it again.
	if s.botCheckHPCrashLocked(b, now) {
		if s.botFarmXPShadowTickLocked(b, now) {
			return
		}
		if s.botConsiderRetreatUtilityLocked(b, now) {
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

	if b.retreating {
		// Recovery is a latched lifecycle state. Do not let the combat pass
		// re-arm an old creep/hero chase before the retreat order is issued: that
		// made a low-HP bot keep swinging in place while creeps and a pursuer
		// finished it. A safe self-defense utility may fire first, but movement
		// toward the recovery destination always wins over optional combat.
		if s.botFarmXPShadowTickLocked(b, now) {
			return
		}
		if s.botConsiderRetreatUtilityLocked(b, now) {
			return
		}
		hx, hy := s.botRetreatPointLocked(b, now)
		if s.botRecoveryFarmTickLocked(b, now) {
			return
		}
		s.botMoveTowardLocked(b, hx, hy, now)
		return
	}
	// Farm coverage is a team-level obligation. A locally attractive hero poke
	// must not pull the assigned lane owner out of XP range while a visible
	// creep wave is still uncovered; that was the route that turned a healthy
	// farmer into a retreating body and caused the next creep to die without XP.
	// The orchestrator still owns the lane assignment, so this only suppresses
	// optional combat for a bot that actually carries a farm lane.
	farmCoverageRequired := botAIProfileForBrain(b).UsesTeamOrchestrator() &&
		s.botTeamFarmCoverageRequiredLocked(c.inst, c.playerTeam(), now)
	if (!farmCoverageRequired || !b.macroAssignment.FarmLaneSet) && s.botCombatTickLocked(b, now) {
		return // an enemy hero is being fought/chased/kited -- that IS this tick's order
	}
	if b.macroAssignment.Mode == botMacroBase {
		// A base defender that also carries a farm lane must not abandon the
		// XP envelope merely because a creep wave is touching the structure.
		// The wave is handled from the same safe rear anchor; only a visible
		// enemy avatar near the objective justifies the dedicated defense tick.
		if b.macroAssignment.FarmLaneSet && farmCoverageRequired {
			objective := c.inst.mobs[b.macroAssignment.ObjectiveID]
			if objective == nil || !s.botVisibleEnemyHeroNearObjectiveLocked(c.inst, c.playerTeam(), objective, now) {
				s.botCoverageTickLocked(b, now)
				return
			}
		}
		// The macro keeps a cover responder in the base plan so it can be
		// promoted to defense without rebuilding the whole team assignment. It
		// is nevertheless still the farm owner of FarmLane; routing it through
		// the defender tick sends it to the objective lane and leaves its line
		// uncovered while the promotion is only precautionary.
		if b.macroAssignment.Role == "cover" && b.macroAssignment.FarmLaneSet {
			s.botCoverageTickLocked(b, now)
			return
		}
		s.botBaseDefenseTickLocked(b, now)
		return
	}
	if b.macroAssignment.Mode == botMacroRecover {
		return
	}
	if b.macroAssignment.Mode == botMacroCover {
		s.botCoverageTickLocked(b, now)
		return
	}
	if b.macroAssignment.Mode == botMacroAltar || b.macroAssignment.Mode == botMacroPush {
		b.phase = botPhaseGroup
		s.botGroupTickLocked(b, now)
		return
	}
	if b.macroAssignment.Mode == botMacroRally {
		b.phase = botPhaseRoam
		s.botRoamTickLocked(b, now)
		return
	}
	if b.macroAssignment.Mode == botMacroLane {
		b.phase = botPhaseLane
		s.botLaneTickLocked(b, now)
		return
	}
	// Compatibility fallback for directly-constructed brains in older callers/tests.
	s.botUpdatePhaseLocked(b, now)
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

// Strategic phase selection is state-driven; no elapsed-time gate is used.
/* botLaneEarlyGame is how long a bot stays committed to pure laning (farm/trade, no
// roaming or grouping) before it starts considering the wider map -- long enough to hit a
// few levels and an early item, short enough that a 15-20 minute «Штурм» match still sees
// real teamfights. */
// botGroupUpRadius is how close living teammates must be to each other to count as
// "grouped" for the botPhaseGroup transition.
const botGroupUpRadius = 25.0

// botUpdatePhaseLocked re-derives this tick's phase from current world state -- not a
// one-way ratchet, so a team that groups, fights and loses correctly scatters back to
// laning instead of being stuck "regrouping" against a stronger enemy forever.
func (s *Server) botUpdatePhaseLocked(b *botBrain, now float64) {
	c := b.c
	cx, cy := c.posAtLocked(float32(now))
	grouped := 1 // counts self
	for _, mem := range c.inst.members {
		if mem == c || mem.huntState == nil || mem.huntState.deadUntil > 0 || mem.playerTeam() != c.playerTeam() {
			continue
		}
		if brain, isBot := c.inst.bots[mem.objID]; isBot && brain != nil && brain.retreating {
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
	sort.Slice(out, func(i, j int) bool { return out[i].objID < out[j].objID })
	return out
}

// botLivingEnemyHeroes returns every living, visible enemy hero.
func botLivingEnemyHeroes(c *conn, now float64) []*conn {
	if c.inst == nil {
		return nil
	}
	visionSources := dotaTeamVisionSourcesLocked(c.inst, c.playerTeam(), now)
	var out []*conn
	for _, mem := range c.inst.members {
		if mem == c || mem.huntState == nil {
			continue
		}
		if mem.playerTeam() == c.playerTeam() {
			continue
		}
		if botVisibleEnemyMemberLocked(c.inst, c.playerTeam(), mem, now, visionSources) {
			out = append(out, mem)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].objID < out[j].objID })
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
	// A recovery latch normally clears at botSafeHPFrac. During a live objective
	// conversion, two bots that have already regrouped can rejoin earlier: waiting
	// for fountain-level safety leaves a winning group without its frontline for
	// most of the conversion window. This is still above the hard retreat floor
	// and is allowed only by botCanRejoinObjectiveLocked's local safety checks.
	botObjectiveRejoinHPFrac = 0.42
	// An exposed altar is the terminal objective: once every gun is gone, a
	// grouped bot that is safely above the hard retreat floor may rejoin the
	// finish before reaching the ordinary 42% objective-rejoin band. This is
	// still gated by live pressure and nearby allies, so a depleted assault
	// does not blindly ignore a real fight.
	botAltarAssaultRejoinHPFrac = 0.36
	// Tactical disengage keeps a bot behind its wave for a short, deterministic minimum
	// hold after the predictive premise ends. It is intentionally much shorter than a
	// fountain trip, but long enough to prevent one-tick re-engagement oscillation.
	botDisengageMinHold = 2.0

	// Predictive retreat only applies below this still-recoverable HP band. A bot at
	// full/high HP therefore does not flee merely because an enemy is nearby.
	botPredictiveRetreatHPFrac = 0.70
	// Require both a meaningful recent loss and a sustained loss rate. This filters
	// harmless regeneration noise/chip while catching a close pursuit before the next
	// burst crosses the 30% baseline.
	botPredictiveRetreatLossFrac = 0.05
	botPredictiveRetreatLossRate = 0.08
	botRetreatPressureRadius     = 24.0
	// Under active incoming damage a bot should create distance before the hard
	// recovery floor. This is deliberately only a tactical disengage; the 30%
	// threshold still escalates to a fountain recovery.
	botPressureRetreatHPFrac = 0.45
	// A targeted spell can deal damage without arming pvpTarget/attackTarget.
	// Keep that attacker as an immediate threat for a short combat-local window
	// so the bot can disengage before the next pulse instead of waiting for the
	// generic HP trend to catch up.
	botRecentHeroDamageThreatWindow = 2.5
)

func botClearRetreatLocked(b *botBrain) {
	b.retreating = false
	b.retreatMode = botRetreatModeRecovery
	b.retreatHoldUntil = 0
}

func botSetRetreatModeLocked(b *botBrain, mode botRetreatMode, now float64) {
	b.retreating = true
	b.retreatMode = mode
	if mode == botRetreatModeDisengage {
		b.retreatHoldUntil = now + botDisengageMinHold
	} else {
		b.retreatHoldUntil = 0
	}
}

// botShouldRetreatLocked updates and returns b.retreating.
func (s *Server) botShouldRetreatLocked(b *botBrain, now float64) bool {
	hs := b.c.huntState
	if hs.deadUntil > 0 {
		botClearRetreatLocked(b)
		return false
	}
	// Full mobilization has a narrow last-stand exception: if the named front
	// structure is already almost dead, a still-viable responder should finish
	// the objective instead of walking away and leaving a one-hit building alive.
	// This never overrides a critically low HP state and does not apply to an
	// ordinary lane/push assignment.
	// A finish-window objective may justify holding a recoverable attacker, but
	// it must never override the hard survival floor. The previous ordering let a
	// 27%-HP bot keep walking through a structure fight and die, which then left
	// its authored lane without an XP body for the next wave.
	if botHPFrac(hs, now) > botRetreatHPFrac && botLastStandObjectiveLocked(b, now) {
		botClearRetreatLocked(b)
		return false
	}
	// A landed hero hit is an authoritative contact signal even when fog or the
	// attack-order lifecycle has not yet exposed the attacker to this bot's
	// ordinary combat query. React on the first meaningful hit, before a second
	// burst can remove the only XP owner from the lane.
	if hs.lastHeroDamager != 0 && hs.lastHeroDamageAt > 0 && now-hs.lastHeroDamageAt <= botRecentHeroDamageThreatWindow &&
		botHPFrac(hs, now) <= botPredictiveRetreatHPFrac {
		botSetRetreatModeLocked(b, botRetreatModeDisengage, now)
		return true
	}
	frac := botHPFrac(hs, now)
	if b.retreating {
		if b.retreatMode == botRetreatModeRecovery {
			if frac >= botSafeHPFrac || s.botCanRejoinObjectiveLocked(b, now, frac) ||
				s.botCanRejoinBaseDefenseLocked(b, now, frac) {
				botClearRetreatLocked(b)
			}
			return b.retreating
		}
		if frac <= botRetreatHPFrac {
			botSetRetreatModeLocked(b, botRetreatModeRecovery, now)
			return true
		}
		loss, rate := botRecentHPLossLocked(b, now)
		baseRejoin := s.botCanRejoinBaseDefenseLocked(b, now, frac)
		if now >= b.retreatHoldUntil && frac <= botPressureRetreatHPFrac &&
			s.botIncomingPressureLocked(b, now) {
			// A short disengage is only useful if it creates safety. If pressure
			// is still landing after the hold, switch to a fountain recovery before
			// the bot drifts all the way through the hard 30% floor.
			botSetRetreatModeLocked(b, botRetreatModeRecovery, now)
			return true
		}
		if baseRejoin || (now >= b.retreatHoldUntil && frac >= botSafeHPFrac &&
			(loss < botPredictiveRetreatLossFrac || rate < botPredictiveRetreatLossRate)) {
			botClearRetreatLocked(b)
		}
		return b.retreating
	}
	if frac <= botRetreatHPFrac {
		botSetRetreatModeLocked(b, botRetreatModeRecovery, now)
	} else if s.botFarmGuardianHoldLocked(b, now) {
		// Preserve a healthy proximity-XP body only while it is not taking active
		// damage. botFarmGuardianHoldLocked deliberately yields to creep pressure.
		return false
	} else if frac <= botPredictiveRetreatHPFrac && s.botFarmCoveragePressureLocked(b, now) {
		// A farm owner should leave a pressured creep pack before its recent
		// damage trend reaches the hard recovery floor, while remaining close
		// enough for the XP shadow hand-off. This is live-state pressure, not an
		// opening-time rule.
		botSetRetreatModeLocked(b, botRetreatModeDisengage, now)
	} else if frac <= botPressureRetreatHPFrac && s.botIncomingPressureLocked(b, now) {
		botSetRetreatModeLocked(b, botRetreatModeDisengage, now)
	} else if s.botPredictiveRetreatLocked(b, now, frac) {
		botSetRetreatModeLocked(b, botRetreatModeDisengage, now)
	}
	return b.retreating
}

// botCanRejoinObjectiveLocked releases a recovery latch only when the bot has
// a concrete, locally safe reason to return. It is deliberately narrower than
// the normal safe-HP hysteresis: a lone injured laner still heals at base, while
// a pair that has regrouped for a ready conversion can restore the team's
// execution power without waiting for full fountain recovery.
func (s *Server) botCanRejoinObjectiveLocked(b *botBrain, now, frac float64) bool {
	return s.botCanRejoinObjectivePlanLocked(b, b.macroAssignment, now, frac, true)
}

// botCanRejoinBaseDefenseLocked releases a recovery latch for a bot that has
// already reached its safe base area while its own structure is under live
// attack. Base defense is a protected regroup point: returning at the normal
// objective-rejoin band is useful, but an avatar still taking incoming damage
// must remain in recovery until the ordinary safe threshold is reached.
func (s *Server) botCanRejoinBaseDefenseLocked(b *botBrain, now, frac float64) bool {
	if b == nil || b.c == nil || b.c.inst == nil || b.c.inst.dota == nil || b.c.huntState == nil ||
		frac < botObjectiveRejoinHPFrac {
		return false
	}
	plan, ok := b.c.inst.dota.teamPlans[b.c.playerTeam()]
	if !ok || plan.Mode != botMacroBase || plan.ObjectiveID == 0 {
		return false
	}
	objective := b.c.inst.mobs[plan.ObjectiveID]
	if objective == nil || objective.dead || !objective.structure || objective.team != b.c.playerTeam() {
		return false
	}
	hx, hy := botHomeLocked(b.c)
	cx, cy := b.c.posAtLocked(float32(now))
	if math.Hypot(float64(cx-hx), float64(cy-hy)) > botRetreatPressureRadius {
		return false
	}
	return !s.botIncomingPressureLocked(b, now)
}

// botCanRejoinObjectivePlanLocked is the plan-aware form used by the
// orchestrator before the new assignment has been copied onto brain. It keeps
// staging releases based on the desired team plan rather than on yesterday's
// lane/recovery assignment.
func (s *Server) botCanRejoinObjectivePlanLocked(b *botBrain, assignment botMacroAssignment, now, frac float64, requireConversion bool) bool {
	if b == nil || b.c == nil || b.c.inst == nil || b.c.huntState == nil ||
		b.retreatMode != botRetreatModeRecovery {
		return false
	}
	altarAssault := assignment.Mode == botMacroAltar && assignment.Reason == "enemy_altar_open"
	rejoinHP := botObjectiveRejoinHPFrac
	if altarAssault {
		rejoinHP = botAltarAssaultRejoinHPFrac
	}
	if frac < rejoinHP {
		return false
	}
	if assignment.Mode != botMacroPush && assignment.Mode != botMacroAltar && assignment.Mode != botMacroCover {
		return false
	}
	if assignment.ObjectiveID == 0 || (!altarAssault && requireConversion && assignment.Reason != "objective_conversion_ready" &&
		assignment.Reason != botMacroReasonPartialMobilization && assignment.Reason != botMacroReasonFullMobilization) {
		return false
	}
	objective := b.c.inst.mobs[assignment.ObjectiveID]
	if objective == nil || objective.dead || !objective.structure ||
		!objective.enemyOf(b.c.playerTeam()) || b.c.altarShieldedLocked(objective) {
		return false
	}
	// Do not clear a recovery retreat while a visible hero or creep is still
	// actively damaging this bot. The local pressure check intentionally keeps
	// the fog-of-war boundary used by the rest of the combat brain.
	if s.botIncomingPressureLocked(b, now) {
		return false
	}
	cx, cy := b.c.posAtLocked(float32(now))
	nearbyAllies := 0
	for _, ally := range b.c.inst.members {
		if ally == nil || ally == b.c || ally.huntState == nil || ally.playerTeam() != b.c.playerTeam() ||
			ally.huntState.deadUntil > 0 {
			continue
		}
		ax, ay := ally.posAtLocked(float32(now))
		allyRadius := botFightRadius
		if altarAssault {
			// The finish group may be reforming along the approach corridor. A
			// fight-radius-only test deadlocks when every member just peeled away
			// from the exposed altar to recover.
			allyRadius = botObjectiveApproachRadius
		}
		if math.Hypot(float64(ax-cx), float64(ay-cy)) > allyRadius {
			continue
		}
		if botHPFrac(ally.huntState, now) < botRetreatHPFrac {
			continue
		}
		nearbyAllies++
	}
	if nearbyAllies == 0 {
		return false
	}
	if altarAssault {
		// All guns are already down, so the terminal altar assault does not wait
		// for another creep-wave conversion predicate. Live pressure and the
		// approach-corridor ally gate above remain authoritative.
		return true
	}
	return !requireConversion || s.botObjectiveConversionReadyLocked(b.c.inst, b.c.playerTeam(), objective, now)
}

func botLastStandObjectiveLocked(b *botBrain, now float64) bool {
	if b == nil || b.c == nil || b.c.huntState == nil || b.c.inst == nil {
		return false
	}
	assignment := b.macroAssignment
	altarAssault := assignment.Reason == "enemy_altar_open" && assignment.Mode == botMacroAltar
	ordinaryFinish := (assignment.Reason == "objective_conversion_ready" || assignment.Reason == botMacroReasonPartialMobilization) && assignment.Mode == botMacroPush
	counterPushFinish := assignment.Role == botMacroCounterPushRole && assignment.Mode == botMacroPush
	// A counter-pusher exists specifically to trade a threatened checkpoint for
	// an already-damaged enemy front objective. Once that objective enters the
	// finish window, preserve the trade until the normal hard retreat floor;
	// otherwise the base-defense overlay can peel the last attacker immediately
	// before the kill. This is a role/objective state, not a roster-size rule.
	if !altarAssault && !ordinaryFinish && !counterPushFinish && (assignment.Reason != botMacroReasonFullMobilization ||
		(assignment.Mode != botMacroPush && assignment.Mode != botMacroCover)) {
		return false
	}
	objective := b.c.inst.mobs[assignment.ObjectiveID]
	if ordinaryFinish {
		// An ordinary conversion is already carrying execution debt in the
		// broader commit band. Keep a recoverable bot on the objective there;
		// reserving the narrower finish window for this check caused a 71%-HP
		// gun to survive while part of the committed group was still retreating.
		if !botMacroObjectiveCommitWindowLocked(objective) {
			return false
		}
	} else if !botMacroObjectiveFinishWindowLocked(objective) {
		return false
	}
	if altarAssault {
		// The assault may trade through the final altar phase, but a critically
		// low bot still has to recover. This keeps the finish exception from
		// converting a 25%-HP avatar into a free death.
		return botHPFrac(b.c.huntState, now) > botRetreatHPFrac
	}
	if ordinaryFinish || counterPushFinish {
		return botHPFrac(b.c.huntState, now) > botRetreatHPFrac
	}
	return botHPFrac(b.c.huntState, now) > 0.12
}

// botPredictiveRetreatLocked combines the current HP floor, a recent loss
// trajectory, and visible nearby hero pressure. The flat 30% threshold remains the
// hard baseline; this is deliberately a narrower early-warning path for a bot that
// is already losing a close fight and is likely to cross that threshold before the
// next think decision.
func (s *Server) botPredictiveRetreatLocked(b *botBrain, now, frac float64) bool {
	if frac > botPredictiveRetreatHPFrac {
		return false
	}
	loss, rate := botRecentHPLossLocked(b, now)
	if loss < botPredictiveRetreatLossFrac || rate < botPredictiveRetreatLossRate {
		return false
	}
	return s.botIncomingPressureLocked(b, now)
}

// botRecentHPLossLocked returns the largest observed fractional loss and its rate
// over the short HP ring. Comparing against every valid older sample makes this
// deterministic even when a damage burst lands between two world ticks.
func botRecentHPLossLocked(b *botBrain, now float64) (loss, rate float64) {
	if b == nil || b.c == nil || b.c.huntState == nil {
		return 0, 0
	}
	current := botHPFrac(b.c.huntState, now)
	for _, sample := range b.hpHistory {
		if sample.t <= 0 || sample.t >= now || now-sample.t > hpCrashWindow {
			continue
		}
		dt := now - sample.t
		if dt < 0.25 {
			continue
		}
		delta := sample.frac - current
		if delta > loss {
			loss = delta
		}
		if candidateRate := delta / dt; candidateRate > rate {
			rate = candidateRate
		}
	}
	return loss, rate
}

// botNearbyEnemyHeroPressureLocked counts visible enemy heroes close enough to
// threaten the current position, including a hero whose PvP target is already this
// bot. It intentionally ignores invisible/dead heroes and never treats nearby enemy
// creeps as hero pressure.
func botNearbyEnemyHeroPressureLocked(b *botBrain, now float64) int {
	if b == nil || b.c == nil || b.c.inst == nil || b.c.huntState == nil {
		return 0
	}
	cx, cy := b.c.posAtLocked(float32(now))
	visionSources := dotaTeamVisionSourcesLocked(b.c.inst, b.c.playerTeam(), now)
	pressure := 0
	for _, enemy := range b.c.inst.members {
		if enemy == nil || enemy == b.c || enemy.huntState == nil ||
			enemy.playerTeam() == b.c.playerTeam() || enemy.huntState.deadUntil > 0 {
			continue
		}
		ex, ey := enemy.posAtLocked(float32(now))
		activelyEngaged := enemy.huntState.pvpTarget == b.c.objID || enemy.huntState.attackTarget == b.c.objID
		if !activelyEngaged && !botVisibleEnemyMemberLocked(b.c.inst, b.c.playerTeam(), enemy, now, visionSources) {
			continue
		}
		if activelyEngaged ||
			b.c.huntState.pvpTarget == enemy.objID ||
			dist2(cx, cy, ex, ey) <= float32(botRetreatPressureRadius*botRetreatPressureRadius) {
			pressure++
		}
	}
	return pressure
}

// botIncomingPressureLocked is the local damage-pressure boundary for tactical
// disengage. It uses only visible enemy units and committed attack state, so a
// hidden creep cannot make a bot flee through fog; a creep that is actually
// hitting the bot is still an immediate survival signal.
func (s *Server) botIncomingPressureLocked(b *botBrain, now float64) bool {
	if b == nil || b.c == nil || b.c.inst == nil || b.c.huntState == nil {
		return false
	}
	if botNearbyEnemyHeroPressureLocked(b, now) > 0 {
		return true
	}
	loss, rate := botRecentHPLossLocked(b, now)
	// A cannon can finish its hit and clear hitTarget before the next bot think
	// tick, so the live-target loop below is not sufficient evidence by itself.
	// Combine the authoritative recent HP trajectory with the visible structure
	// danger envelope: a bot taking sustained damage inside an enemy gun's zone
	// is under pressure even when the gun's one-shot timer has just resolved.
	cx, cy := b.c.posAtLocked(float32(now))
	if s.botEnemyStructureDangerLocked(b.c, cx, cy) {
		if loss >= botPredictiveRetreatLossFrac || rate >= botPredictiveRetreatLossRate {
			return true
		}
	}
	visionSources := dotaTeamVisionSourcesLocked(b.c.inst, b.c.playerTeam(), now)
	// Creep projectiles/melee hits are short-lived state: hitTarget can already
	// be cleared by the time a bot thinks again. A recent material HP loss while
	// a visible enemy creep is still close is therefore authoritative pressure,
	// just like a structure hit. Without this fallback a retreating bot can be
	// sent back into the same wave by XP-shadow logic and lose most of its HP
	// before the next live hit is observable.
	if loss >= botPredictiveRetreatLossFrac || rate >= botPredictiveRetreatLossRate {
		for _, mob := range b.c.inst.mobs {
			if mob == nil || mob.dead || mob.structure || !mob.enemyOf(b.c.playerTeam()) ||
				!botVisibleEnemyMobLocked(b.c.playerTeam(), mob, visionSources) {
				continue
			}
			if math.Hypot(float64(mob.x-cx), float64(mob.y-cy)) <= dotaXPShareRadius+12 {
				return true
			}
		}
	}
	for _, mob := range b.c.inst.mobs {
		if mob == nil || mob.dead || !mob.enemyOf(b.c.playerTeam()) ||
			!botVisibleEnemyMobLocked(b.c.playerTeam(), mob, visionSources) {
			continue
		}
		if mob.hitTarget == b.c.objID || mob.projTarget == b.c.objID || mob.dtarget == b.c.objID {
			return true
		}
	}
	return false
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
	cur := botHPFrac(b.c.huntState, now)
	crash := false
	for _, sample := range b.hpHistory {
		if sample.t == 0 || now-sample.t > hpCrashWindow {
			continue
		}
		if sample.frac-cur >= botHPCrashFrac {
			crash = true
			break
		}
	}
	if !crash {
		return false
	}
	if b.retreating && b.retreatMode == botRetreatModeRecovery {
		return false
	}
	botSetRetreatModeLocked(b, botRetreatModeRecovery, now)
	return true
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
	if b.retreating && b.retreatMode == botRetreatModeDisengage {
		// The short disengage point is useful only after contact is broken. If
		// damage is still landing, that point moves with the creep front and the
		// bot can retreat forever without creating distance. Escalate to the
		// stable recovery route while pressure is live.
		if !s.botNoSafeDisengagePressureLocked(b, now) {
			return s.botDisengagePointLocked(b, now)
		}
	}
	hx, hy := botHomeLocked(b.c)
	if waypoint, ok := s.botRetreatDetourWaypointLocked(b, now, hx, hy); ok {
		return waypoint.x, waypoint.y
	}
	return hx, hy
}

// botNoSafeDisengagePressureLocked distinguishes a nearby, not-yet-committed
// hero from actual contact. A visible hero can make the short step-back safer
// than a fountain trip; an active hero order or a creep currently landing hits
// still forces the stable recovery route.
func (s *Server) botNoSafeDisengagePressureLocked(b *botBrain, now float64) bool {
	if b == nil || b.c == nil || b.c.inst == nil || b.c.huntState == nil {
		return true
	}
	if botNearbyEnemyHeroPressureLocked(b, now) > 0 {
		for _, enemy := range b.c.inst.members {
			if enemy == nil || enemy == b.c || enemy.huntState == nil || enemy.huntState.deadUntil > 0 || enemy.playerTeam() == b.c.playerTeam() {
				continue
			}
			if enemy.huntState.attackTarget == b.c.objID || enemy.huntState.pvpTarget == b.c.objID ||
				b.c.huntState.pvpTarget == enemy.objID {
				return true
			}
		}
		if b.c.huntState.lastHeroDamager != 0 && b.c.huntState.lastHeroDamageAt > 0 &&
			now-b.c.huntState.lastHeroDamageAt <= botRecentHeroDamageThreatWindow {
			return true
		}
	}
	visionSources := dotaTeamVisionSourcesLocked(b.c.inst, b.c.playerTeam(), now)
	for _, mob := range b.c.inst.mobs {
		if mob == nil || mob.dead || !mob.enemyOf(b.c.playerTeam()) ||
			!botVisibleEnemyMobLocked(b.c.playerTeam(), mob, visionSources) {
			continue
		}
		if mob.hitTarget == b.c.objID || mob.projTarget == b.c.objID || mob.dtarget == b.c.objID {
			return true
		}
	}
	return false
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
	// Root is a movement lock, not a full stun: the avatar may still attack or cast,
	// but the bot brain must not re-issue a retreat/lane/coverage move on the next
	// think tick. The player packet and chase paths already honor rooted(); this direct
	// bot path bypasses handleMove, so stop the current leg here as well. Without this,
	// a root applied to a retreating bot was immediately overwritten by its own next
	// movement decision and the client showed it walking through the CC.
	if hs.st.rooted(now) {
		moving := c.hasDest || c.vx != 0 || c.vy != 0
		c.stopArrivalLocked()
		cx, cy := c.posAtLocked(float32(now))
		c.x, c.y, c.vx, c.vy, c.snapT = cx, cy, 0, 0, float32(now)
		c.hasDest = false
		if moving {
			c.sendPosLocked(s, cx, cy, 0, 0, float32(now))
		}
		return
	}
	// Cancel any live attack/order BEFORE the "already there" short-circuit below: a bot
	// that decides to disengage while standing right on top of its destination (e.g.
	// retreating to base while already near base) must still drop its own attack chase,
	// not just skip the no-op move.
	if hs.attackTarget != 0 || (hs.attackActionActive && hs.pvpTarget == 0) {
		s.stopAttackLocked(c, false)
	}
	if hs.pvpTarget != 0 || (hs.attackActionActive && hs.attackTarget == 0) {
		s.stopPvpAttackLocked(c, false)
	}
	s.cancelOrderLocked(c)
	cx, cy := c.posAtLocked(float32(now))
	if dist2(cx, cy, tx, ty) <= 2*2 {
		return // already there
	}
	c.moveToLocked(s, tx, ty)
}
