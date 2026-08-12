package battleserver

import (
	"math"
	"sort"

	"tanatserver/internal/gamedata"
)

// botFarmChoice is the explainable result of the farm director. The target is
// always a live Dota lane creep; callers still pass it through the normal attack
// and ability validation paths.
type botFarmChoice struct {
	target   *mobState
	score    float64
	decision string
	lane     int
	catchUp  bool
}

func botSortedMobs(inst *huntInstance) []*mobState {
	if inst == nil {
		return nil
	}
	out := make([]*mobState, 0, len(inst.mobs))
	for _, m := range inst.mobs {
		if m != nil {
			out = append(out, m)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].id < out[j].id })
	return out
}

// botSameDotaLane compares the authored polyline rather than only its first
// waypoint. The three lanes leave the altar through a shared first point on the
// live map, so a first-point-only comparison silently classified lane 2 as lane
// 1 and starved farm/teleport decisions on that lane.
func botSameDotaLane(left, right []gamedata.Vec2) bool {
	if len(left) != len(right) || len(left) == 0 {
		return false
	}
	if &left[0] == &right[0] {
		return true
	}
	for i := range left {
		if left[i].X != right[i].X || left[i].Y != right[i].Y {
			return false
		}
	}
	return true
}

func botLaneForCreep(c *conn, m *mobState) int {
	if c == nil || c.inst == nil || c.inst.dota == nil || m == nil || len(m.lane) == 0 {
		return -1
	}
	for i, lane := range c.inst.dota.m.Lanes {
		if botSameDotaLane(lane, m.lane) {
			return i
		}
	}
	return -1
}

func botFarmLaneLocked(b *botBrain) int {
	if b == nil {
		return -1
	}
	if b.macroAssignment.FarmLaneSet && b.macroAssignment.FarmLane >= 0 {
		// The orchestrator may retarget a cover assignment during a live-wave
		// hand-off. Baseline coverage is an actual movement contract, so its
		// lane must win over a stale catch-up lane left by the previous plan;
		// otherwise telemetry shows lane 2 while the bot keeps farming lane 0.
		if b.macroAssignment.Role == "lane_cover" && b.macroAssignment.Reason == "baseline_lane_coverage" &&
			b.macroAssignment.Lane >= 0 {
			return b.macroAssignment.Lane
		}
		if b.macroAssignment.Mode == botMacroBase && b.macroAssignment.Lane >= 0 {
			// A base defender's farm opportunity is the threatened structure lane;
			// a stale reserve lane must not pull it away from the nearby wave.
			return b.macroAssignment.Lane
		}
		return b.macroAssignment.FarmLane
	}
	// Strategic assignments may move a bot to a live objective lane, but a
	// baseline lane remains the farm owner whenever the assignment is a cover or
	// emergency response. This prevents every bot from following one push lane.
	if b.macroAssignment.Lane >= 0 && (b.macroAssignment.Mode == botMacroLane ||
		b.macroAssignment.Mode == botMacroCover || b.macroAssignment.Mode == botMacroRecover ||
		b.macroAssignment.Mode == botMacroBase) {
		return b.macroAssignment.Lane
	}
	return b.lane
}

// botFirstFarmWavePendingLocked identifies a zero-XP bot whose assigned lane
// has not yet exposed a visible enemy wave. Own creeps are deliberately ignored:
// they spawn first and are not a useful signal that the bot is staged to receive
// XP. The caller uses this state to pre-stage at the safe lane point, so the bot
// is already close when the first opposing wave appears; it does not authorize
// a blind target through fog of war.
func botFirstFarmWavePendingLocked(b *botBrain, now float64) bool {
	if b == nil || b.c == nil || b.c.huntState == nil || b.c.inst == nil || b.c.inst.dota == nil ||
		b.farmXPEvents != 0 || b.c.huntState.deadUntil > 0 || b.retreating || b.macroAssignment.Reason == "" {
		return false
	}
	lane := botFarmLaneLocked(b)
	if lane < 0 {
		return false
	}
	visionSources := dotaTeamVisionSourcesLocked(b.c.inst, b.c.playerTeam(), now)
	for _, m := range b.c.inst.mobs {
		if m == nil || m.dead || m.structure || !m.enemyOf(b.c.playerTeam()) || !botTeleportLaneCreep(m) ||
			botLaneForCreep(b.c, m) != lane || !botVisibleEnemyMobLocked(b.c.playerTeam(), m, visionSources) {
			continue
		}
		return false
	}
	return true
}

// botFarmLaneWaveScoreLocked reports whether a lane can currently convert a
// farm assignment into XP. The orchestrator uses this live signal to rescue a
// weak bot from an empty lane; local catch-up remains a safety net, not the
// owner of the strategic lane decision.
func botFarmLaneWaveScoreLocked(inst *huntInstance, team int32, lane int) int {
	now := float64(0)
	if inst != nil && inst.dota != nil {
		now = inst.dota.startedAt
	}
	return botFarmLaneWaveScoreAtLocked(inst, team, lane, now)
}

func botFarmLaneWaveScoreAtLocked(inst *huntInstance, team int32, lane int, now float64) int {
	if inst == nil || inst.dota == nil || lane < 0 || lane >= len(inst.dota.m.Lanes) {
		return 0
	}
	score := 0
	visionSources := dotaTeamVisionSourcesLocked(inst, team, now)
	for _, mob := range botSortedMobs(inst) {
		if mob.dead || mob.structure || !mob.enemyOf(team) ||
			!botVisibleEnemyMobLocked(team, mob, visionSources) || !botTeleportLaneCreep(mob) ||
			botLaneForCreep(&conn{inst: inst}, mob) != lane {
			continue
		}
		score++
		if mob.dtarget != 0 || mob.hitTarget != 0 || mob.projTarget != 0 {
			score++
		}
	}
	return score
}

func botWeakestLivingBotLocked(inst *huntInstance, team int32) *botBrain {
	if inst == nil {
		return nil
	}
	var best *botBrain
	for _, b := range inst.bots {
		if b == nil || b.c == nil || b.c.huntState == nil || b.c.playerTeam() != team || b.c.huntState.deadUntil > 0 || b.retreating {
			continue
		}
		if best == nil || b.c.huntState.level < best.c.huntState.level ||
			(b.c.huntState.level == best.c.huntState.level && b.c.huntState.xp < best.c.huntState.xp) ||
			(b.c.huntState.level == best.c.huntState.level && b.c.huntState.xp == best.c.huntState.xp && b.c.objID < best.c.objID) {
			best = b
		}
	}
	return best
}

// botFarmDebtLocked is relative to the most farm XP events currently achieved
// by a living bot on the same team. It is intentionally state-based: no match
// minute or opening-phase constant decides whether a bot is behind.
func botFarmDebtLocked(inst *huntInstance, b *botBrain) int {
	if inst == nil || b == nil || b.c == nil || b.c.huntState == nil {
		return 0
	}
	maxEvents := b.farmXPEvents
	for _, ally := range inst.bots {
		if ally == nil || ally.c == nil || ally.c.huntState == nil || ally.c.playerTeam() != b.c.playerTeam() || ally.c.huntState.deadUntil > 0 {
			continue
		}
		if ally.farmXPEvents > maxEvents {
			maxEvents = ally.farmXPEvents
		}
	}
	return maxEvents - b.farmXPEvents
}

func botFarmPriorityLessLocked(left, right *botBrain) bool {
	if left == nil {
		return false
	}
	if right == nil {
		return true
	}
	leftDebt := botFarmDebtLocked(left.c.inst, left)
	rightDebt := botFarmDebtLocked(right.c.inst, right)
	if leftDebt != rightDebt {
		return leftDebt > rightDebt
	}
	if left.farmXPEvents != right.farmXPEvents {
		return left.farmXPEvents < right.farmXPEvents
	}
	if left.farmLastXPTAt != right.farmLastXPTAt {
		if left.farmLastXPTAt == 0 {
			return true
		}
		if right.farmLastXPTAt == 0 {
			return false
		}
		return left.farmLastXPTAt < right.farmLastXPTAt
	}
	if left.c != nil && right.c != nil && left.c.huntState != nil && right.c.huntState != nil {
		if left.c.huntState.level != right.c.huntState.level {
			return left.c.huntState.level < right.c.huntState.level
		}
		if left.c.huntState.xp != right.c.huntState.xp {
			return left.c.huntState.xp < right.c.huntState.xp
		}
		return left.c.objID < right.c.objID
	}
	return false
}

func botMostIndebtedLivingBotLocked(inst *huntInstance, team int32) *botBrain {
	if inst == nil {
		return nil
	}
	var best *botBrain
	for _, b := range inst.bots {
		if b == nil || b.c == nil || b.c.huntState == nil || b.c.playerTeam() != team || b.c.huntState.deadUntil > 0 || b.retreating {
			continue
		}
		if botFarmPriorityLessLocked(b, best) {
			best = b
		}
	}
	return best
}

func botFarmCatchUpLocked(b *botBrain) bool {
	if b == nil || b.c == nil || b.c.inst == nil {
		return false
	}
	weak := botMostIndebtedLivingBotLocked(b.c.inst, b.c.playerTeam())
	return weak == b && weak != nil
}

// botFarmTargetClaimedByCloserAllyLocked reserves one XP wave for the bot that
// can convert it most efficiently. A farm target is a live decision, not a
// permanent reservation: stale/dead targets and retreating bots do not block
// anyone. If there is no alternative wave, botFarmTargetLocked deliberately
// falls back to the claimed candidate so a bot never becomes idle merely to
// avoid sharing XP.
func botFarmTargetClaimedByCloserAllyLocked(b *botBrain, target *mobState, now float64) bool {
	if b == nil || b.c == nil || b.c.inst == nil || target == nil || target.dead || target.structure {
		return false
	}
	cx, cy := b.c.posAtLocked(float32(now))
	botDistance := dist2(cx, cy, target.x, target.y)
	claimRadius2 := float32(dotaXPShareRadius * dotaXPShareRadius)
	for _, ally := range b.c.inst.bots {
		if ally == nil || ally == b || ally.c == nil || ally.c.huntState == nil || ally.c.playerTeam() != b.c.playerTeam() || ally.c.huntState.deadUntil > 0 || ally.retreating || ally.farmTarget == 0 || ally.farmDecision == "lane_move" {
			continue
		}
		claimed := b.c.inst.mobs[ally.farmTarget]
		if claimed == nil || claimed.dead || claimed.structure || !claimed.enemyOf(b.c.playerTeam()) {
			continue
		}
		if dist2(claimed.x, claimed.y, target.x, target.y) > claimRadius2 {
			continue
		}
		ax, ay := ally.c.posAtLocked(float32(now))
		allyDistance := dist2(ax, ay, target.x, target.y)
		if allyDistance+0.25 < botDistance || (math.Abs(float64(allyDistance-botDistance)) <= 0.25 && ally.c.objID < b.c.objID) {
			return true
		}
	}
	return false
}

func (s *Server) botFarmTargetLocked(b *botBrain, now, radius float64, requireCover bool) *mobState {
	if b == nil || b.c == nil || b.c.huntState == nil || b.c.inst == nil {
		return nil
	}
	c, hs := b.c, b.c.huntState
	cx, cy := c.posAtLocked(float32(now))
	lane := botFarmLaneLocked(b)
	catchUp := botFarmCatchUpLocked(b)
	visionSources := dotaTeamVisionSourcesLocked(c.inst, c.playerTeam(), now)
	// Keep a live target while the bot is converting it. Re-solving from scratch
	// every 200ms made a bot oscillate between equally good creeps and repeatedly
	// cancel its approach before it ever entered the XP radius. The reservation is
	// spatial/state-based: it expires when the creep dies, changes lane, becomes
	// unsafe, or leaves the local farm radius. No match-time lock is involved.
	if b.farmTarget != 0 {
		if committed := c.inst.mobs[b.farmTarget]; committed != nil && !committed.dead &&
			!committed.structure && committed.enemyOf(c.playerTeam()) &&
			botVisibleEnemyMobLocked(c.playerTeam(), committed, visionSources) &&
			botTeleportLaneCreep(committed) && botLaneForCreep(c, committed) == lane {
			distance := math.Hypot(float64(committed.x-cx), float64(committed.y-cy))
			if distance <= radius &&
				(!requireCover || (!s.botCreepFightRiskyLocked(c, cx, cy) && !s.botEnemyStructureDangerLocked(c, committed.x, committed.y))) {
				// Keep a target sticky while approaching it, but let a creep that is
				// already in attack range take precedence over a farther reservation.
				// This avoids walking past a nearly dead wave member merely because
				// the director selected its full-health neighbour one tick earlier.
				attackReach := hs.effAttackRangeLocked(now) + hs.av.Radius() + committed.mob.Radius()
				if distance <= attackReach {
					return committed
				}
			}
			b.farmTarget = 0
		}
	}

	var best, claimedBest botFarmChoice
	best.score = math.Inf(-1)
	claimedBest.score = math.Inf(-1)
	// The team orchestrator owns the line. Catch-up changes priority on this
	// assigned line; it must never silently turn into a local cross-map lane
	// choice, otherwise two bots can abandon an uncovered wave at the same time.
	for _, candidateLane := range []int{lane} {
		for _, m := range botSortedMobs(c.inst) {
			if m.dead || m.structure || !m.enemyOf(c.playerTeam()) ||
				!botVisibleEnemyMobLocked(c.playerTeam(), m, visionSources) || !botTeleportLaneCreep(m) {
				continue
			}
			mLane := botLaneForCreep(c, m)
			// Some older/unit-created creeps do not carry an authored lane tag.
			// They are still legal local farm targets; real lane creeps are always
			// tagged and remain strictly owned by the orchestrator's lane.
			if mLane >= 0 && mLane != candidateLane {
				continue
			}
			distance := math.Hypot(float64(m.x-cx), float64(m.y-cy))
			if distance > radius {
				continue
			}
			if requireCover && (s.botCreepFightRiskyLocked(c, cx, cy) || s.botEnemyStructureDangerLocked(c, m.x, m.y)) {
				continue
			}
			cluster := 0
			for _, other := range botSortedMobs(c.inst) {
				if other.dead || other.structure || !other.enemyOf(c.playerTeam()) ||
					!botVisibleEnemyMobLocked(c.playerTeam(), other, visionSources) || botLaneForCreep(c, other) != candidateLane {
					continue
				}
				if dist2(other.x, other.y, m.x, m.y) <= 5.5*5.5 {
					cluster++
				}
			}
			// XP is awarded by proximity when the creep dies; gold ownership is
			// irrelevant for these bots. Score the whole live wave first, then
			// travel distance. A nearly dead creep has no special value: the bot
			// should remain in the XP radius and help clear the wave instead of
			// competing for a gold last hit it does not need.
			score := float64(cluster*72) + m.xpReward()*0.5 - distance*4
			if distance <= hs.effAttackRangeLocked(now)+hs.av.Radius()+m.mob.Radius() {
				score += 24
			}
			if catchUp {
				score += 18
			}
			for _, ally := range botLivingAllies(c) {
				ax, ay := ally.posAtLocked(float32(now))
				if dist2(ax, ay, m.x, m.y) <= float32(dotaXPShareRadius*dotaXPShareRadius) {
					score -= 18 // avoid needless XP splitting when another ally owns this wave
				}
			}
			// Keep one stable farm label. There is no gold-driven last-hit mode:
			// all creep decisions are about wave coverage and XP proximity.
			decision := "wave_clear"
			choice := botFarmChoice{target: m, score: score, decision: decision, lane: candidateLane, catchUp: catchUp}
			if botFarmTargetClaimedByCloserAllyLocked(b, m, now) {
				if score > claimedBest.score || (score == claimedBest.score && (claimedBest.target == nil || m.id < claimedBest.target.id)) {
					claimedBest = choice
				}
				continue
			}
			if score > best.score || (score == best.score && (best.target == nil || m.id < best.target.id)) {
				best = choice
			}
		}
	}
	if best.target == nil {
		best = claimedBest
	}
	if best.target == nil {
		b.farmTarget, b.farmTargetScore, b.farmDecision = 0, 0, "lane_move"
		b.farmLane, b.farmCatchUp = lane, catchUp
		return nil
	}
	previousTarget := b.farmTarget
	b.farmTarget, b.farmTargetScore, b.farmDecision = best.target.id, best.score, best.decision
	b.farmLane, b.farmCatchUp = best.lane, best.catchUp
	if previousTarget == best.target.id {
		return best.target
	}
	b.farmWaveClears++
	return best.target
}

// botFarmCoveragePointLocked returns the centre of the local visible wave the
// bot is responsible for. Choosing one creep as the movement anchor made a
// cover bot stand on the first creep it reached while a neighbouring creep of
// the same wave died outside XP radius. The point is built from a local cluster
// around the closest live enemy creep, with an own creep fallback when the
// enemy wave is not currently visible.
func (s *Server) botFarmCoveragePointLocked(b *botBrain, now float64) (float32, float32, bool) {
	if b == nil || b.c == nil || b.c.inst == nil || b.c.huntState == nil {
		return 0, 0, false
	}
	c := b.c
	lane := botFarmLaneLocked(b)
	if lane < 0 {
		return 0, 0, false
	}
	cx, cy := c.posAtLocked(float32(now))
	sources := dotaTeamVisionSourcesLocked(c.inst, c.playerTeam(), now)
	var enemies, allies []*mobState
	for _, mob := range botSortedMobs(c.inst) {
		if mob == nil || mob.dead || mob.structure || !botTeleportLaneCreep(mob) || botLaneForCreep(c, mob) != lane {
			continue
		}
		if mob.enemyOf(c.playerTeam()) {
			if botVisibleEnemyMobLocked(c.playerTeam(), mob, sources) {
				enemies = append(enemies, mob)
			}
		} else if mob.team == c.playerTeam() {
			allies = append(allies, mob)
		}
	}
	cluster := enemies
	if len(cluster) == 0 {
		cluster = allies
	}
	if len(cluster) == 0 {
		return 0, 0, false
	}
	var seed *mobState
	bestDistance := math.Inf(1)
	for _, mob := range cluster {
		distance := math.Hypot(float64(mob.x-cx), float64(mob.y-cy))
		if distance < bestDistance || (distance == bestDistance && (seed == nil || mob.id < seed.id)) {
			seed, bestDistance = mob, distance
		}
	}
	const localWaveRadius = 64.0
	var sumX, sumY float64
	count := 0
	for _, mob := range cluster {
		if math.Hypot(float64(mob.x-seed.x), float64(mob.y-seed.y)) > localWaveRadius {
			continue
		}
		sumX += float64(mob.x)
		sumY += float64(mob.y)
		count++
	}
	if count == 0 {
		return seed.x, seed.y, true
	}
	centerX, centerY := float32(sumX/float64(count)), float32(sumY/float64(count))
	return centerX, centerY, true
}

// botHoldFarmXPLocked anchors a laner while a live, visible enemy creep is
// already inside the proximity-XP radius. Chasing the next creep at that point
// is strategically wrong: the current wave member can die behind the bot and
// become an XP miss. The bot may still attack a creep already in swing range;
// otherwise it cancels an old movement leg and waits for the wave to advance.
// Incoming damage and structure danger yield to the normal survival path.
func (s *Server) botHoldFarmXPLocked(b *botBrain, now float64) bool {
	if b == nil || b.c == nil || b.c.inst == nil || b.c.huntState == nil || b.retreating ||
		b.c.huntState.deadUntil > 0 || s.botIncomingPressureLocked(b, now) {
		return false
	}
	c := b.c
	cx, cy := c.posAtLocked(float32(now))
	if s.botEnemyStructureDangerLocked(c, cx, cy) {
		return false
	}
	lane := botFarmLaneLocked(b)
	if lane < 0 {
		return false
	}
	sources := dotaTeamVisionSourcesLocked(c.inst, c.playerTeam(), now)
	var nearest *mobState
	nearestDistance := math.Inf(1)
	for _, mob := range botSortedMobs(c.inst) {
		if mob.dead || mob.structure || !mob.enemyOf(c.playerTeam()) || !botTeleportLaneCreep(mob) ||
			botLaneForCreep(c, mob) != lane || !botVisibleEnemyMobLocked(c.playerTeam(), mob, sources) {
			continue
		}
		distance := math.Hypot(float64(mob.x-cx), float64(mob.y-cy))
		if distance > dotaXPShareRadius || distance >= nearestDistance {
			continue
		}
		nearest, nearestDistance = mob, distance
	}
	if nearest == nil {
		return false
	}
	b.farmTarget = nearest.id
	b.farmTargetScore = 0
	attackReach := c.huntState.effAttackRangeLocked(now) + c.huntState.av.Radius() + nearest.mob.Radius()
	if nearestDistance <= attackReach {
		b.farmDecision = "wave_clear"
		s.startAttackLocked(c, nearest)
	} else {
		b.farmDecision = "wave_anchor"
		c.stopMovementLocked(s, now)
	}
	return true
}

// botFarmGuardianHoldLocked is the survival exception for the last nearby XP
// body on a lane. It only protects a healthy bot while no unit is actively
// damaging it; once a creep starts landing hits, survival takes priority over
// staying inside the XP radius.
func (s *Server) botFarmGuardianHoldLocked(b *botBrain, now float64) bool {
	if b == nil || b.c == nil || b.c.huntState == nil || b.c.inst == nil ||
		botHPFrac(b.c.huntState, now) <= botRetreatHPFrac || botNearbyEnemyHeroPressureLocked(b, now) > 0 {
		return false
	}
	lane := botFarmLaneLocked(b)
	if lane < 0 {
		return false
	}
	c := b.c
	cx, cy := c.posAtLocked(float32(now))
	if s.botEnemyStructureDangerLocked(c, cx, cy) {
		return false
	}
	sources := dotaTeamVisionSourcesLocked(c.inst, c.playerTeam(), now)
	nearest := math.Inf(1)
	var wave *mobState
	for _, mob := range botSortedMobs(c.inst) {
		if mob == nil || mob.dead || mob.structure || !mob.enemyOf(c.playerTeam()) ||
			!botTeleportLaneCreep(mob) || botLaneForCreep(c, mob) != lane ||
			!botVisibleEnemyMobLocked(c.playerTeam(), mob, sources) {
			continue
		}
		distance := math.Hypot(float64(mob.x-cx), float64(mob.y-cy))
		if distance < nearest {
			nearest, wave = distance, mob
		}
	}
	// This is deliberately a narrow latch: only cancel a new retreat when the
	// bot is already inside the reward radius. A 50u look-ahead caused healthy
	// lane bodies to hold against pressure instead of rotating, while a wider
	// radius does not protect the next XP event anyway.
	if wave == nil || nearest > dotaXPShareRadius || s.botEnemyStructureDangerLocked(c, wave.x, wave.y) ||
		s.botIncomingPressureLocked(b, now) {
		return false
	}
	// Only the last local body gets this exception. If another living ally is
	// already close enough, the current bot may safely disengage and preserve HP.
	for _, ally := range botLivingAllies(c) {
		if ally == nil || ally == c || ally.huntState == nil || ally.huntState.deadUntil > 0 {
			continue
		}
		ax, ay := ally.posAtLocked(float32(now))
		if math.Hypot(float64(ax-wave.x), float64(ay-wave.y)) <= dotaXPShareRadius*1.25 {
			return false
		}
	}
	return true
}

// botUrgentFarmCoverageTargetLocked returns a visible enemy creep whose death
// is currently uncovered by every living ally. It is the combat-to-farm
// hand-off signal: optional hero pressure must not consume the only local body
// that can still receive this proximity-XP event.
func (s *Server) botUrgentFarmCoverageTargetLocked(b *botBrain, now float64) *mobState {
	if b == nil || b.c == nil || b.c.inst == nil || b.c.huntState == nil {
		return nil
	}
	c := b.c
	lane := botFarmLaneLocked(b)
	sources := dotaTeamVisionSourcesLocked(c.inst, c.playerTeam(), now)
	allies := botLivingAllies(c)
	cx, cy := c.posAtLocked(float32(now))
	var best *mobState
	bestDistance := math.Inf(1)
	for _, mob := range botSortedMobs(c.inst) {
		if mob == nil || mob.dead || mob.structure || !mob.enemyOf(c.playerTeam()) ||
			!botTeleportLaneCreep(mob) || botLaneForCreep(c, mob) != lane ||
			!botVisibleEnemyMobLocked(c.playerTeam(), mob, sources) {
			continue
		}
		covered := false
		for _, ally := range allies {
			if ally == nil || ally.huntState == nil || ally.huntState.deadUntil > 0 {
				continue
			}
			ax, ay := ally.posAtLocked(float32(now))
			if math.Hypot(float64(ax-mob.x), float64(ay-mob.y)) <= dotaXPShareRadius {
				covered = true
				break
			}
		}
		if covered {
			continue
		}
		distance := math.Hypot(float64(mob.x-cx), float64(mob.y-cy))
		if distance < bestDistance || (distance == bestDistance && (best == nil || mob.id < best.id)) {
			best, bestDistance = mob, distance
		}
	}
	return best
}

const (
	botBarracksImmediateRadius = 32.0
	// Predictive geometry is used only to choose an intercept after a direct
	// defense assignment exists; it does not activate or retain base defense.
	botBarracksPredictiveRadius = 120.0
	botBarracksPredictiveETA    = 30.0
)

// botBarracksInterceptTargetLocked finds the live lane creep that keeps a
// barracks-defense premise alive but is outside the ordinary 96u farm approach
// radius.  The target is deliberately derived from the same barracks threat
// geometry used by the orchestrator, so a defender never abandons its objective
// for an unrelated wave.
func (s *Server) botBarracksInterceptTargetLocked(b *botBrain, now float64) *mobState {
	if b == nil || b.c == nil || b.c.inst == nil || b.c.inst.dota == nil {
		return nil
	}
	objective := b.c.inst.mobs[b.macroAssignment.ObjectiveID]
	if objective == nil || objective.dead || !objective.structure ||
		objective.team != b.c.playerTeam() || objective.dotaRole != gamedata.DotaCreepTower {
		return nil
	}
	lane := botNearestLaneToPointLocked(b.c.inst.dota, objective.x, objective.y)
	cx, cy := b.c.posAtLocked(float32(now))
	var best *mobState
	bestScore := math.Inf(1)
	visionSources := dotaTeamVisionSourcesLocked(b.c.inst, b.c.playerTeam(), now)
	for _, m := range botSortedMobs(b.c.inst) {
		if m.dead || m.structure || !m.enemyOf(b.c.playerTeam()) ||
			!botVisibleEnemyMobLocked(b.c.playerTeam(), m, visionSources) || !botTeleportLaneCreep(m) ||
			botLaneForCreep(b.c, m) != lane {
			continue
		}
		objectiveDistance := math.Hypot(float64(m.x-objective.x), float64(m.y-objective.y))
		if objectiveDistance > botBarracksPredictiveRadius ||
			!botBarracksCreepApproachingWithinRadiusLocked(b.c.inst, objective, m,
				objectiveDistance, botBarracksPredictiveRadius, false) {
			continue
		}
		// Prefer the creep the defender can meet soonest, with a small objective
		// proximity tie-break so the guard does not chase a trailing creep while
		// an imminent one is already on the same lane.
		score := math.Hypot(float64(m.x-cx), float64(m.y-cy)) + objectiveDistance*0.05
		if score < bestScore || (score == bestScore && (best == nil || m.id < best.id)) {
			best, bestScore = m, score
		}
	}
	return best
}

// botBarracksThreatLocked separates a lane's DotaCreepTower from the altar.
// A direct barracks threat is live contact or a committed structure target; a
// merely approaching wave remains a farm assignment, not a base emergency.

// botBarracksThreatSeverityLocked evaluates one barracks independently. Keeping
// this local score separate from the global winner lets the orchestrator defend
// the exact lane under contact without pulling the whole roster to base.
func botBarracksThreatSeverityLocked(inst *huntInstance, team int32, barracks *mobState, now float64) int {
	return botBarracksThreatSeverityWithinRadiusLocked(inst, team, barracks, now, botBarracksImmediateRadius, botBarracksImmediateRadius, false)
}

// botBarracksThreatReleaseSeverityLocked is kept as a named premise check for
// plan/teleport callers, but uses the same direct live threat as activation.
func botBarracksThreatReleaseSeverityLocked(inst *huntInstance, team int32, barracks *mobState, now float64) int {
	return botBarracksThreatSeverityLocked(inst, team, barracks, now)
}

// botStructureDirectThreatSeverityLocked is the shared contact test for
// structures that do not have the barracks lane/progress rules. A structure is
// under direct threat when an enemy is inside its local defense ring or an
// enemy has already committed its attack to that structure. Damage history is
// intentionally ignored: once attackers leave, the defense assignment must
// disappear and the roster may farm or gank again.
func botStructureDirectThreatSeverityLocked(inst *huntInstance, team int32, structure *mobState, now, radius float64) int {
	if inst == nil || structure == nil || structure.dead || !structure.structure || structure.team != team {
		return 0
	}
	r2 := float32(radius * radius)
	severity := 0
	visionSources := dotaTeamVisionSourcesLocked(inst, team, now)
	for _, m := range botSortedMobs(inst) {
		if m.dead || m.structure || !m.enemyOf(team) || !botVisibleEnemyMobLocked(team, m, visionSources) {
			continue
		}
		targeting := m.dtarget == structure.id || m.hitTarget == structure.id || m.projTarget == structure.id
		if !targeting && dist2(m.x, m.y, structure.x, structure.y) > r2 {
			continue
		}
		severity++
		if targeting {
			severity += 2
		}
	}
	for _, mem := range inst.members {
		if !botVisibleEnemyMemberLocked(inst, team, mem, now, visionSources) {
			continue
		}
		x, y := mem.posAtLocked(float32(now))
		if dist2(x, y, structure.x, structure.y) > r2 {
			continue
		}
		severity += 2
		if mem.huntState.attackTarget == structure.id {
			severity += 2
		}
	}
	return severity
}

func (s *Server) botDefenseStructureThreatSeverityLocked(inst *huntInstance, team int32, structure *mobState, now float64) int {
	if structure == nil || structure.dead || !structure.structure || structure.team != team {
		return 0
	}
	switch structure.dotaRole {
	case gamedata.DotaCreepTower:
		return botBarracksThreatSeverityLocked(inst, team, structure, now)
	case gamedata.DotaAltar:
		return botStructureDirectThreatSeverityLocked(inst, team, structure, now, botMacroBasePressureRadius)
	case gamedata.DotaGun:
		return botStructureDirectThreatSeverityLocked(inst, team, structure, now, botBarracksImmediateRadius)
	default:
		return 0
	}
}

// botOwnDefenseStructureThreatLocked selects the one currently endangered
// objective. The priority is altar > barracks > gun when severity ties: the
// altar is match-ending, while barracks are more strategically important than
// a gun because losing them removes the lane's ability to pressure safely.
func (s *Server) botOwnDefenseStructureThreatLocked(inst *huntInstance, team int32, now float64) (*mobState, int) {
	if inst == nil {
		return nil, 0
	}
	priority := func(role gamedata.DotaRole) int {
		switch role {
		case gamedata.DotaAltar:
			return 3
		case gamedata.DotaCreepTower:
			return 2
		case gamedata.DotaGun:
			return 1
		default:
			return 0
		}
	}
	var best *mobState
	bestSeverity, bestPriority := 0, 0
	for _, structure := range botSortedMobs(inst) {
		if structure.team != team || !structure.structure {
			continue
		}
		severity := s.botDefenseStructureThreatSeverityLocked(inst, team, structure, now)
		p := priority(structure.dotaRole)
		if severity > bestSeverity ||
			(severity == bestSeverity && severity > 0 && (p > bestPriority ||
				(p == bestPriority && (best == nil || structure.id < best.id)))) {
			best, bestSeverity, bestPriority = structure, severity, p
		}
	}
	return best, bestSeverity
}

func botBarracksThreatSeverityWithinRadiusLocked(inst *huntInstance, team int32, barracks *mobState, now, localRadius, predictionRadius float64, requireETA bool) int {
	if inst == nil || inst.dota == nil || barracks == nil || barracks.dead || !barracks.structure || barracks.altar ||
		barracks.team != team || barracks.dotaRole != gamedata.DotaCreepTower {
		return 0
	}
	laneIndex := botNearestLaneToPointLocked(inst.dota, barracks.x, barracks.y)
	radius2 := float32(botBarracksImmediateRadius * botBarracksImmediateRadius)
	severity := 0
	visionSources := dotaTeamVisionSourcesLocked(inst, team, now)
	for _, m := range botSortedMobs(inst) {
		if m.dead || m.structure || !m.enemyOf(team) || !botVisibleEnemyMobLocked(team, m, visionSources) {
			continue
		}
		distance := math.Hypot(float64(m.x-barracks.x), float64(m.y-barracks.y))
		targetingBarracks := m.dtarget == barracks.id || m.hitTarget == barracks.id || m.projTarget == barracks.id
		// Contact is an objective-local live threat even if a synthetic/test
		// creep or a briefly displaced wave has not retained the lane tag.
		// Progress/ETA forecasting below still requires the authored lane.
		if distance > localRadius && !targetingBarracks && botLaneForCreep(&conn{inst: inst}, m) != laneIndex {
			continue
		}
		if !targetingBarracks && !botBarracksCreepApproachingWithinRadiusLocked(inst, barracks, m, distance, predictionRadius, requireETA) {
			continue
		}
		severity++
		if distance > botBarracksImmediateRadius {
			severity++ // predictive wave: reserve coverage before contact
		}
		if targetingBarracks {
			severity += 2
		}
	}
	for _, mem := range inst.members {
		if !botVisibleEnemyMemberLocked(inst, team, mem, now, visionSources) {
			continue
		}
		x, y := mem.posAtLocked(float32(now))
		if dist2(x, y, barracks.x, barracks.y) <= radius2 {
			severity += 3
			if mem.huntState.pvpTarget != 0 || mem.huntState.attackTarget != 0 {
				severity++
			}
		}
	}
	return severity
}

func (s *Server) botBarracksThreatLocked(inst *huntInstance, team int32, now float64) (*mobState, int) {
	if inst == nil || inst.dota == nil {
		return nil, 0
	}
	type candidate struct {
		m        *mobState
		severity int
	}
	var cs []candidate
	for _, barracks := range botSortedMobs(inst) {
		severity := botBarracksThreatSeverityLocked(inst, team, barracks, now)
		if severity > 0 {
			cs = append(cs, candidate{m: barracks, severity: severity})
		}
	}
	sort.Slice(cs, func(i, j int) bool {
		if cs[i].severity != cs[j].severity {
			return cs[i].severity > cs[j].severity
		}
		return cs[i].m.id < cs[j].m.id
	})
	if len(cs) == 0 {
		return nil, 0
	}
	return cs[0].m, cs[0].severity
}

func botBarracksCreepApproachingLocked(inst *huntInstance, barracks, creep *mobState, distance float64) bool {
	return botBarracksCreepApproachingWithinRadiusLocked(inst, barracks, creep, distance, botBarracksPredictiveRadius, true)
}

func botBarracksCreepApproachingWithinRadiusLocked(inst *huntInstance, barracks, creep *mobState, distance, predictionRadius float64, requireETA bool) bool {
	if inst == nil || inst.dota == nil || barracks == nil || creep == nil || len(creep.lane) < 2 {
		return false
	}
	if distance <= botBarracksImmediateRadius {
		return true
	}
	lane := botNearestLaneToPointLocked(inst.dota, barracks.x, barracks.y)
	if lane < 0 || lane >= len(inst.dota.m.Lanes) || botLaneForCreep(&conn{inst: inst}, creep) != lane ||
		creep.laneIdx < 0 || creep.laneIdx >= len(creep.lane) {
		return false
	}
	// Lanes are authored from the human altar to the elf altar. A creep is
	// approaching this team's barracks when its authored direction points at
	// that team's endpoint and its remaining waypoint progress is closing.
	targetsForward := barracks.team == dotaTeamElf
	if creep.laneFwd != targetsForward {
		return false
	}
	endpoint := 0
	if targetsForward {
		endpoint = len(creep.lane) - 1
	}
	remaining := endpoint - creep.laneIdx
	if remaining < 0 {
		remaining = -remaining
	}
	progress := 1 - float64(remaining)/float64(len(creep.lane)-1)
	if progress < 0.35 {
		return false
	}
	if distance > predictionRadius {
		return false
	}
	if !requireETA {
		return true
	}
	speed := creep.mob.Speed
	if speed <= 0 {
		speed = 4.0
	}
	eta := distance / speed
	return eta <= botBarracksPredictiveETA
}

// botSelectTeamFocusTargetLocked is shared by every bot on a side. It only
// ranks visible living enemies; the individual combat gate still enforces range,
// structure danger, HP, and local headcount before acting.
func botSelectTeamFocusTargetLocked(inst *huntInstance, team int32, now float64) *conn {
	if inst == nil {
		return nil
	}
	type candidate struct {
		c     *conn
		score float64
	}
	var cs []candidate
	visionSources := dotaTeamVisionSourcesLocked(inst, team, now)
	for _, enemy := range inst.members {
		if !botVisibleEnemyMemberLocked(inst, team, enemy, now, visionSources) {
			continue
		}
		if brain, ok := inst.bots[enemy.objID]; ok && brain != nil && brain.retreating {
			continue
		}
		x, y := enemy.posAtLocked(float32(now))
		score := float64(botEnemyRolePriority(enemy.huntState.av.Type)) * 20
		score += (1 - botHPFrac(enemy.huntState, now)) * 60
		// A visible high-kill enemy is a live macro threat even at full HP. Give
		// the shared focus target enough weight to make nearby bots converge on
		// the carry instead of feeding it isolated lane duels.
		score += math.Min(float64(enemy.huntState.frags), 20) * 8
		score -= math.Min(float64(enemy.huntState.deaths), 10) * 2
		for _, ally := range inst.members {
			if ally == nil || ally.huntState == nil || ally.playerTeam() != team || ally.huntState.deadUntil > 0 {
				continue
			}
			ax, ay := ally.posAtLocked(float32(now))
			if dist2(ax, ay, x, y) <= float32(botFightRadius*botFightRadius) {
				score += 12
				if ally.huntState.pvpTarget == enemy.objID || ally.huntState.attackTarget == enemy.objID ||
					enemy.huntState.pvpTarget == ally.objID || enemy.huntState.attackTarget == ally.objID {
					// Peeling an active attacker is more valuable than the role label of
					// an otherwise attractive target. Without this large state-based
					// bonus, a support or fed carry could win the shared focus vote while
					// another enemy was already killing an ally in the same skirmish.
					score += 120
					score += (1 - botHPFrac(ally.huntState, now)) * 100
				}
			}
		}
		cs = append(cs, candidate{c: enemy, score: score})
	}
	sort.Slice(cs, func(i, j int) bool {
		if cs[i].score != cs[j].score {
			return cs[i].score > cs[j].score
		}
		return cs[i].c.objID < cs[j].c.objID
	})
	if len(cs) == 0 {
		return nil
	}
	return cs[0].c
}

func botEnemyRolePriority(typ int32) int {
	switch typ {
	case gamedata.AvatarTypeSupport:
		return 4
	case gamedata.AvatarTypeMage:
		return 3
	case gamedata.AvatarTypeKiller:
		return 2
	default:
		return 1
	}
}

// botTacticalRiskValue is a small deterministic comparison used by tests and
// the tactical director: objective value must beat travel/risk cost before a
// bot commits to a non-local action.
func botTacticalRiskValue(objectiveValue, travelCost, dangerCost float64) float64 {
	return objectiveValue - travelCost - dangerCost
}
