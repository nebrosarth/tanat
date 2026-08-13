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
		// hand-off. FarmLane is the actual movement contract, including when a
		// duplicate-baseline spare temporarily covers another lane. Reading Lane
		// here discarded that hand-off and left the bot walking toward its authored
		// line while telemetry showed the replacement on the uncovered wave.
		if b.macroAssignment.Role == "lane_cover" && b.macroAssignment.Reason == "baseline_lane_coverage" &&
			b.macroAssignment.FarmLane >= 0 {
			return b.macroAssignment.FarmLane
		}
		if b.macroAssignment.Mode == botMacroBase {
			// A defender remains strategically anchored to the threatened
			// structure, but it is still a farm owner on its authored line. The
			// orchestrator writes FarmLane for that dual role; preferring Lane here
			// silently erased the authored lane and starved it on every base plan.
			if b.macroAssignment.FarmLane >= 0 {
				return b.macroAssignment.FarmLane
			}
			if b.macroAssignment.Lane >= 0 {
				return b.macroAssignment.Lane
			}
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
	if !botAIProfileForBrain(b).UsesFarmDebt() {
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
	if !botAIProfileForBrain(b).UsesFarmRotation() {
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
				(!requireCover || (!s.botCreepFightRiskyLocked(c, cx, cy) && !s.botEnemyStructureDangerLocked(c, committed.x, committed.y))) &&
				(b.phase != botPhaseGroup || s.botPushObjectiveHasPriorityLocked(b, now) || !s.botEnemyStructureDangerLocked(c, committed.x, committed.y)) {
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
			// A grouped bot may only enter an enemy structure's danger envelope
			// when the orchestrator has made the named objective the current
			// conversion priority. A generic group-farm query must not turn a
			// dying wave into an unplanned tower dive.
			if b.phase == botPhaseGroup && !s.botPushObjectiveHasPriorityLocked(b, now) &&
				s.botEnemyStructureDangerLocked(c, m.x, m.y) {
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

// botFarmSafeAnchorLocked returns the rear edge of the visible wave's XP
// envelope. XP is proximity-based, so a healthy bot does not need to stand in
// the creep pack or spend HP for a last hit. The direction is derived from the
// bot's home position, which keeps the anchor behind the wave for either side
// without encoding a lane or match-time exception.
func (s *Server) botFarmSafeAnchorLocked(b *botBrain, now float64) (float32, float32, bool) {
	centerX, centerY, ok := s.botFarmCoveragePointLocked(b, now)
	if !ok || b == nil || b.c == nil {
		return 0, 0, false
	}
	homeX, homeY := botHomeLocked(b.c)
	ax, ay := float64(homeX-centerX), float64(homeY-centerY)
	distance := math.Hypot(ax, ay)
	if distance < 0.01 {
		return centerX, centerY, true
	}
	// The arithmetic centre is not enough: a spread wave can put its nearest
	// creep inside aggro while its dying edge is already outside XP range. Build
	// the same local visible cluster and score candidate points on the home-side
	// ray. A candidate is preferred when it covers more of the wave, then when
	// it leaves the largest clearance from the nearest creep. This is a live
	// geometry decision, not a timer or an opening-game exception.
	lane := botFarmLaneLocked(b)
	sources := dotaTeamVisionSourcesLocked(b.c.inst, b.c.playerTeam(), now)
	cluster := make([]*mobState, 0, 4)
	for _, mob := range botSortedMobs(b.c.inst) {
		if mob == nil || mob.dead || mob.structure || !mob.enemyOf(b.c.playerTeam()) ||
			!botTeleportLaneCreep(mob) || botLaneForCreep(b.c, mob) != lane ||
			!botVisibleEnemyMobLocked(b.c.playerTeam(), mob, sources) {
			continue
		}
		if math.Hypot(float64(mob.x-centerX), float64(mob.y-centerY)) <= 64 {
			cluster = append(cluster, mob)
		}
	}
	if len(cluster) == 0 {
		anchorRadius := dotaXPShareRadius * 0.80
		return centerX + float32(ax/distance*anchorRadius), centerY + float32(ay/distance*anchorRadius), true
	}
	homeDX, homeDY := ax/distance, ay/distance
	const candidateStep = 0.5
	minClearance := dotaCreepAggro + 1.0
	type anchorCandidate struct {
		x, y      float32
		covered   int
		nearest   float64
		clearance float64
		fullySafe bool
	}
	var bestSafe, bestFallback anchorCandidate
	bestSafeSet, bestFallbackSet := false, false
	for radius := dotaCreepAggro; radius <= dotaXPShareRadius; radius += candidateStep {
		candidate := anchorCandidate{
			x:         centerX + float32(homeDX*radius),
			y:         centerY + float32(homeDY*radius),
			nearest:   math.Inf(1),
			clearance: math.Inf(1),
		}
		for _, mob := range cluster {
			d := math.Hypot(float64(candidate.x-mob.x), float64(candidate.y-mob.y))
			if d <= dotaXPShareRadius {
				candidate.covered++
			}
			if d < candidate.nearest {
				candidate.nearest = d
			}
			if d < candidate.clearance {
				candidate.clearance = d
			}
		}
		candidate.fullySafe = candidate.clearance >= minClearance
		better := func(left, right anchorCandidate) bool {
			if left.covered != right.covered {
				return left.covered > right.covered
			}
			if left.clearance != right.clearance {
				return left.clearance > right.clearance
			}
			return left.nearest > right.nearest
		}
		if candidate.covered > 0 && (!bestFallbackSet || better(candidate, bestFallback)) {
			bestFallback, bestFallbackSet = candidate, true
		}
		if candidate.fullySafe && candidate.covered > 0 && (!bestSafeSet || better(candidate, bestSafe)) {
			bestSafe, bestSafeSet = candidate, true
		}
	}
	if bestSafeSet {
		return bestSafe.x, bestSafe.y, true
	}
	if bestFallbackSet {
		return bestFallback.x, bestFallback.y, true
	}
	anchorRadius := dotaXPShareRadius * 0.80
	return centerX + float32(homeDX*anchorRadius), centerY + float32(homeDY*anchorRadius), true
}

// botFarmTargetSafePointLocked keeps the selected creep inside the XP envelope
// when the wave is spread farther than the rear-edge anchor can cover. The
// ordinary anchor remains the preferred point because it covers more of the
// wave, but a target-specific rear point prevents the last creep on the edge
// from dying a few units outside the reward radius. It is derived from the
// target and the bot's home direction, so it is safe geometry rather than a
// match-time or lane-specific exception.
func (s *Server) botFarmTargetSafePointLocked(b *botBrain, target *mobState, now float64) (float32, float32, bool) {
	if b == nil || b.c == nil || target == nil || target.dead {
		return 0, 0, false
	}
	if ax, ay, ok := s.botFarmSafeAnchorLocked(b, now); ok &&
		math.Hypot(float64(ax-target.x), float64(ay-target.y)) <= dotaXPShareRadius {
		return ax, ay, true
	}
	homeX, homeY := botHomeLocked(b.c)
	dx, dy := float64(homeX-target.x), float64(homeY-target.y)
	distance := math.Hypot(dx, dy)
	if distance < 0.01 {
		cx, cy := b.c.posAtLocked(float32(now))
		dx, dy = float64(cx-target.x), float64(cy-target.y)
		distance = math.Hypot(dx, dy)
	}
	if distance < 0.01 {
		return 0, 0, false
	}
	// Stay inside the authoritative XP radius while retaining a small margin
	// from the selected creep. The ordinary wave anchor remains farther back
	// and safer when coverage already exists.
	radius := dotaXPShareRadius * 0.82
	x := target.x + float32(dx/distance*radius)
	y := target.y + float32(dy/distance*radius)
	if s.botEnemyStructureDangerLocked(b.c, x, y) {
		return 0, 0, false
	}
	return x, y, true
}

// botFarmTargetWeakLocked is the only condition that permits a farm bot to
// close the last metres to a creep. XP is proximity-based and gold is already
// abundant for bots, so a healthy laner should use the rear XP anchor for a
// live wave instead of walking into every creep's attack envelope.
func botFarmTargetWeakLocked(target *mobState) bool {
	if target == nil || target.dead {
		return false
	}
	// mob.Health is the authored combat maximum. Prefer it over the mutable
	// runtime maxHP field: fixtures and some summoned creep variants initialize
	// maxHP from the current snapshot, which would make a 1-HP creep look full.
	// maxHP remains the fallback for custom mobs without authored health.
	if target.mob.Health > 0 {
		return target.hp <= target.mob.Health*0.20
	}
	return target.maxHP > 0 && target.hp/target.maxHP <= 0.20
}

// botFarmMayAttackCreepLocked allows an ordinary farm owner to finish a creep
// that is already inside attack range. This is wave maintenance, not a gold
// chase: callers only use it after the creep is already weak and inside the
// authoritative XP radius, so the bot never enters the pack just to compete
// for a last hit. Proximity XP remains the primary farm objective.
func (s *Server) botFarmMayAttackCreepLocked(b *botBrain, now float64) bool {
	if b == nil || !b.macroAssignment.FarmLaneSet {
		return true
	}
	// A healthy owner may clear the wave from its current XP position. Once it
	// is already losing a meaningful amount of HP, stop the attack order and
	// preserve the body for XP coverage instead of chasing a gold reward.
	return botHPFrac(b.c.huntState, now) >= botPredictiveRetreatHPFrac
}

func (s *Server) botFarmMayAttackTargetLocked(b *botBrain, target *mobState, now float64) bool {
	if botUsesAI30(b) {
		// Friendly creep cover is a requirement for walking INTO a wave, not for
		// firing at a creep that is already safely in reach.  Treating it as an
		// attack veto made an AI-30 laner become a spectator the instant its own
		// front creep died: it held its XP position beside hostile creeps but would
		// neither thin them nor defend itself.  The caller still checks real attack
		// reach; retain the hero/structure safety gates here so this exception never
		// turns into an uncovered approach or a tower dive.
		if b != nil && b.c != nil && b.c.huntState != nil && target != nil && !target.dead &&
			botHPFrac(b.c.huntState, now) >= botAI30FarmMinHP &&
			!s.botEnemyStructureDangerLocked(b.c, target.x, target.y) &&
			botNearbyEnemyHeroPressureLocked(b, now) == 0 {
			return true
		}
		_, _, safe := s.botAI30FarmAttackPointLocked(b, target, now)
		return safe
	}
	return s.botFarmMayAttackCreepLocked(b, now)
}

// botMoveToFarmTargetLocked keeps every ordinary farm route behind the same
// safety invariant as botHoldFarmXPLocked. Several macro branches need to
// approach a live wave; allowing one of them to move directly to the target
// would silently undo the XP-preserving behavior on the next tick.
func (s *Server) botMoveToFarmTargetLocked(b *botBrain, target *mobState, now float64) bool {
	if b == nil || b.c == nil || b.c.huntState == nil || target == nil || target.dead {
		return false
	}
	c := b.c
	cx, cy := c.posAtLocked(float32(now))
	distance := math.Hypot(float64(target.x-cx), float64(target.y-cy))
	attackReach := c.huntState.effAttackRangeLocked(now) + c.huntState.av.Radius() + target.mob.Radius()
	b.farmTarget = target.id
	if distance <= attackReach && distance <= dotaXPShareRadius &&
		s.botFarmMayAttackTargetLocked(b, target, now) {
		b.farmDecision = "wave_clear"
		s.startAttackLocked(c, target)
		return true
	}
	// AI-30 converts safe allied creep cover into active wave damage. It walks
	// only to a home-side attack point that still covers proximity XP; without
	// cover or under pressure it falls through to AI-20's rear anchor.
	if x, y, ok := s.botAI30FarmAttackPointLocked(b, target, now); ok {
		b.farmDecision = "active_wave_clear"
		if math.Hypot(float64(x-cx), float64(y-cy)) > 1.0 {
			s.botMoveTowardLocked(b, x, y, now)
		} else {
			s.startAttackLocked(c, target)
		}
		return true
	}
	if ax, ay, ok := s.botFarmTargetSafePointLocked(b, target, now); ok &&
		!s.botEnemyStructureDangerLocked(c, ax, ay) {
		b.farmDecision = "wave_anchor"
		if math.Hypot(float64(ax-cx), float64(ay-cy)) > 1.5 {
			s.botMoveTowardLocked(b, ax, ay, now)
		} else {
			c.stopMovementLocked(s, now)
		}
		return true
	}
	// If the geometry cannot produce an anchor (for example, the map has no
	// authored lane point), keep the XP invariant rather than chasing a target
	// outside the reward radius. A caller may choose its strategic fallback.
	if distance <= dotaXPShareRadius {
		b.farmDecision = "wave_anchor"
		c.stopMovementLocked(s, now)
		return true
	}
	return false
}

// botMoveToFarmCoverageLocked is the target-independent form used while the
// director has a visible wave cluster but has not selected one creep. It still
// uses the rear XP anchor; moving to the arithmetic centre can put the body in
// the middle of the pack and reintroduce avoidable creep damage.
func (s *Server) botMoveToFarmCoverageLocked(b *botBrain, now float64) bool {
	if b == nil || b.c == nil || b.c.huntState == nil {
		return false
	}
	ax, ay, ok := s.botFarmSafeAnchorLocked(b, now)
	if !ok || s.botEnemyStructureDangerLocked(b.c, ax, ay) {
		return false
	}
	cx, cy := b.c.posAtLocked(float32(now))
	b.farmDecision = "wave_anchor"
	if math.Hypot(float64(ax-cx), float64(ay-cy)) > 1.5 {
		s.botMoveTowardLocked(b, ax, ay, now)
	} else {
		b.c.stopMovementLocked(s, now)
	}
	return true
}

// botHoldFarmXPLocked anchors a laner while a live, visible enemy creep is
// already inside the proximity-XP radius. Chasing the next creep at that point
// is strategically wrong: the current wave member can die behind the bot and
// become an XP miss. A healthy bot holds the rear edge of the wave instead of
// standing in the creep pack; only a genuinely weak creep is attacked. Gold is
// not a reason to accept avoidable lane damage.
func (s *Server) botHoldFarmXPLocked(b *botBrain, now float64) bool {
	if b == nil || b.c == nil || b.c.inst == nil || b.c.huntState == nil || b.retreating ||
		b.c.huntState.deadUntil > 0 {
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
	// Being inside the XP radius of one creep does not cover a stretched wave.
	// Resolve the uncovered edge before the nearest-creep hold below; otherwise
	// the bot keeps anchoring on the already-covered member while the outer
	// member dies without a proximity recipient. The target-specific safe point
	// still keeps the move outside the creep pack and returns false when the
	// edge is not locally reachable.
	if b.macroAssignment.FarmLaneSet {
		if urgent := s.botUrgentFarmCoverageTargetLocked(b, now); urgent != nil {
			if s.botMoveToFarmTargetLocked(b, urgent, now) {
				return true
			}
		}
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
	// Once the orchestrator has declared a visible lane uncovered, the owner
	// is a proximity-XP sensor first. Even a valid last hit can pull the body
	// into the creep pack and make the following wave lose its only recipient;
	// the abundant bot economy does not justify that risk. Keep the narrow
	// attack behavior for isolated/unit tests and ordinary non-coverage play.
	if nearestDistance <= attackReach && s.botFarmMayAttackTargetLocked(b, nearest, now) {
		b.farmDecision = "wave_clear"
		s.startAttackLocked(c, nearest)
		return true
	}
	// Being inside XP range of one member of a stretched wave is not enough:
	// the next member may die on the far edge before the ordinary centre anchor
	// catches up. Resolve that uncovered edge after a nearby weak creep has had
	// its legitimate clear action, using the same safe target point as every
	// other farm route.
	if b.farmTarget != 0 {
		if urgent := s.botUrgentFarmCoverageTargetLocked(b, now); urgent != nil {
			if s.botMoveToFarmTargetLocked(b, urgent, now) {
				return true
			}
		}
	}
	b.farmDecision = "wave_anchor"
	if ax, ay, ok := s.botFarmSafeAnchorLocked(b, now); ok {
		if math.Hypot(float64(ax-cx), float64(ay-cy)) > 1.5 {
			s.botMoveTowardLocked(b, ax, ay, now)
		} else {
			c.stopMovementLocked(s, now)
		}
	} else {
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
	if wave == nil || nearest > dotaXPShareRadius || s.botEnemyStructureDangerLocked(c, wave.x, wave.y) {
		return false
	}
	// Creep pressure alone does not invalidate the last XP body when the
	// geometry solver can place it at a safe rear-edge point. The normal farm
	// tick will issue that move immediately; abandoning the lane here would make
	// two owners retreat together and leave the next creep without a recipient.
	// Hero pressure remains a hard veto because the safe XP anchor cannot protect
	// against a visible avatar.
	if s.botIncomingPressureLocked(b, now) && botNearbyEnemyHeroPressureLocked(b, now) > 0 {
		return false
	}
	// Only the last local body gets this exception. If another living ally is
	// already close enough, the current bot may safely disengage and preserve HP.
	for _, ally := range botLivingAllies(c) {
		if ally == nil || ally == c || ally.huntState == nil || ally.huntState.deadUntil > 0 {
			continue
		}
		if allyBrain, ok := c.inst.bots[ally.objID]; ok && allyBrain != nil && allyBrain.retreating {
			// A retreating body may still be inside the XP ring during its
			// disengage, but it is no longer a reliable owner for the lane.
			continue
		}
		ax, ay := ally.posAtLocked(float32(now))
		if math.Hypot(float64(ax-wave.x), float64(ay-wave.y)) <= dotaXPShareRadius*1.25 {
			return false
		}
	}
	return true
}

// botFarmCoveragePressureLocked detects the state in which a lane owner is
// taking creep pressure and is currently the only body that can receive the
// wave's proximity XP. It is deliberately visible-wave bounded and does not
// count hidden creeps or distant optional objectives as a reason to disengage.
func (s *Server) botFarmCoveragePressureLocked(b *botBrain, now float64) bool {
	if b == nil || b.c == nil || b.c.inst == nil || b.c.huntState == nil ||
		botNearbyEnemyHeroPressureLocked(b, now) > 0 {
		return false
	}
	cx, cy := b.c.posAtLocked(float32(now))
	if s.botEnemyStructureDangerLocked(b.c, cx, cy) {
		return false
	}
	target := s.botFarmShadowTargetLocked(b, now)
	return target != nil && s.botIncomingPressureLocked(b, now)
}

// botFarmShadowTargetLocked is the hand-off variant of
// botUrgentFarmCoverageTargetLocked: the retreating bot itself is excluded
// from the coverage test, because it may be inside XP range now but needs to
// move to the outer edge before creep pressure turns into a death.
func (s *Server) botFarmShadowTargetLocked(b *botBrain, now float64) *mobState {
	if b == nil || b.c == nil || b.c.inst == nil || b.c.huntState == nil {
		return nil
	}
	c := b.c
	lane := botFarmLaneLocked(b)
	sources := dotaTeamVisionSourcesLocked(c.inst, c.playerTeam(), now)
	cx, cy := c.posAtLocked(float32(now))
	bestDistance := math.Inf(1)
	var best *mobState
	for _, mob := range botSortedMobs(c.inst) {
		if mob == nil || mob.dead || mob.structure || !mob.enemyOf(c.playerTeam()) ||
			!botTeleportLaneCreep(mob) || botLaneForCreep(c, mob) != lane ||
			!botVisibleEnemyMobLocked(c.playerTeam(), mob, sources) {
			continue
		}
		covered := false
		for _, ally := range botLivingAllies(c) {
			if ally == nil || ally == c || ally.huntState == nil || ally.huntState.deadUntil > 0 {
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

// botFarmXPShadowPointLocked returns a safe point just inside the proximity-XP
// radius of an uncovered visible creep. The point is chosen away from the
// creep, so a pressured laner can preserve the XP event without continuing to
// stand in melee range. Structure and hero danger still veto the shadow route.
func (s *Server) botFarmXPShadowPointLocked(b *botBrain, now float64) (float32, float32, *mobState, bool) {
	if b == nil || b.c == nil || b.c.inst == nil || b.c.huntState == nil ||
		botNearbyEnemyHeroPressureLocked(b, now) > 0 {
		return 0, 0, nil, false
	}
	target := s.botFarmShadowTargetLocked(b, now)
	if target == nil || s.botEnemyStructureDangerLocked(b.c, target.x, target.y) {
		return 0, 0, nil, false
	}
	// A critically injured bot normally must go to the fountain. The one safe
	// exception is a wave already inside a living allied gun/tower's firing
	// envelope: the structure is the protection, and the bot only preserves
	// proximity XP from the rear edge instead of taking another fight.
	if botHPFrac(b.c.huntState, now) <= botRetreatHPFrac &&
		!s.botFarmWaveUnderFriendlyDefenseLocked(b, target) {
		return 0, 0, nil, false
	}
	cx, cy := b.c.posAtLocked(float32(now))
	dx, dy := float64(cx-target.x), float64(cy-target.y)
	distance := math.Hypot(dx, dy)
	if distance < 0.01 {
		// A coincident body has no useful direction. Move toward the lane's
		// authored centre, which keeps the shadow deterministic and away from
		// the creep pack in normal lane geometry.
		lane := botFarmLaneLocked(b)
		if lane >= 0 && lane < len(b.c.inst.dota.m.Lanes) {
			point := b.c.inst.dota.m.Lanes[lane][len(b.c.inst.dota.m.Lanes[lane])/2]
			dx, dy = point.X-float64(target.x), point.Y-float64(target.y)
			distance = math.Hypot(dx, dy)
		}
	}
	if distance < 0.01 {
		return 0, 0, nil, false
	}
	// Shadow the centre of the visible wave at the outer edge of the
	// authoritative XP radius. A retreating bot should cover the whole local
	// cluster without returning to melee range; using the nearest creep alone
	// made it miss a neighbour that died a few units away. The larger 85% offset
	// keeps the body outside the normal creep attack envelope while retaining
	// the XP event. Both the centre and the offset are geometry-derived.
	centerX, centerY, centerOK := s.botFarmCoveragePointLocked(b, now)
	if !centerOK {
		centerX, centerY = target.x, target.y
	}
	awayX, awayY := float64(cx-centerX), float64(cy-centerY)
	awayDistance := math.Hypot(awayX, awayY)
	if awayDistance < 0.01 {
		awayX, awayY = dx, dy
		awayDistance = math.Hypot(awayX, awayY)
	}
	if awayDistance < 0.01 {
		return 0, 0, nil, false
	}
	shadowRadius := dotaXPShareRadius * 0.80
	shadowX := centerX + float32(awayX/awayDistance*shadowRadius)
	shadowY := centerY + float32(awayY/awayDistance*shadowRadius)
	// The centre shadow is optimal only while it covers the selected edge
	// creep. If the wave has stretched, follow the target's rear edge instead;
	// otherwise a retreating owner can remain visually close to the wave while
	// still missing the exact creep that dies next.
	if math.Hypot(float64(shadowX-target.x), float64(shadowY-target.y)) > dotaXPShareRadius {
		if targetX, targetY, targetOK := s.botFarmTargetSafePointLocked(b, target, now); targetOK {
			shadowX, shadowY = targetX, targetY
		}
	}
	if s.botEnemyStructureDangerLocked(b.c, shadowX, shadowY) {
		return 0, 0, nil, false
	}
	return shadowX, shadowY, target, true
}

// botFarmWaveUnderFriendlyDefenseLocked reports whether a visible farm target
// is already inside a living allied shooting structure's attack envelope. It
// is intentionally map-state and geometry based: no match-clock exception and
// no blind movement through fog are introduced.
func (s *Server) botFarmWaveUnderFriendlyDefenseLocked(b *botBrain, target *mobState) bool {
	if b == nil || b.c == nil || b.c.inst == nil || target == nil || target.dead ||
		!target.enemyOf(b.c.playerTeam()) {
		return false
	}
	for _, structure := range botSortedMobs(b.c.inst) {
		if structure == nil || structure.dead || !structure.structure || structure.team != b.c.playerTeam() ||
			structure.mob.AttackRange <= 0 {
			continue
		}
		attackRange := structure.mob.AttackRange + structure.mob.Radius() + target.mob.Radius()
		if math.Hypot(float64(target.x-structure.x), float64(target.y-structure.y)) <= attackRange {
			return true
		}
	}
	return false
}

// botFarmXPShadowTickLocked executes the XP-shadow hand-off for a retreating
// bot. It lives outside the phase ticks because a retreat latch intentionally
// bypasses ordinary lane/group combat logic.
func (s *Server) botFarmXPShadowTickLocked(b *botBrain, now float64) bool {
	if b == nil || b.c == nil || b.c.huntState == nil {
		return false
	}
	shadowX, shadowY, target, ok := s.botFarmXPShadowPointLocked(b, now)
	if !ok {
		return false
	}
	b.engageTarget = 0
	b.farmTarget = target.id
	b.farmDecision = "xp_shadow"
	s.stopPvpAttackLocked(b.c, false)
	s.stopAttackLocked(b.c, false)
	s.botMoveTowardLocked(b, shadowX, shadowY, now)
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
	bestHPFrac := math.Inf(1)
	bestDistance := math.Inf(-1)
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
		hpFrac := 1.0
		if mob.maxHP > 0 {
			hpFrac = float64(mob.hp / mob.maxHP)
		} else if mob.mob.Health > 0 {
			hpFrac = float64(mob.hp / mob.mob.Health)
		}
		// A creep that is already being worn down is the next likely XP event;
		// resolve it before an untouched creep from a newer/farther wave. When
		// health is tied, prefer the farther uncovered edge so a sticky target
		// cannot keep the bot parked on the already-covered side of the lane.
		if hpFrac < bestHPFrac-0.01 ||
			(math.Abs(hpFrac-bestHPFrac) <= 0.01 && (distance > bestDistance || (distance == bestDistance && (best == nil || mob.id < best.id)))) {
			best, bestHPFrac, bestDistance = mob, hpFrac, distance
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
