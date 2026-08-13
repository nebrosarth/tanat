package battleserver

// Laning-phase decisions: where to stand, what to hit, when to nuke a creep wave. This is
// deliberately the bulk of a bot's early game, matching real Dota: most of a match is
// spent farming/trading on a lane, not teamfighting (bot_tactics.go).

import (
	"math"

	"tanatserver/internal/gamedata"
)

// botLaneEngageRadius is how far from itself a laning bot will pick a fight (creep or,
// with friendly creep cover, a structure) -- roughly a creep's own aggro pull
// (dotaCreepAggro=13), so a bot holds its lane instead of wandering into the next one.
const botLaneEngageRadius = 14.0

// A farm approach may acquire a live lane creep before it is attackable. The
// orchestrator has already selected the lane, so this is a wider local response
// radius rather than permission to rotate across the map. The wider radius is
// necessary after a retreat or respawn: waiting at a static lane point while a
// live wave dies outside the old 42u ring was a direct source of lost XP.
const botLaneApproachRadius = 180.0

// botAbilityReadyLocked reports whether slot is learned, off cooldown and affordable.
func (s *Server) botAbilityReadyLocked(hs *huntState, slot int, now float64) bool {
	level := int(hs.skillLevel[slot-1])
	if level < 1 || now < hs.cooldownUntil[slot-1] {
		return false
	}
	def := hs.skillDef(slot)
	if def.Type != "ACTIVE" {
		return false
	}
	cost := skillManaCost(float64(def.ManaCost[level-1]))
	return hs.mana >= cost
}

// botFindLastHitLocked returns an enemy creep already in attack range that will die to
// this bot's next swing. It remains as a small, state-inspection helper for callers and
// tests, but it is no longer a priority in the live farm decision: bots are paid to gain
// XP by staying near the wave, not to compete for gold.
func (s *Server) botFindLastHitLocked(b *botBrain, now float64) *mobState {
	c, hs := b.c, b.c.huntState
	cx, cy := c.posAtLocked(float32(now))
	reach := hs.effAttackRangeLocked(now)
	if reach <= 0 {
		reach = 2.2 // melee fallback, matches dotaMeleeReach's scale
	}
	reach += hs.av.Radius()
	myDmg := hs.baseAttackLocked(now)
	if myDmg <= 0 {
		return nil
	}
	visionSources := dotaTeamVisionSourcesLocked(c.inst, c.playerTeam(), now)
	var best *mobState
	bestHP := math.Inf(1)
	for _, m := range botSortedMobs(c.inst) {
		if m.dead || m.structure || !m.enemyOf(c.playerTeam()) || !botVisibleEnemyMobLocked(c.playerTeam(), m, visionSources) {
			continue
		}
		if botFarmTargetClaimedByCloserAllyLocked(b, m, now) {
			continue
		}
		d := math.Hypot(float64(m.x-cx), float64(m.y-cy)) - m.mob.Radius()
		if d > reach || m.hp > myDmg*1.15 {
			continue
		}
		if m.hp < bestHP || (m.hp == bestHP && (best == nil || m.id < best.id)) {
			bestHP, best = m.hp, m
		}
	}
	return best
}

// botCreepFightRiskyLocked reports whether fighting a creep from the bot's current
// position (cx,cy) means standing in a living enemy structure's own attack range without
// one of our own creeps actually soaking its aggro. A structure fires on whichever enemy
// it finds nearest in range (dotaAcquireTargetLocked) -- "one of our creeps is somewhere
// within botLaneEngageRadius" (ownCreepNearby, the requireCover gate above) is NOT the
// same claim as "our creep is what the structure is actually shooting", so a solo laner
// with a loosely-nearby wave could still be the structure's nearest target and get shot
// down while focused entirely on a creep, never having weighed the structure's own threat
// -- the same gap botEnemyStructureDangerLocked (bot_combat.go) already closes for a hero
// chase, applied here to the plain laning fight-picker instead.
func (s *Server) botCreepFightRiskyLocked(c *conn, cx, cy float32) bool {
	for _, m := range botSortedMobs(c.inst) {
		if m.dead || !m.structure || !m.enemyOf(c.playerTeam()) {
			continue
		}
		rng := float32(m.mob.AttackRange)
		if rng <= 0 {
			continue // a non-shooting structure (e.g. a spring/generator) poses no threat
		}
		myD := dist2(cx, cy, m.x, m.y)
		if myD > rng*rng {
			continue
		}
		soaked := false
		for _, o := range botSortedMobs(c.inst) {
			if o.dead || o.structure || o.team != c.playerTeam() {
				continue
			}
			if dist2(o.x, o.y, m.x, m.y) < myD {
				soaked = true
				break
			}
		}
		if !soaked {
			return true
		}
	}
	return false
}

// botFindLaneTargetLocked is the laning attack-target picker: first the XP-aware live-wave
// director, then the nearest enemy creep worth trading with. It deliberately never falls
// back to an enemy structure. Structure attacks belong to botMacroObjectiveTickLocked,
// where the team orchestrator has named the exact objective and established a conversion
// window. A farm/cover/recovery bot must not turn an empty local wave into an unsolicited
// tower dive: that damages the bot, causes a retreat, and abandons the next XP radius.
func (s *Server) botFindLaneTargetLocked(b *botBrain, now float64, radius float64, requireCover bool) *mobState {
	if target := s.botFarmTargetLocked(b, now, radius, requireCover); target != nil {
		return target
	}
	c := b.c
	cx, cy := c.posAtLocked(float32(now))
	farmLane := botFarmLaneLocked(b)
	visionSources := dotaTeamVisionSourcesLocked(c.inst, c.playerTeam(), now)

	var bestCreep *mobState
	bestCreepD := math.Inf(1)
	for _, m := range botSortedMobs(c.inst) {
		if m.dead {
			continue
		}
		d := math.Hypot(float64(m.x-cx), float64(m.y-cy))
		if m.structure || !m.enemyOf(c.playerTeam()) || d > radius ||
			!botVisibleEnemyMobLocked(c.playerTeam(), m, visionSources) {
			continue
		}
		if !m.structure && botLaneForCreep(c, m) >= 0 && botLaneForCreep(c, m) != farmLane {
			continue
		}
		if d < bestCreepD || (d == bestCreepD && (bestCreep == nil || m.id < bestCreep.id)) {
			bestCreepD, bestCreep = d, m
		}
	}
	if bestCreep != nil {
		if requireCover && s.botCreepFightRiskyLocked(c, cx, cy) {
			return nil // a live enemy structure is shooting from where we're standing, and
			// none of our own creeps are between us and it -- see botCreepFightRiskyLocked.
		}
		return bestCreep
	}
	_ = requireCover // kept in the signature for the creep-safety policy at the call sites
	return nil
}

func (s *Server) botFarmApproachTargetLocked(b *botBrain, now float64) *mobState {
	target := s.botFarmTargetLocked(b, now, botLaneApproachRadius, true)
	if target == nil {
		return nil
	}
	// The farm director may see a wave that is close enough to approach but is
	// already inside an enemy structure's no-dive ring. Keep the structure safety
	// invariant and fall back to the ordinary safe lane point in that case.
	if s.botEnemyStructureDangerLocked(b.c, target.x, target.y) {
		return nil
	}
	return target
}

// botConsiderWaveClearAbilityLocked casts the first ready, affordable AoE damage ability
// at the enemy creep cluster it hits hardest, when at least 2 enemies are caught -- a
// simple but real "use abilities when advantageous" rule for the farming game, distinct
// from bot_combat.go's hero-fight ability logic.
func (s *Server) botConsiderWaveClearAbilityLocked(b *botBrain, now float64) bool {
	c, hs := b.c, b.c.huntState
	cx, cy := c.posAtLocked(float32(now))
	visionSources := dotaTeamVisionSourcesLocked(c.inst, c.playerTeam(), now)
	for slot := 1; slot <= 4; slot++ {
		if !s.botAbilityReadyLocked(hs, slot, now) {
			continue
		}
		def := hs.skillDef(slot)
		if def.AoERadius <= 0 || !botSkillHasOp(def, gamedata.OpDamage) {
			continue // single-target actives are saved for hero fights
		}
		dist := float64(def.Distance)
		if dist <= 0 {
			dist = 6
		}
		bestX, bestY, bestCount := float32(0), float32(0), 0
		bestID := int32(0)
		for _, m := range botSortedMobs(c.inst) {
			if m.dead || m.structure || !m.enemyOf(c.playerTeam()) || !botVisibleEnemyMobLocked(c.playerTeam(), m, visionSources) {
				continue
			}
			if botLaneForCreep(c, m) >= 0 && botLaneForCreep(c, m) != botFarmLaneLocked(b) {
				continue
			}
			if math.Hypot(float64(m.x-cx), float64(m.y-cy)) > dist {
				continue
			}
			n := 0
			for _, o := range botSortedMobs(c.inst) {
				if !o.dead && !o.structure && o.enemyOf(c.playerTeam()) &&
					botVisibleEnemyMobLocked(c.playerTeam(), o, visionSources) &&
					(botLaneForCreep(c, o) < 0 || botLaneForCreep(c, o) == botFarmLaneLocked(b)) &&
					math.Hypot(float64(o.x-m.x), float64(o.y-m.y)) <= float64(def.AoERadius) {
					n++
				}
			}
			if n > bestCount || (n == bestCount && n > 0 && (bestID == 0 || m.id < bestID)) {
				bestCount, bestX, bestY = n, m.x, m.y
				bestID = m.id
			}
		}
		if bestCount >= 2 {
			// Clearing a wave from outside the proximity radius can kill the
			// creeps without granting this bot XP. Move into the reward envelope
			// first; farm abilities are optional, XP coverage is not.
			if math.Hypot(float64(bestX-cx), float64(bestY-cy)) > dotaXPShareRadius {
				continue
			}
			b.farmDecision = "wave_clear_ability"
			b.farmTargetScore = float64(bestCount * 24)
			b.farmWaveClears++
			return s.startSkillOrderLocked(c, slot, 0, bestX, bestY, true)
		}
	}
	return false
}

// botHarassEngageRadius bounds how far outside a genuine hero-fight commitment a laning/
// roaming bot will still poke a ready single-target ability at a visible enemy hero --
// short enough to be a safe poke from the bot's own current position, not a step toward
// the fuller engagement botPickEngageTargetLocked gates on headcount/HP for.
const botHarassEngageRadius = 10.0

// botConsiderHarassAbilityLocked casts a ready single-target offensive ability (CC or
// damage; see botOffensiveOpPriority) at a nearby enemy hero encountered while laning or
// roaming, WITHOUT the full hero-fight commitment botCombatTickLocked/
// botPickEngageTargetLocked gates on (ally/enemy headcount, HP>50% when alone). Those
// single-target actives are deliberately excluded from botConsiderWaveClearAbilityLocked
// ("saved for hero fights"), but a genuine hero fight is rare -- most of a match is spent
// laning/farming (measured: one real hero fight in an entire 9-minute match), so a slot
// with a near-always-ready cooldown otherwise sits completely unused for the whole game.
// This is a low-commitment poke, not an engage: it never issues a move or attack order,
// only a cast, and skips entirely when the target is inside a living enemy structure's
// kill zone (no reason to fish for a trade there). Reports whether it acted.
func (s *Server) botConsiderHarassAbilityLocked(b *botBrain, now float64) bool {
	c, hs := b.c, b.c.huntState
	cx, cy := c.posAtLocked(float32(now))
	var target *conn
	bestD := math.Inf(1)
	for _, e := range botLivingEnemyHeroes(c, now) {
		ex, ey := e.posAtLocked(float32(now))
		if d := math.Hypot(float64(ex-cx), float64(ey-cy)); d <= botHarassEngageRadius && d < bestD {
			bestD, target = d, e
		}
	}
	if target == nil {
		return false
	}
	tx, ty := target.posAtLocked(float32(now))
	if s.botEnemyStructureDangerLocked(c, tx, ty) {
		return false
	}
	bestSlot, bestP := 0, -1
	for slot := 1; slot <= 4; slot++ {
		if !s.botAbilityReadyLocked(hs, slot, now) {
			continue
		}
		def := hs.skillDef(slot)
		if def.AoERadius > 0 {
			continue // AoE actives are botConsiderWaveClearAbilityLocked's job
		}
		dist := float64(def.Distance)
		if dist <= 0 {
			dist = 6
		}
		if bestD > dist {
			continue // enemy hero is outside this ability's own cast range
		}
		if p := botOffensiveOpPriority(def); p > bestP {
			bestP, bestSlot = p, slot
		}
	}
	if bestSlot == 0 || bestP <= 0 {
		return false
	}
	def := hs.skillDef(bestSlot)
	if def.Target == "" || def.Target == "SELF" {
		return s.startSkillOrderLocked(c, bestSlot, 0, 0, 0, false)
	}
	var targetID int32
	if def.Targeting == "TARGET" {
		targetID = target.objID
	}
	return s.startSkillOrderLocked(c, bestSlot, targetID, tx, ty, true)
}

// botSkillHasOp reports whether a skill's op list includes at least one op of kind k.
func botSkillHasOp(def gamedata.Skill, k gamedata.OpKind) bool {
	for _, op := range def.Ops {
		if op.Kind == k {
			return true
		}
	}
	return false
}

// botLanePoint is where a laning bot should stand: the front of its own team's creep wave
// on its assigned lane (the honest "where the lane currently stands" signal already
// driving the creep AI itself), or, before the first wave spawns, a fixed spot partway
// down the lane from home.
func (s *Server) botLanePoint(b *botBrain, now float64) (float32, float32) {
	c := b.c
	d := c.inst.dota
	if b.lane < 0 || b.lane >= len(d.m.Lanes) {
		return botHomeLocked(c)
	}
	lane := d.m.Lanes[b.lane]
	if len(lane) == 0 {
		return botHomeLocked(c)
	}
	if b.laneRedeployPointValid {
		if now >= b.laneRedeployUntil {
			b.laneRedeployPointValid = false
		} else if fx, fy, ok := botLaneFrontLocked(c, lane); !ok ||
			math.Hypot(float64(fx-b.laneRedeployPointX), float64(fy-b.laneRedeployPointY)) > 18.0 {
			return b.laneRedeployPointX, b.laneRedeployPointY
		} else {
			b.laneRedeployPointValid = false
		}
	}
	if fx, fy, ok := botLaneFrontLocked(c, lane); ok {
		return fx, fy
	}
	fwd := c.playerTeam() == dotaTeamHuman
	idx := 2
	if !fwd {
		idx = len(lane) - 1 - 2
	}
	if idx < 0 {
		idx = 0
	}
	if idx >= len(lane) {
		idx = len(lane) - 1
	}
	return float32(lane[idx].X), float32(lane[idx].Y)
}

// botFarmPrestagePointLocked puts a zero-XP laner at the authored lane's
// geometric rendezvous point while the opposing wave is outside vision.
// Waiting for the first enemy creep to become visible was too late on
// accelerated simulation: the creep could die while the cover bot was still
// 20-30 units away, just outside the authoritative 20-unit XP share radius.
// The rendezvous is shifted toward the bot's own side by a safe aggro margin.
// That leaves the body close enough to enter the reward envelope as the wave
// rounds the lane, while still giving the live-wave anchor room to pull it to
// the rear edge. The shift is map geometry plus the authoritative aggro
// distance, not a match clock or an unseen unit position. This only predicts a
// map waypoint; it does not
// select or attack an unseen enemy unit.
func (s *Server) botFarmPrestagePointLocked(b *botBrain, laneIndex int, _ float64) (float32, float32, bool) {
	if b == nil || b.c == nil || b.c.inst == nil || b.c.inst.dota == nil ||
		laneIndex < 0 || laneIndex >= len(b.c.inst.dota.m.Lanes) {
		return 0, 0, false
	}
	lane := b.c.inst.dota.m.Lanes[laneIndex]
	if len(lane) == 0 {
		return 0, 0, false
	}
	if len(lane) == 1 {
		return float32(lane[0].X), float32(lane[0].Y), true
	}
	total := 0.0
	for i := 0; i+1 < len(lane); i++ {
		total += math.Hypot(lane[i+1].X-lane[i].X, lane[i+1].Y-lane[i].Y)
	}
	if total <= 0.001 {
		return float32(lane[0].X), float32(lane[0].Y), true
	}
	remaining := total * 0.5
	// Stage just behind the first safe aggro boundary. Using the full outer
	// XP radius left a bot 20-30u behind a wave after the creeps rounded the
	// lane bend; the handoff then arrived after the first death. The aggro
	// distance is a map/economy invariant, not a match-time opening constant.
	// The meeting point must remain inside the reward envelope while the wave
	// is still hidden. A full aggro-width offset left the body behind a moving
	// wave by just over XP range at the first collision. Keep only a small
	// rear-side safety margin; once the creep is visible, the live safe-anchor
	// logic takes over and keeps the body outside aggro.
	prestageOffset := dotaCreepAggro * 0.35
	if b.c.playerTeam() == dotaTeamHuman {
		remaining -= prestageOffset
	} else {
		remaining += prestageOffset
	}
	if remaining < 0 {
		remaining = 0
	}
	if remaining > total {
		remaining = total
	}
	for i := 0; i+1 < len(lane); i++ {
		x, y := lane[i].X, lane[i].Y
		tx, ty := lane[i+1].X, lane[i+1].Y
		segment := math.Hypot(tx-x, ty-y)
		if segment <= 0.001 {
			continue
		}
		if remaining <= segment {
			fraction := remaining / segment
			return float32(x + (tx-x)*fraction), float32(y + (ty-y)*fraction), true
		}
		remaining -= segment
	}
	last := lane[len(lane)-1]
	return float32(last.X), float32(last.Y), true
}

// botLaneFrontLocked finds the most-advanced living creep of c's own side on `lane`
// (matched by the authored lane polyline) and returns its
// position -- "advanced" means highest laneIdx marching forward, lowest marching backward.
func botLaneFrontLocked(c *conn, lane []gamedata.Vec2) (float32, float32, bool) {
	team := c.playerTeam()
	fwd := team == dotaTeamHuman
	var best *mobState
	bestIdx := -1
	if !fwd {
		bestIdx = len(lane) + 1
	}
	for _, m := range c.inst.mobs {
		if m.dead || m.structure || m.team != team || len(m.lane) == 0 {
			continue
		}
		if !botSameDotaLane(m.lane, lane) {
			continue
		}
		if (fwd && (m.laneIdx > bestIdx || (m.laneIdx == bestIdx && (best == nil || m.id < best.id)))) ||
			(!fwd && (m.laneIdx < bestIdx || (m.laneIdx == bestIdx && (best == nil || m.id < best.id)))) {
			bestIdx, best = m.laneIdx, m
		}
	}
	if best == nil {
		return 0, 0, false
	}
	return best.x, best.y, true
}

// botLaneTickLocked is the default (early-game) decision pass: hold/retreat, clear waves
// with abilities when it pays off, trade, else walk to the lane.
func (s *Server) botLaneTickLocked(b *botBrain, now float64) {
	c, hs := b.c, b.c.huntState
	if s.botConsiderHealLocked(b, now) {
		return
	}
	if s.botShouldRetreatLocked(b, now) {
		hx, hy := s.botRetreatPointLocked(b, now)
		s.botMoveTowardLocked(b, hx, hy, now)
		return
	}
	if hs.attackTarget != 0 {
		if target := hs.mobs[hs.attackTarget]; target != nil && !target.structure &&
			target.enemyOf(c.playerTeam()) && !s.botFarmMayAttackCreepLocked(b, now) {
			s.stopAttackLocked(c, false)
		} else {
			return // hero/structure fight, or an isolated optional last hit
		}
	}
	if botFirstFarmWavePendingLocked(b, now) {
		// Pre-stage before the wave exists. Waiting at the fountain made the
		// first visible wave die before a walking cover bot could enter XP
		// range; the lane point is a safe authored waypoint, not a hidden target.
		lane := botFarmLaneLocked(b)
		if lane >= 0 {
			lx, ly, ok := s.botFarmPrestagePointLocked(b, lane, now)
			if ok && !s.botEnemyStructureDangerLocked(c, lx, ly) {
				s.botMoveTowardLocked(b, lx, ly, now)
				return
			}
		}
		lx, ly := s.botLanePoint(b, now)
		s.botMoveTowardLocked(b, lx, ly, now)
		return
	}
	// XP comes from the dead wave, not from gold ownership. Clear multiple creeps
	// before committing to one already-low creep, then attack the director's live
	// wave target. This keeps the bot inside the XP economy instead of optimizing
	// a reward it already receives for free.
	if s.botConsiderWaveClearAbilityLocked(b, now) {
		return
	}
	if s.botHoldFarmXPLocked(b, now) {
		return
	}
	target := s.botFarmTargetLocked(b, now, botLaneEngageRadius, false)
	if px, py, ok := s.botFarmTargetSafePointLocked(b, target, now); ok {
		cx, cy := c.posAtLocked(float32(now))
		if math.Hypot(float64(px-cx), float64(py-cy)) > dotaXPShareRadius*0.5 &&
			!s.botEnemyStructureDangerLocked(c, px, py) && !s.botIncomingPressureLocked(b, now) {
			if target != nil && s.botMoveToFarmTargetLocked(b, target, now) {
				return
			}
			if s.botMoveToFarmCoverageLocked(b, now) {
				return
			}
		}
	}
	// Resolve the live farm target before a coverage-centre move so telemetry
	// and the attack watchdog retain the authoritative wave commitment even
	// while the bot is repositioning inside that wave.
	_ = s.botFarmApproachTargetLocked(b, now)
	// Once a creep is already in attack range, do not let the coverage-centre
	// move turn the laner into a spectator. The bot can answer the local wave
	// without committing to a chase.
	if target != nil {
		cx, cy := c.posAtLocked(float32(now))
		distance := math.Hypot(float64(target.x-cx), float64(target.y-cy))
		if distance <= hs.effAttackRangeLocked(now)+hs.av.Radius()+target.mob.Radius() &&
			distance <= dotaXPShareRadius &&
			s.botFarmMayAttackCreepLocked(b, now) {
			s.startAttackLocked(c, target)
			return
		}
		if distance <= dotaXPShareRadius && !s.botFarmMayAttackCreepLocked(b, now) {
			// The owner is already under pressure; preserve the XP body instead
			// of holding an attack order that would chase the wave.
			if s.botMoveToFarmCoverageLocked(b, now) {
				return
			}
		}
	}
	if px, py, ok := s.botFarmTargetSafePointLocked(b, target, now); ok {
		cx, cy := c.posAtLocked(float32(now))
		if math.Hypot(float64(px-cx), float64(py-cy)) > dotaXPShareRadius*0.5 &&
			!s.botEnemyStructureDangerLocked(c, px, py) {
			if target != nil && s.botMoveToFarmTargetLocked(b, target, now) {
				return
			}
			if s.botMoveToFarmCoverageLocked(b, now) {
				return
			}
		}
	}
	if s.botConsiderHarassAbilityLocked(b, now) {
		return
	}
	if target := s.botFindLaneTargetLocked(b, now, botLaneEngageRadius, true); target != nil {
		// A generic lane target is still a farm target. Do not turn the
		// fallback selector into a creep chase: gold is not scarce for bots and
		// walking into a full-health wave is exactly how they lose HP and then
		// abandon the next proximity-XP event.
		if s.botMoveToFarmTargetLocked(b, target, now) {
			return
		}
	}
	if target := s.botFarmApproachTargetLocked(b, now); target != nil {
		if s.botMoveToFarmTargetLocked(b, target, now) {
			return
		}
	}
	if b.farmLane >= 0 && b.farmLane != b.lane {
		if lx, ly, ok := s.botFarmPrestagePointLocked(b, b.farmLane, now); ok &&
			!s.botEnemyStructureDangerLocked(c, lx, ly) {
			s.botMoveTowardLocked(b, lx, ly, now)
		} else {
			lx, ly := s.botPushPointLocked(b, b.farmLane, now)
			s.botMoveTowardLocked(b, lx, ly, now)
		}
		return
	}
	if lx, ly, ok := s.botFarmPrestagePointLocked(b, b.lane, now); ok &&
		!s.botEnemyStructureDangerLocked(c, lx, ly) {
		s.botMoveTowardLocked(b, lx, ly, now)
	} else {
		lx, ly := s.botLanePoint(b, now)
		s.botMoveTowardLocked(b, lx, ly, now)
	}
}
