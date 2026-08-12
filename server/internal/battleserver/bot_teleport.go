package battleserver

import (
	"math"
	"sort"

	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
)

const (
	botTeleportCharges         int32 = 10
	botTeleportMaterialSavings       = 5.0
	botTeleportEnemyRadius           = 16.0
	// A ten-second channel must not be tied to the front creep of a wave that is
	// already being collapsed on. This wider forecast radius matches the local
	// fight scale and catches heroes that can reach the creep before completion.
	botTeleportCreepPressureRadius = 24.0
	botTeleportHeroForecastRadius  = 32.0
	botTeleportCreepMinHPFrac      = 0.70
	// Structures are reinforcement points only while a lane wave or meaningful
	// enemy pressure is close enough for the landing to matter. 24u matches the
	// local bot-fight scale and avoids treating distant lane traffic as support.
	botTeleportStructureSupportRadius = 24.0
	// A structure fallback is a temporary redeploy, not a permanent lane switch.
	botTeleportStructureHold = 45.0
	// Base responders only redeploy when the walk is materially long and the
	// threatened objective is close enough for the landing to matter.
	botBaseDefenseTeleportMinDistance = 80.0
	botBaseDefenseTeleportRadius      = 42.0
	// These mainData FX registry IDs map to VFX_ScrollTeleport_prop01 (preparation)
	// and VFX_ScrollTeleport_prop02 (arrival) respectively.
	botTeleportMarkerFx = "TeleportMarker"
	botTeleportTargetFx = "TeleportTarget"
)

type botTeleportOrder struct {
	article        int32
	target         int32
	targetKind     string
	macroObjective int32
	startHPFrac    float64
	complete       float64
	targetX        float32
	targetY        float32
	originX        float32
	originY        float32
	nextTargetFx   float64
	markerFx       int32
	targetFx       int32
	targetAnchor   int32
	arrivalAnchor  int32
}

func botTeleportScroll() (gamedata.Item, bool) {
	items := gamedata.ItemsByKind(gamedata.ItemTeleportScroll)
	if len(items) != 1 {
		return gamedata.Item{}, false
	}
	return items[0], true
}

// seedBotTeleportScrollLocked gives a new bot a battle-only stack. It deliberately
// bypasses grantItemLocked: bot scrolls must never touch session.Store.
func (s *Server) seedBotTeleportScrollLocked(c *conn) {
	it, ok := botTeleportScroll()
	if !ok {
		return
	}
	hs := c.huntState
	hs.bag = map[int32]int32{}
	hs.bagItemID = map[int32]int32{}
	hs.bagArticleByID = map[int32]int32{}
	hs.itemCooldownUntil = map[int32]float64{}

	s.push(c, battleproto.CmdPrototypeInfo, amf.NewArray().
		Set("id", teleportActionProtoID).Set("desc", teleportActionProtoDesc()))
	s.ensureItemProtoLocked(c, it.ArticleID)
	hs.nextBagID++
	hs.bag[it.ArticleID] = botTeleportCharges
	hs.bagItemID[it.ArticleID] = hs.nextBagID
	hs.bagArticleByID[hs.nextBagID] = it.ArticleID
	s.push(c, battleproto.CmdAddToInv, amf.NewArray().
		Set("id", hs.nextBagID).Set("proto", it.ArticleID).Set("count", botTeleportCharges))
}

func (s *Server) botMaybeStartTeleportLocked(b *botBrain, now float64) bool {
	c, hs := b.c, b.c.huntState
	it, ok := botTeleportScroll()
	if !ok || !isBotConn(c) || b.pendingTeleport != nil || hs.deadUntil > 0 ||
		hs.st.stunned(now) || hs.bag[it.ArticleID] <= 0 ||
		now < hs.itemCooldownUntil[it.ArticleID] {
		return false
	}

	recovery := b.retreating && b.retreatMode == botRetreatModeRecovery
	if b.retreating && !recovery {
		return false
	}
	// A mobilization group has a spatial rendezvous contract. The ordinary
	// fountain teleport chooses a nearby allied creep/structure and would split
	// a healthy member away from the shared rally point. Recovery teleports are
	// still allowed for injured members; healthy members must walk to the rally.
	if botAnyMobilizationReason(b.macroAssignment.Reason) && !recovery {
		return false
	}
	var target *mobState
	var destX, destY float32
	firstFarmRedeploy := false
	macroRedeploy := !recovery && botBaseDefenseResponderAssignment(b.macroAssignment) &&
		botHPFrac(hs, now) >= botSafeHPFrac
	if recovery {
		if !s.botRecoveryTeleportReadyLocked(b, now) {
			return false
		}
		var targetOK bool
		target, destX, destY, targetOK = s.botRecoveryTeleportTargetLocked(b, now)
		if !targetOK || !s.botRecoveryTeleportMateriallyFasterLocked(c, now, destX, destY, it) {
			return false
		}
	} else if macroRedeploy {
		if !s.botBaseDefenseTeleportReadyLocked(b, now) {
			return false
		}
		var targetOK bool
		target, destX, destY, targetOK = s.botBaseDefenseTeleportTargetLocked(b, now)
		if !targetOK || s.botBaseDefenseTeleportTargetReasonLocked(b, target, b.macroAssignment.ObjectiveID) != "" {
			return false
		}
		walkDistance := s.botTeleportWalkDistanceLocked(c, destX, destY, now)
		if !s.botTeleportMateriallyFasterLocked(c, destX, destY, walkDistance, it) {
			return false
		}
	} else {
		// The opening farm hand-off is a separate ETA comparison. A bot may still
		// be walking out of base, so the ordinary fountain-only teleport gate is
		// too late; however, the scroll is spent only when an allied structure
		// actually gets the bot to the lane rendezvous sooner than walking there.
		// A stale base objective is still authoritative for this decision. If a
		// test or a live plan has only changed Mode while keeping ObjectiveID, do
		// not fall through into the opening-farm teleport branch.
		farmRedeployAllowed := b.macroAssignment.ObjectiveID == 0 &&
			!botBaseDefenseResponderAssignment(b.macroAssignment)
		if farmRedeployAllowed {
			if firstTarget, firstX, firstY, _, firstOK := s.botFirstFarmTeleportTargetLocked(b, now); firstOK &&
				s.botFirstFarmTeleportMateriallyFasterLocked(b, now, firstX, firstY, it) {
				target, destX, destY = firstTarget, firstX, firstY
				firstFarmRedeploy = true
				b.farmLaneArrivalPending = false
			} else if handoffTarget, handoffX, handoffY, _, handoffOK := s.botFarmHandoffTeleportTargetLocked(b, now); handoffOK &&
				s.botFirstFarmTeleportMateriallyFasterLocked(b, now, handoffX, handoffY, it) {
				target, destX, destY = handoffTarget, handoffX, handoffY
				firstFarmRedeploy = true
				b.farmLaneArrivalPending = false
			} else {
				farmRedeployAllowed = false
			}
		}
		if !farmRedeployAllowed {
			laneRedeploy := s.botStandingBarracksLaneRedeployLocked(b, now)
			if botHPFrac(hs, now) < botSafeHPFrac || (!laneRedeploy && !s.atRespawnFountainLocked(c, now)) {
				return false
			}
			if laneRedeploy && (hs.attackTarget != 0 || hs.pvpTarget != 0 || botNearbyEnemyHeroPressureLocked(b, now) > 0) {
				return false
			}
			var walkDistance float64
			var targetOK bool
			target, destX, destY, walkDistance, targetOK = s.botTeleportTargetLocked(b, now)
			if !targetOK || !s.botTeleportMateriallyFasterLocked(c, destX, destY, walkDistance, it) {
				return false
			}
		}
	}

	// Starting the channel is the one authoritative charge boundary. All later
	// exits are cancellation only; neither the inventory nor cooldown is restored.
	s.botHoldTeleportChannelLocked(b, now)

	hs.bag[it.ArticleID]--
	hs.itemCooldownUntil[it.ArticleID] = now + it.Cooldown
	wireID := hs.bagItemID[it.ArticleID]
	s.push(c, battleproto.CmdRemFromInv, amf.NewArray().Set("id", wireID).Set("count", int32(1)))
	s.push(c, battleproto.CmdActionDone, amf.NewArray().
		Set("id", c.objID).Set("action", it.ArticleID).Set("item", true).
		Set("cooldown", now+it.Cooldown))

	targetKind := botTeleportTargetKind(target)
	if recovery {
		targetKind = "recovery_structure"
	} else if macroRedeploy {
		targetKind = "base_structure"
	} else if firstFarmRedeploy {
		targetKind = "first_farm_rendezvous"
	}
	originX, originY := c.posAtLocked(float32(now))
	markerFx := s.fxStartLocked(c, botTeleportMarkerFx, c.objID, 0, false, 0, 0)
	targetAnchor := s.spawnTrapAnchorLocked(c, destX, destY, now)
	targetFx := s.fxStartLocked(c, botTeleportTargetFx, targetAnchor, target.id, true, destX, destY)
	b.pendingTeleport = &botTeleportOrder{
		article:        it.ArticleID,
		target:         target.id,
		targetKind:     targetKind,
		macroObjective: b.macroAssignment.ObjectiveID,
		startHPFrac:    botHPFrac(hs, now),
		complete:       now + it.Preparation,
		targetX:        destX,
		targetY:        destY,
		originX:        originX,
		originY:        originY,
		nextTargetFx:   now + 2.0,
		markerFx:       markerFx,
		targetFx:       targetFx,
		targetAnchor:   targetAnchor,
	}
	s.telemetryRecordBotTeleportLocked(c, "bot_teleport_start", now, target.id, targetKind, destX, destY, markerFx, targetFx, 0, "")
	return true
}

// botFirstFarmTeleportTargetLocked chooses the allied lane structure that leaves
// the shortest walk from its safe landing point to the map-derived first-wave
// rendezvous. It intentionally does not inspect enemy creep positions: the
// opening decision is valid through fog and is based only on authored geometry,
// the bot's current assignment, and allied structures.
func (s *Server) botFirstFarmTeleportTargetLocked(b *botBrain, now float64) (*mobState, float32, float32, float64, bool) {
	if b == nil || b.c == nil || b.c.huntState == nil || b.retreating ||
		b.c.huntState.attackTarget != 0 || b.c.huntState.pvpTarget != 0 ||
		botNearbyEnemyHeroPressureLocked(b, now) > 0 || !botFirstFarmWavePendingLocked(b, now) {
		return nil, 0, 0, 0, false
	}
	return s.botFarmTeleportTargetLocked(b, now)
}

func (s *Server) botFarmHandoffTeleportTargetLocked(b *botBrain, now float64) (*mobState, float32, float32, float64, bool) {
	if b == nil || !b.farmLaneArrivalPending || b.c == nil || b.c.huntState == nil || b.retreating ||
		b.c.huntState.attackTarget != 0 || b.c.huntState.pvpTarget != 0 ||
		botNearbyEnemyHeroPressureLocked(b, now) > 0 {
		return nil, 0, 0, 0, false
	}
	return s.botFarmTeleportTargetLocked(b, now)
}

func (s *Server) botFarmTeleportTargetLocked(b *botBrain, now float64) (*mobState, float32, float32, float64, bool) {
	lane := botFarmLaneLocked(b)
	if lane < 0 || lane >= len(b.c.inst.dota.m.Lanes) {
		return nil, 0, 0, 0, false
	}
	meetX, meetY, ok := s.botFarmPrestagePointLocked(b, lane, now)
	if !ok {
		return nil, 0, 0, 0, false
	}
	var best *mobState
	var bestX, bestY float32
	bestRemaining := math.Inf(1)
	for _, m := range b.c.inst.mobs {
		if !s.botTeleportTargetValidLocked(b, m) || !m.structure || botTeleportLane(b.c, m) != lane {
			continue
		}
		destX, destY, ok := s.botTeleportDestinationLocked(b.c, m)
		if !ok || s.botEnemyStructureDangerLocked(b.c, destX, destY) {
			continue
		}
		remaining := s.botTeleportWalkDistanceBetweenLocked(b.c, destX, destY, meetX, meetY)
		if math.IsInf(remaining, 1) || remaining >= bestRemaining {
			continue
		}
		best, bestX, bestY, bestRemaining = m, destX, destY, remaining
	}
	if best == nil {
		return nil, 0, 0, 0, false
	}
	return best, bestX, bestY, bestRemaining, true
}

func (s *Server) botFirstFarmTeleportMateriallyFasterLocked(b *botBrain, now float64, destX, destY float32, it gamedata.Item) bool {
	if b == nil || b.c == nil {
		return false
	}
	lane := botFarmLaneLocked(b)
	meetX, meetY, ok := s.botFarmPrestagePointLocked(b, lane, now)
	if !ok {
		return false
	}
	direct := s.botTeleportWalkDistanceLocked(b.c, meetX, meetY, now)
	remaining := s.botTeleportWalkDistanceBetweenLocked(b.c, destX, destY, meetX, meetY)
	if math.IsInf(direct, 1) || math.IsInf(remaining, 1) {
		return false
	}
	speed := float64(b.c.moveSpeedLocked(s))
	if speed <= 0 {
		return false
	}
	return direct/speed >= it.Preparation+remaining/speed+botTeleportMaterialSavings
}

// botStandingBarracksLaneRedeployLocked is a spatial hand-off rule. A living
// enemy barracks guarantees that the lane remains an XP source, so a cover bot
// assigned to that lane may teleport from wherever it is when the orchestrator
// has moved it away from its authored lane. This is deliberately independent of
// match time and still relies on the normal visible/reliable teleport target
// checks before a scroll is consumed.
func (s *Server) botStandingBarracksLaneRedeployLocked(b *botBrain, now float64) bool {
	if b == nil || b.c == nil || b.c.inst == nil || b.c.inst.dota == nil ||
		b.c.huntState == nil || b.retreating || b.macroAssignment.Mode == botMacroRecover ||
		!b.macroAssignment.FarmLaneSet {
		return false
	}
	lane := b.macroAssignment.FarmLane
	if lane < 0 || lane >= len(b.c.inst.dota.m.Lanes) || lane == b.lane ||
		!botLaneHasLivingEnemyBarracksLocked(b.c.inst, b.c.playerTeam(), lane) {
		return false
	}
	// Do not spend a scroll while the bot is already close enough to stage for
	// the lane. The exact walk-vs-channel decision remains in the shared
	// material-faster check after a safe allied target has been selected.
	px, py, ok := s.botFarmCoveragePointLocked(b, now)
	if !ok {
		px, py = s.botPushPointLocked(b, lane, now)
	}
	cx, cy := b.c.posAtLocked(float32(now))
	return math.Hypot(float64(px-cx), float64(py-cy)) > botLaneApproachRadius*2
}

func (s *Server) botBaseDefenseTeleportReadyLocked(b *botBrain, now float64) bool {
	if b == nil || b.c == nil || b.c.huntState == nil || b.retreating ||
		!botBaseDefenseResponderAssignment(b.macroAssignment) ||
		botHPFrac(b.c.huntState, now) < botSafeHPFrac {
		return false
	}
	if s.botBaseDefenseTeleportValidityReasonLocked(b, b.macroAssignment.ObjectiveID, 0, now) != "" {
		return false
	}
	objective := b.c.inst.mobs[b.macroAssignment.ObjectiveID]
	if objective == nil || objective.dead || !objective.structure || objective.team != b.c.playerTeam() ||
		s.botDefenseStructureThreatSeverityLocked(b.c.inst, b.c.playerTeam(), objective, now) <= 0 {
		return false
	}
	cx, cy := b.c.posAtLocked(float32(now))
	minDistance := botBaseDefenseTeleportMinDistance
	// A critically damaged gun is a time-sensitive defensive objective. The
	// ordinary 80u walk-vs-scroll threshold is too conservative here: a bot can
	// be one short lane segment away while the gun has only a few hits left.
	// This is state-driven, not clock-driven; keep the normal threshold until
	// the structure enters the existing finish window.
	if botMacroObjectiveFinishWindowLocked(objective) {
		minDistance *= 0.5
	}
	if dist2(cx, cy, objective.x, objective.y) < float32(minDistance*minDistance) {
		return false
	}
	if !s.botRecoveryTeleportDamageEndedLocked(b, now) {
		return false
	}
	return s.botBaseDefenseTeleportOriginContactBrokenLocked(b, now)
}

// botBaseDefenseTeleportValidityReasonLocked is the authoritative macro premise
// for a base_structure channel. The live team plan is checked when available so
// an assignment replacement cannot leave an old channel running.
func (s *Server) botBaseDefenseTeleportValidityReasonLocked(b *botBrain, objectiveID, orderObjective int32, now float64) string {
	if b == nil || b.c == nil || b.c.inst == nil || b.c.inst.dota == nil {
		return "base_assignment_changed"
	}
	assignment := b.macroAssignment
	if !botBaseDefenseResponderAssignment(assignment) {
		return "base_assignment_changed"
	}
	if b.retreating {
		return "base_retreat"
	}
	if objectiveID == 0 || assignment.ObjectiveID != objectiveID ||
		(orderObjective != 0 && assignment.ObjectiveID != orderObjective) {
		return "base_objective_changed"
	}
	if plan, ok := b.c.inst.dota.teamPlans[b.c.playerTeam()]; ok && len(plan.Assignments) > 0 {
		current, exists := plan.Assignments[b.c.objID]
		if plan.Mode != botMacroBase || plan.ObjectiveID != assignment.ObjectiveID || !exists ||
			!botBaseDefenseResponderAssignment(current) ||
			current.ObjectiveID != assignment.ObjectiveID {
			return "base_assignment_changed"
		}
	}
	premise := botTeamPlan{
		Team: b.c.playerTeam(), Mode: botMacroBase, Lane: assignment.Lane,
		ObjectiveID: assignment.ObjectiveID,
	}
	if !s.botPlanPremiseValidLocked(b.c.inst, b.c.playerTeam(), premise, now) {
		return "base_premise_lost"
	}
	return ""
}

func (s *Server) botBaseDefenseTeleportTargetReasonLocked(b *botBrain, target *mobState, objectiveID int32) string {
	if !s.botTeleportTargetValidLocked(b, target) || !target.structure {
		return "base_target_invalid"
	}
	objective := b.c.inst.mobs[objectiveID]
	if objective == nil || objective.dead {
		return "base_objective_changed"
	}
	if dist2(target.x, target.y, objective.x, objective.y) > float32(botBaseDefenseTeleportRadius*botBaseDefenseTeleportRadius) {
		return "base_target_far"
	}
	return ""
}

func (s *Server) botBaseDefenseTeleportDamageEndedLocked(b *botBrain, order *botTeleportOrder, now float64) bool {
	if order != nil && order.startHPFrac > 0 && (botHPFrac(b.c.huntState, now) < order.startHPFrac-0.01 || botHPFrac(b.c.huntState, now) < botSafeHPFrac) {
		return false
	}
	return s.botRecoveryTeleportDamageEndedLocked(b, now)
}

func (s *Server) botBaseDefenseTeleportOriginContactBrokenLocked(b *botBrain, now float64) bool {
	if b == nil || b.c == nil || b.c.huntState == nil {
		return false
	}
	if botNearbyEnemyHeroPressureLocked(b, now) > 0 || s.botCommittedStructureFocusLocked(b.c, now) != nil ||
		b.c.huntState.attackTarget != 0 || b.c.huntState.pvpTarget != 0 || b.engageTarget != 0 {
		return false
	}
	cx, cy := b.c.posAtLocked(float32(now))
	visionSources := dotaTeamVisionSourcesLocked(b.c.inst, b.c.playerTeam(), now)
	for _, m := range b.c.inst.mobs {
		if m == nil || m.dead || !m.enemyOf(b.c.playerTeam()) || m.structure ||
			!botVisibleEnemyMobLocked(b.c.playerTeam(), m, visionSources) {
			continue
		}
		if dist2(cx, cy, m.x, m.y) <= float32(botTeleportEnemyRadius*botTeleportEnemyRadius) {
			return false
		}
	}
	return !s.botEnemyStructureDangerLocked(b.c, cx, cy)
}

func (s *Server) botBaseDefenseTeleportTargetLocked(b *botBrain, now float64) (*mobState, float32, float32, bool) {
	objective := b.c.inst.mobs[b.macroAssignment.ObjectiveID]
	if objective == nil || objective.dead || !objective.structure || objective.team != b.c.playerTeam() {
		return nil, 0, 0, false
	}
	type candidate struct {
		target    *mobState
		destX     float32
		destY     float32
		clearance float64
		distance  float64
	}
	candidates := make([]candidate, 0)
	for _, m := range b.c.inst.mobs {
		if !s.botTeleportTargetValidLocked(b, m) || !m.structure ||
			dist2(m.x, m.y, objective.x, objective.y) > float32(botBaseDefenseTeleportRadius*botBaseDefenseTeleportRadius) {
			continue
		}
		destX, destY, ok := s.botTeleportDestinationLocked(b.c, m)
		if !ok {
			continue
		}
		candidates = append(candidates, candidate{
			target: m, destX: destX, destY: destY,
			clearance: s.botTeleportEnemyClearanceLocked(b.c, destX, destY, now),
			distance:  math.Sqrt(float64(dist2(m.x, m.y, objective.x, objective.y))),
		})
	}
	if len(candidates) == 0 {
		return nil, 0, 0, false
	}
	sort.Slice(candidates, func(i, j int) bool {
		left, right := candidates[i], candidates[j]
		if left.clearance != right.clearance {
			return left.clearance > right.clearance
		}
		if left.distance != right.distance {
			return left.distance < right.distance
		}
		return left.target.id < right.target.id
	})
	best := candidates[0]
	return best.target, best.destX, best.destY, true
}

func botBaseDefenseResponderAssignment(assignment botMacroAssignment) bool {
	return assignment.Mode == botMacroBase && (assignment.Role == "defender" || assignment.Role == "cover")
}

func (s *Server) botRecoveryTeleportReadyLocked(b *botBrain, now float64) bool {
	c, hs := b.c, b.c.huntState
	emergency := botHPFrac(hs, now) <= botRetreatHPFrac
	// A recovery channel freezes the avatar for the preparation window.  Below
	// the hard HP floor that is only safe when the local damage has stopped;
	// starting it while a creep is still landing hits turned a retreat into a
	// stationary death.  The normal retreat movement is safer in that case and
	// can re-evaluate the teleport on the next tick after contact is broken.
	if emergency && s.botIncomingPressureLocked(b, now) {
		return false
	}
	if !b.retreating || b.retreatMode != botRetreatModeRecovery || s.atRespawnFountainLocked(c, now) ||
		len(hs.channels) > 0 ||
		(!emergency && !s.botRecoveryTeleportDamageEndedLocked(b, now)) ||
		!s.botRecoveryTeleportContactBrokenLocked(b, now) || s.botCommittedStructureFocusLocked(c, now) != nil {
		return false
	}
	// Below the hard retreat floor, waiting for the recent-damage ring to go
	// quiet is not a safety check: it makes a bot walk through the same creep
	// wave until it dies.  Creep damage is still allowed to interrupt a channel
	// above this floor; at or below it, the already-latched recovery state gets a
	// chance to escape. Hero pressure remains a hard block because a channel
	// started in melee contact is overwhelmingly likely to be interrupted.
	cx, cy := c.posAtLocked(float32(now))
	return !s.botEnemyStructureDangerLocked(c, cx, cy)
}

func (s *Server) botRecoveryTeleportDamageEndedLocked(b *botBrain, now float64) bool {
	loss, rate := botRecentHPLossLocked(b, now)
	return loss < botPredictiveRetreatLossFrac && rate < botPredictiveRetreatLossRate
}

func (s *Server) botRecoveryTeleportContactBrokenLocked(b *botBrain, now float64) bool {
	c := b.c
	if botNearbyEnemyHeroPressureLocked(b, now) > 0 {
		return false
	}
	if b.engageTarget != 0 {
		target := c.pvpMember(b.engageTarget)
		if target != nil && target.huntState != nil && target.huntState.deadUntil == 0 {
			tx, ty := target.posAtLocked(float32(now))
			cx, cy := c.posAtLocked(float32(now))
			if target.huntState.pvpTarget == c.objID || target.huntState.attackTarget == c.objID ||
				dist2(cx, cy, tx, ty) <= float32(botRetreatPressureRadius*botRetreatPressureRadius) {
				return false
			}
		}
		b.engageTarget = 0
	}
	return true
}

func (s *Server) botRecoveryTeleportTargetLocked(b *botBrain, now float64) (*mobState, float32, float32, bool) {
	c := b.c
	hx, hy := botHomeLocked(c)
	var best *mobState
	bestDistance := math.Inf(1)
	for _, m := range botSortedMobs(c.inst) {
		if !s.botTeleportTargetValidLocked(b, m) || !m.structure || !m.altar {
			continue
		}
		if distance := float64(dist2(m.x, m.y, hx, hy)); distance < bestDistance ||
			(distance == bestDistance && (best == nil || m.id < best.id)) {
			best, bestDistance = m, distance
		}
	}
	if best == nil {
		return nil, 0, 0, false
	}
	destX, destY, ok := s.botTeleportDestinationLocked(c, best)
	if !ok {
		return nil, 0, 0, false
	}
	return best, destX, destY, true
}

func (s *Server) botRecoveryTeleportMateriallyFasterLocked(c *conn, now float64, destX, destY float32, it gamedata.Item) bool {
	hx, hy := botHomeLocked(c)
	direct := s.botTeleportWalkDistanceLocked(c, hx, hy, now)
	remaining := s.botTeleportWalkDistanceBetweenLocked(c, destX, destY, hx, hy)
	if math.IsInf(direct, 1) || math.IsInf(remaining, 1) {
		return false
	}
	speed := float64(c.moveSpeedLocked(s))
	if speed <= 0 {
		return false
	}
	return direct/speed >= it.Preparation+remaining/speed+botTeleportMaterialSavings
}

func (s *Server) botRecoveryTeleportOriginPressuredLocked(b *botBrain, now float64, order *botTeleportOrder) bool {
	c := b.c
	for _, enemy := range botLivingEnemyHeroes(c, now) {
		ex, ey := enemy.posAtLocked(float32(now))
		if enemy.huntState.pvpTarget == c.objID || enemy.huntState.attackTarget == c.objID ||
			c.huntState.pvpTarget == enemy.objID || b.engageTarget == enemy.objID ||
			dist2(order.originX, order.originY, ex, ey) <= float32(botRetreatPressureRadius*botRetreatPressureRadius) {
			return true
		}
	}
	return s.botCommittedStructureFocusLocked(c, now) != nil ||
		s.botEnemyStructureDangerLocked(c, order.originX, order.originY)
}

func (s *Server) botTickTeleportLocked(b *botBrain, now float64) {
	order := b.pendingTeleport
	if order == nil {
		return
	}
	// A fresh live base assignment outranks a channel that was selected before
	// the plan was refreshed.  The scroll was already charged at channel start;
	// cancellation therefore deliberately does not refund it.
	if reason := s.botUrgentBaseDefenseTeleportReasonLocked(b, order, now); reason != "" {
		s.cancelBotTeleportAtLocked(b, reason, now)
		return
	}
	hs := b.c.huntState
	if hs.deadUntil > 0 || hs.st.stunned(now) {
		reason := "bot_stunned"
		if hs.deadUntil > 0 {
			reason = "bot_dead"
		}
		s.cancelBotTeleportLocked(b, reason)
		return
	}
	if order.targetKind == "recovery_structure" {
		// A recovery channel deliberately freezes the bot in place.  Once its HP
		// has crossed the existing hard retreat floor, cancelling because another
		// creep/shot landed is worse than committing the escape: the bot loses the
		// scroll and remains stationary under the same damage that caused the
		// cancellation.  Keep the ordinary interruption rule while the bot is
		// still above the floor, but commit an already-desperate recovery channel
		// unless the authoritative stun/death/target checks above invalidate it.
		emergencyCommit := botHPFrac(b.c.huntState, now) <= botRetreatHPFrac
		if !s.botRecoveryTeleportDamageEndedLocked(b, now) && !emergencyCommit {
			s.cancelBotTeleportLocked(b, "recovery_recent_damage")
			return
		}
		if s.botRecoveryTeleportOriginPressuredLocked(b, now, order) && !emergencyCommit {
			s.cancelBotTeleportLocked(b, "recovery_origin_pressure")
			return
		}
	}
	if order.targetKind == "base_structure" {
		if reason := s.botBaseDefenseTeleportValidityReasonLocked(b, order.macroObjective, order.macroObjective, now); reason != "" {
			s.cancelBotTeleportLocked(b, reason)
			return
		}
		if !s.botBaseDefenseTeleportDamageEndedLocked(b, order, now) {
			s.cancelBotTeleportLocked(b, "base_recent_damage")
			return
		}
		if !s.botBaseDefenseTeleportOriginContactBrokenLocked(b, now) {
			s.cancelBotTeleportLocked(b, "base_origin_contact")
			return
		}
	}
	target := hs.mobs[order.target]
	if order.targetKind == "base_structure" {
		if reason := s.botBaseDefenseTeleportTargetReasonLocked(b, target, order.macroObjective); reason != "" {
			s.cancelBotTeleportLocked(b, reason)
			return
		}
	}
	if !s.botTeleportTargetValidLocked(b, target) {
		s.cancelBotTeleportLocked(b, "target_invalid")
		return
	}
	if !target.structure {
		if _, reliable := s.botTeleportCreepReliabilityLocked(b.c, target, now); !reliable {
			s.cancelBotTeleportLocked(b, "target_became_risky")
			return
		}
	}
	if currentX, currentY, ok := s.botTeleportDestinationLocked(b.c, target); !ok {
		s.cancelBotTeleportLocked(b, "destination_unsafe")
		return
	} else if order.targetKind == "base_structure" {
		// Tactical redeploys revalidate the safe landing point on every tick;
		// completion performs the same check again before moving the bot.
		order.targetX, order.targetY = currentX, currentY
	}
	if now < order.complete {
		// Member upkeep runs before the bot brain and can resume an old attack
		// when a previously scheduled ACTION_DONE matures. Reassert the channel
		// hold every tick so that resumed order cannot reach the shared combat pass.
		s.botHoldTeleportChannelLocked(b, now)
		if now >= order.nextTargetFx {
			// Dota targets can move or be absent from the client's current view. Keep
			// the target id for clients that can resolve it, and always provide the
			// last safe point as a fallback. A visual refresh must not cancel the
			// authoritative channel just because a new point cannot be computed.
			destX, destY := order.targetX, order.targetY
			if currentX, currentY, ok := s.botTeleportDestinationLocked(b.c, target); ok {
				destX, destY = currentX, currentY
				order.targetX, order.targetY = currentX, currentY
			}
			s.fxEndLocked(b.c, order.targetFx)
			s.removeTrapAnchorLocked(b.c, order.targetAnchor, now)
			order.targetAnchor = s.spawnTrapAnchorLocked(b.c, destX, destY, now)
			order.targetFx = s.fxStartLocked(b.c, botTeleportTargetFx, order.targetAnchor, target.id, true, destX, destY)
			order.nextTargetFx = now + 2.0
			s.telemetryRecordBotTeleportLocked(b.c, "bot_teleport_destination_refresh", now, target.id, order.targetKind, destX, destY, order.markerFx, order.targetFx, 0, "")
		}
		return
	}
	destX, destY, ok := s.botTeleportDestinationLocked(b.c, target)
	if !ok {
		s.cancelBotTeleportLocked(b, "destination_unsafe")
		return
	}

	c := b.c
	markerFx, targetFx := order.markerFx, order.targetFx
	s.endBotTeleportPreparationFXLocked(c, order)
	s.removeTrapAnchorLocked(c, order.targetAnchor, now)
	c.stopArrivalLocked()
	c.x, c.y, c.vx, c.vy, c.snapT = destX, destY, 0, 0, float32(now)
	c.hasDest = false
	// Reset only remote avatars that were already rendered. DELETE_OBJECT followed
	// by the existing render chain makes its first POSITION the destination without
	// revealing the bot to viewers currently behind Dota fog.
	s.hardResetAvatarForLocked(c, now)
	c.sendPosLocked(s, destX, destY, 0, 0, float32(now))
	arrivalAnchor := s.spawnTrapAnchorLocked(c, destX, destY, now)
	arrivalFx := s.fxStartLocked(c, botTeleportTargetFx, arrivalAnchor, 0, true, destX, destY)
	hs.scheduleFxEnd(arrivalFx, now+2.0)
	hs.anchorEnds = append(hs.anchorEnds, anchorEnd{id: arrivalAnchor, at: now + 2.3})
	order.arrivalAnchor = arrivalAnchor
	if order.targetKind == "structure" {
		b.laneRedeployPointX = destX
		b.laneRedeployPointY = destY
		b.laneRedeployUntil = now + botTeleportStructureHold
		b.laneRedeployPointValid = true
	}
	b.pendingTeleport = nil
	s.telemetryRecordBotTeleportLocked(c, "bot_teleport_success", now, order.target, order.targetKind, destX, destY, markerFx, targetFx, arrivalFx, "")
}

// botUrgentBaseDefenseTeleportReasonLocked is the narrow preemption boundary for
// macro defense.  Creep, enemy, and tactical-structure channels are not allowed
// to strand a responder while a live own barracks/altar assignment exists.  A
// recovery/base channel is retained only when its destination is still the same
// live assignment and remains a valid landing point for that objective.
func (s *Server) botUrgentBaseDefenseTeleportReasonLocked(b *botBrain, order *botTeleportOrder, now float64) string {
	if b == nil || order == nil {
		return ""
	}
	// base_structure channels have an established validator below in
	// botTickTeleportLocked. Let it own the exact base_assignment_changed,
	// base_target_*, and base_premise_* reasons; urgent preemption is only for
	// creep/ordinary-structure channels.
	if order.targetKind == "base_structure" {
		return ""
	}
	assignment, urgent := s.botUrgentBaseDefenseAssignmentLocked(b, now)
	if !urgent {
		return ""
	}
	objective := b.c.inst.mobs[assignment.ObjectiveID]
	if objective == nil || objective.dead {
		return "urgent_base_defense"
	}
	switch order.targetKind {
	case "base_structure":
		if order.macroObjective != assignment.ObjectiveID ||
			s.botBaseDefenseTeleportValidityReasonLocked(b, order.macroObjective, order.macroObjective, now) != "" {
			return "urgent_base_defense"
		}
		target := b.c.huntState.mobs[order.target]
		if s.botBaseDefenseTeleportTargetReasonLocked(b, target, assignment.ObjectiveID) != "" {
			return "urgent_base_defense"
		}
		return ""
	case "recovery_structure":
		// Recovery has no independent macro objective. It is compatible only
		// when it is landing on the exact live structure being defended.
		if order.target != assignment.ObjectiveID || !objective.structure ||
			!s.botTeleportTargetValidLocked(b, objective) {
			return "urgent_base_defense"
		}
		return ""
	default:
		return "urgent_base_defense"
	}
}

func (s *Server) botUrgentBaseDefenseAssignmentLocked(b *botBrain, now float64) (botMacroAssignment, bool) {
	if b == nil || b.c == nil || b.c.inst == nil || b.c.huntState == nil ||
		!botBaseDefenseResponderAssignment(b.macroAssignment) {
		return botMacroAssignment{}, false
	}
	a := b.macroAssignment
	objective := b.c.inst.mobs[a.ObjectiveID]
	if objective == nil || objective.dead || !objective.structure || objective.team != b.c.playerTeam() {
		return botMacroAssignment{}, false
	}
	if s.botDefenseStructureThreatSeverityLocked(b.c.inst, b.c.playerTeam(), objective, now) <= 0 {
		return botMacroAssignment{}, false
	}
	// b.macroAssignment is the current responder decision and is authoritative
	// here. The team plan map is a previous planning snapshot during some ticker
	// phases; consulting it would reject a freshly installed live assignment and
	// leave the old channel running. The objective/threat checks above remain the
	// live premise guard.
	return a, true
}

func (s *Server) cancelBotTeleportLocked(b *botBrain, reason string) {
	s.cancelBotTeleportAtLocked(b, reason, float64(s.battleTime()))
}

func (s *Server) cancelBotTeleportAtLocked(b *botBrain, reason string, now float64) {
	order := b.pendingTeleport
	if order == nil {
		return
	}
	b.pendingTeleport = nil
	s.telemetryRecordBotTeleportLocked(b.c, "bot_teleport_cancel", now, order.target, order.targetKind, order.targetX, order.targetY, order.markerFx, order.targetFx, 0, reason)
	s.endBotTeleportPreparationFXLocked(b.c, order)
	s.removeTrapAnchorLocked(b.c, order.targetAnchor, now)
}

func (s *Server) endBotTeleportPreparationFXLocked(c *conn, order *botTeleportOrder) {
	markerFx, targetFx := order.markerFx, order.targetFx
	order.markerFx, order.targetFx = 0, 0
	s.fxEndLocked(c, markerFx)
	s.fxEndLocked(c, targetFx)
}

func botTeleportTargetKind(target *mobState) string {
	if target != nil && target.structure {
		return "structure"
	}
	return "creep"
}

func (s *Server) botHoldTeleportChannelLocked(b *botBrain, now float64) {
	c := b.c
	hs := c.huntState
	moving := c.hasDest || c.arrival != nil || c.vx != 0 || c.vy != 0
	if hs.attackTarget != 0 {
		s.stopAttackLocked(c, true)
	}
	if hs.pvpTarget != 0 {
		s.stopPvpAttackLocked(c, true)
	}
	s.cancelOrderLocked(c)
	c.stopArrivalLocked()
	cx, cy := c.posAtLocked(float32(now))
	c.x, c.y, c.vx, c.vy, c.snapT = cx, cy, 0, 0, float32(now)
	c.hasDest = false
	if moving {
		c.sendPosLocked(s, cx, cy, 0, 0, float32(now))
	}
}

func (s *Server) botTeleportTargetLocked(b *botBrain, now float64) (*mobState, float32, float32, float64, bool) {
	c := b.c
	if c.inst == nil || c.inst.dota == nil || len(c.inst.dota.m.Lanes) == 0 {
		return nil, 0, 0, 0, false
	}

	// Prefer a living allied lane creep globally. If no active creep can accept
	// a safe destination, use an allied lane structure only as the fallback.
	preferred := botFarmLaneLocked(b)
	if preferred < 0 || preferred >= len(c.inst.dota.m.Lanes) {
		preferred = -1
	}
	if target, x, y, walk, ok := s.botTeleportBestTargetLocked(b, now, preferred, false); ok {
		return target, x, y, walk, true
	}
	for lane := range c.inst.dota.m.Lanes {
		if lane == preferred {
			continue
		}
		if target, x, y, walk, ok := s.botTeleportBestTargetLocked(b, now, lane, false); ok {
			return target, x, y, walk, true
		}
	}
	if target, x, y, walk, ok := s.botTeleportBestTargetLocked(b, now, preferred, true); ok {
		return target, x, y, walk, true
	}
	for lane := range c.inst.dota.m.Lanes {
		if lane == preferred {
			continue
		}
		if target, x, y, walk, ok := s.botTeleportBestTargetLocked(b, now, lane, true); ok {
			return target, x, y, walk, true
		}
	}
	return nil, 0, 0, 0, false
}

func (s *Server) botTeleportBestTargetLocked(b *botBrain, now float64, laneIndex int, structuresOnly bool) (*mobState, float32, float32, float64, bool) {
	c := b.c
	if laneIndex < 0 || laneIndex >= len(c.inst.dota.m.Lanes) {
		return nil, 0, 0, 0, false
	}
	lane := c.inst.dota.m.Lanes[laneIndex]
	var best *mobState
	bestScore := -math.MaxFloat64
	var bestX, bestY float32
	var bestWalk float64
	for _, m := range c.inst.mobs {
		if !s.botTeleportTargetValidLocked(b, m) || (structuresOnly != m.structure) || botTeleportLane(c, m) != laneIndex {
			continue
		}
		progress := botTeleportProgress(c, m, lane)
		reliability := 0.0
		if !structuresOnly {
			var reliable bool
			reliability, reliable = s.botTeleportCreepReliabilityLocked(c, m, now)
			if !reliable {
				continue
			}
		}
		dx, dy, ok := s.botTeleportDestinationLocked(c, m)
		if !ok {
			continue
		}
		if structuresOnly && !s.botTeleportStructureUsefulLocked(b, m, laneIndex, dx, dy, now) {
			continue
		}
		walk := s.botTeleportWalkDistanceLocked(c, dx, dy, now)
		// Reliability is deliberately worth several lane waypoints: arriving one
		// creep farther back is better than losing the scroll when the exposed
		// front creep dies during preparation.
		score := progress + reliability
		if score > bestScore || (score == bestScore && (best == nil || m.id < best.id)) {
			best, bestScore, bestX, bestY, bestWalk = m, score, dx, dy, walk
		}
	}
	if best == nil {
		return nil, 0, 0, 0, false
	}
	return best, bestX, bestY, bestWalk, true
}

// botTeleportCreepReliabilityLocked estimates whether an allied creep will remain a
// sensible anchor for the whole preparation channel. It combines current health,
// local wave balance and already-committed attacks. Visible heroes use a wider
// forecast radius than the final landing-safety check because they can close the
// remaining distance while the bot is channeling.
func (s *Server) botTeleportCreepReliabilityLocked(c *conn, target *mobState, now float64) (float64, bool) {
	if c == nil || c.inst == nil || target == nil || target.dead || target.structure || !botTeleportLaneCreep(target) {
		return 0, false
	}
	maxHP := target.maxHealth()
	if maxHP <= 0 {
		return 0, false
	}
	hpFrac := target.hp / maxHP
	if hpFrac < botTeleportCreepMinHPFrac {
		return 0, false
	}

	pressureRadius2 := float32(botTeleportCreepPressureRadius * botTeleportCreepPressureRadius)
	allies, enemies, committed := 0, 0, 0
	visionSources := dotaTeamVisionSourcesLocked(c.inst, c.playerTeam(), now)
	for _, m := range c.inst.mobs {
		if m == nil || m.dead {
			continue
		}
		if m.team == c.playerTeam() && botTeleportLaneCreep(m) && dist2(target.x, target.y, m.x, m.y) <= pressureRadius2 {
			allies++
		}
		if !m.enemyOf(c.playerTeam()) {
			continue
		}
		if !botVisibleEnemyMobLocked(c.playerTeam(), m, visionSources) {
			continue
		}
		if botTeleportLaneCreep(m) &&
			dist2(target.x, target.y, m.x, m.y) <= pressureRadius2 {
			enemies++
		}
		if m.hitTarget == target.id || m.projTarget == target.id || m.dtarget == target.id {
			committed++
		}
	}

	heroRadius2 := float32(botTeleportHeroForecastRadius * botTeleportHeroForecastRadius)
	for _, enemy := range c.inst.members {
		if enemy == nil || enemy.huntState == nil || enemy.playerTeam() == c.playerTeam() ||
			enemy.huntState.deadUntil > 0 {
			continue
		}
		ex, ey := enemy.posAtLocked(float32(now))
		activelyAttackingTarget := enemy.huntState.attackTarget == target.id
		if !activelyAttackingTarget && !botVisibleEnemyMemberLocked(c.inst, c.playerTeam(), enemy, now, visionSources) {
			continue
		}
		if activelyAttackingTarget || dist2(target.x, target.y, ex, ey) <= heroRadius2 {
			return 0, false
		}
	}

	// An isolated creep is valid while uncontested (the normal advanced-wave
	// case), but not once equal-or-greater enemy pressure reaches it. A damaged
	// creep or one with multiple attacks already committed is too fragile even
	// when its allies currently balance the headcount.
	if enemies > 0 && enemies >= allies || committed >= 2 || committed > 0 && hpFrac < 0.90 || enemies > 0 && hpFrac < 0.85 {
		return 0, false
	}

	quality := hpFrac*4 + float64(allies-enemies)*2 - float64(enemies)*2 - float64(committed)*3
	return quality, true
}

// botTeleportStructureUsefulLocked keeps a structure fallback tied to an actual
// reinforcement need. Lane identity comes from the authored lane geometry for
// both creeps and structures; no structure prefab or match-clock assumption is
// part of this decision.
func (s *Server) botTeleportStructureUsefulLocked(b *botBrain, structure *mobState, laneIndex int, destX, destY float32, now float64) bool {
	c := b.c
	visionSources := dotaTeamVisionSourcesLocked(c.inst, c.playerTeam(), now)
	radius2 := float32(botTeleportStructureSupportRadius * botTeleportStructureSupportRadius)
	near := func(x, y float32) bool {
		return dist2(x, y, structure.x, structure.y) <= radius2 || dist2(x, y, destX, destY) <= radius2
	}

	for _, m := range c.inst.mobs {
		if m.dead || m.structure || m.team != c.playerTeam() || !botTeleportLaneCreep(m) || botTeleportLane(c, m) != laneIndex {
			continue
		}
		if near(m.x, m.y) {
			return true
		}
	}
	for _, m := range c.inst.mobs {
		if m.dead || m.structure || !m.enemyOf(c.playerTeam()) ||
			!botVisibleEnemyMobLocked(c.playerTeam(), m, visionSources) || !botTeleportLaneCreep(m) || botTeleportLane(c, m) != laneIndex {
			continue
		}
		if near(m.x, m.y) {
			return true
		}
	}
	for _, enemy := range botLivingEnemyHeroes(c, now) {
		x, y := enemy.posAtLocked(float32(now))
		if near(x, y) {
			return true
		}
	}
	return false
}

func botTeleportLaneCreep(m *mobState) bool {
	return m != nil && !m.structure && len(m.lane) > 1 && m.laneIdx >= 0 && m.laneIdx < len(m.lane)
}

func botTeleportLane(c *conn, m *mobState) int {
	if m == nil || c == nil || c.inst == nil || c.inst.dota == nil {
		return -1
	}
	if !m.structure {
		for i, lane := range c.inst.dota.m.Lanes {
			if botSameDotaLane(m.lane, lane) {
				return i
			}
		}
		return -1
	}
	// Structures do not carry creep waypoints. Classify them from their
	// position against the authored Dota lane geometry; base springs and other
	// off-lane objects correctly return -1 and cannot become teleport targets.
	return c.inst.dota.m.LaneFor(gamedata.DotaStructure{X: float64(m.x), Z: float64(m.y)})
}

func botTeleportProgress(c *conn, m *mobState, lane []gamedata.Vec2) float64 {
	fwd := c.playerTeam() == dotaTeamHuman
	if !m.structure && len(m.lane) > 0 {
		if fwd {
			return float64(m.laneIdx)
		}
		return float64(len(lane) - m.laneIdx)
	}
	hx, hy := botHomeLocked(c)
	return math.Hypot(float64(m.x-hx), float64(m.y-hy))
}

func (s *Server) botTeleportTargetValidLocked(b *botBrain, m *mobState) bool {
	if m == nil || m.dead || m.team != b.c.playerTeam() || m.boss {
		return false
	}
	if m.structure {
		// Only baked Dota structures occupy this id range. Stationary Dota
		// bosses also use the structure combat pipeline, but are creatures and
		// therefore must not become scroll destinations.
		return m.id >= dotaStructIDBase && m.id < dotaCreepIDBase && m.dotaPrefab != ""
	}
	// A non-structure target is eligible only when it is a Dota lane creep.
	// Generic Hunt mobs and summons must never be accepted merely because they
	// happen to share the bot's team.
	return len(m.lane) > 1 && m.laneIdx >= 0 && m.laneIdx < len(m.lane)
}

func (s *Server) botTeleportDestinationLocked(c *conn, target *mobState) (float32, float32, bool) {
	if target == nil || target.dead || target.team != c.playerTeam() {
		return 0, 0, false
	}
	need := float32(target.mob.Radius() + c.huntState.av.Radius() + 0.75)
	hx, hy := botHomeLocked(c)
	base := math.Atan2(float64(hy-target.y), float64(hx-target.x))
	if math.IsNaN(base) {
		base = 0
	}
	var bestX, bestY float32
	bestClearance := -1.0
	found := false
	for i := 0; i < 16; i++ {
		angle := base + float64(i)*math.Pi/8
		x := target.x + need*float32(math.Cos(angle))
		y := target.y + need*float32(math.Sin(angle))
		if s.botTeleportUnsafeLocked(c, x, y, target) {
			continue
		}
		clearance := s.botTeleportEnemyClearanceLocked(c, x, y, float64(s.battleTime()))
		if !found || clearance > bestClearance {
			bestX, bestY, bestClearance, found = x, y, clearance, true
		}
	}
	return bestX, bestY, found
}

func (s *Server) botTeleportEnemyClearanceLocked(c *conn, x, y float32, now float64) float64 {
	clearance := math.Inf(1)
	visionSources := dotaTeamVisionSourcesLocked(c.inst, c.playerTeam(), now)
	for _, m := range c.inst.mobs {
		if m == nil || m.dead || !m.enemyOf(c.playerTeam()) || !botVisibleEnemyMobLocked(c.playerTeam(), m, visionSources) {
			continue
		}
		clearance = math.Min(clearance, math.Sqrt(float64(dist2(x, y, m.x, m.y))))
	}
	for _, mem := range botLivingEnemyHeroes(c, now) {
		mx, my := mem.posAtLocked(float32(now))
		clearance = math.Min(clearance, math.Sqrt(float64(dist2(x, y, mx, my))))
	}
	return clearance
}

func (s *Server) botTeleportUnsafeLocked(c *conn, x, y float32, target *mobState) bool {
	if c.nav != nil && !c.nav.Walkable(float64(x), float64(y)) {
		return true
	}
	if s.botEnemyStructureDangerLocked(c, x, y) {
		return true
	}
	enemyCount := 0
	visionSources := dotaTeamVisionSourcesLocked(c.inst, c.playerTeam(), float64(s.battleTime()))
	for _, m := range c.inst.mobs {
		if m.dead || !m.enemyOf(c.playerTeam()) || !botVisibleEnemyMobLocked(c.playerTeam(), m, visionSources) {
			continue
		}
		if dist2(x, y, m.x, m.y) <= botTeleportEnemyRadius*botTeleportEnemyRadius {
			enemyCount++
		}
	}
	for _, mem := range botLivingEnemyHeroes(c, float64(s.battleTime())) {
		mx, my := mem.posAtLocked(s.battleTime())
		if dist2(x, y, mx, my) <= botTeleportEnemyRadius*botTeleportEnemyRadius {
			return true
		}
	}
	if enemyCount >= 3 {
		return true
	}
	for _, m := range c.inst.mobs {
		if m.dead || m == target {
			continue
		}
		// Unit collision is a physical constraint, but a hidden enemy creep is
		// not a known tactical obstacle. Treating it as one would let the bot
		// reject a teleport destination because the server knows about a unit
		// that the bot's team does not currently see. Enemy structures remain
		// visible through fog via botVisibleEnemyMobLocked, as they do for the
		// client-side map representation.
		if m.enemyOf(c.playerTeam()) && !botVisibleEnemyMobLocked(c.playerTeam(), m, visionSources) {
			continue
		}
		gap := float32(m.mob.Radius() + c.huntState.av.Radius() + 0.5)
		if dist2(x, y, m.x, m.y) < gap*gap {
			return true
		}
	}
	return false
}

func (s *Server) botTeleportWalkDistanceLocked(c *conn, x, y float32, now float64) float64 {
	cx, cy := c.posAtLocked(float32(now))
	return s.botTeleportWalkDistanceBetweenLocked(c, cx, cy, x, y)
}

func (s *Server) botTeleportWalkDistanceBetweenLocked(c *conn, cx, cy, x, y float32) float64 {
	if c.nav == nil {
		return math.Hypot(float64(x-cx), float64(y-cy))
	}
	path := c.nav.Path(float64(cx), float64(cy), float64(x), float64(y))
	if len(path) == 0 {
		return math.Inf(1)
	}
	distance := 0.0
	px, py := float64(cx), float64(cy)
	for _, p := range path {
		distance += math.Hypot(p.X-px, p.Y-py)
		px, py = p.X, p.Y
	}
	return distance
}

func (s *Server) botTeleportMateriallyFasterLocked(c *conn, x, y float32, walkDistance float64, it gamedata.Item) bool {
	if math.IsInf(walkDistance, 1) || walkDistance <= 0 {
		return false
	}
	speed := float64(c.moveSpeedLocked(s))
	if speed <= 0 {
		return false
	}
	return walkDistance/speed >= it.Preparation+botTeleportMaterialSavings
}
