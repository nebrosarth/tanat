package battleserver

import (
	"math"
	"testing"

	"tanatserver/internal/gamedata"
)

func TestObjectiveStagingRallyPointClearsGunDangerRadius(t *testing.T) {
	s, inst, _, _, cleanup := stickyLaneFixture(t, 4)
	defer cleanup()
	for _, team := range []int32{dotaTeamHuman, dotaTeamElf} {
		objective, _ := s.botMacroLaneObjectiveLocked(inst, team, 0)
		if objective == nil {
			t.Fatalf("setup: team %d missing next enemy objective", team)
		}
		rx, ry, ok := botMobilizationRallyPointLocked(inst, team, objective)
		if !ok {
			t.Fatalf("setup: team %d has no objective staging point", team)
		}
		distance := math.Hypot(float64(rx-objective.x), float64(ry-objective.y))
		if distance < botObjectiveStagingClearance {
			t.Fatalf("team %d staging point is inside objective danger envelope: distance=%.2f clearance=%.2f", team, distance, botObjectiveStagingClearance)
		}
	}
}

func TestAI14BarracksPlanHoldsAgainstSmallNeighborThreat(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	var barracks []*mobState
	for _, mob := range botSortedMobs(inst) {
		if mob.structure && !mob.altar && mob.team == dotaTeamHuman && mob.dotaRole == gamedata.DotaCreepTower && !mob.dead {
			barracks = append(barracks, mob)
		}
	}
	if len(barracks) < 2 {
		t.Fatal("setup: need two allied barracks")
	}
	first, second := barracks[0], barracks[1]
	firstLane := botNearestLaneToPointLocked(inst.dota, first.x, first.y)
	secondLane := botNearestLaneToPointLocked(inst.dota, second.x, second.y)
	if firstLane == secondLane {
		t.Fatalf("setup: barracks share lane %d", firstLane)
	}
	addThreat := func(id int32, barracks *mobState, lane int) {
		creep := teleportTestCreep(inst, id, dotaTeamElf, barracks.x-2, barracks.y)
		creep.lane = inst.dota.m.Lanes[lane]
		creep.laneFwd = false
		creep.laneIdx = len(creep.lane) / 2
		creep.dtarget = barracks.id
		inst.mobs[creep.id] = creep
	}
	addThreat(81401, first, firstLane)
	addThreat(81402, second, secondLane)
	previous := botTeamPlan{Team: dotaTeamHuman, Mode: botMacroBase, Lane: secondLane, ObjectiveID: second.id}
	desired := botTeamPlan{Team: dotaTeamHuman, Mode: botMacroBase, Lane: firstLane, ObjectiveID: first.id}

	if !s.botPlanPremiseValidLocked(inst, dotaTeamHuman, previous, now) {
		t.Fatal("setup: previous barracks defense premise is not live")
	}
	if !s.botRetainTeamPlanLocked(inst, dotaTeamHuman, previous, desired, now) {
		t.Fatal("small neighboring barracks threat incorrectly displaced the current defense")
	}
}

func TestObjectiveStagingKeepsOneLaneWaveEscort(t *testing.T) {
	s, inst, bots, now, cleanup := stickyLaneFixture(t, 4)
	defer cleanup()
	objective, _ := s.botMacroLaneObjectiveLocked(inst, dotaTeamHuman, 0)
	if objective == nil {
		t.Fatal("setup: missing next enemy objective")
	}
	plan := botTeamPlan{
		Team: dotaTeamHuman, Mode: botMacroPush, Lane: 0,
		ObjectiveID: objective.id, Reason: botMacroReasonObjectiveStaging,
		Assignments: make(map[int32]botMacroAssignment, len(bots)),
	}
	for _, brain := range bots {
		plan.Assignments[brain.c.objID] = botMacroAssignment{
			Mode: botMacroLane, Lane: brain.lane, BaselineLane: brain.lane,
		}
		macroSetPosition(brain.c, objective.x, objective.y, now)
	}

	inst.mu.Lock()
	assigned := s.botAssignPlanRespondersLocked(&plan, botTeamPlan{}, false, bots, inst, dotaTeamHuman, now)
	inst.mu.Unlock()
	if !assigned {
		t.Fatal("objective staging did not select a live responder group")
	}
	responders, laneOwners := 0, 0
	for _, brain := range bots {
		a := plan.Assignments[brain.c.objID]
		if a.Mode == botMacroPush || a.Mode == botMacroCover {
			responders++
		} else if a.Mode == botMacroLane {
			laneOwners++
		}
	}
	if responders != len(bots)-1 || laneOwners != 1 {
		t.Fatalf("staging assignments responders=%d lane_owners=%d, want %d and 1: %+v",
			responders, laneOwners, len(bots)-1, plan.Assignments)
	}
}

func TestFreshObjectiveStagesWhenWaveAndLocalCoverageAreReady(t *testing.T) {
	s, inst, bots, now, cleanup := stickyLaneFixture(t, 4)
	defer cleanup()
	objective, _ := s.botMacroLaneObjectiveLocked(inst, dotaTeamHuman, 0)
	if objective == nil || objective.maxHealth() <= 0 {
		t.Fatal("setup: missing full-health enemy objective")
	}
	lane := botNearestLaneToPointLocked(inst.dota, objective.x, objective.y)
	wave := teleportTestCreep(inst, 81403, dotaTeamHuman, objective.x-12, objective.y)
	wave.lane = inst.dota.m.Lanes[lane]
	wave.laneIdx = len(wave.lane) / 2
	wave.hp, wave.maxHP = 500, 500
	inst.mobs[wave.id] = wave
	for _, brain := range bots[:2] {
		macroSetPosition(brain.c, objective.x-18, objective.y, now)
	}

	if !s.botObjectiveStagingRequiredLocked(inst, dotaTeamHuman, objective, botTeamPlan{}, false, now) {
		t.Fatal("fresh objective did not stage despite an allied wave and two local lane bodies")
	}
}

func TestAI14BarracksPlanSwitchesForMateriallyStrongerThreat(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	var barracks []*mobState
	for _, mob := range botSortedMobs(inst) {
		if mob.structure && !mob.altar && mob.team == dotaTeamHuman && mob.dotaRole == gamedata.DotaCreepTower && !mob.dead {
			barracks = append(barracks, mob)
		}
	}
	if len(barracks) < 2 {
		t.Fatal("setup: need two allied barracks")
	}
	first, second := barracks[0], barracks[1]
	firstLane := botNearestLaneToPointLocked(inst.dota, first.x, first.y)
	secondLane := botNearestLaneToPointLocked(inst.dota, second.x, second.y)
	if firstLane == secondLane {
		t.Fatalf("setup: barracks share lane %d", firstLane)
	}
	addThreat := func(id int32, barracks *mobState, lane int) {
		creep := teleportTestCreep(inst, id, dotaTeamElf, barracks.x-2, barracks.y)
		creep.lane = inst.dota.m.Lanes[lane]
		creep.laneFwd = false
		creep.laneIdx = len(creep.lane) / 2
		creep.dtarget = barracks.id
		inst.mobs[creep.id] = creep
	}
	addThreat(81411, first, firstLane)
	for i := 0; i < 3; i++ {
		addThreat(81412+int32(i), second, secondLane)
	}
	previous := botTeamPlan{Team: dotaTeamHuman, Mode: botMacroBase, Lane: firstLane, ObjectiveID: first.id}
	desired := botTeamPlan{Team: dotaTeamHuman, Mode: botMacroBase, Lane: secondLane, ObjectiveID: second.id}

	if !s.botPlanPremiseValidLocked(inst, dotaTeamHuman, previous, now) {
		t.Fatal("setup: previous barracks defense premise is not live")
	}
	if s.botRetainTeamPlanLocked(inst, dotaTeamHuman, previous, desired, now) {
		t.Fatal("materially stronger barracks threat did not replace the current defense")
	}
}

func TestAI17BarracksPlanReleasesOutsideContactRing(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	var barracks *mobState
	for _, mob := range botSortedMobs(inst) {
		if mob.structure && !mob.altar && mob.team == dotaTeamHuman && mob.dotaRole == gamedata.DotaCreepTower && !mob.dead {
			barracks = mob
			break
		}
	}
	if barracks == nil {
		t.Fatal("setup: no allied barracks")
	}
	lane := botNearestLaneToPointLocked(inst.dota, barracks.x, barracks.y)
	creep := teleportTestCreep(inst, 81421, dotaTeamElf, barracks.x+130, barracks.y)
	creep.lane = inst.dota.m.Lanes[lane]
	creep.laneFwd = false
	creep.laneIdx = len(creep.lane) / 2
	inst.mobs[creep.id] = creep

	if got := botBarracksThreatSeverityLocked(inst, dotaTeamHuman, barracks, now); got != 0 {
		t.Fatalf("strict barracks threat = %d, want no contact threat outside 32u", got)
	}
	if got := botBarracksThreatReleaseSeverityLocked(inst, dotaTeamHuman, barracks, now); got != 0 {
		t.Fatalf("release threat = %d, want no direct barracks threat outside contact ring", got)
	}
	plan := botTeamPlan{Team: dotaTeamHuman, Mode: botMacroBase, Lane: lane, ObjectiveID: barracks.id}
	if s.botPlanPremiseValidLocked(inst, dotaTeamHuman, plan, now) {
		t.Fatal("barracks defense premise remained valid outside contact ring")
	}
}

func TestCriticalFinishPreservesPushAgainstUndamagedBarracksContact(t *testing.T) {
	s, inst, bots, now, cleanup := stickyLaneFixture(t, 4)
	defer cleanup()

	objective, _ := s.botMacroLaneObjectiveLocked(inst, dotaTeamHuman, 0)
	if objective == nil {
		t.Fatal("setup: missing enemy front objective")
	}
	objective.hp = objective.maxHealth() * 0.50
	var barracks *mobState
	for _, mob := range botSortedMobs(inst) {
		if mob.structure && !mob.altar && mob.team == dotaTeamHuman && mob.dotaRole == gamedata.DotaCreepTower && !mob.dead {
			barracks = mob
			break
		}
	}
	if barracks == nil {
		t.Fatal("setup: missing allied barracks")
	}
	previous := botTeamPlan{
		Team: dotaTeamHuman, Mode: botMacroPush, Lane: 0, ObjectiveID: objective.id,
		Reason: botMacroReasonFullMobilization, Assignments: make(map[int32]botMacroAssignment),
	}
	for _, brain := range bots {
		previous.Assignments[brain.c.objID] = botMacroAssignment{
			Mode: botMacroPush, Role: "push", Lane: 0, ObjectiveID: objective.id,
		}
	}
	if !s.botPreserveCriticalFinishLocked(inst, dotaTeamHuman, previous, true, bots, barracks, now) {
		t.Fatal("finish commitment was discarded by an undamaged barracks contact")
	}
	barracks.hp = barracks.maxHealth() * 0.80
	if s.botPreserveCriticalFinishLocked(inst, dotaTeamHuman, previous, true, bots, barracks, now) {
		t.Fatal("finish commitment survived materially damaged barracks")
	}
}

func TestTeamFocusRalliesAnAlreadySupportedVisibleTarget(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	ally := dotaPlayerConn(t, s, inst, 81431, dotaTeamHuman, 28, 0)
	target := dotaPlayerConn(t, s, inst, 81432, dotaTeamElf, 40, 0)
	bot.x, bot.y, bot.snapT = 0, 0, float32(now)
	ally.snapT = float32(now)
	target.snapT = float32(now)
	inst.bots[bot.objID] = &botBrain{c: bot, lane: 0, phase: botPhaseGroup,
		macroAssignment: botMacroAssignment{Mode: botMacroPush, Aggressive: true}}
	inst.dota.teamPlans = map[int32]botTeamPlan{
		dotaTeamHuman: {Team: dotaTeamHuman, FocusTarget: target.objID},
	}

	brain := inst.bots[bot.objID]
	inst.mu.Lock()
	acted := s.botCombatTickLocked(brain, now)
	inst.mu.Unlock()
	if !acted {
		t.Fatal("supported visible focus did not produce a rally order")
	}
	if !bot.hasDest || bot.destX != target.x || bot.destY != target.y {
		t.Fatalf("focus rally destination=(%.1f,%.1f,%v), want target=(%.1f,%.1f)",
			bot.destX, bot.destY, bot.hasDest, target.x, target.y)
	}
	if bot.huntState.pvpTarget != 0 {
		t.Fatalf("focus rally started a solo attack before entering local range: target=%d", bot.huntState.pvpTarget)
	}
}

func TestPushGroupRescuesNearbyAllyUnderHeroAttack(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	ally := dotaPlayerConn(t, s, inst, 81433, dotaTeamHuman, bot.x+28, bot.y)
	enemy := dotaPlayerConn(t, s, inst, 81434, dotaTeamElf, bot.x+40, bot.y)
	bot.snapT, ally.snapT, enemy.snapT = float32(now), float32(now), float32(now)
	enemy.huntState.attackTarget = ally.objID
	inst.bots[bot.objID] = &botBrain{c: bot, lane: 0, phase: botPhaseGroup,
		macroAssignment: botMacroAssignment{Mode: botMacroPush, Aggressive: true}}

	bot.lock()
	acted := s.botCombatTickLocked(inst.bots[bot.objID], now)
	bot.unlock()
	if !acted {
		t.Fatal("push group ignored an active hero attack on a nearby ally")
	}
	if !bot.hasDest || bot.destX != enemy.x || bot.destY != enemy.y {
		t.Fatalf("rescue destination=(%.1f,%.1f,%v), want attacker=(%.1f,%.1f)",
			bot.destX, bot.destY, bot.hasDest, enemy.x, enemy.y)
	}
	if bot.huntState.pvpTarget != 0 {
		t.Fatalf("rescue movement started a premature solo attack: target=%d", bot.huntState.pvpTarget)
	}
}

func TestCounterPushFinishesDamagedObjectiveBeforeRetreat(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime()) + 10
	brain := macroAddBot(t, s, inst, botIDBase+91, dotaTeamHuman, 0, 0, 0)
	bot := brain.c
	objective := structOfSide(inst, gamedata.DotaGun, dotaTeamElf)
	if objective == nil {
		t.Fatal("setup: missing enemy gun")
	}
	objective.hp = objective.maxHealth() * 0.50
	bot.x, bot.y, bot.snapT = objective.x, objective.y, float32(now)
	brain.macroAssignment = botMacroAssignment{
		Mode: botMacroPush, ObjectiveID: objective.id, Role: botMacroCounterPushRole,
		Reason: "base_pressure_counter_push", Aggressive: true,
	}
	bot.huntState.hp = bot.huntState.maxHPLocked(now) * 0.40

	bot.lock()
	if s.botShouldRetreatLocked(brain, now) {
		bot.unlock()
		t.Fatal("counter-pusher abandoned a damaged objective before the hard retreat floor")
	}
	if brain.retreating {
		bot.unlock()
		t.Fatal("counter-pusher entered retreat despite an objective in the finish window")
	}
	bot.huntState.hp = bot.huntState.maxHPLocked(now) * 0.25
	if !s.botShouldRetreatLocked(brain, now) || brain.retreatMode != botRetreatModeRecovery {
		bot.unlock()
		t.Fatal("counter-pusher did not recover after crossing the hard retreat floor")
	}
	bot.unlock()
}

func TestCommittedPushSurvivesGunThreatAfterCheckpointDestroy(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime()) + 10
	for i := int32(0); i < 3; i++ {
		brain := macroAddBot(t, s, inst, botIDBase+120+i, dotaTeamHuman, 0, 0, 0)
		brain.c.snapT = float32(now)
	}

	var lane int
	var destroyed *mobState
	for candidateLane := range inst.dota.m.Lanes {
		if objective, _ := s.botMacroLaneObjectiveLocked(inst, dotaTeamHuman, candidateLane); objective != nil {
			lane, destroyed = candidateLane, objective
			break
		}
	}
	if destroyed == nil {
		t.Fatal("setup: missing enemy lane objective")
	}
	destroyed.dead = true
	destroyed.hp = 0
	ownGun := structOfSide(inst, gamedata.DotaGun, dotaTeamHuman)
	if ownGun == nil {
		t.Fatal("setup: missing own gun")
	}
	bots := make([]*botBrain, 0, 3)
	for _, brain := range inst.bots {
		if brain != nil && brain.c != nil && brain.c.playerTeam() == dotaTeamHuman {
			bots = append(bots, brain)
		}
	}
	previous := botTeamPlan{
		Mode: botMacroPush, Lane: lane, ObjectiveID: destroyed.id,
		Reason: "objective_conversion_ready",
	}
	if !s.botPreserveCommittedPushAgainstGunLocked(inst, dotaTeamHuman, previous, true, bots, ownGun, 1, now) {
		nextLane, nextObjective, progress, coverage, ok := s.botContinuePushLaneLocked(inst, dotaTeamHuman, previous, true, now)
		t.Fatalf("committed push reset to base after destroying its checkpoint: nextLane=%d nextObjective=%v progress=%v coverage=%d ok=%v healthy=%d", nextLane, nextObjective, progress, coverage, ok, s.botHealthyMacroResponderCountLocked(bots, now))
	}
}

func TestPartialMobilizationPreservesSingleRecoverableGunFinisher(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime()) + 10
	brain := macroAddBot(t, s, inst, botIDBase+140, dotaTeamHuman, 0, 0, 0)
	brain.c.snapT = float32(now)

	var objective *mobState
	for lane := range inst.dota.m.Lanes {
		if candidate, _ := s.botMacroLaneObjectiveLocked(inst, dotaTeamHuman, lane); candidate != nil {
			objective = candidate
			break
		}
	}
	if objective == nil {
		t.Fatal("setup: missing enemy lane objective")
	}
	objective.hp = objective.maxHealth() * 0.50
	ownGun := structOfSide(inst, gamedata.DotaGun, dotaTeamHuman)
	if ownGun == nil {
		t.Fatal("setup: missing own gun")
	}
	previous := botTeamPlan{
		Team: dotaTeamHuman, Mode: botMacroPush, Lane: 0, ObjectiveID: objective.id,
		Reason: botMacroReasonPartialMobilization,
		Assignments: map[int32]botMacroAssignment{
			brain.c.objID: {Mode: botMacroPush, Role: "push", Lane: 0, ObjectiveID: objective.id},
		},
	}
	if !s.botPreserveCommittedPushAgainstGunLocked(inst, dotaTeamHuman, previous, true, []*botBrain{brain}, ownGun, 1, now) {
		t.Fatal("partial mobilization dropped its only recoverable finisher")
	}
}

func TestBotAnswersActiveHeroPressureBeforeObjectiveRetreat(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	enemy := dotaPlayerConn(t, s, inst, 81435, dotaTeamElf, bot.x+22, bot.y)
	bot.x, bot.y, bot.snapT = 0, 0, float32(now)
	enemy.x, enemy.y, enemy.snapT = 22, 0, float32(now)
	enemy.huntState.pvpTarget = bot.objID
	inst.bots[bot.objID] = &botBrain{c: bot, lane: 0, phase: botPhaseGroup,
		macroAssignment: botMacroAssignment{Mode: botMacroPush, Aggressive: true}}

	bot.lock()
	acted := s.botCombatTickLocked(inst.bots[bot.objID], now)
	bot.unlock()
	if !acted {
		t.Fatal("bot did not answer an active hero attacker outside immediate attack range")
	}
	if !bot.hasDest || bot.destX != enemy.x || bot.destY != enemy.y {
		t.Fatalf("active-threat destination=(%.1f,%.1f,%v), want attacker=(%.1f,%.1f)",
			bot.destX, bot.destY, bot.hasDest, enemy.x, enemy.y)
	}
}

func TestDamagedObjectiveKeepsSupportedGroupThroughCommittedGunShot(t *testing.T) {
	s, inst, bots, now, cleanup := stickyLaneFixture(t, 3)
	defer cleanup()
	objective, _ := s.botMacroLaneObjectiveLocked(inst, dotaTeamHuman, 0)
	if objective == nil || objective.maxHealth() <= 0 {
		t.Fatal("setup: missing enemy front objective")
	}
	objective.hp = objective.maxHealth() * 0.50
	for _, brain := range bots {
		macroSetPosition(brain.c, objective.x, objective.y, now)
		brain.macroAssignment = botMacroAssignment{
			Mode: botMacroPush, Lane: 0, ObjectiveID: objective.id,
			Aggressive: true, Role: "push_support",
		}
	}
	objective.hitAt = now + 1
	objective.hitTarget = bots[0].c.objID

	if !s.botCanCommitStructureFocusLocked(bots[0], objective, now) {
		t.Fatal("supported damaged-objective group abandoned the finish after a committed gun shot")
	}
}

func TestConversionReadyGroupKeepsFullObjectiveThroughCommittedGunShot(t *testing.T) {
	s, inst, bots, now, cleanup := stickyLaneFixture(t, 3)
	defer cleanup()
	objective, _ := s.botMacroLaneObjectiveLocked(inst, dotaTeamHuman, 0)
	if objective == nil || objective.maxHealth() <= 0 {
		t.Fatal("setup: missing enemy front objective")
	}
	// The conversion predicate may be true when the wave arrives and false one
	// tick later after the wave dies. The assigned group must keep the attack
	// order instead of resetting at a full-health gun and repeating the same
	// approach forever.
	for _, brain := range bots {
		macroSetPosition(brain.c, objective.x, objective.y, now)
		brain.macroAssignment = botMacroAssignment{
			Mode: botMacroPush, Lane: 0, ObjectiveID: objective.id,
			Reason: "objective_conversion_ready", Aggressive: true,
			Role: "push_support",
		}
	}
	objective.hitAt = now + 1
	objective.hitTarget = bots[0].c.objID

	if !s.botCanCommitStructureFocusLocked(bots[0], objective, now) {
		t.Fatal("conversion-ready objective group abandoned the full-health gun after a committed shot")
	}
}

func TestPartialMobilizationKeepsDamagedObjectiveThroughCommittedGunShot(t *testing.T) {
	s, inst, bots, now, cleanup := stickyLaneFixture(t, 3)
	defer cleanup()
	objective, _ := s.botMacroLaneObjectiveLocked(inst, dotaTeamHuman, 0)
	if objective == nil || objective.maxHealth() <= 0 {
		t.Fatal("setup: missing enemy front objective")
	}
	// A partial strike group is allowed to finish an invested objective even
	// when a neighbouring gun has just committed a shot. Without this lease the
	// group repeatedly retreats from cross-fire and never reaches the named gun.
	objective.hp = objective.maxHealth() * 0.50
	for _, brain := range bots {
		macroSetPosition(brain.c, objective.x, objective.y, now)
		brain.macroAssignment = botMacroAssignment{
			Mode: botMacroPush, Lane: 0, ObjectiveID: objective.id,
			Reason: botMacroReasonPartialMobilization, Aggressive: true,
			Role: "push_support",
		}
	}
	objective.hitAt = now + 1
	objective.hitTarget = bots[0].c.objID

	if !s.botCanCommitStructureFocusLocked(bots[0], objective, now) {
		t.Fatal("partial mobilization abandoned a damaged objective after a committed gun shot")
	}
}

func TestConversionReadyGroupLeavesFullObjectiveDuringSustainedGunDamage(t *testing.T) {
	s, inst, bots, now, cleanup := stickyLaneFixture(t, 3)
	defer cleanup()
	objective, _ := s.botMacroLaneObjectiveLocked(inst, dotaTeamHuman, 0)
	if objective == nil || objective.maxHealth() <= 0 {
		t.Fatal("setup: missing enemy front objective")
	}
	for _, brain := range bots {
		macroSetPosition(brain.c, objective.x, objective.y, now)
		brain.macroAssignment = botMacroAssignment{
			Mode: botMacroPush, Lane: 0, ObjectiveID: objective.id,
			Reason: "objective_conversion_ready", Aggressive: true,
			Role: "push_support",
		}
	}
	objective.hitAt = now + 1
	objective.hitTarget = bots[0].c.objID
	bots[0].c.huntState.hp = bots[0].c.huntState.maxHPLocked(now) * 0.75
	bots[0].hpHistIdx = 0
	bots[0].hpHistory[0] = hpSample{t: now - 0.5, frac: 0.95}

	if s.botCanCommitStructureFocusLocked(bots[0], objective, now) {
		t.Fatal("conversion-ready group kept diving a full-health gun during sustained damage")
	}
}

func TestDamagedConversionDebtDoesNotRequireWaveOrSecondBody(t *testing.T) {
	s, inst, bots, now, cleanup := stickyLaneFixture(t, 3)
	defer cleanup()
	objective, _ := s.botMacroLaneObjectiveLocked(inst, dotaTeamHuman, 0)
	if objective == nil || objective.maxHealth() <= 0 {
		t.Fatal("setup: missing enemy front objective")
	}
	objective.hp = objective.maxHealth() * 0.50
	macroSetPosition(bots[0].c, objective.x, objective.y, now)
	bots[0].macroAssignment = botMacroAssignment{
		Mode: botMacroPush, Lane: 0, ObjectiveID: objective.id,
		Reason: "objective_conversion_ready", Aggressive: true,
		Role: "push",
	}
	for _, brain := range bots[1:] {
		brain.macroAssignment = botMacroAssignment{Mode: botMacroLane, Lane: brain.lane}
	}
	objective.hitAt = now + 1
	objective.hitTarget = bots[0].c.objID

	if botObjectiveFinishGroupReadyLocked(inst, dotaTeamHuman, objective, now) {
		t.Fatal("setup unexpectedly has a second assigned body in objective range")
	}
	if !s.botCanCommitStructureFocusLocked(bots[0], objective, now) {
		t.Fatal("damaged conversion pusher abandoned the objective when its wave/group briefly disappeared")
	}

	bots[0].c.huntState.hp = bots[0].c.huntState.maxHPLocked(now) * (botRetreatHPFrac - 0.01)
	if s.botCanCommitStructureFocusLocked(bots[0], objective, now) {
		t.Fatal("conversion debt ignored the hard retreat HP floor")
	}
}

func TestConversionReasonRestoresObjectivePriorityWithoutAggressiveBit(t *testing.T) {
	s, inst, bots, now, cleanup := stickyLaneFixture(t, 3)
	defer cleanup()
	objective, _ := s.botMacroLaneObjectiveLocked(inst, dotaTeamHuman, 0)
	if objective == nil {
		t.Fatal("setup: missing enemy front objective")
	}
	macroSetPosition(bots[0].c, objective.x, objective.y, now)
	bots[0].macroAssignment = botMacroAssignment{
		Mode: botMacroPush, Lane: 0, ObjectiveID: objective.id,
		Reason: "objective_conversion_ready", Aggressive: false,
	}
	if !s.botPushObjectiveHasPriorityLocked(bots[0], now) {
		t.Fatal("conversion-ready push lost objective priority when its responder flag was stale")
	}
}

func TestExposedDamagedAltarWinsOverOptionalAllyFight(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_HK_Astarot")
	defer cleanup()
	now := float64(s.battleTime())
	altar := altarOf(inst, dotaTeamElf)
	if altar == nil || altar.maxHealth() <= 0 {
		t.Fatal("setup: missing enemy altar")
	}
	for _, gunID := range inst.dota.altarGuards[altar.id] {
		if gun := inst.mobs[gunID]; gun != nil {
			gun.dead = true
		}
	}
	altar.hp = altar.maxHealth() * 0.50
	macroSetPosition(bot, altar.x+1, altar.y, now)
	brain := &botBrain{c: bot, lane: 0}
	brain.macroAssignment = botMacroAssignment{
		Mode: botMacroAltar, ObjectiveID: altar.id, Reason: "enemy_altar_open",
		Aggressive: true, Role: "assault",
	}

	bot.lock()
	defer bot.unlock()
	if !s.botAltarFinishPriorityLocked(brain) {
		t.Fatal("damaged exposed altar did not enter terminal priority")
	}
	if !s.botCombatTickLocked(brain, now) || bot.huntState.attackTarget != altar.id {
		t.Fatalf("terminal altar was not selected: attackTarget=%d", bot.huntState.attackTarget)
	}
}

func TestFullObjectiveWaitsForAlliedWave(t *testing.T) {
	s, inst, bots, now, cleanup := stickyLaneFixture(t, 3)
	defer cleanup()
	objective, _ := s.botMacroLaneObjectiveLocked(inst, dotaTeamHuman, 0)
	if objective == nil || objective.maxHealth() <= 0 {
		t.Fatal("setup: missing enemy front objective")
	}
	for _, mob := range inst.mobs {
		if mob != nil && !mob.structure && mob.team == dotaTeamHuman {
			mob.dead = true
		}
	}
	for _, brain := range bots {
		brain.c.huntState.hp = brain.c.huntState.maxHPLocked(now)
		macroSetPosition(brain.c, objective.x, objective.y, now)
	}
	if s.botObjectiveConversionReadyLocked(inst, dotaTeamHuman, objective, now) {
		t.Fatal("full-health objective became conversion-ready without an allied wave")
	}
}

func TestDamagedObjectiveKeepsAllSelectedRespondersAggressive(t *testing.T) {
	s, inst, bots, now, cleanup := stickyLaneFixture(t, 3)
	defer cleanup()
	objective, _ := s.botMacroLaneObjectiveLocked(inst, dotaTeamHuman, 0)
	if objective == nil || objective.maxHealth() <= 0 {
		t.Fatal("setup: missing enemy front objective")
	}
	objective.hp = objective.maxHealth() * 0.50
	plan := botTeamPlan{
		Team: dotaTeamHuman, Mode: botMacroPush, Lane: 0,
		ObjectiveID: objective.id, Reason: "objective_conversion_ready",
		Assignments: make(map[int32]botMacroAssignment, len(bots)),
	}
	for _, brain := range bots {
		macroSetPosition(brain.c, objective.x, objective.y, now)
		plan.Assignments[brain.c.objID] = botMacroAssignment{
			Mode: botMacroLane, Lane: brain.lane, BaselineLane: brain.lane,
		}
	}

	inst.mu.Lock()
	if !s.botAssignMacroRespondersLocked(&plan, botTeamPlan{}, false, bots, objective.x, objective.y, len(bots), 2, now) {
		inst.mu.Unlock()
		t.Fatal("setup: damaged objective did not produce responders")
	}
	for _, brain := range bots {
		assignment := plan.Assignments[brain.c.objID]
		if assignment.Mode != botMacroPush || !assignment.Aggressive {
			inst.mu.Unlock()
			t.Fatalf("bot %d left damaged-objective finish as non-aggressive assignment: %+v", brain.c.objID, assignment)
		}
	}
	inst.mu.Unlock()
}
