package battleserver

import "testing"

func TestAI20FarmDebtPrioritizesBotWithFewerCreepXPEvents(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	bots := []*botBrain{
		macroAddBot(t, s, inst, botIDBase+81, dotaTeamHuman, 0, 0, 0),
		macroAddBot(t, s, inst, botIDBase+82, dotaTeamHuman, 0, 0, 1),
		macroAddBot(t, s, inst, botIDBase+83, dotaTeamHuman, 0, 0, 2),
	}
	bots[0].farmXPEvents = 4
	bots[1].farmXPEvents = 0
	bots[1].c.huntState.level = 5
	bots[2].farmXPEvents = 2
	bots[2].c.huntState.level = 1
	eligible := make([]*botBrain, len(bots))
	copy(eligible, bots)

	if got := botWeakestEligibleFarmBotLocked(inst, dotaTeamHuman, eligible); got != bots[1] {
		t.Fatalf("farm priority selected bot %d, want debt-heavy bot %d", got.c.objID, bots[1].c.objID)
	}
	if got := botFarmDebtLocked(inst, bots[1]); got != 4 {
		t.Fatalf("farm debt = %d, want 4 events behind the team leader", got)
	}
}

func TestAI20FarmRescueDependsOnLiveWaveAndActualXPState(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	bots := []*botBrain{
		macroAddBot(t, s, inst, botIDBase+91, dotaTeamHuman, 0, 0, 0),
		macroAddBot(t, s, inst, botIDBase+92, dotaTeamHuman, 0, 0, 1),
	}
	lane := inst.dota.m.Lanes[0]
	wave := teleportTestCreep(inst, botIDBase+191, dotaTeamElf, float32(lane[len(lane)/2].X), float32(lane[len(lane)/2].Y))
	wave.lane = lane
	wave.laneIdx = len(lane) / 2
	inst.mobs[wave.id] = wave
	for _, b := range bots {
		b.c.x, b.c.y, b.c.snapT = wave.x, wave.y, float32(now)
	}

	bots[0].farmXPEvents = 1
	if !s.botTeamFarmRescueRequiredLocked(inst, dotaTeamHuman) {
		t.Fatal("farm rescue did not activate while a living bot had zero creep XP")
	}
	bots[1].farmXPEvents = 1
	if s.botTeamFarmRescueRequiredLocked(inst, dotaTeamHuman) {
		t.Fatal("farm rescue stayed active after every living bot received creep XP")
	}
	wave.dead = true
	bots[1].farmXPEvents = 0
	if s.botTeamFarmRescueRequiredLocked(inst, dotaTeamHuman) {
		t.Fatal("farm rescue activated without a live wave")
	}
}

func TestAI20FarmRescueCanReserveDebtBotDuringObjectiveFinish(t *testing.T) {
	s, inst, bots, now, cleanup := stickyLaneFixture(t, 4)
	defer cleanup()
	objective, _ := s.botMacroLaneObjectiveLocked(inst, dotaTeamHuman, 0)
	if objective == nil {
		t.Fatal("setup: missing enemy objective")
	}
	objective.hp = objective.maxHealth() * 0.50
	for _, brain := range bots {
		brain.c.huntState.hp = brain.c.huntState.maxHPLocked(now)
		macroSetPosition(brain.c, objective.x, objective.y, now)
	}
	for _, mob := range inst.mobs {
		if mob != nil && !mob.structure && mob.team == dotaTeamHuman && botTeleportLaneCreep(mob) {
			mob.x, mob.y = objective.x, objective.y
			mob.hp = mob.maxHP
		}
	}
	bots[0].farmXPEvents = 10
	bots[1].farmXPEvents = 10
	bots[2].farmXPEvents = 10
	bots[3].farmXPEvents = 0

	if !s.botObjectiveConversionReadyExcludingLocked(inst, dotaTeamHuman, objective, bots[3].c.objID, now) {
		t.Fatal("setup: objective was not convertible after reserving the debt-heavy bot")
	}
	if !s.botTeamFarmRescueRequiredLocked(inst, dotaTeamHuman) {
		t.Fatal("farm rescue did not reserve debt-heavy bot while the finish group remained viable")
	}
}

func TestAI21FarmReserveRemovesDebtBotFromOrdinaryPush(t *testing.T) {
	s, inst, bots, now, cleanup := stickyLaneFixture(t, 4)
	defer cleanup()
	objective, _ := s.botMacroLaneObjectiveLocked(inst, dotaTeamHuman, 0)
	if objective == nil {
		t.Fatal("setup: missing enemy objective")
	}
	objective.hp = objective.maxHealth() * 0.50
	for _, brain := range bots {
		brain.c.huntState.hp = brain.c.huntState.maxHPLocked(now)
		macroSetPosition(brain.c, objective.x, objective.y, now)
	}
	for _, mob := range inst.mobs {
		if mob != nil && !mob.structure && mob.team == dotaTeamHuman && botTeleportLaneCreep(mob) {
			mob.x, mob.y = objective.x, objective.y
			mob.hp = mob.maxHP
		}
	}
	wave := teleportTestCreep(inst, botIDBase+901, dotaTeamElf, objective.x, objective.y)
	wave.lane = inst.dota.m.Lanes[0]
	wave.laneIdx = len(wave.lane) / 2
	inst.mobs[wave.id] = wave
	for i := range bots {
		bots[i].farmXPEvents = 10
	}
	bots[3].farmXPEvents = 0
	plan := botTeamPlan{
		Team: dotaTeamHuman, Mode: botMacroPush, Lane: 0,
		ObjectiveID: objective.id, Reason: "objective_conversion_ready",
		Assignments: map[int32]botMacroAssignment{},
	}
	for _, brain := range bots {
		plan.Assignments[brain.c.objID] = botMacroAssignment{
			Mode: botMacroPush, Lane: 0, BaselineLane: brain.lane,
			ObjectiveID: objective.id, Role: "push", Reason: plan.Reason,
		}
	}
	if !s.botAssignFarmReserveLocked(&plan, bots, inst, now) {
		t.Fatal("farm reserve did not activate for the debt-heavy bot")
	}
	reserved := plan.Assignments[bots[3].c.objID]
	if reserved.Mode != botMacroLane || reserved.Role != "farm_reserve" || !reserved.FarmLaneSet {
		t.Fatalf("debt-heavy assignment = %+v, want lane/farm_reserve with a live farm lane", reserved)
	}
	for _, brain := range bots[:3] {
		if plan.Assignments[brain.c.objID].Mode != botMacroPush {
			t.Fatalf("strategic bot %d was removed with the farm reserve", brain.c.objID)
		}
	}
}

func TestAI20CreepXPGrantUpdatesFarmDebtState(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	b := newBotBrain(bot, 0, 0)
	inst.bots[bot.objID] = b
	if b.farmXPEvents != 0 || b.farmLastXPTAt != 0 {
		t.Fatalf("initial farm XP state = events %d/last %.1f", b.farmXPEvents, b.farmLastXPTAt)
	}
	s.recordBotCreepXPLocked(bot, 42)
	if b.farmXPEvents != 1 || b.farmLastXPTAt != 42 {
		t.Fatalf("updated farm XP state = events %d/last %.1f, want 1/42", b.farmXPEvents, b.farmLastXPTAt)
	}
}

func TestAI20FarmWaveValueBeatsGoldLastHit(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	lane := inst.dota.m.Lanes[0]
	bot.x, bot.y, bot.snapT = float32(lane[len(lane)/2].X), float32(lane[len(lane)/2].Y), float32(now)
	b := &botBrain{c: bot, lane: 0, phase: botPhaseLane}

	cheap := teleportTestCreep(inst, botIDBase+201, dotaTeamElf, bot.x+1, bot.y)
	cheap.hp, cheap.maxHP = 1, 1
	cheap.lane, cheap.laneIdx = lane, len(lane)/2
	inst.mobs[cheap.id] = cheap
	for i, dx := range []float32{8, 9, 10} {
		wave := teleportTestCreep(inst, botIDBase+202+int32(i), dotaTeamElf, bot.x+dx, bot.y)
		wave.lane, wave.laneIdx = lane, len(lane)/2
		inst.mobs[wave.id] = wave
	}

	bot.lock()
	defer bot.unlock()
	target := s.botFarmTargetLocked(b, now, 20, false)
	if target == nil {
		t.Fatal("farm director returned no live XP target")
	}
	if target == cheap || b.farmDecision == "last_hit" {
		t.Fatalf("farm director still optimized a gold last hit: target=%d decision=%q", target.id, b.farmDecision)
	}
}
