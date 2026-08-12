package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

func TestAI13OrchestratorDistributesFarmLanes(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	lane := inst.dota.m.Lanes[0]
	x, y := float32(lane[len(lane)/2].X), float32(lane[len(lane)/2].Y)
	bots := []*botBrain{
		macroAddBot(t, s, inst, botIDBase+1, dotaTeamHuman, x, y, 0),
		macroAddBot(t, s, inst, botIDBase+2, dotaTeamHuman, x, y, 0),
		macroAddBot(t, s, inst, botIDBase+3, dotaTeamHuman, x, y, 0),
	}
	plan := botTeamPlan{Team: dotaTeamHuman, Assignments: map[int32]botMacroAssignment{}}
	for _, bot := range bots {
		plan.Assignments[bot.c.objID] = botMacroAssignment{
			Mode: botMacroCover, Lane: 0, BaselineLane: 0, Role: "cover",
		}
	}
	s.botAssignFarmLanesLocked(&plan, bots, inst)

	for i, bot := range bots {
		assignment := plan.Assignments[bot.c.objID]
		if !assignment.FarmLaneSet {
			t.Fatalf("bot %d has no orchestrator farm-lane decision", bot.c.objID)
		}
		if assignment.FarmLane != i {
			t.Fatalf("bot %d farm lane = %d, want deterministic spread lane %d", bot.c.objID, assignment.FarmLane, i)
		}
	}
	_ = now
}

func TestAI13CoverageExecutesOrchestratorFarmLane(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_DPS_Gellar")
	defer cleanup()
	now := float64(s.battleTime())
	lane := inst.dota.m.Lanes[0]
	x, y := float32(lane[len(lane)/2].X), float32(lane[len(lane)/2].Y)
	owner := macroAddBot(t, s, inst, botIDBase+11, dotaTeamHuman, x, y, 0)
	cover := macroAddBot(t, s, inst, botIDBase+12, dotaTeamHuman, x, y, 0)
	plan := botTeamPlan{Team: dotaTeamHuman, Assignments: map[int32]botMacroAssignment{
		owner.c.objID: {Mode: botMacroLane, Lane: 0, BaselineLane: 0},
		cover.c.objID: {Mode: botMacroCover, Lane: 0, BaselineLane: 0},
	}}
	s.botAssignFarmLanesLocked(&plan, []*botBrain{owner, cover}, inst)
	cover.macroAssignment = plan.Assignments[cover.c.objID]

	cover.c.lock()
	s.botCoverageTickLocked(cover, now)
	cover.c.unlock()

	if cover.farmLane != 1 {
		t.Fatalf("coverage farm lane = %d, want orchestrator-assigned lane 1", cover.farmLane)
	}
	if !cover.c.hasDest {
		t.Fatal("coverage bot did not receive a movement destination")
	}
	wantX, wantY := s.botPushPointLocked(cover, 1, now)
	if cover.c.destX != wantX || cover.c.destY != wantY {
		t.Fatalf("coverage destination = (%.1f, %.1f), want lane 1 push point (%.1f, %.1f)", cover.c.destX, cover.c.destY, wantX, wantY)
	}
}

func TestAI13ObjectiveAssignmentDoesNotReceiveFarmOverlay(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	lane := inst.dota.m.Lanes[0]
	x, y := float32(lane[len(lane)/2].X), float32(lane[len(lane)/2].Y)
	objective := macroAddBot(t, s, inst, botIDBase+21, dotaTeamHuman, x, y, 0)
	plan := botTeamPlan{Team: dotaTeamHuman, Assignments: map[int32]botMacroAssignment{
		objective.c.objID: {Mode: botMacroAltar, Lane: 0, BaselineLane: 0, Role: "assault"},
	}}
	s.botAssignFarmLanesLocked(&plan, []*botBrain{objective}, inst)
	if assignment := plan.Assignments[objective.c.objID]; assignment.FarmLaneSet {
		t.Fatalf("objective assignment unexpectedly received farm overlay: %+v", assignment)
	}
}

func TestAI13PushAssignmentWalksToItsExactStructureObjective(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	objective := structOfSide(inst, gamedata.DotaGun, dotaTeamElf)
	if objective == nil {
		t.Fatal("missing enemy gun objective")
	}
	bot := macroAddBot(t, s, inst, botIDBase+31, dotaTeamHuman, objective.x-30, objective.y, 0)
	bot.macroAssignment = botMacroAssignment{
		Mode: botMacroPush, Lane: 0, ObjectiveID: objective.id, Role: "push",
	}

	bot.c.lock()
	defer bot.c.unlock()
	if !s.botMacroObjectiveTickLocked(bot, now) {
		t.Fatal("push assignment did not accept its live enemy structure objective")
	}
	if !bot.c.hasDest {
		t.Fatal("push bot did not move toward the assigned structure")
	}
	if bot.c.destX != objective.x || bot.c.destY != objective.y {
		t.Fatalf("push destination = (%.1f, %.1f), want exact objective (%.1f, %.1f)",
			bot.c.destX, bot.c.destY, objective.x, objective.y)
	}
	bot.c.stopArrivalLocked()
	bot.c.x, bot.c.y, bot.c.hasDest = objective.x, objective.y, false
	if !s.botMacroObjectiveTickLocked(bot, now) || bot.c.huntState.attackTarget != objective.id {
		t.Fatalf("push bot did not attack its assigned structure at reach: acted=%v target=%d want=%d",
			bot.c.huntState.attackTarget != 0, bot.c.huntState.attackTarget, objective.id)
	}
}
