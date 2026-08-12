package battleserver

import "testing"

func TestAI15OrchestratorSendsWeakestBotToLiveFarmWave(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	lane := inst.dota.m.Lanes[2]
	wave := teleportTestCreep(inst, 81501, dotaTeamElf, 0, 0)
	wave.lane = lane
	wave.laneIdx = len(lane) / 2
	inst.mobs[wave.id] = wave
	waveLane := botLaneForCreep(&conn{inst: inst}, wave)
	bots := []*botBrain{
		macroAddBot(t, s, inst, botIDBase+31, dotaTeamHuman, 0, 0, 0),
		macroAddBot(t, s, inst, botIDBase+32, dotaTeamHuman, 0, 0, 1),
		macroAddBot(t, s, inst, botIDBase+33, dotaTeamHuman, 0, 0, 2),
	}
	for _, bot := range bots {
		bot.macroAssignment = botMacroAssignment{Mode: botMacroLane, Lane: bot.lane, BaselineLane: bot.lane}
	}
	bots[0].c.huntState.level = 1
	bots[0].c.huntState.xp = 100
	bots[1].c.huntState.level = 4
	bots[1].c.huntState.xp = 900
	bots[2].c.huntState.level = 5
	bots[2].c.huntState.xp = 1200
	plan := botTeamPlan{Team: dotaTeamHuman, Assignments: map[int32]botMacroAssignment{}}
	for _, bot := range bots {
		plan.Assignments[bot.c.objID] = bot.macroAssignment
	}
	s.botAssignFarmLanesLocked(&plan, bots, inst)

	if got := plan.Assignments[bots[0].c.objID].FarmLane; got != waveLane {
		t.Fatalf("weakest bot farm lane = %d, want live-wave lane %d (scores lane0=%d lane1=%d lane2=%d weak=%d)", got, waveLane,
			botFarmLaneWaveScoreLocked(inst, dotaTeamHuman, 0), botFarmLaneWaveScoreLocked(inst, dotaTeamHuman, 1), botFarmLaneWaveScoreLocked(inst, dotaTeamHuman, 2), botWeakestLivingBotLocked(inst, dotaTeamHuman).c.objID)
	}
	if !plan.Assignments[bots[0].c.objID].FarmLaneSet {
		t.Fatal("weakest bot farm rescue was not recorded by orchestrator")
	}
	_ = now
}
