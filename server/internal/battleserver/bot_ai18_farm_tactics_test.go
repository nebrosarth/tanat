package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

func TestAI18BaseFarmRosterPrioritizesWeakestAndSpreadsWaves(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())

	bots := []*botBrain{
		macroAddBot(t, s, inst, botIDBase+41, dotaTeamHuman, 0, 0, 0),
		macroAddBot(t, s, inst, botIDBase+42, dotaTeamHuman, 0, 0, 0),
		macroAddBot(t, s, inst, botIDBase+43, dotaTeamHuman, 0, 0, 0),
	}
	levels := []int32{1, 4, 6}
	xp := []float64{50, 700, 1600}
	plan := botTeamPlan{Team: dotaTeamHuman, Assignments: map[int32]botMacroAssignment{}}
	for i, b := range bots {
		b.c.huntState.level = levels[i]
		b.c.huntState.xp = xp[i]
		b.macroAssignment = botMacroAssignment{Mode: botMacroBase, Lane: 0, BaselineLane: 0, Role: "defender"}
		plan.Assignments[b.c.objID] = b.macroAssignment
		lane := inst.dota.m.Lanes[i]
		wx, wy := float32(lane[len(lane)/2].X), float32(lane[len(lane)/2].Y)
		b.c.x, b.c.y, b.c.snapT = wx, wy, float32(now)
		wave := teleportTestCreep(inst, botIDBase+141+int32(i), dotaTeamElf, wx, wy)
		wave.lane = lane
		wave.laneIdx = len(lane) / 2
		inst.mobs[wave.id] = wave
	}

	s.botAssignFarmLanesLocked(&plan, bots, inst)
	weakest := plan.Assignments[bots[0].c.objID]
	if !weakest.FarmLaneSet {
		t.Fatal("weakest defender did not receive an explicit farm lane")
	}
	if weakest.FarmLane != 0 {
		t.Fatalf("weakest defender farm lane = %d, want live wave lane 0", weakest.FarmLane)
	}
	seen := map[int]bool{}
	for _, b := range bots {
		assignment := plan.Assignments[b.c.objID]
		if !assignment.FarmLaneSet {
			t.Fatalf("defender %d did not receive a farm lane", b.c.objID)
		}
		if seen[assignment.FarmLane] {
			t.Fatalf("farm lane %d was assigned to multiple defenders", assignment.FarmLane)
		}
		seen[assignment.FarmLane] = true
	}
}

func TestAI18FarmTargetClaimAvoidsStackingWhenAnotherWaveIsAvailable(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	lane := inst.dota.m.Lanes[0]
	x, y := float32(lane[len(lane)/2].X), float32(lane[len(lane)/2].Y)
	bots := []*botBrain{
		macroAddBot(t, s, inst, botIDBase+51, dotaTeamHuman, x, y, 0),
		macroAddBot(t, s, inst, botIDBase+52, dotaTeamHuman, x, y, 0),
	}
	for _, b := range bots {
		b.macroAssignment = botMacroAssignment{Mode: botMacroLane, Lane: 0, FarmLane: 0, FarmLaneSet: true}
	}
	first := teleportTestCreep(inst, botIDBase+151, dotaTeamElf, x, y)
	first.lane = lane
	first.laneIdx = len(lane) / 2
	second := teleportTestCreep(inst, botIDBase+152, dotaTeamElf, x+25, y)
	second.lane = lane
	second.laneIdx = len(lane) / 2
	inst.mobs[first.id] = first
	inst.mobs[second.id] = second

	if got := s.botFarmTargetLocked(bots[0], now, 100, false); got != first {
		t.Fatalf("first farmer chose creep %d, want first wave creep %d", got.id, first.id)
	}
	if got := s.botFarmTargetLocked(bots[1], now, 100, false); got != second {
		t.Fatalf("second farmer chose creep %d, want unclaimed nearby wave creep %d", got.id, second.id)
	}
	if bots[1].farmTarget == bots[0].farmTarget {
		t.Fatalf("farmers stacked on target %d despite another live wave being available", bots[1].farmTarget)
	}
}

func TestAI18FarmLaneStaysStableAcrossMacroRecompute(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	bots := []*botBrain{
		macroAddBot(t, s, inst, botIDBase+61, dotaTeamHuman, 0, 0, 0),
		macroAddBot(t, s, inst, botIDBase+62, dotaTeamHuman, 0, 0, 1),
		macroAddBot(t, s, inst, botIDBase+63, dotaTeamHuman, 0, 0, 2),
	}
	for laneIndex, lane := range inst.dota.m.Lanes {
		wave := teleportTestCreep(inst, botIDBase+161+int32(laneIndex), dotaTeamElf, float32(lane[len(lane)/2].X), float32(lane[len(lane)/2].Y))
		wave.lane = lane
		wave.laneIdx = len(lane) / 2
		inst.mobs[wave.id] = wave
	}
	first := botTeamPlan{Team: dotaTeamHuman, Assignments: map[int32]botMacroAssignment{}}
	for _, b := range bots {
		first.Assignments[b.c.objID] = botMacroAssignment{Mode: botMacroLane, Lane: b.lane, BaselineLane: b.lane}
	}
	s.botAssignFarmLanesLocked(&first, bots, inst)
	second := botTeamPlan{Team: dotaTeamHuman, Assignments: map[int32]botMacroAssignment{}}
	for _, b := range bots {
		second.Assignments[b.c.objID] = botMacroAssignment{Mode: botMacroCover, Lane: (b.lane + 1) % 3, BaselineLane: b.lane}
	}
	s.botAssignFarmLanesWithPreviousLocked(&second, first, true, bots, inst)
	for _, b := range bots {
		if got, want := second.Assignments[b.c.objID].FarmLane, first.Assignments[b.c.objID].FarmLane; got != want {
			t.Fatalf("bot %d farm lane changed during equivalent live-wave recompute: got %d want %d", b.c.objID, got, want)
		}
	}
}

func TestAI18BarracksPressureKeepsFarmRosterOutsideBase(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	barracks := structOfSide(inst, gamedata.DotaCreepTower, dotaTeamHuman)
	if barracks == nil {
		t.Fatal("setup: no allied barracks")
	}
	bots := []*botBrain{
		macroAddBot(t, s, inst, botIDBase+71, dotaTeamHuman, barracks.x+20, barracks.y, 0),
		macroAddBot(t, s, inst, botIDBase+72, dotaTeamHuman, barracks.x+25, barracks.y, 1),
		macroAddBot(t, s, inst, botIDBase+73, dotaTeamHuman, barracks.x+30, barracks.y, 2),
		macroAddBot(t, s, inst, botIDBase+74, dotaTeamHuman, barracks.x+35, barracks.y, 0),
	}
	lane := botNearestLaneToPointLocked(inst.dota, barracks.x, barracks.y)
	threat := teleportTestCreep(inst, botIDBase+171, dotaTeamElf, barracks.x-2, barracks.y)
	threat.lane = inst.dota.m.Lanes[lane]
	threat.laneIdx = len(threat.lane) / 2
	threat.dtarget = barracks.id
	inst.mobs[threat.id] = threat

	plan := s.botPlanTeamLocked(inst, dotaTeamHuman, now)
	baseResponders := 0
	farmRoster := 0
	for _, b := range bots {
		a := plan.Assignments[b.c.objID]
		if a.Mode == botMacroBase {
			baseResponders++
		} else if a.FarmLaneSet {
			farmRoster++
		}
	}
	if baseResponders != 1 {
		t.Fatalf("ordinary barracks pressure assigned %d base responders, want one; plan=%+v", baseResponders, plan.Assignments)
	}
	if farmRoster != len(bots)-1 {
		t.Fatalf("ordinary barracks pressure left %d/%d bots in farm roster", farmRoster, len(bots)-1)
	}
}
