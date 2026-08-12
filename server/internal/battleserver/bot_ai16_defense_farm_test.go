package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

func TestAI16BaseDefenderApproachesNearbySafeWave(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	barracks := structOfSide(inst, gamedata.DotaCreepTower, dotaTeamHuman)
	if barracks == nil {
		t.Fatal("setup: no allied barracks")
	}
	lane := botNearestLaneToPointLocked(inst.dota, barracks.x, barracks.y)
	bot.x, bot.y, bot.snapT = barracks.x, barracks.y, float32(now)
	wave := teleportTestCreep(inst, 81601, dotaTeamElf, barracks.x+36, barracks.y)
	wave.lane = inst.dota.m.Lanes[lane]
	wave.laneIdx = len(wave.lane) / 2
	inst.mobs[wave.id] = wave
	b := &botBrain{
		c: bot, lane: lane, phase: botPhaseLane,
		macroAssignment: botMacroAssignment{Mode: botMacroBase, Lane: lane, ObjectiveID: barracks.id, Role: "defender"},
	}

	bot.lock()
	s.botBaseDefenseTickLocked(b, now)
	bot.unlock()

	if b.farmTarget != wave.id || b.farmDecision != "wave_clear" {
		t.Fatalf("defense farm state = target %d/decision %q, want nearby wave %d/wave_clear", b.farmTarget, b.farmDecision, wave.id)
	}
	if !bot.hasDest || bot.destX != wave.x || bot.destY != wave.y {
		t.Fatalf("defender destination = (%.1f, %.1f), want nearby wave (%.1f, %.1f)", bot.destX, bot.destY, wave.x, wave.y)
	}
}

func TestAI16BaseDefenderInterceptsWaveInsidePredictiveRing(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	barracks := structOfSide(inst, gamedata.DotaCreepTower, dotaTeamHuman)
	if barracks == nil {
		t.Fatal("setup: no allied barracks")
	}
	lane := botNearestLaneToPointLocked(inst.dota, barracks.x, barracks.y)
	bot.x, bot.y, bot.snapT = barracks.x, barracks.y, float32(now)
	// The predictive ring is wider than hero vision, but the defender cannot
	// intercept a creep that is still hidden. Keep the wave outside immediate
	// contact and let an allied lane creep provide the forward vision, just as
	// it would in a real push.
	wave := teleportTestCreep(inst, 81602, dotaTeamElf, barracks.x+110, barracks.y)
	wave.lane = inst.dota.m.Lanes[lane]
	wave.laneFwd = false
	wave.laneIdx = len(wave.lane) / 2
	inst.mobs[wave.id] = wave
	scout := teleportTestCreep(inst, 81603, dotaTeamHuman, barracks.x+90, barracks.y)
	scout.lane = inst.dota.m.Lanes[lane]
	scout.laneFwd = true
	scout.laneIdx = len(scout.lane) / 2
	inst.mobs[scout.id] = scout
	b := &botBrain{
		c: bot, lane: lane, phase: botPhaseLane,
		macroAssignment: botMacroAssignment{Mode: botMacroBase, Lane: lane, ObjectiveID: barracks.id, Role: "defender"},
	}

	bot.lock()
	s.botBaseDefenseTickLocked(b, now)
	bot.unlock()

	if b.farmTarget != wave.id || b.farmDecision != "defense_intercept" {
		t.Fatalf("defense intercept state = target %d/decision %q, want wave %d/defense_intercept", b.farmTarget, b.farmDecision, wave.id)
	}
	if !bot.hasDest || bot.destX != wave.x || bot.destY != wave.y {
		t.Fatalf("defender destination = (%.1f, %.1f), want approaching wave (%.1f, %.1f)", bot.destX, bot.destY, wave.x, wave.y)
	}
}
