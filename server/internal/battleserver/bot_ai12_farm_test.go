package battleserver

import "testing"

func TestAI12CoverApproachesAStillSafeFarmWave(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	b := &botBrain{
		c: bot, lane: 0, phase: botPhaseLane,
		macroAssignment: botMacroAssignment{Mode: botMacroCover, Lane: 0, Role: "cover"},
	}
	wave := teleportTestCreep(inst, 81201, dotaTeamElf, bot.x+36, bot.y)
	inst.mobs[wave.id] = wave

	bot.lock()
	s.botCoverageTickLocked(b, now)
	bot.unlock()

	if !bot.hasDest {
		t.Fatal("cover bot did not approach a safe live wave within farm radius")
	}
	if b.farmTarget != wave.id || b.farmDecision != "wave_clear" {
		t.Fatalf("cover farm state = target %d decision %q, want target %d/wave_clear", b.farmTarget, b.farmDecision, wave.id)
	}
}

func TestAI12CoverStaysOnWaveInsteadOfChasingGoldLastHit(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_DPS_Gellar")
	defer cleanup()
	now := float64(s.battleTime())
	b := &botBrain{
		c: bot, lane: 0, phase: botPhaseLane,
		macroAssignment: botMacroAssignment{Mode: botMacroCover, Lane: 0, Role: "cover"},
	}
	last := teleportTestCreep(inst, 81202, dotaTeamElf, bot.x+1, bot.y)
	last.hp, last.maxHP = 1, 1
	cluster := teleportTestCreep(inst, 81203, dotaTeamElf, bot.x+2, bot.y)
	inst.mobs[last.id], inst.mobs[cluster.id] = last, cluster

	bot.lock()
	s.botCoverageTickLocked(b, now)
	bot.unlock()

	if got := bot.huntState.attackTarget; got != last.id {
		t.Fatalf("cover attack target = %d, want nearby wave target %d", got, last.id)
	}
	if b.farmDecision != "wave_clear" {
		t.Fatalf("cover farm decision = %q, want wave_clear", b.farmDecision)
	}
}
