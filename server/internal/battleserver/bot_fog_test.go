package battleserver

import "testing"

func TestBotQueriesRespectDotaFogOfWar(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())

	hiddenX, hiddenY := bot.x+250, bot.y+250
	enemy := dotaPlayerConn(t, s, inst, 83001, dotaTeamElf, hiddenX, hiddenY)
	b := &botBrain{c: bot, lane: 0, phase: botPhaseLane}
	inst.bots[bot.objID] = b

	if got := botLivingEnemyHeroes(bot, now); len(got) != 0 {
		t.Fatalf("hidden enemy heroes returned by local targeting: %d", len(got))
	}
	if got := botSelectTeamFocusTargetLocked(inst, dotaTeamHuman, now); got != nil {
		t.Fatalf("hidden enemy selected as macro focus: %d", got.objID)
	}
	if got := botNearbyEnemyHeroPressureLocked(b, now); got != 0 {
		t.Fatalf("hidden enemy counted as nearby pressure: %d", got)
	}

	wave := teleportTestCreep(inst, 83002, dotaTeamElf, hiddenX, hiddenY)
	wave.lane = inst.dota.m.Lanes[0]
	wave.laneIdx = len(wave.lane) / 2
	inst.mobs[wave.id] = wave
	if got := s.botFarmTargetLocked(b, now, 500, false); got != nil {
		t.Fatalf("hidden enemy creep selected for farm: %d", got.id)
	}

	enemy.x, enemy.y = bot.x+5, bot.y
	wave.x, wave.y = bot.x+6, bot.y
	if got := botLivingEnemyHeroes(bot, now); len(got) != 1 || got[0] != enemy {
		t.Fatalf("visible enemy hero query = %v, want %d", got, enemy.objID)
	}
	if got := s.botFarmTargetLocked(b, now, 500, false); got != wave {
		t.Fatalf("visible enemy creep farm target = %v, want %d", got, wave.id)
	}
}
