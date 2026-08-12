package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

func TestDotaCalibratedCreepWaveXP(t *testing.T) {
	for _, dm := range gamedata.DotaMaps() {
		melee := gamedata.Mobs()[dm.HumanCreepMelee]
		ranged := gamedata.Mobs()[dm.HumanCreepRange]
		if melee.XP != 72 || ranged.XP != 84 {
			t.Fatalf("map %d human creep XP = %.0f/%.0f, want 72/84", dm.ID, melee.XP, ranged.XP)
		}
		if wave := 3*melee.XP + ranged.XP; wave != 300 {
			t.Fatalf("map %d calibrated wave XP = %.0f, want 300", dm.ID, wave)
		}
	}

	// Both factions use the same economy even though their prefab/name entries differ.
	dm := gamedata.DotaMaps()[0]
	elfMelee, elfRange := dm.CreepMobIdx(gamedata.DotaSideElf)
	mobs := gamedata.Mobs()
	if mobs[elfMelee].XP != 72 || mobs[elfRange].XP != 84 {
		t.Fatalf("elf creep XP = %.0f/%.0f, want 72/84", mobs[elfMelee].XP, mobs[elfRange].XP)
	}
}

func TestDotaHeroKillXPFormula(t *testing.T) {
	for _, tc := range []struct {
		victimLevel int32
		want        float64
	}{
		{victimLevel: -1, want: 120},
		{victimLevel: 0, want: 120},
		{victimLevel: 4, want: 152},
		{victimLevel: 9, want: 192},
		{victimLevel: 19, want: 272},
	} {
		if got := dotaHeroKillXP(tc.victimLevel); got != tc.want {
			t.Errorf("dotaHeroKillXP(%d) = %.0f, want %.0f", tc.victimLevel, got, tc.want)
		}
	}
}

func TestDotaHeroKillXPGrantTelemetryRecordsSplit(t *testing.T) {
	s, killer, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	ally := dotaPlayerConn(t, s, inst, 1001, dotaTeamHuman, killer.x+3, killer.y)
	victim := dotaPlayerConn(t, s, inst, 1002, dotaTeamElf, killer.x, killer.y)
	victim.huntState.level = 4 // displayed level 5; bounty is 120 + 8*4 = 152 XP

	dir := t.TempDir()
	rec := newTelemetryRecorder(dir, inst.dota.m.ID)
	if rec == nil {
		t.Fatal("newTelemetryRecorder returned nil")
	}
	inst.dota.telemetry = rec

	killer.lock()
	s.dotaCreditHeroKillLocked(killer, victim, float64(s.battleTime()))
	killer.unlock()
	inst.dota.telemetry = nil
	rec.close()

	got := readTelemetryLines(t, dir)
	if len(got) != 2 {
		t.Fatalf("got %d xp telemetry lines, want one per recipient", len(got))
	}
	for _, line := range got {
		if line["type"] != "xp_grant" || line["source"] != "avatar" {
			t.Errorf("xp telemetry identity = %+v", line)
		}
		if line["source_id"] != float64(victim.objID) || line["raw_xp"] != float64(152) ||
			line["received_xp"] != float64(76) || line["recipients"] != float64(2) || line["victim_level"] != float64(5) {
			t.Errorf("xp telemetry calibration fields = %+v", line)
		}
	}
	if gotXP := killer.huntState.xp; gotXP != 76 {
		t.Errorf("killer XP = %.0f, want 76 after a two-way shared level-5 kill", gotXP)
	}
	if gotXP := ally.huntState.xp; gotXP != 76 {
		t.Errorf("ally XP = %.0f, want 76 after a two-way shared level-5 kill", gotXP)
	}
}
