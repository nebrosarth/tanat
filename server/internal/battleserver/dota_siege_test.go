package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

func siegeCreepsOf(inst *huntInstance) []*mobState {
	var out []*mobState
	for _, m := range inst.mobs {
		if m != nil && !m.dead && m.siege {
			out = append(out, m)
		}
	}
	return out
}

func TestAssaultSiegeCreepCadenceAndDirections(t *testing.T) {
	s, c, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()

	dm, ok := gamedata.DotaMapByID(101)
	if !ok || !dm.SiegeCreepWaves {
		t.Fatal("map 101 must enable Assault siege waves")
	}
	start := inst.dota.startedAt

	c.lock()
	s.dotaSpawnWavesLocked(c, start+gamedata.SiegeCreepFirstWave-0.1)
	c.unlock()
	if got := len(siegeCreepsOf(inst)); got != 0 {
		t.Fatalf("siege creeps spawned %.1fs before the first schedule: got %d", gamedata.SiegeCreepFirstWave-0.1, got)
	}

	c.lock()
	s.dotaSpawnWavesLocked(c, start+gamedata.SiegeCreepFirstWave+0.1)
	c.unlock()
	first := siegeCreepsOf(inst)
	want := len(dm.Lanes) * 2
	if len(first) != want {
		t.Fatalf("first siege wave spawned %d units, want %d (%d lanes x 2 sides)", len(first), want, len(dm.Lanes))
	}

	seenLanes := map[int]map[gamedata.DotaSide]bool{}
	for _, m := range first {
		if m.mob.Health != 500 {
			t.Errorf("siege creep %d has %.0f HP, want 500", m.id, m.mob.Health)
		}
		if m.mob.AttackRange <= 0 || !m.hasProj {
			t.Errorf("siege creep %d is not a ranged projectile unit: range=%.1f hasProj=%v", m.id, m.mob.AttackRange, m.hasProj)
		}
		side := gamedata.DotaSideElf
		if m.team == dotaTeamHuman {
			side = gamedata.DotaSideHuman
		}
		wantIdx := dm.SiegeCreepMobIdx(side)
		if m.mobIdx != wantIdx {
			t.Errorf("siege creep %d uses roster index %d, want side %d index %d", m.id, m.mobIdx, side, wantIdx)
		}
		if side == gamedata.DotaSideHuman && !m.laneFwd {
			t.Errorf("Catapultosaur %d from the Cathedral side walks away from the Exiles", m.id)
		}
		if side == gamedata.DotaSideElf && m.laneFwd {
			t.Errorf("siege bear %d from the Exiles walks away from the Cathedral", m.id)
		}
		lane := -1
		for i, points := range dm.Lanes {
			if sameLane(points, m.lane) {
				lane = i
				break
			}
		}
		if lane < 0 {
			t.Errorf("siege creep %d is not assigned to a map lane", m.id)
			continue
		}
		if seenLanes[lane] == nil {
			seenLanes[lane] = map[gamedata.DotaSide]bool{}
		}
		seenLanes[lane][side] = true
	}
	for lane := range dm.Lanes {
		if !seenLanes[lane][gamedata.DotaSideHuman] || !seenLanes[lane][gamedata.DotaSideElf] {
			t.Errorf("lane %d did not receive both Cathedral and Exiles siege creeps: %v", lane, seenLanes[lane])
		}
	}

	c.lock()
	s.dotaSpawnWavesLocked(c, start+gamedata.SiegeCreepFirstWave+gamedata.SiegeCreepWaveInterval-0.1)
	c.unlock()
	if got := len(siegeCreepsOf(inst)); got != want {
		t.Fatalf("siege wave fired early at 5-minute boundary: got %d, want %d", got, want)
	}

	c.lock()
	s.dotaSpawnWavesLocked(c, start+gamedata.SiegeCreepFirstWave+gamedata.SiegeCreepWaveInterval+0.1)
	c.unlock()
	if got := len(siegeCreepsOf(inst)); got != want*2 {
		t.Fatalf("second siege wave spawned %d units, want %d", got, want*2)
	}
}

func TestAssaultSiegeCreepStopsWithDestroyedBarracks(t *testing.T) {
	s, c, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()

	dm := gamedata.DotaMaps()[0]
	bar := barracksOf(gamedata.DotaSideHuman)[0]
	inst.mobs[dotaStructIDBase+bar.ID].dead = true
	start := inst.dota.startedAt

	c.lock()
	s.dotaSpawnWavesLocked(c, start+gamedata.SiegeCreepFirstWave+0.1)
	c.unlock()

	human, elf := 0, 0
	deadLane := dm.LaneFor(bar)
	for _, m := range siegeCreepsOf(inst) {
		if m.team == dotaTeamHuman {
			human++
			if sameLane(m.lane, dm.Lanes[deadLane]) {
				t.Fatalf("Catapultosaur spawned on destroyed Cathedral barracks lane %d", deadLane)
			}
		} else {
			elf++
		}
	}
	wantHuman := len(dm.Lanes) - 1
	if human != wantHuman || elf != len(dm.Lanes) {
		t.Fatalf("siege cadence ignored destroyed barracks: Cathedral=%d/%d Exiles=%d/%d", human, wantHuman, elf, len(dm.Lanes))
	}
}

func TestAssaultSiegeCreepGetsBuildingDamageBonus(t *testing.T) {
	s, c, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()

	start := inst.dota.startedAt
	c.lock()
	s.dotaSpawnWavesLocked(c, start+gamedata.SiegeCreepFirstWave+0.1)
	var attacker *mobState
	for _, m := range siegeCreepsOf(inst) {
		if m.team == dotaTeamHuman {
			attacker = m
			break
		}
	}
	if attacker == nil {
		c.unlock()
		t.Fatal("first Assault wave has no Cathedral siege creep")
	}
	var target *mobState
	for _, m := range inst.mobs {
		if m.structure && m.team == dotaTeamElf && m.dotaRole == gamedata.DotaGun && !m.dead {
			target = m
			break
		}
	}
	if target == nil {
		c.unlock()
		t.Fatal("map has no live Exiles gun for the siege damage assertion")
	}
	now := start + gamedata.SiegeCreepFirstWave + 1
	attacker.hitTarget = target.id
	attacker.hitDmg = 100
	before := target.hp
	s.dotaLandHitLocked(c, attacker, now)
	got := before - target.hp
	want := 100 * dotaSiegeStructureDamageMultiplier * armorMitigation(target.physArmor(now))
	c.unlock()
	if got < want-0.001 || got > want+0.001 {
		t.Fatalf("siege building hit dealt %.3f damage, want %.3f", got, want)
	}
}
