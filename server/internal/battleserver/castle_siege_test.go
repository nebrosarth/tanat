package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// TestCastleSiegeInstanceStructures pins map_6_0's actual extracted layout: exactly
// the 5 defending GA_ClanWars_Gun structures (win condition) plus the Cerber boss
// guard, all on the defender team, and no altar at all -- the shape that makes
// dotaCheckWinLocked dispatch to castleCheckWinLocked.
func TestCastleSiegeInstanceStructures(t *testing.T) {
	s := New(session.NewStore())
	inst := newDotaInstance(s, 555, 102) // 102 = gamedata's map60 («Битва за замок»)
	if got, want := len(inst.mobs), 6; got != want {
		t.Fatalf("seeded %d objects, want %d (5 GA_ClanWars_Gun_prop01 + 1 boss guard)", got, want)
	}
	guns := 0
	var boss *mobState
	for _, m := range inst.mobs {
		if !m.structure {
			t.Errorf("unexpected non-structure object on the castle map: %+v", m)
		}
		if m.altar {
			t.Error("map_6_0 has no altar -- the «Castle» object is confirmed static scenery")
		}
		if m.team != dotaTeamElf {
			t.Errorf("object team = %d, want %d (defender)", m.team, dotaTeamElf)
		}
		switch m.dotaRole {
		case gamedata.DotaGun:
			guns++
		case gamedata.DotaGenerator:
			boss = m
		default:
			t.Errorf("unexpected dotaRole on the castle map: %+v", m)
		}
	}
	if guns != 5 {
		t.Errorf("gun count = %d, want 5", guns)
	}
	if boss == nil {
		t.Fatal("no boss guard seeded")
	}
	if boss.id != dotaBossID {
		t.Errorf("boss id = %d, want %d", boss.id, dotaBossID)
	}
}

// TestCastleSiegeWinsWhenAllGunsDown: once every defending gun falls, the attacker
// (team 1) wins -- there is no altar to check, unlike «Штурм».
func TestCastleSiegeWinsWhenAllGunsDown(t *testing.T) {
	s := New(session.NewStore())
	inst := newDotaInstance(s, 555, 102)
	attacker := dotaPlayerConn(t, s, inst, 1000, dotaTeamHuman, 0, 0)

	for _, m := range inst.mobs {
		m.dead = true
	}
	now := float64(s.battleTime())
	attacker.lock()
	s.dotaTickLocked(attacker, now)
	attacker.unlock()

	if !inst.dota.ended || inst.dota.winner != dotaTeamHuman {
		t.Fatalf("all guns down: ended=%v winner=%d, want ended=true winner=%d (attacker)",
			inst.dota.ended, inst.dota.winner, dotaTeamHuman)
	}
}

// TestCastleSiegeStaysOpenWithOneGunStanding: the defender holds as long as even one
// GUN survives -- not a majority/percentage threshold, and the boss guard (also a
// structure, but role DotaGenerator) must not be mistaken for a gun by the check.
func TestCastleSiegeStaysOpenWithOneGunStanding(t *testing.T) {
	s := New(session.NewStore())
	inst := newDotaInstance(s, 555, 102)
	attacker := dotaPlayerConn(t, s, inst, 1000, dotaTeamHuman, 0, 0)

	left := false
	for _, m := range inst.mobs {
		if m.dotaRole != gamedata.DotaGun {
			continue // leave the boss alive regardless -- he's not part of the check
		}
		if !left {
			left = true
			continue // leave exactly one gun standing
		}
		m.dead = true
	}
	if !left {
		t.Fatal("test premise broken: no gun found to leave standing")
	}
	now := float64(s.battleTime())
	attacker.lock()
	s.dotaTickLocked(attacker, now)
	attacker.unlock()

	if inst.dota.ended {
		t.Fatal("the match ended with a defending gun still alive")
	}
}

// TestCastleSiegeAltarWinDoesNotFireOnRegularShturm: dotaCheckWinLocked's altar path
// must still be the one used for «Штурм» (map10, which HAS altars) -- the new
// no-altar dispatch must not steal its win condition.
func TestCastleSiegeAltarWinDoesNotFireOnRegularShturm(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	human.lock()
	altarOf(inst, dotaTeamElf).dead = true
	s.dotaTickLocked(human, now)
	human.unlock()
	if !inst.dota.ended || inst.dota.winner != dotaTeamHuman {
		t.Fatalf("«Штурм» altar-fall win broke after adding the no-altar dispatch: ended=%v winner=%d",
			inst.dota.ended, inst.dota.winner)
	}
}

// TestCastleSiegeEndToEndSettlesOwnership: the gun-siege win, not just the altar-fall
// win, must also reach settleCastleBattleLocked when the instance is castle-tagged.
func TestCastleSiegeEndToEndSettlesOwnership(t *testing.T) {
	store := session.NewStore()
	s := New(store)
	u, _, ok := store.LoginOrRegister("stormer@example.com", "pw")
	if !ok {
		t.Fatal("register")
	}
	store.CreateHero(u, 1, false, 0, 0, 0, 0, 0)

	castle, known := gamedata.CastleByID(1)
	if !known {
		t.Fatal("castle 1 not found")
	}
	if castle.MapID != 102 {
		t.Fatalf("castle 1 MapID = %d, want 102 (map_6_0)", castle.MapID)
	}

	inst := newDotaInstance(s, 555, castle.MapID)
	inst.dota.castleID = castle.ID
	attacker := dotaPlayerConn(t, s, inst, 1000, dotaTeamHuman, 0, 0)
	attacker.selfPlayerID = u.ID

	before, _, _ := store.HeroMoney(u.ID)
	for _, m := range inst.mobs {
		m.dead = true
	}
	now := float64(s.battleTime())
	attacker.lock()
	s.dotaTickLocked(attacker, now)
	attacker.unlock()

	after, _, _ := store.HeroMoney(u.ID)
	if after != before+castle.Reward.Money {
		t.Errorf("attacker money after siege win = %d, want %d (settlement did not fire)", after, before+castle.Reward.Money)
	}
}

// TestCastleSiegeCreepCampsSpawnAttackerWaves: map_6_0's 2 un-structured creep camps
// (no barracks to kill) fire on the same CreepFirstWave/CreepsPerWave cadence a
// barracks would, reinforcing the ATTACKER (team 1) side.
func TestCastleSiegeCreepCampsSpawnAttackerWaves(t *testing.T) {
	s := New(session.NewStore())
	inst := newDotaInstance(s, 555, 102)
	before := len(inst.mobs)
	attacker := dotaPlayerConn(t, s, inst, 1000, dotaTeamHuman, 0, 0)

	now := float64(s.battleTime()) + gamedata.CreepFirstWave + 0.1
	attacker.lock()
	s.dotaSpawnWavesLocked(attacker, now)
	attacker.unlock()

	dm, ok := gamedata.DotaMapByID(102)
	if !ok {
		t.Fatal("map60 not found")
	}
	want := len(dm.CreepCamps) * gamedata.CreepsPerWave
	got := len(inst.mobs) - before
	if got != want {
		t.Fatalf("spawned %d creeps, want %d (%d camps x %d)", got, want, len(dm.CreepCamps), gamedata.CreepsPerWave)
	}
	for _, m := range inst.mobs {
		if m.structure {
			continue // a gun or the boss, not a spawned creep
		}
		if m.team != dotaTeamHuman {
			t.Errorf("castle creep team = %d, want %d (attacker)", m.team, dotaTeamHuman)
		}
		if len(m.lane) == 0 {
			t.Error("castle creep has no lane to march")
		}
	}
}

// TestCastleSiegeCreepCampsFireOnlyAfterFirstWaveDelay: a camp must not dump its
// squad on tick 1 -- it gets the same CreepFirstWave grace a barracks gets.
func TestCastleSiegeCreepCampsFireOnlyAfterFirstWaveDelay(t *testing.T) {
	s := New(session.NewStore())
	inst := newDotaInstance(s, 555, 102)
	before := len(inst.mobs)
	attacker := dotaPlayerConn(t, s, inst, 1000, dotaTeamHuman, 0, 0)

	now := float64(s.battleTime()) + 0.1 // well before CreepFirstWave
	attacker.lock()
	s.dotaSpawnWavesLocked(attacker, now)
	attacker.unlock()

	if len(inst.mobs) != before {
		t.Fatalf("a camp fired before CreepFirstWave elapsed: %d new object(s)", len(inst.mobs)-before)
	}
}

// TestCastleSiegeBossGuardStatsAndTeam: the Cerber boss guard carries his real
// (hand-tuned, level-scaling-exempt) roster stats, sits on the defender team, and is
// a stationary structure -- not a lane-marching creep.
func TestCastleSiegeBossGuardStatsAndTeam(t *testing.T) {
	s := New(session.NewStore())
	inst := newDotaInstance(s, 555, 102)
	boss := inst.mobs[dotaBossID]
	if boss == nil {
		t.Fatal("no boss guard seeded")
	}
	cerber := gamedata.MobByIndex(boss.mobIdx)
	if boss.maxHP != cerber.Health {
		t.Errorf("boss maxHP = %g, want %g (Cerber's authored HP, unscaled)", boss.maxHP, cerber.Health)
	}
	if boss.xp != cerber.XP || boss.coins != cerber.Coins {
		t.Errorf("boss reward = {xp=%g coins=%d}, want {xp=%g coins=%d} (Cerber's bounty)",
			boss.xp, boss.coins, cerber.XP, cerber.Coins)
	}
	if boss.team != dotaTeamElf {
		t.Errorf("boss team = %d, want %d (defender)", boss.team, dotaTeamElf)
	}
	if !boss.structure {
		t.Error("boss must be structure=true (stationary-attacker pipeline, no leash/homing risk)")
	}
	if boss.mob.AttackRange <= 0 {
		t.Error("boss has no attack range -- dotaStructCombatLocked would never let him engage anything")
	}
}

// TestCastleSiegeBossDoesNotCountTowardWinCondition: castleCheckWinLocked only scans
// DotaGun structures -- killing every gun ends the siege even with the boss alive,
// and killing the boss alone must never end it.
func TestCastleSiegeBossDoesNotCountTowardWinCondition(t *testing.T) {
	s := New(session.NewStore())
	inst := newDotaInstance(s, 555, 102)
	attacker := dotaPlayerConn(t, s, inst, 1000, dotaTeamHuman, 0, 0)

	// Kill only the boss; every gun stays up.
	inst.mobs[dotaBossID].dead = true
	now := float64(s.battleTime())
	attacker.lock()
	s.dotaTickLocked(attacker, now)
	attacker.unlock()
	if inst.dota.ended {
		t.Fatal("killing only the boss ended the siege -- he must not count toward the win condition")
	}
}
