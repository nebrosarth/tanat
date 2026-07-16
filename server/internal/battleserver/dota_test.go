package battleserver

import (
	"net"
	"testing"

	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// newDotaConn builds a solo «Штурм» (DOTA) member: a Human-side player joined to a
// fresh DOTA instance (all base structures seeded). Pushes drain into a pipe reader so
// *Locked helpers never block.
func newDotaConn(t *testing.T, prefab string) (*Server, *conn, *huntInstance, func()) {
	t.Helper()
	s := New(session.NewStore())
	srv, cli := net.Pipe()
	r := battleproto.NewReader(cli)
	go func() {
		for {
			if _, err := r.Read(); err != nil {
				return
			}
		}
	}()

	dm := gamedata.DotaMaps()[0]
	inst := newDotaInstance(s, dm.ID, dm.ID)
	av := avatarByPrefab(t, prefab)
	c := &conn{Conn: srv}
	c.objID = 1000
	c.x, c.y, c.snapT = float32(dm.SpawnHuman.X), float32(dm.SpawnHuman.Y), s.battleTime()
	hs := &huntState{
		av: av, kit: gamedata.SkillsFor(av),
		mobs: inst.mobs, summons: map[int32]*summonState{},
		hp: av.Health, mana: av.Mana,
	}
	hs.tr.add(c.objID)
	hs.inst = inst
	hs.worldReady = true
	c.huntState = hs
	c.inst = inst
	c.lk = &inst.mu
	inst.members[c.objID] = c
	cleanup := func() { srv.Close(); cli.Close() }
	return s, c, inst, cleanup
}

// altarOf returns the altar of the given in-battle team from the instance.
func altarOf(inst *huntInstance, team int32) *mobState {
	for _, m := range inst.mobs {
		if m.structure && m.altar && m.team == team {
			return m
		}
	}
	return nil
}

// TestDotaInstanceStructures pins the seeded layout: every baked structure is present
// with the player's side on team 1 and the enemy on team -1, altars flagged, and the
// altar/gun/tower roles preserved.
func TestDotaInstanceStructures(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	_ = s
	dm := inst.dota.m
	if got := len(inst.mobs); got != len(dm.Structures) {
		t.Fatalf("seeded %d structures, want %d", got, len(dm.Structures))
	}
	own := altarOf(inst, dotaPlayerTeam)
	enemy := altarOf(inst, dotaEnemyTeam)
	if own == nil || enemy == nil {
		t.Fatalf("missing altar(s): own=%v enemy=%v", own, enemy)
	}
	if !own.structure || !own.altar || own.dotaRole != gamedata.DotaAltar {
		t.Errorf("own altar flags wrong: %+v", own)
	}
	// Human side (the player's) must be team 1, Elf enemy team -1.
	humanGuns, elfGuns := 0, 0
	for _, m := range inst.mobs {
		if m.dotaRole == gamedata.DotaGun {
			if m.team == dotaPlayerTeam {
				humanGuns++
			} else {
				elfGuns++
			}
		}
	}
	if humanGuns == 0 || elfGuns == 0 {
		t.Fatalf("expected guns on both sides, got human=%d elf=%d", humanGuns, elfGuns)
	}
}

// TestDotaAltarGuardedByGuns is the core «Штурм» rule: the enemy altar shrugs off all
// damage while any enemy cannon stands, then becomes destroyable once they are gone.
func TestDotaAltarGuardedByGuns(t *testing.T) {
	s, c, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	enemy := altarOf(inst, dotaEnemyTeam)
	if enemy == nil {
		t.Fatal("no enemy altar")
	}
	now := float64(s.battleTime())

	c.lock()
	if inst.dota.altarVulnerableLocked(enemy) {
		t.Fatal("enemy altar should be invulnerable while its guns stand")
	}
	before := enemy.hp
	s.hitMobLocked(c, enemy, 5000, c.objID) // player smashes the guarded altar
	if enemy.hp != before {
		t.Fatalf("guarded altar took damage: %g -> %g", before, enemy.hp)
	}
	// Raze every enemy cannon.
	for _, m := range inst.mobs {
		if m.dotaRole == gamedata.DotaGun && m.team == dotaEnemyTeam {
			m.dead = true
		}
	}
	if !inst.dota.altarVulnerableLocked(enemy) {
		t.Fatal("enemy altar should be vulnerable after its guns fall")
	}
	s.hitMobLocked(c, enemy, 5000, c.objID)
	c.unlock()
	if enemy.hp >= before {
		t.Fatalf("unguarded altar took no damage: %g -> %g", before, enemy.hp)
	}
	_ = now
}

// TestDotaWinOnEnemyAltar: destroying the enemy altar ends the match with the player's
// side as the winner (BATTLE_END winner team 1).
func TestDotaWinOnEnemyAltar(t *testing.T) {
	s, c, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	enemy := altarOf(inst, dotaEnemyTeam)
	now := float64(s.battleTime())

	c.lock()
	// Clear the guard, then kill the altar via the player damage path.
	for _, m := range inst.mobs {
		if m.dotaRole == gamedata.DotaGun && m.team == dotaEnemyTeam {
			m.dead = true
		}
	}
	// Enough raw damage to overcome the altar's armor mitigation and kill it outright.
	s.hitMobLocked(c, enemy, gamedata.DotaAltarHP*10, c.objID)
	if !enemy.dead {
		t.Fatalf("precondition: altar should be dead, hp=%g", enemy.hp)
	}
	// The win check runs in the tick; drive one pass.
	s.dotaTickLocked(c, now)
	c.unlock()

	if !inst.dota.ended {
		t.Fatal("match did not end after the enemy altar fell")
	}
	if inst.dota.winner != dotaWinTeamSelf {
		t.Fatalf("winner team = %d, want %d (player's side)", inst.dota.winner, dotaWinTeamSelf)
	}
}

// TestDotaLoseOnOwnAltar: losing your own altar ends the match with the enemy as the
// winner (display team 2).
func TestDotaLoseOnOwnAltar(t *testing.T) {
	s, c, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	own := altarOf(inst, dotaPlayerTeam)
	now := float64(s.battleTime())
	c.lock()
	own.dead = true // the enemy pushed it down
	s.dotaTickLocked(c, now)
	c.unlock()
	if !inst.dota.ended || inst.dota.winner != dotaWinTeamEnemy {
		t.Fatalf("own altar loss: ended=%v winner=%d, want ended=true winner=%d",
			inst.dota.ended, inst.dota.winner, dotaWinTeamEnemy)
	}
}

// TestDotaCreepWaveSpawns: each generator releases a creep wave once its cadence
// elapses, and the creeps land on the correct team.
func TestDotaCreepWaveSpawns(t *testing.T) {
	s, c, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	structs := len(inst.mobs)
	now := float64(s.battleTime()) + gamedata.CreepFirstWave + 0.1

	c.lock()
	s.dotaSpawnWavesLocked(c, now)
	c.unlock()

	creeps := len(inst.mobs) - structs
	generators := 0
	for _, sc := range inst.dota.m.Structures {
		if sc.Role == gamedata.DotaGenerator {
			generators++
		}
	}
	want := generators * gamedata.CreepsPerWave
	if creeps != want {
		t.Fatalf("spawned %d creeps, want %d (%d generators × %d)", creeps, want, generators, gamedata.CreepsPerWave)
	}
	// A human generator's creeps must be allies (team 1), an elf generator's enemies.
	var sawAlly, sawEnemy bool
	for _, m := range inst.mobs {
		if m.structure {
			continue
		}
		if m.team == dotaPlayerTeam {
			sawAlly = true
		}
		if m.team == dotaEnemyTeam {
			sawEnemy = true
		}
	}
	if !sawAlly || !sawEnemy {
		t.Fatalf("creep teams: ally=%v enemy=%v, want both", sawAlly, sawEnemy)
	}
}

// TestDotaCreepTargetsEnemyNotAlly: an enemy creep placed next to an ally structure and
// an enemy structure engages the ENEMY, never a same-team object.
func TestDotaCreepTargetsEnemyNotAlly(t *testing.T) {
	s, c, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	// An enemy-team creep (team -1) sitting between an ally altar (team 1) and an enemy
	// gun (team -1). It must pick the ally altar (its enemy), not the enemy gun (ally).
	own := altarOf(inst, dotaPlayerTeam)
	creep := &mobState{
		id: 61000, mobIdx: inst.dota.m.ElfCreepMelee,
		mob:  gamedata.Mobs()[inst.dota.m.ElfCreepMelee],
		x:    own.x + 3, y: own.y,
		team: dotaEnemyTeam,
	}
	inst.mobs[creep.id] = creep

	c.lock()
	tgt := s.dotaAcquireTargetLocked(c, creep, 50, float64(s.battleTime()))
	c.unlock()
	if tgt == nil || tgt.mob == nil {
		t.Fatal("enemy creep acquired no target")
	}
	if tgt.mob.team == creep.team {
		t.Fatalf("creep targeted an ally (team %d == its own)", tgt.mob.team)
	}
	if tgt.mob.id != own.id {
		t.Errorf("creep targeted obj %d, want the nearby ally altar %d", tgt.mob.id, own.id)
	}
}
