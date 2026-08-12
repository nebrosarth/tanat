package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// This file pins the fog-of-war fix for the live report "I see enemy heroes and
// creeps through the fog": «Штурм» used to hand every member's client a CREATE_OBJECT
// for every enemy unit unconditionally (avatars.go/dota.go/mobai.go all fanned reveals
// out over the WHOLE instance, not the unit's own team). dotaVisionPassLocked
// (vision.go) is the fix -- see its doc comment for the mechanism.

// TestDotaEnemyAvatarHiddenUntilAllyInVision: an enemy hero standing far outside every
// human unit's vision must not be tracked on the human's client; once a human unit
// (here, the player themself, walking closer) draws within avatarViewRadius, the very
// next vision pass reveals it.
func TestDotaEnemyAvatarHiddenUntilAllyInVision(t *testing.T) {
	s := New(session.NewStore())
	dm := gamedata.DotaMaps()[0]
	inst := newDotaInstance(s, dm.ID, dm.ID)
	now := float64(s.battleTime())

	human := dotaPlayerConn(t, s, inst, 1000, dotaTeamHuman, 0, 0)
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, avatarViewRadius+50, 0)

	human.mvMu.Lock()
	s.dotaVisionPassLocked(inst, now)
	human.mvMu.Unlock()
	if human.huntState.tr.index(elf.objID) >= 0 {
		t.Fatal("enemy hero tracked while far outside every human unit's vision")
	}

	human.x, human.y = elf.x-avatarViewRadius+5, elf.y
	human.mvMu.Lock()
	s.dotaVisionPassLocked(inst, now)
	human.mvMu.Unlock()
	if human.huntState.tr.index(elf.objID) < 0 {
		t.Error("enemy hero still hidden once a human unit closed within vision range")
	}
}

// TestDotaEnemyAvatarLeavingVisionIsHidden: the reverse of the above -- once a
// previously-visible enemy hero walks well past every source's radius (past the
// hysteresis margin), the next pass must drop it from the tracker again.
func TestDotaEnemyAvatarLeavingVisionIsHidden(t *testing.T) {
	s := New(session.NewStore())
	dm := gamedata.DotaMaps()[0]
	inst := newDotaInstance(s, dm.ID, dm.ID)
	now := float64(s.battleTime())

	human := dotaPlayerConn(t, s, inst, 1000, dotaTeamHuman, 0, 0)
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, 5, 0)

	human.mvMu.Lock()
	s.dotaVisionPassLocked(inst, now)
	human.mvMu.Unlock()
	if human.huntState.tr.index(elf.objID) < 0 {
		t.Fatal("precondition: enemy hero standing 5 units away was not revealed")
	}

	elf.x = human.x + avatarViewRadius + dotaVisionHysteresis + 10
	human.mvMu.Lock()
	s.dotaVisionPassLocked(inst, now)
	human.mvMu.Unlock()
	if human.huntState.tr.index(elf.objID) >= 0 {
		t.Error("enemy hero still tracked well past vision range + hysteresis")
	}
}

// Respawning is a local lifecycle event, not a fog reveal. A dead enemy that
// returns at a distant checkpoint must stay absent from this viewer until the
// ordinary vision pass sees the new position.
func TestDotaRespawnDoesNotRevealHiddenEnemyAvatar(t *testing.T) {
	s := New(session.NewStore())
	dm := gamedata.DotaMaps()[0]
	inst := newDotaInstance(s, dm.ID, dm.ID)
	now := float64(s.battleTime())

	human := dotaPlayerConn(t, s, inst, 1000, dotaTeamHuman, 0, 0)
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, avatarViewRadius+200, 0)
	elf.huntState.respawnX, elf.huntState.respawnY = elf.x, elf.y
	elf.huntState.deadUntil = now + 1

	elf.lock()
	s.respawnPlayerLocked(elf, now)
	elf.unlock()
	if human.huntState.tr.index(elf.objID) >= 0 {
		t.Fatal("enemy respawn leaked through fog before a vision pass")
	}
}

func TestDotaGunRevealsInvisibleAvatar(t *testing.T) {
	s := New(session.NewStore())
	dm := gamedata.DotaMaps()[0]
	inst := newDotaInstance(s, dm.ID, dm.ID)
	now := float64(s.battleTime())

	human := dotaPlayerConn(t, s, inst, 1000, dotaTeamHuman, 0, 0)
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, 10, 0)
	elf.huntState.invisibleUntil = now + 60

	var gun *mobState
	for _, m := range inst.mobs {
		if !m.structure || m.teamVal() != dotaTeamHuman {
			continue
		}
		if m.dotaRole == gamedata.DotaGun && gun == nil {
			gun = m
		}
		m.dead = true
	}
	if gun == nil {
		t.Fatal("precondition: no human-side gun")
	}

	human.mvMu.Lock()
	s.dotaVisionPassLocked(inst, now)
	human.mvMu.Unlock()
	if human.huntState.tr.index(elf.objID) >= 0 {
		t.Fatal("ordinary hero vision revealed an invisible enemy")
	}

	gun.dead = false
	gun.x, gun.y = human.x, human.y
	human.mvMu.Lock()
	s.dotaVisionPassLocked(inst, now)
	human.mvMu.Unlock()
	if human.huntState.tr.index(elf.objID) < 0 {
		t.Fatal("human-side gun did not reveal an invisible enemy")
	}
	if !botVisibleEnemyMemberLocked(inst, dotaTeamHuman, elf, now,
		dotaTeamVisionSourcesLocked(inst, dotaTeamHuman, now)) {
		t.Fatal("bot vision query did not inherit gun True Sight")
	}
}

// TestDotaVisionHysteresisAvoidsBoundaryFlicker: a unit that just barely crosses the
// bare reveal radius outward (but stays within the hysteresis margin) must stay
// visible -- otherwise a unit walking the boundary line flickers CREATE_OBJECT/
// DELETE_OBJECT every tick.
func TestDotaVisionHysteresisAvoidsBoundaryFlicker(t *testing.T) {
	s := New(session.NewStore())
	dm := gamedata.DotaMaps()[0]
	inst := newDotaInstance(s, dm.ID, dm.ID)
	now := float64(s.battleTime())

	human := dotaPlayerConn(t, s, inst, 1000, dotaTeamHuman, 0, 0)
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, avatarViewRadius-5, 0)

	human.mvMu.Lock()
	s.dotaVisionPassLocked(inst, now)
	human.mvMu.Unlock()
	if human.huntState.tr.index(elf.objID) < 0 {
		t.Fatal("precondition: enemy hero inside vision radius was not revealed")
	}

	// Step just past the bare radius, well inside the hysteresis margin.
	elf.x = avatarViewRadius + dotaVisionHysteresis - 1
	human.mvMu.Lock()
	s.dotaVisionPassLocked(inst, now)
	human.mvMu.Unlock()
	if human.huntState.tr.index(elf.objID) < 0 {
		t.Error("already-visible enemy hero flickered hidden inside the hysteresis margin")
	}
}

// TestDotaEnemyCreepHiddenUntilAllyInVision: a fresh enemy wave must not appear on
// the client of a player standing nowhere near it, and must appear once a human unit
// (an allied creep, here) closes within creepViewRadius.
func TestDotaEnemyCreepHiddenUntilAllyInVision(t *testing.T) {
	s := New(session.NewStore())
	dm := gamedata.DotaMaps()[0]
	inst := newDotaInstance(s, dm.ID, dm.ID)
	now := float64(s.battleTime())

	human := dotaPlayerConn(t, s, inst, 1000, dotaTeamHuman, -1000, -1000)
	elfCreep := dotaAlly(t, inst, 62200, dotaTeamElf, 0, 0, now)

	human.mvMu.Lock()
	s.dotaVisionPassLocked(inst, now)
	human.mvMu.Unlock()
	if human.huntState.tr.index(elfCreep.id) >= 0 {
		t.Fatal("enemy creep tracked with no human unit anywhere near it")
	}

	ally := dotaAlly(t, inst, 62201, dotaTeamHuman, elfCreep.x+creepViewRadius-5, elfCreep.y, now)
	human.mvMu.Lock()
	s.dotaVisionPassLocked(inst, now)
	human.mvMu.Unlock()
	if human.huntState.tr.index(elfCreep.id) < 0 {
		t.Errorf("enemy creep still hidden once an allied creep (id %d) closed within vision", ally.id)
	}
}

// TestDotaStructuresStayVisibleRegardlessOfVision: a «Штурм» base is meant to stay on
// the map like Dota's own buildings do -- dotaApplyTeamVisionLocked must never untrack
// a structure, however far every vision source is from it.
func TestDotaStructuresStayVisibleRegardlessOfVision(t *testing.T) {
	s := New(session.NewStore())
	dm := gamedata.DotaMaps()[0]
	inst := newDotaInstance(s, dm.ID, dm.ID)
	now := float64(s.battleTime())

	human := dotaPlayerConn(t, s, inst, 1000, dotaTeamHuman, -100000, -100000)
	s.dotaWorldSetupLocked(human, now)

	var enemyStruct *mobState
	for _, m := range inst.mobs {
		if m.structure && m.teamVal() == dotaTeamElf {
			enemyStruct = m
			break
		}
	}
	if enemyStruct == nil {
		t.Fatal("precondition: no Elf-side structure seeded")
	}
	if human.huntState.tr.index(enemyStruct.id) < 0 {
		t.Fatal("precondition: dotaWorldSetupLocked did not reveal the enemy structure")
	}

	human.mvMu.Lock()
	s.dotaVisionPassLocked(inst, now)
	human.mvMu.Unlock()
	if human.huntState.tr.index(enemyStruct.id) < 0 {
		t.Error("enemy structure was hidden by the vision pass -- structures must stay exempt")
	}
}

// TestDotaCreepWaveHiddenFromEnemyAtSpawn: a freshly spawned wave must render on its
// own side's client immediately (an existing invariant -- see spawnCreepWaveLocked's
// doc comment on late joiners) but not leak to the enemy before any vision pass ran.
func TestDotaCreepWaveHiddenFromEnemyAtSpawn(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, 5000, 5000) // far from the lane

	bars := barracksOf(gamedata.DotaSideHuman)
	if len(bars) == 0 {
		t.Fatal("precondition: no Human-side barracks on this map")
	}
	s.dotaSpawnCreepWaveLocked(c, bars[0], now)

	var creep *mobState
	for _, m := range inst.mobs {
		if !m.structure && m.teamVal() == c.playerTeam() {
			creep = m
			break
		}
	}
	if creep == nil {
		t.Fatal("precondition: wave spawn produced no creep")
	}
	if c.huntState.tr.index(creep.id) < 0 {
		t.Error("own-side player did not immediately see their own freshly spawned creep")
	}
	if elf.huntState.tr.index(creep.id) >= 0 {
		t.Error("enemy player saw a freshly spawned creep before any vision pass ran")
	}
}
