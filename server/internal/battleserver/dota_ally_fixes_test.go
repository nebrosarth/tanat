package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

// This file covers the «Штурм» ally/visibility round: mobState is one type serving two
// roles, and inst.mobs holds the player's OWN creeps and buildings beside the enemy's.
// Every shared pass that scans it was written under the Hunt invariant "every mobState
// is an enemy, at an authored spawn point" -- «Штурм» breaks both halves.
//
// Each filter test is paired with a positive control: a filter that rejects everything
// passes a "does not hit an ally" assertion just as well as a correct one does.

// dotaAlly plants a live creep of the given team next to the avatar and returns it.
func dotaAlly(t *testing.T, inst *huntInstance, id int32, team int32, x, y float32, now float64) *mobState {
	t.Helper()
	idx := inst.dota.m.HumanCreepMelee
	if team == dotaEnemyTeam {
		idx = inst.dota.m.ElfCreepMelee
	}
	m := &mobState{
		id: id, mobIdx: idx, mob: gamedata.Mobs()[idx],
		x: x, y: y, hp: 5000, maxHP: 5000,
		team: team, lastSync: now, active: true, shown: true,
	}
	inst.mobs[m.id] = m
	return m
}

// TestSkillAoESkipsAlliesHitsEnemies: the AoE scans filter the player's own units. The
// enemy in the same blast is the positive control -- without it a scan that returned
// nothing at all would pass.
func TestSkillAoESkipsAlliesHitsEnemies(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	ally := dotaAlly(t, inst, 61001, dotaPlayerTeam, c.x+2, c.y, now)
	enemy := dotaAlly(t, inst, 61002, dotaEnemyTeam, c.x+2, c.y+1, now)

	got := c.mobsWithinLocked(c.x, c.y, 6)
	var sawAlly, sawEnemy bool
	for _, m := range got {
		if m.id == ally.id {
			sawAlly = true
		}
		if m.id == enemy.id {
			sawEnemy = true
		}
	}
	if sawAlly {
		t.Error("mobsWithinLocked returned an ALLY: every damage/CC op routes through it")
	}
	if !sawEnemy {
		t.Fatal("mobsWithinLocked dropped the enemy too -- the filter rejects everything")
	}

	// And through a real op, end to end.
	op := gamedata.Op{Kind: gamedata.OpDamage, Value: gamedata.PerLevel{100, 100, 100, 100}, Radius: 6}
	s.applyOpsLocked(c, []gamedata.Op{op}, opCtx{slot: 1, level: 1, hasPos: true, px: c.x, py: c.y}, now)
	if ally.hp != 5000 {
		t.Errorf("OpDamage hit the player's own creep: hp %g, want 5000", ally.hp)
	}
	if enemy.hp >= 5000 {
		t.Error("OpDamage did not reach the enemy creep -- the op is inert, not filtered")
	}
}

// TestSkillLineSwathSkipsAllies: the same for the line/rift scan, which is a separate
// loop with its own copy of the dead-only filter.
func TestSkillLineSwathSkipsAllies(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	ally := dotaAlly(t, inst, 61003, dotaPlayerTeam, c.x+4, c.y, now)
	enemy := dotaAlly(t, inst, 61004, dotaEnemyTeam, c.x+6, c.y, now)

	got := c.mobsAlongLineLocked(c.x, c.y, c.x+10, c.y, 2, 12)
	for _, m := range got {
		if m.id == ally.id {
			t.Error("mobsAlongLineLocked returned an ALLY")
		}
	}
	var sawEnemy bool
	for _, m := range got {
		if m.id == enemy.id {
			sawEnemy = true
		}
	}
	if !sawEnemy {
		t.Fatal("the swath dropped the enemy too -- the filter rejects everything")
	}
}

// TestAutoAttackDoesNotAcquireAlly: nearestAttackableMobLocked is SHARED and drives the
// server-side auto-attack resume, which bypasses the client-issued path's ally guard
// entirely. The ally is placed CLOSER than the enemy so a scan that ignored teams would
// certainly return it.
func TestAutoAttackDoesNotAcquireAlly(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	ally := dotaAlly(t, inst, 61005, dotaPlayerTeam, c.x+1, c.y, now)
	enemy := dotaAlly(t, inst, 61006, dotaEnemyTeam, c.x+5, c.y, now)

	got := s.nearestAttackableMobLocked(c, now, mobAggroRange)
	if got == nil {
		t.Fatal("auto-attack found nothing at all -- the filter rejects everything")
	}
	if got.id == ally.id {
		t.Fatal("auto-attack acquired the player's own creep")
	}
	if got.id != enemy.id {
		t.Fatalf("auto-attack acquired %d, want the enemy %d", got.id, enemy.id)
	}
}

// TestBounceScansSkipAllies covers the two skull-hop scans, which pick their own targets
// server-side and never see the client's targeting rules.
func TestBounceScansSkipAllies(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	ally := dotaAlly(t, inst, 61007, dotaPlayerTeam, c.x+1, c.y, now)
	enemy := dotaAlly(t, inst, 61008, dotaEnemyTeam, c.x+3, c.y, now)

	if got := s.nearestMobLocked(c, c.x, c.y, 8, 0); got == nil || got.id == ally.id {
		t.Fatalf("nearestMobLocked = %v, want the enemy %d", got, enemy.id)
	}
	for i := 0; i < 30; i++ { // random pick: sample it
		if got := s.randomMobInRangeLocked(c, c.x, c.y, 8, 0); got != nil && got.id == ally.id {
			t.Fatal("randomMobInRangeLocked returned the player's own creep")
		}
	}
}

// TestFriendSkillStillTargetsAlly: the ally filter must be driven by the skill's own
// declared SkillTarget mask, not applied blindly -- four real skills are cast ON an ally
// («Щит хранителя», «Касание спасителя», «Древесный камуфляж», «Дубовая кора»).
func TestFriendSkillStillTargetsAlly(t *testing.T) {
	enemyOnly := gamedata.Skill{Target: "ENEMY+NOT_BUILDING"}
	friendly := gamedata.Skill{Target: "FRIEND"}
	if skillHasTargetFlag(enemyOnly, "FRIEND") {
		t.Error("ENEMY+NOT_BUILDING must not report the FRIEND flag")
	}
	if !skillHasTargetFlag(friendly, "FRIEND") {
		t.Error("FRIEND skill must report the FRIEND flag")
	}
	if !skillHasTargetFlag(enemyOnly, "NOT_BUILDING") {
		t.Error("a '+'-joined flag must be found")
	}
	// Whole-token matching: NOT_FRIEND contains "FRIEND" as a substring, and a naive
	// Contains() test would invert the rule for any skill that ever declares it.
	if skillHasTargetFlag(gamedata.Skill{Target: "NOT_FRIEND"}, "FRIEND") {
		t.Error("NOT_FRIEND must not match a FRIEND probe")
	}
	// The real roster must still carry its four FRIEND skills -- if this drops to zero
	// the guard above is untested in production.
	n := 0
	for _, a := range gamedata.Avatars() {
		for _, sk := range gamedata.SkillsFor(a).Skills {
			if skillHasTargetFlag(sk, "FRIEND") {
				n++
			}
		}
	}
	if n == 0 {
		t.Error("no FRIEND-target skills in the roster: the single-target ally exemption is dead code")
	}
}

// TestRespawnLeavesCreepsAlone: a player respawning near their own base used to fling
// every creep within respawnEvictRange to the map origin, because a creep's spawnX/spawnY
// are the zero value. The homed Hunt mob beside it is the positive control.
func TestRespawnLeavesCreepsAlone(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	// Respawn at the base the player is standing on, which is where a «Штурм» player
	// actually revives -- otherwise the fixture's zero-valued map sends them to the
	// origin and nothing is inside the eviction ring at all.
	c.huntState.respawnX, c.huntState.respawnY = c.x, c.y

	// Creep standing right on the respawn point, as a defending lane does.
	creep := dotaAlly(t, inst, 61009, dotaPlayerTeam, c.x+1, c.y+1, now)
	creep.homed = false
	creep.spawnX, creep.spawnY = 0, 0 // the pre-fix state: no home was ever recorded
	cx, cy := creep.x, creep.y

	// A homed mob in the same ring MUST still be evicted, or this test would pass with
	// the eviction simply deleted.
	idx := mobIndexByPrefab(t, "Mob_Skeleton_1H_Melee_01")
	homed := &mobState{
		id: 2900, mobIdx: idx, mob: gamedata.Mobs()[idx],
		x: c.x + 1, y: c.y - 1, spawnX: c.x + 40, spawnY: c.y,
		hp: 50, maxHP: 92, homed: true, aggro: true, shown: true,
	}
	inst.mobs[homed.id] = homed

	s.respawnPlayerLocked(c, now)

	if creep.x != cx || creep.y != cy {
		t.Errorf("creep was evicted to (%.1f,%.1f) from (%.1f,%.1f) -- it has no home", creep.x, creep.y, cx, cy)
	}
	if creep.x == 0 && creep.y == 0 {
		t.Error("creep teleported to the map origin")
	}
	if homed.x != homed.spawnX || homed.y != homed.spawnY {
		t.Errorf("homed mob NOT evicted: at (%.1f,%.1f), want spawn (%.1f,%.1f) -- eviction is dead",
			homed.x, homed.y, homed.spawnX, homed.spawnY)
	}
}

// TestStructureResistsDisplacement: buildings are Stationary, and the effects engine was
// the one place in the server that ignored the flag. The creep is the positive control.
func TestStructureResistsDisplacement(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	var gun *mobState
	for _, m := range inst.mobs {
		if m.structure && m.dotaRole == gamedata.DotaGun && !m.dead {
			gun = m
			break
		}
	}
	if gun == nil {
		t.Fatal("precondition: no cannon on the map")
	}
	c.x, c.y, c.snapT = gun.x-4, gun.y, float32(now)
	gx, gy := gun.x, gun.y

	s.knockbackMobLocked(c, gun, 5, now)
	if gun.x != gx || gun.y != gy {
		t.Errorf("cannon was knocked back to (%.2f,%.2f) from (%.2f,%.2f)", gun.x, gun.y, gx, gy)
	}
	s.pullMobLocked(c, gun, now)
	if gun.x != gx || gun.y != gy {
		t.Errorf("cannon was pulled to (%.2f,%.2f) from (%.2f,%.2f)", gun.x, gun.y, gx, gy)
	}

	// A creep in the same spot must still move, or the guard is just "never displace".
	creep := dotaAlly(t, inst, 61010, dotaEnemyTeam, c.x+3, c.y, now)
	px := creep.x
	s.knockbackMobLocked(c, creep, 5, now)
	if creep.x <= px {
		t.Errorf("creep was NOT knocked back (%.2f -> %.2f): displacement is dead", px, creep.x)
	}
}
