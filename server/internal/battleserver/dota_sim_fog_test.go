package battleserver

import (
	"sync"
	"testing"
	"time"

	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// This file pins the rule the two modes now share: VISIBILITY IS NOT SIMULATION.
// What a player is shown may never decide what a unit is allowed to do -- an enemy
// creep breaks a tower whether or not anyone can see it. Hunt's interest management is
// allowed to skip an AI only where doing so is provably a no-op, and «Штурм» skips
// nothing at all.

// TestDotaCreepFightsWithNoPlayerAnywhere is the headline: park the player at the far
// end of the map -- outside every Hunt reveal/hide radius, and in fog as far as the
// client is concerned -- and the creep must still march up to a tower and break it.
func TestDotaCreepFightsWithNoPlayerAnywhere(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	var tower *mobState
	for _, m := range inst.mobs {
		if m.structure && m.dotaRole == gamedata.DotaCreepTower && m.team == dotaPlayerTeam && !m.dead {
			tower = m
			break
		}
	}
	if tower == nil {
		t.Fatal("precondition: no friendly tower on the map")
	}
	// The player is 1000 units away: >> mobHideRadius (34). Under the old rule -- where
	// the AI was gated on the render flag -- this froze everything.
	c.x, c.y, c.snapT = 1000, 1000, float32(now)

	idx := inst.dota.m.ElfCreepMelee
	creep := &mobState{
		id: 62001, mobIdx: idx, mob: gamedata.Mobs()[idx],
		x: tower.x + 1, y: tower.y, hp: 5000, maxHP: 5000,
		dmgMin: 40, dmgMax: 40,
		team: dotaEnemyTeam, lastSync: now,
	}
	inst.mobs[creep.id] = creep
	hp0 := tower.hp

	for bt := now; bt <= now+3.0; bt += 0.1 {
		s.dotaTickLocked(c, bt)
	}

	if !creep.active {
		t.Error("creep is not active with no player near -- the AI is gated on visibility again")
	}
	if tower.hp >= hp0 {
		t.Fatalf("tower took no damage (hp %g -> %g): an unseen creep stopped pushing", hp0, tower.hp)
	}
}

// TestDotaUnitsAreRenderedAndSimulated: «Штурм» culls nothing. A late joiner is handed
// the live lane, which is the flag introduceMemberLocked reads.
func TestDotaUnitsAreRenderedAndSimulated(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	var gen gamedata.DotaStructure
	for _, sc := range inst.dota.m.Structures {
		if sc.Role == gamedata.DotaGenerator {
			gen = sc
			break
		}
	}
	s.dotaSpawnCreepWaveLocked(c, gen, now)

	n := 0
	for _, m := range inst.mobs {
		if m.structure || m.dead {
			continue
		}
		n++
		if !m.shown {
			t.Errorf("creep %d is not shown: a joiner mid-match would never receive it", m.id)
		}
	}
	if n == 0 {
		t.Fatal("no creeps spawned -- the assertion above is vacuous")
	}
}

// TestCreepRevealCarriesViewRadius / TestSummonRevealCarriesViewRadius: fog is one
// shared visual layer, so EVERY friendly unit lights it. Creeps and summons were the two
// that did not, which left a «Штурм» lane black beyond the avatar's own circle.
// WarFogObject.Update needs BOTH mViewRadius > 0 and a known friendliness, so each blob
// must carry VIEW_RADIUS and TEAM.
func TestCreepRevealCarriesViewRadius(t *testing.T) {
	s, c, syncs := newDotaFogConn(t)

	idx := gamedata.DotaMaps()[0].HumanCreepMelee
	m := &mobState{id: 62100, mobIdx: idx, mob: gamedata.Mobs()[idx], x: 1, y: 1, hp: 10, maxHP: 10, team: dotaPlayerTeam}
	c.huntState.mobs[m.id] = m

	go func() {
		c.lock()
		s.revealMobToMemberLocked(c, m, float64(s.battleTime()))
		c.unlock()
	}()
	awaitViewRadiusSync(t, syncs)
}

func TestSummonRevealCarriesViewRadius(t *testing.T) {
	s, c, syncs := newDotaFogConn(t)

	sm := &summonState{id: 300009, hp: 100, maxHP: 100, x: 2, y: 2, until: 1e9}
	c.huntState.summons[sm.id] = sm

	go func() {
		c.lock()
		s.revealSummonToMemberLocked(c, sm, float64(s.battleTime()))
		c.unlock()
	}()
	awaitViewRadiusSync(t, syncs)
}

// newDotaFogConn is a «Штурм» member whose SYNC pushes are decoded, for the fog blobs.
func newDotaFogConn(t *testing.T) (*Server, *conn, <-chan []byte) {
	t.Helper()
	s := New(session.NewStore())
	c, syncs := collectSyncs(t)
	dm := gamedata.DotaMaps()[0]
	inst := newDotaInstance(s, dm.ID, dm.ID)
	av := avatarByPrefab(t, "Avtr_Tank_Velial")
	c.objID = 1000
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
	return s, c, syncs
}

// TestSlowActuallySlowsDotaCreep: OpSlow used to land its VFX, broadcast a reduced SPEED
// and change nothing, because the «Штурм» mover read m.mob.Speed raw.
func TestSlowActuallySlowsDotaCreep(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	idx := inst.dota.m.ElfCreepMelee
	creep := &mobState{
		id: 62002, mobIdx: idx, mob: gamedata.Mobs()[idx],
		x: 0, y: 0, hp: 5000, maxHP: 5000, team: dotaEnemyTeam, lastSync: now,
	}
	inst.mobs[creep.id] = creep

	s.dotaMoveTowardLocked(c, creep, 100, 0, now)
	fast := creep.vx
	if fast <= 0 {
		t.Fatal("creep did not move at all")
	}

	creep.st.slowUntil = now + 5
	creep.st.slowFactor = 0.5
	creep.vx, creep.vy = 0, 0
	s.dotaMoveTowardLocked(c, creep, 100, 0, now)
	slow := creep.vx

	if slow >= fast {
		t.Errorf("slow did nothing: vx %.3f slowed vs %.3f unslowed", slow, fast)
	}
	if got, want := slow, fast*0.5; got < want-0.01 || got > want+0.01 {
		t.Errorf("slowed vx = %.3f, want ~%.3f (half)", got, want)
	}
}

// TestMobSpeedAppliesMoveFactorOnce: moveFactor already folds every move_speed_pct mod,
// so the mob paths must not multiply by it a second time. They used to -- identically in
// three places, which is why a 1.4x haste silently ran at 1.96x and matched its own SYNC.
func TestMobSpeedAppliesMoveFactorOnce(t *testing.T) {
	idx := mobIndexByPrefab(t, "Mob_Skeleton_1H_Melee_01")
	mob := gamedata.Mobs()[idx]
	m := &mobState{id: 1, mobIdx: idx, mob: mob}
	now := 100.0
	m.st.mods = append(m.st.mods, statMod{stat: "move_speed_pct", value: 1.4, until: now + 10})

	got := mobSpeed(m, now)
	want := mob.Speed * 1.4
	if got < want-1e-6 || got > want+1e-6 {
		t.Errorf("mobSpeed = %.4f, want %.4f (base %.4f x 1.4 ONCE, not %.4f squared)",
			got, want, mob.Speed, mob.Speed*1.4*1.4)
	}
}

// TestSilenceStopsDotaUnitAttacking: a mob has no spellbook, so a silence on one mutes
// its attack -- the rule the Hunt driver has always enforced and this one never read.
func TestSilenceStopsDotaUnitAttacking(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	gun, victim := structWithEnemyInRange(t, s, c, inst, gamedata.DotaGun, 3, now)

	gun.st.silenceUntil = now + 5
	s.dotaStructCombatLocked(c, gun, now)
	if gun.swingDoneAt != 0 {
		t.Error("a silenced cannon started a swing")
	}

	// Positive control: it fires once the silence lapses.
	gun.st.silenceUntil = 0
	s.dotaStructCombatLocked(c, gun, now)
	if gun.swingDoneAt == 0 {
		t.Fatalf("cannon never swung unsilenced (victim %d in range) -- the test proves nothing", victim.id)
	}
}

// TestStunCancelsCommittedSwing: the two drivers disagreed. Hunt held a committed swing
// through the stun and fired it late; «Штурм» fired dead on schedule mid-stun. Both now
// cancel, which is what the client renders either way.
func TestStunCancelsCommittedSwing(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	gun, _ := structWithEnemyInRange(t, s, c, inst, gamedata.DotaGun, 3, now)
	s.dotaStructCombatLocked(c, gun, now)
	if gun.swingDoneAt == 0 || (gun.projLaunchAt == 0 && gun.hitAt == 0) {
		t.Fatal("precondition: the cannon did not commit a swing")
	}
	nextBefore := gun.nextSwing

	s.stunMobLocked(c, gun, now, 2.0)

	if gun.swingDoneAt != 0 {
		t.Error("stun left the swing animation open")
	}
	if gun.projLaunchAt != 0 || gun.hitAt != 0 {
		t.Error("stun left a committed shot pending: it fires the moment the stun lapses")
	}
	if gun.nextSwing != nextBefore {
		t.Errorf("stun reset the attack cadence (%.2f -> %.2f): being stunned must not refund the cooldown",
			nextBefore, gun.nextSwing)
	}
}

// TestDotaCreepCanHitSummon: pets deal full damage in «Штурм» through the shared summon
// tick, but nothing could hit back -- summons live in their owner's map, and only the
// Hunt targeting was ever taught to look there.
func TestDotaCreepCanHitSummon(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	sm := &summonState{id: 300001, hp: 200, maxHP: 200, x: c.x + 1, y: c.y, until: now + 60}
	c.huntState.summons[sm.id] = sm

	idx := inst.dota.m.ElfCreepMelee
	creep := &mobState{
		id: 62003, mobIdx: idx, mob: gamedata.Mobs()[idx],
		x: sm.x + 1, y: sm.y, hp: 5000, maxHP: 5000,
		dmgMin: 30, dmgMax: 30, team: dotaEnemyTeam, lastSync: now,
	}
	inst.mobs[creep.id] = creep

	tgt := s.dotaAcquireTargetLocked(c, creep, dotaCreepAggro, now)
	if tgt == nil {
		t.Fatal("creep acquired nothing at all")
	}
	if tgt.summon == nil {
		t.Fatalf("creep acquired %d, not the summon standing 1 unit away", tgt.id())
	}

	// And the committed hit must actually resolve onto it.
	creep.hitTarget = sm.id
	creep.hitDmg = 30
	s.dotaLandHitLocked(c, creep, now)
	if sm.hp >= 200 {
		t.Errorf("summon took no damage (hp %g): the swing evaporated -- pets are invulnerable", sm.hp)
	}
}

// TestDotaCreepKilledByPlayerIsNotLeaked: the despawn rule used to follow the ATTACKER --
// a creep killed by a tower was deleted, the same creep killed by the player was kept for
// a Hunt respawn that a «Штурм» world never runs, and accrued for the whole match.
func TestDotaCreepKilledByPlayerIsNotLeaked(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	idx := inst.dota.m.ElfCreepMelee
	creep := &mobState{
		id: 62004, mobIdx: idx, mob: gamedata.Mobs()[idx],
		x: c.x + 1, y: c.y, hp: 10, maxHP: 5000,
		team: dotaEnemyTeam, lastSync: now, homed: false,
	}
	inst.mobs[creep.id] = creep

	s.hitMobLocked(c, creep, 9999, c.objID)

	if !creep.dead {
		t.Fatal("creep survived a lethal hit")
	}
	if creep.respawnAt != 0 {
		t.Errorf("a homeless creep was armed with a respawn timer (%.0f): it would revive at (0,0) "+
			"if this world were ever driven by the Hunt pass", creep.respawnAt)
	}
	_ = now
}

// TestStunDoesNotUnfireAnAirborneShell: a stun cancels the WIND-UP, never a shell that
// is already flying. SET_PROJECTILE hands the client a hit_at it cannot recall (the
// prefab detaches to /objects and arrives regardless), so dropping the damage would
// render a visible impact for nothing. The player path locked this rule long ago --
// projectile_cancel_test.go calls the alternative "the bug".
func TestStunDoesNotUnfireAnAirborneShell(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	gun, victim := structWithEnemyInRange(t, s, c, inst, gamedata.DotaGun, 3, now)
	s.dotaStructCombatLocked(c, gun, now)
	if gun.projLaunchAt == 0 {
		t.Fatal("precondition: the cannon committed no shell")
	}

	// Advance to the release: SET_PROJECTILE goes out and the hit is promised.
	rel := gun.projLaunchAt
	s.dotaResolveSwingLocked(c, gun, rel)
	if !gun.projFlying || gun.hitAt == 0 {
		t.Fatalf("precondition: shell not airborne (flying=%v hitAt=%.2f)", gun.projFlying, gun.hitAt)
	}
	land := gun.hitAt
	hp0 := victim.hp

	// Stun mid-flight.
	s.stunMobLocked(c, gun, rel, 2.0)
	if gun.hitAt == 0 {
		t.Fatal("stun cancelled a shell that is already in the air: the client renders the impact anyway")
	}

	// It must still land on schedule -- not stall until the stun lapses.
	s.dotaResolveSwingLocked(c, gun, land)
	if victim.hp >= hp0 {
		t.Errorf("airborne shell dealt no damage (hp %g -> %g)", hp0, victim.hp)
	}
}

// TestStunDropsAnUnreleasedMeleeSwing is the contrast, mirroring TestMeleeHitDroppedOnCancel
// on the mob path: nothing is promised until the swing connects, so a stun drops it.
func TestStunDropsAnUnreleasedMeleeSwing(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	tower, victim := structWithEnemyInRange(t, s, c, inst, gamedata.DotaCreepTower, 3, now)
	s.dotaStructCombatLocked(c, tower, now)
	if tower.hitAt == 0 || tower.projFlying {
		t.Fatalf("precondition: tower is hitscan, expected a pending non-flying hit (hitAt=%.2f flying=%v)",
			tower.hitAt, tower.projFlying)
	}
	land, hp0 := tower.hitAt, victim.hp

	s.stunMobLocked(c, tower, now, 2.0)
	if tower.hitAt != 0 {
		t.Fatal("stun left an unreleased hitscan swing pending")
	}
	s.dotaResolveSwingLocked(c, tower, land)
	if victim.hp != hp0 {
		t.Errorf("a stunned tower's cancelled swing still dealt damage (hp %g -> %g)", hp0, victim.hp)
	}
}

// TestSecondHitOnDyingSummonIsDropped: the reap runs on the OWNER's tick, so a pet killed
// by one attacker stays in hs.summons with dead==false for the rest of the pass. Two
// creeps focus-firing it in one pass must not double-kill it.
func TestSecondHitOnDyingSummonIsDropped(t *testing.T) {
	s, c, inst, pkts, mu := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	sm := &summonState{id: 300011, hp: 30, maxHP: 30, x: c.x + 1, y: c.y, until: now + 60}
	c.huntState.summons[sm.id] = sm
	s.revealSummonToMemberLocked(c, sm, now)

	idx := inst.dota.m.ElfCreepMelee
	mk := func(id int32) *mobState {
		m := &mobState{id: id, mobIdx: idx, mob: gamedata.Mobs()[idx], x: sm.x + 1, y: sm.y,
			hp: 5000, maxHP: 5000, team: dotaEnemyTeam, lastSync: now}
		inst.mobs[id] = m
		return m
	}
	a, b := mk(62010), mk(62011)
	a.hitTarget, a.hitDmg = sm.id, 40
	b.hitTarget, b.hitDmg = sm.id, 40

	s.dotaLandHitLocked(c, a, now) // kills it: 30 - 40 = -10 (ordinary overkill)
	afterFirst := sm.hp
	if afterFirst > 0 {
		t.Fatalf("precondition: the first hit did not kill the pet (hp %g)", afterFirst)
	}

	s.dotaLandHitLocked(c, b, now) // must be dropped entirely

	if sm.hp != afterFirst {
		t.Errorf("second hit landed on a dead pet: hp %g -> %g", afterFirst, sm.hp)
	}
	if n := countCmd(t, pkts, mu, battleproto.CmdOnKill); n != 1 {
		t.Errorf("ON_KILL broadcast %d times for one pet, want 1", n)
	}
}

// countCmd counts packets of one command id.
func countCmd(t *testing.T, pkts *[]battleproto.Packet, mu *sync.Mutex, cmd battleproto.CmdID) int {
	t.Helper()
	time.Sleep(50 * time.Millisecond)
	mu.Lock()
	defer mu.Unlock()
	n := 0
	for _, p := range *pkts {
		if p.Cmd == cmd {
			n++
		}
	}
	return n
}

// TestSelfCastFriendBuffLandsOnCaster: an On:"target" buff with no unit target is a
// self-cast. It must never reach damageTargetsLocked's radius-4 fallback -- that scan is
// hostile-only, so the buff went to the enemies standing around the caster.
func TestSelfCastFriendBuffLandsOnCaster(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	enemy := dotaAlly(t, inst, 62012, dotaEnemyTeam, c.x+2, c.y, now)
	op := gamedata.Op{Kind: gamedata.OpBuffStat, Value: gamedata.PerLevel{30, 30, 30, 30},
		Dur: gamedata.PerLevel{5, 5, 5, 5}, Stat: "magic_armor", On: "target"}

	s.applyOpsLocked(c, []gamedata.Op{op}, opCtx{slot: 1, level: 1, hasPos: true, px: c.x, py: c.y}, now)

	for _, m := range enemy.st.mods {
		if m.stat == "magic_armor" {
			t.Fatal("a targetless friendly buff was applied to an ENEMY")
		}
	}
	if got := c.huntState.st.modSum(now, "magic_armor"); got != 30 {
		t.Errorf("caster magic_armor mod = %g, want 30 (the buff went nowhere)", got)
	}
}

// TestActiveNeverOutlivesShown: `active` is the simulation flag and must be cleared
// wherever a mob leaves the simulation, or mobSeparation (which reads it) steers live
// mobs around a body that is invisible and possibly teleported. respawnMobLocked runs
// from the TOP of the mob loop and then skips the rest of the tick, so nothing
// downstream would correct a stale value.
func TestActiveNeverOutlivesShown(t *testing.T) {
	s, c, _ := newHuntConn(t, "Avtr_Tank_Velial")
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	idx := mobIndexByPrefab(t, "Mob_Skeleton_1H_Melee_01")
	m := &mobState{
		id: 2600, mobIdx: idx, mob: gamedata.Mobs()[idx],
		x: c.x + 2, y: c.y, spawnX: c.x + 40, spawnY: c.y,
		hp: 1, maxHP: 92, homed: true, shown: true, active: true, dead: true,
	}
	c.huntState.mobs[m.id] = m

	s.respawnMobLocked(c, m, now)
	if m.shown {
		t.Fatal("precondition: a respawned mob is meant to stay hidden")
	}
	if m.active {
		t.Error("respawned mob is still active while hidden: mobSeparation counts a body that is not there")
	}

	m.shown, m.active = true, true
	s.removeMobFromClientsLocked(c, m, now)
	if m.active {
		t.Error("mob removed from the clients is still active")
	}
}
