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
	barracks := 0
	for _, sc := range inst.dota.m.Structures {
		if sc.Role == gamedata.DotaCreepTower {
			barracks++
		}
	}
	// One barracks owns one lane and sends CreepsPerWave down it.
	want := barracks * gamedata.CreepsPerWave
	if creeps != want {
		t.Fatalf("spawned %d creeps, want %d (%d barracks × %d)", creeps, want, barracks, gamedata.CreepsPerWave)
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
	// An enemy-team creep (team -1) sitting next to the player's own altar (team 1). That
	// altar is its enemy; it must pick it, never a same-team object. The altar is guarded by
	// the player's base cannons, so it starts invulnerable and (by the «Штурм» targeting rule)
	// is not a valid target -- raze its guards first so it opens.
	own := altarOf(inst, dotaPlayerTeam)
	creep := &mobState{
		id: 61000, mobIdx: inst.dota.m.ElfCreepMelee,
		mob:  gamedata.Mobs()[inst.dota.m.ElfCreepMelee],
		x:    own.x + 3, y: own.y,
		team: dotaEnemyTeam,
	}
	inst.mobs[creep.id] = creep

	c.lock()
	for _, gid := range inst.dota.altarGuards[own.id] {
		if g := inst.mobs[gid]; g != nil {
			g.dead = true
		}
	}
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

// TestDotaAltarOpensOnBaseCannons pins the fix for "the altar stays invulnerable after I destroy
// the two cannons next to it": the altar is gated by its OWN base cannons, not by all 11 of the
// side's guns. Killing a distant lane gun does nothing; killing the two base cannons opens the
// altar even while lane guns still stand.
func TestDotaAltarOpensOnBaseCannons(t *testing.T) {
	_, c, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	enemy := altarOf(inst, dotaEnemyTeam)
	if enemy == nil {
		t.Fatal("no enemy altar")
	}
	guards := inst.dota.altarGuards[enemy.id]
	if len(guards) != 2 {
		t.Fatalf("enemy altar guard set = %v, want exactly 2 base cannons", guards)
	}
	guardSet := map[int32]bool{guards[0]: true, guards[1]: true}

	c.lock()
	defer c.unlock()
	// Kill a NON-base enemy gun (a lane gun): the altar must stay shielded.
	killedLane := false
	for _, m := range inst.mobs {
		if m.dotaRole == gamedata.DotaGun && m.team == dotaEnemyTeam && !guardSet[m.id] {
			m.dead = true
			killedLane = true
			break
		}
	}
	if !killedLane {
		t.Fatal("no lane gun found to kill")
	}
	if inst.dota.altarVulnerableLocked(enemy) {
		t.Fatal("altar opened after a mere LANE gun died -- it should need its base cannons gone")
	}
	// Raze the two base cannons (only them); lane guns remain.
	for _, gid := range guards {
		inst.mobs[gid].dead = true
	}
	laneAlive := 0
	for _, m := range inst.mobs {
		if m.dotaRole == gamedata.DotaGun && m.team == dotaEnemyTeam && !m.dead && !guardSet[m.id] {
			laneAlive++
		}
	}
	if laneAlive == 0 {
		t.Fatal("precondition: expected lane guns still standing")
	}
	if !inst.dota.altarVulnerableLocked(enemy) {
		t.Fatalf("altar still invulnerable after both base cannons fell (%d lane guns alive)", laneAlive)
	}
}

// TestDotaCreepSkipsShieldedAltar pins the fix for "creeps attack the invulnerable altar while
// cannons stand": a creep next to a shielded enemy altar targets a guarding cannon, and only
// once the cannons fall does the (now open) altar become its target.
func TestDotaCreepSkipsShieldedAltar(t *testing.T) {
	s, c, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	enemyAltar := altarOf(inst, dotaEnemyTeam)
	guards := inst.dota.altarGuards[enemyAltar.id]
	if len(guards) == 0 {
		t.Fatal("enemy altar has no guard cannons")
	}
	// A player-side (team 1) creep standing right on the enemy altar while its cannons stand.
	creep := &mobState{
		id: 60009, mobIdx: inst.dota.m.HumanCreepMelee,
		mob:  gamedata.Mobs()[inst.dota.m.HumanCreepMelee],
		x:    enemyAltar.x - 2, y: enemyAltar.y,
		team: dotaPlayerTeam,
	}
	inst.mobs[creep.id] = creep

	c.lock()
	defer c.unlock()
	tgt := s.dotaAcquireTargetLocked(c, creep, 50, float64(s.battleTime()))
	if tgt == nil || tgt.mob == nil {
		t.Fatal("creep acquired no target next to the enemy base")
	}
	if tgt.mob.id == enemyAltar.id {
		t.Fatal("creep targeted the SHIELDED altar -- it should skip it and hit the guard cannons")
	}
	if tgt.mob.dotaRole != gamedata.DotaGun {
		t.Errorf("creep targeted role %d next to the base, want a guarding cannon (DotaGun)", tgt.mob.dotaRole)
	}
	// Raze the base cannons: the altar opens and becomes the creep's (nearest) target.
	for _, gid := range guards {
		inst.mobs[gid].dead = true
	}
	tgt2 := s.dotaAcquireTargetLocked(c, creep, 50, float64(s.battleTime()))
	if tgt2 == nil || tgt2.mob == nil || tgt2.mob.id != enemyAltar.id {
		t.Fatalf("after the base cannons fell, creep should target the open altar, got %+v", tgt2)
	}
}

// TestDotaCreepVsCreepEncodesSwing reproduces the "server crashes when lane creeps meet"
// bug: two opposing creeps in melee reach swing at each other, which broadcasts a
// CmdAction. The action args carried a nil targetPos, which the AMF encoder mis-handled
// as a typed *MixedArray nil and dereferenced -> panic on the ticker goroutine (whole
// server down). The tick must run and the swing must actually encode (member renders both
// creeps) without panicking, and the hit must land.
func TestDotaCreepVsCreepEncodesSwing(t *testing.T) {
	s, c, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	// Two opposing melee creeps 1m apart, far from any base structure (mid-map), so each
	// picks the other as its nearest enemy.
	ally := &mobState{
		id: 60001, mobIdx: inst.dota.m.HumanCreepMelee,
		mob: gamedata.Mobs()[inst.dota.m.HumanCreepMelee],
		x:   0, y: 0, hp: 500, maxHP: 500, dmgMin: 5, dmgMax: 8,
		team: dotaPlayerTeam, lastSync: now,
	}
	enemy := &mobState{
		id: 60002, mobIdx: inst.dota.m.ElfCreepMelee,
		mob: gamedata.Mobs()[inst.dota.m.ElfCreepMelee],
		x:   1, y: 0, hp: 500, maxHP: 500, dmgMin: 5, dmgMax: 8,
		team: dotaEnemyTeam, lastSync: now,
	}

	c.lock()
	inst.mobs[ally.id] = ally
	inst.mobs[enemy.id] = enemy
	// Reveal both to the member so the CmdAction broadcast actually reaches the encoder
	// (broadcastObjLocked only pushes to members that render the object) -- that encode is
	// where the crash lived.
	s.revealMobToMemberLocked(c, ally, now)
	s.revealMobToMemberLocked(c, enemy, now)
	// World passes: the first makes both creeps acquire each other and swing -- that
	// CmdAction encode is where the crash lived. The swing only COMMITS its hit though
	// (a melee blow connects mid-animation, at 0.5/attackSpeed ~= 0.56s), so keep
	// ticking past the connect point to see the clash actually resolve in damage.
	for bt := now; bt <= now+1.2; bt += 0.2 {
		s.dotaTickLocked(c, bt)
	}
	c.unlock()

	if enemy.hp >= 500 && ally.hp >= 500 {
		t.Fatalf("neither creep took damage -- the melee clash never resolved (ally=%g enemy=%g)", ally.hp, enemy.hp)
	}
}
