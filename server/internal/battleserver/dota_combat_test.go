package battleserver

import (
	"math"
	"net"
	"sync"
	"testing"
	"time"

	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// newDotaCaptureConn is newDotaConn with a packet tap: every push the server fans to
// the member is recorded so a test can assert on the wire, not just on server state.
func newDotaCaptureConn(t *testing.T) (*Server, *conn, *huntInstance, *[]battleproto.Packet, *sync.Mutex) {
	t.Helper()
	s := New(session.NewStore())
	srv, cli := net.Pipe()
	t.Cleanup(func() { srv.Close(); cli.Close() })

	var mu sync.Mutex
	pkts := &[]battleproto.Packet{}
	r := battleproto.NewReader(cli)
	go func() {
		for {
			p, err := r.Read()
			if err != nil {
				return
			}
			mu.Lock()
			*pkts = append(*pkts, p)
			mu.Unlock()
		}
	}()

	dm := gamedata.DotaMaps()[0]
	inst := newDotaInstance(s, dm.ID, dm.ID)
	av := avatarByPrefab(t, "Avtr_Tank_Velial")
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
	// As the real join does (server.go). Without it every «Штурм» test ran on a conn with
	// no navmesh, so anything gated on c.nav -- the walkable clamps, the route-around, the
	// separation guard -- was silently skipped and could not have been tested at all.
	c.nav = inst.nav
	c.lk = &inst.mu
	inst.members[c.objID] = c
	return s, c, inst, pkts, &mu
}

// structWithEnemyInRange returns a live structure of the given role on the player's own
// side, and plants an enemy creep `gap` units away from it -- an ordinary lane creep, so
// the structure acquires it exactly as it would in a real push. Both are rendered on the
// member: broadcasts only reach members that actually track the object, so an unrevealed
// structure would swing in silence and the wire assertions would be vacuous.
func structWithEnemyInRange(t *testing.T, s *Server, c *conn, inst *huntInstance, role gamedata.DotaRole, gap float32, now float64) (*mobState, *mobState) {
	t.Helper()
	var st *mobState
	for _, m := range inst.mobs {
		if m.structure && m.dotaRole == role && m.team == dotaPlayerTeam && !m.dead {
			st = m
			break
		}
	}
	if st == nil {
		t.Fatalf("precondition: no live %v on the player's side", role)
	}
	s.dotaRevealStructureLocked(c, st, now)
	idx := inst.dota.m.ElfCreepMelee
	victim := &mobState{
		id: 61000, mobIdx: idx, mob: gamedata.Mobs()[idx],
		x: st.x + gap, y: st.y, hp: 5000, maxHP: 5000,
		team: dotaEnemyTeam, lastSync: now,
	}
	inst.mobs[victim.id] = victim
	return st, victim
}

// projectilesFrom counts the SET_PROJECTILE packets emitted by one source object, and
// returns the last one's hit_at.
func projectilesFrom(t *testing.T, pkts *[]battleproto.Packet, mu *sync.Mutex, source int32) (int, float64) {
	t.Helper()
	time.Sleep(50 * time.Millisecond) // let the pipe reader drain
	mu.Lock()
	defer mu.Unlock()
	n, hitAt := 0, 0.0
	for _, p := range *pkts {
		if p.Cmd != battleproto.CmdSetProjectile {
			continue
		}
		if src, _ := p.Args.GetInt("source"); src == source {
			n++
			if v, ok := p.Args.Assoc["hit_at"].(float64); ok {
				hitAt = v
			}
		}
	}
	return n, hitAt
}

// actionsFor returns the action ids of the ACTION and ACTION_DONE packets sent for objID.
func actionsFor(t *testing.T, pkts *[]battleproto.Packet, mu *sync.Mutex, objID int32) (started, done []int32) {
	t.Helper()
	time.Sleep(50 * time.Millisecond)
	mu.Lock()
	defer mu.Unlock()
	for _, p := range *pkts {
		id, _ := p.Args.GetInt("id")
		if id != objID {
			continue
		}
		switch p.Cmd {
		case battleproto.CmdAction:
			a, _ := p.Args.GetInt("action")
			started = append(started, a)
		case battleproto.CmdActionDone:
			a, _ := p.Args.GetInt("action")
			done = append(done, a)
		}
	}
	return started, done
}

// --- Bug 1: cannons show no flying shell ---

// TestDotaCannonFiresVisibleShell pins the «Штурм» cannon's projectile. The cannon
// prefabs (GA_Human/Elf_Gun_prop01) are the only buildings carrying a client projectile
// pool (VisualEffectOptions.mProjectiles -> GA_*_Gun_projectile_prop01), but the server
// only ever sent the swing ACTION, so the shell never existed: damage simply appeared on
// the target. SET_PROJECTILE is what flies it.
//
// The client (VisualBattle.OnSetProjectile) drops the shot unless hit_at is strictly in
// the FUTURE -- it sizes the flight as hit_at - Battle.Time -- so the hit may not be
// applied at swing time; it has to be committed and landed on arrival.
func TestDotaCannonFiresVisibleShell(t *testing.T) {
	s, c, inst, pkts, mu := newDotaCaptureConn(t)
	now := float64(s.battleTime())

	c.lock()
	gun, victim := structWithEnemyInRange(t, s, c, inst, gamedata.DotaGun, 6, now)
	s.revealMobToMemberLocked(c, victim, now)
	s.dotaTickLocked(c, now)
	launchAt, hpAtSwing := gun.projLaunchAt, victim.hp
	c.unlock()

	if launchAt == 0 {
		t.Fatal("cannon never committed a shot at an enemy 6u away -- test would be vacuous")
	}
	if hpAtSwing < 5000 {
		t.Fatal("cannon damage landed at swing time; the shell has not even left the barrel yet")
	}
	if n, _ := projectilesFrom(t, pkts, mu, gun.id); n != 0 {
		t.Fatalf("shell left the barrel on the swing frame (%d SET_PROJECTILE); it must wait for the wind-up to finish", n)
	}

	// Tick to the release: the shell flies and the hit is scheduled for its arrival.
	rel := launchAt + 1e-3
	c.lock()
	s.dotaTickLocked(c, rel)
	hitAt := gun.hitAt
	c.unlock()

	n, wireHitAt := projectilesFrom(t, pkts, mu, gun.id)
	if n != 1 {
		t.Fatalf("expected exactly 1 SET_PROJECTILE from the cannon at release, got %d", n)
	}
	if wireHitAt <= rel {
		t.Fatalf("SET_PROJECTILE hit_at %g is not in the future at release %g -- "+
			"OnSetProjectile computes hit_at-Battle.Time and draws nothing when it is <= 0", wireHitAt, rel)
	}
	if hitAt <= rel || hitAt > gun.nextSwing {
		t.Fatalf("hit at %g not scheduled between the release %g and the next swing %g", hitAt, rel, gun.nextSwing)
	}

	// The shell lands its damage on arrival.
	c.lock()
	s.dotaTickLocked(c, hitAt+1e-3)
	hp := victim.hp
	c.unlock()
	if hp >= 5000 {
		t.Fatalf("the shell arrived but dealt no damage (hp %g) -- the committed hit was dropped", hp)
	}
}

// TestDotaTowerFiresNoShell guards the other half of the asset fact: the creep TOWERS
// shoot just as far as a cannon but their prefabs ship an EMPTY projectile pool, so a
// SET_PROJECTILE from one only makes the client log "Object projectile prefab is null"
// and draw nothing. Range must therefore NOT be what decides who shoots a shell.
func TestDotaTowerFiresNoShell(t *testing.T) {
	s, c, inst, pkts, mu := newDotaCaptureConn(t)
	now := float64(s.battleTime())

	c.lock()
	tower, victim := structWithEnemyInRange(t, s, c, inst, gamedata.DotaCreepTower, 5, now)
	s.revealMobToMemberLocked(c, victim, now)
	// Drive well past any wind-up/release point a shell would have used.
	for bt := now; bt <= now+2.0; bt += 0.2 {
		s.dotaTickLocked(c, bt)
	}
	swung, hp := tower.nextSwing > now, victim.hp
	c.unlock()

	if !swung {
		t.Fatal("tower never swung at an enemy 5u away -- test would be vacuous")
	}
	if hp >= 5000 {
		t.Fatal("tower dealt no damage -- it must still hit, just without a visible shell")
	}
	if n, _ := projectilesFrom(t, pkts, mu, tower.id); n != 0 {
		t.Fatalf("tower emitted %d SET_PROJECTILE; its prefab has no projectile pool, so the client "+
			"would only log an error and draw nothing", n)
	}
}

// --- Bug 2: creeps keep moving during the attack animation ---

// TestDotaCreepHoldsStillThroughItsSwing reproduces "у крипов при анимации атаки не
// останавливается движение". The client plays the attack clip on layer 1 BLENDED over
// the run clip on layer 0 (AnimationExt.ActWithBlending) and only stops when ACTION_DONE
// clears the action (Battle.OnActionDone -> InstanceData.StopAction). The server sent the
// ACTION and never the ACTION_DONE, so the swing hung on the creep forever and it marched
// down the lane mid-strike. A creep must stay planted for its whole swing even after the
// thing it was hitting is gone.
func TestDotaCreepHoldsStillThroughItsSwing(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())

	lane := inst.dota.m.Lanes[0]
	idx := inst.dota.m.HumanCreepMelee
	ally := &mobState{
		id: 60001, mobIdx: idx, mob: gamedata.Mobs()[idx],
		x: 0, y: 0, hp: 500, maxHP: 500, dmgMin: 5, dmgMax: 8,
		team: dotaPlayerTeam, lane: lane, laneFwd: true, lastSync: now,
	}
	eidx := inst.dota.m.ElfCreepMelee
	enemy := &mobState{
		id: 60002, mobIdx: eidx, mob: gamedata.Mobs()[eidx],
		x: 1, y: 0, hp: 500, maxHP: 500, dmgMin: 5, dmgMax: 8,
		team: dotaEnemyTeam, lastSync: now,
	}

	c.lock()
	inst.mobs[ally.id] = ally
	inst.mobs[enemy.id] = enemy
	s.revealMobToMemberLocked(c, ally, now)
	s.revealMobToMemberLocked(c, enemy, now)
	s.dotaTickLocked(c, now)
	swingEnds := ally.swingDoneAt
	c.unlock()

	if swingEnds == 0 {
		t.Fatal("creep swung without scheduling an ACTION_DONE -- the client would keep the strike clip forever")
	}
	if ally.vx != 0 || ally.vy != 0 {
		t.Fatalf("creep is moving on its swing frame (v=%.2f,%.2f)", ally.vx, ally.vy)
	}

	// Its target dies mid-swing: with nothing left to fight the creep wants to march on,
	// but the strike animation is still playing -- it must stay put until the swing closes.
	c.lock()
	enemy.dead = true
	mid := (now + swingEnds) / 2
	s.dotaTickLocked(c, mid)
	vxMid, vyMid := ally.vx, ally.vy
	c.unlock()
	if vxMid != 0 || vyMid != 0 {
		t.Fatalf("creep marched off mid-swing (v=%.2f,%.2f) -- it would slide down the lane "+
			"with its weapon swinging", vxMid, vyMid)
	}

	// Once the swing is closed out it is free to march again.
	c.lock()
	s.dotaTickLocked(c, swingEnds+0.05)
	closed := ally.swingDoneAt == 0
	vx, vy := ally.vx, ally.vy
	c.unlock()
	if !closed {
		t.Fatal("swing was never closed out past its end time")
	}
	if vx == 0 && vy == 0 {
		t.Fatal("creep never resumed its lane march after the swing finished")
	}
}

// TestDotaSwingActionDoneMatchesActionId pins the id contract the animation hangs on:
// the client keys actions by id (InstanceData.DoAction/StopAction) and only animates the
// one matching the object's own AttackActionId, so an ACTION_DONE carrying a different id
// clears nothing. A «Штурм» structure's swing uses the shared building effector, NOT a
// mob-roster id -- and the close-out has to agree.
func TestDotaSwingActionDoneMatchesActionId(t *testing.T) {
	s, c, inst, pkts, mu := newDotaCaptureConn(t)
	now := float64(s.battleTime())

	c.lock()
	gun, victim := structWithEnemyInRange(t, s, c, inst, gamedata.DotaGun, 6, now)
	s.revealMobToMemberLocked(c, victim, now)
	for bt := now; bt <= now+2.0; bt += 0.1 {
		s.dotaTickLocked(c, bt)
	}
	c.unlock()

	started, done := actionsFor(t, pkts, mu, gun.id)
	if len(started) == 0 {
		t.Fatal("cannon never swung -- test would be vacuous")
	}
	if len(done) == 0 {
		t.Fatal("cannon's swing was never closed out with an ACTION_DONE")
	}
	if started[0] != dotaStructAttackProtoID {
		t.Fatalf("cannon ACTION action=%d, want the building attack effector %d", started[0], dotaStructAttackProtoID)
	}
	if done[0] != started[0] {
		t.Fatalf("ACTION_DONE action=%d does not match the ACTION's %d -- StopAction would clear "+
			"nothing and the strike clip would hang", done[0], started[0])
	}
}

// --- Bug 3: Grimlok's summoned dinosaur mauls allied creeps ---

// TestSummonSkipsAlliedCreeps reproduces "у Гримлока призываемый 1 скиллом динозавр
// пытается атаковать союзных крипов". A summon fights for its owner, but it picked the
// NEAREST mob of any kind -- and in «Штурм» the mob set also holds the player's own lane
// creeps and base buildings, which march right past the pet.
func TestSummonSkipsAlliedCreeps(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	hs := c.huntState

	sm := &summonState{id: 300001, proto: 800, x: 0, y: 0, hp: 100, maxHP: 100, dmg: 10, until: now + 60}
	hs.summons[sm.id] = sm

	aidx, eidx := inst.dota.m.HumanCreepMelee, inst.dota.m.ElfCreepMelee
	ally := &mobState{ // an ally right on top of the pet
		id: 60001, mobIdx: aidx, mob: gamedata.Mobs()[aidx],
		x: 1, y: 0, hp: 500, maxHP: 500, team: dotaPlayerTeam, lastSync: now,
	}
	enemy := &mobState{ // the real enemy, further away
		id: 60002, mobIdx: eidx, mob: gamedata.Mobs()[eidx],
		x: 6, y: 0, hp: 500, maxHP: 500, team: dotaEnemyTeam, lastSync: now,
	}
	c.lock()
	inst.mobs[ally.id] = ally
	inst.mobs[enemy.id] = enemy
	s.tickSummonsLocked(c, now)
	c.unlock()

	if ally.hp < 500 {
		t.Fatalf("the pet hit its own side's creep (hp %g) -- it must never target team %d", ally.hp, dotaPlayerTeam)
	}
	// It should have gone for the enemy instead: either already swinging, or closing in.
	movingAtEnemy := sm.vx > 0
	if enemy.hp >= 500 && !movingAtEnemy {
		t.Fatalf("the pet ignored the enemy creep entirely (enemy hp %g, pet v=%.2f,%.2f)", enemy.hp, sm.vx, sm.vy)
	}
}

// TestSummonStillFightsHuntMobs guards the other side of that filter: a plain Hunt mob
// carries no team at all (team 0 -> teamVal() -1), and must stay a valid target.
func TestSummonStillFightsHuntMobs(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	hs := c.huntState

	sm := &summonState{id: 300001, proto: 800, x: 0, y: 0, hp: 100, maxHP: 100, dmg: 10, until: now + 60}
	hs.summons[sm.id] = sm
	mob := &mobState{id: 2001, mobIdx: 0, mob: gamedata.Mobs()[0], x: 1, y: 0, hp: 500, maxHP: 500}
	hs.mobs[mob.id] = mob

	c.mvMu.Lock()
	s.tickSummonsLocked(c, now)
	c.mvMu.Unlock()

	if mob.hp >= 500 {
		t.Fatalf("the pet refused to fight an ordinary Hunt mob (hp %g) -- the «Штурм» team filter "+
			"must not swallow team-less mobs", mob.hp)
	}
}

// --- Bug 4: Miriam's «Выстрел бури» does not knock back ---

// TestMiriamStormShotDataKnocksBack pins the DATA: the shipped client description
// (IDS_MiriamSkill1_LongDesc) promises "...отбрасывая назад и обездвиживая на 2 секунды",
// and the knockback was the one part never modelled -- the shot read as a plain stun-nuke.
func TestMiriamStormShotDataKnocksBack(t *testing.T) {
	av := avatarByPrefab(t, "Avtr_DPS_Miriam")
	sk := gamedata.SkillsFor(av).Skills[0]
	if sk.NameRu != "Выстрел бури" {
		t.Fatalf("slot 1 is %q, not «Выстрел бури» -- the roster moved", sk.NameRu)
	}
	var knock, stun *gamedata.Op
	for i := range sk.Ops {
		switch sk.Ops[i].Kind {
		case gamedata.OpKnockback:
			knock = &sk.Ops[i]
		case gamedata.OpStun:
			stun = &sk.Ops[i]
		}
	}
	if knock == nil {
		t.Fatal("«Выстрел бури» has no knockback op, but its description promises «отбрасывая назад»")
	}
	if stun == nil {
		t.Fatal("«Выстрел бури» lost its «обездвиживая» stun")
	}
	if knock.Value.At(0) <= 0 {
		t.Fatalf("knockback distance %g pushes nobody anywhere", knock.Value.At(0))
	}
	if knock.Radius != 0 {
		t.Fatalf("knockback Radius=%g makes it an AoE shove; «Выстрел бури» is single-target "+
			"(a 0 radius resolves to exactly the clicked target)", knock.Radius)
	}
}

// TestKnockbackShovesTargetAwayFromCaster: the engine half. The shove is applied to a
// STUNNED target (its own stun lands with it), so it must not be gated on the target
// being free to move.
func TestKnockbackShovesTargetAwayFromCaster(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Miriam")
	defer cleanup()
	now := float64(s.battleTime())
	cx, cy := c.posAtLocked(float32(now))

	m := &mobState{id: 2001, mobIdx: 0, mob: gamedata.Mobs()[0],
		x: cx + 5, y: cy, hp: 500, maxHP: 500}
	m.st.stunUntil = now + 2 // «обездвиживая»: already rooted when the shove lands
	c.huntState.mobs[m.id] = m
	before := math.Hypot(float64(m.x-cx), float64(m.y-cy))

	c.mvMu.Lock()
	s.knockbackMobLocked(c, m, 4, now)
	c.mvMu.Unlock()

	after := math.Hypot(float64(m.x-cx), float64(m.y-cy))
	if after <= before+0.5 {
		t.Fatalf("a stunned target was not shoved away: %.2f -> %.2f from the caster", before, after)
	}
	if m.vx != 0 || m.vy != 0 {
		t.Fatalf("shoved target kept its velocity (%.2f,%.2f) -- it would dead-reckon away from where it landed", m.vx, m.vy)
	}
}

// TestDotaCreepRespectsStun: «Выстрел бури» stuns for 2s, but the «Штурм» tick read no
// status at all, so a stunned lane creep marched on regardless -- and a shoved one walked
// straight back, which is what made the knockback look like it never happened.
func TestDotaCreepRespectsStun(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())

	lane := inst.dota.m.Lanes[0]
	idx := inst.dota.m.HumanCreepMelee
	creep := &mobState{
		id: 60001, mobIdx: idx, mob: gamedata.Mobs()[idx],
		x: float32(lane[0].X), y: float32(lane[0].Y),
		hp: 500, maxHP: 500, team: dotaPlayerTeam,
		lane: lane, laneFwd: true, lastSync: now,
	}
	c.lock()
	inst.mobs[creep.id] = creep
	s.revealMobToMemberLocked(c, creep, now)

	// Free: it marches.
	s.dotaTickLocked(c, now)
	marching := creep.vx != 0 || creep.vy != 0

	// Stunned: it freezes where it stands.
	creep.st.stunUntil = now + 2
	s.dotaTickLocked(c, now+0.2)
	stunnedVx, stunnedVy := creep.vx, creep.vy
	atStun := [2]float32{creep.x, creep.y}
	s.dotaTickLocked(c, now+1.0)
	drift := math.Hypot(float64(creep.x-atStun[0]), float64(creep.y-atStun[1]))

	// Stun over: it marches again.
	s.dotaTickLocked(c, now+2.5)
	resumed := creep.vx != 0 || creep.vy != 0
	c.unlock()

	if !marching {
		t.Fatal("creep never marched its lane -- test would be vacuous")
	}
	if stunnedVx != 0 || stunnedVy != 0 {
		t.Fatalf("stunned creep kept marching (v=%.2f,%.2f)", stunnedVx, stunnedVy)
	}
	if drift > 0.01 {
		t.Fatalf("stunned creep drifted %.2fu -- a knocked-back creep would walk straight back", drift)
	}
	if !resumed {
		t.Fatal("creep never resumed marching after its stun expired")
	}
}
