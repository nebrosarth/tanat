package battleserver

import (
	"encoding/binary"
	"math"
	"net"
	"strings"
	"testing"
	"time"

	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

func dialTestServerWithStore(t *testing.T, store *session.Store) (net.Conn, *battleproto.Reader) {
	t.Helper()
	s := New(store)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	go func() {
		conn, err := ln.Accept()
		if err != nil {
			return
		}
		s.handleConn(conn)
	}()
	c, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Close() })
	_ = c.SetDeadline(time.Now().Add(20 * time.Second))
	return c, battleproto.NewReader(c)
}

// enterHunt drives CONNECT+READY for a pending hunt battle and returns after
// the READY ack, leaving the world-state packets unread.
func enterHunt(t *testing.T, store *session.Store, avatarID int32) (net.Conn, *battleproto.Reader) {
	t.Helper()
	store.SetPendingBattle(42, session.PendingBattle{
		MapID: 40, AvatarID: avatarID, Passwd: "secret", Scene: "map_4_0",
	})
	c, r := dialTestServerWithStore(t, store)
	send(t, c, battleproto.Packet{Cmd: battleproto.CmdConnect, RequestID: 1, Status: true,
		Args: amf.NewArray().Set("clientId", int32(42)).Set("pass", "secret")})
	if _, err := r.Read(); err != nil {
		t.Fatalf("connect reply: %v", err)
	}
	send(t, c, battleproto.Packet{Cmd: battleproto.CmdReady, RequestID: 2, Status: true, Args: amf.NewArray()})
	if ack, err := r.Read(); err != nil || ack.Cmd != battleproto.CmdReady {
		t.Fatalf("expected READY ack, got %v (err %v)", ack, err)
	}
	return c, r
}

// readWorld drains the hunt world state and indexes the packets by command. It
// reads until the stream goes quiet (a short read deadline) so it stays robust
// as the world-state grows (buff/summon/mob-attack protos and effectors).
func readWorld(t *testing.T, c net.Conn, r *battleproto.Reader) map[battleproto.CmdID][]battleproto.Packet {
	t.Helper()
	got := map[battleproto.CmdID][]battleproto.Packet{}
	for {
		_ = c.SetReadDeadline(time.Now().Add(400 * time.Millisecond))
		p, err := r.Read()
		if err != nil {
			break
		}
		got[p.Cmd] = append(got[p.Cmd], p)
	}
	_ = c.SetReadDeadline(time.Now().Add(20 * time.Second))
	return got
}

// TestHuntWorldState checks the combat world state: avatar prototype with the
// real prefab, the 5-effector skill panel recipe (1 PARAMS + 4 SKILL parents
// with ACTIVE children) + ATTACK, and hostile mobs with TEAM=-1 syncs.
func TestHuntWorldState(t *testing.T) {
	avatar := gamedata.Avatars()[0] // Рогнар
	c, r := enterHunt(t, session.NewStore(), avatar.ID)
	got := readWorld(t, c, r)

	// Avatar prototype.
	var avatarProto battleproto.Packet
	for _, p := range got[battleproto.CmdPrototypeInfo] {
		if id, _ := p.Args.GetInt("id"); id == avatarProtoID(avatar.ID) {
			avatarProto = p
		}
	}
	desc, _ := avatarProto.Args.GetString("desc")
	if !strings.Contains(desc, `PPrefab value="`+avatar.Prefab+`"`) {
		t.Errorf("avatar proto desc missing prefab: %q", desc)
	}
	if !strings.Contains(desc, `<Name value="`+avatar.Name()+`"`) {
		t.Errorf("avatar proto desc missing locale name key: %q", desc)
	}

	// Avatar effectors: 10 of them (params + 4 parents + 4 children + attack)
	// owned by the avatar object; the children must reference their parents
	// with args.level = 1. Mobs also carry an ATTACK effector (owner = mob id),
	// which we exclude by owner.
	avatarObj := int32(1042) // 1000 + user id 42
	var effs []battleproto.Packet
	for _, p := range got[battleproto.CmdAddEffector] {
		if owner, _ := p.Args.GetInt("owner"); owner == avatarObj {
			effs = append(effs, p)
		}
	}
	if len(effs) != 10 {
		t.Fatalf("avatar ADD_EFFECTOR count = %d, want 10", len(effs))
	}
	parents := map[int32]bool{}
	children := 0
	for _, p := range effs {
		parent, _ := p.Args.GetInt("parent")
		if parent == -1 {
			id, _ := p.Args.GetInt("id")
			parents[id] = true
		}
	}
	for _, p := range effs {
		parent, _ := p.Args.GetInt("parent")
		if parent == -1 {
			continue
		}
		children++
		if !parents[parent] {
			t.Errorf("child effector references unknown parent %d", parent)
		}
		args, ok := p.Args.GetArray("args")
		if !ok {
			t.Fatalf("child effector missing args")
		}
		// Every skill starts UNLEARNED at rank 0 (the player buys rank 1 with the
		// starting skill point), so the child effectors ship at level 0.
		if lvl, _ := args.GetInt("level"); lvl != 0 {
			t.Errorf("child effector level = %d, want 0 (skills start unlearned)", lvl)
		}
	}
	if children != 4 {
		t.Errorf("child effector count = %d, want 4", children)
	}

	// SET_AVATAR ships 1 skill point at level 0 so exactly one skill can be raised
	// to rank 1 immediately (everything starts unlearned).
	var sawPoints bool
	for _, p := range got[battleproto.CmdSetAvatar] {
		sawPoints = true
		if pts, _ := p.Args.GetInt("points"); pts != 1 {
			t.Errorf("SET_AVATAR points = %d, want 1 (one point to spend at start)", pts)
		}
	}
	if !sawPoints {
		t.Error("no SET_AVATAR in world state")
	}

	// A skill prototype must carry the battle icon path and locale keys.
	var skillProto battleproto.Packet
	for _, p := range got[battleproto.CmdPrototypeInfo] {
		if id, _ := p.Args.GetInt("id"); id == skillProtoID(avatar, 1) {
			skillProto = p
		}
	}
	sdesc, _ := skillProto.Args.GetString("desc")
	if !strings.Contains(sdesc, `icon value="Gui/Icons/skills/`+avatar.Prefab+`_skill1"`) {
		t.Errorf("skill proto icon wrong: %q", sdesc)
	}
	if !strings.Contains(sdesc, `type value="SKILL"`) {
		t.Errorf("skill proto type wrong: %q", sdesc)
	}

	// Mobs: fog of war means the world state + first ticks create the avatar plus
	// only the mobs within mobRevealRadius of spawn (distant bosses stay hidden).
	// Each revealed mob SYNC has a TEAM int (hostile).
	m, _ := gamedata.HuntMapByID(40)
	sx, sy := m.Spawn()
	wantCreate := 1 // the avatar
	for _, sp := range m.Spawns {
		mx, my := sx+sp.DX, sy+sp.DY
		if sp.Abs {
			mx, my = sp.DX, sp.DY
		}
		if math.Hypot(mx-sx, my-sy) <= mobRevealRadius {
			wantCreate++
		}
	}
	if n := len(got[battleproto.CmdCreateObject]); n != wantCreate {
		t.Fatalf("CREATE_OBJECT count = %d, want %d (fog of war reveals only near mobs)", n, wantCreate)
	}
	// Among the world-state SYNCs, a mob's registration blob carries exactly one
	// add entry and a TEAM/POSITION/MAX_HEALTH mask. (Later tick syncs for mob
	// movement/regen have no add entry, so search rather than take the last.)
	var found bool
	for _, sp := range got[battleproto.CmdSync] {
		data, _ := sp.Args.Assoc["data"].([]byte)
		if len(data) < 18 {
			continue
		}
		if n := binary.LittleEndian.Uint16(data[4:6]); n != 1 {
			continue
		}
		// The add entry must be an object add (high bit), not a removal.
		entry := binary.LittleEndian.Uint32(data[6:10])
		if entry&syncAddMask == 0 {
			continue
		}
		mask := binary.LittleEndian.Uint64(data[10:18])
		if mask&syncTeam != 0 && mask&syncPosition != 0 && mask&syncMaxHealth != 0 {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("no mob registration sync with TEAM+POSITION+MAX_HEALTH found")
	}
}

// TestHuntAutoAttackKillsMob drives DO_ACTION auto-attack against a mob until
// it dies: ACTION push, RECEIVE_HIT + HEALTH syncs, ON_KILL, ACTION_DONE,
// EXPERIENCE sync, and the delayed tracking removal + DELETE_OBJECT.
func TestHuntAutoAttackKillsMob(t *testing.T) {
	avatar := gamedata.Avatars()[3] // Лирвэйн: fast attack, dmg 50-62
	c, r := enterHunt(t, session.NewStore(), avatar.ID)
	readWorld(t, c, r)

	// First mob object id is 2000 (spawned in map spawn order).
	send(t, c, battleproto.Packet{Cmd: battleproto.CmdDoAction, RequestID: 5, Status: true,
		Args: amf.NewArray().
			Set("id", int32(1042)). // avatar obj (1000+42)
			Set("action", attackProtoID(avatar)).
			Set("target", int32(2000)).
			Set("targetPos", amf.NewArray().Set("x", 0.0).Set("y", 0.0))})

	var sawAction, sawHit, sawKill, sawDone, sawDelete bool
	deadline := time.After(15 * time.Second)
	done := make(chan struct{})
	go func() {
		for {
			p, err := r.Read()
			if err != nil {
				close(done)
				return
			}
			switch p.Cmd {
			case battleproto.CmdAction:
				sawAction = true
			case battleproto.CmdReceiveHit:
				sawHit = true
			case battleproto.CmdOnKill:
				sawKill = true
			case battleproto.CmdActionDone:
				sawDone = true
			case battleproto.CmdDeleteObject:
				sawDelete = true
				close(done)
				return
			}
		}
	}()
	select {
	case <-done:
	case <-deadline:
	}
	if !sawAction || !sawHit || !sawKill || !sawDone || !sawDelete {
		t.Fatalf("attack flow incomplete: action=%v hit=%v kill=%v done=%v delete=%v",
			sawAction, sawHit, sawKill, sawDone, sawDelete)
	}
}

// TestHuntSkillCast drives a targeted damage skill (Miriam's "Выстрел бури":
// ACTIVE, target enemy, damage + stun) and checks the effect pipeline: an
// ACTION for the SKILL parent proto, an EFFECT_START (cast + payload fx), and a
// RECEIVE_HIT landing on the mob.
func TestHuntSkillCast(t *testing.T) {
	avatar, ok := gamedata.AvatarByID(25) // Мириам
	if !ok {
		t.Fatal("Miriam (id 25) missing from roster")
	}
	store := session.NewStore()
	store.SetPendingBattle(42, session.PendingBattle{
		MapID: 40, AvatarID: avatar.ID, Passwd: "secret", Scene: "map_4_0",
	})
	// Capture the server-side conn so we can learn skill 1 (all skills now start
	// UNLEARNED at rank 0; a rank-0 skill is correctly uncastable).
	scCh := make(chan *conn, 1)
	testHookNewConn = func(sc *conn) {
		select {
		case scCh <- sc:
		default:
		}
	}
	defer func() { testHookNewConn = nil }()

	c, r := dialTestServerWithStore(t, store)
	send(t, c, battleproto.Packet{Cmd: battleproto.CmdConnect, RequestID: 1, Status: true,
		Args: amf.NewArray().Set("clientId", int32(42)).Set("pass", "secret")})
	if _, err := r.Read(); err != nil {
		t.Fatalf("connect reply: %v", err)
	}
	send(t, c, battleproto.Packet{Cmd: battleproto.CmdReady, RequestID: 2, Status: true, Args: amf.NewArray()})
	if _, err := r.Read(); err != nil {
		t.Fatal(err)
	}
	readWorld(t, c, r)

	sc := <-scCh
	sc.mvMu.Lock()
	sc.huntState.skillLevel[0] = 1
	sc.mvMu.Unlock()

	// Cast skill 1 on the first mob (id 2000). The dungeon anchor now sits ~27m out
	// (past the doubled safe zone), beyond skill range, so this exercises the
	// approach-then-cast path: the avatar chases into range, then the effect fires.
	send(t, c, battleproto.Packet{Cmd: battleproto.CmdDoAction, RequestID: 5, Status: true,
		Args: amf.NewArray().
			Set("id", int32(1042)).
			Set("action", skillProtoID(avatar, 1)).
			Set("target", int32(2000)).
			Set("targetPos", amf.NewArray().Set("x", 0.0).Set("y", 0.0))})

	var sawAction, sawEffect, sawHit bool
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		_ = c.SetReadDeadline(time.Now().Add(500 * time.Millisecond))
		p, err := r.Read()
		if err != nil {
			continue
		}
		switch p.Cmd {
		case battleproto.CmdAction:
			if act, _ := p.Args.GetInt("action"); act == skillProtoID(avatar, 1) {
				sawAction = true
			}
		case battleproto.CmdEffectStart:
			sawEffect = true
		case battleproto.CmdReceiveHit:
			if obj, _ := p.Args.GetInt("object"); obj == 2000 {
				sawHit = true
			}
		}
		if sawAction && sawEffect && sawHit {
			break
		}
	}
	if !sawAction || !sawEffect || !sawHit {
		t.Fatalf("skill cast incomplete: action=%v effect=%v hit=%v", sawAction, sawEffect, sawHit)
	}
}

// TestDoTKillDoesNotPanic reproduces the regression where a DoT tick that kills
// a mob carrying 2+ DoT stacks panicked: hitMobLocked resets the mob's status
// (clearing st.dots) mid-loop, so indexing the live slice went out of range.
func TestDoTKillDoesNotPanic(t *testing.T) {
	s := New(session.NewStore())
	srv, cli := net.Pipe()
	defer srv.Close()
	defer cli.Close()
	// Drain the server->client writes so pushes never block.
	go func() {
		buf := make([]byte, 4096)
		for {
			if _, err := cli.Read(buf); err != nil {
				return
			}
		}
	}()

	c := &conn{Conn: srv}
	c.objID = 1000
	now := float64(s.battleTime())
	hs := &huntState{
		av:      gamedata.Avatars()[0],
		mobs:    map[int32]*mobState{},
		summons: map[int32]*summonState{},
		hp:      500, mana: 100,
	}
	hs.tr.add(c.objID)
	mob := &mobState{id: 2000, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 5, aggro: true}
	hs.tr.add(mob.id)
	hs.mobs[mob.id] = mob
	// Two DoT stacks, both due to tick now, each enough to kill a 5-hp mob.
	for i := 0; i < 2; i++ {
		mob.st.dots = append(mob.st.dots, overTime{
			perSec: 10, until: now + 10, nextTick: now - 1, srcObj: c.objID})
	}
	c.huntState = hs
	defer func() {
		c.mvMu.Lock()
		hs.closed = true // neutralise the delayed-removal AfterFunc
		c.mvMu.Unlock()
	}()

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	// Must not panic even though the first DoT tick kills the mob mid-loop.
	s.tickMobsLocked(c, now)
	if !mob.dead {
		t.Errorf("mob should be dead after lethal DoT ticks, hp=%g", mob.hp)
	}
}

// TestSelfCastProtoIsNoneTarget locks the fix for the client target-selection
// bug: a pure self-cast skill's ACTIVE child proto must expose an EMPTY target
// mask (client IsNoneTarget -> instant cast). A "SELF" flag (0x200) makes the
// client demand a unit target instead. Every SELF-target skill in the roster
// must emit target="" and a zeroed aoeRadius/aoeWidth.
func TestSelfCastProtoIsNoneTarget(t *testing.T) {
	avatars := gamedata.Avatars()
	checked := 0
	for _, a := range avatars {
		kit := gamedata.SkillsFor(a)
		for _, sk := range kit.Skills {
			if sk.Type == "PASSIVE" || sk.Type == "TOGGLE" {
				continue
			}
			// Resolve the effective target the same way activeChildDesc does.
			target := sk.Target
			if target == "" {
				target = "SELF"
			}
			if target != "SELF" {
				continue // POINT / ENEMY skills legitimately keep their flag
			}
			checked++
			desc := activeChildDesc(a, sk)
			if strings.Contains(desc, `<enum name="target" value="SELF"`) {
				t.Errorf("%s slot %d: self-cast proto still emits target=SELF (client will demand a target)", a.Prefab, sk.Slot)
			}
			if !strings.Contains(desc, `<enum name="target" value=""`) {
				t.Errorf("%s slot %d: self-cast proto must emit empty target mask, got: %s", a.Prefab, sk.Slot, desc)
			}
			if !strings.Contains(desc, `name="aoeRadius" value="0"`) || !strings.Contains(desc, `name="aoeWidth" value="0"`) {
				t.Errorf("%s slot %d: self-cast proto must zero aoeRadius/aoeWidth", a.Prefab, sk.Slot)
			}
		}
	}
	if checked == 0 {
		t.Fatal("no self-cast skills found to check -- test is not exercising anything")
	}
	t.Logf("verified %d self-cast skill protos", checked)
}

// TestPointCastProtoTargetingIsNone locks the fix for Elgorm's «Оскверненная
// почва» (and every radius POINT skill): a ground-target AoE must expose
// targeting=NONE (mask 0) so PlayerControl shows the ground cursor (mSkillZoneAoe,
// placed where clicked). A non-zero targeting (the old SELF=1 default) drops the
// client into the avatar-attached line zone, making the AoE follow the caster.
// Line/swath skills (AoEWidth>0) are exempt -- they carry an explicit Targeting
// and the client re-adds the line-zone bits anyway.
func TestPointCastProtoTargetingIsNone(t *testing.T) {
	checked := 0
	for _, a := range gamedata.Avatars() {
		for _, sk := range gamedata.SkillsFor(a).Skills {
			if sk.Target != "POINT" || sk.AoEWidth > 0 || sk.Targeting != "" {
				continue
			}
			checked++
			desc := activeChildDesc(a, sk)
			if strings.Contains(desc, `<enum name="targeting" value="SELF"`) {
				t.Errorf("%s slot %d: point AoE still emits targeting=SELF (AoE will follow the caster)", a.Prefab, sk.Slot)
			}
			if !strings.Contains(desc, `<enum name="targeting" value="NONE"`) {
				t.Errorf("%s slot %d: point AoE must emit targeting=NONE, got: %s", a.Prefab, sk.Slot, desc)
			}
		}
	}
	if checked == 0 {
		t.Fatal("no radius POINT skills found to check -- test is not exercising anything")
	}
	t.Logf("verified %d point-AoE skill protos", checked)
}

// TestSelfCastAoEHitsNearbyMobs locks the fix where a self-cast skill whose
// damage op carries no explicit radius (e.g. Velial's self-AoE lifesteal) fell
// back to an empty target set and hit nothing. It must now strike mobs within
// the skill's authored AoE radius and heal the caster.
func TestSelfCastAoEHitsNearbyMobs(t *testing.T) {
	s := New(session.NewStore())
	srv, cli := net.Pipe()
	defer srv.Close()
	defer cli.Close()
	go func() {
		buf := make([]byte, 4096)
		for {
			if _, err := cli.Read(buf); err != nil {
				return
			}
		}
	}()

	velial, ok := gamedata.AvatarByID(13)
	if !ok {
		t.Fatal("Velial (id 13) missing")
	}
	c := &conn{Conn: srv}
	c.objID = 1000
	now := float64(s.battleTime())
	hs := &huntState{
		av:      velial,
		kit:     gamedata.SkillsFor(velial),
		mobs:    map[int32]*mobState{},
		summons: map[int32]*summonState{},
		hp:      100, mana: 200,
	}
	hs.tr.add(c.objID)
	mob := &mobState{id: 2000, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 140}
	mob.x, mob.y = 2, 0 // within any sane self-AoE radius
	hs.tr.add(mob.id)
	hs.mobs[mob.id] = mob
	c.huntState = hs
	defer func() { c.mvMu.Lock(); hs.closed = true; c.mvMu.Unlock() }()

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	// Self-cast (target nil) lifesteal with NO op radius: must still hit the mob.
	ctx := opCtx{slot: 1, level: 1}
	op := hs.kit.Skills[0].Ops[0] // Velial S1 lifesteal_hit
	s.applyOpsLocked(c, []gamedata.Op{op}, ctx, now)

	if mob.hp >= 140 {
		t.Errorf("nearby mob took no damage from self-cast AoE (hp=%g)", mob.hp)
	}
	if hs.hp <= 100 {
		t.Errorf("caster did not lifesteal from self-cast AoE (hp=%g)", hs.hp)
	}
}

// TestSelfCastDoesNotWalkToOrigin locks the fix for "avatar runs to (0,0) and
// the skill never fires": the client always attaches a targetPos ({0,0} for a
// none-target self-cast), so startSkillOrderLocked must recognise a self-cast
// from the skill definition and fire in place instead of approaching the point.
func TestSelfCastDoesNotWalkToOrigin(t *testing.T) {
	s := New(session.NewStore())
	srv, cli := net.Pipe()
	defer srv.Close()
	defer cli.Close()
	go func() {
		buf := make([]byte, 4096)
		for {
			if _, err := cli.Read(buf); err != nil {
				return
			}
		}
	}()

	velial, ok := gamedata.AvatarByID(13)
	if !ok {
		t.Fatal("Velial (id 13) missing")
	}
	c := &conn{Conn: srv}
	c.objID = 1000
	c.x, c.y, c.snapT = 493, 64, s.battleTime() // far from the origin
	hs := &huntState{
		av:      velial,
		kit:     gamedata.SkillsFor(velial),
		mobs:    map[int32]*mobState{},
		summons: map[int32]*summonState{},
		hp:      100, mana: 200,
	}
	for i := range hs.skillLevel {
		hs.skillLevel[i] = 1
	}
	hs.tr.add(c.objID)
	c.huntState = hs
	defer func() { c.mvMu.Lock(); hs.closed = true; c.mvMu.Unlock() }()

	c.mvMu.Lock()
	manaBefore := hs.mana
	// Avatar is walking when the skill is pressed: casting must root it in place.
	c.moveToLocked(s, 520, 90)
	if !c.hasDest {
		c.mvMu.Unlock()
		t.Fatal("test setup: avatar should be moving before the cast")
	}
	// Slot 1 (Velial S1) is a self-cast. Client sends target=-1, targetPos={0,0},
	// hasPos=true -- exactly the packet that used to send the avatar to the origin.
	s.startSkillOrderLocked(c, 1, -1, 0, 0, true)

	if c.hasDest {
		t.Errorf("cast did not stop movement: avatar still heading to (%.0f,%.0f)", c.destX, c.destY)
	}
	if hs.order != nil {
		t.Errorf("self-cast left a pending approach order instead of casting instantly")
	}
	if hs.mana >= manaBefore {
		t.Errorf("self-cast did not consume mana (before=%g after=%g) -- it never fired", manaBefore, hs.mana)
	}
	// The cast roots the avatar: a manual move issued during the lock window must
	// be rejected (the avatar stays put) until the cast animation finishes.
	if float64(s.battleTime()) >= hs.castLockUntil {
		t.Errorf("cast did not arm a movement lock (castLockUntil=%g, now=%g)", hs.castLockUntil, float64(s.battleTime()))
	}
	c.mvMu.Unlock()

	s.handleMove(c, battleproto.Packet{Cmd: battleproto.CmdMovePlayer, RequestID: 9, Status: true,
		Args: amf.NewArray().Set("targetPos", amf.NewArray().Set("x", 500.0).Set("y", 80.0))})
	c.mvMu.Lock()
	moving := c.hasDest
	c.mvMu.Unlock()
	if moving {
		t.Errorf("avatar accepted a move during the cast lock (should be rooted until the animation ends)")
	}
}

// TestHuntWrongPasswordFallsBackToLobby: a stale/wrong password must not enter
// the battle -- the connection behaves as the central-square lobby.
func TestHuntWrongPasswordFallsBackToLobby(t *testing.T) {
	store := session.NewStore()
	store.SetPendingBattle(42, session.PendingBattle{
		MapID: 40, AvatarID: 1, Passwd: "secret", Scene: "map_4_0",
	})
	c, r := dialTestServerWithStore(t, store)

	send(t, c, battleproto.Packet{Cmd: battleproto.CmdConnect, RequestID: 1, Status: true,
		Args: amf.NewArray().Set("clientId", int32(42)).Set("pass", "wrong")})
	if _, err := r.Read(); err != nil {
		t.Fatalf("connect reply: %v", err)
	}
	send(t, c, battleproto.Packet{Cmd: battleproto.CmdReady, RequestID: 2, Status: true, Args: amf.NewArray()})
	if _, err := r.Read(); err != nil { // READY ack
		t.Fatal(err)
	}
	if _, err := r.Read(); err != nil { // GAME_DATA
		t.Fatal(err)
	}
	proto, err := r.Read()
	if err != nil {
		t.Fatal(err)
	}
	if desc, _ := proto.Args.GetString("desc"); !strings.Contains(desc, `PPrefab value="Hero"`) {
		t.Errorf("wrong password should fall back to the lobby Hero prototype, got %q", desc)
	}
}

// TestSyncBlobLayout locks the multi-object blob format: add entries, type
// blocks in ascending bit order, per-type object bitmask, int32 TEAM.
func TestSyncBlobLayout(t *testing.T) {
	// Two tracked objects: values for both in one blob.
	b := newSyncBlob(0).
		addObject(500).
		setFloats(syncHealth, 0, 0.5).
		setFloats(syncHealth, 1, 1.0).
		setInt(syncTeam, 1, -1)
	data := b.build(2)
	// 4 time + 2 count + 4 add + 8 mask + (1 bitmask + 2*4 health) + (1 bitmask + 4 team)
	if len(data) != 4+2+4+8+1+8+1+4 {
		t.Fatalf("blob length = %d", len(data))
	}
	if n := binary.LittleEndian.Uint16(data[4:6]); n != 1 {
		t.Fatalf("newIds count = %d", n)
	}
	if v := binary.LittleEndian.Uint32(data[6:10]); v != uint32(500)|syncAddMask {
		t.Fatalf("add entry = %#x", v)
	}
	mask := binary.LittleEndian.Uint64(data[10:18])
	if mask != syncHealth|syncTeam {
		t.Fatalf("mask = %#x", mask)
	}
	// HEALTH block: bitmask 0b11, then 0.5 (idx0), 1.0 (idx1).
	if data[18] != 0x03 {
		t.Errorf("health bitmask = %#x, want 0x03", data[18])
	}
	if v := f32(data, 19); v != 0.5 {
		t.Errorf("health[0] = %v", v)
	}
	if v := f32(data, 23); v != 1.0 {
		t.Errorf("health[1] = %v", v)
	}
	// TEAM block: bitmask 0b10, then int32 -1.
	if data[27] != 0x02 {
		t.Errorf("team bitmask = %#x, want 0x02", data[27])
	}
	if v := int32(binary.LittleEndian.Uint32(data[28:32])); v != -1 {
		t.Errorf("team = %d, want -1", v)
	}
}

// TestTrackerSwapRemove mirrors the client's swap-with-last removal rule.
func TestTrackerSwapRemove(t *testing.T) {
	tr := &tracker{}
	tr.add(10) // idx 0
	tr.add(20) // idx 1
	tr.add(30) // idx 2
	if idx := tr.remove(20); idx != 1 {
		t.Fatalf("remove(20) idx = %d, want 1", idx)
	}
	// 30 must have taken index 1 (swap-with-last).
	if idx := tr.index(30); idx != 1 {
		t.Errorf("index(30) = %d, want 1", idx)
	}
	if tr.count() != 2 {
		t.Errorf("count = %d, want 2", tr.count())
	}
}

// TestStopPlayerReleaseDoesNotHalt: STOP_PLAYER{stop:false} (sent on every
// right-button release) must not cancel the move order.
func TestStopPlayerReleaseDoesNotHalt(t *testing.T) {
	c, r := dialTestServerWithStore(t, session.NewStore())
	send(t, c, battleproto.Packet{Cmd: battleproto.CmdConnect, RequestID: 1, Status: true,
		Args: amf.NewArray().Set("clientId", int32(5)).Set("pass", "")})
	if _, err := r.Read(); err != nil {
		t.Fatal(err)
	}
	send(t, c, battleproto.Packet{Cmd: battleproto.CmdReady, RequestID: 2, Status: true, Args: amf.NewArray()})
	for i := 0; i < 8; i++ { // READY ack + 7 lobby world packets
		if _, err := r.Read(); err != nil {
			t.Fatalf("draining handshake #%d: %v", i, err)
		}
	}

	// Move away (~1s at speed 4), then "release the button". targetPos is always an
	// absolute world point (the rel flag is client-side only; see handleMove): from the
	// cs_human spawn (-20,-8) walk west to (-24,-8) -- an obstacle-free straight leg in the
	// open plaza, clear of the central hedge maze.
	send(t, c, battleproto.Packet{Cmd: battleproto.CmdMovePlayer, RequestID: 3, Status: true,
		Args: amf.NewArray().Set("targetPos", amf.NewArray().Set("x", -24.0).Set("y", -8.0)).Set("rel", true)})
	if _, err := r.Read(); err != nil { // MOVE ack
		t.Fatal(err)
	}
	if _, err := r.Read(); err != nil { // leg POSITION sync
		t.Fatal(err)
	}
	send(t, c, battleproto.Packet{Cmd: battleproto.CmdStopPlayer, RequestID: 4, Status: true,
		Args: amf.NewArray().Set("stop", false)})
	if _, err := r.Read(); err != nil { // STOP ack
		t.Fatal(err)
	}

	// The next packet must be the ARRIVAL sync (~1s away) with zero velocity
	// at the target -- NOT an immediate halt sync.
	arrivalDeadline := time.Now().Add(3 * time.Second)
	_ = c.SetDeadline(arrivalDeadline)
	start := time.Now()
	p, err := r.Read()
	if err != nil {
		t.Fatalf("expected arrival sync: %v", err)
	}
	if p.Cmd != battleproto.CmdSync {
		t.Fatalf("expected SYNC, got %s", p.Cmd.Name())
	}
	if elapsed := time.Since(start); elapsed < 500*time.Millisecond {
		t.Errorf("sync arrived after %v -- looks like an immediate halt, not the full arrival", elapsed)
	}
	data, _ := p.Args.Assoc["data"].([]byte)
	// POSITION-only single-object blob (no add entries): header is
	// time(4)+count(2)+mask(8)+bitmask(1) = 15, then x,y,velX,velY,snapT.
	x, y := f32(data, 15), f32(data, 19)
	if x < -24.1 || x > -23.9 || y < -8.1 || y > -7.9 {
		t.Errorf("arrival = (%v,%v), want ~(-24,-8) (full path walked west)", x, y)
	}
}
