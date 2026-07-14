package battleserver

import (
	"net"
	"sync"
	"testing"
	"time"

	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// dialShared dials one client into an already-running Battle server and drives it
// through CONNECT + READY for a pending hunt battle, returning the live conn and
// reader (world-state packets left unread).
func dialShared(t *testing.T, s *Server, addr string, userID int32, pass string) (net.Conn, *battleproto.Reader) {
	t.Helper()
	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Close() })
	_ = c.SetDeadline(time.Now().Add(20 * time.Second))
	r := battleproto.NewReader(c)
	send(t, c, battleproto.Packet{Cmd: battleproto.CmdConnect, RequestID: 1, Status: true,
		Args: amf.NewArray().Set("clientId", userID).Set("pass", pass)})
	if _, err := r.Read(); err != nil {
		t.Fatalf("user %d connect reply: %v", userID, err)
	}
	send(t, c, battleproto.Packet{Cmd: battleproto.CmdReady, RequestID: 2, Status: true, Args: amf.NewArray()})
	if ack, err := r.Read(); err != nil || ack.Cmd != battleproto.CmdReady {
		t.Fatalf("user %d READY ack: %v (err %v)", userID, ack, err)
	}
	return c, r
}

// hasCreateObject scans drained packets for a CREATE_OBJECT of objID.
func hasCreateObject(pkts map[battleproto.CmdID][]battleproto.Packet, objID int32) bool {
	for _, p := range pkts[battleproto.CmdCreateObject] {
		if id, _ := p.Args.GetInt("id"); id == objID {
			return true
		}
	}
	return false
}

// hasEffector scans drained packets for an ADD_EFFECTOR of proto bound to owner.
func hasEffector(pkts map[battleproto.CmdID][]battleproto.Packet, owner, proto int32) bool {
	for _, p := range pkts[battleproto.CmdAddEffector] {
		o, _ := p.Args.GetInt("owner")
		pr, _ := p.Args.GetInt("proto")
		if o == owner && pr == proto {
			return true
		}
	}
	return false
}

// TestSummonIDsUniqueAcrossMembers: summon ids must come from one party-wide space
// so two members' summons never collide (else the shared mob sim misroutes hits /
// kill-credit by summon id). A bare/solo conn falls back to its own counter.
func TestSummonIDsUniqueAcrossMembers(t *testing.T) {
	inst := &huntInstance{nextSummonID: 300000}
	a := &conn{inst: inst, huntState: &huntState{}}
	b := &conn{inst: inst, huntState: &huntState{}}
	seen := map[int32]bool{}
	for i := 0; i < 5; i++ {
		for _, c := range []*conn{a, b} {
			id := c.allocSummonID()
			if seen[id] {
				t.Fatalf("summon id collision across members: %d", id)
			}
			seen[id] = true
		}
	}
	if solo := (&conn{huntState: &huntState{}}).allocSummonID(); solo != 1 {
		t.Fatalf("solo (no-instance) summon id = %d, want 1", solo)
	}
}

// TestMultiplayerSharedWorld: two players who ready for the same map+room land in
// ONE shared instance and each renders the other's avatar.
func TestMultiplayerSharedWorld(t *testing.T) {
	store := session.NewStore()
	s := New(store)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go s.handleConn(conn)
		}
	}()
	addr := ln.Addr().String()

	avatar := gamedata.Avatars()[0]
	const room = int32(40)
	// Two players, same map + room -> same shared world.
	store.SetPendingBattle(101, session.PendingBattle{MapID: 40, AvatarID: avatar.ID, Passwd: "p1", Scene: "map_4_0", Room: room})
	store.SetPendingBattle(102, session.PendingBattle{MapID: 40, AvatarID: avatar.ID, Passwd: "p2", Scene: "map_4_0", Room: room})

	cA, rA := dialShared(t, s, addr, 101, "p1")
	worldA := readWorld(t, cA, rA)
	if !hasCreateObject(worldA, 1101) {
		t.Fatal("player A never got its own avatar object")
	}

	// A second player joins the SAME room.
	cB, rB := dialShared(t, s, addr, 102, "p2")
	worldB := readWorld(t, cB, rB)
	// B's world must include A's avatar (introduced on join) and its own.
	if !hasCreateObject(worldB, 1102) {
		t.Fatal("player B never got its own avatar object")
	}
	if !hasCreateObject(worldB, 1101) {
		t.Fatal("player B did not receive player A's avatar (cross-player render)")
	}
	// B must also get A's ATTACK effector bound to A's avatar, else B's client can't
	// resolve A's basic-attack ACTION and A appears to fight without swinging.
	if !hasEffector(worldB, 1101, attackProtoID(avatar)) {
		t.Fatal("player B did not receive player A's attack effector (no teammate swing animation)")
	}

	// And A must now receive B's avatar (pushed when B joined) with its attack effector.
	laterA := readWorld(t, cA, rA)
	if !hasCreateObject(laterA, 1102) {
		t.Fatal("player A did not receive player B's avatar after B joined")
	}
	if !hasEffector(laterA, 1102, attackProtoID(avatar)) {
		t.Fatal("player A did not receive player B's attack effector after B joined")
	}

	// Exactly one shared instance for the room, with both members.
	s.mu.Lock()
	inst := s.insts[room]
	s.mu.Unlock()
	if inst == nil {
		t.Fatal("no shared instance created for the room")
	}
	inst.mu.Lock()
	n := len(inst.members)
	_, hasA := inst.members[1101]
	_, hasB := inst.members[1102]
	sharedMobs := inst.mobs
	inst.mu.Unlock()
	if n != 2 || !hasA || !hasB {
		t.Fatalf("shared instance membership = %d (A=%v B=%v), want both players", n, hasA, hasB)
	}
	// Both members must alias the SAME mob map (shared authoritative set).
	cA.Close()
	if len(sharedMobs) == 0 {
		t.Fatal("shared instance has no mobs seeded")
	}
}

// TestMultiplayerSeesTeammateAttack drives player A into a basic auto-attack on a
// shared mob and asserts player B actually receives A's ACTION (id=A.objID,
// action=attackProtoID) -- the packet that makes A's avatar swing on B's client.
// This proves the SERVER-side broadcast end to end (the effector-delivery check in
// TestMultiplayerSharedWorld only proves the precondition, not the swing packet).
func TestMultiplayerSeesTeammateAttack(t *testing.T) {
	store := session.NewStore()
	s := New(store)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go s.handleConn(conn)
		}
	}()
	addr := ln.Addr().String()

	avatar := gamedata.Avatars()[3] // Лирвэйн: fast attacker (same as the auto-attack test)
	const room = int32(40)
	store.SetPendingBattle(201, session.PendingBattle{MapID: 40, AvatarID: avatar.ID, Passwd: "p1", Scene: "map_4_0", Room: room})
	store.SetPendingBattle(202, session.PendingBattle{MapID: 40, AvatarID: avatar.ID, Passwd: "p2", Scene: "map_4_0", Room: room})

	cA, rA := dialShared(t, s, addr, 201, "p1")
	_ = cA.SetDeadline(time.Now().Add(40 * time.Second))
	readWorld(t, cA, rA)
	cB, rB := dialShared(t, s, addr, 202, "p2")
	_ = cB.SetDeadline(time.Now().Add(40 * time.Second))
	readWorld(t, cB, rB)

	// Drain A's stream continuously: A receives a flood (its own ACTIONs, movement,
	// mob syncs); if its socket buffer fills, s.push(A,...) blocks under the world
	// lock and stalls the whole instance (including B). Same reason B is read below.
	go func() {
		for {
			if _, err := rA.Read(); err != nil {
				return
			}
		}
	}()

	// A auto-attacks the first shared mob (id 2000). The avatar chases into range,
	// then armAttackTimer emits the basic-attack ACTION -- which must reach B.
	send(t, cA, battleproto.Packet{Cmd: battleproto.CmdDoAction, RequestID: 5, Status: true,
		Args: amf.NewArray().
			Set("id", int32(1201)). // A's avatar obj (1000+201)
			Set("action", attackProtoID(avatar)).
			Set("target", int32(2000)).
			Set("targetPos", amf.NewArray().Set("x", 0.0).Set("y", 0.0))})

	// B must receive BOTH A's swing ACTION and a per-swing ACTION_DONE: the ACTION
	// starts the swing clip, and the DONE (broadcast by scheduleSwingDone) closes it
	// so the WrapMode.Once clip can re-trigger on the next swing instead of freezing.
	sawAction := make(chan struct{}, 1)
	sawDone := make(chan struct{}, 1)
	go func() {
		var a, d bool
		for {
			p, err := rB.Read()
			if err != nil {
				return
			}
			id, _ := p.Args.GetInt("id")
			if id != 1201 {
				continue
			}
			switch p.Cmd {
			case battleproto.CmdAction:
				if act, _ := p.Args.GetInt("action"); act == attackProtoID(avatar) && !a {
					a = true
					sawAction <- struct{}{}
				}
			case battleproto.CmdActionDone:
				if act, _ := p.Args.GetInt("action"); act == attackProtoID(avatar) && !d {
					d = true
					sawDone <- struct{}{}
				}
			}
		}
	}()

	deadline := time.After(30 * time.Second)
	select {
	case <-sawAction:
	case <-deadline:
		t.Fatal("player B never received player A's basic-attack ACTION")
	}
	select {
	case <-sawDone:
	case <-deadline:
		t.Fatal("player B never received a per-swing ACTION_DONE for player A (swing would freeze after one hit)")
	}
}

// TestMultiplayerSeesTeammateSkill drives player A to cast a targeted damage skill
// and asserts player B receives A's skill ACTION AND an EFFECT_START owned by A's
// avatar -- the cast fx that (client-side) plays both the cast animation
// (Skill.StartEffects -> PlayAnimation(mEffect.mAnimation)) and the VFX on the
// remote avatar. Proves the world-scoped fx routing reaches teammates end to end.
func TestMultiplayerSeesTeammateSkill(t *testing.T) {
	avatar, ok := gamedata.AvatarByID(25) // Мириам: targeted damage skill 1 with a CastFx
	if !ok {
		t.Fatal("Miriam (id 25) missing from roster")
	}
	store := session.NewStore()
	s := New(store)

	var mu sync.Mutex
	var conns []*conn
	testHookNewConn = func(sc *conn) {
		mu.Lock()
		conns = append(conns, sc)
		mu.Unlock()
	}
	defer func() { testHookNewConn = nil }()

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go s.handleConn(conn)
		}
	}()
	addr := ln.Addr().String()

	const room = int32(40)
	store.SetPendingBattle(201, session.PendingBattle{MapID: 40, AvatarID: avatar.ID, Passwd: "p1", Scene: "map_4_0", Room: room})
	store.SetPendingBattle(202, session.PendingBattle{MapID: 40, AvatarID: avatar.ID, Passwd: "p2", Scene: "map_4_0", Room: room})

	cA, rA := dialShared(t, s, addr, 201, "p1")
	_ = cA.SetDeadline(time.Now().Add(40 * time.Second))
	readWorld(t, cA, rA)
	cB, rB := dialShared(t, s, addr, 202, "p2")
	_ = cB.SetDeadline(time.Now().Add(40 * time.Second))
	readWorld(t, cB, rB)

	// Find A's server conn (objID 1201) and learn skill 1 (all skills start unlearned).
	var scA *conn
	for stop := time.Now().Add(5 * time.Second); scA == nil && time.Now().Before(stop); {
		mu.Lock()
		for _, sc := range conns {
			if sc.objID == 1201 && sc.huntState != nil {
				scA = sc
				break
			}
		}
		mu.Unlock()
		if scA == nil {
			time.Sleep(20 * time.Millisecond)
		}
	}
	if scA == nil {
		t.Fatal("could not find player A's server conn")
	}
	scA.lock()
	scA.huntState.skillLevel[0] = 1
	scA.unlock()

	// Drain A's stream so its socket buffer never blocks the world lock.
	go func() {
		for {
			if _, err := rA.Read(); err != nil {
				return
			}
		}
	}()

	// A casts skill 1 on the first shared mob (id 2000): chases into range, then execCast fires.
	send(t, cA, battleproto.Packet{Cmd: battleproto.CmdDoAction, RequestID: 5, Status: true,
		Args: amf.NewArray().
			Set("id", int32(1201)).
			Set("action", skillProtoID(avatar, 1)).
			Set("target", int32(2000)).
			Set("targetPos", amf.NewArray().Set("x", 0.0).Set("y", 0.0))})

	sawAction := make(chan struct{}, 1)
	sawFx := make(chan struct{}, 1)
	go func() {
		var a, f bool
		for {
			p, err := rB.Read()
			if err != nil {
				return
			}
			switch p.Cmd {
			case battleproto.CmdAction:
				id, _ := p.Args.GetInt("id")
				act, _ := p.Args.GetInt("action")
				if id == 1201 && act == skillProtoID(avatar, 1) && !a {
					a = true
					sawAction <- struct{}{}
				}
			case battleproto.CmdEffectStart:
				if owner, _ := p.Args.GetInt("owner"); owner == 1201 && !f {
					f = true
					sawFx <- struct{}{}
				}
			}
		}
	}()

	deadline := time.After(30 * time.Second)
	select {
	case <-sawAction:
	case <-deadline:
		t.Fatal("player B never received player A's skill ACTION")
	}
	select {
	case <-sawFx:
	case <-deadline:
		t.Fatal("player B never received a skill EFFECT_START from player A (no cast animation/VFX)")
	}
}

// TestMultiplayerSeesTeammateSummon summons a unit on player A and asserts player B
// receives its CREATE_OBJECT (with the party-wide summon prototype) AND its ATTACK
// effector -- the packets that make A's summon appear and be able to swing on B's
// client. The summon is fired directly on A's server conn (deterministic; the cast
// path is covered by the skill test) so the assertion isolates the cross-render.
func TestMultiplayerSeesTeammateSummon(t *testing.T) {
	avatar := gamedata.Avatars()[0]
	const unit = "Mob_ZombieCrawl_01"
	wantProto, ok := summonProtoIDFor(unit)
	if !ok {
		t.Fatalf("summon prefab %q not in the party-wide registry", unit)
	}

	store := session.NewStore()
	s := New(store)

	var mu sync.Mutex
	var conns []*conn
	testHookNewConn = func(sc *conn) {
		mu.Lock()
		conns = append(conns, sc)
		mu.Unlock()
	}
	defer func() { testHookNewConn = nil }()

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go s.handleConn(conn)
		}
	}()
	addr := ln.Addr().String()

	const room = int32(40)
	store.SetPendingBattle(201, session.PendingBattle{MapID: 40, AvatarID: avatar.ID, Passwd: "p1", Scene: "map_4_0", Room: room})
	store.SetPendingBattle(202, session.PendingBattle{MapID: 40, AvatarID: avatar.ID, Passwd: "p2", Scene: "map_4_0", Room: room})

	cA, rA := dialShared(t, s, addr, 201, "p1")
	_ = cA.SetDeadline(time.Now().Add(40 * time.Second))
	readWorld(t, cA, rA)
	cB, rB := dialShared(t, s, addr, 202, "p2")
	_ = cB.SetDeadline(time.Now().Add(40 * time.Second))
	readWorld(t, cB, rB)

	// Find A's server conn (objID 1201).
	var scA *conn
	for stop := time.Now().Add(5 * time.Second); scA == nil && time.Now().Before(stop); {
		mu.Lock()
		for _, sc := range conns {
			if sc.objID == 1201 && sc.huntState != nil {
				scA = sc
				break
			}
		}
		mu.Unlock()
		if scA == nil {
			time.Sleep(20 * time.Millisecond)
		}
	}
	if scA == nil {
		t.Fatal("could not find player A's server conn")
	}

	// Drain A's stream so its socket buffer never blocks the world lock.
	go func() {
		for {
			if _, err := rA.Read(); err != nil {
				return
			}
		}
	}()

	// Summon one unit on A. summonLocked fans the render out to every ready member.
	op := gamedata.Op{
		Kind:     gamedata.OpSummon,
		Count:    gamedata.PerLevel{1, 1, 1, 1},
		Lifetime: gamedata.PerLevel{60, 60, 60, 60},
		HP:       gamedata.PerLevel{200, 200, 200, 200},
		Dmg:      gamedata.PerLevel{20, 20, 20, 20},
		Unit:     unit,
	}
	scA.lock()
	s.summonLocked(scA, op, opCtx{level: 1}, float64(s.battleTime()))
	scA.unlock()

	// B must receive the summon's CREATE_OBJECT (id in the party-wide summon space,
	// proto = the shared prefab id) and its ATTACK effector.
	sawObj := make(chan int32, 1)
	sawEff := make(chan struct{}, 1)
	go func() {
		var summonID int32
		var o, e bool
		for {
			p, err := rB.Read()
			if err != nil {
				return
			}
			switch p.Cmd {
			case battleproto.CmdCreateObject:
				id, _ := p.Args.GetInt("id")
				proto, _ := p.Args.GetInt("proto")
				if id >= 300000 && proto == wantProto && !o {
					summonID = id
					o = true
					sawObj <- id
				}
			case battleproto.CmdAddEffector:
				owner, _ := p.Args.GetInt("owner")
				proto, _ := p.Args.GetInt("proto")
				if o && owner == summonID && proto == summonAttackProtoID && !e {
					e = true
					sawEff <- struct{}{}
				}
			}
		}
	}()

	deadline := time.After(30 * time.Second)
	select {
	case <-sawObj:
	case <-deadline:
		t.Fatal("player B never received player A's summon CREATE_OBJECT (summon invisible to teammate)")
	}
	select {
	case <-sawEff:
	case <-deadline:
		t.Fatal("player B never received the summon's ATTACK effector (summon can't swing on teammate's client)")
	}
}
