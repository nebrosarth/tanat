package battleserver

import (
	"encoding/binary"
	"net"
	"testing"
	"time"

	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// startLobbyServer starts a Battle server on an ephemeral port and returns it plus
// the dial address. Used by the central-square multiplayer tests.
func startLobbyServer(t *testing.T, store *session.Store) (*Server, string) {
	t.Helper()
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
	return s, ln.Addr().String()
}

// dialLobby dials a client into a running Battle server as a central-square
// occupant: no pending battle, so CONNECT carries no battle pass and READY builds
// the lobby world state. Returns the live conn + reader (world packets left unread).
func dialLobby(t *testing.T, addr string, userID int32) (net.Conn, *battleproto.Reader) {
	t.Helper()
	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Close() })
	_ = c.SetDeadline(time.Now().Add(20 * time.Second))
	r := battleproto.NewReader(c)
	send(t, c, battleproto.Packet{Cmd: battleproto.CmdConnect, RequestID: 1, Status: true,
		Args: amf.NewArray().Set("clientId", userID).Set("pass", "")})
	if _, err := r.Read(); err != nil {
		t.Fatalf("user %d connect reply: %v", userID, err)
	}
	send(t, c, battleproto.Packet{Cmd: battleproto.CmdReady, RequestID: 2, Status: true, Args: amf.NewArray()})
	if ack, err := r.Read(); err != nil || ack.Cmd != battleproto.CmdReady {
		t.Fatalf("user %d READY ack: %v (err %v)", userID, ack, err)
	}
	return c, r
}

// createObjectProto returns the proto id of the CREATE_OBJECT for objID, or -1.
func createObjectProto(pkts map[battleproto.CmdID][]battleproto.Packet, objID int32) int32 {
	for _, p := range pkts[battleproto.CmdCreateObject] {
		if id, _ := p.Args.GetInt("id"); id == objID {
			proto, _ := p.Args.GetInt("proto")
			return proto
		}
	}
	return -1
}

// hasPlayerReg scans drained packets for a PLAYER_REG of player id on the team.
func hasPlayerReg(pkts map[battleproto.CmdID][]battleproto.Packet, id, team int32) bool {
	for _, p := range pkts[battleproto.CmdPlayerReg] {
		pid, _ := p.Args.GetInt("id")
		tm, _ := p.Args.GetInt("team")
		if pid == id && tm == team {
			return true
		}
	}
	return false
}

// decodeMoveSync decodes a plain POSITION update SYNC blob (no add/remove entries,
// POSITION-only type mask, one set object) into the moved object's tracking index
// and velocity. Returns ok=false for any other blob shape. Assumes a 1-byte object
// bitmask (up to 8 tracked objects), which covers the two-player square tests.
func decodeMoveSync(data []byte) (idx int, vx, vy float32, ok bool) {
	if len(data) < 15 {
		return 0, 0, 0, false
	}
	if binary.LittleEndian.Uint16(data[4:6]) != 0 {
		return 0, 0, 0, false // has add/remove entries -> not a plain move
	}
	if binary.LittleEndian.Uint64(data[6:14]) != 1 {
		return 0, 0, 0, false // POSITION type mask is bit 0
	}
	bm := data[14]
	idx = -1
	for i := 0; i < 8; i++ {
		if bm&(1<<uint(i)) != 0 {
			idx = i
			break
		}
	}
	if idx < 0 || len(data) < 15+20 {
		return 0, 0, 0, false
	}
	return idx, f32(data, 15+8), f32(data, 15+12), true
}

// TestLobbyMultiplayerSharedSquare: two players entering the SAME central square
// land in one shared hub and each renders the other's Hero avatar (PLAYER_REG on
// team 1 + CREATE_OBJECT with the shared Hero prototype).
func TestLobbyMultiplayerSharedSquare(t *testing.T) {
	store := session.NewStore()
	s, addr := startLobbyServer(t, store)

	cA, rA := dialLobby(t, addr, 501)
	worldA := readWorld(t, cA, rA)
	if !hasCreateObject(worldA, 1501) {
		t.Fatal("player A never got its own avatar object")
	}
	if hasCreateObject(worldA, 1502) {
		t.Fatal("player A saw player B before B joined")
	}

	cB, rB := dialLobby(t, addr, 502)
	worldB := readWorld(t, cB, rB)
	if !hasCreateObject(worldB, 1502) {
		t.Fatal("player B never got its own avatar object")
	}
	if !hasCreateObject(worldB, 1501) {
		t.Fatal("player B did not receive player A's avatar (cross-player render)")
	}
	if proto := createObjectProto(worldB, 1501); proto != heroPrototypeID {
		t.Fatalf("player A rendered on B with proto %d, want the shared Hero proto %d", proto, heroPrototypeID)
	}
	if !hasPlayerReg(worldB, 501, 1) {
		t.Fatal("player B did not receive player A's PLAYER_REG on team 1")
	}

	// A must now receive B's avatar (pushed when B joined).
	laterA := readWorld(t, cA, rA)
	if !hasCreateObject(laterA, 1502) {
		t.Fatal("player A did not receive player B's avatar after B joined")
	}

	// Exactly one shared hub for the human-square area, holding both members.
	s.mu.Lock()
	linst := s.linsts[367]
	s.mu.Unlock()
	if linst == nil {
		t.Fatal("no shared lobby hub created for area 367")
	}
	linst.mu.Lock()
	n := len(linst.members)
	_, hasA := linst.members[1501]
	_, hasB := linst.members[1502]
	linst.mu.Unlock()
	if n != 2 || !hasA || !hasB {
		t.Fatalf("lobby hub membership = %d (A=%v B=%v), want both players", n, hasA, hasB)
	}
}

// TestLobbyMultiplayerSeesMovement: when one occupant walks, the other receives a
// POSITION sync for that occupant's avatar at its own tracking index (index 1, since
// self is index 0), with the velocity pointing at the move target.
func TestLobbyMultiplayerSeesMovement(t *testing.T) {
	store := session.NewStore()
	s, addr := startLobbyServer(t, store)
	_ = s

	cA, rA := dialLobby(t, addr, 501)
	readWorld(t, cA, rA)
	cB, rB := dialLobby(t, addr, 502)
	readWorld(t, cB, rB)
	readWorld(t, cA, rA) // drain B's render pushed onto A

	// Keep draining A so its socket buffer never blocks the shared lock.
	go func() {
		for {
			if _, err := rA.Read(); err != nil {
				return
			}
		}
	}()

	// A walks north from the cs_human spawn (-20,-8) to (-20,-4): velocity ~ (0,4).
	send(t, cA, battleproto.Packet{Cmd: battleproto.CmdMovePlayer, RequestID: 3, Status: true,
		Args: amf.NewArray().Set("targetPos", amf.NewArray().Set("x", -20.0).Set("y", -4.0)).Set("rel", false)})

	deadline := time.Now().Add(15 * time.Second)
	found := false
	for time.Now().Before(deadline) {
		_ = cB.SetReadDeadline(time.Now().Add(3 * time.Second))
		p, err := rB.Read()
		if err != nil {
			break
		}
		if p.Cmd != battleproto.CmdSync {
			continue
		}
		data, _ := p.Args.Assoc["data"].([]byte)
		idx, vx, vy, ok := decodeMoveSync(data)
		if ok && idx == 1 && vy > 3.9 && vy < 4.1 && vx > -0.1 && vx < 0.1 {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("player B never received player A's movement POSITION sync at tracking index 1")
	}
}

// TestLobbyMultiplayerLeave: when an occupant disconnects, the remaining occupant
// receives a DELETE_OBJECT for the departed avatar (so it stops rendering it).
func TestLobbyMultiplayerLeave(t *testing.T) {
	store := session.NewStore()
	s, addr := startLobbyServer(t, store)
	_ = s

	cA, rA := dialLobby(t, addr, 501)
	readWorld(t, cA, rA)
	cB, rB := dialLobby(t, addr, 502)
	readWorld(t, cB, rB)
	readWorld(t, cA, rA) // drain B's render pushed onto A

	cA.Close() // A leaves the square

	deadline := time.Now().Add(15 * time.Second)
	sawDelete := false
	for time.Now().Before(deadline) {
		_ = cB.SetReadDeadline(time.Now().Add(3 * time.Second))
		p, err := rB.Read()
		if err != nil {
			break
		}
		if p.Cmd == battleproto.CmdDeleteObject {
			if id, _ := p.Args.GetInt("id"); id == 1501 {
				sawDelete = true
				break
			}
		}
	}
	if !sawDelete {
		t.Fatal("player B never received DELETE_OBJECT for the departed player A")
	}
}

// TestLobbyMultiplayerSeparateAreas: occupants of DIFFERENT squares (human 367 vs
// elf 368) are in separate hubs and never cross-render.
func TestLobbyMultiplayerSeparateAreas(t *testing.T) {
	store := session.NewStore()
	store.SetLobbyArea(602, gamedata.AreaCSElf) // put B in the elf square
	s, addr := startLobbyServer(t, store)

	cA, rA := dialLobby(t, addr, 601) // human square (default)
	readWorld(t, cA, rA)
	cB, rB := dialLobby(t, addr, 602) // elf square
	worldB := readWorld(t, cB, rB)
	if hasCreateObject(worldB, 1601) {
		t.Fatal("elf-square player B saw human-square player A -- squares must not cross-render")
	}
	laterA := readWorld(t, cA, rA)
	if hasCreateObject(laterA, 1602) {
		t.Fatal("human-square player A saw elf-square player B -- squares must not cross-render")
	}

	s.mu.Lock()
	h367 := s.linsts[367]
	h368 := s.linsts[gamedata.AreaCSElf]
	s.mu.Unlock()
	if h367 == nil || h368 == nil || h367 == h368 {
		t.Fatalf("expected two distinct lobby hubs, got 367=%p 368=%p", h367, h368)
	}
}
