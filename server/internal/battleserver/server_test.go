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
	"tanatserver/internal/session"
)

// dialTestServer starts the Battle server on an ephemeral port, connects, and
// returns a client conn + reader for the handshake tests.
func dialTestServer(t *testing.T) (net.Conn, *battleproto.Reader) {
	t.Helper()
	s := New(session.NewStore())
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
	_ = c.SetDeadline(time.Now().Add(3 * time.Second))
	return c, battleproto.NewReader(c)
}

// TestConnectHandshake drives the exact packet the client's SendConnect emits
// and checks the reply the client parses into ConnectArg
// {clientId -> selfPlayerId, battleId}.
func TestConnectHandshake(t *testing.T) {
	c, r := dialTestServer(t)

	send(t, c, battleproto.Packet{
		Cmd: battleproto.CmdConnect, RequestID: 5, Status: true,
		Args: amf.NewArray().Set("clientId", int32(77)).Set("pass", ""),
	})

	reply, err := r.Read()
	if err != nil {
		t.Fatalf("read CONNECT reply: %v", err)
	}
	if reply.Cmd != battleproto.CmdConnect {
		t.Errorf("reply cmd = %s, want CONNECT", reply.Cmd.Name())
	}
	if reply.RequestID != 5 {
		t.Errorf("reply requestId = %d, want 5 (must echo request for ping/correlation)", reply.RequestID)
	}
	if !reply.Status {
		t.Errorf("reply status = false, want true")
	}
	if v, _ := reply.Args.GetInt("clientId"); v != 77 {
		t.Errorf("reply clientId = %d, want 77 (echoes user id as self player id)", v)
	}
	if _, ok := reply.Args.GetInt("battleId"); !ok {
		t.Errorf("reply missing battleId")
	}
}

// TestGetTimeHandshake checks GET_TIME returns a numeric "time" the client
// feeds to BattleTimer.Sync.
func TestGetTimeHandshake(t *testing.T) {
	c, r := dialTestServer(t)

	send(t, c, battleproto.Packet{Cmd: battleproto.CmdGetTime, RequestID: 2, Status: true, Args: amf.NewArray()})

	reply, err := r.Read()
	if err != nil {
		t.Fatalf("read GET_TIME reply: %v", err)
	}
	if reply.Cmd != battleproto.CmdGetTime || reply.RequestID != 2 {
		t.Errorf("reply = {%s req=%d}, want {GET_TIME req=2}", reply.Cmd.Name(), reply.RequestID)
	}
	if _, ok := reply.Args.GetFloat("time"); !ok {
		t.Errorf("GET_TIME reply missing numeric time")
	}
}

// TestEnterReadyAcked checks the validation-only ENTER/READY commands get a
// success ack (so BattleServerConnection.OnEntered/OnReady fire).
func TestEnterReadyAcked(t *testing.T) {
	c, r := dialTestServer(t)

	for _, cmd := range []battleproto.CmdID{battleproto.CmdEnter, battleproto.CmdReady} {
		send(t, c, battleproto.Packet{Cmd: cmd, RequestID: 11, Status: true, Args: amf.NewArray()})
		reply, err := r.Read()
		if err != nil {
			t.Fatalf("read %s ack: %v", cmd.Name(), err)
		}
		if reply.Cmd != cmd || reply.RequestID != 11 || !reply.Status {
			t.Errorf("%s ack = {%s req=%d status=%v}, want {%s req=11 status=true}",
				cmd.Name(), reply.Cmd.Name(), reply.RequestID, reply.Status, cmd.Name())
		}
	}
}

// TestReadyTriggersWorldState verifies that after the CONNECT/READY handshake
// the server pushes the self-player registration chain in the order the client
// needs (PROTOTYPE_INFO before CREATE_OBJECT, PLAYER_REG before SET_AVATAR),
// with the ids wired together, so Battle.PerformAvatars -> SelfPlayer.Init fires.
func TestReadyTriggersWorldState(t *testing.T) {
	c, r := dialTestServer(t)

	send(t, c, battleproto.Packet{Cmd: battleproto.CmdConnect, RequestID: 1, Status: true,
		Args: amf.NewArray().Set("clientId", int32(77)).Set("pass", "")})
	if _, err := r.Read(); err != nil { // CONNECT reply
		t.Fatalf("connect reply: %v", err)
	}

	send(t, c, battleproto.Packet{Cmd: battleproto.CmdReady, RequestID: 9, Status: true, Args: amf.NewArray()})

	if ack, err := r.Read(); err != nil || ack.Cmd != battleproto.CmdReady {
		t.Fatalf("expected READY ack first, got %v (err %v)", ack.Cmd.Name(), err)
	}

	wantOrder := []battleproto.CmdID{
		battleproto.CmdGameData, battleproto.CmdPrototypeInfo,
		battleproto.CmdPlayerReg, battleproto.CmdCreateObject, battleproto.CmdSetAvatar,
		battleproto.CmdSync, // POSITION (registers + first sync -> ShowGui)
		battleproto.CmdSync, // SPEED (so the run animation plays)
	}
	got := make(map[battleproto.CmdID]battleproto.Packet)
	for i, want := range wantOrder {
		p, err := r.Read()
		if err != nil {
			t.Fatalf("world packet #%d: %v", i, err)
		}
		if p.Cmd != want {
			t.Fatalf("world packet #%d = %s, want %s", i, p.Cmd.Name(), want.Name())
		}
		got[p.Cmd] = p
	}

	if id, _ := got[battleproto.CmdPlayerReg].Args.GetInt("id"); id != 77 {
		t.Errorf("PLAYER_REG id = %d, want self player 77", id)
	}
	objID, _ := got[battleproto.CmdCreateObject].Args.GetInt("id")
	protoID, _ := got[battleproto.CmdCreateObject].Args.GetInt("proto")
	if protoID != heroPrototypeID {
		t.Errorf("CREATE_OBJECT proto = %d, want %d", protoID, heroPrototypeID)
	}
	sa := got[battleproto.CmdSetAvatar].Args
	if pid, _ := sa.GetInt("playerID"); pid != 77 {
		t.Errorf("SET_AVATAR playerID = %d, want 77", pid)
	}
	if aid, _ := sa.GetInt("avatarID"); aid != objID {
		t.Errorf("SET_AVATAR avatarID = %d, want CREATE_OBJECT id %d", aid, objID)
	}
	if desc, _ := got[battleproto.CmdPrototypeInfo].Args.GetString("desc"); !strings.Contains(desc, `PPrefab value="Hero"`) {
		t.Errorf("PROTOTYPE_INFO desc missing Hero prefab: %q", desc)
	}
	// The SYNC "data" must be a byte blob (client parses it as a raw buffer);
	// it is what finally fires SelfPlayer's first-position callback -> ShowGui.
	if data, ok := got[battleproto.CmdSync].Args.Assoc["data"].([]byte); !ok || len(data) == 0 {
		t.Errorf("SYNC data missing or not a byte array: %#v", got[battleproto.CmdSync].Args.Assoc["data"])
	}
}

// TestFirstPositionSyncLayout locks the exact little-endian byte layout of the
// SYNC blob that SelfPlayer waits on (SyncPacket.Parse consumes it exactly).
func TestFirstPositionSyncLayout(t *testing.T) {
	data := positionSync(1001, true, 0, 0, 0, 0, 0)
	// 4 (time) + 2 (count) + 4 (id) + 8 (typemask) + 1 (objbits) + 20 (5 floats)
	if len(data) != 39 {
		t.Fatalf("sync blob length = %d, want 39", len(data))
	}
	if got := binary.LittleEndian.Uint16(data[4:6]); got != 1 {
		t.Errorf("visibility count = %d, want 1", got)
	}
	if got := binary.LittleEndian.Uint32(data[6:10]); got != uint32(1001)|0x80000000 {
		t.Errorf("object id word = %#x, want %#x", got, uint32(1001)|0x80000000)
	}
	if got := binary.LittleEndian.Uint64(data[10:18]); got != 1 {
		t.Errorf("type mask = %#x, want POSITION (1)", got)
	}
	if data[18] != 0x01 {
		t.Errorf("per-object bitmask = %#x, want 0x01", data[18])
	}
}

// TestSpeedSyncLayout locks the SPEED stat sync blob (count 0, one float value).
func TestSpeedSyncLayout(t *testing.T) {
	data := speedSync(1001, 4.0, 0)
	// 4 (time) + 2 (count=0) + 8 (typemask) + 1 (objbits) + 4 (float)
	if len(data) != 19 {
		t.Fatalf("speed sync length = %d, want 19", len(data))
	}
	if got := binary.LittleEndian.Uint16(data[4:6]); got != 0 {
		t.Errorf("visibility count = %d, want 0 (no add)", got)
	}
	if got := binary.LittleEndian.Uint64(data[6:14]); got != 0x400000 {
		t.Errorf("type mask = %#x, want SPEED (0x400000)", got)
	}
	if got := f32(data, 15); got != 4.0 {
		t.Errorf("speed value = %v, want 4.0", got)
	}
}

// TestMovePlayerStreamsPositionSyncs checks that a MOVE_PLAYER to an absolute
// target is acked and produces a POSITION sync whose velocity points at the
// target (this is what actually walks the hero in the square).
func TestMovePlayerStreamsPositionSyncs(t *testing.T) {
	c, r := dialTestServer(t)
	send(t, c, battleproto.Packet{Cmd: battleproto.CmdConnect, RequestID: 1, Status: true,
		Args: amf.NewArray().Set("clientId", int32(5)).Set("pass", "")})
	if _, err := r.Read(); err != nil {
		t.Fatal(err)
	}
	send(t, c, battleproto.Packet{Cmd: battleproto.CmdReady, RequestID: 2, Status: true, Args: amf.NewArray()})
	// Drain READY ack + 7 world-state packets (5 setup + POSITION + SPEED syncs).
	for i := 0; i < 8; i++ {
		if _, err := r.Read(); err != nil {
			t.Fatalf("draining handshake #%d: %v", i, err)
		}
	}

	// The cs_human spawn is (-20,-8), an open plaza point clear of the central hedge maze.
	// Move 4 units due north to (-20,-4): a clear walkable straight shot -> velocity ~ (0,4).
	send(t, c, battleproto.Packet{Cmd: battleproto.CmdMovePlayer, RequestID: 3, Status: true,
		Args: amf.NewArray().Set("targetPos", amf.NewArray().Set("x", -20.0).Set("y", -4.0)).Set("rel", false)})

	ack, err := r.Read()
	if err != nil || ack.Cmd != battleproto.CmdMovePlayer {
		t.Fatalf("expected MOVE_PLAYER ack, got %v (err %v)", ack.Cmd.Name(), err)
	}
	sync, err := r.Read()
	if err != nil || sync.Cmd != battleproto.CmdSync {
		t.Fatalf("expected POSITION sync, got %v (err %v)", sync.Cmd.Name(), err)
	}
	data, _ := sync.Args.Assoc["data"].([]byte)
	// update-sync layout: time(4) count(2)=0 typemask(8) objbits(1) then 5 floats.
	if len(data) != 35 {
		t.Fatalf("update sync length = %d, want 35", len(data))
	}
	velX := f32(data, 15+8)
	velY := f32(data, 15+12)
	if velY < 3.9 || velY > 4.1 || velX < -0.1 || velX > 0.1 {
		t.Errorf("velocity = (%.2f,%.2f), want ~(0,4) toward target", velX, velY)
	}
}

func f32(b []byte, off int) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(b[off : off+4]))
}

func send(t *testing.T, c net.Conn, p battleproto.Packet) {
	t.Helper()
	if err := battleproto.Write(c, p); err != nil {
		t.Fatalf("write %s: %v", p.Cmd.Name(), err)
	}
}
