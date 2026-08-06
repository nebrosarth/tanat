package battleserver

import (
	"net"
	"testing"
	"time"

	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// TestBotFilledMatchRunsForReal is a real-clock integration smoke test: a real TCP
// CONNECT/READY launch into a fresh «Штурм» room with TANAT_DOTA_BOTS set, then the actual
// runInstanceTicker (real 200ms ticks, real time.AfterFunc swing/arrival timers -- nothing
// stubbed) is left to run for a few seconds. It only asserts loose, timing-tolerant
// conditions (bots exist, the world is still alive, at least one bot has left its exact
// spawn point) -- the precise decision logic itself is covered by bot_test.go's
// synchronous, deterministic unit tests; this exists to catch what those can't: a deadlock,
// panic, or two goroutines fighting over inst.mu under real concurrent timers.
func TestBotFilledMatchRunsForReal(t *testing.T) {
	if testing.Short() {
		t.Skip("real-clock smoke test skipped in -short")
	}
	t.Setenv("TANAT_DOTA_BOTS", "6")

	store := session.NewStore()
	s := New(store)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()
	go s.Serve(ln)

	const userID int32 = 1
	dm := gamedata.DotaMaps()[0]
	av, _ := gamedata.AvatarByID(botRosterAvatarIDs[0])
	store.SetPendingBattle(userID, session.PendingBattle{
		MapID: dm.ID, AvatarID: av.ID, Passwd: "pw", Scene: dm.Scene, Room: dm.ID + 777,
	})

	cl, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer cl.Close()
	_ = cl.SetDeadline(time.Now().Add(15 * time.Second))
	r := battleproto.NewReader(cl)

	if err := battleproto.Write(cl, battleproto.Packet{
		Cmd: battleproto.CmdConnect, RequestID: 1, Status: true,
		Args: amf.NewArray().Set("clientId", userID).Set("pass", "pw"),
	}); err != nil {
		t.Fatalf("send CONNECT: %v", err)
	}
	if _, err := r.Read(); err != nil {
		t.Fatalf("read CONNECT reply: %v", err)
	}
	if err := battleproto.Write(cl, battleproto.Packet{
		Cmd: battleproto.CmdReady, RequestID: 2, Status: true, Args: amf.NewArray(),
	}); err != nil {
		t.Fatalf("send READY: %v", err)
	}

	// Drain every push for the rest of the test -- a real client that stops reading
	// would eventually block the server's writes (and so the whole tick goroutine)
	// once the socket buffer fills, exactly the failure mode this test exists to catch.
	drainDone := make(chan struct{})
	go func() {
		defer close(drainDone)
		for {
			if _, err := r.Read(); err != nil {
				return
			}
		}
	}()

	// Find the room this launch actually landed in (self PLAYER_REG doesn't carry it,
	// but the server logged it and we can just poll the registry instead of parsing logs).
	var inst *huntInstance
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		s.mu.Lock()
		for _, i := range s.insts {
			if i.dota != nil {
				inst = i
			}
		}
		s.mu.Unlock()
		if inst != nil {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if inst == nil {
		t.Fatal("no «Штурм» instance appeared after CONNECT/READY")
	}

	// Let the real ticker run: real 200ms ticks, real swing/arrival timers, nothing
	// simulated. Long enough for several think cycles (0.3s) to fire.
	time.Sleep(4 * time.Second)

	inst.mu.Lock()
	nBots := len(inst.bots)
	moved := false
	for id, b := range inst.bots {
		mem := inst.members[id]
		hx, hy := botHomeLocked(mem)
		if mem.x != hx || mem.y != hy {
			moved = true
		}
		if b.lane < 0 || b.lane >= len(inst.dota.m.Lanes) {
			t.Errorf("bot %d has an out-of-range lane %d", id, b.lane)
		}
	}
	ended := inst.dota.ended
	inst.mu.Unlock()

	if nBots == 0 {
		t.Fatal("TANAT_DOTA_BOTS was set but no bots were spawned")
	}
	if ended {
		t.Fatal("the match ended within the smoke window -- unexpected this early")
	}
	if !moved {
		t.Error("no bot moved from its exact spawn point in 4 real seconds of ticking")
	}

	cl.Close()
	<-drainDone
}
