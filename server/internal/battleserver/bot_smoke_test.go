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

// TestBotFilledMatchRunsForReal uses the real TCP CONNECT/READY path, but
// advances the authoritative Assault clock explicitly. It checks that bots
// join, move, and remain in a live world without relying on wall-clock ticks.
func TestBotFilledMatchRunsForReal(t *testing.T) {
	if testing.Short() {
		t.Skip("real-clock smoke test skipped in -short")
	}
	t.Setenv("TANAT_DOTA_BOTS", "6")

	store := session.NewStore()
	s := New(store)
	clock := newManualBattleClock()
	s.clock = clock
	instanceStarted := make(chan *huntInstance, 1)
	s.instanceTickerStarter = func(*huntInstance) {}
	s.instanceReadyHook = func(inst *huntInstance) { instanceStarted <- inst }
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

	// The hook fires only after the player's world state and bot backfill finish.
	var inst *huntInstance
	select {
	case inst = <-instanceStarted:
	case <-time.After(3 * time.Second):
		t.Fatal("no Assault instance appeared after CONNECT/READY")
		return
	}
	driver := &manualDotaBots{server: s, inst: inst, clock: clock}
	defer driver.close()
	for tick := 0; tick < int((4*time.Second)/AssaultTick); tick++ {
		driver.step()
	}

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
		t.Error("no bot moved from its exact spawn point in 4 simulated seconds of ticking")
	}

	cl.Close()
	<-drainDone
}
