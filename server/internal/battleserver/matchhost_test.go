package battleserver

import (
	"testing"
	"time"

	"tanatserver/internal/mpd"
	"tanatserver/internal/session"
)

func newTestMatchHost() *MatchHost {
	store := session.NewStore()
	return NewMatchHost(store, mpd.NewHub(store), "127.0.0.1")
}

// shutdownAll closes every live match listener (test cleanup).
func (h *MatchHost) shutdownAll() {
	h.mu.Lock()
	defer h.mu.Unlock()
	for room, ms := range h.rooms {
		_ = ms.ln.Close()
		delete(h.rooms, room)
	}
}

// TestMatchServerClockStartsAtMatch is the «Штурм»/«Охота» timer fix: the shared
// Battle server's clock is its whole process uptime (an hour here), but a server
// minted for a fresh match reads ~0. The client HUD clock is that value (synced to
// GET_TIME = time.Since(Server.start)), so a per-match server makes it count from
// match start.
func TestMatchServerClockStartsAtMatch(t *testing.T) {
	shared := New(session.NewStore())
	shared.start = time.Now().Add(-time.Hour) // booted long ago
	if bt := shared.battleTime(); bt < 3000 {
		t.Fatalf("precondition: shared uptime clock too small: %g", bt)
	}

	h := newTestMatchHost()
	h.reapGrace = time.Hour // keep the reaper out of this test
	defer h.shutdownAll()

	_, port, err := h.Launch(101, 500)
	if err != nil {
		t.Fatalf("launch: %v", err)
	}
	h.mu.Lock()
	ms := h.rooms[500]
	h.mu.Unlock()
	if ms == nil || ms.port != port {
		t.Fatalf("room 500 not registered on port %d (ms=%v)", port, ms)
	}
	if bt := ms.srv.battleTime(); bt > 5 {
		t.Fatalf("match server clock should start near 0, got %g", bt)
	}
}

// TestMatchHostIdempotentPerRoom: every player of one match (same room) is routed
// to the same server, while a different room (a different match, or a different
// Hunt map) gets its own.
func TestMatchHostIdempotentPerRoom(t *testing.T) {
	h := newTestMatchHost()
	h.reapGrace = time.Hour
	defer h.shutdownAll()

	_, p1, err := h.Launch(101, 500)
	if err != nil {
		t.Fatalf("launch A: %v", err)
	}
	_, p1b, err := h.Launch(101, 500) // same room -> same server
	if err != nil {
		t.Fatalf("launch A': %v", err)
	}
	if p1 != p1b {
		t.Fatalf("same room got different ports: %d vs %d", p1, p1b)
	}
	_, p2, err := h.Launch(101, 501) // different room -> different server
	if err != nil {
		t.Fatalf("launch B: %v", err)
	}
	if p2 == p1 {
		t.Fatalf("different rooms share a port: %d", p2)
	}
	if got := h.activeRooms(); got != 2 {
		t.Fatalf("activeRooms = %d, want 2", got)
	}
}

// TestMatchHostReapsIdleRoom: a room whose client never reconnects (zero members)
// is torn down once it has been idle past the grace period, so ports/goroutines
// don't leak on a deserted launch.
func TestMatchHostReapsIdleRoom(t *testing.T) {
	h := newTestMatchHost()
	h.reapGrace = 30 * time.Millisecond
	h.reapPoll = 5 * time.Millisecond
	defer h.shutdownAll()

	if _, _, err := h.Launch(101, 700); err != nil {
		t.Fatalf("launch: %v", err)
	}
	if got := h.activeRooms(); got != 1 {
		t.Fatalf("just launched: activeRooms = %d, want 1", got)
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if h.activeRooms() == 0 {
			return // reaped, listener closed
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("idle room not reaped within 2s: activeRooms = %d", h.activeRooms())
}

// TestLiveMemberCount pins the liveness signal the reaper reads: the sum of
// members across the server's match worlds.
func TestLiveMemberCount(t *testing.T) {
	s := New(session.NewStore())
	if got := s.LiveMemberCount(); got != 0 {
		t.Fatalf("fresh server LiveMemberCount = %d, want 0", got)
	}
	inst := &huntInstance{id: 5, members: map[int32]*conn{}}
	s.insts[5] = inst
	inst.members[1000] = &conn{}
	inst.members[1001] = &conn{}
	if got := s.LiveMemberCount(); got != 2 {
		t.Fatalf("LiveMemberCount = %d, want 2", got)
	}
	delete(inst.members, 1000)
	if got := s.LiveMemberCount(); got != 1 {
		t.Fatalf("after one leaves, LiveMemberCount = %d, want 1", got)
	}
}
