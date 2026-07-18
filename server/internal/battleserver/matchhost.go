package battleserver

import (
	"log"
	"net"
	"sync"
	"time"

	"tanatserver/internal/mpd"
	"tanatserver/internal/session"
)

// MatchHost hands out a dedicated Battle server -- its own TCP listener and its
// own battle clock -- for every live «Штурм»/«Охота» room.
//
// Why: the client syncs its BattleTimer to the Battle server's uptime via the
// GET_TIME reply (time.Since(Server.start)) and the HUD clock displays that value
// (see handleGetTime). On the single shared Battle server that uptime is the whole
// process's lifetime, so a match joined an hour after boot shows a 1:00:00 clock.
// A server created at match start reads ~0, so every timestamp it emits -- the
// clock, cooldowns, dead-reckoning -- is match-relative and mutually consistent,
// with NO change to the ~150 s.battleTime() call sites (they all read that
// server's own start).
//
// The central square stays on the main shared Battle server (a hub has no match
// clock); only Штурм/Охота launches route here. Servers are created lazily per
// room (get-or-create, so every player of one match shares one world) and reaped
// once a room has been empty for a grace period -- long enough to cover the
// launch->reconnect gap and brief mid-match disconnects.
type MatchHost struct {
	store *session.Store
	hub   *mpd.Hub
	host  string // advertised to the client; the listener binds all interfaces on an ephemeral port

	mu    sync.Mutex
	rooms map[int32]*matchServer

	// Reap cadence, overridable in tests. A room that has held zero members for
	// reapGrace is torn down; liveness is polled every reapPoll.
	reapGrace time.Duration
	reapPoll  time.Duration
}

type matchServer struct {
	srv  *Server
	ln   net.Listener
	port int32
}

// NewMatchHost builds a per-match Battle server host. store and hub are shared
// with the main server -- the pending-battle handoff (TakePendingBattle) and MPD
// push are process-wide -- and host is the address advertised to the client (the
// same host the main Battle server is advertised as).
func NewMatchHost(store *session.Store, hub *mpd.Hub, host string) *MatchHost {
	return &MatchHost{
		store:     store,
		hub:       hub,
		host:      host,
		rooms:     map[int32]*matchServer{},
		reapGrace: 120 * time.Second,
		reapPoll:  10 * time.Second,
	}
}

// Launch returns the advertised host and port of the Battle server dedicated to
// room, creating it on first use. Idempotent per room: every player of one match
// (DOTA shares a formed-match room; Hunt shares room = map id) gets the same
// server, so they all land in the same world. mapID is informational here (the
// room already pins the world) but kept for symmetry with the launch handlers and
// for logging.
func (h *MatchHost) Launch(mapID, room int32) (string, int32, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if ms, ok := h.rooms[room]; ok {
		return h.host, ms.port, nil
	}
	ln, err := net.Listen("tcp", ":0")
	if err != nil {
		return "", 0, err
	}
	srv := New(h.store)
	srv.MPD = h.hub
	port := int32(ln.Addr().(*net.TCPAddr).Port)
	ms := &matchServer{srv: srv, ln: ln, port: port}
	h.rooms[room] = ms
	go func() {
		// Serve returns when the listener is closed by reap -- expected, not an error.
		_ = srv.Serve(ln)
	}()
	go h.reap(room, ms)
	log.Printf("battle: match room=%d map=%d server started on port %d (own clock)", room, mapID, port)
	return h.host, port, nil
}

// reap tears down a room's server once it has held zero members for reapGrace.
// idleSince starts at creation so a room whose client never reconnects is still
// cleaned up; any live member resets it, so a match in progress is never reaped.
func (h *MatchHost) reap(room int32, ms *matchServer) {
	idleSince := time.Now()
	for {
		time.Sleep(h.reapPoll)
		if ms.srv.LiveMemberCount() > 0 {
			idleSince = time.Now()
			continue
		}
		if time.Since(idleSince) < h.reapGrace {
			continue
		}
		h.mu.Lock()
		if cur, ok := h.rooms[room]; ok && cur == ms {
			delete(h.rooms, room)
		}
		h.mu.Unlock()
		_ = ms.ln.Close()
		log.Printf("battle: match room=%d server on port %d reaped (idle >= %s)", room, ms.port, h.reapGrace)
		return
	}
}

// activeRooms reports how many per-match servers are currently live.
func (h *MatchHost) activeRooms() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.rooms)
}

// LiveMemberCount returns the number of players currently in this server's match
// worlds. MatchHost uses it to decide when a per-match server is idle enough to
// reap. Locks s.mu then each inst.mu, never nested (the instance slice is copied
// under s.mu first), preserving the codebase's one-lock-at-a-time order.
func (s *Server) LiveMemberCount() int {
	s.mu.Lock()
	insts := make([]*huntInstance, 0, len(s.insts))
	for _, inst := range s.insts {
		insts = append(insts, inst)
	}
	s.mu.Unlock()
	n := 0
	for _, inst := range insts {
		inst.mu.Lock()
		n += len(inst.members)
		inst.mu.Unlock()
	}
	return n
}
