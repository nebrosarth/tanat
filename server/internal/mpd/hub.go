// Package mpd implements the raw-TCP "MPD" push channel the client opens after it
// receives a chat|conf packet on the Ctrl channel. Unlike the request/response Ctrl
// (HTTP) and the Battle (TCP) channels, MPD is one-directional server->client push:
// chat lines, party invites, online/offline notices and live hero updates.
//
// The Hub owns the listener AND a registry of connected users -> their socket, so
// any subsystem can push to a specific user by id. It imports only amf + session,
// so both ctrlserver (chat/party, HTTP-driven) and battleserver (square presence)
// can push without an import cycle.
package mpd

import (
	"sync"

	"tanatserver/internal/amf"
	"tanatserver/internal/session"
)

// Hub is the MPD push registry and TCP server.
type Hub struct {
	Store *session.Store

	// OnConnect / OnDisconnect, if set, fire when a user's MPD socket registers /
	// unregisters (used by the party code to push online/offline to co-members).
	// Called without the Hub lock held.
	OnConnect    func(userID int32)
	OnDisconnect func(userID int32)

	mu    sync.Mutex
	conns map[int32]*Conn // userID -> live MPD socket
	areas map[int32]int32 // userID -> lobby square area (for area chat fan-out)
}

// NewHub builds an empty Hub bound to the shared session store.
func NewHub(store *session.Store) *Hub {
	return &Hub{Store: store, conns: map[int32]*Conn{}, areas: map[int32]int32{}}
}

// register installs c as userID's live socket, evicting (and closing) any stale one.
func (h *Hub) register(userID int32, c *Conn) {
	h.mu.Lock()
	old := h.conns[userID]
	h.conns[userID] = c
	h.mu.Unlock()
	if old != nil && old != c {
		old.Close()
	}
}

// unregister drops c as userID's socket if it is still the current one.
func (h *Hub) unregister(userID int32, c *Conn) {
	h.mu.Lock()
	if h.conns[userID] == c {
		delete(h.conns, userID)
	}
	h.mu.Unlock()
}

// SetArea records which central-square area a user is standing in, so area chat can
// fan out to co-occupants. Called by the Battle server on lobby join.
func (h *Hub) SetArea(userID, area int32) {
	h.mu.Lock()
	h.areas[userID] = area
	h.mu.Unlock()
}

// ClearArea forgets a user's square (lobby leave / disconnect).
func (h *Hub) ClearArea(userID int32) {
	h.mu.Lock()
	delete(h.areas, userID)
	h.mu.Unlock()
}

// AreaOf returns the square a user is in, or 0.
func (h *Hub) AreaOf(userID int32) int32 {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.areas[userID]
}

// AreaMembers snapshots every user currently recorded in the given square (area>0).
func (h *Hub) AreaMembers(area int32) []int32 {
	h.mu.Lock()
	defer h.mu.Unlock()
	if area == 0 {
		return nil
	}
	var out []int32
	for uid, a := range h.areas {
		if a == area {
			out = append(out, uid)
		}
	}
	return out
}

// Online reports whether a user has a live MPD socket.
func (h *Hub) Online(userID int32) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	_, ok := h.conns[userID]
	return ok
}

// StatusOK is the success status every pushed packet must carry: the client's
// CtrlPacketValidator (registered for chat.message_mpd, hero.get_data_list_mpd, the
// group commands, ...) drops any packet whose Status != 100 BEFORE its parser/handler
// ever runs. CtrlPacket reads Status from arguments["status"], a sibling of the
// "arguments" payload -- exactly where a Ctrl response puts it (see ctrlproto.Add).
const StatusOK int32 = 100

// Push delivers one command to a user's MPD socket (no-op if the user is offline).
// key is the "object|action" pair WITHOUT the "_mpd" suffix (the client appends it
// in ParseCmd); the payload rides under "arguments", alongside the mandatory
// status:100 the validator checks. The socket is snapshotted under the Hub lock and
// written to OUTSIDE it, so the Hub mutex is the innermost lock and never nests with
// the lobby/session mutexes.
func (h *Hub) Push(userID int32, key string, arguments *amf.MixedArray) {
	h.mu.Lock()
	c := h.conns[userID]
	h.mu.Unlock()
	if c == nil {
		return
	}
	root := amf.NewArray().Set(key, amf.NewArray().
		Set("status", StatusOK).
		Set("arguments", arguments))
	c.writeFrame(encodePush(root))
}
