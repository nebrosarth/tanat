package mpd

import (
	"bufio"
	"log"
	"net"
	"sync"
)

// Conn is one client's MPD socket. wm serializes frame writes so concurrent pushes
// (chat + invite from different goroutines) never interleave bytes on the wire.
type Conn struct {
	net.Conn
	wm     sync.Mutex
	userID int32
}

func (c *Conn) writeFrame(b []byte) {
	c.wm.Lock()
	defer c.wm.Unlock()
	_, _ = c.Write(b)
}

// ListenAndServe accepts MPD connections on addr until the listener errors.
func (h *Hub) ListenAndServe(addr string) error {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	log.Printf("mpd: listening on %s", ln.Addr())
	return h.Serve(ln)
}

// Serve accepts MPD connections on an already-open listener (used by tests).
func (h *Hub) Serve(ln net.Listener) error {
	for {
		nc, err := ln.Accept()
		if err != nil {
			return err
		}
		go h.handleConn(nc)
	}
}

// handleConn runs one MPD socket: read the login frame, authenticate against the
// session store, send the auth-ack (integer 100), register, then hold the socket
// open (the client pushes nothing of interest) until EOF.
func (h *Hub) handleConn(nc net.Conn) {
	defer nc.Close()
	br := bufio.NewReaderSize(nc, 1024)

	body, err := readFrame(br)
	if err != nil {
		return
	}
	id, sid, ok := decodeLogin(body)
	if !ok {
		log.Printf("mpd: %s malformed login frame", nc.RemoteAddr())
		return
	}
	u, valid := h.Store.BySessKey(sid)
	if !valid || u.ID != id {
		log.Printf("mpd: %s auth failed (id=%d)", nc.RemoteAddr(), id)
		return // dropping the socket (no ack) makes the client treat it as failed
	}

	c := &Conn{Conn: nc, userID: id}
	h.register(id, c)
	c.writeFrame(authAck)
	log.Printf("mpd: user %d connected from %s", id, nc.RemoteAddr())
	if h.OnConnect != nil {
		h.OnConnect(id)
	}

	// The client sends nothing after login; block until the socket closes.
	for {
		if _, err := readFrame(br); err != nil {
			break
		}
	}

	h.unregister(id, c)
	if h.OnDisconnect != nil {
		h.OnDisconnect(id)
	}
	log.Printf("mpd: user %d disconnected", id)
}
