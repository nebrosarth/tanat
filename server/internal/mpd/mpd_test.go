package mpd

import (
	"bufio"
	"bytes"
	"net"
	"testing"
	"time"

	"tanatserver/internal/amf"
	"tanatserver/internal/session"
)

// startHub starts a Hub on an ephemeral port and returns it plus the dial address.
func startHub(t *testing.T, store *session.Store) (*Hub, string) {
	t.Helper()
	h := NewHub(store)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	go h.Serve(ln)
	return h, ln.Addr().String()
}

// loginFrame builds the client's first MPD frame: {id, sid} as framed AMF.
func loginFrame(id int32, sid string) []byte {
	var buf bytes.Buffer
	_ = amf.NewRawEncoder().EncodeMessage(&buf, amf.NewArray().Set("id", id).Set("sid", sid))
	return frame(buf.Bytes())
}

// readOneFrame reads a single length-prefixed frame body and AMF-decodes it.
func readOneFrame(t *testing.T, br *bufio.Reader) interface{} {
	t.Helper()
	body, err := readFrame(br)
	if err != nil {
		t.Fatalf("read frame: %v", err)
	}
	v, err := amf.NewDecoder(bytes.NewReader(body)).DecodeValue()
	if err != nil {
		t.Fatalf("decode frame: %v", err)
	}
	return v
}

// TestMPDHandshakeAndPush: a client that sends a valid {id,sid} login frame gets the
// integer-100 auth ack, is registered, and then receives a pushed command framed as
// {"chat|message": {"arguments": {...}}}.
func TestMPDHandshakeAndPush(t *testing.T) {
	store := session.NewStore()
	u, sid, _ := store.LoginOrRegister("a@example.com", "pw")

	h, addr := startHub(t, store)

	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Close() })
	_ = c.SetDeadline(time.Now().Add(10 * time.Second))
	br := bufio.NewReader(c)

	if _, err := c.Write(loginFrame(u.ID, sid)); err != nil {
		t.Fatal(err)
	}

	// Auth ack = a bare integer 100.
	ack := readOneFrame(t, br)
	if n, ok := ack.(int32); !ok || n != 100 {
		t.Fatalf("auth ack = %#v, want int32 100", ack)
	}

	// The hub should now have the user registered; wait briefly for register().
	for i := 0; i < 50 && !h.Online(u.ID); i++ {
		time.Sleep(10 * time.Millisecond)
	}
	if !h.Online(u.ID) {
		t.Fatal("user not registered after handshake")
	}

	// Push a chat line and read it back on the socket.
	h.Push(u.ID, "chat|message", amf.NewArray().
		Set("type", "area").Set("msg", "hi").Set("from", "a"))

	v := readOneFrame(t, br)
	root, ok := v.(*amf.MixedArray)
	if !ok {
		t.Fatalf("push root = %#v, want MixedArray", v)
	}
	cmd, ok := root.GetArray("chat|message")
	if !ok {
		t.Fatalf("push missing chat|message key: %#v", root.Assoc)
	}
	// The client's CtrlPacketValidator drops any packet whose Status != 100 before its
	// handler runs, so every push MUST carry status:100 alongside "arguments".
	if st, _ := cmd.GetInt("status"); st != 100 {
		t.Fatalf("push status = %d, want 100 (validator drops it otherwise)", st)
	}
	args, ok := cmd.GetArray("arguments")
	if !ok {
		t.Fatalf("push missing arguments envelope: %#v", cmd.Assoc)
	}
	if msg, _ := args.GetString("msg"); msg != "hi" {
		t.Errorf("pushed msg = %q, want %q", msg, "hi")
	}
	if from, _ := args.GetString("from"); from != "a" {
		t.Errorf("pushed from = %q, want %q", from, "a")
	}
}

// TestMPDAuthFailDropsSocket: a login frame with a bad session id gets no auth ack
// and the socket is closed (the client then treats it as AUTHORIZATION_FAILED).
func TestMPDAuthFailDropsSocket(t *testing.T) {
	store := session.NewStore()
	u, _, _ := store.LoginOrRegister("a@example.com", "pw")
	_, addr := startHub(t, store)

	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Close() })
	_ = c.SetDeadline(time.Now().Add(5 * time.Second))

	if _, err := c.Write(loginFrame(u.ID, "wrong-sid")); err != nil {
		t.Fatal(err)
	}
	// The server drops the connection; a read must hit EOF, not an auth ack.
	br := bufio.NewReader(c)
	if _, err := readFrame(br); err == nil {
		t.Fatal("expected the socket to be dropped on bad sid, but a frame arrived")
	}
}

// TestPushHeroData: a hero broadcast frames as hero.get_data_list_mpd with the hero's
// appearance keyed by id under {data:{...}}, carrying status:100 so it isn't dropped.
func TestPushHeroData(t *testing.T) {
	store := session.NewStore()
	u, sid, _ := store.LoginOrRegister("a@example.com", "pw")
	store.CreateHero(u, 2, false, 3, 4, 5, 6, 7)

	h, addr := startHub(t, store)
	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Close() })
	_ = c.SetDeadline(time.Now().Add(10 * time.Second))
	br := bufio.NewReader(c)

	if _, err := c.Write(loginFrame(u.ID, sid)); err != nil {
		t.Fatal(err)
	}
	readOneFrame(t, br) // auth ack
	for i := 0; i < 50 && !h.Online(u.ID); i++ {
		time.Sleep(10 * time.Millisecond)
	}

	h.PushHeroData(u.ID, []int32{u.ID})

	root, ok := readOneFrame(t, br).(*amf.MixedArray)
	if !ok {
		t.Fatal("hero push root is not a MixedArray")
	}
	cmd, ok := root.GetArray("hero|get_data_list")
	if !ok {
		t.Fatalf("missing hero|get_data_list key: %#v", root.Assoc)
	}
	if st, _ := cmd.GetInt("status"); st != 100 {
		t.Fatalf("hero push status = %d, want 100", st)
	}
	args, _ := cmd.GetArray("arguments")
	data, ok := args.GetArray("data")
	if !ok {
		t.Fatalf("missing data map: %#v", args.Assoc)
	}
	entry, ok := data.GetArray("1") // hero id == user id == 1
	if !ok {
		t.Fatalf("missing hero entry keyed by id: %#v", data.Assoc)
	}
	load, ok := entry.GetArray("load")
	if !ok {
		t.Fatalf("hero entry missing load: %#v", entry.Assoc)
	}
	if race, _ := load.GetInt("race"); race != 2 {
		t.Errorf("hero load race = %d, want 2", race)
	}
}

// TestAreaMembers: SetArea/AreaMembers/ClearArea track square occupancy for chat.
func TestAreaMembers(t *testing.T) {
	h := NewHub(session.NewStore())
	h.SetArea(1, 367)
	h.SetArea(2, 367)
	h.SetArea(3, 368)
	got := map[int32]bool{}
	for _, id := range h.AreaMembers(367) {
		got[id] = true
	}
	if !got[1] || !got[2] || got[3] || len(got) != 2 {
		t.Fatalf("AreaMembers(367) = %v, want {1,2}", got)
	}
	if h.AreaOf(3) != 368 {
		t.Errorf("AreaOf(3) = %d, want 368", h.AreaOf(3))
	}
	h.ClearArea(2)
	if len(h.AreaMembers(367)) != 1 {
		t.Errorf("after ClearArea(2), area 367 should have 1 member")
	}
	if h.AreaMembers(0) != nil {
		t.Errorf("AreaMembers(0) must be nil (0 = unknown area)")
	}
}
