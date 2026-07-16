package ctrlserver

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"io"
	"net"
	"net/http/httptest"
	"testing"
	"time"

	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
	"tanatserver/internal/mpd"
)

// mpdLoginFrame builds the client's first MPD frame: {id, sid} as framed AMF.
func mpdLoginFrame(id int32, sid string) []byte {
	var b bytes.Buffer
	_ = amf.NewRawEncoder().EncodeMessage(&b, amf.NewArray().Set("id", id).Set("sid", sid))
	body := b.Bytes()
	out := make([]byte, 4+len(body))
	binary.BigEndian.PutUint32(out[:4], uint32(len(body)))
	copy(out[4:], body)
	return out
}

// readMPDFrame reads one length-prefixed frame and AMF-decodes it.
func readMPDFrame(t *testing.T, r *bufio.Reader) interface{} {
	t.Helper()
	var lb [4]byte
	if _, err := io.ReadFull(r, lb[:]); err != nil {
		t.Fatalf("read frame len: %v", err)
	}
	body := make([]byte, binary.BigEndian.Uint32(lb[:]))
	if _, err := io.ReadFull(r, body); err != nil {
		t.Fatalf("read frame body: %v", err)
	}
	v, err := amf.NewDecoder(bytes.NewReader(body)).DecodeValue()
	if err != nil {
		t.Fatalf("decode frame: %v", err)
	}
	return v
}

// dialMPD dials the hub, sends the login frame, and reads the auth ack.
func dialMPD(t *testing.T, addr string, id int32, sid string) (net.Conn, *bufio.Reader) {
	t.Helper()
	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Close() })
	_ = c.SetDeadline(time.Now().Add(10 * time.Second))
	if _, err := c.Write(mpdLoginFrame(id, sid)); err != nil {
		t.Fatal(err)
	}
	br := bufio.NewReader(c)
	if ack, ok := readMPDFrame(t, br).(int32); !ok || ack != 100 {
		t.Fatalf("mpd user %d auth ack = %#v, want 100", id, ack)
	}
	return c, br
}

// TestAreaChatFanOut: player A sends an area chat line over HTTP (chat|add); every
// occupant of the same square (A and B) receives it as a chat|message MPD push.
func TestAreaChatFanOut(t *testing.T) {
	srv := New()
	hub := mpd.NewHub(srv.Store)
	srv.MPD = hub
	srv.MPDHost = "127.0.0.1"
	srv.MPDPorts = []int32{0}

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	go hub.Serve(ln)
	mpdAddr := ln.Addr().String()

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	login := func(email string, counter int32) (int32, string) {
		lr, _ := postEnvelope(t, url, loginEnvelope(email, "pw", "1.11", "0", "", counter)).
			GetArray(ctrlproto.CmdKey("user", "login"))
		sk, _ := lr.GetString("sess_key")
		id, _ := lr.GetInt("id")
		return id, sk
	}
	aID, aSid := login("a@example.com", 1)
	bID, bSid := login("b@example.com", 2)

	// Both connect to MPD and stand in the same square.
	_, _ = dialMPD(t, mpdAddr, aID, aSid)
	_, brB := dialMPD(t, mpdAddr, bID, bSid)
	for i := 0; i < 50 && !(hub.Online(aID) && hub.Online(bID)); i++ {
		time.Sleep(10 * time.Millisecond)
	}
	hub.SetArea(aID, 367)
	hub.SetArea(bID, 367)

	// A sends an area chat line over HTTP.
	postEnvelope(t, url, amf.NewArray().Set("object", "chat").Set("action", "add").
		Set("params", amf.NewArray().Set("type", "area").
			Set("recipient_list", amf.NewArray()).Set("message", "privet ploshchad")).
		Set("sess_uid", int32(0)).Set("sess_key", aSid).Set("counter", int32(10)))

	// B must receive it as a chat|message push.
	v := readMPDFrame(t, brB)
	root, ok := v.(*amf.MixedArray)
	if !ok {
		t.Fatalf("B push root = %#v, want MixedArray", v)
	}
	cmd, ok := root.GetArray("chat|message")
	if !ok {
		t.Fatalf("B push missing chat|message: %#v", root.Assoc)
	}
	args, _ := cmd.GetArray("arguments")
	if msg, _ := args.GetString("msg"); msg != "privet ploshchad" {
		t.Errorf("B received msg = %q, want the sent line", msg)
	}
	if typ, _ := args.GetString("type"); typ != "area" {
		t.Errorf("B received type = %q, want area", typ)
	}
	if _, ok := args.GetFloat("ctime"); !ok {
		t.Errorf("chat push missing ctime")
	}
}

// TestPrivateAndGroupChat: a private message reaches only the addressed nick (+ the
// sender's echo), not a bystander; a group message reaches the sender's whole party.
func TestPrivateAndGroupChat(t *testing.T) {
	srv := New()
	hub := mpd.NewHub(srv.Store)
	srv.MPD = hub

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	go hub.Serve(ln)
	mpdAddr := ln.Addr().String()

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	login := func(email string, counter int32) (int32, string) {
		lr, _ := postEnvelope(t, url, loginEnvelope(email, "pw", "1.11", "0", "", counter)).
			GetArray(ctrlproto.CmdKey("user", "login"))
		sk, _ := lr.GetString("sess_key")
		id, _ := lr.GetInt("id")
		return id, sk
	}
	send := func(sess string, counter int32, typ, message string, nicks ...string) {
		rl := amf.NewArray()
		for _, n := range nicks {
			rl.Add(n)
		}
		postEnvelope(t, url, amf.NewArray().Set("object", "chat").Set("action", "add").
			Set("params", amf.NewArray().Set("type", typ).
				Set("recipient_list", rl).Set("message", message)).
			Set("sess_uid", int32(0)).Set("sess_key", sess).Set("counter", counter))
	}
	// nicks are the login email (Username defaults to the email).
	aID, aSid := login("a@example.com", 1)
	bID, bSid := login("b@example.com", 2)
	cID, cSid := login("c@example.com", 3)

	_, brA := dialMPD(t, mpdAddr, aID, aSid)
	_, brB := dialMPD(t, mpdAddr, bID, bSid)
	cConn, brC := dialMPD(t, mpdAddr, cID, cSid)
	for i := 0; i < 50 && !(hub.Online(aID) && hub.Online(bID) && hub.Online(cID)); i++ {
		time.Sleep(10 * time.Millisecond)
	}

	// A whispers B. B (target) and A (echo) get it; C must not.
	send(aSid, 10, "private_msg", "secret", "b@example.com")
	if got := chatMsg(t, brB); got != "secret" {
		t.Errorf("B private msg = %q, want secret", got)
	}
	if got := chatMsg(t, brA); got != "secret" {
		t.Errorf("A private echo = %q, want secret", got)
	}
	assertNoFrame(t, cConn, brC, "C should not receive A's private message to B")

	// A and B form a party; A posts group chat -> A and B receive it.
	if _, ok := srv.Store.JoinGroup(aID, bID); !ok {
		t.Fatal("failed to form party")
	}
	send(aSid, 11, "group", "party up", "")
	if got := chatMsg(t, brA); got != "party up" {
		t.Errorf("A group msg = %q, want 'party up'", got)
	}
	if got := chatMsg(t, brB); got != "party up" {
		t.Errorf("B group msg = %q, want 'party up'", got)
	}
	assertNoFrame(t, cConn, brC, "C is not in the party and should not receive group chat")
}

// chatMsg reads one chat|message push and returns its msg text.
func chatMsg(t *testing.T, br *bufio.Reader) string {
	t.Helper()
	msg, _ := chatMsgTyped(t, br)
	return msg
}

// chatMsgTyped reads one chat|message push and returns (msg, type).
func chatMsgTyped(t *testing.T, br *bufio.Reader) (string, string) {
	t.Helper()
	root, ok := readMPDFrame(t, br).(*amf.MixedArray)
	if !ok {
		t.Fatal("chat push root is not a MixedArray")
	}
	cmd, ok := root.GetArray("chat|message")
	if !ok {
		t.Fatalf("push missing chat|message: %#v", root.Assoc)
	}
	if st, _ := cmd.GetInt("status"); st != 100 {
		t.Fatalf("chat push status = %d, want 100", st)
	}
	args, _ := cmd.GetArray("arguments")
	msg, _ := args.GetString("msg")
	typ, _ := args.GetString("type")
	return msg, typ
}

// TestDirectedAreaBecomesPrivate: a message typed in the area tab but addressed to a
// nick (the "private message" popup only prepends "[nick]", it doesn't switch tabs) is
// promoted to a private whisper -- delivered only to the target, relayed as
// type="private_msg" so it lands in the private tab, and never broadcast to the square.
func TestDirectedAreaBecomesPrivate(t *testing.T) {
	srv := New()
	hub := mpd.NewHub(srv.Store)
	srv.MPD = hub

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	go hub.Serve(ln)
	mpdAddr := ln.Addr().String()

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	login := func(email string, counter int32) (int32, string) {
		lr, _ := postEnvelope(t, url, loginEnvelope(email, "pw", "1.11", "0", "", counter)).
			GetArray(ctrlproto.CmdKey("user", "login"))
		sk, _ := lr.GetString("sess_key")
		id, _ := lr.GetInt("id")
		return id, sk
	}
	aID, aSid := login("a@example.com", 1)
	bID, bSid := login("b@example.com", 2)
	cID, cSid := login("c@example.com", 3)

	_, brA := dialMPD(t, mpdAddr, aID, aSid)
	_, brB := dialMPD(t, mpdAddr, bID, bSid)
	cConn, brC := dialMPD(t, mpdAddr, cID, cSid)
	for i := 0; i < 50 && !(hub.Online(aID) && hub.Online(bID) && hub.Online(cID)); i++ {
		time.Sleep(10 * time.Millisecond)
	}
	// All three share the square, so a real area line would reach C.
	hub.SetArea(aID, 367)
	hub.SetArea(bID, 367)
	hub.SetArea(cID, 367)

	// A, standing in the area tab, addresses B by nick.
	postEnvelope(t, url, amf.NewArray().Set("object", "chat").Set("action", "add").
		Set("params", amf.NewArray().Set("type", "area").
			Set("recipient_list", amf.NewArray().Add("b@example.com")).
			Set("message", "just for you")).
		Set("sess_uid", int32(0)).Set("sess_key", aSid).Set("counter", int32(10)))

	msg, typ := chatMsgTyped(t, brB)
	if msg != "just for you" || typ != "private_msg" {
		t.Errorf("B got (%q,%q), want (\"just for you\",\"private_msg\")", msg, typ)
	}
	if _, typA := chatMsgTyped(t, brA); typA != "private_msg" {
		t.Errorf("A echo type = %q, want private_msg", typA)
	}
	assertNoFrame(t, cConn, brC, "C shares the square but the directed line must not reach them")
}

// assertNoFrame fails if a frame arrives within a short window, using a read deadline
// (buffered data already delivered counts as an arrival).
func assertNoFrame(t *testing.T, conn net.Conn, br *bufio.Reader, msg string) {
	t.Helper()
	_ = conn.SetReadDeadline(time.Now().Add(300 * time.Millisecond))
	defer conn.SetReadDeadline(time.Now().Add(10 * time.Second))
	var lb [4]byte
	if _, err := io.ReadFull(br, lb[:]); err == nil {
		t.Error(msg)
	}
}
