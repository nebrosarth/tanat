package ctrlserver

import (
	"net"
	"net/http/httptest"
	"testing"
	"time"

	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
	"tanatserver/internal/mpd"
)

// friendsHarness spins up a Ctrl server + MPD hub and returns helpers.
type friendsHarness struct {
	srv   *Server
	hub   *mpd.Hub
	url   string
	mpd   string
	login func(email string, counter int32) (int32, string)
}

func newFriendsHarness(t *testing.T) *friendsHarness {
	t.Helper()
	srv := New()
	hub := mpd.NewHub(srv.Store)
	srv.MPD = hub

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	go hub.Serve(ln)

	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	url := ts.URL + "/entry_point.php"

	h := &friendsHarness{srv: srv, hub: hub, url: url, mpd: ln.Addr().String()}
	h.login = func(email string, counter int32) (int32, string) {
		lr, _ := postEnvelope(t, url, loginEnvelope(email, "pw", "1.11", "0", "", counter)).
			GetArray(ctrlproto.CmdKey("user", "login"))
		sk, _ := lr.GetString("sess_key")
		id, _ := lr.GetInt("id")
		return id, sk
	}
	return h
}

func (h *friendsHarness) post(t *testing.T, sess, action string, params *amf.MixedArray, counter int32) *amf.MixedArray {
	return postEnvelope(t, h.url, amf.NewArray().Set("object", "user").Set("action", action).
		Set("params", params).Set("sess_uid", int32(0)).Set("sess_key", sess).Set("counter", counter))
}

// friendIDs pulls the white-list ids from a user|get_bw_list response.
func (h *friendsHarness) friendIDs(t *testing.T, sess string, counter int32) map[int32]bool {
	gl, _ := h.post(t, sess, "get_bw_list", amf.NewArray().Set("type", int32(0)), counter).
		GetArray(ctrlproto.CmdKey("user", "get_bw_list"))
	white, _ := gl.GetArray("white")
	out := map[int32]bool{}
	for _, v := range white.Dense {
		if row, ok := v.(*amf.MixedArray); ok {
			if id, ok := row.GetInt("id"); ok {
				out[id] = true
			}
		}
	}
	return out
}

// TestFriendRequestAcceptFlow: A adds B -> B gets a friend_request push -> B accepts ->
// A gets an add_to_list refresh push -> both get_bw_list list each other.
func TestFriendRequestAcceptFlow(t *testing.T) {
	h := newFriendsHarness(t)
	aID, aSid := h.login("a@example.com", 1)
	bID, bSid := h.login("b@example.com", 2)

	_, brA := dialMPD(t, h.mpd, aID, aSid)
	_, brB := dialMPD(t, h.mpd, bID, bSid)
	for i := 0; i < 50 && !(h.hub.Online(aID) && h.hub.Online(bID)); i++ {
		time.Sleep(10 * time.Millisecond)
	}

	// A adds B as a friend -> B receives a friend_request push carrying A's id.
	h.post(t, aSid, "add_to_list", amf.NewArray().Set("user_id", bID).Set("type", int32(1)), 10)
	if args := readPushArgs(t, brB, "user|friend_request"); func() bool { id, _ := args.GetInt("user_id"); return id != aID }() {
		t.Fatal("B's friend_request push did not carry A's id")
	}

	// B accepts -> A gets an add_to_list refresh push (answer=true).
	h.post(t, bSid, "friend_answer", amf.NewArray().Set("user_id", aID).Set("answer", true), 11)
	if args := readPushArgs(t, brA, "user|add_to_list"); func() bool { ok, _ := args.GetBool("answer"); return !ok }() {
		t.Fatal("A's add_to_list push should report answer=true")
	}

	// Both lists now contain each other.
	if !h.friendIDs(t, aSid, 12)[bID] {
		t.Error("A's friend list is missing B")
	}
	if !h.friendIDs(t, bSid, 13)[aID] {
		t.Error("B's friend list is missing A")
	}
}

// TestMutualAddInstantFriends: when both players add each other, the second add
// completes the pair immediately (no accept needed) and both are pushed a refresh.
func TestMutualAddInstantFriends(t *testing.T) {
	h := newFriendsHarness(t)
	cID, cSid := h.login("c@example.com", 1)
	dID, dSid := h.login("d@example.com", 2)

	_, brC := dialMPD(t, h.mpd, cID, cSid)
	_, brD := dialMPD(t, h.mpd, dID, dSid)
	for i := 0; i < 50 && !(h.hub.Online(cID) && h.hub.Online(dID)); i++ {
		time.Sleep(10 * time.Millisecond)
	}

	// C adds D (pending request; D gets prompted).
	h.post(t, cSid, "add_to_list", amf.NewArray().Set("user_id", dID).Set("type", int32(1)), 10)
	readPushArgs(t, brD, "user|friend_request")

	// D adds C back -> mutual -> both become friends now, both get an add_to_list push.
	h.post(t, dSid, "add_to_list", amf.NewArray().Set("user_id", cID).Set("type", int32(1)), 11)
	readPushArgs(t, brC, "user|add_to_list")
	readPushArgs(t, brD, "user|add_to_list")

	if !h.friendIDs(t, cSid, 12)[dID] || !h.friendIDs(t, dSid, 13)[cID] {
		t.Error("mutual add did not make C and D friends")
	}
}
