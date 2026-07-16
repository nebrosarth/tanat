package ctrlserver

import (
	"bufio"
	"net"
	"net/http/httptest"
	"testing"
	"time"

	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
	"tanatserver/internal/mpd"
)

// readPushArgs reads MPD frames until one carries the given "obj|act" key, and
// returns its arguments payload (fails on timeout / too many unrelated frames).
func readPushArgs(t *testing.T, br *bufio.Reader, key string) *amf.MixedArray {
	t.Helper()
	for i := 0; i < 16; i++ {
		v := readMPDFrame(t, br)
		root, ok := v.(*amf.MixedArray)
		if !ok {
			continue
		}
		if cmd, ok := root.GetArray(key); ok {
			args, _ := cmd.GetArray("arguments")
			return args
		}
	}
	t.Fatalf("did not receive push %q", key)
	return nil
}

// TestGroupInviteAcceptListLeave drives the full party flow over HTTP (requests) +
// MPD (pushes): A invites B, B accepts, both list the party, then B leaves and A is
// told the party disbanded.
func TestGroupInviteAcceptListLeave(t *testing.T) {
	srv := New()
	hub := mpd.NewHub(srv.Store)
	srv.MPD = hub
	hub.OnConnect = srv.NotifyOnline
	hub.OnDisconnect = srv.NotifyOffline

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
	post := func(sess, obj, action string, params *amf.MixedArray, counter int32) *amf.MixedArray {
		return postEnvelope(t, url, amf.NewArray().Set("object", obj).Set("action", action).
			Set("params", params).Set("sess_uid", int32(0)).Set("sess_key", sess).Set("counter", counter))
	}

	aID, aSid := login("a@example.com", 1)
	bID, bSid := login("b@example.com", 2)
	_, brA := dialMPD(t, mpdAddr, aID, aSid)
	_, brB := dialMPD(t, mpdAddr, bID, bSid)

	// A invites B.
	inv, _ := post(aSid, "user", "join_from_group_request",
		amf.NewArray().Set("user_id", bID).Set("referred", false), 10).
		GetArray(ctrlproto.CmdKey("user", "join_from_group_request"))
	if inGroup, _ := inv.GetBool("user_in_group"); inGroup {
		t.Fatal("invite reply said B already in a group")
	}
	// B receives the invite push with A's id.
	if args := readPushArgs(t, brB, "user|join_from_group_request"); func() bool { id, _ := args.GetInt("user_id"); return id != aID }() {
		t.Fatal("B's invite push did not carry A's id")
	}

	// B accepts.
	post(bSid, "user", "join_from_group_answer",
		amf.NewArray().Set("user_id", aID).Set("answer", true), 11)
	// A is told B joined.
	if args := readPushArgs(t, brA, "user|joined_to_group"); func() bool { id, _ := args.GetInt("user"); return id != bID }() {
		t.Fatal("A's joined_to_group push did not carry B's id")
	}

	// A lists the party: both members, A is leader (leader==0).
	gl, _ := post(aSid, "user", "group_list", amf.NewArray(), 12).
		GetArray(ctrlproto.CmdKey("user", "group_list"))
	users, _ := gl.GetArray("users")
	if len(users.Dense) != 2 {
		t.Fatalf("A's group_list has %d members, want 2", len(users.Dense))
	}
	if ldr, _ := gl.GetInt("leader"); ldr != 0 {
		t.Errorf("A's group_list leader = %d, want 0 (self is leader)", ldr)
	}
	// B lists the party: leader is A's id (someone else leads).
	glB, _ := post(bSid, "user", "group_list", amf.NewArray(), 13).
		GetArray(ctrlproto.CmdKey("user", "group_list"))
	if ldr, _ := glB.GetInt("leader"); ldr != aID {
		t.Errorf("B's group_list leader = %d, want A %d", ldr, aID)
	}

	// B leaves -> the 2-person party collapses -> A gets group_deleted.
	post(bSid, "user", "leave_group", amf.NewArray(), 14)
	readPushArgs(t, brA, "user|group_deleted")
	if srv.Store.InGroup(aID) || srv.Store.InGroup(bID) {
		t.Error("party should be fully disbanded after B left a 2-person group")
	}
}

// TestGroupListSoloIsSelfLeader: a solo player's group_list reports leader=0 (self is
// their own leader) with no members, so the client's Group.IsLeader is true and the
// central-square popup offers "invite to group".
func TestGroupListSoloIsSelfLeader(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	lr, _ := postEnvelope(t, url, loginEnvelope("solo@example.com", "pw", "1.11", "0", "", 1)).
		GetArray(ctrlproto.CmdKey("user", "login"))
	sid, _ := lr.GetString("sess_key")

	gl, _ := postEnvelope(t, url, amf.NewArray().Set("object", "user").Set("action", "group_list").
		Set("params", amf.NewArray()).Set("sess_uid", int32(0)).Set("sess_key", sid).Set("counter", int32(2))).
		GetArray(ctrlproto.CmdKey("user", "group_list"))
	if ldr, _ := gl.GetInt("leader"); ldr != 0 {
		t.Errorf("solo group_list leader = %d, want 0 (self-leader so invite shows)", ldr)
	}
	if users, _ := gl.GetArray("users"); len(users.Dense) != 0 {
		t.Errorf("solo group_list should have no members, got %d", len(users.Dense))
	}
}

// TestGroupPresenceOnConnect: a co-member's MPD connect pushes user|online to the
// other member.
func TestGroupPresenceOnConnect(t *testing.T) {
	srv := New()
	hub := mpd.NewHub(srv.Store)
	srv.MPD = hub
	hub.OnConnect = srv.NotifyOnline
	hub.OnDisconnect = srv.NotifyOffline

	ln, _ := net.Listen("tcp", "127.0.0.1:0")
	t.Cleanup(func() { ln.Close() })
	go hub.Serve(ln)

	// Two logged-in users forming a party.
	u1, s1 := srv.Store.LoginOrRegister("m1@example.com", "pw")
	u2, s2 := srv.Store.LoginOrRegister("m2@example.com", "pw")
	if _, ok := srv.Store.JoinGroup(u1.ID, u2.ID); !ok {
		t.Fatal("failed to form the test party")
	}

	_, br1 := dialMPD(t, ln.Addr().String(), u1.ID, s1)
	// Give member 1 a moment to register.
	for i := 0; i < 50 && !hub.Online(u1.ID); i++ {
		time.Sleep(10 * time.Millisecond)
	}
	// Now member 2 connects -> member 1 must get user|online for u2.
	dialMPD(t, ln.Addr().String(), u2.ID, s2)
	args := readPushArgs(t, br1, "user|online")
	if id, _ := args.GetInt("user_id"); id != u2.ID {
		t.Errorf("member 1 got online for %d, want %d", id, u2.ID)
	}
}
