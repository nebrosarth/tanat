package ctrlserver

import (
	"net"
	"net/http/httptest"
	"testing"

	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
	"tanatserver/internal/mpd"
	"tanatserver/internal/session"
)

// clanTestPost / clanTestLogin mirror the closures the party tests use.
func clanLogin(t *testing.T, url, email string, counter int32) (int32, string) {
	t.Helper()
	lr, _ := postEnvelope(t, url, loginEnvelope(email, "pw", "1.11", "0", "", counter)).
		GetArray(ctrlproto.CmdKey("user", "login"))
	sk, _ := lr.GetString("sess_key")
	id, _ := lr.GetInt("id")
	return id, sk
}

func clanPost(t *testing.T, url, sess, action string, params *amf.MixedArray, counter int32) *amf.MixedArray {
	t.Helper()
	return postEnvelope(t, url, amf.NewArray().Set("object", "clan").Set("action", action).
		Set("params", params).Set("sess_uid", int32(0)).Set("sess_key", sess).Set("counter", counter))
}

// TestClanCreateAndInfoWire: create a clan over the wire, then read it back with clan|info
// and check the exact response shape the client's parsers expect (root {id}; info root
// {id,tag,name,level,rating} + a DENSE users array of {id,nick,role}).
func TestClanCreateAndInfoWire(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	id, sid := clanLogin(t, url, "founder@example.com", 1)
	u, _ := srv.Store.ByID(id)
	srv.Store.CreateHero(u, 1, false, 0, 0, 0, 0, 0)
	srv.Store.AddHeroMoney(id, session.ClanCreatePrice) // afford the create

	cr, ok := clanPost(t, url, sid, "create",
		amf.NewArray().Set("name", "WireClan").Set("tag", "WC"), 2).
		GetArray(ctrlproto.CmdKey("clan", "create"))
	if !ok {
		t.Fatal("no clan|create in response")
	}
	clanID, _ := cr.GetInt("id")
	if clanID == 0 {
		t.Fatalf("clan|create returned no id (status=%v error=%v)", cr.Assoc["status"], cr.Assoc["error"])
	}

	ci, ok := clanPost(t, url, sid, "info", amf.NewArray().Set("user_id", id), 3).
		GetArray(ctrlproto.CmdKey("clan", "info"))
	if !ok {
		t.Fatal("no clan|info in response")
	}
	if got, _ := ci.GetInt("id"); got != clanID {
		t.Errorf("info id = %d, want %d", got, clanID)
	}
	if tag, _ := ci.GetString("tag"); tag != "WC" {
		t.Errorf("info tag = %q, want WC", tag)
	}
	if name, _ := ci.GetString("name"); name != "WireClan" {
		t.Errorf("info name = %q, want WireClan", name)
	}
	if _, ok := ci.GetInt("level"); !ok {
		t.Error("info missing level")
	}
	if _, ok := ci.GetInt("rating"); !ok {
		t.Error("info missing rating")
	}
	users, ok := ci.GetArray("users")
	if !ok || len(users.Dense) != 1 {
		t.Fatalf("info users dense len = %d, want 1", len(users.Dense))
	}
	m := users.Dense[0].(*amf.MixedArray)
	if role, _ := m.GetInt("role"); role != session.ClanRoleHead {
		t.Errorf("founder role = %d, want HEAD(5)", role)
	}
	if nick, _ := m.GetString("nick"); nick == "" {
		t.Error("founder roster row missing nick")
	}
}

// TestClanCreateInsufficientFundsWire: a broke founder gets error 7010 (not a clan).
func TestClanCreateInsufficientFundsWire(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	id, sid := clanLogin(t, url, "broke@example.com", 1)
	u, _ := srv.Store.ByID(id)
	srv.Store.CreateHero(u, 1, false, 0, 0, 0, 0, 0) // only the starter wallet, < 200000

	cr, _ := clanPost(t, url, sid, "create",
		amf.NewArray().Set("name", "BrokeClan").Set("tag", "BC"), 2).
		GetArray(ctrlproto.CmdKey("clan", "create"))
	if errc, _ := cr.GetInt("error"); errc != session.ClanErrNotEnoughMoney {
		t.Errorf("create error = %d, want 7010", errc)
	}
}

// TestClanInviteAcceptWire drives the invite loop over HTTP + MPD: A (a clan HEAD) invites
// B by nick; B receives clan|invite_mpd {nick, clan_name}; B accepts; A receives
// clan|invite_answer_mpd {nick, answer:true}; B is now in the clan.
func TestClanInviteAcceptWire(t *testing.T) {
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

	aID, aSid := clanLogin(t, url, "head@example.com", 1)
	bID, bSid := clanLogin(t, url, "recruit@example.com", 2)
	ua, _ := srv.Store.ByID(aID)
	srv.Store.CreateHero(ua, 1, false, 0, 0, 0, 0, 0)
	srv.Store.AddHeroMoney(aID, session.ClanCreatePrice)
	ub, _ := srv.Store.ByID(bID)
	srv.Store.CreateHero(ub, 2, false, 0, 0, 0, 0, 0)
	if _, code := srv.Store.CreateClan(aID, "RecruitClan", "RC"); code != session.ClanOK {
		t.Fatalf("clan create failed: %d", code)
	}

	_, brA := dialMPD(t, mpdAddr, aID, aSid)
	_, brB := dialMPD(t, mpdAddr, bID, bSid)

	// A invites B by nick.
	inv, _ := clanPost(t, url, aSid, "invite", amf.NewArray().Set("nick", "recruit@example.com"), 10).
		GetArray(ctrlproto.CmdKey("clan", "invite"))
	if st, _ := inv.GetInt("status"); st != ctrlproto.StatusOK {
		t.Fatalf("invite status = %d (error=%v)", st, inv.Assoc["error"])
	}
	// B receives the invitation push carrying the clan name.
	iargs := readPushArgs(t, brB, "clan|invite")
	if cn, _ := iargs.GetString("clan_name"); cn != "RecruitClan" {
		t.Errorf("invite push clan_name = %q, want RecruitClan", cn)
	}

	// B accepts.
	clanPost(t, url, bSid, "invite_answer", amf.NewArray().Set("answer", true), 11)
	// A is told B accepted.
	aargs := readPushArgs(t, brA, "clan|invite_answer")
	if ans, _ := aargs.GetBool("answer"); !ans {
		t.Error("A's invite_answer push should say accepted")
	}

	if v, ok := srv.Store.ClanOfUser(bID); !ok || len(v.Members) != 2 {
		t.Fatalf("B did not join the clan (ok=%v members=%d)", ok, func() int {
			if v != nil {
				return len(v.Members)
			}
			return 0
		}())
	}
}
