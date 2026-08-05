package ctrlserver

import (
	"net"
	"net/http/httptest"
	"strconv"
	"testing"

	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/mpd"
)

func castlePost(t *testing.T, url, sess, action string, params *amf.MixedArray, counter int32) *amf.MixedArray {
	t.Helper()
	return postEnvelope(t, url, amf.NewArray().Set("object", "castle").Set("action", action).
		Set("params", params).Set("sess_uid", int32(0)).Set("sess_key", sess).Set("counter", counter))
}

// TestCastleListShape: castle|list emits an ASSOCIATIVE castles map keyed by castle-id
// string, each entry carrying the exact keys CastleListArgParser reads (incl. a nested
// rewards block and a relative start_time).
func TestCastleListShape(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	_, sid := clanLogin(t, url, "browser@example.com", 1)
	cl, ok := castlePost(t, url, sid, "list", amf.NewArray(), 2).
		GetArray(ctrlproto.CmdKey("castle", "list"))
	if !ok {
		t.Fatal("no castle|list in response")
	}
	castles, ok := cl.GetArray("castles")
	if !ok {
		t.Fatal("castle|list missing required 'castles' key")
	}
	want := gamedata.Castles()
	if len(castles.Assoc) != len(want) {
		t.Fatalf("castles map has %d entries, want %d", len(castles.Assoc), len(want))
	}
	first := want[0]
	entry, ok := castles.GetArray(strconv.Itoa(int(first.ID)))
	if !ok {
		t.Fatalf("castles not keyed by id string %d", first.ID)
	}
	if name, _ := entry.GetString("name"); name != first.NameKey {
		t.Errorf("castle name = %q, want locale key %q", name, first.NameKey)
	}
	if _, ok := entry.GetInt("start_time"); !ok {
		t.Error("castle entry missing start_time")
	}
	rewards, ok := entry.GetArray("rewards")
	if !ok {
		t.Fatal("castle entry missing rewards block")
	}
	if money, _ := rewards.GetInt("money"); money != first.Reward.Money {
		t.Errorf("reward money = %d, want %d", money, first.Reward.Money)
	}
	if _, ok := rewards.GetInt("money_d"); !ok {
		t.Error("rewards missing money_d (diamonds)")
	}
}

// TestCastleInfoFlags: castle|info returns the members payload with all the boolean/int
// eligibility flags the client reads.
func TestCastleInfoFlags(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	id, sid := clanLogin(t, url, "viewer@example.com", 1)
	u, _ := srv.Store.ByID(id)
	srv.Store.CreateHero(u, 1, false, 0, 0, 0, 0, 0) // level 1 -> right_level for castle 1

	ci, ok := castlePost(t, url, sid, "info", amf.NewArray().Set("castle_id", int32(1)), 2).
		GetArray(ctrlproto.CmdKey("castle", "info"))
	if !ok {
		t.Fatal("no castle|info in response")
	}
	for _, key := range []string{"members", "queue"} {
		if _, ok := ci.GetArray(key); !ok {
			t.Errorf("castle|info missing %q array", key)
		}
	}
	if _, ok := ci.GetBool("joined"); !ok {
		t.Error("castle|info missing joined")
	}
	if _, ok := ci.GetBool("editable"); !ok {
		t.Error("castle|info missing editable")
	}
	if _, ok := ci.GetBool("in_progress"); !ok {
		t.Error("castle|info missing in_progress")
	}
	if rl, _ := ci.GetBool("right_level"); !rl {
		t.Error("level-1 hero should be right_level for castle 1 (levels 1-30)")
	}
	if _, ok := ci.GetInt("ban_count"); !ok {
		t.Error("castle|info missing ban_count")
	}
	// An unknown castle is a clean error, not a crash.
	if bad, _ := castlePost(t, url, sid, "info", amf.NewArray().Set("castle_id", int32(9999)), 3).
		GetArray(ctrlproto.CmdKey("castle", "info")); func() bool { e, _ := bad.GetInt("error"); return e == 0 }() {
		t.Error("castle|info for unknown id should return an error")
	}
}

// TestCastleEnrollDesert: set_fighters enrolls the caller (round-trips through fighters),
// desert removes them.
func TestCastleEnrollDesert(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	id, sid := clanLogin(t, url, "fighter@example.com", 1)
	u, _ := srv.Store.ByID(id)
	srv.Store.CreateHero(u, 1, false, 0, 0, 0, 0, 0)

	// Enroll self into group 2 of castle 1.
	fighters := amf.NewArray().Set(strconv.Itoa(int(id)), int32(2))
	sf, _ := castlePost(t, url, sid, "set_fighters",
		amf.NewArray().Set("castle_id", int32(1)).Set("fighters", fighters), 2).
		GetArray(ctrlproto.CmdKey("castle", "set_fighters"))
	if st, _ := sf.GetInt("status"); st != ctrlproto.StatusOK {
		t.Fatalf("set_fighters status = %d", st)
	}

	fr, _ := castlePost(t, url, sid, "fighters", amf.NewArray().Set("castle_id", int32(1)), 3).
		GetArray(ctrlproto.CmdKey("castle", "fighters"))
	fmap, _ := fr.GetArray("fighters")
	if grp, ok := fmap.GetInt(strconv.Itoa(int(id))); !ok || grp != 2 {
		t.Fatalf("enrolled fighter group = %d ok=%v, want 2", grp, ok)
	}
	if cl, _ := fr.GetBool("can_leave"); !cl {
		t.Error("fighters should report can_leave=true")
	}

	// Desert -> roster empties.
	castlePost(t, url, sid, "desert", amf.NewArray().Set("castle_id", int32(1)), 4)
	fr2, _ := castlePost(t, url, sid, "fighters", amf.NewArray().Set("castle_id", int32(1)), 5).
		GetArray(ctrlproto.CmdKey("castle", "fighters"))
	fmap2, _ := fr2.GetArray("fighters")
	if _, ok := fmap2.GetInt(strconv.Itoa(int(id))); ok {
		t.Error("fighter still enrolled after desert")
	}
}

// TestCastleBattleWindowDraftAndLaunch drives the full scheduler-fired flow over HTTP +
// MPD: a fighter enrolls, the scheduler is fast-forwarded to fire the battle window,
// the fighter receives castle|start_request + select_avatar_timer, picks an avatar,
// readies, and gets castle|launch backed by a PendingBattle tagged with the castle.
func TestCastleBattleWindowDraftAndLaunch(t *testing.T) {
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

	id, sid := clanLogin(t, url, "besieger@example.com", 1)
	u, _ := srv.Store.ByID(id)
	srv.Store.CreateHero(u, 1, false, 0, 0, 0, 0, 0)

	fighters := amf.NewArray().Set(strconv.Itoa(int(id)), int32(1))
	castlePost(t, url, sid, "set_fighters",
		amf.NewArray().Set("castle_id", int32(1)).Set("fighters", fighters), 2)

	_, br := dialMPD(t, mpdAddr, id, sid)

	// Fast-forward the scheduler well past castle 1's countdown.
	srv.tickCastles(999999)

	sr := readPushArgs(t, br, "castle|start_request")
	if cid, _ := sr.GetInt("castle_id"); cid != 1 {
		t.Errorf("start_request castle_id = %d, want 1", cid)
	}
	fmap, ok := sr.GetArray("fighters")
	if !ok {
		t.Fatal("start_request missing fighters")
	}
	if _, ok := fmap.GetArray(strconv.Itoa(int(id))); !ok {
		t.Error("start_request fighters missing the enrolled fighter")
	}
	timerArgs := readPushArgs(t, br, "castle|select_avatar_timer")
	if tm, _ := timerArgs.GetInt("time"); tm != castleSelectTimeoutSec {
		t.Errorf("select_avatar_timer time = %d, want %d", tm, castleSelectTimeoutSec)
	}

	if !srv.castleInProgress(1) {
		t.Error("castle 1 should show in_progress once the window has fired")
	}

	// Pick an avatar.
	av := gamedata.Avatars()[0]
	sa, _ := castlePost(t, url, sid, "select_avatar", amf.NewArray().Set("avatar_id", av.ID), 3).
		GetArray(ctrlproto.CmdKey("castle", "select_avatar"))
	if st, _ := sa.GetInt("status"); st != ctrlproto.StatusOK {
		t.Fatalf("select_avatar status = %d", st)
	}
	readPushArgs(t, br, "castle|select_avatar") // roster-tile broadcast

	// Ready -> launch.
	rd, _ := castlePost(t, url, sid, "ready", amf.NewArray(), 4).
		GetArray(ctrlproto.CmdKey("castle", "ready"))
	if st, _ := rd.GetInt("status"); st != ctrlproto.StatusOK {
		t.Fatalf("ready status = %d", st)
	}
	readPushArgs(t, br, "castle|ready")
	c1 := gamedata.Castles()[0]
	launch := readPushArgs(t, br, "castle|launch")
	if mid, _ := launch.GetInt("map_id"); mid != c1.MapID {
		t.Errorf("launch map_id = %d, want %d", mid, c1.MapID)
	}
	if scene, _ := launch.GetString("scene"); scene != c1.Scene {
		t.Errorf("launch scene = %q, want %q", scene, c1.Scene)
	}

	pb, ok := srv.Store.TakePendingBattle(id)
	if !ok {
		t.Fatal("no PendingBattle stored after castle|ready")
	}
	if pb.CastleID != 1 {
		t.Errorf("PendingBattle.CastleID = %d, want 1", pb.CastleID)
	}
	if pb.AvatarID != av.ID {
		t.Errorf("PendingBattle.AvatarID = %d, want %d", pb.AvatarID, av.ID)
	}

	if srv.castleInProgress(1) {
		t.Error("castle 1 should no longer be in_progress once the only fighter has readied")
	}
}

// TestCastleDesertBattleLeavesDraft: castle|desert_battle clears the in-flight
// selection without touching the (separate) registration roster.
func TestCastleDesertBattleLeavesDraft(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	id, sid := clanLogin(t, url, "waverer@example.com", 1)
	u, _ := srv.Store.ByID(id)
	srv.Store.CreateHero(u, 1, false, 0, 0, 0, 0, 0)
	fighters := amf.NewArray().Set(strconv.Itoa(int(id)), int32(1))
	castlePost(t, url, sid, "set_fighters",
		amf.NewArray().Set("castle_id", int32(1)).Set("fighters", fighters), 2)

	srv.tickCastles(999999)
	if !srv.castleInProgress(1) {
		t.Fatal("window did not fire")
	}

	db, _ := castlePost(t, url, sid, "desert_battle", amf.NewArray(), 3).
		GetArray(ctrlproto.CmdKey("castle", "desert_battle"))
	if st, _ := db.GetInt("status"); st != ctrlproto.StatusOK {
		t.Fatalf("desert_battle status = %d", st)
	}
	if srv.castleInProgress(1) {
		t.Error("desert_battle should clear the draft selection")
	}
	// The registration roster itself is untouched by desert_battle.
	if !srv.castleContains(1, id) {
		t.Error("desert_battle should not remove the fighter from the registration roster")
	}
}

// TestCastleWindowWithNoFightersResetsQuietly: an empty roster lets the battle window
// pass with no draft and no crash; the countdown still resets for the next window.
func TestCastleWindowWithNoFightersResetsQuietly(t *testing.T) {
	srv := New()
	before := srv.castleCountdownRemaining(gamedata.Castles()[0])
	srv.tickCastles(before + 1)
	if srv.castleInProgress(1) {
		t.Error("an empty roster should never start a draft")
	}
	after := srv.castleCountdownRemaining(gamedata.Castles()[0])
	if after <= 0 {
		t.Errorf("countdown after an empty window = %d, want it reset to a positive CycleSec", after)
	}
}

// TestCastleBattleInfoEmpty: the deferred bracket returns an empty-but-valid stage list.
func TestCastleBattleInfoEmpty(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	_, sid := clanLogin(t, url, "spectator@example.com", 1)
	bi, ok := castlePost(t, url, sid, "battle_info", amf.NewArray().Set("castle_id", int32(1)), 2).
		GetArray(ctrlproto.CmdKey("castle", "battle_info"))
	if !ok {
		t.Fatal("no castle|battle_info in response")
	}
	stages, ok := bi.GetArray("stages")
	if !ok {
		t.Fatal("battle_info missing 'stages'")
	}
	if len(stages.Dense) != 0 {
		t.Errorf("stages len = %d, want 0 (bracket deferred)", len(stages.Dense))
	}
}
