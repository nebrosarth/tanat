package ctrlserver

import (
	"net/http/httptest"
	"strconv"
	"testing"

	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
	"tanatserver/internal/gamedata"
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
	if name, _ := entry.GetString("name"); name != first.Name {
		t.Errorf("castle name = %q, want %q", name, first.Name)
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
