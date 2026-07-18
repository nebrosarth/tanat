package ctrlserver

import (
	"net/http/httptest"
	"strconv"
	"testing"

	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
	"tanatserver/internal/gamedata"
)

// TestDotaMatchmakingFlow drives the Ctrl side of the «Штурм» (DOTA) solo
// instant-match: fight|join records the selection and acks the queue; fight|
// select_avatar records the avatar; fight|ready records a PendingBattle for the
// Battle server with the DOTA scene/room. (MPD pushes are skipped when srv.MPD is
// nil, so this exercises the Ctrl state transitions.)
func TestDotaMatchmakingFlow(t *testing.T) {
	srv := New()
	srv.BattleHost = "127.0.0.1"
	srv.BattlePorts = []int32{9339}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	login := postEnvelope(t, url, loginEnvelope("storm@example.com", "pw", "1.11", "0", "", 1))
	lr, _ := login.GetArray(ctrlproto.CmdKey("user", "login"))
	sessKey, _ := lr.GetString("sess_key")
	userID, _ := lr.GetInt("id")

	mkReq := func(obj, action string, params *amf.MixedArray, counter int32) *amf.MixedArray {
		return amf.NewArray().Set("object", obj).Set("action", action).
			Set("params", params).
			Set("sess_uid", userID).Set("sess_key", sessKey).Set("counter", counter)
	}

	dm := gamedata.DotaMaps()[0]

	// fight|join: queue ack + selection recorded.
	join, _ := postEnvelope(t, url,
		mkReq("fight", "join", amf.NewArray().Set("map_id", dm.ID), 2)).
		GetArray(ctrlproto.CmdKey("fight", "join"))
	if join == nil {
		t.Fatal("no fight|join response")
	}
	if mid, _ := join.GetInt("map_id"); mid != dm.ID {
		t.Errorf("fight|join map_id = %d, want %d", mid, dm.ID)
	}
	if _, ok := srv.getFightSel(userID); !ok {
		t.Fatal("fight|join did not record a selection")
	}

	// fight|select_avatar: choose an avatar.
	av := gamedata.Avatars()[0]
	sa, _ := postEnvelope(t, url,
		mkReq("fight", "select_avatar", amf.NewArray().Set("avatar_id", av.ID), 3)).
		GetArray(ctrlproto.CmdKey("fight", "select_avatar"))
	if status, _ := sa.GetInt("status"); status != ctrlproto.StatusOK {
		t.Fatalf("fight|select_avatar status = %d, want 100", status)
	}
	if sel, _ := srv.getFightSel(userID); sel.avatarID != av.ID {
		t.Fatalf("selected avatar = %d, want %d", sel.avatarID, av.ID)
	}

	// fight|ready: PendingBattle for the Battle server.
	postEnvelope(t, url, mkReq("fight", "ready", amf.NewArray(), 4))
	pb, ok := srv.Store.TakePendingBattle(userID)
	if !ok {
		t.Fatal("fight|ready did not record a PendingBattle")
	}
	if pb.MapID != dm.ID || pb.Scene != dm.Scene {
		t.Errorf("PendingBattle = {map=%d scene=%q}, want {map=%d scene=%q}", pb.MapID, pb.Scene, dm.ID, dm.Scene)
	}
	if pb.AvatarID != av.ID {
		t.Errorf("PendingBattle avatar = %d, want %d", pb.AvatarID, av.ID)
	}
	if pb.Passwd == "" {
		t.Error("PendingBattle has no battle password")
	}
	// The selection is consumed once ready fires.
	if _, ok := srv.getFightSel(userID); ok {
		t.Error("fight selection not cleared after fight|ready")
	}
}

// TestDotaMatchSizeGating: with DotaMatchSize=2 a match must not form until the
// SECOND player queues, and both matched players share one room.
func TestDotaMatchSizeGating(t *testing.T) {
	srv := New()
	if got := srv.SetDotaMatchSize(2); got != 2 {
		t.Fatalf("SetDotaMatchSize(2) = %d", got)
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"
	dm := gamedata.DotaMaps()[0]

	joinAs := func(email string, counter int32) int32 {
		login := postEnvelope(t, url, loginEnvelope(email, "pw", "1.11", "0", "", counter))
		lr, _ := login.GetArray(ctrlproto.CmdKey("user", "login"))
		sessKey, _ := lr.GetString("sess_key")
		uid, _ := lr.GetInt("id")
		req := amf.NewArray().Set("object", "fight").Set("action", "join").
			Set("params", amf.NewArray().Set("map_id", dm.ID)).
			Set("sess_uid", uid).Set("sess_key", sessKey).Set("counter", counter+1)
		postEnvelope(t, url, req)
		return uid
	}

	// First player queues -> waiting, no selection yet (no match formed).
	a := joinAs("stormA@example.com", 1)
	if _, ok := srv.getFightSel(a); ok {
		t.Fatal("match formed with only 1 of 2 players queued")
	}

	// Second player queues -> match forms; both get a selection with the same room.
	b := joinAs("stormB@example.com", 10)
	selA, okA := srv.getFightSel(a)
	selB, okB := srv.getFightSel(b)
	if !okA || !okB {
		t.Fatalf("match did not form for both players: A=%v B=%v", okA, okB)
	}
	if selA.room == 0 || selA.room != selB.room {
		t.Fatalf("matched players must share one room: A.room=%d B.room=%d", selA.room, selB.room)
	}
}

// TestArenaTabListAndMatchmaking exercises the «Арена» (DM) path the client's Arena tab
// drives: arena|get_maps {type:DM} must return the arena map (an empty response is the
// "empty tab" bug), get_map_type_descs must carry a DM blurb, and the map must join and
// launch through the SAME fight|* handlers as «Штурм» (only HUNT uses hunt|*).
func TestArenaTabListAndMatchmaking(t *testing.T) {
	srv := New()
	srv.BattleHost = "127.0.0.1"
	srv.BattlePorts = []int32{9339}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	login := postEnvelope(t, url, loginEnvelope("arena@example.com", "pw", "1.11", "0", "", 1))
	lr, _ := login.GetArray(ctrlproto.CmdKey("user", "login"))
	sessKey, _ := lr.GetString("sess_key")
	userID, _ := lr.GetInt("id")
	mkReq := func(obj, action string, params *amf.MixedArray, counter int32) *amf.MixedArray {
		return amf.NewArray().Set("object", obj).Set("action", action).
			Set("params", params).
			Set("sess_uid", userID).Set("sess_key", sessKey).Set("counter", counter)
	}

	am := gamedata.ArenaMaps()[0]

	// arena|get_maps {type: DM} -> the assoc is keyed by map id; the arena map must be
	// there, or the Arena tab renders empty.
	gm, _ := postEnvelope(t, url,
		mkReq("arena", "get_maps", amf.NewArray().Set("type", gamedata.MapTypeDM), 2)).
		GetArray(ctrlproto.CmdKey("arena", "get_maps"))
	if gm == nil {
		t.Fatal("no arena|get_maps response")
	}
	mapsAssoc, ok := gm.GetArray("maps")
	if !ok {
		t.Fatal("arena|get_maps has no maps")
	}
	entry, ok := mapsAssoc.GetArray(strconv.Itoa(int(am.ID)))
	if !ok {
		t.Fatalf("arena map %d absent from get_maps{type:DM} -> Arena tab is empty (assoc=%#v)", am.ID, mapsAssoc.Assoc)
	}
	if sc, _ := entry.GetString("scene"); sc != am.Scene {
		t.Errorf("arena map scene = %q, want %s", sc, am.Scene)
	}

	// arena|get_map_type_descs must include the DM blurb.
	descsResp, _ := postEnvelope(t, url,
		mkReq("arena", "get_map_type_descs", amf.NewArray(), 3)).
		GetArray(ctrlproto.CmdKey("arena", "get_map_type_descs"))
	descs, _ := descsResp.GetArray("descs")
	foundDM := false
	for _, e := range descs.Dense {
		d, _ := e.(*amf.MixedArray)
		if d == nil {
			continue
		}
		if typ, _ := d.GetInt("type_id"); typ == gamedata.MapTypeDM {
			foundDM = true
		}
	}
	if !foundDM {
		t.Error("get_map_type_descs missing a DM (Арена) blurb")
	}

	// The arena map joins and launches through fight|* (not hunt|*).
	postEnvelope(t, url, mkReq("fight", "join", amf.NewArray().Set("map_id", am.ID), 4))
	if _, ok := srv.getFightSel(userID); !ok {
		t.Fatal("fight|join for the arena map did not record a selection")
	}
	postEnvelope(t, url, mkReq("fight", "ready", amf.NewArray(), 5))
	pb, ok := srv.Store.TakePendingBattle(userID)
	if !ok {
		t.Fatal("fight|ready for the arena map recorded no PendingBattle")
	}
	if pb.MapID != am.ID || pb.Scene != am.Scene {
		t.Errorf("PendingBattle = {map=%d scene=%q}, want {map=%d scene=%q}", pb.MapID, pb.Scene, am.ID, am.Scene)
	}
}
