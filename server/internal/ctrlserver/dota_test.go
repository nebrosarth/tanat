package ctrlserver

import (
	"net/http/httptest"
	"strconv"
	"testing"

	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
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

// TestDotaMatchSplitsTeams: a «Штурм» match of two players is true PvP -- the matchmaker
// puts them on OPPOSITE sides (teams 1 and 2), and the assigned side rides through to the
// launch (PendingBattle.Team), so the pre-battle roster and the battle agree.
func TestDotaMatchSplitsTeams(t *testing.T) {
	srv := New()
	srv.BattleHost = "127.0.0.1"
	srv.BattlePorts = []int32{9339}
	srv.SetDotaMatchSize(2)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"
	dm := gamedata.DotaMaps()[0]

	// Log in and queue one player; return (uid, sessKey).
	joinAs := func(email string, counter int32) (int32, string) {
		login := postEnvelope(t, url, loginEnvelope(email, "pw", "1.11", "0", "", counter))
		lr, _ := login.GetArray(ctrlproto.CmdKey("user", "login"))
		sessKey, _ := lr.GetString("sess_key")
		uid, _ := lr.GetInt("id")
		req := amf.NewArray().Set("object", "fight").Set("action", "join").
			Set("params", amf.NewArray().Set("map_id", dm.ID)).
			Set("sess_uid", uid).Set("sess_key", sessKey).Set("counter", counter+1)
		postEnvelope(t, url, req)
		return uid, sessKey
	}

	a, keyA := joinAs("splitA@example.com", 1)
	b, _ := joinAs("splitB@example.com", 10)

	selA, okA := srv.getFightSel(a)
	selB, okB := srv.getFightSel(b)
	if !okA || !okB {
		t.Fatalf("match did not form for both players: A=%v B=%v", okA, okB)
	}
	// The two must be on OPPOSING sides: {1,2}, not both the same team.
	teams := map[int32]bool{selA.team: true, selB.team: true}
	if !teams[1] || !teams[2] || selA.team == selB.team {
		t.Fatalf("match not split into two sides: A.team=%d B.team=%d (want one 1 and one 2)", selA.team, selB.team)
	}

	// The assigned side survives to the launch handoff.
	ready := amf.NewArray().Set("object", "fight").Set("action", "ready").
		Set("params", amf.NewArray()).
		Set("sess_uid", a).Set("sess_key", keyA).Set("counter", 3)
	postEnvelope(t, url, ready)
	pb, ok := srv.Store.TakePendingBattle(a)
	if !ok {
		t.Fatal("fight|ready recorded no PendingBattle")
	}
	if pb.Team != selA.team {
		t.Fatalf("PendingBattle.Team = %d, want the matchmaker-assigned side %d", pb.Team, selA.team)
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

// TestFightJoinRejectsCastleOnlyMap: map_6_0 («Битва за замок») must NOT be enterable
// through the general fight|join queue -- only through castle|ready's scheduled
// window (see gamedata.DotaMap.CastleOnly) -- because that queue skips the castle's
// team/ownership/reward wiring entirely.
func TestFightJoinRejectsCastleOnlyMap(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	login := postEnvelope(t, url, loginEnvelope("sneak@example.com", "pw", "1.11", "0", "", 1))
	lr, _ := login.GetArray(ctrlproto.CmdKey("user", "login"))
	sessKey, _ := lr.GetString("sess_key")
	userID, _ := lr.GetInt("id")

	var castleMapID int32
	for _, m := range gamedata.DotaMaps() {
		if m.CastleOnly {
			castleMapID = m.ID
		}
	}
	if castleMapID == 0 {
		t.Fatal("no CastleOnly DotaMap seeded -- test premise broken")
	}

	req := amf.NewArray().Set("object", "fight").Set("action", "join").
		Set("params", amf.NewArray().Set("map_id", castleMapID)).
		Set("sess_uid", userID).Set("sess_key", sessKey).Set("counter", 2)
	resp, _ := postEnvelope(t, url, req).GetArray(ctrlproto.CmdKey("fight", "join"))
	if resp == nil {
		t.Fatal("no fight|join response")
	}
	if errc, ok := resp.GetInt("error"); !ok || errc == 0 {
		t.Errorf("fight|join on the castle-only map = %#v, want an error", resp.Assoc)
	}
	if _, ok := srv.getFightSel(userID); ok {
		t.Error("fight|join on the castle-only map recorded a selection -- it should have been rejected outright")
	}
}

// TestCastleTestMapEnvVarExposesItUnderFightTab: with TANAT_CASTLE_TEST_MAP=1 set,
// map_6_0 shows up in arena|get_maps_info under the «Штурм» type and fight|join/
// fight|ready launch onto it directly -- a manual-testing shortcut for exercising the
// siege mechanics without running the whole castle draft/schedule flow.
func TestCastleTestMapEnvVarExposesItUnderFightTab(t *testing.T) {
	t.Setenv("TANAT_CASTLE_TEST_MAP", "1")

	srv := New()
	srv.BattleHost = "127.0.0.1"
	srv.BattlePorts = []int32{9339}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	login := postEnvelope(t, url, loginEnvelope("tester@example.com", "pw", "1.11", "0", "", 1))
	lr, _ := login.GetArray(ctrlproto.CmdKey("user", "login"))
	sessKey, _ := lr.GetString("sess_key")
	userID, _ := lr.GetInt("id")
	mkReq := func(obj, action string, params *amf.MixedArray, counter int32) *amf.MixedArray {
		return amf.NewArray().Set("object", obj).Set("action", action).
			Set("params", params).
			Set("sess_uid", userID).Set("sess_key", sessKey).Set("counter", counter)
	}

	var castleMap gamedata.DotaMap
	for _, m := range gamedata.DotaMaps() {
		if m.CastleOnly {
			castleMap = m
		}
	}
	if castleMap.ID == 0 {
		t.Fatal("no CastleOnly DotaMap seeded -- test premise broken")
	}

	mi, _ := postEnvelope(t, url, mkReq("arena", "get_maps_info", amf.NewArray(), 2)).
		GetArray(ctrlproto.CmdKey("arena", "get_maps_info"))
	maps, _ := mi.GetArray("maps_info")
	var found *amf.MixedArray
	for _, e := range maps.Dense {
		m, _ := e.(*amf.MixedArray)
		if m == nil {
			continue
		}
		if id, _ := m.GetInt("id"); id == castleMap.ID {
			found = m
		}
	}
	if found == nil {
		t.Fatal("castle map not listed in arena|get_maps_info under TANAT_CASTLE_TEST_MAP=1")
	}
	if typ, _ := found.GetInt("type_id"); typ != gamedata.MapTypeDota {
		t.Errorf("castle map type_id = %d, want %d (Штурм tab)", typ, gamedata.MapTypeDota)
	}
	if name, _ := found.GetString("name"); name != "Castle_War_Text" {
		t.Errorf("castle map name = %q, want a real locale key", name)
	}

	postEnvelope(t, url, mkReq("fight", "join", amf.NewArray().Set("map_id", castleMap.ID), 3))
	if _, ok := srv.getFightSel(userID); !ok {
		t.Fatal("fight|join on the exposed castle map did not record a selection")
	}
	postEnvelope(t, url, mkReq("fight", "ready", amf.NewArray(), 4))
	pb, ok := srv.Store.TakePendingBattle(userID)
	if !ok {
		t.Fatal("fight|ready on the exposed castle map recorded no PendingBattle")
	}
	if pb.MapID != castleMap.ID || pb.Scene != castleMap.Scene {
		t.Errorf("PendingBattle = {map=%d scene=%q}, want {map=%d scene=%q}",
			pb.MapID, pb.Scene, castleMap.ID, castleMap.Scene)
	}
	if pb.CastleID != 0 {
		t.Error("a fight|* launch onto the castle map must NOT carry a CastleID -- it bypasses the castle flow's ownership/reward wiring by design (test path only)")
	}
}

// TestFightLogServesPublishedScoreboard: once battleserver has published a match's
// scoreboard (session.Store.SetFightLog -- see battleserver/rating.go), fight|log{fight_id}
// must actually return it as {log:{heroes:{...}}}. Before this handler existed, fight|log
// fell through to the generic UNHANDLED ack with no "log" key at all, which is why the
// client's end-of-match table came up empty no matter what battleserver published.
func TestFightLogServesPublishedScoreboard(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	login := postEnvelope(t, url, loginEnvelope("scoreboard@example.com", "pw", "1.11", "0", "", 1))
	lr, _ := login.GetArray(ctrlproto.CmdKey("user", "login"))
	sessKey, _ := lr.GetString("sess_key")
	userID, _ := lr.GetInt("id")

	srv.Store.SetFightLog(4242, map[int32]session.FightLogEntry{
		1000: {AvatarID: 1000, Nick: "Hero", Team: 1, Kills: 5, Assists: 2, Deaths: 1,
			Level: 6, Money: 300, OldRating: 1000, NewRating: 1024},
	})

	req := amf.NewArray().Set("object", "fight").Set("action", "log").
		Set("params", amf.NewArray().Set("fight_id", int32(4242))).
		Set("sess_uid", userID).Set("sess_key", sessKey).Set("counter", int32(2))
	resp, _ := postEnvelope(t, url, req).GetArray(ctrlproto.CmdKey("fight", "log"))
	if resp == nil {
		t.Fatal("no fight|log response")
	}
	logArr, ok := resp.GetArray("log")
	if !ok {
		t.Fatal("fight|log response has no \"log\" key")
	}
	heroes, ok := logArr.GetArray("heroes")
	if !ok {
		t.Fatal("fight|log's \"log\" has no \"heroes\" key")
	}
	row, ok := heroes.GetArray("1000")
	if !ok {
		t.Fatal("fight|log heroes has no entry for avatar 1000 -- the table would render empty")
	}
	if kills, _ := row.GetInt("AvatarKills"); kills != 5 {
		t.Errorf("AvatarKills = %d, want 5", kills)
	}
	if assists, _ := row.GetInt("Assists"); assists != 2 {
		t.Errorf("Assists = %d, want 2", assists)
	}
	if deaths, _ := row.GetInt("Deaths"); deaths != 1 {
		t.Errorf("Deaths = %d, want 1", deaths)
	}
	if rating, _ := row.GetInt("rating"); rating != 1024 {
		t.Errorf("rating = %d, want 1024", rating)
	}
	if old, _ := row.GetInt("old_rating"); old != 1000 {
		t.Errorf("old_rating = %d, want 1000", old)
	}
	if nick, _ := row.GetString("nick"); nick != "Hero" {
		t.Errorf("nick = %q, want %q", nick, "Hero")
	}
}

// TestFightLogUnknownFightIDIsEmptyNotError: a stale/expired/never-published fight_id
// (server restarted mid-match, or a stray duplicate request) must answer with an empty
// scoreboard rather than failing the request.
func TestFightLogUnknownFightIDIsEmptyNotError(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	login := postEnvelope(t, url, loginEnvelope("noscoreboard@example.com", "pw", "1.11", "0", "", 1))
	lr, _ := login.GetArray(ctrlproto.CmdKey("user", "login"))
	sessKey, _ := lr.GetString("sess_key")
	userID, _ := lr.GetInt("id")

	req := amf.NewArray().Set("object", "fight").Set("action", "log").
		Set("params", amf.NewArray().Set("fight_id", int32(99999))).
		Set("sess_uid", userID).Set("sess_key", sessKey).Set("counter", int32(2))
	resp, _ := postEnvelope(t, url, req).GetArray(ctrlproto.CmdKey("fight", "log"))
	if resp == nil {
		t.Fatal("no fight|log response")
	}
	logArr, ok := resp.GetArray("log")
	if !ok {
		t.Fatal("fight|log response has no \"log\" key")
	}
	if _, ok := logArr.GetArray("heroes"); !ok {
		t.Fatal("fight|log's \"log\" has no \"heroes\" key even when empty")
	}
}
