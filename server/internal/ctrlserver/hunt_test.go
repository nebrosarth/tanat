package ctrlserver

import (
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"

	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
	"tanatserver/internal/gamedata"
)

// TestHuntMatchmakingFlow drives the full Ctrl side of the hunt launch:
// get_maps_info (real maps) -> hunt|join -> hunt|ready (launch params +
// PendingBattle recorded for the Battle server).
func TestHuntMatchmakingFlow(t *testing.T) {
	srv := New()
	srv.BattleHost = "127.0.0.1"
	srv.BattlePorts = []int32{9339}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	login := postEnvelope(t, url, loginEnvelope("hunter@example.com", "pw", "1.11", "0", "", 1))
	lr, _ := login.GetArray(ctrlproto.CmdKey("user", "login"))
	sessKey, _ := lr.GetString("sess_key")
	userID, _ := lr.GetInt("id")

	mkReq := func(obj, action string, params *amf.MixedArray, counter int32) *amf.MixedArray {
		return amf.NewArray().Set("object", obj).Set("action", action).
			Set("params", params).
			Set("sess_uid", userID).Set("sess_key", sessKey).Set("counter", counter)
	}

	// arena|get_maps_info must list every hunt map (type_id 4) AND every «Штурм»
	// (DOTA, type_id 1) map -- FightHelper.FindMapById needs both so JoinBattle can
	// route by mType.
	mi, _ := postEnvelope(t, url, mkReq("arena", "get_maps_info", amf.NewArray(), 2)).
		GetArray(ctrlproto.CmdKey("arena", "get_maps_info"))
	if mi == nil {
		t.Fatal("no arena|get_maps_info response")
	}
	maps, ok := mi.GetArray("maps_info")
	wantMaps := len(gamedata.HuntMaps()) + len(gamedata.DotaMaps())
	if !ok || len(maps.Dense) != wantMaps {
		t.Fatalf("maps_info: want %d dense entries, got %#v", wantMaps, mi.Assoc)
	}
	first, ok := maps.Dense[0].(*amf.MixedArray)
	if !ok {
		t.Fatalf("maps_info[0] is %T", maps.Dense[0])
	}
	if typ, _ := first.GetInt("type_id"); typ != gamedata.MapTypeHunt {
		t.Errorf("maps_info[0].type_id = %d, want %d", typ, gamedata.MapTypeHunt)
	}
	if sc, _ := first.GetString("scene"); sc != "map_4_0" {
		t.Errorf("maps_info[0].scene = %q, want map_4_0", sc)
	}

	huntMap := gamedata.HuntMaps()[0]

	// hunt|join opens the avatar-select lobby.
	join, _ := postEnvelope(t, url,
		mkReq("hunt", "join", amf.NewArray().Set("map_id", huntMap.ID), 3)).
		GetArray(ctrlproto.CmdKey("hunt", "join"))
	if join == nil {
		t.Fatal("no hunt|join response")
	}
	if status, _ := join.GetInt("status"); status != ctrlproto.StatusOK {
		t.Fatalf("hunt|join status = %d, want 100", status)
	}
	if uid, _ := join.GetInt("user_id"); uid != userID {
		t.Errorf("hunt|join user_id = %d, want %d", uid, userID)
	}
	if _, ok := join.GetArray("deny_for_map"); !ok {
		t.Errorf("hunt|join missing deny_for_map")
	}

	// hunt|ready returns the launch params and records the pending battle.
	avatar := gamedata.Avatars()[2] // Шарли
	ready, _ := postEnvelope(t, url,
		mkReq("hunt", "ready", amf.NewArray().
			Set("map_id", huntMap.ID).Set("avatar_id", avatar.ID), 4)).
		GetArray(ctrlproto.CmdKey("hunt", "ready"))
	if ready == nil {
		t.Fatal("no hunt|ready response")
	}
	params, ok := ready.GetArray("params")
	if !ok {
		t.Fatalf("hunt|ready missing params: %#v", ready.Assoc)
	}
	if ip, _ := params.GetString("ip"); ip != "127.0.0.1" {
		t.Errorf("params.ip = %q", ip)
	}
	if sc, _ := params.GetString("scene"); sc != huntMap.Scene {
		t.Errorf("params.scene = %q, want %s", sc, huntMap.Scene)
	}
	passwd, _ := params.GetString("passwd")
	if passwd == "" {
		t.Fatal("params.passwd empty")
	}
	ports, ok := params.GetArray("port")
	if !ok || len(ports.Dense) != 1 {
		t.Fatalf("params.port malformed: %#v", params.Assoc["port"])
	}

	pb, ok := srv.Store.TakePendingBattle(userID)
	if !ok {
		t.Fatal("no pending battle recorded")
	}
	if pb.MapID != huntMap.ID || pb.AvatarID != avatar.ID || pb.Passwd != passwd || pb.Scene != huntMap.Scene {
		t.Errorf("pending battle = %+v, want map=%d avatar=%d passwd=%s scene=%s",
			pb, huntMap.ID, avatar.ID, passwd, huntMap.Scene)
	}

	// The select-window "random" button sends avatar_id -1; the server must resolve
	// it to a real roster avatar and launch (not reject it).
	rnd, _ := postEnvelope(t, url,
		mkReq("hunt", "ready", amf.NewArray().
			Set("map_id", huntMap.ID).Set("avatar_id", int32(-1)), 6)).
		GetArray(ctrlproto.CmdKey("hunt", "ready"))
	if status, _ := rnd.GetInt("status"); status != ctrlproto.StatusOK {
		t.Fatalf("hunt|ready with random avatar (-1) failed: status %d", status)
	}
	if _, ok := rnd.GetArray("params"); !ok {
		t.Fatal("hunt|ready random: missing launch params")
	}
	rpb, ok := srv.Store.TakePendingBattle(userID)
	if !ok {
		t.Fatal("hunt|ready random: no pending battle recorded")
	}
	if _, ok := gamedata.AvatarByID(rpb.AvatarID); !ok || rpb.AvatarID == -1 {
		t.Errorf("random avatar not resolved: pending avatar id = %d", rpb.AvatarID)
	}

	// Unknown map must fail, not launch.
	bad, _ := postEnvelope(t, url,
		mkReq("hunt", "ready", amf.NewArray().
			Set("map_id", int32(999)).Set("avatar_id", avatar.ID), 5)).
		GetArray(ctrlproto.CmdKey("hunt", "ready"))
	if status, _ := bad.GetInt("status"); status == ctrlproto.StatusOK {
		t.Errorf("hunt|ready with unknown map returned OK")
	}
}

// TestAvatarsAmfAndList checks /xml/avatars.amf carries the roster in
// CtrlAvatarStore.Retrieve's shape and avatar|list unlocks every avatar.
func TestAvatarsAmfAndList(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/xml/avatars.amf")
	if err != nil {
		t.Fatalf("get avatars.amf: %v", err)
	}
	defer resp.Body.Close()
	dec := amf.NewDecoder(resp.Body)
	v, err := dec.DecodeMessage()
	if err != nil {
		t.Fatalf("decode avatars.amf: %v", err)
	}
	root, ok := v.(*amf.MixedArray)
	if !ok {
		t.Fatalf("avatars.amf root is %T", v)
	}
	if len(root.Dense) != len(gamedata.Avatars()) {
		t.Fatalf("avatars.amf: %d entries, want %d", len(root.Dense), len(gamedata.Avatars()))
	}
	a0, ok := root.Dense[0].(*amf.MixedArray)
	if !ok {
		t.Fatalf("avatars.amf[0] is %T", root.Dense[0])
	}
	if id, _ := a0.GetInt("id"); id != gamedata.Avatars()[0].ID {
		t.Errorf("avatars.amf[0].id = %d", id)
	}
	if pf, _ := a0.GetString("prefab"); pf != gamedata.Avatars()[0].Prefab {
		t.Errorf("avatars.amf[0].prefab = %q", pf)
	}
	skills, ok := a0.GetArray("skills")
	if !ok || len(skills.Dense) != 4 {
		t.Fatalf("avatars.amf[0].skills malformed")
	}

	// avatar|list: every roster avatar present and available.
	login := postEnvelope(t, ts.URL+"/entry_point.php",
		loginEnvelope("a@example.com", "pw", "1.11", "0", "", 1))
	lr, _ := login.GetArray(ctrlproto.CmdKey("user", "login"))
	sessKey, _ := lr.GetString("sess_key")
	userID, _ := lr.GetInt("id")

	listResp := postEnvelope(t, ts.URL+"/entry_point.php", amf.NewArray().
		Set("object", "avatar").Set("action", "list").
		Set("params", amf.NewArray()).
		Set("sess_uid", userID).Set("sess_key", sessKey).Set("counter", int32(2)))
	list, _ := listResp.GetArray(ctrlproto.CmdKey("avatar", "list"))
	if list == nil {
		t.Fatal("no avatar|list response")
	}
	avatars, ok := list.GetArray("avatars")
	if !ok {
		t.Fatalf("avatar|list missing avatars")
	}
	for _, a := range gamedata.Avatars() {
		entry, ok := avatars.GetArray(strconv.Itoa(int(a.ID)))
		if !ok {
			t.Errorf("avatar|list missing avatar %d", a.ID)
			continue
		}
		if av, _ := entry.GetBool("available"); !av {
			t.Errorf("avatar %d not available", a.ID)
		}
	}
}

// TestItemsAmfCarriesCtrlPrototypeFields guards the "item never appears in
// any bag" bug: CachedCtrlPrototypeProvider only resolves a non-null
// CtrlPrototype.Article/Desc for an article id if PropertyHolder.
// RetrieveProperties found ALL of PCtrlDesc's five required keys
// (id/title/short/long/icon) for that entry -- a missing one drops the whole
// entry (PArticle included), silently: the lobby bag then skips the item
// (SelfHero.CreateThing's Article==null guard) and the in-battle bag
// NullReferenceExceptions instead, aborting its refresh. Serving this empty
// (the old handleEmptyProto stub) was the actual root cause, not anything in
// the Battle-channel PROTOTYPE_INFO XML.
func TestItemsAmfCarriesCtrlPrototypeFields(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/xml/items.amf")
	if err != nil {
		t.Fatalf("get items.amf: %v", err)
	}
	defer resp.Body.Close()
	dec := amf.NewDecoder(resp.Body)
	v, err := dec.DecodeMessage()
	if err != nil {
		t.Fatalf("decode items.amf: %v", err)
	}
	root, ok := v.(*amf.MixedArray)
	if !ok {
		t.Fatalf("items.amf root is %T", v)
	}
	if len(root.Dense) != len(gamedata.Items()) {
		t.Fatalf("items.amf: %d entries, want %d", len(root.Dense), len(gamedata.Items()))
	}
	for i, want := range gamedata.Items() {
		entry, ok := root.Dense[i].(*amf.MixedArray)
		if !ok {
			t.Fatalf("items.amf[%d] is %T", i, root.Dense[i])
		}
		if id, _ := entry.GetInt("id"); id != want.ArticleID {
			t.Errorf("items.amf[%d].id = %d, want %d", i, id, want.ArticleID)
		}
		// PCtrlDesc's five keys are ALL required -- assert every one is present.
		for _, key := range []string{"title", "short", "long", "icon"} {
			if _, ok := entry.GetString(key); !ok {
				t.Errorf("items.amf[%d] (article %d) missing required PCtrlDesc key %q", i, want.ArticleID, key)
			}
		}
		if icon, _ := entry.GetString("icon"); icon != want.Icon {
			t.Errorf("items.amf[%d].icon = %q, want %q", i, icon, want.Icon)
		}
		// kind_id must NOT default to 0 (ShopGUI.ItemType.QUEST_ITEM): the
		// live client rendered every potion's tooltip with the generic
		// "Предмет, требующийся для задания" quest-item description because
		// this field was missing entirely.
		if kind, ok := entry.GetInt("kind_id"); !ok || kind == 0 {
			t.Errorf("items.amf[%d] (article %d) kind_id = %d, ok=%v -- must not be 0 (QUEST_ITEM)", i, want.ArticleID, kind, ok)
		}
	}
}
