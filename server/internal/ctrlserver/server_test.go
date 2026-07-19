package ctrlserver

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"

	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
	"tanatserver/internal/session"
)

func postEnvelope(t *testing.T, url string, root *amf.MixedArray) *amf.MixedArray {
	t.Helper()
	var buf bytes.Buffer
	enc := amf.NewEncoder()
	if err := enc.EncodeMessage(&buf, root); err != nil {
		t.Fatalf("encode request: %v", err)
	}
	resp, err := http.Post(url, "application/octet-stream", &buf)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	dec := amf.NewDecoder(resp.Body)
	v, err := dec.DecodeMessage()
	if err != nil {
		t.Fatalf("decode response: %v", err)
	}
	m, ok := v.(*amf.MixedArray)
	if !ok {
		t.Fatalf("response root is %T, want *amf.MixedArray", v)
	}
	return m
}

func loginEnvelope(email, passwd, version, sessUID, sessKey string, counter int32) *amf.MixedArray {
	return amf.NewArray().
		Set("object", "user").
		Set("action", "login").
		Set("params", amf.NewArray().
			Set("email", email).
			Set("passwd", passwd).
			Set("version", version)).
		Set("sess_uid", sessUID).
		Set("sess_key", sessKey).
		Set("counter", counter)
}

func TestLoginThenHeroCreateEndToEnd(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	// 1. Login (first time -> auto-registers).
	resp := postEnvelope(t, url, loginEnvelope("player@example.com", "pw", "3.4.1.27345", "0", "", 1))

	loginResp, ok := resp.GetArray(ctrlproto.CmdKey("user", "login"))
	if !ok {
		t.Fatalf("missing user|login in response: %#v", resp.Assoc)
	}
	status, _ := loginResp.GetInt("status")
	if status != ctrlproto.StatusOK {
		t.Fatalf("login status = %d, want 100", status)
	}
	sessKey, ok := loginResp.GetString("sess_key")
	if !ok || sessKey == "" {
		t.Fatalf("missing sess_key in login response")
	}
	userID, _ := loginResp.GetInt("id")
	if userID == 0 {
		t.Fatalf("expected nonzero user id")
	}

	heroConf, ok := resp.GetArray(ctrlproto.CmdKey("common", "hero_conf"))
	if !ok {
		t.Fatalf("missing common|hero_conf in login response")
	}
	if _, ok := heroConf.GetArray("create"); !ok {
		t.Fatalf("expected hero_conf.create for a fresh account, got %#v", heroConf.Assoc)
	}

	// 2. Create the hero using the freshly issued session.
	createReq := amf.NewArray().
		Set("object", "hero").
		Set("action", "create").
		Set("params", amf.NewArray().
			Set("race", int32(0)).
			Set("gender", int32(1)).
			Set("face", int32(2)).
			Set("hair", int32(3)).
			Set("dist_mark", int32(0)).
			Set("skin_color", int32(0)).
			Set("hair_color", int32(0))).
		Set("sess_uid", int32(userID)).
		Set("sess_key", sessKey).
		Set("counter", int32(2))

	resp2 := postEnvelope(t, url, createReq)
	createResp, ok := resp2.GetArray(ctrlproto.CmdKey("hero", "create"))
	if !ok {
		t.Fatalf("missing hero|create in response: %#v", resp2.Assoc)
	}
	if status, _ := createResp.GetInt("status"); status != ctrlproto.StatusOK {
		t.Fatalf("hero create status = %d, want 100", status)
	}

	// 3. Logging in again with a stale/blank session should reflect the
	// hero now existing.
	resp3 := postEnvelope(t, url, loginEnvelope("player@example.com", "pw", "3.4.1.27345", "0", "", 3))
	heroConf2, ok := resp3.GetArray(ctrlproto.CmdKey("common", "hero_conf"))
	if !ok {
		t.Fatalf("missing common|hero_conf on second login")
	}
	if _, ok := heroConf2.GetArray("load"); !ok {
		t.Fatalf("expected hero_conf.load after hero creation, got %#v", heroConf2.Assoc)
	}
}

// TestLoginWrongPasswordRejected checks that, once an account exists, a login
// with the wrong password replies with the WRONG_PASS (6014) error the client's
// CtrlPacketValidator routes to LoginPerformer.OnLoginFailed -- and does NOT
// bundle hero_conf/area_conf (which would look like a successful login).
func TestLoginWrongPasswordRejected(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	// Register the account with its real password.
	first := postEnvelope(t, url, loginEnvelope("wp@example.com", "right", "3.4.1.27345", "0", "", 1))
	if lr, ok := first.GetArray(ctrlproto.CmdKey("user", "login")); !ok {
		t.Fatal("missing user|login on first login")
	} else if st, _ := lr.GetInt("status"); st != ctrlproto.StatusOK {
		t.Fatalf("first login status = %d, want 100", st)
	}

	// A wrong password must be rejected with 6014 and no hero bundle.
	bad := postEnvelope(t, url, loginEnvelope("wp@example.com", "WRONG", "3.4.1.27345", "0", "", 2))
	lr, ok := bad.GetArray(ctrlproto.CmdKey("user", "login"))
	if !ok {
		t.Fatalf("missing user|login on rejected login: %#v", bad.Assoc)
	}
	if st, _ := lr.GetInt("status"); st != loginWrongPass {
		t.Errorf("rejected login status = %d, want %d (WRONG_PASS)", st, loginWrongPass)
	}
	if errCode, _ := lr.GetInt("error"); errCode != loginWrongPass {
		t.Errorf("rejected login error = %d, want %d", errCode, loginWrongPass)
	}
	if _, ok := bad.GetArray(ctrlproto.CmdKey("common", "hero_conf")); ok {
		t.Error("a rejected login must not bundle hero_conf")
	}

	// The correct password still works afterward.
	good := postEnvelope(t, url, loginEnvelope("wp@example.com", "right", "3.4.1.27345", "0", "", 3))
	if lr, ok := good.GetArray(ctrlproto.CmdKey("user", "login")); !ok {
		t.Fatal("missing user|login on retry")
	} else if st, _ := lr.GetInt("status"); st != ctrlproto.StatusOK {
		t.Errorf("correct-password login status = %d, want 100", st)
	}
}

// TestLoginBannedRejected: a banned account authenticates but is refused with the
// BANNED (6011) code and no hero bundle, even with the correct password.
func TestLoginBannedRejected(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	// Register, then ban via the admin store path.
	first := postEnvelope(t, url, loginEnvelope("ban@example.com", "pw", "3.4.1.27345", "0", "", 1))
	lr, _ := first.GetArray(ctrlproto.CmdKey("user", "login"))
	userID, _ := lr.GetInt("id")
	if !srv.Store.SetBanned(int32(userID), true) {
		t.Fatal("SetBanned failed")
	}

	banned := postEnvelope(t, url, loginEnvelope("ban@example.com", "pw", "3.4.1.27345", "0", "", 2))
	br, ok := banned.GetArray(ctrlproto.CmdKey("user", "login"))
	if !ok {
		t.Fatalf("missing user|login on banned login: %#v", banned.Assoc)
	}
	if st, _ := br.GetInt("status"); st != loginBanned {
		t.Errorf("banned login status = %d, want %d (BANNED)", st, loginBanned)
	}
	if _, ok := banned.GetArray(ctrlproto.CmdKey("common", "hero_conf")); ok {
		t.Error("a banned login must not bundle hero_conf")
	}

	// Unbanning lets them back in.
	srv.Store.SetBanned(int32(userID), false)
	good := postEnvelope(t, url, loginEnvelope("ban@example.com", "pw", "3.4.1.27345", "0", "", 3))
	if gr, ok := good.GetArray(ctrlproto.CmdKey("user", "login")); !ok {
		t.Fatal("missing user|login after unban")
	} else if st, _ := gr.GetInt("status"); st != ctrlproto.StatusOK {
		t.Errorf("post-unban login status = %d, want 100", st)
	}
}

// TestHeroCreateDoesNotOverwriteExisting: a second hero|create for an account
// that already has a hero must NOT wipe its progress (money/level/gear/quests).
func TestHeroCreateDoesNotOverwriteExisting(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	login := postEnvelope(t, url, loginEnvelope("keep@example.com", "pw", "3.4.1.27345", "0", "", 1))
	lr, _ := login.GetArray(ctrlproto.CmdKey("user", "login"))
	sessKey, _ := lr.GetString("sess_key")
	userID, _ := lr.GetInt("id")

	createReq := func(counter int32) *amf.MixedArray {
		return amf.NewArray().
			Set("object", "hero").Set("action", "create").
			Set("params", amf.NewArray().
				Set("race", int32(1)).Set("gender", int32(1)).
				Set("face", int32(0)).Set("hair", int32(0)).
				Set("dist_mark", int32(0)).Set("skin_color", int32(0)).Set("hair_color", int32(0))).
			Set("sess_uid", int32(userID)).Set("sess_key", sessKey).Set("counter", counter)
	}
	postEnvelope(t, url, createReq(2)) // create the hero

	// Simulate earned progress.
	u, ok := srv.Store.ByID(int32(userID))
	if !ok || u.Hero == nil {
		t.Fatal("hero not created")
	}
	u.Hero.Money = 99999
	u.Hero.Level = 12
	srv.Store.Save()

	// A second (spurious) hero|create must be a no-op, not a wipe.
	postEnvelope(t, url, createReq(3))
	u2, _ := srv.Store.ByID(int32(userID))
	if u2.Hero.Money != 99999 || u2.Hero.Level != 12 {
		t.Errorf("second hero|create overwrote progress: money=%d level=%d, want 99999/12", u2.Hero.Money, u2.Hero.Level)
	}
}

// TestAreaConfReturnsBattleCoordinates checks the common|area_conf reply has
// the exact shape ServerDataArgParser reads ({area_conf:{ip,port,scene,passwd,
// area_id}, log}) and that the scene tracks the hero's race (elf -> cs_elf).
func TestAreaConfReturnsBattleCoordinates(t *testing.T) {
	srv := New()
	srv.BattleHost = "127.0.0.1"
	srv.BattlePorts = []int32{9339}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	login := postEnvelope(t, url, loginEnvelope("elf@example.com", "pw", "3.4.1.27345", "0", "", 1))
	loginResp, _ := login.GetArray(ctrlproto.CmdKey("user", "login"))
	sessKey, _ := loginResp.GetString("sess_key")
	userID, _ := loginResp.GetInt("id")

	// Create an ELF hero (race=2) so area_conf should pick cs_elf.
	create := amf.NewArray().
		Set("object", "hero").Set("action", "create").
		Set("params", amf.NewArray().Set("race", int32(2)).Set("gender", int32(0)).
			Set("face", int32(0)).Set("hair", int32(0)).Set("dist_mark", int32(0)).
			Set("skin_color", int32(0)).Set("hair_color", int32(0))).
		Set("sess_uid", userID).Set("sess_key", sessKey).Set("counter", int32(2))
	postEnvelope(t, url, create)

	areaReq := amf.NewArray().
		Set("object", "common").Set("action", "area_conf").
		Set("params", amf.NewArray()).
		Set("sess_uid", userID).Set("sess_key", sessKey).Set("counter", int32(3))
	resp := postEnvelope(t, url, areaReq)

	fields, ok := resp.GetArray(ctrlproto.CmdKey("common", "area_conf"))
	if !ok {
		t.Fatalf("missing common|area_conf in response: %#v", resp.Assoc)
	}
	if status, _ := fields.GetInt("status"); status != ctrlproto.StatusOK {
		t.Fatalf("area_conf status = %d, want 100", status)
	}
	if _, ok := fields.GetInt("log"); !ok {
		t.Fatalf("area_conf missing top-level log field")
	}
	area, ok := fields.GetArray("area_conf")
	if !ok {
		t.Fatalf("missing nested area_conf: %#v", fields.Assoc)
	}
	if ip, _ := area.GetString("ip"); ip != "127.0.0.1" {
		t.Errorf("ip = %q, want 127.0.0.1", ip)
	}
	if scene, _ := area.GetString("scene"); scene != "cs_elf" {
		t.Errorf("scene = %q, want cs_elf (elf hero)", scene)
	}
	ports, ok := area.GetArray("port")
	if !ok || len(ports.Dense) != 1 {
		t.Fatalf("port must be a dense array of one int, got %#v", area.Assoc["port"])
	}
	if p, ok := ports.Dense[0].(int32); !ok || p != 9339 {
		t.Errorf("port[0] = %v, want int32 9339", ports.Dense[0])
	}
	if _, ok := area.GetInt("area_id"); !ok {
		t.Errorf("area_conf missing area_id")
	}
}

// TestLobbyHeroData checks the Phase-2 lobby responses have the shapes the
// client's parsers require (money, full hero info, hero data list).
func TestLobbyHeroData(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	login := postEnvelope(t, url, loginEnvelope("h@example.com", "pw", "1.11", "0", "", 1))
	lr, _ := login.GetArray(ctrlproto.CmdKey("user", "login"))
	sessKey, _ := lr.GetString("sess_key")
	userID, _ := lr.GetInt("id")

	mkReq := func(obj, action string, counter int32) *amf.MixedArray {
		return amf.NewArray().Set("object", obj).Set("action", action).
			Set("params", amf.NewArray()).
			Set("sess_uid", userID).Set("sess_key", sessKey).Set("counter", counter)
	}

	// Create a human hero (race 1).
	postEnvelope(t, url, amf.NewArray().Set("object", "hero").Set("action", "create").
		Set("params", amf.NewArray().Set("race", int32(1)).Set("gender", int32(1)).
			Set("face", int32(0)).Set("hair", int32(0)).Set("dist_mark", int32(0)).
			Set("skin_color", int32(0)).Set("hair_color", int32(0))).
		Set("sess_uid", userID).Set("sess_key", sessKey).Set("counter", int32(2)))

	// user|money
	money, _ := postEnvelope(t, url, mkReq("user", "money", 3)).GetArray(ctrlproto.CmdKey("user", "money"))
	if money == nil {
		t.Fatal("no user|money response")
	}
	if _, ok := money.GetInt("money"); !ok {
		t.Errorf("user|money missing money")
	}
	if _, ok := money.GetInt("money_d"); !ok {
		t.Errorf("user|money missing money_d")
	}

	// user|bag: real persisted consumable stacks, not the old hardcoded empty
	// array (the bug this session fixed).
	srv.Store.AddBagItem(userID, 5000, 3)
	bagResp, _ := postEnvelope(t, url, mkReq("user", "bag", 10)).GetArray(ctrlproto.CmdKey("user", "bag"))
	if bagResp == nil {
		t.Fatal("no user|bag response")
	}
	if _, ok := bagResp.GetInt("user_money"); !ok {
		t.Errorf("user|bag missing user_money")
	}
	bag, ok := bagResp.GetArray("bag")
	if !ok || len(bag.Dense) != 1 {
		t.Fatalf("user|bag.bag = %#v, want a 1-entry dense array", bagResp.Assoc["bag"])
	}
	item, ok := bag.Dense[0].(*amf.MixedArray)
	if !ok {
		t.Fatalf("bag entry is not a MixedArray: %#v", bag.Dense[0])
	}
	if art, _ := item.GetInt("artikul_id"); art != 5000 {
		t.Errorf("bag entry artikul_id = %d, want 5000", art)
	}
	if cnt, _ := item.GetInt("cnt"); cnt != 3 {
		t.Errorf("bag entry cnt = %d, want 3", cnt)
	}

	// user|full_hero_info: visual_data{"<id>":{load,...}}, hero_data{user_id,...}
	fhi, _ := postEnvelope(t, url, mkReq("user", "full_hero_info", 4)).GetArray(ctrlproto.CmdKey("user", "full_hero_info"))
	if fhi == nil {
		t.Fatal("no user|full_hero_info response")
	}
	vis, ok := fhi.GetArray("visual_data")
	if !ok {
		t.Fatalf("full_hero_info missing visual_data")
	}
	key := strconv.Itoa(int(userID))
	entry, ok := vis.GetArray(key)
	if !ok {
		t.Fatalf("visual_data missing hero entry %q: %#v", key, vis.Assoc)
	}
	load, ok := entry.GetArray("load")
	if !ok {
		t.Fatalf("hero entry missing load")
	}
	if race, _ := load.GetInt("race"); race != 1 {
		t.Errorf("load.race = %d, want 1", race)
	}
	if _, ok := entry.GetArray("dressed_items"); !ok {
		t.Errorf("hero entry missing dressed_items")
	}
	hd, ok := fhi.GetArray("hero_data")
	if !ok {
		t.Fatalf("full_hero_info missing hero_data")
	}
	if uid, _ := hd.GetInt("user_id"); uid != userID {
		t.Errorf("hero_data.user_id = %d, want %d", uid, userID)
	}
	if _, ok := hd.GetArray("stats"); !ok {
		t.Errorf("hero_data missing stats")
	}

	// hero|get_data_list -> data{"<id>":{...}}
	gdl, _ := postEnvelope(t, url, mkReq("hero", "get_data_list", 5)).GetArray(ctrlproto.CmdKey("hero", "get_data_list"))
	if gdl == nil {
		t.Fatal("no hero|get_data_list response")
	}
	data, ok := gdl.GetArray("data")
	if !ok {
		t.Fatalf("get_data_list missing data")
	}
	if _, ok := data.GetArray(key); !ok {
		t.Errorf("get_data_list.data missing hero %q", key)
	}
}

// TestHeroDataByIdForOtherPlayer covers the central-square multiplayer appearance
// path: when a client binds ANOTHER occupant's "Hero" avatar it auto-fires
// hero|get_data_list{id:[otherId]} and user|game_info{user_id:otherId} on the Ctrl
// channel, and the server must answer for the requested id (not the requester's own
// hero) or the other player renders bodiless. Here player A asks for player B's data.
func TestHeroDataByIdForOtherPlayer(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	// login + create returns the user's id and session key.
	loginCreate := func(email string, race int32, counterBase int32) (int32, string) {
		lr, _ := postEnvelope(t, url, loginEnvelope(email, "pw", "1.11", "0", "", counterBase)).
			GetArray(ctrlproto.CmdKey("user", "login"))
		sk, _ := lr.GetString("sess_key")
		uid, _ := lr.GetInt("id")
		postEnvelope(t, url, amf.NewArray().Set("object", "hero").Set("action", "create").
			Set("params", amf.NewArray().Set("race", race).Set("gender", int32(1)).
				Set("face", int32(0)).Set("hair", int32(0)).Set("dist_mark", int32(0)).
				Set("skin_color", int32(0)).Set("hair_color", int32(0))).
			Set("sess_uid", uid).Set("sess_key", sk).Set("counter", counterBase+1))
		return uid, sk
	}

	_, skA := loginCreate("a@example.com", 1 /*human*/, 1)
	bID, _ := loginCreate("b@example.com", 2 /*elf*/, 10)

	// A requests B's hero data by id.
	gdl, _ := postEnvelope(t, url, amf.NewArray().Set("object", "hero").Set("action", "get_data_list").
		Set("params", amf.NewArray().Set("id", amf.NewArray().Add(bID))).
		Set("sess_uid", int32(0)).Set("sess_key", skA).Set("counter", int32(20))).
		GetArray(ctrlproto.CmdKey("hero", "get_data_list"))
	if gdl == nil {
		t.Fatal("no hero|get_data_list response")
	}
	data, ok := gdl.GetArray("data")
	if !ok {
		t.Fatal("get_data_list missing data")
	}
	entry, ok := data.GetArray(strconv.Itoa(int(bID)))
	if !ok {
		t.Fatalf("get_data_list.data missing player B's hero %d: %#v", bID, data.Assoc)
	}
	load, _ := entry.GetArray("load")
	if race, _ := load.GetInt("race"); race != 2 {
		t.Errorf("player B's load.race = %d, want 2 (elf) -- served the wrong hero", race)
	}

	// A binds B's avatar AFTER load -> hero|get_data (singular) for B. Must return
	// {load:{id, race,...}} (load key => mPersExists=true) or B renders bodiless.
	gd, _ := postEnvelope(t, url, amf.NewArray().Set("object", "hero").Set("action", "get_data").
		Set("params", amf.NewArray().Set("id", bID)).
		Set("sess_uid", int32(0)).Set("sess_key", skA).Set("counter", int32(22))).
		GetArray(ctrlproto.CmdKey("hero", "get_data"))
	if gd == nil {
		t.Fatal("no hero|get_data response")
	}
	gdLoad, ok := gd.GetArray("load")
	if !ok {
		t.Fatalf("hero|get_data missing load (=> mPersExists=false, no body): %#v", gd.Assoc)
	}
	if hid, _ := gdLoad.GetInt("id"); hid != bID {
		t.Errorf("get_data load.id = %d, want player B %d", hid, bID)
	}
	if race, _ := gdLoad.GetInt("race"); race != 2 {
		t.Errorf("get_data load.race = %d, want 2 (elf)", race)
	}

	// A requests B's game info by user_id (needed for the online roster row too).
	gi, _ := postEnvelope(t, url, amf.NewArray().Set("object", "user").Set("action", "game_info").
		Set("params", amf.NewArray().Set("user_id", bID)).
		Set("sess_uid", int32(0)).Set("sess_key", skA).Set("counter", int32(21))).
		GetArray(ctrlproto.CmdKey("user", "game_info"))
	if gi == nil {
		t.Fatal("no user|game_info response")
	}
	if uid, _ := gi.GetInt("user_id"); uid != bID {
		t.Errorf("game_info.user_id = %d, want player B %d", uid, bID)
	}
}

func TestPingReturnsValidEmptyArray(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	ping := amf.NewArray().Set("sess_uid", "1").Set("sess_key", "abc").Set("counter", int32(5))
	resp := postEnvelope(t, ts.URL+"/entry_point.php", ping)
	if len(resp.Assoc) != 0 {
		t.Fatalf("expected empty ack for ping, got %#v", resp.Assoc)
	}
}

func TestUnknownCommandGetsGenericAck(t *testing.T) {
	srv := New()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	req := amf.NewArray().
		Set("object", "store").
		Set("action", "list").
		Set("params", amf.NewArray()).
		Set("sess_uid", "1").
		Set("sess_key", "abc").
		Set("counter", int32(1))
	resp := postEnvelope(t, ts.URL+"/entry_point.php", req)
	sub, ok := resp.GetArray(ctrlproto.CmdKey("store", "list"))
	if !ok {
		t.Fatalf("missing generic ack for store|list: %#v", resp.Assoc)
	}
	if status, _ := sub.GetInt("status"); status != ctrlproto.StatusOK {
		t.Fatalf("status = %d, want 100", status)
	}
}

// TestSceneForAreaCrossCity pins the portal-travel fix: the central-square scene is
// chosen by the requested AREA (client Location enum: CS_HUMAN=367, CS_ELF=368), not
// the hero's race -- so an elf using the human-city portal (area_id 367) loads
// cs_human instead of respawning in cs_elf. Also checks the store round-trips the area.
func TestSceneForAreaCrossCity(t *testing.T) {
	if got := sceneForArea(areaCSHuman); got != "cs_human" {
		t.Errorf("sceneForArea(367) = %q, want cs_human", got)
	}
	if got := sceneForArea(areaCSElf); got != "cs_elf" {
		t.Errorf("sceneForArea(368) = %q, want cs_elf", got)
	}
	if got := sceneForArea(0); got != "cs_human" {
		t.Errorf("sceneForArea(0) = %q, want cs_human (default)", got)
	}
	st := session.NewStore()
	st.SetLobbyArea(42, areaCSHuman)
	if got := st.LobbyArea(42); got != areaCSHuman {
		t.Errorf("LobbyArea round-trip = %d, want %d", got, areaCSHuman)
	}
	if got := st.LobbyArea(99); got != 0 {
		t.Errorf("LobbyArea for unknown user = %d, want 0", got)
	}
}
