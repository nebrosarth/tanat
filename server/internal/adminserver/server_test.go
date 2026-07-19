package adminserver

import (
	"encoding/json"
	"net/http"
	"net/http/cookiejar"
	"net/http/httptest"
	"strings"
	"testing"

	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

func newTestServer(t *testing.T) (*httptest.Server, *session.Store, int32) {
	t.Helper()
	store := session.NewStore() // in-memory (no DB); SetMeta is a no-op
	u, _, ok := store.LoginOrRegister("p@x", "pw")
	if !ok {
		t.Fatal("register failed")
	}
	store.CreateHero(u, 1, false, 0, 0, 0, 0, 0)
	ts := httptest.NewServer(New(store, "secret").Handler())
	t.Cleanup(ts.Close)
	return ts, store, u.ID
}

// login returns an authenticated client (cookie jar carries the session) and its
// CSRF token.
func login(t *testing.T, ts *httptest.Server, password string) (*http.Client, string) {
	t.Helper()
	jar, _ := cookiejar.New(nil)
	c := &http.Client{Jar: jar}
	res, err := c.Post(ts.URL+"/api/login", "application/json", strings.NewReader(`{"password":"`+password+`"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		t.Fatalf("login status = %d", res.StatusCode)
	}
	var out struct {
		CSRF string `json:"csrf"`
	}
	json.NewDecoder(res.Body).Decode(&out)
	if out.CSRF == "" {
		t.Fatal("no csrf returned")
	}
	return c, out.CSRF
}

func post(t *testing.T, c *http.Client, url, csrf, body string) *http.Response {
	t.Helper()
	req, _ := http.NewRequest("POST", url, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if csrf != "" {
		req.Header.Set(csrfHeader, csrf)
	}
	res, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return res
}

func restoreGamedata(t *testing.T) {
	t.Helper()
	orig := gamedata.Snapshot()
	t.Cleanup(func() { gamedata.Apply(orig) })
}

func TestUnauthedStateRejected(t *testing.T) {
	ts, _, _ := newTestServer(t)
	res, err := http.Get(ts.URL + "/api/state")
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != 401 {
		t.Fatalf("want 401, got %d", res.StatusCode)
	}
}

func TestWrongPasswordRejected(t *testing.T) {
	ts, _, _ := newTestServer(t)
	res, err := http.Post(ts.URL+"/api/login", "application/json", strings.NewReader(`{"password":"nope"}`))
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != 401 {
		t.Fatalf("want 401, got %d", res.StatusCode)
	}
}

func TestLoginServesState(t *testing.T) {
	ts, _, _ := newTestServer(t)
	c, _ := login(t, ts, "secret")
	res, err := c.Get(ts.URL + "/api/state")
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		t.Fatalf("state status = %d", res.StatusCode)
	}
	var st struct {
		CSRF     string            `json:"csrf"`
		Settings gamedata.Settings `json:"settings"`
		Avatars  []any             `json:"avatars"`
		Mobs     []any             `json:"mobs"`
	}
	if err := json.NewDecoder(res.Body).Decode(&st); err != nil {
		t.Fatal(err)
	}
	if st.CSRF == "" || len(st.Avatars) == 0 || len(st.Mobs) == 0 {
		t.Fatalf("state incomplete: csrf=%q avatars=%d mobs=%d", st.CSRF, len(st.Avatars), len(st.Mobs))
	}
}

func TestCSRFEnforcedOnMutation(t *testing.T) {
	ts, store, id := newTestServer(t)
	c, csrf := login(t, ts, "secret")

	// Without the CSRF header: rejected, store unchanged.
	res := post(t, c, ts.URL+"/api/player/money", "", `{"id":`+itoa(id)+`,"money":999,"diamonds":9}`)
	res.Body.Close()
	if res.StatusCode != 403 {
		t.Fatalf("missing CSRF: want 403, got %d", res.StatusCode)
	}
	if m, _, _ := store.HeroMoney(id); m == 999 {
		t.Fatal("money changed without CSRF")
	}

	// With the CSRF header: accepted, store updated.
	res = post(t, c, ts.URL+"/api/player/money", csrf, `{"id":`+itoa(id)+`,"money":999,"diamonds":9}`)
	res.Body.Close()
	if res.StatusCode != 200 {
		t.Fatalf("with CSRF: want 200, got %d", res.StatusCode)
	}
	if m, d, _ := store.HeroMoney(id); m != 999 || d != 9 {
		t.Fatalf("money not set: %d,%d", m, d)
	}
}

func TestSettingsUpdateAffectsGamedata(t *testing.T) {
	restoreGamedata(t)
	ts, _, _ := newTestServer(t)
	c, csrf := login(t, ts, "secret")
	res := post(t, c, ts.URL+"/api/settings", csrf,
		`{"fog_of_war":false,"mob_hp_per_level":0.2,"mob_dmg_per_level":0.1,"mob_xp_per_level":0.4,"mob_coin_per_level":0.15,"hero_power_per_level":0.06,"hero_health_per_level":0.06,"xp_multiplier":2,"coin_multiplier":1,"new_hero_money":1000,"new_hero_diamond":100}`)
	res.Body.Close()
	if res.StatusCode != 200 {
		t.Fatalf("settings status = %d", res.StatusCode)
	}
	if gamedata.FogOfWar() {
		t.Error("fog should be disabled")
	}
	if gamedata.Snapshot().XPMultiplier != 2 {
		t.Error("xp multiplier not applied")
	}
}

func TestAvatarOverrideEndpoint(t *testing.T) {
	restoreGamedata(t)
	ts, _, _ := newTestServer(t)
	c, csrf := login(t, ts, "secret")
	res := post(t, c, ts.URL+"/api/avatar", csrf, `{"id":8,"override":{"Health":9999}}`)
	res.Body.Close()
	if res.StatusCode != 200 {
		t.Fatalf("avatar override status = %d", res.StatusCode)
	}
	a, _ := gamedata.AvatarByID(8)
	if a.Health != 9999 {
		t.Errorf("avatar override not applied: Health=%g", a.Health)
	}
	// Clearing (empty override) restores authored stats.
	res = post(t, c, ts.URL+"/api/avatar", csrf, `{"id":8,"override":{}}`)
	res.Body.Close()
	a, _ = gamedata.AvatarByID(8)
	if a.Health == 9999 {
		t.Error("override not cleared")
	}
}

func TestDeletePlayerEndpoint(t *testing.T) {
	ts, store, id := newTestServer(t)
	c, csrf := login(t, ts, "secret")
	res := post(t, c, ts.URL+"/api/player/delete", csrf, `{"id":`+itoa(id)+`}`)
	res.Body.Close()
	if res.StatusCode != 200 {
		t.Fatalf("delete status = %d", res.StatusCode)
	}
	if len(store.ListPlayers()) != 0 {
		t.Fatal("player not deleted")
	}
}

func TestQuestAndGrantEndpoints(t *testing.T) {
	ts, store, id := newTestServer(t)
	c, csrf := login(t, ts, "secret")

	// A real catalog quest id (the first authored quest).
	quests := gamedata.Quests()
	if len(quests) == 0 {
		t.Fatal("no quests in catalog")
	}
	qid := quests[0].ID
	qcount := quests[0].Count

	// Catalog endpoint exposes quests + both item families.
	getRes, err := c.Get(ts.URL + "/api/catalog")
	if err != nil {
		t.Fatal(err)
	}
	var cat struct {
		Quests    []map[string]any `json:"quests"`
		Potions   []map[string]any `json:"potions"`
		Wearables []map[string]any `json:"wearables"`
	}
	json.NewDecoder(getRes.Body).Decode(&cat)
	getRes.Body.Close()
	if len(cat.Quests) == 0 || len(cat.Potions) == 0 || len(cat.Wearables) == 0 {
		t.Fatalf("catalog incomplete: q=%d p=%d w=%d", len(cat.Quests), len(cat.Potions), len(cat.Wearables))
	}

	// Set quest progress to DONE at the objective count.
	res := post(t, c, ts.URL+"/api/player/quest", csrf,
		`{"id":`+itoa(id)+`,"quest_id":`+itoa(qid)+`,"status":2,"progress":9999}`)
	res.Body.Close()
	if res.StatusCode != 200 {
		t.Fatalf("set quest status = %d", res.StatusCode)
	}
	qs, _ := store.AdminHeroQuests(id)
	if len(qs) != 1 || qs[0].QuestID != qid || qs[0].Status != 2 || qs[0].Progress != qcount {
		t.Fatalf("quest not set/clamped: %+v (count=%d)", qs, qcount)
	}

	// Unknown quest id is rejected.
	res = post(t, c, ts.URL+"/api/player/quest", csrf, `{"id":`+itoa(id)+`,"quest_id":999999,"status":1,"progress":0}`)
	if res.StatusCode != 400 {
		t.Errorf("unknown quest: want 400, got %d", res.StatusCode)
	}
	res.Body.Close()

	// Remove it.
	res = post(t, c, ts.URL+"/api/player/quest", csrf, `{"id":`+itoa(id)+`,"quest_id":`+itoa(qid)+`,"remove":true}`)
	res.Body.Close()
	if qs, _ := store.AdminHeroQuests(id); len(qs) != 0 {
		t.Fatalf("quest not removed: %+v", qs)
	}

	// Grant a potion (auto-detected by article range) and a wearable.
	pArt := int32(cat.Potions[0]["article"].(float64))
	wArt := int32(cat.Wearables[0]["article"].(float64))
	res = post(t, c, ts.URL+"/api/player/grant", csrf, `{"id":`+itoa(id)+`,"article_id":`+itoa(pArt)+`,"count":5}`)
	res.Body.Close()
	if res.StatusCode != 200 {
		t.Fatalf("grant potion status = %d", res.StatusCode)
	}
	res = post(t, c, ts.URL+"/api/player/grant", csrf, `{"id":`+itoa(id)+`,"article_id":`+itoa(wArt)+`,"count":2}`)
	res.Body.Close()
	if res.StatusCode != 200 {
		t.Fatalf("grant wearable status = %d", res.StatusCode)
	}
	if bag := store.HeroBag(id); len(bag) != 1 || bag[0].Count != 5 {
		t.Errorf("potion not granted: %+v", bag)
	}
	if owned := store.HeroOwned(id); len(owned) != 2 {
		t.Errorf("wearable not granted: %d", len(owned))
	}

	// Unknown article rejected.
	res = post(t, c, ts.URL+"/api/player/grant", csrf, `{"id":`+itoa(id)+`,"article_id":12345,"count":1}`)
	if res.StatusCode != 400 {
		t.Errorf("unknown article: want 400, got %d", res.StatusCode)
	}
	res.Body.Close()

	// Inventory read reflects the grants.
	invRes, err := c.Get(ts.URL + "/api/player/inventory?id=" + itoa(id))
	if err != nil {
		t.Fatal(err)
	}
	var inv struct {
		Bag   []map[string]any `json:"bag"`
		Owned []map[string]any `json:"owned"`
	}
	json.NewDecoder(invRes.Body).Decode(&inv)
	invRes.Body.Close()
	if len(inv.Bag) != 1 || len(inv.Owned) != 2 {
		t.Errorf("inventory read wrong: bag=%d owned=%d", len(inv.Bag), len(inv.Owned))
	}
}

func itoa(i int32) string {
	// small helper to avoid importing strconv just for the test bodies
	if i == 0 {
		return "0"
	}
	neg := i < 0
	if neg {
		i = -i
	}
	var b [12]byte
	p := len(b)
	for i > 0 {
		p--
		b[p] = byte('0' + i%10)
		i /= 10
	}
	if neg {
		p--
		b[p] = '-'
	}
	return string(b[p:])
}
