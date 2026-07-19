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

// newShopHero registers a hero of the given race/level/gold and returns its user id and
// session key so the Ctrl handlers (which resolve the caller via SessKey) can be driven.
func newShopHero(t *testing.T, srv *Server, race, level, money int32) (int32, string) {
	t.Helper()
	u, key, _ := srv.Store.LoginOrRegister("shop@example.com", "pw")
	h := srv.Store.CreateHero(u, race, false, 0, 0, 0, 0, 0)
	h.Level = level
	h.Money = money
	return u.ID, key
}

func shopReq(sessKey, obj, action string, params *amf.MixedArray) ctrlproto.Request {
	if params == nil {
		params = amf.NewArray()
	}
	return ctrlproto.Request{Object: obj, Action: action, Params: params, SessKey: sessKey}
}

// affordableWearable returns a Human wearable a level-`level` hero with `money` gold can buy.
func affordableWearable(t *testing.T, level, money int32) gamedata.Wearable {
	t.Helper()
	for _, w := range gamedata.WearablesForRace(1) {
		if w.MinHeroLevel <= level && w.Price <= money {
			return w
		}
	}
	t.Fatal("no affordable Human wearable in catalog")
	return gamedata.Wearable{}
}

func subResp(t *testing.T, resp *ctrlproto.Response, key string) *amf.MixedArray {
	t.Helper()
	sub, ok := resp.Root().GetArray(key)
	if !ok {
		t.Fatalf("response missing %q: %#v", key, resp.Root().Assoc)
	}
	return sub
}

// TestStoreListGroupsRaceGear: the city shop lists only the hero's own-race wearables, keyed
// by category, as bare article ids (the client resolves them via items.amf).
func TestStoreListGroupsRaceGear(t *testing.T) {
	srv := New()
	_, key := newShopHero(t, srv, 1, 1, 1000) // Human
	resp := ctrlproto.NewResponse()
	srv.handleStoreList(shopReq(key, "store", "list", nil), resp)

	store := subResp(t, resp, ctrlproto.CmdKey("store", "list"))
	inner, ok := store.GetArray("store")
	if !ok {
		t.Fatal("store|list missing 'store' map")
	}
	if len(inner.Assoc) == 0 {
		t.Fatal("city shop is empty for a Human hero")
	}
	humanArticles := map[int32]bool{}
	for _, w := range gamedata.WearablesForRace(1) {
		humanArticles[w.ArticleID] = true
	}
	total := 0
	for _, catVal := range inner.Assoc {
		cat := catVal.(*amf.MixedArray)
		items, ok := cat.GetArray("items")
		if !ok {
			t.Fatal("category missing 'items'")
		}
		for _, v := range items.Dense {
			total++
			if !humanArticles[coerceInt32(v)] {
				t.Errorf("store lists article %v that is not a Human wearable", v)
			}
		}
	}
	if total != len(gamedata.WearablesForRace(1)) {
		t.Errorf("store lists %d items, want %d Human wearables", total, len(gamedata.WearablesForRace(1)))
	}
}

// TestStoreBuyDebitsAndOwns: a valid buy debits gold, adds the instance to Owned, and the
// response carries the ack plus the new balance.
func TestStoreBuyDebitsAndOwns(t *testing.T) {
	srv := New()
	uid, key := newShopHero(t, srv, 1, 80, 100000)
	w := affordableWearable(t, 80, 100000)

	basket := amf.NewArray().Set("basket", amf.NewArray().Set(strconv.Itoa(int(w.ArticleID)), int32(1)))
	resp := ctrlproto.NewResponse()
	srv.handleStoreBuy(shopReq(key, "store", "buy", basket), resp)

	if st, _ := subResp(t, resp, ctrlproto.CmdKey("store", "buy")).GetInt("status"); st != ctrlproto.StatusOK {
		t.Fatalf("buy status = %d, want 100", st)
	}
	money, _, _ := srv.Store.HeroMoney(uid)
	if money != 100000-w.Price {
		t.Errorf("money = %d after buy, want %d", money, 100000-w.Price)
	}
	if m, _ := subResp(t, resp, ctrlproto.CmdKey("user", "money")).GetInt("money"); m != money {
		t.Errorf("user|money in response = %d, want %d", m, money)
	}
	owned := srv.Store.HeroOwned(uid)
	if len(owned) != 1 || owned[0].ArticleID != w.ArticleID {
		t.Fatalf("Owned = %+v, want one instance of article %d", owned, w.ArticleID)
	}
	if owned[0].ID < 1000 {
		t.Errorf("owned instance id %d is not in the stable (>=1000) range", owned[0].ID)
	}
}

// TestStoreBuyGates: an unaffordable buy, and a buy of another race's gear, are both rejected
// with a non-100 status and no gold spent / nothing owned.
func TestStoreBuyGates(t *testing.T) {
	srv := New()
	uid, key := newShopHero(t, srv, 1, 80, 10) // 10 gold: can't afford anything real

	w := affordableWearable(t, 80, 100000)
	basket := amf.NewArray().Set("basket", amf.NewArray().Set(strconv.Itoa(int(w.ArticleID)), int32(1)))
	resp := ctrlproto.NewResponse()
	srv.handleStoreBuy(shopReq(key, "store", "buy", basket), resp)
	if st, _ := subResp(t, resp, ctrlproto.CmdKey("store", "buy")).GetInt("status"); st == ctrlproto.StatusOK {
		t.Error("unaffordable buy succeeded")
	}
	if money, _, _ := srv.Store.HeroMoney(uid); money != 10 {
		t.Errorf("gold changed on a failed buy: %d", money)
	}
	if len(srv.Store.HeroOwned(uid)) != 0 {
		t.Error("owned an item after a failed buy")
	}

	// An Elf item is not buyable by a Human hero even with money.
	srv.Store.AddHeroMoney(uid, 1000000)
	var elf gamedata.Wearable
	for _, e := range gamedata.WearablesForRace(2) {
		if e.MinHeroLevel <= 80 {
			elf = e
			break
		}
	}
	eb := amf.NewArray().Set("basket", amf.NewArray().Set(strconv.Itoa(int(elf.ArticleID)), int32(1)))
	resp = ctrlproto.NewResponse()
	srv.handleStoreBuy(shopReq(key, "store", "buy", eb), resp)
	if st, _ := subResp(t, resp, ctrlproto.CmdKey("store", "buy")).GetInt("status"); st == ctrlproto.StatusOK {
		t.Error("Human hero bought an Elf item")
	}
}

// buyOne is a helper: buy one affordable wearable and return its owned instance id + article.
func buyOne(t *testing.T, srv *Server, uid int32, key string, level, money int32) (int32, gamedata.Wearable) {
	t.Helper()
	w := affordableWearable(t, level, money)
	basket := amf.NewArray().Set("basket", amf.NewArray().Set(strconv.Itoa(int(w.ArticleID)), int32(1)))
	srv.handleStoreBuy(shopReq(key, "store", "buy", basket), ctrlproto.NewResponse())
	owned := srv.Store.HeroOwned(uid)
	if len(owned) == 0 {
		t.Fatal("buyOne: nothing owned after buy")
	}
	return owned[len(owned)-1].ID, w
}

// TestUserDressMovesToSlot: dressing a valid item moves it out of Owned into Dressed at its
// slot, and the reply carries {slot,item,artikul}.
func TestUserDressMovesToSlot(t *testing.T) {
	srv := New()
	uid, key := newShopHero(t, srv, 1, 80, 100000)
	inst, w := buyOne(t, srv, uid, key, 80, 100000)

	params := amf.NewArray().Set("item", inst).Set("slot", w.SlotBit)
	resp := ctrlproto.NewResponse()
	srv.handleUserDress(shopReq(key, "user", "dress", params), resp)

	sub := subResp(t, resp, ctrlproto.CmdKey("user", "dress"))
	if st, _ := sub.GetInt("status"); st != ctrlproto.StatusOK {
		t.Fatalf("dress status = %d, want 100", st)
	}
	if slot, _ := sub.GetInt("slot"); slot != w.SlotBit {
		t.Errorf("reply slot = %d, want %d", slot, w.SlotBit)
	}
	if it, _ := sub.GetInt("item"); it != inst {
		t.Errorf("reply item = %d, want instance %d", it, inst)
	}
	if art, _ := sub.GetInt("artikul"); art != w.ArticleID {
		t.Errorf("reply artikul = %d, want %d", art, w.ArticleID)
	}
	dressed := srv.Store.HeroDressed(uid)
	if len(dressed) != 1 || dressed[0].ID != inst || dressed[0].Slot != w.SlotBit {
		t.Fatalf("Dressed = %+v, want instance %d at slot %d", dressed, inst, w.SlotBit)
	}
	if len(srv.Store.HeroOwned(uid)) != 0 {
		t.Error("dressed instance is still in Owned")
	}
}

// TestUserDressWrongSlotRejected: the server rejects a dress whose slot bit does not match
// the item's kind (the client never validates this).
func TestUserDressWrongSlotRejected(t *testing.T) {
	srv := New()
	uid, key := newShopHero(t, srv, 1, 80, 100000)
	inst, w := buyOne(t, srv, uid, key, 80, 100000)

	wrong := int32(8192) // Soulshot bit -- no Set item is a soulshot
	if w.SlotBit == wrong {
		wrong = 1
	}
	params := amf.NewArray().Set("item", inst).Set("slot", wrong)
	resp := ctrlproto.NewResponse()
	srv.handleUserDress(shopReq(key, "user", "dress", params), resp)
	if st, _ := subResp(t, resp, ctrlproto.CmdKey("user", "dress")).GetInt("status"); st == ctrlproto.StatusOK {
		t.Error("dress into the wrong slot succeeded")
	}
	if len(srv.Store.HeroDressed(uid)) != 0 {
		t.Error("item was dressed despite the slot mismatch")
	}
}

// TestUserDressSwapReturnsOldItem: dressing a second item into an occupied slot swaps the
// first one back into Owned.
func TestUserDressSwapReturnsOldItem(t *testing.T) {
	srv := New()
	uid, key := newShopHero(t, srv, 1, 80, 1000000)

	// Two different-color items in the SAME slot.
	var a, b gamedata.Wearable
	for _, w := range gamedata.WearablesForRace(1) {
		if w.MinHeroLevel > 80 {
			continue
		}
		if a.ArticleID == 0 {
			a = w
		} else if w.SlotBit == a.SlotBit && w.ArticleID != a.ArticleID {
			b = w
			break
		}
	}
	if b.ArticleID == 0 {
		t.Skip("no two same-slot wearables to swap")
	}
	buy := func(art int32) int32 {
		basket := amf.NewArray().Set("basket", amf.NewArray().Set(strconv.Itoa(int(art)), int32(1)))
		srv.handleStoreBuy(shopReq(key, "store", "buy", basket), ctrlproto.NewResponse())
		o := srv.Store.HeroOwned(uid)
		return o[len(o)-1].ID
	}
	instA := buy(a.ArticleID)
	dress := func(inst, slot int32) {
		p := amf.NewArray().Set("item", inst).Set("slot", slot)
		srv.handleUserDress(shopReq(key, "user", "dress", p), ctrlproto.NewResponse())
	}
	dress(instA, a.SlotBit)
	instB := buy(b.ArticleID)
	dress(instB, b.SlotBit)

	dressed := srv.Store.HeroDressed(uid)
	if len(dressed) != 1 || dressed[0].ID != instB {
		t.Fatalf("after swap Dressed = %+v, want only instance %d", dressed, instB)
	}
	// The displaced first item is back in the bag.
	found := false
	for _, o := range srv.Store.HeroOwned(uid) {
		if o.ID == instA {
			found = true
		}
	}
	if !found {
		t.Error("swapped-out item did not return to Owned")
	}
}

// TestUserUndressReturnsToBag: undressing a slot moves the worn item back into Owned.
func TestUserUndressReturnsToBag(t *testing.T) {
	srv := New()
	uid, key := newShopHero(t, srv, 1, 80, 100000)
	inst, w := buyOne(t, srv, uid, key, 80, 100000)
	dressParams := amf.NewArray().Set("item", inst).Set("slot", w.SlotBit)
	srv.handleUserDress(shopReq(key, "user", "dress", dressParams), ctrlproto.NewResponse())

	resp := ctrlproto.NewResponse()
	srv.handleUserUndress(shopReq(key, "user", "undress", amf.NewArray().Set("slot", w.SlotBit)), resp)
	if st, _ := subResp(t, resp, ctrlproto.CmdKey("user", "undress")).GetInt("status"); st != ctrlproto.StatusOK {
		t.Fatalf("undress status = %d, want 100", st)
	}
	if len(srv.Store.HeroDressed(uid)) != 0 {
		t.Error("slot still worn after undress")
	}
	owned := srv.Store.HeroOwned(uid)
	if len(owned) != 1 || owned[0].ID != inst {
		t.Fatalf("Owned = %+v after undress, want instance %d back", owned, inst)
	}
}

// TestSellRemovesAndRefunds: selling an owned instance refunds its sell price and removes it.
func TestSellRemovesAndRefunds(t *testing.T) {
	srv := New()
	uid, key := newShopHero(t, srv, 1, 80, 100000)
	inst, w := buyOne(t, srv, uid, key, 80, 100000)
	afterBuy, _, _ := srv.Store.HeroMoney(uid)

	basket := amf.NewArray().Set("basket", amf.NewArray().Set(strconv.Itoa(int(inst)), int32(1)))
	resp := ctrlproto.NewResponse()
	srv.handleStoreSell(shopReq(key, "store", "sell", basket), resp)
	if st, _ := subResp(t, resp, ctrlproto.CmdKey("store", "sell")).GetInt("status"); st != ctrlproto.StatusOK {
		t.Fatalf("sell status = %d, want 100", st)
	}
	if money, _, _ := srv.Store.HeroMoney(uid); money != afterBuy+w.SellPrice {
		t.Errorf("money = %d after sell, want %d", money, afterBuy+w.SellPrice)
	}
	if len(srv.Store.HeroOwned(uid)) != 0 {
		t.Error("sold instance still owned")
	}
}

// TestUserBagListsOwnedWearables: the bag response concatenates potion stacks and owned
// wearable instances (the latter with their stable ids).
func TestUserBagListsOwnedWearables(t *testing.T) {
	srv := New()
	uid, key := newShopHero(t, srv, 1, 80, 100000)
	inst, w := buyOne(t, srv, uid, key, 80, 100000)
	srv.Store.AddBagItem(uid, gamedata.Items()[0].ArticleID, 3) // a potion stack too

	resp := ctrlproto.NewResponse()
	srv.handleUserBag(shopReq(key, "user", "bag", nil), resp)
	bag, _ := subResp(t, resp, ctrlproto.CmdKey("user", "bag")).GetArray("bag")
	var sawWearable, sawPotion bool
	for _, v := range bag.Dense {
		row := v.(*amf.MixedArray)
		id, _ := row.GetInt("id")
		art, _ := row.GetInt("artikul_id")
		if id == inst && art == w.ArticleID {
			sawWearable = true
		}
		if art == gamedata.Items()[0].ArticleID {
			sawPotion = true
		}
	}
	if !sawWearable {
		t.Error("owned wearable not listed in user|bag")
	}
	if !sawPotion {
		t.Error("potion stack not listed in user|bag")
	}
}

// TestStoreBuyRejectsHugeCount: a count above the per-article cap is rejected with no mint and
// no spend. Guards the int32 price-overflow free-mint (a big count could wrap the total to a
// small positive value that slips past the affordability check).
func TestStoreBuyRejectsHugeCount(t *testing.T) {
	srv := New()
	uid, key := newShopHero(t, srv, 1, 80, 100000)
	w := affordableWearable(t, 80, 100000)

	basket := amf.NewArray().Set("basket", amf.NewArray().Set(strconv.Itoa(int(w.ArticleID)), int32(100000000)))
	resp := ctrlproto.NewResponse()
	srv.handleStoreBuy(shopReq(key, "store", "buy", basket), resp)

	if st, _ := subResp(t, resp, ctrlproto.CmdKey("store", "buy")).GetInt("status"); st == ctrlproto.StatusOK {
		t.Error("buy with an absurd count succeeded (overflow guard missing)")
	}
	if n := len(srv.Store.HeroOwned(uid)); n != 0 {
		t.Errorf("minted %d items from an over-count buy", n)
	}
	if money, _, _ := srv.Store.HeroMoney(uid); money != 100000 {
		t.Errorf("gold changed on a rejected over-count buy: %d", money)
	}
}

// TestStoreBuyDeliversBagRowOverMPD drives a real buy over HTTP with a live MPD socket and
// asserts the bought instance is actually delivered as a user|update_bag row. Guards the
// double-wrap regression where the rows were nested under a redundant "arguments" key, leaving
// the client's parser iterating an empty array.
func TestStoreBuyDeliversBagRowOverMPD(t *testing.T) {
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
	defer ts.Close()
	url := ts.URL + "/entry_point.php"

	uid, key := newShopHero(t, srv, 1, 80, 100000)
	_, br := dialMPD(t, ln.Addr().String(), uid, key)

	w := affordableWearable(t, 80, 100000)
	basket := amf.NewArray().Set("basket", amf.NewArray().Set(strconv.Itoa(int(w.ArticleID)), int32(1)))
	postEnvelope(t, url, amf.NewArray().
		Set("object", "store").Set("action", "buy").
		Set("params", basket).
		Set("sess_uid", int32(0)).Set("sess_key", key).Set("counter", int32(1)))

	args := readPushArgs(t, br, "user|update_bag")
	if args == nil || len(args.Dense) != 1 {
		got := 0
		if args != nil {
			got = len(args.Dense)
		}
		t.Fatalf("update_bag delivered %d rows, want 1 (double-wrap regression?)", got)
	}
	row, ok := args.Dense[0].(*amf.MixedArray)
	if !ok {
		t.Fatalf("update_bag row is %T", args.Dense[0])
	}
	if art, _ := row.GetInt("artikul_id"); art != w.ArticleID {
		t.Errorf("delivered row artikul_id = %d, want %d", art, w.ArticleID)
	}
	if cnt, _ := row.GetInt("cnt"); cnt != 1 {
		t.Errorf("delivered row cnt = %d, want 1", cnt)
	}
}
