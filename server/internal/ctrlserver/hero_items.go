package ctrlserver

import (
	"math"
	"strconv"

	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// Hero gear ("предметы героев") on the Ctrl channel: the city shop (store|list/buy/sell)
// that sells the baked Set wearables, and the paperdoll (user|dress/undress) that equips
// them for permanent, cross-match stat bonuses. Verified wire facts (see the research
// dossier):
//   - store|list (bare) returns {store:{"<catId>":{weight, items:[articleId...]}}} -- only
//     article ids, grouped by category; the client resolves each id against the PArticle
//     catalog already delivered in /xml/items.amf. The money_d/avatar/type variants select
//     other stores (out of scope here) and get an empty-but-valid map.
//   - store|buy sends {basket:{"<articleId>":cnt}} (buy keys are ARTICLE ids). Success is a
//     bare status:100 ack (SelfHero.OnBought reads nothing); the client does NOT re-fetch,
//     so the server must also push the new balance (user|money) and deliver the bought
//     instances (user|update_bag_mpd).
//   - store|sell sends {basket:{"<instanceId>":cnt}} (sell keys are owned INSTANCE ids).
//   - user|dress sends {item:<instanceId>, slot:<bit>}; reply {slot,item,artikul}. The
//     client adds it to the paperdoll but does NOT drop it from the bag, so the server
//     pushes user|update_bag_mpd (cnt=0 to remove the worn instance, re-adding any item it
//     displaced) and broadcasts user|dress_mpd to co-located players so they see it live.
//   - user|undress sends {slot}; bare ack + a bag re-add + a user|undress_mpd broadcast.
//
// Failure codes are any non-100 int (CtrlPacketValidator treats status==100 as the sole
// success); the client just fires OnBuyFailed/OnDressFailed(code). One generic code suffices.
const errHeroItem int32 = 1

// maxBuyPerArticle / maxBasketEntries bound a single store|buy so a hostile basket cannot
// (a) overflow the int32 price total into a small positive value that slips past the
// affordability gate (a free mint), nor (b) OOM the process by minting a giant Owned slice.
const (
	maxBuyPerArticle = 10
	maxBasketEntries = 64
)

// wearableArticleEntry builds one hero-gear PArticle for /xml/items.amf. Beyond the five
// structurally-required PCtrlDesc keys (id/title/short/long/icon) it carries the fields the
// shop + tooltip read: price/sell_price, price_type (gold), type_id (ItemType.WEARABLE),
// kind_id (ShopGUI.ItemType, drives the tooltip category + the paperdoll slot), min_hero_level
// (the dress gate) and params (the stat tooltip, resolved against the item's baked LongDesc
// placeholders). No tree_* keys -- gear is not a battle-tree item.
func wearableArticleEntry(w gamedata.Wearable) *amf.MixedArray {
	params := amf.NewArray()
	for _, st := range w.Stats {
		params.Add(amf.NewArray().
			Set("skill_id", st.Name).
			Set("impact", st.Impact()).
			Set("value", st.Value))
	}
	return amf.NewArray().
		Set("id", w.ArticleID).
		Set("title", w.NameKey).
		Set("short", "").
		Set("long", w.DescKey).
		Set("icon", w.Icon).
		Set("price", w.Price).
		Set("sell_price", w.SellPrice).
		Set("type_id", int32(0)). // ItemType.WEARABLE
		Set("kind_id", w.KindID). // ShopGUI.ItemType (1..12)
		Set("min_hero_level", w.MinHeroLevel).
		Set("min_ava_level", int32(0)).
		Set("cnt", int32(1)).
		Set("price_type", int32(1)). // virtual money (hero gold)
		Set("sort", w.SlotBit).
		Set("flags", int32(0)).
		Set("params", params)
}

// handleStoreList answers store|list. The bare request is the city gold shop: it returns the
// requesting hero's own-race gear grouped by kind into {store:{"<catId>":{weight,items}}}.
// The diamond (money_d), avatar and battle (type) variants are out of scope and return an
// empty-but-valid store map so their menus don't error.
func (s *Server) handleStoreList(req ctrlproto.Request, resp *ctrlproto.Response) {
	store := amf.NewArray()
	if isCityGearShop(req) {
		if h := s.heroFromSession(req); h != nil {
			byCat := map[int32]*amf.MixedArray{}
			for _, w := range gamedata.WearablesForRace(h.Race) {
				items := byCat[w.KindID]
				if items == nil {
					items = amf.NewArray()
					byCat[w.KindID] = items
				}
				items.Add(w.ArticleID)
			}
			for cat, items := range byCat {
				store.Set(strconv.Itoa(int(cat)), amf.NewArray().
					Set("weight", cat).
					Set("items", items))
			}
		}
	}
	resp.Add("store", "list", amf.NewArray().Set("store", store))
}

// isCityGearShop is true for the bare store|list (ShopType.SIMPLE); the presence of money_d,
// avatar or type selects one of the other stores.
func isCityGearShop(req ctrlproto.Request) bool {
	if req.Params == nil {
		return true
	}
	if b, ok := req.Params.GetBool("money_d"); ok && b {
		return false
	}
	if b, ok := req.Params.GetBool("avatar"); ok && b {
		return false
	}
	if _, ok := req.Params.GetInt("type"); ok {
		return false
	}
	return true
}

// handleStoreBuy serves store|buy for city gear: it validates every basket entry (real
// wearable, right race, hero high enough level), atomically debits gold for the lot, and
// delivers the minted instances. On success it acks store|buy, pushes the new user|money,
// and pushes user|update_bag_mpd carrying the new rows; any validation miss fails the whole
// basket with no spend (the client's own gates should already prevent it).
func (s *Server) handleStoreBuy(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil || u.Hero == nil {
		resp.Fail("store", "buy", errHeroItem)
		return
	}
	level, race, ok := s.Store.HeroLevelRace(u.ID)
	if !ok {
		resp.Fail("store", "buy", errHeroItem)
		return
	}
	basket, ok := req.Params.GetArray("basket")
	if !ok || len(basket.Assoc) == 0 || len(basket.Assoc) > maxBasketEntries {
		resp.Fail("store", "buy", errHeroItem)
		return
	}
	var articles []int32
	var total int64 // int64 so a hostile count can't overflow the running price
	for key, val := range basket.Assoc {
		art := atoi32(key)
		cnt := coerceInt32(val)
		if cnt < 1 || cnt > maxBuyPerArticle {
			resp.Fail("store", "buy", errHeroItem)
			return
		}
		w, ok := gamedata.WearableByArticle(art)
		if !ok || gamedata.WearableRaceCode(w.Race) != race || w.MinHeroLevel > level {
			resp.Fail("store", "buy", errHeroItem)
			return
		}
		for i := int32(0); i < cnt; i++ {
			articles = append(articles, art)
		}
		total += int64(w.Price) * int64(cnt)
	}
	if len(articles) == 0 || total > math.MaxInt32 {
		resp.Fail("store", "buy", errHeroItem)
		return
	}
	money, diamonds, added, ok := s.Store.BuyWearables(u.ID, int32(total), articles)
	if !ok {
		resp.Fail("store", "buy", errHeroItem)
		return
	}
	resp.Ack("store", "buy")
	resp.Add("user", "money", amf.NewArray().Set("money", money).Set("money_d", diamonds))
	rows := amf.NewArray()
	for _, it := range added {
		rows.Add(bagRow(it.ID, it.ArticleID, 1))
	}
	s.pushBagUpdate(u.ID, rows)
}

// handleStoreSell serves store|sell: each basket key is an owned instance id. It refunds the
// item's sell price, removes the instance, and pushes the new balance + a cnt=0 bag removal.
func (s *Server) handleStoreSell(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil || u.Hero == nil {
		resp.Fail("store", "sell", errHeroItem)
		return
	}
	basket, ok := req.Params.GetArray("basket")
	if !ok || len(basket.Assoc) == 0 {
		resp.Fail("store", "sell", errHeroItem)
		return
	}
	owned := s.Store.HeroOwned(u.ID)
	removals := amf.NewArray()
	var money, diamonds int32
	sold := false
	for key := range basket.Assoc {
		instance := atoi32(key)
		art, ok := findOwnedArticle(owned, instance)
		if !ok {
			continue
		}
		refund := int32(0)
		if w, ok := gamedata.WearableByArticle(art); ok {
			refund = w.SellPrice
		}
		m, d, _, ok := s.Store.SellWearable(u.ID, instance, refund)
		if !ok {
			continue
		}
		money, diamonds = m, d
		removals.Add(bagRow(instance, art, 0))
		sold = true
	}
	if !sold {
		resp.Fail("store", "sell", errHeroItem)
		return
	}
	resp.Ack("store", "sell")
	resp.Add("user", "money", amf.NewArray().Set("money", money).Set("money_d", diamonds))
	s.pushBagUpdate(u.ID, removals)
}

// handleUserDress serves user|dress {item, slot}: equip an owned wearable into a paperdoll
// slot. The server is the sole authority for the kind<->slot rule (the client never checks
// it), so it re-validates ownership, that the item's SlotBit is the requested slot, the race
// match and the level gate. On success it replies {slot,item,artikul}, refreshes the bag
// (drop the worn instance, re-add any displaced one) and broadcasts the change to the square.
func (s *Server) handleUserDress(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil || u.Hero == nil {
		resp.Fail("user", "dress", errHeroItem)
		return
	}
	level, race, ok := s.Store.HeroLevelRace(u.ID)
	if !ok {
		resp.Fail("user", "dress", errHeroItem)
		return
	}
	heroID := u.Hero.ID // immutable, safe to read directly
	instance, _ := req.Params.GetInt("item")
	slot, _ := req.Params.GetInt("slot")
	art, ok := findOwnedArticle(s.Store.HeroOwned(u.ID), instance)
	if !ok {
		resp.Fail("user", "dress", errHeroItem)
		return
	}
	w, ok := gamedata.WearableByArticle(art)
	if !ok || w.SlotBit != slot || gamedata.WearableRaceCode(w.Race) != race || w.MinHeroLevel > level {
		resp.Fail("user", "dress", errHeroItem)
		return
	}
	res, ok := s.Store.DressWearable(u.ID, instance, slot)
	if !ok {
		resp.Fail("user", "dress", errHeroItem)
		return
	}
	resp.Add("user", "dress", amf.NewArray().
		Set("slot", slot).
		Set("item", instance).
		Set("artikul", art))
	rows := amf.NewArray()
	rows.Add(bagRow(instance, art, 0)) // worn instance leaves the bag
	if res.Displaced {
		rows.Add(bagRow(res.DisplacedID, res.DisplacedArtcl, 1)) // swapped-out item returns
	}
	s.pushBagUpdate(u.ID, rows)
	s.broadcastToArea(u.ID, "user|dress", amf.NewArray().
		Set("user_id", heroID).
		Set("item", art). // the _mpd broadcast carries the ARTICLE id, not the instance
		Set("slot", slot))
}

// handleUserUndress serves user|undress {slot}: unequip whatever is worn in that slot back
// into the bag. Bare ack, a bag re-add push, and an undress broadcast to the square.
func (s *Server) handleUserUndress(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil || u.Hero == nil {
		resp.Fail("user", "undress", errHeroItem)
		return
	}
	slot, _ := req.Params.GetInt("slot")
	freedID, freedArt, ok := s.Store.UndressWearable(u.ID, slot)
	if !ok {
		resp.Fail("user", "undress", errHeroItem)
		return
	}
	resp.Ack("user", "undress")
	rows := amf.NewArray()
	rows.Add(bagRow(freedID, freedArt, 1))
	s.pushBagUpdate(u.ID, rows)
	s.broadcastToArea(u.ID, "user|undress", amf.NewArray().
		Set("user_id", u.Hero.ID).
		Set("slot", slot).
		Set("item", freedArt))
}

// pushBagUpdate delivers a set of bag deltas to a user's own MPD socket as
// user|update_bag_mpd. rows is the DENSE list of {id,artikul_id,cnt,used} deltas (cnt==0
// removes a row). It is handed STRAIGHT to Hub.Push, which already wraps its payload as the
// "arguments" the client's UpdateBagArgMpdParser iterates over (mArguments.Dense); wrapping
// it again in {arguments:rows} here would hand the parser an assoc whose own .Dense is empty,
// silently dropping every delta. No-op if the user has no live MPD socket (e.g. mid-battle);
// their next user|bag reflects the persisted state anyway.
func (s *Server) pushBagUpdate(userID int32, rows *amf.MixedArray) {
	if s.MPD == nil || len(rows.Dense) == 0 {
		return
	}
	s.MPD.Push(userID, "user|update_bag", rows)
}

// broadcastToArea pushes key+args to every OTHER occupant of the sender's square, so a live
// gear change reaches the players who can see the sender (dress_mpd/undress_mpd).
func (s *Server) broadcastToArea(selfID int32, key string, args *amf.MixedArray) {
	if s.MPD == nil {
		return
	}
	area := s.MPD.AreaOf(selfID)
	if area == 0 {
		return
	}
	for _, uid := range s.MPD.AreaMembers(area) {
		if uid == selfID {
			continue
		}
		s.MPD.Push(uid, key, args)
	}
}

// bagRow builds one user|bag / update_bag row: {id, artikul_id, cnt, used}. used is always 0
// for gear (a consumable concept); cnt=0 signals a removal in an update_bag push.
func bagRow(id, article, cnt int32) *amf.MixedArray {
	return amf.NewArray().
		Set("id", id).
		Set("artikul_id", article).
		Set("cnt", cnt).
		Set("used", int32(0))
}

// findOwnedArticle returns the article id of the owned instance with the given id.
func findOwnedArticle(owned []session.OwnedItem, instanceID int32) (int32, bool) {
	for _, it := range owned {
		if it.ID == instanceID {
			return it.ArticleID, true
		}
	}
	return 0, false
}

// atoi32 parses a decimal string to int32, returning -1 on error (an id that matches nothing).
func atoi32(s string) int32 {
	n, err := strconv.Atoi(s)
	if err != nil {
		return -1
	}
	return int32(n)
}

// coerceInt32 turns an AMF-decoded number (int32/float64/int/int64) into an int32 count.
func coerceInt32(v interface{}) int32 {
	switch n := v.(type) {
	case int32:
		return n
	case float64:
		return int32(n)
	case int:
		return int32(n)
	case int64:
		return int32(n)
	}
	return 0
}
