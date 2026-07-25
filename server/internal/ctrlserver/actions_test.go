package ctrlserver

import (
	"strconv"
	"testing"

	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
	"tanatserver/internal/gamedata"
)

// TestChestOpenGrantsAndConsumes: opening a fortune chest via common|action consumes one
// chest from the bag, always credits coins (the response carries the new user|money), and any
// bonus item lands in the persistent bag.
func TestChestOpenGrantsAndConsumes(t *testing.T) {
	srv := New()
	uid, key := newShopHero(t, srv, 1, 10, 0) // 0 gold to start
	chest := gamedata.Chests()[0]
	// Give the hero two of the chest so we can verify the stack decrements (not just removal).
	srv.Store.AddBagItem(uid, chest.ArticleID, 2)

	// The chest is the only bag stack -> its user|bag row id is 1.
	params := amf.NewArray().Set("action_id", chest.ArticleID).Set("artifact_id", int32(1))
	resp := ctrlproto.NewResponse()
	srv.handleCommonAction(shopReq(key, "common", "action", params), resp)

	if st, _ := subResp(t, resp, ctrlproto.CmdKey("common", "action")).GetInt("status"); st != ctrlproto.StatusOK {
		t.Fatalf("action status = %d, want 100", st)
	}
	// One chest consumed (2 -> 1).
	bag := srv.Store.HeroBag(uid)
	var chestCount int32
	for _, b := range bag {
		if b.ArticleID == chest.ArticleID {
			chestCount = b.Count
		}
	}
	if chestCount != 1 {
		t.Fatalf("chest count = %d after one open, want 1", chestCount)
	}
	// Coins were credited and echoed in the response.
	money, _, _ := srv.Store.HeroMoney(uid)
	if money <= 0 {
		t.Fatalf("hero money = %d after chest open, want > 0", money)
	}
	if m, _ := subResp(t, resp, ctrlproto.CmdKey("user", "money")).GetInt("money"); m != money {
		t.Errorf("user|money in response = %d, want %d", m, money)
	}
}

// TestUseRuneStartsGlobalBuff: using a rune via common|action consumes it and starts a timed
// global buff that user|get_global_buffs then reports.
func TestUseRuneStartsGlobalBuff(t *testing.T) {
	srv := New()
	uid, key := newShopHero(t, srv, 1, 10, 0)
	rune := gamedata.Runes()[0]
	srv.Store.AddBagItem(uid, rune.ArticleID, 1)

	params := amf.NewArray().Set("action_id", rune.ArticleID).Set("artifact_id", int32(1))
	resp := ctrlproto.NewResponse()
	srv.handleCommonAction(shopReq(key, "common", "action", params), resp)

	// Rune consumed.
	if bag := srv.Store.HeroBag(uid); len(bag) != 0 {
		t.Fatalf("rune not consumed: %+v", bag)
	}
	// Buff active.
	buffs := srv.Store.HeroActiveBuffs(uid)
	if len(buffs) != 1 || buffs[0].ArticleID != rune.ArticleID {
		t.Fatalf("buff not active: %+v", buffs)
	}
	// get_global_buffs reports it.
	gresp := ctrlproto.NewResponse()
	srv.handleGetGlobalBuffs(shopReq(key, "user", "get_global_buffs", nil), gresp)
	got := subResp(t, gresp, ctrlproto.CmdKey("user", "get_global_buffs"))
	bmap, ok := got.GetArray("buffs")
	if !ok || len(bmap.Assoc) != 1 {
		t.Fatalf("get_global_buffs missing buff: %#v", got.Assoc)
	}
	if _, ok := bmap.Assoc[itoaTest(rune.ArticleID)]; !ok {
		t.Errorf("buff for article %d not in reply: %#v", rune.ArticleID, bmap.Assoc)
	}
}

func itoaTest(i int32) string {
	return strconv.Itoa(int(i))
}

// bagCount returns how many of articleID the hero currently holds (0 if none).
func bagCount(srv *Server, uid, articleID int32) int32 {
	for _, b := range srv.Store.HeroBag(uid) {
		if b.ArticleID == articleID {
			return b.Count
		}
	}
	return 0
}

// twoRunesSameCategory returns two distinct runes that share a category (Stat).
func twoRunesSameCategory(t *testing.T) (gamedata.Rune, gamedata.Rune) {
	t.Helper()
	byStat := map[string][]gamedata.Rune{}
	for _, r := range gamedata.Runes() {
		byStat[r.Stat] = append(byStat[r.Stat], r)
	}
	for _, rs := range byStat {
		if len(rs) >= 2 {
			return rs[0], rs[1]
		}
	}
	t.Fatal("catalog has no two runes sharing a category")
	return gamedata.Rune{}, gamedata.Rune{}
}

// TestUseRuneRefusesSameCategory: while a rune of a category is active a hero cannot activate
// ANOTHER rune of that same category, NOR re-use the same rune (no refresh, no re-consume) --
// the use is refused with nothing consumed. A rune of another category is still allowed.
func TestUseRuneRefusesSameCategory(t *testing.T) {
	srv := New()
	uid, key := newShopHero(t, srv, 1, 10, 0)
	r1, r2 := twoRunesSameCategory(t)
	// r1 already active (as if used earlier / carried over from a prior session).
	srv.Store.AddGlobalBuff(uid, r1.ArticleID, 3600)
	srv.Store.AddBagItem(uid, r2.ArticleID, 1)
	srv.Store.AddBagItem(uid, r1.ArticleID, 1) // also hold a spare r1 to prove re-use is blocked

	// Re-using the SAME rune (r1) is refused: no refresh, nothing consumed.
	before := srv.Store.HeroActiveBuffs(uid)[0].ExpiresUnix
	reuse := amf.NewArray().Set("action_id", r1.ArticleID).Set("artifact_id", int32(2)) // r1 spare is bag row 2
	srv.handleCommonAction(shopReq(key, "common", "action", reuse), ctrlproto.NewResponse())
	if buffs := srv.Store.HeroActiveBuffs(uid); len(buffs) != 1 || buffs[0].ExpiresUnix != before {
		t.Fatalf("re-using the same rune must be refused (no refresh), got %+v", buffs)
	}

	// Using r2 (same category as r1) is refused.
	params := amf.NewArray().Set("action_id", r2.ArticleID).Set("artifact_id", int32(1))
	srv.handleCommonAction(shopReq(key, "common", "action", params), ctrlproto.NewResponse())

	if buffs := srv.Store.HeroActiveBuffs(uid); len(buffs) != 1 || buffs[0].ArticleID != r1.ArticleID {
		t.Fatalf("same-category rune should be refused: want only r1 buff, got %+v", buffs)
	}
	// Both refused runes stay unconsumed in the bag (r2 row 1, spare r1 row 2).
	if r2n := bagCount(srv, uid, r2.ArticleID); r2n != 1 {
		t.Fatalf("refused r2 must stay unconsumed, count=%d", r2n)
	}
	if r1n := bagCount(srv, uid, r1.ArticleID); r1n != 1 {
		t.Fatalf("refused spare r1 must stay unconsumed, count=%d", r1n)
	}

	// A rune of a DIFFERENT category IS allowed (control).
	var rOther gamedata.Rune
	for _, r := range gamedata.Runes() {
		if r.Stat != r1.Stat {
			rOther = r
			break
		}
	}
	if rOther.ArticleID == 0 {
		t.Skip("no rune outside r1's category to test the allow-path")
	}
	srv.Store.AddBagItem(uid, rOther.ArticleID, 1)
	// rOther now sits behind r2 in the bag; resolve its row by re-reading the bag.
	var otherRow int32
	for i, b := range srv.Store.HeroBag(uid) {
		if b.ArticleID == rOther.ArticleID {
			otherRow = int32(i + 1)
		}
	}
	p2 := amf.NewArray().Set("action_id", rOther.ArticleID).Set("artifact_id", otherRow)
	srv.handleCommonAction(shopReq(key, "common", "action", p2), ctrlproto.NewResponse())
	if buffs := srv.Store.HeroActiveBuffs(uid); len(buffs) != 2 {
		t.Fatalf("different-category rune should activate: want 2 buffs, got %+v", buffs)
	}
}

// TestCommonActionOnNonChestIsNoop: pointing common|action at a non-openable bag item (a
// potion) consumes nothing and pays nothing.
func TestCommonActionOnNonChestIsNoop(t *testing.T) {
	srv := New()
	uid, key := newShopHero(t, srv, 1, 10, 0)
	potion := gamedata.Items()[0]
	srv.Store.AddBagItem(uid, potion.ArticleID, 3)

	params := amf.NewArray().Set("action_id", potion.ArticleID).Set("artifact_id", int32(1))
	resp := ctrlproto.NewResponse()
	srv.handleCommonAction(shopReq(key, "common", "action", params), resp)

	// Nothing consumed, no money.
	bag := srv.Store.HeroBag(uid)
	if len(bag) != 1 || bag[0].Count != 3 {
		t.Errorf("potion stack changed: %+v", bag)
	}
	if money, _, _ := srv.Store.HeroMoney(uid); money != 0 {
		t.Errorf("money = %d, want 0 (non-chest action)", money)
	}
}
