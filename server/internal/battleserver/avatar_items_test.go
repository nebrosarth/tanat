package battleserver

import (
	"strconv"
	"strings"
	"testing"

	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
)

// buyPkt builds the Battle BUY the item tree sends. For a tree buy the wire
// itemId is the battle proto id, which for us equals the article id.
func buyPkt(article int32) battleproto.Packet {
	return battleproto.Packet{
		Cmd:       battleproto.CmdBuy,
		Args:      amf.NewArray().Set("shopId", int32(0)).Set("sellerId", int32(0)).Set("itemId", article).Set("count", int32(1)),
		RequestID: 1,
	}
}

// attackTreeRootAndChild returns an ATTACK-tree root (no parents) and one of its
// direct children, for the parent-gate tests.
func attackTreeRootAndChild(t *testing.T) (root, child gamedata.AvatarItem) {
	t.Helper()
	for _, it := range gamedata.AvatarItems() {
		if it.TreeID == gamedata.AvatarTreeAttack && len(it.Parents) == 0 {
			root = it
		}
	}
	if root.ArticleID == 0 {
		t.Fatal("no ATTACK-tree root item")
	}
	for _, it := range gamedata.AvatarItems() {
		if len(it.Parents) == 1 && it.Parents[0] == root.ArticleID {
			child = it
			break
		}
	}
	if child.ArticleID == 0 {
		t.Fatalf("root %s has no child", root.NameKey)
	}
	return root, child
}

// TestAvatarItemProtoDescCarriesArticle: every tree item's PROTOTYPE_INFO must
// carry a <PItem><Article> (the bridge that fills Battle.ArticleToProto, without
// which a buy is silently dropped) and must NOT carry a <PTool> (a tree item is
// never a click-to-drink bag entry).
func TestAvatarItemProtoDescCarriesArticle(t *testing.T) {
	for _, it := range gamedata.AvatarItems() {
		desc := avatarItemProtoDesc(it)
		if !strings.Contains(desc, "<PItem>") {
			t.Fatalf("%s: proto missing <PItem>: %s", it.NameKey, desc)
		}
		want := `<Article value="` + strconv.Itoa(int(it.ArticleID)) + `"/>`
		if !strings.Contains(desc, want) {
			t.Errorf("%s: proto missing %s", it.NameKey, want)
		}
		if strings.Contains(desc, "<PTool>") {
			t.Errorf("%s: tree item proto must not carry <PTool> (not drinkable)", it.NameKey)
		}
	}
}

// TestBuyAvatarTreeItem drives the happy path: buying a root debits the hero's
// gold, records ownership, and applies the item's stat as a permanent mod;
// buying its child afterwards succeeds and both bonuses stack.
func TestBuyAvatarTreeItem(t *testing.T) {
	s, c, u, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()

	root, child := attackTreeRootAndChild(t)
	// Root is DamageMin-only at stage 1; child adds more DamageMin (both -> dmg_flat).
	startMoney, _, _ := s.Store.HeroMoney(u.ID)
	if startMoney < root.Price+child.Price {
		t.Fatalf("fixture: hero money %d too low for root+child %d", startMoney, root.Price+child.Price)
	}

	s.handleBuy(c, buyPkt(root.ArticleID))

	if !c.huntState.ownedTreeItems[root.ArticleID] {
		t.Fatalf("root %s not marked owned after buy", root.NameKey)
	}
	if money, _, _ := s.Store.HeroMoney(u.ID); money != startMoney-root.Price {
		t.Errorf("money = %d after root buy, want %d", money, startMoney-root.Price)
	}
	now := float64(s.battleTime())
	wantFlat := statSum(root.Stats, "DamageMin")
	if got := c.huntState.st.modSum(now, "dmg_flat"); got != wantFlat {
		t.Errorf("dmg_flat = %v after root, want %v", got, wantFlat)
	}

	// Child now unlocked (parent owned) and affordable.
	s.handleBuy(c, buyPkt(child.ArticleID))
	if !c.huntState.ownedTreeItems[child.ArticleID] {
		t.Fatalf("child %s not owned after buy", child.NameKey)
	}
	if money, _, _ := s.Store.HeroMoney(u.ID); money != startMoney-root.Price-child.Price {
		t.Errorf("money = %d after child buy, want %d", money, startMoney-root.Price-child.Price)
	}
	wantFlat += statSum(child.Stats, "DamageMin")
	if got := c.huntState.st.modSum(now, "dmg_flat"); got != wantFlat {
		t.Errorf("dmg_flat = %v after root+child, want %v (stacked)", got, wantFlat)
	}
}

// TestBuyAvatarTreeItemMaxHP: buying a Health item raises max HP by exactly the
// authored amount and tops the current pool up too.
func TestBuyAvatarTreeItemMaxHP(t *testing.T) {
	s, c, _, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	// A DEFENCE root is Health-only.
	var health gamedata.AvatarItem
	for _, it := range gamedata.AvatarItems() {
		if it.TreeID == gamedata.AvatarTreeDefence && len(it.Parents) == 0 {
			health = it
		}
	}
	hpBonus := statSum(health.Stats, "Health")
	if hpBonus <= 0 {
		t.Fatalf("fixture: defence root %s has no Health stat", health.NameKey)
	}
	baseMax := hs.maxHPLocked(now)
	hs.hp = baseMax // full before the buy

	s.handleBuy(c, buyPkt(health.ArticleID))

	if got := hs.maxHPLocked(now); got != baseMax+hpBonus {
		t.Errorf("maxHP = %v after Health buy, want %v", got, baseMax+hpBonus)
	}
	if hs.hp != baseMax+hpBonus {
		t.Errorf("current hp = %v, want topped up to %v", hs.hp, baseMax+hpBonus)
	}
}

// TestBuyAvatarTreeItemGates: a LOCKED child (parent unowned), an UNAFFORDABLE
// item, and a re-buy of an already-owned item must all be rejected with no gold
// spent and no duplicate ownership.
func TestBuyAvatarTreeItemGates(t *testing.T) {
	s, c, u, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	root, child := attackTreeRootAndChild(t)

	// 1. Child before its parent -> LOCKED, rejected, no spend.
	before, _, _ := s.Store.HeroMoney(u.ID)
	s.handleBuy(c, buyPkt(child.ArticleID))
	if c.huntState.ownedTreeItems[child.ArticleID] {
		t.Error("locked child was bought before its parent")
	}
	if money, _, _ := s.Store.HeroMoney(u.ID); money != before {
		t.Errorf("gold spent on a locked buy: %d -> %d", before, money)
	}

	// 2. Unaffordable: drain the hero below the root price.
	s.Store.AddHeroMoney(u.ID, -(before - (root.Price - 1))) // leave root.Price-1
	poor, _, _ := s.Store.HeroMoney(u.ID)
	s.handleBuy(c, buyPkt(root.ArticleID))
	if c.huntState.ownedTreeItems[root.ArticleID] {
		t.Error("root bought without enough gold")
	}
	if money, _, _ := s.Store.HeroMoney(u.ID); money != poor {
		t.Errorf("gold changed on an unaffordable buy: %d -> %d", poor, money)
	}

	// 3. Afford it, buy once, then re-buy -> second buy is a no-op (no double debit).
	s.Store.AddHeroMoney(u.ID, root.Price*3)
	rich, _, _ := s.Store.HeroMoney(u.ID)
	s.handleBuy(c, buyPkt(root.ArticleID))
	afterFirst, _, _ := s.Store.HeroMoney(u.ID)
	if afterFirst != rich-root.Price {
		t.Fatalf("first buy money = %d, want %d", afterFirst, rich-root.Price)
	}
	s.handleBuy(c, buyPkt(root.ArticleID))
	if money, _, _ := s.Store.HeroMoney(u.ID); money != afterFirst {
		t.Errorf("re-buying an owned item spent gold again: %d -> %d", afterFirst, money)
	}
}

func statSum(stats []gamedata.AvatarItemStat, name string) float64 {
	var v float64
	for _, s := range stats {
		if s.Name == name {
			v += s.Value
		}
	}
	return v
}
