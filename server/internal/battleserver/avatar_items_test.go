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

// TestBuyAvatarTreeItemsFromDistinctTrees drives the happy path: buying roots
// from two different trees debits the hero's gold, records both ownerships,
// and applies both permanent stat bonuses.
func TestBuyAvatarTreeItemsFromDistinctTrees(t *testing.T) {
	s, c, u, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	c.huntState.level = 4 // displayed character level 5 unlocks the selected tier-2 roots

	root := avatarTreeChain(t, gamedata.AvatarTreeAttack, 1)[0]
	other := avatarTreeChain(t, gamedata.AvatarTreeDefence, 1)[0]
	startMoney, _, _ := s.Store.HeroMoney(u.ID)
	if startMoney < root.Price+other.Price {
		t.Fatalf("fixture: hero money %d too low for two roots %d", startMoney, root.Price+other.Price)
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

	// A root from a different tree is independently available and affordable.
	s.handleBuy(c, buyPkt(other.ArticleID))
	if !c.huntState.ownedTreeItems[other.ArticleID] {
		t.Fatalf("root %s from a different tree was not owned after buy", other.NameKey)
	}
	if money, _, _ := s.Store.HeroMoney(u.ID); money != startMoney-root.Price-other.Price {
		t.Errorf("money = %d after two distinct roots, want %d", money, startMoney-root.Price-other.Price)
	}
	wantFlat += statSum(other.Stats, "DamageMin")
	if got := c.huntState.st.modSum(now, "dmg_flat"); got != wantFlat {
		t.Errorf("dmg_flat = %v after two distinct roots, want %v (stacked)", got, wantFlat)
	}
}

func TestBuyAvatarTreeItemsRespectAvatarLevelGates(t *testing.T) {
	s, c, u, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	if !s.Store.SetHeroMoney(u.ID, 1_000_000, 0) {
		t.Fatal("fixture: failed to fund hero")
	}

	var magicTierOne, magicTierTwo gamedata.AvatarItem
	for _, it := range gamedata.AvatarItems() {
		if it.TreeID != gamedata.AvatarTreeMagic {
			continue
		}
		if it.Stage == 1 {
			magicTierOne = it
		}
		if it.Stage == 2 && magicTierTwo.ArticleID == 0 {
			magicTierTwo = it
		}
	}
	if magicTierOne.ArticleID == 0 || magicTierTwo.ArticleID == 0 {
		t.Fatal("fixture: Magic tree is missing tier-1 or tier-2 item")
	}

	// The first Magic-tier item is available at the starting character level.
	s.handleBuy(c, buyPkt(magicTierOne.ArticleID))
	if !c.huntState.ownedTreeItems[magicTierOne.ArticleID] {
		t.Fatal("Magic tier-1 item was not buyable at character level 1")
	}

	// A fresh connection is used so the level-gate check starts with a clean
	// tree. Buy the central T1 first: T2 is an upgrade branch, not a second root.
	s2, c2, u2, cleanup2 := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup2()
	if !s2.Store.SetHeroMoney(u2.ID, 1_000_000, 0) {
		t.Fatal("fixture: failed to fund second hero")
	}
	s2.handleBuy(c2, buyPkt(magicTierOne.ArticleID))
	s2.handleBuy(c2, buyPkt(magicTierTwo.ArticleID))
	if c2.huntState.ownedTreeItems[magicTierTwo.ArticleID] {
		t.Fatal("Magic tier-2 item was bought below character level 5")
	}

	c2.huntState.level = 4 // displayed character level 5
	s2.handleBuy(c2, buyPkt(magicTierTwo.ArticleID))
	if !c2.huntState.ownedTreeItems[magicTierTwo.ArticleID] {
		t.Fatal("Magic tier-2 item was not buyable at character level 5")
	}
}

// TestBuyAvatarTreeItemMaxHP: buying a Health item raises max HP by exactly the
// authored amount and tops the current pool up too.
func TestBuyAvatarTreeItemMaxHP(t *testing.T) {
	s, c, _, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	c.huntState.level = 4 // the selected defence root is tier 2
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
	c.huntState.level = 4 // the selected attack root is tier 2
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

func TestBuyAvatarTreeItemsCapsAtClientLimit(t *testing.T) {
	s, c, u, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	c.huntState.level = 4 // all selected roots are tier 2

	treeIDs := []int32{gamedata.AvatarTreeAttack, gamedata.AvatarTreeDefence, gamedata.AvatarTreeMagic, gamedata.AvatarTreeControl}
	roots := make([]gamedata.AvatarItem, 0, len(treeIDs))
	for _, treeID := range treeIDs {
		roots = append(roots, avatarTreeChain(t, treeID, 1)[0])
	}
	if !s.Store.SetHeroMoney(u.ID, 1_000_000, 0) {
		t.Fatal("fixture: failed to fund hero")
	}

	for _, it := range roots[:gamedata.AvatarTreeMaxItems] {
		s.handleBuy(c, buyPkt(it.ArticleID))
	}
	if got := len(c.huntState.ownedTreeItems); got != gamedata.AvatarTreeMaxItems {
		t.Fatalf("owned avatar items = %d after valid buys, want %d", got, gamedata.AvatarTreeMaxItems)
	}

	beforeMoney, _, _ := s.Store.HeroMoney(u.ID)
	s.handleBuy(c, buyPkt(roots[gamedata.AvatarTreeMaxItems].ArticleID))
	if c.huntState.ownedTreeItems[roots[gamedata.AvatarTreeMaxItems].ArticleID] {
		t.Fatal("fourth avatar item was bought despite the three-item cap")
	}
	if got := len(c.huntState.ownedTreeItems); got != gamedata.AvatarTreeMaxItems {
		t.Fatalf("owned avatar items = %d after fourth buy, want %d", got, gamedata.AvatarTreeMaxItems)
	}
	if afterMoney, _, _ := s.Store.HeroMoney(u.ID); afterMoney != beforeMoney {
		t.Fatalf("fourth buy changed money: %d -> %d", beforeMoney, afterMoney)
	}
}

func TestBuyAvatarTreeItemUpgradeReplacesActiveTreeSlot(t *testing.T) {
	s, c, u, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	c.huntState.level = 4 // unlock the tier-2 root selected by avatarTreeChain

	chain := avatarTreeChain(t, gamedata.AvatarTreeAttack, 2)
	if !s.Store.SetHeroMoney(u.ID, 1_000_000, 0) {
		t.Fatal("fixture: failed to fund hero")
	}
	s.handleBuy(c, buyPkt(chain[0].ArticleID))
	beforeMoney, _, _ := s.Store.HeroMoney(u.ID)

	c.huntState.level = 9 // displayed character level 10 unlocks the tier-3 upgrade
	s.handleBuy(c, buyPkt(chain[1].ArticleID))
	if !c.huntState.ownedTreeItems[chain[1].ArticleID] {
		t.Fatal("upgrade from the same avatar tree was not bought")
	}
	if got := len(c.huntState.ownedTreeItems); got != 2 {
		t.Fatalf("owned avatar items = %d after upgrade, want parent plus child", got)
	}
	if got := avatarTreeActiveLocked(c.huntState, chain[0].TreeID); got != chain[1].ArticleID {
		t.Fatalf("active tree article = %d after upgrade, want %d", got, chain[1].ArticleID)
	}
	if got := c.huntState.st.modSum(float64(s.battleTime()), "dmg_flat"); got != statSum(chain[1].Stats, "DamageMin") {
		t.Fatalf("active item damage bonus = %v after upgrade, want only tier-%d value %v", got, chain[1].Stage, statSum(chain[1].Stats, "DamageMin"))
	}
	if afterMoney, _, _ := s.Store.HeroMoney(u.ID); afterMoney != beforeMoney-chain[1].Price {
		t.Fatalf("upgrade money = %d, want %d", afterMoney, beforeMoney-chain[1].Price)
	}
}

func avatarTreeChain(t *testing.T, treeID int32, count int) []gamedata.AvatarItem {
	t.Helper()
	var current gamedata.AvatarItem
	for _, it := range gamedata.AvatarItems() {
		if it.TreeID == treeID && len(it.Parents) == 0 && (current.ArticleID == 0 || it.ArticleID < current.ArticleID) {
			current = it
		}
	}
	if current.ArticleID == 0 {
		t.Fatalf("no root in avatar tree %d", treeID)
	}

	chain := make([]gamedata.AvatarItem, 0, count)
	for len(chain) < count {
		chain = append(chain, current)
		if len(chain) == count {
			break
		}
		var next gamedata.AvatarItem
		for _, it := range gamedata.AvatarItems() {
			if it.TreeID == treeID && len(it.Parents) == 1 && it.Parents[0] == current.ArticleID {
				next = it
				break
			}
		}
		if next.ArticleID == 0 {
			t.Fatalf("avatar tree %d root %d has no chain item %d", treeID, current.ArticleID, len(chain)+1)
		}
		current = next
	}
	return chain
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
