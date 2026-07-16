package session

import (
	"path/filepath"
	"testing"
)

// TestMoneyPersistsAcrossRestart guards the bug this session fixed: mob-kill
// coin bounties (AddHeroMoney, called from the Battle server's
// awardCoinsLocked) must survive a process restart, not just live in memory.
func TestMoneyPersistsAcrossRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "accounts.json")
	s1 := NewPersistentStore(path)
	u, _ := s1.LoginOrRegister("a@b.c", "pw")
	s1.CreateHero(u, 1, true, 0, 0, 0, 0, 0)

	money, _, ok := s1.AddHeroMoney(u.ID, 250)
	if !ok || money != 1250 { // starter 1000 + 250
		t.Fatalf("AddHeroMoney = %d, %v, want 1250, true", money, ok)
	}

	s2 := NewPersistentStore(path)
	got, ok := s2.usersByEmail["a@b.c"]
	if !ok || got.Hero == nil {
		t.Fatal("account/hero not reloaded")
	}
	if got.Hero.Money != 1250 {
		t.Errorf("reloaded money = %d, want 1250 (money did not persist)", got.Hero.Money)
	}
}

// TestAddHeroExpLevelsUp checks the persistent character-XP curve
// (heroExpNextLevel = 100*level) advances Level/Exp/NextExp and persists.
func TestAddHeroExpLevelsUp(t *testing.T) {
	path := filepath.Join(t.TempDir(), "accounts.json")
	s1 := NewPersistentStore(path)
	u, _ := s1.LoginOrRegister("a@b.c", "pw")
	s1.CreateHero(u, 1, true, 0, 0, 0, 0, 0) // starts Level 1, Exp 0, NextExp 100

	// Start: Level 1, Exp 0, NextExp 100. +250 exp: consumes the 100 for level 2
	// (150 left, next=200 for level 3); 150 < 200, so it stops there.
	level, exp, next, ok := s1.AddHeroExp(u.ID, 250)
	if !ok {
		t.Fatal("AddHeroExp returned ok=false")
	}
	if level != 2 || exp != 150 || next != 200 {
		t.Errorf("AddHeroExp(250) = level %d exp %d next %d, want 2 150 200", level, exp, next)
	}

	s2 := NewPersistentStore(path)
	got := s2.usersByEmail["a@b.c"].Hero
	if got.Level != 2 || got.Exp != 150 || got.NextExp != 200 {
		t.Errorf("reloaded hero = level %d exp %d next %d, want 2 150 200 (exp did not persist)",
			got.Level, got.Exp, got.NextExp)
	}
}

// TestBagPersistsAndMerges checks AddBagItem merges same-article stacks,
// RemoveBagItem drops an emptied stack, and the bag survives a reload.
func TestBagPersistsAndMerges(t *testing.T) {
	path := filepath.Join(t.TempDir(), "accounts.json")
	s1 := NewPersistentStore(path)
	u, _ := s1.LoginOrRegister("a@b.c", "pw")
	s1.CreateHero(u, 1, true, 0, 0, 0, 0, 0)

	if !s1.AddBagItem(u.ID, 5000, 2) {
		t.Fatal("AddBagItem failed")
	}
	if !s1.AddBagItem(u.ID, 5000, 3) { // merges into the same stack
		t.Fatal("AddBagItem (merge) failed")
	}
	if !s1.AddBagItem(u.ID, 5008, 1) { // a different article, new stack
		t.Fatal("AddBagItem (second article) failed")
	}
	bag := s1.HeroBag(u.ID)
	if len(bag) != 2 {
		t.Fatalf("bag has %d stacks, want 2: %+v", len(bag), bag)
	}
	var got5000 int32
	for _, bi := range bag {
		if bi.ArticleID == 5000 {
			got5000 = bi.Count
		}
	}
	if got5000 != 5 {
		t.Errorf("article 5000 count = %d, want 5 (merge failed)", got5000)
	}

	if !s1.RemoveBagItem(u.ID, 5000, 5) { // empties and drops the stack
		t.Fatal("RemoveBagItem failed")
	}
	if len(s1.HeroBag(u.ID)) != 1 {
		t.Errorf("bag has %d stacks after emptying 5000, want 1", len(s1.HeroBag(u.ID)))
	}

	s2 := NewPersistentStore(path)
	bag2 := s2.HeroBag(u.ID)
	if len(bag2) != 1 || bag2[0].ArticleID != 5008 || bag2[0].Count != 1 {
		t.Errorf("reloaded bag = %+v, want [{5008 1}] (bag did not persist)", bag2)
	}
}
