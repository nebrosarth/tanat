package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

func TestBotItemShoppingFollowsThreeItemCapAndSpeedCap(t *testing.T) {
	s, bot, _, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	s.Store.CreateBotHero(bot.selfPlayerID, "item-progression-bot")
	if !s.Store.SetHeroMoney(bot.selfPlayerID, 1000000, 0) {
		t.Fatal("setup: failed to give bot the test wallet")
	}
	b := newBotBrain(bot, 0, 0)
	bot.huntState.level = 0 // level must not gate the three affordable root items
	now := float64(s.battleTime())

	bot.lock()
	for i := 0; i < 100; i++ {
		s.botBuyItemsLocked(b, now)
	}
	owned := len(bot.huntState.ownedTreeItems)
	factor := botItemMoveSpeedFactorLocked(bot.huntState)
	bot.unlock()

	if owned != gamedata.AvatarTreeMaxItems {
		t.Fatalf("bot bought %d items at level %d, want the three-item cap %d", owned, bot.huntState.level, gamedata.AvatarTreeMaxItems)
	}
	if owned > gamedata.AvatarTreeMaxItems {
		t.Fatalf("bot bought %d items, want no more than %d", owned, gamedata.AvatarTreeMaxItems)
	}
	trees := make(map[int32]bool, owned)
	for article := range bot.huntState.ownedTreeItems {
		it, ok := gamedata.AvatarItemByArticle(article)
		if !ok {
			t.Fatalf("bot owns unknown avatar item %d", article)
		}
		if trees[it.TreeID] {
			t.Fatalf("bot bought two items from avatar tree %d", it.TreeID)
		}
		trees[it.TreeID] = true
	}
	if factor > botMaxItemMoveSpeedFactor+1e-9 {
		t.Fatalf("bot item move-speed factor = %.3f, want <= %.3f", factor, botMaxItemMoveSpeedFactor)
	}
}

func TestBotItemShoppingUpgradesFilledSlotAtUnlockedTier(t *testing.T) {
	s, bot, _, cleanup := newDotaConn(t, "Avtr_HK_Tangren")
	defer cleanup()
	s.Store.CreateBotHero(bot.selfPlayerID, "item-upgrade-bot")
	if !s.Store.SetHeroMoney(bot.selfPlayerID, 1_000_000, 0) {
		t.Fatal("setup: failed to give bot the test wallet")
	}
	b := newBotBrain(bot, 0, 0)
	now := float64(s.battleTime())

	// Start with the same state as a real match: three distinct root items fill
	// the active slots, but their owned parents remain available for upgrades.
	for i := 0; i < gamedata.AvatarTreeMaxItems; i++ {
		s.botBuyItemsLocked(b, now)
	}
	if got := avatarEquippedTreeCountLocked(bot.huntState); got != gamedata.AvatarTreeMaxItems {
		t.Fatalf("active tree count at start = %d, want %d", got, gamedata.AvatarTreeMaxItems)
	}

	// Internal level 4 is displayed as level 5, the first level at which the
	// bot stage policy opens tier 3. A filled three-slot build must progress,
	// rather than being blocked by the active-slot cap.
	bot.huntState.level = 4
	for i := 0; i < gamedata.AvatarTreeMaxItems; i++ {
		s.botBuyItemsLocked(b, now)
	}

	upgraded := 0
	for treeID, article := range bot.huntState.activeTreeItems {
		it, ok := gamedata.AvatarItemByArticle(article)
		if !ok {
			t.Fatalf("active tree %d has unknown article %d", treeID, article)
		}
		if it.Stage >= 3 {
			upgraded++
		}
	}
	if upgraded == 0 {
		t.Fatalf("bot stayed on tier-2 roots after reaching level %d: active=%v owned=%v", bot.huntState.level+1, bot.huntState.activeTreeItems, bot.huntState.ownedTreeItems)
	}
	if got := avatarEquippedTreeCountLocked(bot.huntState); got != gamedata.AvatarTreeMaxItems {
		t.Fatalf("active tree count after upgrades = %d, want %d", got, gamedata.AvatarTreeMaxItems)
	}
}
