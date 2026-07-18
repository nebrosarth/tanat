package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

func humanWearableWith(stat string, avoidSlots map[int32]bool) (gamedata.Wearable, bool) {
	for _, w := range gamedata.WearablesForRace(1) {
		if avoidSlots[w.SlotBit] {
			continue
		}
		for _, s := range w.Stats {
			if s.Name == stat {
				return w, true
			}
		}
	}
	return gamedata.Wearable{}, false
}

// dressOnHero buys (free) and dresses a wearable on the conn's hero, returning it.
func dressOnHero(t *testing.T, s *Server, uid int32, w gamedata.Wearable) {
	t.Helper()
	_, _, added, ok := s.Store.BuyWearables(uid, 0, []int32{w.ArticleID})
	if !ok || len(added) != 1 {
		t.Fatalf("BuyWearables(%d) failed", w.ArticleID)
	}
	if _, ok := s.Store.DressWearable(uid, added[0].ID, w.SlotBit); !ok {
		t.Fatalf("DressWearable(%d) failed", w.ArticleID)
	}
}

// TestDressedGearAppliesStats: at battle entry every worn wearable's stats are folded into
// the avatar as permanent mods matching the authored values, and hp/mana are refilled to the
// raised maxima.
func TestDressedGearAppliesStats(t *testing.T) {
	s, c, u, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	hs := c.huntState

	w, ok := humanWearableWith("Health", nil)
	if !ok {
		t.Fatal("no Human wearable carries Health")
	}
	dressOnHero(t, s, u.ID, w)

	now := float64(s.battleTime())
	c.lock()
	s.applyDressedItemStatsLocked(c, now)
	c.unlock()

	// Every authored stat became a permanent (until=0) engine mod of the right value.
	for _, st := range w.Stats {
		engine := wearableModStat(st.Name)
		if engine == "" {
			t.Errorf("stat %s has no engine mapping", st.Name)
			continue
		}
		if got := hs.st.modSum(now, engine); got < st.Value-0.001 || got > st.Value+0.001 {
			t.Errorf("%s -> %s modSum = %v, want %v", st.Name, engine, got, st.Value)
		}
	}
	// The dressed mods are permanent so they survive death/respawn.
	wantSrc := "dress_"
	perm := 0
	for _, m := range hs.st.mods {
		if len(m.src) >= len(wantSrc) && m.src[:len(wantSrc)] == wantSrc {
			if m.until != 0 {
				t.Errorf("dressed mod %s has until=%v, want 0 (permanent)", m.stat, m.until)
			}
			perm++
		}
	}
	if perm != len(w.Stats) {
		t.Errorf("got %d dressed mods, want %d", perm, len(w.Stats))
	}
	// hp/mana were refilled to include the gear's max_hp/max_mana bonus.
	if hs.hp != hs.maxHPLocked(now) {
		t.Errorf("hp = %v, want maxHP %v (should start full incl. gear)", hs.hp, hs.maxHPLocked(now))
	}
	if hs.mana != hs.maxManaLocked(now) {
		t.Errorf("mana = %v, want maxMana %v", hs.mana, hs.maxManaLocked(now))
	}
}

// TestDressedGearRegenAndPenetration: the four gear-only stats (regen + armor penetration)
// map onto engine stats the combat/regen systems already consume.
func TestDressedGearRegenAndPenetration(t *testing.T) {
	s, c, u, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	hs := c.huntState

	used := map[int32]bool{}
	for _, stat := range []string{"HealthRegen", "AntiPhysArmor", "AntiMagicArmor"} {
		w, ok := humanWearableWith(stat, used)
		if !ok {
			t.Fatalf("no Human wearable with %s in a free slot", stat)
		}
		used[w.SlotBit] = true
		dressOnHero(t, s, u.ID, w)
	}

	now := float64(s.battleTime())
	c.lock()
	s.applyDressedItemStatsLocked(c, now)
	c.unlock()

	for _, engine := range []string{"hp_regen", "mana_regen", "phys_armor_pen", "magic_armor_pen"} {
		if got := hs.st.modSum(now, engine); got <= 0 {
			t.Errorf("%s modSum = %v, want > 0 (gear stat not applied)", engine, got)
		}
	}
}
