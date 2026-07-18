package gamedata

import (
	"bufio"
	"math/bits"
	"os"
	"strings"
	"testing"
)

// The hero-gear ("предметы героев") catalog is held to the same client contract as the
// avatar tree: every Name/Desc/Icon must be a real baked id, every tooltip stat name
// (params.skill_id) must match a {placeholder} in that item's baked LongDesc, and the
// slot/kind must obey the paperdoll bitmask. These tests pin the authored catalog against
// those invariants so an invented string or a mismatched stat can never ship.

// bakedSetPlaceholders loads, per item Name key, the exact stat-placeholder set transcribed
// straight from the client's Set LongDescs (suffixes %/^/# stripped) -- an INDEPENDENT
// extraction the authored params must equal.
func bakedSetPlaceholders(t *testing.T) map[string][]string {
	t.Helper()
	f, err := os.Open("testdata/set_placeholders.txt")
	if err != nil {
		t.Fatalf("open set_placeholders.txt: %v", err)
	}
	defer f.Close()
	out := map[string][]string{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, " ", 2)
		var names []string
		if len(parts) == 2 && parts[1] != "" {
			names = strings.Split(parts[1], ",")
		}
		out[parts[0]] = names
	}
	return out
}

func TestWearableCount(t *testing.T) {
	if got := len(Wearables()); got != 348 {
		t.Fatalf("Wearables() = %d, want 348 (174 Human + 174 Elf baked Set pieces)", got)
	}
}

// TestWearableArticleIDsUnique: ids fill the authored 80000-range without gaps or
// duplicates and never collide with the potion (50000..) or avatar-tree (60000..)
// articles that share the same client prototype id space.
func TestWearableArticleIDsUnique(t *testing.T) {
	seen := map[int32]bool{}
	for _, w := range Wearables() {
		if w.ArticleID < wearableArticleBase || w.ArticleID >= wearableArticleBase+int32(len(Wearables())) {
			t.Errorf("%s article %d outside the 80000-range", w.NameKey, w.ArticleID)
		}
		if seen[w.ArticleID] {
			t.Errorf("duplicate article id %d", w.ArticleID)
		}
		seen[w.ArticleID] = true
		if _, ok := ItemByArticle(w.ArticleID); ok {
			t.Errorf("article %d collides with a consumable article", w.ArticleID)
		}
		if _, ok := AvatarItemByArticle(w.ArticleID); ok {
			t.Errorf("article %d collides with an avatar-tree article", w.ArticleID)
		}
	}
}

// TestWearableSlotKindValid: every item's SlotBit is a single paperdoll bit (a power of
// two in 1..8192), its KindID is a real ShopGUI.ItemType (1..12, never 0=QUEST_ITEM), and
// the two agree with the canonical setSlot table -- the server is the sole authority for
// the kind<->slot rule the client never enforces.
func TestWearableSlotKindValid(t *testing.T) {
	for _, w := range Wearables() {
		if bits.OnesCount32(uint32(w.SlotBit)) != 1 || w.SlotBit < 1 || w.SlotBit > 8192 {
			t.Errorf("%s SlotBit %d is not a single 1..8192 paperdoll bit", w.NameKey, w.SlotBit)
		}
		if w.KindID < 1 || w.KindID > 12 {
			t.Errorf("%s KindID %d out of 1..12 (0 renders as QUEST_ITEM)", w.NameKey, w.KindID)
		}
		si, ok := setSlot[w.Slot]
		if !ok {
			t.Errorf("%s unknown slot token %q", w.NameKey, w.Slot)
			continue
		}
		if w.SlotBit != si.Bit || w.KindID != si.Kind {
			t.Errorf("%s slot %q bit/kind = %d/%d, want %d/%d", w.NameKey, w.Slot, w.SlotBit, w.KindID, si.Bit, si.Kind)
		}
		if w.MinHeroLevel < 1 {
			t.Errorf("%s MinHeroLevel %d < 1", w.NameKey, w.MinHeroLevel)
		}
	}
}

// TestWearableLocaleAndIconsResolve: every Name/Desc key is baked in the locale and every
// icon is a real container entry (after the 22 gap substitutions). A miss ships an "EMPTY!"
// tooltip or a blank square.
func TestWearableLocaleAndIconsResolve(t *testing.T) {
	locale := validLocaleKeys(t)
	icons := validItemIcons(t)
	for _, w := range Wearables() {
		if !locale[w.NameKey] {
			t.Errorf("%s NameKey not in locale", w.NameKey)
		}
		if !locale[w.DescKey] {
			t.Errorf("%s DescKey not in locale", w.DescKey)
		}
		key := strings.ToLower("Gui/Icons/Items/" + w.Icon)
		if !icons[key] {
			t.Errorf("%s icon %q (lookup %q) not in container", w.NameKey, w.Icon, key)
		}
	}
}

// TestWearableParamsMatchPlaceholders is the load-bearing check: the authored stat set of
// every item must equal, as a set, the placeholders baked into that item's LongDesc; every
// value must be > 0; and no wearable stat is multiplicative (all impact 0).
func TestWearableParamsMatchPlaceholders(t *testing.T) {
	baked := bakedSetPlaceholders(t)
	for _, w := range Wearables() {
		want, ok := baked[w.NameKey]
		if !ok {
			t.Errorf("%s: no baked placeholder set", w.NameKey)
			continue
		}
		wantSet := map[string]bool{}
		for _, n := range want {
			wantSet[n] = true
		}
		gotSet := map[string]bool{}
		for _, s := range w.Stats {
			gotSet[s.Name] = true
			if s.Value <= 0 {
				t.Errorf("%s stat %s value %v <= 0", w.NameKey, s.Name, s.Value)
			}
			if s.Mul {
				t.Errorf("%s stat %s is multiplicative (wearable stats are all additive)", w.NameKey, s.Name)
			}
		}
		if len(gotSet) != len(wantSet) {
			t.Errorf("%s params %v != baked placeholders %v", w.NameKey, keysOf(gotSet), want)
			continue
		}
		for n := range wantSet {
			if !gotSet[n] {
				t.Errorf("%s missing baked stat %s (has %v)", w.NameKey, n, keysOf(gotSet))
			}
		}
	}
}

// TestWearableRaceFilter: the shop filter partitions the catalog cleanly by race, 174 each.
func TestWearableRaceFilter(t *testing.T) {
	h, e := WearablesForRace(1), WearablesForRace(2)
	if len(h) != 174 || len(e) != 174 {
		t.Fatalf("race filter = %d Human / %d Elf, want 174/174", len(h), len(e))
	}
	for _, w := range h {
		if w.Race != "Human" {
			t.Errorf("Human filter leaked %s (race %s)", w.NameKey, w.Race)
		}
	}
	for _, w := range e {
		if w.Race != "Elf" {
			t.Errorf("Elf filter leaked %s (race %s)", w.NameKey, w.Race)
		}
	}
}
