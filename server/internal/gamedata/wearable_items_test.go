package gamedata

import (
	"bufio"
	"math"
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

// findWearable returns the single catalog piece matching race/tier/color/slot.
func findWearable(t *testing.T, race string, tier int32, color, slot string) Wearable {
	t.Helper()
	for _, w := range Wearables() {
		if w.Race == race && w.Tier == tier && w.Color == color && w.Slot == slot {
			return w
		}
	}
	t.Fatalf("no wearable %s S%d %s %s", race, tier, color, slot)
	return Wearable{}
}

func wearableStat(w Wearable, name string) (float64, bool) {
	for _, s := range w.Stats {
		if s.Name == name {
			return s.Value, true
		}
	}
	return 0, false
}

// TestWearableWikiStatsApplied pins that the exact numbers the official wiki «Геройские
// комплекты» published are what the catalog now serves -- not the old synthetic formula. Spot
// checks span the range (Novice grey -> Властелин violet) and both races, since the wiki is
// race-agnostic. CritChance is checked as a FRACTION (wiki % / 100) because {CritChance%} is the
// one placeholder the client multiplies by 100 for display / the engine reads as a probability.
func TestWearableWikiStatsApplied(t *testing.T) {
	type check struct {
		race, color, slot string
		tier              int32
		stat              string
		want              float64
	}
	cases := []check{
		// Сапоги Новичка (Grey S0 Foots): +10 HP, +0.04 hp-regen, +0.5 phys armor
		{"Elf", "Grey", "Foots", 0, "Health", 10},
		{"Human", "Grey", "Foots", 0, "HealthRegen", 0.04},
		{"Elf", "Grey", "Foots", 0, "PhysArmor", 0.5},
		// S0 faction pieces get the wiki Novice stats: Пояс Новичка (Elf Belt) and Кираса
		// Новичка (Human Body) both = +0.04 hp-regen, +0.5 phys armor.
		{"Elf", "Grey", "Belt", 0, "PhysArmor", 0.5},
		{"Human", "Grey", "Body", 0, "HealthRegen", 0.04},
		{"Human", "Grey", "Body", 0, "PhysArmor", 0.5},
		// Наручи Новичка (Grey S0 Hands): +0.4% crit -> 0.004 fraction, +0.01 atk speed
		{"Human", "Grey", "Hands", 0, "CritChance", 0.004},
		{"Elf", "Grey", "Hands", 0, "AttackSpeed", 0.01},
		// Шлем Воителя (Grey S1 Helm): +0.008 anti-magic, +9 mana
		{"Elf", "Grey", "Helm", 1, "AntiMagicArmor", 0.008},
		{"Human", "Grey", "Helm", 1, "Mana", 9},
		// Наплечники Защитника (Grey S7 Shoulders): +135 HP, +3.9 magic armor
		{"Elf", "Grey", "Shoulders", 7, "Health", 135},
		{"Human", "Grey", "Shoulders", 7, "MagicArmor", 3.9},
		// Перчатки Чемпиона (Blue S7 Hands): +6% crit -> 0.06, +0.18 atk speed, +96 HP
		{"Elf", "Blue", "Hands", 7, "CritChance", 0.06},
		{"Human", "Blue", "Hands", 7, "AttackSpeed", 0.18},
		// Сапоги Властелина (Violet S7 Foots): +110 HP, +0.44 hp-regen, +8.6 phys armor
		{"Elf", "Violet", "Foots", 7, "Health", 110},
		{"Human", "Violet", "Foots", 7, "PhysArmor", 8.6},
		// Шлем Властелина (Violet S7 Helm): +11.4 anti-magic, +88 mana, +0.44 mana-regen
		{"Elf", "Violet", "Helm", 7, "AntiMagicArmor", 11.4},
		{"Human", "Violet", "Helm", 7, "ManaRegen", 0.44},
	}
	for _, c := range cases {
		w := findWearable(t, c.race, c.tier, c.color, c.slot)
		got, ok := wearableStat(w, c.stat)
		if !ok {
			t.Errorf("%s: no %s stat", w.NameKey, c.stat)
			continue
		}
		if math.Abs(got-c.want) > 1e-9 {
			t.Errorf("%s %s = %v, want %v (wiki value)", w.NameKey, c.stat, got, c.want)
		}
	}
}

// TestWearableCritIsFraction guards the {CritChance%} conversion: the wiki prints crit as a
// percent (0.4% .. 7%), but the engine's crit_pct is a probability and the client re-multiplies
// by 100 for display, so every stored CritChance must be a small fraction (< 0.5), never a raw
// percent like 7. A regression here would make gear crit 100x too strong.
func TestWearableCritIsFraction(t *testing.T) {
	seen := false
	for _, w := range Wearables() {
		if v, ok := wearableStat(w, "CritChance"); ok {
			seen = true
			if v <= 0 || v >= 0.5 {
				t.Errorf("%s CritChance = %v, want a small fraction in (0,0.5)", w.NameKey, v)
			}
		}
	}
	if !seen {
		t.Fatal("no wearable carries CritChance -- test is vacuous")
	}
}

// TestWearableElfHumanStatsIdentical: the wiki gives one stat block per set (no race split),
// so where both races share a tier+color+slot piece their stats must be identical. The ONLY
// asymmetry is the S0 Novice set's faction pieces -- Elf gets the Пояс (Belt, «только
// Изгнанники»), Human gets the Кираса (Body, «только Собор») -- which the test pins exactly.
func TestWearableElfHumanStatsIdentical(t *testing.T) {
	type key struct {
		tier        int32
		color, slot string
	}
	elf, human := map[key]Wearable{}, map[key]Wearable{}
	for _, w := range Wearables() {
		if w.Race == "Elf" {
			elf[key{w.Tier, w.Color, w.Slot}] = w
		} else {
			human[key{w.Tier, w.Color, w.Slot}] = w
		}
	}
	for k, e := range elf {
		h, ok := human[k]
		if !ok {
			continue // race-only slot, checked below
		}
		if len(e.Stats) != len(h.Stats) {
			t.Errorf("S%d %s %s: Elf %d stats vs Human %d", k.tier, k.color, k.slot, len(e.Stats), len(h.Stats))
			continue
		}
		for _, s := range e.Stats {
			hv, ok := wearableStat(h, s.Name)
			if !ok || math.Abs(hv-s.Value) > 1e-9 {
				t.Errorf("S%d %s %s %s: Elf=%v Human=%v (races must match)", k.tier, k.color, k.slot, s.Name, s.Value, hv)
			}
		}
	}
	// Pin the faction asymmetry: exactly one Elf-only and one Human-only slot, both at S0 Grey.
	var elfOnly, humanOnly []key
	for k := range elf {
		if _, ok := human[k]; !ok {
			elfOnly = append(elfOnly, k)
		}
	}
	for k := range human {
		if _, ok := elf[k]; !ok {
			humanOnly = append(humanOnly, k)
		}
	}
	if len(elfOnly) != 1 || elfOnly[0] != (key{0, "Grey", "Belt"}) {
		t.Errorf("Elf-only slots = %v, want exactly [S0 Grey Belt]", elfOnly)
	}
	if len(humanOnly) != 1 || humanOnly[0] != (key{0, "Grey", "Body"}) {
		t.Errorf("Human-only slots = %v, want exactly [S0 Grey Body]", humanOnly)
	}
}

// TestWikiStatsAllConsumed: every authored wiki value is actually served by a catalog piece
// (no dead keys), and the Grey S1 belt anomaly correctly falls back to the synthetic Mana/
// MagicArmor its baked placeholders demand rather than the wiki's inapplicable HealthRegen/
// PhysArmor row.
func TestWikiStatsAllConsumed(t *testing.T) {
	for k, v := range wikiWearableStats {
		// Search any race: the S0 Novice faction pieces exist for only one race each.
		found := false
		for _, w := range Wearables() {
			if w.Tier != k.Tier || w.Color != k.Color || w.Slot != k.Slot {
				continue
			}
			got, ok := wearableStat(w, k.Stat)
			if !ok {
				continue
			}
			found = true
			if math.Abs(got-v) > 1e-9 {
				t.Errorf("wiki stat %+v = %v on %s, want %v", k, got, w.NameKey, v)
			}
		}
		if !found {
			t.Errorf("wiki stat %+v is served by no catalog piece (dead key)", k)
		}
	}
	// The wiki's Воитель belt row must NOT leak in (its placeholders are Mana/MagicArmor).
	belt := findWearable(t, "Elf", 1, "Grey", "Belt")
	if _, ok := wearableStat(belt, "HealthRegen"); ok {
		t.Errorf("%s picked up the wiki belt anomaly (HealthRegen); should stay Mana/MagicArmor", belt.NameKey)
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
