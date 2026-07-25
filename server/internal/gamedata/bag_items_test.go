package gamedata

import (
	"bufio"
	"math/rand"
	"os"
	"strings"
	"testing"
)

// TestBagCatalogsPopulated guards against an empty codegen (a broken extraction/regen).
func TestBagCatalogsPopulated(t *testing.T) {
	if len(Runes()) < 50 {
		t.Errorf("runes: want >=50, got %d", len(Runes()))
	}
	if len(Elixirs()) < 40 {
		t.Errorf("elixirs: want >=40, got %d", len(Elixirs()))
	}
	if len(Chests()) < 20 {
		t.Errorf("chests: want >=20, got %d", len(Chests()))
	}
	if len(Totems()) < 10 {
		t.Errorf("totems: want >=10, got %d", len(Totems()))
	}
}

// TestBagArticleRangesDisjoint asserts every family's article ids sit in its own window and
// no two families (nor potions/wearables) overlap -- the invariant ResolveBagArticle relies on.
func TestBagArticleRangesDisjoint(t *testing.T) {
	seen := map[int32]string{}
	add := func(fam string, id int32) {
		if prev, ok := seen[id]; ok {
			t.Fatalf("article id %d used by both %s and %s", id, prev, fam)
		}
		seen[id] = fam
	}
	for _, it := range Items() {
		add("potion", it.ArticleID)
	}
	// Avatar battle-tree items share the one items.amf id space and sit at 60000+ -- the
	// exact range a naive bag base would collide with. Include them so any future overlap
	// (or a range typo) fails here instead of silently clobbering a tree item in the catalog.
	for _, it := range AvatarItems() {
		add("avatar_item", it.ArticleID)
	}
	for _, r := range Runes() {
		if r.ArticleID < runeArticleBase || r.ArticleID >= elixirArticleBase {
			t.Errorf("rune %d out of range", r.ArticleID)
		}
		add("rune", r.ArticleID)
	}
	for _, e := range Elixirs() {
		if e.ArticleID < elixirArticleBase || e.ArticleID >= chestArticleBase {
			t.Errorf("elixir %d out of range", e.ArticleID)
		}
		add("elixir", e.ArticleID)
	}
	for _, c := range Chests() {
		if c.ArticleID < chestArticleBase || c.ArticleID >= totemArticleBase {
			t.Errorf("chest %d out of range", c.ArticleID)
		}
		add("chest", c.ArticleID)
	}
	for _, tt := range Totems() {
		if tt.ArticleID < totemArticleBase || tt.ArticleID >= bagArticleTop {
			t.Errorf("totem %d out of range", tt.ArticleID)
		}
		add("totem", tt.ArticleID)
	}
	for _, w := range Wearables() {
		add("wearable", w.ArticleID)
	}
}

// TestBagIconsValid confirms every new item's icon resolves to a real client asset (the same
// allowlist the potion/wearable icons are checked against), so nothing renders as a missing
// texture.
func TestBagIconsValid(t *testing.T) {
	allow := loadIconAllowlist(t)
	check := func(kind, icon string) {
		key := "gui/icons/items/" + strings.ToLower(icon)
		if !allow[key] {
			t.Errorf("%s icon not in allowlist: %s", kind, icon)
		}
	}
	for _, r := range Runes() {
		check("rune", r.Icon)
	}
	for _, e := range Elixirs() {
		check("elixir", e.Icon)
	}
	for _, c := range Chests() {
		check("chest", c.Icon)
	}
	for _, tt := range Totems() {
		check("totem", tt.Icon)
	}
}

// TestResolveBagArticle round-trips one of each family + rejects an unknown id.
func TestResolveBagArticle(t *testing.T) {
	cases := []struct {
		id   int32
		fam  BagFamily
		open bool
		use  bool
	}{
		{Items()[0].ArticleID, BagPotion, false, true},
		{Runes()[0].ArticleID, BagRune, false, true},
		{Elixirs()[0].ArticleID, BagElixir, false, true},
		{Chests()[0].ArticleID, BagChest, true, false},
		{Totems()[0].ArticleID, BagTotem, false, false},
	}
	for _, c := range cases {
		got, ok := ResolveBagArticle(c.id)
		if !ok || got.Family != c.fam || got.Openable != c.open || got.Usable != c.use {
			t.Errorf("resolve %d: got %+v ok=%v", c.id, got, ok)
		}
		if got.NameKey == "" || got.Icon == "" {
			t.Errorf("resolve %d: empty name/icon", c.id)
		}
	}
	if _, ok := ResolveBagArticle(1); ok {
		t.Error("unknown article resolved")
	}
	if _, ok := ResolveBagArticle(80000); ok {
		t.Error("wearable article should not resolve as a bag stack")
	}
}

// TestRollChestAlwaysPaysCoins samples every chest across many seeds: a roll must always pay
// positive coins, and any bonus item must be a real catalog article.
func TestRollChestAlwaysPaysCoins(t *testing.T) {
	valid := map[int32]bool{}
	for _, it := range Items() {
		valid[it.ArticleID] = true
	}
	for _, r := range Runes() {
		valid[r.ArticleID] = true
	}
	for _, e := range Elixirs() {
		valid[e.ArticleID] = true
	}
	for _, c := range Chests() {
		rng := rand.New(rand.NewSource(int64(c.ArticleID)))
		for i := 0; i < 200; i++ {
			rew := RollChest(c, rng)
			if rew.Coins <= 0 {
				t.Fatalf("chest %s paid %d coins", c.Infix, rew.Coins)
			}
			for _, g := range rew.Items {
				if !valid[g.Article] || g.Count <= 0 {
					t.Fatalf("chest %s dropped invalid item %+v", c.Infix, g)
				}
			}
		}
	}
}

// TestRuneCardParamsMatchPlaceholders guards the items.amf rune tooltip against the real client
// LongDesc: the set of params CardParams emits must EXACTLY equal the {placeholder} set baked in
// the rune's LongDesc (testdata/rune_placeholders.txt, transcribed from the live locale). A
// missing param renders "-1" on the card; an extra one is a placeholder the client can't resolve.
// Literal-desc runes (S1-S3) have no placeholders and must emit no params. Every emitted value
// must be > 0 so the number is real.
func TestRuneCardParamsMatchPlaceholders(t *testing.T) {
	baked := loadRunePlaceholders(t)
	for _, r := range Runes() {
		base := strings.TrimSuffix(r.DescKey, "_LongDesc")
		want, ok := baked[base]
		if !ok {
			t.Errorf("%s: no baked placeholder set", base)
			continue
		}
		wantSet := map[string]bool{}
		for _, n := range want {
			wantSet[n] = true
		}
		gotSet := map[string]bool{}
		for _, p := range r.CardParams() {
			gotSet[p.Key] = true
			if p.Value <= 0 {
				t.Errorf("%s param %s value %v <= 0", r.NameKey, p.Key, p.Value)
			}
		}
		if len(gotSet) != len(wantSet) {
			t.Errorf("%s params %v != baked placeholders %v", r.NameKey, keysOf(gotSet), want)
			continue
		}
		for n := range wantSet {
			if !gotSet[n] {
				t.Errorf("%s missing baked placeholder %s (has %v)", r.NameKey, n, keysOf(gotSet))
			}
		}
	}
}

// loadRunePlaceholders reads testdata/rune_placeholders.txt ("<base_key> ph1,ph2" per line, the
// placeholder list empty for a literal-desc rune) into base_key -> placeholder names.
func loadRunePlaceholders(t *testing.T) map[string][]string {
	t.Helper()
	f, err := os.Open("testdata/rune_placeholders.txt")
	if err != nil {
		t.Fatalf("open rune_placeholders.txt: %v", err)
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

// loadIconAllowlist reads testdata/valid_item_icons.txt into a set (lowercased lines).
func loadIconAllowlist(t *testing.T) map[string]bool {
	t.Helper()
	f, err := os.Open("testdata/valid_item_icons.txt")
	if err != nil {
		t.Fatalf("open allowlist: %v", err)
	}
	defer f.Close()
	set := map[string]bool{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if l := strings.TrimSpace(sc.Text()); l != "" {
			set[strings.ToLower(l)] = true
		}
	}
	return set
}
