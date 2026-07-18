package gamedata

import (
	"bufio"
	"os"
	"strings"
	"testing"
)

// The avatar battle-tree items (предметы аватаров) are built to a hard contract
// with the shipped client: names/descriptions/icons must be real baked ids (an
// invented one renders "EMPTY!" or a blank square), tree_id/tree_slot/tree_parents
// must obey BattleItemMenu's grid/graph rules, and every tooltip stat name
// (params.skill_id) must match a {placeholder} in that item's baked LongDesc or
// the client logs "Can't find param" and prints -1. These tests hold the authored
// catalog against those invariants.

// validItemIcons loads the legal item-icon container keys (all gui/icons/items/*
// entries, lowercase as the ResourceManager stores them). The client resolves an
// item icon as "Gui/Icons/Items/" + Icon (no "_03"), so the lookup key is that
// concatenation lowercased.
func validItemIcons(t *testing.T) map[string]bool {
	return loadSet(t, "testdata/valid_item_icons.txt")
}

// bakedItemPlaceholders loads, per item base id, the exact stat placeholder set
// transcribed straight from the client's locale LongDesc (suffixes %/^/# already
// stripped) -- an INDEPENDENT extraction the authored params must match.
func bakedItemPlaceholders(t *testing.T) map[string][]string {
	t.Helper()
	f, err := os.Open("testdata/item_placeholders.txt")
	if err != nil {
		t.Fatalf("open item_placeholders.txt: %v", err)
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

func TestAvatarItemCount(t *testing.T) {
	if got := len(AvatarItems()); got != 60 {
		t.Fatalf("AvatarItems() = %d, want 60 (5 trees x 12 cells)", got)
	}
}

// TestAvatarItemArticleIDsUnique: ids are unique, in the authored 60000-range,
// and do not collide with the potion article range (50000..) that shares the
// same client connection's prototype id space.
func TestAvatarItemArticleIDsUnique(t *testing.T) {
	seen := map[int32]bool{}
	for _, it := range AvatarItems() {
		if it.ArticleID < avatarItemArticleBase || it.ArticleID >= avatarItemArticleBase+60 {
			t.Errorf("%s article %d outside 60000-range", it.NameKey, it.ArticleID)
		}
		if seen[it.ArticleID] {
			t.Errorf("duplicate article id %d", it.ArticleID)
		}
		seen[it.ArticleID] = true
		if _, ok := ItemByArticle(it.ArticleID); ok {
			t.Errorf("article %d collides with a consumable article", it.ArticleID)
		}
	}
}

// TestAvatarItemTreeGrid: each of the 5 trees has exactly 12 items filling the
// 3x4 grid slots 1..12 with no duplicate slot (a duplicate slot inside one tree
// makes the client drop the button with a Log.Error).
func TestAvatarItemTreeGrid(t *testing.T) {
	byTree := map[int32][]AvatarItem{}
	for _, it := range AvatarItems() {
		if it.TreeID < AvatarTreeDefence || it.TreeID > AvatarTreeSupport {
			t.Errorf("%s tree_id %d out of 1..5", it.NameKey, it.TreeID)
		}
		byTree[it.TreeID] = append(byTree[it.TreeID], it)
	}
	if len(byTree) != 5 {
		t.Fatalf("got %d trees, want 5", len(byTree))
	}
	for tid, items := range byTree {
		if len(items) != 12 {
			t.Errorf("tree %d has %d items, want 12", tid, len(items))
		}
		slots := map[int32]bool{}
		for _, it := range items {
			if it.TreeSlot < 1 || it.TreeSlot > 12 {
				t.Errorf("%s tree_slot %d out of 1..12", it.NameKey, it.TreeSlot)
			}
			if slots[it.TreeSlot] {
				t.Errorf("tree %d duplicate slot %d", tid, it.TreeSlot)
			}
			slots[it.TreeSlot] = true
		}
	}
}

// TestAvatarItemTreeIsAcyclicDAG: every parent id is a real item IN THE SAME
// tree, at least one root (empty parents) exists per tree, and the parent graph
// has no cycle -- the client walks tree_parents with unguarded recursion, so a
// cycle would overflow its stack.
func TestAvatarItemTreeIsAcyclicDAG(t *testing.T) {
	roots := map[int32]int{}
	for _, it := range AvatarItems() {
		if len(it.Parents) == 0 {
			roots[it.TreeID]++
			continue
		}
		for _, p := range it.Parents {
			pit, ok := AvatarItemByArticle(p)
			if !ok {
				t.Errorf("%s parent %d is not an item", it.NameKey, p)
				continue
			}
			if pit.TreeID != it.TreeID {
				t.Errorf("%s parent %s is in a different tree", it.NameKey, pit.NameKey)
			}
		}
	}
	for tid := AvatarTreeDefence; tid <= AvatarTreeSupport; tid++ {
		if roots[tid] == 0 {
			t.Errorf("tree %d has no root (every item LOCKED forever)", tid)
		}
	}
	// Cycle check: walk parents upward from each node with a visited set.
	for _, it := range AvatarItems() {
		seen := map[int32]bool{it.ArticleID: true}
		stack := append([]int32(nil), it.Parents...)
		for len(stack) > 0 {
			id := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if seen[id] {
				t.Fatalf("cycle in tree_parents reachable from %s (revisits %d)", it.NameKey, id)
			}
			seen[id] = true
			if p, ok := AvatarItemByArticle(id); ok {
				stack = append(stack, p.Parents...)
			}
		}
	}
}

// TestAvatarItemLocaleAndIconsResolve: every Name/Desc key is in the baked
// locale and every icon is a real container entry. A miss ships an "EMPTY!"
// tooltip or a blank icon.
func TestAvatarItemLocaleAndIconsResolve(t *testing.T) {
	locale := validLocaleKeys(t)
	icons := validItemIcons(t)
	for _, it := range AvatarItems() {
		if !locale[it.NameKey] {
			t.Errorf("%s NameKey not in locale", it.NameKey)
		}
		if !locale[it.DescKey] {
			t.Errorf("%s DescKey not in locale", it.DescKey)
		}
		key := strings.ToLower("Gui/Icons/Items/" + it.Icon)
		if !icons[key] {
			t.Errorf("%s icon %q (lookup %q) not in container", it.NameKey, it.Icon, key)
		}
	}
}

// TestAvatarItemParamsMatchPlaceholders is the load-bearing check: the authored
// stat set of every item must equal, as a set, the placeholders baked into that
// item's LongDesc (extracted independently into testdata). A drift here means a
// tooltip param the client can't resolve (Log.Error + "-1"), or a stat we apply
// that the player is never told about.
func TestAvatarItemParamsMatchPlaceholders(t *testing.T) {
	baked := bakedItemPlaceholders(t)
	for _, it := range AvatarItems() {
		base := strings.TrimSuffix(it.DescKey, "_LongDesc")
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
		for _, s := range it.Stats {
			gotSet[s.Name] = true
			if s.Value <= 0 {
				t.Errorf("%s stat %s value %v <= 0", it.NameKey, s.Name, s.Value)
			}
			if (s.Name == "Speed") != s.Mul {
				t.Errorf("%s stat %s Mul=%v (only Speed is multiplicative)", it.NameKey, s.Name, s.Mul)
			}
		}
		if len(gotSet) != len(wantSet) {
			t.Errorf("%s params %v != baked placeholders %v", it.NameKey, keysOf(gotSet), want)
			continue
		}
		for n := range wantSet {
			if !gotSet[n] {
				t.Errorf("%s missing baked stat %s (has %v)", it.NameKey, n, keysOf(gotSet))
			}
		}
	}
}

func keysOf(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
