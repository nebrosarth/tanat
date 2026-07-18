package gamedata

import (
	"bufio"
	"os"
	"strings"
	"testing"
)

// The object card (ObjectInfo) is the panel that names whatever you have selected. It
// reads two strings off the prototype we send, and it resolves BOTH by exact lookup
// against tables the client ships. There is no fallback on either path and no way for
// the server to supply the string itself, so a name or icon we invent is not a
// harmless miss -- it is a card that reads "EMPTY!" over a blank square. These tests
// hold every such string against the real client tables.
//
// Both manifests are dumped from the shipped client (scratchpad/dump_manifests.py).

// validCardIcons loads the legal <PDesc><Icon> values. ObjectInfo.cs:165 loads
// mIcon + "_03", so the legal set is the container's *_03 textures with the suffix
// stripped -- that suffix is the client's to add and ours to leave off. Keys are
// lowercase, matching how the container stores (and therefore compares) them.
func validCardIcons(t *testing.T) map[string]bool {
	return loadSet(t, "testdata/valid_card_icons.txt")
}

// validLocaleKeys loads every id in the baked locale. LocaleState.GetText renders an
// unknown id as the literal text "EMPTY!" and an empty one as "" plus a log warning.
// The locale is baked into the client: the server cannot add a key, only cite one.
func validLocaleKeys(t *testing.T) map[string]bool {
	return loadSet(t, "testdata/valid_locale_keys.txt")
}

func loadSet(t *testing.T, path string) map[string]bool {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer f.Close()
	set := map[string]bool{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if s := strings.TrimSpace(sc.Text()); s != "" {
			set[s] = true
		}
	}
	return set
}

// iconlessMobs: roster entries deliberately shipped with no card icon, because the
// client ships no art for them at card resolution. Anything NOT listed here must have
// one -- an empty icon is worse than a wrong one, since ObjectInfo appends "_03"
// before loading and so asks for a texture literally named "_03".
//
// The panther is the only one, and it is unreachable: index 0 exists to pin the
// roster numbering that several battle tests key off, and no spawn pack places it.
// No panther texture exists in mainData, resources.assets or sharedassets0.
var iconlessMobs = map[string]bool{"IDS_MobMeleePanter_Name": true}

func TestMobCardsResolve(t *testing.T) {
	icons, keys := validCardIcons(t), validLocaleKeys(t)
	for i, m := range Mobs() {
		if !keys[m.NameKey] {
			t.Errorf("mob %d (%s) name key is not in the baked locale: the card would read \"EMPTY!\"", i, m.NameKey)
		}
		switch {
		case m.Icon == "":
			if !iconlessMobs[m.NameKey] {
				t.Errorf("mob %d (%s) has no card icon: the client would try to load a texture named \"_03\"", i, m.NameKey)
			}
		case !icons[strings.ToLower(m.Icon)]:
			t.Errorf("mob %d (%s) icon %q does not exist in the client (card renders blank)", i, m.NameKey, m.Icon)
		case strings.HasSuffix(m.Icon, "_03"):
			t.Errorf("mob %d (%s) icon %q carries the _03 suffix: ObjectInfo appends it, so this loads %q", i, m.NameKey, m.Icon, m.Icon+"_03")
		}
	}
}

// TestMapCardsResolve is the "no EMPTY! on the map card" guard. SelectGameMenu runs a map's
// Name, Desc and WinDesc each through GuiSystem.GetLocaleText (:81, :92-93), so all three must
// be baked locale KEYS -- a display string renders the literal "EMPTY!" on the card. This is the
// test that was missing when «Штурм» shipped Name="IDS_DOTA_Text" (absent) and literal-Russian
// Desc/WinDesc, and the DM map shipped literal Desc/WinDesc. It covers every mode's maps at once.
func TestMapCardsResolve(t *testing.T) {
	keys := validLocaleKeys(t)
	check := func(scene, field, key string) {
		if key == "" {
			t.Errorf("map %s: empty %s -> the card renders \"\"", scene, field)
			return
		}
		if !keys[key] {
			t.Errorf("map %s: %s key %q is not in the baked locale -> the card renders \"EMPTY!\"", scene, field, key)
		}
	}
	for _, m := range HuntMaps() {
		check(m.Scene, "Name", m.Name)
		check(m.Scene, "Desc", m.Desc)
		check(m.Scene, "WinDesc", m.WinDesc)
	}
	for _, m := range DotaMaps() {
		check(m.Scene, "Name", m.Name)
		check(m.Scene, "Desc", m.Desc)
		check(m.Scene, "WinDesc", m.WinDesc)
	}
	for _, m := range ArenaMaps() {
		check(m.Scene, "Name", m.Name)
		check(m.Scene, "Desc", m.Desc)
		check(m.Scene, "WinDesc", m.WinDesc)
	}
}

// TestDotaBuildingCardsResolve covers all four «Штурм» structure roles on both sides.
func TestDotaBuildingCardsResolve(t *testing.T) {
	icons, keys := validCardIcons(t), validLocaleKeys(t)
	roles := []DotaRole{DotaAltar, DotaGun, DotaCreepTower, DotaGenerator}
	for _, role := range roles {
		for _, side := range []DotaSide{DotaSideHuman, DotaSideElf} {
			name, icon := DotaBuildingDesc(role, side)
			if !keys[name] {
				t.Errorf("building role=%d side=%d name key %q is not in the baked locale", role, side, name)
			}
			if !icons[strings.ToLower(icon)] {
				t.Errorf("building role=%d side=%d icon %q does not exist in the client", role, side, icon)
			}
		}
	}
	// Every baked structure must be one of the roles covered above: a role added to the
	// map without a card entry would silently reintroduce the empty-string bug.
	for _, m := range DotaMaps() {
		for _, sc := range m.Structures {
			if name, _ := DotaBuildingDesc(sc.Role, sc.Side); name == "" {
				t.Errorf("map %d structure %d (%s) has no card entry for role=%d", m.ID, sc.ID, sc.Prefab, sc.Role)
			}
		}
	}
}

// TestSummonUnitsHaveCards is the guard that was missing: summons look their strings up
// by prefab through the mob roster, and the lookup used to fall back to empty strings on
// a miss, so the two summons whose prefab is not a mob shipped blank for months without
// a single failing test. UnitDesc reports the miss and this fails on it.
func TestSummonUnitsHaveCards(t *testing.T) {
	icons, keys := validCardIcons(t), validLocaleKeys(t)
	seen := map[string]bool{}
	for _, ks := range skillsByPrefab {
		for _, sk := range ks.Skills {
			for _, op := range sk.Ops {
				collectSummonUnits(op, seen)
			}
		}
	}
	if len(seen) == 0 {
		t.Fatal("no summon units found: the walk over the skill kit is broken, not the data")
	}
	// The contract itself, not just today's data: every real prefab resolves, so nothing
	// in the roster reaches the miss branch, and a UnitDesc that quietly reported success
	// with two empty strings -- exactly the old behaviour -- would sail through the loop
	// below. Reporting the miss is the whole point of the rewrite, so pin it directly.
	if name, icon, ok := UnitDesc("Avtr_This_Prefab_Does_Not_Exist"); ok {
		t.Errorf("UnitDesc claims strings for an unknown prefab (%q/%q): a future summon would ship blank instead of failing here", name, icon)
	}
	for unit := range seen {
		name, icon, ok := UnitDesc(unit)
		if !ok {
			t.Errorf("summon unit %q has no card strings: it would render a nameless blank card", unit)
			continue
		}
		if !keys[name] {
			t.Errorf("summon unit %q name key %q is not in the baked locale", unit, name)
		}
		if !icons[strings.ToLower(icon)] {
			t.Errorf("summon unit %q icon %q does not exist in the client", unit, icon)
		}
	}
}
