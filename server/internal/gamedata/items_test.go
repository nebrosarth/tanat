package gamedata

import (
	"strings"
	"testing"
)

// TestItemFamilyCounts pins the exact item count per kind, derived from the
// real shipped client's locale asset (Tanat_Data/mainData's "configs/locale"
// TextAsset) -- Health/Mana are NOT a flat 8-tier ladder, they're an
// irregular (level-bracket x rarity) grid that only has 24 real cells each
// (S1-S3 only shipped Grey+Green, S4-S5 stop at Blue).
func TestItemFamilyCounts(t *testing.T) {
	cases := []struct {
		kind ItemKind
		want int
	}{
		{ItemHealthPotion, 24},
		{ItemManaPotion, 24},
		{ItemHealthFlask, 1},
		{ItemManaFlask, 1},
		{ItemSpeedPotion, 4},
		{ItemInvisibilityPotion, 4},
		{ItemRevelationPotion, 4},
		{ItemDodgeChancePotion, 4},
		{ItemCritStrikePotion, 4},
		{ItemAntiPhysArmorPotion, 4},
		{ItemAntiMagicArmorPotion, 4},
	}
	total := 0
	for _, c := range cases {
		got := ItemsByKind(c.kind)
		if len(got) != c.want {
			t.Errorf("kind %v has %d items, want %d", c.kind, len(got), c.want)
		}
		total += c.want
	}
	if len(Items()) != total {
		t.Errorf("Items() has %d entries, want %d (sum of all kinds)", len(Items()), total)
	}
}

// TestItemArticlesUnique: every item gets a distinct catalog id (shared by the
// Ctrl bag's artikul_id and the Battle inventory's proto), and every item
// carries a non-empty name/description/icon.
func TestItemArticlesUnique(t *testing.T) {
	seen := map[int32]bool{}
	for _, it := range Items() {
		if seen[it.ArticleID] {
			t.Errorf("duplicate ArticleID %d", it.ArticleID)
		}
		seen[it.ArticleID] = true
		if it.Icon == "" {
			t.Errorf("item %+v has no icon", it)
		}
		if it.NameKey == "" {
			t.Errorf("item %+v has no name key", it)
		}
		if it.DescKey == "" {
			t.Errorf("item %+v has no description key", it)
		}
		if it.Cooldown <= 0 {
			t.Errorf("item %+v has no cooldown", it)
		}
		got, ok := ItemByArticle(it.ArticleID)
		if !ok || got.ArticleID != it.ArticleID {
			t.Errorf("ItemByArticle(%d) = %+v, %v", it.ArticleID, got, ok)
		}
	}
}

// TestItemIconIsBareName guards the blank-icon bug: every item-icon render
// site in the client (BattleItemMenu, InventoryMenu, DropMenu, ShopGUI, ...)
// prepends "Gui/Icons/Items/" to Desc.mIcon itself, so Icon must never
// re-bake that prefix (that would double-prefix the path and the client
// would silently render no icon). Icon MAY (and, for most potions, must)
// still contain its own subfolder segment below that prefix.
func TestItemIconIsBareName(t *testing.T) {
	for _, it := range Items() {
		if strings.Contains(strings.ToLower(it.Icon), "gui/icons/items") {
			t.Errorf("item %+v: Icon must not re-bake the client's own \"Gui/Icons/Items/\" prefix, got %q", it, it.Icon)
		}
	}
}

// TestItemIconPathHasPrefix guards the buff-bar icon path: unlike the bag/
// shop/drop menus (which add "Gui/Icons/Items/" themselves), BuffRenderer's
// battle-buff path passes PEffectDesc.Desc.mIcon straight to GuiSystem.GetImage
// with no prefix, so IconPath() must bake it in or the buff icon renders as
// the client's default placeholder star.
func TestItemIconPathHasPrefix(t *testing.T) {
	for _, it := range Items() {
		want := "Gui/Icons/Items/" + it.Icon
		if got := it.IconPath(); got != want {
			t.Errorf("item %+v: IconPath() = %q, want %q", it, got, want)
		}
	}
}

// TestItemNameKeysAreRealLocaleIds guards against ever reintroducing invented
// "IDS_Item_Potion_*"-style ids (the original bug that rendered as "EMPTY!"
// in the live client): every NameKey/DescKey must follow the REAL client's
// naming convention (IDS_Potion_<Family>_... / IDS_HealthPotion_.. /
// IDS_ManaPotion_..), never the old invented "IDS_Item_Potion_" prefix.
func TestItemNameKeysAreRealLocaleIds(t *testing.T) {
	for _, it := range Items() {
		if strings.HasPrefix(it.NameKey, "IDS_Item_") {
			t.Errorf("item %+v: NameKey %q looks like an invented id, not a real client locale id", it, it.NameKey)
		}
		if !strings.HasPrefix(it.NameKey, "IDS_") {
			t.Errorf("item %+v: NameKey %q doesn't look like a locale id at all", it, it.NameKey)
		}
	}
}

// TestHealthRegenRealValues pins the exact real numbers (transcribed from the
// shipped client's own locale LongDesc text) for the HealthRegen family, tier
// by tier, so a future edit can't silently drift from the authoritative
// source without a test failure pointing at the exact cell that's wrong.
func TestHealthRegenRealValues(t *testing.T) {
	want := []struct {
		nameKey            string
		amount, cd float64
	}{
		{"IDS_Potion_HealthRegen_S0_Grey_Name", 250, 40},
		{"IDS_Potion_HealthRegen_S0_Green_Name", 320, 36},
		{"IDS_Potion_HealthRegen_S0_Blue_Name", 370, 34},
		{"IDS_Potion_HealthRegen_S0_Violet_Name", 420, 30},
		{"IDS_Potion_HealthRegen_S1_Grey_Name", 370, 40},
		{"IDS_Potion_HealthRegen_S1_Green_Name", 400, 40},
		{"IDS_Potion_HealthRegen_S2_Grey_Name", 500, 40},
		{"IDS_Potion_HealthRegen_S2_Green_Name", 650, 36},
		{"IDS_Potion_HealthRegen_S3_Grey_Name", 620, 40},
		{"IDS_Potion_HealthRegen_S3_Green_Name", 800, 36},
		{"IDS_Potion_HealthRegen_S4_Grey_Name", 750, 30},
		{"IDS_Potion_HealthRegen_S4_Green_Name", 970, 26},
		{"IDS_Potion_HealthRegen_S4_Blue_Name", 1100, 24},
		{"IDS_Potion_HealthRegen_S5_Grey_Name", 870, 30},
		{"IDS_Potion_HealthRegen_S5_Green_Name", 1150, 26},
		{"IDS_Potion_HealthRegen_S5_Blue_Name", 1350, 24},
		{"IDS_Potion_HealthRegen_S6_Grey_Name", 1000, 30},
		{"IDS_Potion_HealthRegen_S6_Green_Name", 1300, 26},
		{"IDS_Potion_HealthRegen_S6_Blue_Name", 1500, 24},
		{"IDS_Potion_HealthRegen_S6_Violet_Name", 1700, 20},
		{"IDS_Potion_HealthRegen_S7_Grey_Name", 1200, 30},
		{"IDS_Potion_HealthRegen_S7_Green_Name", 1460, 26},
		{"IDS_Potion_HealthRegen_S7_Blue_Name", 1750, 24},
		{"IDS_Potion_HealthRegen_S7_Violet_Name", 1900, 20},
	}
	byKey := map[string]Item{}
	for _, it := range ItemsByKind(ItemHealthPotion) {
		byKey[it.NameKey] = it
	}
	if len(byKey) != len(want) {
		t.Fatalf("ItemsByKind(ItemHealthPotion) has %d entries, want %d", len(byKey), len(want))
	}
	for _, w := range want {
		it, ok := byKey[w.nameKey]
		if !ok {
			t.Errorf("missing HealthRegen item %q", w.nameKey)
			continue
		}
		if it.Value != w.amount || it.Cooldown != w.cd || it.Duration != 10 {
			t.Errorf("%s: got {amount:%v cd:%v dur:%v}, want {amount:%v cd:%v dur:10}", w.nameKey, it.Value, it.Cooldown, it.Duration, w.amount, w.cd)
		}
	}
}

// TestManaRegenRealValues mirrors TestHealthRegenRealValues for the
// ManaRegen family.
func TestManaRegenRealValues(t *testing.T) {
	want := []struct {
		nameKey    string
		amount, cd float64
	}{
		{"IDS_Potion_ManaRegen_S0_Grey_Name", 100, 40},
		{"IDS_Potion_ManaRegen_S0_Green_Name", 150, 36},
		{"IDS_Potion_ManaRegen_S0_Blue_Name", 200, 34},
		{"IDS_Potion_ManaRegen_S0_Violet_Name", 260, 30},
		{"IDS_Potion_ManaRegen_S1_Grey_Name", 150, 40},
		{"IDS_Potion_ManaRegen_S1_Green_Name", 180, 40},
		{"IDS_Potion_ManaRegen_S2_Grey_Name", 200, 40},
		{"IDS_Potion_ManaRegen_S2_Green_Name", 260, 36},
		{"IDS_Potion_ManaRegen_S3_Grey_Name", 250, 40},
		{"IDS_Potion_ManaRegen_S3_Green_Name", 320, 36},
		{"IDS_Potion_ManaRegen_S4_Grey_Name", 300, 30},
		{"IDS_Potion_ManaRegen_S4_Green_Name", 390, 26},
		{"IDS_Potion_ManaRegen_S4_Blue_Name", 450, 24},
		{"IDS_Potion_ManaRegen_S5_Grey_Name", 350, 30},
		{"IDS_Potion_ManaRegen_S5_Green_Name", 460, 26},
		{"IDS_Potion_ManaRegen_S5_Blue_Name", 520, 24},
		{"IDS_Potion_ManaRegen_S6_Grey_Name", 400, 30},
		{"IDS_Potion_ManaRegen_S6_Green_Name", 530, 26},
		{"IDS_Potion_ManaRegen_S6_Blue_Name", 600, 24},
		{"IDS_Potion_ManaRegen_S6_Violet_Name", 680, 20},
		{"IDS_Potion_ManaRegen_S7_Grey_Name", 450, 30},
		{"IDS_Potion_ManaRegen_S7_Green_Name", 580, 26},
		{"IDS_Potion_ManaRegen_S7_Blue_Name", 670, 24},
		{"IDS_Potion_ManaRegen_S7_Violet_Name", 760, 20},
	}
	byKey := map[string]Item{}
	for _, it := range ItemsByKind(ItemManaPotion) {
		byKey[it.NameKey] = it
	}
	if len(byKey) != len(want) {
		t.Fatalf("ItemsByKind(ItemManaPotion) has %d entries, want %d", len(byKey), len(want))
	}
	for _, w := range want {
		it, ok := byKey[w.nameKey]
		if !ok {
			t.Errorf("missing ManaRegen item %q", w.nameKey)
			continue
		}
		if it.Value != w.amount || it.Cooldown != w.cd || it.Duration != 10 {
			t.Errorf("%s: got {amount:%v cd:%v dur:%v}, want {amount:%v cd:%v dur:10}", w.nameKey, it.Value, it.Cooldown, it.Duration, w.amount, w.cd)
		}
	}
}

// TestFourTierFamiliesRealValues pins the exact real numbers for every
// 4-rarity-tier family (Speed/Invisibility/Revelation/Dodge/Crit/
// AntiPhysArmor/AntiMagicArmor), all of which share a flat 150s cooldown.
func TestFourTierFamiliesRealValues(t *testing.T) {
	type want struct {
		value, value2, duration float64
	}
	cases := []struct {
		kind  ItemKind
		tiers [4]want
	}{
		{ItemSpeedPotion, [4]want{{0.15, 0, 10}, {0.25, 0, 12}, {0.50, 0, 14}, {0.55, 0, 16}}},
		{ItemInvisibilityPotion, [4]want{{0, 0, 10}, {0, 0, 15}, {0, 0, 25}, {0, 0, 30}}},
		{ItemRevelationPotion, [4]want{{0, 0, 10}, {0, 0, 15}, {0, 0, 25}, {0, 0, 30}}},
		{ItemDodgeChancePotion, [4]want{{0.10, 0, 10}, {0.15, 0, 12}, {0.30, 0, 14}, {0.35, 0, 16}}},
		{ItemCritStrikePotion, [4]want{{0.06, 0.10, 10}, {0.10, 0.25, 12}, {0.20, 0.30, 14}, {0.25, 0.35, 16}}},
		{ItemAntiPhysArmorPotion, [4]want{{6, 0, 10}, {10, 0, 12}, {20, 0, 14}, {25, 0, 16}}},
		{ItemAntiMagicArmorPotion, [4]want{{6, 0, 10}, {10, 0, 12}, {20, 0, 14}, {25, 0, 16}}},
	}
	for _, c := range cases {
		tiers := ItemsByKind(c.kind)
		if len(tiers) != 4 {
			t.Fatalf("kind %v has %d tiers, want 4", c.kind, len(tiers))
		}
		for i, it := range tiers {
			w := c.tiers[i]
			if it.Value != w.value || it.Value2 != w.value2 || it.Duration != w.duration || it.Cooldown != 150 {
				t.Errorf("kind %v tier %d: got {value:%v value2:%v dur:%v cd:%v}, want {value:%v value2:%v dur:%v cd:150}",
					c.kind, i+1, it.Value, it.Value2, it.Duration, it.Cooldown, w.value, w.value2, w.duration)
			}
			if !strings.HasSuffix(it.Icon, "_"+rarityColor[i]) {
				t.Errorf("kind %v tier %d: icon %q doesn't end with rarity color %q", c.kind, i+1, it.Icon, rarityColor[i])
			}
		}
	}
}

// TestFlaskRealValues pins the simple Health/Mana flask numbers.
func TestFlaskRealValues(t *testing.T) {
	h := ItemsByKind(ItemHealthFlask)
	if len(h) != 1 || h[0].Value != 120 || h[0].Duration != 10 || h[0].Cooldown != 40 {
		t.Errorf("HealthFlask = %+v, want {Value:120 Duration:10 Cooldown:40}", h)
	}
	m := ItemsByKind(ItemManaFlask)
	if len(m) != 1 || m[0].Value != 60 || m[0].Duration != 10 || m[0].Cooldown != 40 {
		t.Errorf("ManaFlask = %+v, want {Value:60 Duration:10 Cooldown:40}", m)
	}
}
