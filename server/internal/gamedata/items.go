package gamedata

import "strconv"

// Consumable items (potions/flasks). Every NameKey/DescKey below is a REAL
// locale id already baked into the shipped client's own locale asset
// (Tanat_Data/mainData's "configs/locale" TextAsset, read via UnityPy --
// Resources.Load-based, see LocaleState/AssetLoader.ReadText). We do not
// invent display text: GuiSystem.GetLocaleText resolves these ids to real,
// professionally-written Russian names/descriptions with zero client editing
// needed, because the original devs' potion catalog was simply never used by
// this Go server before (a much earlier attempt invented its own fake
// "IDS_Item_Potion_*" ids, which the client's locale table has never heard
// of, rendering as the literal string "EMPTY!").
//
// Balance numbers (Value/Duration/Cooldown) are likewise the REAL numbers
// transcribed from that same locale text's LongDesc lines, not invented --
// e.g. IDS_Potion_HealthRegen_S0_Grey_LongDesc literally reads "Восстановление
// 250 здоровья в течение 10 секунд. Перезарядка 40 секунд." Icon paths are
// separately verified against Tanat_Data/mainData's ResourceManager.m_Container
// lookup table (the real Resources.Load path index), not just the texture's
// bare name -- see the "neutral/potion/" note below.

// ItemKind identifies a consumable's effect family.
type ItemKind int

const (
	ItemHealthPotion ItemKind = iota
	ItemManaPotion
	ItemHealthFlask
	ItemManaFlask
	ItemSpeedPotion
	ItemInvisibilityPotion
	ItemRevelationPotion
	ItemDodgeChancePotion
	ItemCritStrikePotion
	ItemAntiPhysArmorPotion
	ItemAntiMagicArmorPotion
)

// Item is an authored consumable definition. ArticleID is the stable catalog
// id used as both the Ctrl-channel bag's artikul_id and the Battle-channel
// inventory/prototype id -- one id space for the same item everywhere.
type Item struct {
	ArticleID int32
	Kind      ItemKind
	Tier      int // 1-based position within this Kind's authored list (see
	// the HealthPotion/ManaPotion doc below for why this is NOT a strict
	// potency ladder for those two kinds specifically)
	NameKey string // real client locale id for the display name
	DescKey string // real client locale id for the long description
	Icon    string // asset path relative to "Gui/Icons/Items/" (client prepends that itself)

	// Value is interpreted per Kind:
	//   Health/Mana potion & flask: flat HP/mana restored over Duration (a
	//                               heal/mana-over-time, not instant).
	//   Speed potion:               move-speed multiplier bonus (0.15 = +15%).
	//   Dodge potion:               dodge-chance bonus (0.10 = +10%).
	//   Crit potion:                crit-CHANCE bonus (Value2 is the paired
	//                               crit-DAMAGE bonus -- see stat "crit_dmg_pct").
	//   AntiPhysArmor/AntiMagicArmor: flat armor-penetration bonus against the
	//                               target's armor. Currently INERT in this
	//                               PvE-only hunt mode: mobs carry no armor
	//                               stat for a "phys_armor_pen"/"magic_armor_pen"
	//                               mod to reduce (see mobai.go's hit-mob damage
	//                               path, which applies damage directly with no
	//                               armor mitigation for mobs). Forward-compatible
	//                               infrastructure, not a functional no-op bug.
	//   Invisibility/Revelation:    unused (0); Duration is all that matters.
	Value float64
	// Value2 is a second Kind-specific number; only CritStrikePotion uses it
	// (crit-DAMAGE bonus, added on top of the fixed 1.5x crit multiplier --
	// see hunt.go's "dmg *= 1.5 + crit_dmg_pct" basic-attack crit roll).
	Value2 float64
	// Duration is how long the (buff/HoT) effect lasts, in seconds. Always 10
	// for every Health/Mana potion and flask in the real data.
	Duration float64
	// Cooldown is how long, in seconds, before THIS SPECIFIC item can be used
	// again -- a real per-item value (30-40s for Health/Mana, scaling down at
	// higher level brackets; a flat 150s for every other family), replacing an
	// earlier simplification that shared one flat 15s cooldown across every
	// potion kind.
	Cooldown float64
}

// potionArticleBase anchors the consumable catalog's article/proto id range.
// Deliberately far above any avatar's effector-proto range
// (battleserver.effBase(a) = 1000 + a.ID*100, currently topping out at 5000
// for Avrora, avatar id 40) -- these ids are pushed to the SAME client
// connection via PROTOTYPE_INFO, so a shared id between an avatar's own
// skill/buff prototype and an item's catalog prototype would have the second
// registration silently clobber the first on the client. See
// TestAvatarEffectorProtosDontCollideWithItemArticles (battleserver) for the
// standing invariant this must keep satisfying as the avatar roster grows.
const potionArticleBase int32 = 50000

var items []Item
var itemsByArticle map[int32]Item

// rarityColor is the quality/rarity axis shared by every 4-tier potion family
// (Speed/Invisibility/Revelation/DodgeChance/CritStrike/AntiPhysArmor/
// AntiMagicArmor), in REAL ascending-potency order: Grey < Green < Blue <
// Violet (an earlier attempt had this as Grey<Blue<Green<Violet, which was
// wrong -- confirmed by the real LongDesc numbers, e.g. MovementSpeed goes
// +15%/+25%/+50%/+55% for Grey/Green/Blue/Violet respectively).
var rarityColor = [4]string{"Grey", "Green", "Blue", "Violet"}

func addItem(kind ItemKind, tier int, nameKey, descKey, icon string, value, value2, duration, cooldown float64) {
	items = append(items, Item{
		ArticleID: potionArticleBase + int32(len(items)),
		Kind:      kind, Tier: tier, NameKey: nameKey, DescKey: descKey, Icon: icon,
		Value: value, Value2: value2, Duration: duration, Cooldown: cooldown,
	})
}

// regenTier is one real (level-bracket, rarity) cell of the HealthRegen/
// ManaRegen grid. The shipped game did NOT give every level bracket all 4
// rarities: S1-S3 only ever got Grey+Green, S4-S5 stop at Blue, and only
// S0/S6/S7 have the full Grey-Green-Blue-Violet spread. Every combo listed
// below is confirmed to have BOTH a resolvable name/description (the real
// locale text) AND a resolvable icon (the real ResourceManager.m_Container
// entry) -- i.e. this is not a guess at which combos "should" exist, it's
// exactly the set the original client actually shipped.
type regenTier struct {
	sTier     int
	colors    []string
	amounts   []float64 // one per color, same order as colors
	cooldowns []float64 // one per color, same order as colors
}

// addRegenFamily authors one HealthRegen/ManaRegen-shaped family (health/mana
// potions), flattening the irregular tier x rarity grid into Item entries.
// NOTE: Tier here is just "1-based position in this flattened list", NOT a
// strict potency ladder -- e.g. S0_Violet (420 hp) is stronger than S1_Grey
// (370 hp) despite S1 sorting after S0, because rarity and level-bracket are
// two different real axes being flattened into one list for the drop table
// (drops.go picks uniformly among a kind's items, which doesn't care about
// ordering). Use TestItem*RealValues (items_test.go) to verify exact numbers
// instead of a monotonic-tier invariant.
func addRegenFamily(kind ItemKind, family string, tiers []regenTier) {
	tierNum := 0
	for _, t := range tiers {
		suf := "S" + strconv.Itoa(t.sTier)
		for i, color := range t.colors {
			tierNum++
			addItem(kind, tierNum,
				"IDS_Potion_"+family+"_"+suf+"_"+color+"_Name",
				"IDS_Potion_"+family+"_"+suf+"_"+color+"_LongDesc",
				"neutral/potion/Potion_"+family+"_"+suf+"_"+color,
				t.amounts[i], 0, 10, t.cooldowns[i])
		}
	}
}

// rarityTier is one cell of a 4-tier (rarity-only) potion family.
type rarityTier struct {
	color                    string
	value, value2, duration float64
}

// addRarityFamily authors one 4-tier (Grey/Green/Blue/Violet) family sharing
// one flat cooldown across every tier (real data: always 150s for these).
func addRarityFamily(kind ItemKind, family string, tiers []rarityTier, cooldown float64) {
	for i, t := range tiers {
		addItem(kind, i+1,
			"IDS_Potion_"+family+"_"+t.color+"_Name",
			"IDS_Potion_"+family+"_"+t.color+"_LongDesc",
			"neutral/potion/Potion_"+family+"_"+t.color,
			t.value, t.value2, t.duration, cooldown)
	}
}

func init() {
	// Health/Mana regen potions: heal/restore-over-10-seconds, real per-tier
	// cooldown (40s at low level brackets, tapering to 30s at high ones).
	addRegenFamily(ItemHealthPotion, "HealthRegen", []regenTier{
		{0, []string{"Grey", "Green", "Blue", "Violet"}, []float64{250, 320, 370, 420}, []float64{40, 36, 34, 30}},
		{1, []string{"Grey", "Green"}, []float64{370, 400}, []float64{40, 40}},
		{2, []string{"Grey", "Green"}, []float64{500, 650}, []float64{40, 36}},
		{3, []string{"Grey", "Green"}, []float64{620, 800}, []float64{40, 36}},
		{4, []string{"Grey", "Green", "Blue"}, []float64{750, 970, 1100}, []float64{30, 26, 24}},
		{5, []string{"Grey", "Green", "Blue"}, []float64{870, 1150, 1350}, []float64{30, 26, 24}},
		{6, []string{"Grey", "Green", "Blue", "Violet"}, []float64{1000, 1300, 1500, 1700}, []float64{30, 26, 24, 20}},
		{7, []string{"Grey", "Green", "Blue", "Violet"}, []float64{1200, 1460, 1750, 1900}, []float64{30, 26, 24, 20}},
	})
	addRegenFamily(ItemManaPotion, "ManaRegen", []regenTier{
		{0, []string{"Grey", "Green", "Blue", "Violet"}, []float64{100, 150, 200, 260}, []float64{40, 36, 34, 30}},
		{1, []string{"Grey", "Green"}, []float64{150, 180}, []float64{40, 40}},
		{2, []string{"Grey", "Green"}, []float64{200, 260}, []float64{40, 36}},
		{3, []string{"Grey", "Green"}, []float64{250, 320}, []float64{40, 36}},
		{4, []string{"Grey", "Green", "Blue"}, []float64{300, 390, 450}, []float64{30, 26, 24}},
		{5, []string{"Grey", "Green", "Blue"}, []float64{350, 460, 520}, []float64{30, 26, 24}},
		{6, []string{"Grey", "Green", "Blue", "Violet"}, []float64{400, 530, 600, 680}, []float64{30, 26, 24, 20}},
		{7, []string{"Grey", "Green", "Blue", "Violet"}, []float64{450, 580, 670, 760}, []float64{30, 26, 24, 20}},
	})

	// Simple health/mana "flasks" -- an older, single-tier consumable
	// distinct from the Potion_*Regen_* family (real locale uses "Колба"
	// [flask] rather than "Зелье" [potion/brew] and a plain "IDS_HealthPotion_
	// Name"/"IDS_ManaPotion_Name" id with no family/tier suffix at all).
	addItem(ItemHealthFlask, 1, "IDS_HealthPotion_Name", "IDS_HealthPotion_LongDesc", "icon_healthpotion", 120, 0, 10, 40)
	addItem(ItemManaFlask, 1, "IDS_ManaPotion_Name", "IDS_ManaPotion_LongDesc", "icon_manapotion", 60, 0, 10, 40)

	// Speed potions: +15/25/50/55% move speed for 10/12/14/16s, 150s cooldown.
	addRarityFamily(ItemSpeedPotion, "MovementSpeed", []rarityTier{
		{"Grey", 0.15, 0, 10}, {"Green", 0.25, 0, 12}, {"Blue", 0.50, 0, 14}, {"Violet", 0.55, 0, 16},
	}, 150)

	// Invisibility potions: 10/15/25/30s of stealth from mob aggro, 150s cooldown.
	addRarityFamily(ItemInvisibilityPotion, "Invisibility", []rarityTier{
		{"Grey", 0, 0, 10}, {"Green", 0, 0, 15}, {"Blue", 0, 0, 25}, {"Violet", 0, 0, 30},
	}, 150)

	// Revelation potions: 10/15/25/30s of seeing invisible enemies, 150s
	// cooldown. Currently a no-op beyond its buff icon/timer in this co-op
	// PvE hunt mode -- there are no enemy avatars to reveal and mobs have no
	// invisibility of their own; forward-compatible for whenever PvP or
	// invisible mobs exist.
	addRarityFamily(ItemRevelationPotion, "Revelation", []rarityTier{
		{"Grey", 0, 0, 10}, {"Green", 0, 0, 15}, {"Blue", 0, 0, 25}, {"Violet", 0, 0, 30},
	}, 150)

	// Dodge-chance potions: +10/15/30/35% dodge for 10/12/14/16s, 150s cooldown.
	addRarityFamily(ItemDodgeChancePotion, "DodgeChance", []rarityTier{
		{"Grey", 0.10, 0, 10}, {"Green", 0.15, 0, 12}, {"Blue", 0.30, 0, 14}, {"Violet", 0.35, 0, 16},
	}, 150)

	// Crit-strike potions: +chance/+damage on crits for 10/12/14/16s, 150s cooldown.
	addRarityFamily(ItemCritStrikePotion, "CritStrike", []rarityTier{
		{"Grey", 0.06, 0.10, 10}, {"Green", 0.10, 0.25, 12}, {"Blue", 0.20, 0.30, 14}, {"Violet", 0.25, 0.35, 16},
	}, 150)

	// Armor-penetration potions (physical/magic): flat +6/10/20/25 penetration
	// for 10/12/14/16s, 150s cooldown.
	addRarityFamily(ItemAntiPhysArmorPotion, "AntiPhysArmor", []rarityTier{
		{"Grey", 6, 0, 10}, {"Green", 10, 0, 12}, {"Blue", 20, 0, 14}, {"Violet", 25, 0, 16},
	}, 150)
	addRarityFamily(ItemAntiMagicArmorPotion, "AntiMagicArmor", []rarityTier{
		{"Grey", 6, 0, 10}, {"Green", 10, 0, 12}, {"Blue", 20, 0, 14}, {"Violet", 25, 0, 16},
	}, 150)

	itemsByArticle = make(map[int32]Item, len(items))
	for _, it := range items {
		itemsByArticle[it.ArticleID] = it
	}
}

// Items returns every authored consumable definition.
func Items() []Item { return items }

// ItemByArticle looks up a consumable by its catalog article id.
func ItemByArticle(articleID int32) (Item, bool) {
	it, ok := itemsByArticle[articleID]
	return it, ok
}

// IconPath is the full path GuiSystem.GetImage expects wherever the client
// does NOT prepend "Gui/Icons/Items/" itself -- unlike the bag/shop/drop menus
// (which read Icon bare and add that prefix on their own, see the Icon field
// doc), the battle buff-bar (BuffRenderer.CreateBattleBuffBtn) passes
// PEffectDesc.Desc.mIcon straight to GuiSystem.GetImage with no prefix at all,
// so a potion's BUFF-type effector prototype needs this baked-in form instead.
func (it Item) IconPath() string {
	return "Gui/Icons/Items/" + it.Icon
}

// ItemsByKind returns every tier of one consumable family, in authored order.
func ItemsByKind(kind ItemKind) []Item {
	var out []Item
	for _, it := range items {
		if it.Kind == kind {
			out = append(out, it)
		}
	}
	return out
}
