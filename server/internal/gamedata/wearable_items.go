package gamedata

import (
	"fmt"
	"math"
)

// Hero-gear ("предметы героев") -- the persistent player equipment (ItemType.WEARABLE):
// the 348 baked "Set" pieces a player buys in the city shop (store|list/buy) and dresses
// onto a paperdoll slot (user|dress) for permanent, cross-match stat bonuses. This is a
// SEPARATE system from AvatarItem (the in-battle DotA tree, avatar_items.go) and from the
// consumable potions (items.go). Names/icons/descriptions come 1:1 from the client's baked
// IDS_Set_* assets (transcribed in wearable_items_gen.go); the stat VALUES come from the
// official wiki «Геройские комплекты» (wearable_stats_gen.go, keyed by color+tier+slot+stat)
// where the wiki publishes them, falling back to the synthetic wearableStatValue for the
// pieces the wiki does not cover (weapons, the Violet S5 «Каратель» set, the Grey S1 belt
// anomaly). Prices are authored here. All values are kept numerically identical to what the
// Battle server later applies (wearableModStat) and to what the client tooltip renders.

const wearableArticleBase int32 = 80000

// Wearable is one baked Set gear piece. SlotBit is the paperdoll bitmask the client sends
// in user|dress; KindID is the ShopGUI.ItemType (1..12) that drives the tooltip category and
// the slot rule. MinHeroLevel is the dress gate, read from the item's baked Name ("[N]").
type Wearable struct {
	ArticleID    int32
	Race         string // "Human" | "Elf"
	Tier         int32  // 0..7 (S0..S7)
	Color        string // Grey|Green|Blue|Violet (rarity)
	Slot         string // Helm|Body|Legs|Foots|Shoulders|Hands|Belt|Weapon
	SlotBit      int32  // 1<<n paperdoll bit
	KindID       int32  // ShopGUI.ItemType
	MinHeroLevel int32
	NameKey      string
	DescKey      string
	Icon         string
	Price        int32
	SellPrice    int32
	Stats        []AvatarItemStat
}

// setSlot maps each of the 8 Set slot tokens to its paperdoll bit and ShopGUI ItemType kind.
// HeroInfo bits: 1=Helm 2=Body 4=Legs 8=Foots 128=Shoulders 1024=Gloves 2048=Belt 4096=Weapon.
// The client never validates kind<->slot on drop, so the SERVER is the sole authority: a dress
// is legal only when the item's SlotBit equals the requested slot.
var setSlot = map[string]struct {
	Bit  int32
	Kind int32
}{
	"Helm":      {1, 1},     // ItemType.HELM
	"Body":      {2, 2},     // ItemType.ARMOR
	"Legs":      {4, 3},     // ItemType.TROUSERS
	"Foots":     {8, 4},     // ItemType.BOOTS
	"Shoulders": {128, 7},   // ItemType.SHOULDERS
	"Hands":     {1024, 10}, // ItemType.CLOTHERS (gloves)
	"Belt":      {2048, 11}, // ItemType.BELT
	"Weapon":    {4096, 12}, // ItemType.WEAPON
}

// wearableColorMul scales stat magnitude by rarity (the color frame): Grey < Green < Blue <
// Violet. Prices use a steeper curve (wearablePriceMul).
func wearableColorMul(color string) float64 {
	switch color {
	case "Green":
		return 1.25
	case "Blue":
		return 1.6
	case "Violet":
		return 2.0
	}
	return 1.0 // Grey (starter/common)
}

func wearablePriceMul(color string) float64 {
	switch color {
	case "Green":
		return 1.6
	case "Blue":
		return 2.4
	case "Violet":
		return 3.6
	}
	return 1.0 // Grey
}

// wearableStatValue is the SYNTHETIC fallback for the stat values the wiki does not publish:
// weapon pieces (the wiki lists no weapons -> DamageMin here), the Violet S5 «Каратель» set
// (wiki: coming soon) and the Grey S1 belt (its wiki row lists stats the baked item cannot
// render). It authors one stat's additive bonus (impact 0) scaling with the item's baked level
// and rarity, kept > 0 so no tooltip placeholder renders blank. Where wikiWearableStats has an
// entry, that authored wiki number is used instead (see init). The Battle server applies these
// exact numbers (wearableModStat maps each name to an engine stat). The eleven possible stat
// names are the placeholders found across all 348 baked LongDescs.
func wearableStatValue(stat string, level int32, color string) float64 {
	l := float64(level)
	cm := wearableColorMul(color)
	switch stat {
	case "Health":
		return math.Round(l * 9 * cm)
	case "Mana":
		return math.Round(l * 6 * cm)
	case "PhysArmor", "MagicArmor":
		return math.Max(1, math.Round(l*0.7*cm))
	case "AntiPhysArmor", "AntiMagicArmor":
		return math.Max(1, math.Round(l*0.5*cm))
	case "DamageMin":
		return math.Max(1, math.Round(l*1.1*cm))
	case "HealthRegen":
		return wearableRoundTo((0.5+l*0.12)*cm, 1)
	case "ManaRegen":
		return wearableRoundTo((0.4+l*0.09)*cm, 1)
	case "AttackSpeed":
		return wearableRoundTo((0.015+l*0.0035)*cm, 2)
	case "CritChance":
		return wearableRoundTo((0.01+l*0.0018)*cm, 3)
	}
	return 0
}

func wearableRoundTo(v float64, places int) float64 {
	p := math.Pow(10, float64(places))
	return math.Round(v*p) / p
}

// wearablePrice is the city buy cost: quadratic in level, scaled by rarity.
func wearablePrice(level int32, color string) int32 {
	base := 40 + level*level
	return int32(math.Round(float64(base) * wearablePriceMul(color)))
}

var (
	wearables         []Wearable
	wearableByArticle map[int32]Wearable
)

func init() {
	wearableByArticle = make(map[int32]Wearable, len(setItemDefs))
	wearables = make([]Wearable, 0, len(setItemDefs))
	for i, d := range setItemDefs {
		si, ok := setSlot[d.Slot]
		if !ok {
			panic("gamedata: wearable with unknown slot token " + d.Slot)
		}
		art := wearableArticleBase + int32(i)
		w := Wearable{
			ArticleID:    art,
			Race:         d.Race,
			Tier:         d.Tier,
			Color:        d.Color,
			Slot:         d.Slot,
			SlotBit:      si.Bit,
			KindID:       si.Kind,
			MinHeroLevel: d.Level,
			NameKey:      fmt.Sprintf("IDS_Set_%s_S%d_%s_%s_Name", d.Race, d.Tier, d.Color, d.Slot),
			DescKey:      fmt.Sprintf("IDS_Set_%s_S%d_%s_%s_LongDesc", d.Race, d.Tier, d.Color, d.Slot),
			Icon:         d.Icon,
			Price:        wearablePrice(d.Level, d.Color),
		}
		w.SellPrice = w.Price / 4
		for _, name := range d.Stats {
			// Prefer the exact value the wiki published for this piece; the synthetic formula
			// only fills the stats the wiki does not cover (weapons, Violet S5, the S1 belt).
			val, ok := wikiWearableStats[wikiStatKey{Color: d.Color, Tier: d.Tier, Slot: d.Slot, Stat: name}]
			if !ok {
				val = wearableStatValue(name, d.Level, d.Color)
			}
			w.Stats = append(w.Stats, AvatarItemStat{
				Name:  name,
				Value: val,
				Mul:   false,
			})
		}
		wearables = append(wearables, w)
		wearableByArticle[art] = w
	}
}

// Wearables returns the full baked hero-gear catalog (read-only, both races).
func Wearables() []Wearable { return wearables }

// WearableByArticle resolves a wearable by its article/proto id.
func WearableByArticle(id int32) (Wearable, bool) {
	w, ok := wearableByArticle[id]
	return w, ok
}

// IsWearableArticle reports whether id names a hero-gear article.
func IsWearableArticle(id int32) bool {
	_, ok := wearableByArticle[id]
	return ok
}

// WearableArticleBase is the first wearable article id (80000).
func WearableArticleBase() int32 { return wearableArticleBase }

// WearableRaceCode maps a wearable's Race to the client hero-race int (HeroRace: HUMAN=1, ELF=2).
func WearableRaceCode(race string) int32 {
	if race == "Elf" {
		return 2
	}
	return 1
}

// WearablesForRace returns the catalog filtered to one hero race code (1=Human, 2=Elf), so the
// city shop only offers gear the requesting hero can actually wear.
func WearablesForRace(raceCode int32) []Wearable {
	out := make([]Wearable, 0, len(wearables)/2)
	for _, w := range wearables {
		if WearableRaceCode(w.Race) == raceCode {
			out = append(out, w)
		}
	}
	return out
}
