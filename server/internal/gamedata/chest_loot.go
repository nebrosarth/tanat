package gamedata

import "math/rand"

// Fortune-chest loot. The client ships chest NAMES/ICONS but no loot numbers (contents are
// unauthored), so the reward tables here are AUTHORED balance: a chest always pays coins
// scaled by its grade (1..3) and rarity color (Grey<Green<Blue<Violet), and with a
// grade-dependent chance drops one bonus consumable drawn from the real catalogs (potions
// for small chests, runes/elixirs for bigger ones). Avatar-chests pay a premium coin purse
// for now (the avatar-unlock they originally granted is deferred). Rolling is server-side
// and authoritative -- the client sends no reward choice (see the research dossier).

// GrantedItem is one item reward (article + count) from a chest roll.
type GrantedItem struct {
	Article int32
	Count   int32
}

// ChestReward is the full outcome of opening one chest: coins plus zero or more bonus items.
type ChestReward struct {
	Coins int32
	Items []GrantedItem
}

// chestColorMul scales the coin purse by rarity.
func chestColorMul(color string) float64 {
	switch color {
	case "Green":
		return 1.6
	case "Blue":
		return 2.6
	case "Violet":
		return 4.2
	}
	return 1.0 // Grey / unknown
}

// chestGradeBase returns the (base, span) bronze-coin range for a grade before the color
// multiplier, and the probability of an additional bonus item.
func chestGradeBase(grade int32) (base, span int32, itemChance float64) {
	switch grade {
	case 3:
		return 800, 1200, 0.85
	case 2:
		return 250, 350, 0.55
	default: // grade 1
		return 60, 120, 0.30
	}
}

// RollChest rolls one chest's reward with the provided RNG (so callers/tests control the
// seed). Coins are always granted; a bonus item is added with grade-scaled probability.
func RollChest(c Chest, r *rand.Rand) ChestReward {
	base, span, itemChance := chestGradeBase(c.Grade)
	mul := chestColorMul(c.Color)
	if c.Avatar != "" {
		mul *= 2.0 // avatar-chest premium (stands in for the deferred avatar unlock)
	}
	coins := int32(float64(base+int32(r.Intn(int(span)+1))) * mul)
	rew := ChestReward{Coins: coins}
	if r.Float64() < itemChance {
		if it, ok := chestBonusItem(c.Grade, r); ok {
			rew.Items = append(rew.Items, it)
		}
	}
	return rew
}

// chestBonusItem picks one bonus consumable appropriate to the chest grade from the real
// catalogs: grade 1 -> a potion; grade 2 -> potion or rune; grade 3 -> rune or elixir. ok is
// false only if the relevant catalog is somehow empty (then the chest just pays coins).
func chestBonusItem(grade int32, r *rand.Rand) (GrantedItem, bool) {
	pick := func(pool []int32) (GrantedItem, bool) {
		if len(pool) == 0 {
			return GrantedItem{}, false
		}
		return GrantedItem{Article: pool[r.Intn(len(pool))], Count: 1}, true
	}
	potions := articleIDs(len(items), func(i int) int32 { return items[i].ArticleID })
	runes := articleIDs(len(runeCatalog), func(i int) int32 { return runeCatalog[i].ArticleID })
	elixirs := articleIDs(len(elixirCatalog), func(i int) int32 { return elixirCatalog[i].ArticleID })
	switch grade {
	case 3:
		if len(elixirs) > 0 && r.Float64() < 0.5 {
			return pick(elixirs)
		}
		if g, ok := pick(runes); ok {
			return g, true
		}
		return pick(potions)
	case 2:
		if len(runes) > 0 && r.Float64() < 0.5 {
			return pick(runes)
		}
		return pick(potions)
	default:
		return pick(potions)
	}
}

// articleIDs collects the article ids of a catalog via an index accessor.
func articleIDs(n int, get func(int) int32) []int32 {
	out := make([]int32, n)
	for i := 0; i < n; i++ {
		out[i] = get(i)
	}
	return out
}
