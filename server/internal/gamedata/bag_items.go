package gamedata

// Extended bag items ("прочие предметы"): the non-wearable, non-avatar-tree catalog
// items that live in the hero's consumable bag alongside potions -- RUNES (kind 15),
// ELIXIRS (kind 18) and mastery elixirs (kind 24), FORTUNE CHESTS (kind 21), and TOTEMS
// (kind 16). Each is a plain PArticle in /xml/items.amf (title/short/long/icon + the shop
// PArticle fields) whose per-family BEHAVIOR is driven by the ShopGUI.ItemType kind_id and
// the optional "action" field, NOT by a distinct message shape (see the research dossier).
//
// This file owns the shared model + the article-id ranges + the unified ResolveBagArticle
// lookup that the Ctrl catalog, the admin grant, and the common|action handler all key off,
// so a new family is added by transcribing its data (a *_gen.go file that calls the
// register* funcs from init) without touching the plumbing. Names/descs/icons are the REAL
// baked client locale ids (transcribed, never invented); only chest LOOT weights and the
// buff MAGNITUDES the locale leaves as placeholders are authored here.

// ShopGUI.ItemType kind ids (the PArticle kind_id, drives the tooltip/shop category and,
// for us, the server-side behavior family). POTION(19) + the wearable slots (1..12) live in
// their own files; these are the ones this file introduces.
const (
	KindTotem   int32 = 16
	KindRune    int32 = 15
	KindElixir  int32 = 18
	KindChest   int32 = 21
	KindMastery int32 = 24
)

// Article-id ranges for the new families. Chosen to sit ABOVE the potion range (50000..~50077)
// AND the avatar battle-tree range (avatarItemArticleBase 60000..~60xxx), and BELOW the
// wearable range (80000+), never overlapping each other -- so a bag article id maps to exactly
// one family (asserted by TestBagArticleRangesDisjoint, which also cross-checks the potion,
// avatar-tree and wearable ranges). Each family gets a generous 1000-id window.
const (
	runeArticleBase   int32 = 70000
	elixirArticleBase int32 = 71000
	chestArticleBase  int32 = 72000
	totemArticleBase  int32 = 73000
	bagArticleTop     int32 = 74000 // exclusive upper bound of the extended-bag range
)

// ---- Rune (kind 15): a timed hero-wide flat-stat buff consumable ----

// Rune is one stat rune. Stat is an avatar stat-field name (Health/Mana/DmgMin.. or the
// engine's own key) the buff adds Value to for Duration seconds while active. The buff is
// hero-wide (applies to whichever avatar the hero takes into battle), started by using the
// rune out of combat.
type Rune struct {
	ArticleID int32
	Infix     string // key infix, e.g. "AttackPower_S1_Grey"
	NameKey   string
	DescKey   string
	Icon      string
	Tier      int32
	Color     string
	Stat      string  // buff stat key
	Value     float64 // flat bonus applied while active
	Duration  int64   // seconds
}

// EffectiveValue returns the rune's flat stat bonus. It is the faithful literal value when the
// LongDesc shipped one (S1-S3); for the higher tiers whose text uses a {DamageMin} placeholder
// (transcribed as 0), it SYNTHESIZES a tier/rarity-scaled value so a Violet rune still does
// something, in the ballpark of the real low-tier numbers (~11-24). Synthetic and
// clearly-derived -- the real numbers were never in the locale.
func (r Rune) EffectiveValue() float64 {
	if r.Value > 0 {
		return r.Value // literal-desc tiers (S1-S3): the faithful number from the locale
	}
	// S4+ descs use a {placeholder}; the real value lived in the original server's add_stats and
	// is unrecoverable, so synthesize ANCHORED on the real S1-S3 literals rather than a flat
	// formula. The old 8+4*tier ignored per-family scale, so a Health rune rendered +24 next to
	// the S3 literal's +180. Instead: a Grey baseline that continues the real Grey curve, a rarity
	// multiplier fit to the real Green/Grey ratio (~1.3), and the exact cross-family law the
	// literals reveal -- Health = 10x AttackPower, Mana ~= 4x (holds on every S1-S3 anchor). Keeps
	// S4-S7 monotonic above S3 and believable on the card.
	return runeSynthValue(r.Tier, r.Color, r.Stat)
}

// runeGreyBaseline is the Grey (base-rarity) value ladder in AttackPower units per tier: the real
// AttackPower Grey literals for S1-S3 (11/15/18), continued +3/tier for the placeholder tiers
// S4-S7. Index by tier (1..7); index 0 unused.
var runeGreyBaseline = [8]float64{0, 11, 15, 18, 21, 24, 27, 30}

// runeSynthValue is the anchored synthetic primary value for a placeholder-desc rune: the Grey
// baseline for its tier, scaled by rarity then by the family's cross-stat multiplier. Rounded to
// a whole number to match the integer literals.
func runeSynthValue(tier int32, color, stat string) float64 {
	base := runeGreyBaseline[len(runeGreyBaseline)-1]
	if tier >= 1 && int(tier) < len(runeGreyBaseline) {
		base = runeGreyBaseline[tier]
	}
	switch color {
	case "Green":
		base *= 1.3
	case "Blue":
		base *= 1.6
	case "Violet":
		base *= 2.0
	}
	switch stat {
	case "health":
		base *= 10.0
	case "mana":
		base *= 4.0
		// attack_power keeps the x1 base unit.
	}
	return float64(int(base + 0.5))
}

// hasLiteralDesc reports whether this rune's LongDesc shipped a literal number (the S1-S3 tiers)
// rather than a {placeholder} the client substitutes at render time (S4+). The generator encodes
// exactly this: it transcribed the literal into Value for literal-desc runes and left Value==0
// where the desc used a placeholder (see the runes_gen.go header). A literal-desc rune needs no
// items.amf params -- its card already reads the number straight from the text -- so this also
// gates CardParams/EffectiveRegen off for those tiers.
func (r Rune) hasLiteralDesc() bool { return r.Value != 0 }

// runePrimaryPlaceholder maps a rune's engine stat key to the PRIMARY {placeholder} token its
// client LongDesc uses. Verified against the real locale (locale_TRUE.xml): health->{Health},
// mana->{Mana}, attack_power->{DamageMin}. "" for an unmodeled family.
func runePrimaryPlaceholder(stat string) string {
	switch stat {
	case "health":
		return "Health"
	case "mana":
		return "Mana"
	case "attack_power":
		return "DamageMin"
	}
	return ""
}

// runeRegenPlaceholder returns the SECONDARY regen {placeholder} a rune's LongDesc carries, or ""
// for families that have none. The S4+ Health/Mana runes show a second "+{HealthRegen}/
// {ManaRegen} восстановление" line; AttackPower runes have only the one placeholder. A rune's
// card lists EVERY placeholder, so both must be supplied or the client renders the missing one
// as "-1" (see FormatedTipMgr.GetItemFormatedText).
func runeRegenPlaceholder(stat string) string {
	switch stat {
	case "health":
		return "HealthRegen"
	case "mana":
		return "ManaRegen"
	}
	return ""
}

// EffectiveRegen synthesizes the flat regen bonus behind {HealthRegen}/{ManaRegen} (per-second,
// the same unit the battle engine sums into passive regen -- gear pieces sit around 0.04-0.09).
// 0 for AttackPower runes and for every literal-desc tier (S1-S3, whose card states regen as a
// prose multiplier, not a flat placeholder -- so nothing is shown OR applied for them). Like
// EffectiveValue this is a server-authored number: the real per-item regen lived only in the
// original server's add_stats and is unrecoverable. Kept modest and tier/rarity-scaled so the
// number the card shows is exactly the number the buff applies.
func (r Rune) EffectiveRegen() float64 {
	if r.hasLiteralDesc() || runeRegenPlaceholder(r.Stat) == "" {
		return 0
	}
	base := 0.05 * float64(r.Tier)
	switch r.Color {
	case "Green":
		base *= 1.3
	case "Blue":
		base *= 1.7
	case "Violet":
		base *= 2.2
	}
	return float64(int(base*100+0.5)) / 100 // 2 decimals, matches the client's "0.##" render
}

// RuneParam is one {placeholder -> value} the client card substitutes, emitted into items.amf
// "params" as {skill_id:Key, impact:0(ADD), value:Value}.
type RuneParam struct {
	Key   string
	Value float64
}

// CardParams returns the tooltip params a rune's card needs so the client substitutes every
// {placeholder} in its LongDesc with a real number instead of printing the raw token. Empty for
// literal-desc runes (S1-S3): their card already shows literals, so shipping params would only
// add a spurious "Параметры" header. For S4+ it lists the primary stat (EffectiveValue) plus,
// for Health/Mana, the regen (EffectiveRegen) -- the SAME values applyGlobalBuffs folds into the
// avatar, so display == effect (mirrors the avatar-tree item contract).
func (r Rune) CardParams() []RuneParam {
	if r.hasLiteralDesc() {
		return nil
	}
	var out []RuneParam
	if k := runePrimaryPlaceholder(r.Stat); k != "" {
		out = append(out, RuneParam{Key: k, Value: r.EffectiveValue()})
	}
	if k := runeRegenPlaceholder(r.Stat); k != "" {
		out = append(out, RuneParam{Key: k, Value: r.EffectiveRegen()})
	}
	return out
}

// ---- Elixir (kind 18) / mastery elixir (kind 24): a timed multiplier buff ----

// Elixir is one booster elixir. Boost picks which reward it multiplies (xp|money|drop|
// avatar_xp|mastery); Mult is the FINAL multiplier transcribed from the real LongDesc (an
// "×2" elixir is Mult 2.0, a "+50%" one is 1.5). Mastery elixirs (kind 24) are transcribed
// here too so they exist + grant, but their per-avatar application is deferred (see the
// memory notes); the hero-wide xp/money/drop elixirs are the ones wired to an effect.
type Elixir struct {
	ArticleID int32
	Infix     string
	NameKey   string
	DescKey   string
	Icon      string
	Kind      int32 // KindElixir or KindMastery
	Boost     string
	Mult      float64
	Duration  int64
}

// ---- Chest (kind 21): an openable loot box ----

// ChestDrop is one weighted entry of a chest's loot table. Exactly one of {CoinMax>0,
// Article>0} is set. A roll picks one entry by Weight, then a coin roll in [CoinMin,CoinMax]
// or an item count in [CntMin,CntMax]. Loot tables are AUTHORED (see chest_loot.go) from the
// chest's grade/color, since the client ships chest names/icons but no loot numbers.
type ChestDrop struct {
	Weight  int
	CoinMin int32
	CoinMax int32
	Article int32
	CntMin  int32
	CntMax  int32
}

// Chest is one fortune chest. Grade (1..3) + Color (rarity) come from the real key and drive
// the authored loot table (chestLoot). Avatar is non-empty for the two avatar-chests (their
// reward is premium coins for now; avatar unlock is deferred).
type Chest struct {
	ArticleID int32
	Infix     string
	NameKey   string
	DescKey   string
	Icon      string
	Grade     int32
	Color     string
	Avatar    string // "" for a normal loot chest; avatar name for an avatar-chest
}

// ---- Totem (kind 16): a battle ward (catalog-only for now) ----

// Totem is one deployable ward. Modeled in the catalog with its real name/icon so it exists
// and can be granted; actual deployment (a battle vision/reveal subsystem) is deferred.
type Totem struct {
	ArticleID int32
	Infix     string
	NameKey   string
	DescKey   string
	Icon      string
	Family    string
	Color     string
	Duration  int64
}

// ---- registries (populated by the transcribed *_gen.go files via init) ----

var (
	runeCatalog   []Rune
	elixirCatalog []Elixir
	chestCatalog  []Chest
	totemCatalog  []Totem

	runeByArticle   = map[int32]Rune{}
	elixirByArticle = map[int32]Elixir{}
	chestByArticle  = map[int32]Chest{}
	totemByArticle  = map[int32]Totem{}
)

// register* append one transcribed item to its catalog AND index it by article, from the
// generated data files' init(). Insertion-time indexing keeps the maps correct regardless of
// init() ordering across files.
func registerRune(r Rune)   { runeCatalog = append(runeCatalog, r); runeByArticle[r.ArticleID] = r }
func registerElixir(e Elixir) {
	elixirCatalog = append(elixirCatalog, e)
	elixirByArticle[e.ArticleID] = e
}
func registerChest(c Chest) { chestCatalog = append(chestCatalog, c); chestByArticle[c.ArticleID] = c }
func registerTotem(t Totem) { totemCatalog = append(totemCatalog, t); totemByArticle[t.ArticleID] = t }

// Accessors (stable order = registration order = article-id order).
func Runes() []Rune     { return runeCatalog }
func Elixirs() []Elixir { return elixirCatalog }
func Chests() []Chest   { return chestCatalog }
func Totems() []Totem   { return totemCatalog }

func RuneByArticle(id int32) (Rune, bool)     { r, ok := runeByArticle[id]; return r, ok }
func ElixirByArticle(id int32) (Elixir, bool) { e, ok := elixirByArticle[id]; return e, ok }
func ChestByArticle(id int32) (Chest, bool)   { c, ok := chestByArticle[id]; return c, ok }
func TotemByArticle(id int32) (Totem, bool)   { t, ok := totemByArticle[id]; return t, ok }

// ---- unified bag-article resolver ----

// BagFamily classifies a grantable bag article for the callers that must treat every family
// uniformly (admin grant, catalog emit, common|action dispatch, bag rendering).
type BagFamily int

const (
	BagPotion BagFamily = iota
	BagRune
	BagElixir
	BagChest
	BagTotem
)

// BagArticle is the resolved, family-agnostic view of one bag article.
type BagArticle struct {
	Family    BagFamily
	ArticleID int32
	KindID    int32 // ShopGUI.ItemType
	NameKey   string
	DescKey   string
	Icon      string
	MinLevel  int32
	Openable  bool // chest -> gets an "open" action button
	Usable    bool // rune/elixir -> gets a "use" action button
}

// ResolveBagArticle returns the family-agnostic descriptor for any grantable bag article
// (potion, rune, elixir, chest, totem). ok=false for an unknown id (e.g. a wearable, which is
// an owned INSTANCE, not a bag stack -- resolve those with WearableByArticle). This is the one
// lookup the admin grant + city use/open handlers switch on.
func ResolveBagArticle(id int32) (BagArticle, bool) {
	if it, ok := ItemByArticle(id); ok { // potion
		return BagArticle{Family: BagPotion, ArticleID: id, KindID: ctrlItemKindPotion,
			NameKey: it.NameKey, DescKey: it.DescKey, Icon: it.Icon, MinLevel: 1, Usable: true}, true
	}
	if r, ok := runeByArticle[id]; ok {
		return BagArticle{Family: BagRune, ArticleID: id, KindID: KindRune,
			NameKey: r.NameKey, DescKey: r.DescKey, Icon: r.Icon, MinLevel: 1, Usable: true}, true
	}
	if e, ok := elixirByArticle[id]; ok {
		return BagArticle{Family: BagElixir, ArticleID: id, KindID: e.Kind,
			NameKey: e.NameKey, DescKey: e.DescKey, Icon: e.Icon, MinLevel: 1, Usable: e.Kind == KindElixir}, true
	}
	if c, ok := chestByArticle[id]; ok {
		return BagArticle{Family: BagChest, ArticleID: id, KindID: KindChest,
			NameKey: c.NameKey, DescKey: c.DescKey, Icon: c.Icon, MinLevel: 1, Openable: true}, true
	}
	if t, ok := totemByArticle[id]; ok {
		return BagArticle{Family: BagTotem, ArticleID: id, KindID: KindTotem,
			NameKey: t.NameKey, DescKey: t.DescKey, Icon: t.Icon, MinLevel: 1}, true
	}
	return BagArticle{}, false
}

// ctrlItemKindPotion mirrors the ctrlserver constant (ShopGUI.ItemType.POTION = 19); kept
// here too so the resolver can label a potion without importing ctrlserver.
const ctrlItemKindPotion int32 = 19
