package gamedata

import "strconv"

// Avatar battle-tree items ("предметы аватаров") -- the in-battle, DotA-style
// item build shown in BattleItemMenu's five tabs. This is a DIFFERENT system
// from the drinkable consumables in items.go: a tree item carries
// tree_id/tree_slot/tree_parents and grants a PERMANENT per-match stat bonus
// (bought with in-battle gold, gone at match end), never a timed potion effect.
//
// Every Name/Desc/Icon below is a REAL baked client id
// (IDS_ItemAvtr_<Class>_Ln<L>_St<S>_{Name,LongDesc}, and the matching
// gui/icons/items/itemavatar/icon_itemavtr_* texture): the original client
// shipped all 60 items with professionally-written names and tooltips, so the
// server only CITES them -- GuiSystem.GetLocaleText/GetImage resolve them with
// zero server-side text. The stat NUMBERS and the tree TOPOLOGY (slots/parents/
// prices) are baked NOWHERE in the client (proven by the items dossier) -- the
// server authors them, and the params tooltip is kept numerically identical to
// the stat actually applied in battle so the two can never disagree.
//
// Client wiring the layout below is built to satisfy (all verified against the
// decompiled client, see the avatar-items dossier):
//   - The tree is a PERMANENT HUD element fed ONLY by this Ctrl catalog
//     (CtrlServerConnection.GetGroupedItems groups by Article.mTreeId, keeps
//     mTreeSlot != -1). No PShop object is needed.
//   - tree_id must be BattleItemMenu.TreeType (1..5) or the item lands in no tab.
//   - tree_slot must be 1..12 (the 3-col x 4-row grid); duplicates within one
//     tree are dropped by the client with a Log.Error.
//   - A root (empty tree_parents) is always available; a child is LOCKED until
//     its parent is OWNED. tree_parents must be an acyclic DAG (the client walks
//     it with unguarded recursion -- a cycle overflows its stack).

// Avatar item tree tabs (mirror BattleItemMenu.TreeType). The class prefix in
// the baked ids maps to a tab: Pr->DEFENCE, At->ATTACK, Mg->MAGIC, Cn->CONTROL,
// Sp->SUPPORT.
const (
	AvatarTreeDefence int32 = 1 // Pr
	AvatarTreeAttack  int32 = 2 // At
	AvatarTreeMagic   int32 = 3 // Mg
	AvatarTreeControl int32 = 4 // Cn
	AvatarTreeSupport int32 = 5 // Sp
)

// avatarItemArticleBase anchors the tree-item article/proto id range. Chosen
// clear of every other id space that reaches the same client connection: the
// potion articles (potionArticleBase=50000..50077) and their buff-proto shadow
// (itemBuffProtoBase+article = 70000..70077), plus every avatar/mob/summon/
// action proto (all < 6000). 60 items -> 60000..60059. As with potions, ONE id
// serves as BOTH the Ctrl article id and the Battle prototype id for an item.
const avatarItemArticleBase int32 = 60000

// kindAvatarItem is ShopGUI.ItemType.AVATAR (=20): FormatedTipMgr renders the
// tooltip type-label "AVATAR_ITEM_TEXT". The valid kind range is {0..16,18..25}.
const kindAvatarItem int32 = 20

// AvatarItemStat is one stat bonus on a tree item. Name is the EXACT locale
// placeholder key found inside the item's baked LongDesc (bare -- the client
// strips a trailing %/^/# before its mParams lookup). The same (Name, Value,
// Mul) triple feeds BOTH the client tooltip (emitted into items.amf params with
// impact=1 when Mul, else 0) AND the real stat applied to the avatar in battle,
// so the number a player reads is exactly the number they get.
type AvatarItemStat struct {
	Name  string
	Value float64
	Mul   bool // multiplicative (Speed): tooltip shows (value-1)*100, impact=1
}

// Impact is the items.amf "impact" value the client's PArticle.Load switches on
// (0 -> AddStat.mAdd, 1 -> AddStat.mMul).
func (s AvatarItemStat) Impact() int32 {
	if s.Mul {
		return 1
	}
	return 0
}

// AvatarItem is one authored battle-tree item.
type AvatarItem struct {
	ArticleID int32
	Class     string // At/Cn/Mg/Pr/Sp
	TreeID    int32  // AvatarTree* (BattleItemMenu.TreeType)
	Line      int    // 1..3 (grid column)
	Stage     int    // 1..5 (progression tier within the line)
	TreeSlot  int32  // 1..12 grid slot
	Parents   []int32
	NameKey   string
	DescKey   string
	Icon      string // path relative to "Gui/Icons/Items/" (client prepends it)
	Price     int32  // in-battle gold (Article.mBuyCost, checked vs VirtualMoney)
	MinAvaLvl int32
	Stats     []AvatarItemStat
}

// avatarItemStatLadder gives each stat's authored value at stage 1..5 (index 0
// unused). Additive stats are flat bonuses; Speed is a move-speed MULTIPLIER
// (1.10 = +10%) and CritChance is a fraction the client renders x100 (0.06 =
// "6%"). These are server-invented balance numbers, not baked data.
var avatarItemStatLadder = map[string][6]float64{
	"Health":      {0, 150, 220, 340, 480, 640},
	"Mana":        {0, 100, 140, 210, 300, 400},
	"DamageMin":   {0, 7, 11, 18, 28, 40},
	"AttackSpeed": {0, 0.10, 0.15, 0.24, 0.34, 0.48},
	"PhysArmor":   {0, 3, 5, 8, 12, 17},
	"MagicArmor":  {0, 3, 5, 8, 12, 17},
	"SpellPower":  {0, 10, 15, 24, 38, 54},
	"CritChance":  {0, 0.03, 0.04, 0.06, 0.09, 0.13},
	"Speed":       {0, 1.05, 1.07, 1.10, 1.14, 1.20},
}

// avatarItemPriceByStage is the gold cost per stage (index 0 unused). A build
// walks from a cheap stage-1/2 root up to an expensive stage-5 tip.
var avatarItemPriceByStage = [6]int32{0, 150, 280, 600, 1050, 1650}

// avatarTreeLines lists, per line (grid column 1..3), the stages present
// top-to-bottom (grid row order). Transcribed from the baked cells: line 2
// holds the only stage-1 item and skips stage 2; lines 1 and 3 start at stage 2.
// The 12 cells map bijectively onto the client's 3-col x 4-row grid:
// slot = 1 + (line-1) + row*3.
var avatarTreeLines = [3][4]int{
	{2, 3, 4, 5}, // line 1
	{1, 3, 4, 5}, // line 2 (stage-1 root here)
	{2, 3, 4, 5}, // line 3
}

// avatarTree is one class's tree: its baked-id prefix, its tab, and the stat
// placeholder set of each cell keyed by [line,stage]. The stat-name lists are
// transcribed 1:1 (in locale order) from each item's baked LongDesc placeholders
// -- TestAvatarItemParamsMatchPlaceholders guards this against a typo that would
// otherwise ship a tooltip param the client can't resolve.
type avatarTree struct {
	prefix string
	treeID int32
	cells  map[[2]int][]string
}

var avatarTrees = []avatarTree{
	{"Pr", AvatarTreeDefence, map[[2]int][]string{
		{1, 2}: {"Health", "MagicArmor"},
		{1, 3}: {"Health", "MagicArmor"},
		{1, 4}: {"Health", "MagicArmor"},
		{1, 5}: {"Health", "MagicArmor"},
		{2, 1}: {"Health"},
		{2, 3}: {"Health", "MagicArmor", "PhysArmor"},
		{2, 4}: {"Health", "MagicArmor", "PhysArmor"},
		{2, 5}: {"Health", "MagicArmor", "PhysArmor"},
		{3, 2}: {"Health", "PhysArmor"},
		{3, 3}: {"Health", "PhysArmor"},
		{3, 4}: {"Health", "PhysArmor"},
		{3, 5}: {"Health", "PhysArmor"},
	}},
	{"At", AvatarTreeAttack, map[[2]int][]string{
		{1, 2}: {"DamageMin", "AttackSpeed"},
		{1, 3}: {"DamageMin", "AttackSpeed", "CritChance"},
		{1, 4}: {"DamageMin", "AttackSpeed", "CritChance"},
		{1, 5}: {"DamageMin", "AttackSpeed", "CritChance"},
		{2, 1}: {"DamageMin"},
		{2, 3}: {"DamageMin"},
		{2, 4}: {"DamageMin", "AttackSpeed"},
		{2, 5}: {"DamageMin", "AttackSpeed"},
		{3, 2}: {"AttackSpeed"},
		{3, 3}: {"DamageMin", "AttackSpeed"},
		{3, 4}: {"DamageMin", "AttackSpeed"},
		{3, 5}: {"DamageMin", "AttackSpeed"},
	}},
	{"Mg", AvatarTreeMagic, map[[2]int][]string{
		{1, 2}: {"SpellPower", "DamageMin"},
		{1, 3}: {"SpellPower", "DamageMin"},
		{1, 4}: {"SpellPower", "DamageMin"},
		{1, 5}: {"SpellPower", "DamageMin"},
		{2, 1}: {"SpellPower", "Mana"},
		{2, 3}: {"SpellPower", "Mana"},
		{2, 4}: {"SpellPower", "Mana"},
		{2, 5}: {"SpellPower", "Mana"},
		{3, 2}: {"SpellPower", "Mana"},
		{3, 3}: {"SpellPower", "Mana"},
		{3, 4}: {"SpellPower", "Mana"},
		{3, 5}: {"SpellPower", "Mana"},
	}},
	{"Cn", AvatarTreeControl, map[[2]int][]string{
		{1, 2}: {"DamageMin", "Speed", "SpellPower"},
		{1, 3}: {"DamageMin", "SpellPower", "Speed"},
		{1, 4}: {"DamageMin", "SpellPower", "Speed"},
		{1, 5}: {"DamageMin", "SpellPower", "Speed"},
		{2, 1}: {"Speed"},
		{2, 3}: {"DamageMin", "AttackSpeed", "Speed"},
		{2, 4}: {"DamageMin", "AttackSpeed", "Speed"},
		{2, 5}: {"DamageMin", "AttackSpeed", "Speed"},
		{3, 2}: {"DamageMin", "Speed", "Health"},
		{3, 3}: {"DamageMin", "Speed", "Health"},
		{3, 4}: {"DamageMin", "Speed", "Health"},
		{3, 5}: {"DamageMin", "Speed", "Health"},
	}},
	{"Sp", AvatarTreeSupport, map[[2]int][]string{
		{1, 2}: {"DamageMin", "Health", "PhysArmor", "MagicArmor"},
		{1, 3}: {"DamageMin", "Health", "PhysArmor", "MagicArmor"},
		{1, 4}: {"DamageMin", "Health", "PhysArmor", "MagicArmor"},
		{1, 5}: {"DamageMin", "Health", "PhysArmor", "MagicArmor"},
		{2, 1}: {"DamageMin", "Health"},
		{2, 3}: {"DamageMin", "Health"},
		{2, 4}: {"DamageMin", "Health"},
		{2, 5}: {"DamageMin", "Health"},
		{3, 2}: {"DamageMin", "Health", "SpellPower"},
		{3, 3}: {"DamageMin", "Health", "SpellPower"},
		{3, 4}: {"DamageMin", "Health", "SpellPower"},
		{3, 5}: {"DamageMin", "Health", "SpellPower"},
	}},
}

var avatarItems []AvatarItem
var avatarItemsByArticle map[int32]AvatarItem

func init() {
	next := avatarItemArticleBase
	for _, tc := range avatarTrees {
		// Pass 1: assign a stable article id per cell (line-then-row order) so
		// parent edges can reference already-known ids.
		idByCell := map[[2]int]int32{}
		for line := 1; line <= 3; line++ {
			for row := 0; row < 4; row++ {
				stage := avatarTreeLines[line-1][row]
				idByCell[[2]int{line, stage}] = next
				next++
			}
		}
		// Pass 2: build the items.
		for line := 1; line <= 3; line++ {
			for row := 0; row < 4; row++ {
				stage := avatarTreeLines[line-1][row]
				cell := [2]int{line, stage}
				names := tc.cells[cell]
				if len(names) == 0 {
					panic("avatar_items: no stat placeholders for " + tc.prefix +
						" Ln" + strconv.Itoa(line) + "_St" + strconv.Itoa(stage))
				}
				var parents []int32
				if row > 0 {
					prevStage := avatarTreeLines[line-1][row-1]
					parents = []int32{idByCell[[2]int{line, prevStage}]}
				}
				stats := make([]AvatarItemStat, 0, len(names))
				for _, n := range names {
					ladder, ok := avatarItemStatLadder[n]
					if !ok {
						panic("avatar_items: no stat ladder for " + n)
					}
					stats = append(stats, AvatarItemStat{Name: n, Value: ladder[stage], Mul: n == "Speed"})
				}
				suffix := tc.prefix + "_Ln" + strconv.Itoa(line) + "_St" + strconv.Itoa(stage)
				avatarItems = append(avatarItems, AvatarItem{
					ArticleID: idByCell[cell],
					Class:     tc.prefix,
					TreeID:    tc.treeID,
					Line:      line,
					Stage:     stage,
					TreeSlot:  int32(1 + (line - 1) + row*3),
					Parents:   parents,
					NameKey:   "IDS_ItemAvtr_" + suffix + "_Name",
					DescKey:   "IDS_ItemAvtr_" + suffix + "_LongDesc",
					Icon:      "ItemAvatar/Icon_ItemAvtr_" + suffix,
					Price:     avatarItemPriceByStage[stage],
					MinAvaLvl: 0,
					Stats:     stats,
				})
			}
		}
	}
	avatarItemsByArticle = make(map[int32]AvatarItem, len(avatarItems))
	for _, it := range avatarItems {
		avatarItemsByArticle[it.ArticleID] = it
	}
}

// AvatarItems returns every authored battle-tree item.
func AvatarItems() []AvatarItem { return avatarItems }

// AvatarItemByArticle looks up a tree item by its article/proto id.
func AvatarItemByArticle(id int32) (AvatarItem, bool) {
	it, ok := avatarItemsByArticle[id]
	return it, ok
}

// AvatarItemKindID is the ShopGUI.ItemType the catalog tags tree items with.
func AvatarItemKindID() int32 { return kindAvatarItem }
