package battleserver

// Leveling and item-shop decisions: spend a banked skill point the instant one is
// available, and greedily buy the currently-unlocked item in one of three distinct,
// role-appropriate avatar trees. A filled slot is upgraded along its branch instead
// of blocking all later progression. Both ride the bot's always-on housekeeping (botTickLocked),
// not the think cadence, so a bot never sits on gold or an unspent point.

import "tanatserver/internal/gamedata"

const botMaxItemMoveSpeedFactor = 1.35

// botItemStageLimit keeps a large starting wallet from turning into an instant
// full-tree build. Progression is tied to the avatar's current level, not to
// elapsed match time: a bot can spend freely when it has earned the level, but
// cannot buy late-game movement multipliers at spawn.
func botItemStageLimit(level int32) int {
	characterLevel := level + 1 // battle state is 0-based; catalog gates are 1-based
	switch {
	case characterLevel < 1:
		return 0
	case characterLevel < 5:
		return 1
	case characterLevel < 10:
		return 2
	case characterLevel < 15:
		return 3
	case characterLevel < 20:
		return 4
	default:
		return 5
	}
}

func botItemMoveSpeedFactorLocked(hs *huntState) float64 {
	if hs == nil {
		return 1
	}
	factor := 1.0
	for _, mod := range hs.st.mods {
		if mod.stat == "move_speed_pct" && len(mod.src) >= 5 && mod.src[:5] == "item_" {
			factor *= mod.value
		}
	}
	return factor
}

func botItemSpeedAllowedLocked(hs *huntState, it gamedata.AvatarItem) bool {
	factor := botItemMoveSpeedFactorLocked(hs)
	for _, stat := range it.Stats {
		if stat.Name == "Speed" {
			factor *= stat.Value
		}
	}
	return factor <= botMaxItemMoveSpeedFactor+1e-9
}

// botSpendSkillPointLocked spends a single banked skill point, if any, through the exact
// UPGRADE_SKILL validation a real client's request goes through.
func (s *Server) botSpendSkillPointLocked(b *botBrain) {
	c, hs := b.c, b.c.huntState
	if hs.points <= 0 {
		return
	}
	slot := botChooseSkillToLevelLocked(hs)
	if slot == 0 {
		return
	}
	s.upgradeSkillLocked(c, skillProtoID(hs.av, slot))
}

// botChooseSkillToLevelLocked uses authored operations plus the avatar role. A
// support therefore learns a heal/control tool before a damage filler, while a
// mage/killer invests in reliable damage and a warrior in survivability/damage.
// The ultimate remains a strong context-independent breakpoint when its level
// gate is live; all ties are resolved by slot number.
func botChooseSkillToLevelLocked(hs *huntState) int {
	best, bestScore := 0, -1e9
	for slot := 1; slot <= 4; slot++ {
		if !botCanLevelLocked(hs, slot) {
			continue
		}
		def := hs.skillDef(slot)
		score := botSkillRolePriority(hs.av.Type, def)
		if slot == 4 {
			score += 100
		}
		// Keep a role's core tool ahead of a neglected utility skill, but do not
		// let one skill starve forever once the core is capped.
		if max := hs.kit.Skills[slot-1].MaxRank(); max > 0 {
			score += float64(max-int(hs.skillLevel[slot-1])) * 2
		}
		if score > bestScore || (score == bestScore && (best == 0 || slot < best)) {
			bestScore, best = score, slot
		}
	}
	return best
}

func botSkillRolePriority(avatarType int32, def gamedata.Skill) float64 {
	damage := botSkillHasOpDeep(def, gamedata.OpDamage) || botSkillHasOpDeep(def, gamedata.OpExecute) || botSkillHasOpDeep(def, gamedata.OpManaScaledDamage)
	control := botSkillHasOpDeep(def, gamedata.OpStun) || botSkillHasOpDeep(def, gamedata.OpRoot) || botSkillHasOpDeep(def, gamedata.OpSilence) || botSkillHasOpDeep(def, gamedata.OpSlow)
	defence := botSkillHasOpDeep(def, gamedata.OpHeal) || botSkillHasOpDeep(def, gamedata.OpHot) || botSkillHasOpDeep(def, gamedata.OpShield) || botSkillHasOpDeep(def, gamedata.OpStealth)
	score := 0.0
	switch avatarType {
	case gamedata.AvatarTypeSupport:
		if defence {
			score += 60
		}
		if control {
			score += 32
		}
		if damage {
			score += 14
		}
	case gamedata.AvatarTypeMage:
		if damage {
			score += 58
		}
		if control {
			score += 34
		}
		if defence {
			score += 22
		}
	case gamedata.AvatarTypeKiller:
		if damage {
			score += 58
		}
		if control {
			score += 38
		}
		if defence {
			score += 18
		}
	default:
		if damage {
			score += 46
		}
		if defence {
			score += 42
		}
		if control {
			score += 28
		}
	}
	return score
}

func botSkillHasOpDeep(def gamedata.Skill, kind gamedata.OpKind) bool {
	for _, op := range def.Ops {
		if op.Kind == kind || botSkillHasNestedOp(op.Ops, kind) {
			return true
		}
	}
	return false
}

func botSkillHasNestedOp(ops []gamedata.Op, kind gamedata.OpKind) bool {
	for _, op := range ops {
		if op.Kind == kind || botSkillHasNestedOp(op.Ops, kind) {
			return true
		}
	}
	return false
}

func botCanLevelLocked(hs *huntState, slot int) bool {
	cur := int(hs.skillLevel[slot-1])
	if cur >= hs.kit.Skills[slot-1].MaxRank() {
		return false
	}
	req := skillReqLevel(slot, cur)
	return req < 0 || int(hs.level) >= req
}

// botPreferredTrees maps an avatar archetype to the three distinct item trees
// it is allowed to shop from. The server permits only one ACTIVE item per tree;
// later purchases may upgrade that slot without creating a fourth active item.
var botPreferredTrees = map[int32][3]int32{
	gamedata.AvatarTypeWarrior: {gamedata.AvatarTreeDefence, gamedata.AvatarTreeAttack, gamedata.AvatarTreeSupport},
	gamedata.AvatarTypeKiller:  {gamedata.AvatarTreeAttack, gamedata.AvatarTreeControl, gamedata.AvatarTreeDefence},
	gamedata.AvatarTypeMage:    {gamedata.AvatarTreeMagic, gamedata.AvatarTreeControl, gamedata.AvatarTreeSupport},
	gamedata.AvatarTypeSupport: {gamedata.AvatarTreeSupport, gamedata.AvatarTreeMagic, gamedata.AvatarTreeDefence},
}

// botBuyItemsLocked spends available gold on the best affordable frontier item
// in the role's preferred trees. The frontier is restricted to real unlocked
// items, including upgrades of already occupied trees. Low-health/recovery
// contexts bias toward health/armor/speed utility without inventing gold, stats,
// or a separate purchase path.
func (s *Server) botBuyItemsLocked(b *botBrain, now float64) {
	c, hs := b.c, b.c.huntState
	trees, ok := botPreferredTrees[hs.av.Type]
	if !ok {
		trees = [3]int32{gamedata.AvatarTreeAttack, gamedata.AvatarTreeDefence, gamedata.AvatarTreeSupport}
	}
	money, _, ok := s.Store.HeroMoney(c.selfPlayerID)
	if !ok {
		return
	}
	var best gamedata.AvatarItem
	bestScore := -1e9
	for _, it := range gamedata.AvatarItems() {
		if it.TreeID != trees[0] && it.TreeID != trees[1] && it.TreeID != trees[2] {
			continue
		}
		if hs.ownedTreeItems[it.ArticleID] || it.Price > money || it.MinAvaLvl > hs.level+1 || it.Stage > botItemStageLimit(hs.level) || !botItemSpeedAllowedLocked(hs, it) {
			continue
		}
		activeArticle := avatarTreeActiveLocked(hs, it.TreeID)
		if activeArticle == 0 && avatarEquippedTreeCountLocked(hs) >= gamedata.AvatarTreeMaxItems {
			continue
		}
		if activeArticle != 0 && len(it.Parents) == 0 {
			continue
		}
		locked := false
		for _, par := range it.Parents {
			if !hs.ownedTreeItems[par] {
				locked = true
				break
			}
		}
		if locked {
			continue
		}
		score := botItemRolePriority(hs.av.Type, it)
		if botItemRecoveryUtility(it) && (botHPFrac(hs, now) < botRetreatHPFrac || b.retreating) {
			score += 38
		}
		if score > bestScore || (score == bestScore && (best.ArticleID == 0 || it.Price < best.Price)) ||
			(score == bestScore && it.Price == best.Price && it.ArticleID < best.ArticleID) {
			bestScore, best = score, it
		}
	}
	if bestScore < -1e8 {
		return
	}
	s.buyItemLocked(c, best.ArticleID)
}

func botItemRecoveryUtility(it gamedata.AvatarItem) bool {
	for _, stat := range it.Stats {
		switch stat.Name {
		case "Health", "PhysArmor", "MagicArmor", "Speed":
			return true
		}
	}
	return false
}

func botItemRolePriority(avatarType int32, it gamedata.AvatarItem) float64 {
	score := float64(5 - it.Stage) // take a useful frontier step before a distant tip
	for _, stat := range it.Stats {
		weight := 0.0
		switch stat.Name {
		case "Health", "PhysArmor", "MagicArmor":
			weight = 20
		case "DamageMin", "AttackSpeed", "CritChance":
			weight = 20
		case "SpellPower", "Mana":
			weight = 20
		case "Speed":
			weight = 16
		}
		switch avatarType {
		case gamedata.AvatarTypeSupport:
			if stat.Name == "Health" || stat.Name == "Mana" || stat.Name == "SpellPower" || stat.Name == "PhysArmor" {
				weight += 12
			}
		case gamedata.AvatarTypeMage:
			if stat.Name == "SpellPower" || stat.Name == "Mana" {
				weight += 15
			}
		case gamedata.AvatarTypeKiller:
			if stat.Name == "DamageMin" || stat.Name == "AttackSpeed" || stat.Name == "CritChance" {
				weight += 15
			}
		default:
			if stat.Name == "Health" || stat.Name == "PhysArmor" || stat.Name == "DamageMin" {
				weight += 12
			}
		}
		score += weight * stat.Value
	}
	return score
}
