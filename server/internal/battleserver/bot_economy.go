package battleserver

// Leveling and item-shop decisions: spend a banked skill point the instant one is
// available, and greedily buy the cheapest currently-unlocked item in a role-appropriate
// pair of avatar-tree trees. Both ride the bot's always-on housekeeping (botTickLocked),
// not the think cadence, so a bot never sits on gold or an unspent point.

import "tanatserver/internal/gamedata"

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

// botChooseSkillToLevelLocked picks which slot to spend the next point on: the ultimate
// whenever it is learnable and not maxed (levelling it the instant it is available is
// close to always correct in a MOBA), else whichever of slots 1-3 is furthest BEHIND, so
// the kit fills out evenly instead of one skill maxing while the other two sit at 0.
func botChooseSkillToLevelLocked(hs *huntState) int {
	if botCanLevelLocked(hs, 4) {
		return 4
	}
	best, bestRank := 0, 999
	for slot := 1; slot <= 3; slot++ {
		if !botCanLevelLocked(hs, slot) {
			continue
		}
		if r := int(hs.skillLevel[slot-1]); r < bestRank {
			bestRank, best = r, slot
		}
	}
	return best
}

func botCanLevelLocked(hs *huntState, slot int) bool {
	cur := int(hs.skillLevel[slot-1])
	if cur >= hs.kit.Skills[slot-1].MaxRank() {
		return false
	}
	req := skillReqLevel(slot, cur)
	return req < 0 || int(hs.level) >= req
}

// botPreferredTrees maps an avatar archetype to the two item trees a bot of that
// archetype shops from -- the class fantasy each tree's own naming already implies
// (AvatarTreeDefence/Attack/Magic/Control/Support).
var botPreferredTrees = map[int32][2]int32{
	gamedata.AvatarTypeWarrior: {gamedata.AvatarTreeDefence, gamedata.AvatarTreeAttack},
	gamedata.AvatarTypeKiller:  {gamedata.AvatarTreeAttack, gamedata.AvatarTreeControl},
	gamedata.AvatarTypeMage:    {gamedata.AvatarTreeMagic, gamedata.AvatarTreeControl},
	gamedata.AvatarTypeSupport: {gamedata.AvatarTreeSupport, gamedata.AvatarTreeMagic},
}

// botBuyItemsLocked spends available gold on the cheapest currently-unlocked item (parent
// already owned, or a root) in this avatar's two preferred trees -- a simple greedy build
// order: a bot always owns the affordable frontier of its trees, walking up from the roots
// as gold allows, rather than hoarding silently for one specific expensive item.
func (s *Server) botBuyItemsLocked(b *botBrain, now float64) {
	c, hs := b.c, b.c.huntState
	trees, ok := botPreferredTrees[hs.av.Type]
	if !ok {
		trees = [2]int32{gamedata.AvatarTreeAttack, gamedata.AvatarTreeDefence}
	}
	money, _, ok := s.Store.HeroMoney(c.selfPlayerID)
	if !ok {
		return
	}
	var best gamedata.AvatarItem
	bestPrice := int32(-1)
	for _, it := range gamedata.AvatarItems() {
		if it.TreeID != trees[0] && it.TreeID != trees[1] {
			continue
		}
		if hs.ownedTreeItems[it.ArticleID] || it.Price > money {
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
		if bestPrice < 0 || it.Price < bestPrice {
			bestPrice, best = it.Price, it
		}
	}
	if bestPrice < 0 {
		return
	}
	s.buyItemLocked(c, best.ArticleID)
}
