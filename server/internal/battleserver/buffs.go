package battleserver

import (
	"strconv"

	"tanatserver/internal/gamedata"
)

// Global buffs ("руны/эликсиры") in battle: the timed hero-wide effects a player started in
// the city (session.Hero.Buffs, keyed by article) that must actually change combat/rewards.
// Two kinds, both snapshotted at match entry (they can't be used mid-hunt):
//   - RUNE (kind 15): a flat stat bonus folded into the avatar as a permanent statMod, exactly
//     like a dressed wearable (so it survives death/respawn for the match).
//   - ELIXIR (kind 18): an xp/money multiplier cached on the conn and applied when rewards are
//     granted (grantXPLocked / awardCoinsLocked). The drop multiplier is noted but not yet
//     wired (the loot-drop path is a later hook).

// runeModStat maps a rune's stat key to the battle engine's internal statMod name (the same
// mods the wearable fold and avatar tree use). An unknown key returns "" and is skipped.
func runeModStat(stat string) string {
	switch stat {
	case "attack_power":
		return "dmg_flat"
	case "health":
		return "max_hp"
	case "mana":
		return "max_mana"
	}
	return ""
}

// runeRegenModStat maps a rune's stat key to the engine's per-second regen statMod (summed into
// passive regen by mobai's tick, same as gear/potions). "" for families whose card carries no
// regen line. Kept in lockstep with gamedata.runeRegenPlaceholder so the fold matches the card.
func runeRegenModStat(stat string) string {
	switch stat {
	case "health":
		return "hp_regen"
	case "mana":
		return "mana_regen"
	}
	return ""
}

// applyGlobalBuffsLocked folds the hero's active rune stat-buffs into the avatar and caches
// the active elixir multipliers on the conn. Called at world-build, right after the dressed
// gear fold, inside the build lock. Defaults the multipliers to 1x when the hero has no
// elixir active.
func (s *Server) applyGlobalBuffsLocked(c *conn, now float64) {
	c.buffXPMult, c.buffCoinMult = 1, 1
	hs := c.huntState
	if hs == nil {
		return
	}
	for _, b := range s.Store.HeroActiveBuffs(c.selfPlayerID) {
		if r, ok := gamedata.RuneByArticle(b.ArticleID); ok {
			mod := runeModStat(r.Stat)
			if mod == "" {
				continue
			}
			hs.st.mods = append(hs.st.mods, statMod{
				stat: mod, value: r.EffectiveValue(), until: 0,
				src: "buff_" + strconv.Itoa(int(b.ArticleID)),
			})
			// S4+ Health/Mana runes also grant the flat regen their card shows (EffectiveRegen>0
			// only there); fold it as an hp_regen/mana_regen mod so the number on the card is the
			// number the player gets.
			if reg := r.EffectiveRegen(); reg > 0 {
				if rmod := runeRegenModStat(r.Stat); rmod != "" {
					hs.st.mods = append(hs.st.mods, statMod{
						stat: rmod, value: reg, until: 0,
						src: "buffreg_" + strconv.Itoa(int(b.ArticleID)),
					})
				}
			}
			continue
		}
		if e, ok := gamedata.ElixirByArticle(b.ArticleID); ok {
			switch e.Boost {
			case "xp":
				c.buffXPMult *= e.Mult
			case "money":
				c.buffCoinMult *= e.Mult
				// "drop" and "mastery" elixirs exist + grant, but their effect (loot-drop rate /
				// per-avatar mastery) is a later hook -- see [[item-types]].
			}
		}
	}
	// The rune stat-fold raised max HP/mana; top the pools up so entry starts full (mirrors
	// applyDressedItemStatsLocked, which runs just before this).
	hs.hp = hs.maxHPLocked(now)
	hs.mana = hs.maxManaLocked(now)
}

// buffCoinScale multiplies a coin reward by the hero's active money-elixir multiplier
// (defaulting an unset/zero mult to 1x), rounding to the nearest coin.
func buffCoinScale(c *conn, coins int32) int32 {
	m := c.buffCoinMult
	if m <= 0 {
		return coins
	}
	return int32(float64(coins)*m + 0.5)
}
