package battleserver

import (
	"strconv"

	"tanatserver/internal/gamedata"
)

// Hero gear ("предметы героев") in battle: the persistent WEARABLE Set pieces a player
// dressed in the city (session.Hero.Dressed) fold their stats into the battle avatar at
// world-build, so equipped gear actually affects combat in every mode (Hunt/Arena/Штурм,
// which all share sendHuntWorldState). Unlike the avatar tree, gear is NOT bought in
// battle -- it is already worn when the match starts, so there is only an apply-at-entry
// path, no BUY handler.

// wearableModStat maps a wearable's stat-placeholder name to the battle engine's internal
// statMod name. Nine of the eleven are combat-LIVE: max_hp/max_mana/dmg_flat/atk_speed_flat/
// phys_armor/crit_pct (shared with the avatar tree), hp_regen/mana_regen (summed by the
// per-tick regen, like the Health/Mana flasks) and phys_armor_pen (penetrates a target's
// physical armor on the damage path, like the AntiPhysArmor elixir). The other two --
// magic_armor and magic_armor_pen -- are applied and shown on the character sheet but
// currently mitigate no damage: the engine has only physical-armor mitigation (armorMitigation
// over phys armor), with no magic-damage type yet (see hunt.go's magic_armor_pen note). This
// matches the avatar tree's identical MagicArmor behaviour; both go live automatically if magic
// mitigation is ever added. An unmapped name returns "" and is skipped.
func wearableModStat(name string) string {
	switch name {
	case "Health":
		return "max_hp"
	case "Mana":
		return "max_mana"
	case "DamageMin":
		return "dmg_flat"
	case "AttackSpeed":
		return "atk_speed_flat"
	case "PhysArmor":
		return "phys_armor"
	case "MagicArmor":
		return "magic_armor"
	case "CritChance":
		return "crit_pct"
	case "HealthRegen":
		return "hp_regen"
	case "ManaRegen":
		return "mana_regen"
	case "AntiPhysArmor":
		return "phys_armor_pen"
	case "AntiMagicArmor":
		return "magic_armor_pen"
	}
	return ""
}

// applyDressedItemStatsLocked folds every worn wearable's stats into the avatar as PERMANENT
// mods (until=0, which playerDieLocked keeps across death/respawn, so gear lasts the whole
// match) and refills hp/mana to the raised maxima so the player starts full. The mod src is
// "dress_<article>" -- distinct from the tree items' "item_<id>" so the two never alias.
// Called inside the world-build lock (before the stat packets are built) so the sim sees the
// boosted pools immediately; a pushPlayerStatsLocked later carries the display values.
func (s *Server) applyDressedItemStatsLocked(c *conn, now float64) {
	hs := c.huntState
	if hs == nil {
		return
	}
	for _, di := range s.Store.HeroDressed(c.selfPlayerID) {
		w, ok := gamedata.WearableByArticle(di.ArticleID)
		if !ok {
			continue
		}
		src := "dress_" + strconv.Itoa(int(di.ArticleID))
		for _, st := range w.Stats {
			mod := wearableModStat(st.Name)
			if mod == "" {
				continue
			}
			hs.st.mods = append(hs.st.mods, statMod{stat: mod, value: st.Value, until: 0, src: src})
		}
	}
	hs.hp = hs.maxHPLocked(now)
	hs.mana = hs.maxManaLocked(now)
}
