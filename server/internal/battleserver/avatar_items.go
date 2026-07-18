package battleserver

import (
	"math"
	"strconv"

	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
)

// Avatar battle-tree items ("предметы аватаров") -- the in-battle DotA-style item
// build shown in BattleItemMenu. This file is the Battle-channel half; the Ctrl
// catalog (tree_id/tree_slot/tree_parents/params) rides in items.amf
// (ctrlserver). Verified client wiring (see the avatar-items dossier):
//
//   - The tree needs each item's BATTLE prototype (a <PItem> with <Article>): the
//     client maps the clicked Ctrl article id to a battle proto id via
//     Battle.ArticleToProto (populated ONLY from such a prototype) and a buy for
//     an unmapped article is silently dropped. No <PShop> object is involved.
//   - Buying sends BUY{shopId:0, sellerId:0, itemId:<battle proto id>, count:1}.
//     For us the battle proto id equals the article id (one id space, like
//     potions), so itemId resolves straight back to the article.
//   - A buy is reflected by SET_MONEY (new balance -> re-check affordability),
//     ADD_TO_INVENTORY (marks the slot USED and unlocks children -- this, NOT the
//     BUY ack, feeds the client's owned set), and a stats SYNC. The BUY reply
//     itself only needs to be a success ack (the client reads itemId off its own
//     request; SelfPlayer.OnBought just logs it).
//   - The client pre-checks price <= VirtualMoney and only makes affordable items
//     clickable, so the server MUST send a real balance via SET_MONEY at battle
//     start, else every priced item is greyed out.

// avatarItemProtoDesc is the PROTOTYPE_INFO XML for one tree item: a name/desc/
// icon PDesc plus a PItem block carrying the Article (the bridge that fills
// Battle.ArticleToProto and the hover tooltip's battle-side data). No PTool: a
// tree item is never a click-to-use bag entry (tree_id>0 routes it to the item
// panel, not the normal bag), and no PShop/PDestructible: it is never a world
// object. Type=CONSUMABLE mirrors the potion path; the client routes tree items
// by the Ctrl-side IsConsumable() (mTreeId>0), not by this battle enum.
func avatarItemProtoDesc(it gamedata.AvatarItem) string {
	return `<Proto>` +
		`<PDesc>` +
		`<Name value="` + xmlEsc(it.NameKey) + `"/>` +
		`<Short value=""/><Long value="` + xmlEsc(it.DescKey) + `"/>` +
		`<Icon value="` + xmlEsc(it.Icon) + `"/>` +
		`</PDesc>` +
		`<PItem>` +
		`<Type value="CONSUMABLE"/>` +
		`<BuyCost value="` + itoa(int(it.Price)) + `"/>` +
		`<SellCost value="0"/>` +
		`<Level value="1"/>` +
		`<Article value="` + itoa(int(it.ArticleID)) + `"/>` +
		`</PItem>` +
		`</Proto>`
}

// avatarItemProtoPkts is the PROTOTYPE_INFO batch for every tree item, added to
// the world-build sequence so every buy resolves and every tooltip has data. 60
// small packets, comparable to the mob/skill proto registration already there.
func avatarItemProtoPkts() []battleproto.Packet {
	items := gamedata.AvatarItems()
	pkts := make([]battleproto.Packet, 0, len(items))
	for _, it := range items {
		pkts = append(pkts, protoInfoPkt(it.ArticleID, avatarItemProtoDesc(it)))
	}
	return pkts
}

// avatarItemModStat maps a tree-item stat placeholder to the battle engine's
// internal statMod name (status.go). Every one is applied as a PERMANENT mod
// (until=0), which playerDieLocked deliberately keeps across death/respawn, so a
// bought item stays in effect for the whole match. Three of these
// (max_mana/dmg_flat/atk_speed_flat) are flat bonuses this file also teaches the
// live-stat helpers to read.
func avatarItemModStat(name string) string {
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
	case "SpellPower":
		return "spell_power"
	case "CritChance":
		return "crit_pct"
	case "Speed":
		return "move_speed_pct"
	}
	return ""
}

func itemModSrc(article int32) string { return "item_" + strconv.Itoa(int(article)) }

// sendInitialMoneyLocked pushes the hero's current persistent gold as the
// in-battle balance, so the item tree's client-side affordability check
// (price <= VirtualMoney) has a real number to compare against. Without this,
// VirtualMoney is 0 and every priced tree item renders greyed-out/unclickable.
func (s *Server) sendInitialMoneyLocked(c *conn) {
	money, diamonds, ok := s.Store.HeroMoney(c.selfPlayerID)
	if !ok {
		return
	}
	s.push(c, battleproto.CmdSetMoney, amf.NewArray().
		Set("money", amf.NewArray().Set("v", money).Set("r", diamonds)))
}

// applyAvatarItemStatsLocked appends the item's stat bonuses as permanent mods
// and re-syncs the avatar's displayed stats. Health/Mana also top up the current
// pool by the bought amount (clamped to the raised max), so a defensive buy feels
// immediate rather than only helping after the next regen tick.
func (s *Server) applyAvatarItemStatsLocked(c *conn, it gamedata.AvatarItem, now float64) {
	hs := c.huntState
	var hpAdd, manaAdd float64
	for _, st := range it.Stats {
		modName := avatarItemModStat(st.Name)
		if modName == "" {
			continue
		}
		hs.st.mods = append(hs.st.mods, statMod{stat: modName, value: st.Value, until: 0, src: itemModSrc(it.ArticleID)})
		switch st.Name {
		case "Health":
			hpAdd += st.Value
		case "Mana":
			manaAdd += st.Value
		}
	}
	if hpAdd > 0 {
		hs.hp = math.Min(hs.maxHPLocked(now), hs.hp+hpAdd)
	}
	if manaAdd > 0 {
		hs.mana = math.Min(hs.maxManaLocked(now), hs.mana+manaAdd)
	}
	s.pushPlayerStatsLocked(c, now)
}

// handleBuy serves the Battle BUY (13) the item tree sends. It validates the
// purchase server-side (real item, not already owned, parents owned, affordable),
// atomically debits the hero's gold, marks the item owned, and reflects it with
// SET_MONEY + ADD_TO_INVENTORY + a stats SYNC, then acks the request. Any
// validation miss just acks with no state change (the client's own gates should
// already prevent it, so this is defence, not a user-facing error path).
func (s *Server) handleBuy(c *conn, p battleproto.Packet) {
	c.lock()
	defer c.unlock()
	hs := c.huntState
	if hs == nil || hs.closed {
		s.ack(c, p) // not in a battle (lobby): nothing to buy
		return
	}
	// itemId is the battle proto id; for tree items that equals the article id.
	article := p.Args.IntOr("itemId", -1)
	it, ok := gamedata.AvatarItemByArticle(article)
	if !ok {
		s.ack(c, p)
		return
	}
	if hs.ownedTreeItems[it.ArticleID] {
		s.ack(c, p) // already bought this match
		return
	}
	for _, par := range it.Parents {
		if !hs.ownedTreeItems[par] {
			s.ack(c, p) // parent not owned -> still LOCKED
			return
		}
	}
	money, diamonds, ok := s.Store.SpendHeroMoney(c.selfPlayerID, it.Price)
	if !ok {
		s.ack(c, p) // no hero, or can't afford
		return
	}

	if hs.ownedTreeItems == nil {
		hs.ownedTreeItems = map[int32]bool{}
	}
	hs.ownedTreeItems[it.ArticleID] = true
	hs.nextBagID++
	invID := hs.nextBagID
	now := float64(s.battleTime())

	// New balance so the tree re-evaluates affordability of the rest.
	s.push(c, battleproto.CmdSetMoney, amf.NewArray().
		Set("money", amf.NewArray().Set("v", money).Set("r", diamonds)).
		Set("delta", amf.NewArray().Set("v", -it.Price).Set("r", int32(0))))
	// This is what marks the slot USED and unlocks its children client-side.
	s.push(c, battleproto.CmdAddToInv, amf.NewArray().
		Set("id", invID).Set("proto", it.ArticleID).Set("count", int32(1)))
	// Permanent stat bonuses + re-sync.
	s.applyAvatarItemStatsLocked(c, it, now)
	// Success ack for the BUY request (client reads itemId off its own request).
	s.ack(c, p)
}
