package battleserver

import (
	"log"
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

// avatarItemEquipArgs is the client-side active-item notification. The decompiled
// ItemEquipedArgParser reads the exact wire keys id/item/proto/equip; the client then
// passes proto to Player.Equip, which is what populates Player.ActiveItems for the
// avatar card. Tree articles are also the battle prototype ids in our catalog.
func avatarItemEquipArgs(ownerObjID, articleID int32, equip bool) *amf.MixedArray {
	return amf.NewArray().
		Set("id", ownerObjID).
		Set("item", articleID).
		Set("proto", articleID).
		Set("equip", equip)
}

// sendAvatarItemStateToViewerLocked delivers one active-item state transition
// to a viewer. The client keeps Player.ActiveItems as a list, so upgrades must
// explicitly unequip the old article before equipping the replacement; sending
// only the new article would create a fourth visible slot after an upgrade.
func (s *Server) sendAvatarItemStateToViewerLocked(viewer, owner *conn, articleID int32, equip bool) {
	if viewer == nil || owner == nil || viewer.huntState == nil || owner.huntState == nil {
		return
	}
	if viewer != owner && viewer.huntState.tr.index(owner.objID) < 0 {
		return
	}
	if viewer.huntState.sentAvatarItems == nil {
		viewer.huntState.sentAvatarItems = map[int32]map[int32]bool{}
	}
	sent := viewer.huntState.sentAvatarItems[owner.objID]
	if sent == nil {
		sent = map[int32]bool{}
		viewer.huntState.sentAvatarItems[owner.objID] = sent
	}
	if equip {
		if sent[articleID] {
			return
		}
		s.push(viewer, battleproto.CmdItemEquip, avatarItemEquipArgs(owner.objID, articleID, true))
		sent[articleID] = true
		return
	}
	if !sent[articleID] {
		return
	}
	s.push(viewer, battleproto.CmdItemEquip, avatarItemEquipArgs(owner.objID, articleID, false))
	delete(sent, articleID)
}

// sendAvatarItemToViewerLocked is the idempotent equip/replay wrapper retained
// for callers and tests that only need to introduce an active article.
func (s *Server) sendAvatarItemToViewerLocked(viewer, owner *conn, articleID int32) {
	s.sendAvatarItemStateToViewerLocked(viewer, owner, articleID, true)
}

// pushAvatarItemEquipAllLocked is the live-purchase path. It mirrors
// pushAvatarAllLocked's owner-plus-currently-rendering-viewers fan-out while
// retaining the per-viewer de-duplication used by late reveals.
func (s *Server) pushAvatarItemEquipAllLocked(owner *conn, articleID int32) {
	if owner == nil {
		return
	}
	for _, viewer := range owner.members() {
		s.sendAvatarItemToViewerLocked(viewer, owner, articleID)
	}
}

func (s *Server) pushAvatarItemUnequipAllLocked(owner *conn, articleID int32) {
	if owner == nil {
		return
	}
	for _, viewer := range owner.members() {
		s.sendAvatarItemStateToViewerLocked(viewer, owner, articleID, false)
	}
}

// replayAvatarItemsToViewerLocked restores any not-yet-delivered active
// battle-tree items when an avatar is introduced or re-created on a client.
// ADD_TO_INVENTORY is owner-local; ITEM_EQUIP is the shared-world packet that
// fills the remote Player.ActiveItems list used by EnemyInfoWindow.
func (s *Server) replayAvatarItemsToViewerLocked(viewer, owner *conn) {
	if viewer == nil || owner == nil || owner.huntState == nil {
		return
	}
	if len(owner.huntState.activeTreeItems) > 0 {
		for _, articleID := range owner.huntState.activeTreeItems {
			s.sendAvatarItemToViewerLocked(viewer, owner, articleID)
		}
		return
	}
	// Compatibility for older test fixtures that seed ownedTreeItems directly.
	// Production sessions populate activeTreeItems on the first purchase.
	for _, it := range gamedata.AvatarItems() {
		if owner.huntState.ownedTreeItems[it.ArticleID] {
			s.sendAvatarItemToViewerLocked(viewer, owner, it.ArticleID)
		}
	}
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
	s.buyItemLocked(c, p.Args.IntOr("itemId", -1))
	// Any validation miss just acks with no state change (the client's own gates should
	// already prevent it, so this is defence, not a user-facing error path) -- buyItemLocked
	// reports success/failure only for a bot's benefit; the wire reply is unconditional.
	s.ack(c, p)
}

// buyItemLocked is BUY's validated core, split out of handleBuy so a bot (bot.go) can
// spend gold on an avatar-tree item through the exact same rules (real item, not already
// owned, parents owned, affordable) a real client's request is checked against. article is
// the battle proto id, which for tree items equals the Ctrl catalog article id. Reports
// whether the purchase happened. Caller holds the lock.
func (s *Server) buyItemLocked(c *conn, article int32) bool {
	hs := c.huntState
	if hs == nil || hs.closed {
		return false // not in a battle (lobby): nothing to buy
	}
	it, ok := gamedata.AvatarItemByArticle(article)
	if !ok {
		return false
	}
	if hs.ownedTreeItems[it.ArticleID] {
		return false // already bought this match
	}
	if it.MinAvaLvl > hs.level+1 {
		return false // catalog uses the displayed 1-based avatar level
	}
	activeArticle := avatarTreeActiveLocked(hs, it.TreeID)
	if activeArticle == it.ArticleID {
		return false // already the active item in this slot
	}
	if activeArticle == 0 && avatarEquippedTreeCountLocked(hs) >= gamedata.AvatarTreeMaxItems {
		return false // the client has three active battle-item slots
	}
	if activeArticle != 0 && len(it.Parents) == 0 {
		return false // an occupied tree can only move forward/up a branch
	}
	for _, par := range it.Parents {
		if !hs.ownedTreeItems[par] {
			return false // parent not owned -> still LOCKED
		}
	}
	money, diamonds, ok := s.Store.SpendHeroMoney(c.selfPlayerID, it.Price)
	if !ok {
		return false // no hero, or can't afford
	}
	moneyBefore := money + it.Price
	now := float64(s.battleTime())

	if hs.ownedTreeItems == nil {
		hs.ownedTreeItems = map[int32]bool{}
	}
	if hs.activeTreeItems == nil {
		hs.activeTreeItems = map[int32]int32{}
	}
	upgradedFrom := activeArticle
	if upgradedFrom != 0 {
		// The catalog values are total values for the current tier, not deltas.
		// Remove the old active item's permanent mods before applying the new tier.
		s.replaceAvatarItemStatsLocked(c, upgradedFrom, it, now)
		s.pushAvatarItemUnequipAllLocked(c, upgradedFrom)
	} else {
		s.applyAvatarItemStatsLocked(c, it, now)
	}
	hs.ownedTreeItems[it.ArticleID] = true
	hs.activeTreeItems[it.TreeID] = it.ArticleID
	hs.nextBagID++
	invID := hs.nextBagID

	// New balance so the tree re-evaluates affordability of the rest.
	s.push(c, battleproto.CmdSetMoney, amf.NewArray().
		Set("money", amf.NewArray().Set("v", money).Set("r", diamonds)).
		Set("delta", amf.NewArray().Set("v", -it.Price).Set("r", int32(0))))
	// This is what marks the slot USED and unlocks its children client-side.
	s.push(c, battleproto.CmdAddToInv, amf.NewArray().
		Set("id", invID).Set("proto", it.ArticleID).Set("count", int32(1)))
	// ADD_TO_INVENTORY updates only the buyer's own bag. ITEM_EQUIP updates the
	// avatar's ActiveItems list on the owner and every client that renders this avatar.
	s.pushAvatarItemEquipAllLocked(c, it.ArticleID)
	if isBotConn(c) {
		log.Printf("battle: bot %d bought avatar item article=%d tree=%d stage=%d price=%d money=%d->%d upgrade_from=%d", c.objID, it.ArticleID, it.TreeID, it.Stage, it.Price, moneyBefore, money, upgradedFrom)
		s.telemetryRecordBotPurchaseLocked(c, it, moneyBefore, money, now)
	}
	return true
}

// replaceAvatarItemStatsLocked changes one active slot from old to new. Item
// stats are authored as the complete value of a tier, so only the difference
// in Health/Mana tops up the current pools instead of stacking every tier.
func (s *Server) replaceAvatarItemStatsLocked(c *conn, oldArticle int32, newItem gamedata.AvatarItem, now float64) {
	hs := c.huntState
	oldItem, ok := gamedata.AvatarItemByArticle(oldArticle)
	if !ok {
		s.applyAvatarItemStatsLocked(c, newItem, now)
		return
	}
	removePermanentItemModsBySrcLocked(hs, itemModSrc(oldArticle))
	var hpDelta, manaDelta float64
	for _, st := range newItem.Stats {
		modName := avatarItemModStat(st.Name)
		if modName == "" {
			continue
		}
		hs.st.mods = append(hs.st.mods, statMod{stat: modName, value: st.Value, until: 0, src: itemModSrc(newItem.ArticleID)})
		switch st.Name {
		case "Health":
			hpDelta += st.Value - avatarItemStatValue(oldItem, "Health")
		case "Mana":
			manaDelta += st.Value - avatarItemStatValue(oldItem, "Mana")
		}
	}
	if hpDelta > 0 {
		hs.hp = math.Min(hs.maxHPLocked(now), hs.hp+hpDelta)
	}
	if manaDelta > 0 {
		hs.mana = math.Min(hs.maxManaLocked(now), hs.mana+manaDelta)
	}
	s.pushPlayerStatsLocked(c, now)
}

func removePermanentItemModsBySrcLocked(hs *huntState, src string) {
	if hs == nil {
		return
	}
	keep := hs.st.mods[:0]
	for _, mod := range hs.st.mods {
		if mod.src != src {
			keep = append(keep, mod)
		}
	}
	hs.st.mods = keep
}

func avatarItemStatValue(it gamedata.AvatarItem, name string) float64 {
	for _, st := range it.Stats {
		if st.Name == name {
			return st.Value
		}
	}
	return 0
}

// avatarTreeOwnedLocked reports whether this avatar already owns an item from
// treeID. The caller holds the hunt connection lock. Tree membership is derived
// from the authoritative article catalog, so the rule also covers old state
// reconstructed from a saved/test fixture.
func avatarTreeOwnedLocked(hs *huntState, treeID int32) bool {
	return avatarTreeActiveLocked(hs, treeID) != 0
}

func avatarTreeActiveLocked(hs *huntState, treeID int32) int32 {
	if hs == nil {
		return 0
	}
	if hs.activeTreeItems != nil {
		if article := hs.activeTreeItems[treeID]; article != 0 {
			return article
		}
	}
	// Legacy fixtures may only seed ownedTreeItems. Choose the highest purchased
	// tier as the active article so the upgrade path remains deterministic.
	var best gamedata.AvatarItem
	for article, owned := range hs.ownedTreeItems {
		if !owned {
			continue
		}
		it, ok := gamedata.AvatarItemByArticle(article)
		if !ok || it.TreeID != treeID {
			continue
		}
		if best.ArticleID == 0 || it.Stage > best.Stage || (it.Stage == best.Stage && it.ArticleID > best.ArticleID) {
			best = it
		}
	}
	return best.ArticleID
}

func avatarEquippedTreeCountLocked(hs *huntState) int {
	if hs == nil {
		return 0
	}
	if hs.activeTreeItems != nil {
		return len(hs.activeTreeItems)
	}
	trees := map[int32]bool{}
	for article, owned := range hs.ownedTreeItems {
		if !owned {
			continue
		}
		if it, ok := gamedata.AvatarItemByArticle(article); ok {
			trees[it.TreeID] = true
		}
	}
	return len(trees)
}
