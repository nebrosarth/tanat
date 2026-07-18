package battleserver

import (
	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
)

// Consumable potions/flasks. A potion is drunk via DO_ACTION with action=-1
// (SelfPlayer.UseItem's wire contract -- see handleDoAction's "action == -1"
// case), decrements the player's persisted bag, and applies its effect.
// Real per-item cooldowns (huntState.itemCooldownUntil, keyed by article id)
// replace an earlier v1 simplification that shared one flat 15s cooldown
// across every potion kind -- the real client's own locale text gives each
// item its own cooldown (30-40s for Health/Mana, tapering by level bracket;
// a flat 150s for every buff-type potion).

const (
	// itemUseActionProtoID is the shared "action" prototype every consumable's
	// <PTool> points at (see itemProtoDesc). One proto suffices for all of them:
	// every potion is a plain instant self-cast with no target/point selection.
	// Free ids: trap anchor=900, drop chest=901 (drops.go).
	itemUseActionProtoID int32 = 908

	// itemBuffProtoBase anchors the per-item buff-icon proto id space (see
	// itemBuffProtoID) -- deliberately far past every other range in use
	// (summon protos 800-804, trap anchor/drop chest/action proto 900-908,
	// and every avatar's effBase(a)=1000+a.ID*100 effector range) so the full
	// 78-item catalog can never collide with any of them. Since it's an
	// offset ADDED to article (gamedata.potionArticleBase=50000+), the actual
	// wire ids land at 70000+ -- comfortably clear of everything else too.
	itemBuffProtoBase int32 = 20000
)

// itemBuffProtoID is the buff-bar prototype id for one SPECIFIC item's own
// timed effect -- one per item, not one per Kind, so e.g. a Health Potion (I)
// and a Health Potion (VIII) each show their own real name/icon on the buff
// bar instead of a shared "tier-1" placeholder.
func itemBuffProtoID(article int32) int32 { return itemBuffProtoBase + article }

// itemUseActionProtoDesc is a bare PEffectDesc with an empty target/targeting
// enum (mask 0 = TargetValidator.IsNoneTarget), which is what makes
// PlayerControl.ActivateAbility fire the item instantly on click instead of
// arming a target-selection cursor that never gets satisfied. See the
// itemProtoDesc doc for why <PTool> itself is mandatory, not just this proto.
func itemUseActionProtoDesc() string {
	return effectProtoDesc("", "", "", "ACTIVE",
		attrEnum("target", "")+attrEnum("targeting", "NONE")+
			attrItem("distance", "0")+attrItem("aoeRadius", "0")+attrItem("aoeWidth", "0"))
}

// potionBuffProtoDesc is a bare BUFF-type effector prototype for a potion's
// buff-bar icon (no per-avatar tooltip tokens -- potions aren't skills).
// nameKey is a real client locale id (see items.go), not an invented one.
// icon must be the item's IconPath() (full "Gui/Icons/Items/..." path) --
// BuffRenderer's battle-buff path (unlike the bag/shop/drop menus) passes
// this string straight to GuiSystem.GetImage with no prefix of its own, so an
// empty or bare icon here renders as the client's default placeholder star.
func potionBuffProtoDesc(nameKey, icon string) string {
	return effectProtoDesc(nameKey, "", icon, "BUFF", "")
}

// itemProtoDesc is the PROTOTYPE_INFO XML for a consumable article: a
// name/desc/icon PDesc plus a PItem block. It is never instantiated as a
// world object itself (only referenced by proto id from
// ADD_TO_INVENTORY/GET_DROP_INFO), so it carries no PPrefab/PDestructible.
//
// Name/Long carry the item's REAL client locale ids (gamedata.Item.NameKey/
// DescKey) -- GuiSystem.GetLocaleText resolves these against the client's own
// baked locale table with zero server-side text needed. PItem is NOT optional
// decoration either: DropMenu.SetData (Assembly-CSharp\DropMenu.cs:100-115)
// looks up the prototype by proto id and reads battlePrototype.Item.mArticle
// to build each loot button -- omitting <PItem> left BattlePrototype.Item
// null, so every dropped item was silently skipped (Log.Warning("No battle
// data item...") + continue), making every loot chest look empty even though
// the server-side drop itself was correct.
//
// <PTool> is likewise mandatory, not decoration: clicking a bag item calls
// PlayerControl.SetActiveItem -> CreateActiveItem, which reads
// _item.BattleProto.Tool.mAction and returns null (silently, no error, no
// packet ever sent to the server) if Tool is absent or mAction is 0 --
// exactly "клика — не тратится, лечения нет" with zero server-side symptom to
// chase, since the client never even attempts DO_ACTION. Action points at
// itemUseActionProtoID, a single shared self-cast PEffectDesc (every potion
// is an instant no-target use), mirroring how skills reference their own
// PEffectDesc child proto.
func itemProtoDesc(it gamedata.Item) string {
	return `<Proto>` +
		`<PDesc>` +
		`<Name value="` + xmlEsc(it.NameKey) + `"/>` +
		`<Short value=""/><Long value="` + xmlEsc(it.DescKey) + `"/>` +
		`<Icon value="` + xmlEsc(it.Icon) + `"/>` +
		`</PDesc>` +
		`<PItem>` +
		`<Type value="CONSUMABLE"/>` +
		`<BuyCost value="0"/>` +
		`<SellCost value="0"/>` +
		`<Level value="1"/>` +
		`<Article value="` + itoa(int(it.ArticleID)) + `"/>` +
		`</PItem>` +
		`<PTool><Action value="` + itoa(int(itemUseActionProtoID)) + `"/></PTool>` +
		`</Proto>`
}

// ensureItemBuffProtoLocked lazily PROTOTYPE_INFOs an item's OWN buff-bar
// prototype to c the first time this session shows it (mirrors
// ensureItemProtoLocked's lazy-registration pattern for the item's bag
// entry). One dedicated proto id per item, keyed off its ArticleID via
// itemBuffProtoID, instead of one shared id per Kind -- see itemBuffProtoID's
// doc for why that matters (tier/rarity must be visible on the buff bar).
func (s *Server) ensureItemBuffProtoLocked(c *conn, it gamedata.Item) int32 {
	hs := c.huntState
	protoID := itemBuffProtoID(it.ArticleID)
	if hs.sentBuffProtos == nil {
		hs.sentBuffProtos = map[int32]bool{}
	}
	if !hs.sentBuffProtos[protoID] {
		hs.sentBuffProtos[protoID] = true
		s.push(c, battleproto.CmdPrototypeInfo, amf.NewArray().
			Set("id", protoID).Set("desc", potionBuffProtoDesc(it.NameKey, it.IconPath())))
	}
	return protoID
}

// ensureItemProtoLocked PROTOTYPE_INFOs an article id to c the first time this
// session references it (owned at world-build, or freshly looted), so the
// client can resolve its name/icon before any ADD_TO_INVENTORY/GET_DROP_INFO
// naming it arrives.
func (s *Server) ensureItemProtoLocked(c *conn, article int32) {
	hs := c.huntState
	if hs.sentItemProtos == nil {
		hs.sentItemProtos = map[int32]bool{}
	}
	if hs.sentItemProtos[article] {
		return
	}
	hs.sentItemProtos[article] = true
	it, ok := gamedata.ItemByArticle(article)
	if !ok {
		return
	}
	s.push(c, battleproto.CmdPrototypeInfo, amf.NewArray().
		Set("id", article).Set("desc", itemProtoDesc(it)))
}

// sendInitialBagLocked registers this hero's persisted consumable stacks as
// Battle-channel inventory (one PROTOTYPE_INFO + ADD_TO_INVENTORY per owned
// article), so a hunt session starts with the same bag the lobby shows.
func (s *Server) sendInitialBagLocked(c *conn) {
	hs := c.huntState
	hs.bag = map[int32]int32{}
	hs.bagItemID = map[int32]int32{}
	hs.bagArticleByID = map[int32]int32{}
	for _, bi := range s.Store.HeroBag(c.selfPlayerID) {
		if bi.Count <= 0 {
			continue
		}
		s.ensureItemProtoLocked(c, bi.ArticleID)
		hs.nextBagID++
		id := hs.nextBagID
		hs.bagItemID[bi.ArticleID] = id
		hs.bagArticleByID[id] = bi.ArticleID
		hs.bag[bi.ArticleID] = bi.Count
		s.push(c, battleproto.CmdAddToInv, amf.NewArray().
			Set("id", id).Set("proto", bi.ArticleID).Set("count", bi.Count))
	}
}

// grantItemLocked credits delta of article to the picker's persistent bag and
// pushes ADD_TO_INVENTORY (delta semantics: BattleInventory increments an
// existing stack by whatever count is sent, it is not an absolute total).
// Shared by loot pickup and (indirectly, via sendInitialBagLocked) session start.
func (s *Server) grantItemLocked(c *conn, article, delta int32) {
	if delta <= 0 {
		return
	}
	hs := c.huntState
	if hs.bag == nil {
		hs.bag = map[int32]int32{}
		hs.bagItemID = map[int32]int32{}
		hs.bagArticleByID = map[int32]int32{}
	}
	s.Store.AddBagItem(c.selfPlayerID, article, delta)
	s.ensureItemProtoLocked(c, article)
	id, existed := hs.bagItemID[article]
	if !existed {
		hs.nextBagID++
		id = hs.nextBagID
		hs.bagItemID[article] = id
		hs.bagArticleByID[id] = article
	}
	hs.bag[article] += delta
	s.push(c, battleproto.CmdAddToInv, amf.NewArray().
		Set("id", id).Set("proto", article).Set("count", delta))
}

// useItemLocked drinks the bag stack behind wireID (the Battle-channel
// inventory id DO_ACTION's "id" field named), applying its effect and
// consuming one unit -- a no-op if the id/stack/cooldown don't check out
// (stale client state, empty stack, still on cooldown).
func (s *Server) useItemLocked(c *conn, wireID int32, now float64) {
	hs := c.huntState
	article, ok := hs.bagArticleByID[wireID]
	if !ok || hs.bag[article] <= 0 {
		return
	}
	it, ok := gamedata.ItemByArticle(article)
	if !ok {
		return
	}
	if hs.itemCooldownUntil == nil {
		hs.itemCooldownUntil = map[int32]float64{}
	}
	if now < hs.itemCooldownUntil[article] {
		return
	}
	switch it.Kind {
	case gamedata.ItemHealthPotion, gamedata.ItemHealthFlask:
		s.applyRegenHoTLocked(c, it, "hp_regen", it.Value, it.Duration, now)
	case gamedata.ItemManaPotion, gamedata.ItemManaFlask:
		s.applyRegenHoTLocked(c, it, "mana_regen", it.Value, it.Duration, now)
	case gamedata.ItemSpeedPotion:
		s.applyPotionBuffLocked(c, []statMod{{stat: "move_speed_pct", value: 1 + it.Value}},
			it, it.Duration, now)
	case gamedata.ItemInvisibilityPotion:
		s.applyInvisibilityLocked(c, it, it.Duration, now)
	case gamedata.ItemRevelationPotion:
		s.applyRevelationLocked(c, it, it.Duration, now)
	case gamedata.ItemDodgeChancePotion:
		s.applyPotionBuffLocked(c, []statMod{{stat: "dodge_pct", value: it.Value}},
			it, it.Duration, now)
	case gamedata.ItemCritStrikePotion:
		s.applyPotionBuffLocked(c, []statMod{
			{stat: "crit_pct", value: it.Value},
			{stat: "crit_dmg_pct", value: it.Value2},
		}, it, it.Duration, now)
	case gamedata.ItemAntiPhysArmorPotion:
		s.applyPotionBuffLocked(c, []statMod{{stat: "phys_armor_pen", value: it.Value}},
			it, it.Duration, now)
	case gamedata.ItemAntiMagicArmorPotion:
		s.applyPotionBuffLocked(c, []statMod{{stat: "magic_armor_pen", value: it.Value}},
			it, it.Duration, now)
	}
	hs.itemCooldownUntil[article] = now + it.Cooldown
	hs.bag[article]--
	s.Store.RemoveBagItem(c.selfPlayerID, article, 1)
	s.push(c, battleproto.CmdRemFromInv, amf.NewArray().Set("id", wireID).Set("count", int32(1)))
}

// applyRegenHoTLocked grants a Health/Mana potion's heal/mana-over-time: the
// real client tooltip always describes "N points over 10 seconds", which
// reuses the SAME generic hp_regen/mana_regen stat mobai.go's per-tick regen
// already sums into the passive per-avatar regen amount -- no new engine
// plumbing needed. Routed through applyPotionBuffLocked so the HoT gets the
// same buff-bar icon every other timed potion effect shows; omitting it was a
// v1 simplification, not intentional -- the icon is exactly how a player
// confirms the potion actually caught (SelfPlayer.UseItem gives no other
// client-side feedback that the HoT is ticking).
func (s *Server) applyRegenHoTLocked(c *conn, it gamedata.Item, stat string, totalAmount, dur float64, now float64) {
	s.applyPotionBuffLocked(c, []statMod{{stat: stat, value: totalAmount / dur}}, it, dur, now)
}

// applyPotionBuffLocked grants one or more timed self statMods sharing a
// single buff-bar icon; expiry (icon removal + fx end + re-sync) is handled
// generically by tickPlayerStatusLocked's existing mods-expiry pass, same as
// any skill buff. Only the FIRST mod carries the buffEffID: tickPlayerStatusLocked's
// expiry loop reverses each mod independently, so giving every mod in the
// group the same id would just double-send the same REM_EFFECTOR. The icon
// itself is the item's OWN dedicated proto (ensureItemBuffProtoLocked), not
// one shared across every tier of its Kind -- see itemBuffProtoID's doc.
func (s *Server) applyPotionBuffLocked(c *conn, mods []statMod, it gamedata.Item, dur float64, now float64) {
	hs := c.huntState
	effID := hs.newEffID()
	buffProto := s.ensureItemBuffProtoLocked(c, it)
	for i := range mods {
		mods[i].until = now + dur
		mods[i].src = "potion_" + mods[i].stat
		if i == 0 {
			mods[i].buffEffID = effID
		}
		hs.st.mods = append(hs.st.mods, mods[i])
	}
	s.push(c, battleproto.CmdAddEffector, addEffectorArgs(effID, buffProto, c.objID, -1, now,
		amf.NewArray().Set("duration", dur)))
	s.pushPlayerStatsLocked(c, now)
}

// applyInvisibilityLocked grants timed stealth from mob aggro (Invisibility
// potion): mobTargetLocked ignores this avatar as a target candidate while
// hs.invisibleUntil is live (suppresses NEW aggro; doesn't break an existing
// chase -- see the huntState.invisibleUntil doc). A shared translucent shade fx
// (same visual mobs use, InvisibilityEffect) shows every party member the
// player has gone invisible; expiry is handled in tickPlayerStatusLocked.
func (s *Server) applyInvisibilityLocked(c *conn, it gamedata.Item, dur float64, now float64) {
	hs := c.huntState
	hs.invisibleUntil = now + dur
	if hs.invisFxUID == 0 {
		hs.invisFxUID = s.worldFxStartLocked(c, mobShadeFx, c.objID, 0, false, 0, 0)
	}
	if hs.invisBuffEffID == 0 {
		hs.invisBuffEffID = hs.newEffID()
		buffProto := s.ensureItemBuffProtoLocked(c, it)
		s.push(c, battleproto.CmdAddEffector, addEffectorArgs(hs.invisBuffEffID, buffProto, c.objID, -1, now,
			amf.NewArray().Set("duration", dur)))
	}
}

// applySkillStealthLocked grants timed stealth from an avatar SKILL (OpStealth), reusing
// the exact aggro-suppression the Invisibility potion uses: mobTargetLocked ignores a
// hidden avatar (hs.invisibleUntil) and the shared shade fx shows the party the vanish.
// Unlike the potion it mints no separate buff-effector -- the skill's own BuffFx/BuffIcon
// (e.g. InvisibilityEffect) carries the buff-bar timer. Stealth breaks the moment the
// player next acts (breakInvisibilityLocked at cast/attack time).
func (s *Server) applySkillStealthLocked(c *conn, dur, now float64) {
	if dur <= 0 {
		return
	}
	hs := c.huntState
	if hs == nil {
		return
	}
	hs.invisibleUntil = now + dur
	if hs.invisFxUID == 0 {
		hs.invisFxUID = s.worldFxStartLocked(c, mobShadeFx, c.objID, 0, false, 0, 0)
	}
}

// breakInvisibilityLocked ends active stealth immediately -- called when the player
// attacks or casts, so acting reveals them (mobs re-aggro at once). A cheap no-op when
// not stealthed. Mirrors the potion-expiry cleanup in tickPlayerStatusLocked (end the
// shared shade fx + drop the buff icon), but forces invisibleUntil to 0 rather than
// waiting for the timer, so the very next mobTargetLocked can see them again.
func (s *Server) breakInvisibilityLocked(c *conn, now float64) {
	hs := c.huntState
	if hs == nil || now >= hs.invisibleUntil {
		return // not currently stealthed
	}
	hs.invisibleUntil = 0
	if hs.invisFxUID != 0 {
		s.worldFxEndLocked(c, hs.invisFxUID)
		hs.invisFxUID = 0
	}
	if hs.invisBuffEffID != 0 {
		s.push(c, battleproto.CmdRemEffector, amf.NewArray().Set("id", hs.invisBuffEffID))
		hs.invisBuffEffID = 0
	}
}

// applyRevelationLocked grants timed "see invisible enemies" state
// (Revelation potion). Currently a no-op beyond its own buff icon/timer --
// see huntState.revealInvisibleUntil's doc for why (co-op PvE hunt mode, no
// enemy avatars, mobs have no invisibility of their own). Expiry is handled
// in tickPlayerStatusLocked, mirroring applyInvisibilityLocked.
func (s *Server) applyRevelationLocked(c *conn, it gamedata.Item, dur float64, now float64) {
	hs := c.huntState
	hs.revealInvisibleUntil = now + dur
	if hs.revealBuffEffID == 0 {
		hs.revealBuffEffID = hs.newEffID()
		buffProto := s.ensureItemBuffProtoLocked(c, it)
		s.push(c, battleproto.CmdAddEffector, addEffectorArgs(hs.revealBuffEffID, buffProto, c.objID, -1, now,
			amf.NewArray().Set("duration", dur)))
	}
}
