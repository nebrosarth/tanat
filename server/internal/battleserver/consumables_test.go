package battleserver

import (
	"strconv"
	"strings"
	"testing"

	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// TestItemProtoDescCarriesPItem guards the empty-chest bug: DropMenu.SetData
// reads battlePrototype.Item.mArticle to build each loot button, so every
// consumable's PROTOTYPE_INFO must carry a <PItem><Article value="..."/></PItem>
// block or the client silently drops the item from the menu.
func TestItemProtoDescCarriesPItem(t *testing.T) {
	for _, it := range gamedata.Items() {
		desc := itemProtoDesc(it)
		if !strings.Contains(desc, "<PItem>") {
			t.Fatalf("article %d: itemProtoDesc missing <PItem>: %s", it.ArticleID, desc)
		}
		want := `<Article value="` + strconv.Itoa(int(it.ArticleID)) + `"/>`
		if !strings.Contains(desc, want) {
			t.Errorf("article %d: itemProtoDesc missing %s: %s", it.ArticleID, want, desc)
		}
	}
}

// TestItemProtoDescCarriesPTool guards the "click does nothing" bug:
// PlayerControl.CreateActiveItem returns null (no DO_ACTION ever sent) unless
// the item's own prototype carries <PTool><Action value="..."/></PTool>
// pointing at a proto with a PEffectDesc.
func TestItemProtoDescCarriesPTool(t *testing.T) {
	want := `<PTool><Action value="` + strconv.Itoa(int(itemUseActionProtoID)) + `"/></PTool>`
	for _, it := range gamedata.Items() {
		desc := itemProtoDesc(it)
		if !strings.Contains(desc, want) {
			t.Fatalf("article %d: itemProtoDesc missing %s: %s", it.ArticleID, want, desc)
		}
	}
	actionDesc := itemUseActionProtoDesc()
	if !strings.Contains(actionDesc, "<PEffectDesc>") {
		t.Fatalf("itemUseActionProtoDesc missing <PEffectDesc>: %s", actionDesc)
	}
	// Must be a genuine no-target self-cast: an empty target enum (mask 0),
	// not "SELF" (which the client treats as needing a unit target).
	if !strings.Contains(actionDesc, `<enum name="target" value=""/>`) {
		t.Errorf("itemUseActionProtoDesc target attrib not empty (must be mask 0 = IsNoneTarget): %s", actionDesc)
	}
}

// TestPotionBuffProtoDescCarriesIcon guards the "buff icon shows as the
// client's default placeholder star" bug: BuffRenderer's battle-buff path
// passes PEffectDesc.Desc.mIcon straight to GuiSystem.GetImage with no prefix
// of its own (unlike the bag/shop/drop menus), so potionBuffProtoDesc must be
// given the item's full IconPath(), not an empty string or the bare Icon.
func TestPotionBuffProtoDescCarriesIcon(t *testing.T) {
	it := gamedata.ItemsByKind(gamedata.ItemHealthPotion)[0]
	desc := potionBuffProtoDesc(it.NameKey, it.IconPath())
	want := `<icon value="` + it.IconPath() + `"/>`
	if !strings.Contains(desc, want) {
		t.Errorf("potionBuffProtoDesc missing %s: %s", want, desc)
	}
	if strings.Contains(desc, `<icon value=""/>`) {
		t.Errorf("potionBuffProtoDesc has an empty icon: %s", desc)
	}
}

// TestAvatarEffectorProtosDontCollideWithItemArticles guards a REAL bug found
// 2026-07-16: Avrora (avatar id 40) has effBase(a)=1000+40*100=5000, which
// landed directly on the old potionArticleBase=5000 -- both an avatar's own
// skill/active/params/attack/buff effector prototypes AND the item catalog's
// article-based prototypes are PROTOTYPE_INFO'd to the SAME client
// connection keyed by this same "id", so if a player plays that avatar AND
// owns/uses any item whose ArticleID falls in the collision range, the
// second registration silently clobbers the first on the client (broken
// skill icons/effects or broken item behavior, depending on order). Fixed by
// giving potionArticleBase (gamedata/items.go) a lot more headroom; this test
// is the standing invariant so a future avatar-roster addition can't quietly
// reintroduce the same collision.
func TestAvatarEffectorProtosDontCollideWithItemArticles(t *testing.T) {
	const maxAvatarProtoOffset = 44 // buffProtoID: effBase(a) + 40 + slot, slot<=4
	minArticle := gamedata.Items()[0].ArticleID
	for _, it := range gamedata.Items() {
		if it.ArticleID < minArticle {
			minArticle = it.ArticleID
		}
	}
	for _, a := range gamedata.Avatars() {
		if maxProto := effBase(a) + maxAvatarProtoOffset; maxProto >= minArticle {
			t.Errorf("avatar %s (id %d): effector protos reach %d, colliding with the item catalog's article range starting at %d",
				a.Prefab, a.ID, maxProto, minArticle)
		}
	}
}

// TestItemBuffProtoIDsAreUnique guards against ever going back to one shared
// buff proto per Kind: every one of the 78 items must map to its own
// collision-free itemBuffProtoID.
func TestItemBuffProtoIDsAreUnique(t *testing.T) {
	seen := map[int32]int32{}
	for _, it := range gamedata.Items() {
		id := itemBuffProtoID(it.ArticleID)
		if other, dup := seen[id]; dup {
			t.Fatalf("itemBuffProtoID collision: articles %d and %d both map to %d", it.ArticleID, other, id)
		}
		seen[id] = it.ArticleID
	}
}

// TestPerItemBuffProtoIsDistinctPerTier guards the "every healing potion
// shows the same icon and name" bug report: two different tiers of the same
// Kind must register two DIFFERENT buff prototypes (their own real
// NameKey/IconPath), not share one "tier-1" placeholder.
func TestPerItemBuffProtoIsDistinctPerTier(t *testing.T) {
	s, c, _, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	tiers := gamedata.ItemsByKind(gamedata.ItemHealthPotion)
	low, high := tiers[0], tiers[len(tiers)-1]
	if low.NameKey == high.NameKey || low.Icon == high.Icon {
		t.Fatalf("fixture assumption broken: want distinct NameKey/Icon between tiers, got %q/%q vs %q/%q",
			low.NameKey, low.Icon, high.NameKey, high.Icon)
	}

	c.lock()
	lowProto := s.ensureItemBuffProtoLocked(c, low)
	highProto := s.ensureItemBuffProtoLocked(c, high)
	c.unlock()

	if lowProto == highProto {
		t.Errorf("two different Health Potion tiers share buff proto id %d -- the buff bar would show one tier's name/icon for both", lowProto)
	}
	if got := itemBuffProtoID(low.ArticleID); got != lowProto {
		t.Errorf("ensureItemBuffProtoLocked proto = %d, want itemBuffProtoID(%d) = %d", lowProto, low.ArticleID, got)
	}
}

// newHuntConnWithHero extends newHuntConn with a real session.Store user/hero
// so Store-backed effects (persistent bag, money, character exp) have
// somewhere to land, and points c.selfPlayerID at it.
func newHuntConnWithHero(t *testing.T, prefab string) (*Server, *conn, *session.User, func()) {
	t.Helper()
	s, c, cleanup := newHuntConn(t, prefab)
	u, _ := s.Store.LoginOrRegister("potiontest@test.test", "pw")
	s.Store.CreateHero(u, 1, true, 0, 0, 0, 0, 0)
	c.selfPlayerID = u.ID
	return s, c, u, cleanup
}

// seedBagLocked wires article as bag-slot wireID on hs (bypassing
// sendInitialBagLocked, which a bare test conn never runs) and mirrors it into
// the persistent store, matching what a real session would have loaded.
func seedBagLocked(s *Server, c *conn, wireID, article, count int32) {
	hs := c.huntState
	hs.bag = map[int32]int32{article: count}
	hs.bagItemID = map[int32]int32{article: wireID}
	hs.bagArticleByID = map[int32]int32{wireID: article}
	s.Store.AddBagItem(c.selfPlayerID, article, count)
}

func TestUseHealthPotionHeals(t *testing.T) {
	s, c, u, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	hs := c.huntState
	it := gamedata.ItemsByKind(gamedata.ItemHealthPotion)[0] // S0 Grey: 250 hp over 10s
	seedBagLocked(s, c, 1, it.ArticleID, 1)

	now := float64(s.battleTime())
	c.lock()
	s.useItemLocked(c, 1, now)
	c.unlock()

	// Real potions heal-over-time, not instantly: this should have armed an
	// hp_regen rate of Value/Duration, not moved hs.hp directly.
	wantRate := it.Value / it.Duration
	if got := hs.st.modSum(now, "hp_regen"); got < wantRate-0.001 || got > wantRate+0.001 {
		t.Errorf("hp_regen rate = %.4f, want ~%.4f (%.0f hp over %.0fs)", got, wantRate, it.Value, it.Duration)
	}
	if hs.bag[it.ArticleID] != 0 {
		t.Errorf("session bag count = %d, want 0", hs.bag[it.ArticleID])
	}
	if bag := s.Store.HeroBag(u.ID); len(bag) != 0 {
		t.Errorf("persisted bag = %+v, want empty (potion not consumed on disk)", bag)
	}
	if hs.itemCooldownUntil[it.ArticleID] <= now {
		t.Error("itemCooldownUntil not advanced after use")
	}

	// Rate drops back to 0 once the HoT window (its real Duration) elapses.
	after := now + it.Duration + 1
	if got := hs.st.modSum(after, "hp_regen"); got != 0 {
		t.Errorf("hp_regen rate after expiry = %.4f, want 0", got)
	}
}

// TestHealthPotionShowsBuffIcon guards against the HoT being a "silent"
// effect: applyRegenHoTLocked must route through applyPotionBuffLocked (like
// every other timed potion) so the client gets an ADD_EFFECTOR buff-bar icon,
// not just the invisible hp_regen statMod.
func TestHealthPotionShowsBuffIcon(t *testing.T) {
	s, c, _, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	hs := c.huntState
	it := gamedata.ItemsByKind(gamedata.ItemHealthPotion)[0]
	seedBagLocked(s, c, 1, it.ArticleID, 1)

	now := float64(s.battleTime())
	c.lock()
	s.useItemLocked(c, 1, now)
	c.unlock()

	found := false
	for _, m := range hs.st.mods {
		if m.stat == "hp_regen" && m.buffEffID != 0 {
			found = true
		}
	}
	if !found {
		t.Error("hp_regen HoT mod missing buffEffID (no buff icon shown to the client)")
	}
}

func TestUseManaPotionRestores(t *testing.T) {
	s, c, _, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	hs := c.huntState
	it := gamedata.ItemsByKind(gamedata.ItemManaPotion)[0] // S0 Grey: 100 mana over 10s
	seedBagLocked(s, c, 1, it.ArticleID, 1)

	now := float64(s.battleTime())
	c.lock()
	s.useItemLocked(c, 1, now)
	c.unlock()

	wantRate := it.Value / it.Duration
	if got := hs.st.modSum(now, "mana_regen"); got < wantRate-0.001 || got > wantRate+0.001 {
		t.Errorf("mana_regen rate = %.4f, want ~%.4f (%.0f mana over %.0fs)", got, wantRate, it.Value, it.Duration)
	}
}

func TestFlasksHealAndRestore(t *testing.T) {
	s, c, _, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	hs := c.huntState
	hf := gamedata.ItemsByKind(gamedata.ItemHealthFlask)[0]
	mf := gamedata.ItemsByKind(gamedata.ItemManaFlask)[0]
	seedBagLocked(s, c, 1, hf.ArticleID, 1)
	hs.bagItemID[mf.ArticleID] = 2
	hs.bagArticleByID[2] = mf.ArticleID
	hs.bag[mf.ArticleID] = 1
	s.Store.AddBagItem(c.selfPlayerID, mf.ArticleID, 1)

	now := float64(s.battleTime())
	c.lock()
	s.useItemLocked(c, 1, now) // health flask
	s.useItemLocked(c, 2, now) // mana flask
	c.unlock()

	if got, want := hs.st.modSum(now, "hp_regen"), hf.Value/hf.Duration; got < want-0.001 || got > want+0.001 {
		t.Errorf("health flask hp_regen = %.4f, want ~%.4f", got, want)
	}
	if got, want := hs.st.modSum(now, "mana_regen"), mf.Value/mf.Duration; got < want-0.001 || got > want+0.001 {
		t.Errorf("mana flask mana_regen = %.4f, want ~%.4f", got, want)
	}
}

func TestPotionCooldownBlocksSecondUse(t *testing.T) {
	s, c, _, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	hs := c.huntState
	it := gamedata.ItemsByKind(gamedata.ItemHealthPotion)[0]
	seedBagLocked(s, c, 1, it.ArticleID, 5)

	now := float64(s.battleTime())
	c.lock()
	s.useItemLocked(c, 1, now) // consumes 1, arms this article's cooldown
	rateAfterFirst := hs.st.modSum(now, "hp_regen")
	s.useItemLocked(c, 1, now) // still on cooldown: must be a no-op
	c.unlock()

	if hs.bag[it.ArticleID] != 4 {
		t.Errorf("bag count after 2 uses within cooldown = %d, want 4 (second use should be blocked)", hs.bag[it.ArticleID])
	}
	if got := hs.st.modSum(now, "hp_regen"); got != rateAfterFirst {
		t.Errorf("hp_regen rate changed on the blocked second use: %.4f -> %.4f", rateAfterFirst, got)
	}
}

// TestPerItemCooldownIsIndependent guards the point of the per-item cooldown
// rework: drinking one potion must NOT block a DIFFERENT kind's potion (the
// old v1 code shared one flat cooldown across every kind; the real client
// gives each item its own).
func TestPerItemCooldownIsIndependent(t *testing.T) {
	s, c, _, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	hs := c.huntState
	health := gamedata.ItemsByKind(gamedata.ItemHealthPotion)[0]
	speed := gamedata.ItemsByKind(gamedata.ItemSpeedPotion)[0]
	seedBagLocked(s, c, 1, health.ArticleID, 1)
	hs.bagItemID[speed.ArticleID] = 2
	hs.bagArticleByID[2] = speed.ArticleID
	hs.bag[speed.ArticleID] = 1
	s.Store.AddBagItem(c.selfPlayerID, speed.ArticleID, 1)

	now := float64(s.battleTime())
	c.lock()
	s.useItemLocked(c, 1, now) // drink the health potion; arms ONLY its own cooldown
	s.useItemLocked(c, 2, now) // speed potion must still be usable right away
	buffed := c.curSpeedLocked(now)
	c.unlock()

	baseSpeed := float64(lobbyMoveSpeed)
	wantSpeed := baseSpeed * (1 + speed.Value)
	if buffed < wantSpeed-0.01 || buffed > wantSpeed+0.01 {
		t.Errorf("speed buff after an unrelated potion's cooldown = %.2f, want ~%.2f (should not have been blocked)", buffed, wantSpeed)
	}
	if hs.bag[speed.ArticleID] != 0 {
		t.Errorf("speed potion bag count = %d, want 0 (should have been consumed, not blocked)", hs.bag[speed.ArticleID])
	}
}

func TestSpeedPotionBuffsAndExpires(t *testing.T) {
	s, c, _, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	it := gamedata.ItemsByKind(gamedata.ItemSpeedPotion)[0]
	seedBagLocked(s, c, 1, it.ArticleID, 1)

	now := float64(s.battleTime())
	c.lock()
	s.useItemLocked(c, 1, now)
	baseSpeed := float64(lobbyMoveSpeed)
	buffed := c.curSpeedLocked(now)
	c.unlock()

	wantSpeed := baseSpeed * (1 + it.Value)
	if buffed < wantSpeed-0.01 || buffed > wantSpeed+0.01 {
		t.Errorf("speed while buffed = %.2f, want ~%.2f (base %.2f)", buffed, wantSpeed, baseSpeed)
	}

	// After the duration elapses, the tick's mods-expiry pass should drop it.
	c.lock()
	s.tickPlayerStatusLocked(c, now+it.Duration+1)
	after := c.curSpeedLocked(now + it.Duration + 1)
	c.unlock()
	if after < baseSpeed-0.01 || after > baseSpeed+0.01 {
		t.Errorf("speed after expiry = %.2f, want back to base %.2f", after, baseSpeed)
	}
}

func TestInvisibilityBreaksMobTargeting(t *testing.T) {
	s, c, _, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	it := gamedata.ItemsByKind(gamedata.ItemInvisibilityPotion)[0]
	seedBagLocked(s, c, 1, it.ArticleID, 1)

	now := float64(s.battleTime())
	m := &mobState{id: 2000, x: 0, y: 0}
	members := []*conn{c}

	if tgt := mobTargetLocked(m, members, now); tgt.obj != c.objID {
		t.Fatalf("visible player should be targetable, got obj=%d", tgt.obj)
	}

	c.lock()
	s.useItemLocked(c, 1, now)
	c.unlock()

	if tgt := mobTargetLocked(m, members, now); tgt.obj != 0 {
		t.Errorf("invisible player should not be targetable, got obj=%d", tgt.obj)
	}

	// After the duration elapses, targeting resumes.
	after := now + it.Duration + 1
	if tgt := mobTargetLocked(m, members, after); tgt.obj != c.objID {
		t.Errorf("player should be targetable again after invisibility expires, got obj=%d", tgt.obj)
	}
}

func TestDodgeChancePotionAppliesDodgePct(t *testing.T) {
	s, c, _, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	hs := c.huntState
	it := gamedata.ItemsByKind(gamedata.ItemDodgeChancePotion)[0]
	seedBagLocked(s, c, 1, it.ArticleID, 1)

	now := float64(s.battleTime())
	c.lock()
	s.useItemLocked(c, 1, now)
	c.unlock()

	if got := hs.st.modSum(now, "dodge_pct"); got < it.Value-0.001 || got > it.Value+0.001 {
		t.Errorf("dodge_pct = %.4f, want ~%.4f", got, it.Value)
	}
	after := now + it.Duration + 1
	if got := hs.st.modSum(after, "dodge_pct"); got != 0 {
		t.Errorf("dodge_pct after expiry = %.4f, want 0", got)
	}
}

func TestCritStrikePotionAppliesBothCritStats(t *testing.T) {
	s, c, _, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	hs := c.huntState
	it := gamedata.ItemsByKind(gamedata.ItemCritStrikePotion)[0]
	seedBagLocked(s, c, 1, it.ArticleID, 1)

	now := float64(s.battleTime())
	c.lock()
	s.useItemLocked(c, 1, now)
	c.unlock()

	if got := hs.st.modSum(now, "crit_pct"); got < it.Value-0.001 || got > it.Value+0.001 {
		t.Errorf("crit_pct = %.4f, want ~%.4f", got, it.Value)
	}
	if got := hs.st.modSum(now, "crit_dmg_pct"); got < it.Value2-0.001 || got > it.Value2+0.001 {
		t.Errorf("crit_dmg_pct = %.4f, want ~%.4f", got, it.Value2)
	}
}

func TestArmorPenPotionsApplyPenStats(t *testing.T) {
	s, c, _, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	hs := c.huntState
	phys := gamedata.ItemsByKind(gamedata.ItemAntiPhysArmorPotion)[0]
	magic := gamedata.ItemsByKind(gamedata.ItemAntiMagicArmorPotion)[0]
	seedBagLocked(s, c, 1, phys.ArticleID, 1)
	hs.bagItemID[magic.ArticleID] = 2
	hs.bagArticleByID[2] = magic.ArticleID
	hs.bag[magic.ArticleID] = 1
	s.Store.AddBagItem(c.selfPlayerID, magic.ArticleID, 1)

	now := float64(s.battleTime())
	c.lock()
	s.useItemLocked(c, 1, now)
	s.useItemLocked(c, 2, now)
	c.unlock()

	if got := hs.st.modSum(now, "phys_armor_pen"); got != phys.Value {
		t.Errorf("phys_armor_pen = %v, want %v", got, phys.Value)
	}
	if got := hs.st.modSum(now, "magic_armor_pen"); got != magic.Value {
		t.Errorf("magic_armor_pen = %v, want %v", got, magic.Value)
	}
}

func TestRevelationPotionArmsRevealState(t *testing.T) {
	s, c, _, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	hs := c.huntState
	it := gamedata.ItemsByKind(gamedata.ItemRevelationPotion)[0]
	seedBagLocked(s, c, 1, it.ArticleID, 1)

	now := float64(s.battleTime())
	c.lock()
	s.useItemLocked(c, 1, now)
	c.unlock()

	if hs.revealInvisibleUntil <= now {
		t.Error("revealInvisibleUntil not armed after drinking a Revelation potion")
	}
	if hs.revealBuffEffID == 0 {
		t.Error("revealBuffEffID not set (no buff icon shown)")
	}

	c.lock()
	s.tickPlayerStatusLocked(c, now+it.Duration+1)
	c.unlock()
	if hs.revealBuffEffID != 0 {
		t.Error("revealBuffEffID should be cleared after the potion's duration elapses")
	}
}
