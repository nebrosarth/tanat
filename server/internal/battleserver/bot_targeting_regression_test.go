package battleserver

import (
	"testing"
	"time"

	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
)

// TestBotPlusMinusKeepsHeroTargetThroughChannel covers the live failure where the
// bot aimed PlusMinus's second skill at a point, then the next combat tick replaced
// the cast with a normal PvP order. The channel must retain the enemy avatar id and
// continue pulsing against that avatar even after it moves away from the cast point.
func TestBotPlusMinusKeepsHeroTargetThroughChannel(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Dsb_PlusMinus")
	defer cleanup()
	enemy := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, bot.x+4, bot.y)
	bot.huntState.team = dotaTeamHuman
	bot.huntState.skillLevel[1] = 1 // PlusMinus skill 2
	brain := &botBrain{c: bot}

	castAt := float64(s.battleTime())
	bot.mvMu.Lock()
	// Reproduce the live race: a PvP attack timer is still armed when the
	// targeted channel is selected. The cast must invalidate it before the delayed
	// payload creates the channel.
	bot.huntState.pvpTarget = enemy.objID
	bot.huntState.attackSeq++
	if !s.botConsiderOffensiveAbilityLocked(brain, enemy, castAt) {
		bot.mvMu.Unlock()
		t.Fatal("bot did not cast PlusMinus skill 2")
	}
	if len(bot.huntState.payloads) != 1 || bot.huntState.payloads[0].target != enemy.objID {
		got := int32(0)
		if len(bot.huntState.payloads) == 1 {
			got = bot.huntState.payloads[0].target
		}
		bot.mvMu.Unlock()
		t.Fatalf("PlusMinus payload target = %d, want enemy hero %d", got, enemy.objID)
	}
	if bot.huntState.pvpTarget != 0 || bot.hasDest || bot.vx != 0 || bot.vy != 0 {
		bot.mvMu.Unlock()
		t.Fatalf("PlusMinus left an attack/movement order during cast wind-up: pvpTarget=%d dest=%v velocity=(%g,%g)",
			bot.huntState.pvpTarget, bot.hasDest, bot.vx, bot.vy)
	}
	if !botHasPendingBlockingChannelLocked(bot.huntState, castAt+0.2) {
		bot.mvMu.Unlock()
		t.Fatal("PlusMinus delayed payload was not treated as a blocking channel")
	}
	if !s.botCombatTickLocked(brain, castAt+0.2) {
		bot.mvMu.Unlock()
		t.Fatal("bot combat tick did not yield to the pending PlusMinus channel")
	}
	if bot.huntState.pvpTarget != 0 || bot.hasDest || bot.vx != 0 || bot.vy != 0 {
		bot.mvMu.Unlock()
		t.Fatalf("bot combat tick reissued movement during PlusMinus wind-up: pvpTarget=%d dest=%v velocity=(%g,%g)",
			bot.huntState.pvpTarget, bot.hasDest, bot.vx, bot.vy)
	}
	s.runDuePayloadsLocked(bot, castAt+0.31)
	if len(bot.huntState.channels) != 1 {
		bot.mvMu.Unlock()
		t.Fatalf("PlusMinus opened %d channels, want 1", len(bot.huntState.channels))
	}
	if enemy.huntState.st.stunUntil <= castAt+0.31 {
		bot.mvMu.Unlock()
		t.Fatalf("PlusMinus did not stun hero: stun=%g", enemy.huntState.st.stunUntil)
	}
	if got := bot.huntState.channels[0].target; got != enemy.objID {
		bot.mvMu.Unlock()
		t.Fatalf("PlusMinus channel target = %d, want enemy hero %d", got, enemy.objID)
	}
	bot.mvMu.Unlock()

	// Move the live hero away from the original aim point. A target channel follows
	// the hero; it is not a four-unit ground AoE accidentally left at the old point.
	enemy.mvMu.Lock()
	enemy.x = bot.x + 25
	enemy.y = bot.y
	enemy.vx, enemy.vy = 0, 0
	enemy.snapT = float32(castAt + 1.4)
	enemy.mvMu.Unlock()

	bot.mvMu.Lock()
	before := enemy.huntState.hp
	s.tickChannelsLocked(bot, castAt+1.4)
	after := enemy.huntState.hp
	if after >= before {
		bot.mvMu.Unlock()
		t.Fatalf("PlusMinus pulse did not damage moved hero: hp %.1f -> %.1f", before, after)
	}
	// The next bot decision must keep sustaining the channel, not start a PvP attack
	// that cancels/replaces the cast order.
	if !s.botCombatTickLocked(brain, castAt+1.4) || bot.huntState.pvpTarget != 0 {
		bot.mvMu.Unlock()
		t.Fatalf("bot issued a PvP order while PlusMinus channel was active: pvpTarget=%d", bot.huntState.pvpTarget)
	}
	bot.mvMu.Unlock()
}

// TestBotTeridinUltTracksHeroUntilImpact covers the delayed sniper payload. The
// payload must retain the target object id and resolve the hero at impact time, so
// walking away after the cast does not make the shot disappear at the old point.
func TestBotTeridinUltTracksHeroUntilImpact(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_HK_Teridin")
	defer cleanup()
	enemy := dotaPlayerConn(t, s, inst, 1002, dotaTeamElf, bot.x+4, bot.y)
	bot.huntState.team = dotaTeamHuman
	bot.huntState.skillLevel[3] = 1 // Teridin ultimate
	brain := &botBrain{c: bot}

	castAt := float64(s.battleTime())
	bot.mvMu.Lock()
	if !s.botConsiderOffensiveAbilityLocked(brain, enemy, castAt) {
		bot.mvMu.Unlock()
		t.Fatal("bot did not cast Teridin ultimate")
	}
	if len(bot.huntState.payloads) != 1 || bot.huntState.payloads[0].target != enemy.objID {
		got := int32(0)
		if len(bot.huntState.payloads) == 1 {
			got = bot.huntState.payloads[0].target
		}
		bot.mvMu.Unlock()
		t.Fatalf("Teridin payload target = %d, want enemy hero %d", got, enemy.objID)
	}
	bot.mvMu.Unlock()

	enemy.mvMu.Lock()
	enemy.x = bot.x + 30
	enemy.y = bot.y
	enemy.vx, enemy.vy = 0, 0
	enemy.snapT = float32(castAt + 2.1)
	enemy.mvMu.Unlock()

	bot.mvMu.Lock()
	before := enemy.huntState.hp
	s.runDuePayloadsLocked(bot, castAt+2.1)
	after := enemy.huntState.hp
	bot.mvMu.Unlock()
	if after >= before {
		t.Fatalf("Teridin ultimate missed moved hero at impact: hp %.1f -> %.1f", before, after)
	}
}

// TestAvatarItemReplayUsesArticleIds pins the wire shape used by the remote inspect
// panel: the client parser reads ITEM_EQUIP.item/proto and Player.Equip stores proto
// in ActiveItems, which the card resolves through the article-keyed BattleItemData
// catalog. Internal bag ids must never be replayed.
func TestAvatarItemReplayUsesArticleIds(t *testing.T) {
	s, viewer, inst, packets, packetMu := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	owner := dotaPlayerConn(t, s, inst, 1003, dotaTeamHuman, viewer.x+2, viewer.y)
	viewer.huntState.team = dotaTeamHuman
	owner.huntState.ownedTreeItems = map[int32]bool{}
	items := gamedata.AvatarItems()
	if len(items) < 2 {
		t.Fatal("avatar item catalog is too small for replay test")
	}
	owner.huntState.ownedTreeItems[items[0].ArticleID] = true
	owner.huntState.ownedTreeItems[items[1].ArticleID] = true

	viewer.mvMu.Lock()
	s.renderAvatarWithStatsLocked(viewer, owner, now)
	viewer.mvMu.Unlock()
	time.Sleep(50 * time.Millisecond)

	packetMu.Lock()
	defer packetMu.Unlock()
	seen := map[int32]bool{}
	for _, p := range *packets {
		if p.Cmd != battleproto.CmdItemEquip {
			continue
		}
		seen[p.Args.IntOr("proto", -1)] = true
		if p.Args.IntOr("id", -1) != owner.objID {
			t.Fatalf("ITEM_EQUIP id = %d, want owner %d", p.Args.IntOr("id", -1), owner.objID)
		}
		if p.Args.IntOr("item", -1) != p.Args.IntOr("proto", -1) {
			t.Fatalf("ITEM_EQUIP item=%d proto=%d: tree item ids must match", p.Args.IntOr("item", -1), p.Args.IntOr("proto", -1))
		}
	}
	for _, it := range items[:2] {
		if !seen[it.ArticleID] {
			t.Fatalf("missing replayed ITEM_EQUIP for article %d; seen=%v", it.ArticleID, seen)
		}
	}
}

// TestAvatarItemReplayDoesNotDuplicateOnRecreate mirrors a fog reveal or
// respawn. The client keeps Player.ActiveItems after DELETE_OBJECT, therefore
// the server must not replay the same owned articles a second time.
func TestAvatarItemReplayDoesNotDuplicateOnRecreate(t *testing.T) {
	s, viewer, inst, packets, packetMu := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	owner := dotaPlayerConn(t, s, inst, 1004, dotaTeamHuman, viewer.x+2, viewer.y)
	owner.huntState.ownedTreeItems = map[int32]bool{}
	items := gamedata.AvatarItems()
	if len(items) < 2 {
		t.Fatal("avatar item catalog is too small for replay test")
	}
	owner.huntState.ownedTreeItems[items[0].ArticleID] = true
	owner.huntState.ownedTreeItems[items[1].ArticleID] = true

	viewer.mvMu.Lock()
	s.renderAvatarWithStatsLocked(viewer, owner, now)
	s.removeAvatarForLocked(viewer, owner)
	s.renderAvatarWithStatsLocked(viewer, owner, now+1)
	viewer.mvMu.Unlock()
	time.Sleep(50 * time.Millisecond)

	packetMu.Lock()
	defer packetMu.Unlock()
	count := 0
	for _, p := range *packets {
		if p.Cmd == battleproto.CmdItemEquip && p.Args.IntOr("id", -1) == owner.objID {
			count++
		}
	}
	if count != 2 {
		t.Fatalf("recreated avatar replayed %d ITEM_EQUIP packets, want exactly 2", count)
	}
}

// TestAvatarItemStatsReachRenderedViewer verifies the live target-card path:
// buying/applying a Health item must update max HP on clients that already
// render the avatar, not only on the buyer's private connection.
func TestAvatarItemStatsReachRenderedViewer(t *testing.T) {
	s, viewer, inst, packets, packetMu := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	owner := dotaPlayerConn(t, s, inst, 1005, dotaTeamHuman, viewer.x+2, viewer.y)
	var health gamedata.AvatarItem
	for _, it := range gamedata.AvatarItems() {
		if it.TreeID == gamedata.AvatarTreeDefence && len(it.Parents) == 0 {
			health = it
			break
		}
	}
	if health.ArticleID == 0 {
		t.Fatal("missing defence root health item")
	}
	hpBonus := statSum(health.Stats, "Health")
	baseMax := owner.huntState.maxHPLocked(now)
	owner.huntState.hp = baseMax

	viewer.mvMu.Lock()
	s.renderAvatarWithStatsLocked(viewer, owner, now)
	viewer.mvMu.Unlock()

	owner.lock()
	s.applyAvatarItemStatsLocked(owner, health, now)
	owner.unlock()
	time.Sleep(50 * time.Millisecond)

	idx := viewer.huntState.tr.index(owner.objID)
	want := float32(baseMax + hpBonus)
	var got float32
	var found bool
	packetMu.Lock()
	for _, p := range *packets {
		if p.Cmd != battleproto.CmdSync || p.Args == nil {
			continue
		}
		data, ok := p.Args.Assoc["data"].([]byte)
		if !ok {
			continue
		}
		if v, ok := syncFloatForIndex(data, syncMaxHealth, idx); ok {
			got, found = v, true
		}
	}
	packetMu.Unlock()
	if !found {
		t.Fatalf("viewer received no MAX_HEALTH sync for owner index %d", idx)
	}
	if got != want {
		t.Fatalf("viewer max HP = %v, want %v after +%v Health item", got, want, hpBonus)
	}
}

// TestBotRetreatClosesVisibleAttackSwing covers the client-facing half of the bot
// retreat bug: an already-emitted ACTION can outlive attackTarget by one timer tick.
// A move decision must close that animation before starting the retreat leg.
func TestBotRetreatClosesVisibleAttackSwing(t *testing.T) {
	s, bot, _, cleanup := newDotaConn(t, "Avtr_Tank_Sigilion")
	defer cleanup()
	brain := &botBrain{c: bot}

	bot.lock()
	bot.huntState.attackTarget = 60090
	bot.huntState.attackActionActive = true
	s.botMoveTowardLocked(brain, bot.x-10, bot.y, float64(s.battleTime()))
	if bot.huntState.attackTarget != 0 || bot.huntState.attackActionActive {
		bot.unlock()
		t.Fatalf("retreat left attack state active: target=%d animation=%v", bot.huntState.attackTarget, bot.huntState.attackActionActive)
	}
	bot.unlock()
}

// TestDotaCreepKillCreditsRecentHero verifies that a creep-delivered lethal blow
// keeps the last successful enemy-avatar damage as the hero kill, including the
// normal «Штурм» frag/XP/gold path.
func TestDotaCreepKillCreditsRecentHero(t *testing.T) {
	s, victim, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	victim.huntState.team = dotaTeamElf
	attacker := dotaPlayerConn(t, s, inst, 1001, dotaTeamHuman, victim.x+2, victim.y)
	creep := &mobState{
		id: 60090, mobIdx: inst.dota.m.ElfCreepMelee,
		mob: gamedata.Mobs()[inst.dota.m.ElfCreepMelee], team: dotaTeamElf,
		hp: 5000, maxHP: 5000, x: victim.x, y: victim.y,
	}
	now := float64(s.battleTime())

	inst.mu.Lock()
	victim.huntState.hp = victim.huntState.av.Health
	// A non-lethal landed hero hit establishes the credit source.
	s.hitPlayerFromLocked(victim, attacker.objID, 1, now, nil, attacker)
	if victim.huntState.lastHeroDamager != attacker.objID {
		inst.mu.Unlock()
		t.Fatalf("last hero damager=%d, want %d", victim.huntState.lastHeroDamager, attacker.objID)
	}
	s.hitPlayerFromLocked(victim, creep.id, 100000, now+1, creep, nil)
	gotFrags := attacker.huntState.frags
	gotLastKiller := victim.huntState.lastKiller
	gotXP := attacker.huntState.xp
	inst.mu.Unlock()

	if gotFrags != 1 {
		t.Fatalf("creep-delivered hero kill frags=%d, want 1", gotFrags)
	}
	if gotLastKiller != attacker.selfPlayerID {
		t.Fatalf("victim lastKiller=%d, want attacker player id %d", gotLastKiller, attacker.selfPlayerID)
	}
	if gotXP <= 0 {
		t.Fatalf("credited hero received no XP: %g", gotXP)
	}
}
