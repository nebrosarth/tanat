package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

// TestMobKillAwardsCoinsAndXP verifies that killing a level-scaled mob credits its
// scaled coin bounty to the player's persistent hero money and grants its scaled
// XP (the reward half of the mob-level system).
func TestMobKillAwardsCoinsAndXP(t *testing.T) {
	s, c, _, sx, sy := newNavConn(t)
	// Bind a real hero to the battle connection so coins have somewhere to land.
	u, _ := s.Store.LoginOrRegister("hunter@test.io", "pw")
	s.Store.CreateHero(u, 0, false, 0, 0, 0, 0, 0)
	c.selfPlayerID = u.ID
	startMoney := u.Hero.Money

	idx := mobIndexByPrefab(t, "Mob_ZombieCrawl_01") // ghoul
	// A level-10 ghoul as it would be spawned: scaled maxHP/xp/coins set.
	m := &mobState{
		id: 2100, mobIdx: idx, mob: gamedata.Mobs()[idx],
		x: sx + 3, y: sy, spawnX: sx + 3, spawnY: sy,
		level: 10, maxHP: 800, dmgMin: 8, dmgMax: 8, xp: 100, coins: 60,
		hp: 10, shown: true, aggro: true,
	}

	c.mvMu.Lock()
	c.huntState.mobs[m.id] = m
	c.huntState.tr.add(m.id)
	xpBefore := c.huntState.xp
	s.hitMobLocked(c, m, 999, c.objID) // kill
	xpGained := c.huntState.xp - xpBefore
	c.mvMu.Unlock()

	if got := u.Hero.Money - startMoney; got != 60 {
		t.Errorf("coins credited = %d, want the scaled 60", got)
	}
	if xpGained != 100 {
		t.Errorf("XP granted = %g, want the scaled 100", xpGained)
	}
}

// TestMobRespawn drives the death -> 5-minute-timer -> revive flow: a killed mob
// keeps its state (not deleted), schedules a respawn, and the combat tick revives
// it at full HP at its authored spawn point once the timer elapses.
func TestMobRespawn(t *testing.T) {
	s, c, _, sx, sy := newNavConn(t)
	idx := mobIndexByPrefab(t, "Mob_Skeleton_1H_Melee_01")
	mob := gamedata.Mobs()[idx]
	id := int32(2000)
	m := &mobState{
		id: id, mobIdx: idx, mob: mob,
		x: sx + 3, y: sy, spawnX: sx + 3, spawnY: sy,
		hp: 10, shown: true, aggro: true,
	}

	c.mvMu.Lock()
	c.huntState.mobs[id] = m
	c.huntState.tr.add(id)
	// Kill it.
	s.hitMobLocked(c, m, 999, c.objID)
	dead := m.dead
	respawnAt := m.respawnAt
	stillTracked := c.huntState.mobs[id] != nil
	c.mvMu.Unlock()

	if !dead {
		t.Fatal("mob should be dead after a lethal hit")
	}
	if !stillTracked {
		t.Fatal("a respawnable mob must NOT be deleted from hs.mobs on death")
	}
	now := float64(s.battleTime())
	if respawnAt <= now {
		t.Fatalf("respawnAt %.0f should be ~5 min in the future (now %.0f)", respawnAt, now)
	}
	if respawnAt-now < 250 { // ~mobRespawnDelay (300s), generous lower bound
		t.Errorf("respawn delay %.0fs shorter than expected ~300s", respawnAt-now)
	}

	// Move the mob off its spawn and simulate the 5 minutes elapsing by ticking at
	// a battle-time just past its respawn timer.
	c.mvMu.Lock()
	m.x, m.y = sx+50, sy+50
	s.tickMobsLocked(c, m.respawnAt+1)
	revived := !m.dead
	hp := m.hp
	px, py := m.x, m.y
	c.mvMu.Unlock()

	if !revived {
		t.Fatal("mob should have revived once its respawn timer elapsed")
	}
	if hp != mob.Health {
		t.Errorf("revived HP = %g, want full %g", hp, mob.Health)
	}
	if px != sx+3 || py != sy {
		t.Errorf("revived at (%.1f,%.1f), want spawn (%.1f,%.1f)", px, py, sx+3, sy)
	}
}

// TestRespawnEvictsCampingPack is the guard for the spawn-camp fix: a pack dragged
// onto a respawn checkpoint mid-fight is sent home when the player respawns, so the
// fresh player never resurrects inside its 9m aggro bubble. (The mob-free spawn ring
// only governs where mobs SPAWN; leash is player-relative, so mobs CAN be chased onto
// a checkpoint -- respawnPlayerLocked's eviction is what keeps the checkpoint safe.)
func TestRespawnEvictsCampingPack(t *testing.T) {
	s, c, _, sx, sy := newNavConn(t)
	hs := c.huntState
	hs.respawnX, hs.respawnY = sx, sy // the player's active checkpoint

	idx := mobIndexByPrefab(t, "Mob_ZombieCrawl_01")
	// Camper: HOME is 20m away (outside evict+aggro range) but it was dragged right
	// onto the checkpoint (2m) and froze there, still shown, mid-fight HP.
	camper := &mobState{
		id: 2200, mobIdx: idx, mob: gamedata.Mobs()[idx],
		x: sx + 2, y: sy, spawnX: sx + 20, spawnY: sy,
		hp: 30, maxHP: 92, shown: true, aggro: true,
	}
	// Distant mob genuinely at its far home must not be repositioned.
	far := &mobState{
		id: 2201, mobIdx: idx, mob: gamedata.Mobs()[idx],
		x: sx + 40, y: sy, spawnX: sx + 40, spawnY: sy,
		hp: 92, maxHP: 92, shown: true, aggro: true,
	}

	c.mvMu.Lock()
	hs.mobs[camper.id], hs.mobs[far.id] = camper, far
	hs.tr.add(camper.id)
	hs.tr.add(far.id)
	s.respawnPlayerLocked(c, float64(s.battleTime()))
	cx, cy, cAggro, cShown, cHP := camper.x, camper.y, camper.aggro, camper.shown, camper.hp
	fx := far.x
	c.mvMu.Unlock()

	// Camper: sent home, de-aggroed, hidden -- but HP preserved (reposition, not respawn).
	if cx != sx+20 || cy != sy {
		t.Errorf("camper not sent home: at (%.1f,%.1f), want (%.1f,%.1f)", cx, cy, sx+20, sy)
	}
	if cAggro {
		t.Error("camper should have dropped aggro on player respawn")
	}
	if cShown {
		t.Error("camper should be hidden after eviction (fog re-reveals at home next tick)")
	}
	if cHP != 30 {
		t.Errorf("eviction must preserve HP (reposition, not respawn): hp %g, want 30", cHP)
	}
	// Distant mob: left where it stood.
	if fx != sx+40 {
		t.Errorf("distant mob wrongly moved to %.1f (should be untouched)", fx)
	}
}

// TestLevelScalingInCombat checks the per-level power curve is actually applied
// to the live stats: max HP and skill damage both scale up with avatar level, and
// are unchanged at level 0.
func TestLevelScalingInCombat(t *testing.T) {
	s, c, _, _, _ := newNavConn(t)
	hs := c.huntState
	now := float64(s.battleTime())

	// Max HP: base at level 0, scaled at level 19.
	hs.level = 0
	base := hs.maxHPLocked(now)
	if base != hs.av.Health {
		t.Errorf("level-0 maxHP = %g, want base %g", base, hs.av.Health)
	}
	hs.level = 19
	if hi := hs.maxHPLocked(now); hi < base*2.0 || hi > base*2.3 {
		t.Errorf("level-20 maxHP = %g, want ~%g", hi, base*2.14)
	}

	// Skill damage: a flat magic op with no spell power scales purely by level.
	op := gamedata.Op{Kind: gamedata.OpDamage, Value: gamedata.PerLevel{100, 100, 100, 100}, Scale: "magic"}
	ctx := opCtx{level: 1}
	hs.level = 0
	c.mvMu.Lock()
	d0 := s.skillDamageLocked(c, op, ctx, nil)
	hs.level = 9
	d9 := s.skillDamageLocked(c, op, ctx, nil)
	c.mvMu.Unlock()
	if d0 != 100 {
		t.Errorf("level-0 skill damage = %g, want 100 (unscaled)", d0)
	}
	if want := 100 * gamedata.LevelPowerMul(9); d9 != want {
		t.Errorf("level-10 skill damage = %g, want %g", d9, want)
	}
}
