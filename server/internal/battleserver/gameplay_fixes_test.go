package battleserver

import (
	"math"
	"testing"

	"tanatserver/internal/gamedata"
)

// ---- Task 2: spawn leash (boss pinned to its arena, trash roams wider) ----

func TestMobSpawnLeashRadius(t *testing.T) {
	if r := mobSpawnLeashRadius(&mobState{boss: true}); r != bossSpawnLeash {
		t.Errorf("boss leash = %g, want %g", r, bossSpawnLeash)
	}
	if r := mobSpawnLeashRadius(&mobState{boss: false}); r != mobSpawnLeash {
		t.Errorf("regular leash = %g, want %g", r, mobSpawnLeash)
	}
	if bossSpawnLeash >= mobSpawnLeash {
		t.Errorf("a boss must be pinned tighter than trash (%g vs %g)", bossSpawnLeash, mobSpawnLeash)
	}
}

// TestBossSpawnLeashPinsToArena: a boss dragged past its 2m spawn leash gives up and
// returns home even while a player stands right next to it -- it cannot be kited out of
// its arena. A regular mob at the same offset keeps fighting (its leash is far wider).
func TestBossSpawnLeashPinsToArena(t *testing.T) {
	s, c, _, sx, sy := newNavConn(t)
	idx := mobIndexByPrefab(t, "Mob_Skeleton_1H_Melee_01")
	mob := gamedata.Mobs()[idx]

	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	// Boss standing 4m from spawn (> bossSpawnLeash 2m), player 1m away (well inside aggro).
	boss := &mobState{
		id: 2400, mobIdx: idx, mob: mob,
		x: sx + 4, y: sy, spawnX: sx, spawnY: sy,
		hp: mob.Health, shown: true, aggro: true, homed: true, boss: true,
	}
	c.x, c.y, c.snapT = boss.x - 1, boss.y, 0
	c.huntState.mobs = map[int32]*mobState{boss.id: boss}
	c.huntState.tr.add(boss.id)
	s.tickMobsLocked(c, 0.2)
	if !boss.returning {
		t.Fatal("a boss dragged past its 2m spawn leash must give up even with a player adjacent")
	}

	// A regular mob at the SAME 4m offset with the player adjacent keeps fighting (4 < 12).
	reg := &mobState{
		id: 2401, mobIdx: idx, mob: mob,
		x: sx + 4, y: sy, spawnX: sx, spawnY: sy,
		hp: mob.Health, shown: true, aggro: true, homed: true, boss: false,
	}
	c.x, c.y, c.snapT = reg.x - 1, reg.y, 0
	c.huntState.mobs = map[int32]*mobState{reg.id: reg}
	c.huntState.tr.add(reg.id)
	s.tickMobsLocked(c, 0.4)
	if reg.returning {
		t.Fatal("a regular mob only 4m from spawn (< 12m leash) must keep fighting an adjacent player")
	}
	if !reg.aggro {
		t.Fatal("regular mob should still be aggroed")
	}
}

// ---- Task 3: knockback is an animated glide, not a zero-velocity teleport snap ----

// TestKnockbackGlide: the shove moves the mob's authoritative position out at once but
// broadcasts a velocity (a glide), keeping the SERVER velocity 0 so it can't dead-reckon
// past the landing spot; the glide window (kbUntil) then closes on the next tick and the
// mob rests where the server held it.
func TestKnockbackGlide(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Miriam")
	defer cleanup()
	now := float64(s.battleTime())
	cx, cy := c.posAtLocked(float32(now))
	m := &mobState{id: 2500, mobIdx: 0, mob: gamedata.Mobs()[0],
		x: cx + 5, y: cy, hp: 500, maxHP: 500, shown: true}
	c.huntState.mobs[m.id] = m
	c.huntState.tr.add(m.id)

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	before := math.Hypot(float64(m.x-cx), float64(m.y-cy))
	s.knockbackMobLocked(c, m, 4, now)

	if m.kbUntil <= now {
		t.Fatalf("knockback opened no glide window (kbUntil=%g, now=%g)", m.kbUntil, now)
	}
	if m.vx != 0 || m.vy != 0 {
		t.Fatalf("server velocity must be 0 during the glide (got %.2f,%.2f) so it can't overshoot the landing spot", m.vx, m.vy)
	}
	if after := math.Hypot(float64(m.x-cx), float64(m.y-cy)); after <= before+0.5 {
		t.Fatalf("mob was not shoved away from the caster: %.2f -> %.2f", before, after)
	}
	landedX, landedY := m.x, m.y

	// Still gliding just before the window ends.
	if !s.advanceKnockbackLocked(c, m, m.kbUntil-0.01) {
		t.Fatal("glide should still be active just before kbUntil")
	}
	// Window elapsed: glide ends, flag cleared, resting exactly where the server held it.
	if s.advanceKnockbackLocked(c, m, m.kbUntil+0.01) {
		t.Fatal("glide should be over past kbUntil")
	}
	if m.kbUntil != 0 {
		t.Errorf("kbUntil not cleared after the glide: %g", m.kbUntil)
	}
	if m.x != landedX || m.y != landedY {
		t.Errorf("mob drifted after the glide ended: (%.2f,%.2f) -> (%.2f,%.2f)", landedX, landedY, m.x, m.y)
	}
}

// ---- Task 5: attacking / casting breaks stealth ----

func TestBreakInvisibilityClearsStealth(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Sandariel")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	// A cheap no-op when not stealthed (must not panic or touch state).
	hs.invisibleUntil = 0
	s.breakInvisibilityLocked(c, now)

	// Active stealth is revealed at once, buff icon dropped.
	hs.invisibleUntil = now + 10
	hs.invisBuffEffID = 777 // has a buff icon to remove (invisFxUID stays 0: no tracked fx)
	s.breakInvisibilityLocked(c, now)
	if hs.invisibleUntil != 0 {
		t.Errorf("stealth not broken: invisibleUntil=%g", hs.invisibleUntil)
	}
	if hs.invisBuffEffID != 0 {
		t.Errorf("stealth buff icon not removed: %d", hs.invisBuffEffID)
	}
}

// TestCastBreaksInvisibility: a successful skill cast reveals the player.
func TestCastBreaksInvisibility(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Miriam")
	defer cleanup()
	hs := c.huntState
	hs.skillLevel[2] = 1 // learn skill 3 (self-buff, no target needed)
	hs.mana = 500
	now := float64(s.battleTime())
	hs.invisibleUntil = now + 10

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	s.execCastLocked(c, 3, nil, c.x, c.y, false, 0)
	if hs.invisibleUntil != 0 {
		t.Errorf("a successful cast must break stealth, invisibleUntil=%g", hs.invisibleUntil)
	}
}

// ---- Task 1: fountain regen at the respawn point ----

// TestHuntFountainRegen: a living player standing on their respawn checkpoint recovers
// +10 HP/s on top of the normal trickle; away from it, only the trickle.
func TestHuntFountainRegen(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Miriam")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.respawnX, hs.respawnY = 50, 50

	// On the checkpoint.
	hs.hp = 1
	c.x, c.y, c.snapT = 50, 50, float32(now)
	s.memberTickLocked(c, now)
	atPoint := hs.hp - 1

	// Away from it.
	hs.hp = 1
	c.x, c.y, c.snapT = 200, 200, float32(now)
	s.memberTickLocked(c, now)
	away := hs.hp - 1

	if atPoint <= away {
		t.Fatalf("fountain gave no bonus: at-point +%g vs away +%g HP", atPoint, away)
	}
	// +10 HP/s over one 0.2s tick = +2 HP above the trickle.
	if got := atPoint - away; math.Abs(got-2) > 0.05 {
		t.Errorf("hunt fountain bonus = %g HP/tick, want ~2 (10 HP/s * 0.2s)", got)
	}
}

// TestDotaFountainFastRecovery: a «Штурм» player at their base heals a big fraction of
// max HP per tick (~2s to full), far faster than the Hunt trickle.
func TestDotaFountainFastRecovery(t *testing.T) {
	s, c, _, cleanup := newDotaConn(t, "Avtr_DPS_Miriam")
	defer cleanup()
	hs := c.huntState

	c.lock()
	defer c.unlock()
	now := float64(s.battleTime())
	// Anchor the base fountain at the player's live position.
	px, py := c.posAtLocked(float32(now))
	hs.respawnX, hs.respawnY = px, py
	maxHP := hs.maxHPLocked(now)
	hs.hp = maxHP * 0.1
	before := hs.hp
	s.memberTickLocked(c, now)
	gained := hs.hp - before

	want := maxHP * dotaFountainFracPerSec * tickInterval.Seconds()
	if gained < want*0.9 {
		t.Errorf("dota fountain healed %g in one tick, want ~%g (fast base recovery)", gained, want)
	}
}
