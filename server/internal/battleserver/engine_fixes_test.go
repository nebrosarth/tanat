package battleserver

import (
	"math"
	"testing"

	"tanatserver/internal/gamedata"
)

// mkMob is a small helper: a shown, living mob at (x,y).
func mkMob(t *testing.T, id int32, x, y float32) *mobState {
	t.Helper()
	idx := mobIndexByPrefab(t, "Mob_Skeleton_1H_Melee_01")
	return &mobState{id: id, mobIdx: idx, mob: gamedata.Mobs()[idx], x: x, y: y,
		hp: gamedata.Mobs()[idx].Health, shown: true}
}

// TestCerberHealOnKill verifies «Кровавый пир»: killing an enemy heals the killer
// for a fraction of the slain foe's max HP (capped by Value2).
func TestCerberHealOnKill(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Cerber")
	defer cleanup()
	hs := c.huntState
	hs.skillLevel[2] = 1 // «Кровавый пир» learned
	hs.healOnKillSlot = 3

	m := mkMob(t, 3100, 1, 0)
	c.mvMu.Lock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	hs.hp = 50 // well below max so a heal is observable
	before := hs.hp
	s.hitMobLocked(c, m, m.hp+1, c.objID) // lethal
	after := hs.hp
	c.mvMu.Unlock()

	if !m.dead {
		t.Fatal("mob should be dead")
	}
	want := math.Min(0.1*m.maxHealth(), 80) // rank-1 coeff 0.1, cap 80
	if math.Abs((after-before)-want) > 0.5 {
		t.Fatalf("heal-on-kill = %g, want %g (hp %g -> %g)", after-before, want, before, after)
	}
}

// TestTeridinAttackRange verifies «Прицеливание»: an attack_range self-buff extends
// the avatar's effective auto-attack reach.
func TestTeridinAttackRange(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_HK_Teridin")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())
	base := hs.effAttackRangeLocked(now)
	hs.st.mods = append(hs.st.mods, statMod{stat: "attack_range", value: 4, until: 0})
	if got := hs.effAttackRangeLocked(now); math.Abs(got-(base+4)) > 1e-6 {
		t.Fatalf("effAttackRange = %g, want %g", got, base+4)
	}
}

// TestDutnikKnockback verifies the blast shoves a mob directly away from the caster.
func TestDutnikKnockback(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_HK_Dutnik")
	defer cleanup()
	hs := c.huntState
	m := mkMob(t, 3200, 2, 0) // 2 units in front of the caster at origin
	now := float64(s.battleTime())
	c.mvMu.Lock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	s.knockbackMobLocked(c, m, 5, now)
	dist := math.Hypot(float64(m.x), float64(m.y))
	c.mvMu.Unlock()
	if dist < 6.5 { // was 2, pushed +5 away => ~7
		t.Fatalf("knockback distance = %g, want ~7 (pushed away)", dist)
	}
}

// TestMaxTargetsCap verifies MaxTargets keeps only the N nearest victims.
func TestMaxTargetsCap(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Rognar")
	defer cleanup()
	hs := c.huntState
	c.mvMu.Lock()
	for i, d := range []float32{1, 2, 3, 4, 5} {
		m := mkMob(t, int32(3300+i), d, 0)
		hs.mobs[m.id] = m
		hs.tr.add(m.id)
	}
	op := gamedata.Op{Radius: 6, MaxTargets: 2}
	// slot 3 = Rognar's self-AoE «Могильный холод» (AoEWidth 0); centres on the caster.
	got := s.opTargetsLocked(c, opCtx{slot: 3, level: 1}, op)
	c.mvMu.Unlock()
	if len(got) != 2 {
		t.Fatalf("MaxTargets 2 returned %d victims", len(got))
	}
	for _, m := range got {
		if math.Hypot(float64(m.x), float64(m.y)) > 2.01 {
			t.Fatalf("kept a far victim at x=%g; MaxTargets should keep the 2 nearest", m.x)
		}
	}
}

// TestOnKillCooldownReset verifies OpOnKill runs its nested OpCooldownReset only when
// the cast's primary target died from the cast.
func TestOnKillCooldownReset(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Lirvein")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())
	ops := []gamedata.Op{{Kind: gamedata.OpOnKill, Ops: []gamedata.Op{{Kind: gamedata.OpCooldownReset}}}}

	// Live target: the on-kill branch must NOT fire.
	live := mkMob(t, 3400, 1, 0)
	c.mvMu.Lock()
	hs.cooldownUntil = [4]float64{10, 10, 10, 10}
	s.applyOpsLocked(c, ops, opCtx{slot: 2, level: 1, target: live}, now)
	liveCd := hs.cooldownUntil
	c.mvMu.Unlock()
	if liveCd[0] == 0 {
		t.Fatal("cooldowns reset on a LIVE target -- on-kill branch fired incorrectly")
	}

	// Dead target: the on-kill branch fires and clears every cooldown.
	dead := mkMob(t, 3401, 1, 0)
	dead.dead = true
	c.mvMu.Lock()
	hs.cooldownUntil = [4]float64{10, 10, 10, 10}
	s.applyOpsLocked(c, ops, opCtx{slot: 2, level: 1, target: dead}, now)
	deadCd := hs.cooldownUntil
	c.mvMu.Unlock()
	for slot, cd := range deadCd {
		if cd != 0 {
			t.Fatalf("cooldown slot %d = %g after on-kill reset, want 0", slot+1, cd)
		}
	}
}
