package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

// Engine/behavioral tests for the 2026-07-21 pass-9 audit batch: a second-opinion
// re-sweep of the 10 avatars from pass 5 (the pre-adversarial-verify era) found 20
// confirmed findings (2 candidates were refuted). None of the 10 came back clean this
// time -- even Cerber, clean in pass 5. Structural (data-shape) assertions for the same
// fixes live in gamedata/skill_fidelity_test.go; these confirm the actual runtime
// behavior of the new engine primitives (GrowthPerSP, ExcludeCenterTarget, OnKill/
// OnAnyDamage procs, toggle-stealth, negative dmg_reduction_pct, ExplodeSP).

// TestTitanidEarthquakeGrowthAddsSPToRamp: Op.GrowthPerSP must add an EXTRA, SP-scaled
// term to the per-pulse damage ramp on top of the flat Growth increment.
func TestTitanidEarthquakeGrowthAddsSPToRamp(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Titanid")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	plain := mkMob(t, 6300, 1, 0)
	plain.hp, plain.maxHP = 100000, 100000
	withSP := mkMob(t, 6301, -1, 0)
	withSP.hp, withSP.maxHP = 100000, 100000

	c.mvMu.Lock()
	hs.mobs[plain.id] = plain
	hs.mobs[withSP.id] = withSP
	hs.tr.add(plain.id)
	hs.tr.add(withSP.id)
	hs.st.mods = append(hs.st.mods, statMod{stat: "spell_power", value: 50})

	// Two synthetic channels, identical except one carries GrowthPerSP -- isolates its
	// contribution to the per-pulse damage bonus.
	hs.channels = []channelState{
		{slot: 1, level: 1, until: now + 10, interval: 100, nextPulse: now, target: plain.id,
			ops: []gamedata.Op{{Kind: gamedata.OpDamage, Radius: 0}}, growth: 10, pulseCount: 2},
		{slot: 1, level: 1, until: now + 10, interval: 100, nextPulse: now, target: withSP.id,
			ops: []gamedata.Op{{Kind: gamedata.OpDamage, Radius: 0}}, growth: 10, growthSP: 1, pulseCount: 2},
	}
	s.tickChannelsLocked(c, now)
	dmgPlain := 100000 - plain.hp
	dmgWithSP := 100000 - withSP.hp
	c.mvMu.Unlock()

	if dmgPlain <= 0 {
		t.Fatal("baseline flat-growth pulse must deal some damage")
	}
	if dmgWithSP <= dmgPlain {
		t.Errorf("GrowthPerSP must add extra ramp damage on top of flat growth: plain=%g, withSP=%g", dmgPlain, dmgWithSP)
	}
}

// TestTitanidShockwaveExcludesPrimaryTarget: the splash must skip the primary target
// itself («все ДРУГИЕ враги вокруг него»), not hit it a second time.
func TestTitanidShockwaveExcludesPrimaryTarget(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Titanid")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	primary := mkMob(t, 6400, 0, 0)
	primary.hp, primary.maxHP = 100000, 100000
	other := mkMob(t, 6401, 1, 0)
	other.hp, other.maxHP = 100000, 100000

	var splash gamedata.Op
	for _, op := range hs.kit.Skills[1].Ops {
		if op.Kind == gamedata.OpDamage && op.Radius > 0 {
			splash = op
		}
	}
	if !splash.ExcludeCenterTarget {
		t.Fatal("Titanid «Ударная волна» splash is no longer ExcludeCenterTarget; test needs updating")
	}

	c.mvMu.Lock()
	hs.mobs[primary.id] = primary
	hs.mobs[other.id] = other
	hs.tr.add(primary.id)
	hs.tr.add(other.id)
	s.applyOpsLocked(c, []gamedata.Op{splash},
		opCtx{slot: 2, level: 1, target: primary, px: primary.x, py: primary.y, hasPos: true}, now)
	primaryAfter := primary.hp
	otherAfter := other.hp
	c.mvMu.Unlock()

	if primaryAfter != 100000 {
		t.Errorf("splash must exclude the primary target, but its hp dropped to %g", primaryAfter)
	}
	if otherAfter >= 100000 {
		t.Error("splash must still hit other enemies in radius")
	}
}

// TestGayalZombieRaisesOnKillNotOnHit: the zombie-raise proc must be registered into
// killProcs (rolled once per KILL), not procs (rolled on every landing hit).
func TestGayalZombieRaisesOnKillNotOnHit(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Gayal")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	var procOp gamedata.Op
	for _, op := range hs.kit.Skills[1].Ops {
		if op.Kind == gamedata.OpProc {
			procOp = op
		}
	}
	if !procOp.OnKill {
		t.Fatal("Gayal «Аура погибших» is no longer an OnKill proc; test needs updating")
	}
	hs.skillLevel[1] = 1
	// newHuntConn skips world-build registration -- register manually, mirroring how the
	// real registration loop would route an OnKill op into killProcs, not procs.
	hs.killProcs = []procState{{slot: 2, chance: gamedata.PerLevel{1}, ops: procOp.Ops}}
	hs.summonProtos = map[string]int32{"Mob_ZombieCrawl_01": 800}

	victim := mkMob(t, 6200, 1, 0)
	c.mvMu.Lock()
	hs.mobs[victim.id] = victim
	hs.tr.add(victim.id)

	before := len(hs.summons)
	s.runProcsLocked(c, victim, now) // a landing hit (not a kill) on the on-hit path
	afterHit := len(hs.summons)
	s.runKillProcsLocked(c, victim, now) // the kill itself
	afterKill := len(hs.summons)
	c.mvMu.Unlock()

	if afterHit != before {
		t.Errorf("a landing hit must not raise a zombie (registered in killProcs, not procs): %d -> %d", before, afterHit)
	}
	if afterKill <= afterHit {
		t.Error("a kill must raise a zombie")
	}
}

// TestAstarotToggleGrantsAndRevokesStealth: «Слуга тьмы» must grant real invisibility
// while the toggle is on (kept alive every tick), and break it immediately on toggle-off.
func TestAstarotToggleGrantsAndRevokesStealth(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_HK_Astarot")
	defer cleanup()
	hs := c.huntState
	hs.skillLevel[1] = 1 // «Слуга тьмы» learned
	now := float64(s.battleTime())

	c.mvMu.Lock()
	s.toggleSkillLocked(c, 2) // click ON
	onUntil := hs.invisibleUntil
	onSlot := hs.toggleStealthSlot
	c.mvMu.Unlock()

	if onSlot != 2 {
		t.Fatalf("toggleStealthSlot = %d, want 2", onSlot)
	}
	if onUntil <= now {
		t.Fatalf("stealth not granted on activation: invisibleUntil=%g now=%g", onUntil, now)
	}

	c.mvMu.Lock()
	s.tickTogglesLocked(c, now+0.5) // one server tick later
	refreshed := hs.invisibleUntil
	c.mvMu.Unlock()
	if refreshed <= onUntil {
		t.Errorf("stealth must be re-armed every tick while the toggle stays on: %g -> %g", onUntil, refreshed)
	}

	c.mvMu.Lock()
	s.toggleSkillLocked(c, 2) // click again -> toggles OFF
	offUntil := hs.invisibleUntil
	offSlot := hs.toggleStealthSlot
	c.mvMu.Unlock()

	if offSlot != 0 {
		t.Errorf("toggleStealthSlot must clear on toggle-off, got %d", offSlot)
	}
	if offUntil > now+0.5 {
		t.Errorf("stealth must break immediately on toggle-off, invisibleUntil=%g", offUntil)
	}
}

// TestAnhelPhantomCallFiresFromSkillDamage: «Зов фантомов» must also roll when Anhel's
// own ACTIVE skill damage lands, not only a basic-attack landing.
func TestAnhelPhantomCallFiresFromSkillDamage(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Psh_Anhel")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	var procOp gamedata.Op
	for _, op := range hs.kit.Skills[2].Ops {
		if op.Kind == gamedata.OpProc {
			procOp = op
		}
	}
	if !procOp.OnAnyDamage {
		t.Fatal("Anhel «Зов фантомов» is no longer an OnAnyDamage proc; test needs updating")
	}
	hs.skillLevel[2] = 1
	hs.anyDamageProcs = []procState{{slot: 3, chance: gamedata.PerLevel{1}, ops: procOp.Ops}}
	hs.summonProtos = map[string]int32{"Avtr_Psh_Anhel": 801}

	target := mkMob(t, 6600, 1, 0)
	target.hp, target.maxHP = 100000, 100000

	c.mvMu.Lock()
	hs.mobs[target.id] = target
	hs.tr.add(target.id)
	before := len(hs.summons)
	// Simulate a skill-cast OpDamage landing (Anhel's s1 nuke).
	s.applyOpsLocked(c, []gamedata.Op{{Kind: gamedata.OpDamage, Value: gamedata.PerLevel{50}, Scale: "magic"}},
		opCtx{slot: 1, level: 1, target: target}, now)
	after := len(hs.summons)
	c.mvMu.Unlock()

	if after <= before {
		t.Error("Anhel's own skill damage landing must roll the OnAnyDamage proc (clone summon)")
	}
}

// TestBlackDragonFrenzyAmplifiesIncomingDamage: a NEGATIVE dmg_reduction_pct must
// AMPLIFY incoming damage by roughly its magnitude, the opposite of the positive
// (Rognar) branch which reduces it.
func TestBlackDragonFrenzyAmplifiesIncomingDamage(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_BlackDragon")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	m := mkMob(t, 6700, 1, 0)
	c.mvMu.Lock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)

	startHP := hs.maxHPLocked(now)
	hs.hp = startHP
	s.hitPlayerFromLocked(c, m.id, 100, now, m, nil)
	dmgWithout := startHP - hs.hp

	hs.hp = startHP
	hs.st.mods = append(hs.st.mods, statMod{stat: "dmg_reduction_pct", value: -0.2})
	s.hitPlayerFromLocked(c, m.id, 100, now, m, nil)
	dmgWith := startHP - hs.hp
	c.mvMu.Unlock()

	if dmgWithout <= 0 {
		t.Fatal("baseline hit must deal some damage")
	}
	if dmgWith <= dmgWithout {
		t.Errorf("negative dmg_reduction_pct must AMPLIFY incoming damage: without=%g with=%g", dmgWithout, dmgWith)
	}
	if diff := dmgWith - dmgWithout*1.2; diff > 0.5 || diff < -0.5 {
		t.Errorf("dmg_reduction_pct=-0.2 should add ~20%% damage: without=%g with=%g (want ~%g)", dmgWithout, dmgWith, dmgWithout*1.2)
	}
}

// TestPlusMinusSuperconductivityDoesNotDoubleHitStruckMob: the chain proc must exclude
// the already-struck mob it centers on, hitting only genuine neighbors.
func TestPlusMinusSuperconductivityDoesNotDoubleHitStruckMob(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_PlusMinus")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	var procOp gamedata.Op
	for _, op := range hs.kit.Skills[2].Ops {
		if op.Kind == gamedata.OpProc {
			procOp = op
		}
	}
	excludes := false
	for _, nested := range procOp.Ops {
		if nested.Kind == gamedata.OpDamage && nested.ExcludeCenterTarget {
			excludes = true
		}
	}
	if !excludes {
		t.Fatal("PlusMinus «Сверхпроводимость» chain is no longer ExcludeCenterTarget; test needs updating")
	}
	hs.skillLevel[2] = 1
	hs.procs = []procState{{slot: 3, chance: gamedata.PerLevel{1}, ops: procOp.Ops}}

	struckMob := mkMob(t, 6500, 0, 0)
	struckMob.hp, struckMob.maxHP = 100000, 100000
	neighbor := mkMob(t, 6501, 1, 0)
	neighbor.hp, neighbor.maxHP = 100000, 100000

	c.mvMu.Lock()
	hs.mobs[struckMob.id] = struckMob
	hs.mobs[neighbor.id] = neighbor
	hs.tr.add(struckMob.id)
	hs.tr.add(neighbor.id)
	s.runProcsLocked(c, struckMob, now)
	struckAfter := struckMob.hp
	neighborAfter := neighbor.hp
	c.mvMu.Unlock()

	if struckAfter != 100000 {
		t.Errorf("the already-struck mob must not be hit again by its own chain, hp dropped to %g", struckAfter)
	}
	if neighborAfter >= 100000 {
		t.Error("the chain must still reach a genuine neighbor")
	}
}

// TestWilfangPoisonBiteExplosionUsesExplodeSP: the periodic tick must deal zero damage,
// while the death explosion scales with spell power via its OWN ExplodeSP field,
// decoupled from the (now zero) tick PerSP.
func TestWilfangPoisonBiteExplosionUsesExplodeSP(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Wilfang")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	var dotOp gamedata.Op
	for _, op := range hs.kit.Skills[2].Ops {
		for _, nested := range op.Ops {
			if nested.Kind == gamedata.OpDot {
				dotOp = nested
			}
		}
	}
	if dotOp.Kind != gamedata.OpDot {
		t.Fatal("Wilfang «Ядовитый укус» must carry a nested OpDot")
	}

	m := mkMob(t, 6800, 1, 0)
	c.mvMu.Lock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	hs.st.mods = append(hs.st.mods, statMod{stat: "spell_power", value: 100})
	s.applyOpsLocked(c, []gamedata.Op{dotOp}, opCtx{slot: 3, level: 1, target: m}, now)
	var tickDmg float64
	if len(m.st.dots) > 0 {
		tickDmg = m.st.dots[0].perSec
	}
	explodeDmg := m.st.poisonExplodeDmg
	c.mvMu.Unlock()

	if tickDmg != 0 {
		t.Errorf("poison tick must deal no damage, perSec=%g", tickDmg)
	}
	wantExplode := dotOp.ExplodeDamage.At(1) + 100*dotOp.ExplodeSP
	if explodeDmg != wantExplode {
		t.Errorf("death explosion = %g, want %g (base + spellPower*ExplodeSP)", explodeDmg, wantExplode)
	}
}
