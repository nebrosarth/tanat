package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

// Engine/behavioral tests for the 2026-07-20 pass-7 client-locale audit batch: the last 6
// avatars that had never had a full parallel sweep -- Teridin, Zamaran, Veritas, Hekata,
// Tangren fixed (Mihalych came back clean). Structural (data-shape) assertions for the
// same fixes live in gamedata/skill_fidelity_test.go; these confirm the actual runtime
// behavior through applyOpsLocked/the tick engine.

// TestTeridinSniperShotHitsOnce: the prior OpChannel{Dur:2,Interval:2} wrapper let a
// stationary target take a second, undescribed damage tick ~2s after the first. The fixed
// encoding is a single top-level OpDamage with no channel state at all.
func TestTeridinSniperShotHitsOnce(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_HK_Teridin")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	m := mkMob(t, 5000, 3, 0)
	m.hp, m.maxHP = 100000, 100000
	startHP := m.hp
	sp := hs.spellPowerLocked(now)

	c.mvMu.Lock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	ctx := opCtx{slot: 4, level: 1, target: m}
	s.applyOpsLocked(c, hs.kit.Skills[3].Ops, ctx, now)
	dmgAfterCast := startHP - m.hp
	nChannels := len(hs.channels)
	c.mvMu.Unlock()

	if nChannels != 0 {
		t.Fatalf("Teridin «Снайперский выстрел» must not open a channel, got %d live channels", nChannels)
	}
	if want := 130 + sp; dmgAfterCast != want { // rank-1 Value + PerSP, no double-hit (PVP balance redesign round 2, was 150 -> 160 -> 130)
		t.Fatalf("Teridin «Снайперский выстрел» dealt %g damage on a single application, want exactly %g (rank-1, no double-hit)", dmgAfterCast, want)
	}
}

// TestZamaranChargeDamageWaitsForArrival: «Таран»'s slow+AoE damage must land when the
// dash actually reaches the clicked point, not the instant the cast starts.
func TestZamaranChargeDamageWaitsForArrival(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	m := mkMob(t, 5100, 5, 0) // sits at the dash's landing point
	m.hp, m.maxHP = 100000, 100000
	startHP := m.hp

	c.mvMu.Lock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	ctx := opCtx{slot: 1, level: 1, px: 5, py: 0, hasPos: true}
	s.applyOpsLocked(c, hs.kit.Skills[0].Ops, ctx, now)
	dashUntil := hs.dashUntil
	hpAfterCast := m.hp
	slowAfterCast := m.st.slowUntil

	if dashUntil <= now {
		c.mvMu.Unlock()
		t.Fatalf("Zamaran «Таран» did not start a dash (dashUntil=%.2f now=%.2f)", dashUntil, now)
	}
	s.runDuePayloadsLocked(c, dashUntil+0.01)
	hpAfterArrival := m.hp
	slowAfterArrival := m.st.slowUntil
	c.mvMu.Unlock()

	if hpAfterCast != startHP {
		t.Fatalf("Zamaran «Таран» damaged the target on CAST (hp %g -> %g); must wait for arrival", startHP, hpAfterCast)
	}
	if slowAfterCast > now {
		t.Fatal("Zamaran «Таран» slowed the target on CAST; must wait for arrival")
	}
	if hpAfterArrival >= hpAfterCast {
		t.Fatalf("Zamaran «Таран» never damaged the target on arrival (hp stayed %g)", hpAfterArrival)
	}
	if slowAfterArrival <= dashUntil {
		t.Fatal("Zamaran «Таран» never slowed the target on arrival")
	}
}

// TestVeritasBlessingRegenIncludesSpellPower: «Благословение жизни»'s hp_regen buff must
// add the caster's own spell power on top of the flat rank value (PerSP), matching the
// client's "{hpInc}+{damageSP}" construction.
func TestVeritasBlessingRegenIncludesSpellPower(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Veritas")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())
	sp := hs.spellPowerLocked(now)

	c.mvMu.Lock()
	ctx := opCtx{slot: 3, level: 1}
	s.applyOpsLocked(c, hs.kit.Skills[2].Ops, ctx, now)
	var regen float64
	for _, m := range hs.st.mods {
		if m.stat == "hp_regen" {
			regen = m.value
		}
	}
	c.mvMu.Unlock()

	want := 2 + sp // rank-1 Value=2, PerSP=1
	if regen != want {
		t.Fatalf("Veritas «Благословение жизни» hp_regen mod = %g, want %g (2 flat + %g spell power)", regen, want, sp)
	}
}

// TestVeritasMetamorphosisDamageIsFlatAndSPScaled: «Метаморфоза» must grant a flat,
// SP-scaled dmg_flat bonus (matching the surviving damageBoost/damageSP TipArgs), not the
// old 40-70% dmg_pct total-damage multiplier.
func TestVeritasMetamorphosisDamageIsFlatAndSPScaled(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Veritas")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())
	sp := hs.spellPowerLocked(now)

	c.mvMu.Lock()
	ctx := opCtx{slot: 4, level: 1}
	s.applyOpsLocked(c, hs.kit.Skills[3].Ops, ctx, now)
	var dmgFlat, dmgPct, maxHP float64
	sawDmgPct := false
	for _, m := range hs.st.mods {
		switch m.stat {
		case "dmg_flat":
			dmgFlat = m.value
		case "dmg_pct":
			sawDmgPct = true
			dmgPct = m.value
		case "max_hp":
			maxHP = m.value
		}
	}
	c.mvMu.Unlock()

	if sawDmgPct {
		t.Fatalf("Veritas «Метаморфоза» must not grant a dmg_pct multiplier, got %g", dmgPct)
	}
	if want := 20 + sp; dmgFlat != want {
		t.Fatalf("Veritas «Метаморфоза» dmg_flat = %g, want %g (20 flat + %g spell power)", dmgFlat, want, sp)
	}
	if want := 150 + sp; maxHP != want {
		t.Fatalf("Veritas «Метаморфоза» max_hp = %g, want %g (150 flat + %g spell power)", maxHP, want, sp)
	}
}

// TestHekataSulfurCloudSilencesTarget: the target loses the ability to act (silenced) for
// the DoT's full 5s, and is never stunned.
func TestHekataSulfurCloudSilencesTarget(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Hekata")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	m := mkMob(t, 5200, 1, 0)
	c.mvMu.Lock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	ctx := opCtx{slot: 1, level: 1, target: m}
	s.applyOpsLocked(c, hs.kit.Skills[0].Ops, ctx, now)
	silenceUntil := m.st.silenceUntil
	stunned := m.st.stunUntil > now
	c.mvMu.Unlock()

	if stunned {
		t.Fatal("Hekata «Серное облако» must not stun the target")
	}
	if silenceUntil-now < 4.99 {
		t.Fatalf("Hekata «Серное облако» silence = %.2fs, want ~5s", silenceUntil-now)
	}
}

// TestTangrenCounterattackDealsOwnAttackOnDamage: «Контратака» is a chance-gated
// ON-DAMAGED proc -- once it rolls, it must deal Tangren's OWN attack damage to whoever
// struck him, not a percentage-of-incoming-damage thorns reflect.
func TestTangrenCounterattackDealsOwnAttackOnDamage(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_HK_Tangren")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	// Pull the real authored ops (so a future data change is caught) but force the roll
	// deterministic -- the actual Chance (15-30%) is covered by the gamedata structural test.
	realOp := hs.kit.Skills[1].Ops[0]
	if realOp.Kind != gamedata.OpProc || !realOp.OnDamaged {
		t.Fatal("Tangren «Контратака» is no longer an OnDamaged proc; test needs updating")
	}

	attacker := &mobState{id: 5300, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 100000, maxHP: 100000, shown: true}
	startHP := attacker.hp
	wantDmg := hs.baseAttackLocked(now)

	c.mvMu.Lock()
	hs.mobs[attacker.id] = attacker
	hs.tr.add(attacker.id)
	hs.defenseProcs = append(hs.defenseProcs, procState{slot: 2, chance: gamedata.PerLevel{1, 1, 1, 1}, ops: realOp.Ops})
	hs.skillLevel[1] = 1
	s.runDefenseProcsLocked(c, attacker, 10, now)
	dealt := startHP - attacker.hp
	c.mvMu.Unlock()

	if dealt <= 0 {
		t.Fatal("Tangren «Контратака» proc did not counter-attack the striker")
	}
	if diff := dealt - wantDmg; diff > 0.5 || diff < -0.5 {
		t.Fatalf("Tangren «Контратака» dealt %g, want ~%g (own base attack)", dealt, wantDmg)
	}
}
