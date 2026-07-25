package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

// Engine/behavioral tests for the 2026-07-22 pass-10 audit batch: a re-sweep of pass-6's
// 10 avatars, this time run through a local Ollama harness (qwen3-coder/devstral/gpt-oss)
// with every candidate finding re-derived by hand against the actual dispatch code before
// being accepted. 7 confirmed (5 PerSP additions + a BasicAttackOnly gate + a new
// PctOfAttack primitive), several refuted along the way (Avrora s2's locale citation didn't
// even exist; Edilia s2's armor_pct debuff and Inshari s2's OpZoneArmor both already do what
// their locale text asks, just through a named mechanism the auditing model didn't trace).
// Structural (data-shape) assertions for the PerSP/BasicAttackOnly fixes live in
// gamedata/skill_fidelity_test.go; this file covers the new PctOfAttack primitive's actual
// runtime math.

// TestNerlagMeatGrinderDamageTracksLiveAttack: Op.PctOfAttack must scale off the caster's
// CURRENT base attack (hs.baseAttackLocked), not a value frozen at authoring time - so
// buffing/debuffing the caster's attack power between two otherwise-identical hits must
// change the damage.
func TestNerlagMeatGrinderDamageTracksLiveAttack(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Nerlag")
	defer cleanup()
	hs := c.huntState

	op := gamedata.Op{Kind: gamedata.OpDamage, PctOfAttack: gamedata.PerLevel{0.5, 0.6, 0.7, 0.8}, Scale: "magic"}
	ctx := opCtx{slot: 2, level: 1}

	m := mkMob(t, 6400, 1, 0)
	baseline := s.skillDamageLocked(c, op, ctx, m)

	c.mvMu.Lock()
	hs.st.mods = append(hs.st.mods, statMod{stat: "dmg_flat", value: 1000})
	c.mvMu.Unlock()
	buffed := s.skillDamageLocked(c, op, ctx, m)

	if buffed <= baseline {
		t.Fatalf("PctOfAttack damage did not track a live attack-power buff: baseline=%v buffed=%v", baseline, buffed)
	}
	wantDelta := 1000 * 0.5 // rank-1 coefficient applied to the +1000 dmg_flat buff
	gotDelta := buffed - baseline
	if gotDelta < wantDelta*0.99 || gotDelta > wantDelta*1.01 {
		t.Errorf("PctOfAttack delta = %v, want ~%v (50%% of the +1000 attack buff)", gotDelta, wantDelta)
	}
}

// TestNerlagMeatGrinderZeroWithoutBaseAttack: a rank with PctOfAttack unset (the zero-value
// PerLevel every other avatar's OpDamage still uses) must be a complete no-op, so this new
// field can never silently perturb existing skills.
func TestNerlagMeatGrinderZeroWithoutBaseAttack(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Nerlag")
	defer cleanup()

	withField := gamedata.Op{Kind: gamedata.OpDamage, Value: gamedata.PerLevel{10}, Scale: "phys"}
	withoutField := gamedata.Op{Kind: gamedata.OpDamage, Value: gamedata.PerLevel{10}, Scale: "phys", PctOfAttack: gamedata.PerLevel{}}
	ctx := opCtx{slot: 1, level: 1}
	m := mkMob(t, 6401, 1, 0)

	a := s.skillDamageLocked(c, withField, ctx, m)
	b := s.skillDamageLocked(c, withoutField, ctx, m)
	if a != b {
		t.Fatalf("an empty PctOfAttack must be a no-op: got %v vs %v", a, b)
	}
}
