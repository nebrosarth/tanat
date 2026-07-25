package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

// Behavioral test for the 2026-07-23 pass-12 audit batch: a third-opinion re-sweep of
// pass-8's 10 avatars (Rognar/Gektor/Lirvein/Sigilion/ShinDalar via qwen3-coder, Dutnik/
// Abominator/Arianna/Neirofim/Morlokay via gpt-oss:20b). Only Arianna's CC-immunity is new
// engine behavior; the three Morlokay fixes (PerSP/GrowthPerSP/DmgPerSP) reuse primitives
// already exercised by earlier passes' behavioral tests, so structural coverage in
// gamedata/skill_fidelity_test.go is sufficient for those.

// TestShieldGrantsCCImmuneBlocksStunForItsDuration: an OpShield with GrantsCCImmune must make
// ccImmuneBlockLocked report immunity for exactly the shield's own duration, with nothing to
// consume or reset (Arianna's «Щит хранителя» -- a cast-granted window, not Wilfang's
// passive consume-then-cooldown).
func TestShieldGrantsCCImmuneBlocksStunForItsDuration(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Sp_Arianna")
	defer cleanup()
	now := float64(s.battleTime())

	if s.ccImmuneBlockLocked(c, now) {
		t.Fatal("must not be CC-immune before the shield is cast")
	}

	op := gamedata.Op{Kind: gamedata.OpShield, Value: gamedata.PerLevel{120}, Dur: gamedata.PerLevel{5}, On: "ally", GrantsCCImmune: true}
	ctx := opCtx{slot: 1, level: 1, allyTarget: c}
	s.applyOpsLocked(c, []gamedata.Op{op}, ctx, now)

	if !s.ccImmuneBlockLocked(c, now+1) {
		t.Fatal("must be CC-immune while the shield is active")
	}
	if !s.ccImmuneBlockLocked(c, now+4.9) {
		t.Fatal("must still be CC-immune right up to the shield's duration")
	}
	if s.ccImmuneBlockLocked(c, now+5.1) {
		t.Fatal("must no longer be CC-immune once the shield's duration has elapsed")
	}
}

// TestOrdinaryShieldDoesNotGrantCCImmune: a plain OpShield without the flag must not touch
// tempCCImmuneUntil at all -- regression guard against GrantsCCImmune leaking onto every
// shield in the roster (Rognar's «Костяной щит», the many other OpShield casts, ...).
func TestOrdinaryShieldDoesNotGrantCCImmune(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Sp_Arianna")
	defer cleanup()
	now := float64(s.battleTime())

	op := gamedata.Op{Kind: gamedata.OpShield, Value: gamedata.PerLevel{120}, Dur: gamedata.PerLevel{5}, On: "ally"}
	ctx := opCtx{slot: 1, level: 1, allyTarget: c}
	s.applyOpsLocked(c, []gamedata.Op{op}, ctx, now)

	if s.ccImmuneBlockLocked(c, now+1) {
		t.Fatal("an ordinary shield (GrantsCCImmune=false) must not grant CC immunity")
	}
}
