package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

// Engine/behavioral tests for the 2026-07-23 pass-11 audit batch: a re-sweep of pass-7's 6
// avatars (Teridin/Zamaran/Veritas/Hekata/Tangren/Mihalych), the last of passes 4-7 to get a
// second-opinion re-verification. Structural (data-shape) assertions live in
// gamedata/skill_fidelity_test.go; this file covers the four fixes with real new engine
// behavior (OpOnKillStack PerSP, OpConsecutiveHit PerSP, view_radius_pct, OpDash.PushAside).

// TestHekataKillStackPerKillIncludesSP: activating «Культ жнеца» must fold the caster's
// spell power into the per-kill attack increment, not just the flat per-rank Value.
func TestHekataKillStackPerKillIncludesSP(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Hekata")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	c.mvMu.Lock()
	hs.st.mods = append(hs.st.mods, statMod{stat: "spell_power", value: 50})
	c.mvMu.Unlock()

	op := gamedata.Op{Kind: gamedata.OpOnKillStack, Value: gamedata.PerLevel{4}, Value2: gamedata.PerLevel{10}, Dur: gamedata.PerLevel{10}, PerSP: 1}
	ctx := opCtx{slot: 2, level: 1}
	s.applyOpsLocked(c, []gamedata.Op{op}, ctx, now)

	if hs.killWindowPerKill <= 4 {
		t.Fatalf("killWindowPerKill = %v, want > 4 (flat Value) once spell power is folded in", hs.killWindowPerKill)
	}
}

// TestMihalychConsecutiveHitPerHitIncludesSP: each stack of «Трепка» must scale with spell
// power, not just the flat per-rank Value.
func TestMihalychConsecutiveHitPerHitIncludesSP(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_HK_Mihalych")
	defer cleanup()
	hs := c.huntState
	hs.consecutiveHitSlot = 2
	hs.skillLevel[1] = 1

	baseline := s.consecutiveHitBonusLocked(hs, 999) // first hit on a fresh target: streak 0, bonus 0 either way
	_ = baseline
	second := s.consecutiveHitBonusLocked(hs, 999) // second hit, same target: streak 1, bonus = 1*per

	c.mvMu.Lock()
	hs.st.mods = append(hs.st.mods, statMod{stat: "spell_power", value: 50})
	c.mvMu.Unlock()
	s.consecutiveHitBonusLocked(hs, 998)                 // a different target: fresh streak
	buffedSecond := s.consecutiveHitBonusLocked(hs, 998) // second hit, same target, WITH spell power

	if buffedSecond <= second {
		t.Fatalf("consecutive-hit bonus did not grow with spell power: unbuffed=%v buffed=%v", second, buffedSecond)
	}
}

// TestVeritasViewRadiusShrinksEffectiveMobDistance: a member with an active view_radius_pct
// buff must reveal/hide mobs as if standing closer than they really are.
func TestVeritasViewRadiusShrinksEffectiveMobDistance(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Veritas")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	m := mkMob(t, 7700, 100, 0)
	c.mvMu.Lock()
	hs.mobs[m.id] = m
	c.mvMu.Unlock()

	baseline := nearestMemberDistLocked([]*conn{c}, m, now)

	c.mvMu.Lock()
	hs.st.mods = append(hs.st.mods, statMod{stat: "view_radius_pct", value: 1.2})
	c.mvMu.Unlock()
	buffed := nearestMemberDistLocked([]*conn{c}, m, now)

	if buffed >= baseline {
		t.Fatalf("view_radius_pct did not shrink the effective mob distance: baseline=%v buffed=%v", baseline, buffed)
	}
}

// TestZamaranChargePushAsideKnocksBackMobsOnPath: a mob standing on Zamaran's charge line
// (but not at the destination) must be knocked back the instant the dash starts.
func TestZamaranChargePushAsideKnocksBackMobsOnPath(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	// Caster at origin, charging to (20, 0); a mob sits at (10, 0) -- squarely on the path,
	// well short of the destination, so only PushAside (not the arrival AoE) can move it.
	onPath := mkMob(t, 7701, 10, 0)
	onPath.hp, onPath.maxHP = 1000, 1000
	c.mvMu.Lock()
	hs.mobs[onPath.id] = onPath
	hs.tr.add(onPath.id)
	c.mvMu.Unlock()
	startX, startY := onPath.x, onPath.y

	op := gamedata.Op{Kind: gamedata.OpDash, Value: gamedata.PerLevel{22}, StrikeOnArrival: true, PushAside: 3}
	ctx := opCtx{slot: 1, level: 1, px: 20, py: 0, hasPos: true}
	s.applyOpsLocked(c, []gamedata.Op{op}, ctx, now)

	if onPath.x == startX && onPath.y == startY {
		t.Fatal("PushAside did not move a mob standing on the dash's path")
	}
}
