package battleserver

import (
	"math"
	"testing"

	"tanatserver/internal/gamedata"
)

// TestVelialWillToWinBonus pins «Воля к победе»: the proc's bonus damage equals the
// rank coefficient × Velial's own missing-HP fraction (flat, unaffected by max HP or
// level), reproducing the in-game screenshot values.
func TestVelialWillToWinBonus(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())
	maxHP := hs.maxHPLocked(now)

	// The slot-3 proc's inner damage op (CasterMissingHP {40,60,80,100,100}).
	sk := hs.kit.Skills[2]
	if sk.Ops[0].Kind != gamedata.OpProc {
		t.Fatalf("Velial slot 3 op0 = %q, want proc", sk.Ops[0].Kind)
	}
	dmgOp := sk.Ops[0].Ops[0]

	cases := []struct {
		frac  float64 // missing-HP fraction
		level int
		want  float64
	}{
		{0.207, 5, 20.7}, // 3600/4540 -> +21
		{0.455, 5, 45.5}, // 2473/4540 -> +46
		{0.542, 5, 54.2}, // 908/1984  -> +54
		{0.296, 4, 29.6}, // 901/1280  -> +30 (rank 4 == rank 5 coeff)
		{0.186, 5, 18.6}, // 1862/2286 -> +19
	}
	for _, tc := range cases {
		hs.hp = maxHP * (1 - tc.frac)
		got := s.skillDamageLocked(c, dmgOp, opCtx{slot: 3, level: tc.level}, nil)
		if math.Abs(got-tc.want) > 1.0 {
			t.Errorf("missing=%.3f rank=%d: bonus=%.1f, want ~%.1f", tc.frac, tc.level, got, tc.want)
		}
	}

	// Full HP -> no bonus.
	hs.hp = maxHP
	if got := s.skillDamageLocked(c, dmgOp, opCtx{slot: 3, level: 5}, nil); got != 0 {
		t.Errorf("at full HP bonus=%.1f, want 0", got)
	}
}
