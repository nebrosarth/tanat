package battleserver

import (
	"math"
	"testing"

	"tanatserver/internal/gamedata"
)

// TestVigilansUltIsCharge pins the data: the ult's dash must ignore obstacles
// (NoClip) and defer its strike to arrival (StrikeOnArrival).
func TestVigilansUltIsCharge(t *testing.T) {
	vig, _ := gamedata.AvatarByID(20)
	kit := gamedata.SkillsFor(vig)
	var dash *gamedata.Op
	for i := range kit.Skills[3].Ops {
		if kit.Skills[3].Ops[i].Kind == gamedata.OpDash {
			dash = &kit.Skills[3].Ops[i]
		}
	}
	if dash == nil {
		t.Fatal("Vigilans ult has no dash")
	}
	if !dash.NoClip {
		t.Error("ult dash must be NoClip (leap through obstacles)")
	}
	if !dash.StrikeOnArrival {
		t.Error("ult dash must StrikeOnArrival (damage on impact, not on cast)")
	}
}

// TestChargeDashIgnoresObstacles: a no-clip charge drives straight to the exact
// target even when a wall blocks the straight line, whereas the normal (clipping)
// dash stops short at the wall.
func TestChargeDashIgnoresObstacles(t *testing.T) {
	s, c, nav, sx, sy := newNavConn(t)

	// Find a walkable target whose STRAIGHT line from spawn is blocked by geometry
	// (nav.Clip stops well short of it).
	var gx, gy float32
	found := false
	for r := float32(6); r <= 40 && !found; r += 2 {
		for a := 0; a < 24 && !found; a++ {
			ang := float64(a) * math.Pi / 12
			tx := sx + r*float32(math.Cos(ang))
			ty := sy + r*float32(math.Sin(ang))
			if !nav.Walkable(float64(tx), float64(ty)) {
				continue
			}
			cx, cy := nav.Clip(float64(sx), float64(sy), float64(tx), float64(ty))
			if math.Hypot(float64(tx)-cx, float64(ty)-cy) > 1.5 {
				gx, gy, found = tx, ty, true
			}
		}
	}
	if !found {
		t.Skip("no wall-blocked straight target near spawn on this map")
	}

	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	// Clipping dash stops short of the blocked target.
	c.moveStraightExLocked(s, gx, gy, true)
	clipped := c.path[len(c.path)-1]
	if math.Hypot(clipped.X-float64(gx), clipped.Y-float64(gy)) < 0.5 {
		t.Fatalf("clipping dash reached the target %v — the chosen target was not actually blocked", clipped)
	}

	// No-clip charge drives straight to the exact target, through the wall.
	c.moveStraightExLocked(s, gx, gy, false)
	end := c.path[len(c.path)-1]
	if math.Hypot(end.X-float64(gx), end.Y-float64(gy)) > 0.01 {
		t.Fatalf("no-clip charge endpoint %v is not the target (%.1f,%.1f) — obstacles were not ignored", end, gx, gy)
	}
}
