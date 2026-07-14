package gamedata

import (
	"math"
	"math/rand"
	"testing"
)

// openGrid returns a WxH grid with every cell walkable.
func openGrid(w, h int) *NavGrid {
	g := &NavGrid{MinX: 0, MinY: 0, Cell: 1, W: w, H: h,
		bits: make([]uint64, (w*h+63)/64), spawn: Vec2{0.5, 0.5}}
	for i := 0; i < w; i++ {
		for j := 0; j < h; j++ {
			b := i*h + j
			g.bits[b>>6] |= 1 << (uint(b) & 63)
		}
	}
	return g
}

func (g *NavGrid) clear(i, j int) {
	b := i*g.H + j
	g.bits[b>>6] &^= 1 << (uint(b) & 63)
}

// TestLineWalkableOctants: the DDA must traverse correctly in all 8 octants,
// axis-aligned both ways, and handle degenerate (same-cell, zero-length) and
// off-grid segments. The classic bug it guards is the tMax init for negative
// step directions (distance to the LOWER grid line).
func TestLineWalkableOctants(t *testing.T) {
	g := openGrid(9, 9)
	cx, cy := 4.5, 4.5 // centre cell (4,4)

	// All-open grid: every ray from the centre to any border cell centre passes.
	for _, d := range [][2]float64{
		{4, 1}, {4, -1}, {-4, 1}, {-4, -1}, // shallow octants
		{1, 4}, {1, -4}, {-1, 4}, {-1, -4}, // steep octants
		{4, 0}, {-4, 0}, {0, 4}, {0, -4}, // axis-aligned
		{4, 4}, {-4, -4}, {4, -4}, {-4, 4}, // exact diagonals
	} {
		if !g.lineWalkable(cx, cy, cx+d[0], cy+d[1]) {
			t.Errorf("open grid: segment to offset (%+.0f,%+.0f) reported blocked", d[0], d[1])
		}
	}

	// Zero-length and same-cell segments reduce to the start cell's bit.
	if !g.lineWalkable(cx, cy, cx, cy) {
		t.Error("zero-length segment on a walkable cell reported blocked")
	}
	if !g.lineWalkable(4.1, 4.1, 4.9, 4.9) {
		t.Error("same-cell segment reported blocked")
	}

	// Unwalkable start or end fails; off-grid fails (cellWalkable bound checks).
	g.clear(4, 4)
	if g.lineWalkable(cx, cy, cx+2, cy) {
		t.Error("segment starting on a blocked cell reported walkable")
	}
	if g.lineWalkable(cx+2, cy, cx, cy) {
		t.Error("segment ending on a blocked cell reported walkable")
	}
	if g.lineWalkable(1.5, 1.5, -3.5, 1.5) {
		t.Error("segment leaving the grid reported walkable")
	}

	// A single blocked cell in each octant's path must block that ray only.
	g = openGrid(9, 9)
	g.clear(6, 5) // on the ray toward (+4,+2), off the ray toward (+4,-2)
	if g.lineWalkable(cx, cy, cx+4, cy+2) {
		t.Error("ray through blocked cell (6,5) reported walkable")
	}
	if !g.lineWalkable(cx, cy, cx+4, cy-2) {
		t.Error("ray avoiding blocked cell (6,5) reported blocked")
	}
}

// TestLineWalkableCornerRule: a segment through the exact shared corner of two
// blocked cells must be rejected (the same no-corner-cut rule as A*'s diagonal
// step), while the same corner with walkable side cells passes.
func TestLineWalkableCornerRule(t *testing.T) {
	// Blocked side cells: checkerboard around the corner (2,2).
	//   . #        walls at (1,2) and (2,1); the diagonal (1,1)->(2,2) crosses
	//   # .        their shared corner and must NOT pass.
	g := openGrid(5, 5)
	g.clear(1, 2)
	g.clear(2, 1)
	if g.lineWalkable(1.5, 1.5, 2.5, 2.5) {
		t.Error("diagonal cut between two corner-adjacent blocked cells")
	}
	// One blocked side cell is enough to refuse (matches astar's rule).
	g = openGrid(5, 5)
	g.clear(2, 1)
	if g.lineWalkable(1.5, 1.5, 2.5, 2.5) {
		t.Error("corner crossing with one blocked side cell must be rejected")
	}
	// Both side cells open: the exact-corner diagonal passes.
	g = openGrid(5, 5)
	if !g.lineWalkable(1.5, 1.5, 2.5, 2.5) {
		t.Error("corner crossing with open side cells reported blocked")
	}
	// The corner rule must not fire for a wall the segment merely passes NEAR:
	// a blocked cell diagonally adjacent to the path but not on it.
	g = openGrid(5, 5)
	g.clear(1, 3)
	if !g.lineWalkable(1.5, 1.5, 2.5, 2.5) {
		t.Error("blocked cell off the traversed diagonal blocked the segment")
	}
}

// TestLineWalkableSupersetOfDenseSampling: on the real map, whenever brutally
// dense point sampling (0.01 cell) finds a blocked point on a segment, the DDA
// must reject that segment too — exactness means the DDA's rejections are a
// superset of any sampler's. (The reverse is allowed: the DDA also catches
// corner chords and corner cuts that samplers slip past.)
func TestLineWalkableSupersetOfDenseSampling(t *testing.T) {
	g := navGrid40
	rng := rand.New(rand.NewSource(7))
	denseBlocked := func(ax, ay, bx, by float64) bool {
		dx, dy := bx-ax, by-ay
		n := int(math.Hypot(dx, dy)/(g.Cell*0.01)) + 1
		for i := 0; i <= n; i++ {
			t := float64(i) / float64(n)
			if !g.Walkable(ax+dx*t, ay+dy*t) {
				return true
			}
		}
		return false
	}
	checked, blockedN := 0, 0
	for n := 0; n < 10000; n++ {
		ax := g.MinX + rng.Float64()*float64(g.W)*g.Cell
		ay := g.MinY + rng.Float64()*float64(g.H)*g.Cell
		bx := ax + (rng.Float64()-0.5)*40
		by := ay + (rng.Float64()-0.5)*40
		if !denseBlocked(ax, ay, bx, by) {
			continue // sampler found nothing; the DDA may still (rightly) reject
		}
		checked++
		if g.lineWalkable(ax, ay, bx, by) {
			t.Fatalf("segment %d (%.3f,%.3f)->(%.3f,%.3f): dense sampling found a blocked point but the DDA passed it",
				n, ax, ay, bx, by)
		}
		blockedN++
	}
	if checked < 100 {
		t.Fatalf("too few blocked segments sampled (%d) for the superset check to mean anything", checked)
	}
	t.Logf("verified %d dense-blocked segments are all DDA-blocked", blockedN)
}

// TestRawAstarLegsPassLineWalkable pins the smoothing safety invariant: every
// consecutive cell-centre leg of a RAW astar() path (including the exact-corner
// 45° diagonals) must pass lineWalkable, so stringPull can never do worse than
// the unsmoothed path. Checked on the real map and the synthetic wall grid.
func TestRawAstarLegsPassLineWalkable(t *testing.T) {
	check := func(t *testing.T, g *NavGrid, cells []int, tag string) {
		t.Helper()
		for k := 1; k < len(cells); k++ {
			ax, ay := g.cellCenterX(cells[k-1]/g.H), g.cellCenterY(cells[k-1]%g.H)
			bx, by := g.cellCenterX(cells[k]/g.H), g.cellCenterY(cells[k]%g.H)
			if !g.lineWalkable(ax, ay, bx, by) {
				t.Fatalf("%s: raw astar leg (%.2f,%.2f)->(%.2f,%.2f) rejected by lineWalkable", tag, ax, ay, bx, by)
			}
		}
	}

	// Synthetic wall grid: the detour through the gap.
	wg := wallGrid()
	if cells := wg.astar(1, 0, 3, 0); cells == nil {
		t.Fatal("wallGrid astar found no route")
	} else {
		check(t, wg, cells, "wallGrid")
	}

	// Real map: random walkable start/goal pairs.
	g := navGrid40
	var walk []int
	for i := 0; i < g.W; i++ {
		for j := 0; j < g.H; j++ {
			if g.cellWalkable(i, j) {
				walk = append(walk, i*g.H+j)
			}
		}
	}
	rng := rand.New(rand.NewSource(3))
	legs := 0
	for n := 0; n < 200; n++ {
		a, b := walk[rng.Intn(len(walk))], walk[rng.Intn(len(walk))]
		cells := g.astar(a/g.H, a%g.H, b/g.H, b%g.H)
		if cells == nil {
			t.Fatalf("astar found no route between two walkable cells of the same component")
		}
		check(t, g, cells, "map_4_0")
		legs += len(cells) - 1
	}
	t.Logf("verified %d raw astar legs pass the exact lineWalkable", legs)
}
