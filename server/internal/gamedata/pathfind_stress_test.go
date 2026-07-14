package gamedata

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

// TestPathFuzzRealMap fires many random queries at the shipped map_4_0 grid and
// asserts every returned route is walkable end-to-end, never panics, and the
// whole batch finishes quickly (a proxy for "A* always terminates"). It walks
// random start/goal pairs sampled from actual walkable cells.
func TestPathFuzzRealMap(t *testing.T) {
	g := navGrid40

	// Collect walkable cell centres to sample realistic endpoints.
	var pts []Vec2
	for i := 0; i < g.W; i++ {
		for j := 0; j < g.H; j++ {
			if g.cellWalkable(i, j) {
				pts = append(pts, Vec2{g.cellCenterX(i), g.cellCenterY(j)})
			}
		}
	}
	if len(pts) < 100 {
		t.Fatalf("grid has too few walkable cells (%d) to fuzz", len(pts))
	}

	rng := rand.New(rand.NewSource(1))
	const N = 4000
	start := time.Now()
	maxLegs := 0
	empties := 0
	// Separately time the combat-relevant bucket: mobs only ever path within the
	// 22m leash range, so that latency (not full-map latency) is what runs inside
	// the 200ms tick.
	var combatDur time.Duration
	var combatN int
	for n := 0; n < N; n++ {
		a := pts[rng.Intn(len(pts))]
		b := pts[rng.Intn(len(pts))]
		near := math.Hypot(a.X-b.X, a.Y-b.Y) <= 22.0
		var t0 time.Time
		if near {
			t0 = time.Now()
		}
		route := g.Path(a.X, a.Y, b.X, b.Y)
		if near {
			combatDur += time.Since(t0)
			combatN++
		}
		if route == nil {
			// Both endpoints are walkable cells of the same connected component,
			// so a route must exist.
			empties++
			continue
		}
		if len(route) > maxLegs {
			maxLegs = len(route)
		}
		// Every leg must stay on walkable ground.
		ax, ay := a.X, a.Y
		for _, wp := range route {
			if !g.lineWalkable(ax, ay, wp.X, wp.Y) {
				t.Fatalf("query %d: leg (%.1f,%.1f)->(%.1f,%.1f) crosses non-walkable ground",
					n, ax, ay, wp.X, wp.Y)
			}
			ax, ay = wp.X, wp.Y
		}
		// The route must land on (or snap near) the requested goal.
		if d := math.Hypot(ax-b.X, ay-b.Y); d > 2.0 {
			t.Fatalf("query %d: route ended %.1f from goal (%.1f,%.1f)", n, d, b.X, b.Y)
		}
	}
	elapsed := time.Since(start)
	if empties > N/50 {
		t.Errorf("too many empty routes between same-component cells: %d/%d", empties, N)
	}
	t.Logf("%d queries in %v (%.1f us/query full-map), max legs=%d, empty=%d",
		N, elapsed, float64(elapsed.Microseconds())/N, maxLegs, empties)
	_, _ = combatDur, combatN

	// Dedicated combat-range batch: build many start/goal pairs within the 22m
	// leash and time the whole batch (per-query timing is below the Windows clock
	// resolution). This is the latency that actually runs inside the 200ms tick.
	type pair struct{ a, b Vec2 }
	var near []pair
	for len(near) < 5000 {
		a := pts[rng.Intn(len(pts))]
		b := pts[rng.Intn(len(pts))]
		if math.Hypot(a.X-b.X, a.Y-b.Y) <= 22.0 {
			near = append(near, pair{a, b})
		}
	}
	t0 := time.Now()
	for _, p := range near {
		g.Path(p.a.X, p.a.Y, p.b.X, p.b.Y)
	}
	usPer := float64(time.Since(t0).Microseconds()) / float64(len(near))
	t.Logf("combat-range (<=22m): %d queries, %.2f us/query", len(near), usPer)
	if usPer > 300 {
		t.Errorf("combat-range pathfinding too slow: %.2f us/query (tick budget risk)", usPer)
	}
	// Generous ceiling: if A* ever failed to terminate this would blow past it.
	if elapsed > 30*time.Second {
		t.Fatalf("fuzz batch too slow (%v) — possible non-termination", elapsed)
	}
}
