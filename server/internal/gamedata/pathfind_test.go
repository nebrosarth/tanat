package gamedata

import (
	"math"
	"testing"
)

// wallGrid builds a small grid with a single vertical wall that has a gap, so a
// route from the left of the wall to the right must detour through the gap.
//
//	col x:  0 1 2 3 4   (W=5)
//	        . . # . .   row 0
//	        . . # . .   row 1
//	        . . . . .   row 2  <- gap at (2,2)
//	        . . # . .   row 3
//	        . . # . .   row 4
func wallGrid() *NavGrid {
	const W, H = 5, 5
	g := &NavGrid{MinX: 0, MinY: 0, Cell: 1, W: W, H: H,
		bits: make([]uint64, (W*H+63)/64), spawn: Vec2{0.5, 0.5}}
	set := func(i, j int) {
		b := i*H + j
		g.bits[b>>6] |= 1 << (uint(b) & 63)
	}
	for i := 0; i < W; i++ {
		for j := 0; j < H; j++ {
			if i == 2 && j != 2 {
				continue // the wall column, except the gap at j==2
			}
			set(i, j)
		}
	}
	return g
}

func TestPathClearLineSingleWaypoint(t *testing.T) {
	g := wallGrid()
	// Two points on the same open side (left column), clear straight line.
	p := g.Path(0.5, 0.5, 0.5, 3.5)
	if len(p) != 1 {
		t.Fatalf("clear line should be a single leg, got %d waypoints: %v", len(p), p)
	}
	if math.Abs(p[0].X-0.5) > 1e-6 || math.Abs(p[0].Y-3.5) > 1e-6 {
		t.Fatalf("single waypoint should be the goal, got %v", p[0])
	}
}

func TestPathRoutesAroundWall(t *testing.T) {
	g := wallGrid()
	// From left of the wall to right of the wall: must detour through the gap row.
	p := g.Path(1.5, 0.5, 3.5, 0.5)
	if len(p) == 0 {
		t.Fatal("no route found around the wall")
	}
	// The route must end at the goal.
	last := p[len(p)-1]
	if math.Abs(last.X-3.5) > 1e-6 || math.Abs(last.Y-0.5) > 1e-6 {
		t.Fatalf("route should end at goal (3.5,0.5), got %v", last)
	}
	// Every leg must stay on walkable ground (no cutting through the wall).
	ax, ay := 1.5, 0.5
	for _, wp := range p {
		if !g.lineWalkable(ax, ay, wp.X, wp.Y) {
			t.Fatalf("leg (%.1f,%.1f)->(%.1f,%.1f) crosses a wall", ax, ay, wp.X, wp.Y)
		}
		ax, ay = wp.X, wp.Y
	}
	// A straight shot would pass through the wall column at x=2, y=0.5 (blocked),
	// so the path must visit the gap region (some waypoint with y around 2).
	viaGap := false
	for _, wp := range p {
		if wp.Y > 1.5 {
			viaGap = true
		}
	}
	if !viaGap {
		t.Fatalf("route did not detour through the gap: %v", p)
	}
}

func TestPathNoCornerCut(t *testing.T) {
	// A grid where a diagonal step would clip a wall corner.
	//   . #
	//   # .
	// walkable at (0,0) and (1,1); walls at (0,1) and (1,0). A diagonal from
	// (0,0) to (1,1) must NOT be allowed (it cuts the corner); with both
	// diagonal neighbours walls, the two cells are disconnected.
	const W, H = 2, 2
	g := &NavGrid{MinX: 0, MinY: 0, Cell: 1, W: W, H: H,
		bits: make([]uint64, 1), spawn: Vec2{0.5, 0.5}}
	set := func(i, j int) { b := i*H + j; g.bits[b>>6] |= 1 << (uint(b) & 63) }
	set(0, 0)
	set(1, 1)
	if p := g.astar(0, 0, 1, 1); p != nil {
		t.Fatalf("A* cut a wall corner: %v", p)
	}
}

func TestPathGoalInWallSnaps(t *testing.T) {
	g := wallGrid()
	// Aim straight at a wall cell (2,0); Path should snap to a nearby walkable
	// cell and still return a usable route from the left side.
	p := g.Path(1.5, 0.5, 2.5, 0.5)
	if len(p) == 0 {
		t.Fatal("no route to a goal that landed in a wall")
	}
	last := p[len(p)-1]
	if !g.Walkable(last.X, last.Y) {
		t.Fatalf("snapped goal %v is not walkable", last)
	}
}

func TestNearestWalkableTrulyNearest(t *testing.T) {
	// 9x9 grid, query (4,4). Only two walkable cells: the ring-3 corner (7,7)
	// at Euclidean 4.243 and the ring-4 edge (8,4) at 4.0. The nearest is (8,4)
	// even though (7,7) is found in an earlier Chebyshev ring.
	const W, H = 9, 9
	g := &NavGrid{MinX: 0, MinY: 0, Cell: 1, W: W, H: H,
		bits: make([]uint64, (W*H+63)/64), spawn: Vec2{0.5, 0.5}}
	set := func(i, j int) { b := i*H + j; g.bits[b>>6] |= 1 << (uint(b) & 63) }
	set(7, 7)
	set(8, 4)
	i, j, ok := g.nearestWalkable(4, 4, 48)
	if !ok {
		t.Fatal("nearestWalkable found nothing")
	}
	if i != 8 || j != 4 {
		t.Fatalf("nearestWalkable = (%d,%d), want the closer (8,4) not the ring-3 corner (7,7)", i, j)
	}
}

func TestPathUnreachableReturnsNil(t *testing.T) {
	// Two disconnected rooms: cells (0,0) and (4,4) only, nothing between.
	const W, H = 5, 5
	g := &NavGrid{MinX: 0, MinY: 0, Cell: 1, W: W, H: H,
		bits: make([]uint64, (W*H+63)/64), spawn: Vec2{0.5, 0.5}}
	set := func(i, j int) { b := i*H + j; g.bits[b>>6] |= 1 << (uint(b) & 63) }
	set(0, 0)
	set(4, 4)
	if p := g.astar(0, 0, 4, 4); p != nil {
		t.Fatalf("expected no route between disconnected cells, got %v", p)
	}
}

// TestPathOnRealMap exercises the shipped map_4_0 grid: a route from spawn to a
// far reachable point must exist, stay walkable, and (when obstructed) contain
// more than one leg.
func TestPathOnRealMap(t *testing.T) {
	sx, sy := navGrid40.Spawn()
	// Walk to the first dungeon mob's location (known reachable from spawn).
	tx, ty := sx+dungeonPack40[0].DX, sy+dungeonPack40[0].DY
	p := navGrid40.Path(sx, sy, tx, ty)
	if len(p) == 0 {
		t.Fatalf("no route from spawn (%.1f,%.1f) to mob (%.1f,%.1f)", sx, sy, tx, ty)
	}
	ax, ay := sx, sy
	for _, wp := range p {
		if !navGrid40.lineWalkable(ax, ay, wp.X, wp.Y) {
			t.Fatalf("leg (%.1f,%.1f)->(%.1f,%.1f) leaves walkable ground", ax, ay, wp.X, wp.Y)
		}
		ax, ay = wp.X, wp.Y
	}
	if d := math.Hypot(ax-tx, ay-ty); d > 1.0 {
		t.Fatalf("route ended %.1f from the goal", d)
	}
}
