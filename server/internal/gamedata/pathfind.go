package gamedata

import (
	"math"
	"sync"
)

// Pathfinding turns the flat walkability grid into routes that go AROUND walls
// instead of stopping at them. Nav.Clip only walks a straight line until it hits
// geometry; Nav.Path runs A* over the grid cells and then string-pulls the cell
// path down to a minimal list of corner waypoints. The client is fully
// server-authoritative over movement (it renders whatever POSITION syncs we
// send, via NetSyncTransform's SmoothErrorCorrector), so a multi-waypoint route
// is drawn simply by syncing each leg's velocity in turn.
//
// A* uses a pooled scratch buffer with generation-stamped visited/closed marks
// (flat slices, no per-call zeroing or map allocation) and a hand-rolled binary
// heap, so a query costs tens of microseconds even on the 547x547 map and is
// cheap enough to run inside the 200ms combat tick for several chasing mobs.

const sqrt2 = 1.4142135623730951

// maxExpansions caps A* work so a pathological query (e.g. a goal walled off in
// its own pocket) can't stall the combat tick. Point-to-point queries on the
// real map expand far fewer cells than this thanks to the octile heuristic.
const maxExpansions = 200000

type step struct {
	di, dj int
	diag   bool
}

// dirs8 is the 8-neighbourhood; diagonals are marked so the search can refuse to
// cut through a wall corner.
var dirs8 = [8]step{
	{1, 0, false}, {-1, 0, false}, {0, 1, false}, {0, -1, false},
	{1, 1, true}, {1, -1, true}, {-1, 1, true}, {-1, -1, true},
}

// Path returns a smoothed sequence of world waypoints from (fx,fy) to a point at
// or nearest to (tx,ty), routing around walls. The start point itself is
// implicit: the slice begins at the first waypoint to walk to and ends at the
// goal. When the straight line is already clear the result is a single waypoint
// (the goal), so an unobstructed move stays a single leg. Returns nil when no
// route exists (caller should fall back to Clip or stay put).
func (g *NavGrid) Path(fx, fy, tx, ty float64) []Vec2 {
	si := int(math.Floor((fx - g.MinX) / g.Cell))
	sj := int(math.Floor((fy - g.MinY) / g.Cell))
	si, sj, ok := g.nearestWalkable(si, sj, 4)
	if !ok {
		return nil
	}
	gi := int(math.Floor((tx - g.MinX) / g.Cell))
	gj := int(math.Floor((ty - g.MinY) / g.Cell))
	// A click into a wall/void snaps to the nearest reachable cell (searched
	// wider than the start snap so distant off-mesh clicks still resolve).
	gi, gj, ok = g.nearestWalkable(gi, gj, 48)
	if !ok {
		return nil
	}

	// The exact goal is the click point when it is itself walkable, otherwise the
	// centre of the walkable cell we snapped to.
	goalX, goalY := tx, ty
	if !g.Walkable(tx, ty) {
		goalX = g.cellCenterX(gi)
		goalY = g.cellCenterY(gj)
	}

	// Fast path: a clear straight shot needs no search and no smoothing.
	if g.lineWalkable(fx, fy, goalX, goalY) {
		return []Vec2{{goalX, goalY}}
	}

	cells := g.astar(si, sj, gi, gj)
	if cells == nil {
		return nil
	}
	pts := make([]Vec2, len(cells))
	for i, c := range cells {
		pts[i] = Vec2{g.cellCenterX(c / g.H), g.cellCenterY(c % g.H)}
	}
	// Land on the exact goal rather than the last cell centre.
	pts[len(pts)-1] = Vec2{goalX, goalY}
	return g.stringPull(fx, fy, pts)
}

func (g *NavGrid) cellCenterX(i int) float64 { return g.MinX + (float64(i)+0.5)*g.Cell }
func (g *NavGrid) cellCenterY(j int) float64 { return g.MinY + (float64(j)+0.5)*g.Cell }

// nearestWalkable returns the truly nearest (min-Euclidean) walkable cell to
// (i,j) within maxR Chebyshev rings. Returns (i,j,true) immediately if it is
// already walkable. Because a Chebyshev ring r can hold cells as far as r*sqrt2
// (its corners) while the next ring's edge is only r+1 away, finding a hit in
// ring r is not enough — we keep scanning until the ring index exceeds the best
// distance found (no farther ring can beat it), then return the global best.
func (g *NavGrid) nearestWalkable(i, j, maxR int) (int, int, bool) {
	if g.cellWalkable(i, j) {
		return i, j, true
	}
	bi, bj, bd, found := 0, 0, math.Inf(1), false
	for r := 1; r <= maxR; r++ {
		if found && float64(r) > math.Sqrt(bd) {
			break // every cell in ring r is >= r away, farther than the best hit
		}
		for di := -r; di <= r; di++ {
			for dj := -r; dj <= r; dj++ {
				// ring only: skip the interior already covered by smaller r
				if di > -r && di < r && dj > -r && dj < r {
					continue
				}
				ni, nj := i+di, j+dj
				if !g.cellWalkable(ni, nj) {
					continue
				}
				if d := float64(di*di + dj*dj); d < bd {
					bd, bi, bj, found = d, ni, nj, true
				}
			}
		}
	}
	return bi, bj, found
}

// pathScratch is a reusable A* working set. Visited/closed membership is encoded
// by generation stamps (a cell belongs to "this query" iff its stamp == gen), so
// the buffers never need zeroing between queries. Pulled from scratchPool, so
// concurrent Path calls on the shared *NavGrid each get their own buffer — no
// shared mutable state, no lock.
type pathScratch struct {
	g      []float64 // gScore[c], valid iff seen[c]==gen
	came   []int32   // predecessor cell, valid iff seen[c]==gen
	seen   []uint32  // stamp: c has a gScore this query iff seen[c]==gen
	closed []uint32  // stamp: c finalised this query iff closed[c]==gen
	gen    uint32
	heap   []hNode
}

var scratchPool = sync.Pool{New: func() any { return &pathScratch{} }}

func (s *pathScratch) prepare(n int) uint32 {
	if cap(s.g) < n {
		s.g = make([]float64, n)
		s.came = make([]int32, n)
		s.seen = make([]uint32, n)
		s.closed = make([]uint32, n)
	} else {
		s.g = s.g[:n]
		s.came = s.came[:n]
		s.seen = s.seen[:n]
		s.closed = s.closed[:n]
	}
	s.gen++
	if s.gen == 0 { // wrapped: clear stamps so a stale 0 stamp isn't read as current
		for i := range s.seen {
			s.seen[i] = 0
			s.closed[i] = 0
		}
		s.gen = 1
	}
	s.heap = s.heap[:0]
	return s.gen
}

// hNode is one binary-heap entry, ordered by f-score.
type hNode struct {
	cell int32
	f    float64
}

func (s *pathScratch) push(cell int32, f float64) {
	s.heap = append(s.heap, hNode{cell, f})
	i := len(s.heap) - 1
	for i > 0 {
		p := (i - 1) / 2
		if s.heap[p].f <= s.heap[i].f {
			break
		}
		s.heap[p], s.heap[i] = s.heap[i], s.heap[p]
		i = p
	}
}

func (s *pathScratch) pop() int32 {
	h := s.heap
	top := h[0].cell
	n := len(h) - 1
	h[0] = h[n]
	h = h[:n]
	i := 0
	for {
		l, r, small := 2*i+1, 2*i+2, i
		if l < n && h[l].f < h[small].f {
			small = l
		}
		if r < n && h[r].f < h[small].f {
			small = r
		}
		if small == i {
			break
		}
		h[i], h[small] = h[small], h[i]
		i = small
	}
	s.heap = h
	return top
}

// astar finds a cell path from (si,sj) to (gi,gj) over the 8-neighbourhood with
// an octile heuristic, refusing to cut wall corners on diagonal steps. Returns
// the cell indices start..goal inclusive, or nil if unreachable / capped.
func (g *NavGrid) astar(si, sj, gi, gj int) []int {
	start := int32(si*g.H + sj)
	goal := int32(gi*g.H + gj)
	if start == goal {
		return []int{int(start)}
	}
	sc := scratchPool.Get().(*pathScratch)
	defer scratchPool.Put(sc)
	gen := sc.prepare(g.W * g.H)

	sc.g[start] = 0
	sc.seen[start] = gen
	sc.push(start, 0)
	expansions := 0
	for len(sc.heap) > 0 {
		c := sc.pop()
		if sc.closed[c] == gen {
			continue // stale duplicate heap entry
		}
		sc.closed[c] = gen
		if c == goal {
			return reconstruct(sc, start, goal)
		}
		if expansions++; expansions > maxExpansions {
			return nil
		}
		ci, cj := int(c)/g.H, int(c)%g.H
		gc := sc.g[c]
		for _, d := range dirs8 {
			ni, nj := ci+d.di, cj+d.dj
			if !g.cellWalkable(ni, nj) {
				continue
			}
			if d.diag && (!g.cellWalkable(ci+d.di, cj) || !g.cellWalkable(ci, cj+d.dj)) {
				continue // would clip the shared corner of two walls
			}
			nc := int32(ni*g.H + nj)
			if sc.closed[nc] == gen {
				continue
			}
			cost := 1.0
			if d.diag {
				cost = sqrt2
			}
			ng := gc + cost
			if sc.seen[nc] == gen && ng >= sc.g[nc] {
				continue
			}
			sc.g[nc] = ng
			sc.came[nc] = c
			sc.seen[nc] = gen
			sc.push(nc, ng+octile(ni, nj, gi, gj))
		}
	}
	return nil
}

// reconstruct walks the came-from chain from goal back to start and reverses it.
func reconstruct(sc *pathScratch, start, goal int32) []int {
	path := []int{int(goal)}
	for c := goal; c != start; {
		c = sc.came[c]
		path = append(path, int(c))
	}
	for l, r := 0, len(path)-1; l < r; l, r = l+1, r-1 {
		path[l], path[r] = path[r], path[l]
	}
	return path
}

// octile is the admissible 8-neighbour heuristic (straight + diagonal blend).
func octile(i, j, gi, gj int) float64 {
	dx := math.Abs(float64(i - gi))
	dy := math.Abs(float64(j - gj))
	if dx > dy {
		return dx + (sqrt2-1)*dy
	}
	return dy + (sqrt2-1)*dx
}

// stringPull collapses the cell-centre polyline to corner waypoints: from each
// anchor it advances to the farthest point whose straight segment is walkable
// (lineWalkable is an exact cell traversal, so smoothed legs can never graze a
// wall, not even a corner chord shorter than a cell), emits it, and re-anchors
// there. Linear: consecutive cells are adjacent, so re-anchoring always makes
// forward progress.
func (g *NavGrid) stringPull(sx, sy float64, pts []Vec2) []Vec2 {
	if len(pts) <= 1 {
		return pts
	}
	out := make([]Vec2, 0, len(pts))
	ax, ay := sx, sy
	i := 0
	for i < len(pts) {
		j := i
		for k := i + 1; k < len(pts); k++ {
			if g.lineWalkable(ax, ay, pts[k].X, pts[k].Y) {
				j = k
			} else {
				break // beyond here the anchor's view is occluded; re-anchor
			}
		}
		out = append(out, pts[j])
		ax, ay = pts[j].X, pts[j].Y
		i = j + 1
	}
	return out
}

// cornerEps is the tMax tie window lineWalkable treats as an exact corner
// crossing. Crossings within this of each other only arise from segments aimed
// (up to float rounding) through a lattice corner — e.g. the 45° cell-centre
// diagonals of a raw A* path; genuinely distinct crossings differ by far more.
const cornerEps = 1e-12

// lineWalkable reports whether the segment (ax,ay)->(bx,by) crosses walkable
// cells only. It is EXACT: an Amanatides–Woo grid traversal visits precisely
// the cells the segment passes through, so no blocked cell — however short its
// chord — can slip between samples (the old point-sampling version could graze
// a wall corner by up to a sample step). A crossing that hits a cell corner
// applies the same rule as A*'s diagonal step: both side cells must be walkable
// too, so smoothing can never squeeze between two corner-adjacent walls the raw
// path could not.
func (g *NavGrid) lineWalkable(ax, ay, bx, by float64) bool {
	// Work in grid space, where cell (i,j) spans [i,i+1)x[j,j+1) — the same
	// floor convention as Walkable.
	gx0, gy0 := (ax-g.MinX)/g.Cell, (ay-g.MinY)/g.Cell
	gx1, gy1 := (bx-g.MinX)/g.Cell, (by-g.MinY)/g.Cell
	i, j := int(math.Floor(gx0)), int(math.Floor(gy0))
	ie, je := int(math.Floor(gx1)), int(math.Floor(gy1))
	if !g.cellWalkable(i, j) {
		return false
	}
	dx, dy := gx1-gx0, gy1-gy0

	// Per-axis stepping state: direction, t of the first grid-line crossing,
	// and t per whole cell. A zero component never crosses its axis (+Inf).
	stepI, tMaxX, tDeltaX := 0, math.Inf(1), math.Inf(1)
	if dx > 0 {
		stepI, tDeltaX = 1, 1/dx
		tMaxX = (math.Floor(gx0) + 1 - gx0) / dx
	} else if dx < 0 {
		stepI, tDeltaX = -1, -1/dx
		tMaxX = (gx0 - math.Floor(gx0)) / -dx
	}
	stepJ, tMaxY, tDeltaY := 0, math.Inf(1), math.Inf(1)
	if dy > 0 {
		stepJ, tDeltaY = 1, 1/dy
		tMaxY = (math.Floor(gy0) + 1 - gy0) / dy
	} else if dy < 0 {
		stepJ, tDeltaY = -1, -1/dy
		tMaxY = (gy0 - math.Floor(gy0)) / -dy
	}

	// Walk cell to cell until the end cell. Every iteration advances at least
	// one axis one cell toward the end, so the Manhattan gap bounds the loop; if
	// float jitter ever steps past the end cell instead of onto it, the cap runs
	// out and the answer is a conservative "no" (never a false "yes").
	for n := iabs(ie-i) + iabs(je-j); n > 0 && (i != ie || j != je); n-- {
		switch d := tMaxX - tMaxY; {
		case d < -cornerEps: // vertical grid line first
			i += stepI
			tMaxX += tDeltaX
		case d > cornerEps: // horizontal grid line first
			j += stepJ
			tMaxY += tDeltaY
		default: // exact corner crossing: refuse to cut between two blocked cells
			if !g.cellWalkable(i+stepI, j) || !g.cellWalkable(i, j+stepJ) {
				return false
			}
			i += stepI
			j += stepJ
			tMaxX += tDeltaX
			tMaxY += tDeltaY
		}
		if !g.cellWalkable(i, j) {
			return false
		}
	}
	return i == ie && j == je
}

func iabs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
