package gamedata

import "math"

// regionmap.go levels and themes mob packs by NAVMESH (path) distance instead of
// straight-line distance -- the fix for "level 11 right next to the spawn".
//
// The old regionLevelAt measured straight-line Hypot distance to region anchors
// and blended toward the LIST-adjacent anchor. That is fine on a LINEAR map (the
// crypt), where the region list runs start->deep along one corridor. But the
// jungle is a STAR of four trails forking from one hub: straight-line distance
// leaks across the gaps between trails, so a pack physically a few steps up one
// trail could be geometrically closest to a DEEP anchor on a neighbouring trail
// and inherit its level. The player saw exactly that: adjacent packs reading
// 5, 3, 2, 11, 3 instead of a smooth ramp.
//
// A player never crosses those gaps -- they WALK a corridor. So level follows the
// distance actually WALKED from the battle start: a single radial gradient over the
// WHOLE map, level 1 at the spawn rising to the map's ceiling at the far end (levelAt).
// This is monotonic everywhere -- difficulty rises the whole way in, never dipping
// back down as you cross the map -- which the per-anchor blend could not guarantee
// (the authored anchor bands are not consistent with walk distance: e.g. an anchor
// authored level 10 sits a short walk from the spawn while a level-7 anchor is much
// farther, so blending toward them made the ramp climb, plateau, then dip). The four
// bosses happen to sit at increasing walk distances (Grimlok < Fairy < Titanid <
// Anhel), so the radial ramp also lands them in ladder order for free.
//
// geoField turns "walk distance from X to any cell" into an O(1) lookup. The spawn
// field drives the LEVEL; the per-anchor fields drive the creature POOL (nearest).
type regionMap struct {
	ng         *NavGrid
	regions    []dungeonRegion
	fields     [][]float32 // fields[a][i*H+j] = walk cost from anchor a to that cell (POOL)
	spawnField []float32   // walk cost from the REAL battle start to every cell (LEVEL)
	maxDist    float64     // walk distance of the deepest-level anchor: the ramp's far end
	maxLevel   int         // that anchor's band: the map's difficulty ceiling
}

// newRegionMap floods a geodesic field from each anchor (for the pool) and one from
// the real battle start (spawnX,spawnY -- NOT ng.Spawn(), which on the jungle is a
// far-side seed marker; the HuntMap overrides it). `ceiling` is the level the ramp
// reaches at the far end of the map (the map's difficulty cap); maxDist is calibrated
// to the deepest AUTHORED anchor's walk distance, so the ramp hits `ceiling` there and
// clamps beyond. A ceiling higher than the authored bands (e.g. jungle 13 -> 20) just
// steepens the same ramp -- it does not touch the anchor bands that theme the pools.
func newRegionMap(ng *NavGrid, regions []dungeonRegion, spawnX, spawnY float64, ceiling int) *regionMap {
	rm := &regionMap{ng: ng, regions: regions}
	rm.fields = make([][]float32, len(regions))
	for i, r := range regions {
		rm.fields[i] = ng.geoField(r.x, r.y)
	}
	rm.spawnField = ng.geoField(spawnX, spawnY)

	// Calibrate maxDist to the deepest AUTHORED anchor (the far end of progression).
	deepest := 1
	for _, r := range regions {
		if r.level > deepest {
			deepest = r.level
			rm.maxDist = rm.spawnDistAt(r.x, r.y)
		}
	}
	rm.maxLevel = ceiling
	if rm.maxLevel < deepest {
		rm.maxLevel = deepest // never cap below an authored band
	}
	if rm.maxLevel < 1 {
		rm.maxLevel = 1
	}
	if rm.maxDist <= 0 || math.IsInf(rm.maxDist, 1) {
		rm.maxDist = rm.spawnFieldMax() // deepest anchor off-mesh: fall back to map radius
	}
	if rm.maxDist <= 0 {
		rm.maxDist = 1 // degenerate single-cell map: avoid div-by-zero
	}
	return rm
}

// spawnDistAt is the walk distance (cell units) from the battle start to world
// (wx,wy), +Inf if unreachable. The point is snapped to the nearest walkable cell.
func (rm *regionMap) spawnDistAt(wx, wy float64) float64 {
	g := rm.ng
	ci := int(math.Floor((wx - g.MinX) / g.Cell))
	cj := int(math.Floor((wy - g.MinY) / g.Cell))
	ci, cj, ok := g.nearestWalkable(ci, cj, anchorSnapRings)
	if !ok {
		return math.Inf(1)
	}
	d := rm.spawnField[ci*g.H+cj]
	if d >= math.MaxFloat32 {
		return math.Inf(1)
	}
	return float64(d)
}

// spawnFieldMax is the farthest walkable cell from the spawn -- the map's walk radius.
func (rm *regionMap) spawnFieldMax() float64 {
	best := 0.0
	for _, d := range rm.spawnField {
		if d < math.MaxFloat32 && float64(d) > best {
			best = float64(d)
		}
	}
	return best
}

// huntLevelCeiling is the mob level the radial ramp reaches at the far end of a Hunt
// map. 20 matches the crypt's deepest authored band; the jungle's authored anchors
// only reach 13, so this stretches its ramp to a full 1..20 endgame at the Anhel side.
const huntLevelCeiling = 20

// anchorSnapRings is how far (Chebyshev rings) an off-mesh point is snapped to the
// nearest walkable cell, shared by geoField (its source) and pathDist (its query) so
// the two agree. Generous, because the hand-placed region anchors are thematic
// centres that often sit in non-walkable void (they only ever fed Euclidean
// nearestRegion before). Pack members are walkable, so the snap is a no-op for them.
const anchorSnapRings = 48

// pathDist is the geodesic distance (cell units) from anchor a to world (wx,wy),
// or +Inf if the point is off-mesh or a can't reach it. The point's cell is snapped
// to the nearest walkable cell first, so an anchor in void resolves to the same
// walkable cell geoField flooded from (distance ~0), and a pack centre rounded onto a
// wall edge still resolves.
func (rm *regionMap) pathDist(a int, wx, wy float64) float64 {
	g := rm.ng
	ci := int(math.Floor((wx - g.MinX) / g.Cell))
	cj := int(math.Floor((wy - g.MinY) / g.Cell))
	ci, cj, ok := g.nearestWalkable(ci, cj, anchorSnapRings)
	if !ok {
		return math.Inf(1)
	}
	d := rm.fields[a][ci*g.H+cj]
	if d >= math.MaxFloat32 {
		return math.Inf(1)
	}
	return float64(d)
}

// nearest returns the region nearest to (wx,wy) BY PATH and its index. Drop-in for
// the old Euclidean nearestRegion; falls back to it only if every field is
// unreachable (which cannot happen on the spawn-connected grid the packs live on).
func (rm *regionMap) nearest(wx, wy float64) (dungeonRegion, int) {
	best, bi := math.Inf(1), -1
	for a := range rm.regions {
		if d := rm.pathDist(a, wx, wy); d < best {
			best, bi = d, a
		}
	}
	if bi < 0 {
		return nearestRegion(rm.regions, wx, wy)
	}
	return rm.regions[bi], bi
}

// levelAt returns the mob level for a pack at (wx,wy): a linear radial ramp on the
// WALK distance from the battle start -- level 1 at the spawn, rising to maxLevel at
// maxDist (the deepest-level anchor) and clamped there beyond. Because walk distance
// grows monotonically as a player heads deeper, difficulty rises the WHOLE way across
// the map and never dips back; two packs a step apart differ by at most one level.
// The far side of the map is hardest regardless of which trail leads there, and a
// pack that is close in a straight line but a long walk away (across the central
// massif) is correctly leveled by the walk, not the straight line.
func (rm *regionMap) levelAt(wx, wy float64) int {
	if rm.maxLevel <= 1 {
		return rm.maxLevel
	}
	d := rm.spawnDistAt(wx, wy)
	if math.IsInf(d, 1) {
		return 1 // off-mesh / unreachable: treat as the entrance
	}
	t := d / rm.maxDist
	if t < 0 {
		t = 0
	}
	if t > 1 {
		t = 1
	}
	n := int(math.Round(1 + float64(rm.maxLevel-1)*t))
	if n < 1 {
		n = 1
	}
	if n > rm.maxLevel {
		n = rm.maxLevel
	}
	return n
}

// geoField floods geodesic (navmesh) distances in CELL units from the walkable cell
// nearest world (wx,wy) across the whole grid, using the SAME 8-neighbour moves,
// diagonal cost (sqrt2) and corner-cut refusal as astar/Path -- so a field value
// equals the length of the route A* would walk. dist[i*H+j] is the shortest walk
// cost to that cell, or math.MaxFloat32 where unreachable. Dijkstra with a binary
// min-heap; runs at pack-generation time only.
func (g *NavGrid) geoField(wx, wy float64) []float32 {
	n := g.W * g.H
	dist := make([]float32, n)
	for i := range dist {
		dist[i] = math.MaxFloat32
	}
	si := int(math.Floor((wx - g.MinX) / g.Cell))
	sj := int(math.Floor((wy - g.MinY) / g.Cell))
	si, sj, ok := g.nearestWalkable(si, sj, anchorSnapRings)
	if !ok {
		return dist // degenerate: source off-mesh, everything stays unreachable
	}
	src := si*g.H + sj
	dist[src] = 0

	h := []fNode{{int32(src), 0}}
	for len(h) > 0 {
		top := heapPopF(&h)
		c := int(top.cell)
		if top.d > dist[c] {
			continue // stale heap entry
		}
		ci, cj := c/g.H, c%g.H
		for _, d := range dirs8 {
			ni, nj := ci+d.di, cj+d.dj
			if !g.cellWalkable(ni, nj) {
				continue
			}
			if d.diag && (!g.cellWalkable(ci+d.di, cj) || !g.cellWalkable(ci, cj+d.dj)) {
				continue // would clip the shared corner of two walls (same rule as A*)
			}
			step := float32(1)
			if d.diag {
				step = float32(sqrt2)
			}
			nc := ni*g.H + nj
			if nd := dist[c] + step; nd < dist[nc] {
				dist[nc] = nd
				heapPushF(&h, fNode{int32(nc), nd})
			}
		}
	}
	return dist
}

// fNode is one binary-heap entry for geoField, ordered by distance.
type fNode struct {
	cell int32
	d    float32
}

func heapPushF(h *[]fNode, v fNode) {
	s := append(*h, v)
	i := len(s) - 1
	for i > 0 {
		p := (i - 1) / 2
		if s[p].d <= s[i].d {
			break
		}
		s[p], s[i] = s[i], s[p]
		i = p
	}
	*h = s
}

func heapPopF(h *[]fNode) fNode {
	s := *h
	top := s[0]
	n := len(s) - 1
	s[0] = s[n]
	s = s[:n]
	i := 0
	for {
		l, r, small := 2*i+1, 2*i+2, i
		if l < n && s[l].d < s[small].d {
			small = l
		}
		if r < n && s[r].d < s[small].d {
			small = r
		}
		if small == i {
			break
		}
		s[i], s[small] = s[small], s[i]
		i = small
	}
	*h = s
	return top
}
