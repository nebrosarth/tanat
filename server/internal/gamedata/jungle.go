package gamedata

import (
	"math"
	"sort"
)

// jungle.go: procedural mob generation for map_4_2 («Заповедные джунгли»). Mirrors the
// crypt generator (buildDungeonPack40) -- medial-axis pack centres (navClearance),
// Poisson-disk spacing, Vogel-spiral members, region-mixed pools + rising level -- but
// with a JUNGLE roster (spiders/tribesmen/dinosaurs/gorillas/golems) and one extra
// rule: GOLEMS are gated to the trail to Titanid (golemAllowed). The four bosses are
// pinned last. Fully deterministic (fixed scan + stable sort + index cycling, no rand).

// jungleSpawn is the in-game-confirmed battle-start pocket (a real Reborn marker,
// ~(35,30)). Single source shared by the HuntMap (SpawnAt) and the generator's
// mob-free safe ring, so the two never drift apart.
var jungleSpawn = Vec2{X: 35.32, Y: 30.23}

// jungleBosses are the four arena bosses (absolute world X,Z, measured in-game),
// listed in the rising difficulty ladder the party clears in order: Grimlok < Fairy <
// Titanid < Anhel (final). Pinned after the generated trash so they own their arenas.
var jungleBosses = []struct {
	mob  int
	x, y float64
}{
	{mobBossGrimlok, 361.02, -190.31}, // 1 weakest, far SE
	{mobBossFairy, -32.56, 5.96},      // 2 central-SW (ranged)
	{mobBossTitanid, -239.89, 85.21},  // 3 far W
	{mobBossAnhel, 104.94, 322.66},    // 4 final, N (ranged)
}

// jungleBurntVillageCenter / Radius bound the map's ACTUAL burnt village -- the cluster of baked
// `House_0*_Burned` / `Stone_Burned*` / decorative `Skeleton_burned` props (extracted from
// map_4_2.unity3d) spread over ~44m around world (60,-128). It is a POOL-OVERRIDE zone: the normal
// pack generator runs inside it (medial-axis pack centres, Vogel-spiral members, radial level) but
// its roster is forced to BURNING SKELETONS only (mobSkeletonBurning / Mob_Skeleton_1H_Melee_05
// «Горящий скелет»), so the ruins fill with proper skeleton packs instead of six hand-placed pins.
// This gives the quest «Деревенский пожар» (Map_4_2 Stage2_2 «Уничтожить горящих скелетов в центре
// карты Заповедные джунгли») a real, natural target. Level comes from the same radial field as every
// other pack (rm.levelAt), so it reads as part of the map (~L13 there).
var jungleBurntVillageCenter = Vec2{X: 60.0, Y: -128.0}

const jungleBurntVillageRadius = 44.0 // covers the burned-house cluster (max prop ~r44 from centre)

// jungleBurntVillagePool is the skeleton-only roster for the ruins (the single burning-skeleton
// creature -- a cross-roster pin, not a native jungle mob).
var jungleBurntVillagePool = []int{mobSkeletonBurning}

// inJungleBurntVillageZone reports whether a world point lies in the burnt village (a simple
// disc around the burned-house cluster). Membership is tested per mob, so any creature standing
// in the ruins is a burning skeleton.
func inJungleBurntVillageZone(x, y float64) bool {
	return math.Hypot(x-jungleBurntVillageCenter.X, y-jungleBurntVillageCenter.Y) <= jungleBurntVillageRadius
}

// golemGate is the point on the trail to Titanid the user marked: golems (heavy stone
// guardians) may only spawn FROM here on -- the western Titanid approach. Enforced as a
// longitude gate west of the gate/Titanid (x -226/-239) but east of Fairy (x -138), so
// golems never leak toward the other bosses even if a western pack drifts.
var golemGate = Vec2{X: -226.48, Y: -154.89}

// golemAllowed reports whether a golem may spawn at (x,y): only on the western Titanid
// trail (x at or west of ~-180, a margin east of the gate to cover the whole approach).
func golemAllowed(x, _ float64) bool { return x <= golemGate.X+46 } // <= -180.48

// isGolem reports whether a mob index is a golem (gated creature).
func isGolem(mob int) bool { return mob == mobGolem || mob == mobGolemElite }

// jungleTribesmanZone is the southern "village" the user marked: inside it ONLY tribesmen
// spawn, overriding the nearest region's mixed pool. The points are CORRIDOR CENTRES the
// player walked; the zone is the WHOLE walkable area between them, not the thin boundary loop
// through them. So the region is a FLOOD FILL: every walkable cell reachable from a marked
// point, bounded by the convex hull of the marks (jungleTribesmanRegion). That fills each
// corridor to its full width and the pockets between them, while the hull stops the flood
// from leaking out into the rest of the connected jungle. First three points are the corridor
// centres the user gave as borders; the rest are interior spots that were still spawning
// dinos/gorillas and must also be tribesman. (Bosses are pinned outside the pack loop.)
var jungleTribesmanZone = []Vec2{
	{X: 246.49, Y: -207.35}, // corridor centre (E border)
	{X: 116.47, Y: -232.71}, // corridor centre (W border)
	{X: 184.90, Y: -272.25}, // corridor centre (S border)
	{X: 201.96, Y: -164.87}, // interior: was dino
	{X: 180.46, Y: -151.41}, // interior: was dinos
	{X: 166.70, Y: -131.81}, // interior (N reach): was gorilla
	{X: 166.11, Y: -270.57}, // interior: was dinos+gorilla
	{X: 113.30, Y: -255.83}, // interior: was dinos+tribe
}

// jungleTribesmanRegion floods the walkable village bounded by the hull of those marks.
var jungleTribesmanRegion = newNavRegion(navGrid42, jungleTribesmanZone)

// jungleTribesmanPool is the tribesman-only roster for that zone (melee/ranged/big --
// the pure tribe creatures, no zombie-tribe hybrid).
var jungleTribesmanPool = []int{mobTribesman, mobTribesmanRange, mobTribesmanBig}

func inJungleTribesmanZone(x, y float64) bool {
	return jungleTribesmanRegion.contains(x, y)
}

// jungleFlowerZone is the corridor the user marked with two clicks: inside a band around
// the segment A-B ONLY hunter-flowers spawn. Two points define a line, not an area, so the
// zone is a capsule (everything within jungleFlowerZoneWidth of the segment) -- a strip of
// flower patch along the walked path. (Bosses and the pinned burnt village are added outside
// the pack loop, so they are unaffected; this corridor doesn't overlap the tribesman zone.)
var jungleFlowerZone = [2]Vec2{
	{X: 5.39, Y: -297.97},
	{X: -66.66, Y: -190.32},
}

const jungleFlowerZoneWidth = 20.0 // half-width of the flower corridor around the segment

// jungleFlowerPool is the flower-only roster for that corridor (the rooted ranged
// hunter-flower; the jungle has a single flower creature).
var jungleFlowerPool = []int{mobFlowerHunter}

func inJungleFlowerZone(x, y float64) bool {
	return distToSeg(jungleFlowerZone[0], jungleFlowerZone[1], x, y) <= jungleFlowerZoneWidth
}

// navRegion is a per-area zone marked by a scatter of world points and FILLED over the
// navmesh: the region is the connected walkable area between the points, not a thin polygon
// through them. It is built by flood-filling walkable cells outward from every marked point,
// confined to the convex hull of the marks. The flood gives the full width of every corridor
// and the pockets between them (respecting walls, since it only steps to walkable cells); the
// hull caps the extent so the fill can't leak out through the village's open mouths into the
// rest of the (single connected) jungle. Membership is an O(1) precomputed per-cell mask.
type navRegion struct {
	ng   *NavGrid
	mask []bool // per-cell: true == in the region
}

// newNavRegion builds the fill mask: seed every marked point's cell, then BFS to walkable
// 4-neighbours that lie inside the hull. Seeds are marked unconditionally so a point the user
// gave is always in the region even if it sits exactly on the hull edge.
func newNavRegion(ng *NavGrid, marks []Vec2) *navRegion {
	hull := convexHull(marks)
	mask := make([]bool, ng.W*ng.H)
	queue := make([]int, 0, 1024)
	cellOf := func(v Vec2) (int, int, bool) {
		i := int(math.Floor((v.X - ng.MinX) / ng.Cell))
		j := int(math.Floor((v.Y - ng.MinY) / ng.Cell))
		return ng.nearestWalkable(i, j, 8)
	}
	// seed: mark a walkable cell unconditionally (used for the user's marks).
	seed := func(i, j int) {
		if i < 0 || i >= ng.W || j < 0 || j >= ng.H {
			return
		}
		idx := i*ng.H + j
		if mask[idx] || !ng.cellWalkable(i, j) {
			return
		}
		mask[idx] = true
		queue = append(queue, idx)
	}
	// grow: mark a walkable cell only if its centre is inside the hull.
	grow := func(i, j int) {
		if i < 0 || i >= ng.W || j < 0 || j >= ng.H {
			return
		}
		idx := i*ng.H + j
		if mask[idx] || !ng.cellWalkable(i, j) {
			return
		}
		if !pointInPolygon(hull, ng.cellCenterX(i), ng.cellCenterY(j)) {
			return
		}
		mask[idx] = true
		queue = append(queue, idx)
	}
	for _, m := range marks {
		if i, j, ok := cellOf(m); ok {
			seed(i, j)
		}
	}
	for h := 0; h < len(queue); h++ {
		idx := queue[h]
		i, j := idx/ng.H, idx%ng.H
		grow(i+1, j)
		grow(i-1, j)
		grow(i, j+1)
		grow(i, j-1)
	}
	return &navRegion{ng: ng, mask: mask}
}

// contains reports whether (x,y) falls in a filled region cell (O(1) mask lookup).
func (r *navRegion) contains(x, y float64) bool {
	i := int(math.Floor((x - r.ng.MinX) / r.ng.Cell))
	j := int(math.Floor((y - r.ng.MinY) / r.ng.Cell))
	if i < 0 || i >= r.ng.W || j < 0 || j >= r.ng.H {
		return false
	}
	return r.mask[i*r.ng.H+j]
}

// convexHull returns the convex hull (CCW) of pts via Andrew's monotone chain. The hull
// bounds how far the region fill may spread from the marked points.
func convexHull(pts []Vec2) []Vec2 {
	n := len(pts)
	if n < 3 {
		return append([]Vec2{}, pts...)
	}
	p := append([]Vec2{}, pts...)
	sort.Slice(p, func(i, j int) bool {
		if p[i].X != p[j].X {
			return p[i].X < p[j].X
		}
		return p[i].Y < p[j].Y
	})
	cross := func(o, a, b Vec2) float64 {
		return (a.X-o.X)*(b.Y-o.Y) - (a.Y-o.Y)*(b.X-o.X)
	}
	hull := make([]Vec2, 0, 2*n)
	for _, pt := range p { // lower chain
		for len(hull) >= 2 && cross(hull[len(hull)-2], hull[len(hull)-1], pt) <= 0 {
			hull = hull[:len(hull)-1]
		}
		hull = append(hull, pt)
	}
	lower := len(hull) + 1
	for i := n - 2; i >= 0; i-- { // upper chain
		pt := p[i]
		for len(hull) >= lower && cross(hull[len(hull)-2], hull[len(hull)-1], pt) <= 0 {
			hull = hull[:len(hull)-1]
		}
		hull = append(hull, pt)
	}
	return hull[:len(hull)-1]
}

// pointInPolygon is the even-odd (ray-cast) test for a closed polyline.
func pointInPolygon(poly []Vec2, x, y float64) bool {
	in := false
	j := len(poly) - 1
	for i := 0; i < len(poly); i++ {
		pi, pj := poly[i], poly[j]
		if (pi.Y > y) != (pj.Y > y) &&
			x < (pj.X-pi.X)*(y-pi.Y)/(pj.Y-pi.Y)+pi.X {
			in = !in
		}
		j = i
	}
	return in
}

// jungleRegions are the themed zone anchors: every generated pack takes the LEVEL and
// creature POOL of the nearest anchor, so difficulty rises with distance from the start
// and each approach is themed. Golems appear ONLY in the two western (Titanid-trail)
// pools; combined with golemAllowed that confines them to the western approach.
func jungleRegions() []dungeonRegion {
	// The jungle is a STAR: four trails fork out of the start pocket, each with its own
	// difficulty ramp, wrapped around a central massif. Because the anchors are leveled
	// by NAVMESH distance (regionMap.levelAt), a deep anchor that is close in a straight
	// line but far to WALK (across the massif) no longer inflates an early pack -- the
	// leak that put a level-11 pack a few steps from the spawn.
	return []dungeonRegion{
		// Start pocket (35,30): easy -- spiders + tribe scouts.
		{35, 30, 1, []int{mobSpider, mobSpider, mobTribesman, mobTribesmanRange, mobSpiderElite}},
		// SE trail to Grimlok (361,-190): dinosaurs + tribe.
		{200, -100, 4, []int{mobDino, mobTribesman, mobDinoRange, mobSpiderElite, mobTribesmanBig}},
		{340, -175, 7, []int{mobDino, mobDinoElite, mobDinoRange, mobGorilla, mobTribesmanBig}},
		// NW trail to Fairy (-138,124): tribe + spiders + dinos (NO golems).
		{-60, 110, 5, []int{mobTribesman, mobTribesmanRange, mobSpiderElite, mobDino, mobTribesmanZombie}},
		{-130, 150, 8, []int{mobDinoElite, mobGorilla, mobTribesmanBig, mobDinoRange, mobTribesmanZombie}},
		// N trail to Anhel (104,322): tough gorillas + dinos + brutes.
		{80, 200, 10, []int{mobGorilla, mobDinoElite, mobTribesmanBig, mobDinoRange, mobGorillaElite}},
		{104, 300, 13, []int{mobGorillaElite, mobDinoElite, mobTribesmanBig, mobGorilla, mobDinoRange}},
		// W / SW trail to Titanid -- GOLEM ZONE (gate -226,-154; Titanid -239,85).
		{-200, -80, 9, []int{mobGolem, mobGorilla, mobDinoElite, mobTribesmanBig, mobGolemElite}},
		{-235, 40, 12, []int{mobGolem, mobGolemElite, mobGorillaElite, mobDinoElite, mobTribesmanBig}},
	}
}

// junglePack is GENERATED from navGrid42 at package init (deterministic). See
// buildJunglePack42.
var junglePack = buildJunglePack42()

func buildJunglePack42() []MobSpawn {
	ng := navGrid42
	sx, sy := jungleSpawn.X, jungleSpawn.Y // REAL battle start (not navGrid42.Spawn(), a far-side seed)
	regions := jungleRegions()
	// Level by WALK distance from the spawn (radial ramp over the whole map), pool by
	// nearest anchor -- difficulty follows the trail a player walks, not straight-line
	// distance that leaks across the gaps between trails. See regionmap.go.
	rm := newRegionMap(ng, regions, sx, sy, huntLevelCeiling)

	// clearOf keeps packs off the start pocket and out of the boss arenas.
	clearOf := func(wx, wy float64) bool {
		if math.Hypot(wx-sx, wy-sy) < dungeonSpawnClear {
			return false
		}
		for _, b := range jungleBosses {
			if math.Hypot(wx-b.x, wy-b.y) < dungeonBossClear {
				return false
			}
		}
		return true
	}

	// Pass 1: seed pack centres on the medial axis (highest wall-clearance first),
	// greedily accepting while >= dungeonPackSpacing apart. Same as the crypt.
	clr := navClearance(ng)
	type cand struct {
		x, y float64
		clr  int
	}
	var cands []cand
	for j := 0; j < ng.H; j++ {
		for i := 0; i < ng.W; i++ {
			cv := clr[j*ng.W+i]
			if cv < dungeonMinClear {
				continue
			}
			wx, wy := ng.cellCenterX(i), ng.cellCenterY(j)
			if !clearOf(wx, wy) {
				continue
			}
			// The flower corridor is populated by its own wall-hugging pass
			// (buildJungleFlowers), not the medial-axis packs -- skip its centres.
			if inJungleFlowerZone(wx, wy) {
				continue
			}
			cands = append(cands, cand{wx, wy, cv})
		}
	}
	sort.SliceStable(cands, func(a, b int) bool { return cands[a].clr > cands[b].clr })
	var centres [][3]float64 // x, y, clearance(m)
	for _, cd := range cands {
		ok := true
		for _, c := range centres {
			if math.Hypot(cd.x-c[0], cd.y-c[1]) < dungeonPackSpacing {
				ok = false
				break
			}
		}
		if ok {
			centres = append(centres, [3]float64{cd.x, cd.y, float64(cd.clr)})
		}
	}

	// Pass 2: a pack per centre. Size skews bigger with region depth; members spread on
	// a Vogel spiral inside a clearance-capped radius; each member's stored coords are
	// proven walkable. A per-region counter cycles that region's mixed pool. Golems are
	// swapped to the region's first non-golem creature outside the Titanid trail.
	out := make([]MobSpawn, 0, len(centres)*3+len(jungleBosses))
	regionCounter := make([]int, len(regions))
	// Dedicated cycle counters for the pool-override zones (level/size stay the region's).
	var tribeCtr, villageCtr int
	for pi, c := range centres {
		cx, cy := c[0], c[1]
		effRad := dungeonPackRadius
		if lim := c[2] - 0.6; lim < effRad {
			effRad = lim
		}
		if effRad < 1.0 {
			effRad = 1.0
		}
		rg, ri := rm.nearest(cx, cy) // POOL + pack-size band from the nearest region (by path)
		size := dungeonPackSize(rg.level, pi)
		if fit := 1 + int((effRad/1.8)*(effRad/1.8)); size > fit {
			size = fit
		}
		spread := dungeonMemberSpacing
		if maxR := spread * math.Sqrt(float64(size-1)); maxR > effRad && maxR > 0 {
			spread *= effRad / maxR
		}
		for k := 0; k < size; k++ {
			var mx, my float64
			placed := false
			rad := spread * math.Sqrt(float64(k))
			for attempt := 0; attempt < 4; attempt++ {
				ang := float64(k)*2.399963 + float64(attempt)*0.7
				rx := math.Round((cx+rad*math.Cos(ang))*10) / 10
				ry := math.Round((cy+rad*math.Sin(ang))*10) / 10
				if ng.Walkable(rx, ry) {
					mx, my, placed = rx, ry, true
					break
				}
			}
			if !placed {
				continue
			}
			// Pool chosen PER MEMBER at its own position, so each zone's boundary is exact
			// even where a pack straddles it: tribesmen inside the southern nav-region, skeletons
			// in the burnt village, else the region's mix. FLOWERS are the exception -- they are
			// rooted plants placed along the corridor WALLS by buildJungleFlowers, so any medial-
			// axis member that lands in the flower corridor is dropped here (no centre flowers).
			var mob int
			switch {
			case inJungleTribesmanZone(mx, my):
				mob = jungleTribesmanPool[tribeCtr%len(jungleTribesmanPool)]
				tribeCtr++
			case inJungleFlowerZone(mx, my):
				continue
			case inJungleBurntVillageZone(mx, my):
				mob = jungleBurntVillagePool[villageCtr%len(jungleBurntVillagePool)]
				villageCtr++
			default:
				mob = rg.pool[regionCounter[ri]%len(rg.pool)]
				regionCounter[ri]++
				if isGolem(mob) && !golemAllowed(mx, my) {
					mob = firstNonGolem(rg.pool)
				}
			}
			// Level PER MEMBER at its own position: the radial field is continuous, so
			// packmates a couple of metres apart read the same level while packs a
			// step apart differ by at most one -- no cliff at the zone borders.
			out = append(out, MobSpawn{Mob: mob, DX: mx, DY: my, Abs: true, Level: rm.levelAt(mx, my)})
		}
	}

	// The flower corridor: rooted hunter-flowers lining the WALLS (not the open centre).
	out = append(out, buildJungleFlowers(ng, rm, clearOf)...)

	for _, b := range jungleBosses {
		out = append(out, MobSpawn{Mob: b.mob, DX: b.x, DY: b.y, Abs: true})
	}
	return out
}

// Flower placement tuning. Flowers spawn in CLUSTERS (a random 2..5 flowers each) strung along
// the corridor walls: cluster centres are spread jungleFlowerClusterSpacing apart, each cluster's
// flowers gathered from wall cells within jungleFlowerClusterRadius and kept jungleFlowerMemberSpacing
// apart. The per-cluster count is a DETERMINISTIC pseudo-random of the centre cell (the whole
// generator is rand-free so the map is reproducible), varying [Min,Max].
const (
	jungleFlowerClusterSpacing = 22.0 // min gap between flower clusters
	jungleFlowerClusterRadius  = 6.0  // how far a cluster's flowers spread along the wall
	jungleFlowerMemberSpacing  = 3.0  // gap between flowers inside a cluster
	jungleFlowerMin            = 2    // smallest cluster
	jungleFlowerMax            = 5    // largest cluster
)

// buildJungleFlowers places the flower corridor's rooted hunter-flowers in CLUSTERS along the
// corridor WALLS instead of the medial axis: it scans walkable EDGE cells (a walkable cell
// touching a non-walkable neighbour) inside the flower zone, seeds cluster centres spaced apart,
// then fills each with a random-sized group of nearby wall flowers. Level comes from the same
// radial field as every pack.
func buildJungleFlowers(ng *NavGrid, rm *regionMap, clearOf func(float64, float64) bool) []MobSpawn {
	isEdge := func(i, j int) bool {
		// A walkable cell is an edge if any 8-neighbour is off-mesh (a wall/void).
		for di := -1; di <= 1; di++ {
			for dj := -1; dj <= 1; dj++ {
				if di == 0 && dj == 0 {
					continue
				}
				if !ng.cellWalkable(i+di, j+dj) {
					return true
				}
			}
		}
		return false
	}
	// Every wall cell in the corridor, in a stable scan order.
	var edges []Vec2
	for j := 0; j < ng.H; j++ {
		for i := 0; i < ng.W; i++ {
			if !ng.cellWalkable(i, j) || !isEdge(i, j) {
				continue
			}
			wx, wy := ng.cellCenterX(i), ng.cellCenterY(j)
			if !inJungleFlowerZone(wx, wy) || !clearOf(wx, wy) {
				continue
			}
			edges = append(edges, Vec2{wx, wy})
		}
	}
	// Cluster centres: greedy Poisson-disk over the wall cells.
	var centres []Vec2
	for _, e := range edges {
		ok := true
		for _, c := range centres {
			if math.Hypot(e.X-c.X, e.Y-c.Y) < jungleFlowerClusterSpacing {
				ok = false
				break
			}
		}
		if ok {
			centres = append(centres, e)
		}
	}
	// Fill each cluster with a deterministic-random count of nearby wall flowers.
	var placed []Vec2
	var out []MobSpawn
	for _, c := range centres {
		h := uint32(int32(math.Round(c.X)))*73856093 ^ uint32(int32(math.Round(c.Y)))*19349663
		size := jungleFlowerMin + int(h%uint32(jungleFlowerMax-jungleFlowerMin+1))
		cnt := 0
		for _, e := range edges {
			if cnt >= size {
				break
			}
			if math.Hypot(e.X-c.X, e.Y-c.Y) > jungleFlowerClusterRadius {
				continue
			}
			tooClose := false
			for _, p := range placed {
				if math.Hypot(e.X-p.X, e.Y-p.Y) < jungleFlowerMemberSpacing {
					tooClose = true
					break
				}
			}
			if tooClose {
				continue
			}
			placed = append(placed, e)
			out = append(out, MobSpawn{Mob: mobFlowerHunter, DX: e.X, DY: e.Y, Abs: true, Level: rm.levelAt(e.X, e.Y)})
			cnt++
		}
	}
	return out
}

// firstNonGolem returns the first non-golem creature in a pool (the substitute when a
// golem rolls outside the Titanid trail). Pools always contain at least one.
func firstNonGolem(pool []int) int {
	for _, m := range pool {
		if !isGolem(m) {
			return m
		}
	}
	return pool[0]
}
