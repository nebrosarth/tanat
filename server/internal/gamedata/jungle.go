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

// jungleRegions are the themed zone anchors: every generated pack takes the LEVEL and
// creature POOL of the nearest anchor, so difficulty rises with distance from the start
// and each approach is themed. Golems appear ONLY in the two western (Titanid-trail)
// pools; combined with golemAllowed that confines them to the western approach.
func jungleRegions() []dungeonRegion {
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
	sx, sy := jungleSpawn.X, jungleSpawn.Y
	regions := jungleRegions()

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
	for pi, c := range centres {
		cx, cy := c[0], c[1]
		effRad := dungeonPackRadius
		if lim := c[2] - 0.6; lim < effRad {
			effRad = lim
		}
		if effRad < 1.0 {
			effRad = 1.0
		}
		rg, ri := nearestRegion(regions, cx, cy)
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
			mob := rg.pool[regionCounter[ri]%len(rg.pool)]
			regionCounter[ri]++
			if isGolem(mob) && !golemAllowed(mx, my) {
				mob = firstNonGolem(rg.pool)
			}
			out = append(out, MobSpawn{Mob: mob, DX: mx, DY: my, Abs: true, Level: rg.level})
		}
	}

	for _, b := range jungleBosses {
		out = append(out, MobSpawn{Mob: b.mob, DX: b.x, DY: b.y, Abs: true})
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
