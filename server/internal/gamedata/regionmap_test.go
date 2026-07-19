package gamedata

import (
	"math"
	"testing"
)

// polyLen sums the leg lengths of a Path polyline (implicit start at fx,fy).
func polyLen(p []Vec2, fx, fy float64) float64 {
	if len(p) == 0 {
		return math.Inf(1)
	}
	tot, px, py := 0.0, fx, fy
	for _, v := range p {
		tot += math.Hypot(v.X-px, v.Y-py)
		px, py = v.X, v.Y
	}
	return tot
}

// TestGeoFieldMatchesAStar: the Dijkstra distance flood that levels the mob packs
// must measure the SAME route length as the shipped A* pathfinder. The octile flood
// is a touch longer than the string-pulled polyline, but within ~1.15x; a bigger gap
// (or an Inf where A* finds a route) would mean geoField is walking a different mesh.
func TestGeoFieldMatchesAStar(t *testing.T) {
	sx, sy := navGrid42.Spawn()
	targets := [][2]float64{
		{217.63, -20.90}, {212.17, 61.72}, {162.89, 62.42},
		{83.97, 108.80}, {286.26, 106.53}, {104, 300}, {-235, 40},
	}
	for _, tg := range targets {
		ap := polyLen(navGrid42.Path(sx, sy, tg[0], tg[1]), sx, sy)
		if math.IsInf(ap, 1) {
			continue // A* found no route; nothing to compare
		}
		// Distance FROM the spawn point: build a field rooted at the real spawn.
		field := navGrid42.geoField(sx, sy)
		ci := int(math.Floor((tg[0] - navGrid42.MinX) / navGrid42.Cell))
		cj := int(math.Floor((tg[1] - navGrid42.MinY) / navGrid42.Cell))
		ci, cj, ok := navGrid42.nearestWalkable(ci, cj, 8)
		if !ok {
			t.Fatalf("target (%.1f,%.1f) has no walkable cell", tg[0], tg[1])
		}
		geo := float64(field[ci*navGrid42.H+cj])
		if geo >= math.MaxFloat32 {
			t.Errorf("geoField unreachable at (%.1f,%.1f) but A* routes there (%.0f)", tg[0], tg[1], ap)
			continue
		}
		if r := geo / ap; r < 0.95 || r > 1.15 {
			t.Errorf("geoField/A* mismatch at (%.1f,%.1f): geo=%.0f a*=%.0f ratio=%.3f", tg[0], tg[1], geo, ap, r)
		}
	}
}

// maxAdjacentLevelJump returns the largest level difference between any two non-boss
// packs that are close (<=22m) AND have a clear walkable line between them -- i.e.
// two mobs a player perceives as neighbours and can step between. A high value is
// exactly the bug the player reported (a level-2 mob next to a level-11 one).
func maxAdjacentLevelJump(spawns []MobSpawn, ng *NavGrid, isBoss func(int) bool) (maxJump, pairs int) {
	type pk struct {
		x, y float64
		l    int
	}
	var ps []pk
	for _, sp := range spawns {
		if isBoss(sp.Mob) || sp.Level == 0 {
			continue
		}
		ps = append(ps, pk{sp.DX, sp.DY, sp.Level})
	}
	for i := 0; i < len(ps); i++ {
		for j := i + 1; j < len(ps); j++ {
			if math.Hypot(ps[i].x-ps[j].x, ps[i].y-ps[j].y) > 22 {
				continue
			}
			if !ng.lineWalkable(ps[i].x, ps[i].y, ps[j].x, ps[j].y) {
				continue
			}
			pairs++
			d := ps[i].l - ps[j].l
			if d < 0 {
				d = -d
			}
			if d > maxJump {
				maxJump = d
			}
		}
	}
	return
}

func jungleBossMob(m int) bool {
	return m == mobBossGrimlok || m == mobBossFairy || m == mobBossTitanid || m == mobBossAnhel
}

// TestJungleLevelRampIsSmooth is the direct regression for the reported bug: with
// path-based leveling no two walkable-adjacent jungle mobs may differ by more than a
// single level, so difficulty rises smoothly with the walk instead of jumping (the
// player saw levels 5,3,2,11,3 on packs a few metres apart). Includes the burnt
// village, which is leveled from the same field.
func TestJungleLevelRampIsSmooth(t *testing.T) {
	mx, pairs := maxAdjacentLevelJump(junglePack, navGrid42, jungleBossMob)
	if pairs < 100 {
		t.Fatalf("expected many adjacent pack pairs to check, got %d", pairs)
	}
	if mx > 1 {
		t.Errorf("jungle level ramp not smooth: adjacent packs differ by up to %d levels (want <=1)", mx)
	}
}

// TestCryptLevelRampIsSmooth: the crypt (linear) also stays smooth under the shared
// path-based leveling. Its authored bands are spaced wider, so allow up to 2 between
// walkable-adjacent packs (still no cliff).
func TestCryptLevelRampIsSmooth(t *testing.T) {
	mx, pairs := maxAdjacentLevelJump(dungeonPack40, navGrid40, func(int) bool { return false })
	if pairs < 100 {
		t.Fatalf("expected many adjacent pack pairs to check, got %d", pairs)
	}
	if mx > 2 {
		t.Errorf("crypt level ramp not smooth: adjacent packs differ by up to %d levels (want <=2)", mx)
	}
}

// TestJungleLevelSpansRange: the entrance reads low and the far (Anhel) side reaches
// the endgame ceiling, so the ramp uses the full 1..huntLevelCeiling (20) range.
func TestJungleLevelSpansRange(t *testing.T) {
	minL, maxL := 99, 0
	for _, sp := range junglePack {
		if jungleBossMob(sp.Mob) || sp.Level == 0 {
			continue
		}
		if sp.Level < minL {
			minL = sp.Level
		}
		if sp.Level > maxL {
			maxL = sp.Level
		}
	}
	if minL > 2 {
		t.Errorf("no low-level packs near the entrance: min level %d (want <=2)", minL)
	}
	if maxL < huntLevelCeiling-2 {
		t.Errorf("far side doesn't reach the endgame ceiling: max level %d (want >=%d)", maxL, huntLevelCeiling-2)
	}
}

// TestJungleLevelRisesWithWalk is the positive statement of the fix: difficulty must
// track the distance a player WALKS from the battle start (jungleSpawn, the overridden
// pocket -- NOT navGrid42's seed marker, which sits on the far side of the map). Packs
// a short walk from the spawn stay trivial; packs a long walk away are high. Measured
// on the real geodesic field, so it fails if leveling ever drifts back to straight-line
// distance (which put deep mobs a straight step -- but a long walk -- from the spawn).
func TestJungleLevelRisesWithWalk(t *testing.T) {
	field := navGrid42.geoField(jungleSpawn.X, jungleSpawn.Y)
	nearMax := 0           // hardest pack within an 80-unit walk (must stay trivial)
	nearSum, nearN := 0, 0 // level average within 120 units
	farSum, farN := 0, 0   // level average past 350 units
	for _, sp := range junglePack {
		if jungleBossMob(sp.Mob) || sp.Level == 0 {
			continue
		}
		ci := int(math.Floor((sp.DX - navGrid42.MinX) / navGrid42.Cell))
		cj := int(math.Floor((sp.DY - navGrid42.MinY) / navGrid42.Cell))
		ci, cj, ok := navGrid42.nearestWalkable(ci, cj, 8)
		if !ok {
			continue
		}
		d := float64(field[ci*navGrid42.H+cj])
		if d <= 80 && sp.Level > nearMax {
			nearMax = sp.Level
		}
		if d <= 120 {
			nearSum += sp.Level
			nearN++
		}
		if d >= 350 {
			farSum += sp.Level
			farN++
		}
	}
	if nearN == 0 || farN == 0 {
		t.Fatalf("need packs both near and far by path: near=%d far=%d", nearN, farN)
	}
	if nearMax > 3 {
		t.Errorf("packs within an 80-unit walk of the spawn should be trivial, got level up to %d", nearMax)
	}
	// Different trails cap at different bands, so far packs aren't uniformly high;
	// but the AVERAGE difficulty must clearly rise with the walk from the spawn.
	nearAvg, farAvg := float64(nearSum)/float64(nearN), float64(farSum)/float64(farN)
	if farAvg < nearAvg+3 {
		t.Errorf("difficulty should rise clearly with the walk: near-avg %.1f, far-avg %.1f", nearAvg, farAvg)
	}
}

// TestTribesmanZoneOnlyTribesmen: inside the user-marked southern "village" every generated
// pack mob is a tribesman (bosses are added outside the pack loop and excluded), and the zone
// actually holds packs. Also asserts the region FILL covers every point the user marked --
// including the interior spots that were previously spawning dinos/gorillas.
func TestTribesmanZoneOnlyTribesmen(t *testing.T) {
	isTribe := func(m int) bool {
		return m == mobTribesman || m == mobTribesmanRange || m == mobTribesmanBig || m == mobTribesmanZombie
	}
	// The whole area between the marks must be filled: every mark is in the region.
	for _, m := range jungleTribesmanZone {
		if !inJungleTribesmanZone(m.X, m.Y) {
			t.Errorf("marked village point (%.1f,%.1f) is not inside the filled region", m.X, m.Y)
		}
	}
	n := 0
	for _, sp := range junglePack {
		if !inJungleTribesmanZone(sp.DX, sp.DY) || jungleBossMob(sp.Mob) || sp.Mob == mobSkeletonBurning {
			continue
		}
		n++
		if !isTribe(sp.Mob) {
			t.Errorf("mob idx %d at (%.1f,%.1f) in the tribesman zone is not a tribesman", sp.Mob, sp.DX, sp.DY)
		}
	}
	if n < 5 {
		t.Fatalf("tribesman zone holds too few packs (%d) -- is the region right?", n)
	}
}

// TestFlowerZoneOnlyFlowers: inside the user-marked western corridor every mob is a rooted
// hunter-flower, they line the WALLS (each is a walkable cell touching a wall -- not standing
// in the open corridor centre), and the corridor is populated.
func TestFlowerZoneOnlyFlowers(t *testing.T) {
	ng := navGrid42
	isWall := func(x, y float64) bool { // walkable cell with an off-mesh 8-neighbour
		i := int(math.Floor((x - ng.MinX) / ng.Cell))
		j := int(math.Floor((y - ng.MinY) / ng.Cell))
		for di := -1; di <= 1; di++ {
			for dj := -1; dj <= 1; dj++ {
				if (di != 0 || dj != 0) && !ng.cellWalkable(i+di, j+dj) {
					return true
				}
			}
		}
		return false
	}
	n := 0
	for _, sp := range junglePack {
		if !inJungleFlowerZone(sp.DX, sp.DY) || jungleBossMob(sp.Mob) || sp.Mob == mobSkeletonBurning {
			continue
		}
		n++
		if sp.Mob != mobFlowerHunter {
			t.Errorf("mob idx %d at (%.1f,%.1f) in the flower zone is not a flower", sp.Mob, sp.DX, sp.DY)
		}
		if !isWall(sp.DX, sp.DY) {
			t.Errorf("flower at (%.1f,%.1f) is not against a wall (should hug the corridor edge)", sp.DX, sp.DY)
		}
	}
	if n < 5 {
		t.Fatalf("flower zone holds too few flowers (%d) -- is the corridor right?", n)
	}
}

// TestJungleLevelMonotonicWithWalk is the whole-map guarantee: because level is a
// radial ramp on walk distance from the spawn, a pack FARTHER to walk is never a
// lower level than a nearer one -- difficulty rises the whole way across the map and
// never dips back (the roller-coaster the per-anchor blend produced: climb, plateau,
// dip, climb). Checks every pack pair against its spawn walk distance.
func TestJungleLevelMonotonicWithWalk(t *testing.T) {
	field := navGrid42.geoField(jungleSpawn.X, jungleSpawn.Y)
	type pk struct {
		d float64
		l int
	}
	var ps []pk
	for _, sp := range junglePack {
		if jungleBossMob(sp.Mob) || sp.Level == 0 {
			continue
		}
		ci := int(math.Floor((sp.DX - navGrid42.MinX) / navGrid42.Cell))
		cj := int(math.Floor((sp.DY - navGrid42.MinY) / navGrid42.Cell))
		ci, cj, ok := navGrid42.nearestWalkable(ci, cj, 8)
		if !ok {
			continue
		}
		d := field[ci*navGrid42.H+cj]
		if d >= math.MaxFloat32 {
			continue
		}
		ps = append(ps, pk{float64(d), sp.Level})
	}
	for i := range ps {
		for j := range ps {
			// A clearly-farther pack must never be a lower level than a nearer one.
			if ps[i].d+1 < ps[j].d && ps[j].l < ps[i].l {
				t.Fatalf("non-monotonic: pack at walk %.0f is L%d but a nearer pack at walk %.0f is L%d",
					ps[j].d, ps[j].l, ps[i].d, ps[i].l)
			}
		}
	}
}
