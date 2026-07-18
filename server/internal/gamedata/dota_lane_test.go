package gamedata

import (
	"math"
	"testing"
)

// The lanes are hand-baked geometry: nothing in map_1_0 authors a creep path, so these
// polylines are our reconstruction of where the map's own gun placement says the lanes
// must run. That makes them exactly the kind of data that rots silently -- a lane can
// be plausible on a diagram and still walk creeps through a cliff. navGrid10 is the
// scene's real collision, so these tests hold the lanes against it.

func lanes(t *testing.T) [][]Vec2 {
	t.Helper()
	m, ok := DotaMapByID(101)
	if !ok {
		t.Fatal("map 101 (map_1_0) missing")
	}
	return m.Lanes
}

// TestLanesAreWalkable is the guard for the bug the shipped centre lane had: every one
// of its waypoints sat in free space, but the straight run between two of them cut 3.5u
// into a river-bank rock. Creeps interpolate BETWEEN waypoints, so checking the corners
// proves nothing -- this marches the whole polyline.
func TestLanesAreWalkable(t *testing.T) {
	for li, lane := range lanes(t) {
		for i, wp := range lane {
			if !navGrid10.Walkable(wp.X, wp.Y) {
				t.Errorf("lane %d waypoint %d (%.1f,%.1f) is inside geometry", li, i, wp.X, wp.Y)
			}
		}
		for i := 0; i+1 < len(lane); i++ {
			a, b := lane[i], lane[i+1]
			d := math.Hypot(b.X-a.X, b.Y-a.Y)
			steps := int(d / 0.5)
			for k := 1; k < steps; k++ {
				f := float64(k) / float64(steps)
				x, y := a.X+(b.X-a.X)*f, a.Y+(b.Y-a.Y)*f
				if !navGrid10.Walkable(x, y) {
					t.Errorf("lane %d segment %d (%.1f,%.1f)->(%.1f,%.1f) crosses geometry at (%.1f,%.1f)",
						li, i, a.X, a.Y, b.X, b.Y, x, y)
					break
				}
			}
		}
	}
}

// TestLanesRunAltarToAltar: a lane is a siege route. If it stops short of the enemy
// altar the creeps that survive it mill about in the open instead of pushing, and the
// mode has no way to end. Both ends must sit on an altar.
func TestLanesRunAltarToAltar(t *testing.T) {
	m, _ := DotaMapByID(101)
	var human, elf DotaStructure
	for _, sc := range m.Structures {
		if sc.Role != DotaAltar {
			continue
		}
		if sc.Side == DotaSideHuman {
			human = sc
		} else {
			elf = sc
		}
	}
	const reach = 3.0 // dotaWaypointHit: closer than this and the creep has arrived
	for li, lane := range m.Lanes {
		if len(lane) < 2 {
			t.Errorf("lane %d has %d waypoints", li, len(lane))
			continue
		}
		first, last := lane[0], lane[len(lane)-1]
		if d := math.Hypot(first.X-human.X, first.Y-human.Z); d > reach {
			t.Errorf("lane %d starts %.1fu from the human altar, not on it", li, d)
		}
		if d := math.Hypot(last.X-elf.X, last.Y-elf.Z); d > reach {
			t.Errorf("lane %d ends %.1fu from the elf altar, not on it", li, d)
		}
	}
}

// TestLanesThreadTheirGuns is what makes the lanes evidence rather than opinion: the map
// itself placed 18 lane guns, 3 per side per lane, and each one sits within a hair of
// exactly one lane. If a lane is edited into the wrong place, it stops threading its
// guns long before it stops being walkable.
func TestLanesThreadTheirGuns(t *testing.T) {
	m, _ := DotaMapByID(101)
	// The 4 base guns hug the altars rather than a lane, so they are not lane evidence.
	baseGuns := map[int32]bool{5: true, 6: true, 20: true, 21: true}
	perLane := map[int]int{}
	for _, sc := range m.Structures {
		if sc.Role != DotaGun || baseGuns[sc.ID] {
			continue
		}
		best, bestD, nextD := -1, math.Inf(1), math.Inf(1)
		for li, lane := range m.Lanes {
			d := distToPolyline(lane, sc.X, sc.Z)
			if d < bestD {
				best, nextD, bestD = li, bestD, d
			} else if d < nextD {
				nextD = d
			}
		}
		if bestD > 1.0 {
			t.Errorf("gun %d is %.2fu from its nearest lane (%d): the lanes no longer thread the map's own guns",
				sc.ID, bestD, best)
		}
		if nextD-bestD < 10 {
			t.Errorf("gun %d is ambiguous: %.2fu from lane %d but %.2fu from the next -- lanes overlap",
				sc.ID, bestD, best, nextD)
		}
		perLane[best]++
	}
	for li := range m.Lanes {
		if perLane[li] != 6 {
			t.Errorf("lane %d threads %d guns, want 6 (3 per side)", li, perLane[li])
		}
	}
}

// TestLaneForMeasuresToSegmentsNotCorners pins LaneFor's contract rather than today's
// map. Measuring to the waypoints instead of to the legs between them happens to give
// the same six answers on map_1_0, so the real map can never tell the two apart -- but
// they are not the same rule, and the difference bites the moment a barracks stands
// beside the middle of a long leg, far from either of its ends. Synthetic geometry is
// the only thing that can hold this.
func TestLaneForMeasuresToSegmentsNotCorners(t *testing.T) {
	m := DotaMap{Lanes: [][]Vec2{
		// Lane 0: one long leg. Its ENDS are 100u from the barracks below; its LINE is 5u.
		{{X: 0, Y: 0}, {X: 200, Y: 0}},
		// Lane 1: a decoy whose nearest CORNER (100,60) is closer than lane 0's corners.
		{{X: 100, Y: 60}, {X: 100, Y: 260}},
	}}
	bar := DotaStructure{ID: 99, Role: DotaCreepTower, X: 100, Z: 5}
	if li := m.LaneFor(bar); li != 0 {
		t.Errorf("LaneFor = %d, want 0: the barracks stands 5u from lane 0's leg and 55u from lane 1, "+
			"so only corner-measuring (100u vs 55u) could miss it", li)
	}
}

// distToPolyline: distToSeg itself is production code now (DotaMap.LaneFor uses it).
func distToPolyline(lane []Vec2, x, y float64) float64 {
	best := math.Inf(1)
	for i := 0; i+1 < len(lane); i++ {
		if d := distToSeg(lane[i], lane[i+1], x, y); d < best {
			best = d
		}
	}
	return best
}

// TestDotaStructuresAndSpawnsAreWalkable: the frame check, kept live. If navGrid10 is
// ever regenerated in a mismatched frame (map_4_0's polygon shipped in the wrong one,
// which is why Hunt needed a rescue), the symptom is buildings buried in rock -- this
// catches that before a player walks into it.
func TestDotaStructuresAndSpawnsAreWalkable(t *testing.T) {
	m, _ := DotaMapByID(101)
	for _, sc := range m.Structures {
		if !navGrid10.Walkable(sc.X, sc.Z) {
			t.Errorf("structure %d (%s) at (%.1f,%.1f) is not on open ground: wrong frame?", sc.ID, sc.Prefab, sc.X, sc.Z)
		}
	}
	for _, sp := range []struct {
		name string
		v    Vec2
	}{{"human", m.SpawnHuman}, {"elf", m.SpawnElf}} {
		if !navGrid10.Walkable(sp.v.X, sp.v.Y) {
			t.Errorf("%s spawn (%.1f,%.1f) is not walkable", sp.name, sp.v.X, sp.v.Y)
		}
	}
	// Both bases must be mutually reachable, or one side can never be pushed.
	if p := navGrid10.Path(m.SpawnHuman.X, m.SpawnHuman.Y, m.SpawnElf.X, m.SpawnElf.Y); len(p) == 0 {
		t.Error("no route from the human spawn to the elf spawn: the arena is not connected")
	}
}
