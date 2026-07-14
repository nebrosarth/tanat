package gamedata

import (
	"math"
	"testing"
)

func TestNavGrid40SpawnWalkable(t *testing.T) {
	sx, sy := navGrid40.Spawn()
	if !navGrid40.Walkable(sx, sy) {
		t.Fatalf("spawn (%.1f,%.1f) is not walkable", sx, sy)
	}
	// Battle-start Reborn zone extracted from the scene.
	if math.Abs(sx-493.0) > 1 || math.Abs(sy-64.0) > 1 {
		t.Errorf("spawn drifted from the start Reborn zone: got (%.1f,%.1f)", sx, sy)
	}
}

func TestNavGrid40MobsWalkable(t *testing.T) {
	bx, by := navGrid40.Spawn()
	// mobAggroRange in the battle server is 9.0; the dungeon pack must start
	// outside it so nothing aggros the moment the player spawns. Keep a margin.
	const startSafeRadius = 9.5
	for i, sp := range dungeonPack40 {
		mx, my := bx+sp.DX, by+sp.DY
		if sp.Abs {
			mx, my = sp.DX, sp.DY
		}
		if !navGrid40.Walkable(mx, my) {
			t.Errorf("mob %d at (%.1f,%.1f) is not walkable", i, mx, my)
		}
		if sp.Abs {
			// Bosses sit in far arenas across the map: no straight line of sight,
			// but they must be reachable by pathfinding so the player can get to them.
			if p := navGrid40.Path(bx, by, mx, my); len(p) == 0 {
				t.Errorf("boss mob %d at (%.1f,%.1f) is not reachable from spawn", i, mx, my)
			}
			continue
		}
		if d := math.Hypot(sp.DX, sp.DY); d <= startSafeRadius {
			t.Errorf("mob %d is %.1f from spawn -- inside the start aggro range, it would aggro immediately", i, d)
		}
		// Reachable from spawn (so the player can walk over and engage it).
		cx, cy := navGrid40.Clip(bx, by, mx, my)
		if math.Hypot(cx-mx, cy-my) > 1.0 {
			t.Errorf("mob %d at (%.1f,%.1f) is not reachable from spawn (a wall blocks the path)", i, mx, my)
		}
	}
}

func TestNavGrid40OutsideBlocked(t *testing.T) {
	if navGrid40.Walkable(10000, 10000) {
		t.Fatal("a point far outside the map is walkable")
	}
	// Far below the connected region (the void) must be blocked.
	if navGrid40.Walkable(378.5, -400) {
		t.Fatal("a point in the void is walkable")
	}
}

func TestNavGrid40ClipStopsAtEdge(t *testing.T) {
	sx, sy := navGrid40.Spawn()
	// Walk far south (toward the map edge/void); clip must stay walkable and stop
	// short of the target.
	tx, ty := sx, sy-1000
	cx, cy := navGrid40.Clip(sx, sy, tx, ty)
	if !navGrid40.Walkable(cx, cy) {
		t.Fatalf("clipped point (%.1f,%.1f) is not walkable", cx, cy)
	}
	if math.Hypot(cx-tx, cy-ty) < 1 {
		t.Fatalf("clip did not stop before the void target (landed %.1f,%.1f)", cx, cy)
	}
}

// map_4_2 («Заповедные джунгли») nav grid: walkability from the authored polygon
// (shuffled into the map_4_1 bundle). All 6 of the scene's Reborn_point markers must
// be on walkable floor and reachable from spawn (they share one connected component).
func TestNavGrid42RebornsWalkableAndReachable(t *testing.T) {
	sx, sy := navGrid42.Spawn()
	if !navGrid42.Walkable(sx, sy) {
		t.Fatalf("map_4_2 spawn (%.1f,%.1f) is not walkable", sx, sy)
	}
	reborns := [][2]float64{
		{-335.7, 102.6}, {5.0, -80.8}, {-43.1, 365.1},
		{321.6, -165.6}, {332.9, 107.5}, {35.0, 30.0},
	}
	for i, r := range reborns {
		if !navGrid42.Walkable(r[0], r[1]) {
			t.Errorf("map_4_2 Reborn %d (%.1f,%.1f) is not walkable", i, r[0], r[1])
		}
		if p := navGrid42.Path(sx, sy, r[0], r[1]); len(p) == 0 {
			t.Errorf("map_4_2 Reborn %d (%.1f,%.1f) is not reachable from spawn", i, r[0], r[1])
		}
	}
}

func TestNavGrid42OutsideBlocked(t *testing.T) {
	if navGrid42.Walkable(10000, 10000) {
		t.Fatal("a point far outside map_4_2 is walkable")
	}
}
