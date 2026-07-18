package gamedata

import (
	"math"
	"testing"
)

// TestArenaMapExists is the smoke test: the deathmatch arena must be registered and
// distinguishable from «Штурм».
func TestArenaMapExists(t *testing.T) {
	m, ok := ArenaMapByID(1)
	if !ok {
		t.Fatal("arena map id 1 missing")
	}
	if m.Scene != "map_0_0" {
		t.Errorf("arena scene = %q, want map_0_0", m.Scene)
	}
	if MapTypeDM != 0 {
		t.Errorf("MapTypeDM = %d, want 0 (first MapType enum member)", MapTypeDM)
	}
	if ArenaTeamA == ArenaTeamB {
		t.Fatal("the two arena teams share a value: hostile() could never tell them apart")
	}
	// Side A reuses the «Штурм» player-team value so the shared combat that keys on it
	// keeps working.
	if ArenaTeamA != 1 {
		t.Errorf("ArenaTeamA = %d, want 1 (dotaPlayerTeam) so shared combat treats it as the local side", ArenaTeamA)
	}
}

// TestArenaSpawnsAreWalkable: every respawn point must be on open ground, or a player
// materialises stuck in geometry. Also proves the spawns come from the scene's markers
// (5 of them) and sit inside the nav bounds.
func TestArenaSpawnsAreWalkable(t *testing.T) {
	m, _ := ArenaMapByID(1)
	if len(m.Spawns) != 5 {
		t.Fatalf("arena has %d spawns, want 5 (the scene's Reborn_point markers)", len(m.Spawns))
	}
	for i, sp := range m.Spawns {
		if !m.Nav.Walkable(sp.X, sp.Y) {
			t.Errorf("spawn %d (%.1f,%.1f) is outside the walkable field", i, sp.X, sp.Y)
		}
	}
}

// TestArenaSpawnsAreSpreadOut: a deathmatch respawn picker chooses the point farthest
// from enemies, which only works if the points are actually far apart. If two markers
// were nearly coincident the picker would have no room to separate fighters.
func TestArenaSpawnsAreSpreadOut(t *testing.T) {
	m, _ := ArenaMapByID(1)
	const minGap = 20.0
	for i := 0; i < len(m.Spawns); i++ {
		for j := i + 1; j < len(m.Spawns); j++ {
			d := math.Hypot(m.Spawns[i].X-m.Spawns[j].X, m.Spawns[i].Y-m.Spawns[j].Y)
			if d < minGap {
				t.Errorf("spawns %d and %d are only %.1fu apart (want >= %.0f): a respawn picker "+
					"can't keep fighters apart", i, j, d, minGap)
			}
		}
	}
}

// TestArenaNavIsRasterisedPolygon: the arena nav must be the scene's real collision
// polygon (a street network riddled with building blocks), NOT a bounding rectangle. A
// rectangle would report ~100% of its bbox walkable; the real town is ~26%. Sampling the
// grid's own bbox and finding a large blocked fraction is what tells them apart.
func TestArenaNavIsRasterisedPolygon(t *testing.T) {
	m, _ := ArenaMapByID(1)
	g, ok := m.Nav.(*NavGrid)
	if !ok {
		t.Fatalf("arena nav is %T, want *NavGrid (the rasterised PassibilityData polygon)", m.Nav)
	}
	walk, total := 0, 0
	for i := 0; i < g.W; i++ {
		for j := 0; j < g.H; j++ {
			x := g.MinX + (float64(i)+0.5)*g.Cell
			y := g.MinY + (float64(j)+0.5)*g.Cell
			total++
			if g.Walkable(x, y) {
				walk++
			}
		}
	}
	frac := float64(walk) / float64(total)
	if frac > 0.60 {
		t.Errorf("walkable fraction %.0f%% is too high: nav looks like a rectangle, not a town with building blocks", frac*100)
	}
	if frac < 0.10 {
		t.Errorf("walkable fraction %.0f%% is implausibly low: rasterisation likely dropped the streets", frac*100)
	}
}

// TestArenaNavBlocksBuildingsAndVoid pins the frame and the polygon choice. The three raw
// Reborn markers that had to be snapped land, at their ORIGINAL coordinates, either inside
// a building block (interior holes) or in the jungle margin off the outer boundary -- so
// they must read as not walkable. If the frame were wrong (rotated/mirrored) or the stale
// "map_0_0.xml" placeholder had been loaded, these specific points would not line up with
// real obstacles.
func TestArenaNavBlocksBuildingsAndVoid(t *testing.T) {
	m, _ := ArenaMapByID(1)
	blockedInterior := []Vec2{{X: 18.46, Y: -5.04}, {X: -77.85, Y: 46.33}} // raw Reborn in building holes
	blockedVoid := []Vec2{{X: 26.75, Y: 78.89}, {X: 61.49, Y: -70.59}, {X: 500, Y: 500}}
	for _, p := range append(blockedInterior, blockedVoid...) {
		if m.Nav.Walkable(p.X, p.Y) {
			t.Errorf("(%.1f,%.1f) should be blocked (building block or off-field), but nav says walkable", p.X, p.Y)
		}
	}
}

// TestArenaNavClipStopsAtWall: walking from a spawn toward a far off-field target must
// stop on walkable ground short of the target -- the per-tick movement clamp relies on
// Clip never handing back an off-field point.
func TestArenaNavClipStopsAtWall(t *testing.T) {
	m, _ := ArenaMapByID(1)
	sp := m.Spawns[4] // (-124.68, 15.34), on the far-west street; +x heads into town/walls
	cx, cy := m.Nav.Clip(sp.X, sp.Y, sp.X+300, sp.Y)
	if !m.Nav.Walkable(cx, cy) {
		t.Errorf("Clip returned off-field point (%.1f,%.1f)", cx, cy)
	}
	if cx >= sp.X+300 {
		t.Errorf("Clip reached the far off-field target at x=%.1f without stopping at a wall", cx)
	}
}

// TestArenaNavPath: a route between two walkable spawns exists (non-nil, ends at the
// goal), and a route to an off-grid goal is nil.
func TestArenaNavPath(t *testing.T) {
	m, _ := ArenaMapByID(1)
	a, b := m.Spawns[2], m.Spawns[0] // both walkable streets
	if p := m.Nav.Path(a.X, a.Y, b.X, b.Y); p == nil {
		t.Errorf("Path between two walkable spawns is nil, want a route")
	} else if last := p[len(p)-1]; last != b {
		t.Errorf("Path ends at %v, want the goal %v", last, b)
	}
	if p := m.Nav.Path(a.X, a.Y, 999, 999); p != nil {
		t.Errorf("Path to an off-grid goal = %v, want nil", p)
	}
}
