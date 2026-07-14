package gamedata

import "testing"

// TestLobbyNavByArea: LobbyNav picks cs_human for the human city (367) and any other
// area, cs_elf for the elf city (368 = Location.CS_ELF), and each grid's spawn point
// is itself walkable (so a freshly spawned hero isn't stuck off-grid, where Clip would
// freeze all movement). Keyed by AREA so cross-city portal travel resolves correctly.
func TestLobbyNavByArea(t *testing.T) {
	if LobbyNav(367) != navGridCSHuman {
		t.Error("human city area (367) should map to the cs_human (cathedral) nav")
	}
	if LobbyNav(0) != navGridCSHuman {
		t.Error("unknown area should default to the human lobby nav")
	}
	if LobbyNav(AreaCSElf) != navGridCSElf {
		t.Error("elf city area (368) should map to the cs_elf nav")
	}
	for _, tc := range []struct {
		name string
		g    *NavGrid
	}{{"cs_human", navGridCSHuman}, {"cs_elf", navGridCSElf}} {
		sx, sy := tc.g.Spawn()
		if !tc.g.Walkable(sx, sy) {
			t.Errorf("%s spawn (%.1f,%.1f) is not walkable -- hero would spawn stuck", tc.name, sx, sy)
		}
	}
}

// TestCSHumanWalkable pins the cathedral-square walkability: base = inside poly0 minus
// the two hedge arches (poly3/poly4) from the map_4_0 PassibilityData, with the bogus
// central "cross" filled back in and the hedge re-blocked from the minimap. The spawn and
// a sample of user-verified walkable plaza points are open, the central hedge maze (the
// "C") is blocked, and a point far outside the square is blocked.
func TestCSHumanWalkable(t *testing.T) {
	g := navGridCSHuman
	sx, sy := g.Spawn()
	if !g.Walkable(sx, sy) {
		t.Errorf("cs_human spawn (%.2f,%.2f) should be walkable", sx, sy)
	}
	// A sample of the dense walkable points the user clicked across the square,
	// including the filled-in central cross.
	for _, p := range []struct{ x, y float64 }{
		{-3.06, 33.29}, {21.56, 53.55}, {-58.83, 25.84}, {-13.89, -13.26}, {28.13, 29.01},
		{-9.14, 64.63}, {7.19, 64.81},
	} {
		if !g.Walkable(p.x, p.y) {
			t.Errorf("cs_human walkable plaza point (%.1f,%.1f) should be walkable", p.x, p.y)
		}
	}
	// Zones expanded from the latest ground-truth click session: the south-center band
	// below the hedge, the right cluster (over the House_03 floor decal), the guild's
	// north strip (top edge of the oversized GuildBuild_Selector), and the NW tree line
	// (inside the Combined-tree AABB, above the guild). All were over-blocked before.
	for _, p := range []struct{ x, y float64 }{
		{5.0, -8.0}, {24.0, 11.0}, {-50.0, -4.8}, {-53.8, 56.3},
	} {
		if !g.Walkable(p.x, p.y) {
			t.Errorf("cs_human expanded plaza point (%.1f,%.1f) should be walkable", p.x, p.y)
		}
	}
	// A sample of the user-traced hedge-maze points (bottom channel, right arch, top
	// channel) -- these must be blocked.
	for _, p := range []struct{ x, y float64 }{
		{-15.46, 2.42}, {13.74, 28.36}, {8.72, 43.03}, {-1.81, 49.26}, {-50.20, 36.03},
	} {
		if g.Walkable(p.x, p.y) {
			t.Errorf("cs_human hedge point (%.1f,%.1f) should be blocked", p.x, p.y)
		}
	}
	// Scene-object footprints (from cs_human colliders + meshes) must be blocked:
	// guild building, human tower, a shop, a house.
	for _, p := range []struct{ x, y float64 }{
		{-55, -20}, {-40, -45}, {-39, 40}, {-3, 74},
	} {
		if g.Walkable(p.x, p.y) {
			t.Errorf("cs_human object interior (%.1f,%.1f) should be blocked", p.x, p.y)
		}
	}
	if g.Walkable(1000, 1000) {
		t.Error("a point far outside the cathedral square should be blocked")
	}
}

// TestCSElfWalkable pins the approximate cs_elf grid: the portal spawn is open and
// the guild-building footprint is blocked.
func TestCSElfWalkable(t *testing.T) {
	g := navGridCSElf
	if !g.Walkable(-46.4, -36.0) {
		t.Error("cs_elf town-portal spawn should be walkable")
	}
	if g.Walkable(13, -45) { // CS_Elf_GuildBuilding01 footprint
		t.Error("cs_elf guild building footprint should be blocked")
	}
}
