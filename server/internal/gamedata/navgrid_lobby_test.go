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

// TestCSElfWalkable pins the reconstructed cs_elf island: the elf city (area 368) is a
// mesh-terrain island with no authored PassibilityData polygon, so its grid is built from
// the client minimap (map_cs_elf_267.png) -- bright-green land + wooden bridges walkable,
// dark-blue water blocked -- with the minimap->world transform constellation-fit to the
// scene's own anchors (town portal, battle portal, bank, shop, guild, NPCs, bridges). The
// spawn is the town-portal Reborn; the whole island is one flood-connected component.
func TestCSElfWalkable(t *testing.T) {
	g := navGridCSElf
	sx, sy := g.Spawn()
	if !g.Walkable(sx, sy) {
		t.Errorf("cs_elf spawn (%.1f,%.1f) should be walkable", sx, sy)
	}
	// Verified walkable landmarks across the island (world X,Z from cs_elf.unity3d):
	// battle portal, bank, shop, plaza centre, the eastern sub-island, the southern lobe.
	for _, p := range []struct{ x, y float64 }{
		{-44.3, 9.8}, {-42.8, -8.1}, {-35.1, 22.4}, {0.0, 0.0}, {50.0, 10.0}, {20.0, 55.0},
	} {
		if !g.Walkable(p.x, p.y) {
			t.Errorf("cs_elf landmark (%.1f,%.1f) should be walkable", p.x, p.y)
		}
	}
	// The four wooden bridges (their collider centres) must be walkable -- they are the
	// only crossings between the island's water-separated lobes.
	for _, p := range []struct{ x, y float64 }{
		{-11.0, -40.4}, {56.8, -15.4}, {-5.9, 51.5}, {4.7, 35.3},
	} {
		if !g.Walkable(p.x, p.y) {
			t.Errorf("cs_elf bridge (%.1f,%.1f) should be walkable", p.x, p.y)
		}
	}
	// Water (all four sides of the island) and the guild-building footprint must be blocked.
	for _, p := range []struct{ x, y float64 }{
		{-90.0, -10.0}, {110.0, 0.0}, {0.0, 95.0}, {-40.0, -90.0}, // open water
		{18.1, -58.0},    // CS_Elf_GuildBuilding01 interior
		{1000.0, 1000.0}, // far outside
	} {
		if g.Walkable(p.x, p.y) {
			t.Errorf("cs_elf point (%.1f,%.1f) should be blocked", p.x, p.y)
		}
	}
	// Cross-island reachability: from the town-portal spawn the player can path across the
	// isthmus + bridges to the far battle-portal plaza and the southern lobe (one component).
	for _, p := range []struct{ x, y float64 }{
		{-44.3, 9.8}, {20.0, 55.0},
	} {
		if path := g.Path(sx, sy, p.x, p.y); len(path) == 0 {
			t.Errorf("cs_elf point (%.1f,%.1f) is not reachable from spawn", p.x, p.y)
		}
	}
}
