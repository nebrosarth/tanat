package gamedata

// Vec2 is a point on the scene ground plane (world X, Z) — the same coordinate
// space as MOVE_PLAYER.targetPos and SYNC positions.
type Vec2 struct{ X, Y float64 }

// AreaCSElf is the client Location.CS_ELF id (see ctrlserver areaCSElf); the elf
// central square. Anything else routes to the human cathedral square.
const AreaCSElf int32 = 368

// LobbyNav returns the walkability grid for a central-square lobby AREA: CS_ELF (368)
// = the elf city (cs_elf), everything else = the human cathedral square (cs_human).
// Keyed by area (not race) so a hero visiting the other race's city via the portal
// gets that city's walls. The returned Nav's Spawn() is the scene's spawn point (town
// portal / Reborn). Matches ctrlserver's sceneForArea.
func LobbyNav(area int32) Nav {
	if area == AreaCSElf {
		return navGridCSElf
	}
	return navGridCSHuman
}

// Nav is a scene's walkability oracle: which world points are walkable, where the
// avatar spawns, and how far a straight move may travel before leaving the
// walkable area. Implemented by *NavGrid (rasterised surface).
type Nav interface {
	Walkable(x, y float64) bool
	Spawn() (float64, float64)
	Clip(fx, fy, tx, ty float64) (float64, float64)
	// Path returns waypoints routing around walls from (fx,fy) to (tx,ty); nil
	// if no route exists. A clear straight shot returns a single waypoint.
	Path(fx, fy, tx, ty float64) []Vec2
}
