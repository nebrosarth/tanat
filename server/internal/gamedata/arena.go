package gamedata

// «Арена» (MapType.DM = 0) is the player-versus-player deathmatch on map_0_0 -- a
// burned-out village with no creeps, no structures and no lanes: just avatars fighting
// avatars. Two sides share the same open ground and respawn from a pool of points the
// scene ships as `Reborn_point` markers. A solo launch is the N==1 case (an empty arena
// to walk), exactly as «Штурм» is playable alone.
//
// MapTypeDM mirrors TanatKernel.MapType.DM (the enum's first member, so 0). The mode's
// display name is the locale key DM_Text = «Арена».
const MapTypeDM int32 = 0

// ArenaTeamA / ArenaTeamB are the two sides. They reuse the «Штурм» player-team value
// for side A so all the shared combat that already treats dotaPlayerTeam as "the local
// player" keeps working, and a distinct value for side B so hostile() tells them apart.
// A free-for-all would need a team per player; the client renders a binary own/enemy
// split, so two teams is what it can actually show.
const (
	ArenaTeamA int32 = 1
	ArenaTeamB int32 = 2
)

// ArenaMap is one deathmatch arena. It is deliberately much smaller than DotaMap: no
// structures, no creep roster, no lanes -- the only authored geometry a match needs is
// where players (re)spawn and where the walkable ground ends.
type ArenaMap struct {
	ID         int32
	Name       string // display name (locale key: DM_Text)
	Scene      string // map bundle (map_0_0)
	LevelMin   int32
	LevelMax   int32
	Desc       string
	WinDesc    string
	MinPlayers int32
	MaxPlayers int32

	// FragLimit ends the match when a team reaches this many kills (0 = endless).
	FragLimit int32

	// Spawns is the pool of respawn points, taken from the scene's `Reborn_point`
	// markers. A (re)spawning player is placed at the point farthest from its living
	// enemies (chosen at spawn time, not fixed per team), so neither side owns a base
	// and nobody materialises on top of an enemy.
	Spawns []Vec2

	// Nav is the walkability oracle: the rasterised PassibilityData street network
	// (navGrid00), so avatars route around the burned village's building blocks.
	Nav Nav
}

// map00 is the map_0_0 «Арена» layout. Spawns are the scene's five Reborn_point markers,
// each SNAPPED to the nearest walkable cell of navGrid00: three of the authored markers
// sit in the thin margin between the passability boundary and the surrounding jungle
// ring (5-24u off-field), so placing a player at the raw coordinate would materialise it
// in a wall. The comment on each line is the original marker (world X, Z).
var map00 = ArenaMap{
	ID: 1, // client map id for the DM launch; distinct from «Штурм» (101)
	// Locale key the client resolves for the map's display name: Map_0_0_Name =
	// «Осаждённое поселение». (The mode tab itself is named from DM_Text = «Арена».)
	Name:       "Map_0_0_Name",
	Scene:      "map_0_0",
	LevelMin:   1,
	LevelMax:   20,
	// Desc/WinDesc are locale KEYS too: SelectGameMenu resolves them via GetLocaleText
	// (:92-93), so a literal string renders "EMPTY!" in the card's history/win-condition
	// detail panel. Map_0_0_Desc/WinDesc are the baked keys for scene map_0_0.
	Desc:       "Map_0_0_Desc",
	WinDesc:    "Map_0_0_WinDesc",
	MinPlayers: 1,
	MaxPlayers: 10,
	FragLimit:  20,
	Spawns: []Vec2{
		{X: -77.68, Y: 47.34},  // Reborn (-77.85, 46.33), snapped 1.0u
		{X: 19.32, Y: 66.34},   // Reborn (26.75, 78.89), snapped 14.6u
		{X: 15.32, Y: -4.66},   // Reborn (18.46, -5.04), snapped 3.2u
		{X: 41.32, Y: -57.66},  // Reborn (61.49, -70.59), snapped 24.0u
		{X: -124.68, Y: 15.34}, // Reborn (-128.36, 19.73), snapped 5.7u
	},
	Nav: navGrid00,
}

var arenaMaps = []ArenaMap{map00}

// ArenaMaps returns every deathmatch arena.
func ArenaMaps() []ArenaMap { return arenaMaps }

// ArenaMapByID returns the arena with this client map id.
func ArenaMapByID(id int32) (ArenaMap, bool) {
	for _, m := range arenaMaps {
		if m.ID == id {
			return m, true
		}
	}
	return ArenaMap{}, false
}
