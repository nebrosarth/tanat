package gamedata

// CASTLES (clan-siege targets), "Битва за замок". This is the static registry the
// castle|list / castle|info screens read.
//
// The two castles and their locale keys are BAKED into the client's own asset data
// (Tanat_Data/resources.assets, recovered via byte-offset search): the client has
// exactly two named castles, "Castle_1_Name" ("Рейан'Тар") and "Castle_2_Name"
// ("Норкин лис") -- there is no third. CastleMenu.cs resolves a castle's displayed
// name via GuiSystem.GetLocaleText(mCastleInfo.mName), so the wire "name" field must
// be the LOCALE KEY, not raw text -- the client looks it up itself.
//
// The siege plays out on its OWN dedicated scene, `data/scenes/map_6_0.unity3d` (not
// listed in resources.xml -- scene bundles live as loose files under data/scenes/,
// invisible to that manifest entirely; found by directory listing, confirmed distinct
// from every other scene by MD5 and by its own baked minimap texture "Map_6_0" in
// resources.assets). See gamedata/dota.go's map60 for the extracted layout (5
// defending cannons; no altar -- "every gun destroyed" is the win condition) and
// battleserver/castle.go for the battle-side wiring.
//
// Time fields on the wire are RELATIVE: castle|list emits start_time = seconds until
// the next battle window. StartInSec is that countdown; CycleSec is how far it resets
// after the window fires (see the scheduler in ctrlserver/castle.go).

// CastleReward is the loot bundle advertised for winning a castle, credited to every
// enrolled fighter on the winning side when the siege ends.
type CastleReward struct {
	Money     int32
	Diamonds  int32
	Exp       int32
	Item      int32
	ItemCount int32
}

// Castle is one siege target.
type Castle struct {
	ID          int32
	NameKey     string // locale key, e.g. "Castle_1_Name" -- resolved client-side
	MapID       int32  // gamedata.DotaMapByID id the siege battle plays out on
	Scene       string // battle scene bundle (mirrors the DotaMap's own Scene)
	LevelMin    int32
	LevelMax    int32
	OwnerID     int32  // owning clan id (0 = unowned/neutral)
	OwnerName   string // owning clan name ("" when neutral)
	FightersMin int32  // minimum fighters a side must field
	Reward      CastleReward
	StartInSec  int32 // seconds until the next battle window
	CycleSec    int32 // window recurrence once a battle has fired (win or no-show)
}

// castles is the seeded registry: the two castles the client's own locale data
// names. Both fight over the same map_6_0 siege scene (map60) -- one castle per
// physical scene wouldn't make sense given only one was ever built; they're two
// separate ownership/schedule/reward slots contesting the same battlefield.
var castles = []Castle{
	{
		ID: 1, NameKey: "Castle_1_Name", MapID: 102, Scene: "map_6_0",
		LevelMin: 1, LevelMax: 30, OwnerID: 0, OwnerName: "", FightersMin: 1,
		Reward:     CastleReward{Money: 50000, Diamonds: 10, Exp: 5000, Item: 0, ItemCount: 0},
		StartInSec: 3600, CycleSec: 3600,
	},
	{
		ID: 2, NameKey: "Castle_2_Name", MapID: 102, Scene: "map_6_0",
		LevelMin: 1, LevelMax: 30, OwnerID: 0, OwnerName: "", FightersMin: 1,
		Reward:     CastleReward{Money: 100000, Diamonds: 25, Exp: 10000, Item: 0, ItemCount: 0},
		StartInSec: 7200, CycleSec: 7200,
	},
}

// Castles returns the full castle registry (do not mutate the returned slice).
func Castles() []Castle { return castles }

// CastleByID returns the castle with the given id.
func CastleByID(id int32) (Castle, bool) {
	for _, c := range castles {
		if c.ID == id {
			return c, true
		}
	}
	return Castle{}, false
}

// MapTypeCwSiege mirrors the decompiled client's MapType.CW_SIEGE (=8): the
// clan-siege "Битва за замок" battle mode (BattleStats.cs groups it with DOTA/
// CW_DOTA under mTeamRequired=true). Castle uses its own castle|* screens instead of
// the arena|get_maps* menu, so this constant isn't threaded through that registry --
// kept for parity with MapTypeDota/MapTypeHunt and for the in-battle MapType report.
const MapTypeCwSiege int32 = 8
