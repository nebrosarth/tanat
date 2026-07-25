package gamedata

// CASTLES (clan-siege targets). This is the static registry the castle|list / castle|info
// screens read. It is intentionally a small seeded set: the LIVE siege battle (scheduling,
// bracket, real-time combat, ownership transfer, reward payout) is deferred, so these serve
// the registration/info screens with self-consistent data and no crash.
//
// Time fields on the wire are RELATIVE: castle|list emits start_time = seconds until the
// next battle window. StartInSec here is that countdown (a fixed placeholder schedule).

// CastleReward is the loot bundle advertised for winning a castle (displayed only for now;
// payout is deferred).
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
	Name        string // displayed raw (not a locale key)
	Scene       string // battle map/scene name (used once the live siege lands)
	LevelMin    int32
	LevelMax    int32
	OwnerID     int32  // owning clan id (0 = unowned/neutral)
	OwnerName   string // owning clan name ("" when neutral)
	FightersMin int32  // minimum fighters a side must field
	Reward      CastleReward
	StartInSec  int32 // seconds until the next battle window (placeholder schedule)
}

// castles is the seeded registry. Names are readable Russian labels; scenes reuse existing
// battle maps so a future siege has somewhere to land.
var castles = []Castle{
	{
		ID: 1, Name: "Замок Севера", Scene: "map_1_0",
		LevelMin: 1, LevelMax: 30, OwnerID: 0, OwnerName: "", FightersMin: 3,
		Reward:     CastleReward{Money: 50000, Diamonds: 0, Exp: 5000, Item: 0, ItemCount: 0},
		StartInSec: 3600,
	},
	{
		ID: 2, Name: "Крепость Запада", Scene: "map_1_0",
		LevelMin: 10, LevelMax: 30, OwnerID: 0, OwnerName: "", FightersMin: 5,
		Reward:     CastleReward{Money: 100000, Diamonds: 10, Exp: 10000, Item: 0, ItemCount: 0},
		StartInSec: 7200,
	},
	{
		ID: 3, Name: "Цитадель Востока", Scene: "map_1_0",
		LevelMin: 15, LevelMax: 30, OwnerID: 0, OwnerName: "", FightersMin: 5,
		Reward:     CastleReward{Money: 150000, Diamonds: 25, Exp: 15000, Item: 0, ItemCount: 0},
		StartInSec: 10800,
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
