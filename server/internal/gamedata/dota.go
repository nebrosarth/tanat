package gamedata

import "math"

// «Штурм» (MapType.DOTA = 1) is the DotA-like lane pusher: two bases, each with an
// altar (Fortress Crystal) guarded by cannons, creep-spawning barracks and springs,
// connected by lanes. Destroy the enemy altar to win. The battle map is
// map_1_0.unity3d; its static structures are baked in the scene as ExportObjectData
// markers (id + world position), which the CLIENT discards on load -- so the server
// owns their placement and spawns each as a battle object by that id. All positions,
// ids and per-side prefab names below were extracted from map_1_0 with UnityPy
// (tools: scratchpad extract_export.py / dota_layout.py; see the shturm-dota-research
// memory). Human = «Собор» (Cathedral), Elf = «Изгнанники» (Exiles).

// MapTypeDota mirrors TanatKernel.MapType.DOTA.
const MapTypeDota int32 = 1

// DotaSide identifies which base a structure belongs to, as baked in map_1_0
// (GA_Human_* vs GA_Elf_*). It is NOT the in-battle team number -- the battle
// instance maps the player's chosen side to team 1 (self/allies) and the opponent
// to team -1 (enemies), reusing Hunt's team convention.
type DotaSide int

const (
	DotaSideHuman DotaSide = 0
	DotaSideElf   DotaSide = 1
)

// DotaRole classifies a static structure.
type DotaRole int

// Roles are named after the PREFAB each one is built from, which is the only stable
// ground truth here -- what a building DOES was originally the opposite of what this
// server assumed. The client's own labels settle it: the Creep_Tower prefab is
// «Казарма» (barracks) and carries the icon literally named *_creep_generator, while
// the Generator prefab is «Источник» (spring) with *_base_buffer. The map agrees:
// there are exactly 3 Creep_Towers per side and exactly 3 lanes, one tower each, while
// the 2 generators sit back at the base and match no lane (see DotaMap.LaneFor).
const (
	DotaAltar      DotaRole = iota // Fortress Crystal: destroying the enemy's wins the match
	DotaGun                        // cannon: stationary, shoots enemies in range; guards the altar
	DotaCreepTower                 // «Казарма»: the barracks of ONE lane; sends that lane's waves
	DotaGenerator                  // «Источник»: a base spring; destructible, but drives nothing yet
)

// DotaStructure is one baked map_1_0 object: its stable client object id (the
// ExportObjectData id, kept so a future map-bound client and the server agree),
// role, owning side, world (x,z) and render prefab.
type DotaStructure struct {
	ID     int32
	Role   DotaRole
	Side   DotaSide
	X, Z   float64
	Prefab string
}

// Structure hit points. The altar is a boss-sized objective; cannons and towers are
// sturdy but a coordinated push (creeps + hero) grinds them down. Armor makes them
// shrug off chip damage so creeps alone take a while (as in the real mode). The altar
// is INVULNERABLE until its side's cannons are gone (dotaAltarGuardedByGuns), the
// core «Штурм» rule ("нельзя нанести повреждения алтарю, пока не уничтожены пушки").
const (
	DotaAltarHP      = 8000.0
	DotaAltarArmor   = 20.0
	DotaGunHP        = 1400.0
	DotaGunArmor     = 12.0
	DotaGunDmgMin    = 40
	DotaGunDmgMax    = 55
	DotaGunRange     = 11.0
	DotaGunAtkSpeed  = 0.7
	DotaTowerHP      = 1100.0
	DotaTowerArmor   = 10.0
	DotaTowerDmgMin  = 30
	DotaTowerDmgMax  = 42
	DotaTowerRange   = 10.0
	DotaTowerAtk     = 0.7
)

// dotaAltarGuardedByGuns: the enemy altar only takes damage once every gun on its
// side is destroyed. true reproduces the mode; barracks/springs do not gate it.
const dotaAltarGuardedByGuns = true

// Creep-wave cadence (racial troops). Every CreepWaveInterval seconds each live
// BARRACKS (DotaCreepTower) sends CreepsPerWave troops down its own lane; the first
// wave after CreepFirstWave. Kill a barracks and its lane goes quiet -- that is the
// whole point of a 1:1 building-to-lane map, and it only works because the map really
// does place one barracks per lane (LaneFor is unambiguous by 22u or more).
//
// CreepsPerWave is therefore per lane. 4 is the size «Штурм» always shipped; it used to
// be per generator with both generators feeding the single baked lane (8 per lane per
// wave), so a lane is now a little thinner than it was while the side as a whole
// fields 12 -- three real lanes instead of one crowded one.
const (
	CreepWaveInterval = 30.0
	CreepFirstWave    = 8.0
	CreepsPerWave     = 4 // per barracks, i.e. per lane: mostly melee, ~1 archer
)

// DotaMap is the «Штурм» arena.
type DotaMap struct {
	ID         int32
	Name       string // display name (locale key)
	Scene      string // map bundle (map_1_0)
	LevelMin   int32
	LevelMax   int32
	Desc       string
	WinDesc    string
	MinPlayers int32
	MaxPlayers int32

	Structures []DotaStructure

	// Spawn points per side (near the own altar, facing the centre). The player of a
	// side starts here with a protective field, as in the mode.
	SpawnHuman Vec2
	SpawnElf   Vec2

	// Lanes are the creep march polylines, ordered from the HUMAN base end to the ELF
	// base end. A human barracks' creeps walk their lane forward (index 0..n); an elf
	// barracks' creeps walk it in reverse. Both endpoints are the ALTARS, so a lane
	// that survives to its far end walks its creeps onto the win object; the entry
	// waypoint is picked per spawner (see laneEntryIdx), not assumed to be index 0.
	Lanes [][]Vec2

	// CreepMelee/CreepRange are the mob-roster indices of each side's troops.
	HumanCreepMelee, HumanCreepRange int
	ElfCreepMelee, ElfCreepRange     int

	// Nav is the walkability oracle for this arena (nil = unrestricted movement), the
	// same contract HuntMap.Nav has. Its Spawn() is NOT used: «Штурм» starts each
	// player at their own base, so the spawn comes from the side, not from the grid.
	Nav Nav
}

// map10 is the map_1_0 «Штурм» layout, ids/positions from the bundle extraction.
var map10 = DotaMap{
	ID:         101,
	Name:       "IDS_DOTA_Text", // mode blurb key; falls back gracefully if absent
	Scene:      "map_1_0",
	LevelMin:   1,
	LevelMax:   20,
	Desc:       "Штурм — командный захват: сокрушите пушки противника и уничтожьте вражеский алтарь.",
	WinDesc:    "Уничтожьте вражеский алтарь.",
	MinPlayers: 1,
	MaxPlayers: 16,
	SpawnHuman: Vec2{X: -186, Y: -4},
	SpawnElf:   Vec2{X: 175, Y: -11},
	Nav:        navGrid10,
	HumanCreepMelee: mobHumanCreepMelee, HumanCreepRange: mobHumanCreepRange,
	ElfCreepMelee: mobElfCreepMelee, ElfCreepRange: mobElfCreepRange,
	// The three lanes, altar to altar. map_1_0 ships NO authored path of any kind --
	// the scene's 16,919 GameObjects contain no lane/waypoint marker and no path
	// script, and the client has no class that would read one (the whole route is the
	// server's business). What the map DOES pin down is where the lanes must run: the
	// 22 cannons fall into a dead-symmetric 3-per-side-per-lane layout, and each of the
	// 18 lane guns sits within 0.6u of exactly one of these polylines while the next
	// nearest is 36u+ away. The 3 creep towers per side land one per lane the same way.
	// So the lanes are not a guess -- they are the only routes that thread the guns the
	// map already placed. Between the guns each polyline was routed around the rock by
	// A* over the scene's own collision data (PassibilityData "map_1_0.xml", the same
	// polygon navGrid10 is rasterised from) and every segment re-measured against the
	// exact polygon: zero blocked segments, worst clearance 2.24u north / 2.65u centre
	// / 2.13u south, all comfortably over the 0.6u the guns themselves sit at.
	//
	// The centre lane is the one v1 shipped and is otherwise unchanged -- except that
	// its {0,-6} waypoint is now {-18.2,5.2}: the straight run from {-60,-7} to {0,-6}
	// cut 3.5u INTO a river-bank rock, so creeps walked through solid geometry. Only
	// the segment was bad; every original vertex was in free space, which is why
	// nothing caught it (and why nothing will catch the next one -- verify lane edits
	// against the polygon, not against a rasterised grid, whose cells round any
	// sub-cell clearance to zero and cry wolf).
	Lanes: [][]Vec2{
		// North: bows over the top of the diamond (Human 13/15/8, Elf 24/25/26).
		{
			{X: -202, Y: -5}, {X: -163, Y: 31}, {X: -125.8, Y: 73.8}, {X: -112, Y: 79},
			{X: -76.2, Y: 109.8}, {X: -62, Y: 114}, {X: 14.2, Y: 184.8}, {X: 33, Y: 185},
			{X: 83.2, Y: 116.2}, {X: 83, Y: 112}, {X: 150, Y: 34}, {X: 191, Y: -11},
		},
		// Centre: straight across the middle (Human 9/7/11, Elf 31/28/27).
		{
			{X: -202, Y: -5}, {X: -148, Y: -6}, {X: -101, Y: 3}, {X: -60, Y: -7},
			{X: -18.2, Y: 5.2}, {X: 29, Y: -21}, {X: 78, Y: 1}, {X: 126.2, Y: -6.8},
			{X: 130, Y: -11}, {X: 191, Y: -11},
		},
		// South: bows under the bottom of the diamond (Human 12/10/14, Elf 30/22/23).
		{
			{X: -202, Y: -5}, {X: -171, Y: -46}, {X: -146.2, Y: -83.2}, {X: -107, Y: -118},
			{X: -36, Y: -211}, {X: 27.8, Y: -201.8}, {X: 69, Y: -143}, {X: 121, Y: -100},
			{X: 162, Y: -53}, {X: 191, Y: -11},
		},
	},
	Structures: []DotaStructure{
		// Altars (win objects).
		{ID: 16, Role: DotaAltar, Side: DotaSideHuman, X: -202.16, Z: -4.94, Prefab: "GA_Human_Fortress_Crystal_prop01"},
		{ID: 33, Role: DotaAltar, Side: DotaSideElf, X: 191.02, Z: -10.94, Prefab: "GA_Elf_Fortress_Crystal_prop01"},
		// Barracks: one per lane per side (LaneFor derives which).
		{ID: 17, Role: DotaCreepTower, Side: DotaSideHuman, X: -174.56, Z: 33.55, Prefab: "GA_Human_Creep_Tower_prop01"},
		{ID: 18, Role: DotaCreepTower, Side: DotaSideHuman, X: -179.29, Z: -43.86, Prefab: "GA_Human_Creep_Tower_prop01"},
		{ID: 19, Role: DotaCreepTower, Side: DotaSideHuman, X: -158.13, Z: -1.12, Prefab: "GA_Human_Creep_Tower_prop01"},
		{ID: 34, Role: DotaCreepTower, Side: DotaSideElf, X: 154.47, Z: 23.40, Prefab: "GA_Elf_CreepTower_prop01"},
		{ID: 35, Role: DotaCreepTower, Side: DotaSideElf, X: 163.59, Z: -42.77, Prefab: "GA_Elf_CreepTower_prop01"},
		{ID: 36, Role: DotaCreepTower, Side: DotaSideElf, X: 135.57, Z: -3.22, Prefab: "GA_Elf_CreepTower_prop01"},
		// Springs (no role in the simulation yet; see DotaRole).
		{ID: 1, Role: DotaGenerator, Side: DotaSideHuman, X: -171.98, Z: 14.89, Prefab: "GA_Human_Generator_prop01"},
		{ID: 2, Role: DotaGenerator, Side: DotaSideHuman, X: -176.70, Z: -22.98, Prefab: "GA_Human_Generator_prop01"},
		{ID: 3, Role: DotaGenerator, Side: DotaSideElf, X: 152.78, Z: 4.64, Prefab: "GA_Elf_Generator_prop01"},
		{ID: 4, Role: DotaGenerator, Side: DotaSideElf, X: 157.28, Z: -22.86, Prefab: "GA_Elf_Generator_prop01"},
		// Cannons (guns).
		{ID: 5, Role: DotaGun, Side: DotaSideHuman, X: -200.45, Z: 7.39, Prefab: "GA_Human_Gun_prop01"},
		{ID: 6, Role: DotaGun, Side: DotaSideHuman, X: -200.45, Z: -16.40, Prefab: "GA_Human_Gun_prop01"},
		{ID: 7, Role: DotaGun, Side: DotaSideHuman, X: -101.14, Z: 2.62, Prefab: "GA_Human_Gun_prop01"},
		{ID: 8, Role: DotaGun, Side: DotaSideHuman, X: -62.03, Z: 113.95, Prefab: "GA_Human_Gun_prop01"},
		{ID: 9, Role: DotaGun, Side: DotaSideHuman, X: -147.95, Z: -6.40, Prefab: "GA_Human_Gun_prop01"},
		{ID: 10, Role: DotaGun, Side: DotaSideHuman, X: -106.50, Z: -118.09, Prefab: "GA_Human_Gun_prop01"},
		{ID: 11, Role: DotaGun, Side: DotaSideHuman, X: -59.64, Z: -7.44, Prefab: "GA_Human_Gun_prop01"},
		{ID: 12, Role: DotaGun, Side: DotaSideHuman, X: -171.32, Z: -46.44, Prefab: "GA_Human_Gun_prop01"},
		{ID: 13, Role: DotaGun, Side: DotaSideHuman, X: -163.17, Z: 31.43, Prefab: "GA_Human_Gun_prop01"},
		{ID: 14, Role: DotaGun, Side: DotaSideHuman, X: -35.57, Z: -211.26, Prefab: "GA_Human_Gun_prop01"},
		{ID: 15, Role: DotaGun, Side: DotaSideHuman, X: -112.03, Z: 79.45, Prefab: "GA_Human_Gun_prop01"},
		{ID: 20, Role: DotaGun, Side: DotaSideElf, X: 186.38, Z: -23.48, Prefab: "GA_Elf_Gun_prop01"},
		{ID: 21, Role: DotaGun, Side: DotaSideElf, X: 186.35, Z: 1.89, Prefab: "GA_Elf_Gun_prop01"},
		{ID: 22, Role: DotaGun, Side: DotaSideElf, X: 121.18, Z: -99.95, Prefab: "GA_Elf_Gun_prop01"},
		{ID: 23, Role: DotaGun, Side: DotaSideElf, X: 162.24, Z: -53.00, Prefab: "GA_Elf_Gun_prop01"},
		{ID: 24, Role: DotaGun, Side: DotaSideElf, X: 33.41, Z: 185.49, Prefab: "GA_Elf_Gun_prop01"},
		{ID: 25, Role: DotaGun, Side: DotaSideElf, X: 82.95, Z: 112.36, Prefab: "GA_Elf_Gun_prop01"},
		{ID: 26, Role: DotaGun, Side: DotaSideElf, X: 149.64, Z: 33.72, Prefab: "GA_Elf_Gun_prop01"},
		{ID: 27, Role: DotaGun, Side: DotaSideElf, X: 129.64, Z: -11.28, Prefab: "GA_Elf_Gun_prop01"},
		{ID: 28, Role: DotaGun, Side: DotaSideElf, X: 77.63, Z: 1.03, Prefab: "GA_Elf_Gun_prop01"},
		{ID: 30, Role: DotaGun, Side: DotaSideElf, X: 69.47, Z: -142.78, Prefab: "GA_Elf_Gun_prop01"},
		{ID: 31, Role: DotaGun, Side: DotaSideElf, X: 29.12, Z: -20.97, Prefab: "GA_Elf_Gun_prop01"},
	},
}

var dotaMaps = []DotaMap{map10}

// DotaMaps returns the «Штурм» arenas.
func DotaMaps() []DotaMap { return dotaMaps }

// DotaMapByID finds a «Штурм» map by id.
func DotaMapByID(id int32) (DotaMap, bool) {
	for _, m := range dotaMaps {
		if m.ID == id {
			return m, true
		}
	}
	return DotaMap{}, false
}

// AltarGuardedByGuns reports whether an altar is invulnerable until its side's guns
// are all destroyed (the «Штурм» push rule).
func (DotaMap) AltarGuardedByGuns() bool { return dotaAltarGuardedByGuns }

// StructByID finds a baked structure by its ExportObjectData id (the Battle server derives this
// from a structure object's id via id-dotaStructIDBase). Used by PvP battle-tasks to resolve a
// destroyed structure's role and lane. Returns false for an unknown id.
func (m DotaMap) StructByID(id int32) (DotaStructure, bool) {
	for _, sc := range m.Structures {
		if sc.ID == id {
			return sc, true
		}
	}
	return DotaStructure{}, false
}

// dotaBuildingDesc: the client's OWN name key and object-card icon for each structure
// role, per side. Every one of these was verified present in the baked locale and the
// mainData resource container -- the server may not invent either (the client resolves
// both by exact name against tables it ships, and has no fallback).
//
// The mapping is unambiguous because the prefabs mirror the keys one-for-one
// (GA_Human_Fortress_Crystal_prop01 <-> IDS_Sobor_Fortress_Crystal_Name). Sobor =
// Human, Apostate = Elf, as everywhere else in the client.
//
// These labels are what revealed that the server had two roles backwards: the
// Creep_Tower prefab is «Казарма» with the icon literally named *_creep_generator,
// while the Generator prefab is «Источник» with *_base_buffer. The simulation now
// matches -- barracks send the waves (see DotaRole) -- so name and behaviour agree.
var dotaBuildingDesc = map[DotaRole][2]struct{ NameKey, Icon string }{
	DotaAltar: {
		{"IDS_Sobor_Fortress_Crystal_Name", "Gui/Buildings/Icons/human_base_cristal"},
		{"IDS_Apostate_Fortress_Crystal_Name", "Gui/Buildings/Icons/elf_base_cristal"},
	},
	DotaGun: {
		{"IDS_Sobor_Gun_Name", "Gui/Buildings/Icons/human_gun"},
		{"IDS_Apostate_Gun_Name", "Gui/Buildings/Icons/elf_gun"},
	},
	DotaCreepTower: {
		{"IDS_Sobor_Creep_Tower_Name", "Gui/Buildings/Icons/human_creep_generator"},
		{"IDS_Apostate_Creep_Tower_Name", "Gui/Buildings/Icons/elf_creep_generator"},
	},
	DotaGenerator: {
		{"IDS_Sobor_Generator_Name", "Gui/Buildings/Icons/human_base_buffer"},
		{"IDS_Apostate_Generator_Name", "Gui/Buildings/Icons/elf_base_buffer"},
	},
}

// DotaBuildingDesc returns the name key and icon the client should show for a
// structure. The icon is the path WITHOUT the "_03" suffix: ObjectInfo appends that
// itself before loading, which is also why an empty icon is not a harmless no-op --
// it becomes a load of a texture literally named "_03".
func DotaBuildingDesc(role DotaRole, side DotaSide) (nameKey, icon string) {
	d := dotaBuildingDesc[role][side]
	return d.NameKey, d.Icon
}

// LaneFor returns the index of the lane a structure stands on, or -1 if it stands too
// far from any to belong to one. It is derived from the map's own geometry rather than
// hand-assigned, because the geometry IS the evidence: the 3 barracks per side sit
// 3.8-9.7u from their lane and 22u+ from the next -- a 1:1 assignment with no room to
// argue, which is what lets a barracks own a lane's waves.
//
// Only meaningful for the barracks (and the lane guns, which sit within 0.6u). Do NOT
// ask it about a spring: those are 5.7-15.6u out, overlapping the barracks' own range,
// and one is a mere 2u from calling it either way. Their distance says nothing, which
// is precisely why waves could never have been assigned from them.
//
// The laneClaim cutoff is a sanity net, not the discriminator: it fires only if a
// barracks drifts right off its lane, and TestEachBarracksOwnsOneLane fails loudly if
// one ever does.
func (m DotaMap) LaneFor(sc DotaStructure) int {
	const laneClaim = 12.0
	best, bestD := -1, math.Inf(1)
	for li, lane := range m.Lanes {
		for i := 0; i+1 < len(lane); i++ {
			if d := distToSeg(lane[i], lane[i+1], sc.X, sc.Z); d < bestD {
				bestD, best = d, li
			}
		}
	}
	if bestD > laneClaim {
		return -1
	}
	return best
}

// distToSeg is the distance from (x,z) to segment a-b, not to the infinite line: a
// structure can stand beside the middle of a long leg and be far from both its ends.
func distToSeg(a, b Vec2, x, z float64) float64 {
	dx, dz := b.X-a.X, b.Y-a.Y
	l2 := dx*dx + dz*dz
	if l2 == 0 {
		return math.Hypot(x-a.X, z-a.Y)
	}
	t := math.Max(0, math.Min(1, ((x-a.X)*dx+(z-a.Y)*dz)/l2))
	return math.Hypot(x-(a.X+t*dx), z-(a.Y+t*dz))
}

// CreepMobIdx returns the (melee, ranged) mob-roster indices for a side's troops.
func (m DotaMap) CreepMobIdx(side DotaSide) (melee, ranged int) {
	if side == DotaSideElf {
		return m.ElfCreepMelee, m.ElfCreepRange
	}
	return m.HumanCreepMelee, m.HumanCreepRange
}
