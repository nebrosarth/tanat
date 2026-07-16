package gamedata

// «Штурм» (MapType.DOTA = 1) is the DotA-like lane pusher: two bases, each with an
// altar (Fortress Crystal) guarded by cannons, creep-spawning generators, and creep
// towers, connected by lanes. Destroy the enemy altar to win. The battle map is
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

const (
	DotaAltar      DotaRole = iota // Fortress Crystal: destroying the enemy's wins the match
	DotaGun                        // cannon: stationary, shoots enemies in range; guards the altar
	DotaCreepTower                 // creep tower: stationary defender near a base
	DotaGenerator                  // barracks: periodically spawns a creep wave down a lane
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
// side is destroyed. true reproduces the mode; the towers/generators do not gate it.
const dotaAltarGuardedByGuns = true

// Creep-wave cadence (racial troops). A wave of CreepsPerWave leaves each generator
// every CreepWaveInterval seconds and marches its lane; the first wave after
// CreepFirstWave. Kept modest so lanes stay readable in a solo match.
const (
	CreepWaveInterval = 30.0
	CreepFirstWave    = 8.0
	CreepsPerWave     = 4 // per generator (mix of melee + ranged)
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
	// base end. A human generator's creeps walk a lane forward (index 0..n); an elf
	// generator's creeps walk it in reverse. v1 ships the centre lane; the flank lanes
	// (guns are already placed on them) are a later addition.
	Lanes [][]Vec2

	// CreepMelee/CreepRange are the mob-roster indices of each side's troops.
	HumanCreepMelee, HumanCreepRange int
	ElfCreepMelee, ElfCreepRange     int
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
	HumanCreepMelee: mobHumanCreepMelee, HumanCreepRange: mobHumanCreepRange,
	ElfCreepMelee: mobElfCreepMelee, ElfCreepRange: mobElfCreepRange,
	// Centre lane: base to base through the middle guns (Human 9/7/11, Elf 31/28/27).
	Lanes: [][]Vec2{{
		{X: -175, Y: -8}, {X: -148, Y: -6}, {X: -101, Y: 3}, {X: -60, Y: -7},
		{X: 0, Y: -6}, {X: 29, Y: -21}, {X: 78, Y: 1}, {X: 130, Y: -11}, {X: 175, Y: -11},
	}},
	Structures: []DotaStructure{
		// Altars (win objects).
		{ID: 16, Role: DotaAltar, Side: DotaSideHuman, X: -202.16, Z: -4.94, Prefab: "GA_Human_Fortress_Crystal_prop01"},
		{ID: 33, Role: DotaAltar, Side: DotaSideElf, X: 191.02, Z: -10.94, Prefab: "GA_Elf_Fortress_Crystal_prop01"},
		// Creep towers.
		{ID: 17, Role: DotaCreepTower, Side: DotaSideHuman, X: -174.56, Z: 33.55, Prefab: "GA_Human_Creep_Tower_prop01"},
		{ID: 18, Role: DotaCreepTower, Side: DotaSideHuman, X: -179.29, Z: -43.86, Prefab: "GA_Human_Creep_Tower_prop01"},
		{ID: 19, Role: DotaCreepTower, Side: DotaSideHuman, X: -158.13, Z: -1.12, Prefab: "GA_Human_Creep_Tower_prop01"},
		{ID: 34, Role: DotaCreepTower, Side: DotaSideElf, X: 154.47, Z: 23.40, Prefab: "GA_Elf_CreepTower_prop01"},
		{ID: 35, Role: DotaCreepTower, Side: DotaSideElf, X: 163.59, Z: -42.77, Prefab: "GA_Elf_CreepTower_prop01"},
		{ID: 36, Role: DotaCreepTower, Side: DotaSideElf, X: 135.57, Z: -3.22, Prefab: "GA_Elf_CreepTower_prop01"},
		// Generators (barracks).
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

// CreepMobIdx returns the (melee, ranged) mob-roster indices for a side's troops.
func (m DotaMap) CreepMobIdx(side DotaSide) (melee, ranged int) {
	if side == DotaSideElf {
		return m.ElfCreepMelee, m.ElfCreepRange
	}
	return m.HumanCreepMelee, m.HumanCreepRange
}
