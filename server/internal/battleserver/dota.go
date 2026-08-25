package battleserver

// «Штурм» (MapType.DOTA) is the DotA-like lane pusher on map_1_0: two bases, each
// with an altar guarded by cannons, creep-spawning generators, and creep towers,
// connected by lanes. This file is the server-side simulation that replaces the
// Hunt mob pass for a DOTA world: it spawns creep waves that march the lanes, drives
// cannon/tower/creep combat with team-aware targeting, and ends the match when an
// altar falls.
//
// It reuses the Hunt object machinery wholesale: structures and creeps are ordinary
// mobStates in inst.mobs (so the player attacks them through the existing player
// combat, earns XP/coins, and the client renders them via CREATE_OBJECT/SYNC). The
// only DOTA-specific pieces are the team field (mobState.team), the building-prototype
// render for structures, and this tick's targeting/movement/win logic.
//
// Team convention (bulletproof, reused from Hunt): the player's side = team 1 (the
// client's self team → allies), the opponent = team -1 (the client's unconditional
// ENEMY). No GAME_DATA team table is needed.

import (
	"cmp"
	"log"
	"math"
	"slices"
	"sort"

	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
)

const (
	// Object id spaces, clear of avatar (1000+), Hunt mobs (2000+) and summons (300000+).
	dotaStructIDBase int32 = 50000 // structure objID = base + DotaStructure.ID
	dotaCreepIDBase  int32 = 60000 // creeps counted up from here

	// Prototype id spaces, clear of avatar/mob/summon protos (<=1100).
	dotaStructAttackProtoID int32 = 950 // shared attack effector for cannons/towers
	dotaStructProtoBase     int32 = 960 // building proto = base + DotaStructure.ID

	// Combat tuning.
	dotaCreepAggro                     = 13.0 // a creep engages an enemy within this radius, else marches
	dotaMeleeReach                     = 2.2  // melee attack reach added to the two body radii
	dotaWaypointHit                    = 3.0  // a creep advances to the next lane waypoint within this
	dotaSiegeStructureDamageMultiplier = 1.75

	// Team model. «Штурм» is ABSOLUTE-team, not player-relative: the Human base
	// («Собор») and everything fighting for it are team 1, the Elf base
	// («Изгнанники») and its side are team 2, on EVERY client. Two distinct positive
	// teams with no <Teams> table read as mutually hostile on the client
	// (TeamRecognizer.GetFriendliness: neither team is in a group and the battle is
	// not all-neutral, so team != selfTeam -> ENEMY), while same-team is FRIEND -- so a
	// Human player and an Elf player are enemies to each other AND to the other side's
	// creeps/cannons, with no per-viewer team rewriting. This replaces the old relative
	// model (own side = 1, opponent = -1) that only ever worked because the opponent was
	// pure bots: an Elf-side player needs the Human side to be team 1 on their screen too.
	dotaTeamHuman = int32(1)
	dotaTeamElf   = int32(2)

	// dotaPlayerTeam is the DEFAULT side of a player whose team was never set (a solo
	// launch, a direct enter, a bare test conn): the Human side. playerTeam() falls back
	// to it, so a lone «Штурм» pusher fights for «Собор» exactly as before.
	dotaPlayerTeam = dotaTeamHuman
	// dotaEnemyTeam is the teamVal() fallback for a Hunt mob that never set a team (0):
	// an unconditional enemy of the default side. It is NOT a «Штурм» structure team --
	// every DOTA structure and creep carries an explicit absolute team (1 or 2).
	dotaEnemyTeam = int32(-1)
)

// dotaSpatialCell is deliberately wider than one creep's move in a 200ms tick.
// The index is a start-of-tick snapshot; querySpatialMobs expands every query by
// the exact maximum possible creep displacement recorded for that snapshot and
// retains the old distance predicate afterwards.  It therefore only removes
// impossible candidates and cannot change target or collision results.
const dotaSpatialCell float32 = 16

type dotaMobSpatialIndex struct {
	buckets         map[uint64][]*mobState
	used            []uint64
	maxRadius       float32
	maxDisplacement float32
}

func dotaSpatialKey(x, y float32) uint64 {
	cx := int32(math.Floor(float64(x / dotaSpatialCell)))
	cy := int32(math.Floor(float64(y / dotaSpatialCell)))
	return uint64(uint32(cx))<<32 | uint64(uint32(cy))
}

func dotaSpatialRange(value, radius float32) (int32, int32) {
	return int32(math.Floor(float64((value - radius) / dotaSpatialCell))),
		int32(math.Floor(float64((value + radius) / dotaSpatialCell)))
}

func (idx *dotaMobSpatialIndex) rebuild(mobs []*mobState, now float64) {
	if idx.buckets == nil {
		idx.buckets = make(map[uint64][]*mobState)
	}
	for _, key := range idx.used {
		idx.buckets[key] = idx.buckets[key][:0]
	}
	idx.used = idx.used[:0]
	idx.maxRadius = 0
	idx.maxDisplacement = 0
	for _, mob := range mobs {
		if mob == nil || mob.dead {
			continue
		}
		key := dotaSpatialKey(mob.x, mob.y)
		bucket := idx.buckets[key]
		if len(bucket) == 0 {
			idx.used = append(idx.used, key)
		}
		idx.buckets[key] = append(bucket, mob)
		if radius := float32(mob.mob.Radius()); radius > idx.maxRadius {
			idx.maxRadius = radius
		}
		// DOTA units only move through dotaMoveTowardLocked.  Capture the
		// largest legal distance a candidate can travel before the next rebuild;
		// this makes a snapshot lookup exact even late in the sequential tick.
		speed := mobSpeed(mob, now)
		if speed <= 0 {
			speed = 4.0 // same defensive fallback as dotaMoveTowardLocked
		}
		if step := float32(speed * tickInterval.Seconds()); step > idx.maxDisplacement {
			idx.maxDisplacement = step
		}
	}
}

func (idx *dotaMobSpatialIndex) forEachNearby(x, y, radius float32, visit func(*mobState)) {
	if idx == nil || len(idx.buckets) == 0 {
		return
	}
	radius += idx.maxDisplacement
	minX, maxX := dotaSpatialRange(x, radius)
	minY, maxY := dotaSpatialRange(y, radius)
	for cellX := minX; cellX <= maxX; cellX++ {
		for cellY := minY; cellY <= maxY; cellY++ {
			key := uint64(uint32(cellX))<<32 | uint64(uint32(cellY))
			for _, mob := range idx.buckets[key] {
				visit(mob)
			}
		}
	}
}

// dotaState is the per-instance «Штурм» simulation, hung on huntInstance.dota.
type dotaState struct {
	m gamedata.DotaMap

	// Stable, allocation-free view of this tick's mob set. Waves are spawned
	// before it is rebuilt. Target selection and body separation then share the
	// same sorted pointers instead of repeatedly ranging and sorting the map for
	// every creep.
	tickMobs   []*mobState
	tickMobsAt float64
	tickMobsOK bool
	spatial    dotaMobSpatialIndex

	// nextSide alternates as players join so the two bases fill evenly (Human, Elf,
	// Human, ...). Read/advanced under the instance lock in the world-state build, the
	// twin of arenaState.nextTeam. Only consulted when a joiner arrives with no
	// pre-assigned side (PendingBattle.Team == 0): a solo/test launch, or the fallback
	// if the matchmaker didn't split. A matched player carries their side from Ctrl.
	nextSide gamedata.DotaSide

	nextWave   map[int32]float64 // generator objID -> next creep-wave battle-time
	nextCreep  int32             // rolling creep objID
	waveParity int               // alternates melee/ranged bias per wave

	// nextSiegeWaveAt is a single match-time schedule shared by all lanes. When it
	// is due, one siege creep is released from every living lane on both sides.
	// Zero means the map does not use Assault siege waves.
	nextSiegeWaveAt float64

	// instMobs aliases the instance's mob map (Go maps are references, so it stays
	// current as creeps are added/removed) -- lets altarVulnerableLocked scan the
	// guns without a back-pointer to the huntInstance.
	instMobs map[int32]*mobState

	// altarGuards maps an altar's object id to the object ids of the BASE cannons guarding
	// it (the guns flanking that altar, per gamedata.AltarGuardGunIDs). An altar opens to
	// damage once every gun in its OWN list is dead -- not once every gun on the whole side
	// is: map_1_0 gives each side 11 guns but only the 2 base cannons guard the altar, so
	// the old "any same-side gun alive" rule kept the altar invulnerable while 9 far-off
	// lane guns still stood. An empty list means "not gun-guarded" (always vulnerable).
	altarGuards map[int32][]int32

	ended  bool
	winner int32

	// castleID is nonzero when this «Штурм»-shaped match is actually a «Битва за
	// замок» siege (PendingBattle.CastleID, set by ctrlserver's castle scheduler --
	// see castle.go). Everything else about the instance (structures, teams, win
	// check) is identical to a plain «Штурм» match; this field is only consulted at
	// the very end, in dotaEndLocked, to run the castle-specific payout.
	castleID int32

	// botsFilled marks that this instance's one-time bot backfill (see bot.go,
	// TANAT_DOTA_BOTS) has already run, so a second real player joining the same
	// room doesn't spawn a second wave of bots on top of the first.
	botsFilled bool

	// startedAt is the battle-time this match's world was built -- bot_tactics.go's
	// early/late-game phase heuristic reads elapsed match time off it.
	startedAt float64

	// Periodic lane assignment accounts only for real players who have actually left
	// their fountain, so an AFK player cannot reserve a lane slot forever.
	nextLaneRebalanceAt float64
	laneActiveHumans    map[int32]bool

	// teamPlans is rebuilt from the live world once before each member loop. The
	// shared plan makes bot behavior independent of member map iteration order.
	teamPlans            map[int32]botTeamPlan
	teamPlanTelemetryKey map[int32]string
	// botAIVersionByTeam is sampled at match creation. Team 1 is Собор and
	// team 2 is Изгнанники; a missing entry means the default AI-30 profile.
	botAIVersionByTeam map[int32]int
	ai40Runtime        ai40PolicyRuntime
	ai40StartAttempted bool
	ai40FallbackByTeam map[int32]string

	// telemetry is this match's optional recording session (see telemetry.go),
	// non-nil only when TANAT_BOT_TELEMETRY is set. nil-safe on every method, so
	// every call site can use it unconditionally without an extra guard.
	telemetry *telemetryRecorder

	// earlyCreepXPMisses is an acceptance metric for bot farm coverage. It counts
	// lane-creep deaths during the first four simulated minutes for which no
	// living teammate received proximity XP. It is updated at the authoritative
	// XP-grant boundary, never inferred from an intended attack order.
	earlyCreepXPMisses int
}

const dotaEarlyCreepXPMissWindow = 4 * 60.0

// telemetryMatchTimeLocked is elapsed match time (dotaState.startedAt to now), the "t"
// every telemetry event line carries.
func (d *dotaState) telemetryMatchTimeLocked(now float64) float64 {
	return now - d.startedAt
}

// teamForSide maps a baked map side to its ABSOLUTE in-battle team: Human -> team 1,
// Elf -> team 2, the same on every client (see the team-model note on the constants).
func teamForSide(side gamedata.DotaSide) int32 {
	if side == gamedata.DotaSideElf {
		return dotaTeamElf
	}
	return dotaTeamHuman
}

// sideForTeam is the inverse: a team back to its baked side (used to place a player
// whose side arrived pre-assigned from the matchmaker).
func sideForTeam(team int32) gamedata.DotaSide {
	if team == dotaTeamElf {
		return gamedata.DotaSideElf
	}
	return gamedata.DotaSideHuman
}

// assignSide hands the next joining player a side, alternating Human/Elf so the two
// bases fill evenly. Caller holds the instance lock. The twin of arenaState.assignTeam.
func (d *dotaState) assignSide() gamedata.DotaSide {
	s := d.nextSide
	if d.nextSide == gamedata.DotaSideHuman {
		d.nextSide = gamedata.DotaSideElf
	} else {
		d.nextSide = gamedata.DotaSideHuman
	}
	return s
}

// sideSpawn is where a player of `side` starts and respawns: near that side's own altar.
func (d *dotaState) sideSpawn(side gamedata.DotaSide) (float64, float64) {
	if side == gamedata.DotaSideElf {
		return d.m.SpawnElf.X, d.m.SpawnElf.Y
	}
	return d.m.SpawnHuman.X, d.m.SpawnHuman.Y
}

// altarVulnerableLocked reports whether an altar can currently take damage: true once every
// BASE cannon guarding THIS altar is destroyed (the «Штурм» push rule -- "smash the cannons
// next to the altar"). It consults only the altar's own guard set (altarGuards), NOT every gun
// on the side, so razing the two flanking cannons opens the altar even while the side's distant
// lane guns still stand. Caller holds the world lock.
func (d *dotaState) altarVulnerableLocked(altar *mobState) bool {
	if !d.m.AltarGuardedByGuns() {
		return true
	}
	for _, gid := range d.altarGuards[altar.id] {
		if g := d.instMobs[gid]; g != nil && !g.dead {
			return false
		}
	}
	return true
}

// altarPhysImm is the client's PHYS_IMM value for an altar: 1 while its guns still
// stand (untargetable), 0 once they are gone (open season). The server-side damage guard
// (altarVulnerableLocked, consulted on the incoming-damage path) is the authority; this
// is the same rule mirrored to the client so it stops the player wasting swings.
func altarPhysImm(d *dotaState, altar *mobState) int32 {
	if d == nil || d.altarVulnerableLocked(altar) {
		return 0
	}
	return 1
}

// dotaSyncAltarImmunityLocked re-pushes every altar's PHYS_IMM. Called whenever a gun
// dies, which is the only thing that can change the answer: the moment a side's LAST gun
// falls its altar becomes attackable, and the client must hear about it or the player
// would have to guess when the push is finally on.
func (s *Server) dotaSyncAltarImmunityLocked(rep *conn, now float64) {
	d := rep.inst.dota
	if d == nil {
		return
	}
	mobIDs := make([]int32, 0, len(rep.inst.mobs))
	for id := range rep.inst.mobs {
		mobIDs = append(mobIDs, id)
	}
	sort.Slice(mobIDs, func(i, j int) bool { return mobIDs[i] < mobIDs[j] })
	for _, id := range mobIDs {
		m := rep.inst.mobs[id]
		if m == nil {
			continue
		}
		if !m.altar || m.dead {
			continue
		}
		imm := altarPhysImm(d, m)
		if imm == m.physImm {
			continue // unchanged: do not spam a stat sync every time a gun dies
		}
		m.physImm = imm
		t := float32(now)
		id := m.id
		rep.mobViewersLocked(id, func(mem *conn, idx, count int) {
			s.push(mem, battleproto.CmdSync, amf.NewArray().Set("data",
				newSyncBlob(t).setInt(syncPhysImm, idx, imm).build(count)))
		})
	}
}

// newDotaInstance builds a «Штурм» world: a huntInstance whose object set is the
// map's baked structures (altars, cannons, towers, generators) seeded as mobStates,
// plus a dotaState. Creeps are added later by the tick. The solo player always fights
// for the Human side in v1.
func newDotaInstance(s *Server, id, mapID int32) *huntInstance {
	dm, _ := gamedata.DotaMapByID(mapID)
	inst := &huntInstance{
		s:              s,
		id:             id,
		mapID:          mapID,
		nav:            dm.Nav,
		mobs:           map[int32]*mobState{},
		members:        map[int32]*conn{},
		nextFxUID:      1 << 20,
		nextSummonID:   300000,
		nextAnchorID:   400000,
		drops:          map[int32]*dropState{},
		nextDropID:     dropChestBaseID,
		nextDropItemID: dropItemBaseID,
		bots:           map[int32]*botBrain{},
	}
	d := &dotaState{
		m:                    dm,
		nextSide:             gamedata.DotaSideHuman, // the first (or solo) joiner fights for «Собор»
		nextWave:             map[int32]float64{},
		nextCreep:            dotaCreepIDBase,
		startedAt:            float64(s.battleTime()),
		laneActiveHumans:     map[int32]bool{},
		teamPlans:            map[int32]botTeamPlan{},
		teamPlanTelemetryKey: map[int32]string{},
		botAIVersionByTeam: map[int32]int{
			dotaTeamHuman: botAIVersionForTeam(dotaTeamHuman),
			dotaTeamElf:   botAIVersionForTeam(dotaTeamElf),
		},
		ai40FallbackByTeam: map[int32]string{},
	}
	if dm.SiegeCreepWaves {
		d.nextSiegeWaveAt = d.startedAt + gamedata.SiegeCreepFirstWave
	}
	if dir := botTelemetryDir(); dir != "" {
		d.telemetry = newTelemetryRecorder(dir, mapID)
		d.telemetry.record(telemetryMatchStart{
			telemetryEvent: newTelemetryEvent("match_start", 0),
			MapID:          mapID,
		})
	}
	inst.dota = d
	for _, sc := range dm.Structures {
		ms := newDotaStructure(sc, teamForSide(sc.Side))
		inst.mobs[ms.id] = ms
		if sc.Role == gamedata.DotaCreepTower {
			d.nextWave[ms.id] = float64(s.battleTime()) + gamedata.CreepFirstWave
		}
	}
	// Give every standalone creep camp the same first-wave grace as a barracks --
	// without this, an un-seeded d.nextWave key defaults to zero and the camp's very
	// first wave fires on tick 1 instead of after CreepFirstWave.
	for i := range dm.CreepCamps {
		d.nextWave[dotaCampWaveIDBase+int32(i)] = float64(s.battleTime()) + gamedata.CreepFirstWave
	}
	if dm.Boss != nil {
		boss := newDotaBossGuard(*dm.Boss)
		inst.mobs[boss.id] = boss
	}
	d.instMobs = inst.mobs
	// Resolve each altar's base-cannon guard set from the map geometry, keyed by object id so
	// altarVulnerableLocked never has to re-derive it. The altar opens once these -- and only
	// these -- are dead.
	d.altarGuards = map[int32][]int32{}
	for _, sc := range dm.Structures {
		if sc.Role != gamedata.DotaAltar {
			continue
		}
		altarObjID := dotaStructIDBase + sc.ID
		for _, gid := range dm.AltarGuardGunIDs(sc.ID) {
			d.altarGuards[altarObjID] = append(d.altarGuards[altarObjID], dotaStructIDBase+gid)
		}
	}
	log.Printf("battle: created «Штурм» (DOTA) instance room=%d map=%d structures=%d", id, mapID, len(dm.Structures))
	log.Printf("battle: bot AI profiles room=%d team1=AI-%d team2=AI-%d", id,
		d.botAIVersionByTeam[dotaTeamHuman], d.botAIVersionByTeam[dotaTeamElf])
	return inst
}

// newDotaStructure builds the mobState for one baked structure: a stationary
// destructible object with the structure's HP/armor and (for cannons/towers) attack.
func newDotaStructure(sc gamedata.DotaStructure, team int32) *mobState {
	var hp, armor, atkSpeed, atkRange, radius float64
	var dmgLo, dmgHi float64
	switch sc.Role {
	case gamedata.DotaAltar:
		hp, armor, radius = gamedata.DotaAltarHP, gamedata.DotaAltarArmor, 3.0
	case gamedata.DotaGun:
		hp, armor, radius = gamedata.DotaGunHP, gamedata.DotaGunArmor, 1.6
		atkSpeed, atkRange = gamedata.DotaGunAtkSpeed, gamedata.DotaGunRange
		dmgLo, dmgHi = gamedata.DotaGunDmgMin, gamedata.DotaGunDmgMax
	case gamedata.DotaCreepTower:
		hp, armor, radius = gamedata.DotaTowerHP, gamedata.DotaTowerArmor, 1.4
		atkSpeed, atkRange = gamedata.DotaTowerAtk, gamedata.DotaTowerRange
		dmgLo, dmgHi = gamedata.DotaTowerDmgMin, gamedata.DotaTowerDmgMax
	default: // generator: destructible barracks, no attack
		hp, armor, radius = gamedata.DotaTowerHP, gamedata.DotaTowerArmor, 1.8
	}
	m := &mobState{
		id:     dotaStructIDBase + sc.ID,
		mobIdx: -1,
		mob: gamedata.Mob{
			PhysArmor: armor, AttackSpeed: atkSpeed, AttackRange: atkRange,
			CollisionRadius: radius, Stationary: true,
		},
		x: float32(sc.X), y: float32(sc.Z),
		spawnX: float32(sc.X), spawnY: float32(sc.Z),
		// Not homed: a building never leashes, is never evicted and never respawns --
		// once it falls it is gone and the match moves on. It has spawn coords only
		// because that is where it was built.
		homed: false,
		hp:    hp, maxHP: hp, dmgMin: dmgLo, dmgMax: dmgHi,
		team: team, structure: true, altar: sc.Role == gamedata.DotaAltar,
		dotaRole: sc.Role, dotaPrefab: sc.Prefab,
		// Only the CANNONS fire a visible shell: their prefabs are the only buildings
		// carrying a projectile pool (and the only ones flagged mIsCannon). The towers
		// shoot just as far but ship an empty pool, so they stay hitscan (see hasProj).
		hasProj: sc.Role == gamedata.DotaGun,
	}
	return m
}

// dotaBossID is the object id for a DotaMap.Boss guardian (one per map today --
// map_6_0's Cerber). Clear of structures (dotaStructIDBase 50000+, well under a
// hundred of them) and creeps (dotaCreepIDBase 60000+).
const dotaBossID int32 = 70000

// newDotaBossGuard builds the mobState for a DotaMap.Boss: a hand-tuned mob-roster
// entry (bosses are exempt from level scaling -- Health/damage/XP/coins are used
// exactly as authored) planted at a fixed point. Routed through the STATIONARY
// attacker pipeline (dotaStructCombatLocked, via structure=true) rather than the
// creep chase/march pipeline: the DOTA tick has no leash-to-home logic for a roaming
// unit, so a planted guard is the reliable path -- identical machinery to a cannon.
// dotaRole is DotaGenerator (not DotaGun), so he is never counted by
// castleCheckWinLocked -- a bonus target, never a gate on the win condition.
func newDotaBossGuard(b gamedata.DotaBossSpawn) *mobState {
	mob := gamedata.MobByIndex(b.MobIdx)
	mob.Stationary = true
	if mob.AttackRange <= 0 {
		// A melee boss (Cerber has none baked -- the roster's AttackRange field is only
		// set for ranged mobs) needs SOME reach for the structure pipeline to ever pick
		// a target: dotaStructCombatLocked's acquire radius is m.mob.AttackRange
		// directly, with no melee fallback of its own (every real gun/tower already
		// bakes a real range). Boss-sized melee reach, matching his CollisionRadius.
		mob.AttackRange = 3.0
	}
	return &mobState{
		id: dotaBossID, mobIdx: b.MobIdx, mob: mob,
		x: float32(b.X), y: float32(b.Z),
		spawnX: float32(b.X), spawnY: float32(b.Z),
		homed: false, // never moves in the first place -- see the routing note above
		hp:    mob.Health, maxHP: mob.Health,
		dmgMin: float64(mob.DmgMin), dmgMax: float64(mob.DmgMax),
		// mobState.xpReward/coinReward fall back to these (not m.mob.XP/Coins) once
		// maxHP > 0 -- unlike newDotaStructure's guns/altars (which leave them at zero,
		// so a Штурм structure kill currently pays nothing), the boss is meant to be a
		// real bounty target, so they're set explicitly, exactly like a creep spawn.
		xp: mob.XP, coins: mob.Coins,
		team: teamForSide(b.Side), structure: true,
		dotaRole: gamedata.DotaGenerator, dotaPrefab: mob.Prefab,
		// Cerber is melee; a ranged DotaBossSpawn would need hasProj wired to whatever
		// projectile pool its prefab ships -- not needed yet, only Cerber exists.
		hasProj: false,
	}
}

// dotaWorldSetupLocked registers the DOTA object prototypes on the joining member and
// renders every base structure on their client. Creeps' prototypes are registered here
// too so the tick can create them without a PROTOTYPE_INFO race. Caller holds the lock.
func (s *Server) dotaWorldSetupLocked(mem *conn, now float64) {
	d := mem.inst.dota
	// Shared cannon/tower attack effector prototype.
	s.push(mem, battleproto.CmdPrototypeInfo, amf.NewArray().
		Set("id", dotaStructAttackProtoID).Set("desc", unitAttackProtoDesc()))
	// One building prototype per structure (positions/prefab differ; cheap to send all).
	for _, sc := range d.m.Structures {
		s.push(mem, battleproto.CmdPrototypeInfo, amf.NewArray().
			Set("id", dotaStructProtoBase+sc.ID).
			Set("desc", dotaBuildingProtoDesc(sc, dotaStructHP(sc.Role))))
	}
	// Creep mob + attack prototypes (both sides, melee + ranged).
	for _, idx := range d.creepMobIdxs() {
		s.push(mem, battleproto.CmdPrototypeInfo, amf.NewArray().
			Set("id", mobProtoID(idx)).Set("desc", mobProtoDesc(gamedata.Mobs()[idx])))
		s.push(mem, battleproto.CmdPrototypeInfo, amf.NewArray().
			Set("id", mobAttackProtoID(idx)).Set("desc", unitAttackProtoDesc()))
	}
	// Render every structure.
	for _, sc := range d.m.Structures {
		if m := mem.inst.mobs[dotaStructIDBase+sc.ID]; m != nil && !m.dead {
			s.dotaRevealStructureLocked(mem, m, now)
		}
	}
}

// creepMobIdxs returns the creep roster indices used by this map, including
// scheduled siege units when the map is regular Assault.
func (d *dotaState) creepMobIdxs() []int {
	hm, hr := d.m.CreepMobIdx(gamedata.DotaSideHuman)
	em, er := d.m.CreepMobIdx(gamedata.DotaSideElf)
	idxs := []int{hm, hr, em, er}
	if siege := d.m.SiegeCreepMobIdx(gamedata.DotaSideHuman); siege >= 0 {
		idxs = append(idxs, siege)
	}
	if siege := d.m.SiegeCreepMobIdx(gamedata.DotaSideElf); siege >= 0 {
		idxs = append(idxs, siege)
	}
	return idxs
}

// dotaStructHP returns a role's authored HP (for the prototype's declared max).
func dotaStructHP(role gamedata.DotaRole) float64 {
	switch role {
	case gamedata.DotaAltar:
		return gamedata.DotaAltarHP
	case gamedata.DotaGun:
		return gamedata.DotaGunHP
	default:
		return gamedata.DotaTowerHP
	}
}

// dotaBuildingProtoDesc builds a BUILDING prototype (PDestructible + PBuilding): the
// client classifies it as a building (GameData.InitMapSymbol) and renders the Fn_*
// prefab. The cannon-vs-altar distinction is baked in the prefab (mIsCannon), so no
// extra flag is sent.
//
// The name and icon come from the client's own tables (DotaBuildingDesc). This used to
// send four empty strings, on the theory that a building card would then simply not be
// drawn. It isn't so: ObjectInfo feeds mName straight to the locale, where "" is an
// error and not a request for silence, and it appends "_03" to mIcon BEFORE loading,
// so "" asked the client for a texture named "_03" on every click. Both duly failed and
// logged, and «Штурм» had 34 nameless buildings.
func dotaBuildingProtoDesc(sc gamedata.DotaStructure, hp float64) string {
	name, icon := gamedata.DotaBuildingDesc(sc.Role, sc.Side)
	return `<Proto>` +
		`<PPrefab value="` + xmlEsc(sc.Prefab) + `"/>` +
		`<PBuilding value="true"/>` +
		`<PDesc>` +
		`<Name value="` + xmlEsc(name) + `"/>` +
		`<Short value=""/><Long value=""/>` +
		`<Icon value="` + xmlEsc(icon) + `"/>` +
		`</PDesc>` +
		`<PDestructible><Health value="` + ftoa(hp) + `"/></PDestructible>` +
		`</Proto>`
}

// dotaRevealStructureLocked creates one structure on a member's client: a building
// CREATE_OBJECT plus a SYNC carrying its position, HP, team and (for cannons) combat
// stats, and an attack effector so its swing animates.
func (s *Server) dotaRevealStructureLocked(mem *conn, m *mobState, now float64) {
	hs := mem.huntState
	if hs == nil || hs.tr.index(m.id) >= 0 {
		return
	}
	bt := float32(now)
	idx := hs.tr.add(m.id)
	frac := float32(1.0)
	if m.maxHP > 0 {
		frac = float32(math.Max(m.hp, 0) / m.maxHP)
	}
	s.push(mem, battleproto.CmdCreateObject, amf.NewArray().
		Set("id", m.id).Set("proto", dotaStructProtoBase+(m.id-dotaStructIDBase)))
	blob := newSyncBlob(bt).addObject(m.id).
		position(idx, m.x, m.y, 0, 0, bt).
		setFloats(syncHealth, idx, frac).
		setFloats(syncMaxHealth, idx, float32(m.maxHP)).
		setFloats(syncRadius, idx, float32(m.mob.Radius())).
		setFloats(syncViewRadius, idx, effectiveViewRadius(structViewRadius)).
		setInt(syncTeam, idx, m.teamVal())
	if m.dmgMax > 0 { // a cannon/tower: give the client its attack stats
		blob = blob.setFloats(syncDmgMin, idx, float32(m.dmgMin)).
			setFloats(syncDmgMax, idx, float32(m.dmgMax)).
			setFloats(syncAttackSpeed, idx, float32(m.mob.AttackSpeed)).
			setFloats(syncAttackRange, idx, float32(m.mob.AttackRange))
	}
	// A guarded altar is untouchable, and the client has to be TOLD so -- see syncPhysImm.
	// Without it the altar looked like an ordinary target: the avatar walked over, swung
	// at it forever and nothing happened, because the damage was refused server-side with
	// no way for the player to know why.
	if m.altar {
		blob = blob.setInt(syncPhysImm, idx, altarPhysImm(mem.inst.dota, m))
	}
	s.push(mem, battleproto.CmdSync, amf.NewArray().Set("data", blob.build(hs.tr.count())))
	if m.dmgMax > 0 {
		s.addAttackEffectorLocked(mem, m.id, dotaStructAttackProtoID, now)
	}
}

// dotaTickLocked is the «Штурм» world pass, run once per instance tick in place of
// the Hunt mob simulation. Caller holds the world lock (rep is a live member).
func (s *Server) dotaTickLocked(rep *conn, now float64) {
	d := rep.inst.dota
	if d.ended {
		return
	}
	s.dotaCheckWinLocked(rep, now)
	if d.ended {
		return
	}
	// Re-mirror altar targetability to the clients. Driven from the tick rather than
	// hooked onto a gun's death on purpose: death forks by ATTACKER here, not by mode
	// (a gun killed by a creep goes down dotaDamageLocked, the same gun killed by the
	// player goes down hitMobFlagsLocked), so a hook would have to be remembered in both
	// -- and the sync is a no-op unless the answer actually changed.
	s.dotaSyncAltarImmunityLocked(rep, now)
	s.dotaSpawnWavesLocked(rep, now)
	// Fail any timed PvP battle-task whose deadline passed with its objective unmet.
	s.sweepPvpTaskTimersLocked(rep, now)
	// Fog of war: decide what each side's clients are actually SHOWN this tick. See
	// vision.go -- this never touches m.active, only the object list.
	if dotaHasRenderableMemberLocked(rep.inst) {
		s.dotaVisionPassLocked(rep.inst, now)
	}
	// Drive every live combatant: creeps march + fight, cannons/towers shoot.
	d.tickMobs = d.tickMobs[:0]
	for _, mob := range rep.inst.mobs {
		d.tickMobs = append(d.tickMobs, mob)
	}
	slices.SortFunc(d.tickMobs, func(a, b *mobState) int {
		return cmp.Compare(a.id, b.id)
	})
	d.tickMobsAt = now
	d.tickMobsOK = true
	d.spatial.rebuild(d.tickMobs, now)
	for _, m := range d.tickMobs {
		if m == nil {
			continue
		}
		if m.dead {
			continue
		}
		// «Штурм» runs no interest management: every unit stays simulated and rendered
		// for the whole match. Creeps and buildings fight each other with no player
		// within 200 units, and a lane must keep pushing while it sits in fog -- which
		// is exactly the coupling Hunt's pass has and this one must not grow. Setting
		// active explicitly is also what lets the SHARED helpers see these units:
		// mobSeparation skips !active, and would silently return (0,0) for the whole
		// mode otherwise.
		m.active = true
		// Statuses, DoTs and the swing close-out, exactly as the Hunt pass does them --
		// this tick replaces that pass, so without the shared call none of it runs here.
		if !s.mobUpkeepLocked(rep, m, now) {
			continue // a DoT finished it off
		}
		if !s.dotaResolveSwingLocked(rep, m, now) {
			continue // its own shot killed the thing it was standing on top of
		}
		if m.structure {
			if m.dmgMax > 0 { // cannons + towers attack; altars/generators are passive
				s.dotaStructCombatLocked(rep, m, now)
			}
			continue
		}
		s.dotaCreepTickLocked(rep, m, now)
	}
}

// Headless training policies consume authoritative observations directly and
// never decode client object-list packets. Fog remains part of those observations
// through dotaVisibleEnemy*; only the expensive render tracker/SYNC pass is skipped.
func dotaHasRenderableMemberLocked(inst *huntInstance) bool {
	for _, member := range inst.members {
		if member != nil && !member.headless {
			return true
		}
	}
	return false
}

// dotaResolveSwingLocked advances a «Штурм» unit's in-flight swing: it releases a
// committed projectile once the wind-up ends and lands a committed hit when it arrives
// (or mid-swing, for a melee/hitscan attacker). Reports whether m is still alive -- the
// hit it lands can kill m itself via a thorns/reflect path, and the caller must then
// stop driving it.
//
// Splitting release from impact is what makes a shell VISIBLE: the client flies the
// prefab over hit_at-now and drops it if hit_at is not in the future, so damage cannot
// be applied at swing time and still show a shot crossing the gap.
func (s *Server) dotaResolveSwingLocked(rep *conn, m *mobState, now float64) bool {
	// Loose the shell at the end of the wind-up, then size its flight from the LIVE gap
	// so the hit lands exactly on arrival.
	if m.projLaunchAt > 0 && now >= m.projLaunchAt {
		m.projLaunchAt = 0
		m.hitAt = now + s.dotaArrowFlightLocked(rep, m, now)
		// The shell is now the client's to fly, and its hit can no longer be called off
		// -- see mobState.projFlying.
		m.projFlying = true
		s.broadcastObjLocked(rep, m.id, battleproto.CmdSetProjectile, amf.NewArray().
			Set("source", m.id).
			Set("target", m.projTarget).
			Set("hit_at", m.hitAt))
	}
	// Land a committed hit even if the target has since moved: the swing connects
	// because it was in range when it STARTED, matching the client, which locks onto
	// the target at the start of the action.
	if m.hitAt > 0 && now >= m.hitAt {
		m.hitAt = 0
		m.projFlying = false
		s.dotaLandHitLocked(rep, m, now)
	}
	return !m.dead
}

// dotaArrowFlightLocked is how long a just-released «Штурм» shell flies, from the live
// gap to its target, capped so it lands before the shooter's next swing (which would
// otherwise overwrite the pending hit and drop the damage).
func (s *Server) dotaArrowFlightLocked(rep *conn, m *mobState, now float64) float64 {
	dist := m.mob.AttackRange // fallback if the target vanished this tick
	if tx, ty, ok := s.dotaTargetPosLocked(rep, m.projTarget, now); ok {
		dist = math.Hypot(float64(tx-m.x), float64(ty-m.y))
	}
	f := clampArrowFlight(dist)
	if hi := m.nextSwing - now - 0.02; hi > 0.04 && f > hi {
		f = hi
	}
	return f
}

// dotaTargetPosLocked locates a «Штурм» object by id -- an enemy mob/structure, a
// player's avatar, or one of their summons.
func (s *Server) dotaTargetPosLocked(rep *conn, objID int32, now float64) (float32, float32, bool) {
	if o := rep.inst.mobs[objID]; o != nil {
		return o.x, o.y, true
	}
	if mem := rep.inst.members[objID]; mem != nil && mem.huntState != nil {
		px, py := mem.posAtLocked(float32(now))
		return px, py, true
	}
	// Summons: same third map dotaLandHitLocked has to search. Without this arm a shell
	// loosed at a pet would size its flight from the shooter's nominal range instead of
	// the real gap, and land early or late. Same liveness test as the acquire scan --
	// reporting a corpse's coordinates here would size the flight to a body the client is
	// about to DELETE_OBJECT, instead of falling back to the nominal range.
	for _, mem := range rep.inst.members {
		hs := mem.huntState
		if hs == nil {
			continue
		}
		if sm, ok := hs.summons[objID]; ok && sm.alive(now) {
			return sm.x, sm.y, true
		}
	}
	return 0, 0, false
}

// dotaLandHitLocked applies a committed swing's damage to whatever it was aimed at.
func (s *Server) dotaLandHitLocked(rep *conn, m *mobState, now float64) {
	if victim := rep.inst.mobs[m.hitTarget]; victim != nil {
		dmg := m.hitDmg
		if m.siege && victim.structure {
			dmg *= dotaSiegeStructureDamageMultiplier
		}
		s.dotaDamageLocked(rep, victim, dmg, m.id, now)
		return
	}
	if mem := rep.inst.members[m.hitTarget]; mem != nil && mem.huntState != nil {
		s.hitPlayerLocked(mem, m, m.hitDmg, now)
		return
	}
	// A summon. Pets are keyed in their OWNER's map, so a committed swing against one
	// resolves against neither of the two maps above and used to evaporate silently --
	// which is what made them invulnerable here. Its Hunt twin (resolveMobHitLocked)
	// has had this arm all along.
	for _, mem := range rep.inst.members {
		hs := mem.huntState
		if hs == nil {
			continue
		}
		if sm, ok := hs.summons[m.hitTarget]; ok && !sm.dead {
			s.hitSummonLocked(mem, m, sm, m.hitDmg, now)
			return
		}
	}
}

// dotaCheckWinLocked ends the match when a base altar is gone: whichever side's altar
// fell, the OTHER side wins. Absolute-team, so the answer is the same for every viewer
// (the end screen colours team 1 blue, team 2 red). Sends BATTLE_END once to everyone.
func (s *Server) dotaCheckWinLocked(rep *conn, now float64) {
	var humanAltar, elfAltar *mobState
	anyAltar := false
	for _, m := range rep.inst.mobs {
		if !m.structure || !m.altar {
			continue
		}
		anyAltar = true
		if m.team == dotaTeamElf {
			elfAltar = m
		} else {
			humanAltar = m
		}
	}
	if !anyAltar {
		// «Битва за замок» (map_6_0): no altar exists on this map at all -- the win
		// condition is "every defending gun destroyed" instead. See castle.go.
		s.castleCheckWinLocked(rep, now)
		return
	}
	switch {
	case humanAltar == nil || humanAltar.dead:
		s.dotaEndLocked(rep, dotaTeamElf, now) // «Собор» fell -> «Изгнанники» win
	case elfAltar == nil || elfAltar.dead:
		s.dotaEndLocked(rep, dotaTeamHuman, now) // «Изгнанники» fell -> «Собор» wins
	}
}

// dotaEndLocked marks the match ended and broadcasts BATTLE_END{id: winner team}.
// dotaFreezeLocked is the single authoritative match-end barrier. It tears down
// every live player/unit action before the result scene is shown; the ticker's
// ended guard then leaves that frozen scene untouched.
func (s *Server) dotaFreezeLocked(inst *huntInstance, now float64) {
	if inst == nil {
		return
	}
	rep := repForInstanceLocked(inst)
	for _, mem := range inst.members {
		hs := mem.huntState
		if hs == nil {
			continue
		}
		if brain := inst.bots[mem.objID]; brain != nil {
			s.cancelBotTeleportAtLocked(brain, "match_end", now)
			brain.macroAssignment = botMacroAssignment{Mode: botMacroRecover, Lane: -1, Reason: "match_end"}
		}
		s.cancelOrderLocked(mem)
		s.stopAttackLocked(mem, false)
		s.stopPvpAttackLocked(mem, false)
		mem.stopArrivalLocked()
		x, y := mem.posAtLocked(float32(now))
		mem.x, mem.y, mem.vx, mem.vy, mem.snapT = x, y, 0, 0, float32(now)
		mem.hasDest = false
		hs.castLockUntil, hs.dashUntil = 0, 0
		for _, ch := range hs.channels {
			for _, uid := range ch.fxUIDs {
				s.fxEndLocked(mem, uid)
			}
			if ch.holdsTarget && ch.target != 0 {
				if target := hs.mobs[ch.target]; target != nil && !target.dead {
					target.st.rootUntil = math.Min(target.st.rootUntil, now)
					target.st.silenceUntil = math.Min(target.st.silenceUntil, now)
					target.st.atkSlowUntil = math.Min(target.st.atkSlowUntil, now)
				}
			}
		}
		hs.channels = nil
		// Deferred casts/hits cannot mature after the result is authoritative.
		hs.payloads = nil
		hs.actionDones = nil
		mem.sendPosLocked(s, x, y, 0, 0, float32(now))
	}
	for _, m := range inst.mobs {
		if m == nil || rep == nil {
			continue
		}
		s.closeMobSwingLocked(rep, m, now)
		m.resetInFlight()
		m.dtarget = 0
		m.pf = pathState{}
		m.vx, m.vy = 0, 0
		s.broadcastPosLocked(rep, m.id, m.x, m.y, 0, 0, float32(now))
	}
}

// repForInstanceLocked returns any member for shared-object fan-out.
func repForInstanceLocked(inst *huntInstance) *conn {
	for _, mem := range inst.members {
		if mem != nil {
			return mem
		}
	}
	return nil
}

func (s *Server) dotaEndLocked(rep *conn, winner int32, now float64) {
	d := rep.inst.dota
	if d.ended {
		return
	}
	d.ended = true
	d.winner = winner
	if d.ai40Runtime != nil {
		runtime := d.ai40Runtime
		d.ai40Runtime = nil
		go runtime.Close()
	}
	s.dotaFreezeLocked(rep.inst, now)
	log.Printf("battle: «Штурм» room=%d ended, winner team=%d", rep.inst.id, winner)
	for _, mem := range rep.inst.members {
		s.push(mem, battleproto.CmdBattleEnd, amf.NewArray().Set("id", winner))
	}
	s.settleMatchLocked(rep.inst, winner, now)
	if d.castleID != 0 {
		s.settleCastleBattleLocked(rep.inst, d.castleID, winner)
	}
	s.telemetryRecordMatchEndLocked(d.telemetry, rep.inst, winner, d.telemetryMatchTimeLocked(now))
	// telemetryRecordMatchEndLocked closed the recorder's channel, but the instance (and
	// this dotaState) lives on after the match ends -- players linger to see the result,
	// and runInstanceTicker keeps ticking every member, including its per-tick telemetry
	// snapshot call. Every recording call site already nil-checks d.telemetry before
	// using it; clearing it here is what makes that check mean "there is nothing left to
	// record into", not just "there was something once". Without this, the very next
	// tick's snapshot call sends on the closed channel and panics the whole server (crash
	// observed live: "panic: send on closed channel" in telemetrySnapshotLocked).
	d.telemetry = nil
}

// dotaCampWaveIDBase is the d.nextWave key space for standalone DotaCreepCamps
// (map_6_0), clear of every real structure's mobState id (dotaStructIDBase 50000 +
// at most a few dozen) and the creep id counter (dotaCreepIDBase 60000+).
const dotaCampWaveIDBase int32 = 90000

// dotaSpawnWavesLocked releases a creep wave from each live BARRACKS on its cadence
// (a dead barracks stops its lane -- the reason the map places one per lane), plus
// every standalone DotaCreepCamp (map_6_0's un-structured creep spawns -- these have
// no structure to kill, so they fire on the same cadence for the whole match).
func (s *Server) dotaSpawnWavesLocked(rep *conn, now float64) {
	d := rep.inst.dota
	for _, sc := range d.m.Structures {
		if sc.Role != gamedata.DotaCreepTower {
			continue
		}
		bar := rep.inst.mobs[dotaStructIDBase+sc.ID]
		if bar == nil || bar.dead {
			continue
		}
		if now < d.nextWave[bar.id] {
			continue
		}
		d.nextWave[bar.id] = now + gamedata.CreepWaveInterval
		s.dotaSpawnCreepWaveLocked(rep, sc, now)
	}
	for i, camp := range d.m.CreepCamps {
		key := dotaCampWaveIDBase + int32(i)
		if now < d.nextWave[key] {
			continue
		}
		d.nextWave[key] = now + gamedata.CreepWaveInterval
		s.dotaSpawnCreepWaveFromCampLocked(rep, camp, now)
	}
	s.dotaSpawnSiegeWavesLocked(rep, now)
}

// dotaSpawnSiegeWavesLocked releases one siege creep from every living lane on
// both sides when the shared Assault schedule is due. A lane without its
// barracks is intentionally silent, matching ordinary wave production.
func (s *Server) dotaSpawnSiegeWavesLocked(rep *conn, now float64) {
	d := rep.inst.dota
	if d.nextSiegeWaveAt <= 0 {
		return
	}
	for now >= d.nextSiegeWaveAt {
		s.dotaSpawnSiegeWaveLocked(rep, now)
		d.nextSiegeWaveAt += gamedata.SiegeCreepWaveInterval
	}
}

func (s *Server) dotaSpawnSiegeWaveLocked(rep *conn, now float64) {
	d := rep.inst.dota
	for _, side := range []gamedata.DotaSide{gamedata.DotaSideHuman, gamedata.DotaSideElf} {
		idx := d.m.SiegeCreepMobIdx(side)
		if idx < 0 {
			continue
		}
		for li := range d.m.Lanes {
			var bar gamedata.DotaStructure
			found := false
			for _, candidate := range d.m.Structures {
				if candidate.Role != gamedata.DotaCreepTower || candidate.Side != side || d.m.LaneFor(candidate) != li {
					continue
				}
				live := rep.inst.mobs[dotaStructIDBase+candidate.ID]
				if live == nil || live.dead {
					break
				}
				bar, found = candidate, true
				break
			}
			if !found {
				continue
			}
			entry := laneEntryIdx(d.m.Lanes[li], bar.X, bar.Z, side == gamedata.DotaSideHuman)
			s.spawnDotaSiegeCreepLocked(rep, idx, bar.X, bar.Z, side, li, entry, now)
		}
	}
}

// archerEveryNth: one creep in three carries a bow. Kept apart from CreepsPerWave so
// the mix stays a ratio -- tuning the squad size must not silently change the mix.
const archerEveryNth = 3

// laneEntryIdx picks the waypoint a creep spawning at (x,z) should head for FIRST.
//
// A lane runs altar to altar, so index 0 is not automatically ahead of whoever walks
// it: the generators sit off to one side of the lanes, and a creep entering at index 0
// would trudge 30u+ back into its OWN altar before turning round and marching out. So:
// find the lane segment the spawn is nearest to -- that is where this creep meets the
// lane -- and head for that segment's far end. "Walk to the lane, then along it".
//
// Distance to the segment, not to the vertices: a spawn can sit beside the middle of a
// long leg and be far from both of its ends. Nor "the first waypoint closer to the goal
// than I am", which sounds equivalent and is not -- a flank lane leaves the base almost
// perpendicular to the enemy, so its first legitimate leg barely closes any distance at
// all, and that test decided by 0.2u which way a creep turned.
func laneEntryIdx(lane []gamedata.Vec2, x, z float64, fwd bool) int {
	best, bestD := 0, math.Inf(1)
	for i := 0; i+1 < len(lane); i++ {
		if d := distToSeg(lane[i], lane[i+1], x, z); d < bestD {
			bestD, best = d, i
		}
	}
	if fwd {
		return best + 1
	}
	return best
}

// distToSeg is the distance from (x,z) to segment a-b (not to the infinite line).
func distToSeg(a, b gamedata.Vec2, x, z float64) float64 {
	dx, dz := b.X-a.X, b.Y-a.Y
	l2 := dx*dx + dz*dz
	if l2 == 0 {
		return math.Hypot(x-a.X, z-a.Y)
	}
	t := math.Max(0, math.Min(1, ((x-a.X)*dx+(z-a.Y)*dz)/l2))
	return math.Hypot(x-(a.X+t*dx), z-(a.Y+t*dz))
}

// dotaSpawnCreepWaveLocked spawns one barracks' wave: CreepsPerWave troops onto the
// ONE lane that barracks belongs to, marching in its side's direction, then rendered on
// every member's client.
func (s *Server) dotaSpawnCreepWaveLocked(rep *conn, bar gamedata.DotaStructure, now float64) {
	d := rep.inst.dota
	li := d.m.LaneFor(bar)
	if li < 0 || li >= len(d.m.Lanes) {
		// A barracks that stands by no lane has nowhere to send anyone. Unreachable on
		// map_1_0 (all 6 are 3.8-9.7u from their lane), but silence beats a nil lane.
		log.Printf("battle: «Штурм» barracks %d (%s) matches no lane, wave skipped", bar.ID, bar.Prefab)
		return
	}
	s.spawnCreepWaveLocked(rep, bar.X, bar.Z, bar.Side, li, now)
}

// dotaSpawnCreepWaveFromCampLocked spawns one un-structured DotaCreepCamp's wave
// (map_6_0) -- the same marching squad, just anchored to a bare point instead of a
// killable barracks.
func (s *Server) dotaSpawnCreepWaveFromCampLocked(rep *conn, camp gamedata.DotaCreepCamp, now float64) {
	if camp.Lane < 0 || camp.Lane >= len(rep.inst.dota.m.Lanes) {
		log.Printf("battle: castle creep camp at (%.1f,%.1f) matches no lane, wave skipped", camp.X, camp.Z)
		return
	}
	s.spawnCreepWaveLocked(rep, camp.X, camp.Z, camp.Side, camp.Lane, now)
}

// spawnCreepWaveLocked is the shared squad-spawn core for both a barracks wave and a
// castle creep-camp wave: CreepsPerWave troops onto lane index li, marching in side's
// direction from (x,z), then rendered on every member's client.
func (s *Server) spawnCreepWaveLocked(rep *conn, x, z float64, side gamedata.DotaSide, li int, now float64) {
	d := rep.inst.dota
	lane := d.m.Lanes[li]
	melee, ranged := d.m.CreepMobIdx(side)
	team := teamForSide(side)
	fwd := side == gamedata.DotaSideHuman // human marches lane forward, elf reverses
	entry := laneEntryIdx(lane, x, z, fwd)
	for i := 0; i < gamedata.CreepsPerWave; i++ {
		// Mostly melee with an archer in the ranks; the lane and parity terms rotate which
		// slot he is, so repeated waves don't leave in lockstep formation.
		idx := melee
		if (i+li+d.waveParity)%archerEveryNth == archerEveryNth-1 {
			idx = ranged
		}
		mob := gamedata.MobByIndex(idx) // authored stats + any live admin override
		d.nextCreep++
		// Fan the spawn a touch so they don't stack on one point.
		off := float32(i) * 0.8
		px, py := float32(x)+off, float32(z)-off
		cm := &mobState{
			id: d.nextCreep, mobIdx: idx, mob: mob,
			x: px, y: py,
			// A creep is HOMELESS: its spawn produced it, it marches a one-way lane and
			// it is deleted on death -- it never leashes, is never evicted and never
			// respawns. spawnX/spawnY are still filled in so that no shared Hunt pass that
			// reaches for a home can read the zero value and fling it to the map origin,
			// which is exactly what a player respawning near their own base used to do.
			homed:  false,
			spawnX: px, spawnY: py,
			hp: mob.Health, maxHP: mob.Health,
			dmgMin: float64(mob.DmgMin), dmgMax: float64(mob.DmgMax),
			xp: mob.XP, coins: mob.Coins,
			team: team, lane: lane, laneFwd: fwd, laneIdx: entry,
			lastSync: now,
			// The roster's ranged creep (Creep2, the only one with an AttackRange) is
			// also the only one whose prefab carries a projectile pool -- unlike the
			// towers, where range and pool come apart. Melee footmen shoot nothing.
			hasProj: mob.AttackRange > 0,
		}
		rep.inst.mobs[cm.id] = cm
		s.telemetryRecordCreepSpawnLocked(rep, cm, now)
		// Render on the creep's own side immediately; the enemy only gets it once
		// dotaVisionPassLocked (vision.go) finds it within vision, on a later tick -- the
		// same fog delay a fresh wave gets in Dota. shown=true is still set here (same
		// bookkeeping revealMobLocked used), which is what introduceMemberLocked hands a
		// late joiner on the creep's own side, so before this a player who joined a match
		// in progress saw an empty lane and took damage from nothing until the next wave.
		// It also gates mobSeparation.
		s.dotaRevealCreepToOwnTeamLocked(rep, cm, now)
	}
	d.waveParity++
}

// spawnDotaSiegeCreepLocked creates the one Creep3 unit released for a lane's
// scheduled siege wave. It intentionally shares the ordinary creep reveal and
// telemetry paths, while marking the unit for building-focused combat below.
func (s *Server) spawnDotaSiegeCreepLocked(rep *conn, idx int, x, z float64, side gamedata.DotaSide, li, laneIndex int, now float64) {
	d := rep.inst.dota
	mob := gamedata.MobByIndex(idx)
	d.nextCreep++
	cm := &mobState{
		id: d.nextCreep, mobIdx: idx, mob: mob,
		x: float32(x), y: float32(z), spawnX: float32(x), spawnY: float32(z),
		homed: false, hp: mob.Health, maxHP: mob.Health,
		dmgMin: float64(mob.DmgMin), dmgMax: float64(mob.DmgMax),
		xp: mob.XP, coins: mob.Coins,
		team: teamForSide(side), lane: d.m.Lanes[li],
		laneFwd: side == gamedata.DotaSideHuman, laneIdx: laneIndex,
		siege: true, lastSync: now, hasProj: mob.AttackRange > 0,
	}
	rep.inst.mobs[cm.id] = cm
	s.telemetryRecordCreepSpawnLocked(rep, cm, now)
	s.dotaRevealCreepToOwnTeamLocked(rep, cm, now)
}

// dotaCreepTickLocked drives one creep: engage the nearest enemy in reach (attack) or
// aggro (chase), else march its lane toward the enemy base.
func (s *Server) dotaCreepTickLocked(rep *conn, m *mobState, now float64) {
	// Integrate the creep's position from its last velocity (client dead-reckons the
	// same way, so they stay aligned between heading syncs).
	dt := now - m.lastSync
	clampedToMap := false
	if dt > 0 && (m.vx != 0 || m.vy != 0) {
		px0, py0 := m.x, m.y
		m.x += m.vx * float32(dt)
		m.y += m.vy * float32(dt)
		// Clamp to walkable ground, as every other mover already does (Hunt mobs and
		// summons both clip here). A lane creep never needed it -- the lanes are authored
		// walkable and it marched them dead straight -- but it steers around other bodies
		// now, and the lanes are narrow: the north one has ~2.24u of clearance, so a wave
		// wide enough to spread is a wave wide enough to put its flank in the rock.
		if rep.nav != nil && !rep.nav.Walkable(float64(m.x), float64(m.y)) {
			cx, cy := rep.nav.Clip(float64(px0), float64(py0), float64(m.x), float64(m.y))
			m.x, m.y = float32(cx), float32(cy)
			// The server position is now clamped at the last walkable point. The
			// client, however, dead-reckons from the last POSITION packet; leaving
			// the old velocity alive makes its model continue through the boundary
			// even though the authoritative server has stopped there. Publish the
			// clamp and stop atomically so the visual and authoritative positions
			// cannot diverge at the map edge.
			m.vx, m.vy = 0, 0
			s.broadcastPosLocked(rep, m.id, m.x, m.y, 0, 0, float32(now))
			clampedToMap = true
		}
	}
	m.lastSync = now
	if clampedToMap {
		// Do not immediately issue the same blocked heading again below. The next
		// tick will choose a fresh lane/chase direction from this valid position.
		return
	}

	// Knockback glide (see knockbackMobLocked): the shove already moved the creep to its
	// landing spot and broadcast a client-side glide there. Hold its AI until the window
	// elapses, then advanceKnockbackLocked broadcasts the stop.
	if m.kbUntil > 0 && s.advanceKnockbackLocked(rep, m, now) {
		return
	}

	// Stand still while the strike animation plays: a creature never marches or chases
	// mid-swing. This is what makes the creep plant its feet for the blow instead of
	// sliding down the lane with its weapon swinging (the client blends the attack clip
	// over the run clip, so a moving attacker reads as broken). swingDoneAt is cleared
	// by the shared upkeep when the ACTION_DONE fires.
	if m.swingDoneAt > 0 {
		s.stopMobLocked(rep, m, now)
		return
	}
	// Rooted or stunned: frozen in place. It may still swing if something is in reach
	// (root only holds the feet), but a stun locks the swing too.
	if m.st.rooted(now) {
		s.stopMobLocked(rep, m, now)
		if m.st.stunned(now) {
			return
		}
		if t := s.dotaAcquireTargetLocked(rep, m, dotaCreepAggro, now); t != nil {
			tx, ty, tr := t.pos()
			reach := s.dotaReach(m, tr)
			if float64(dist2(m.x, m.y, tx, ty)) <= reach*reach {
				s.dotaAttackLocked(rep, m, t, now)
			}
		}
		return
	}

	target := s.dotaAcquireTargetLocked(rep, m, dotaCreepAggro, now)
	if target != nil {
		tx, ty, tr := target.pos()
		reach := s.dotaReach(m, tr)
		if float64(dist2(m.x, m.y, tx, ty)) <= reach*reach {
			// Commit the swing first, THEN decide whether to hold or shuffle. Two enemy
			// waves that meet on a lane both stop here and, without a nudge, weld onto one
			// point forever (measured: a natural engagement freezes at a 0.24u gap). The
			// sidestep parts them -- but only BETWEEN swings: dotaAttackLocked sets
			// swingDoneAt when it actually strikes, and a creep mid-strike must plant its
			// feet (task #7: the client blends the attack clip over the run clip). So this
			// only ever moves a creep in the gap while its weapon is down.
			s.dotaAttackLocked(rep, m, target, now)
			if !s.dotaSidestepLocked(rep, m, now) {
				s.stopMobLocked(rep, m, now)
			}
			return
		}
		s.dotaMoveTowardLocked(rep, m, tx, ty, now)
		return
	}
	s.dotaMarchLaneLocked(rep, m, now)
}

// dotaSidestepLocked shuffles a creep that is standing in its target's reach off a
// crowding neighbour, so an engagement doesn't pile onto one point. Returns false when it
// did not move (mid-swing, rooted, stationary, or already clear) so the caller stops it.
//
// This is the in-range twin of the push folded into dotaMoveTowardLocked, at reduced
// speed: the same steering rule Hunt runs between swings (mobai.go in-range arm), sharing
// mobSepStep/mobSidestepFrac so the two drivers cannot drift apart.
func (s *Server) dotaSidestepLocked(rep *conn, m *mobState, now float64) bool {
	// Mid-strike stays planted (task #7); a rooted/stunned or stationary body never steps.
	if m.swingDoneAt > 0 || m.st.rooted(now) || m.mob.Stationary {
		return false
	}
	sepx, sepy := rep.huntState.bodySeparation(rep.instMembers(), now, m.id, m.x, m.y, m.mob.Radius())
	sn := float32(math.Hypot(float64(sepx), float64(sepy)))
	if sn <= mobSepStep {
		return false // not crowded: hold, so a lone attacker doesn't wander
	}
	speed := float32(mobSpeed(m, now)) * mobSidestepFrac
	if speed <= 0 {
		return false
	}
	vx, vy := sepx/sn*speed, sepy/sn*speed
	// Even the short separation step goes through the map path service. This
	// prevents a lateral crowding correction from cutting into a forest wall.
	if rep.nav == nil {
		return false
	}
	step := float32(tickInterval.Seconds())
	stepRoute := rep.nav.Path(float64(m.x), float64(m.y), float64(m.x+vx*step), float64(m.y+vy*step))
	if len(stepRoute) == 0 {
		return false
	}
	wp := stepRoute[0]
	dx, dy := float32(wp.X)-m.x, float32(wp.Y)-m.y
	d := float32(math.Hypot(float64(dx), float64(dy)))
	if d < 0.01 {
		return false
	}
	vx, vy = dx/d*speed, dy/d*speed
	// Same throttle as the march arm: sync a material change or a stale heading, otherwise
	// leave m.vx/m.vy alone so server integration keeps riding the velocity the client was
	// last told. Coming out of a stop (m.vx==0) headingChanged is true, so the first step
	// always goes out. Either way the creep is steering, so tell the caller not to stop it.
	changed := headingChanged(m.vx, m.vy, vx, vy)
	material := math.Hypot(float64(vx-m.vx), float64(vy-m.vy)) > float64(speed)*0.3
	if material || (changed && now-m.posSyncAt > 0.7) {
		m.vx, m.vy = vx, vy
		m.posSyncAt = now
		s.broadcastPosLocked(rep, m.id, m.x, m.y, vx, vy, float32(now))
	}
	return true
}

// dotaStructCombatLocked lets a stationary cannon/tower shoot the nearest enemy in
// range on its attack cadence.
func (s *Server) dotaStructCombatLocked(rep *conn, m *mobState, now float64) {
	// Mid-swing, or stunned: hold fire. A gun emplacement has no feet to root, so only
	// a stun stops it.
	if m.swingDoneAt > 0 || m.st.stunned(now) {
		return
	}
	rng := m.mob.AttackRange
	target := s.dotaAcquireTargetLocked(rep, m, rng, now)
	if target == nil {
		return
	}
	tx, ty, tr := target.pos()
	reach := rng + float64(tr)
	if float64(dist2(m.x, m.y, tx, ty)) > reach*reach {
		return
	}
	s.dotaAttackLocked(rep, m, target, now)
}

// dotaTarget is what a «Штурм» unit picked to swing at this tick: an enemy mobState, a
// player conn, or one of a player's summons -- plus the position and body radius the
// reach test needs. Only its id() survives the swing commit (into m.hitTarget), so the
// impact re-resolves the object itself; nothing here is carried across ticks.
type dotaTarget struct {
	mob    *mobState
	player *conn
	summon *summonState
	x, y   float32
	radius float32
}

func (t dotaTarget) pos() (float32, float32, float32) { return t.x, t.y, t.radius }
func (t dotaTarget) id() int32 {
	switch {
	case t.player != nil:
		return t.player.objID
	case t.summon != nil:
		return t.summon.id
	}
	return t.mob.id
}

// dotaAcquireTargetLocked returns the nearest ENEMY of m (an enemy-team mob or an
// enemy-team player) within `radius`, or nil.
func (s *Server) dotaAcquireTargetLocked(rep *conn, m *mobState, radius, now float64) *dotaTarget {
	dota := rep.inst.dota
	r2 := radius * radius
	var best dotaTarget
	hasBest := false
	bestD := math.Inf(1)
	bestPriority := 99
	consider := func(t dotaTarget, priority int) {
		d := float64(dist2(m.x, m.y, t.x, t.y))
		if d > r2 {
			return
		}
		// A siege creep does not abandon an exposed structure for a hero that
		// happens to be a little closer. Structures are its strategic target;
		// ordinary units retain nearest-target behavior.
		if priority < bestPriority || (priority == bestPriority &&
			(d < bestD || (d == bestD && (!hasBest || t.id() < best.id())))) {
			bestPriority, bestD, best = priority, d, t
			hasBest = true
		}
	}
	// enemy mobs (creeps + structures). During the DOTA world pass this slice is
	// already sorted and hot in cache; the fallback preserves direct-call tests.
	mobs := dota.tickMobs
	if !dota.tickMobsOK || dota.tickMobsAt != now {
		mobs = mobs[:0]
		for _, o := range rep.inst.mobs {
			mobs = append(mobs, o)
		}
	}
	considerMob := func(o *mobState) {
		if o.dead || o.id == m.id || o.teamVal() == m.teamVal() {
			return
		}
		// An enemy altar still shielded by its base cannons is not a valid target: creeps
		// (and cannons/towers, which share this picker) would otherwise plant on it and swing
		// into the void -- dotaDamageLocked refuses the hit -- while its guns still stand. Skip
		// it so the nearest-enemy pick falls to a guarding cannon instead; once the guns die
		// altarVulnerableLocked flips and the altar re-enters the candidate set. This mirrors
		// the player's altarShieldedLocked gate and the client PHYS_IMM the altar advertises.
		if o.altar && dota != nil && !dota.altarVulnerableLocked(o) {
			return
		}
		priority := 0
		if m.siege && !o.structure {
			priority = 1
		}
		consider(dotaTarget{mob: o, x: o.x, y: o.y, radius: float32(o.mob.Radius())}, priority)
	}
	if dota != nil && dota.tickMobsOK && dota.tickMobsAt == now {
		dota.spatial.forEachNearby(m.x, m.y, float32(radius), considerMob)
	} else {
		for _, o := range mobs {
			considerMob(o)
		}
	}
	// enemy players AND their summons: a creep/cannon strikes any player whose side
	// differs from its own -- Human units hunt Elf heroes and vice versa (true PvP), and
	// a solo Human pusher is hunted only by the Elf-side bots exactly as before. A pet
	// fights for its owner, so it shares that player's side and is gated the same way; a
	// same-side hero or pet is skipped, so no unit ever shoots its own team.
	for _, mem := range rep.inst.members {
		hs := mem.huntState
		if hs == nil || mem.playerTeam() == m.teamVal() {
			continue
		}
		if hs.deadUntil == 0 && now >= hs.invisibleUntil {
			px, py := mem.posAtLocked(float32(now))
			priority := 0
			if m.siege {
				priority = 1
			}
			consider(dotaTarget{player: mem, x: px, y: py, radius: float32(hs.av.Radius())}, priority)
		}
		// Summons live in their owner's map, never in inst.mobs, so the mob scan above
		// cannot see them. Only the Hunt driver was ever taught that, which made a pet
		// in «Штурм» untouchable: it dealt full damage through the shared summon tick
		// and no creep, cannon or tower could hit back for the whole match.
		// Liveness matches tickSummonsLocked's own test, not just the lazily-set flag.
		for _, sm := range hs.summons {
			if !sm.alive(now) {
				continue
			}
			priority := 0
			if m.siege {
				priority = 1
			}
			consider(dotaTarget{summon: sm, x: sm.x, y: sm.y, radius: summonRadius}, priority)
		}
	}
	if !hasBest {
		return nil
	}
	m.dotaTargetScratch = best
	return &m.dotaTargetScratch
}

// dotaReach is m's attack reach against a target of body radius tr (melee default, or
// the ranged AttackRange), plus both body radii.
func (s *Server) dotaReach(m *mobState, tr float32) float64 {
	reach := dotaMeleeReach
	if m.mob.AttackRange > 0 {
		reach = m.mob.AttackRange
	}
	return reach + m.mob.Radius() + float64(tr)
}

// dotaAttackLocked has m swing at target on its cadence: it plays the attack ACTION,
// commits the hit and schedules the swing's close-out. The hit is COMMITTED, not applied
// here -- a melee/hitscan attacker connects mid-swing, a cannon looses its shell at the
// end of the wind-up and connects when it arrives (dotaResolveSwingLocked).
func (s *Server) dotaAttackLocked(rep *conn, m *mobState, target *dotaTarget, now float64) {
	if now < m.nextSwing {
		return
	}
	// Silenced (which st.silenced also reads as stunned): no swing. A mob has no
	// spellbook, so a silence on one is a mute on its ATTACK -- effects.go says exactly
	// that where it applies the status, and the Hunt driver gates on it. This driver
	// never read the field, so silencing a creep or a tower merely throttled it to a
	// tenth of its cadence, and only by accident: the silence op sets atkSlowFactor=0.1
	// alongside, and dotaAttackLocked does honour attackFactor. Gating at the swing
	// commit covers creeps and structures in one place.
	if m.st.silenced(now) {
		return
	}
	speed := m.mob.AttackSpeed * m.st.attackFactor(now)
	if speed <= 0 {
		speed = 1.0
	}
	m.nextSwing = now + 1.0/speed
	m.dtarget = target.id()
	tx, ty, _ := target.pos()
	// Face + swing on every viewer. The swing is object-targeted (targetObj), but the
	// wire form still carries a targetPos point like every mob/summon swing -- never a
	// nil, which the AMF encoder would crash on (see newActionArgs).
	s.broadcastObjLocked(rep, m.id, battleproto.CmdAction,
		newActionArgs(m.id, m.attackProtoID(), target.id(), now,
			amf.NewArray().Set("x", float64(tx)).Set("y", float64(ty))))
	// Close the swing a hair before the next one. This is load-bearing, not polish: the
	// client only drops the attack clip when ACTION_DONE clears the action, and it plays
	// that clip BLENDED OVER the run clip -- so an unclosed swing leaves a creep marching
	// down the lane mid-strike forever. Worse, InstanceData.DoAction REJECTS an action id
	// the object is already doing, so the next swing would never animate at all.
	m.swingDoneAt = now + math.Min(0.9/speed, 1.2)
	m.hitDmg = m.rollDamage(s)
	m.hitTarget = target.id()
	if m.hasProj {
		// Ranged with a real shell: it must leave the muzzle at the END of the wind-up,
		// not partway through, so it streaks the last stretch instead of drifting across
		// the whole animation. Only if that leaves too little room for a visible flight
		// before the next swing is the release pulled earlier (never before mid-swing).
		// The hit lands on ARRIVAL, sized at release -- a ranged hit can't precede its
		// own projectile -- so hitAt stays 0 here.
		release := m.swingDoneAt
		if latest := m.nextSwing - clampArrowFlight(float64(m.mob.AttackRange)) - 0.02; release > latest {
			release = latest
		}
		if earliest := now + 0.5/speed; release < earliest {
			release = earliest
		}
		m.projLaunchAt = release
		m.projTarget = target.id()
		m.hitAt = 0
		return
	}
	// Melee, or a tower whose prefab ships no shell: the blow connects mid-swing.
	m.hitAt = now + 0.5/speed
}

// dotaDamageLocked applies creep/cannon damage to a structure or creep: armor
// mitigation, RECEIVE_HIT, HP sync, and on death ON_KILL + a scheduled corpse removal
// Nearby heroes receive creep XP even when another unit lands the last hit; gold still
// requires a hero last hit on the player damage path. Altar death ends the match.
func (s *Server) dotaDamageLocked(rep *conn, victim *mobState, dmg float64, attackerID int32, now float64) {
	if victim.dead {
		return
	}
	d := rep.inst.dota
	if victim.altar && !d.altarVulnerableLocked(victim) {
		return
	}
	if armor := victim.physArmor(now); armor != 0 {
		dmg *= armorMitigation(armor)
	}
	victim.hp -= dmg
	victim.aggro = true
	s.broadcastObjLocked(rep, victim.id, battleproto.CmdReceiveHit, amf.NewArray().
		Set("object", victim.id).Set("damager", attackerID).Set("flags", int32(0)).Set("damage", dmg))
	if victim.hp > 0 {
		frac := float32(math.Max(victim.hp, 0) / victim.maxHealth())
		s.broadcastStatLocked(rep, victim.id, syncHealth, frac, float32(now))
		return
	}
	victim.hp = 0
	victim.dead = true
	var killer *conn
	if rep.inst != nil {
		killer = rep.inst.members[attackerID]
	}
	if !victim.structure {
		s.telemetryRecordCreepDeathByIDLocked(rep, victim, attackerID, killer, now)
	}
	s.broadcastObjLocked(rep, victim.id, battleproto.CmdOnKill, amf.NewArray().
		Set("killer", attackerID).Set("id", victim.id))
	s.broadcastStatLocked(rep, victim.id, syncHealth, 0, float32(now))
	if !victim.structure {
		if attacker := rep.inst.mobs[attackerID]; attacker != nil && attacker.teamVal() != victim.teamVal() {
			s.grantDotaProximityXPLockedMeta(rep, attacker.teamVal(), victim.xpReward(), victim.x, victim.y, dotaXPGrantMeta{
				source: "creep", sourceID: victim.id,
			})
		}
	}
	// A creep/tower can land the final blow on an enemy cannon/barracks; credit the «Штурм» PvP
	// tasks here too, or a team objective met by a friendly creep would strand (or falsely FAIL,
	// for a timed task) the player's task. The player-landed twin is in hitMobFlagsLocked.
	s.creditPvpStructureKillLocked(rep.inst, victim)
	if victim.structure && rep.inst.dota.telemetry != nil {
		s.telemetryRecordStructureDestroyLocked(rep.inst.dota.telemetry, victim, killer, rep.inst.dota.telemetryMatchTimeLocked(now))
	}
	if victim.altar {
		s.dotaEndLocked(rep, teamForVictorOver(victim.team), now)
	}
	s.dotaScheduleCorpseLocked(rep.inst, victim.id)
}

// teamForVictorOver returns the BATTLE_END winner team when the altar of `loserTeam`
// falls: simply the OTHER side. Absolute-team, so it does not depend on any viewer.
func teamForVictorOver(loserTeam int32) int32 {
	if loserTeam == dotaTeamHuman {
		return dotaTeamElf
	}
	return dotaTeamHuman
}

// dotaScheduleCorpseLocked removes a dead structure/creep from every client and the
// mob set after the corpse-display window.
func (s *Server) dotaScheduleCorpseLocked(inst *huntInstance, mobID int32) {
	s.simulationAfter(corpseDeleteDelay, func() {
		inst.mu.Lock()
		defer inst.mu.Unlock()
		if inst.closed {
			return
		}
		m := inst.mobs[mobID]
		if m == nil || !m.dead {
			return
		}
		t := float32(s.battleTime())
		for _, mem := range inst.members {
			if mem.huntState != nil {
				s.untrackObjForMemberLocked(mem, mobID, t)
			}
		}
		delete(inst.mobs, mobID)
	})
}

// dotaMarchLaneLocked walks a creep along its lane toward the enemy base, advancing to
// the next waypoint on arrival. Past the last waypoint the creep holds at the enemy
// base (its dtarget acquisition then engages the altar/structures).
func (s *Server) dotaMarchLaneLocked(rep *conn, m *mobState, now float64) {
	if m.laneIdx < 0 || m.laneIdx >= len(m.lane) {
		if s.dotaMarchToAltarGuardLocked(rep, m, now) {
			return
		}
		s.stopMobLocked(rep, m, now)
		return
	}
	wp := m.lane[m.laneIdx]
	if dist2(m.x, m.y, float32(wp.X), float32(wp.Y)) <= dotaWaypointHit*dotaWaypointHit {
		if m.laneFwd {
			m.laneIdx++
		} else {
			m.laneIdx--
		}
		if m.laneIdx < 0 || m.laneIdx >= len(m.lane) {
			if s.dotaMarchToAltarGuardLocked(rep, m, now) {
				return
			}
			s.stopMobLocked(rep, m, now)
			return
		}
		wp = m.lane[m.laneIdx]
	}
	s.dotaMoveTowardLocked(rep, m, float32(wp.X), float32(wp.Y), now)
}

// dotaMarchToAltarGuardLocked: a creep that has run out of lane (arrived at the enemy
// base) but found no target within dotaCreepAggro does not just plant itself at an altar
// it cannot scratch -- dotaAcquireTargetLocked already refuses an altar still shielded by
// its base cannons (altarVulnerableLocked), so a wave that finished off the ONE cannon on
// its own lane fell through to stopMobLocked and idled there forever, even though the
// altar's OTHER base cannon (guarding from a different lane, per AltarGuardGunIDs) was
// still very much alive and killable -- exactly "крипы упираются в Алтарь... не идут
// атаковать вторую пушку". This closes that gap: reach for whichever of the enemy altar's
// guard cannons is still alive, regardless of aggro radius, so the wave finishes the guns
// instead of stalling. Returns false (caller falls back to stopMobLocked) when there is no
// enemy altar, it is already open, or every guard is already dead.
func (s *Server) dotaMarchToAltarGuardLocked(rep *conn, m *mobState, now float64) bool {
	d := rep.inst.dota
	if d == nil {
		return false
	}
	var altar *mobState
	for _, o := range rep.inst.mobs {
		if o.altar && !o.dead && o.enemyOf(m.teamVal()) {
			altar = o
			break
		}
	}
	if altar == nil || d.altarVulnerableLocked(altar) {
		return false
	}
	var nearest *mobState
	bestD := math.Inf(1)
	for _, gid := range d.altarGuards[altar.id] {
		g := d.instMobs[gid]
		if g == nil || g.dead {
			continue
		}
		if dd := float64(dist2(m.x, m.y, g.x, g.y)); dd < bestD {
			bestD, nearest = dd, g
		}
	}
	if nearest == nil {
		return false
	}
	s.dotaMoveTowardLocked(rep, m, nearest.x, nearest.y, now)
	return true
}

// dotaMoveTowardLocked heads a creep toward (tx,ty) at its move speed, steering around
// the bodies it would otherwise walk into, and emits a POSITION sync only when the
// course changes materially (the client dead-reckons in between).
//
// This is the ONE arm that may push a creep. The other two both end in stopMobLocked --
// a creep mid-swing, and a creep already in reach of its target -- and shoving either
// would slide it out from under its own attack clip, which is the exact break that made
// «крипы двигаются во время замаха» a bug. Pushing on the approach is enough: a wave
// that arrives already fanned out then plants its feet apart.
func (s *Server) dotaMoveTowardLocked(rep *conn, m *mobState, tx, ty float32, now float64) {
	dx, dy := tx-m.x, ty-m.y
	dist := float32(math.Hypot(float64(dx), float64(dy)))
	if dist < 0.01 {
		return
	}
	// Through the SHARED speed helper: this used to read m.mob.Speed raw, so a slow
	// landed its VFX, broadcast a reduced SPEED stat and then did nothing at all --
	// the creep marched at full pace while the client was told it was crawling. Root
	// and stun did work here, which is exactly why nobody noticed the slow did not.
	speed := float32(mobSpeed(m, now))
	if speed <= 0 {
		speed = 4.0
	}
	// All «Штурм» creep movement goes through A*: lane waypoints, target chases,
	// and structure approaches use the same cached route. A failed route stops the
	// creep instead of falling back to a direct heading into the obstacle.
	gx, gy, ok := rep.aimAlongAStar(&m.pf, m.x, m.y, tx, ty, now)
	if !ok {
		s.stopMobLocked(rep, m, now)
		return
	}
	dx, dy = gx-m.x, gy-m.y
	dist = float32(math.Hypot(float64(dx), float64(dy)))
	if dist < 0.01 {
		s.stopMobLocked(rep, m, now)
		return
	}
	ux, uy := dx/dist, dy/dist
	// Every creep of a wave chases the same target's exact centre and files down the
	// same lane, so without a push they converge by construction.
	sepx, sepy := rep.huntState.bodySeparation(rep.instMembers(), now, m.id, m.x, m.y, m.mob.Radius())
	stx, sty := ux+sepx*mobSepWeight, uy+sepy*mobSepWeight
	sn := float32(math.Hypot(float64(stx), float64(sty)))
	if sn < 1e-3 {
		stx, sty, sn = ux, uy, 1
	}
	vx, vy := stx/sn*speed, sty/sn*speed
	// Separation is only a steering correction. Keep its one-tick endpoint on
	// the map; if it is not, use the already path-validated route heading.
	if rep.nav != nil {
		step := float32(tickInterval.Seconds())
		if !rep.nav.Walkable(float64(m.x+vx*step), float64(m.y+vy*step)) {
			vx, vy = ux*speed, uy*speed
			if !rep.nav.Walkable(float64(m.x+vx*step), float64(m.y+vy*step)) {
				s.stopMobLocked(rep, m, now)
				return
			}
		}
	}
	// Sync a material course change at once; let a slight one wait for the staleness
	// bound. Skipping a sync is self-consistent rather than a lie: the tick above
	// integrates on m.vx/m.vy -- the velocity the client was last told -- so the two
	// stay aligned, and a push applies once it is worth a packet.
	//
	// Both halves earn their place. Without the 0.3 arm a hard shove would wait up to
	// 0.7s to reach the client, and the creep would visibly snap. Without the
	// headingChanged gate on the timer arm, a lone creep marching dead straight would
	// re-sync every 0.7s forever, spending a packet to repeat itself -- a whole wave
	// does that on every lane, and the rule here has always been one sync per waypoint.
	changed := headingChanged(m.vx, m.vy, vx, vy)
	material := math.Hypot(float64(vx-m.vx), float64(vy-m.vy)) > float64(speed)*0.3
	if material || (changed && now-m.posSyncAt > 0.7) {
		m.vx, m.vy = vx, vy
		m.posSyncAt = now
		s.broadcastPosLocked(rep, m.id, m.x, m.y, vx, vy, float32(now))
	}
}

// helpers ------------------------------------------------------------------

func dist2(ax, ay, bx, by float32) float32 {
	dx, dy := ax-bx, ay-by
	return dx*dx + dy*dy
}

// headingChanged reports whether the new velocity differs enough from the old to be
// worth a fresh POSITION sync (angle/magnitude change beyond a small tolerance).
func headingChanged(ovx, ovy, nvx, nvy float32) bool {
	if ovx == 0 && ovy == 0 {
		return true
	}
	return math.Abs(float64(ovx-nvx)) > 0.05 || math.Abs(float64(ovy-nvy)) > 0.05
}
