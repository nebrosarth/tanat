package battleserver

// «Штурм» (MapType.DOTA) is the DotA-like lane pusher on map_1_0: two bases, each
// with an altar guarded by cannons, creep-spawning generators, and creep towers,
// connected by a lane. This file is the server-side simulation that replaces the
// Hunt mob pass for a DOTA world: it spawns creep waves that march the lane, drives
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
	"log"
	"math"
	"time"

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
	dotaCreepAggro   = 13.0 // a creep engages an enemy within this radius, else marches
	dotaMeleeReach   = 2.2  // melee attack reach added to the two body radii
	dotaWaypointHit  = 3.0  // a creep advances to the next lane waypoint within this
	dotaPlayerTeam   = int32(1)
	dotaEnemyTeam    = int32(-1)
	dotaWinTeamSelf  = int32(1) // BATTLE_END winner when the player's side wins
	dotaWinTeamEnemy = int32(2) // ... and when it loses (enemy = display team 2)
)

// dotaState is the per-instance «Штурм» simulation, hung on huntInstance.dota.
type dotaState struct {
	m          gamedata.DotaMap
	playerSide gamedata.DotaSide // the side every (solo v1) player fights for

	nextWave   map[int32]float64 // generator objID -> next creep-wave battle-time
	nextCreep  int32             // rolling creep objID
	waveParity int               // alternates melee/ranged bias per wave

	// instMobs aliases the instance's mob map (Go maps are references, so it stays
	// current as creeps are added/removed) -- lets altarVulnerableLocked scan the
	// guns without a back-pointer to the huntInstance.
	instMobs map[int32]*mobState

	ended  bool
	winner int32
}

// teamForSide maps a baked map side to the in-battle team from the player's point of
// view: the player's own side is team 1 (allies), the opponent team -1 (enemies).
func (d *dotaState) teamForSide(side gamedata.DotaSide) int32 {
	if side == d.playerSide {
		return dotaPlayerTeam
	}
	return dotaEnemyTeam
}

// enemySide is the side opposite the player's.
func (d *dotaState) enemySide() gamedata.DotaSide {
	if d.playerSide == gamedata.DotaSideHuman {
		return gamedata.DotaSideElf
	}
	return gamedata.DotaSideHuman
}

// playerSpawn is where a player of this world's side starts (near its own altar).
func (d *dotaState) playerSpawn() (float64, float64) {
	if d.playerSide == gamedata.DotaSideElf {
		return d.m.SpawnElf.X, d.m.SpawnElf.Y
	}
	return d.m.SpawnHuman.X, d.m.SpawnHuman.Y
}

// altarVulnerableLocked reports whether an altar can currently take damage: true once
// every cannon on its side is destroyed (the «Штурм» push rule). Caller holds the
// world lock.
func (d *dotaState) altarVulnerableLocked(altar *mobState) bool {
	if !d.m.AltarGuardedByGuns() {
		return true
	}
	for _, m := range d.instMobs {
		if m.structure && m.dotaRole == gamedata.DotaGun && m.team == altar.team && !m.dead {
			return false
		}
	}
	return true
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
		mobs:           map[int32]*mobState{},
		members:        map[int32]*conn{},
		nextFxUID:      1 << 20,
		nextSummonID:   300000,
		nextAnchorID:   400000,
		drops:          map[int32]*dropState{},
		nextDropID:     dropChestBaseID,
		nextDropItemID: dropItemBaseID,
	}
	d := &dotaState{
		m:          dm,
		playerSide: gamedata.DotaSideHuman,
		nextWave:   map[int32]float64{},
		nextCreep:  dotaCreepIDBase,
	}
	inst.dota = d
	for _, sc := range dm.Structures {
		ms := newDotaStructure(sc, d.teamForSide(sc.Side))
		inst.mobs[ms.id] = ms
		if sc.Role == gamedata.DotaGenerator {
			d.nextWave[ms.id] = float64(s.battleTime()) + gamedata.CreepFirstWave
		}
	}
	d.instMobs = inst.mobs
	log.Printf("battle: created «Штурм» (DOTA) instance room=%d map=%d structures=%d", id, mapID, len(dm.Structures))
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
		hp: hp, maxHP: hp, dmgMin: dmgLo, dmgMax: dmgHi,
		team: team, structure: true, altar: sc.Role == gamedata.DotaAltar,
		dotaRole: sc.Role, dotaPrefab: sc.Prefab,
	}
	return m
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
			Set("desc", dotaBuildingProtoDesc(sc.Prefab, dotaStructHP(sc.Role), sc.Role == gamedata.DotaAltar)))
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

// creepMobIdxs returns the four creep roster indices used by this map.
func (d *dotaState) creepMobIdxs() []int {
	hm, hr := d.m.CreepMobIdx(gamedata.DotaSideHuman)
	em, er := d.m.CreepMobIdx(gamedata.DotaSideElf)
	return []int{hm, hr, em, er}
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
func dotaBuildingProtoDesc(prefab string, hp float64, altar bool) string {
	return `<Proto>` +
		`<PPrefab value="` + xmlEsc(prefab) + `"/>` +
		`<PBuilding value="true"/>` +
		`<PDesc><Name value=""/><Short value=""/><Long value=""/><Icon value=""/></PDesc>` +
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
		setInt(syncTeam, idx, m.teamVal())
	if m.dmgMax > 0 { // a cannon/tower: give the client its attack stats
		blob = blob.setFloats(syncDmgMin, idx, float32(m.dmgMin)).
			setFloats(syncDmgMax, idx, float32(m.dmgMax)).
			setFloats(syncAttackSpeed, idx, float32(m.mob.AttackSpeed)).
			setFloats(syncAttackRange, idx, float32(m.mob.AttackRange))
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
	s.dotaSpawnWavesLocked(rep, now)
	// Drive every live combatant: creeps march + fight, cannons/towers shoot.
	for _, m := range rep.inst.mobs {
		if m.dead {
			continue
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

// dotaCheckWinLocked ends the match when an altar is gone: the player's side wins if
// the enemy altar fell, loses if its own did. Sends BATTLE_END once to every member.
func (s *Server) dotaCheckWinLocked(rep *conn, now float64) {
	d := rep.inst.dota
	enemyTeam := d.teamForSide(d.enemySide())
	var enemyAltar, ownAltar *mobState
	for _, m := range rep.inst.mobs {
		if !m.structure || !m.altar {
			continue
		}
		if m.team == enemyTeam {
			enemyAltar = m
		} else {
			ownAltar = m
		}
	}
	switch {
	case enemyAltar == nil || enemyAltar.dead:
		s.dotaEndLocked(rep, dotaWinTeamSelf, now)
	case ownAltar == nil || ownAltar.dead:
		s.dotaEndLocked(rep, dotaWinTeamEnemy, now)
	}
}

// dotaEndLocked marks the match ended and broadcasts BATTLE_END{id: winner team}.
func (s *Server) dotaEndLocked(rep *conn, winner int32, now float64) {
	d := rep.inst.dota
	if d.ended {
		return
	}
	d.ended = true
	d.winner = winner
	log.Printf("battle: «Штурм» room=%d ended, winner team=%d", rep.inst.id, winner)
	for _, mem := range rep.inst.members {
		s.push(mem, battleproto.CmdBattleEnd, amf.NewArray().Set("id", winner))
	}
}

// dotaSpawnWavesLocked releases a creep wave from each live generator on its cadence.
func (s *Server) dotaSpawnWavesLocked(rep *conn, now float64) {
	d := rep.inst.dota
	for _, sc := range d.m.Structures {
		if sc.Role != gamedata.DotaGenerator {
			continue
		}
		gen := rep.inst.mobs[dotaStructIDBase+sc.ID]
		if gen == nil || gen.dead {
			continue
		}
		if now < d.nextWave[gen.id] {
			continue
		}
		d.nextWave[gen.id] = now + gamedata.CreepWaveInterval
		s.dotaSpawnCreepWaveLocked(rep, sc, now)
	}
}

// dotaSpawnCreepWaveLocked spawns one generator's wave: CreepsPerWave troops at the
// generator, each assigned the centre lane and its side's march direction, then
// rendered on every member's client.
func (s *Server) dotaSpawnCreepWaveLocked(rep *conn, gen gamedata.DotaStructure, now float64) {
	d := rep.inst.dota
	if len(d.m.Lanes) == 0 {
		return
	}
	lane := d.m.Lanes[0]
	melee, ranged := d.m.CreepMobIdx(gen.Side)
	team := d.teamForSide(gen.Side)
	fwd := gen.Side == gamedata.DotaSideHuman // human marches lane forward, elf reverses
	for i := 0; i < gamedata.CreepsPerWave; i++ {
		// Mostly melee with one ranged per wave (parity alternates which slot).
		idx := melee
		if (i+d.waveParity)%3 == 2 {
			idx = ranged
		}
		mob := gamedata.Mobs()[idx]
		d.nextCreep++
		// Fan the spawn a touch so they don't stack on one point.
		off := float32(i) * 0.8
		cm := &mobState{
			id: d.nextCreep, mobIdx: idx, mob: mob,
			x: float32(gen.X) + off, y: float32(gen.Z) - off,
			hp: mob.Health, maxHP: mob.Health,
			dmgMin: float64(mob.DmgMin), dmgMax: float64(mob.DmgMax),
			xp: mob.XP, coins: mob.Coins,
			team: team, lane: lane, laneFwd: fwd,
			lastSync: now,
		}
		if fwd {
			cm.laneIdx = 0
		} else {
			cm.laneIdx = len(lane) - 1
		}
		rep.inst.mobs[cm.id] = cm
		for _, mem := range rep.inst.members {
			if mem.huntState != nil {
				s.revealMobToMemberLocked(mem, cm, now)
			}
		}
	}
	d.waveParity++
}

// dotaCreepTickLocked drives one creep: engage the nearest enemy in reach (attack) or
// aggro (chase), else march its lane toward the enemy base.
func (s *Server) dotaCreepTickLocked(rep *conn, m *mobState, now float64) {
	// Integrate the creep's position from its last velocity (client dead-reckons the
	// same way, so they stay aligned between heading syncs).
	dt := now - m.lastSync
	if dt > 0 && (m.vx != 0 || m.vy != 0) {
		m.x += m.vx * float32(dt)
		m.y += m.vy * float32(dt)
	}
	m.lastSync = now

	target := s.dotaAcquireTargetLocked(rep, m, dotaCreepAggro, now)
	if target != nil {
		tx, ty, tr := target.pos()
		reach := s.dotaReach(m, tr)
		if float64(dist2(m.x, m.y, tx, ty)) <= reach*reach {
			s.dotaStopLocked(rep, m, now)
			s.dotaAttackLocked(rep, m, target, mobAttackProtoID(m.mobIdx), now)
			return
		}
		s.dotaMoveTowardLocked(rep, m, tx, ty, now)
		return
	}
	s.dotaMarchLaneLocked(rep, m, now)
}

// dotaStructCombatLocked lets a stationary cannon/tower shoot the nearest enemy in
// range on its attack cadence.
func (s *Server) dotaStructCombatLocked(rep *conn, m *mobState, now float64) {
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
	s.dotaAttackLocked(rep, m, target, dotaStructAttackProtoID, now)
}

// dotaTarget is either an enemy mobState or a player conn.
type dotaTarget struct {
	mob    *mobState
	player *conn
	x, y   float32
	radius float32
}

func (t dotaTarget) pos() (float32, float32, float32) { return t.x, t.y, t.radius }
func (t dotaTarget) id() int32 {
	if t.player != nil {
		return t.player.objID
	}
	return t.mob.id
}

// dotaAcquireTargetLocked returns the nearest ENEMY of m (an enemy-team mob or an
// enemy-team player) within `radius`, or nil.
func (s *Server) dotaAcquireTargetLocked(rep *conn, m *mobState, radius, now float64) *dotaTarget {
	r2 := radius * radius
	var best *dotaTarget
	bestD := math.Inf(1)
	consider := func(t *dotaTarget) {
		d := float64(dist2(m.x, m.y, t.x, t.y))
		if d <= r2 && d < bestD {
			bestD, best = d, t
		}
	}
	// enemy mobs (creeps + structures)
	for _, o := range rep.inst.mobs {
		if o.dead || o.id == m.id || o.teamVal() == m.teamVal() {
			continue
		}
		consider(&dotaTarget{mob: o, x: o.x, y: o.y, radius: float32(o.mob.Radius())})
	}
	// enemy players: only if m is on the opposite side (in solo v1 the player is
	// team 1, so only enemy-team creeps/cannons ever target them).
	if m.teamVal() != dotaPlayerTeam {
		for _, mem := range rep.inst.members {
			hs := mem.huntState
			if hs == nil || hs.deadUntil > 0 {
				continue
			}
			px, py := mem.posAtLocked(float32(now))
			consider(&dotaTarget{player: mem, x: px, y: py, radius: float32(hs.av.Radius())})
		}
	}
	return best
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

// dotaAttackLocked has m swing at target on its cadence, playing the attack ACTION and
// applying the hit (to a mob via dotaDamage, to a player via the Hunt hitPlayer path).
func (s *Server) dotaAttackLocked(rep *conn, m *mobState, target *dotaTarget, actionProto int32, now float64) {
	if now < m.nextSwing {
		return
	}
	speed := m.mob.AttackSpeed
	if speed <= 0 {
		speed = 1.0
	}
	m.nextSwing = now + 1.0/speed
	m.dtarget = target.id()
	// Face + swing on every viewer.
	s.broadcastObjLocked(rep, m.id, battleproto.CmdAction,
		newActionArgs(m.id, actionProto, target.id(), now, nil))
	dmg := m.rollDamage()
	if target.player != nil {
		s.hitPlayerLocked(target.player, m, dmg, now)
		return
	}
	s.dotaDamageLocked(rep, target.mob, dmg, m.id, now)
}

// dotaDamageLocked applies creep/cannon damage to a structure or creep: armor
// mitigation, RECEIVE_HIT, HP sync, and on death ON_KILL + a scheduled corpse removal
// (no XP -- unit-vs-unit kills don't reward a player). Altar death ends the match.
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
	s.broadcastObjLocked(rep, victim.id, battleproto.CmdOnKill, amf.NewArray().
		Set("killer", attackerID).Set("id", victim.id))
	s.broadcastStatLocked(rep, victim.id, syncHealth, 0, float32(now))
	if victim.altar {
		s.dotaEndLocked(rep, d.teamForVictorOver(victim.team), now)
	}
	s.dotaScheduleCorpseLocked(rep.inst, victim.id)
}

// teamForVictorOver returns the BATTLE_END winner team when the altar of `loserTeam`
// falls: the player's display team (1) if the ENEMY altar fell, else the enemy's (2).
func (d *dotaState) teamForVictorOver(loserTeam int32) int32 {
	if loserTeam == dotaPlayerTeam {
		return dotaWinTeamEnemy
	}
	return dotaWinTeamSelf
}

// dotaScheduleCorpseLocked removes a dead structure/creep from every client and the
// mob set after the corpse-display window.
func (s *Server) dotaScheduleCorpseLocked(inst *huntInstance, mobID int32) {
	time.AfterFunc(corpseDeleteDelay, func() {
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
		s.dotaStopLocked(rep, m, now)
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
			s.dotaStopLocked(rep, m, now)
			return
		}
		wp = m.lane[m.laneIdx]
	}
	s.dotaMoveTowardLocked(rep, m, float32(wp.X), float32(wp.Y), now)
}

// dotaMoveTowardLocked heads a creep toward (tx,ty) at its move speed, emitting a
// POSITION sync only when the heading changes materially (the client dead-reckons in
// between), so a straight march is one sync per waypoint.
func (s *Server) dotaMoveTowardLocked(rep *conn, m *mobState, tx, ty float32, now float64) {
	dx, dy := tx-m.x, ty-m.y
	dist := float32(math.Hypot(float64(dx), float64(dy)))
	if dist < 0.01 {
		return
	}
	speed := float32(m.mob.Speed)
	if speed <= 0 {
		speed = 4.0
	}
	vx, vy := dx/dist*speed, dy/dist*speed
	// Re-sync only on a real heading change (or coming out of a stop).
	if headingChanged(m.vx, m.vy, vx, vy) {
		m.vx, m.vy = vx, vy
		s.broadcastPosLocked(rep, m.id, m.x, m.y, vx, vy, float32(now))
	}
}

// dotaStopLocked freezes a creep in place and syncs the halt (once).
func (s *Server) dotaStopLocked(rep *conn, m *mobState, now float64) {
	if m.vx == 0 && m.vy == 0 {
		return
	}
	m.vx, m.vy = 0, 0
	s.broadcastPosLocked(rep, m.id, m.x, m.y, 0, 0, float32(now))
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
