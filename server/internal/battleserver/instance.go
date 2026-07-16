package battleserver

import (
	"log"
	"sync"
	"time"

	"tanatserver/internal/gamedata"
)

// A huntInstance is one shared hunt world. Several player connections routed to
// the same room (open instance per map -- see the Ctrl matchmaking) join one
// instance: a single authoritative mob simulation, a single 200ms ticker, and a
// membership list so mob syncs, aggro and cross-player rendering fan out to
// everyone. A lone player is just an instance with one member, so single-player
// behaviour is the N==1 case of the shared world.
//
// Locking: inst.mu guards the simulation (mobs + members + every member's hunt
// state, since each member's conn.lk points at it). The Server's registry
// (s.insts) is guarded by s.mu. The two locks are NEVER held nested -- join and
// disposal always release one before taking the other -- so no lock-order cycle
// exists.
type huntInstance struct {
	mu      sync.Mutex
	s       *Server
	id      int32
	mapID   int32
	m       gamedata.HuntMap
	nav     gamedata.Nav
	mobs    map[int32]*mobState
	members map[int32]*conn // keyed by avatar objID
	closed  bool

	// nextFxUID allocates world-scoped effect ids (mob debuffs, boss telegraphs,
	// fog-ring shade) that are broadcast to every member with the SAME id, so a
	// mob's stun/slow/shade looks identical on all clients and can be ended
	// everywhere. Based high above the per-connection fx counter (hs.nextFxUID,
	// used for owner-only self visuals) so the two spaces never collide on a client.
	nextFxUID int32

	// nextSummonID allocates summon object ids from ONE space for the whole party,
	// so two members' summons never share an id. The shared mob simulation resolves
	// a committed hit / kill credit by summon id across every member (resolveMobHit,
	// creditConn), which would misroute to the wrong player if ids collided. Based
	// clear of avatar ids (1000+) and mob ids (2000+).
	nextSummonID int32

	// nextAnchorID allocates ids for invisible trap-fx anchor objects (a SELF-mode
	// ground fx parents to a stationary object so it holds the cast point). One space
	// for the whole party, based clear of summon ids (300000+).
	nextAnchorID int32

	// drops holds every currently-spawned loot chest, keyed by container object id
	// (see drops.go). nextDropID/nextDropItemID allocate the container's object id
	// and the single item-inside-it's own wire id from two more party-wide spaces,
	// based clear of anchor ids (400000+).
	drops          map[int32]*dropState
	nextDropID     int32
	nextDropItemID int32

	// dota is non-nil for a «Штурм» (MapType.DOTA) world: the two-base lane-pusher
	// simulation (structures + creeps + win condition) that replaces the Hunt mob
	// pass in the tick loop. See dota.go. nil = a normal Hunt world.
	dota *dotaState
}

// newHuntInstance builds an instance for a map and seeds its shared mob set from
// the map's authored spawns (the roster all members fight). Level scaling that
// used to happen per connection now happens once, here.
func newHuntInstance(s *Server, id, mapID int32) *huntInstance {
	m, _ := gamedata.HuntMapByID(mapID)
	inst := &huntInstance{
		s:            s,
		id:           id,
		mapID:        mapID,
		m:            m,
		nav:          m.Nav,
		mobs:         map[int32]*mobState{},
		members:      map[int32]*conn{},
		nextFxUID:      1 << 20, // world fx id base, far above per-conn hs.nextFxUID
		nextSummonID:   300000,  // party-wide summon id base, clear of avatar/mob ids
		nextAnchorID:   400000,  // party-wide trap-anchor id base, clear of summon ids
		drops:          map[int32]*dropState{},
		nextDropID:     dropChestBaseID,
		nextDropItemID: dropItemBaseID,
	}
	baseX, baseY := m.Spawn()
	for i, sp := range m.Spawns {
		mobT := gamedata.Mobs()[sp.Mob]
		mid := int32(2000 + i)
		// Abs spawns (bosses) carry absolute world coords; the rest are offsets from
		// the player spawn.
		mx, my := baseX+sp.DX, baseY+sp.DY
		if sp.Abs {
			mx, my = sp.DX, sp.DY
		}
		// Defensive: if a spawn lands off the walkable floor (e.g. a hand-authored boss
		// coord that drifted, or an offset pack near a wall), snap it to the nearest
		// reachable cell so nothing spawns inside geometry. A no-op for the current maps,
		// whose placements are all walkable by construction (crypt/jungle generators and
		// invasionPack41 round-then-test; bosses pinned to verified cells).
		if inst.nav != nil && !inst.nav.Walkable(mx, my) {
			if p := inst.nav.Path(baseX, baseY, mx, my); len(p) > 0 {
				mx, my = p[len(p)-1].X, p[len(p)-1].Y
			} else {
				mx, my = inst.nav.Clip(baseX, baseY, mx, my)
			}
		}
		lvl := sp.Level
		if lvl < 1 {
			lvl = 1
		}
		ms := &mobState{
			id: mid, mobIdx: sp.Mob, mob: mobT,
			x:      float32(mx),
			y:      float32(my),
			spawnX: float32(mx),
			spawnY: float32(my),
			level:  lvl,
		}
		ms.maxHP, ms.dmgMin, ms.dmgMax, ms.xp, ms.coins = mobT.ScaledStats(lvl)
		ms.hp = ms.maxHP
		if len(mobT.Skills) > 0 {
			ms.skillReady = make([]float64, len(mobT.Skills))
		}
		inst.mobs[mid] = ms
	}
	return inst
}

// joinInstance places c into the shared world for roomID (creating it, seeded
// from mapID, if this is the first arrival), points c's lock at the instance and
// starts the ticker for a fresh world. Returns the instance c joined.
//
// Concurrency: s.mu and inst.mu are taken sequentially, never nested. If the
// found instance is mid-disposal (closed) we retry, which re-creates a fresh one.
func (s *Server) joinInstance(roomID, mapID int32, c *conn) *huntInstance {
	for {
		s.mu.Lock()
		inst := s.insts[roomID]
		if inst == nil {
			if _, ok := gamedata.DotaMapByID(mapID); ok {
				inst = newDotaInstance(s, roomID, mapID)
			} else {
				inst = newHuntInstance(s, roomID, mapID)
			}
			s.insts[roomID] = inst
			s.mu.Unlock()
			inst.mu.Lock()
			inst.members[c.objID] = c
			c.inst = inst
			c.lk = &inst.mu
			inst.mu.Unlock()
			go s.runInstanceTicker(inst)
			log.Printf("battle: created hunt instance room=%d map=%d (member %d)", roomID, mapID, c.objID)
			return inst
		}
		s.mu.Unlock()

		inst.mu.Lock()
		if inst.closed {
			inst.mu.Unlock()
			continue // being disposed; loop re-creates a fresh instance
		}
		inst.members[c.objID] = c
		c.inst = inst
		c.lk = &inst.mu
		n := len(inst.members)
		inst.mu.Unlock()
		log.Printf("battle: joined hunt instance room=%d map=%d (member %d, now %d players)",
			roomID, mapID, c.objID, n)
		return inst
	}
}

// memberList snapshots the instance's READY members -- those whose world state is
// fully built. A just-joined member is in inst.members (so the world isn't
// disposed) but excluded here until worldReady, so the ticker and every broadcast
// fan-out skip it while its scene is still loading. Caller holds inst.mu.
func (inst *huntInstance) memberList() []*conn {
	out := make([]*conn, 0, len(inst.members))
	for _, c := range inst.members {
		if c.huntState != nil && c.huntState.worldReady {
			out = append(out, c)
		}
	}
	return out
}

// leaveInstance removes c from its instance (disconnect) and tells the remaining
// members to drop c's avatar. The ticker disposes the world once it is empty.
// Caller holds inst.mu (via c.lock() in closeHunt).
func (inst *huntInstance) leaveInstanceLocked(c *conn) {
	if _, ok := inst.members[c.objID]; !ok {
		return
	}
	delete(inst.members, c.objID)
	now := float64(inst.s.battleTime())
	// Tell everyone still here to remove the departing avatar and its summons.
	for _, other := range inst.members {
		if other == c {
			continue
		}
		inst.s.removeAvatarForLocked(other, c)
		if c.huntState != nil {
			for _, sm := range c.huntState.summons {
				inst.s.hideSummonFromMemberLocked(other, sm, now)
			}
			// Delete the leaver's trap-fx anchors too, else they leak on the remaining
			// members (the leaver's tick that would have removed them is gone).
			for _, t := range c.huntState.traps {
				if t.anchor != 0 {
					inst.s.untrackObjForMemberLocked(other, t.anchor, float32(now))
				}
			}
			for _, a := range c.huntState.anchorEnds {
				inst.s.untrackObjForMemberLocked(other, a.id, float32(now))
			}
		}
	}
}

// runInstanceTicker is the single simulation loop for one shared world: every
// 200ms it runs each member's per-player upkeep and one shared mob pass, then
// disposes the world once the last member has left.
func (s *Server) runInstanceTicker(inst *huntInstance) {
	t := time.NewTicker(tickInterval)
	defer t.Stop()
	for range t.C {
		inst.mu.Lock()
		if inst.closed {
			inst.mu.Unlock()
			return
		}
		if len(inst.members) == 0 {
			inst.closed = true
			inst.mu.Unlock()
			s.mu.Lock()
			if s.insts[inst.id] == inst {
				delete(s.insts, inst.id)
			}
			s.mu.Unlock()
			log.Printf("battle: disposed empty hunt instance room=%d", inst.id)
			return
		}
		now := float64(s.battleTime())
		var rep *conn
		for _, c := range inst.memberList() { // ready members only
			if !c.huntState.closed {
				s.memberTickLocked(c, now)
				rep = c
			}
		}
		// One shared object pass for the whole world, driven through any live member.
		// «Штурм» (DOTA) swaps the Hunt mob simulation for its lane-pusher pass
		// (creeps + cannons + win condition); everything else about the shared world
		// (per-member upkeep above, locking, disposal) is identical.
		if rep != nil {
			if inst.dota != nil {
				s.dotaTickLocked(rep, now)
			} else {
				s.tickMobsLocked(rep, now)
			}
		}
		inst.mu.Unlock()
	}
}
