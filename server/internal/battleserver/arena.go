package battleserver

import (
	"log"
	"math"
	"math/rand"
	"time"

	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
)

// «Арена» (MapType.DM, map_0_0) is the player-versus-player deathmatch. It reuses the
// shared hunt world wholesale -- the same instance, ticker, movement, death/respawn and
// avatar rendering -- and adds only what PvE never needed: two hostile teams, an avatar
// that can auto-attack another avatar, a frag count and a win condition. There are no
// mobs, so the shared mob pass is a no-op (inst.mobs is empty); the win check runs in its
// place. See instance.go for where the tick loop dispatches here.

// arenaState is the per-instance «Арена» simulation, hung on huntInstance.arena.
type arenaState struct {
	m         gamedata.ArenaMap
	frags     map[int32]int32 // team -> kills by that team
	fragLimit int32           // team frags to win (0 = endless)
	ended     bool
	winner    int32
	nextTeam  int32 // alternates as players join, so sides stay balanced
}

// newArenaInstance builds a deathmatch world: a huntInstance with an arenaState and no
// mobs. The id-space bases match newHuntInstance so summons/fx/drops never collide.
func newArenaInstance(s *Server, id, mapID int32) *huntInstance {
	m, _ := gamedata.ArenaMapByID(mapID)
	inst := &huntInstance{
		s:              s,
		id:             id,
		mapID:          mapID,
		nav:            m.Nav,
		mobs:           map[int32]*mobState{},
		members:        map[int32]*conn{},
		nextFxUID:      1 << 20,
		nextSummonID:   300000,
		nextAnchorID:   400000,
		drops:          map[int32]*dropState{},
		nextDropID:     dropChestBaseID,
		nextDropItemID: dropItemBaseID,
	}
	inst.arena = &arenaState{
		m:         m,
		frags:     map[int32]int32{},
		fragLimit: m.FragLimit,
		nextTeam:  gamedata.ArenaTeamA,
	}
	return inst
}

// assignArenaTeamLocked places a joining player on a side, alternating A/B so an even
// number of players splits evenly. Caller holds inst.mu.
func (a *arenaState) assignTeam() int32 {
	t := a.nextTeam
	if a.nextTeam == gamedata.ArenaTeamA {
		a.nextTeam = gamedata.ArenaTeamB
	} else {
		a.nextTeam = gamedata.ArenaTeamA
	}
	return t
}

// arenaSpawnLocked chooses c's (re)spawn point: the marker farthest from c's nearest
// living enemy, so nobody materialises on top of an opponent and the two sides start
// apart. With no living enemies (a solo launch, or everyone dead) it falls back to the
// point farthest from the map centre-ish first marker, which is deterministic and fine.
// Caller holds inst.mu.
func (s *Server) arenaSpawnLocked(inst *huntInstance, c *conn, now float64) (float32, float32) {
	a := inst.arena
	best := a.m.Spawns[0]
	bestScore := -1.0
	for _, sp := range a.m.Spawns {
		// Score a candidate by its distance to the CLOSEST living enemy: a high score means
		// even the nearest opponent is far, which is what we want.
		nearest := math.Inf(1)
		for _, mem := range inst.members {
			if mem == c || mem.huntState == nil || !arenaEnemies(c, mem) {
				continue
			}
			if mem.huntState.deadUntil > 0 {
				continue
			}
			mx, my := mem.posAtLocked(float32(now))
			if d := math.Hypot(sp.X-float64(mx), sp.Y-float64(my)); d < nearest {
				nearest = d
			}
		}
		if nearest > bestScore {
			bestScore = nearest
			best = sp
		}
	}
	return float32(best.X), float32(best.Y)
}

// arenaInitialSpawn is a joining player's FIRST spawn, before anyone has moved: side A
// takes the first marker, side B the marker farthest from it, so the two sides start
// apart. Every later respawn uses arenaSpawnLocked, which scans live enemies instead.
func arenaInitialSpawn(a *arenaState, team int32) (float64, float64) {
	sp := a.m.Spawns
	if team == gamedata.ArenaTeamA || len(sp) == 1 {
		return sp[0].X, sp[0].Y
	}
	far, best := sp[0], -1.0
	for _, p := range sp[1:] {
		if d := math.Hypot(p.X-sp[0].X, p.Y-sp[0].Y); d > best {
			best, far = d, p
		}
	}
	return far.X, far.Y
}

// arenaCreditKillLocked records that killer's side scored a frag on victim and ends the
// match if that reaches the frag limit. Caller holds inst.mu.
func (s *Server) arenaCreditKillLocked(killer, victim *conn, now float64) {
	inst := killer.inst
	if inst == nil || inst.arena == nil || inst.arena.ended {
		return
	}
	a := inst.arena
	kh := killer.huntState
	if kh != nil {
		kh.frags++
	}
	team := killer.playerTeam()
	a.frags[team]++
	log.Printf("battle: «Арена» room=%d frag: team %d killed %d (team total %d/%d)",
		inst.id, team, victim.objID, a.frags[team], a.fragLimit)
	if a.fragLimit > 0 && a.frags[team] >= a.fragLimit {
		s.arenaEndLocked(inst, team, now)
	}
}

// arenaEndLocked marks the match ended and broadcasts BATTLE_END{id: winner team} to
// every member, exactly as «Штурм» does. Caller holds inst.mu.
func (s *Server) arenaEndLocked(inst *huntInstance, winner int32, now float64) {
	a := inst.arena
	if a.ended {
		return
	}
	a.ended = true
	a.winner = winner
	log.Printf("battle: «Арена» room=%d ended, winner team=%d", inst.id, winner)
	for _, mem := range inst.members {
		s.push(mem, battleproto.CmdBattleEnd, amf.NewArray().Set("id", winner))
	}
}

// ---- PvP auto-attack: the twin of hunt.go's mob attack loop, aimed at an avatar ----

// startPvpAttackLocked begins auto-attacking an enemy avatar. Mirrors startAttackLocked
// but the target is a conn, tracked in hs.pvpTarget instead of hs.attackTarget.
func (s *Server) startPvpAttackLocked(c *conn, victim *conn) {
	hs := c.huntState
	if hs.deadUntil > 0 {
		return
	}
	s.cancelOrderLocked(c)
	hs.attackTarget = 0
	hs.pvpTarget = victim.objID
	c.resetChaseLocked()
	hs.attackSeq++
	s.armPvpAttackTimer(c, hs.attackSeq, victim.objID, 0,
		time.Duration(float64(time.Second)/s.attackPeriodLocked(hs)))
}

// armPvpAttackTimer is the PvP attack tick: chase the enemy avatar, and once in reach
// play the swing ACTION and schedule the hit, re-arming on the attacker's cadence. A
// close twin of armAttackTimer; the differences are that the target is resolved from
// inst.members (not hs.mobs) and its liveness is deadUntil, not a dead flag.
func (s *Server) armPvpAttackTimer(c *conn, seq int, targetID int32, delay, interval time.Duration) {
	time.AfterFunc(delay, func() {
		c.lock()
		defer c.unlock()
		hs := c.huntState
		if hs == nil || hs.closed || hs.attackSeq != seq || hs.deadUntil > 0 {
			return
		}
		victim := c.arenaMember(targetID)
		if victim == nil || !arenaEnemies(c, victim) || victim.huntState.deadUntil > 0 {
			s.stopAttackLocked(c, false)
			return
		}
		now := s.battleTime()
		cx, cy := c.posAtLocked(now)
		vx, vy := victim.posAtLocked(now)
		reach := hs.effAttackRangeLocked(float64(now)) + hs.av.Radius() + victim.huntState.av.Radius()
		if math.Hypot(float64(vx-cx), float64(vy-cy)) > reach {
			c.chaseMoveLocked(s, vx, vy)
			s.armPvpAttackTimer(c, seq, targetID, 250*time.Millisecond, interval)
			return
		}
		if c.vx != 0 || c.vy != 0 {
			c.stopArrivalLocked()
			c.hasDest = false
			c.x, c.y, c.vx, c.vy, c.snapT = cx, cy, 0, 0, now
			c.sendPosLocked(s, cx, cy, 0, 0, now)
		}
		actionArgs := newActionArgs(c.objID, attackProtoID(hs.av), targetID, float64(now),
			amf.NewArray().Set("x", 0.0).Set("y", 0.0))
		s.pushAvatarAllLocked(c, battleproto.CmdAction, actionArgs)
		if hs.hasProjectile {
			release := time.Duration(hs.av.AttackWindup * float64(interval))
			s.schedulePvpProjectileLocked(c, seq, targetID, release)
		} else {
			s.schedulePvpHitLocked(c, seq, targetID, interval/2)
		}
		s.scheduleSwingDone(c, seq, interval)
		s.armPvpAttackTimer(c, seq, targetID, interval, interval)
	})
}

// schedulePvpProjectileLocked flies a ranged avatar's basic-attack bolt at an enemy
// avatar, landing the hit on arrival. Mirrors scheduleProjectileLocked's avatar path.
func (s *Server) schedulePvpProjectileLocked(c *conn, seq int, targetID int32, release time.Duration) {
	time.AfterFunc(release, func() {
		c.lock()
		defer c.unlock()
		hs := c.huntState
		if hs == nil || hs.closed || hs.attackSeq != seq || hs.deadUntil > 0 {
			return
		}
		victim := c.arenaMember(targetID)
		if victim == nil || victim.huntState.deadUntil > 0 {
			return
		}
		now := s.battleTime()
		cx, cy := c.posAtLocked(now)
		vx, vy := victim.posAtLocked(now)
		flight := math.Hypot(float64(vx-cx), float64(vy-cy))/24 + 0.1
		s.pushAvatarAllLocked(c, battleproto.CmdSetProjectile, amf.NewArray().
			Set("source", c.objID).
			Set("target", targetID).
			Set("hit_at", float64(now)+flight))
		// The bolt is loosed and committed: land it on arrival regardless of a later
		// cancel/retarget (seq unchecked), like the PvE projectile hit.
		s.schedulePvpHitAfterLocked(c, 0, targetID, time.Duration(flight*float64(time.Second)), true)
	})
}

func (s *Server) schedulePvpHitLocked(c *conn, seq int, targetID int32, windup time.Duration) {
	s.schedulePvpHitAfterLocked(c, seq, targetID, windup, false)
}

// schedulePvpHitAfterLocked applies a landed basic-attack blow to an enemy avatar,
// reusing the exact avatar damage math the PvE path uses (dmg roll, dmg_pct, power, crit,
// lifesteal). committed=true skips the seq check for an in-flight projectile.
func (s *Server) schedulePvpHitAfterLocked(c *conn, seq int, targetID int32, windup time.Duration, committed bool) {
	time.AfterFunc(windup, func() {
		c.lock()
		defer c.unlock()
		hs := c.huntState
		if hs == nil || hs.closed {
			return
		}
		if !committed && hs.attackSeq != seq {
			return
		}
		victim := c.arenaMember(targetID)
		if victim == nil || victim.huntState.deadUntil > 0 {
			return
		}
		now := s.battleTime()
		av := hs.av
		flat := hs.st.modSum(float64(now), "dmg_flat") // +attack from avatar tree items
		dmg := (float64(av.DmgMin) + flat + rand.Float64()*float64(av.DmgMax-av.DmgMin)) *
			hs.st.modMul(float64(now), "dmg_pct") * hs.powerMul()
		if crit := hs.st.modSum(float64(now), "crit_pct"); crit > 0 && rand.Float64() < crit {
			dmg *= 1.5 + hs.st.modSum(float64(now), "crit_dmg_pct")
		}
		s.hitPlayerFromLocked(victim, c.objID, dmg, float64(now), nil, c)
		if ls := hs.st.modSum(float64(now), "lifesteal_pct"); ls > 0 {
			s.healPlayerLocked(c, dmg*ls)
		}
	})
}

// arenaMember returns the enemy avatar this player is targeting if it is still a live
// member of the same arena, else nil (left, or not an arena at all).
func (c *conn) arenaMember(objID int32) *conn {
	if c.inst == nil || c.inst.arena == nil {
		return nil
	}
	m := c.inst.members[objID]
	if m == nil || m.huntState == nil {
		return nil
	}
	return m
}
