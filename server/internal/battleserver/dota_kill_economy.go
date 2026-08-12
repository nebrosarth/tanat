package battleserver

import (
	"log"
	"math"
)

// dotaXPShareRadius is how close a living teammate must be to a kill to share its XP --
// the mechanical reason «впятером на линии невыгодно качаться»: XP DIVIDES among everyone
// in range instead of each hero drawing the kill's full value. Sized to comfortably cover
// one lane's width plus a support hanging a little behind the front line, without reaching
// into a neighbouring lane (dotaCreepAggro=13 is a creep's much shorter aggro pull).
const dotaXPShareRadius = 20.0

// dotaHeroKillCreditWindow is the recent-damage window used when a lane creep
// lands the lethal blow on an avatar. It is intentionally short: a hero that
// merely tagged an avatar long ago must not receive a later unrelated kill.
const dotaHeroKillCreditWindow = 8.0

// enemyHeroByObjIDLocked is a strict lookup. creditConnLocked intentionally falls
// back to the acting connection for unknown ids, which is correct for PvE unit
// ownership but would incorrectly turn a creep id into a hero for kill credit.
func (s *Server) enemyHeroByObjIDLocked(victim *conn, objID int32) *conn {
	if victim == nil || victim.inst == nil || objID == 0 {
		return nil
	}
	for _, mem := range victim.inst.members {
		if mem == nil || mem.huntState == nil || mem.objID != objID || !arenaEnemies(mem, victim) {
			continue
		}
		return mem
	}
	return nil
}

// noteHeroDamageLocked remembers the last enemy avatar that actually landed
// damage. It is called after mitigation/absorb, so blocked, dodged and fully
// absorbed hits cannot steal a later creep kill.
func (s *Server) noteHeroDamageLocked(victim, attacker *conn, now float64) {
	if victim == nil || victim.huntState == nil || attacker == nil ||
		!arenaEnemies(attacker, victim) {
		return
	}
	victim.huntState.lastHeroDamager = attacker.objID
	victim.huntState.lastHeroDamageAt = now
}

// recentHeroKillCreditLocked returns the enemy avatar whose recent landed hit
// should receive a creep-delivered hero kill. The lookup is revalidated against
// the current instance/team because bots can reconnect or leave while timers are
// still in flight.
func (s *Server) recentHeroKillCreditLocked(victim *conn, now float64) *conn {
	if victim == nil || victim.huntState == nil || victim.huntState.lastHeroDamager == 0 ||
		now-victim.huntState.lastHeroDamageAt > dotaHeroKillCreditWindow {
		return nil
	}
	return s.enemyHeroByObjIDLocked(victim, victim.huntState.lastHeroDamager)
}

// grantKillXPLocked awards a kill's XP either to the killer alone (Hunt/PvE, «Арена», or
// any world without a live «Штурм» simulation) or split EVENLY among every living teammate
// of killer's side within dotaXPShareRadius of (x,y) -- a real «Штурм» match. This is the
// mechanic behind "пять аватаров на одной линии невыгодно": the same wave gives each of
// them proportionally less. rep is any live member of the instance, used only to reach
// inst.members the same way every other *Locked helper does. Returns the sharing set (just
// [killer] outside a live «Штурм» sim) so a hero kill's caller (dotaCreditHeroKillLocked)
// can credit the SAME roster with assists -- one radius, one notion of "was in on this
// kill", not two that could quietly drift apart.
func (s *Server) grantKillXPLocked(rep *conn, killer *conn, xp float64, x, y float32) []*conn {
	return s.grantKillXPLockedMeta(rep, killer, xp, x, y, dotaXPGrantMeta{})
}

// dotaXPGrantMeta carries the source needed by XP telemetry. An empty source keeps the
// legacy Hunt/Arena call path quiet; Dota creep and avatar rewards opt in explicitly.
type dotaXPGrantMeta struct {
	source      string
	sourceID    int32
	victimLevel int32 // displayed level, not the internal zero-based level
}

func (s *Server) grantKillXPLockedMeta(rep *conn, killer *conn, xp float64, x, y float32, meta dotaXPGrantMeta) []*conn {
	if rep == nil || rep.inst == nil || rep.inst.dota == nil || killer == nil || killer.huntState == nil {
		s.grantXPLocked(killer, xp)
		return []*conn{killer}
	}
	sharing := s.grantDotaProximityXPLockedMeta(rep, killer.playerTeam(), xp, x, y, meta)
	if len(sharing) == 0 {
		sharing = []*conn{killer} // the credited killer's own kill always counts
		s.grantXPLocked(killer, xp)
		if meta.source == "creep" {
			s.recordBotCreepXPLocked(killer, float64(s.battleTime()))
		}
		s.telemetryRecordXPGrantLocked(killer, killer.playerTeam(), meta.source, meta.sourceID,
			xp, dotaReceivedXP(killer, xp), 1, meta.victimLevel, float64(s.battleTime()))
	}
	return sharing
}

// grantDotaProximityXPLocked awards a DOTA unit death's XP to nearby living heroes on
// team, independent of who delivered the last hit. Creep-vs-creep deaths use this helper;
// gold remains exclusive to a hero last hit on the existing player damage path.
func (s *Server) grantDotaProximityXPLocked(rep *conn, team int32, xp float64, x, y float32) []*conn {
	return s.grantDotaProximityXPLockedMeta(rep, team, xp, x, y, dotaXPGrantMeta{})
}

func (s *Server) grantDotaProximityXPLockedMeta(rep *conn, team int32, xp float64, x, y float32, meta dotaXPGrantMeta) []*conn {
	if rep == nil || rep.inst == nil || rep.inst.dota == nil || xp <= 0 {
		return nil
	}
	now := float64(s.battleTime())
	var sharing []*conn
	for _, mem := range rep.inst.members {
		hs := mem.huntState
		if hs == nil || hs.deadUntil > 0 || mem.playerTeam() != team {
			continue
		}
		mx, my := mem.posAtLocked(float32(now))
		if math.Hypot(float64(mx-x), float64(my-y)) <= dotaXPShareRadius {
			sharing = append(sharing, mem)
		}
	}
	if len(sharing) == 0 {
		if meta.source == "creep" && rep.inst.dota != nil &&
			now-rep.inst.dota.startedAt <= dotaEarlyCreepXPMissWindow {
			rep.inst.dota.earlyCreepXPMisses++
			if rep.inst.dota.telemetry != nil {
				rep.inst.dota.telemetry.record(telemetryCreepXPMiss{
					telemetryEvent: newTelemetryEvent("creep_xp_miss", rep.inst.dota.telemetryMatchTimeLocked(now)),
					CreepID:        meta.sourceID, Team: team, X: x, Y: y,
				})
			}
		}
		return nil
	}
	share := xp / float64(len(sharing))
	for _, mem := range sharing {
		s.grantXPLocked(mem, share)
		if meta.source == "creep" {
			s.recordBotCreepXPLocked(mem, now)
		}
		s.telemetryRecordXPGrantLocked(mem, team, meta.source, meta.sourceID, xp,
			dotaReceivedXP(mem, share), len(sharing), meta.victimLevel, now)
	}
	return sharing
}

// recordBotCreepXPLocked updates only battle-local bot farm state. It is fed by
// the authoritative XP grant path, so farm debt measures real received creep
// XP rather than a bot's intended target or a movement decision.
func (s *Server) recordBotCreepXPLocked(c *conn, now float64) {
	if c == nil || c.inst == nil {
		return
	}
	if brain := c.inst.bots[c.objID]; brain != nil {
		brain.farmXPEvents++
		brain.farmLastXPTAt = now
	}
}

func dotaReceivedXP(c *conn, xp float64) float64 {
	if c != nil && c.buffXPMult > 0 {
		return xp * c.buffXPMult
	}
	return xp
}

// dotaHeroKillBaseXP/dotaHeroKillPerLevelXP/dotaHeroKillBaseCoins/dotaHeroKillPerLevelCoins
// size a «Штурм» hero-kill bounty against the calibrated creep economy: one complete lane
// wave (3 melee + 1 ranged) pays 300 XP. A hero kill starts at 120 XP and increases with
// the victim's level, so a kill is material without eclipsing an uncontested farming lane.
// Gold is NOT split like XP is -- see grantKillXPLocked's own doc comment for why XP is --
// a kill's gold goes to the credited killer alone, same as any Hunt kill.
const (
	dotaHeroKillBaseXP        = 120.0
	dotaHeroKillPerLevelXP    = 8.0
	dotaHeroKillBaseCoins     = int32(40)
	dotaHeroKillPerLevelCoins = int32(6)
)

func dotaHeroKillXP(victimLevel int32) float64 {
	if victimLevel < 0 {
		victimLevel = 0
	}
	return dotaHeroKillBaseXP + dotaHeroKillPerLevelXP*float64(victimLevel)
}

// dotaCreditHeroKillLocked is «Штурм»'s hero-kill bounty: unlike «Арена»
// (arenaCreditKillLocked) there is no frag counter and no win condition here (only an
// altar ends a «Штурм» match) -- just the XP/gold a real Dota-like kill should pay, which
// was entirely missing before this (a «Штурм» hero kill paid nothing at all:
// arenaCreditKillLocked no-ops outside inst.arena, and this was the only caller).
func (s *Server) dotaCreditHeroKillLocked(killer, victim *conn, now float64) {
	inst := killer.inst
	if inst == nil || inst.dota == nil || victim.huntState == nil {
		return
	}
	lvl := victim.huntState.level
	xp := dotaHeroKillXP(lvl)
	coins := dotaHeroKillBaseCoins + dotaHeroKillPerLevelCoins*lvl
	kx, ky := killer.posAtLocked(float32(now))
	sharing := s.grantKillXPLockedMeta(killer, killer, xp, kx, ky, dotaXPGrantMeta{
		source: "avatar", sourceID: victim.objID, victimLevel: lvl + 1,
	})
	s.awardCoinsLocked(killer, victim.objID, coins)
	// The scoreboard's AvatarKills/Assists (fight|log -- see rating.go): the credited
	// killer gets the kill, and every OTHER teammate who shared the kill's XP (same
	// sharing set, so "was in on this kill" means one consistent thing) gets an assist.
	if killer.huntState != nil {
		killer.huntState.frags++
		s.broadcastPlayerStatsLocked(killer)
		s.publishLiveFightLogLocked(inst)
	}
	for _, mem := range sharing {
		if mem != killer && mem.huntState != nil {
			mem.huntState.assists++
			s.broadcastPlayerStatsLocked(mem)
			s.publishLiveFightLogLocked(inst)
		}
	}
	log.Printf("battle: «Штурм» room=%d hero kill: %d killed %d (xp=%.0f coins=%d)",
		inst.id, killer.objID, victim.objID, xp, coins)
}
