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
	if rep == nil || rep.inst == nil || rep.inst.dota == nil || killer == nil || killer.huntState == nil {
		s.grantXPLocked(killer, xp)
		return []*conn{killer}
	}
	now := float64(s.battleTime())
	team := killer.playerTeam()
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
		sharing = []*conn{killer} // the killer's own kill always counts, even at a radius edge case
	}
	share := xp / float64(len(sharing))
	for _, mem := range sharing {
		s.grantXPLocked(mem, share)
	}
	return sharing
}

// dotaHeroKillBaseXP/dotaHeroKillPerLevelXP/dotaHeroKillBaseCoins/dotaHeroKillPerLevelCoins
// size a «Штурм» hero-kill bounty against the creep economy it competes with: one lane wave
// (4 creeps, mostly the 12XP/4-coin melee troop) pays ~50XP/17 coins, so a lone hero kill at
// level 0 (60XP/40 coins) already beats the wave it likely interrupted, and both grow with
// the VICTIM's level (a Dota staple: picking off a fed/high-level enemy pays more than a
// fresh spawn). Gold is NOT split like XP is -- see grantKillXPLocked's own doc comment for
// why XP is -- a kill's gold goes to the credited killer alone, same as any Hunt kill.
const (
	dotaHeroKillBaseXP        = 60.0
	dotaHeroKillPerLevelXP    = 10.0
	dotaHeroKillBaseCoins     = int32(40)
	dotaHeroKillPerLevelCoins = int32(6)
)

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
	lvl := float64(victim.huntState.level)
	xp := dotaHeroKillBaseXP + dotaHeroKillPerLevelXP*lvl
	coins := dotaHeroKillBaseCoins + dotaHeroKillPerLevelCoins*int32(lvl)
	kx, ky := killer.posAtLocked(float32(now))
	sharing := s.grantKillXPLocked(killer, killer, xp, kx, ky)
	s.awardCoinsLocked(killer, victim.objID, coins)
	// The scoreboard's AvatarKills/Assists (fight|log -- see rating.go): the credited
	// killer gets the kill, and every OTHER teammate who shared the kill's XP (same
	// sharing set, so "was in on this kill" means one consistent thing) gets an assist.
	if killer.huntState != nil {
		killer.huntState.frags++
	}
	for _, mem := range sharing {
		if mem != killer && mem.huntState != nil {
			mem.huntState.assists++
		}
	}
	log.Printf("battle: «Штурм» room=%d hero kill: %d killed %d (xp=%.0f coins=%d)",
		inst.id, killer.objID, victim.objID, xp, coins)
}
