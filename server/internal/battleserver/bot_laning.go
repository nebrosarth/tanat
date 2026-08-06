package battleserver

// Laning-phase decisions: where to stand, what to hit, when to nuke a creep wave. This is
// deliberately the bulk of a bot's early game, matching real Dota: most of a match is
// spent farming/trading on a lane, not teamfighting (bot_tactics.go).

import (
	"math"

	"tanatserver/internal/gamedata"
)

// botLaneEngageRadius is how far from itself a laning bot will pick a fight (creep or,
// with friendly creep cover, a structure) -- roughly a creep's own aggro pull
// (dotaCreepAggro=13), so a bot holds its lane instead of wandering into the next one.
const botLaneEngageRadius = 14.0

// botAbilityReadyLocked reports whether slot is learned, off cooldown and affordable.
func (s *Server) botAbilityReadyLocked(hs *huntState, slot int, now float64) bool {
	level := int(hs.skillLevel[slot-1])
	if level < 1 || now < hs.cooldownUntil[slot-1] {
		return false
	}
	def := hs.skillDef(slot)
	if def.Type != "ACTIVE" {
		return false
	}
	cost := skillManaCost(float64(def.ManaCost[level-1]))
	return hs.mana >= cost
}

// botFindLastHitLocked returns an enemy creep already in attack range that will die to
// this bot's next swing (a 15% slack margin covers roll variance/regen), the highest-value
// laning decision: never skip a free last hit for a slower kill elsewhere.
func (s *Server) botFindLastHitLocked(b *botBrain, now float64) *mobState {
	c, hs := b.c, b.c.huntState
	cx, cy := c.posAtLocked(float32(now))
	reach := hs.effAttackRangeLocked(now)
	if reach <= 0 {
		reach = 2.2 // melee fallback, matches dotaMeleeReach's scale
	}
	reach += hs.av.Radius()
	myDmg := hs.baseAttackLocked(now)
	if myDmg <= 0 {
		return nil
	}
	var best *mobState
	bestHP := math.Inf(1)
	for _, m := range hs.mobs {
		if m.dead || m.structure || !m.enemyOf(c.playerTeam()) {
			continue
		}
		d := math.Hypot(float64(m.x-cx), float64(m.y-cy)) - m.mob.Radius()
		if d > reach || m.hp > myDmg*1.15 {
			continue
		}
		if m.hp < bestHP {
			bestHP, best = m.hp, m
		}
	}
	return best
}

// botFindLaneTargetLocked is the laning attack-target picker: a guaranteed last hit first,
// else the nearest enemy creep worth trading with, else an enemy structure in reach --
// gated on requireCover (our own creep wave standing here to soak its aggro) for a solo
// laner, ungated for a grouped push (the party itself is the aggro soak).
func (s *Server) botFindLaneTargetLocked(b *botBrain, now float64, radius float64, requireCover bool) *mobState {
	if last := s.botFindLastHitLocked(b, now); last != nil {
		return last
	}
	c, hs := b.c, b.c.huntState
	cx, cy := c.posAtLocked(float32(now))

	var bestCreep *mobState
	bestCreepD := math.Inf(1)
	ownCreepNearby := false
	for _, m := range hs.mobs {
		if m.dead {
			continue
		}
		d := math.Hypot(float64(m.x-cx), float64(m.y-cy))
		if !m.structure && m.team == c.playerTeam() && d <= radius {
			ownCreepNearby = true
		}
		if m.structure || !m.enemyOf(c.playerTeam()) || d > radius {
			continue
		}
		if d < bestCreepD {
			bestCreepD, bestCreep = d, m
		}
	}
	if bestCreep != nil {
		return bestCreep
	}
	if requireCover && !ownCreepNearby {
		return nil // no creep of ours to soak tower aggro -- don't dive alone
	}
	var bestStruct *mobState
	bestStructD := math.Inf(1)
	for _, m := range hs.mobs {
		if m.dead || !m.structure || !m.enemyOf(c.playerTeam()) || c.altarShieldedLocked(m) {
			continue
		}
		d := math.Hypot(float64(m.x-cx), float64(m.y-cy))
		if d <= radius && d < bestStructD {
			bestStructD, bestStruct = d, m
		}
	}
	return bestStruct
}

// botConsiderWaveClearAbilityLocked casts the first ready, affordable AoE damage ability
// at the enemy creep cluster it hits hardest, when at least 2 enemies are caught -- a
// simple but real "use abilities when advantageous" rule for the farming game, distinct
// from bot_combat.go's hero-fight ability logic.
func (s *Server) botConsiderWaveClearAbilityLocked(b *botBrain, now float64) {
	c, hs := b.c, b.c.huntState
	cx, cy := c.posAtLocked(float32(now))
	for slot := 1; slot <= 4; slot++ {
		if !s.botAbilityReadyLocked(hs, slot, now) {
			continue
		}
		def := hs.skillDef(slot)
		if def.AoERadius <= 0 || !botSkillHasOp(def, gamedata.OpDamage) {
			continue // single-target actives are saved for hero fights
		}
		dist := float64(def.Distance)
		if dist <= 0 {
			dist = 6
		}
		bestX, bestY, bestCount := float32(0), float32(0), 0
		for _, m := range hs.mobs {
			if m.dead || m.structure || !m.enemyOf(c.playerTeam()) {
				continue
			}
			if math.Hypot(float64(m.x-cx), float64(m.y-cy)) > dist {
				continue
			}
			n := 0
			for _, o := range hs.mobs {
				if !o.dead && !o.structure && o.enemyOf(c.playerTeam()) &&
					math.Hypot(float64(o.x-m.x), float64(o.y-m.y)) <= float64(def.AoERadius) {
					n++
				}
			}
			if n > bestCount {
				bestCount, bestX, bestY = n, m.x, m.y
			}
		}
		if bestCount >= 2 {
			s.startSkillOrderLocked(c, slot, 0, bestX, bestY, true)
			return
		}
	}
}

// botSkillHasOp reports whether a skill's op list includes at least one op of kind k.
func botSkillHasOp(def gamedata.Skill, k gamedata.OpKind) bool {
	for _, op := range def.Ops {
		if op.Kind == k {
			return true
		}
	}
	return false
}

// botLanePoint is where a laning bot should stand: the front of its own team's creep wave
// on its assigned lane (the honest "where the lane currently stands" signal already
// driving the creep AI itself), or, before the first wave spawns, a fixed spot partway
// down the lane from home.
func (s *Server) botLanePoint(b *botBrain, now float64) (float32, float32) {
	c := b.c
	d := c.inst.dota
	if b.lane < 0 || b.lane >= len(d.m.Lanes) {
		return botHomeLocked(c)
	}
	lane := d.m.Lanes[b.lane]
	if len(lane) == 0 {
		return botHomeLocked(c)
	}
	if fx, fy, ok := botLaneFrontLocked(c, lane); ok {
		return fx, fy
	}
	fwd := c.playerTeam() == dotaTeamHuman
	idx := 2
	if !fwd {
		idx = len(lane) - 1 - 2
	}
	if idx < 0 {
		idx = 0
	}
	if idx >= len(lane) {
		idx = len(lane) - 1
	}
	return float32(lane[idx].X), float32(lane[idx].Y)
}

// botLaneFrontLocked finds the most-advanced living creep of c's own side on `lane`
// (matched by its first waypoint, which is unique per lane on the map) and returns its
// position -- "advanced" means highest laneIdx marching forward, lowest marching backward.
func botLaneFrontLocked(c *conn, lane []gamedata.Vec2) (float32, float32, bool) {
	team := c.playerTeam()
	fwd := team == dotaTeamHuman
	var best *mobState
	bestIdx := -1
	if !fwd {
		bestIdx = len(lane) + 1
	}
	for _, m := range c.inst.mobs {
		if m.dead || m.structure || m.team != team || len(m.lane) == 0 {
			continue
		}
		if len(m.lane) != len(lane) || m.lane[0] != lane[0] {
			continue
		}
		if (fwd && m.laneIdx > bestIdx) || (!fwd && m.laneIdx < bestIdx) {
			bestIdx, best = m.laneIdx, m
		}
	}
	if best == nil {
		return 0, 0, false
	}
	return best.x, best.y, true
}

// botLaneTickLocked is the default (early-game) decision pass: hold/retreat, clear waves
// with abilities when it pays off, last-hit or trade, else walk to the lane.
func (s *Server) botLaneTickLocked(b *botBrain, now float64) {
	c, hs := b.c, b.c.huntState
	if s.botShouldRetreatLocked(b, now) {
		hx, hy := botHomeLocked(c)
		s.botMoveTowardLocked(b, hx, hy, now)
		return
	}
	if s.botConsiderHealLocked(b, now) {
		return
	}
	s.botConsiderWaveClearAbilityLocked(b, now)
	if hs.attackTarget != 0 {
		return // already swinging at something; let the timer chain run
	}
	if target := s.botFindLaneTargetLocked(b, now, botLaneEngageRadius, true); target != nil {
		s.startAttackLocked(c, target)
		return
	}
	lx, ly := s.botLanePoint(b, now)
	s.botMoveTowardLocked(b, lx, ly, now)
}
