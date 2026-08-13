package battleserver

import "math"

// botAI30Profile is the aggressive scripted teacher/opponent for AI-40. It
// inherits the complete AI-20 team and XP director, then changes local
// execution: actively clear a safely covered wave, value ready combat tools,
// and abandon a chase as soon as its live safety conditions disappear.
type botAI30Profile struct{}

func (botAI30Profile) Version() int               { return 30 }
func (botAI30Profile) UsesTeamOrchestrator() bool { return true }
func (botAI30Profile) UsesFarmSafeWave() bool     { return true }
func (botAI30Profile) UsesFarmLanePlan() bool     { return true }
func (botAI30Profile) UsesPlanHysteresis() bool   { return true }
func (botAI30Profile) UsesFarmRescue() bool       { return true }
func (botAI30Profile) UsesFarmRotation() bool     { return true }
func (botAI30Profile) UsesFarmStability() bool    { return true }
func (botAI30Profile) UsesFarmDebt() bool         { return true }

const (
	botAI30FarmMinHP       = 0.72
	botAI30ChaseMinHP      = 0.58
	botAI30WaveCoverRadius = 9.0
)

func botUsesAI30(b *botBrain) bool { return botAIVersionForBrain(b) == 30 }

// botAI30FarmAttackPointLocked returns a home-side point from which the hero
// can attack target. A friendly creep must already be tanking the wave, and
// hero/structure pressure vetoes the step. The point remains inside XP range;
// AI-30 helps kill the wave but never sacrifices proximity coverage for gold.
func (s *Server) botAI30FarmAttackPointLocked(b *botBrain, target *mobState, now float64) (float32, float32, bool) {
	if b == nil || b.c == nil || b.c.huntState == nil || target == nil || target.dead || target.structure ||
		!target.enemyOf(b.c.playerTeam()) || !botUsesAI30(b) ||
		botHPFrac(b.c.huntState, now) < botAI30FarmMinHP || s.botIncomingPressureLocked(b, now) ||
		botNearbyEnemyHeroPressureLocked(b, now) > 0 {
		return 0, 0, false
	}
	covered := false
	targetLane := botLaneForCreep(b.c, target)
	for _, mob := range botSortedMobs(b.c.inst) {
		if mob == nil || mob.dead || mob.structure || mob.team != b.c.playerTeam() {
			continue
		}
		if lane := botLaneForCreep(b.c, mob); targetLane >= 0 && lane >= 0 && lane != targetLane {
			continue
		}
		if math.Hypot(float64(mob.x-target.x), float64(mob.y-target.y)) <= botAI30WaveCoverRadius {
			covered = true
			break
		}
	}
	if !covered {
		return 0, 0, false
	}
	homeX, homeY := botHomeLocked(b.c)
	dx, dy := float64(homeX-target.x), float64(homeY-target.y)
	d := math.Hypot(dx, dy)
	if d < 0.01 {
		return 0, 0, false
	}
	reach := b.c.huntState.effAttackRangeLocked(now) + b.c.huntState.av.Radius() + target.mob.Radius()
	if reach <= 0 {
		reach = 2.2
	}
	standOff := math.Max(1.0, math.Min(reach*0.82, dotaXPShareRadius*0.82))
	x := target.x + float32(dx/d*standOff)
	y := target.y + float32(dy/d*standOff)
	if math.Hypot(float64(x-target.x), float64(y-target.y)) > dotaXPShareRadius ||
		s.botEnemyStructureDangerLocked(b.c, x, y) {
		return 0, 0, false
	}
	return x, y, true
}

// botAI30ReadyPowerLocked adds only currently usable combat tools to the
// ordinary HP/level comparison. This makes the trade test react to cooldowns
// and mana rather than treating two equal-level heroes as permanently equal.
func (s *Server) botAI30ReadyPowerLocked(c *conn, now float64) float64 {
	if c == nil || c.huntState == nil || c.huntState.deadUntil > 0 {
		return 0
	}
	hs := c.huntState
	power := 0.0
	for slot := 1; slot <= 4; slot++ {
		if !s.botAbilityReadyLocked(hs, slot, now) {
			continue
		}
		switch priority := botOffensiveOpPriority(hs.skillDef(slot)); priority {
		case 4:
			power += 0.16
		case 3:
			power += 0.12
		case 2:
			power += 0.08
		}
	}
	return power
}

// botAI30PreferredTargetLocked ranks visible, locally reachable heroes. Active
// attackers and wounded valuable roles beat a merely closer full-HP target.
func (s *Server) botAI30PreferredTargetLocked(b *botBrain, enemies []*conn, now float64) *conn {
	if b == nil || b.c == nil || !botUsesAI30(b) {
		return nil
	}
	cx, cy := b.c.posAtLocked(float32(now))
	var best *conn
	bestScore := math.Inf(-1)
	for _, enemy := range enemies {
		if enemy == nil || enemy.huntState == nil || enemy.huntState.deadUntil > 0 {
			continue
		}
		ex, ey := enemy.posAtLocked(float32(now))
		distance := math.Hypot(float64(ex-cx), float64(ey-cy))
		if distance > botEngageRadius || s.botEnemyStructureDangerLocked(b.c, ex, ey) {
			continue
		}
		score := (1-botHPFrac(enemy.huntState, now))*48 - distance*1.8
		score += float64(botEnemyRolePriority(enemy.huntState.av.Type)) * 4
		if enemy.huntState.pvpTarget == b.c.objID || enemy.huntState.attackTarget == b.c.objID {
			score += 36
		}
		if score > bestScore || (score == bestScore && (best == nil || enemy.objID < best.objID)) {
			best, bestScore = enemy, score
		}
	}
	return best
}

func (s *Server) botAI30ChaseSafeLocked(b *botBrain, target *conn, now float64) bool {
	if b == nil || b.c == nil || target == nil || target.huntState == nil || target.huntState.deadUntil > 0 {
		return false
	}
	if !botUsesAI30(b) {
		return true
	}
	tx, ty := target.posAtLocked(float32(now))
	cx, cy := b.c.posAtLocked(float32(now))
	return botHPFrac(b.c.huntState, now) >= botAI30ChaseMinHP &&
		math.Hypot(float64(tx-cx), float64(ty-cy)) <= botEngageRadius*1.25 &&
		!s.botEnemyStructureDangerLocked(b.c, tx, ty)
}
