package battleserver

// Mid/late-game tactics: roaming toward wherever teammates already are, and once enough
// of the team is grouped up, pushing the lane the group is actually standing on together
// -- structures included, since the party itself (not a creep wave) is the tower-aggro
// soak. botUpdatePhaseLocked (bot_brain.go) decides which of these runs each tick.

import "math"

func botStrategicAllyLocked(c *conn, mem *conn) bool {
	if c == nil || c.inst == nil || mem == nil {
		return false
	}
	brain, isBot := c.inst.bots[mem.objID]
	return !isBot || brain == nil || !brain.retreating
}

// botGroupEngageRadius is the group phase's attack/push search radius -- wider than solo
// laning (botLaneEngageRadius) since a grouped push commits harder.
const botGroupEngageRadius = 20.0

// botNearestAllyLocked returns the closest living teammate, or nil if alone.
func (s *Server) botNearestAllyLocked(b *botBrain, now float64) *conn {
	c := b.c
	cx, cy := c.posAtLocked(float32(now))
	var best *conn
	bestD := math.Inf(1)
	for _, mem := range botLivingAllies(c) {
		mx, my := mem.posAtLocked(float32(now))
		if d := math.Hypot(float64(mx-cx), float64(my-cy)); d < bestD {
			bestD, best = d, mem
		}
	}
	return best
}

// botIsActiveLocked reports whether ally is currently doing something worth following --
// mid-attack or mid-PvP-engagement. Without this, botRoamTickLocked's "chase the nearest
// ally" fallback treats an equally idle teammate as just as valid a destination as an
// engaged one: two bots with nothing to do converge on each other, satisfy each other's
// botUpdatePhaseLocked "grouped" check, and anchor there -- see the fix below.
func botIsActiveLocked(ally *conn) bool {
	hs := ally.huntState
	return hs != nil && (hs.attackTarget != 0 || hs.pvpTarget != 0)
}

// botRoamTickLocked: a pair (or a laner between waves) drifts toward wherever the rest of
// the team already is, still farming/fighting anything it passes along the way.
//
// The ally-chase fallback below used to be unconditional: once within botMoveTowardLocked's
// own "already there" threshold (dist2<=2*2) of its nearest ally, the move became a
// permanent no-op every subsequent think tick, and mutual proximity is exactly the
// condition that put both bots into roam phase in the first place -- so any isolated pair
// with nothing to fight froze in place indefinitely (measured: up to 72% of a match, never
// leveling). Now the ally is only followed while doing so is an actual move AND the ally
// itself is doing something worth following; otherwise this falls through to a genuine
// redirect toward wherever the team as a whole still is (not just this bot's own,
// possibly-exhausted lane).
func (s *Server) botRoamTickLocked(b *botBrain, now float64) {
	c, hs := b.c, b.c.huntState
	if s.botConsiderHealLocked(b, now) {
		return
	}
	if s.botShouldRetreatLocked(b, now) {
		hx, hy := s.botRetreatPointLocked(b, now)
		s.botMoveTowardLocked(b, hx, hy, now)
		return
	}
	if b.macroAssignment.Reason == botMacroReasonMobilizationPreparation &&
		(hs.attackTarget != 0 || (hs.attackActionActive && hs.pvpTarget == 0)) {
		// A preparation transition can happen immediately after a previous
		// full-mobilization attack. Clear that stale structure swing before the
		// ordinary attack-state guard, otherwise the bot remains frozen on the
		// old objective and can never walk to the staging point.
		s.stopAttackLocked(c, false)
	}
	if hs.attackTarget != 0 {
		return
	}
	// Roaming does not suspend lane economics: a multi-creep XP opportunity is
	// converted before a single low-HP creep or a hero poke.
	if s.botConsiderWaveClearAbilityLocked(b, now) {
		return
	}
	if target := s.botFindLaneTargetLocked(b, now, botLaneEngageRadius, true); target != nil {
		if s.botMoveToFarmTargetLocked(b, target, now) {
			return
		}
	}
	if s.botConsiderHarassAbilityLocked(b, now) {
		return
	}
	cx, cy := c.posAtLocked(float32(now))
	if ally := s.botNearestAllyLocked(b, now); ally != nil && botIsActiveLocked(ally) {
		ax, ay := ally.posAtLocked(float32(now))
		if dist2(cx, cy, ax, ay) > 2*2 {
			s.botMoveTowardLocked(b, ax, ay, now)
			return
		}
	}
	lane := s.botRedirectLaneLocked(b, now)
	lx, ly := s.botPushPointLocked(b, lane, now)
	s.botMoveTowardLocked(b, lx, ly, now)
}

// botNearestLaneToPointLocked returns the lane index whose polyline passes closest to
// (x,y) -- the shared "which lane is this point on" geometry behind botPushLaneLocked and
// botRedirectLaneLocked.
func botNearestLaneToPointLocked(d *dotaState, x, y float32) int {
	best, bestD := 0, math.Inf(1)
	for i, lane := range d.m.Lanes {
		for _, wp := range lane {
			dd := math.Hypot(float64(float32(wp.X)-x), float64(float32(wp.Y)-y))
			if dd < bestD {
				bestD, best = dd, i
			}
		}
	}
	return best
}

// botPushLaneLocked returns the lane index closest to the grouped party's centroid --
// "which lane the group is actually standing on/heading toward" -- falling back to this
// bot's own assigned lane when nobody else is close enough to average in.
func (s *Server) botPushLaneLocked(b *botBrain, now float64) int {
	if b.macroAssignment.Lane >= 0 && (b.macroAssignment.Mode == botMacroPush ||
		b.macroAssignment.Mode == botMacroRally || b.macroAssignment.Mode == botMacroAltar) {
		return b.macroAssignment.Lane
	}
	c := b.c
	cx, cy := c.posAtLocked(float32(now))
	sx, sy, n := cx, cy, float32(1)
	for _, mem := range botLivingAllies(c) {
		if !botStrategicAllyLocked(c, mem) {
			continue
		}
		mx, my := mem.posAtLocked(float32(now))
		if dist2(cx, cy, mx, my) <= float32(botGroupUpRadius*botGroupUpRadius) {
			sx, sy, n = sx+mx, sy+my, n+1
		}
	}
	return botNearestLaneToPointLocked(c.inst.dota, sx/n, sy/n)
}

// botRedirectLaneLocked is botPushLaneLocked's un-gated twin: it averages EVERY living
// teammate's position, not just whoever already happens to be within botGroupUpRadius of
// the caller. botPushLaneLocked's own radius gate is exactly the wrong signal for a bot
// that has nothing left to do and no useful ally nearby (see botRoamTickLocked) -- in that
// situation the only teammate within range is typically another equally-stuck bot, so a
// gated centroid just points back at the same dead spot. Averaging in every living
// teammate regardless of distance points instead at wherever the team as a whole actually
// still is.
func (s *Server) botRedirectLaneLocked(b *botBrain, now float64) int {
	if b.macroAssignment.Lane >= 0 && (b.macroAssignment.Mode == botMacroPush ||
		b.macroAssignment.Mode == botMacroRally || b.macroAssignment.Mode == botMacroAltar) {
		return b.macroAssignment.Lane
	}
	c := b.c
	cx, cy := c.posAtLocked(float32(now))
	sx, sy, n := cx, cy, float32(1)
	for _, mem := range botLivingAllies(c) {
		if !botStrategicAllyLocked(c, mem) {
			continue
		}
		mx, my := mem.posAtLocked(float32(now))
		sx, sy, n = sx+mx, sy+my, n+1
	}
	return botNearestLaneToPointLocked(c.inst.dota, sx/n, sy/n)
}

// botPushPointLocked mirrors botLanePoint but for the group's chosen push lane rather than
// this bot's own default one.
func (s *Server) botPushPointLocked(b *botBrain, lane int, now float64) (float32, float32) {
	c := b.c
	d := c.inst.dota
	if lane < 0 || lane >= len(d.m.Lanes) || len(d.m.Lanes[lane]) == 0 {
		return botHomeLocked(c)
	}
	pts := d.m.Lanes[lane]
	if fx, fy, ok := botLaneFrontLocked(c, pts); ok {
		return fx, fy
	}
	fwd := c.playerTeam() == dotaTeamHuman
	idx := 2
	if !fwd {
		idx = len(pts) - 1 - 2
	}
	if idx < 0 {
		idx = 0
	} else if idx >= len(pts) {
		idx = len(pts) - 1
	}
	return float32(pts[idx].X), float32(pts[idx].Y)
}

// botMacroObjectiveTickLocked is the execution half of the team orchestrator's
// push decision.  A plan that names an enemy structure must eventually become an
// attack order on that exact structure; walking to a lane front is only a rally
// fallback and can otherwise leave a match farming forever once waves stop
// advancing.  Cover assignments inherit the same objective through ObjectiveID,
// while base-defense cover points at an allied structure and therefore fail the
// enemy check below.
func (s *Server) botMacroObjectiveTickLocked(b *botBrain, now float64) bool {
	if b == nil || b.c == nil || b.c.inst == nil || b.c.huntState == nil {
		return false
	}
	assignment := b.macroAssignment
	if assignment.Mode != botMacroPush && assignment.Mode != botMacroAltar && assignment.Mode != botMacroCover {
		return false
	}
	objective := b.c.inst.mobs[assignment.ObjectiveID]
	if objective == nil || objective.dead || !objective.structure || !objective.enemyOf(b.c.playerTeam()) ||
		b.c.altarShieldedLocked(objective) {
		return false
	}
	cx, cy := b.c.posAtLocked(float32(now))
	groupReady := botObjectiveFinishGroupReadyLocked(b.c.inst, b.c.playerTeam(), objective, now)
	commitGroupReady := botMacroObjectiveCommitWindowLocked(objective) && groupReady
	conversionCommit := assignment.Aggressive && assignment.Reason == "objective_conversion_ready" && groupReady
	// Do not walk an ordinary push into structure range before the local
	// conversion premise is true. Without this hold point, a bot reaches a
	// full-health gun a few seconds before its wave, absorbs a shot, retreats,
	// and repeats the same wasteful loop. A materially damaged objective in the
	// finish window is the deliberate exception: the group may finish it without
	// waiting for another wave. Full mobilization has already passed its own
	// launch gate and is also allowed to commit.
	if assignment.Reason != "" && assignment.Reason != botMacroReasonFullMobilization &&
		!botMacroObjectiveFinishWindowLocked(objective) && !commitGroupReady && !conversionCommit &&
		!s.botObjectiveConversionReadyLocked(b.c.inst, b.c.playerTeam(), objective, now) {
		rx, ry, ok := botMobilizationRallyPointLocked(b.c.inst, b.c.playerTeam(), objective)
		if ok {
			distance := math.Hypot(float64(objective.x-cx), float64(objective.y-cy))
			if distance <= b.c.huntState.effAttackRangeLocked(now)+float64(b.c.huntState.av.Radius())+objective.mob.Radius()+botMobilizationGatherRadius {
				// The rally point is already inside the normal approach band; keep
				// the bot there until the wave/group predicate becomes true.
				s.botMoveTowardLocked(b, rx, ry, now)
				return true
			}
		}
	}
	reach := b.c.huntState.effAttackRangeLocked(now) + float64(b.c.huntState.av.Radius()) + objective.mob.Radius()
	if math.Hypot(float64(objective.x-cx), float64(objective.y-cy)) <= reach {
		s.startAttackLocked(b.c, objective)
	} else {
		s.botMoveTowardLocked(b, objective.x, objective.y, now)
	}
	return true
}

// botMobilizationGatherTickLocked is the non-aggressive preparation phase of
// a full team conversion. It deliberately issues no creep or structure attack:
// healthy members walk to one staging point, while injured members go through
// the normal fountain recovery route.
func (s *Server) botMobilizationGatherTickLocked(b *botBrain, now float64) bool {
	if b == nil || b.c == nil || b.c.inst == nil || b.c.huntState == nil {
		return false
	}
	objective := b.c.inst.mobs[b.macroAssignment.ObjectiveID]
	if objective == nil || objective.dead {
		return false
	}
	if botHPFrac(b.c.huntState, now) < botSafeHPFrac {
		if !b.retreating {
			botSetRetreatModeLocked(b, botRetreatModeRecovery, now)
		}
		x, y := s.botRetreatPointLocked(b, now)
		s.botMoveTowardLocked(b, x, y, now)
		return true
	}
	rx, ry, ok := botMobilizationRallyPointLocked(b.c.inst, b.c.playerTeam(), objective)
	if !ok {
		return false
	}
	cx, cy := b.c.posAtLocked(float32(now))
	if math.Hypot(float64(cx-rx), float64(cy-ry)) > botMobilizationGatherRadius {
		s.botMoveTowardLocked(b, rx, ry, now)
	} else {
		b.c.stopMovementLocked(s, now)
	}
	return true
}

// botGroupTickLocked: with enough of the team grouped up (botPhaseGroup), press whatever
// lane the party is on -- creeps, then structures without needing a creep escort, since
// the group itself is the tower-aggro soak a solo laner doesn't have.
func (s *Server) botGroupTickLocked(b *botBrain, now float64) {
	c, hs := b.c, b.c.huntState
	if s.botConsiderHealLocked(b, now) {
		return
	}
	if s.botShouldRetreatLocked(b, now) {
		hx, hy := s.botRetreatPointLocked(b, now)
		s.botMoveTowardLocked(b, hx, hy, now)
		return
	}
	objectiveFormation := b.macroAssignment.Reason == botMacroReasonObjectiveStaging ||
		b.macroAssignment.Reason == "objective_conversion_ready" ||
		b.macroAssignment.Reason == botMacroReasonFullMobilization
	if (b.macroAssignment.Reason == botMacroReasonMobilizationPreparation || objectiveFormation) &&
		(hs.attackTarget != 0 || (hs.attackActionActive && hs.pvpTarget == 0)) {
		// Preparation and objective formation are synchronization/execution phases.
		// A plan refresh can arrive immediately after an ordinary creep order; clear
		// that stale swing before the attack-state guard, otherwise the bot stays
		// frozen on the old target and never reaches the shared rally point or the
		// named structure. The exact objective order is preserved if it is already
		// the active target.
		if b.macroAssignment.Reason == botMacroReasonMobilizationPreparation ||
			hs.attackTarget == 0 || hs.attackTarget != b.macroAssignment.ObjectiveID {
			s.stopAttackLocked(c, false)
		}
	}
	if hs.attackTarget != 0 {
		return
	}
	if b.macroAssignment.Reason == botMacroReasonMobilizationPreparation {
		s.botMobilizationGatherTickLocked(b, now)
		return
	}
	// Once an aggressive responder is within the objective commit radius, issue
	// the exact structure order before considering nearby creeps. The ordinary
	// farm-first order is valuable on the approach, but near the building it
	// turns a winning push into another wave-clear loop.
	// During staging/conversion the objective route itself is the order even when
	// a creep is nearby; otherwise the group can remain farming forever at the
	// rally point and never close the last distance to the gun.
	if (objectiveFormation || s.botPushObjectiveHasPriorityLocked(b, now)) && s.botMacroObjectiveTickLocked(b, now) {
		return
	}
	// A grouped push spends AoE on the live wave before selecting one creep. The
	// XP grant is proximity-based, so deleting a cluster is strictly more useful
	// than preserving a gold last-hit opportunity.
	if s.botConsiderWaveClearAbilityLocked(b, now) {
		return
	}
	// A grouped defender or responder can still be the only body covering its
	// assigned lane. When no committed objective has priority, intercept that
	// visible wave before falling back to the narrow group attack radius; this
	// prevents a 30u approach gap from becoming a guaranteed XP miss.
	if !objectiveFormation && !s.botPushObjectiveHasPriorityLocked(b, now) {
		if s.botHoldFarmXPLocked(b, now) {
			return
		}
		if target := s.botFarmApproachTargetLocked(b, now); target != nil {
			if s.botMoveToFarmTargetLocked(b, target, now) {
				return
			}
		}
	}
	if target := s.botFindLaneTargetLocked(b, now, botGroupEngageRadius, false); target != nil {
		if s.botMoveToFarmTargetLocked(b, target, now) {
			return
		}
	}
	if s.botMacroObjectiveTickLocked(b, now) {
		return
	}
	lane := s.botPushLaneLocked(b, now)
	px, py := s.botPushPointLocked(b, lane, now)
	s.botMoveTowardLocked(b, px, py, now)
}
