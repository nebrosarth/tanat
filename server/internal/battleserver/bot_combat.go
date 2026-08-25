package battleserver

// Hero-vs-hero combat decisions: whether a fight is worth starting, who to focus, which
// ability to spend on it, and healing/shielding a hurt ally. Everything here leans on the
// PVP ability-vs-hero engine fix (pvp_hero_targets.go) -- an ability aimed at an enemy
// hero's position now actually damages/CCs them, so "use abilities when advantageous"
// means something real here, not just an animation.

import (
	"math"

	"tanatserver/internal/gamedata"
)

// botEngageRadius is how far a bot will notice and commit to an opportunistic hero fight.
const botEngageRadius = 16.0

// botFocusRallyRadius is the local convergence band for a shared team focus.
// It is intentionally wider than the attack acquisition radius: a bot that sees
// the director's target must be able to join the fight before deciding that the
// target is merely an optional lane encounter. Four fight radii still describe
// one local skirmish, not a blind cross-map chase; the target must also already
// have an allied body beside it.
const botFocusRallyRadius = botFightRadius * 4

// botFightRadius is the radius around the FIGHT LOCATION (not the bot itself) used to
// count allies/enemies for the favourability check -- wider than the engage radius since a
// teammate a little further back still arrives in time to matter.
const botFightRadius = 24.0

// botNoDiveRadius keeps a chase from continuing into an enemy tower/cannon's kill zone.
// DotaGunRange/DotaTowerRange are both ~10-11u; a few extra units of margin means a bot
// breaks off BEFORE it's actually taking structure damage, not after. Without this, a
// fleeing hero kited a bot straight through the enemy's own cannons and all the way to
// their fountain -- the numbers-favourability check alone never catches this, since it
// only counts nearby HEROES, not the structures doing the actual killing.
const botNoDiveRadius = 15.0

const (
	botStructureAvoidHold   = 2.0
	botStructureAvoidMargin = 2.0
	botStructureDetourClear = 2.5
)

type botStructureWaypoint struct {
	x, y float32
}

func (s *Server) botRecordStructureEscapeLocked(b *botBrain, threat *mobState, x, y float32, reason string, now float64) {
	if b == nil || b.c == nil {
		return
	}
	threatID := int32(0)
	if threat != nil {
		threatID = threat.id
	}
	if b.structureEscapeTelemetryThreat == threatID && b.structureEscapeTelemetryReason == reason &&
		math.Hypot(float64(x-b.structureEscapeTelemetryX), float64(y-b.structureEscapeTelemetryY)) <= 0.5 {
		return
	}
	b.structureEscapeTelemetryThreat = threatID
	b.structureEscapeTelemetryReason = reason
	b.structureEscapeTelemetryX, b.structureEscapeTelemetryY = x, y
	s.telemetryRecordBotStructureEscapeLocked(b.c, threat, x, y, reason, now)
}

// botCommittedStructureFocusLocked returns a living enemy structure whose
// already-committed swing or projectile is aimed at c. dtarget alone is only
// current intent; the hit/projectile timers make this an active shot state and
// avoid reacting to an idle structure's last target.
func (s *Server) botCommittedStructureFocusLocked(c *conn, now float64) *mobState {
	if c == nil || c.huntState == nil || c.inst == nil {
		return nil
	}
	for _, m := range botSortedMobs(c.inst) {
		if m == nil || m.dead || !m.structure || !m.enemyOf(c.playerTeam()) || m.mob.AttackRange <= 0 {
			continue
		}
		if m.hitAt > now && m.hitTarget == c.objID {
			return m
		}
		if m.projLaunchAt > now && m.projTarget == c.objID {
			return m
		}
		if m.projFlying && m.hitAt > now && m.projTarget == c.objID {
			return m
		}
		if m.swingDoneAt > now && m.dtarget == c.objID &&
			(m.hitTarget == c.objID || m.projTarget == c.objID) {
			return m
		}
	}
	return nil
}

// botCanCommitStructureFocusLocked is the push-side exception to the ordinary
// tower safety rule. A bot may trade through a committed shot only when its
// assignment is an aggressive objective push and the assigned objective has a
// favourable local power state. This keeps solo laners safe while allowing a
// properly supported group to finish a structure instead of resetting on every
// cannon projectile.
func (s *Server) botCanCommitStructureFocusLocked(b *botBrain, threat *mobState, now float64) bool {
	objective := (*mobState)(nil)
	if b != nil && b.c != nil && b.c.inst != nil {
		objective = b.c.inst.mobs[b.macroAssignment.ObjectiveID]
	}
	if b == nil || b.c == nil || b.c.inst == nil || threat == nil ||
		!threat.structure || !threat.enemyOf(b.c.playerTeam()) || objective == nil {
		return false
	}
	// The structure-focus exception only permits a recoverable trade. A bot
	// below the normal hard retreat floor must be allowed to leave even during
	// full mobilization or an objective conversion; otherwise the safety layer
	// correctly detects the cannon shot but the macro layer turns it back into
	// a forced death.
	minimumHP := botRetreatHPFrac
	fullAI30Assault := botUsesAI30(b) && b.macroAssignment.Reason == botMacroReasonFullMobilization
	if fullAI30Assault {
		minimumHP = botAI30ChaseMinHP
	}
	if b.c.huntState == nil || botHPFrac(b.c.huntState, now) <= minimumHP {
		return false
	}
	// A conversion lease is not permission to stand under a full-health gun
	// while the HP trajectory is already collapsing. The live telemetry failure
	// was exactly this sequence: the bot remained committed at 75% -> 67% ->
	// 57% HP, then reached the retreat floor before it could clear the wave.
	// One committed shot may be traded; sustained local damage must still force
	// an escape unless the named structure is already in the finish window.
	loss, rate := botRecentHPLossLocked(b, now)
	if !fullAI30Assault && !botMacroObjectiveFinishWindowLocked(objective) &&
		(loss >= botPredictiveRetreatLossFrac || rate >= botPredictiveRetreatLossRate) {
		return false
	}
	aggressiveObjective := (b.macroAssignment.Aggressive ||
		b.macroAssignment.Reason == "objective_conversion_ready") &&
		(b.macroAssignment.Mode == botMacroPush || b.macroAssignment.Mode == botMacroAltar)
	// A push support assignment is deliberately non-aggressive during the
	// approach so it can preserve lane coverage. Once the named front structure
	// is in the finish window, however, every assigned push/cover body is part
	// of the conversion group and must not peel away on every committed cannon
	// shot. The hard retreat HP floor below still protects a critically injured
	// avatar.
	finishSupport := (b.macroAssignment.Mode == botMacroPush || b.macroAssignment.Mode == botMacroCover) &&
		botMacroObjectiveFinishWindowLocked(objective)
	mobilized := b.macroAssignment.Reason == botMacroReasonFullMobilization
	if !aggressiveObjective && !finishSupport && !mobilized {
		return false
	}
	conversionReady := s.botObjectiveConversionReadyLocked(b.c.inst, b.c.playerTeam(), objective, now)
	// A damaged front structure is already invested conversion debt. Once at
	// least two assigned, recoverable bodies are close enough to finish it, a
	// committed gun shot is part of the trade rather than a reason to reset the
	// entire group. Without this exception the objective predicate often flips
	// false precisely after the gun fires, so every attacker repeatedly escapes
	// and the structure never receives the finishing damage.
	finishDebt := botObjectiveFinishGroupReadyLocked(b.c.inst, b.c.playerTeam(), objective, now) &&
		objective.maxHealth() > 0 && objective.hp/objective.maxHealth() <= botMacroCommitDamageHPFrac
	// A conversion group can lose its wave or a nearby member for one or two
	// think ticks while the objective is already damaged. Do not require the
	// group count again in that state: the named aggressive pusher owns the
	// execution debt and should keep attacking until the target is dead or its
	// own HP crosses the hard retreat floor. This is deliberately keyed to the
	// tactical assignment, not the match roster, so it also works for a
	// counter-push in an uneven game.
	objectiveDebt := aggressiveObjective &&
		(botMacroObjectiveCommitWindowLocked(objective)) &&
		botHPFrac(b.c.huntState, now) > botRetreatHPFrac &&
		(b.macroAssignment.Reason == "objective_conversion_ready" ||
			b.macroAssignment.Reason == botMacroReasonPartialMobilization ||
			b.macroAssignment.Role == botMacroCounterPushRole)
	// The conversion reason is itself an execution lease issued by the
	// orchestrator after the live wave/power gate passed. A wave can die, or a
	// support can cross the approach boundary, in the same tick that the gun
	// commits its shot. Do not turn that transient bookkeeping change into a
	// structure escape for the already-assigned pusher: the hard HP floor above
	// remains the only veto once the bot is actually in the gun's firing zone.
	conversionExecutionLease := aggressiveObjective &&
		b.macroAssignment.Reason == "objective_conversion_ready" &&
		botHPFrac(b.c.huntState, now) > botRetreatHPFrac
	// Conversion readiness is evaluated from live wave/pressure state. Once a
	// real aggressive group has entered the objective approach band, that wave
	// can die between two think ticks while the building remains the correct
	// target. Keep the already-issued attack committed for that local group;
	// otherwise the first gun shot flips the predicate and every bot performs
	// the same retreat/re-enter loop.
	conversionCommit := aggressiveObjective && b.macroAssignment.Reason == "objective_conversion_ready" &&
		botObjectiveFinishGroupReadyLocked(b.c.inst, b.c.playerTeam(), objective, now)
	if !conversionReady && !finishSupport && !mobilized && !finishDebt && !objectiveDebt &&
		!conversionCommit && !conversionExecutionLease {
		return false
	}
	// The orchestrator has already committed this bot to the first live
	// structure on the lane. The local power predicate above is recalculated as
	// bodies, HP, waves, and enemy pressure change; it never depends on roster
	// size.
	return true
}

// botObjectiveFinishGroupReadyLocked reports whether the current assignment
// contains a real local execution group for this objective. It intentionally
// uses assigned objective IDs and spatial proximity, not the total roster: a
// distant lane bot must not make a lone diver trade through a gun shot.
func botObjectiveFinishGroupReadyLocked(inst *huntInstance, team int32, objective *mobState, now float64) bool {
	if inst == nil || objective == nil || objective.dead || !objective.structure {
		return false
	}
	ready := 0
	for _, mem := range inst.members {
		if mem == nil || mem.huntState == nil || mem.playerTeam() != team || mem.huntState.deadUntil > 0 {
			continue
		}
		brain := inst.bots[mem.objID]
		if brain == nil || brain.retreating || brain.macroAssignment.ObjectiveID != objective.id {
			continue
		}
		switch brain.macroAssignment.Mode {
		case botMacroPush, botMacroAltar, botMacroCover:
		default:
			continue
		}
		if botHPFrac(mem.huntState, now) <= botRetreatHPFrac {
			continue
		}
		x, y := mem.posAtLocked(float32(now))
		if math.Hypot(float64(x-objective.x), float64(y-objective.y)) > botObjectiveApproachRadius {
			continue
		}
		ready++
		if ready >= 2 {
			return true
		}
	}
	return false
}

// botStructureDangerRadiusLocked is the circle a bot must not enter. The shared
// no-dive policy is deliberately the floor: the structure's actual attack reach,
// both bodies and a small margin remain covered when a map or avatar changes.
func (s *Server) botStructureDangerRadiusLocked(c *conn, threat *mobState) float64 {
	if c == nil || c.huntState == nil || threat == nil {
		return botNoDiveRadius
	}
	return math.Max(botNoDiveRadius,
		threat.mob.AttackRange+float64(c.huntState.av.Radius())+botStructureAvoidMargin)
}

func botSegmentIntersectsCircle(ax, ay, bx, by, cx, cy float64, radius float64) bool {
	dx, dy := bx-ax, by-ay
	length2 := dx*dx + dy*dy
	t := 0.0
	if length2 > 0 {
		t = ((cx-ax)*dx + (cy-ay)*dy) / length2
		if t < 0 {
			t = 0
		} else if t > 1 {
			t = 1
		}
	}
	px, py := ax+t*dx, ay+t*dy
	return (px-cx)*(px-cx)+(py-cy)*(py-cy) <= radius*radius
}

// botRouteLocked mirrors conn.moveToLocked's route choice. The bool is false
// when navigation could not produce a route; callers that promise a nav-safe
// detour must reject that route rather than silently falling back through walls.
func botRouteLocked(c *conn, fx, fy, tx, ty float32) ([]gamedata.Vec2, bool) {
	if c != nil && c.nav != nil && !calibrateNav {
		if route := c.nav.Path(float64(fx), float64(fy), float64(tx), float64(ty)); len(route) > 0 {
			return route, true
		}
		return []gamedata.Vec2{{X: float64(tx), Y: float64(ty)}}, false
	}
	return []gamedata.Vec2{{X: float64(tx), Y: float64(ty)}}, true
}

func (s *Server) botRouteClearOfStructureLocked(c *conn, fx, fy, tx, ty float32, threat *mobState, requireNav bool) bool {
	route, navOK := botRouteLocked(c, fx, fy, tx, ty)
	if requireNav && !navOK {
		return false
	}
	radius := s.botStructureDangerRadiusLocked(c, threat)
	ax, ay := float64(fx), float64(fy)
	for _, point := range route {
		bx, by := point.X, point.Y
		if botSegmentIntersectsCircle(ax, ay, bx, by, float64(threat.x), float64(threat.y), radius) {
			return false
		}
		ax, ay = bx, by
	}
	return true
}

// botEscapeRouteClearOfStructureLocked is the one exception to the ordinary
// no-circle route rule: an emergency escape may start inside the circle, but it
// must move monotonically farther from the threat until it exits, and it may
// never enter the circle again. Outside starts retain the strict predicate above.
func (s *Server) botEscapeRouteClearOfStructureLocked(c *conn, fx, fy, tx, ty float32, threat *mobState, requireNav bool) bool {
	route, navOK := botRouteLocked(c, fx, fy, tx, ty)
	if requireNav && !navOK {
		return false
	}
	radius := s.botStructureDangerRadiusLocked(c, threat)
	prevX, prevY := float64(fx), float64(fy)
	prevDist2 := (prevX-float64(threat.x))*(prevX-float64(threat.x)) + (prevY-float64(threat.y))*(prevY-float64(threat.y))
	inside := prevDist2 < radius*radius
	for _, point := range route {
		nextX, nextY := point.X, point.Y
		dx, dy := nextX-prevX, nextY-prevY
		if inside {
			// Squared distance along a segment is quadratic. Its derivative is
			// increasing, so a non-negative derivative at the start proves the
			// whole segment is monotonically outward.
			fromX, fromY := prevX-float64(threat.x), prevY-float64(threat.y)
			if fromX*dx+fromY*dy < -1e-6 {
				return false
			}
		} else if botSegmentIntersectsCircle(prevX, prevY, nextX, nextY,
			float64(threat.x), float64(threat.y), radius) {
			return false
		}
		nextDist2 := (nextX-float64(threat.x))*(nextX-float64(threat.x)) + (nextY-float64(threat.y))*(nextY-float64(threat.y))
		if inside && nextDist2+1e-6 < prevDist2 {
			return false
		}
		if inside && nextDist2 >= radius*radius {
			inside = false
		}
		prevX, prevY, prevDist2 = nextX, nextY, nextDist2
	}
	return true
}

// botStructureEscapePointLocked finds a point farther out on the bot's CURRENT
// side of the firing structure. This is intentionally not homeward: when home
// lies beyond the threat, the first escape leg must never cross the danger circle.
func (s *Server) botStructureEscapePointLocked(b *botBrain, threat *mobState) (float32, float32, bool) {
	c := b.c
	now := float32(s.battleTime())
	cx, cy := c.posAtLocked(now)
	dx, dy := float64(cx-threat.x), float64(cy-threat.y)
	if length := math.Hypot(dx, dy); length > 0.001 {
		dx, dy = dx/length, dy/length
	} else {
		hx, hy := botHomeLocked(c)
		dx, dy = float64(hx-threat.x), float64(hy-threat.y)
		length := math.Hypot(dx, dy)
		if length <= 0.001 {
			dx, dy = -1, 0
		} else {
			dx, dy = dx/length, dy/length
		}
	}
	radius := s.botStructureDangerRadiusLocked(c, threat)
	currentDistance := math.Hypot(float64(cx-threat.x), float64(cy-threat.y))
	minimumDistance := math.Max(radius+2, currentDistance+6)
	for distance := minimumDistance; distance <= minimumDistance+24; distance += 4 {
		for _, angle := range []float64{0, math.Pi / 12, -math.Pi / 12, math.Pi / 6, -math.Pi / 6} {
			cos, sin := math.Cos(angle), math.Sin(angle)
			x := threat.x + float32((dx*cos-dy*sin)*distance)
			y := threat.y + float32((dx*sin+dy*cos)*distance)
			if c.nav != nil && !c.nav.Walkable(float64(x), float64(y)) {
				continue
			}
			// Keep the destination on the same radial half-plane as the bot and
			// verify the actual routed leg, not only its endpoint.
			if float64(x-threat.x)*float64(cx-threat.x)+float64(y-threat.y)*float64(cy-threat.y) <= 0 {
				continue
			}
			if !s.botEscapeRouteClearOfStructureLocked(c, cx, cy, x, y, threat, true) || s.botEnemyStructureDangerLocked(c, x, y) {
				continue
			}
			return x, y, true
		}
	}
	return 0, 0, false
}

// botStructureEscapeFallbackLocked is used only when navigation cannot produce
// a validated route. It still points strictly away from the threat; falling
// back to home here would recreate the original through-the-tower failure.
func (s *Server) botStructureEscapeFallbackLocked(b *botBrain, threat *mobState) (float32, float32) {
	c := b.c
	now := float32(s.battleTime())
	cx, cy := c.posAtLocked(now)
	dx, dy := float64(cx-threat.x), float64(cy-threat.y)
	if length := math.Hypot(dx, dy); length <= 0.001 {
		hx, hy := botHomeLocked(c)
		dx, dy = float64(hx-threat.x), float64(hy-threat.y)
		length = math.Hypot(dx, dy)
		if length <= 0.001 {
			dx, dy, length = -1, 0, 1
		}
		dx, dy = dx/length, dy/length
	} else {
		dx, dy = dx/length, dy/length
	}
	distance := s.botStructureDangerRadiusLocked(c, threat) + 4
	return threat.x + float32(dx*distance), threat.y + float32(dy*distance)
}

func (s *Server) botEscapeStructureFocusLocked(b *botBrain, threat *mobState, now float64) {
	wasHolding := now < b.structureAvoidUntil
	b.structureAvoidUntil = now + botStructureAvoidHold
	newThreat := !wasHolding || b.structureAvoidTarget != threat.id
	b.structureAvoidTarget = threat.id
	tx, ty := b.structureAvoidDestinationX, b.structureAvoidDestinationY
	valid := !newThreat && b.structureAvoidDestinationValid &&
		s.botStructureEscapeDestinationValidLocked(b, threat, tx, ty, now)
	if newThreat || !valid {
		var ok bool
		tx, ty, ok = s.botStructureEscapePointLocked(b, threat)
		if !ok {
			tx, ty = s.botStructureEscapeFallbackLocked(b, threat)
		}
		b.structureAvoidDestinationX, b.structureAvoidDestinationY = tx, ty
		b.structureAvoidDestinationValid = true
		if newThreat {
			b.structureEscapeTelemetryThreat = 0
		}
		s.botRecordStructureEscapeLocked(b, threat, tx, ty, "committed_focus", now)
		s.botMoveTowardLocked(b, tx, ty, now)
		return
	}
	// The existing order keeps running; re-issuing it every world tick was the
	// source of the live destination churn.
}

func (s *Server) botHoldStructureAvoidanceLocked(b *botBrain, now float64) {
	if target := b.c.inst.mobs[b.structureAvoidTarget]; target != nil && !target.dead {
		tx, ty := b.structureAvoidDestinationX, b.structureAvoidDestinationY
		if b.structureAvoidDestinationValid &&
			s.botStructureEscapeDestinationValidLocked(b, target, tx, ty, now) {
			return // keep the existing order without extending the fixed hold
		}
		var ok bool
		tx, ty, ok = s.botStructureEscapePointLocked(b, target)
		if !ok {
			tx, ty = s.botStructureEscapeFallbackLocked(b, target)
		}
		b.structureAvoidDestinationX, b.structureAvoidDestinationY = tx, ty
		b.structureAvoidDestinationValid = true
		s.botRecordStructureEscapeLocked(b, target, tx, ty, "committed_focus", now)
		s.botMoveTowardLocked(b, tx, ty, now)
		return
	}
	b.structureAvoidDestinationValid = false
	tx, ty := botHomeLocked(b.c)
	s.botMoveTowardLocked(b, tx, ty, now)
}

func (s *Server) botStructureEscapeDestinationValidLocked(b *botBrain, threat *mobState, tx, ty float32, now float64) bool {
	if b == nil || b.c == nil || threat == nil {
		return false
	}
	if b.c.nav != nil && !b.c.nav.Walkable(float64(tx), float64(ty)) {
		return false
	}
	cx, cy := b.c.posAtLocked(float32(now))
	if dist2(cx, cy, tx, ty) <= 2*2 {
		return false // reached: choose the next outward-safe leg
	}
	return s.botEscapeRouteClearOfStructureLocked(b.c, cx, cy, tx, ty, threat, true) &&
		!s.botEnemyStructureDangerLocked(b.c, tx, ty)
}

func botStructureIsFiringLocked(threat *mobState, now float64) bool {
	if threat == nil || threat.dead || !threat.structure || threat.mob.AttackRange <= 0 {
		return false
	}
	return threat.hitAt > now || threat.projLaunchAt > now ||
		(threat.projFlying && threat.hitAt > now) || threat.swingDoneAt > now ||
		(threat.dtarget != 0 && threat.nextSwing > now)
}

func (s *Server) botRetreatThreatLocked(b *botBrain, now float64, hx, hy float32) *mobState {
	c := b.c
	if c == nil || c.inst == nil {
		return nil
	}
	// Finish an already-selected route for this threat even if the current
	// waypoint itself no longer intersects the original homeward segment.
	if b.structureDetourThreat != 0 && len(b.structureDetour) > 0 {
		if threat := c.inst.mobs[b.structureDetourThreat]; botStructureIsFiringLocked(threat, now) {
			return threat
		}
	}
	cx, cy := c.posAtLocked(float32(now))
	route, _ := botRouteLocked(c, cx, cy, hx, hy)
	var best *mobState
	bestID := int32(0)
	for _, threat := range c.inst.mobs {
		if !botStructureIsFiringLocked(threat, now) {
			continue
		}
		ax, ay := float64(cx), float64(cy)
		intersects := false
		for _, point := range route {
			if botSegmentIntersectsCircle(ax, ay, point.X, point.Y,
				float64(threat.x), float64(threat.y), s.botStructureDangerRadiusLocked(c, threat)) {
				intersects = true
				break
			}
			ax, ay = point.X, point.Y
		}
		if intersects && (best == nil || threat.id < bestID) {
			best, bestID = threat, threat.id
		}
	}
	return best
}

func (s *Server) botBuildStructureDetourLocked(b *botBrain, threat *mobState, side int, now float64, hx, hy float32) ([]botStructureWaypoint, float64, bool) {
	c := b.c
	sx, sy := c.posAtLocked(float32(now))
	dx, dy := float64(hx-sx), float64(hy-sy)
	lineLength := math.Hypot(dx, dy)
	if lineLength <= 0.001 {
		return nil, 0, false
	}
	radius := s.botStructureDangerRadiusLocked(c, threat) + botStructureDetourClear
	const samples = 48
	type detourPoint struct {
		waypoint       botStructureWaypoint
		startOK, endOK bool
	}
	points := make([]detourPoint, 0, samples/2)
	threatRadius := s.botStructureDangerRadiusLocked(c, threat)
	for i := 0; i < samples; i++ {
		angle := 2 * math.Pi * float64(i) / samples
		x := threat.x + float32(math.Cos(angle)*radius)
		y := threat.y + float32(math.Sin(angle)*radius)
		cross := dx*float64(y-sy) - dy*float64(x-sx)
		if float64(side)*cross <= 0 {
			continue
		}
		if c.nav != nil && !c.nav.Walkable(float64(x), float64(y)) {
			continue
		}
		startInside := dist2(sx, sy, threat.x, threat.y) < float32(threatRadius*threatRadius)
		startOK := startInside || !botSegmentIntersectsCircle(float64(sx), float64(sy), float64(x), float64(y),
			float64(threat.x), float64(threat.y), threatRadius)
		if startOK {
			if startInside {
				startOK = s.botEscapeRouteClearOfStructureLocked(c, sx, sy, x, y, threat, true)
			} else {
				startOK = s.botRouteClearOfStructureLocked(c, sx, sy, x, y, threat, true)
			}
		}
		endOK := !botSegmentIntersectsCircle(float64(x), float64(y), float64(hx), float64(hy),
			float64(threat.x), float64(threat.y), threatRadius) &&
			s.botRouteClearOfStructureLocked(c, x, y, hx, hy, threat, true)
		points = append(points, detourPoint{waypoint: botStructureWaypoint{x: x, y: y}, startOK: startOK, endOK: endOK})
	}

	// A single waypoint is preferred when both precomputed legs are safe.
	bestSingleCost := math.Inf(1)
	var bestSingle botStructureWaypoint
	for _, point := range points {
		if !point.startOK || !point.endOK {
			continue
		}
		cost := math.Hypot(float64(point.waypoint.x-sx), float64(point.waypoint.y-sy)) +
			math.Hypot(float64(hx-point.waypoint.x), float64(hy-point.waypoint.y))
		if cost < bestSingleCost {
			bestSingleCost, bestSingle = cost, point.waypoint
		}
	}
	if bestSingleCost < math.Inf(1) {
		return []botStructureWaypoint{bestSingle}, bestSingleCost, true
	}

	// Opposite-side routes generally need two tangent-like points. Pairing is
	// geometry-only here; at most eight cheapest pairs incur the middle Nav.Path
	// call, keeping navigation work O(samples) instead of O(samples^2).
	type detourCandidate struct {
		first, second botStructureWaypoint
		cost          float64
	}
	pairs := make([]detourCandidate, 0, 8)
	addPair := func(candidate detourCandidate) {
		if len(pairs) < cap(pairs) {
			pairs = append(pairs, candidate)
			return
		}
		worst := 0
		for i := 1; i < len(pairs); i++ {
			if pairs[i].cost > pairs[worst].cost {
				worst = i
			}
		}
		if candidate.cost < pairs[worst].cost {
			pairs[worst] = candidate
		}
	}
	for _, first := range points {
		if !first.startOK {
			continue
		}
		for _, second := range points {
			if !second.endOK || math.Hypot(float64(first.waypoint.x-second.waypoint.x), float64(first.waypoint.y-second.waypoint.y)) < 1 {
				continue
			}
			if botSegmentIntersectsCircle(float64(first.waypoint.x), float64(first.waypoint.y),
				float64(second.waypoint.x), float64(second.waypoint.y), float64(threat.x), float64(threat.y), threatRadius) {
				continue
			}
			candidate := detourCandidate{
				first: first.waypoint, second: second.waypoint,
				cost: math.Hypot(float64(first.waypoint.x-sx), float64(first.waypoint.y-sy)) +
					math.Hypot(float64(first.waypoint.x-second.waypoint.x), float64(first.waypoint.y-second.waypoint.y)) +
					math.Hypot(float64(hx-second.waypoint.x), float64(hy-second.waypoint.y)),
			}
			addPair(candidate)
		}
	}
	for _, pair := range pairs {
		if s.botRouteClearOfStructureLocked(c, pair.first.x, pair.first.y, pair.second.x, pair.second.y, threat, true) {
			return []botStructureWaypoint{pair.first, pair.second}, pair.cost, true
		}
	}
	return nil, 0, false
}

func (s *Server) botRetreatDetourWaypointLocked(b *botBrain, now float64, hx, hy float32) (botStructureWaypoint, bool) {
	c := b.c
	if c == nil || c.huntState == nil {
		return botStructureWaypoint{}, false
	}
	threat := s.botRetreatThreatLocked(b, now, hx, hy)
	if threat == nil {
		b.structureDetour = nil
		b.structureDetourThreat = 0
		b.structureDetourSide = 0
		return botStructureWaypoint{}, false
	}
	if b.structureDetourThreat != threat.id || b.structureDetourGoalX != hx || b.structureDetourGoalY != hy || len(b.structureDetour) == 0 {
		var side int
		if b.structureDetourThreat == threat.id && b.structureDetourSide != 0 {
			side = b.structureDetourSide
		} else {
			_, plusCost, plusOK := s.botBuildStructureDetourLocked(b, threat, 1, now, hx, hy)
			_, minusCost, minusOK := s.botBuildStructureDetourLocked(b, threat, -1, now, hx, hy)
			switch {
			case plusOK && (!minusOK || plusCost < minusCost):
				side = 1
			case minusOK:
				side = -1
			}
			if side == 0 {
				return botStructureWaypoint{}, false
			}
		}
		waypoints, _, ok := s.botBuildStructureDetourLocked(b, threat, side, now, hx, hy)
		if !ok {
			return botStructureWaypoint{}, false
		}
		b.structureDetourThreat, b.structureDetourSide = threat.id, side
		b.structureDetourGoalX, b.structureDetourGoalY = hx, hy
		b.structureDetour = waypoints
	}
	cx, cy := c.posAtLocked(float32(now))
	for len(b.structureDetour) > 0 && dist2(cx, cy, b.structureDetour[0].x, b.structureDetour[0].y) <= 2*2 {
		b.structureDetour = b.structureDetour[1:]
	}
	if len(b.structureDetour) == 0 {
		return botStructureWaypoint{}, false
	}
	waypoint := b.structureDetour[0]
	s.botRecordStructureEscapeLocked(b, threat, waypoint.x, waypoint.y, "retreat_detour", now)
	return waypoint, true
}

// botEnemyStructureDangerLocked reports whether (x,y) is within botNoDiveRadius of any
// living enemy structure (gun/tower/altar) -- i.e. whether continuing a fight AT that spot
// means tower-diving, not hero-fighting.
func (s *Server) botEnemyStructureDangerLocked(c *conn, x, y float32) bool {
	for _, m := range c.huntState.mobs {
		if m.dead || !m.structure || !m.enemyOf(c.playerTeam()) {
			continue
		}
		if dist2(x, y, m.x, m.y) <= botNoDiveRadius*botNoDiveRadius {
			return true
		}
	}
	return false
}

// botCombatTickLocked looks for a nearby enemy hero worth fighting and, if one is found,
// drives that engagement for this tick (ability, then attack order). Reports whether it
// issued an order, so the caller (botTickLocked) skips the phase-specific logic below it.
func (s *Server) botCombatTickLocked(b *botBrain, now float64) bool {
	c, hs := b.c, b.c.huntState
	// The cast-to-impact part of a targeted channel is already committed. It must
	// claim this decision tick before heal/retreat/hero targeting can issue a new
	// order and before the channelState exists for botHasBlockingChannelLocked.
	if botHasBlockingChannelLocked(hs, now) || botHasPendingBlockingChannelLocked(hs, now) {
		return true
	}
	// A hurting ally (or self) always gets a heal/shield first, fight or no fight --
	// this is what makes Ariana/Neirofim actually play their support role. Checked BEFORE
	// the retreat gate below: retreating exits combat handling entirely (returns false),
	// so a heal ready on the very tick a bot crosses the retreat threshold used to never
	// fire at all -- the bot fled instead of spending its one instant self-save.
	if s.botConsiderHealLocked(b, now) {
		return true
	}
	enemies := botLivingEnemyHeroes(c, now)
	immediateThreat := s.botImmediateHeroThreatLocked(b, enemies, now)
	directHeroThreat := immediateThreat != nil && immediateThreat.huntState != nil &&
		(immediateThreat.huntState.pvpTarget == c.objID || immediateThreat.huntState.attackTarget == c.objID)
	if s.botAltarFinishPriorityLocked(b) && (immediateThreat == nil || immediateThreat.objID != c.objID) {
		// An ally being chased is not allowed to pull the terminal assault away
		// from an altar that is already close to destruction. The threatened ally
		// has its own retreat authority; this bot converts the objective instead.
		b.engageTarget = 0
		s.stopPvpAttackLocked(c, false)
		if s.botMacroObjectiveTickLocked(b, now) {
			return true
		}
	}
	// A visible uncovered wave is a hard local XP obligation. It outranks an
	// optional hero focus and even a rescue of a nearby ally; only a hero that
	// is directly hitting this bot may pre-empt it. Otherwise the cover body
	// walks to the exact creep and preserves the next proximity-XP event.
	if !directHeroThreat && !botUsesAI30(b) {
		if farmTarget := s.botUrgentFarmCoverageTargetLocked(b, now); farmTarget != nil {
			b.engageTarget = 0
			s.stopPvpAttackLocked(c, false)
			cx, cy := c.posAtLocked(float32(now))
			distance := math.Hypot(float64(farmTarget.x-cx), float64(farmTarget.y-cy))
			attackReach := hs.effAttackRangeLocked(now) + hs.av.Radius() + farmTarget.mob.Radius()
			// Ranged attack reach is not XP reach. A farm rescue must first
			// enter the authoritative proximity radius; otherwise the bot can
			// hit a creep from 25u away and still miss its XP on death.
			// Combat has first refusal, but a creep rescue still obeys the
			// farm safety invariant. Only a genuinely weak creep is attacked;
			// otherwise the helper holds the rear XP anchor instead of walking
			// into the wave just because it was selected as urgent.
			if !s.botMoveToFarmTargetLocked(b, farmTarget, now) &&
				distance <= attackReach && distance <= dotaXPShareRadius &&
				s.botFarmMayAttackTargetLocked(b, farmTarget, now) {
				s.startAttackLocked(c, farmTarget)
			}
			return true
		}
	}
	// Before the first visible wave, a zero-XP farm owner must stage on its
	// assigned lane instead of taking an optional hero duel. This is a live
	// obligation (no XP yet + lane wave pending), not a match-clock phase.
	if immediateThreat == nil && !botUsesAI30(b) && botFirstFarmWavePendingLocked(b, now) {
		b.engageTarget = 0
		s.stopPvpAttackLocked(c, false)
		return false
	}
	if immediateThreat != nil && immediateThreat.huntState != nil &&
		(immediateThreat.huntState.pvpTarget == c.objID || immediateThreat.huntState.attackTarget == c.objID) {
		assignment := b.macroAssignment
		farmOwnerUnderPressure := assignment.FarmLaneSet && assignment.FarmLane >= 0 &&
			(assignment.Mode == botMacroLane || assignment.Mode == botMacroCover || assignment.Mode == botMacroBase) &&
			!assignment.Aggressive && s.botFarmApproachTargetLocked(b, now) != nil
		if farmOwnerUnderPressure && !botUsesAI30(b) && !botLastStandObjectiveLocked(b, now) {
			// A farm owner is not a duelist. Once an enemy hero has committed
			// contact on a lane with a live wave, preserve the avatar and let the
			// orchestrator hand the XP obligation to a healthy cover body. Walking
			// into a fair-looking PvP exchange here is strategically dominated by
			// the death/recovery gap it creates for the next creep wave.
			botSetRetreatModeLocked(b, botRetreatModeDisengage, now)
			hx, hy := s.botRetreatPointLocked(b, now)
			s.botMoveTowardLocked(b, hx, hy, now)
			return true
		}
		// A targeted spell may be the only authoritative evidence of contact.
		// Treat recent landed hero damage as a predictive disengage trigger for
		// every ordinary assignment, including push responders; a farm bot that
		// loses 30% more HP before its next thought is already too late to cover
		// the following wave.
		if !botUsesAI30(b) && !botLastStandObjectiveLocked(b, now) && botHPFrac(hs, now) <= botPredictiveRetreatHPFrac {
			botSetRetreatModeLocked(b, botRetreatModeDisengage, now)
			hx, hy := s.botRetreatPointLocked(b, now)
			s.botMoveTowardLocked(b, hx, hy, now)
			return true
		}
		if farmOwnerUnderPressure && !botUsesAI30(b) && botHPFrac(hs, now) <= botSafeHPFrac {
			// A farm owner is strategically replaceable; a death creates an
			// unrecoverable XP gap. Leave an active hero contact once HP has
			// crossed the normal safe band instead of accepting a fair duel on
			// the wave. The orchestrator can hand the lane to a nearby spare.
			botSetRetreatModeLocked(b, botRetreatModeDisengage, now)
			hx, hy := s.botRetreatPointLocked(b, now)
			s.botMoveTowardLocked(b, hx, hy, now)
			return true
		}
		// An active hero attacker is different from an optional duel. While the
		// bot is above the pressure-retreat floor, close a short local gap and
		// answer the attack before the objective/retreat phase can strand it.
		tx, ty := immediateThreat.posAtLocked(float32(now))
		cx, cy := c.posAtLocked(float32(now))
		distance := math.Hypot(float64(tx-cx), float64(ty-cy))
		chaseFloor := botPressureRetreatHPFrac
		if botUsesAI30(b) {
			chaseFloor = botAI30ChaseMinHP
		}
		if botHPFrac(hs, now) > chaseFloor &&
			distance > botEngageRadius && distance <= botFocusRallyRadius+botFightRadius &&
			!s.botEnemyStructureDangerLocked(c, tx, ty) {
			b.engageTarget = immediateThreat.objID
			s.stopPvpAttackLocked(c, false)
			s.botMoveTowardLocked(b, tx, ty, now)
			return true
		}
	}
	if s.botShouldRetreatLocked(b, now) {
		if s.botFarmXPShadowTickLocked(b, now) {
			return true
		}
		return s.botConsiderRetreatUtilityLocked(b, now)
	}
	// Preparation is a synchronization phase. A bot may answer an enemy that
	// is actively hitting it, but it must not keep an optional chase alive and
	// strand the whole mobilization group on a side fight.
	if (b.macroAssignment.Reason == botMacroReasonMobilizationPreparation ||
		b.macroAssignment.Reason == botMacroReasonObjectiveStaging) && hs.pvpTarget != 0 {
		target := c.pvpMember(hs.pvpTarget)
		if target == nil || !botMobilizationPreparationEnemyThreatLocked(c, target) {
			b.engageTarget = 0
			s.stopPvpAttackLocked(c, false)
			return false
		}
		return true
	}
	if immediateThreat != nil && immediateThreat.objID != c.objID &&
		immediateThreat.huntState != nil &&
		immediateThreat.huntState.pvpTarget != c.objID &&
		immediateThreat.huntState.attackTarget != c.objID {
		// A nearby ally is under active hero pressure. If the rescuer is not yet
		// in attack range, close the local gap first; otherwise the objective
		// phase wins the tick and the ally can die in plain sight of the group.
		tx, ty := immediateThreat.posAtLocked(float32(now))
		cx, cy := c.posAtLocked(float32(now))
		distance := math.Hypot(float64(tx-cx), float64(ty-cy))
		if distance > botEngageRadius && distance <= botFocusRallyRadius+botFightRadius &&
			!s.botEnemyStructureDangerLocked(c, tx, ty) {
			b.engageTarget = immediateThreat.objID
			s.stopPvpAttackLocked(c, false)
			s.botMoveTowardLocked(b, tx, ty, now)
			return true
		}
	}
	// Objective staging is a formation order. With no active local threat, an
	// optional visible hero must not pull a responder away from the rally point;
	// otherwise the group arrives piecemeal and loses the conversion window.
	if b.macroAssignment.Reason == botMacroReasonObjectiveStaging && immediateThreat == nil {
		b.engageTarget = 0
		s.stopPvpAttackLocked(c, false)
		return false
	}
	// A push assignment has a concrete structure objective. Once the group is
	// close enough to commit, an opportunistic hero fight must not steal the
	// decision tick from the building attack: that was the reason a numerically
	// superior team could win kills while leaving every barracks untouched. An
	// enemy hero that is actively attacking this bot or a nearby ally is the
	// exception; ignoring that contact lets a pusher die beside the objective.
	if s.botPushObjectiveHasPriorityLocked(b, now) && immediateThreat == nil {
		b.engageTarget = 0
		s.stopPvpAttackLocked(c, false)
		return false
	}
	// A sustained unit/self channel is an active combat decision, not a cast that
	// should be followed by the generic attack loop. PlusMinus's second skill must
	// keep its held enemy target while its five pulses resolve. Planted point
	// channels are excluded because they are fire-and-forget by design.
	if botHasBlockingChannelLocked(hs, now) || botHasPendingBlockingChannelLocked(hs, now) {
		return true
	}
	if len(enemies) == 0 {
		wasEngaged := b.engageTarget != 0 || hs.pvpTarget != 0
		b.engageTarget = 0
		s.stopPvpAttackLocked(c, false)
		if botUsesAI30(b) && wasEngaged {
			x, y := s.botDisengagePointLocked(b, now)
			s.botMoveTowardLocked(b, x, y, now)
			return true
		}
		return false
	}
	if botUsesAI30(b) && hs.pvpTarget != 0 {
		if chased := c.pvpMember(hs.pvpTarget); chased != nil && chased.huntState != nil &&
			chased.huntState.pvpTarget != c.objID && chased.huntState.attackTarget != c.objID &&
			!s.botAI30ChaseSafeLocked(b, chased, now) {
			b.engageTarget = 0
			s.stopPvpAttackLocked(c, false)
			x, y := s.botDisengagePointLocked(b, now)
			s.botMoveTowardLocked(b, x, y, now)
			return true
		}
	}
	// A shared focus is a coordination order, not just a tie-breaker between
	// heroes that happen to be inside one bot's attack radius. When a teammate is
	// already near that visible target, converge on the same fight before the
	// ordinary lane/farm phase gets a chance to split the group into isolated
	// duels. The visibility and structure-danger checks remain local and
	// authoritative; a hidden target or a target under its own gun is never used.
	if s.botMoveToTeamFocusLocked(b, enemies, now) {
		return true
	}
	target := s.botPickEngageTargetLocked(b, enemies, now)
	if target == nil {
		wasEngaged := b.engageTarget != 0 || hs.pvpTarget != 0
		beingPursued := false
		for _, enemy := range enemies {
			if enemy == nil || enemy.huntState == nil {
				continue
			}
			if enemy.huntState.pvpTarget == c.objID || enemy.huntState.attackTarget == c.objID {
				beingPursued = true
				break
			}
		}
		b.engageTarget = 0
		// The brain just decided this fight is no longer worth it (out of range,
		// unfavourable numbers, or -- see botEnemyStructureDangerLocked -- about to walk
		// into the enemy's own tower/cannon range). Without this, hs.pvpTarget stays set
		// and armPvpAttackTimer's own chase-and-rearm loop keeps running on its own
		// cadence regardless of what the brain thinks now: reported live, this is how a
		// bot ended up chasing a fleeing hero straight through the enemy's cannons and
		// onto their fountain. armPvpAttackTimer only ever stops itself when the victim
		// actually dies/leaves/goes invisible -- it has no notion of "too deep," so the
		// brain has to be the one pulling the plug every think tick it reconsiders.
		s.stopPvpAttackLocked(c, false)
		if wasEngaged {
			tx, ty := s.botDisengagePointLocked(b, now)
			s.botMoveTowardLocked(b, tx, ty, now)
			return true
		}
		// A stronger enemy can commit to this bot before the local favourability
		// gate ever accepts a reciprocal fight. Stopping the target order alone
		// leaves the bot walking its old lane while the pursuer keeps attacking;
		// explicitly break contact so the power comparison has a real tactical
		// consequence.
		if beingPursued {
			tx, ty := s.botDisengagePointLocked(b, now)
			s.botMoveTowardLocked(b, tx, ty, now)
			return true
		}
		return false
	}
	if b.macroAssignment.Reason == botMacroReasonMobilizationPreparation &&
		!botMobilizationPreparationEnemyThreatLocked(c, target) {
		b.engageTarget = 0
		s.stopPvpAttackLocked(c, false)
		return false
	}
	if s.botShouldPassOptionalHeroFightLocked(b, target, now) {
		// A numerically superior push should not turn every lane crossing into a
		// chase. Keep the group moving toward its building unless this hero is
		// actively hitting us; the objective is the source of the advantage's
		// value, while an optional kill only spends time and HP.
		b.engageTarget = 0
		s.stopPvpAttackLocked(c, false)
		return false
	}
	if s.botShouldProtectFarmFromOptionalFightLocked(b, target, now) {
		// XP debt is a strategic deficit, not a reason to take a low-value duel.
		// Let the lane phase acquire the next live wave; the direct-threat check
		// inside the helper preserves a response when this hero is actually
		// attacking the bot.
		b.engageTarget = 0
		s.stopPvpAttackLocked(c, false)
		return false
	}
	b.engageTarget = target.objID
	if s.botConsiderOffensiveAbilityLocked(b, target, now) {
		return true
	}
	if hs.pvpTarget != target.objID {
		s.startPvpAttackLocked(c, target)
	}
	return true
}

// botImmediateHeroThreatLocked returns a visible enemy that is already
// attacking this bot or a nearby ally. Objective priority may suppress an
// optional duel, but it must never suppress an active contact: doing so leaves
// the structure attacker standing still while a teammate is killed beside it.
// The rescue radius is deliberately local and uses the same visible enemy list
// as ordinary combat, so this is not an omniscient team-wide alarm.
func (s *Server) botImmediateHeroThreatLocked(b *botBrain, enemies []*conn, now float64) *conn {
	if b == nil || b.c == nil || b.c.huntState == nil {
		return nil
	}
	c := b.c
	// Skills can land damage without arming the normal PvP attack order. The
	// authoritative last-damager record closes that gap: after a targeted hit,
	// the bot knows which enemy just made contact even if that enemy's next
	// order has not been installed yet.
	if c.huntState.lastHeroDamager != 0 && now-c.huntState.lastHeroDamageAt <= botRecentHeroDamageThreatWindow {
		if attacker := c.pvpMember(c.huntState.lastHeroDamager); attacker != nil && attacker.huntState != nil &&
			attacker.huntState.deadUntil == 0 && arenaEnemies(attacker, c) {
			return attacker
		}
	}
	for _, enemy := range enemies {
		if enemy == nil || enemy.huntState == nil || enemy.huntState.deadUntil > 0 {
			continue
		}
		if enemy.huntState.pvpTarget == c.objID || enemy.huntState.attackTarget == c.objID {
			return enemy
		}
	}
	cx, cy := c.posAtLocked(float32(now))
	var best *conn
	bestDistance := math.Inf(1)
	for _, ally := range botLivingAllies(c) {
		if ally == nil || ally.huntState == nil || ally.huntState.deadUntil > 0 {
			continue
		}
		ax, ay := ally.posAtLocked(float32(now))
		if math.Hypot(float64(ax-cx), float64(ay-cy)) > botFocusRallyRadius {
			continue
		}
		for _, enemy := range enemies {
			if enemy == nil || enemy.huntState == nil || enemy.huntState.deadUntil > 0 ||
				(enemy.huntState.pvpTarget != ally.objID && enemy.huntState.attackTarget != ally.objID) {
				continue
			}
			ex, ey := enemy.posAtLocked(float32(now))
			if math.Hypot(float64(ex-ax), float64(ey-ay)) > botFightRadius {
				continue
			}
			distance := math.Hypot(float64(ex-cx), float64(ey-cy))
			if distance < bestDistance {
				best, bestDistance = enemy, distance
			}
		}
	}
	return best
}

// botMoveToTeamFocusLocked moves a bot into a nearby team rendezvous. It
// deliberately does not start a PVP order: the next think tick re-evaluates
// local power and lets botPickEngageTargetLocked commit the attack only when
// the group is actually in range. The director's shared target is the signal
// that starts the rendezvous; requiring an ally to be there already would make
// the first bot impossible to join and preserve isolated lane deaths.
func (s *Server) botMoveToTeamFocusLocked(b *botBrain, enemies []*conn, now float64) bool {
	if b == nil || b.c == nil || b.c.inst == nil || b.c.inst.dota == nil || b.c.huntState == nil {
		return false
	}
	plan, ok := b.c.inst.dota.teamPlans[b.c.playerTeam()]
	if !ok || plan.FocusTarget == 0 ||
		b.macroAssignment.Reason == botMacroReasonMobilizationPreparation ||
		b.macroAssignment.Reason == botMacroReasonObjectiveStaging {
		return false
	}
	if !b.macroAssignment.Aggressive && b.macroAssignment.Mode != botMacroPush &&
		b.macroAssignment.Mode != botMacroAltar && b.macroAssignment.Mode != botMacroCover {
		return false
	}
	var target *conn
	for _, enemy := range enemies {
		if enemy != nil && enemy.objID == plan.FocusTarget && enemy.huntState != nil && enemy.huntState.deadUntil == 0 {
			target = enemy
			break
		}
	}
	if target == nil {
		return false
	}
	// Once the director has established a safe conversion window, the building
	// is the team's highest-value target. An active attacker threatening a nearby
	// low-HP ally is the state-based exception: the pusher joins the peel briefly
	// so the objective group does not lose a body while the enemy gets free kills.
	if (b.macroAssignment.Mode == botMacroPush || b.macroAssignment.Mode == botMacroAltar) &&
		b.macroAssignment.ObjectiveID != 0 {
		objective := b.c.inst.mobs[b.macroAssignment.ObjectiveID]
		if s.botObjectiveConversionReadyLocked(b.c.inst, b.c.playerTeam(), objective, now) &&
			!s.botTeamFocusNeedsRescueLocked(b, target, now) {
			return false
		}
	}
	cx, cy := b.c.posAtLocked(float32(now))
	tx, ty := target.posAtLocked(float32(now))
	distance := math.Hypot(float64(tx-cx), float64(ty-cy))
	if distance <= botEngageRadius || distance > botFocusRallyRadius || s.botEnemyStructureDangerLocked(b.c, tx, ty) {
		return false
	}
	b.engageTarget = target.objID
	s.stopPvpAttackLocked(b.c, false)
	s.botMoveTowardLocked(b, tx, ty, now)
	return true
}

// botTeamFocusNeedsRescueLocked allows a push assignment to answer the shared
// focus when that focus is an active attacker on a nearby ally. The range and
// HP checks keep this local: an objective pusher does not abandon a conversion
// for a remote fight or for a healthy ally who can disengage on their own.
func (s *Server) botTeamFocusNeedsRescueLocked(b *botBrain, target *conn, now float64) bool {
	if b == nil || b.c == nil || b.c.inst == nil || target == nil || target.huntState == nil {
		return false
	}
	tx, ty := target.posAtLocked(float32(now))
	cx, cy := b.c.posAtLocked(float32(now))
	if math.Hypot(float64(tx-cx), float64(ty-cy)) > botFocusRallyRadius+botFightRadius {
		return false
	}
	for _, ally := range botLivingAllies(b.c) {
		if ally == nil || ally.huntState == nil || ally.objID == b.c.objID {
			continue
		}
		if target.huntState.pvpTarget != ally.objID && target.huntState.attackTarget != ally.objID {
			continue
		}
		ax, ay := ally.posAtLocked(float32(now))
		if math.Hypot(float64(ax-cx), float64(ay-cy)) > botFocusRallyRadius ||
			math.Hypot(float64(tx-ax), float64(ty-ay)) > botFightRadius {
			continue
		}
		return botHPFrac(ally.huntState, now) <= botSafeHPFrac
	}
	return false
}

func botMobilizationPreparationEnemyThreatLocked(c *conn, target *conn) bool {
	if c == nil || c.huntState == nil || target == nil || target.huntState == nil {
		return false
	}
	return target.huntState.pvpTarget == c.objID || target.huntState.attackTarget == c.objID ||
		c.huntState.pvpTarget == target.objID
}

func (s *Server) botPushObjectiveHasPriorityLocked(b *botBrain, now float64) bool {
	if b == nil || b.c == nil || b.c.inst == nil || b.c.huntState == nil {
		return false
	}
	assignment := b.macroAssignment
	if (!assignment.Aggressive && assignment.Reason != "objective_conversion_ready" &&
		assignment.Reason != botMacroReasonFullMobilization) ||
		(assignment.Mode != botMacroPush && assignment.Mode != botMacroAltar) {
		return false
	}
	objective := b.c.inst.mobs[assignment.ObjectiveID]
	if objective == nil || objective.dead || !objective.structure ||
		!objective.enemyOf(b.c.playerTeam()) || b.c.altarShieldedLocked(objective) {
		return false
	}
	cx, cy := b.c.posAtLocked(float32(now))
	distance := math.Hypot(float64(objective.x-cx), float64(objective.y-cy))
	if distance <= botGroupEngageRadius*2 {
		return true
	}
	// A nearly destroyed front object is a time-critical conversion. Once the
	// assigned group is already in the approach ring, an optional defender duel
	// must not consume the final attack orders needed to close it.
	return botMacroObjectiveCommitWindowLocked(objective) &&
		botObjectiveFinishGroupReadyLocked(b.c.inst, b.c.playerTeam(), objective, now) &&
		distance <= botObjectiveApproachRadius
}

// botAltarFinishPriorityLocked is the terminal-objective override. Once an
// exposed altar is already in its finish window, an enemy hero press on a
// nearby ally is no longer a reason to leave the altar and start an optional
// duel. A direct attack on the bot itself is still handled by the normal threat
// path; this only prevents a retreating ally from pulling the execution group
// away from a nearly destroyed win condition.
func (s *Server) botAltarFinishPriorityLocked(b *botBrain) bool {
	if b == nil || b.c == nil || b.c.inst == nil || b.c.huntState == nil {
		return false
	}
	assignment := b.macroAssignment
	if assignment.Mode != botMacroAltar || assignment.Reason != "enemy_altar_open" {
		return false
	}
	objective := b.c.inst.mobs[assignment.ObjectiveID]
	return objective != nil && !objective.dead && objective.altar &&
		objective.enemyOf(b.c.playerTeam()) && !b.c.altarShieldedLocked(objective) &&
		botMacroObjectiveFinishWindowLocked(objective)
}

func (s *Server) botShouldPassOptionalHeroFightLocked(b *botBrain, target *conn, now float64) bool {
	if b == nil || b.c == nil || b.c.inst == nil || target == nil || target.huntState == nil {
		return false
	}
	assignment := b.macroAssignment
	if (!assignment.Aggressive && assignment.Reason != "objective_conversion_ready" &&
		assignment.Reason != botMacroReasonFullMobilization) ||
		(assignment.Mode != botMacroPush && assignment.Mode != botMacroAltar) {
		return false
	}
	objective := b.c.inst.mobs[assignment.ObjectiveID]
	if objective == nil || objective.dead || !objective.structure || !objective.enemyOf(b.c.playerTeam()) ||
		b.c.altarShieldedLocked(objective) {
		return false
	}
	conversionReady := s.botObjectiveConversionReadyLocked(b.c.inst, b.c.playerTeam(), objective, now)
	finishDebt := botMacroObjectiveCommitWindowLocked(objective) &&
		botObjectiveFinishGroupReadyLocked(b.c.inst, b.c.playerTeam(), objective, now)
	conversionCommit := assignment.Aggressive && assignment.Reason == "objective_conversion_ready" &&
		botObjectiveFinishGroupReadyLocked(b.c.inst, b.c.playerTeam(), objective, now)
	if !conversionReady && !finishDebt && !conversionCommit {
		return false
	}
	// A hero actively attacking this bot is not optional. It is a blocking
	// threat and must be answered even during an objective conversion.
	if target.huntState.pvpTarget == b.c.objID || target.huntState.attackTarget == b.c.objID ||
		b.c.huntState.pvpTarget == target.objID {
		return false
	}
	cx, cy := b.c.posAtLocked(float32(now))
	ox, oy := objective.x, objective.y
	return math.Hypot(float64(ox-cx), float64(oy-cy)) > botGroupEngageRadius*2
}

// botShouldProtectFarmFromOptionalFightLocked keeps a farm-deficient laner from
// converting a visible but non-threatening hero into a time-consuming duel. The
// guard is deliberately local and state-based: a live XP debt plus an available
// lane target is enough to prefer farming, while a direct attack on this bot,
// an aggressive objective assignment, or a bot with no farm route is allowed to
// use the ordinary combat policy.
func (s *Server) botShouldProtectFarmFromOptionalFightLocked(b *botBrain, target *conn, now float64) bool {
	if b == nil || b.c == nil || b.c.inst == nil || target == nil || target.huntState == nil {
		return false
	}
	// AI-30 is the aggressive scripted opponent. A visible local enemy hero is
	// a legitimate lane target, not a reason to keep both sides staring across a
	// creep wave. Its combat path still refuses a tower dive and breaks below the
	// AI-30 chase-health floor.
	if botUsesAI30(b) {
		return false
	}
	assignment := b.macroAssignment
	if assignment.Aggressive || assignment.Mode == botMacroPush || assignment.Mode == botMacroAltar {
		return false
	}
	// A baseline cover assignment carries a stronger obligation than relative
	// farm debt: it is the named owner of a live lane.  If that owner is not
	// inside XP range yet, an optional hero fight must not pull it away from the
	// wave simply because another ally happened to receive the previous creep's
	// XP.  The rule is state-driven (live target + coverage assignment), not a
	// match-time opening window.
	farmRoute := assignment.FarmLaneSet && assignment.FarmLane >= 0 &&
		(assignment.Mode == botMacroLane || assignment.Mode == botMacroCover || assignment.Mode == botMacroBase)
	coverageNeedsFarm := assignment.Mode == botMacroCover && assignment.Coverage && farmRoute
	// A live wave is itself a farm obligation, even when this bot has not yet
	// accumulated measurable XP debt. Otherwise a healthy lane owner can take
	// an optional duel simply because an ally happened to receive the previous
	// proximity grant, and the resulting retreat/death strands the next wave.
	liveWaveNeedsFarm := s.botFarmApproachTargetLocked(b, now) != nil
	if !farmRoute && !coverageNeedsFarm && !liveWaveNeedsFarm && botFarmDebtLocked(b.c.inst, b) < 1 {
		return false
	}
	// A hero that is actively hitting this bot is not an optional fight. The
	// normal combat path must answer it even when the bot is behind on XP.
	if target.huntState.pvpTarget == b.c.objID || target.huntState.attackTarget == b.c.objID {
		return false
	}
	if !farmRoute && !liveWaveNeedsFarm {
		return false
	}
	return botHPFrac(b.c.huntState, now) > botRetreatHPFrac
}

// botPickEngageTargetLocked chooses which nearby enemy hero to fight, or nil if no fight
// is currently favourable. Sticky on the already-committed target so it doesn't flip-flop
// between two similarly-close enemies every think tick.
func (s *Server) botPickEngageTargetLocked(b *botBrain, enemies []*conn, now float64) *conn {
	c, hs := b.c, b.c.huntState
	cx, cy := c.posAtLocked(float32(now))

	nearest := s.botAI30PreferredTargetLocked(b, enemies, now)
	nearestD := math.Inf(1)
	if nearest != nil {
		ex, ey := nearest.posAtLocked(float32(now))
		nearestD = math.Hypot(float64(ex-cx), float64(ey-cy))
	}
	for _, e := range enemies {
		ex, ey := e.posAtLocked(float32(now))
		if d := math.Hypot(float64(ex-cx), float64(ey-cy)); nearest == nil && d <= botEngageRadius &&
			(d < nearestD || (d == nearestD && (nearest == nil || e.objID < nearest.objID))) {
			nearestD, nearest = d, e
		}
	}
	// The team plan supplies one shared focus target. Local range and danger
	// gates below remain authoritative, so a remote/unsafe focus never forces a
	// bot to walk into a bad fight.
	if c.inst != nil && c.inst.dota != nil {
		if plan, ok := c.inst.dota.teamPlans[c.playerTeam()]; ok && plan.FocusTarget != 0 {
			for _, e := range enemies {
				if e.objID != plan.FocusTarget {
					continue
				}
				ex, ey := e.posAtLocked(float32(now))
				if d := math.Hypot(float64(ex-cx), float64(ey-cy)); d <= botEngageRadius*1.4 {
					nearest, nearestD = e, d
				}
				break
			}
		}
	}
	if b.engageTarget != 0 {
		if cur := c.pvpMember(b.engageTarget); cur != nil && cur.huntState.deadUntil == 0 {
			ux, uy := cur.posAtLocked(float32(now))
			if math.Hypot(float64(ux-cx), float64(uy-cy)) <= botEngageRadius*1.4 {
				nearest = cur // keep pressing the committed target over a marginally-closer one
			}
		}
	}
	if nearest == nil {
		return nil
	}
	tx, ty := nearest.posAtLocked(float32(now))
	if s.botEnemyStructureDangerLocked(c, tx, ty) {
		return nil // the fight (or the chase toward it) is inside the enemy's own tower/cannon range -- not worth pressing
	}
	// AI-30 must initiate ordinary visible lane duels instead of waiting for an
	// enemy to attack first or for a favorable power estimate. Its only opening
	// gates are meaningful: do not fight inside enemy structure range and do not
	// start a new chase while already below its retreat floor.
	if botUsesAI30(b) {
		if botHPFrac(hs, now) >= botAI30ChaseMinHP {
			return nearest
		}
		return nil
	}

	// Favourability: count living allies vs enemies near the FIGHT LOCATION. A lone
	// laner does not start a fight next to two enemy teammates, and won't press a fight
	// while alone and already hurt even if the numbers are even. The headcount is
	// only the first gate: a fed, high-level carry is materially more dangerous
	// than a level-one support, so the director also compares weighted live power.
	allyN, enemyN := 1, 1 // self + target
	heroPower := func(mem *conn) float64 {
		if mem == nil || mem.huntState == nil || mem.huntState.deadUntil > 0 {
			return 0
		}
		hp := botHPFrac(mem.huntState, now)
		if hp <= 0 {
			return 0
		}
		levelMul := 1 + math.Min(float64(mem.huntState.level), 19)*0.06
		killMul := 1 + math.Min(float64(mem.huntState.frags), 20)*0.015
		power := hp * levelMul * killMul
		// Cooldowns and mana of an enemy are not visible through the ordinary
		// client contract. AI-30 may value its own ready kit, but must not inspect
		// hidden opponent readiness through authoritative server state.
		if botUsesAI30(b) && mem == c {
			power += s.botAI30ReadyPowerLocked(mem, now)
		}
		return power
	}
	allyPower, enemyPower := heroPower(c), heroPower(nearest)
	visionSources := dotaTeamVisionSourcesLocked(c.inst, c.playerTeam(), now)
	for _, mem := range c.inst.members {
		if mem == c || mem == nearest || mem.huntState == nil || mem.huntState.deadUntil > 0 {
			continue
		}
		mx, my := mem.posAtLocked(float32(now))
		if math.Hypot(float64(mx-tx), float64(my-ty)) > botFightRadius {
			continue
		}
		if mem.playerTeam() == c.playerTeam() {
			allyN++
			allyPower += heroPower(mem)
		} else {
			if !botVisibleEnemyMemberLocked(c.inst, c.playerTeam(), mem, now, visionSources) {
				continue
			}
			enemyN++
			enemyPower += heroPower(mem)
		}
	}
	if allyN < enemyN {
		return nil
	}
	// Equal live power is a fair fight and must remain engageable (the
	// headcount gate above already rejects a local numerical disadvantage).
	// Requiring an arbitrary 5% surplus made an otherwise identical 1v1 fail
	// the basic combat contract and caused bots to abandon safe opportunities.
	if allyPower < enemyPower {
		return nil
	}
	if allyN == 1 && botHPFrac(hs, now) < 0.5 {
		return nil
	}
	objectiveValue := 28.0 + float64(botEnemyRolePriority(nearest.huntState.av.Type))*3
	objectiveValue += (1 - botHPFrac(nearest.huntState, now)) * 18
	objectiveValue += math.Min(float64(nearest.huntState.frags), 20) * 2
	dangerCost := float64(enemyN-allyN) * 35
	if dangerCost < 0 {
		dangerCost = 0
	}
	if botTacticalRiskValue(objectiveValue, nearestD, dangerCost) <= 0 {
		return nil
	}
	return nearest
}

// botOffensiveOpPriority scores a ready ability for use against an enemy hero: hard CC
// first (a stunned/rooted/silenced target can't fight back or escape), then damage,
// everything else last.
func botOffensiveOpPriority(def gamedata.Skill) int {
	switch {
	case botSkillHasOp(def, gamedata.OpStun):
		return 4
	case botSkillHasOp(def, gamedata.OpRoot), botSkillHasOp(def, gamedata.OpSilence):
		return 3
	case botSkillHasOp(def, gamedata.OpDamage), botSkillHasOp(def, gamedata.OpExecute),
		botSkillHasOp(def, gamedata.OpManaScaledDamage), botSkillHasOp(def, gamedata.OpSlow):
		return 2
	case botSkillHasOp(def, gamedata.OpHeal), botSkillHasOp(def, gamedata.OpShield), botSkillHasOp(def, gamedata.OpHot):
		return 0 // handled separately by botConsiderHealLocked, never picked here
	default:
		return 1
	}
}

// botConsiderOffensiveAbilityLocked casts the highest-priority ready ability at target's
// current position. A point-targeted cast at the hero's own coordinates resolves the hero
// itself (see pvp_hero_targets.go), so this works uniformly for AoE, line and single-target
// kits alike without needing to special-case each skill's exact Target mask.
func (s *Server) botConsiderOffensiveAbilityLocked(b *botBrain, target *conn, now float64) bool {
	c, hs := b.c, b.c.huntState
	bestSlot, bestP := 0, -1
	for slot := 1; slot <= 4; slot++ {
		if !s.botAbilityReadyLocked(hs, slot, now) {
			continue
		}
		def := hs.skillDef(slot)
		if p := botOffensiveOpPriority(def); p > bestP {
			bestP, bestSlot = p, slot
		}
	}
	if bestSlot == 0 || bestP <= 0 {
		return false
	}
	def := hs.skillDef(bestSlot)
	if def.Target == "" || def.Target == "SELF" {
		return s.startSkillOrderLocked(c, bestSlot, 0, 0, 0, false)
	}
	tx, ty := target.posAtLocked(float32(now))
	var targetID int32
	if def.Targeting == "TARGET" {
		// Unit-targeted skills must retain the live hero object id through delayed
		// impact/channel ticks. Ground/AoE skills still receive only the current point
		// so their authored area/line shape is preserved.
		targetID = target.objID
	}
	return s.startSkillOrderLocked(c, bestSlot, targetID, tx, ty, true)
}

// botHealNeedFrac is the HP fraction below which a hero (self or ally) is worth spending a
// heal/shield on.
const botHealNeedFrac = 0.65

// botSkillTargetsAlliesLocked reports whether a heal/hot/shield slot can actually be aimed
// at (or land on) a hurt ally: an explicit FRIEND-flagged single-target cast, a
// self-centered cast (Target==""/"SELF" -- e.g. Tangren's "Целительный тотем", an
// AoE-on-self totem with On:"allies"), or a ground-targeted AoE (Target=="POINT" -- e.g.
// Ariana's "Исцеление", aimed at the hurt ally's position). Without the latter two, an
// entire class of real, already-balanced heals was structurally invisible to the bot: it
// only ever considered the FRIEND-flagged case, so a hero whose sole heal is a self-AoE
// totem or a ground-target AoE never cast it once, in any match, regardless of how hurt it
// or its allies were.
func botSkillTargetsAlliesLocked(def gamedata.Skill) bool {
	return skillHasTargetFlag(def, "FRIEND") || def.Target == "" || def.Target == "SELF" || def.Target == "POINT"
}

// botFindHealTargetLocked returns the most hurt nearby ally (self included) below
// botHealNeedFrac and a ready ally-reaching heal/shield/hot slot for it, or (nil, 0).
func (s *Server) botFindHealTargetLocked(b *botBrain, now float64) (*conn, int) {
	c, hs := b.c, b.c.huntState
	healSlot := 0
	for slot := 1; slot <= 4; slot++ {
		if !s.botAbilityReadyLocked(hs, slot, now) {
			continue
		}
		def := hs.skillDef(slot)
		if !botSkillTargetsAlliesLocked(def) {
			continue
		}
		if botSkillHasOp(def, gamedata.OpHeal) || botSkillHasOp(def, gamedata.OpHot) || botSkillHasOp(def, gamedata.OpShield) {
			healSlot = slot
			break
		}
	}
	if healSlot == 0 {
		return nil, 0
	}
	dist := float64(hs.skillDef(healSlot).Distance)
	if dist <= 0 {
		dist = 8
	}
	cx, cy := c.posAtLocked(float32(now))

	var worst *conn
	worstFrac := botHealNeedFrac
	if f := botHPFrac(hs, now); f < worstFrac {
		worst, worstFrac = c, f
	}
	for _, mem := range botLivingAllies(c) {
		f := botHPFrac(mem.huntState, now)
		if f >= worstFrac {
			continue
		}
		mx, my := mem.posAtLocked(float32(now))
		if math.Hypot(float64(mx-cx), float64(my-cy)) > dist+6 {
			continue // too far to reasonably reach with this cast right now
		}
		worst, worstFrac = mem, f
	}
	return worst, healSlot
}

// botConsiderHealLocked casts a support ability on the most hurt nearby ally (or self) if
// one is found. Reports whether it acted.
func (s *Server) botConsiderHealLocked(b *botBrain, now float64) bool {
	target, slot := s.botFindHealTargetLocked(b, now)
	if target == nil {
		return false
	}
	c := b.c
	def := c.huntState.skillDef(slot)
	tx, ty := target.posAtLocked(float32(now))
	switch def.Target {
	case "", "SELF":
		// Self-centered (or self-only) cast: startSkillOrderLocked's own self-cast branch
		// fires unconditionally on def.Target=="" || "SELF" and ignores target/position
		// entirely, so this only needs to happen at all -- who it ends up helping (e.g.
		// Tangren's totem healing whichever allies stand in its radius) is the skill's own
		// business, not this call's.
		return s.startSkillOrderLocked(c, slot, 0, 0, 0, false)
	case "POINT":
		// Ground-targeted AoE heal (e.g. Ariana's "Исцеление"): must NOT carry the ally's
		// objID -- startSkillOrderLocked only resolves an object target through the
		// FRIEND-flagged ally path, and a bare POINT skill doesn't carry that flag, so
		// passing target.objID here would silently fizzle the order (falls through to
		// orderDoneLocked with nothing cast). Aim at the hurt ally's position instead.
		return s.startSkillOrderLocked(c, slot, 0, tx, ty, true)
	default:
		return s.startSkillOrderLocked(c, slot, target.objID, tx, ty, true)
	}
}

func botRetreatOpHarmful(ops []gamedata.Op) bool {
	for _, op := range ops {
		// Enemy-facing damage is compatible with a retreat utility (for example,
		// Astarot's point dash). Reject only authored self-cost effects; the
		// explicit HP-spend ops are self-harm even when they have no Apply field.
		if op.Apply == "self" {
			switch op.Kind {
			case gamedata.OpDamage, gamedata.OpDot, gamedata.OpExecute, gamedata.OpManaScaledDamage,
				gamedata.OpLifestealHit, gamedata.OpDrainMaxHP, gamedata.OpManaBurnHit,
				gamedata.OpConsumeDots, gamedata.OpAttackDamage:
				return true
			}
		}
		switch op.Kind {
		case gamedata.OpEmpowerNextHit, gamedata.OpSelfRecoil:
			return true
		}
		if botRetreatOpHarmful(op.Ops) {
			return true
		}
	}
	return false
}

func botRetreatOpDeep(ops []gamedata.Op, kind gamedata.OpKind) bool {
	for _, op := range ops {
		if op.Kind == kind || botRetreatOpDeep(op.Ops, kind) {
			return true
		}
	}
	return false
}

// botRetreatHasBlockingChannel rejects sustained casts from the emergency
// escape path. A channel can grant armour or another positive self buff while
// still pinning the avatar in place; choosing it while retreating turns a
// defensive decision into an intentional death under incoming damage.
func botRetreatHasBlockingChannel(def gamedata.Skill) bool {
	return botRetreatOpDeep(def.Ops, gamedata.OpChannel)
}

func botRetreatPositiveSelfBuff(def gamedata.Skill) bool {
	for _, op := range def.Ops {
		if op.Kind != gamedata.OpBuffStat || (op.On != "" && op.On != "self") || op.Value.At(1) <= 0 {
			continue
		}
		switch op.Stat {
		case "move_speed_pct", "phys_armor", "magic_armor", "dmg_reduction_pct", "dodge_pct":
			return true
		}
	}
	return false
}

func (s *Server) botRetreatPursuerLocked(b *botBrain, now float64) *conn {
	c := b.c
	cx, cy := c.posAtLocked(float32(now))
	var pursuer *conn
	best := math.Inf(1)
	for _, enemy := range botLivingEnemyHeroes(c, now) {
		if enemy.huntState.pvpTarget != c.objID && enemy.huntState.attackTarget != c.objID {
			continue
		}
		ex, ey := enemy.posAtLocked(float32(now))
		if d := math.Hypot(float64(ex-cx), float64(ey-cy)); d < best {
			best, pursuer = d, enemy
		}
	}
	return pursuer
}

func botRetreatPointDashLocked(b *botBrain, def gamedata.Skill, now float64, pursuer *conn) (float32, float32, bool) {
	cx, cy := b.c.posAtLocked(float32(now))
	hx, hy := botHomeLocked(b.c)
	dx, dy := float64(hx-cx), float64(hy-cy)
	distance := math.Hypot(dx, dy)
	if distance <= 0.001 || def.Target != "POINT" {
		return 0, 0, false
	}
	maxDash := float64(def.Distance)
	if maxDash <= 0 {
		maxDash = 8
	}
	step := math.Min(distance, maxDash)
	tx := cx + float32(dx/distance*step)
	ty := cy + float32(dy/distance*step)
	if pursuer != nil {
		px, py := pursuer.posAtLocked(float32(now))
		before := dist2(cx, cy, px, py)
		after := dist2(tx, ty, px, py)
		if after <= before {
			return 0, 0, false
		}
	}
	return tx, ty, true
}

// botConsiderRetreatUtilityLocked runs before plain retreat movement. It only accepts
// operations whose authored gamedata says they save the caster or safely control a nearby
// pursuer; ordinary damage and self-cost skills are deliberately excluded.
func (s *Server) botConsiderRetreatUtilityLocked(b *botBrain, now float64) bool {
	if s.botConsiderHealLocked(b, now) {
		return true
	}
	hs := b.c.huntState
	pursuer := s.botRetreatPursuerLocked(b, now)
	bestSlot, bestScore := 0, -1
	var dashX, dashY float32
	hasDashDestination := false
	for slot := 1; slot <= 4; slot++ {
		if !s.botAbilityReadyLocked(hs, slot, now) {
			continue
		}
		def := hs.skillDef(slot)
		if botRetreatOpHarmful(def.Ops) || botRetreatHasBlockingChannel(def) {
			continue
		}
		score := -1
		var tx, ty float32
		var pointDash bool
		if def.Target == "" || def.Target == "SELF" {
			if botRetreatOpDeep(def.Ops, gamedata.OpStealth) || botRetreatPositiveSelfBuff(def) || botRetreatOpDeep(def.Ops, gamedata.OpShield) {
				score = 3
			}
		} else if botRetreatOpDeep(def.Ops, gamedata.OpDash) && def.Target == "POINT" {
			tx, ty, pointDash = botRetreatPointDashLocked(b, def, now, pursuer)
			if pointDash {
				score = 2
			}
		}
		if score > bestScore {
			bestSlot, bestScore = slot, score
			dashX, dashY, hasDashDestination = tx, ty, pointDash
		}
	}
	if bestSlot != 0 {
		if s.startSkillOrderLocked(b.c, bestSlot, 0, dashX, dashY, hasDashDestination) {
			reason, category := "retreat_self_utility", "self_defense"
			if hasDashDestination {
				reason, category = "retreat_home_dash", "point_dash"
			}
			s.telemetryRecordBotRetreatUtilityLocked(b.c, bestSlot, 0, reason, category, now, dashX, dashY, hasDashDestination)
			return true
		}
	}
	if pursuer == nil {
		return false
	}
	cx, cy := b.c.posAtLocked(float32(now))
	for slot := 1; slot <= 4; slot++ {
		if !s.botAbilityReadyLocked(hs, slot, now) {
			continue
		}
		def := hs.skillDef(slot)
		if botRetreatOpHarmful(def.Ops) || botRetreatHasBlockingChannel(def) ||
			(!botRetreatOpDeep(def.Ops, gamedata.OpStun) && !botRetreatOpDeep(def.Ops, gamedata.OpRoot) &&
				!botRetreatOpDeep(def.Ops, gamedata.OpSilence) && !botRetreatOpDeep(def.Ops, gamedata.OpSlow)) {
			continue
		}
		if def.Target == "" || def.Target == "SELF" || def.Target == "POINT" {
			continue
		}
		px, py := pursuer.posAtLocked(float32(now))
		maxDist := float64(def.Distance)
		if maxDist <= 0 || math.Hypot(float64(px-cx), float64(py-cy)) > maxDist+0.5 {
			continue
		}
		if s.startSkillOrderLocked(b.c, slot, pursuer.objID, px, py, true) {
			s.telemetryRecordBotRetreatUtilityLocked(b.c, slot, pursuer.objID, "retreat_pursuer_control", "nearby_control", now, 0, 0, false)
			return true
		}
	}
	return false
}
