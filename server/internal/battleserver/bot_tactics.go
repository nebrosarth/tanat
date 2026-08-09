package battleserver

// Mid/late-game tactics: roaming toward wherever teammates already are, and once enough
// of the team is grouped up, pushing the lane the group is actually standing on together
// -- structures included, since the party itself (not a creep wave) is the tower-aggro
// soak. botUpdatePhaseLocked (bot_brain.go) decides which of these runs each tick.

import "math"

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
	s.botConsiderWaveClearAbilityLocked(b, now)
	s.botConsiderHarassAbilityLocked(b, now)
	if hs.attackTarget != 0 {
		return
	}
	if target := s.botFindLaneTargetLocked(b, now, botLaneEngageRadius, true); target != nil {
		s.startAttackLocked(c, target)
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
	c := b.c
	cx, cy := c.posAtLocked(float32(now))
	sx, sy, n := cx, cy, float32(1)
	for _, mem := range botLivingAllies(c) {
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
	c := b.c
	cx, cy := c.posAtLocked(float32(now))
	sx, sy, n := cx, cy, float32(1)
	for _, mem := range botLivingAllies(c) {
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
	s.botConsiderWaveClearAbilityLocked(b, now)
	if hs.attackTarget != 0 {
		return
	}
	if target := s.botFindLaneTargetLocked(b, now, botGroupEngageRadius, false); target != nil {
		s.startAttackLocked(c, target)
		return
	}
	lane := s.botPushLaneLocked(b, now)
	px, py := s.botPushPointLocked(b, lane, now)
	s.botMoveTowardLocked(b, px, py, now)
}
