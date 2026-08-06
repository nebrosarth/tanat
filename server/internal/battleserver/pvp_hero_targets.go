package battleserver

import (
	"math"
	"time"

	"tanatserver/internal/gamedata"
)

// This file closes the PVP ability-vs-hero gap: before it, a skill's OpDamage/OpStun/
// OpRoot/OpSlow/OpSilence could only ever land on a mobState (a creep, a tower, a boss),
// because every op-target scan (damageTargetsLocked/mobsWithinLocked/mobsAlongLineLocked)
// walks c.huntState.mobs -- which is inst.mobs, and enemy PLAYERS live in inst.members,
// a completely different map. Only the plain auto-attack path (startPvpAttackLocked ->
// hitPlayerFromLocked) ever touched an enemy hero's HP; a teamfight where casters throw
// spells at each other did nothing but animate.
//
// The fix does not rewire the ops engine around an interface (too invasive for a combat
// core this heavily tuned). Instead, a living enemy hero is represented to the existing
// scans by a disposable "shadow" mobState (heroOwner set, never stored in inst.mobs,
// rebuilt fresh on every op resolution). The two or three places that actually WRITE
// damage/status check heroOwner and redirect into the hero's real conn/huntState instead
// of mutating the throwaway shadow, which would vanish with the next GC cycle unread.

// heroShadowRadius is the body radius given to every hero shadow: a fixed, avatar-average
// figure rather than the exact caster's own gamedata.Avatar.Radius(), because building a
// shadow must not have to know which avatar mem is playing beyond what it already reads
// (av.Radius() IS read -- see dotaHeroShadowLocked -- this constant only backstops it if a
// future avatar's radius comes back zero, matching mobState.mob.Radius()'s own "0 falls
// back to a default" contract).
const heroShadowRadius = 0.6

// dotaHeroShadowLocked builds a transient mobState standing in for mem's avatar, usable
// anywhere a *mobState target is expected (damage/status ops), or nil if mem cannot
// currently be targeted (dead, or hidden by an active invisibility). Caller holds the
// world lock. The shadow is never inserted into inst.mobs -- see the file doc comment.
func dotaHeroShadowLocked(mem *conn, now float64) *mobState {
	hs := mem.huntState
	if hs == nil || hs.deadUntil > 0 || now < hs.invisibleUntil {
		return nil
	}
	px, py := mem.posAtLocked(float32(now))
	radius := hs.av.Radius()
	if radius <= 0 {
		radius = heroShadowRadius
	}
	return &mobState{
		id:        mem.objID,
		x:         px,
		y:         py,
		hp:        hs.hp,
		maxHP:     hs.maxHPLocked(now),
		mana:      hs.mana,
		maxMana:   hs.maxManaLocked(now),
		st:        hs.st,
		mob:       gamedata.Mob{CollisionRadius: radius},
		team:      mem.playerTeam(),
		shown:     true,
		active:    true,
		heroOwner: mem,
	}
}

// dotaLivingEnemyHeroShadowsLocked returns a shadow for every living, targetable enemy
// hero of c's side -- the raw candidate list dotaEnemyHeroShadowsLocked and its
// along-the-line twin both filter by geometry.
func (s *Server) dotaLivingEnemyHeroShadowsLocked(c *conn) []*mobState {
	if c.inst == nil {
		return nil
	}
	now := float64(s.battleTime())
	actingTeam := c.playerTeam()
	var out []*mobState
	for _, mem := range c.inst.members {
		if mem == c || mem.huntState == nil || mem.playerTeam() == actingTeam {
			continue
		}
		if sh := dotaHeroShadowLocked(mem, now); sh != nil {
			out = append(out, sh)
		}
	}
	return out
}

// dotaEnemyHeroShadowsLocked returns a shadow for every living, targetable enemy hero
// within radius of (x,y) -- the hero-side twin of the mob scan mobsWithinLocked already
// runs, appended by its caller (damageTargetsLocked) so a single AoE loop naturally
// catches creeps, structures AND heroes in one pass.
func (s *Server) dotaEnemyHeroShadowsLocked(c *conn, x, y float32, r float64) []*mobState {
	var out []*mobState
	for _, sh := range s.dotaLivingEnemyHeroShadowsLocked(c) {
		if math.Hypot(float64(sh.x-x), float64(sh.y-y)) <= r+sh.mob.Radius() {
			out = append(out, sh)
		}
	}
	return out
}

// dotaEnemyHeroShadowsAlongLineLocked is dotaEnemyHeroShadowsLocked's twin for a line/rift
// skill (mobsAlongLineLocked's hero-inclusive counterpart): the same along-ray/perpendicular
// test, run over enemy hero shadows instead of inst.mobs.
func (s *Server) dotaEnemyHeroShadowsAlongLineLocked(c *conn, cx, cy, tx, ty float32, halfWidth, maxLen float64) []*mobState {
	dx, dy := float64(tx-cx), float64(ty-cy)
	dlen := math.Hypot(dx, dy)
	if dlen < 1e-6 {
		return nil
	}
	ux, uy := dx/dlen, dy/dlen
	var out []*mobState
	for _, sh := range s.dotaLivingEnemyHeroShadowsLocked(c) {
		rx, ry := float64(sh.x-cx), float64(sh.y-cy)
		along := rx*ux + ry*uy
		perp := math.Abs(rx*uy - ry*ux)
		br := sh.mob.Radius()
		if along >= -br && along <= maxLen+br && perp <= halfWidth+br {
			out = append(out, sh)
		}
	}
	return out
}

// dotaEnemyHeroShadowLocked resolves a single enemy hero by objID to a shadow, for the
// unit-target (non-AoE) cast path -- startSkillOrderLocked's twin of hs.mobs[target] when
// the clicked id isn't a mob at all but a rival avatar. Returns nil if objID doesn't name a
// live, targetable enemy hero of c's side.
func (s *Server) dotaEnemyHeroShadowLocked(c *conn, objID int32, now float64) *mobState {
	if c.inst == nil || objID == c.objID {
		return nil
	}
	mem := c.inst.members[objID]
	if mem == nil || mem.huntState == nil || mem.playerTeam() == c.playerTeam() {
		return nil
	}
	return dotaHeroShadowLocked(mem, now)
}

// ---- CC ops on a hero shadow: write straight to the real huntState.st, not the shadow ----
//
// huntState.st and mobState.st are the SAME unitStatus type (see status.go's cloak-field
// comment: "an enemy mob OR an ally player"), so every existing player-side gate --
// startSkillOrderLocked's stunned()/silenced() check, resumeAutoAttackLocked's stunned()
// check, handleMove's rooted() guard -- honours these fields the instant they're set, with
// no separate plumbing. The shadow's own .st (a value COPY taken at shadow-build time) is
// discarded; these helpers address the real mem.huntState.st directly.

// ensureHeroStatusFxLocked starts a looped status VFX on an enemy hero exactly once,
// mirroring ensureMobStatusFxLocked but targeting a player's objID.
func (s *Server) ensureHeroStatusFxLocked(c *conn, mem *conn, slot *int32, fx string) {
	if *slot == 0 {
		*slot = s.worldFxStartLocked(c, fx, mem.objID, 0, false, 0, 0)
	}
}

// freezeStunnedHeroLocked stops mem in place and drops its in-flight attack/order, the
// player-side twin of stunMobLocked's stopMobLocked+cancelSwingLocked pair.
func (s *Server) freezeStunnedHeroLocked(mem *conn, now float64) {
	s.stopAttackLocked(mem, true)
	s.cancelOrderLocked(mem)
	mem.stopArrivalLocked()
	mem.hasDest = false
	cx, cy := mem.posAtLocked(float32(now))
	mem.x, mem.y, mem.vx, mem.vy, mem.snapT = cx, cy, 0, 0, float32(now)
	mem.sendPosLocked(s, cx, cy, 0, 0, float32(now))
}

// dotaStunHeroLocked applies OpStun to an enemy hero.
func (s *Server) dotaStunHeroLocked(c *conn, mem *conn, now, dur float64) {
	hs := mem.huntState
	hs.st.stunUntil = math.Max(hs.st.stunUntil, now+dur)
	s.ensureHeroStatusFxLocked(c, mem, &hs.st.stunFx, "StunEffect")
	s.freezeStunnedHeroLocked(mem, now)
}

// dotaRootHeroLocked applies OpRoot to an enemy hero: it may still act (attack/cast) but
// cannot move -- rooted() reads exactly like stunned() does for mobs.
func (s *Server) dotaRootHeroLocked(c *conn, mem *conn, now, dur float64) {
	hs := mem.huntState
	hs.st.rootUntil = math.Max(hs.st.rootUntil, now+dur)
	s.ensureHeroStatusFxLocked(c, mem, &hs.st.rootFx, "StunEffect")
	mem.stopArrivalLocked()
	mem.hasDest = false
	cx, cy := mem.posAtLocked(float32(now))
	mem.x, mem.y, mem.vx, mem.vy, mem.snapT = cx, cy, 0, 0, float32(now)
	mem.sendPosLocked(s, cx, cy, 0, 0, float32(now))
}

// dotaSlowHeroLocked applies OpSlow to an enemy hero -- the player-side twin of the
// OpSlow case's mob branch (same fields, same optional DecayTo ramp).
func (s *Server) dotaSlowHeroLocked(c *conn, mem *conn, now float64, factor, dur float64, decayTo float64) {
	hs := mem.huntState
	hs.st.slowUntil = now + dur
	hs.st.slowFactor = factor
	hs.st.slowFactorEnd = factor
	if decayTo > 0 {
		hs.st.slowFactorEnd = decayTo
	}
	hs.st.slowStart = now
	s.ensureHeroStatusFxLocked(c, mem, &hs.st.slowFx, "SlowMoveEffect")
}

// dotaSilenceHeroLocked applies OpSilence to an enemy hero: unlike a mob (whose only
// "spell" is its attack), a hero keeps auto-attacking while silenced -- only casting is
// blocked, matching silenced()'s player-side consumer in startSkillOrderLocked.
func (s *Server) dotaSilenceHeroLocked(c *conn, mem *conn, now, dur float64) {
	hs := mem.huntState
	hs.st.silenceUntil = now + dur
	s.ensureHeroStatusFxLocked(c, mem, &hs.st.silenceFx, "SilenceEffect")
}

// dotaAttackSlowHeroLocked applies OpAttackSlow to an enemy hero.
func (s *Server) dotaAttackSlowHeroLocked(c *conn, mem *conn, now, dur, factor float64) {
	hs := mem.huntState
	hs.st.atkSlowUntil = now + dur
	hs.st.atkSlowFactor = factor
	s.ensureHeroStatusFxLocked(c, mem, &hs.st.atkSlowFx, "SlowAttackEffect")
}

// ---- position ops on a hero: knockback/pull actually move the real conn ----
//
// knockbackMobLocked/pullMobLocked mutate a *mobState's x/y directly -- on a hero shadow
// that is a silent no-op (the shadow is discarded unread), which is exactly what Miriam's
// «Выстрел бури» reported live: it damaged and rooted an enemy hero (OpDamage/OpRoot
// already redirected) but never actually shoved them. These two write straight to the
// real mem.x/mem.y instead.

// dotaKnockbackHeroLocked shoves an enemy hero directly away from the caster by dist
// units, clipped to walkable ground -- the hero-side twin of knockbackMobLocked. Same wire
// pattern: the authoritative position moves to the landing spot at once, but the broadcast
// carries the OLD position with a real glide velocity so clients animate a shove instead of
// a teleport; a one-shot timer then sends the matching stop once the glide has had time to
// land. Any in-flight move order is dropped first, or the player's own destination could
// still "arrive" afterward and stomp the shoved position.
func (s *Server) dotaKnockbackHeroLocked(c *conn, mem *conn, dist float64, now float64) {
	if dist <= 0 {
		dist = 3
	}
	cx, cy := c.posAtLocked(float32(now))
	mx, my := mem.posAtLocked(float32(now))
	dx, dy := float64(mx-cx), float64(my-cy)
	d := math.Hypot(dx, dy)
	if d < 1e-6 {
		dx, dy, d = 1, 0, 1
	}
	ux, uy := dx/d, dy/d
	fromX, fromY := mx, my
	tx := mx + float32(ux*dist)
	ty := my + float32(uy*dist)
	if mem.nav != nil {
		nx, ny := mem.nav.Clip(float64(mx), float64(my), float64(tx), float64(ty))
		tx, ty = float32(nx), float32(ny)
	}
	adist := math.Hypot(float64(tx-fromX), float64(ty-fromY))
	if adist < 1e-3 {
		return
	}
	dur := adist / knockbackSpeed
	if dur < knockbackMinTime {
		dur = knockbackMinTime
	}
	mem.stopArrivalLocked()
	mem.hasDest = false
	mem.x, mem.y = tx, ty
	mem.vx, mem.vy = 0, 0
	mem.snapT = float32(now)
	mem.sendPosLocked(s, fromX, fromY, float32(ux*knockbackSpeed), float32(uy*knockbackSpeed), float32(now))

	time.AfterFunc(time.Duration(dur*float64(time.Second)), func() {
		mem.lock()
		defer mem.unlock()
		if mem.huntState == nil || mem.huntState.closed || mem.hasDest {
			return // closed, or a real move order has since taken over -- don't stomp it
		}
		at := s.battleTime()
		mem.x, mem.y, mem.vx, mem.vy, mem.snapT = tx, ty, 0, 0, at
		mem.sendPosLocked(s, tx, ty, 0, 0, at)
	})
}

// dotaPullHeroLocked yanks an enemy hero to within 1.5 units of the caster -- the
// hero-side twin of pullMobLocked. An instant reposition (unlike knockback's timed glide),
// so no follow-up timer is needed.
func (s *Server) dotaPullHeroLocked(c *conn, mem *conn, now float64) {
	cx, cy := c.posAtLocked(float32(now))
	mx, my := mem.posAtLocked(float32(now))
	d := math.Hypot(float64(mx-cx), float64(my-cy))
	if d < 1.5 {
		return
	}
	nx := cx + float32(float64(mx-cx)*1.5/d)
	ny := cy + float32(float64(my-cy)*1.5/d)
	mem.stopArrivalLocked()
	mem.hasDest = false
	mem.x, mem.y, mem.vx, mem.vy, mem.snapT = nx, ny, 0, 0, float32(now)
	mem.sendPosLocked(s, nx, ny, 0, 0, float32(now))
}
