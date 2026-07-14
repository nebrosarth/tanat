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

// Combat tick: a single 200ms loop per hunt connection drives everything
// timed — mob AI, DoT/HoT streams, buff expiry, toggles/auras, traps,
// channels, summons, regen and the player's death/respawn. One loop instead
// of scattered AfterFuncs keeps ordering deterministic under mvMu.

const (
	tickInterval = 200 * time.Millisecond

	// dotTickInterval is how often a DoT (poison) deals its slice of damage. It
	// matches the combat tick (5x/sec) so the health bar bleeds down smoothly
	// instead of lurching once per second. Damage numbers are throttled separately
	// (see the DoT stream in tickMobsLocked) so the fine cadence doesn't spam them.
	dotTickInterval = 0.2

	mobAggroRange = 9.0  // mobs turn hostile inside this radius
	mobLeashRange = 22.0 // and give up beyond this distance from the player

	// mobReturnRegenPerSec is how fast a leashed mob heals while walking home, as a
	// fraction of its max HP per second. At ~40%/s a full reset takes ~2.5s, which
	// comfortably tops it off over the walk back from leash range so it arrives (and
	// re-reveals to a returning player) at full health.
	mobReturnRegenPerSec = 0.4
	// mobHomeEpsilon: a returning mob within this of its spawn is considered home.
	mobHomeEpsilon = 0.3

	// respawnEvictRange: on player respawn, any mob within this of the checkpoint is
	// sent home. Leash is player-relative, so a pack chased onto a checkpoint would
	// otherwise freeze there and re-aggro the fresh-respawned player (a spawn-camp
	// death loop). Pack spawns sit >~10m from a checkpoint by construction, so a mob
	// sent home lands outside mobAggroRange -- the checkpoint stays the safe rally
	// point the mob-free spawn ring (dungeonRebornClear) intends.
	respawnEvictRange = 14.0

	// meleeReach is the base edge-to-edge gap a melee swing crosses; the real
	// reach adds BOTH bodies' radii (attacker + target), matching the client's
	// AvatarAI (reach = range + attacker.mRadius + target.mRadius). A ranged
	// mob's own AttackRange replaces this base.
	meleeReach = 0.6

	summonSeek        = 12.0 // summons pick fights inside this radius
	summonRing        = 2.8  // escort radius: summons hold a ring slot AROUND the follow point
	summonSpawnRadius = 1.2  // tighter ring the burst spawns on, around its cast point
	summonSlotTol     = 0.6  // deadband: don't re-issue a follow until this far off the slot
	summonRadius      = 0.5  // body radius of a summoned crawler (has no gamedata.Mob entry)
	// summonSpeed is the summon's run speed. Deliberately above the avatar's
	// lobbyMoveSpeed (4.0) and on par with a normal mob (~4.2-4.6) so a pet can
	// actually catch up to and keep pace with its moving owner and its prey.
	summonSpeed = 4.4

	// mobSepRange is the desired minimum centre-to-centre spacing: mobs closer
	// than this push apart so a pack surrounds the target instead of stacking on
	// one spot. Kept below the effective melee reach so a spread pack still
	// reaches the target. mobSepWeight scales the push against the chase direction.
	mobSepRange  = 1.8
	mobSepWeight = 0.9

	// mobArrowSpeed is a ranged mob's projectile speed (world units/sec) used to size
	// the arrow's flight from the release point to the target. Fast, like a real arrow.
	// minArrowFlight/maxArrowFlight bound it so a point-blank shot still animates a
	// visible streak and a long shot never dawdles; the flight is additionally capped
	// to land before the next swing so the attack cycle stays clean.
	mobArrowSpeed  = 26.0
	minArrowFlight = 0.12
	maxArrowFlight = 0.35

	// Fog of war: a mob is only created on the client while the player is within
	// mobRevealRadius, and removed again past mobHideRadius (hysteresis avoids
	// flicker at the boundary). Both exceed mobLeashRange (22) so a mob is always
	// visible before it can aggro and until it has walked home. Server-side so it
	// works without the client's scene-driven fog plane.
	mobRevealRadius = 28.0
	mobHideRadius   = 34.0

	// Graded client fog: a revealed mob is rendered translucent while it sits in
	// the outer ring of the reveal range (a "DeathShader" 50%-alpha overlay via
	// the InvisibilityEffect fx) and snaps to full opacity as the player closes
	// in. The two radii give the shade toggle its own hysteresis, and both are
	// below mobRevealRadius so a freshly-revealed distant mob always appears
	// shaded first. Purely visual; server-driven, no client change.
	mobShadeRadius   = 24.0 // become shaded at/beyond this distance
	mobUnshadeRadius = 22.0 // clear shade once nearer than this
	mobShadeFx       = "InvisibilityEffect"

	respawnDelay = 6.0 // seconds from death to revive
	corpseHide   = 3.0 // seconds from death to SYNC-remove (death anim shows)

	// Mob respawn: every creature (trash and boss) revives mobRespawnDelay after
	// death at its authored spawn point, so the location can be farmed. The corpse
	// is removed from the client corpseDeleteDelay after death (death anim plays),
	// then the mobState waits, hidden, until the respawn timer elapses.
	mobRespawnDelay   = 300.0 // 5 minutes
	corpseDeleteDelay = 4 * time.Second
)

// trapState is one armed ground trap.
type trapState struct {
	x, y      float32
	radius    float64
	until     float64
	slot      int
	level     int
	ops       []gamedata.Op
	fx        int32
	triggerFx string
	// anchor is the invisible stationary object the SELF-mode trap fx is parented to
	// (0 = the fx is owned by the caster directly). Deleted when the trap ends.
	anchor int32
}

// channelState is one running channeled skill.
type channelState struct {
	slot, level int
	until       float64
	interval    float64
	nextPulse   float64
	target      int32
	px, py      float32
	hasPos      bool
	ops         []gamedata.Op
	// interruptible marks a ground-anchored channel the caster actively sustains
	// (Elgorm's «Стрелы Аркана») rather than a planted fire-and-forget ground
	// effect (Titanid's quake): it breaks the moment the caster acts again -- moves,
	// is stunned, or casts another skill. See channelInterruptible.
	interruptible bool
}

// summonState is one allied summoned unit. In a shared world it is rendered by
// every ready member (owner + teammates), each in its own tracker index, so proto
// (the party-wide prototype id) is stored to rebuild the model on a newcomer's
// client.
type summonState struct {
	id          int32
	proto       int32
	hp, maxHP   float64
	dmg         float64
	until       float64
	x, y        float32
	vx, vy      float32
	fang        float64 // fixed formation angle: this summon's slot on the ring around its anchor
	nextSwing   float64
	swingDoneAt float64 // battle-time to send ACTION_DONE closing the current swing
	dead        bool

	pf pathState // routed chase waypoints when the straight line is wall-blocked
}

// pathState caches an A* route a tick-driven chaser (mob/summon) is following:
// the remaining waypoints, the cursor into them, the anchor of the current leg
// (fromX/Y — the point we last steered from, for pass-detection), the goal the
// route was computed for (to detect the target drifting) and when it was
// computed (to bound staleness). Used only while the straight line is blocked.
type pathState struct {
	pts          []gamedata.Vec2
	idx          int
	fromX, fromY float32
	goalX, goalY float32
	at           float64
	tried        bool // a route has been computed at least once (gates the first attempt)
}

// aimAlong returns the point a chaser at (fx,fy) should steer toward to reach
// (tx,ty). With a clear line (or no nav) it heads straight at the target and
// drops any cached route. While blocked it (re)computes an A* route — when the
// cache is empty, exhausted, stale (>1s), or the target has drifted >2m — then
// advances through the waypoints as they are reached, returning the current one.
//
// A waypoint is retired when the chaser is within 0.6m of it OR has moved at/past
// it along the travel direction (a dot-product test). The projection test makes
// advancement independent of per-tick step length, so a fast/hasted unit that
// overshoots a waypoint still advances instead of oscillating around it, without
// loosening the corner radius (which would let it cut into a wall and stick).
func (c *conn) aimAlong(ps *pathState, fx, fy, tx, ty float32, blocked bool, now float64) (float32, float32) {
	if c.nav == nil || !blocked {
		ps.pts = nil
		return tx, ty
	}
	// (Re)compute the route when it is exhausted, stale (>1s), or the target has
	// drifted >2m. Crucially, an EMPTY route (Path returned nil for a target with
	// no walkable route — e.g. walled off within leash range) is gated by the
	// staleness throttle as well, so it retries at most once per second instead of
	// re-running a worst-case component-exhausting A* on every 200ms tick.
	stale := now-ps.at > 1.0
	drift := math.Hypot(float64(tx-ps.goalX), float64(ty-ps.goalY)) > 2.0
	recompute := !ps.tried || stale // first attempt is unconditional; empty retries gated by staleness
	if len(ps.pts) > 0 {
		recompute = ps.idx >= len(ps.pts) || stale || drift
	}
	if recompute {
		ps.pts = c.nav.Path(float64(fx), float64(fy), float64(tx), float64(ty))
		ps.idx = 0
		ps.fromX, ps.fromY = fx, fy
		ps.goalX, ps.goalY = tx, ty
		ps.at = now
		ps.tried = true
	}
	for ps.idx < len(ps.pts) {
		wp := ps.pts[ps.idx]
		dxw := float64(fx) - wp.X
		dyw := float64(fy) - wp.Y
		reached := dxw*dxw+dyw*dyw < 0.36 // within 0.6m
		if !reached {
			// Passed it? travel dir = wp - from; (chaser - wp)·dir >= 0 means at/beyond wp.
			tdx := wp.X - float64(ps.fromX)
			tdy := wp.Y - float64(ps.fromY)
			if tdx*tdx+tdy*tdy > 1e-6 && dxw*tdx+dyw*tdy >= 0 {
				reached = true
			}
		}
		if reached {
			ps.fromX, ps.fromY = float32(wp.X), float32(wp.Y)
			ps.idx++
			continue
		}
		return float32(wp.X), float32(wp.Y)
	}
	return tx, ty // route failed or fully consumed — head straight (clamp keeps it legal)
}

// procState is a registered passive on-hit proc.
type procState struct {
	slot   int
	chance gamedata.PerLevel
	ops    []gamedata.Op
}

// memberTickLocked runs one 200ms step of a single player's own upkeep: timed
// fx/payloads, statuses, toggles, traps, channels, summons, death/respawn and
// regen. The shared mob simulation is NOT here -- the instance ticker runs it
// exactly once per world (tickMobsLocked) after stepping every member, so mobs
// aren't double-simulated by a multi-player party. Caller holds the world lock.
func (s *Server) memberTickLocked(c *conn, now float64) {
	hs := c.huntState

	// 1. Timed fx ends and scheduled packets.
	var fxKeep []fxEnd
	for _, f := range hs.fxEnds {
		if f.at > now {
			fxKeep = append(fxKeep, f)
			continue
		}
		s.fxEndLocked(c, f.uid)
	}
	hs.fxEnds = fxKeep

	s.runDuePayloadsLocked(c, now)

	var adKeep []actionDone
	for _, ad := range hs.actionDones {
		if ad.at > now {
			adKeep = append(adKeep, ad)
			continue
		}
		doneArgs := amf.NewArray().
			Set("id", c.objID).
			Set("action", ad.action).
			Set("item", false).
			Set("cooldown", ad.cooldown)
		// Close the cast action on this player AND on teammates (mirrors execCast) so
		// the remote avatar's DoingAction clears -- otherwise the skill action lingers
		// in the teammate's mCurActions.
		s.pushAvatarAllLocked(c, battleproto.CmdActionDone, doneArgs)
		if ad.order {
			s.orderDoneLocked(c, ad.action)
			// A finished ability flows straight back into auto-attack, mirroring how
			// a kill rolls the avatar onto the next mob -- unless it started a channel
			// the caster is meant to keep sustaining (noResume).
			if !ad.noResume {
				s.resumeAutoAttackLocked(c, now, ad.resumeTarget)
			}
		}
	}
	hs.actionDones = adKeep

	// 2. Pending approach-cast.
	s.tickOrderLocked(c, now)

	// 3. Player statuses.
	s.tickPlayerStatusLocked(c, now)

	// 3b. Respawn checkpoints: activate the Reborn_point the player walks onto.
	s.tickRebornLocked(c, now)

	// 4. Toggles: mana drain + aura pulses.
	s.tickTogglesLocked(c, now)
	// 4b. Always-on passive auras (e.g. BlackDragon's «Крылья тьмы»).
	s.tickPassiveAurasLocked(c, now)
	// 4c. Keep the number on a dynamic passive's buff icon current (Velial's «Воля к
	// победе» bonus tracks missing HP).
	s.refreshPassiveBuffCountersLocked(c, now)

	// 5. Traps (+ deferred trap-anchor cleanup).
	s.tickTrapsLocked(c, now)
	s.tickAnchorEndsLocked(c, now)

	// 6. Channels.
	s.tickChannelsLocked(c, now)

	// 6b. Bounces (chaining skull).
	s.tickBouncesLocked(c, now)

	// 7. Summons.
	s.tickSummonsLocked(c, now)

	// 8. Death/respawn.
	if hs.deadUntil > 0 {
		if !hs.corpseHidden && now >= hs.diedAt+corpseHide {
			hs.corpseHidden = true
			if idx := hs.tr.remove(c.objID); idx >= 0 {
				s.push(c, battleproto.CmdSync, amf.NewArray().Set("data",
					newSyncBlob(float32(now)).removeIndex(idx).build(hs.tr.count())))
			}
			// Drop this corpse from every teammate's client too, so a dead player
			// doesn't linger as a frozen body on their screen.
			for _, other := range c.members() {
				if other != c {
					s.removeAvatarForLocked(other, c)
				}
			}
		}
		if now >= hs.deadUntil {
			s.respawnPlayerLocked(c, now)
		}
	}

	// 10. Regen (2s cadence).
	if now >= hs.nextRegen && hs.deadUntil == 0 {
		hs.nextRegen = now + 2
		var types []uint64
		maxHP := hs.maxHPLocked(now)
		if hs.hp < maxHP {
			hs.hp = math.Min(maxHP, hs.hp+2*(hs.av.HealthRegen+hs.st.modSum(now, "hp_regen")))
			types = append(types, syncHealth)
		}
		if maxMana := hs.maxManaLocked(now); hs.mana < maxMana {
			hs.mana = math.Min(maxMana, hs.mana+2*(hs.av.ManaRegen+hs.st.modSum(now, "mana_regen")))
			types = append(types, syncMana)
		}
		if len(types) > 0 {
			s.syncSelfLocked(c, types...)
		}
	}
}

// ---- player status upkeep ----

func (s *Server) tickPlayerStatusLocked(c *conn, now float64) {
	hs := c.huntState
	st := &hs.st

	// Expired mods: reverse icons/fx and re-sync stats.
	var keep []statMod
	changed := false
	for _, m := range st.mods {
		if m.until == 0 || m.until > now {
			keep = append(keep, m)
			continue
		}
		changed = true
		if m.buffEffID != 0 {
			s.push(c, battleproto.CmdRemEffector, amf.NewArray().Set("id", m.buffEffID))
		}
		s.fxEndLocked(c, m.fxUID)
	}
	st.mods = keep
	if changed {
		s.pushPlayerStatsLocked(c, now)
	}

	if st.shieldFx != 0 && (now >= st.shieldUntil || st.shield <= 0) {
		s.fxEndLocked(c, st.shieldFx)
		st.shieldFx = 0
	}
	if st.silenceFx != 0 && now >= st.silenceUntil {
		s.fxEndLocked(c, st.silenceFx)
		st.silenceFx = 0
		s.syncSelfIntLocked(c, syncSilence, 0)
	}
	if st.slowFx != 0 && now >= st.slowUntil {
		s.fxEndLocked(c, st.slowFx)
		st.slowFx = 0
		s.pushPlayerStatsLocked(c, now)
	}

	// HoT streams.
	var hotKeep []overTime
	healed := false
	for i := range st.hots {
		h := st.hots[i]
		for h.nextTick <= now && h.nextTick <= h.until {
			hs.hp = math.Min(hs.maxHPLocked(now), hs.hp+h.perSec)
			healed = true
			h.nextTick++
		}
		if h.until > now {
			hotKeep = append(hotKeep, h)
		}
	}
	st.hots = hotKeep
	if healed {
		s.syncSelfLocked(c, syncHealth)
	}
}

// ---- toggles / auras ----

func (s *Server) tickTogglesLocked(c *conn, now float64) {
	hs := c.huntState
	dt := tickInterval.Seconds()
	for slot := 1; slot <= 4; slot++ {
		if !hs.toggleOn[slot-1] {
			continue
		}
		def := hs.skillDef(slot)
		level := int(hs.skillLevel[slot-1])
		for _, op := range def.Ops {
			if op.Kind != gamedata.OpAura {
				continue
			}
			hs.mana -= skillManaCost(op.TickCost.At(level) * dt)
			if hs.mana <= 0 {
				hs.mana = 0
				s.syncSelfLocked(c, syncMana)
				s.toggleOffLocked(c, slot, now, false)
				break
			}
			if now >= hs.toggleNextPulse[slot-1] {
				hs.toggleNextPulse[slot-1] = now + math.Max(op.Interval, 0.5)
				cx, cy := c.posAtLocked(float32(now))
				for _, m := range c.mobsWithinLocked(cx, cy, op.Radius) {
					ctx := opCtx{slot: slot, level: level, target: m, toggle: true}
					s.applyOpsLocked(c, op.Ops, ctx, now)
				}
				s.syncSelfLocked(c, syncMana)
			}
		}
	}
}

// tickPassiveAurasLocked pulses always-on PASSIVE auras (e.g. BlackDragon's «Крылья
// тьмы» enemy attack-speed aura). Unlike a toggle aura it has no mana cost and no
// on/off state -- it runs whenever the passive is learned and the player is alive.
// Pulse cadence reuses toggleNextPulse[slot-1] (a slot is never both a toggle and a
// passive, so the field is free on a passive slot).
func (s *Server) tickPassiveAurasLocked(c *conn, now float64) {
	hs := c.huntState
	if hs.deadUntil > 0 {
		return
	}
	for slot := 1; slot <= 4; slot++ {
		def := hs.skillDef(slot)
		if def.Type != "PASSIVE" {
			continue
		}
		level := int(hs.skillLevel[slot-1])
		if level < 1 {
			continue
		}
		for _, op := range def.Ops {
			if op.Kind != gamedata.OpAura {
				continue
			}
			if now < hs.toggleNextPulse[slot-1] {
				continue
			}
			hs.toggleNextPulse[slot-1] = now + math.Max(op.Interval, 0.5)
			cx, cy := c.posAtLocked(float32(now))
			for _, m := range c.mobsWithinLocked(cx, cy, op.Radius) {
				ctx := opCtx{slot: slot, level: level, target: m}
				s.applyOpsLocked(c, op.Ops, ctx, now)
			}
		}
	}
}

// ---- traps ----

func (s *Server) tickTrapsLocked(c *conn, now float64) {
	hs := c.huntState
	var keep []trapState
	for _, t := range hs.traps {
		if now > t.until {
			s.fxEndLocked(c, t.fx)
			// Lifetime elapsed with no trigger: no trigger fx follows, so the anchor
			// can be removed at once.
			s.removeTrapAnchorLocked(c, t.anchor, now)
			continue
		}
		var victim *mobState
		for _, m := range c.mobsWithinLocked(t.x, t.y, t.radius) {
			victim = m
			break
		}
		if victim == nil {
			keep = append(keep, t)
			continue
		}
		s.fxEndLocked(c, t.fx)
		// The trigger fx is also SELF-mode, so own it to the same anchor (if any) so it
		// erupts at the trap point, and keep the anchor alive until it plays out.
		fxOwner := c.objID
		if t.anchor != 0 {
			fxOwner = t.anchor
		}
		uid := s.fxStartLocked(c, t.triggerFx, fxOwner, 0, true, t.x, t.y)
		hs.scheduleFxEnd(uid, now+3)
		if t.anchor != 0 {
			hs.anchorEnds = append(hs.anchorEnds, anchorEnd{id: t.anchor, at: now + 3})
		}
		ctx := opCtx{slot: t.slot, level: t.level, target: victim,
			px: t.x, py: t.y, hasPos: true}
		s.applyOpsLocked(c, t.ops, ctx, now)
	}
	hs.traps = keep
}

// ---- channels ----

func (s *Server) tickChannelsLocked(c *conn, now float64) {
	hs := c.huntState
	var keep []channelState
	for _, ch := range hs.channels {
		// A ground-anchored channel (POINT cast: a target-less effect pinned to a
		// world point, e.g. Titanid's «Землетрясение» 3-wave quake or Elgorm's
		// arcane-arrow rain) is fire-and-forget -- it keeps erupting at its point
		// while the caster walks off or is stunned; only death or its own duration
		// ends it. A self/unit channel (e.g. Abominator's «Пожирание» drain) needs
		// the caster to keep channeling, so moving/stun/death breaks it. hasDest is
		// the truthful "is-moving" flag (c.arrival lingers non-nil after a leg).
		groundAnchored := ch.hasPos && ch.target == 0
		broken := hs.deadUntil > 0
		// A self/unit channel, OR an interruptible ground channel (a caster-sustained
		// one like Elgorm's arrow rain), ends the moment the caster moves or is
		// stunned. A plain fire-and-forget ground channel (Titanid's quake) keeps
		// erupting regardless.
		if !groundAnchored || ch.interruptible {
			broken = broken || hs.st.stunned(now) || c.hasDest
		}
		if broken || now > ch.until {
			continue
		}
		if now >= ch.nextPulse {
			// Advance the schedule by the interval (accumulate), NOT now+interval.
			// The tick runs on a coarse 0.2s grid; recomputing from the snapped `now`
			// rounds every gap UP to the next tick, so a 0.46s cadence drifts to ~0.6s
			// and loses pulses (Elgorm's arrow rain dropped from 9 ticks to ~7).
			// Accumulating keeps the ideal schedule (0.2, 0.66, 1.12, ...) so each pulse
			// fires on the first tick at/after its ideal time -> the full 9 ticks land.
			ch.nextPulse += math.Max(ch.interval, 0.4)
			var ms *mobState
			if ch.target > 0 {
				ms = hs.mobs[ch.target]
				if ms != nil && ms.dead {
					ms = nil
				}
			}
			ctx := opCtx{slot: ch.slot, level: ch.level, target: ms,
				px: ch.px, py: ch.py, hasPos: ch.hasPos}
			s.applyOpsLocked(c, ch.ops, ctx, now)
		}
		keep = append(keep, ch)
	}
	hs.channels = keep
}

// ---- bounces (chaining skull, Paralyzing-Cask-style) ----

// skullMoverSpeed replicates the client's SmoothMove mover on the server so hit
// timing matches the visual exactly. The Elgorm skill-1 skull prop
// (VFX_Avtr_Dsb_Elgorm_skill1_prop01) carries a SmoothMove with mBySpeed=true and
// mSpeed=15 (units/sec, read from the prefab in Avtr_Dsb_Elgorm.unity3d): its flight
// time is Distance/mSpeed, independent of the parabolic arc (mMaxHeight) and of the
// target moving (mTargetTime is fixed at launch). skullMinFlight floors near-zero
// hops so a coincident target still shows a brief throw rather than an instant hit.
const (
	skullMoverSpeed = 15.0
	skullMinFlight  = 0.05
)

// skullFlightTime is the server's copy of SmoothMove: seconds for the skull to cross
// the straight-line gap between two points at skullMoverSpeed.
func skullFlightTime(ax, ay, bx, by float32) float64 {
	t := math.Hypot(float64(bx-ax), float64(by-ay)) / skullMoverSpeed
	if t < skullMinFlight {
		t = skullMinFlight
	}
	return t
}

// bounceState is one in-flight chaining skull: it is currently flying toward
// flyingTo and lands its hit at arriveAt, then hops to a RANDOM enemy within radius
// that ISN'T the one it just struck. Like Paralyzing Cask the hop is unpredictable
// and may bounce back to enemies it already hit (so two mobs ping-pong the full step
// budget) -- it only avoids landing on the very mob it is standing on. Fire-and-
// forget once thrown.
type bounceState struct {
	slot, level    int
	remaining      int           // impacts still to land, incl. the skull currently in flight
	arriveAt       float64       // battle-time the in-flight skull reaches flyingTo
	radius         float64       // search radius around the last impact for the next target
	flyingTo       int32         // the mob the skull is flying toward now (damage lands on arrival)
	flyToX, flyToY float32       // that target's position at launch (fallback origin if it dies mid-flight)
	ops            []gamedata.Op // per-hit ops (single-target damage + stun)
	fx             string        // the skull fx spawned for each new hop
}

// startBounceLocked resolves an OpBounce: the skull is thrown from the caster and
// FLIES to the first enemy (the cast target, or nearest to the aim point), then
// keeps hopping to the nearest fresh enemy. Damage lands on IMPACT -- when the skull
// actually reaches each enemy, never on cast -- so every number ticks down as the
// skull touches its target. The first skull fx was already thrown by the payload
// system (PayloadFxAt "target", caster->target); this schedules that first impact,
// and each hop launches its own fx and schedules the next impact on arrival.
//
// The impact times come from skullFlightTime (Distance/skullMoverSpeed), a byte-for-
// byte copy of the client's SmoothMove mover, so server damage and the client visual
// land together at any hop distance.
func (s *Server) startBounceLocked(c *conn, op gamedata.Op, ctx opCtx, now float64) {
	hs := c.huntState
	total := int(op.Count.At(ctx.level))
	if total < 1 {
		total = 1
	}
	first := ctx.target
	if first == nil || first.dead {
		first = s.nearestMobLocked(c, ctx.px, ctx.py, op.Radius, 0)
	}
	if first == nil {
		return // nothing to hit -- the skull fizzles
	}
	cx, cy := c.posAtLocked(float32(now))
	hs.bounces = append(hs.bounces, bounceState{
		slot: ctx.slot, level: ctx.level,
		remaining: total, arriveAt: now + skullFlightTime(cx, cy, first.x, first.y),
		radius: op.Radius, flyingTo: first.id, flyToX: first.x, flyToY: first.y,
		ops: op.Ops, fx: hs.skillDef(ctx.slot).PayloadFx,
	})
}

// nearestMobLocked returns the closest living mob within r of (x,y), skipping
// `exclude` (0 = none), or nil. Used only to resolve the skull's FIRST target when
// the cast had no explicit target (nearest to the aim point).
func (s *Server) nearestMobLocked(c *conn, x, y float32, r float64, exclude int32) *mobState {
	var best *mobState
	bestD := r
	for _, m := range c.huntState.mobs {
		if m.dead || m.id == exclude {
			continue
		}
		if d := math.Hypot(float64(m.x-x), float64(m.y-y)); d <= bestD {
			bestD, best = d, m
		}
	}
	return best
}

// randomMobInRangeLocked picks a uniformly RANDOM living mob within r of (x,y),
// skipping `exclude` (0 = none), or nil if none qualify. This is the skull's hop
// target rule -- Paralyzing-Cask-style, the chain jumps to a random nearby enemy
// (which may be one it already hit), not the nearest, so its path is unpredictable.
// The choice is made once here on the server and the resulting fx is broadcast, so
// every client sees the same hop sequence.
func (s *Server) randomMobInRangeLocked(c *conn, x, y float32, r float64, exclude int32) *mobState {
	var cands []*mobState
	for _, m := range c.huntState.mobs {
		if m.dead || m.id == exclude {
			continue
		}
		if math.Hypot(float64(m.x-x), float64(m.y-y)) <= r {
			cands = append(cands, m)
		}
	}
	if len(cands) == 0 {
		return nil
	}
	return cands[rand.Intn(len(cands))]
}

// tickBouncesLocked advances every in-flight chaining skull: when it reaches its
// target it applies its ops, then hops to a random other enemy in range and re-arms
// -- or ends when its hits are spent or no other enemy is in range.
func (s *Server) tickBouncesLocked(c *conn, now float64) {
	hs := c.huntState
	if len(hs.bounces) == 0 {
		return
	}
	var keep []bounceState
	for _, b := range hs.bounces {
		if b.remaining <= 0 {
			continue
		}
		if now < b.arriveAt {
			keep = append(keep, b)
			continue
		}
		// The in-flight skull just reached its target: apply the hit there (skip if it
		// died mid-flight), then note where the skull now sits for the next search.
		originX, originY := b.flyToX, b.flyToY
		if victim := hs.mobs[b.flyingTo]; victim != nil && !victim.dead {
			s.applyOpsLocked(c, b.ops, opCtx{slot: b.slot, level: b.level,
				target: victim, px: victim.x, py: victim.y, hasPos: true}, now)
			originX, originY = victim.x, victim.y
		}
		b.remaining--
		if b.remaining <= 0 {
			continue // every hit landed
		}
		// Launch the skull at the next enemy -- a RANDOM one in range that isn't the mob
		// it just struck (Paralyzing-Cask hop: unpredictable, may revisit earlier
		// enemies, only fizzles when no other enemy is nearby). It flies from the struck
		// enemy and its hit is scheduled for the moment it arrives (Distance/
		// skullMoverSpeed, matching the client SmoothMove), so damage coincides with the
		// impact.
		next := s.randomMobInRangeLocked(c, originX, originY, b.radius, b.flyingTo)
		if next == nil {
			continue // chain fizzles: no other enemy in range
		}
		if uid := s.fxStartLocked(c, b.fx, b.flyingTo, next.id, false, 0, 0); uid != 0 {
			hs.scheduleFxEnd(uid, now+2)
		}
		b.arriveAt = now + skullFlightTime(originX, originY, next.x, next.y)
		b.flyingTo, b.flyToX, b.flyToY = next.id, next.x, next.y
		keep = append(keep, b)
	}
	hs.bounces = keep
}

// ---- summons ----

// allocSummonID returns a globally-unique summon object id. In a shared world the
// id comes from the instance-wide counter so two members' summons can never
// collide (the mob sim resolves hits/kill-credit by summon id across all members).
// A bare/solo conn with no instance falls back to its own per-conn counter.
func (c *conn) allocSummonID() int32 {
	if c.inst != nil {
		c.inst.nextSummonID++
		return c.inst.nextSummonID
	}
	c.huntState.nextSummonID++
	return c.huntState.nextSummonID
}

func (s *Server) summonLocked(c *conn, op gamedata.Op, ctx opCtx, now float64) {
	hs := c.huntState
	protoID, ok := hs.summonProtos[op.Unit]
	if !ok {
		return
	}
	n := int(op.Count.At(ctx.level))
	if n < 1 {
		n = 1
	}
	// Spawn at the cast point for a ground-targeted summon (Elgorm's «Гвардия
	// мертвых» is aimed where the player clicks); fall back to the caster for a
	// self/untargeted summon.
	cx, cy := c.posAtLocked(float32(now))
	if ctx.hasPos {
		cx, cy = ctx.px, ctx.py
	}
	for i := 0; i < n; i++ {
		id := c.allocSummonID()
		// Give each summon a fixed slot angle on the ring so the group spawns
		// spread out (tightly, around the cast point) and keeps holding distinct
		// positions around the owner/target instead of stacking on one point.
		ang := float64(i) * 2 * math.Pi / float64(n)
		sx := cx + float32(summonSpawnRadius*math.Cos(ang))
		sy := cy + float32(summonSpawnRadius*math.Sin(ang))
		sm := &summonState{
			id: id, proto: protoID, x: sx, y: sy, fang: ang,
			hp: op.HP.At(ctx.level), maxHP: op.HP.At(ctx.level),
			dmg:   op.Dmg.At(ctx.level),
			until: now + op.Lifetime.At(ctx.level),
		}
		hs.summons[id] = sm
		// Render on every ready member (owner + teammates), each in its own tracker.
		// Outside an instance c.members() is [c], so this is the owner-only path.
		for _, mem := range c.members() {
			s.revealSummonToMemberLocked(mem, sm, now)
		}
	}
}

// revealSummonToMemberLocked builds a summon on ONE member's client (tracker add +
// CREATE_OBJECT + stats SYNC + attack effector). No-op if already tracked. Mirrors
// revealMobToMemberLocked -- the summon prototype is party-wide-registered in every
// member's world-state chain, so no PrototypeInfo is needed here.
func (s *Server) revealSummonToMemberLocked(mem *conn, sm *summonState, now float64) {
	hs := mem.huntState
	if hs == nil || hs.tr.index(sm.id) >= 0 {
		return
	}
	idx := hs.tr.add(sm.id)
	frac := float32(1.0)
	if sm.maxHP > 0 {
		frac = float32(math.Max(sm.hp, 0) / sm.maxHP)
	}
	s.push(mem, battleproto.CmdCreateObject, amf.NewArray().
		Set("id", sm.id).Set("proto", sm.proto))
	s.push(mem, battleproto.CmdSync, amf.NewArray().Set("data",
		newSyncBlob(float32(now)).addObject(sm.id).
			position(idx, sm.x, sm.y, sm.vx, sm.vy, float32(now)).
			setFloats(syncHealth, idx, frac).
			setFloats(syncMaxHealth, idx, float32(sm.maxHP)).
			setFloats(syncSpeed, idx, summonSpeed).
			// Attack speed matches the summon's 1/s swing cadence (nextSwing += 1.0).
			// Without it mAttackSpeed=0 on the client freezes the swing clip at playback
			// speed 0 -- the ACTION_DONE re-shows the pose each swing but it never
			// actually animates (same latent bug the ally avatar had).
			setFloats(syncAttackSpeed, idx, 1.0).
			setFloats(syncRadius, idx, float32(summonRadius)).
			setInt(syncTeam, idx, 1).
			build(hs.tr.count())))
	// ATTACK effector so the model swings when we push its ACTIONs.
	s.addAttackEffectorLocked(mem, sm.id, summonAttackProtoID, now)
}

// hideSummonFromMemberLocked removes a summon from ONE member's client (tracker
// swap-remove + SYNC-remove + DELETE_OBJECT). No-op if the member wasn't tracking it.
func (s *Server) hideSummonFromMemberLocked(mem *conn, sm *summonState, now float64) {
	s.untrackObjForMemberLocked(mem, sm.id, float32(now))
}

// removeSummonFromClientsLocked drops a summon from every member's client (expiry,
// death, or owner leave).
func (s *Server) removeSummonFromClientsLocked(c *conn, sm *summonState, now float64) {
	for _, mem := range c.members() {
		s.hideSummonFromMemberLocked(mem, sm, now)
	}
}

func (s *Server) tickSummonsLocked(c *conn, now float64) {
	hs := c.huntState
	for id, sm := range hs.summons {
		if sm.dead {
			continue
		}
		if now > sm.until || sm.hp <= 0 {
			sm.dead = true
			sm.swingDoneAt = 0
			delete(hs.summons, id)
			// Remove from every viewer's client (owner + teammates).
			s.removeSummonFromClientsLocked(c, sm, now)
			continue
		}
		// Close out a finished swing so the client drops the attack-action overlay
		// and the next DO_ACTION can re-trigger the WrapMode.Once attack clip.
		// Without this ACTION_DONE the summon swings exactly once then never
		// animates again (InstanceData.DoAction rejects the duplicate action id).
		// Broadcast so teammates' copies re-trigger too.
		if sm.swingDoneAt > 0 && now >= sm.swingDoneAt {
			sm.swingDoneAt = 0
			s.broadcastObjLocked(c, sm.id, battleproto.CmdActionDone, amf.NewArray().
				Set("id", sm.id).
				Set("action", summonAttackProtoID).
				Set("item", false).
				Set("cooldown", now))
		}
		// Find something to fight.
		var target *mobState
		best := summonSeek
		for _, m := range hs.mobs {
			if m.dead {
				continue
			}
			d := math.Hypot(float64(m.x-sm.x), float64(m.y-sm.y))
			if d < best {
				best, target = d, m
			}
		}
		if target == nil {
			// No enemy nearby: escort the owner like a pet instead of freezing in
			// place. Each summon holds its own slot on a ring AROUND the owner
			// (surrounding them from all sides) rather than every pet piling onto
			// the owner's exact point.
			ax, ay := c.posAtLocked(float32(now))
			gx, gy := sm.ringPoint(ax, ay, summonRing)
			// Near a map edge some ring slots fall off the walkable area; clip each
			// slot back onto the map (from the on-map owner toward the slot) so a pet
			// never targets -- and slides onto -- ground outside the map.
			if c.nav != nil && !c.nav.Walkable(float64(gx), float64(gy)) {
				cgx, cgy := c.nav.Clip(float64(ax), float64(ay), float64(gx), float64(gy))
				gx, gy = float32(cgx), float32(cgy)
			}
			if math.Hypot(float64(gx-sm.x), float64(gy-sm.y)) > summonSlotTol {
				s.moveSummonLocked(c, sm, gx, gy, true, now)
			} else {
				s.moveSummonLocked(c, sm, 0, 0, false, now)
			}
			continue
		}
		reach := meleeReach + summonRadius + target.mob.Radius()
		if best > reach {
			// Approach from this summon's slot angle so several pets surround the
			// mob at melee distance instead of stacking on its centre. Aim just
			// inside the stop radius so best drops below `reach` and it commits to
			// the swing.
			gx, gy := sm.ringPoint(target.x, target.y, math.Max(0.4, reach-0.4))
			s.moveSummonLocked(c, sm, gx, gy, true, now)
			continue
		}
		s.moveSummonLocked(c, sm, 0, 0, false, now)
		if now >= sm.nextSwing {
			sm.nextSwing = now + 1.0
			s.broadcastObjLocked(c, sm.id, battleproto.CmdAction,
				newActionArgs(sm.id, summonAttackProtoID, target.id, now,
					amf.NewArray().Set("x", 0.0).Set("y", 0.0)))
			// Close the swing a hair before the next one so a continuous attacker
			// re-triggers cleanly (mirrors the mob path).
			sm.swingDoneAt = now + 0.9
			s.hitMobLocked(c, target, sm.dmg, sm.id)
		}
	}
}

// ringPoint returns this summon's formation slot: the point at its fixed angle on
// a ring of the given radius around (cx,cy). Anchoring the goal at a per-summon
// angle keeps a group spread out around the owner (escort) or the target (melee
// circle) instead of every unit converging on one point.
func (sm *summonState) ringPoint(cx, cy float32, radius float64) (float32, float32) {
	return cx + float32(radius*math.Cos(sm.fang)), cy + float32(radius*math.Sin(sm.fang))
}

// moveSummonLocked dead-reckons a summon and syncs velocity changes. Movement is
// clamped to walkable ground and routes around walls (same nav treatment as
// mobs) so a summon can't chase its target through geometry.
func (s *Server) moveSummonLocked(c *conn, sm *summonState, tx, ty float32, chase bool, now float64) {
	dt := tickInterval.Seconds()
	px0, py0 := sm.x, sm.y
	sm.x += sm.vx * float32(dt)
	sm.y += sm.vy * float32(dt)
	if c.nav != nil && (sm.vx != 0 || sm.vy != 0) && !c.nav.Walkable(float64(sm.x), float64(sm.y)) {
		cx, cy := c.nav.Clip(float64(px0), float64(py0), float64(sm.x), float64(sm.y))
		sm.x, sm.y = float32(cx), float32(cy)
		// Same as the mob clamp: a clipped summon's cached route may point at an
		// occluded waypoint — exhaust it so aimAlong re-paths from the clipped spot.
		if len(sm.pf.pts) > 0 {
			sm.pf.idx = len(sm.pf.pts)
		}
	}
	var nvx, nvy float32
	if chase {
		blocked := c.nav != nil && !c.mobHasLoSLocked(sm.x, sm.y, tx, ty)
		gx, gy := c.aimAlong(&sm.pf, sm.x, sm.y, tx, ty, blocked, now)
		dx, dy := gx-sm.x, gy-sm.y
		d := float32(math.Hypot(float64(dx), float64(dy)))
		if d > 0.3 {
			nvx, nvy = dx/d*summonSpeed, dy/d*summonSpeed
		}
	} else {
		sm.pf.pts = nil
	}
	if nvx == sm.vx && nvy == sm.vy {
		return
	}
	sm.vx, sm.vy = nvx, nvy
	// Re-sync the new velocity to every viewer, each with its own tracking index.
	s.broadcastPosLocked(c, sm.id, sm.x, sm.y, sm.vx, sm.vy, float32(now))
}

// ---- mob statuses + AI ----

// ensureMobStatusFxLocked starts a looped status VFX on a mob exactly once, storing
// its world-fx id in *slot (idempotent: a nonzero slot means it is already running).
// An empty fx name is a no-op -- worldFxStartLocked returns 0, leaving *slot at 0.
func (s *Server) ensureMobStatusFxLocked(c *conn, m *mobState, slot *int32, fx string) {
	if *slot == 0 {
		*slot = s.worldFxStartLocked(c, fx, m.id, 0, false, 0, 0)
	}
}

// stunMobLocked applies a stun with its generic fx and stops the mob.
func (s *Server) stunMobLocked(c *conn, m *mobState, now, dur float64) {
	m.st.stunUntil = math.Max(m.st.stunUntil, now+dur)
	s.ensureMobStatusFxLocked(c, m, &m.st.stunFx, "StunEffect")
	s.stopMobLocked(c, m, now)
}

// stopMobLocked zeroes a mob's velocity (with sync) if it was moving. The stop
// is broadcast to every member that renders the mob (each with its own index).
func (s *Server) stopMobLocked(c *conn, m *mobState, now float64) {
	if m.vx == 0 && m.vy == 0 {
		return
	}
	m.vx, m.vy = 0, 0
	s.broadcastPosLocked(c, m.id, m.x, m.y, 0, 0, float32(now))
}

// syncMobSpeedLocked pushes the mob's current SPEED stat (slow visuals) to every
// viewer.
func (s *Server) syncMobSpeedLocked(c *conn, m *mobState, now float64) {
	sp := m.mob.Speed * m.st.moveFactor(now) * m.st.modMul(now, "move_speed_pct")
	s.broadcastStatLocked(c, m.id, syncSpeed, float32(sp), float32(now))
}

// mobTarget is the enemy a mob has picked this tick: the nearest member's avatar
// or one of the members' summons, with the owning connection for hit resolution.
type mobTarget struct {
	x, y   float32
	dist   float64
	radius float64
	obj    int32 // objID to swing at (0 = no valid target this tick)
	owner  *conn // member the target belongs to (for hit resolution)
	summon *summonState
}

// mobTargetLocked picks a mob's target across the whole party: the closest of
// every alive member's avatar and every member's summons. dist is +Inf (obj 0)
// when nothing is targetable, which drops the mob out of combat via the leash.
func mobTargetLocked(m *mobState, members []*conn, now float64) mobTarget {
	best := mobTarget{dist: math.Inf(1)}
	for _, mem := range members {
		hs := mem.huntState
		if hs == nil {
			continue
		}
		if hs.deadUntil == 0 {
			px, py := mem.posAtLocked(float32(now))
			if d := math.Hypot(float64(px-m.x), float64(py-m.y)); d < best.dist {
				best = mobTarget{x: px, y: py, dist: d, radius: hs.av.Radius(), obj: mem.objID, owner: mem}
			}
		}
		for _, sm := range hs.summons {
			if sm.dead {
				continue
			}
			if d := math.Hypot(float64(sm.x-m.x), float64(sm.y-m.y)); d < best.dist {
				best = mobTarget{x: sm.x, y: sm.y, dist: d, radius: summonRadius, obj: sm.id, owner: mem, summon: sm}
			}
		}
	}
	return best
}

// resolveMobHitLocked applies a mob's committed basic-attack hit to whatever it
// was aimed at (a member avatar or a summon), looked up by the stored objID.
func (s *Server) resolveMobHitLocked(m *mobState, members []*conn, now float64) {
	for _, mem := range members {
		hs := mem.huntState
		if hs == nil {
			continue
		}
		if mem.objID == m.hitTarget {
			s.hitPlayerLocked(mem, m, m.hitDmg, now)
			return
		}
		if sm, ok := hs.summons[m.hitTarget]; ok && !sm.dead {
			s.hitSummonLocked(mem, m, sm, m.hitDmg, now)
			return
		}
	}
}

// mobTargetPosLocked returns the live position of the objID a ranged mob is aimed
// at (a member avatar or a summon) and whether it was found -- used to size the
// arrow's flight from the mob's current spot at the moment of release.
func mobTargetPosLocked(objID int32, members []*conn, now float64) (float32, float32, bool) {
	for _, mem := range members {
		hs := mem.huntState
		if hs == nil {
			continue
		}
		if mem.objID == objID {
			px, py := mem.posAtLocked(float32(now))
			return px, py, true
		}
		if sm, ok := hs.summons[objID]; ok && !sm.dead {
			return sm.x, sm.y, true
		}
	}
	return 0, 0, false
}

// clampArrowFlight sizes an arrow's flight from the caster→target gap: fast, but
// bounded so a point-blank shot still streaks visibly and a long shot never dawdles.
func clampArrowFlight(dist float64) float64 {
	f := dist / mobArrowSpeed
	if f < minArrowFlight {
		f = minArrowFlight
	}
	if f > maxArrowFlight {
		f = maxArrowFlight
	}
	return f
}

// mobArrowFlightLocked returns how long a just-released arrow should fly to its
// target, from the live gap at release, additionally capped so the hit lands before
// the mob's next swing (which would otherwise overwrite the pending hit and drop the
// damage). The release was scheduled to reserve this room, so the cap rarely bites.
func mobArrowFlightLocked(m *mobState, members []*conn, now float64) float64 {
	dist := m.mob.AttackRange // sensible fallback if the target vanished this tick
	if tx, ty, ok := mobTargetPosLocked(m.projTarget, members, now); ok {
		dist = math.Hypot(float64(tx-m.x), float64(ty-m.y))
	}
	f := clampArrowFlight(dist)
	if hi := m.nextSwing - now - 0.02; hi > 0.04 && f > hi {
		f = hi
	}
	return f
}

// closeMobSwingLocked ends a mob's in-flight basic-attack / cast animation on every
// viewer -- the ACTION_DONE that drops the attack-animation overlay (AnimationExt
// gates the swing clip on DoingAction, so without it the client loops the attack pose
// forever). Idempotent: a no-op when the mob isn't mid-swing. Used by the normal
// per-tick close-out AND by any path that abandons a live swing (leash/give-up on a
// player kill) -- there resetInFlight would otherwise zero swingDoneAt WITHOUT ever
// telling the client the swing ended, leaving the attack clip looping as the mob
// walks home.
func (s *Server) closeMobSwingLocked(c *conn, m *mobState, now float64) {
	if m.swingDoneAt == 0 {
		return
	}
	m.swingDoneAt = 0
	s.broadcastObjLocked(c, m.id, battleproto.CmdActionDone, amf.NewArray().
		Set("id", m.id).
		Set("action", mobAttackProtoID(m.mobIdx)).
		Set("item", false).
		Set("cooldown", now))
}

// tickMobsLocked is the shared mob simulation, run once per world by the instance
// ticker (c is any live member -- all members share the mob set, nav and roster).
// Mobs target the nearest party member and every visual is broadcast to all
// viewers. With one member it is the old single-player pass, unchanged.
func (s *Server) tickMobsLocked(c *conn, now float64) {
	hs := c.huntState
	dt := tickInterval.Seconds()
	members := c.members()

	for _, m := range hs.mobs {
		if m.dead {
			// Revive at the authored spawn point once the respawn timer elapses; the
			// fog-of-war pass re-creates it on the client when the player is near.
			if m.respawnAt > 0 && now >= m.respawnAt {
				s.respawnMobLocked(c, m, now)
			}
			continue
		}
		// Fog of war (unified for the party): create/remove the mob when ANY member
		// is near/far. A mob no member is near runs no AI and pushes nothing.
		s.updateMobVisibilityLocked(c, m, now)
		if !m.shown {
			continue
		}
		st := &m.st

		// Status fx expiry (world-scoped: ended on every viewer).
		if st.stunFx != 0 && now >= st.stunUntil {
			s.worldFxEndLocked(c, st.stunFx)
			st.stunFx = 0
		}
		if st.rootFx != 0 && now >= st.rootUntil {
			s.worldFxEndLocked(c, st.rootFx)
			st.rootFx = 0
		}
		if st.slowFx != 0 && now >= st.slowUntil {
			s.worldFxEndLocked(c, st.slowFx)
			st.slowFx = 0
			s.syncMobSpeedLocked(c, m, now)
		}
		if st.atkSlowFx != 0 && now >= st.atkSlowUntil {
			s.worldFxEndLocked(c, st.atkSlowFx)
			st.atkSlowFx = 0
		}
		if st.silenceFx != 0 && now >= st.silenceUntil {
			s.worldFxEndLocked(c, st.silenceFx)
			st.silenceFx = 0
		}
		var expMods []statMod
		for _, mod := range st.mods {
			if mod.until == 0 || mod.until > now {
				expMods = append(expMods, mod)
			}
		}
		if len(expMods) != len(st.mods) {
			st.mods = expMods
			s.syncMobSpeedLocked(c, m, now)
		}

		// DoT streams. Snapshot the slice first: a tick that kills the mob resets
		// m.st (clearing st.dots) mid-loop, so indexing the live slice would panic.
		// Damage is applied in small dotTickInterval slices and the health bar is
		// synced ONCE per combat tick (5x/sec) so poison bleeds down smoothly instead
		// of lurching once a second. No RECEIVE_HIT per slice -- that would spray an
		// impact particle every tick; the acid VFX already signals the poison. Only
		// the KILLING slice goes through hitMobLocked (for the death sequence).
		dots := st.dots
		var dotKeep []overTime
		drained := false
		for i := range dots {
			if m.dead {
				break
			}
			d := dots[i]
			for d.nextTick <= now && d.nextTick <= d.until && !m.dead {
				slice := d.perSec * dotTickInterval
				if m.hp-slice <= 0 {
					s.hitMobLocked(c, m, slice, d.srcObj) // finishing tick: full death path
				} else {
					m.hp -= slice
					drained = true
				}
				d.nextTick += dotTickInterval
			}
			if !m.dead && d.until > now {
				dotKeep = append(dotKeep, d)
			}
		}
		if m.dead {
			continue
		}
		if drained {
			s.syncMobHealthLocked(c, m)
		}
		st.dots = dotKeep
		// Poison fully cleared -> drop the shared acid visual.
		if len(dotKeep) == 0 && st.dotFx != 0 {
			s.worldFxEndLocked(c, st.dotFx)
			st.dotFx = 0
		}

		// Close out a finished swing: tell every viewer the attack action is done so
		// it drops the attack-animation overlay (and attacker highlight). Runs in
		// every branch, so a mob that leaves attack range mid-swing still stops.
		if m.swingDoneAt > 0 && now >= m.swingDoneAt {
			s.closeMobSwingLocked(c, m, now)
		}

		// Dead-reckon our copy of the mob position, clamped to walkable ground so
		// a chasing mob can never end up standing inside a wall/void.
		px0, py0 := m.x, m.y
		m.x += m.vx * float32(dt)
		m.y += m.vy * float32(dt)
		if c.nav != nil && (m.vx != 0 || m.vy != 0) && !c.nav.Walkable(float64(m.x), float64(m.y)) {
			cx, cy := c.nav.Clip(float64(px0), float64(py0), float64(m.x), float64(m.y))
			m.x, m.y = float32(cx), float32(cy)
			m.vx, m.vy = 0, 0
			// Separation steering can push a route-follower off its verified line
			// into a wall, leaving the cached waypoint occluded from the clipped
			// spot. Exhaust the route so aimAlong re-paths from here this same
			// tick instead of grinding at the corner for up to the 1s staleness
			// window. The len guard keeps the nil-route 1/s retry throttle intact.
			if len(m.pf.pts) > 0 {
				m.pf.idx = len(m.pf.pts)
			}
		}

		if st.stunned(now) {
			continue
		}

		// Leashed and walking home: regenerate and steer back to spawn, ignoring the
		// world -- UNLESS a member has come back inside aggro range, which cancels the
		// retreat and re-engages (with whatever HP the mob has regained by now).
		if m.returning {
			if t := mobTargetLocked(m, members, now); t.obj != 0 && t.dist <= mobAggroRange {
				m.returning = false
				m.aggro = true
			} else {
				s.returnHomeStepLocked(c, m, now)
				continue
			}
		}

		// Target: the nearest of every party member's avatar and their summons.
		tgt := mobTargetLocked(m, members, now)
		tx, ty := tgt.x, tgt.y
		dist := tgt.dist
		targetRadius := tgt.radius

		if !m.aggro {
			if dist <= mobAggroRange {
				m.aggro = true
			} else {
				continue
			}
		}
		if dist > mobLeashRange || tgt.obj == 0 {
			// Gave up: nobody (avatar or summon) is in reach. Walk back to spawn and
			// regenerate to full instead of freezing wherever the chase ended -- the
			// early returning branch above drives it home on subsequent ticks.
			// End any in-flight swing FIRST: a player killed mid-swing drops the mob
			// straight here while swingDoneAt is still in the future, and resetInFlight
			// would zero it without the ACTION_DONE -- leaving the attack clip looping
			// all the way home. closeMobSwingLocked sends it before we clear the state.
			m.aggro = false
			m.returning = true
			s.closeMobSwingLocked(c, m, now)
			m.resetInFlight()
			s.stopMobLocked(c, m, now)
			continue
		}

		// Release a ranged mob's committed arrow once the draw animation ends (its
		// scheduled bow-release point). The client flies the projectile prefab over
		// hit_at-now, so compute the flight NOW from the live gap and set the hit to
		// land on arrival. Capped to resolve before the next swing so the cycle stays
		// clean. Fired from the same "committed" gate as the hit below.
		if m.projLaunchAt > 0 && now >= m.projLaunchAt {
			m.projLaunchAt = 0
			m.hitAt = now + mobArrowFlightLocked(m, members, now)
			s.broadcastObjLocked(c, m.id, battleproto.CmdSetProjectile, amf.NewArray().
				Set("source", m.id).
				Set("target", m.projTarget).
				Set("hit_at", m.hitAt))
		}
		// Land committed hits even if the target has since moved: a swing/cast
		// connects because range (or aim) was met when it STARTED, matching the
		// client, which locks onto the target at the start of the action. Basic
		// attack hit -- resolved to whichever member/summon it was aimed at:
		if m.hitAt > 0 && now >= m.hitAt {
			m.hitAt = 0
			s.resolveMobHitLocked(m, members, now)
		}
		// Boss skill impact:
		if m.skillHitAt > 0 && now >= m.skillHitAt {
			m.skillHitAt = 0
			s.landBossSkillLocked(c, m, members, now)
		}
		// Stand still while an attack/cast animation is playing: a creature never
		// chases or shuffles mid-swing. swingDoneAt is cleared when the animation's
		// ACTION_DONE fires; skillHitAt while a cast wind-up is still pending.
		if m.swingDoneAt > 0 || m.skillHitAt > 0 {
			s.stopMobLocked(c, m, now)
			continue
		}
		// Off cooldown and in range? Cast a skill instead of a basic attack.
		if s.tryBossSkillLocked(c, m, members, now) {
			continue
		}

		// A mob may only attack a target it can actually reach: if a wall sits
		// between them (the mob chased to a chokepoint the player is behind),
		// treat it as out of range so it never hits "through" geometry -- the
		// cause of phantom damage from an unseen attacker. blocked also forces the
		// chase branch, where movement is nav-clipped so the mob stops at the wall.
		blocked := !c.mobHasLoSLocked(m.x, m.y, tx, ty)

		// Reach = base gap + both body radii (client's AvatarAI math): a melee mob
		// uses meleeReach, a ranged mob its own AttackRange. Big bosses reach from
		// farther because their radius is large.
		base := meleeReach
		if m.mob.AttackRange > 0 {
			base = m.mob.AttackRange
		}
		atkRange := base + m.mob.Radius() + targetRadius

		if dist > atkRange || blocked {
			// Stationary mobs (rooted plants/turrets) never chase or reposition:
			// they hold their spawn and wait for the target to re-enter range.
			if m.mob.Stationary || st.rooted(now) {
				s.stopMobLocked(c, m, now)
				continue
			}
			// Chase: with a clear line, head straight at the target; when a wall is
			// in the way (blocked), follow an A* route around it. Either way the
			// per-tick walkable clamp above keeps the mob out of geometry.
			gx, gy := c.aimAlong(&m.pf, m.x, m.y, tx, ty, blocked, now)
			dx, dy := gx-m.x, gy-m.y
			d := float32(math.Hypot(float64(dx), float64(dy)))
			if d < 0.2 {
				s.stopMobLocked(c, m, now) // wall-blocked: hold position
				continue
			}
			sp := m.mob.Speed * st.moveFactor(now) * st.modMul(now, "move_speed_pct")
			// Steer = chase direction + separation from packmates, so a group
			// converging on the target fans out instead of overlapping.
			ux, uy := dx/d, dy/d
			sepx, sepy := hs.mobSeparation(m)
			stx, sty := ux+sepx*mobSepWeight, uy+sepy*mobSepWeight
			sn := float32(math.Hypot(float64(stx), float64(sty)))
			if sn < 1e-3 {
				stx, sty, sn = ux, uy, 1
			}
			nvx, nvy := stx/sn*float32(sp), sty/sn*float32(sp)
			if now-m.lastSync > 0.7 ||
				math.Hypot(float64(nvx-m.vx), float64(nvy-m.vy)) > float64(sp)*0.3 {
				m.vx, m.vy = nvx, nvy
				m.lastSync = now
				s.broadcastMobPosLocked(c, m, now)
			}
			continue
		}

		// In range with a clear path. If crowding a packmate, sidestep so the pack
		// spreads around the target rather than stacking on one point; otherwise
		// hold position. Either way it still swings (attacking while shuffling).
		sepx, sepy := hs.mobSeparation(m)
		sn := float32(math.Hypot(float64(sepx), float64(sepy)))
		if sn > 0.5 && !st.rooted(now) && !m.mob.Stationary {
			sp := m.mob.Speed * st.moveFactor(now) * st.modMul(now, "move_speed_pct")
			m.vx, m.vy = sepx/sn*float32(sp)*0.6, sepy/sn*float32(sp)*0.6
			m.lastSync = now
			s.broadcastMobPosLocked(c, m, now)
		} else {
			s.stopMobLocked(c, m, now)
		}
		if now < st.silenceUntil {
			continue
		}
		atkSpeed := m.mob.AttackSpeed * st.attackFactor(now)
		if now >= m.nextSwing && atkSpeed > 0 {
			m.nextSwing = now + 1/atkSpeed
			target := tgt.obj
			s.broadcastObjLocked(c, m.id, battleproto.CmdAction,
				newActionArgs(m.id, mobAttackProtoID(m.mobIdx), target, now,
					amf.NewArray().Set("x", 0.0).Set("y", 0.0)))
			// Schedule the matching ACTION_DONE so the client stops "doing the attack
			// action" once the swing plays. Without it the client keeps the attack
			// animation overlaid forever (AnimationExt gates it on DoingAction), so a
			// mob that chases out of range still visibly swings. Fire it a hair before
			// the next swing so a continuous attacker re-triggers cleanly.
			animEnd := now + math.Min(0.9/atkSpeed, 1.2)
			m.swingDoneAt = animEnd
			dmg := m.rollDamage()
			m.hitDmg = dmg
			m.hitTarget = target
			if m.mob.AttackRange > 0 {
				// Ranged mobs (skeleton archers, the shooter plant, caster bosses -- any
				// with an explicit AttackRange) fire a visible arrow/bolt so the player sees
				// it cross the gap instead of taking damage from afar. The arrow must leave
				// the bow at the END of the draw animation (like Elgorm's bolt), NOT partway
				// through it, so the release -- and the SET_PROJECTILE that spawns the
				// prefab -- is scheduled for animEnd (the ACTION_DONE moment). Only if that
				// end leaves too little room for a visible flight before the next swing is
				// the release pulled a touch earlier (never before the mid-swing point). The
				// hit lands on ARRIVAL, computed at release from the live gap: unlike a melee
				// swing (fixed mid-swing connect) a ranged hit can't precede its own
				// projectile, so hitAt is left 0 here and set when the arrow flies.
				release := animEnd
				if latest := m.nextSwing - clampArrowFlight(dist) - 0.02; release > latest {
					release = latest
				}
				if earliest := now + 0.5/atkSpeed; release < earliest {
					release = earliest
				}
				m.projLaunchAt = release
				m.projTarget = target
				m.hitAt = 0
			} else {
				// Melee: the weapon connects mid-swing, in place, no projectile.
				m.hitAt = now + 0.5/atkSpeed
			}
		}
	}
}

// broadcastMobPosLocked pushes a mob's current position+velocity to every viewer,
// each addressed by its own tracking index.
func (s *Server) broadcastMobPosLocked(c *conn, m *mobState, now float64) {
	s.broadcastPosLocked(c, m.id, m.x, m.y, m.vx, m.vy, float32(now))
}

// mobSeparation returns a steering push that moves m away from other living mobs
// within mobSepRange, so a pack spreads around the target instead of piling onto
// one point. The magnitude grows as mobs overlap more (0 at mobSepRange, ~1 when
// touching). Two mobs sharing the exact same point are parted deterministically
// by id so they never stay welded together.
func (hs *huntState) mobSeparation(m *mobState) (float32, float32) {
	var sx, sy float32
	for _, o := range hs.mobs {
		// Only shown mobs can be within mobSepRange (1.8m): a hidden mob is >34m
		// away (past mobHideRadius), so skipping them is a free optimization that
		// keeps this O(shown^2) instead of O(shown x total) -- important now that a
		// map can hold thousands of spawns.
		if o == m || o.dead || !o.shown {
			continue
		}
		dx, dy := m.x-o.x, m.y-o.y
		d := float32(math.Hypot(float64(dx), float64(dy)))
		if d >= mobSepRange {
			continue
		}
		if d < 1e-3 {
			a := float64(m.id) * 2.3999632 // golden angle: distinct dir per id
			dx, dy, d = float32(math.Cos(a)), float32(math.Sin(a)), 1
		}
		w := (mobSepRange - d) / mobSepRange
		sx += dx / d * w
		sy += dy / d * w
	}
	return sx, sy
}

// nearestMemberDistLocked is the distance from a mob to the closest party member
// (avatars only; +Inf if none). Drives the unified fog: a mob is shown to the
// whole party while ANY member is near, hidden once EVERY member is far.
func nearestMemberDistLocked(members []*conn, m *mobState, now float64) float64 {
	best := math.Inf(1)
	for _, mem := range members {
		if mem.huntState == nil {
			continue
		}
		px, py := mem.posAtLocked(float32(now))
		if d := math.Hypot(float64(px-m.x), float64(py-m.y)); d < best {
			best = d
		}
	}
	return best
}

// updateMobVisibilityLocked reveals a mob to the party when any member draws
// within mobRevealRadius and hides it once every member is past mobHideRadius
// (fog of war). m.shown is the shared "on the clients / simulate" flag; the
// per-member CREATE/DELETE fan out to every member. With one member this is the
// old single-player pass verbatim.
func (s *Server) updateMobVisibilityLocked(c *conn, m *mobState, now float64) {
	d := nearestMemberDistLocked(c.members(), m, now)
	if m.shown {
		if d >= mobHideRadius {
			s.hideMobLocked(c, m, now)
			// Nobody is near (all members past the hide radius): fully heal and snap
			// home so the mob is pristine at its spawn when the party returns. Covers
			// the case a mob is abandoned mid-fight faster than the leash walk-home can
			// finish (blink/teleport away, party wipe far off).
			m.resetToSpawn()
			return
		}
		s.updateMobShadeLocked(c, m, d)
	} else if d <= mobRevealRadius {
		s.revealMobLocked(c, m, now)
		s.updateMobShadeLocked(c, m, d)
	}
}

// updateMobShadeLocked toggles the translucent fog-ring overlay on a shown mob
// (world-scoped, so every member sees it identically): mobShadeFx (50%-alpha
// DeathShader) once the NEAREST member drifts to the outer reveal ring, ended as
// the party closes in. Hysteresis prevents a per-tick flicker at the boundary.
func (s *Server) updateMobShadeLocked(c *conn, m *mobState, d float64) {
	if m.shaded {
		if d <= mobUnshadeRadius {
			s.worldFxEndLocked(c, m.shadeFxUID)
			m.shadeFxUID = 0
			m.shaded = false
		}
	} else if d >= mobShadeRadius {
		m.shadeFxUID = s.worldFxStartLocked(c, mobShadeFx, m.id, 0, false, 0, 0)
		m.shaded = true
	}
}

// revealMobToMemberLocked creates the mob on ONE member's client (tracker add +
// CREATE_OBJECT + stats SYNC + attack effector). No-op if already tracked -- used
// both by the fog pass (fan out to all) and when a newcomer joins an ongoing world.
func (s *Server) revealMobToMemberLocked(mem *conn, m *mobState, now float64) {
	hs := mem.huntState
	if hs == nil || hs.tr.index(m.id) >= 0 {
		return
	}
	bt := float32(now)
	idx := hs.tr.add(m.id)
	maxHP := m.maxHealth()
	hpFrac := float32(1.0)
	if maxHP > 0 {
		hpFrac = float32(m.hp / maxHP)
	}
	dmgLo, dmgHi := m.dmgRange()
	s.push(mem, battleproto.CmdCreateObject, amf.NewArray().
		Set("id", m.id).Set("proto", mobProtoID(m.mobIdx)))
	s.push(mem, battleproto.CmdSync, amf.NewArray().Set("data",
		newSyncBlob(bt).addObject(m.id).
			position(idx, m.x, m.y, m.vx, m.vy, bt).
			setFloats(syncHealth, idx, hpFrac).
			setFloats(syncMaxHealth, idx, float32(maxHP)).
			setFloats(syncDmgMin, idx, float32(dmgLo)).
			setFloats(syncDmgMax, idx, float32(dmgHi)).
			setFloats(syncAttackSpeed, idx, float32(m.mob.AttackSpeed)).
			setFloats(syncSpeed, idx, float32(m.mob.Speed)).
			setFloats(syncRadius, idx, float32(m.mob.Radius())).
			setInt(syncTeam, idx, -1).
			build(hs.tr.count())))
	s.addAttackEffectorLocked(mem, m.id, mobAttackProtoID(m.mobIdx), now)
}

// revealMobLocked marks the mob shown and creates it on every party member's
// client.
func (s *Server) revealMobLocked(c *conn, m *mobState, now float64) {
	m.shown = true
	for _, mem := range c.members() {
		s.revealMobToMemberLocked(mem, m, now)
	}
	m.lastSync = now
}

// respawnMobLocked revives a dead mob at its authored spawn point with full HP,
// all state cleared. It stays hidden (m.shown=false) so the fog-of-war pass
// re-creates it on the client the next tick if the player is in range.
// resetInFlight zeros a mob's in-flight basic-attack and skill state (the fields a
// committed-but-unresolved swing/cast reads). Shared by the respawn and
// return-home paths so the two can't drift as new in-flight fields are added.
func (m *mobState) resetInFlight() {
	m.hitAt, m.hitDmg, m.hitTarget = 0, 0, 0
	m.projLaunchAt, m.projTarget = 0, 0
	m.swingDoneAt, m.nextSwing = 0, 0
	m.skillHitAt, m.skillDmg, m.skillRadius = 0, 0, 0
}

// resetToSpawn restores a LIVING mob to a pristine state at its spawn point: full
// HP, no aggro/return, no statuses, no in-flight action, cooldowns cleared. Called
// when the party abandons it past the fog-hide radius so a re-revealed mob is
// always fresh at home. The mob is hidden when this runs, so no client sync is
// needed -- the fog pass re-creates it at these values next time a member nears.
func (m *mobState) resetToSpawn() {
	m.x, m.y = m.spawnX, m.spawnY
	m.vx, m.vy = 0, 0
	m.hp = m.maxHealth()
	m.aggro = false
	m.returning = false
	m.st = unitStatus{}
	m.pf = pathState{}
	m.resetInFlight()
	m.lastSync = 0
	for i := range m.skillReady {
		m.skillReady[i] = 0
	}
}

func (s *Server) respawnMobLocked(c *conn, m *mobState, now float64) {
	m.dead = false
	m.respawnAt = 0
	m.hp = m.maxHealth()
	m.x, m.y = m.spawnX, m.spawnY
	m.vx, m.vy = 0, 0
	m.aggro = false
	m.returning = false
	m.shown = false
	m.shaded = false
	m.shadeFxUID = 0
	m.st = unitStatus{}
	m.pf = pathState{}
	m.resetInFlight()
	m.lastSync = 0
	for i := range m.skillReady {
		m.skillReady[i] = 0
	}
}

// returnMobHomeLocked sends a LIVING mob back to its spawn and drops any in-flight
// action, hiding it so the fog pass re-reveals it at home next tick. Used to evict
// mobs that were dragged onto a respawn checkpoint mid-fight (their HP/dead state is
// preserved -- this is a reposition, not a respawn).
func (s *Server) returnMobHomeLocked(c *conn, m *mobState, now float64) {
	m.x, m.y = m.spawnX, m.spawnY
	m.vx, m.vy = 0, 0
	m.aggro = false
	m.returning = false
	m.pf = pathState{}
	m.resetInFlight()
	if m.shown {
		s.hideMobLocked(c, m, now)
	}
}

// returnHomeStepLocked advances a leashed (returning) mob one tick toward its
// spawn: it regenerates HP (broadcast so the bar visibly refills for anyone
// watching it retreat) and steers home, routing around walls like the chase path.
// On arrival it tops off to full HP and clears the returning state. A returning
// mob ignores all targets -- the caller only enters here while no member is inside
// aggro range.
func (s *Server) returnHomeStepLocked(c *conn, m *mobState, now float64) {
	maxHP := m.maxHealth()
	// Regenerate toward full while walking back.
	if m.hp < maxHP {
		m.hp = math.Min(maxHP, m.hp+maxHP*mobReturnRegenPerSec*tickInterval.Seconds())
		s.syncMobHealthLocked(c, m)
	}
	sp := m.mob.Speed // reset ignores slows -- it heads home at full pace
	// Arrive when within a single step of home; a fixed per-tick step would
	// otherwise overshoot the sub-step epsilon and oscillate across the spawn point.
	arrive := math.Max(mobHomeEpsilon, sp*tickInterval.Seconds())
	dx, dy := m.spawnX-m.x, m.spawnY-m.y
	if math.Hypot(float64(dx), float64(dy)) <= arrive {
		// Home: finish the reset (guarantee full HP even if regen hadn't caught up).
		m.x, m.y = m.spawnX, m.spawnY
		m.returning = false
		m.pf = pathState{}
		if m.hp < maxHP {
			m.hp = maxHP
			s.syncMobHealthLocked(c, m)
		}
		s.stopMobLocked(c, m, now)
		return
	}
	// Steer home: straight line when clear, A* route around walls when blocked.
	blocked := c.nav != nil && !c.mobHasLoSLocked(m.x, m.y, m.spawnX, m.spawnY)
	gx, gy := c.aimAlong(&m.pf, m.x, m.y, m.spawnX, m.spawnY, blocked, now)
	sdx, sdy := gx-m.x, gy-m.y
	d := float32(math.Hypot(float64(sdx), float64(sdy)))
	if d < 0.2 {
		s.stopMobLocked(c, m, now) // wall-blocked corner: hold, re-path next tick
		return
	}
	nvx, nvy := sdx/d*float32(sp), sdy/d*float32(sp)
	if now-m.lastSync > 0.7 ||
		math.Hypot(float64(nvx-m.vx), float64(nvy-m.vy)) > float64(sp)*0.3 {
		m.vx, m.vy = nvx, nvy
		m.lastSync = now
		s.broadcastMobPosLocked(c, m, now)
	}
}

// hideMobFromMemberLocked removes the mob from ONE member's client (tracker
// swap-remove + SYNC-remove + DELETE_OBJECT). No-op if the member wasn't tracking it.
func (s *Server) hideMobFromMemberLocked(mem *conn, m *mobState, now float64) {
	s.untrackObjForMemberLocked(mem, m.id, float32(now))
}

// removeMobFromClientsLocked drops the mob from every member's client (fog-hide or
// corpse cleanup). The DELETE_OBJECT tears down the fog-ring fx child with the
// model, so the shade state is just cleared (no EFFECT_END needed).
func (s *Server) removeMobFromClientsLocked(c *conn, m *mobState, now float64) {
	m.shown = false
	m.shaded = false
	m.shadeFxUID = 0
	for _, mem := range c.members() {
		s.hideMobFromMemberLocked(mem, m, now)
	}
}

// hideMobLocked removes the mob from every client and quiesces it (a hidden mob is
// far past leash range, so it drops aggro and any in-flight action).
func (s *Server) hideMobLocked(c *conn, m *mobState, now float64) {
	m.aggro = false
	m.vx, m.vy = 0, 0
	m.hitAt = 0
	m.projLaunchAt = 0
	m.skillHitAt = 0
	m.swingDoneAt = 0
	s.removeMobFromClientsLocked(c, m, now)
}

// nearestMemberLocked returns the closest alive party member's avatar to the mob
// (nil if the whole party is down), with its distance and position. Bosses aim
// their abilities at this member.
func nearestMemberLocked(members []*conn, m *mobState, now float64) (best *conn, dist float64, bx, by float32) {
	dist = math.Inf(1)
	for _, mem := range members {
		hs := mem.huntState
		if hs == nil || hs.deadUntil > 0 {
			continue
		}
		px, py := mem.posAtLocked(float32(now))
		if d := math.Hypot(float64(px-m.x), float64(py-m.y)); d < dist {
			best, dist, bx, by = mem, d, px, py
		}
	}
	return
}

// tryBossSkillLocked casts the first ready boss ability whose range covers the
// nearest party member, playing its attack animation, spawning its VFX, and
// scheduling the hit after the wind-up. Returns true if a skill was started this
// tick (the caller then skips the basic-attack/chase branch).
func (s *Server) tryBossSkillLocked(c *conn, m *mobState, members []*conn, now float64) bool {
	if len(m.mob.Skills) == 0 || m.skillHitAt > 0 {
		return false
	}
	tgt, dist, px, py := nearestMemberLocked(members, m, now)
	if tgt == nil {
		return false // no target: the whole party is down
	}
	if !c.mobHasLoSLocked(m.x, m.y, px, py) {
		return false // don't cast through walls
	}
	for i := range m.mob.Skills {
		sk := m.mob.Skills[i]
		if now < m.skillReady[i] || dist > sk.Range {
			continue
		}
		m.skillReady[i] = now + sk.Cooldown
		// Play an attack animation (reuse the boss's attack action) turned to the
		// target, broadcast to every viewer, and close it after the wind-up.
		s.broadcastObjLocked(c, m.id, battleproto.CmdAction,
			newActionArgs(m.id, mobAttackProtoID(m.mobIdx), tgt.objID, now,
				amf.NewArray().Set("x", float64(px)).Set("y", float64(py))))
		m.swingDoneAt = now + math.Min(sk.CastTime+0.4, 1.5)
		// Spawn the ability VFX (prefab ships in the boss bundle), world-scoped so the
		// whole party sees the telegraph: at the target's feet for lobbed/ranged
		// skills, on the boss for point-blank ones.
		if sk.OnTarget {
			s.worldFxStartLocked(c, sk.Fx, m.id, tgt.objID, true, px, py)
		} else {
			s.worldFxStartLocked(c, sk.Fx, m.id, 0, false, 0, 0)
		}
		// Schedule the impact; root the boss during the cast.
		m.skillHitAt = now + sk.CastTime
		m.skillDmg = sk.Dmg
		m.skillRadius = sk.Radius
		m.skillCX, m.skillCY = px, py
		m.skillTargetObj = tgt.objID
		s.stopMobLocked(c, m, now)
		if debugCombat {
			log.Printf("battle: %s boss %d casts %q (dmg %.0f radius %.1f)",
				c.RemoteAddr(), m.id, sk.Name, sk.Dmg, sk.Radius)
		}
		return true
	}
	return false
}

// landBossSkillLocked applies a boss skill's damage when its wind-up completes. An
// AoE (Radius>0) hits every member still within Radius of the cast point (so
// moving out dodges it); a single-target skill hits the member it was aimed at.
func (s *Server) landBossSkillLocked(c *conn, m *mobState, members []*conn, now float64) {
	if m.skillRadius > 0 {
		for _, mem := range members {
			hs := mem.huntState
			if hs == nil || hs.deadUntil > 0 {
				continue
			}
			px, py := mem.posAtLocked(float32(now))
			if math.Hypot(float64(px-m.skillCX), float64(py-m.skillCY)) <= m.skillRadius {
				s.hitPlayerLocked(mem, m, m.skillDmg, now)
			}
		}
		return
	}
	for _, mem := range members {
		if mem.objID == m.skillTargetObj {
			if hs := mem.huntState; hs != nil && hs.deadUntil == 0 {
				s.hitPlayerLocked(mem, m, m.skillDmg, now)
			}
			return
		}
	}
}

// mobHasLoSLocked reports whether a straight walkable segment connects the mob
// to (tx,ty): nav.Clip returns the farthest reachable point, so if it stops well
// short of the target a wall is in the way.
func (c *conn) mobHasLoSLocked(mx, my, tx, ty float32) bool {
	if c.nav == nil {
		return true
	}
	cx, cy := c.nav.Clip(float64(mx), float64(my), float64(tx), float64(ty))
	return math.Hypot(cx-float64(tx), cy-float64(ty)) < 1.0
}

// ---- damage to the player / summons, death, respawn ----

func (s *Server) hitPlayerLocked(c *conn, m *mobState, dmg float64, now float64) {
	hs := c.huntState
	if hs.deadUntil > 0 {
		return
	}
	if debugCombat {
		alive := 0
		for _, mm := range hs.mobs {
			if !mm.dead {
				alive++
			}
		}
		px, py := c.posAtLocked(float32(now))
		log.Printf("battle: %s HIT-BY mob=%d mobpos=(%.1f,%.1f) selfpos=(%.1f,%.1f) dmg=%.0f aliveMobs=%d",
			c.RemoteAddr(), m.id, m.x, m.y, px, py, dmg, alive)
	}
	// Dodge?
	if rand.Float64() < hs.st.modSum(now, "dodge_pct") {
		s.broadcastAvatarObjLocked(c, battleproto.CmdReceiveHit, amf.NewArray().
			Set("object", c.objID).
			Set("damager", m.id).
			Set("flags", int32(1)).
			Set("damage", 0.0))
		return
	}
	armor := (hs.av.PhysArmor + hs.st.modSum(now, "phys_armor")) * hs.st.modMul(now, "armor_pct")
	dmg *= 1 - armor/(armor+50)
	dmg = hs.st.absorb(now, dmg)
	if thorns := hs.st.modSum(now, "thorns_pct"); thorns > 0 && dmg > 0 {
		s.hitMobLocked(c, m, dmg*thorns, c.objID)
	}
	if dmg <= 0 {
		return
	}
	hs.hp -= dmg
	s.broadcastAvatarObjLocked(c, battleproto.CmdReceiveHit, amf.NewArray().
		Set("object", c.objID).
		Set("damager", m.id).
		Set("flags", int32(0)).
		Set("damage", dmg))
	if hs.hp <= 0 {
		hs.hp = 0
		s.syncSelfLocked(c, syncHealth)
		s.playerDieLocked(c, m.id, now)
		return
	}
	s.syncSelfLocked(c, syncHealth)
}

func (s *Server) hitSummonLocked(c *conn, m *mobState, sm *summonState, dmg float64, now float64) {
	sm.hp -= dmg
	// Fan the hit flash + HP-bar drop out to every viewer (owner + teammates), each
	// with its own tracking index for the health SYNC.
	s.broadcastObjLocked(c, sm.id, battleproto.CmdReceiveHit, amf.NewArray().
		Set("object", sm.id).
		Set("damager", m.id).
		Set("flags", int32(0)).
		Set("damage", dmg))
	frac := float32(math.Max(sm.hp, 0) / sm.maxHP)
	s.broadcastStatLocked(c, sm.id, syncHealth, frac, float32(now))
	if sm.hp <= 0 {
		s.broadcastObjLocked(c, sm.id, battleproto.CmdOnKill, amf.NewArray().
			Set("killer", m.id).Set("id", sm.id))
		// removal happens on the next summon tick
	}
}

// tryReviveLocked resurrects the player in place instead of dying, when a learned
// OpRevive passive (Zamaran's «Возрождение») is off its internal cooldown. It
// restores the op's HP (scaled by powerMul, capped at max HP), starts the internal
// cooldown, plays the skill's resurrection fx, and returns true so the death
// sequence is skipped. Called at the top of playerDieLocked so it covers every
// death path (not just the direct melee hit).
func (s *Server) tryReviveLocked(c *conn, now float64) bool {
	hs := c.huntState
	slot := hs.reviveSlot
	if slot == 0 {
		return false
	}
	level := int(hs.skillLevel[slot-1])
	if level < 1 { // unlearned ult slot
		return false
	}
	if now < hs.reviveReadyAt {
		return false
	}
	def := hs.skillDef(slot)
	var rev *gamedata.Op
	for i := range def.Ops {
		if def.Ops[i].Kind == gamedata.OpRevive {
			rev = &def.Ops[i]
			break
		}
	}
	if rev == nil {
		return false
	}
	heal := rev.Value.At(level) * hs.powerMul()
	if maxHP := hs.maxHPLocked(now); heal > maxHP {
		heal = maxHP
	}
	if heal <= 0 {
		return false
	}
	hs.hp = heal
	hs.reviveReadyAt = now + rev.Dur.At(level)
	// Resurrection cleanses lingering DoTs so the revived hero doesn't instantly bleed
	// back out (mirrors the death cleanse below, without the death/respawn).
	hs.st.dots = nil
	s.syncSelfLocked(c, syncHealth)
	if uid := s.fxStartLocked(c, def.CastFx, c.objID, 0, false, 0, 0); uid != 0 {
		hs.scheduleFxEnd(uid, now+1.5)
	}
	return true
}

// ccImmuneBlockLocked reports whether an incoming crowd-control effect on the player
// is blocked by a learned OpImmune passive (Wilfang's «Защитный покров» — immunity
// to root/stun/silence), consuming it and starting its recovery cooldown. Returns
// false when there is no immunity or it is on cooldown, so the caller then applies
// the CC. No mob applies player-facing CC in the PvE hunt today, so this gate is
// latent until such a source is added; it is exercised directly by unit tests.
func (s *Server) ccImmuneBlockLocked(c *conn, now float64) bool {
	hs := c.huntState
	slot := hs.ccImmuneSlot
	if slot == 0 {
		return false
	}
	level := int(hs.skillLevel[slot-1])
	if level < 1 {
		return false
	}
	if now < hs.ccImmuneReadyAt {
		return false
	}
	def := hs.skillDef(slot)
	for _, op := range def.Ops {
		if op.Kind == gamedata.OpImmune {
			hs.ccImmuneReadyAt = now + op.Dur.At(level)
			return true
		}
	}
	return false
}

// playerDieLocked runs the death sequence: everything cancels, the corpse
// shows for corpseHide seconds, RESPAWN announces the timer, and the tick
// revives at deadUntil.
func (s *Server) playerDieLocked(c *conn, killer int32, now float64) {
	hs := c.huntState
	// Auto-revive passive (Zamaran): resurrect in place instead of dying.
	if s.tryReviveLocked(c, now) {
		return
	}
	hs.deadUntil = now + respawnDelay
	hs.diedAt = now
	hs.corpseHidden = false

	s.cancelOrderLocked(c)
	hs.channels = nil
	for slot := 1; slot <= 4; slot++ {
		if hs.toggleOn[slot-1] {
			s.toggleOffLocked(c, slot, now, false)
		}
	}
	s.stopAttackLocked(c, true)
	c.stopArrivalLocked()
	c.hasDest = false
	cx, cy := c.posAtLocked(float32(now))
	c.x, c.y, c.vx, c.vy, c.snapT = cx, cy, 0, 0, float32(now)

	// Drop timed buffs (icons/fx) — death cleanses.
	var keep []statMod
	for _, mod := range hs.st.mods {
		if mod.until == 0 {
			keep = append(keep, mod)
			continue
		}
		if mod.buffEffID != 0 {
			s.push(c, battleproto.CmdRemEffector, amf.NewArray().Set("id", mod.buffEffID))
		}
		s.fxEndLocked(c, mod.fxUID)
	}
	hs.st.mods = keep
	hs.st.shield = 0
	hs.st.hots = nil // death cleanses; stop healing the corpse
	hs.st.dots = nil

	// Freeze the avatar in place (velocity 0) BEFORE ON_KILL so the client stops
	// dead-reckoning it and the death clip plays at the death spot instead of
	// sliding along the last run leg.
	c.sendPosLocked(s, cx, cy, 0, 0, float32(now))

	// Broadcast the death so teammates see this avatar drop (the client plays the
	// death clip on the object they render for this player).
	s.broadcastAvatarObjLocked(c, battleproto.CmdOnKill, amf.NewArray().
		Set("killer", killer).Set("id", c.objID))
	// Respawn timer (absolute battle time) is this player's own UI only.
	s.push(c, battleproto.CmdRespawn, amf.NewArray().
		Set("id", c.selfPlayerID).
		Set("time", hs.deadUntil))
}

// respawnPlayerLocked revives at the spawn point: the SYNC re-add re-enables
// the disabled GameObject client-side (GameObjManager.EnableGameObject:
// DeathBehaviour.Reborn + Animation.Reset + NetSyncTransform on), the no-time
// RESPAWN plays the reborn fx, and — because the client drops every effector
// of an object that left relevance — the whole effector set is re-sent.
// rebornActivateRange is how close the avatar must get to a Reborn_point for it
// to become the active respawn checkpoint.
const rebornActivateRange = 6.0

// tickRebornLocked activates the nearest Reborn_point the avatar is standing on,
// making it the respawn checkpoint (and deactivating the previous). The player
// starts on the battle-start Reborn and picks up later ones as they advance.
func (s *Server) tickRebornLocked(c *conn, now float64) {
	hs := c.huntState
	if hs == nil || hs.deadUntil > 0 || len(hs.m.Reborn) == 0 {
		return
	}
	px, py := c.posAtLocked(float32(now))
	for i, r := range hs.m.Reborn {
		if i == hs.rebornIdx {
			continue
		}
		if math.Hypot(float64(px)-r.X, float64(py)-r.Y) <= rebornActivateRange {
			hs.rebornIdx = i
			hs.respawnX, hs.respawnY = float32(r.X), float32(r.Y)
			if debugCombat {
				log.Printf("battle: %s reborn checkpoint %d activated at (%.1f,%.1f)",
					c.RemoteAddr(), i, r.X, r.Y)
			}
			break
		}
	}
}

func (s *Server) respawnPlayerLocked(c *conn, now float64) {
	hs := c.huntState
	hs.deadUntil = 0
	hs.diedAt = 0
	hs.hp = hs.maxHPLocked(now)
	hs.mana = hs.maxManaLocked(now)

	// Respawn at the last-activated checkpoint (falls back to Spawn() before any
	// Reborn_point has been reached / when the map has no checkpoints).
	sx, sy := hs.respawnX, hs.respawnY
	if sx == 0 && sy == 0 {
		sx64, sy64 := hs.m.Spawn()
		sx, sy = float32(sx64), float32(sy64)
	}
	c.x, c.y, c.vx, c.vy, c.snapT = sx, sy, 0, 0, float32(now)

	if hs.corpseHidden {
		idx := hs.tr.add(c.objID)
		s.push(c, battleproto.CmdSync, amf.NewArray().Set("data",
			newSyncBlob(float32(now)).addObject(c.objID).
				position(idx, sx, sy, 0, 0, float32(now)).
				build(hs.tr.count())))
		hs.corpseHidden = false
	} else if idx := hs.tr.index(c.objID); idx >= 0 {
		s.push(c, battleproto.CmdSync, amf.NewArray().Set("data",
			newSyncBlob(float32(now)).position(idx, sx, sy, 0, 0, float32(now)).
				build(hs.tr.count())))
	}
	s.push(c, battleproto.CmdRespawn, amf.NewArray().Set("id", c.selfPlayerID))
	s.pushPlayerStatsLocked(c, now)
	s.syncSelfLocked(c, syncHealth, syncMana)
	s.sendEffectorsLocked(c, now)
	// Re-show this avatar on every teammate's client (its corpse was removed from
	// them on death); renderAvatarForLocked is a no-op for any who still track it.
	for _, other := range c.members() {
		if other != c {
			s.renderAvatarForLocked(other, c, now)
		}
	}
	// Mobs lose interest in the fresh corpse walker until re-aggroed; and any pack
	// dragged onto this checkpoint mid-fight is sent home, so the player never
	// resurrects inside an aggro bubble (spawn-camp death loop). The mob-free spawn
	// ring only controls where mobs SPAWN -- leash is player-relative, so without
	// this a chased pack freezes on the checkpoint and re-aggros the fresh player.
	for _, m := range hs.mobs {
		m.aggro = false
		if !m.dead && math.Hypot(float64(m.x-sx), float64(m.y-sy)) < respawnEvictRange {
			s.returnMobHomeLocked(c, m, now)
		}
	}
}
