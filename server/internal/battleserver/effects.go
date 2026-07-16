package battleserver

import (
	"log"
	"math"
	"sort"

	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
)

// The skill-effect engine. Wire recipe per cast (verified against the
// decompiled client):
//
//	DO_ACTION reply (echo, status true)          -- client arms the order
//	ACTION {id, action, targetObj, start, ...}   -- rotates model, opens action
//	EFFECT_START {effect, owner, fx, args}       -- cast anim + VFX (fx registry)
//	  ... payload at PayloadDelay: EFFECT_START payload fx + ops applied ...
//	ACTION_DONE {id, action, cooldown}           -- closes anim, sets cooldown
//	ORDER_DONE {id, action}                      -- unblocks AvatarAI orders
//	EFFECT_END {id}                              -- stops looped/stop-flagged fx
//
// Statuses on mobs are visualized with the registry's generic loops
// (StunEffect, SlowMoveEffect, ...); on the self player additionally with
// BUFF-type effectors (icons) and stat SYNCs the HUD reads live.

// ---- fx helpers ----

// fxStartLocked pushes EFFECT_START and returns its uid (0 if fx is empty).
//
// In a shared world EVERY player-owned visual (cast splash, payload impact, toggle
// aura, buff glow, shield, trap, morph) must reach teammates -- and, crucially, the
// cast ANIMATION itself is driven client-side by the fx (Skill.StartEffects plays
// mEffect.mAnimation on the owner), not by the ACTION. So an instance member routes
// through the world-scoped path (one EFFECT_START to every member that renders the
// owner, one shared uid, endable everywhere). Solo / bare-conn (c.inst == nil) keeps
// the per-connection path below, byte-for-byte identical. worldFxStartLocked falls
// back here when c.inst == nil, so the two never recurse.
func (s *Server) fxStartLocked(c *conn, fx string, owner, target int32, hasPos bool, px, py float32) int32 {
	if fx == "" {
		return 0
	}
	if c.inst != nil {
		return s.worldFxStartLocked(c, fx, owner, target, hasPos, px, py)
	}
	hs := c.huntState
	hs.nextFxUID++
	uid := hs.nextFxUID
	args := amf.NewArray()
	if target != 0 {
		args.Set("target", target)
	}
	if hasPos {
		args.Set("targetPos", amf.NewArray().Set("x", float64(px)).Set("y", float64(py)))
	}
	s.push(c, battleproto.CmdEffectStart, amf.NewArray().
		Set("effect", uid).
		Set("owner", owner).
		Set("fx", fx).
		Set("args", args))
	return uid
}

// fxEndLocked pushes EFFECT_END for a live effect uid (no-op for 0). Note the
// wire key is "id" here, not "effect" (EffectEndArgParser). In an instance it ends
// on every member (worldFxEndLocked) so a teammate's copy of a persistent player fx
// is torn down too; an unknown uid is a harmless client no-op. Solo path unchanged.
func (s *Server) fxEndLocked(c *conn, uid int32) {
	if uid == 0 {
		return
	}
	if c.inst != nil {
		s.worldFxEndLocked(c, uid)
		return
	}
	s.push(c, battleproto.CmdEffectEnd, amf.NewArray().Set("id", uid))
}

// scheduleFxEnd ends a cast fx after d seconds of battle time (via the tick
// loop's timed queue, so it survives bursts and honors mvMu).
func (hs *huntState) scheduleFxEnd(uid int32, at float64) {
	if uid == 0 {
		return
	}
	hs.fxEnds = append(hs.fxEnds, fxEnd{uid: uid, at: at})
}

type fxEnd struct {
	uid int32
	at  float64
}

// ---- orders (approach-then-cast) ----

// pendingCast is a skill order waiting for the avatar to get in range.
type pendingCast struct {
	slot   int
	target int32 // mob id, 0 for point/self casts
	px, py float32
	hasPos bool
}

// orderDoneLocked tells AvatarAI the order finished (mandatory: without it the
// client-side DEFENCE auto-attack blocks forever on a non-empty order list).
func (s *Server) orderDoneLocked(c *conn, action int32) {
	s.push(c, battleproto.CmdOrderDone, amf.NewArray().
		Set("id", c.objID).Set("action", action))
}

// ---- cast pipeline ----

// startSkillOrderLocked validates a DO_ACTION for a skill and either casts
// now, or starts the approach chase. Caller holds mvMu.
func (s *Server) startSkillOrderLocked(c *conn, slot int, target int32, px, py float32, hasPos bool) {
	hs := c.huntState
	def := hs.skillDef(slot)
	now := float64(s.battleTime())
	parent := skillProtoID(hs.av, slot)

	// A new order supersedes any pending approach-cast: flush the old one's
	// ORDER_DONE first (else its client mOrders entry leaks and AvatarAI's
	// DEFENCE auto-attack hangs forever). Mirrors startAttackLocked.
	s.cancelOrderLocked(c)

	if hs.deadUntil > now || hs.st.stunned(now) || hs.st.silenced(now) {
		s.orderDoneLocked(c, parent)
		return
	}
	if def.Type == "PASSIVE" {
		s.orderDoneLocked(c, parent)
		return
	}
	if def.Type == "TOGGLE" {
		s.toggleSkillLocked(c, slot)
		return
	}
	level := int(hs.skillLevel[slot-1])
	// A rank-0 skill is UNLEARNED (the ult before avatar level 5) -- uncastable.
	if level < 1 || now < hs.cooldownUntil[slot-1] || hs.mana < skillManaCost(float64(def.ManaCost[level-1])) {
		s.orderDoneLocked(c, parent)
		return
	}

	// Self casts fire instantly on the caster, ignoring target/position. The
	// client always ships a targetPos ({0,0} for a none-target cast -- see
	// BattleServerConnection.SendDoAction), so we must key this off the skill's
	// own target type, NOT the presence of a position, or the avatar would run
	// toward the origin treating {0,0} as a ground-target point.
	if def.Target == "" || def.Target == "SELF" {
		s.execCastLocked(c, slot, nil, 0, 0, false)
		return
	}

	// Resolve where the cast lands.
	var ms *mobState
	tx, ty := px, py
	if target > 0 && target != c.objID {
		ms = hs.mobs[target]
		if ms == nil || ms.dead {
			s.orderDoneLocked(c, parent)
			return
		}
		tx, ty = ms.x, ms.y
		hasPos = true
	}

	// A unit-target skill cast with no valid target/position: fire in place.
	if ms == nil && !hasPos {
		s.execCastLocked(c, slot, nil, 0, 0, false)
		return
	}

	// In range? Cast. Otherwise chase toward the cast point.
	cx, cy := c.posAtLocked(s.battleTime())
	maxDist := float64(def.Distance)
	if maxDist <= 0 {
		maxDist = 2.5
	}
	if math.Hypot(float64(tx-cx), float64(ty-cy)) <= maxDist+0.5 {
		s.execCastLocked(c, slot, ms, tx, ty, hasPos)
		return
	}
	hs.order = &pendingCast{slot: slot, target: target, px: px, py: py, hasPos: hasPos}
	c.resetChaseLocked() // new chase session: path now, then throttle the tick re-issues
	c.chaseMoveLocked(s, tx, ty)
}

// tickOrderLocked advances the pending approach-cast (called from the tick).
func (s *Server) tickOrderLocked(c *conn, now float64) {
	hs := c.huntState
	o := hs.order
	if o == nil {
		return
	}
	def := hs.skillDef(o.slot)
	tx, ty := o.px, o.py
	var ms *mobState
	if o.target > 0 {
		ms = hs.mobs[o.target]
		if ms == nil || ms.dead {
			hs.order = nil
			s.orderDoneLocked(c, skillProtoID(hs.av, o.slot))
			return
		}
		tx, ty = ms.x, ms.y
	}
	cx, cy := c.posAtLocked(float32(now))
	maxDist := float64(def.Distance)
	if maxDist <= 0 {
		maxDist = 2.5
	}
	if math.Hypot(float64(tx-cx), float64(ty-cy)) <= maxDist+0.5 {
		hs.order = nil
		// stop and cast
		c.stopArrivalLocked()
		c.x, c.y, c.vx, c.vy, c.snapT = cx, cy, 0, 0, float32(now)
		c.sendPosLocked(s, cx, cy, 0, 0, float32(now))
		s.execCastLocked(c, o.slot, ms, tx, ty, o.hasPos)
		return
	}
	c.chaseMoveLocked(s, tx, ty) // retarget a moving mob (throttled for a static one)
}

// cancelOrderLocked drops a pending cast (manual move, stun, death).
func (s *Server) cancelOrderLocked(c *conn) {
	hs := c.huntState
	if hs == nil || hs.order == nil {
		return
	}
	slot := hs.order.slot
	hs.order = nil
	s.orderDoneLocked(c, skillProtoID(hs.av, slot))
}

// channelInterruptible reports whether a ground-anchored channel is one the caster
// actively sustains -- so any new action (move, stun, another cast) ends it --
// rather than a planted fire-and-forget ground effect that erupts on its own. The
// skill specs carry no such flag, so it is hand-maintained here, keyed by avatar
// prefab + slot. Elgorm's «Стрелы Аркана» (slot 4) is a channel; Titanid's
// «Землетрясение» quake, by contrast, stays fire-and-forget.
func channelInterruptible(prefab string, slot int) bool {
	return prefab == "Avtr_Dsb_Elgorm" && slot == 4
}

// channelPulseDelay is the lead-in before a channel's FIRST damage pulse, matching
// the client payload fx's own start delay so the server ticks land in step with the
// visual. Elgorm's «Стрелы Аркана» arrow burst (ProjectileBurst on
// VFX_Avtr_Dsb_Elgorm_skill4_prop01) has mDelay=0.2 before its first arrow; every
// other channel starts pulsing immediately. Hand-maintained (the specs carry no
// such field), keyed by prefab+slot like channelInterruptible.
func channelPulseDelay(prefab string, slot int) float64 {
	if prefab == "Avtr_Dsb_Elgorm" && slot == 4 {
		return 0.2
	}
	return 0
}

// skillChannelDur returns the longest OpChannel duration in a skill at the given
// rank (0 if the skill has no channel). Used to keep a channel skill's payload fx
// alive for the whole channel instead of a fixed short bound, so a long arrow rain
// renders all of its arrows.
func skillChannelDur(def gamedata.Skill, level int) float64 {
	var d float64
	for _, op := range def.Ops {
		if op.Kind == gamedata.OpChannel {
			if v := op.Dur.At(level); v > d {
				d = v
			}
		}
	}
	return d
}

// skillHasChannel reports whether a skill's ops include an OpChannel (a sustained
// channel the caster should stand and hold, so it must NOT roll into auto-attack
// when its cast action closes).
func skillHasChannel(def gamedata.Skill) bool {
	for _, op := range def.Ops {
		if op.Kind == gamedata.OpChannel {
			return true
		}
	}
	return false
}

// breakInterruptibleChannelsLocked ends every interruptible channel the caster is
// sustaining -- called when a new action supersedes it (a fresh skill cast; the
// tick handles movement/stun). Fire-and-forget ground channels are left running.
func (s *Server) breakInterruptibleChannelsLocked(c *conn) {
	hs := c.huntState
	if len(hs.channels) == 0 {
		return
	}
	keep := hs.channels[:0:0]
	for _, ch := range hs.channels {
		if ch.interruptible {
			continue
		}
		keep = append(keep, ch)
	}
	hs.channels = keep
}

// execCastLocked performs the actual cast: mana, packets, payload scheduling.
func (s *Server) execCastLocked(c *conn, slot int, ms *mobState, px, py float32, hasPos bool) {
	hs := c.huntState
	def := hs.skillDef(slot)
	level := int(hs.skillLevel[slot-1])
	now := float64(s.battleTime())
	parent := skillProtoID(hs.av, slot)

	if level < 1 { // rank-0 (unlearned ult) is uncastable
		s.orderDoneLocked(c, parent)
		return
	}
	cost := skillManaCost(float64(def.ManaCost[level-1]))
	if hs.mana < cost || now < hs.cooldownUntil[slot-1] {
		s.orderDoneLocked(c, parent)
		return
	}
	hs.mana -= cost
	s.syncSelfLocked(c, syncMana)

	cd := skillCooldown(float64(def.Cooldown[level-1]))
	hs.cooldownUntil[slot-1] = now + cd

	// A new cast supersedes a sustained (interruptible) channel the caster was
	// holding -- Elgorm's arrow rain ends the instant he casts something else. This
	// runs BEFORE the payload that would create THIS cast's own channel, so a
	// channel skill never cancels itself.
	s.breakInterruptibleChannelsLocked(c)

	// Casting roots the avatar: stop any in-flight movement at the live position
	// and push a velocity-0 sync so the client plays the cast in place instead of
	// sliding on to the old click destination. (An approach-cast already stops in
	// tickOrderLocked; this also covers an instant cast issued mid-walk.)
	if c.hasDest || c.arrival != nil {
		cx, cy := c.posAtLocked(float32(now))
		c.stopArrivalLocked()
		c.hasDest = false
		c.x, c.y, c.vx, c.vy, c.snapT = cx, cy, 0, 0, float32(now)
		c.sendPosLocked(s, cx, cy, 0, 0, float32(now))
	}

	var targetObj int32 = -1
	if ms != nil {
		targetObj = ms.id
	}
	tp := amf.NewArray().Set("x", 0.0).Set("y", 0.0)
	if hasPos && ms == nil {
		tp = amf.NewArray().Set("x", float64(px)).Set("y", float64(py))
	}
	// The skill ACTION goes to the caster AND teammates. The cast ANIMATION and VFX
	// come from the fx (now world-scoped), not this ACTION; broadcasting it turns the
	// remote avatar to face the target (VisualBattle.OnAction RotateTo) and marks it
	// DoingAction. It's a single one-shot per cast (closed by the ACTION_DONE in the
	// actionDones drain), so no per-swing re-trigger dance is needed. The client
	// resolves it without a skill effector on the remote avatar (Battle.OnAction just
	// adds the action; a point cast with no effector simply skips the rotate).
	castAction := newActionArgs(c.objID, parent, targetObj, now, tp)
	s.pushAvatarAllLocked(c, battleproto.CmdAction, castAction)

	// Cast-moment fx (plays the Cast clip + caster props).
	castUID := s.fxStartLocked(c, def.CastFx, c.objID, targetObj, hasPos, px, py)
	castDur := def.CastFxDur
	if castDur <= 0 {
		castDur = 2.0
	}
	hs.scheduleFxEnd(castUID, now+castDur)

	// Payload: fx at the victim/point + the actual ops.
	delay := def.PayloadDelay
	targetID := int32(0)
	if ms != nil {
		targetID = ms.id
	}
	hs.payloads = append(hs.payloads, payload{
		at: now + delay, slot: slot, level: level,
		target: targetID, px: px, py: py, hasPos: hasPos,
	})
	if delay <= 0 {
		s.runDuePayloadsLocked(c, now)
	}

	// Close the action so animations settle and the cooldown sweep starts. Remember
	// the cast's target so the avatar rolls back into auto-attack on it when the
	// action closes (nearest enemy if it's a self/point cast or the target died).
	doneAt := math.Max(delay, 0.3)
	var resumeTarget int32
	if ms != nil {
		resumeTarget = ms.id
	}
	hs.actionDones = append(hs.actionDones, actionDone{
		at: now + doneAt, action: parent, cooldown: now + cd, order: true,
		resumeTarget: resumeTarget,
		// A channel skill holds the caster in place sustaining it; do NOT roll into
		// auto-attack when the cast action closes (that would visually break the
		// channel pose and start swinging mid-channel).
		noResume: skillHasChannel(def),
	})

	// Root the avatar for the cast's committed motion only: the wind-up
	// (PayloadDelay, when the effect lands) plus a short recovery. This is
	// deliberately NOT CastFxDur -- that is how long the VFX lingers, not how long
	// the character is animating, and locking for it felt ~0.5s too long. Capped
	// so an unusually long wind-up never freezes the player excessively.
	const castRecovery = 0.0
	lockDur := def.PayloadDelay + castRecovery
	if lockDur < doneAt {
		lockDur = doneAt
	}
	if lockDur > 2.0 {
		lockDur = 2.0
	}
	hs.castLockUntil = now + lockDur
}

// payload is a scheduled skill impact.
type payload struct {
	at     float64
	slot   int
	level  int
	target int32
	px, py float32
	hasPos bool
	// ops, when non-nil, is the exact op list to run instead of the whole skill's
	// def.Ops -- used to defer a dash's follow-up ops (damage/root) until arrival.
	ops []gamedata.Op
	// resume, on a StrikeOnArrival continuation, rolls the avatar back into
	// auto-attack once the charge lands. The action-done's own resume attempt fires
	// mid-dash (hasDest set) and bails, so the charge needs its own post-arrival one.
	resume bool
}

type actionDone struct {
	at       float64
	action   int32
	cooldown float64
	order    bool
	// resumeTarget is the mob the avatar should keep swinging at once this skill's
	// action closes (0 = fall back to the nearest enemy). Lets a cast flow straight
	// back into auto-attack, the way a kill rolls the avatar onto the next mob.
	resumeTarget int32
	// noResume suppresses that auto-attack roll-back -- set for a channel cast, whose
	// caster stays put sustaining the channel instead of swinging.
	noResume bool
}

// runDuePayloadsLocked fires every payload whose time has come.
func (s *Server) runDuePayloadsLocked(c *conn, now float64) {
	hs := c.huntState
	// Take the current queue and clear it: firePayloadLocked may APPEND new payloads
	// (a StrikeOnArrival dash schedules its follow-up strike at dashUntil), and those
	// must survive. Rebuild the queue from the not-yet-due ones PLUS anything fired
	// payloads appended -- overwriting with a snapshot would drop the deferred strike,
	// so the charge would dash but never damage/root/drop its barrier.
	pending := hs.payloads
	hs.payloads = nil
	var keep []payload
	for _, p := range pending {
		if p.at > now {
			keep = append(keep, p)
			continue
		}
		s.firePayloadLocked(c, p, now)
	}
	hs.payloads = append(hs.payloads, keep...)
}

// lineFxEndpoint returns where to place a point payload fx. For a STATIONARY line
// skill (AoEWidth>0, no dash) it projects the click point (px,py) out to the skill's
// full range in the aim direction from the caster (cx,cy), so a caster->targetPos fx
// (e.g. Elgorm's arrow rain) sweeps the whole beam instead of stopping at the click.
// For any other skill it returns the click point unchanged.
func lineFxEndpoint(cx, cy, px, py float32, sk gamedata.Skill) (float32, float32) {
	if sk.AoEWidth <= 0 || skillIsDashCleave(sk) || sk.Distance <= 0 {
		return px, py
	}
	dx, dy := float64(px-cx), float64(py-cy)
	d := math.Hypot(dx, dy)
	if d < 1e-6 {
		return px, py
	}
	k := float64(sk.Distance) / d
	return cx + float32(dx*k), cy + float32(dy*k)
}

// targetBuffTTL is how long a skill's target-mode BuffFx should linger on the
// victim: the longest duration among its TOP-LEVEL OpBuffStat ops that land On the
// target (e.g. Velial's «Трибунал» 30s armor break). Nested ops -- aura/channel/proc
// re-applications -- are excluded on purpose: their short, repeated windows would
// strobe a persistent aura. 0 means the skill applies no top-level target stat-buff,
// so no target BuffFx is shown (its debuff visual, if any, comes from an op like OpSlow).
func targetBuffTTL(def gamedata.Skill, level int) float64 {
	var ttl float64
	for _, op := range def.Ops {
		if op.Kind == gamedata.OpBuffStat && op.On == "target" {
			if d := op.Dur.At(level); d > ttl {
				ttl = d
			}
		}
	}
	return ttl
}

func (s *Server) firePayloadLocked(c *conn, p payload, now float64) {
	hs := c.huntState
	def := hs.skillDef(p.slot)
	var ms *mobState
	if p.target > 0 {
		ms = hs.mobs[p.target]
		if ms != nil && ms.dead {
			ms = nil
		}
	}
	// Payload fx placement -- only for the primary payload, not a deferred
	// dash-arrival continuation (which re-uses this path with its own op subset).
	if p.ops == nil {
		// A one-shot payload fx plays out over a fixed short window; a channel skill's
		// payload (Elgorm's arrow rain) must live the whole channel so every arrow is
		// spawned (a 4s rain cut at 3s dropped the last ~2 arrows). Ended by EFFECT_END.
		fxLife := 3.0
		if d := skillChannelDur(def, p.level); d > fxLife {
			fxLife = d
		}
		// A stationary line skill's point payload (Elgorm's arrow rain, a
		// SELF_TO_TARGETPOS ProjectileBurst that shoots caster->targetPos) must sweep
		// the FULL beam, so aim its endpoint at the skill's max range in the click
		// direction rather than the exact click point -- matching the full-range damage
		// swath (damageTargetsLocked) and the client's full-length line cursor.
		fpx, fpy := p.px, p.py
		if p.hasPos {
			cx, cy := c.posAtLocked(float32(now))
			fpx, fpy = lineFxEndpoint(cx, cy, p.px, p.py, def)
		}
		switch def.PayloadFxAt {
		case "target":
			tid := p.target
			if tid == 0 {
				tid = c.objID
			}
			uid := s.fxStartLocked(c, def.PayloadFx, c.objID, tid, p.hasPos, fpx, fpy)
			hs.scheduleFxEnd(uid, now+fxLife)
		case "point":
			// A SELF-baked ground fx trails the caster; for a skill whose point payload
			// is SELF-mode (Titanid's «Землетрясение» quake) pin it to an invisible
			// stationary anchor at the point instead of owning it to the moving avatar.
			fxOwner, anchor := c.objID, int32(0)
			if payloadFxUsesAnchor(hs.av.Prefab, p.slot) {
				anchor = s.spawnTrapAnchorLocked(c, fpx, fpy, now)
				fxOwner = anchor
			}
			uid := s.fxStartLocked(c, def.PayloadFx, fxOwner, 0, true, fpx, fpy)
			hs.scheduleFxEnd(uid, now+fxLife)
			if anchor != 0 {
				hs.anchorEnds = append(hs.anchorEnds, anchorEnd{id: anchor, at: now + fxLife + 0.3})
			}
		case "self":
			uid := s.fxStartLocked(c, def.PayloadFx, c.objID, 0, false, 0, 0)
			hs.scheduleFxEnd(uid, now+fxLife)
		}
		// Target-mode BuffFx: a persistent debuff/buff visual pinned ON the primary
		// victim for the effect's own duration -- e.g. Velial's «Трибунал» armor-break
		// aura. The self/ground variants live in addPlayerModLocked (which explicitly
		// SKIPS BuffFxOn=="target"), and the per-op loop is the wrong home too: it would
		// double the visual on a multi-buff ult (Urg stacks phys+magic armor in one cast)
		// and strobe on aura/channel re-application. So it fires once here, on ms, and
		// self-ends after the buff's own TTL. World-scoped (fxStartLocked -> instance),
		// so every party member sees the debuffed mob. Parented to the mob (owner=ms.id),
		// so it dies with the body if the mob is killed before the TTL elapses.
		if ms != nil && def.BuffFxOn == "target" && def.BuffFx != "" {
			if ttl := targetBuffTTL(def, p.level); ttl > 0 {
				uid := s.fxStartLocked(c, def.BuffFx, ms.id, 0, false, 0, 0)
				hs.scheduleFxEnd(uid, now+ttl)
			}
		}
	}
	ops := def.Ops
	if p.ops != nil {
		ops = p.ops
	}
	ctx := opCtx{slot: p.slot, level: p.level, target: ms, px: p.px, py: p.py, hasPos: p.hasPos}
	s.applyOpsLocked(c, ops, ctx, now)
	// A charge's strike lands here, AFTER the dash cleared hasDest -- so this is the
	// point where auto-attack can actually re-engage (the earlier action-done attempt
	// bailed mid-dash). Prefer the struck target.
	if p.resume {
		s.resumeAutoAttackLocked(c, now, p.target)
	}
}

// ---- toggles ----

func (s *Server) toggleSkillLocked(c *conn, slot int) {
	hs := c.huntState
	def := hs.skillDef(slot)
	level := int(hs.skillLevel[slot-1])
	now := float64(s.battleTime())
	parent := skillProtoID(hs.av, slot)

	if hs.toggleOn[slot-1] {
		s.toggleOffLocked(c, slot, now, true)
		return
	}
	if level < 1 || now < hs.cooldownUntil[slot-1] || hs.mana < skillManaCost(float64(def.ManaCost[level-1])) {
		return
	}
	hs.mana -= skillManaCost(float64(def.ManaCost[level-1]))
	s.syncSelfLocked(c, syncMana)
	hs.toggleOn[slot-1] = true
	hs.toggleNextPulse[slot-1] = now
	toggleAction := newActionArgs(c.objID, parent, int32(-1), now,
		amf.NewArray().Set("x", 0.0).Set("y", 0.0))
	s.pushAvatarAllLocked(c, battleproto.CmdAction, toggleAction)
	// The persistent toggle visual (e.g. Abominator's tentacles) MUST be the
	// BuffFx: its prefab's gfx carries stopOnDone=true, so the toggle-off
	// EFFECT_END can actually remove it. The CastFx is a fire-and-forget cast
	// splash whose prefab can't be stopped -- holding it as the persistent handle
	// left it stuck on forever. Fall back to CastFx only when a toggle has no
	// BuffFx (e.g. Zamaran), preserving its old visual.
	toggleVisual := def.BuffFx
	if toggleVisual == "" {
		toggleVisual = def.CastFx
	}
	hs.toggleFx[slot-1] = s.fxStartLocked(c, toggleVisual, c.objID, 0, false, 0, 0)
	// Self-buff ops of a toggle apply while it is on (aura ops pulse in tick).
	ctx := opCtx{slot: slot, level: level, toggle: true}
	for _, op := range def.Ops {
		if op.Kind == gamedata.OpBuffStat && op.On != "target" {
			s.applyOpsLocked(c, []gamedata.Op{op}, ctx, now)
		}
	}
}

// toggleOffLocked switches a toggle off (player click, mana starvation, death).
func (s *Server) toggleOffLocked(c *conn, slot int, now float64, byUser bool) {
	hs := c.huntState
	if !hs.toggleOn[slot-1] {
		return
	}
	def := hs.skillDef(slot)
	level := int(hs.skillLevel[slot-1])
	hs.toggleOn[slot-1] = false
	s.fxEndLocked(c, hs.toggleFx[slot-1])
	hs.toggleFx[slot-1] = 0
	// Drop the toggle's self-buff mods immediately.
	s.removeModsBySrcLocked(c, toggleSrc(slot), now)
	cd := skillCooldown(float64(def.Cooldown[level-1]))
	hs.cooldownUntil[slot-1] = now + cd
	toggleDone := amf.NewArray().
		Set("id", c.objID).
		Set("action", skillProtoID(hs.av, slot)).
		Set("item", false).
		Set("cooldown", now+cd)
	s.pushAvatarAllLocked(c, battleproto.CmdActionDone, toggleDone)
	_ = byUser
}

func toggleSrc(slot int) string { return "toggle" + string(rune('0'+slot)) }

// ---- op execution ----

// opCtx carries the resolution context of one ops batch.
type opCtx struct {
	slot   int
	level  int
	target *mobState // nil for self/point casts
	px, py float32
	hasPos bool
	toggle bool
}

// centerLocked returns the AoE center: target mob, else point, else caster.
func (s *Server) centerLocked(c *conn, ctx opCtx) (float32, float32) {
	if ctx.target != nil {
		return ctx.target.x, ctx.target.y
	}
	if ctx.hasPos {
		return ctx.px, ctx.py
	}
	return c.posAtLocked(s.battleTime())
}

// mobsWithinLocked collects living mobs whose body (centre within r + the mob's
// own radius) overlaps the circle of radius r at (x,y), so an AoE that reaches a
// big boss's edge still hits it.
func (c *conn) mobsWithinLocked(x, y float32, r float64) []*mobState {
	var out []*mobState
	for _, m := range c.huntState.mobs {
		if m.dead {
			continue
		}
		if math.Hypot(float64(m.x-x), float64(m.y-y)) <= r+m.mob.Radius() {
			out = append(out, m)
		}
	}
	return out
}

// mobsAlongLineLocked collects living mobs inside a line/rift swath: within
// halfWidth of the ray from the caster (cx,cy) toward the aim point, from the
// caster out to maxLen. The aim point fixes only the DIRECTION -- the swath
// starts at the caster, so enemies standing BETWEEN the caster and a far aim
// point are hit (the bug where aiming past a mob missed it entirely).
func (c *conn) mobsAlongLineLocked(cx, cy, tx, ty float32, halfWidth, maxLen float64) []*mobState {
	dx, dy := float64(tx-cx), float64(ty-cy)
	dlen := math.Hypot(dx, dy)
	if dlen < 1e-6 {
		return nil
	}
	ux, uy := dx/dlen, dy/dlen // unit direction toward the aim point
	var out []*mobState
	for _, m := range c.huntState.mobs {
		if m.dead {
			continue
		}
		rx, ry := float64(m.x-cx), float64(m.y-cy)
		along := rx*ux + ry*uy          // distance along the ray
		perp := math.Abs(rx*uy - ry*ux) // perpendicular offset from the ray
		br := m.mob.Radius()            // per-mob body pad (a boss's wide edge still catches)
		if along >= -br && along <= maxLen+br && perp <= halfWidth+br {
			out = append(out, m)
		}
	}
	return out
}

// skillIsDashCleave reports whether a line skill (AoEWidth>0) is a dash-cleave --
// the caster lunges to the aim point, so the damage lane is the path travelled and
// a short dash cuts a short lane. A stationary line skill (beam/rift/thrown volley)
// has no dash and instead projects the full skill range in the aim direction.
func skillIsDashCleave(sk gamedata.Skill) bool {
	for _, op := range sk.Ops {
		if op.Kind == gamedata.OpDash {
			return true
		}
	}
	return false
}

// damageTargetsLocked resolves the victims of a damaging op.
func (s *Server) damageTargetsLocked(c *conn, ctx opCtx, radius float64) []*mobState {
	sk := c.huntState.skillDef(ctx.slot)
	// Line/rift skills (AoEWidth>0, aimed at a ground point, e.g. Velial's
	// "Разлом"): the swath runs from the caster toward the aim point, so mobs
	// standing in front of a far-aimed point are still caught. Circle radius
	// (below) would center on the point and miss them.
	if sk.AoEWidth > 0 && ctx.target == nil && ctx.hasPos {
		cx, cy := c.posAtLocked(s.battleTime())
		// A STATIONARY line skill (a beam/rift/thrown volley: Velial's «Разлом»,
		// Elgorm's «Стрелы Аркана», Nerlag's «Метание топоров») projects the FULL skill
		// range in the aim direction regardless of exactly where the player clicked --
		// matching the client's SkillLineZone.SelfNoClamp, which always draws the beam
		// Distance-long. A DASH-cleave is different: the caster lunges to the aim point
		// and the damage lane is only the path actually travelled, so a short dash cuts
		// a short lane (length = click distance, capped at range).
		length := float64(sk.Distance)
		if skillIsDashCleave(sk) {
			length = math.Hypot(float64(ctx.px-cx), float64(ctx.py-cy))
			if d := float64(sk.Distance); d > 0 && length > d {
				length = d
			}
			if length <= 0 {
				length = float64(sk.Distance)
			}
		}
		return c.mobsAlongLineLocked(cx, cy, ctx.px, ctx.py, float64(sk.AoEWidth)/2, length)
	}
	if radius <= 0 {
		if ctx.target != nil {
			return []*mobState{ctx.target}
		}
		// A self/point cast with no explicit op radius: these are "around
		// self/point" skills (e.g. Velial's self-AoE lifesteal). Fall back to the
		// skill's authored AoE radius so the op actually hits something rather
		// than resolving to an empty target set.
		radius = float64(sk.AoERadius)
		if radius <= 0 {
			radius = 4
		}
	}
	x, y := s.centerLocked(c, ctx)
	return c.mobsWithinLocked(x, y, radius)
}

// opTargetsLocked resolves a damaging/CC op's victims and applies its MaxTargets cap
// (the N nearest to the AoE centre). A capped op (Rognar's «Могильный холод», two
// targets) hits only that subset of everything in range; uncapped ops are unchanged.
func (s *Server) opTargetsLocked(c *conn, ctx opCtx, op gamedata.Op) []*mobState {
	targets := s.damageTargetsLocked(c, ctx, op.Radius)
	if op.MaxTargets > 0 && len(targets) > op.MaxTargets {
		cx, cy := s.centerLocked(c, ctx)
		sort.Slice(targets, func(i, j int) bool {
			return math.Hypot(float64(targets[i].x-cx), float64(targets[i].y-cy)) <
				math.Hypot(float64(targets[j].x-cx), float64(targets[j].y-cy))
		})
		targets = targets[:op.MaxTargets]
	}
	return targets
}

// skillDamageLocked computes a damage op's amount for the caster. Spell power
// is added as SP×PerSP (magic-scaled ops carry PerSP=1 from the generator);
// phys-scaled ops instead ride the attack-damage buff multiplier.
func (s *Server) skillDamageLocked(c *conn, op gamedata.Op, ctx opCtx, victim *mobState) float64 {
	hs := c.huntState
	now := float64(s.battleTime())
	// Base skill damage scales with avatar level; the spell-power contribution
	// already carries its own level scaling via spellPowerLocked, so only the flat
	// per-rank value is multiplied here (no double-count).
	dmg := op.Value.At(ctx.level) * hs.powerMul()
	if op.PerSP > 0 {
		dmg += hs.spellPowerLocked(now) * op.PerSP
	} else if op.Scale == "magic" {
		dmg += hs.spellPowerLocked(now)
	}
	if op.Scale == "phys" {
		dmg *= hs.st.modMul(now, "dmg_pct")
	}
	if b := op.BonusMissingHP.At(ctx.level); b > 0 && victim != nil {
		missing := 1 - victim.hp/victim.maxHealth()
		dmg *= 1 + b*missing
	}
	// Bonus damage scaling with the CASTER's own missing HP (Velial's «Воля к победе»).
	// Added flat AFTER the multipliers so it is not scaled by power/attack buffs --
	// matching the in-game values (~100 × missing at max rank, independent of level).
	if b := op.CasterMissingHP.At(ctx.level); b > 0 {
		if maxHP := hs.maxHPLocked(now); maxHP > 0 {
			if missing := 1 - hs.hp/maxHP; missing > 0 {
				dmg += b * missing
			}
		}
	}
	return dmg
}

// applyOpsLocked executes a batch of ops in a context. Caller holds mvMu.
func (s *Server) applyOpsLocked(c *conn, ops []gamedata.Op, ctx opCtx, now float64) {
	hs := c.huntState
	for i := 0; i < len(ops); i++ {
		op := ops[i]
		switch op.Kind {
		case gamedata.OpDamage:
			if op.Apply == "self" {
				// A self-sacrifice cost (Abominator): pure health drain, no armor.
				dmg := op.Value.At(ctx.level)
				hs.hp = math.Max(1, hs.hp-dmg) // never suicide on a cost
				s.syncSelfLocked(c, syncHealth)
				break
			}
			for _, m := range s.opTargetsLocked(c, ctx, op) {
				s.hitMobLocked(c, m, s.skillDamageLocked(c, op, ctx, m), c.objID)
			}

		case gamedata.OpConsumeDots:
			for _, m := range s.opTargetsLocked(c, ctx, op) {
				stacks := len(m.st.dots)
				bonus := op.Value.At(ctx.level) * hs.powerMul() * float64(stacks)
				if op.PerSP > 0 {
					bonus += hs.spellPowerLocked(now) * op.PerSP * float64(stacks)
				}
				m.st.dots = nil // consumed
				if m.st.dotFx != 0 {
					s.worldFxEndLocked(c, m.st.dotFx) // acid gone -> drop its visual
					m.st.dotFx = 0
				}
				if bonus > 0 {
					s.hitMobLocked(c, m, bonus, c.objID)
				}
			}

		case gamedata.OpLifestealHit:
			for _, m := range s.opTargetsLocked(c, ctx, op) {
				dmg := s.skillDamageLocked(c, op, ctx, m)
				s.hitMobLocked(c, m, dmg, c.objID)
				s.healPlayerLocked(c, dmg*op.Value2.At(ctx.level))
			}

		case gamedata.OpDot:
			for _, m := range s.opTargetsLocked(c, ctx, op) {
				m.st.dots = append(m.st.dots, overTime{
					perSec: op.Value.At(ctx.level), until: now + op.Dur.At(ctx.level),
					nextTick: now + dotTickInterval, srcObj: c.objID,
				})
				// Persistent acid/poison visual on the victim (one shared copy, shown
				// to the whole party). An empty DotFx is a no-op inside the helper.
				s.ensureMobStatusFxLocked(c, m, &m.st.dotFx, op.DotFx)
			}

		case gamedata.OpHeal:
			amt := op.Value.At(ctx.level) * hs.powerMul()
			if op.PerSP > 0 {
				amt += hs.spellPowerLocked(now) * op.PerSP
			}
			s.healPlayerLocked(c, amt)

		case gamedata.OpHot:
			hs.st.hots = append(hs.st.hots, overTime{
				perSec: op.Value.At(ctx.level) * hs.powerMul(), until: now + op.Dur.At(ctx.level),
				nextTick: now + 1,
			})

		case gamedata.OpManaRestore:
			hs.mana = math.Min(hs.maxManaLocked(now), hs.mana+op.Value.At(ctx.level)*hs.powerMul())
			s.syncSelfLocked(c, syncMana)

		case gamedata.OpStun:
			for _, m := range s.opTargetsLocked(c, ctx, op) {
				s.stunMobLocked(c, m, now, op.Dur.At(ctx.level))
			}

		case gamedata.OpRoot:
			for _, m := range s.opTargetsLocked(c, ctx, op) {
				m.st.rootUntil = math.Max(m.st.rootUntil, now+op.Dur.At(ctx.level))
				s.ensureMobStatusFxLocked(c, m, &m.st.rootFx, "StunEffect")
				s.stopMobLocked(c, m, now)
			}

		case gamedata.OpSlow:
			for _, m := range s.opTargetsLocked(c, ctx, op) {
				m.st.slowUntil = now + op.Dur.At(ctx.level)
				m.st.slowFactor = op.Value.At(ctx.level)
				s.ensureMobStatusFxLocked(c, m, &m.st.slowFx, "SlowMoveEffect")
				s.syncMobSpeedLocked(c, m, now)
			}

		case gamedata.OpAttackSlow:
			for _, m := range s.opTargetsLocked(c, ctx, op) {
				m.st.atkSlowUntil = now + op.Dur.At(ctx.level)
				m.st.atkSlowFactor = op.Value.At(ctx.level)
				s.ensureMobStatusFxLocked(c, m, &m.st.atkSlowFx, "SlowAttackEffect")
			}

		case gamedata.OpSilence:
			// Mobs have no skills: silencing one also stops its attacks.
			for _, m := range s.opTargetsLocked(c, ctx, op) {
				m.st.silenceUntil = now + op.Dur.At(ctx.level)
				m.st.atkSlowUntil = math.Max(m.st.atkSlowUntil, now+op.Dur.At(ctx.level))
				m.st.atkSlowFactor = 0.1
				s.ensureMobStatusFxLocked(c, m, &m.st.silenceFx, "SilenceEffect")
			}

		case gamedata.OpBuffStat:
			if op.On == "target" {
				for _, m := range s.opTargetsLocked(c, ctx, op) {
					m.st.mods = append(m.st.mods, statMod{
						stat: op.Stat, value: op.Value.At(ctx.level),
						until: now + op.Dur.At(ctx.level), src: castSrc(ctx),
					})
					s.syncMobSpeedLocked(c, m, now)
				}
			} else {
				s.addPlayerModLocked(c, ctx, op, now)
			}

		case gamedata.OpShield:
			hs.st.shield = op.Value.At(ctx.level) * hs.powerMul()
			hs.st.shieldUntil = now + op.Dur.At(ctx.level)
			if hs.st.shieldFx == 0 {
				hs.st.shieldFx = s.fxStartLocked(c, "RuneShieldEffect3", c.objID, 0, false, 0, 0)
			}

		case gamedata.OpBlink:
			s.blinkLocked(c, ctx)

		case gamedata.OpDash:
			s.dashLocked(c, ctx, op.Value.At(ctx.level), now, op.NoClip)
			// Strike on arrival: defer the ops AFTER this dash until the lunge lands
			// (hs.dashUntil), so damage/root/barrier hit on impact, not on cast.
			if op.StrikeOnArrival && i+1 < len(ops) {
				rest := append([]gamedata.Op(nil), ops[i+1:]...)
				tid := int32(0)
				if ctx.target != nil {
					tid = ctx.target.id
				}
				hs.payloads = append(hs.payloads, payload{
					at: hs.dashUntil, slot: ctx.slot, level: ctx.level,
					target: tid, px: ctx.px, py: ctx.py, hasPos: ctx.hasPos, ops: rest,
					resume: true,
				})
				return
			}

		case gamedata.OpPull:
			if ctx.target != nil {
				s.pullMobLocked(c, ctx.target, now)
			}

		case gamedata.OpKnockback:
			for _, m := range s.opTargetsLocked(c, ctx, op) {
				s.knockbackMobLocked(c, m, op.Value.At(ctx.level), now)
			}

		case gamedata.OpOnKill:
			// Run the nested ops only if this cast's primary target died from it
			// (Lirvein's «Изощренный бросок» reset+empower on a kill).
			if ctx.target != nil && ctx.target.dead {
				s.applyOpsLocked(c, op.Ops, ctx, now)
			}

		case gamedata.OpCooldownReset:
			s.resetCooldownsLocked(c, now)

		case gamedata.OpSummon:
			s.summonLocked(c, op, ctx, now)

		case gamedata.OpTrap:
			px, py := s.centerLocked(c, ctx)
			// A SELF-mode ground fx would trail the caster; pin it by owning it to an
			// invisible stationary anchor at the point instead of to the avatar.
			fxOwner, anchor := c.objID, int32(0)
			if trapUsesAnchor(hs.av.Prefab, ctx.slot) {
				anchor = s.spawnTrapAnchorLocked(c, px, py, now)
				fxOwner = anchor
			}
			uid := s.fxStartLocked(c, op.TrapFx, fxOwner, 0, true, px, py)
			hs.traps = append(hs.traps, trapState{
				x: px, y: py, radius: op.TriggerRadius,
				until: now + op.Lifetime.At(ctx.level),
				ops:   op.Ops, level: ctx.level, slot: ctx.slot,
				fx: uid, triggerFx: op.TriggerFx, anchor: anchor,
			})

		case gamedata.OpBounce:
			s.startBounceLocked(c, op, ctx, now)

		case gamedata.OpChannel:
			hs.channels = append(hs.channels, channelState{
				slot: ctx.slot, level: ctx.level,
				until: now + op.Dur.At(ctx.level), interval: op.Interval,
				// Delay the FIRST pulse by the client fx's own lead-in so the damage
				// ticks land in step with the visual (Elgorm's arrow rain waits 0.2s
				// before its first arrow); every other channel starts immediately.
				nextPulse: now + channelPulseDelay(hs.av.Prefab, ctx.slot),
				target:    mobID(ctx.target),
				px:        ctx.px, py: ctx.py, hasPos: ctx.hasPos, ops: op.Ops,
				interruptible: channelInterruptible(hs.av.Prefab, ctx.slot),
			})

		case gamedata.OpProc:
			// procs are registered from passives at world-state time; a proc op
			// inside an active cast applies its nested ops immediately instead.
			s.applyOpsLocked(c, op.Ops, ctx, now)

		case gamedata.OpAura:
			// aura pulses run from the tick while the toggle is on; nothing here.

		case gamedata.OpRevive, gamedata.OpImmune, gamedata.OpHealOnKill:
			// Passive-only mechanics: registered at world-build and honored in the
			// death / player-CC / on-kill gates. Nothing to run inside an ops batch.

		default:
			log.Printf("battle: %s unknown op kind %q", c.RemoteAddr(), op.Kind)
		}
	}
}

func castSrc(ctx opCtx) string {
	if ctx.toggle {
		return toggleSrc(ctx.slot)
	}
	return "skill" + string(rune('0'+ctx.slot))
}

func mobID(m *mobState) int32 {
	if m == nil {
		return 0
	}
	return m.id
}

// ---- player-side helpers ----

// powerMul / hpMul are the per-level scaling multipliers at the avatar's current
// battle level (1.0 at level 0). powerMul lifts basic + skill damage/heals and
// spell power; hpMul lifts the max health/mana pools. Together they make a
// level-20 hero ~2.1x a level-1 one -- the curve behind the boss level-gating.
func (hs *huntState) powerMul() float64 { return gamedata.LevelPowerMul(hs.level) }
func (hs *huntState) hpMul() float64    { return gamedata.LevelHealthMul(hs.level) }

// spellPowerLocked / maxHPLocked / maxManaLocked are the live effective stats.
func (hs *huntState) spellPowerLocked(now float64) float64 {
	return hs.av.SpellPower*hs.powerMul() + hs.st.modSum(now, "spell_power")
}

func (hs *huntState) maxHPLocked(now float64) float64 {
	return hs.av.Health*hs.hpMul() + hs.st.modSum(now, "max_hp")
}

func (hs *huntState) maxManaLocked(float64) float64 { return hs.av.Mana * hs.hpMul() }

// effAttackRangeLocked is the avatar's live auto-attack reach: its base AttackRange
// plus any attack_range buff (Teridin's «Прицеливание» passive).
func (hs *huntState) effAttackRangeLocked(now float64) float64 {
	return hs.av.AttackRange + hs.st.modSum(now, "attack_range")
}

func (s *Server) healPlayerLocked(c *conn, amount float64) {
	hs := c.huntState
	now := float64(s.battleTime())
	if hs.deadUntil > 0 || amount <= 0 {
		return
	}
	hs.hp = math.Min(hs.maxHPLocked(now), hs.hp+amount)
	s.syncSelfLocked(c, syncHealth)
}

// addPlayerModLocked applies a self buffstat: status mod + buff icon + fx +
// live stat syncs.
func (s *Server) addPlayerModLocked(c *conn, ctx opCtx, op gamedata.Op, now float64) {
	hs := c.huntState
	def := hs.skillDef(ctx.slot)
	dur := op.Dur.At(ctx.level)
	until := now + dur
	if dur <= 0 { // permanent (passives)
		until = 0
	}
	mod := statMod{stat: op.Stat, value: op.Value.At(ctx.level), until: until, src: castSrc(ctx)}

	// Buff-bar icon (only for timed, non-toggle, icon-enabled skills).
	if def.BuffIcon && dur > 0 {
		mod.buffEffID = hs.newEffID()
		args := amf.NewArray().Set("duration", dur).Set("level", int32(ctx.level))
		for k, v := range def.TipArgs {
			args.Set(k, v.At(ctx.level))
		}
		s.push(c, battleproto.CmdAddEffector, addEffectorArgs(mod.buffEffID,
			buffProtoID(hs.av, ctx.slot), c.objID, -1, now, args))
	}
	// A toggle owns its persistent BuffFx via hs.toggleFx (started in
	// toggleSkillLocked); don't start a second copy here or it would leak the
	// duplicate on toggle-off.
	if def.BuffFx != "" && def.BuffFxOn != "target" && !ctx.toggle {
		if def.BuffFxOn == "ground" {
			// A stationary barrier (e.g. Vigilans' ult). CONFIRMED on the client: this
			// prefab's barrier gfx is SELF-mode -- it PARENTS to the EFFECT_START owner
			// GameObject and follows it (owner=caster made it trail Vigilans; a point
			// targetPos is ignored by a SELF gfx, so owner=-1/point shows nothing). The
			// only stationary anchor is the ROOTED target: this ult roots the enemy for
			// the same duration as the buff, so parenting to the mob pins the barrier on
			// the trapped foe. (The pad sub-gfx is positional and stays regardless.)
			owner := c.objID
			if ctx.target != nil {
				owner = ctx.target.id
				// If the ult kills this target, its corpse must outlive the barrier so
				// the SELF-mode VFX keeps its anchor (else it orphans onto the caster
				// when the body is deleted -- the intermittent "barrier follows me").
				if until > ctx.target.st.anchorFxUntil {
					ctx.target.st.anchorFxUntil = until
				}
			}
			mod.fxUID = s.fxStartLocked(c, def.BuffFx, owner, 0, false, 0, 0)
		} else {
			mod.fxUID = s.fxStartLocked(c, def.BuffFx, c.objID, 0, false, 0, 0)
		}
	}
	hs.st.mods = append(hs.st.mods, mod)
	s.pushPlayerStatsLocked(c, now)
}

// removeModsBySrcLocked drops all player mods tagged src, reversing visuals.
func (s *Server) removeModsBySrcLocked(c *conn, src string, now float64) {
	hs := c.huntState
	var keep []statMod
	for _, m := range hs.st.mods {
		if m.src != src {
			keep = append(keep, m)
			continue
		}
		if m.buffEffID != 0 {
			s.push(c, battleproto.CmdRemEffector, amf.NewArray().Set("id", m.buffEffID))
		}
		s.fxEndLocked(c, m.fxUID)
	}
	hs.st.mods = keep
	s.pushPlayerStatsLocked(c, now)
}

// pushPlayerStatsLocked syncs every mod-affected stat so the HUD/target frame
// and animations track buffs live.
func (s *Server) pushPlayerStatsLocked(c *conn, now float64) {
	hs := c.huntState
	idx := hs.tr.index(c.objID)
	if idx < 0 {
		return
	}
	a := hs.av
	st := &hs.st
	// dmgMul is the buff multiplier; pMul is the per-level power multiplier -- both
	// apply to the displayed basic-attack damage (matching the real hit calc).
	dmgMul := st.modMul(now, "dmg_pct") * hs.powerMul()
	armMul := st.modMul(now, "armor_pct")
	maxHP := hs.maxHPLocked(now)
	if hs.hp > maxHP {
		hs.hp = maxHP
	}
	maxMana := hs.maxManaLocked(now)
	if hs.mana > maxMana {
		hs.mana = maxMana
	}
	b := newSyncBlob(float32(now)).
		setFloats(syncDmgMin, idx, float32(float64(a.DmgMin)*dmgMul)).
		setFloats(syncDmgMax, idx, float32(float64(a.DmgMax)*dmgMul)).
		setFloats(syncAttackSpeed, idx, float32(a.AttackSpeed*st.attackFactor(now))).
		setFloats(syncMaxHealth, idx, float32(maxHP)).
		setFloats(syncHealth, idx, float32(hs.hp/maxHP)).
		setFloats(syncMaxMana, idx, float32(maxMana)).
		setFloats(syncMana, idx, float32(hs.mana/maxMana)).
		setFloats(syncPhysArmor, idx, float32((a.PhysArmor+st.modSum(now, "phys_armor"))*armMul)).
		setFloats(syncMagicArmor, idx, float32((a.MagicArmor+st.modSum(now, "magic_armor"))*armMul)).
		setFloats(syncSpellPower, idx, float32(hs.spellPowerLocked(now))).
		setFloats(syncAttackRange, idx, float32(hs.effAttackRangeLocked(now))).
		setFloats(syncSpeed, idx, float32(c.curSpeedLocked(now)))
	s.push(c, battleproto.CmdSync, amf.NewArray().Set("data", b.build(hs.tr.count())))
	c.applySpeedLocked(s, now)
}

// curSpeedLocked is the player's current move speed in units/sec.
func (c *conn) curSpeedLocked(now float64) float64 {
	hs := c.huntState
	if hs == nil {
		return float64(lobbyMoveSpeed)
	}
	if now < hs.dashUntil {
		return hs.dashSpeed
	}
	return float64(lobbyMoveSpeed) * hs.st.moveFactor(now)
}

// applySpeedLocked re-issues the current movement leg at the current speed so
// slows/hastes take effect mid-run.
func (c *conn) applySpeedLocked(s *Server, now float64) {
	if c.arrival == nil || !c.hasDest {
		return
	}
	c.moveToLocked(s, c.destX, c.destY)
	_ = now
}

// ---- blink / dash / pull ----

func (s *Server) blinkLocked(c *conn, ctx opCtx) {
	hs := c.huntState
	def := hs.skillDef(ctx.slot)
	now := s.battleTime()
	cx, cy := c.posAtLocked(now)
	tx, ty := ctx.px, ctx.py
	if !ctx.hasPos {
		return
	}
	// Clamp to cast range, then to walkable ground.
	d := math.Hypot(float64(tx-cx), float64(ty-cy))
	maxD := float64(def.Distance)
	if maxD > 0 && d > maxD {
		tx = cx + float32(float64(tx-cx)*maxD/d)
		ty = cy + float32(float64(ty-cy)*maxD/d)
	}
	if c.nav != nil {
		nx, ny := c.nav.Clip(float64(cx), float64(cy), float64(tx), float64(ty))
		tx, ty = float32(nx), float32(ny)
	}
	c.stopArrivalLocked()
	c.hasDest = false
	c.x, c.y, c.vx, c.vy, c.snapT = tx, ty, 0, 0, now
	c.sendPosLocked(s, tx, ty, 0, 0, now)
}

func (s *Server) dashLocked(c *conn, ctx opCtx, speed float64, now float64, noClip bool) {
	hs := c.huntState
	if speed <= 0 {
		speed = 12
	}
	tx, ty := ctx.px, ctx.py
	if ctx.target != nil {
		tx, ty = ctx.target.x, ctx.target.y
	} else if !ctx.hasPos {
		return
	}
	cx, cy := c.posAtLocked(float32(now))
	// A dash is a straight lunge, not a routed walk. A normal dash clips to the wall
	// (stops at the first obstacle); a noClip "charge" drives straight to the target
	// THROUGH obstacles. Size the dash-speed window to the ACTUAL travel distance so
	// the whole lunge runs at dash speed.
	dtx, dty := tx, ty
	if c.nav != nil && !noClip {
		nx, ny := c.nav.Clip(float64(cx), float64(cy), float64(tx), float64(ty))
		dtx, dty = float32(nx), float32(ny)
	}
	dist := math.Hypot(float64(dtx-cx), float64(dty-cy))
	hs.dashSpeed = speed
	hs.dashUntil = now + dist/speed + 0.05
	c.moveStraightExLocked(s, tx, ty, !noClip)
}

// knockbackMobLocked shoves a mob directly away from the caster by dist units (the
// inverse of pullMobLocked), clipped to walkable ground -- Dutnik's «Взрыв» blast.
func (s *Server) knockbackMobLocked(c *conn, m *mobState, dist float64, now float64) {
	hs := c.huntState
	if dist <= 0 {
		dist = 3
	}
	cx, cy := c.posAtLocked(float32(now))
	dx, dy := float64(m.x-cx), float64(m.y-cy)
	d := math.Hypot(dx, dy)
	if d < 1e-6 {
		dx, dy, d = 1, 0, 1 // degenerate overlap: push along +x
	}
	tx := m.x + float32(dx/d*dist)
	ty := m.y + float32(dy/d*dist)
	if c.nav != nil {
		nx, ny := c.nav.Clip(float64(m.x), float64(m.y), float64(tx), float64(ty))
		tx, ty = float32(nx), float32(ny)
	}
	m.x, m.y = tx, ty
	idx := hs.tr.index(m.id)
	if idx >= 0 {
		s.push(c, battleproto.CmdSync, amf.NewArray().Set("data",
			newSyncBlob(float32(now)).position(idx, m.x, m.y, 0, 0, float32(now)).
				build(hs.tr.count())))
	}
	s.stopMobLocked(c, m, now)
}

// resetCooldownsLocked clears every skill cooldown for the caster, server-side AND on
// the client (one ACTION_DONE per slot carrying an already-elapsed cooldown), so a
// kill-triggered reset (Lirvein's «Изощренный бросок») lets the player recast at once.
func (s *Server) resetCooldownsLocked(c *conn, now float64) {
	hs := c.huntState
	for i := range hs.cooldownUntil {
		hs.cooldownUntil[i] = 0
	}
	for slot := 1; slot <= 4; slot++ {
		s.pushAvatarAllLocked(c, battleproto.CmdActionDone, amf.NewArray().
			Set("id", c.objID).
			Set("action", skillProtoID(hs.av, slot)).
			Set("item", false).
			Set("cooldown", now))
	}
}

func (s *Server) pullMobLocked(c *conn, m *mobState, now float64) {
	hs := c.huntState
	cx, cy := c.posAtLocked(float32(now))
	d := math.Hypot(float64(m.x-cx), float64(m.y-cy))
	if d < 1.5 {
		return
	}
	nx := cx + float32(float64(m.x-cx)*1.5/d)
	ny := cy + float32(float64(m.y-cy)*1.5/d)
	m.x, m.y = nx, ny
	idx := hs.tr.index(m.id)
	if idx >= 0 {
		s.push(c, battleproto.CmdSync, amf.NewArray().Set("data",
			newSyncBlob(float32(now)).position(idx, m.x, m.y, 0, 0, float32(now)).
				build(hs.tr.count())))
	}
	s.stopMobLocked(c, m, now)
}
