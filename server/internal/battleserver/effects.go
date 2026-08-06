package battleserver

import (
	"log"
	"math"
	"math/rand"
	"sort"
	"strings"

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
	return s.fxStartCounterLocked(c, fx, owner, target, hasPos, px, py, 0)
}

// fxStartCounterLocked is fxStartLocked plus a "counter" wire arg -- the client's
// VisualEffectHolder.Init only applies it when > 1 (VisualEffect.SetEffectCounter ->
// ParticlesMgr.SetCounter, which multiplies the effect's OWN particle emitter's
// minEmission/maxEmission by it, confirmed straight off the decompiled client). It is how
// a server-side COUNT (Gellar's banked souls) becomes a visually proportional number of
// "soul" particles in one EFFECT_START, rather than one EFFECT_START per soul. counter<=1
// omits the arg entirely, identical to plain fxStartLocked.
func (s *Server) fxStartCounterLocked(c *conn, fx string, owner, target int32, hasPos bool, px, py float32, counter int32) int32 {
	if fx == "" {
		return 0
	}
	if c.inst != nil {
		return s.worldFxStartCounterLocked(c, fx, owner, target, hasPos, px, py, counter)
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
	if counter > 1 {
		args.Set("counter", counter)
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

// noteCastFxLocked records an fx uid as belonging to the current cast of a slot, and
// resets the list first when this is the cast's own opening effect. See huntState.castFx.
func (hs *huntState) noteCastFxLocked(slot int, uid int32, first bool) {
	if slot < 1 || slot > len(hs.castFx) {
		return
	}
	if first {
		hs.castFx[slot-1] = hs.castFx[slot-1][:0]
	}
	if uid != 0 {
		hs.castFx[slot-1] = append(hs.castFx[slot-1], uid)
	}
}

// liveCastFxLocked returns a copy of the current cast's fx uids for a slot.
func (hs *huntState) liveCastFxLocked(slot int) []int32 {
	if slot < 1 || slot > len(hs.castFx) || len(hs.castFx[slot-1]) == 0 {
		return nil
	}
	return append([]int32(nil), hs.castFx[slot-1]...)
}

// scheduleFxEnd ends a cast fx after d seconds of battle time (via the tick
// loop's timed queue, so it survives bursts and honors mvMu).
func (hs *huntState) scheduleFxEnd(uid int32, at float64) {
	hs.scheduleFxEndThen(uid, at, "", 0)
}

// scheduleFxEndThen additionally starts `then` on `thenOwner` at the same moment the uid
// ends -- what the effect turns INTO when it lapses (Frost's ice block shattering).
func (hs *huntState) scheduleFxEndThen(uid int32, at float64, then string, thenOwner int32) {
	if uid == 0 {
		return
	}
	hs.fxEnds = append(hs.fxEnds, fxEnd{uid: uid, at: at, then: then, thenOwner: thenOwner})
}

type fxEnd struct {
	uid       int32
	at        float64
	then      string
	thenOwner int32
}

// ---- orders (approach-then-cast) ----

// pendingCast is a skill order waiting for the avatar to get in range.
type pendingCast struct {
	slot    int
	target  int32 // mob id, 0 for point/self casts
	allyObj int32 // friendly avatar objID for a FRIEND cast, 0 otherwise
	px, py  float32
	hasPos  bool
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
		s.execCastLocked(c, slot, nil, 0, 0, false, 0)
		return
	}

	// Resolve where the cast lands.
	var ms *mobState
	var allyObj int32
	tx, ty := px, py
	if target > 0 && target != c.objID {
		ms = hs.mobs[target]
		switch {
		case ms == nil:
			// Not a mob: either a FRIEND-castable skill (Arianna's «Щит хранителя» /
			// «Касание спасителя») aimed at a party member's avatar, or an ENEMY-castable
			// skill aimed directly at a rival HERO (Teridin's sniper shot clicked on the
			// enemy carry) -- heroes live in inst.members, not hs.mobs, so neither ever
			// resolved as ms. A FRIEND cast carries the ally objID through so its heal/
			// shield/buff lands on THAT ally, not the caster; an ENEMY cast against a hero
			// resolves to a disposable shadow (see pvp_hero_targets.go) so it flows through
			// exactly like a mob target from here on. Anything else (a stale/dead id, or a
			// hero aimed at with a FRIEND-only skill) fizzles.
			if ally := c.friendlyMember(target); ally != nil && skillHasTargetFlag(def, "FRIEND") {
				allyObj = target
				tx, ty = ally.posAtLocked(s.battleTime())
				hasPos = true
			} else if sh := s.dotaEnemyHeroShadowLocked(c, target, float64(s.battleTime())); sh != nil {
				ms = sh
				tx, ty = sh.x, sh.y
				hasPos = true
			} else {
				s.orderDoneLocked(c, parent)
				return
			}
		case ms.dead:
			s.orderDoneLocked(c, parent)
			return
		default:
			// «Штурм» friendly fire, single-target arm. The AoE scans filter allies
			// themselves (mobsWithinLocked), but damageTargetsLocked hands an op with no
			// radius its ctx.target VERBATIM -- and that target starts here. This is also
			// where OpPull's victim comes from, which bypasses opTargetsLocked entirely.
			// Gate on the skill's own declared mask: FRIEND skills are castable on an ally
			// creep/building; a hostile-only skill turns an ally target away.
			if !ms.enemyOf(c.playerTeam()) && !skillHasTargetFlag(def, "FRIEND") {
				s.orderDoneLocked(c, parent)
				return
			}
			tx, ty = ms.x, ms.y
			hasPos = true
		}
	}

	// A self-cast of a friend-or-foe skill (the client now lets NOT_BUILDING accept a click on
	// the caster -- Frost s3, Kiona s4): route it through the ally path so the ally-side ops
	// (heal/shield/buff) land on the caster. A pure enemy skill aimed at self still fizzles.
	if target == c.objID && allyObj == 0 && skillHasTargetFlag(def, "FRIEND") {
		allyObj = c.objID
		tx, ty = c.posAtLocked(s.battleTime())
		hasPos = true
	}

	// A unit-target skill cast with no valid target/position: fire in place.
	if ms == nil && allyObj == 0 && !hasPos {
		s.execCastLocked(c, slot, nil, 0, 0, false, 0)
		return
	}

	// In range? Cast. Otherwise chase toward the cast point.
	cx, cy := c.posAtLocked(s.battleTime())
	maxDist := float64(def.Distance)
	if maxDist <= 0 {
		maxDist = 2.5
	}
	if math.Hypot(float64(tx-cx), float64(ty-cy)) <= maxDist+0.5 {
		s.execCastLocked(c, slot, ms, tx, ty, hasPos, allyObj)
		return
	}
	hs.order = &pendingCast{slot: slot, target: target, allyObj: allyObj, px: px, py: py, hasPos: hasPos}
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
	switch {
	case o.allyObj != 0:
		// Chasing to cast a FRIEND skill on a party member: track the ally's position.
		ally := c.friendlyMember(o.allyObj)
		if ally == nil {
			hs.order = nil
			s.orderDoneLocked(c, skillProtoID(hs.av, o.slot))
			return
		}
		tx, ty = ally.posAtLocked(float32(now))
	case o.target > 0:
		ms = hs.mobs[o.target]
		if ms == nil || ms.dead {
			// Not a mob (or it died mid-chase): re-try as a live enemy hero, the twin of
			// startSkillOrderLocked's own fallback -- an approach-cast started on a rival
			// hero must keep tracking them tick by tick, exactly like chasing a mob.
			ms = s.dotaEnemyHeroShadowLocked(c, o.target, now)
		}
		if ms == nil {
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
		s.execCastLocked(c, o.slot, ms, tx, ty, o.hasPos, o.allyObj)
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
	// Elgorm's «Стрелы Аркана» (ground arrow rain) and Morlokai's «Кабала» (a caster-held
	// hold-target siphon) both END when the caster acts again -- moves, is stunned, casts,
	// or attacks. The tick already breaks a unit channel on move/stun; the interruptible
	// flag additionally lets a new cast or attack cancel it (breakInterruptibleChannelsLocked).
	return (prefab == "Avtr_Dsb_Elgorm" && slot == 4) ||
		(prefab == "Avtr_Dsb_Morlokay" && slot == 2)
}

// channelSustainsThroughDisruption reports whether a SELF-only channel (no ground point,
// no held unit) keeps ticking through movement and stuns instead of breaking, like a
// ground-anchored channel does. Every self/unit channel breaks on move/stun by default
// (Abominator's «Пожирание» drain is the model this defaults to: it needs the caster to
// keep concentrating). BlackDragon's «Взмах погибели» is a different shape -- a self-buff
// rage that flaps and pulses AoE damage around him for its own 8s window, not something
// that requires him to stand still or go uninterrupted; nothing in this file's engine
// distinguished "a buff that happens to tick" from "a channel you must actively sustain"
// until this flag existed. Hand-maintained, keyed by prefab+slot like channelInterruptible.
func channelSustainsThroughDisruption(prefab string, slot int) bool {
	// Gellar's «Армия душ»: the souls are already loosed the instant it is cast (each
	// pulse is one banked soul being spent, not the caster actively sustaining a beam/
	// hold), so it is not something walking or being stunned should be able to cut off.
	return (prefab == "Avtr_DPS_BlackDragon" && slot == 4) ||
		(prefab == "Avtr_DPS_Gellar" && slot == 4)
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
	// Miriam's «Убийственный залп» (slot 4): the arrow rain is a spawner-of-spawners --
	// MiriamSkill4Effect's PrefabTimeSpawn only starts dropping arrows at m_SpawnTime=1.0
	// (matching this skill's own Interval:1), and each arrow then falls for a further,
	// unmeasured stretch before its own TimeDestroy(1.5) despawns it -- no SmoothMove/
	// velocity field on the arrow prefab gives an exact fall speed the way Frost's bolt
	// did. What IS hard data is the floor: damage cannot land before the FIRST arrow even
	// exists, so the channel's first tick moves from t=0 (instant, before any arrow had
	// spawned -- «урон наносится сразу») to t=1.0, in step with every following tick.
	if prefab == "Avtr_DPS_Miriam" && slot == 4 {
		return 1.0
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

// skillOverTimeDur returns the longest DoT/HoT duration a skill applies at the given rank
// (0 if it applies none) -- how long an effect that RIDES a unit is actually meant to last.
func skillOverTimeDur(def gamedata.Skill, level int) float64 {
	var d float64
	for _, op := range def.Ops {
		switch op.Kind {
		case gamedata.OpDot, gamedata.OpHot:
			if v := op.Dur.At(level); v > d {
				d = v
			}
		}
	}
	return d
}

// skillHasAnyBuffStat reports whether a skill applies an OpBuffStat anywhere -- at the top
// level or nested one level inside another op (a channel's per-tick ops, a proc's, ...).
// Used to tell a channel that IS a stat buff (Hekata's «Пепельный смерч», which gets its
// self BuffFx from its own nested OpBuffStat re-applying every tick) from a channel that
// has no stat mod to ride at all (BlackDragon/Gayal/Gellar's ults), which otherwise never
// starts its BuffFx.
func skillHasAnyBuffStat(def gamedata.Skill) bool {
	for _, op := range def.Ops {
		if op.Kind == gamedata.OpBuffStat {
			return true
		}
		for _, nested := range op.Ops {
			if nested.Kind == gamedata.OpBuffStat {
				return true
			}
		}
	}
	return false
}

// skillHoldsTarget reports whether a skill both CHANNELS and pins its victim with a
// root/silence -- i.e. the channel is a GRIP, and the crowd control is that grip made
// visible rather than an independent debuff that should outlive it.
func skillHoldsTarget(def gamedata.Skill) bool {
	var channels, holds bool
	for _, op := range def.Ops {
		switch op.Kind {
		case gamedata.OpChannel:
			channels = true
		case gamedata.OpRoot, gamedata.OpSilence:
			holds = true
		}
	}
	return channels && holds
}

// skillStealthsCaster reports whether casting this skill makes the CASTER invisible --
// On:"allies" is excluded, since that cloaks OTHERS while the caster (Sandariel's
// «Сокрывающая вуаль») explicitly stays visible and should resume auto-attacking normally.
func skillStealthsCaster(def gamedata.Skill) bool {
	for _, op := range def.Ops {
		if op.Kind == gamedata.OpStealth && op.On != "allies" {
			return true
		}
	}
	return false
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
func (s *Server) execCastLocked(c *conn, slot int, ms *mobState, px, py float32, hasPos bool, allyObj int32) {
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

	// Acting breaks stealth: a successful skill cast reveals the player, so mobs can
	// re-aggro at once ("атаки и способности не снимают невидимость"). Placed after the
	// mana/cooldown gate so a fizzled cast doesn't reveal them. This runs BEFORE this cast's
	// ops, so a stealth skill's own OpStealth (Lirvein/Wilfang) survives -- it re-cloaks after
	// this reveal and then breaks on the NEXT action.
	s.breakInvisibilityLocked(c, now)

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
	// A CHANNEL's cast fx carries the sustained pose: the client's baked holder for these
	// skills sets mLoopAnimation on the Cast clip, so the loop runs until EFFECT_END. Its
	// natural end is the end of the CHANNEL, not the authored one-shot CastFxDur -- which
	// is sized for a normal cast and would drop the pose (and, for Einzenhaim's volley, the
	// shot spawner riding the same effect) part-way through. The channel itself starts at
	// the payload, hence the +PayloadDelay.
	if d := skillChannelDur(def, level); d > 0 {
		if full := def.PayloadDelay + d; full > castDur {
			castDur = full
		}
	}
	hs.scheduleFxEnd(castUID, now+castDur)
	hs.noteCastFxLocked(slot, castUID, true)

	// Payload: fx at the victim/point + the actual ops.
	delay := def.PayloadDelay
	targetID := int32(0)
	if ms != nil {
		targetID = ms.id
	}
	hs.payloads = append(hs.payloads, payload{
		at: now + delay, slot: slot, level: level,
		target: targetID, allyObj: allyObj, px: px, py: py, hasPos: hasPos,
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
		// channel pose and start swinging mid-channel). A skill that cloaks the CASTER
		// (not Sandariel's «Сокрывающая вуаль», which stealths allies while she explicitly
		// stays visible) must not roll into auto-attack either -- the whole point of going
		// invisible is not immediately undone by the engine itself swinging at the nearest
		// enemy the instant the cast closes (Wilfang's «Засада»: "аватар не должен
		// переключаться на автоатаку после применения этой способности").
		noResume: skillHasChannel(def) || skillStealthsCaster(def),
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
	// allyObj is the friendly avatar objID a FRIEND skill was cast on (0 otherwise);
	// firePayloadLocked resolves it to ctx.allyTarget at impact time.
	allyObj int32
	px, py  float32
	hasPos  bool
	// ops, when non-nil, is the exact op list to run instead of the whole skill's
	// def.Ops -- used to defer a dash's follow-up ops (damage/root) until arrival.
	ops []gamedata.Op
	// resume, on a StrikeOnArrival continuation, rolls the avatar back into
	// auto-attack once the charge lands. The action-done's own resume attempt fires
	// mid-dash (hasDest set) and bails, so the charge needs its own post-arrival one.
	resume bool
	// anchor, on the ARRIVAL half of a thrown payload (Skill.PayloadFlight), is the
	// invisible object standing at the aim point: the impact and linger fx are owned to
	// it so they play THERE and not on the caster.
	anchor int32
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
		// A target stat-buff, as before -- and also the CROWD CONTROL a skill pins on its
		// target, which is how long a visual REPRESENTING that state should stand. Frost's
		// «Гробница холода» is the case: the ice block IS its BuffFx, but the skill's only
		// OpBuffStat targets an ALLY, so the TTL came out 0 and the block was never shown
		// at all -- «не создаётся prop». The encasement itself is an OpStun (both halves of
		// the dual cast carry one), and that is exactly how long the ice should stand.
		buff := op.Kind == gamedata.OpBuffStat && op.On == "target"
		encase := op.Kind == gamedata.OpStun || op.Kind == gamedata.OpRoot
		if !buff && !encase {
			continue
		}
		if d := op.Dur.At(level); d > ttl {
			ttl = d
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
		if ms == nil {
			// Not a live mob: p.target may name an enemy HERO instead (see
			// pvp_hero_targets.go) -- re-resolved fresh here, at IMPACT time rather than
			// cast time, so a delayed payload lands on the hero's current position/HP,
			// not a stale snapshot from when the cast started.
			ms = s.dotaEnemyHeroShadowLocked(c, p.target, now)
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
			// A SELF-baked target-mode fx follows its OWNER; for those skills own it to the
			// struck enemy so the visual lands on the victim, not the caster (Sharli s1).
			// A friend-or-foe cast (Kiona's «Страж леса») may have been aimed at an ALLY
			// instead, in which case the ally's object -- not the mob slot -- is the thing to
			// hang it on; otherwise the owl fell back to the caster.
			fxOwner := c.objID
			if payloadTargetFxOwnedToTarget(hs.av.Prefab, p.slot) {
				switch {
				case p.allyObj != 0:
					fxOwner, tid = p.allyObj, p.allyObj
				case tid != 0:
					fxOwner = tid
				}
				// A visual pinned to its charge must last as long as the thing it
				// represents: the owl guards its target for the full 10s of the
				// heal/damage over time, not the default one-shot window.
				if d := skillOverTimeDur(def, p.level); d > fxLife {
					fxLife = d
				}
			}
			uid := s.fxStartLocked(c, def.PayloadFx, fxOwner, tid, p.hasPos, fpx, fpy)
			hs.scheduleFxEnd(uid, now+fxLife)
			hs.noteCastFxLocked(p.slot, uid, false)
			// A visual RIDING a mob dies with the mob. Remember it on the body so the
			// death branch can end it: the scheduled end is minutes of game-time away in
			// owl terms, and a guardian left circling a corpse (or orphaned when the body
			// is deleted) is exactly what the player sees.
			if ms != nil && fxOwner == ms.id {
				s.worldFxEndLocked(c, ms.st.riderFx) // a re-cast replaces the old one
				ms.st.riderFx = uid
			}
			// A projectile the CLIENT flies at a fixed speed: hold the ops back until it
			// actually arrives, or the victim loses health before anything reaches them.
			if def.PayloadFlightSpeed > 0 && ms != nil {
				cx, cy := c.posAtLocked(float32(now))
				flight := math.Hypot(float64(ms.x-cx), float64(ms.y-cy)) / def.PayloadFlightSpeed
				if flight > 0.02 {
					hs.payloads = append(hs.payloads, payload{
						at: now + flight, slot: p.slot, level: p.level,
						target: p.target, allyObj: p.allyObj,
						px: p.px, py: p.py, hasPos: p.hasPos, ops: def.Ops,
					})
					return
				}
			}
		case "point":
			// A SELF-baked ground fx trails the caster; for a skill whose point payload
			// is SELF-mode (Titanid's «Землетрясение» quake) pin it to an invisible
			// stationary anchor at the point instead of owning it to the moving avatar.
			// A DELAYED ground payload (PlusMinus's ball lightning, which sits on the
			// ground and detonates later) always needs one: the thing it plants has to
			// stay where it was planted for the whole fuse, whatever the caster does.
			fxOwner, anchor := c.objID, int32(0)
			if payloadFxUsesAnchor(hs.av.Prefab, p.slot) || def.PayloadFlight > 0 {
				anchor = s.spawnTrapAnchorLocked(c, fpx, fpy, now)
				fxOwner = anchor
			}
			// A fused payload's fx is the FUSE: it ends when the thing goes off, and the
			// ops (plus ImpactFx) are re-scheduled for that moment.
			life := fxLife
			if def.PayloadFlight > 0 {
				life = def.PayloadFlight
			}
			uid := s.fxStartLocked(c, def.PayloadFx, fxOwner, 0, true, fpx, fpy)
			hs.scheduleFxEnd(uid, now+life)
			hs.noteCastFxLocked(p.slot, uid, false)
			if def.PayloadFlight > 0 {
				hs.payloads = append(hs.payloads, payload{
					at: now + def.PayloadFlight, slot: p.slot, level: p.level,
					target: p.target, allyObj: p.allyObj,
					px: p.px, py: p.py, hasPos: p.hasPos,
					ops: def.Ops, anchor: anchor,
				})
				return // the arrival half removes the anchor
			}
			if anchor != 0 {
				hs.anchorEnds = append(hs.anchorEnds, anchorEnd{id: anchor, at: now + fxLife + 0.3})
			}
		case "self":
			uid := s.fxStartLocked(c, def.PayloadFx, c.objID, 0, false, 0, 0)
			hs.scheduleFxEnd(uid, now+fxLife)
			hs.noteCastFxLocked(p.slot, uid, false)
		case "throw":
			// A THROWN payload. The client's prefab for it is SELF_TO_TARGET -- it flies
			// from the caster to a target OBJECT -- so a bare point cast gives it nothing to
			// aim at and it silently never renders. Pin an invisible anchor at the aim point
			// and fly to that. The ops do NOT run here: they are re-scheduled for the moment
			// the throw lands (PayloadFlight), which is the whole point -- «бутылка должна
			// лететь и приземляться».
			anchor := s.spawnTrapAnchorLocked(c, fpx, fpy, now)
			flight := def.PayloadFlight
			uid := s.fxStartLocked(c, def.PayloadFx, c.objID, anchor, true, fpx, fpy)
			hs.scheduleFxEnd(uid, now+flight+0.3)
			hs.payloads = append(hs.payloads, payload{
				at: now + flight, slot: p.slot, level: p.level,
				target: p.target, allyObj: p.allyObj,
				px: p.px, py: p.py, hasPos: p.hasPos,
				ops: def.Ops, anchor: anchor,
			})
			return
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
		//
		// It also covers an ALLY-side cast. Frost's «Гробница холода» encases «врага ИЛИ
		// союзника», and the ice block is this very BuffFx -- owned to a mob it never even
		// appeared when the ice was cast on a friend.
		if def.BuffFxOn == "target" && def.BuffFx != "" {
			owner := int32(0)
			switch {
			case ms != nil:
				owner = ms.id
			case p.allyObj != 0:
				owner = p.allyObj
			}
			if ttl := targetBuffTTL(def, p.level); ttl > 0 && owner != 0 {
				uid := s.fxStartLocked(c, def.BuffFx, owner, 0, false, 0, 0)
				// TargetFxEnd is what the visual turns INTO when it lapses (Frost's ice block
				// SHATTERING, sound and all): started on the same owner at the same moment.
				hs.scheduleFxEndThen(uid, now+ttl, def.TargetFxEnd, owner)
				if ms != nil {
					s.worldFxEndLocked(c, ms.st.riderFx)
					ms.st.riderFx = uid
				}
			}
		}
	}
	// ARRIVAL of a thrown payload: the explosion and the lingering ground effect, both
	// owned by the anchor so they play at the landing point rather than on the caster.
	// Started BEFORE the ops so a channel created below picks them up as its own fx (and
	// therefore ends them if it breaks early).
	if p.anchor != 0 {
		lingerLife := skillChannelDur(def, p.level)
		if lingerLife <= 0 {
			lingerLife = 3.0
		}
		if def.ImpactFx != "" {
			uid := s.fxStartLocked(c, def.ImpactFx, p.anchor, 0, false, 0, 0)
			hs.scheduleFxEnd(uid, now+3.0)
		}
		if def.LingerFx != "" {
			uid := s.fxStartLocked(c, def.LingerFx, p.anchor, 0, false, 0, 0)
			hs.scheduleFxEnd(uid, now+lingerLife)
			hs.noteCastFxLocked(p.slot, uid, false)
		}
		// The anchor must outlive every fx parented to it, or they are torn down with it.
		hs.anchorEnds = append(hs.anchorEnds, anchorEnd{id: p.anchor, at: now + math.Max(lingerLife, 3.0) + 0.3})
	}

	ops := def.Ops
	if p.ops != nil {
		ops = p.ops
	}
	ctx := opCtx{slot: p.slot, level: p.level, target: ms, px: p.px, py: p.py, hasPos: p.hasPos}
	if p.allyObj != 0 {
		ctx.allyTarget = c.friendlyMember(p.allyObj) // may be nil if the ally left/died
	}
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
		if op.Kind == gamedata.OpShieldExplode {
			// Rognar's «Костяной щит»: arm the hit-counted blast (three incoming hits detonate
			// it). Remembered so the incoming-damage path can tick it down.
			hs.shieldExplodeSlot = slot
			hs.shieldStartedAt = now
			hs.shieldHitsLeft = shieldExplodeHits
		}
		if op.Kind == gamedata.OpZoneArmor {
			// Inshari's «Угнетение»: remember which toggle carries the zone-gated armor so
			// hitPlayerFromLocked can read its live Value/Radius against the attacker's
			// position.
			hs.zoneArmorSlot = slot
		}
		if op.Kind == gamedata.OpMeleeForm {
			// Grimlok's «Темная сторона»: force melee range/no-projectile for the duration,
			// restored on toggle-off.
			hs.meleeFormSlot = slot
			hs.meleeFormWasProjectile = hs.hasProjectile
			hs.hasProjectile = false
		}
		if op.Kind == gamedata.OpStealth {
			// Astarot's «Слуга тьмы»: real invisibility for as long as the toggle stays on
			// (mana holds out). tickTogglesLocked re-arms invisibleUntil every tick; start it
			// here too so stealth is live from the moment of activation, not one tick later.
			hs.toggleStealthSlot = slot
			s.applySkillStealthLocked(c, 1.0, op.BreakOnMove, now)
		}
		if op.Kind == gamedata.OpAttackManaBonus {
			// Miriam's «Зачарованные стрелы»: arm the mana-fueled attack window for as long
			// as the toggle stays on -- a far horizon here, cleared explicitly in
			// toggleOffLocked, rather than the op's own (unused for a toggle) Dur.
			hs.manaShotSlot = slot
			hs.manaShotDmg = op.Value.At(level)
			if op.PerSP > 0 {
				hs.manaShotDmg += hs.spellPowerLocked(now) * op.PerSP
			}
			hs.manaShotCost = op.Value2.At(level)
			hs.manaShotUntil = now + 1e9
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
	if hs.shieldExplodeSlot == slot {
		hs.shieldExplodeSlot = 0 // bone shield down: stop counting hits
	}
	if hs.zoneArmorSlot == slot {
		hs.zoneArmorSlot = 0
	}
	if hs.meleeFormSlot == slot {
		hs.meleeFormSlot = 0
		hs.hasProjectile = hs.meleeFormWasProjectile
	}
	if hs.toggleStealthSlot == slot {
		hs.toggleStealthSlot = 0
		s.breakInvisibilityLocked(c, now)
	}
	if hs.manaShotSlot == slot {
		hs.manaShotSlot = 0
		hs.manaShotUntil = 0
	}
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

const (
	// shieldExplodeHits is how many incoming hits Rognar's «Костяной щит» absorbs before it
	// detonates («при получении трёх ударов»).
	shieldExplodeHits = 3
	// shieldExplodeFullWindow is the age (seconds) at which the blast has decayed all the way
	// from its max to its min. The client says only «чем меньше времени — тем больше урон»
	// with no explicit ceiling, so this is a chosen span that makes an instant pop hit
	// hardest and a long-standing shield hit softest.
	shieldExplodeFullWindow = 8.0
)

// explodeBoneShieldLocked detonates Rognar's «Костяной щит»: it blasts enemies within the
// op's radius for a magnitude that decays from Value2 (max, just cast) toward Value (min,
// stood the full window), then switches the toggle off. A no-op if no bone shield is up.
func (s *Server) explodeBoneShieldLocked(c *conn, now float64) {
	hs := c.huntState
	slot := hs.shieldExplodeSlot
	if slot < 1 {
		return
	}
	level := int(hs.skillLevel[slot-1])
	for _, op := range hs.skillDef(slot).Ops {
		if op.Kind != gamedata.OpShieldExplode || level < 1 {
			continue
		}
		frac := (now - hs.shieldStartedAt) / shieldExplodeFullWindow
		if frac < 0 {
			frac = 0
		} else if frac > 1 {
			frac = 1
		}
		mn := op.Value.At(level) * hs.powerMul()
		mx := op.Value2.At(level) * hs.powerMul()
		if op.PerSP > 0 {
			sp := hs.spellPowerLocked(now) * op.PerSP
			mn, mx = mn+sp, mx+sp
		}
		dmg := mx - frac*(mx-mn) // max when fresh, min when it stood the full window
		px, py := c.posAtLocked(float32(now))
		for _, m := range c.mobsWithinLocked(px, py, op.Radius) {
			s.hitMobLocked(c, m, dmg, c.objID)
		}
		break
	}
	// The shield is spent -> switch it off (also clears shieldExplodeSlot).
	s.toggleOffLocked(c, slot, now, false)
}

// ---- op execution ----

// opCtx carries the resolution context of one ops batch.
type opCtx struct {
	slot   int
	level  int
	target *mobState // nil for self/point casts
	px, py float32
	hasPos bool
	toggle bool
	dmgIn  float64 // size of the hit that triggered an on-damaged proc (0 otherwise)
	// dmgBonus is flat extra damage folded into this call's OpDamage (Op.Growth's
	// per-pulse channel ramp, set by tickChannelsLocked; 0 otherwise).
	dmgBonus float64
	// durBonus is flat extra duration folded into this call's OpStun (Op.Growth's
	// per-pulse channel ramp on a nested stun, set by tickChannelsLocked; 0 otherwise).
	durBonus float64
	// radiusBonus widens this call's AoE (Op.RadiusGrowth's per-pulse channel ramp, set
	// by tickChannelsLocked; 0 otherwise).
	radiusBonus float64
	// allyTarget is the friendly avatar a FRIEND-castable skill was aimed at (nil for a
	// self/AoE cast). Ops with On=="ally" apply to it (or the caster if nil).
	allyTarget *conn
	// hitCount is how many enemies this cast's OpDamage actually landed on, set by that
	// case for a LATER op in the SAME ops list to read (Op.ScalePerHit, Nerlag's
	// «Поголовная бойня»). 0 until an OpDamage has run.
	hitCount int
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

// mobsWithinLocked collects living ENEMIES whose body (centre within r + the mob's
// own radius) overlaps the circle of radius r at (x,y), so an AoE that reaches a
// big boss's edge still hits it. See mobState.enemyOf: in «Штурм» this map also
// holds the caster's own creeps and buildings, and every op routed through here
// (damage, DoT, stun, root, slow, silence, knockback) was landing on them.
func (c *conn) mobsWithinLocked(x, y float32, r float64) []*mobState {
	var out []*mobState
	for _, m := range c.huntState.mobs {
		if m.dead || !m.enemyOf(c.playerTeam()) {
			continue
		}
		if math.Hypot(float64(m.x-x), float64(m.y-y)) <= r+m.mob.Radius() {
			out = append(out, m)
		}
	}
	return out
}

// mobHasAllyNearLocked reports whether the target mob has ANY OTHER living enemy within
// radius of it -- i.e. it is NOT isolated. Used by the TargetIsolated op gate (Vigilans's
// «Свидание со смертью»). Every hostile mob is an ally of every other, so any other live
// mob in range counts (a Hunt pack, a «Штурм» creep wave).
func (s *Server) mobHasAllyNearLocked(c *conn, target *mobState, radius float64) bool {
	for _, m := range c.mobsWithinLocked(target.x, target.y, radius) {
		if m.id != target.id {
			return true
		}
	}
	return false
}

// friendlyMember resolves an object id to a same-instance party member's conn (a
// world-ready avatar on the caster's side), or nil. Used to aim FRIEND-castable skills
// at another player. In solo (no instance) only the caster's own id resolves.
func (c *conn) friendlyMember(objID int32) *conn {
	if c.inst == nil {
		if objID == c.objID {
			return c
		}
		return nil
	}
	mem := c.inst.members[objID]
	if mem == nil || mem.huntState == nil || arenaEnemies(c, mem) {
		return nil
	}
	return mem
}

// allyTargetsLocked resolves the friendly-avatar recipients of an ally-targeting op:
//   - On=="ally"   -> the single aimed ally (ctx.allyTarget) or the caster if none/gone.
//   - On=="allies" -> every living party member within the op's radius of the AoE centre.
//
// Self is an ally: a self-centred AoE always catches the caster (distance 0), and a
// point AoE catches the caster only if they stand in it -- matching «все союзники в
// области». In solo (members() == [caster]) both forms collapse to the caster, so these
// skills stay visible in single-player.
func (s *Server) allyTargetsLocked(c *conn, ctx opCtx, op gamedata.Op) []*conn {
	switch op.On {
	case "ally":
		if a := ctx.allyTarget; a != nil && a.huntState != nil && a.huntState.deadUntil == 0 {
			return []*conn{a}
		}
		return []*conn{c}
	case "allies":
		cx, cy := s.centerLocked(c, ctx)
		r := op.Radius
		if r <= 0 {
			r = float64(c.huntState.skillDef(ctx.slot).AoERadius)
		}
		if r <= 0 {
			r = 4
		}
		now := s.battleTime()
		var out []*conn
		for _, mem := range c.members() {
			hs := mem.huntState
			if hs == nil || hs.deadUntil > 0 || arenaEnemies(c, mem) {
				continue
			}
			mx, my := mem.posAtLocked(now)
			if math.Hypot(float64(mx-cx), float64(my-cy)) <= r {
				out = append(out, mem)
			}
		}
		return out
	}
	return nil
}

// applyShieldLocked sets an absorb shield on any member (self or ally) and starts the
// shared shield VFX on that avatar, so every viewer sees it. Mirrors the self OpShield
// path but is parameterized by target conn.
func (s *Server) applyShieldLocked(target *conn, amount, until float64) {
	hs := target.huntState
	if hs == nil {
		return
	}
	hs.st.shield = amount
	hs.st.shieldUntil = until
	if hs.st.shieldFx == 0 {
		hs.st.shieldFx = s.fxStartLocked(target, "RuneShieldEffect3", target.objID, 0, false, 0, 0)
	}
}

// addAllyHotLocked arms a heal-over-time on any member (self or ally); the recipient's
// own tick drains it. perSec/until come from the CASTER's power (the heal is the
// caster's), applied to the ally's status block.
func (s *Server) addAllyHotLocked(target *conn, perSec, until, now float64) {
	hs := target.huntState
	if hs == nil {
		return
	}
	hs.st.hots = append(hs.st.hots, overTime{perSec: perSec, until: until, nextTick: now + 1})
}

// addAllyModLocked applies a stat mod to ANOTHER member (an ally) and re-syncs that
// member's affected stats to its own client. Unlike addPlayerModLocked it does NOT add a
// buff-bar effector, because the buff icon/fx protos resolve against the CASTER's kit,
// not the ally's -- the stat still works and syncs, only the icon is omitted.
func (s *Server) addAllyModLocked(target *conn, op gamedata.Op, ctx opCtx, now float64) {
	hs := target.huntState
	if hs == nil {
		return
	}
	until := 0.0
	if d := op.Dur.At(ctx.level); d > 0 {
		until = now + d
	}
	hs.st.mods = append(hs.st.mods, statMod{
		stat: op.Stat, value: op.Value.At(ctx.level), until: until, src: castSrc(ctx),
	})
	s.pushPlayerStatsLocked(target, now)
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
		if m.dead || !m.enemyOf(c.playerTeam()) { // allies are not in the swath -- see enemyOf
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

// skillHasTargetFlag reports whether a skill's declared target mask names `flag`.
//
// The mask is the client's own TanatKernel.SkillTarget enum (FRIEND=1, ENEMY=4,
// BUILDING=0x10, NOT_BUILDING=0x20 ...), stored '+'-joined -- "ENEMY+NOT_BUILDING" --
// and shipped verbatim to the client in the effector description, where
// TargetValidator enforces it on the player's click. It is authored game data, so it
// is the right authority for "may this cast land on that unit" on the server too.
//
// Tokens are matched WHOLE: a naive substring test would see FRIEND inside NOT_FRIEND
// and invert the rule.
func skillHasTargetFlag(sk gamedata.Skill, flag string) bool {
	for _, f := range strings.Split(sk.Target, "+") {
		if strings.TrimSpace(f) == flag {
			return true
		}
	}
	return false
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
		out := c.mobsAlongLineLocked(cx, cy, ctx.px, ctx.py, float64(sk.AoEWidth)/2, length)
		// Enemy HEROES stand in this same swath via a disposable shadow (see
		// pvp_hero_targets.go) -- without this a line skill aimed through the enemy
		// team only ever caught their creeps, never the heroes among them.
		out = append(out, s.dotaEnemyHeroShadowsAlongLineLocked(c, cx, cy, ctx.px, ctx.py, float64(sk.AoEWidth)/2, length)...)
		return out
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
	out := c.mobsWithinLocked(x, y, radius)
	// Enemy HEROES stand in this same radius scan via a disposable shadow (see
	// pvp_hero_targets.go) -- without this an AoE nuke aimed at the enemy team only
	// ever caught their creeps, never the heroes standing among them.
	out = append(out, s.dotaEnemyHeroShadowsLocked(c, x, y, radius)...)
	return out
}

// opTargetsLocked resolves a damaging/CC op's victims and applies its MaxTargets cap
// (the N nearest to the AoE centre). A capped op (Rognar's «Могильный холод», two
// targets) hits only that subset of everything in range; uncapped ops are unchanged.
func (s *Server) opTargetsLocked(c *conn, ctx opCtx, op gamedata.Op) []*mobState {
	targets := s.damageTargetsLocked(c, ctx, op.Radius+ctx.radiusBonus)
	// ExcludeCenterTarget drops the AoE's own center (the struck/aimed unit) from its own
	// splash/chain -- Titanid's «Ударная волна» ("все ДРУГИЕ враги"), PlusMinus's
	// «Сверхпроводимость» ("четырем СОСЕДНИМ целям") -- so the primary target isn't hit a
	// second time as one of its own splash victims.
	if op.ExcludeCenterTarget && ctx.target != nil {
		filtered := targets[:0]
		for _, m := range targets {
			if m != ctx.target {
				filtered = append(filtered, m)
			}
		}
		targets = filtered
	}
	// Sort nearest-first whenever a cap OR a per-target decay cares about hop order
	// (PlusMinus's «Сверхпроводимость» chain decays by distance-from-epicentre order),
	// not only when the cap actually trims the list -- otherwise an uncapped-count hit
	// (fewer live enemies than MaxTargets) would decay in undefined map-iteration order.
	if op.MaxTargets > 0 {
		if op.Randomize {
			rand.Shuffle(len(targets), func(i, j int) { targets[i], targets[j] = targets[j], targets[i] })
		} else {
			cx, cy := s.centerLocked(c, ctx)
			sort.Slice(targets, func(i, j int) bool {
				return math.Hypot(float64(targets[i].x-cx), float64(targets[i].y-cy)) <
					math.Hypot(float64(targets[j].x-cx), float64(targets[j].y-cy))
			})
		}
		if len(targets) > op.MaxTargets {
			targets = targets[:op.MaxTargets]
		}
	}
	// PerTargetGrowth (Nerlag's «Метание топоров») walks OUTWARD from the caster along the
	// throw, so the "first, second, third..." index has to be nearest-to-CASTER first --
	// unlike MaxTargets' nearest-to-CENTER sort (the center is the aim point for a point/
	// beam cast, which would rank the growth backwards).
	if op.PerTargetGrowth.At(ctx.level) > 0 || op.PerTargetGrowthSP > 0 {
		cx, cy := c.posAtLocked(s.battleTime())
		sort.Slice(targets, func(i, j int) bool {
			return math.Hypot(float64(targets[i].x-cx), float64(targets[i].y-cy)) <
				math.Hypot(float64(targets[j].x-cx), float64(targets[j].y-cy))
		})
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
	if pct := op.PctOfAttack.At(ctx.level); pct > 0 {
		dmg += hs.baseAttackLocked(now) * pct
	}
	// Soul-scaled bonus (Gellar's «Армия душ»: +damagePerSoul per banked soul).
	if ps := op.PerSoul.At(ctx.level); ps > 0 {
		dmg += ps * float64(hs.soulStacks) * hs.powerMul()
		if op.PerSoulSP > 0 {
			dmg += op.PerSoulSP * hs.spellPowerLocked(now) * float64(hs.soulStacks)
		}
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
		// A non-proc op may carry its own Chance to fire PROBABILISTICALLY inside an
		// aura/channel tick -- Zamaran's «Пламя войны» roots the enemy it damages only
		// «с вероятностью 20%». OpProc keeps its own semantics (registered passives, or
		// unconditional when nested in an active cast), so it is exempt here. The roll is
		// server-side and its result is broadcast, so every client sees the same outcome.
		if op.Kind != gamedata.OpProc {
			if ch := op.Chance.At(ctx.level); ch > 0 && ch < 1 && rand.Float64() >= ch {
				continue
			}
		}
		// Isolation gate: an op flagged TargetIsolated fires only if the aimed enemy has NO
		// living ally within TriggerRadius of it (Vigilans's «Свидание со смертью» punishes a
		// foe caught away from its pack). No target, or an ally nearby, skips the op.
		if op.TargetIsolated {
			r := op.TriggerRadius
			if r <= 0 {
				r = 6
			}
			if ctx.target == nil || s.mobHasAllyNearLocked(c, ctx.target, r) {
				continue
			}
		}
		// Friend-or-foe DUAL cast: an "enemy" op fires only when a foe was aimed, an "ally"
		// op only when a friend was aimed (Kiona's «Страж леса», Frost's «Гробница холода»,
		// Hekata's «Выбор скверны»). This keeps the enemy half from splashing a friend's
		// surroundings and the ally half from defaulting to self on an enemy cast.
		switch op.TargetSide {
		case "enemy":
			if ctx.target == nil {
				continue
			}
		case "ally":
			if ctx.allyTarget == nil {
				continue
			}
		}
		switch op.Kind {
		case gamedata.OpDamage:
			if op.Apply == "self" {
				// RefundIfHit (Abominator's «Бросок плоти»: «теряет здоровье, которое он может
				// восстановить, ударив цель») -- skip the self-cost entirely when the cast's
				// paired enemy-facing hit actually connected (ctx.target alive). A cast that
				// resolved with no live target (rare: it died/left before the payload landed)
				// still pays the cost, matching the client's "can restore BY HITTING" framing.
				if op.RefundIfHit && ctx.target != nil && !ctx.target.dead {
					break
				}
				// A self-sacrifice cost (Abominator): pure health drain, no armor.
				dmg := op.Value.At(ctx.level)
				hs.hp = math.Max(1, hs.hp-dmg) // never suicide on a cost
				s.syncSelfLocked(c, syncHealth)
				break
			}
			targets := s.opTargetsLocked(c, ctx, op)
			for i, m := range targets {
				dmg := s.skillDamageLocked(c, op, ctx, m)
				// EdgeValue (PlusMinus's «Электрошок»): damage is a function of the TARGET'S
				// OWN DISTANCE from the epicenter, continuously -- «в зависимости от их
				// положения. Чем дальше находится враг от эпицентра, тем больший урон он
				// получает» -- not of how many other targets this cast happened to catch or
				// what rank they came in at. skillDamageLocked already gave the CENTER end
				// (Value+PerSP); interpolate it toward the EDGE end (EdgeValue+EdgeValueSP) by
				// how far out of Radius this target actually stands.
				if len(op.EdgeValue) > 0 {
					cx, cy := s.centerLocked(c, ctx)
					frac := 1.0
					if op.Radius > 0 {
						frac = math.Min(1, math.Hypot(float64(m.x-cx), float64(m.y-cy))/op.Radius)
					}
					edge := op.EdgeValue.At(ctx.level) * hs.powerMul()
					if op.EdgeValueSP > 0 {
						edge += hs.spellPowerLocked(now) * op.EdgeValueSP
					}
					dmg = dmg*(1-frac) + edge*frac
				}
				dmg += ctx.dmgBonus
				if pct := op.SelfMaxHPPct.At(ctx.level); pct > 0 {
					dmg = pct * hs.maxHPLocked(now)
				}
				// MissingHPLinear (Inshari's «Возмездие»): a true execute -- damage scales
				// linearly with the TARGET's own missing HP, capped at DamageCap(+PerSP×SP)
				// -- «наносит {*damagePerHP} ... за каждую единицу здоровья отсутствующую у
				// цели, но не более {*damageMax}+{*@damageSP}».
				if lin := op.MissingHPLinear.At(ctx.level); lin > 0 {
					dmg = lin * (m.maxHealth() - m.hp)
					if cap := op.DamageCap.At(ctx.level); cap > 0 {
						capVal := cap
						if op.PerSP > 0 {
							capVal += hs.spellPowerLocked(now) * op.PerSP
						}
						if dmg > capVal {
							dmg = capVal
						}
					}
				}
				if op.PerTargetDecay > 0 {
					dmg *= math.Pow(1-op.PerTargetDecay, float64(i))
				}
				// PerTargetGrowth (Nerlag's «Метание топоров»): each successive target (nearest-
				// to-caster-first, see opTargetsLocked) takes MORE than the last, not the same
				// flat number.
				if g := op.PerTargetGrowth.At(ctx.level); g > 0 || op.PerTargetGrowthSP > 0 {
					bonus := g
					if op.PerTargetGrowthSP > 0 {
						bonus += op.PerTargetGrowthSP * hs.spellPowerLocked(now)
					}
					dmg += bonus * float64(i)
				}
				s.hitMobLocked(c, m, dmg, c.objID)
			}
			// Lets a LATER op in this same cast scale off how many enemies actually got hit
			// (Nerlag's «Поголовная бойня», Op.ScalePerHit).
			ctx.hitCount = len(targets)
			// Op.OnAnyDamage passives (Anhel's «Зов фантомов») also get a roll here: the
			// caster's own skill damage landing, not just a basic attack.
			if len(targets) > 0 {
				s.runAnyDamageProcsLocked(c, now)
			}

		case gamedata.OpExecute:
			// «Казнь»: instant-kill the target if its HP is at/below the threshold (Value2),
			// otherwise deal Value damage. The kill deals exactly the target's remaining HP
			// (pre-divided by armor mitigation) so the client shows a clean lethal blow, not
			// an overkill number, and armor can't save a target under the threshold.
			if m := ctx.target; m != nil && !m.dead {
				if m.hp <= op.Value2.At(ctx.level) {
					mult := s.armorMultLocked(c, m, c.objID, now)
					if mult <= 0 {
						mult = 1
					}
					s.hitMobLocked(c, m, m.hp/mult, c.objID)
				} else {
					s.hitMobLocked(c, m, s.skillDamageLocked(c, op, ctx, m), c.objID)
				}
			}

		case gamedata.OpAttackDamage:
			// Gektor's «Разящий удар»: bonus = Value × the caster's base attack, dealt to the
			// struck target. Sits in a Chance-1 on-hit proc, so ctx.target is the swing's mob.
			if m := ctx.target; m != nil && !m.dead {
				if dmg := op.Value.At(ctx.level) * hs.baseAttackLocked(now); dmg > 0 {
					s.hitMobLocked(c, m, dmg, c.objID)
				}
			}

		case gamedata.OpManaScaledDamage:
			// Neirofim's «Паралич воли»: the base hit ALWAYS lands («наносит магический урон и
			// замедляет цель, в зависимости от количества маны у цели» -- only the MAGNITUDE is a
			// function of mana, not whether it connects at all). A mana-less (melee) mob still
			// takes the flat base+PerSP damage; only the missing-mana bonus and the slow (which
			// have nothing to scale from) are skipped for it.
			if m := ctx.target; m != nil && !m.dead {
				dmg := op.Value.At(ctx.level) * hs.powerMul()
				if op.PerSP > 0 {
					dmg += hs.spellPowerLocked(now) * op.PerSP
				}
				if m.maxMana > 0 {
					dmg += op.Value2.At(ctx.level) * (m.maxMana - m.mana)
				}
				s.hitMobLocked(c, m, dmg, c.objID)
				if dur := op.Dur.At(ctx.level); dur > 0 && m.maxMana > 0 {
					m.st.slowUntil = now + dur
					m.st.slowFactor = 1 - 0.5*(m.mana/m.maxMana)
					s.ensureMobStatusFxLocked(c, m, &m.st.slowFx, "SlowMoveEffect")
					s.syncMobSpeedLocked(c, m, now)
				}
			}

		case gamedata.OpManaBurnHit:
			// BlackDragon's «Выжигание маны» / Neirofim's «Пожирание магии» / Inshari's siphon:
			// drain mana from the struck target on a basic attack (nested in a Chance-1 proc).
			if m := ctx.target; m != nil && !m.dead {
				amt := op.Value.At(ctx.level)
				if op.Apply == "own_mana" {
					amt *= hs.maxManaLocked(now) // a % of the caster's own pool
				} else if op.PerSP > 0 {
					amt += hs.spellPowerLocked(now) * op.PerSP
				}
				if drained := m.drainManaLocked(amt); drained > 0 {
					switch {
					case op.Apply == "restore":
						hs.mana = math.Min(hs.maxManaLocked(now), hs.mana+drained)
						s.syncSelfLocked(c, syncMana)
					default:
						if frac := op.Value2.At(ctx.level); frac > 0 {
							s.hitMobLocked(c, m, drained*frac, c.objID)
						}
					}
				}
			}

		case gamedata.OpManaBurnArea:
			// PlusMinus's «Шаровая молния»: burn mana from every enemy in the blast, no
			// damage-back component.
			cx, cy := s.centerLocked(c, ctx)
			drain := op.Value.At(ctx.level)
			if op.PerSP > 0 {
				drain += hs.spellPowerLocked(now) * op.PerSP
			}
			for _, m := range c.mobsWithinLocked(cx, cy, op.Radius) {
				m.drainManaLocked(drain)
			}

		case gamedata.OpSilenceAll:
			// Neirofim's «Молчание»: silence every hostile mob on the map, and drain mana from
			// those nearby. Boss casting honours silenceUntil (tryBossSkillLocked).
			dur := op.Dur.At(ctx.level)
			for _, m := range hs.mobs {
				if m.dead || !m.enemyOf(c.playerTeam()) {
					continue
				}
				m.st.silenceUntil = math.Max(m.st.silenceUntil, now+dur)
				if m.shown {
					s.ensureMobStatusFxLocked(c, m, &m.st.silenceFx, "SilenceEffect")
				}
			}
			if drain := op.Value.At(ctx.level); drain > 0 {
				cx, cy := s.centerLocked(c, ctx)
				for _, m := range c.mobsWithinLocked(cx, cy, op.Radius) {
					m.drainManaLocked(drain)
				}
			}

		case gamedata.OpChill:
			// Frost «озноб»: chilling an already-chilled target instead stuns it and clears the
			// chill (the signature combo); otherwise it just marks the chill window.
			for _, m := range s.opTargetsLocked(c, ctx, op) {
				if now < m.st.chillUntil {
					s.stunMobLocked(c, m, now, op.Value2.At(ctx.level))
					m.st.chillUntil = 0
					if m.st.chillFx != 0 {
						s.worldFxEndLocked(c, m.st.chillFx)
						m.st.chillFx = 0
					}
				} else if !op.OnlyIfChilled {
					m.st.chillUntil = now + op.Dur.At(ctx.level)
					s.ensureMobStatusFxLocked(c, m, &m.st.chillFx, "FrozenEffect")
				}
			}

		case gamedata.OpEmpowerNextHit:
			// Rognar's «Окропление кровью»: spend Value2 fraction of current HP, store a bonus
			// magic hit of Value × the HP spent onto the next basic attack.
			cost := op.Value2.At(ctx.level) * hs.hp
			if cost > 0 {
				hs.hp = math.Max(1, hs.hp-cost)
				s.syncSelfLocked(c, syncHealth)
				hs.nextHitBonus += op.Value.At(ctx.level) * cost
			}

		case gamedata.OpSelfRecoil:
			// Sigilion's «Мощь берсерка»: arm the per-attack self-punish window, consumed
			// in scheduleHitAfterLocked against the swing's LIVE dmg_pct bonus.
			hs.recoilFrac = op.Value.At(ctx.level)
			hs.recoilUntil = now + op.Dur.At(ctx.level)

		case gamedata.OpAttackManaBonus:
			// Miriam's «Зачарованные стрелы»: arm the mana-fueled attack window, consumed
			// in scheduleHitAfterLocked.
			hs.manaShotDmg = op.Value.At(ctx.level)
			if op.PerSP > 0 {
				hs.manaShotDmg += hs.spellPowerLocked(now) * op.PerSP
			}
			hs.manaShotCost = op.Value2.At(ctx.level)
			hs.manaShotUntil = now + op.Dur.At(ctx.level)

		case gamedata.OpAttackCleave:
			// BlackDragon's «Неистовство»: arm the cleave window, consumed in
			// scheduleHitAfterLocked.
			hs.cleaveRadius = op.Radius
			hs.cleaveUntil = now + op.Dur.At(ctx.level)

		case gamedata.OpHitStack:
			// Gayal's «Меч жажды»: arm the per-hit stacking window, consumed in
			// scheduleHitAfterLocked. Re-casting refreshes the window and the per-stack
			// magnitudes, but leaves any stacks already banked alone.
			hs.hitStackUntil = now + op.Dur.At(ctx.level)
			hs.hitStackCap = int(op.Count.At(ctx.level))
			hs.hitStackLSPer = op.Value.At(ctx.level)
			hs.hitStackSpdPer = op.Value2.At(ctx.level)
			hs.hitStackBurstDmg = op.StackBurstDamage.At(ctx.level)
			hs.hitStackBurstSP = op.PerSP

		case gamedata.OpCastMark:
			// Einzenhaim's «Изгнание колдовства»: mark every enemy in the spray so
			// tryBossSkillLocked can punish one that casts within the window.
			for _, m := range s.opTargetsLocked(c, ctx, op) {
				dmg := op.Value.At(ctx.level)
				if op.PerSP > 0 {
					dmg += hs.spellPowerLocked(now) * op.PerSP
				}
				m.st.castMarkUntil = now + op.Dur.At(ctx.level)
				m.st.castMarkDmg = dmg
				m.st.castMarkOwner = c.objID
			}

		case gamedata.OpConsumeSouls:
			// Gellar's «Армия душ»: «теряет половину из накопленных душ» on cast.
			hs.soulStacks /= 2

		case gamedata.OpDeathLink:
			// Rognar's «Канал смерти»: link the target so a share of incoming blows forwards
			// to it (or heals it, if a friend).
			if a := ctx.allyTarget; a != nil {
				hs.deathLinkObj, hs.deathLinkAlly = a.objID, true
			} else if m := ctx.target; m != nil && !m.dead {
				hs.deathLinkObj, hs.deathLinkAlly = m.id, false
			} else {
				break
			}
			hs.deathLinkUntil = now + op.Dur.At(ctx.level)
			hs.deathLinkFrac = op.Value2.At(ctx.level)

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

		case gamedata.OpDrainMaxHP:
			for _, m := range s.opTargetsLocked(c, ctx, op) {
				amt := op.Value.At(ctx.level) * hs.powerMul()
				if op.PerSP > 0 {
					amt += hs.spellPowerLocked(now) * op.PerSP
				}
				m.maxHP = math.Max(1, m.maxHealth()-amt)
				if m.hp > m.maxHP {
					m.hp = m.maxHP
				}
			}

		case gamedata.OpDot:
			for _, m := range s.opTargetsLocked(c, ctx, op) {
				perSec := op.Value.At(ctx.level)
				// VictimMaxHPPct (Elgorm's «Оскверненная почва»): the DoT punishes tanky/
				// high-HP targets harder -- a fraction of the TARGET's own max HP, not a
				// flat per-rank number.
				if pct := op.VictimMaxHPPct.At(ctx.level); pct > 0 {
					perSec = pct * m.maxHealth()
				}
				if op.PerSP > 0 {
					perSec += hs.spellPowerLocked(now) * op.PerSP
				}
				perSecEnd := perSec
				if len(op.DecayTo) > 0 {
					perSecEnd = op.DecayTo.At(ctx.level)
				}
				m.st.dots = append(m.st.dots, overTime{
					perSec: perSec, perSecEnd: perSecEnd, startAt: now,
					until:    now + op.Dur.At(ctx.level),
					nextTick: now + dotTickInterval, srcObj: c.objID,
				})
				// Persistent acid/poison visual on the victim (one shared copy, shown
				// to the whole party). An empty DotFx is a no-op inside the helper.
				s.ensureMobStatusFxLocked(c, m, &m.st.dotFx, op.DotFx)
				// Wilfang's «Ядовитый укус»: a death while still poisoned detonates an AoE
				// burst, read in the mob-death branch before the status wipe.
				if op.ExplodeOnDeath {
					dmg := op.ExplodeDamage.At(ctx.level)
					if op.ExplodeSP > 0 {
						dmg += hs.spellPowerLocked(now) * op.ExplodeSP
					}
					m.st.poisonExplodeUntil = now + op.Dur.At(ctx.level)
					m.st.poisonExplodeOwner = c.objID
					m.st.poisonExplodeDmg = dmg
					m.st.poisonExplodeRadius = op.ExplodeRadius
					m.st.poisonExplodeFx = op.ExplodeFx
				}
			}

		case gamedata.OpChainHeal:
			// Kiona's «Лечебная волна»: hops between up to Count living allies (self first),
			// healing each; enemies near WHICHEVER ally is currently being healed take
			// Value2 magic damage. Solo (no teammates) simply heals+splashes around Kiona
			// herself, matching the self-only fallback used elsewhere in this file.
			steps := int(op.Count.At(ctx.level))
			if steps < 1 {
				steps = 1
			}
			healAmt := op.Value.At(ctx.level) * hs.powerMul()
			if op.PerSP > 0 {
				healAmt += hs.spellPowerLocked(now) * op.PerSP
			}
			dmgAmt := op.Value2.At(ctx.level) * hs.powerMul()
			radius := op.Radius
			if radius <= 0 {
				radius = 4
			}
			hit := 0
			for _, mem := range c.members() {
				if hit >= steps {
					break
				}
				mh := mem.huntState
				if mh == nil || mh.deadUntil > 0 {
					continue
				}
				s.healPlayerLocked(mem, healAmt)
				if dmgAmt > 0 {
					mx, my := mem.posAtLocked(float32(now))
					for _, m := range c.mobsWithinLocked(mx, my, radius) {
						s.hitMobLocked(c, m, dmgAmt, c.objID)
					}
				}
				hit++
			}

		case gamedata.OpDamageShare:
			// Kiona's «Лесной покров»: mark whichever unit this cast resolved (enemy mob OR
			// ally player) to share Value fraction of any damage it takes as healing to
			// nearby allies for the duration.
			coeff := op.Value.At(ctx.level)
			dur := op.Dur.At(ctx.level)
			radius := op.Radius
			if radius <= 0 {
				radius = 5
			}
			switch {
			case ctx.target != nil:
				ctx.target.st.cloakUntil = now + dur
				ctx.target.st.cloakOwner = c.objID
				ctx.target.st.cloakHealCoeff = coeff
				ctx.target.st.cloakRadius = radius
			case ctx.allyTarget != nil:
				ahs := ctx.allyTarget.huntState
				ahs.st.cloakUntil = now + dur
				ahs.st.cloakOwner = c.objID
				ahs.st.cloakHealCoeff = coeff
				ahs.st.cloakRadius = radius
			default:
				hs.st.cloakUntil = now + dur
				hs.st.cloakOwner = c.objID
				hs.st.cloakHealCoeff = coeff
				hs.st.cloakRadius = radius
			}

		case gamedata.OpRevealTarget:
			// Velial's «Трибунал»: the marked enemy stays fully revealed to the whole team
			// (bypasses mobViewDistLocked's distance-based fog) for the duration.
			if m := ctx.target; m != nil {
				m.st.revealUntil = now + op.Dur.At(ctx.level)
			}

		case gamedata.OpHeal:
			amt := op.Value.At(ctx.level) * hs.powerMul()
			if op.PerSP > 0 {
				amt += hs.spellPowerLocked(now) * op.PerSP
			}
			if pct := op.SelfMaxHPPct.At(ctx.level); pct > 0 {
				amt = pct * hs.maxHPLocked(now)
			}
			// Value2 scales the heal by the size of the hit that triggered this op --
			// Nerlag's «Прилив крови» (on-damaged proc) heals for the damage just taken.
			if v2 := op.Value2.At(ctx.level); v2 > 0 && ctx.dmgIn > 0 {
				amt += ctx.dmgIn * v2
			}
			// On:"allies"/"ally" spreads the heal to friendly avatars (self included) --
			// Arianna's «Исцеление», Kiona/Edilia's «heal allies», Tangren's totem tick.
			if op.On == "allies" || op.On == "ally" {
				for _, mem := range s.allyTargetsLocked(c, ctx, op) {
					s.healPlayerLocked(mem, amt)
				}
			} else {
				s.healPlayerLocked(c, amt)
			}

		case gamedata.OpHot:
			perSec := op.Value.At(ctx.level) * hs.powerMul()
			if op.PerSP > 0 {
				perSec += hs.spellPowerLocked(now) * op.PerSP
			}
			if op.On == "allies" || op.On == "ally" {
				for _, mem := range s.allyTargetsLocked(c, ctx, op) {
					s.addAllyHotLocked(mem, perSec, now+op.Dur.At(ctx.level), now)
				}
			} else {
				hs.st.hots = append(hs.st.hots, overTime{
					perSec: perSec, until: now + op.Dur.At(ctx.level), nextTick: now + 1,
				})
			}

		case gamedata.OpManaRestore:
			amt := op.Value.At(ctx.level) * hs.powerMul()
			if op.PerSP > 0 {
				amt += hs.spellPowerLocked(now) * op.PerSP
			}
			hs.mana = math.Min(hs.maxManaLocked(now), hs.mana+amt)
			s.syncSelfLocked(c, syncMana)

		case gamedata.OpStun:
			for _, m := range s.opTargetsLocked(c, ctx, op) {
				// A disposable enemy-hero shadow (see pvp_hero_targets.go): write the CC
				// straight into the real huntState.st via the hero-side twin, not the
				// shadow's own copy, which is discarded unread the instant this op returns.
				if m.heroOwner != nil {
					s.dotaStunHeroLocked(c, m.heroOwner, now, op.Dur.At(ctx.level)+ctx.durBonus)
					continue
				}
				s.stunMobLocked(c, m, now, op.Dur.At(ctx.level)+ctx.durBonus)
			}

		case gamedata.OpRoot:
			for _, m := range s.opTargetsLocked(c, ctx, op) {
				if m.heroOwner != nil {
					s.dotaRootHeroLocked(c, m.heroOwner, now, op.Dur.At(ctx.level))
					continue
				}
				m.st.rootUntil = math.Max(m.st.rootUntil, now+op.Dur.At(ctx.level))
				s.ensureMobStatusFxLocked(c, m, &m.st.rootFx, "StunEffect")
				s.stopMobLocked(c, m, now)
			}

		case gamedata.OpSlow:
			for _, m := range s.opTargetsLocked(c, ctx, op) {
				if m.heroOwner != nil {
					decayTo := 0.0
					if len(op.DecayTo) > 0 {
						decayTo = op.DecayTo.At(ctx.level)
					}
					s.dotaSlowHeroLocked(c, m.heroOwner, now, op.Value.At(ctx.level), op.Dur.At(ctx.level), decayTo)
					continue
				}
				m.st.slowUntil = now + op.Dur.At(ctx.level)
				m.st.slowFactor = op.Value.At(ctx.level)
				m.st.slowFactorEnd = m.st.slowFactor
				if len(op.DecayTo) > 0 {
					m.st.slowFactorEnd = op.DecayTo.At(ctx.level)
				}
				m.st.slowStart = now
				s.ensureMobStatusFxLocked(c, m, &m.st.slowFx, "SlowMoveEffect")
				s.syncMobSpeedLocked(c, m, now)
			}

		case gamedata.OpAttackSlow:
			for _, m := range s.opTargetsLocked(c, ctx, op) {
				if m.heroOwner != nil {
					s.dotaAttackSlowHeroLocked(c, m.heroOwner, now, op.Dur.At(ctx.level), op.Value.At(ctx.level))
					continue
				}
				m.st.atkSlowUntil = now + op.Dur.At(ctx.level)
				m.st.atkSlowFactor = op.Value.At(ctx.level)
				s.ensureMobStatusFxLocked(c, m, &m.st.atkSlowFx, "SlowAttackEffect")
			}

		case gamedata.OpSilence:
			// Mobs have no skills: silencing one also stops its attacks. A hero keeps
			// auto-attacking while silenced -- see dotaSilenceHeroLocked.
			for _, m := range s.opTargetsLocked(c, ctx, op) {
				if m.heroOwner != nil {
					s.dotaSilenceHeroLocked(c, m.heroOwner, now, op.Dur.At(ctx.level))
					continue
				}
				m.st.silenceUntil = now + op.Dur.At(ctx.level)
				m.st.atkSlowUntil = math.Max(m.st.atkSlowUntil, now+op.Dur.At(ctx.level))
				m.st.atkSlowFactor = 0.1
				s.ensureMobStatusFxLocked(c, m, &m.st.silenceFx, "SilenceEffect")
			}

		case gamedata.OpBuffStat:
			// ScalePerHit (Nerlag's «Поголовная бойня»): rescale the authored Value by how
			// many enemies THIS SAME cast's OpDamage just hit (ctx.hitCount, set earlier in
			// this ops list) before any of the branches below read op.Value. 0 hits = no
			// buff at all.
			if op.ScalePerHit {
				n := float64(ctx.hitCount)
				if n <= 0 {
					break
				}
				v := op.Value.At(ctx.level)
				if strings.HasSuffix(op.Stat, "_pct") {
					v = 1 + (v-1)*n
				} else {
					v = v * n
				}
				op.Value = gamedata.PerLevel{v}
			}
			// PerSP (Veritas's «Благословение жизни»/«Метаморфоза»): add the caster's own
			// spell power to the authored Value, same as OpDamage/OpHeal already do. Collapses
			// to a single-level PerLevel so every branch below reads the scaled total
			// regardless of ctx.level, mirroring the ScalePerHit rescale above.
			if op.PerSP > 0 {
				op.Value = gamedata.PerLevel{op.Value.At(ctx.level) + hs.spellPowerLocked(now)*op.PerSP}
			}
			// On:"target" with NO unit target is a self-cast (the client always ships a
			// targetPos, so hasPos alone does not mean a unit was picked): buff the
			// caster. It must not fall through to opTargetsLocked -- damageTargetsLocked's
			// "no target, no radius" arm substitutes a 4-unit circle, which is a heuristic
			// for DAMAGE ops and hands a friendly buff to whatever stands nearby. That
			// scan is hostile-only, so a self-cast «Щит хранителя» was handing +30
			// magic_armor to the enemies around it and nothing to the caster.
			if op.On == "own_summons" {
				// Anhel's «Гнев океана»: "себе, и всем своим клонам" -- the caster's own
				// self-buff is a SEPARATE op (On:"self"/"target"); this one only reaches her
				// live summoned units. Grimlok's «Дикость» additionally buffs move speed
				// (Stat=="move_speed_pct"), a separate live multiplier from attack speed.
				until := now + op.Dur.At(ctx.level)
				mul := op.Value.At(ctx.level)
				for _, sm := range hs.summons {
					if sm.dead {
						continue
					}
					if op.Stat == "move_speed_pct" {
						sm.moveSpeedMul, sm.moveSpeedMulUntil = mul, until
					} else {
						sm.atkSpeedMul, sm.atkSpeedMulUntil = mul, until
					}
				}
			} else if op.On == "allies" || op.On == "ally" {
				// Buff friendly avatars (self + nearby / aimed allies): Arianna's «Аура
				// стойкости», Sandariel's «Прыжок» speed, Hekata's «Пепельный смерч» ally
				// attack. Self keeps the caster's own buff icon; allies get the stat only.
				for _, mem := range s.allyTargetsLocked(c, ctx, op) {
					if mem == c {
						s.addPlayerModLocked(c, ctx, op, now)
					} else {
						s.addAllyModLocked(mem, op, ctx, now)
					}
				}
			} else if op.On == "enemies" {
				// AoE hostile stat debuff regardless of an aimed unit (Hekata's «Пепельный
				// смерч» weakens every enemy's attack around her); uses op.Radius.
				for _, m := range s.opTargetsLocked(c, ctx, op) {
					m.st.mods = append(m.st.mods, statMod{
						stat: op.Stat, value: op.Value.At(ctx.level),
						until: now + op.Dur.At(ctx.level), src: castSrc(ctx),
					})
					s.syncMobSpeedLocked(c, m, now)
				}
			} else if op.On == "target" && ctx.target == nil {
				s.addPlayerModLocked(c, ctx, op, now)
			} else if op.On == "target" {
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
			amount := op.Value.At(ctx.level) * hs.powerMul()
			until := now + op.Dur.At(ctx.level)
			// On:"ally"/"allies" puts the absorb shield on the aimed friend / nearby allies
			// (Arianna's «Щит хранителя» / «Касание спасителя») instead of the caster.
			if op.On == "allies" || op.On == "ally" {
				for _, mem := range s.allyTargetsLocked(c, ctx, op) {
					s.applyShieldLocked(mem, amount, until)
					if op.GrantsCCImmune {
						mem.huntState.tempCCImmuneUntil = until
					}
				}
			} else {
				s.applyShieldLocked(c, amount, until)
				if op.GrantsCCImmune {
					hs.tempCCImmuneUntil = until
				}
			}

		case gamedata.OpBlink:
			s.blinkLocked(c, ctx)

		case gamedata.OpDash:
			if op.PushAside > 0 {
				// «Отпихивая всех врагов в стороны»: shove enemies along the charge's path,
				// separate from (and before) the arrival-point slow+damage AoE below.
				tx, ty := ctx.px, ctx.py
				if ctx.target != nil {
					tx, ty = ctx.target.x, ctx.target.y
				}
				if ctx.hasPos || ctx.target != nil {
					cx, cy := c.posAtLocked(float32(now))
					for _, m := range c.mobsAlongLineLocked(cx, cy, tx, ty, 2.5, math.Hypot(float64(tx-cx), float64(ty-cy))) {
						s.knockbackMobLocked(c, m, op.PushAside, now)
					}
				}
			}
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
			// Apply:"self" makes it RECOIL -- the caster is thrown, not the enemy
			// (Einzenhaim's «Выстрел с отдачей»). Same field Apply already uses to point a
			// damage op at the caster.
			if op.Apply == "self" {
				s.recoilSelfLocked(c, ctx, op.Value.At(ctx.level), now)
				break
			}
			for _, m := range s.opTargetsLocked(c, ctx, op) {
				s.knockbackMobLocked(c, m, op.Value.At(ctx.level), now)
			}

		case gamedata.OpStealth:
			if op.On == "allies" {
				// Sandariel's «Сокрывающая вуаль»: cloaks nearby ALLIES, not herself -- she
				// "остаётся видимой" (stays visible; her own dodge_pct buff is a separate op).
				allyOp := op
				if allyOp.Radius <= 0 {
					allyOp.Radius = 5
				}
				for _, mem := range s.allyTargetsLocked(c, ctx, allyOp) {
					if mem != c {
						s.applySkillStealthLocked(mem, op.Dur.At(ctx.level), op.BreakOnMove, now)
					}
				}
				break
			}
			// Cloak the caster (Lirvein/Sandariel/Astarot/Wilfang stealth skills). The
			// cast's own breakInvisibilityLocked already fired at the top of doSkillLocked
			// (before ops run), so this grant survives until the NEXT attack/cast reveals it.
			// op.BreakOnMove additionally reveals it on any move order (Wilfang's «Засада»).
			s.applySkillStealthLocked(c, op.Dur.At(ctx.level), op.BreakOnMove, now)
			// A stealth op carrying nested Ops arms a reveal burst, detonated the instant
			// invisibility breaks -- «при этом окружающие враги получают урон» (Wilfang's
			// «Засада»: damage lands when the ambush ends, not at cast).
			if len(op.Ops) > 0 {
				hs.stealthBurst = op.Ops
				hs.stealthBurstSlot = ctx.slot
				hs.stealthBurstLevel = ctx.level
			}

		case gamedata.OpTreeForm:
			// Urg's «Древесный камуфляж»: turn the ally (self in solo) into a tree — the
			// reveal burst detonates when they leave the form (urg.go).
			s.applyTreeFormLocked(c, op, ctx, now)

		case gamedata.OpGrove:
			// Urg's «Непроглядные дебри»: grow the tree ring; the fall-damage is deferred
			// to when the trees vanish (urg.go). The while-standing silence is a sibling op.
			s.applyGroveLocked(c, op, ctx, now)

		case gamedata.OpOnKill:
			// Run the nested ops immediately if this cast's primary target died from it
			// (Lirvein's «Изощренный бросок» reset+empower on a kill). Otherwise, if the
			// op carries a Dur, mark the (surviving) target: a later kill by ANY source
			// before the mark expires still credits this caster (see the mob-death branch).
			if ctx.target != nil && ctx.target.dead {
				s.applyOpsLocked(c, op.Ops, ctx, now)
			} else if ctx.target != nil && !ctx.target.dead {
				if dur := op.Dur.At(ctx.level); dur > 0 {
					ctx.target.st.killMarkUntil = now + dur
					ctx.target.st.killMarkOwner = c.objID
					ctx.target.st.killMarkOps = op.Ops
					ctx.target.st.killMarkLevel = ctx.level
					ctx.target.st.killMarkSlot = ctx.slot
				}
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

		case gamedata.OpVisionWard:
			s.spawnVisionWardLocked(c, op, ctx, now)

		case gamedata.OpBounce:
			s.startBounceLocked(c, op, ctx, now)

		case gamedata.OpChannel:
			// Op.Growth on a nested OpDamage (Miriam's «Убийственный залп») ramps the
			// channel's damage additively per pulse; on a nested OpStun (Titanid's
			// «Землетрясение») it ramps the stun duration instead; Op.RadiusGrowth widens
			// either's AoE per pulse -- read them once here into the channelState so
			// tickChannelsLocked can apply pulseCount×growth each tick.
			var growth, growthSP, stunGrowth, radiusGrowth, growthPerEnemy, growthRadius float64
			for _, nested := range op.Ops {
				if nested.Kind == gamedata.OpDamage && len(nested.Growth) > 0 {
					growth = nested.Growth.At(ctx.level)
					growthSP = nested.GrowthPerSP
				}
				if nested.Kind == gamedata.OpStun && len(nested.Growth) > 0 {
					stunGrowth = nested.Growth.At(ctx.level)
				}
				if nested.RadiusGrowth > 0 {
					radiusGrowth = nested.RadiusGrowth
				}
				if nested.Kind == gamedata.OpDamage && len(nested.GrowthPerEnemy) > 0 {
					growthPerEnemy = nested.GrowthPerEnemy.At(ctx.level)
					growthRadius = nested.Radius
				}
			}
			// Op.Count on a channel = an exact pulse count (Einzenhaim's «{shots} выстрелов»).
			// Its `until` still bounds the channel, sized generously from the count so the
			// counter -- not tick rounding -- is what ends it.
			until := now + op.Dur.At(ctx.level)
			pulses := int(op.Count.At(ctx.level))
			interval := op.Interval
			if len(op.Intervals) > 0 {
				interval = op.Intervals.At(ctx.level)
			}
			// firstPulse/fired: a counted channel delivers pulse #1 right here, so both its
			// schedule and its growth index start one ahead.
			firstPulse, fired := now+channelPulseDelay(hs.av.Prefab, ctx.slot), 0
			if pulses > 0 {
				if end := now + interval*float64(pulses) + 0.2; end > until {
					until = end
				}
				// A COUNTED channel is "N shots starting NOW", not "a stream over N seconds":
				// fire shot 1 here rather than on the next tick. The tick runs on a 0.2s grid,
				// so waiting for it would push every shot up to a fifth of a second late --
				// enough to visibly lag the client's own muzzle flashes, which are what the
				// player is watching. Duration-bounded channels keep their old lead-in.
				s.applyOpsLocked(c, op.Ops, ctx, now)
				pulses--
				fired = 1
				if pulses == 0 {
					continue // a one-shot "volley" is just the shot; nothing to sustain
				}
				firstPulse = now + interval
			}
			// A SELF-only channel whose ONLY mechanic is the channel itself (no OpBuffStat
			// anywhere to ride) never gets its BuffFx started: the sole two places that ever
			// start a "self" BuffFx are addPlayerModLocked (fires when a stat MOD is applied)
			// and the toggle path -- neither of which this shape ever reaches. BlackDragon's
			// «Взмах погибели», Gayal's «Поглощение жизни» and Gellar's «Армия душ» are all
			// exactly this shape (OpChannel wrapping only OpDamage/OpSlow/OpHeal), and all
			// three silently show nothing but the opening cast flourish -- «не создаёт никаких
			// эффектов». Hekata's «Пепельный смерч» is NOT this shape (its channel's own
			// nested OpBuffStat legitimately re-triggers the fx every tick) and must be left
			// alone, hence the skillHasAnyBuffStat guard.
			fxUIDs := hs.liveCastFxLocked(ctx.slot)
			if chDef := hs.skillDef(ctx.slot); chDef.BuffFx != "" && chDef.BuffFxOn == "self" &&
				ctx.target == nil && !ctx.toggle && !skillHasAnyBuffStat(chDef) {
				// hs.soulStacks is Gellar-only state (0 for every other avatar, since nothing
				// else ever sets it), so passing it here unconditionally is harmless for
				// BlackDragon/Gayal: fxStartCounterLocked omits the counter arg entirely at
				// <=1. For Gellar's «Армия душ» it is how many "soul" particles
				// GellarSkill4Effect visibly bursts out -- «визуально количество душ,
				// вылетающих равное накопленным стакам» -- via the client's own
				// ParticlesMgr.SetCounter (confirmed off the decompiled VisualEffectHolder/
				// ParticlesMgr: it multiplies the effect's particle emitter's min/maxEmission).
				buffUID := s.fxStartCounterLocked(c, chDef.BuffFx, c.objID, 0, false, 0, 0, int32(hs.soulStacks))
				hs.scheduleFxEnd(buffUID, until)
				fxUIDs = append(fxUIDs, buffUID)
			}
			hs.channels = append(hs.channels, channelState{
				slot: ctx.slot, level: ctx.level,
				until: until, interval: interval, pulsesLeft: pulses, pulseCount: fired,
				// The caster's live skill visuals, so an early break can end them.
				fxUIDs: fxUIDs,
				// A channel cast alongside a root/silence on the same victim IS the grip
				// holding them (Morlokai's «Кабала»); breaking it must let them go.
				holdsTarget: skillHoldsTarget(hs.skillDef(ctx.slot)),
				// Delay the FIRST pulse by the client fx's own lead-in so the damage
				// ticks land in step with the visual (Elgorm's arrow rain waits 0.2s
				// before its first arrow); every other channel starts immediately.
				nextPulse: firstPulse,
				target:    mobID(ctx.target),
				px:        ctx.px, py: ctx.py, hasPos: ctx.hasPos, ops: op.Ops,
				interruptible:  channelInterruptible(hs.av.Prefab, ctx.slot),
				breakDist:      op.TriggerRadius,        // >0 = leash (Inshari siphon)
				stunOnBreak:    op.Value2.At(ctx.level), // stun seconds when the leash snaps
				growth:         growth,
				growthSP:       growthSP,
				stunGrowth:     stunGrowth,
				radiusGrowth:   radiusGrowth,
				growthPerEnemy: growthPerEnemy,
				growthRadius:   growthRadius,
			})

		case gamedata.OpProc:
			// procs are registered from passives at world-state time; a proc op
			// inside an active cast applies its nested ops immediately instead.
			s.applyOpsLocked(c, op.Ops, ctx, now)

		case gamedata.OpAura:
			// aura pulses run from the tick while the toggle is on; nothing here.

		case gamedata.OpRevive, gamedata.OpImmune, gamedata.OpHealOnKill, gamedata.OpManaOnKill,
			gamedata.OpConsecutiveHit, gamedata.OpAttackSpeedStreak, gamedata.OpShieldExplode,
			gamedata.OpCleanseOnHit, gamedata.OpMoveChargeAttack, gamedata.OpZoneArmor, gamedata.OpMeleeForm:
			// Passive/toggle-only mechanics: registered at world-build (or on toggle-on) and
			// honored in the death / player-CC / on-kill / basic-attack / incoming-hit gates.
			// Nothing to run inside an ops batch.

		case gamedata.OpOnKillStack:
			// Hekata's «Культ жнеца» cast: open (or refresh) the kill-window so kills during
			// it grow her base attack. The persistent soul flavour (Dur 0, Gellar) is passive
			// and never reaches here. Stacks reset on each activation.
			if hs := c.huntState; hs != nil {
				if dur := op.Dur.At(ctx.level); dur > 0 {
					hs.killWindowUntil = now + dur
					hs.killWindowPerKill = op.Value.At(ctx.level)
					if op.PerSP > 0 {
						hs.killWindowPerKill += hs.spellPowerLocked(now) * op.PerSP
					}
					hs.killWindowCap = int(op.Value2.At(ctx.level))
					hs.killWindowStacks = 0
				}
			}

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

func (hs *huntState) maxManaLocked(now float64) float64 {
	return hs.av.Mana*hs.hpMul() + hs.st.modSum(now, "max_mana")
}

// effAttackRangeLocked is the avatar's live auto-attack reach: its base AttackRange
// plus any attack_range buff (Teridin's «Прицеливание» passive) -- or a flat melee
// reach while Grimlok's «Темная сторона» (Op.MeleeForm) toggle is active.
func (hs *huntState) effAttackRangeLocked(now float64) float64 {
	if hs.meleeFormSlot > 0 {
		return meleeReach
	}
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

	// StackCap (Titanid's «Каменная кожа»): clamp this stack to the remaining headroom
	// under the cap, shared across every live mod from the SAME skill+stat; drop it
	// entirely once the cap is already full.
	if cap := op.StackCap.At(ctx.level); cap > 0 {
		var sum float64
		for _, m := range hs.st.mods {
			if m.stat == mod.stat && m.src == mod.src && (m.until == 0 || now < m.until) {
				sum += m.value
			}
		}
		if room := cap - sum; room < mod.value {
			mod.value = room
		}
		if mod.value <= 0 {
			return
		}
	}

	// A multi-stat self/ally buff (Urg's «Дубовая кора» stacks block + armor + regen in one
	// cast) must show ONE icon and ONE glow, not one per stat op. If a live mod from THIS same
	// cast already carries the icon/fx, this op only contributes its stat -- no duplicate.
	srcHasIcon, srcHasFx := false, false
	for _, m := range hs.st.mods {
		if m.src != mod.src {
			continue
		}
		if m.buffEffID != 0 {
			srcHasIcon = true
		}
		if m.fxUID != 0 {
			srcHasFx = true
		}
	}

	// Buff-bar icon (only for timed, non-toggle, icon-enabled skills).
	if def.BuffIcon && dur > 0 && !srcHasIcon {
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
	if def.BuffFx != "" && def.BuffFxOn != "target" && !ctx.toggle && !srcHasFx {
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
	// Flat basic-attack bonuses from avatar tree items (DamageMin/AttackSpeed), plus any
	// banked on-kill attack (Gellar's souls / Hekata's kill-window) -- it grows the base
	// attack floor exactly like gear does in the REAL hit formula (scheduleHitAfterLocked),
	// but was missing here, so the character card showed plain base damage while actual
	// swings already carried the soul bonus.
	dmgFlat := st.modSum(now, "dmg_flat") + s.killAttackBonusLocked(hs, now)
	// Velial's «Воля к победе» adds a flat, post-multiplier bonus scaling with his own missing
	// HP. Fold it into the DISPLAYED damage so the avatar card shows it added to the attack
	// (it tracks HP live via refreshPassiveBuffCountersLocked, which re-pushes these stats).
	missBonus := hs.casterMissingHPBonusLocked(now)
	atkSpeed := a.AttackSpeed + st.modSum(now, "atk_speed_flat")
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
		setFloats(syncDmgMin, idx, float32((float64(a.DmgMin)+dmgFlat)*dmgMul+missBonus)).
		setFloats(syncDmgMax, idx, float32((float64(a.DmgMax)+dmgFlat)*dmgMul+missBonus)).
		setFloats(syncAttackSpeed, idx, float32(atkSpeed*st.attackFactor(now))).
		setFloats(syncMaxHealth, idx, float32(maxHP)).
		setFloats(syncHealth, idx, float32(hs.hp/maxHP)).
		setFloats(syncMaxMana, idx, float32(maxMana)).
		setFloats(syncMana, idx, float32(hs.mana/maxMana)).
		setFloats(syncPhysArmor, idx, float32((a.PhysArmor+st.modSum(now, "phys_armor"))*armMul)).
		setFloats(syncMagicArmor, idx, float32((a.MagicArmor+st.modSum(now, "magic_armor"))*armMul)).
		setFloats(syncSpellPower, idx, float32(hs.spellPowerLocked(now))).
		setFloats(syncAttackRange, idx, float32(hs.effAttackRangeLocked(now))).
		setFloats(syncSpeed, idx, float32(c.curSpeedLocked(now))).
		setFloats(syncViewRadius, idx, effectiveViewRadius(avatarViewRadius*float32(st.modMul(now, "view_radius_pct"))))
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

// recoilSelfLocked shoves the CASTER straight backwards, away from what they just shot
// at -- Einzenhaim's «Выстрел с отдачей»: «В момент выстрела аватар отлетает назад из-за
// отдачи». Nothing in the client does this on its own: the baked holder for
// EinzenhaimSkill3 carries only the Cast03 clip and a SELF_TO_TARGET beam, and the legacy
// Animation component never moves the transform (the server owns position). So the recoil
// has to be a real server-side displacement or it does not exist at all.
//
// Implemented as a reversed dash rather than a teleport: the same lunge machinery every
// dash skill uses, so the client dead-reckons a fast glide backwards instead of the
// avatar blinking. Clipped to walkable ground, so recoiling into a wall simply stops
// short. The locale gives no distance ("отлетает назад", no number), so the caller
// passes one.
func (s *Server) recoilSelfLocked(c *conn, ctx opCtx, dist float64, now float64) {
	if dist <= 0 {
		return
	}
	hs := c.huntState
	cx, cy := c.posAtLocked(float32(now))
	// Direction: from the target (or aim point) back through the caster.
	var ax, ay float32
	switch {
	case ctx.target != nil:
		ax, ay = ctx.target.x, ctx.target.y
	case ctx.hasPos:
		ax, ay = ctx.px, ctx.py
	default:
		return
	}
	dx, dy := float64(cx-ax), float64(cy-ay)
	d := math.Hypot(dx, dy)
	if d < 1e-6 {
		return // standing exactly on the target: no meaningful "backwards"
	}
	tx := cx + float32(dx/d*dist)
	ty := cy + float32(dy/d*dist)
	if c.nav != nil {
		nx, ny := c.nav.Clip(float64(cx), float64(cy), float64(tx), float64(ty))
		tx, ty = float32(nx), float32(ny)
	}
	travel := math.Hypot(float64(tx-cx), float64(ty-cy))
	if travel < 1e-3 {
		return // backed against a wall: nowhere to be thrown
	}
	hs.dashSpeed = knockbackSpeed
	hs.dashUntil = now + travel/knockbackSpeed + 0.05
	c.moveStraightExLocked(s, tx, ty, true)
}

// knockbackSpeed is how fast a shoved mob slides to its landing spot (world units/sec):
// fast, so the push is snappy, but a real velocity the client can dead-reckon rather than
// a teleport. knockbackMinTime floors the glide so a tiny shove still animates a beat.
const (
	knockbackSpeed   = 18.0
	knockbackMinTime = 0.12
)

// knockbackMobLocked shoves a mob directly away from the caster by dist units (the
// inverse of pullMobLocked), clipped to walkable ground -- Dutnik's «Взрыв» blast and
// Miriam's «Выстрел бури».
//
// The mob's authoritative position moves to the landing spot at once (it is stunned /
// being shoved, so nothing walks it further, and hit resolution stays simple). What
// changed is the WIRE: instead of the old zero-velocity teleport snapshot -- which the
// client could only render as an instant jump ("визуально не отталкивает + рассинхрон")
// -- we broadcast the shove FROM the old spot carrying a real velocity, so every client
// glides the model to the landing point. The mob tick loop sends the matching stop at
// kbUntil and keeps the AI/stun gates off the mob until then, so the server never fights
// the glide.
func (s *Server) knockbackMobLocked(c *conn, m *mobState, dist float64, now float64) {
	if m.structure || m.mob.Stationary {
		return // immovable: an altar/cannon is not shoved out of its emplacement
	}
	if dist <= 0 {
		dist = 3
	}
	cx, cy := c.posAtLocked(float32(now))
	dx, dy := float64(m.x-cx), float64(m.y-cy)
	d := math.Hypot(dx, dy)
	if d < 1e-6 {
		dx, dy, d = 1, 0, 1 // degenerate overlap: push along +x
	}
	ux, uy := dx/d, dy/d
	fromX, fromY := m.x, m.y
	tx := m.x + float32(ux*dist)
	ty := m.y + float32(uy*dist)
	if c.nav != nil {
		nx, ny := c.nav.Clip(float64(m.x), float64(m.y), float64(tx), float64(ty))
		tx, ty = float32(nx), float32(ny)
	}
	adist := math.Hypot(float64(tx-fromX), float64(ty-fromY))
	if adist < 1e-3 {
		return // nowhere to go (already against a wall on the away side): no move, no glide
	}
	dur := adist / knockbackSpeed
	if dur < knockbackMinTime {
		dur = knockbackMinTime
	}
	// Authoritative rest position (server velocity stays 0 so the dead-reckon can't carry
	// it past the spot); the client is sent the glide velocity separately.
	m.x, m.y = tx, ty
	m.vx, m.vy = 0, 0
	m.kbUntil = now + dur
	s.broadcastPosLocked(c, m.id, fromX, fromY,
		float32(ux*knockbackSpeed), float32(uy*knockbackSpeed), float32(now))
}

// advanceKnockbackLocked reports whether a shoved mob is still mid-glide (caller then
// keeps every AI/stun gate off it). When the window elapses it clears the flag and
// broadcasts the stop at the landing spot the server has held all along, so the client
// settles exactly where the server is.
func (s *Server) advanceKnockbackLocked(c *conn, m *mobState, now float64) bool {
	if now < m.kbUntil {
		return true
	}
	m.kbUntil = 0
	m.vx, m.vy = 0, 0
	s.broadcastPosLocked(c, m.id, m.x, m.y, 0, 0, float32(now))
	return false
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
	cx, cy := c.posAtLocked(float32(now))
	d := math.Hypot(float64(m.x-cx), float64(m.y-cy))
	if d < 1.5 {
		return
	}
	nx := cx + float32(float64(m.x-cx)*1.5/d)
	ny := cy + float32(float64(m.y-cy)*1.5/d)
	s.teleportMobLocked(c, m, nx, ny, now)
}

// teleportMobLocked drops a mob at (x,y) and tells EVERY viewer, not just the caster:
// a displaced mob is halted there, so its clients must be snapped to the new spot
// rather than left dead-reckoning from the old one. Shared by the pull and knockback
// paths, which both used to push the new position to the caster alone and then lean on
// stopMobLocked to inform the rest -- but that call bails out early on an already-
// stationary mob, so yanking or shoving a standing mob simply did not move it on a
// teammate's screen.
//
// Immovable units are refused outright. Mob.Stationary is honoured by the two AI
// movement drivers (they never integrate a stationary mob), so before this guard the
// effects engine was the ONE place in the server that could move something bolted to
// the ground: a «Штурм» cannon could be shoved out of its emplacement by a knockback
// and dragged around by a pull, and the illegal position went out to every client.
func (s *Server) teleportMobLocked(c *conn, m *mobState, x, y float32, now float64) {
	if m.structure || m.mob.Stationary {
		return
	}
	m.x, m.y = x, y
	m.vx, m.vy = 0, 0
	s.broadcastPosLocked(c, m.id, m.x, m.y, 0, 0, float32(now))
}
