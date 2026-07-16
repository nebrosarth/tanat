package battleserver

import (
	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
)

// Fan-out helpers for the shared hunt world. A mob (or any shared object) is
// rendered by several members at once, each with its OWN tracking index, so a
// SYNC blob has to be rebuilt per member; object-id-addressed packets (ACTION,
// RECEIVE_HIT, ...) are identical and just replayed to each viewer. Outside an
// instance c.members() is [c], so every one of these collapses to a single push
// -- the single-player path is byte-for-byte unchanged.

// mobViewersLocked calls fn for every member that currently renders objID, with
// that member's own tracking index and post-update object count. Caller holds
// the world lock.
func (c *conn) mobViewersLocked(objID int32, fn func(mem *conn, idx, count int)) {
	for _, mem := range c.members() {
		tr := mem.renderTr()
		if tr == nil {
			continue
		}
		if idx := tr.index(objID); idx >= 0 {
			fn(mem, idx, tr.count())
		}
	}
}

// broadcastObjLocked replays an object-id-addressed packet to every member that
// renders objID (the args are identical for all, so one build, many pushes).
func (s *Server) broadcastObjLocked(c *conn, objID int32, cmd battleproto.CmdID, args *amf.MixedArray) {
	c.mobViewersLocked(objID, func(mem *conn, _, _ int) {
		s.push(mem, cmd, args)
	})
}

// broadcastPosLocked fans a POSITION SYNC (x, y, velocity, snapshot time) for objID
// out to every member that renders it, each with its OWN tracking index. Used for
// mob, summon and avatar movement. Outside an instance c.members() is [c], so it
// collapses to a single push -- byte-identical to the pre-fan-out code.
func (s *Server) broadcastPosLocked(c *conn, objID int32, x, y, vx, vy, t float32) {
	c.mobViewersLocked(objID, func(mem *conn, idx, count int) {
		s.push(mem, battleproto.CmdSync, amf.NewArray().Set("data",
			newSyncBlob(t).position(idx, x, y, vx, vy, t).build(count)))
	})
}

// broadcastStatLocked fans a single typed-float SYNC (e.g. HEALTH fraction, SPEED)
// for objID out to every member that renders it, each with its own tracking index.
// Callers do any clamp/frac math before the call.
func (s *Server) broadcastStatLocked(c *conn, objID int32, typ uint64, val, t float32) {
	c.mobViewersLocked(objID, func(mem *conn, idx, count int) {
		s.push(mem, battleproto.CmdSync, amf.NewArray().Set("data",
			newSyncBlob(t).setFloats(typ, idx, val).build(count)))
	})
}

// addAttackEffectorLocked pushes the ATTACK effector bound to objID on mem's client
// -- the effector that lets the client turn objID's basic-attack ACTION into a
// swing/projectile. Shared by the avatar/mob/summon reveal helpers (parent=-1,
// empty args, a fresh per-member effector id).
func (s *Server) addAttackEffectorLocked(mem *conn, objID, attackProto int32, now float64) {
	s.push(mem, battleproto.CmdAddEffector,
		addEffectorArgs(mem.huntState.newEffID(), attackProto, objID, -1, now, nil))
}

// untrackObjForMemberLocked removes objID from ONE member's client via the delicate
// swap-remove SYNC path (tracker.remove -> removeIndex SYNC -> DELETE_OBJECT); no-op
// if the member wasn't tracking it. Shared by the avatar/mob/summon hide helpers so
// the removal-index/count math lives in exactly one place.
func (s *Server) untrackObjForMemberLocked(mem *conn, objID int32, now float32) {
	tr := mem.renderTr()
	if tr == nil {
		return
	}
	if idx := tr.remove(objID); idx >= 0 {
		s.push(mem, battleproto.CmdSync, amf.NewArray().Set("data",
			newSyncBlob(now).removeIndex(idx).build(tr.count())))
		s.push(mem, battleproto.CmdDeleteObject, amf.NewArray().Set("id", objID))
	}
}

// broadcastAvatarObjLocked replays an object-id-addressed packet about THIS
// player's own avatar (ACTION, ACTION_DONE, RECEIVE_HIT, ...) to every member
// that renders it -- so teammates see this player swing, cast and get hit.
func (s *Server) broadcastAvatarObjLocked(c *conn, cmd battleproto.CmdID, args *amf.MixedArray) {
	s.broadcastObjLocked(c, c.objID, cmd, args)
}

// broadcastAvatarToOthersLocked replays an avatar-owned packet (ACTION,
// ACTION_DONE, SET_PROJECTILE) to every OTHER member that renders this player's
// avatar. The owner already pushed the packet to itself, so this only adds the
// teammate copies -- keeping the self path (and every solo/bare-conn test) byte
// identical. A teammate receives it only if it renders the avatar (it does, once
// introduced) and, for the effector-driven ACTIONs, carries the matching attack
// effector added in renderAvatarForLocked. Outside an instance c.members() is
// [c], so this is a no-op.
func (s *Server) broadcastAvatarToOthersLocked(c *conn, cmd battleproto.CmdID, args *amf.MixedArray) {
	for _, mem := range c.members() {
		if mem == c {
			continue
		}
		hs := mem.huntState
		if hs == nil {
			continue
		}
		if hs.tr.index(c.objID) >= 0 {
			s.push(mem, cmd, args)
		}
	}
}

// pushAvatarAllLocked emits a self-owned avatar packet (ACTION, ACTION_DONE,
// SET_PROJECTILE) to the owner AND every teammate that renders the avatar: the
// owner push comes first (broadcastAvatarToOthersLocked's contract), then the
// teammate copies. This is the canonical "render on self + viewers" op; the two
// calls must stay in lockstep, so they live here in one place. Outside an instance
// the broadcast is a no-op, so the solo path is byte-identical to a bare s.push.
func (s *Server) pushAvatarAllLocked(c *conn, cmd battleproto.CmdID, args *amf.MixedArray) {
	s.push(c, cmd, args)
	s.broadcastAvatarToOthersLocked(c, cmd, args)
}

// worldFxStartLocked pushes an EFFECT_START for a SHARED visual (mob debuff, boss
// telegraph, fog-ring shade) to every member that renders the owner object, using
// one world-scoped effect id so it looks identical everywhere and can be ended on
// all clients. Returns the id (0 if fx is empty). Outside an instance it falls
// back to the per-connection path (owner-only), identical to fxStartLocked.
func (s *Server) worldFxStartLocked(c *conn, fx string, owner, target int32, hasPos bool, px, py float32) int32 {
	if fx == "" {
		return 0
	}
	if c.inst == nil {
		return s.fxStartLocked(c, fx, owner, target, hasPos, px, py)
	}
	c.inst.nextFxUID++
	uid := c.inst.nextFxUID
	args := amf.NewArray()
	if target != 0 {
		args.Set("target", target)
	}
	if hasPos {
		args.Set("targetPos", amf.NewArray().Set("x", float64(px)).Set("y", float64(py)))
	}
	p := amf.NewArray().Set("effect", uid).Set("owner", owner).Set("fx", fx).Set("args", args)
	for _, mem := range c.members() {
		hs := mem.huntState
		if hs == nil {
			continue
		}
		// Only push to a member that actually has the owner object (mobs are
		// per-member fog-gated); a point fx (owner<=0) goes to everyone.
		if owner > 0 && hs.tr.index(owner) < 0 {
			continue
		}
		s.push(mem, battleproto.CmdEffectStart, p)
	}
	return uid
}

// worldFxEndLocked ends a world-scoped effect on every member. An unknown id is a
// harmless no-op client-side, so ending everywhere is safe even if a member never
// saw the start. Falls back to the per-connection end outside an instance.
func (s *Server) worldFxEndLocked(c *conn, uid int32) {
	if uid == 0 {
		return
	}
	if c.inst == nil {
		s.fxEndLocked(c, uid)
		return
	}
	p := amf.NewArray().Set("id", uid)
	for _, mem := range c.members() {
		if mem.huntState == nil {
			continue
		}
		s.push(mem, battleproto.CmdEffectEnd, p)
	}
}
