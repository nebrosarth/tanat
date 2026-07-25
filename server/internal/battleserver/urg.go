package battleserver

import (
	"math"

	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
)

// This file implements Urg's two "grow real trees on the map" abilities, which the
// generic effector engine has no home for:
//   - «Древесный камуфляж» (slot 1, OpTreeForm): turn a friendly avatar into a TREE —
//     break-on-move stealth plus a standing disguise prop — and detonate a magic-damage
//     + silence burst on the surrounding enemies the instant they leave tree form.
//   - «Непроглядные дебри» (slot 4, OpGrove): a ring of trees around Urg that silences the
//     enemies caught inside, then FALLS after its duration and hits them with a magic burst.
// Both plant visible world props through spawnPropLocked and reuse the anchor-end queue for
// deletion, so nothing here re-implements object/tracker plumbing.

// spawnPropLocked creates one visible, inert prop (identified by an already-registered
// prototype id) at (x,y) on every party member's client and returns its object id. It
// mirrors spawnTrapAnchorLocked but for a VISIBLE prefab (the invisible "Dummy" anchor is
// its sibling); deletion is id-based via removeTrapAnchorLocked / the anchorEnds queue.
func (s *Server) spawnPropLocked(c *conn, proto int32, x, y float32, now float64) int32 {
	id := c.allocAnchorID()
	bt := float32(now)
	for _, mem := range c.members() {
		hs := mem.huntState
		if hs == nil || hs.tr.index(id) >= 0 {
			continue
		}
		idx := hs.tr.add(id)
		s.push(mem, battleproto.CmdCreateObject, amf.NewArray().
			Set("id", id).Set("proto", proto))
		s.push(mem, battleproto.CmdSync, amf.NewArray().Set("data",
			newSyncBlob(bt).addObject(id).
				position(idx, x, y, 0, 0, bt).
				build(hs.tr.count())))
	}
	return id
}

// applyTreeFormLocked arms Urg's tree camouflage (OpTreeForm) on the aimed friendly avatar
// (the caster in solo): it cloaks them as break-on-move stealth, plants the disguise prop at
// their feet, and stores the reveal burst so leaving tree form detonates it. The camouflage
// state lives on the TARGET's huntState, so the target's own reveal path (its move/attack/
// cast, or expiry) fires the burst around the target — correct for both solo and an ally cast.
func (s *Server) applyTreeFormLocked(c *conn, op gamedata.Op, ctx opCtx, now float64) {
	tc := c
	if ctx.allyTarget != nil {
		tc = ctx.allyTarget
	}
	hs := tc.huntState
	if hs == nil {
		return
	}
	dur := op.Dur.At(ctx.level)
	if dur <= 0 {
		return
	}
	// A tree form still armed on this avatar resolves first, so a re-cast never leaks a
	// disguise prop or a stale burst.
	s.fireTreeFormBurstLocked(tc, now)
	// Cloak as break-on-move stealth: the shared shade fx shows the vanish, and stepping out
	// (server.go's move handler) reveals via breakInvisibilityLocked -> fireTreeFormBurstLocked.
	s.applySkillStealthLocked(tc, dur, true, now)
	if proto, ok := wardProtoIDFor(op.Unit); ok {
		px, py := tc.posAtLocked(float32(now))
		hs.treeFormObj = s.spawnPropLocked(tc, proto, px, py, now)
	}
	hs.treeFormBurst = op.Ops
	hs.treeFormSlot = ctx.slot
	hs.treeFormLevel = ctx.level
	hs.treeFormUntil = now + dur
}

// fireTreeFormBurstLocked ends an active tree form: it removes the disguise prop and detonates
// the armed reveal burst (magic damage + silence) on the enemies around the avatar. Called
// from breakInvisibilityLocked (move/attack/cast reveal) and on natural expiry. No-op when no
// tree form is armed on this conn.
func (s *Server) fireTreeFormBurstLocked(c *conn, now float64) {
	hs := c.huntState
	if hs == nil {
		return
	}
	if hs.treeFormObj != 0 {
		s.removeTrapAnchorLocked(c, hs.treeFormObj, now)
		hs.treeFormObj = 0
	}
	if hs.treeFormBurst == nil {
		return
	}
	burst := hs.treeFormBurst
	hs.treeFormBurst = nil
	hs.treeFormUntil = 0
	px, py := c.posAtLocked(float32(now))
	ctx := opCtx{slot: hs.treeFormSlot, level: hs.treeFormLevel, px: px, py: py, hasPos: true}
	s.applyOpsLocked(c, burst, ctx, now)
}

// fireStealthBurstLocked detonates an armed OpStealth reveal burst (Wilfang's «Засада»:
// «при этом окружающие враги получают урон») around the avatar's CURRENT position. Called
// from breakInvisibilityLocked (move/attack/cast reveal) and on natural stealth expiry.
// No-op when no burst is armed on this conn.
func (s *Server) fireStealthBurstLocked(c *conn, now float64) {
	hs := c.huntState
	if hs == nil || hs.stealthBurst == nil {
		return
	}
	burst := hs.stealthBurst
	hs.stealthBurst = nil
	px, py := c.posAtLocked(float32(now))
	ctx := opCtx{slot: hs.stealthBurstSlot, level: hs.stealthBurstLevel, px: px, py: py, hasPos: true}
	s.applyOpsLocked(c, burst, ctx, now)
}

// applyGroveLocked grows Urg's dense-grove ring (OpGrove) around the caster: it plants Count
// tree props on a circle of Radius, schedules their fall at Dur (anchor-end queue), and queues
// the fall-damage burst (the nested Ops) as a deferred payload at the same instant, centred on
// the ring. The while-standing silence is a sibling top-level OpSilence on the skill.
func (s *Server) applyGroveLocked(c *conn, op gamedata.Op, ctx opCtx, now float64) {
	hs := c.huntState
	if hs == nil {
		return
	}
	dur := op.Dur.At(ctx.level)
	if dur <= 0 {
		return
	}
	cx, cy := c.posAtLocked(float32(now))
	r := op.Radius
	if r <= 0 {
		r = 6
	}
	if proto, ok := wardProtoIDFor(op.Unit); ok {
		n := int(op.Count.At(ctx.level))
		if n < 1 {
			n = 8
		}
		for i := 0; i < n; i++ {
			ang := 2 * math.Pi * float64(i) / float64(n)
			tx := cx + float32(r*math.Cos(ang))
			ty := cy + float32(r*math.Sin(ang))
			id := s.spawnPropLocked(c, proto, tx, ty, now)
			// The trees "fall" (delete) when the ring's duration elapses.
			hs.anchorEnds = append(hs.anchorEnds, anchorEnd{id: id, at: now + dur})
		}
	}
	// When the trees fall, the nested magic burst hits every enemy still inside the ring.
	if len(op.Ops) > 0 {
		hs.payloads = append(hs.payloads, payload{
			at: now + dur, slot: ctx.slot, level: ctx.level,
			px: cx, py: cy, hasPos: true, ops: op.Ops,
		})
	}
}
