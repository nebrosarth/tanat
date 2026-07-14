package battleserver

import (
	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
)

// A SELF-mode ground fx (its FxSkillTarget baked SELF in the client) PARENTS itself
// to the owner object's transform, so owning it to the caster makes it trail the
// avatar. Elgorm's «Оскверненная почва» (slot 3) ground hazard is exactly this. To
// pin such a fx at the cast point we spawn an INVISIBLE, STATIONARY anchor object
// there and own the fx to that instead -- the same trick the Vigilans barrier uses
// (owned to the rooted mob). The anchor is the shipped "Dummy" prefab: it carries a
// VisualEffectOptions component (required or the SELF fx silently fails to attach),
// renders nothing (its MeshRenderer has zero materials), keeps localScale=(1,1,1) (so
// the client's fx-scale division stays finite), and has no ExportObjectData/collider
// (so it isn't auto-destroyed and shows no selection ring).
const (
	trapAnchorProtoID = 900     // free: summon protos 800..804, avatar effBase >=1100
	trapAnchorBaseID  = 400000  // object-id base, clear of avatar/mob/summon(300000) ids
	trapAnchorPrefab  = "Dummy" // invisible marker prefab with VisualEffectOptions
)

// trapAnchorProtoDesc is the prototype the client instantiates for an anchor: just the
// invisible prefab. No PDestructible (no client path requires it for a plain object)
// and no PAvatar/PBuilding/PShop (those route into unrelated handling).
func trapAnchorProtoDesc() string {
	return `<Proto><PPrefab value="` + xmlEsc(trapAnchorPrefab) + `"/></Proto>`
}

// trapUsesAnchor reports whether a trap's ground fx is SELF-mode (parents to its
// owner) and so must be pinned to a stationary anchor to hold the cast point. The fx
// target mode is baked in the client, not in the skill data, so this is hand-
// maintained by prefab+slot (like channelInterruptible): Elgorm's «Оскверненная
// почва» (slot 3) uses SELF-baked fx (ElgormSkill3Effect1 trap + ElgormSkill3Effect2
// trigger).
func trapUsesAnchor(prefab string, slot int) bool {
	return prefab == "Avtr_Dsb_Elgorm" && slot == 3
}

// allocAnchorID hands out a party-wide anchor object id (instance space) or a per-conn
// one for a solo/bare-conn, both based clear of every other id space.
func (c *conn) allocAnchorID() int32 {
	if c.inst != nil {
		c.inst.nextAnchorID++
		return c.inst.nextAnchorID
	}
	if c.huntState.nextAnchorID < trapAnchorBaseID {
		c.huntState.nextAnchorID = trapAnchorBaseID
	}
	c.huntState.nextAnchorID++
	return c.huntState.nextAnchorID
}

// anchorEnd defers deleting an anchor object until its trigger fx has played out (the
// SELF-mode trigger fx is parented to the anchor, so the anchor must outlive it).
type anchorEnd struct {
	id int32
	at float64
}

// spawnTrapAnchorLocked creates the invisible stationary anchor at (px,py) on every
// member's client (owner + teammates, each with its own tracker index) and returns its
// object id. A SELF-mode trap fx owned to this id then holds the point.
func (s *Server) spawnTrapAnchorLocked(c *conn, px, py float32, now float64) int32 {
	id := c.allocAnchorID()
	for _, mem := range c.members() {
		hs := mem.huntState
		if hs == nil || hs.tr.index(id) >= 0 {
			continue
		}
		idx := hs.tr.add(id)
		s.push(mem, battleproto.CmdCreateObject, amf.NewArray().
			Set("id", id).Set("proto", trapAnchorProtoID))
		s.push(mem, battleproto.CmdSync, amf.NewArray().Set("data",
			newSyncBlob(float32(now)).addObject(id).
				position(idx, px, py, 0, 0, float32(now)).
				build(hs.tr.count())))
	}
	return id
}

// removeTrapAnchorLocked deletes an anchor from every member's client (tracker swap-
// remove + DELETE_OBJECT). No-op for id 0.
func (s *Server) removeTrapAnchorLocked(c *conn, id int32, now float64) {
	if id == 0 {
		return
	}
	for _, mem := range c.members() {
		s.untrackObjForMemberLocked(mem, id, float32(now))
	}
}

// tickAnchorEndsLocked deletes anchors whose deferred-removal time has passed.
func (s *Server) tickAnchorEndsLocked(c *conn, now float64) {
	hs := c.huntState
	if len(hs.anchorEnds) == 0 {
		return
	}
	keep := hs.anchorEnds[:0:0]
	for _, a := range hs.anchorEnds {
		if now < a.at {
			keep = append(keep, a)
			continue
		}
		s.removeTrapAnchorLocked(c, a.id, now)
	}
	hs.anchorEnds = keep
}
