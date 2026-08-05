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

// payloadFxUsesAnchor reports whether a skill's POINT payload fx is SELF-mode (parents
// to its owner) and so must be pinned to a stationary anchor to hold the cast point,
// rather than trailing the caster. Like trapUsesAnchor the fx mode is baked in the
// client, hand-maintained by prefab+slot: Titanid's «Землетрясение» (slot 1) quake fx.
func payloadFxUsesAnchor(prefab string, slot int) bool {
	// Titanid's «Землетрясение» (slot 1) quake fx; Edilia's «Дерево жизни» (slot 4) tree --
	// both are SELF-mode ground fx that would otherwise trail the caster instead of standing
	// at the cast point (the tree "follows Edilia" report).
	//
	// Miriam's «Убийственный залп» (slot 4) is the same shape: MiriamSkill4Effect's gfx
	// (VFX_Avtr_DPS_Miriam_skill4_prop01, a PrefabTimeSpawn dropping arrows every 1s with
	// m_UseParentTransorm=false) is baked SELF, so its whole arrow-spawner -- and every
	// arrow it drops -- rained down wherever Miriam stood instead of the clicked area.
	//
	// Morlokai's «Проклятие Вуду» (slot 1) is the same shape and then some. Its
	// MorlokaySkill1Effect1 gfx is the ground decal VFX_Avtr_Dsb_Morlokay_Skill1_prop01 --
	// a Projector under a VisualEffect with mMaxDuration=0 and mLoopEffect=true, so it
	// never expires on its own, AND the registry marks that gfx stop_on_done=false, which
	// means Skill.StartEffects never records a handle for it and EFFECT_END physically
	// cannot stop it. Owned to the caster it therefore did exactly what was reported:
	// followed Morlokai forever. Pinning it to an anchor fixes both halves at once -- the
	// circle stands where it was cast, and deleting the anchor (anchorEnds, below) takes
	// the parented decal with it, which is the only lever the server has over an
	// unstoppable looping fx.
	// Sharli's «Лавовые оковы» (slot 3): SharliSkill3Effect's gfx VFX_Avtr_DPS_Sharli_
	// skill3_prop01 is a SELF-baked ground Projector + falling-hammer mesh (its own
	// VisualEffectTargets carries no targets), so it parented to the caster instead of
	// landing at the clicked point -- «молот создаётся не в точке каста, а на аватаре».
	return (prefab == "Avtr_Tank_Titanid" && slot == 1) ||
		(prefab == "Avtr_Dsb_Edilia" && slot == 4) ||
		(prefab == "Avtr_Dsb_Morlokay" && slot == 1) ||
		(prefab == "Avtr_DPS_Sharli" && slot == 3) ||
		(prefab == "Avtr_DPS_Miriam" && slot == 4)
}

// payloadTargetFxOwnedToTarget reports whether a skill's target-mode payload fx is a
// SELF-baked prefab that must be OWNED by (parented to) the struck enemy, not the caster.
// A SELF-baked fx follows its owner GameObject; started with owner=caster it renders on the
// avatar instead of the victim (Charlie/Sharli s1 «Ожог» burn appearing on Sharli). Kept a
// narrow prefab+slot allowlist so genuine caster-anchored or TARGET-baked payloads are
// untouched.
func payloadTargetFxOwnedToTarget(prefab string, slot int) bool {
	// Kiona's «Страж леса» (slot 4) is the same shape: KionaSkill4Effect's gfx
	// VFX_Avtr_SP_Kiona_Skill4_prop01 is baked SELF, so owning it to the caster parented
	// the guardian owl to KIONA -- it followed her around instead of watching over the
	// unit she cast it on, for the whole 10s the buff/DoT runs.
	// Frost's «Гробница холода» (slot 3): FrostSkill3Effect1 is the ice FORMING over the
	// victim -- a SELF-baked FreezeEffect with a ground projector -- so owned to the caster
	// it froze Frost instead of whoever she aimed at.
	return (prefab == "Avtr_DPS_Sharli" && slot == 1) ||
		(prefab == "Avtr_Sp_Kiona" && slot == 4) ||
		(prefab == "Avtr_Dsb_Frost" && slot == 3)
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
