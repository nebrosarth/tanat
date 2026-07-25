package battleserver

import (
	"sync"

	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
)

// A vision ward (Urg's «Росток» acorn) is a stationary friendly SCOUT prop planted at
// the cast point. It is a pure vision utility -- no damage, no CC:
//   - the visible prop is rendered on EVERY member's client (a real world object, so
//     enemies see it too), carrying the caster's TEAM and a fog-of-war VIEW_RADIUS, so
//     the WarFogObject on a friendly client lights up the patch of terrain around it
//     («открывает небольшой участок местности для всей дружественной команды»);
//   - each world tick it REVEALS any stealthed enemy within its radius by breaking their
//     invisibility (balance patch 1.08's «видеть невидимых врагов»);
//   - it also acts as an EYE for Hunt interest-management (mobViewDistLocked in mobai.go):
//     a mob standing inside the ward's radius is created + unshaded on every client even
//     with no player nearby, so «Росток» shows the mobs on the patch it opens (on a Hunt
//     map distant mobs are otherwise never sent to the client at all).
// It is NOT a trap: it never triggers or consumes -- it simply stands for its lifetime.

// wardProtoBaseID must dodge every other fixed proto id: summon protos 800..804,
// trapAnchor=900, dropChest=901 (a COLLISION here made the client render the scout tree
// as a loot chest), avatar effBase>=1100. 950 is clear.
const wardProtoBaseID = 950

// wardProto is one distinct scout-prop prototype (stable id/desc), like summonProto.
type wardProto struct {
	id     int32
	prefab string
	desc   string
}

var (
	wardProtoOnce  sync.Once
	wardProtoList  []wardProto
	wardProtoByKey map[string]int32
)

// wardProtoDesc is the prototype the client instantiates for a scout prop: just the
// visible prefab. No PBuilding/PDestructible/PAvatar (those route into combat/selection
// handling) and no collider, so it is an inert prop -- not a target, not auto-destroyed.
func wardProtoDesc(prefab string) string {
	return `<Proto><PPrefab value="` + xmlEsc(prefab) + `"/></Proto>`
}

func buildWardProtos() {
	wardProtoByKey = map[string]int32{}
	for _, a := range gamedata.Avatars() {
		for _, sk := range gamedata.SkillsFor(a).Skills {
			for _, op := range sk.Ops {
				// Every op that plants a visible prop needs a client prototype: the scout
				// ward (OpVisionWard), Urg's tree-camouflage disguise (OpTreeForm) and his
				// dense-grove ring trees (OpGrove) all resolve their prefab through here.
				switch op.Kind {
				case gamedata.OpVisionWard, gamedata.OpTreeForm, gamedata.OpGrove:
				default:
					continue
				}
				if op.Unit == "" {
					continue
				}
				if _, seen := wardProtoByKey[op.Unit]; seen {
					continue
				}
				id := wardProtoBaseID + int32(len(wardProtoList))
				wardProtoByKey[op.Unit] = id
				wardProtoList = append(wardProtoList, wardProto{
					id: id, prefab: op.Unit, desc: wardProtoDesc(op.Unit),
				})
			}
		}
	}
}

// wardProtos returns every distinct scout-prop prototype (stable order/ids), so every
// member's world-state chain can register the whole set up front -- like summonProtos.
func wardProtos() []wardProto {
	wardProtoOnce.Do(buildWardProtos)
	return wardProtoList
}

// wardProtoIDFor maps a scout-prop prefab to its prototype id.
func wardProtoIDFor(prefab string) (int32, bool) {
	wardProtoOnce.Do(buildWardProtos)
	id, ok := wardProtoByKey[prefab]
	return id, ok
}

// wardState is one planted scout ward.
type wardState struct {
	obj    int32   // the visible prop object id (allocAnchorID space)
	x, y   float32 // fixed plant point
	radius float64 // fog view + stealth-detect radius
	until  float64 // battle-time expiry
	fx     int32   // optional persistent ground fx (0 = none)
}

// spawnVisionWardLocked plants a scout ward at the cast point: it creates the visible
// prop on every member's client (owner + teammates AND enemies -- it is a real world
// object), synced with the caster's team and a fog VIEW_RADIUS so friendly clients
// reveal the area, starts the optional ground fx, and records the ward for the tick.
func (s *Server) spawnVisionWardLocked(c *conn, op gamedata.Op, ctx opCtx, now float64) {
	hs := c.huntState
	if hs == nil {
		return
	}
	proto, ok := wardProtoIDFor(op.Unit)
	if !ok {
		return // unknown prop prefab -> nothing to render
	}
	px, py := s.centerLocked(c, ctx)
	team := c.playerTeam()
	view := effectiveViewRadius(float32(op.Radius))
	id := c.allocAnchorID()
	bt := float32(now)
	for _, mem := range c.members() {
		mhs := mem.huntState
		if mhs == nil || mhs.tr.index(id) >= 0 {
			continue
		}
		idx := mhs.tr.add(id)
		s.push(mem, battleproto.CmdCreateObject, amf.NewArray().
			Set("id", id).Set("proto", proto))
		s.push(mem, battleproto.CmdSync, amf.NewArray().Set("data",
			newSyncBlob(bt).addObject(id).
				position(idx, px, py, 0, 0, bt).
				setFloats(syncViewRadius, idx, view).
				setInt(syncTeam, idx, team).
				build(mhs.tr.count())))
	}
	// The ground fx is owned to the (stationary) ward object so it stays pinned at the
	// plant point rather than trailing the caster. fxStartLocked broadcasts across the
	// instance, so every member sees it.
	fx := s.fxStartLocked(c, op.TrapFx, id, 0, true, px, py)
	hs.wards = append(hs.wards, wardState{
		obj: id, x: px, y: py, radius: op.Radius, until: now + op.Lifetime.At(ctx.level), fx: fx,
	})
}

// tickWardsLocked expires wards past their lifetime (end fx + delete the prop on every
// member) and, for live wards, reveals stealthed enemies within radius.
func (s *Server) tickWardsLocked(c *conn, now float64) {
	hs := c.huntState
	if len(hs.wards) == 0 {
		return
	}
	keep := hs.wards[:0:0]
	for _, w := range hs.wards {
		if now > w.until {
			s.fxEndLocked(c, w.fx)
			s.removeTrapAnchorLocked(c, w.obj, now) // same delete-on-every-member path
			continue
		}
		s.revealStealthedEnemiesLocked(c, w, now)
		keep = append(keep, w)
	}
	hs.wards = keep
}

// revealStealthedEnemiesLocked breaks the invisibility of every ENEMY-team member whose
// body is within the ward's radius (patch 1.08). In co-op Hunt every member shares the
// player's team, so this is a no-op there; it bites in PvP modes where enemy avatars can
// stealth. Mobs carry no invisibility of their own, so only players are scanned.
func (s *Server) revealStealthedEnemiesLocked(c *conn, w wardState, now float64) {
	myTeam := c.playerTeam()
	r2 := w.radius * w.radius
	for _, mem := range c.members() {
		mhs := mem.huntState
		if mhs == nil || mem.playerTeam() == myTeam {
			continue
		}
		if now >= mhs.invisibleUntil { // not currently stealthed
			continue
		}
		mx, my := mem.posAtLocked(float32(now))
		dx, dy := float64(mx-w.x), float64(my-w.y)
		if dx*dx+dy*dy <= r2 {
			s.breakInvisibilityLocked(mem, now)
		}
	}
}
