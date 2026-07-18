package battleserver

import (
	"math/rand"

	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
)

// Loot drops: a mob death may spawn a ground chest (Drop_Chest_prop01) holding
// one random consumable, shared party-wide loot rights (anyone who was in the
// world when it dropped may pick it up -- no need/greed rules in v1).
//
// Wire contract (decompiled, see BattleServerConnection.cs/DropContent.cs):
// the client auto-requests GET_DROP_INFO(19) the moment DropContent.Init sees
// the container object, and separately sends PICK_UP(20) with the DROPPED
// ITEM's own id (not the container's) once the player takes it. We never need
// to originate REFRESH_DROP_CONTENT(537): our chest holds exactly one item, so
// a pickup deletes the whole container object for every viewer immediately,
// which is simpler than the general "still has other items" case the real
// client also supports.
const (
	trashDropChance             = 15 // 1-in-N chance per trash kill; bosses always drop
	dropChestProtoID     int32  = 901
	dropChestBaseID      int32  = 500000 // object-id base, clear of trap anchors (400000+)
	dropItemBaseID       int32  = 550000 // dropped-item wire-id base (its own namespace)
	dropChestPrefab      string = "Drop_Chest_prop01"
)

// dropState is one spawned loot chest.
type dropState struct {
	id        int32 // chest world-object id
	itemObjID int32 // the single item's wire id (GET_DROP_INFO/PICK_UP address it, not id)
	article   int32 // gamedata item article id
	x, y      float32
	allowed   []int32 // avatar objIDs permitted to loot (every member present at drop time)
}

func dropChestProtoDesc() string {
	return `<Proto><PPrefab value="` + xmlEsc(dropChestPrefab) + `"/><PItemContainer value="true"/></Proto>`
}

// rollDropLocked reports whether ms should drop loot: bosses (any Skills)
// always do; trash rolls a flat 1-in-trashDropChance chance.
//
// «Штурм» drops NOTHING. It is a match, not a farm -- there is no dungeon to stock up
// for and no run to carry loot out of. Lane creeps, cannons, towers, generators and the
// enemy Fortress Crystal are all ordinary mobStates on the shared death path, so without
// this gate the mode rolled a 1-in-15 chest off every one of them and dropped a
// consumable on the ground at the exact moment the match was won.
func (s *Server) rollDropLocked(c *conn, ms *mobState) bool {
	if c.inst != nil && c.inst.dota != nil {
		return false
	}
	if len(ms.mob.Skills) > 0 {
		return true
	}
	return rand.Intn(trashDropChance) == 0
}

// randomDropArticle picks a uniformly random consumable KIND, then a uniformly
// random item within it -- so Health/Mana (24 items each, an irregular
// level-bracket x rarity grid) don't drop many times more often than the
// 4-tier buff families.
func randomDropArticle() int32 {
	kinds := [...]gamedata.ItemKind{
		gamedata.ItemHealthPotion, gamedata.ItemManaPotion,
		gamedata.ItemHealthFlask, gamedata.ItemManaFlask,
		gamedata.ItemSpeedPotion, gamedata.ItemInvisibilityPotion,
		gamedata.ItemRevelationPotion, gamedata.ItemDodgeChancePotion,
		gamedata.ItemCritStrikePotion,
		gamedata.ItemAntiPhysArmorPotion, gamedata.ItemAntiMagicArmorPotion,
	}
	pool := gamedata.ItemsByKind(kinds[rand.Intn(len(kinds))])
	return pool[rand.Intn(len(pool))].ArticleID
}

// allocDropChestID / allocDropItemID hand out party-wide (instance) or
// per-conn (solo/bare-conn test) ids, both based clear of every other id
// space -- mirrors allocAnchorID exactly.
func (c *conn) allocDropChestID() int32 {
	if c.inst != nil {
		c.inst.nextDropID++
		return c.inst.nextDropID
	}
	if c.huntState.nextDropID < dropChestBaseID {
		c.huntState.nextDropID = dropChestBaseID
	}
	c.huntState.nextDropID++
	return c.huntState.nextDropID
}

func (c *conn) allocDropItemID() int32 {
	if c.inst != nil {
		c.inst.nextDropItemID++
		return c.inst.nextDropItemID
	}
	if c.huntState.nextDropItemID < dropItemBaseID {
		c.huntState.nextDropItemID = dropItemBaseID
	}
	c.huntState.nextDropItemID++
	return c.huntState.nextDropItemID
}

// dropsMapLocked returns the shared (instance) or fallback (solo/bare-conn)
// drop set, lazily initialized -- mirrors how hs.mobs aliases inst.mobs.
func (c *conn) dropsMapLocked() map[int32]*dropState {
	if c.inst != nil {
		if c.inst.drops == nil {
			c.inst.drops = map[int32]*dropState{}
		}
		return c.inst.drops
	}
	if c.huntState.drops == nil {
		c.huntState.drops = map[int32]*dropState{}
	}
	return c.huntState.drops
}

// findDropByItemLocked finds the chest holding wireID (the ITEM's own id, as
// carried in PICK_UP), returning it plus its container id.
func (c *conn) findDropByItemLocked(wireID int32) (*dropState, int32, bool) {
	for id, d := range c.dropsMapLocked() {
		if d.itemObjID == wireID {
			return d, id, true
		}
	}
	return nil, 0, false
}

func allowedContains(allowed []int32, objID int32) bool {
	for _, a := range allowed {
		if a == objID {
			return true
		}
	}
	return false
}

func allowedArray(ids []int32) *amf.MixedArray {
	a := amf.NewArray()
	for _, id := range ids {
		a.Add(id)
	}
	return a
}

// spawnDropLocked creates a loot chest at (px,py) with one random consumable,
// visible to every member currently in the world (mirrors spawnTrapAnchorLocked).
// Newcomers who join the instance after this point won't see it -- a known v1
// limitation (chests aren't fog-of-war-revealed like mobs), acceptable since a
// drop is transient and the party is normally already together for the kill.
func (s *Server) spawnDropLocked(c *conn, px, py float32, now float64) {
	article := randomDropArticle()
	chestID := c.allocDropChestID()
	itemID := c.allocDropItemID()
	members := c.members()
	allowed := make([]int32, 0, len(members))
	for _, mem := range members {
		allowed = append(allowed, mem.objID)
	}
	d := &dropState{id: chestID, itemObjID: itemID, article: article, x: px, y: py, allowed: allowed}
	c.dropsMapLocked()[chestID] = d

	for _, mem := range members {
		hs := mem.huntState
		if hs == nil || hs.tr.index(chestID) >= 0 {
			continue
		}
		idx := hs.tr.add(chestID)
		s.push(mem, battleproto.CmdCreateObject, amf.NewArray().
			Set("id", chestID).Set("proto", dropChestProtoID))
		s.push(mem, battleproto.CmdSync, amf.NewArray().Set("data",
			newSyncBlob(float32(now)).addObject(chestID).
				position(idx, px, py, 0, 0, float32(now)).
				build(hs.tr.count())))
		s.ensureItemProtoLocked(mem, article)
	}
}

// handleGetDropInfo answers a client's auto-fired GET_DROP_INFO(19) (sent when
// DropContent.Init sees the chest object) with the container's single item
// entry: {id, proto, count, allowed}. This must be answered with EXACTLY ONE
// reply carrying the real payload, on the request's own RequestID -- unlike
// PICK_UP/EQUIP_ITEM (pure validator commands where a bare success/fail ack is
// the whole reply and any resulting state change is a SEPARATE push to
// whoever it affects), GET_DROP_INFO's reply IS the data
// (BattleScreen.OnDropInfo/DropInfoArgParser reads {id,info} straight off it).
// Sending a bare empty ack first and the real data as a second, unprompted
// push (RequestID -1) under the same cmd was the bug behind "chest looks wrong
// and clicking it does nothing": the client's DropContent.SetContent ran once
// with empty content from the ack, so its loot-glow VFX never started and
// mContent stayed empty for any click to act on.
func (s *Server) handleGetDropInfo(c *conn, p battleproto.Packet) {
	if c.hunt == nil || c.huntState == nil {
		s.ack(c, p)
		return
	}
	containerID := p.Args.IntOr("id", -1)
	c.lock()
	defer c.unlock()
	d, ok := c.dropsMapLocked()[containerID]
	if !ok {
		s.ack(c, p) // unknown container (already looted/gone): bare success, no data
		return
	}
	info := amf.NewArray()
	info.Add(amf.NewArray().
		Set("id", d.itemObjID).Set("proto", d.article).Set("count", int32(1)).
		Set("allowed", allowedArray(d.allowed)))
	_ = c.send(battleproto.Packet{
		Cmd: p.Cmd, RequestID: p.RequestID, Status: true,
		Args: amf.NewArray().Set("id", containerID).Set("info", info),
	})
}

// handlePickUp answers PICK_UP(20), whose "id" is the dropped ITEM's id (not
// the container's). On success: delete the chest for every viewer (v1 chests
// hold exactly one item, so there's never anything left to refresh) and credit
// the item to the picker's persistent bag.
func (s *Server) handlePickUp(c *conn, p battleproto.Packet) {
	if c.hunt == nil || c.huntState == nil {
		s.ack(c, p)
		return
	}
	itemID := p.Args.IntOr("id", -1)
	c.lock()
	defer c.unlock()
	d, containerID, found := c.findDropByItemLocked(itemID)
	if !found || !allowedContains(d.allowed, c.objID) {
		_ = c.send(battleproto.Packet{Cmd: p.Cmd, RequestID: p.RequestID, Status: false, Error: "not allowed"})
		return
	}
	s.ack(c, p)
	now := float64(s.battleTime())
	delete(c.dropsMapLocked(), containerID)
	for _, mem := range c.members() {
		s.untrackObjForMemberLocked(mem, containerID, float32(now))
	}
	s.grantItemLocked(c, d.article, 1)
}
