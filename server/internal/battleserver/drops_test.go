package battleserver

import (
	"testing"

	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// TestRollDropChances pins the two odds the user asked for: a boss (any
// Skills) always drops, and a trash mob rolls a flat 1-in-trashDropChance
// chance (currently 15).
func TestRollDropChances(t *testing.T) {
	s, c, _, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()

	boss := &mobState{mob: gamedata.Mob{Skills: []gamedata.BossSkill{{}}}}
	for i := 0; i < 50; i++ {
		if !s.rollDropLocked(c, boss) {
			t.Fatal("boss should always drop")
		}
	}

	trash := &mobState{}
	const trials = 30000
	hits := 0
	for i := 0; i < trials; i++ {
		if s.rollDropLocked(c, trash) {
			hits++
		}
	}
	want := trials / trashDropChance
	lo, hi := want*70/100, want*130/100
	if hits < lo || hits > hi {
		t.Errorf("trash drop hits = %d over %d trials, want ~%d (%d..%d)", hits, trials, want, lo, hi)
	}
}

// TestDropPickupFlow exercises the full loot loop: a mob death spawns a chest
// with one random consumable, GET_DROP_INFO reports it, and PICK_UP credits
// the item to the picker's persistent bag and removes the chest for everyone.
func TestDropPickupFlow(t *testing.T) {
	s, c, u, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	c.hunt = &session.PendingBattle{} // handleGetDropInfo/handlePickUp require a live hunt

	now := float64(s.battleTime())
	c.lock()
	s.spawnDropLocked(c, 10, 20, now)
	c.unlock()

	drops := c.dropsMapLocked()
	if len(drops) != 1 {
		t.Fatalf("dropsMapLocked() has %d entries, want 1", len(drops))
	}
	var chestID int32
	var d *dropState
	for id, dd := range drops {
		chestID, d = id, dd
	}
	if !allowedContains(d.allowed, c.objID) {
		t.Fatalf("dropped chest doesn't allow the only party member: %+v", d.allowed)
	}
	it, ok := gamedata.ItemByArticle(d.article)
	if !ok {
		t.Fatalf("dropped article %d isn't a known item", d.article)
	}

	s.handleGetDropInfo(c, battleproto.Packet{Cmd: battleproto.CmdGetDropInfo, RequestID: 1,
		Args: amf.NewArray().Set("id", chestID)})

	s.handlePickUp(c, battleproto.Packet{Cmd: battleproto.CmdPickUp, RequestID: 2,
		Args: amf.NewArray().Set("id", d.itemObjID)})

	if len(c.dropsMapLocked()) != 0 {
		t.Error("chest still present after pickup")
	}
	bag := s.Store.HeroBag(u.ID)
	if len(bag) != 1 || bag[0].ArticleID != it.ArticleID || bag[0].Count != 1 {
		t.Errorf("persisted bag after pickup = %+v, want [{%d 1}]", bag, it.ArticleID)
	}
}

// TestPickUpRejectsNonMember: a player who was never allowed to loot this
// chest (not present when it dropped) gets a failure reply and no item.
func TestPickUpRejectsNonMember(t *testing.T) {
	s, c, u, cleanup := newHuntConnWithHero(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	c.hunt = &session.PendingBattle{}

	now := float64(s.battleTime())
	c.lock()
	s.spawnDropLocked(c, 0, 0, now)
	c.unlock()

	var d *dropState
	for _, dd := range c.dropsMapLocked() {
		d = dd
	}
	d.allowed = nil // simulate: this conn was not present at drop time

	s.handlePickUp(c, battleproto.Packet{Cmd: battleproto.CmdPickUp, RequestID: 3,
		Args: amf.NewArray().Set("id", d.itemObjID)})

	if len(c.dropsMapLocked()) != 1 {
		t.Error("chest was removed despite a rejected pickup")
	}
	if bag := s.Store.HeroBag(u.ID); len(bag) != 0 {
		t.Errorf("bag credited despite a rejected pickup: %+v", bag)
	}
}
