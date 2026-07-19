package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

// questProgress / questStatus read a hero's live state for one quest.
func questProgress(s *Server, uid, questID int32) int32 {
	for _, qs := range s.Store.HeroQuests(uid) {
		if qs.QuestID == questID {
			return qs.Progress
		}
	}
	return -1
}

func questStatus(s *Server, uid, questID int32) int32 {
	for _, qs := range s.Store.HeroQuests(uid) {
		if qs.QuestID == questID {
			return qs.Status
		}
	}
	return -99
}

// TestQuestKillAdvancesProgress drives the Battle-side hook end to end: a homed Hunt mob kill on
// the quest's map advances an accepted quest to DONE, while a homeless «Штурм» creep and a quest
// bound to a different map are left untouched.
func TestQuestKillAdvancesProgress(t *testing.T) {
	s, c, _, sx, sy := newNavConn(t) // killer is on map_4_0 (id 40)
	u, _, _ := s.Store.LoginOrRegister("qhunter@test.io", "pw")
	s.Store.CreateHero(u, 1, false, 0, 0, 0, 0, 0)
	c.selfPlayerID = u.ID

	// An accepted map-40 quest that needs several kills, plus a map-41 quest that must NOT move.
	var here gamedata.Quest
	for _, q := range gamedata.QuestsOnMap(40) {
		if q.Count >= 3 {
			here = q
			break
		}
	}
	if here.ID == 0 {
		t.Fatal("no map_40 quest with count>=3 in the catalog")
	}
	elsewhere := gamedata.QuestsOnMap(41)[0]
	s.Store.AcceptQuest(u.ID, here.ID)
	s.Store.AcceptQuest(u.ID, elsewhere.ID)

	idx := mobIndexByPrefab(t, "Mob_ZombieCrawl_01")
	kill := func(id int32, homed bool) {
		m := &mobState{
			id: id, mobIdx: idx, mob: gamedata.Mobs()[idx],
			x: sx + 3, y: sy, spawnX: sx + 3, spawnY: sy,
			hp: 10, shown: true, aggro: true, homed: homed,
		}
		c.mvMu.Lock()
		c.huntState.mobs[id] = m
		c.huntState.tr.add(id)
		s.hitMobLocked(c, m, 999, c.objID)
		c.mvMu.Unlock()
	}

	// A homeless creep (Штурм) does not advance a PvE quest.
	kill(3001, false)
	if p := questProgress(s, u.ID, here.ID); p != 0 {
		t.Fatalf("homeless-creep kill advanced quest to %d, want 0", p)
	}

	// A homed Hunt mob advances it one step.
	kill(3002, true)
	if p := questProgress(s, u.ID, here.ID); p != 1 {
		t.Fatalf("progress = %d after one homed kill, want 1", p)
	}

	// Finish the objective; status flips to DONE.
	for id := int32(3003); questProgress(s, u.ID, here.ID) < here.Count; id++ {
		kill(id, true)
	}
	if st := questStatus(s, u.ID, here.ID); st != gamedata.QuestStatusDone {
		t.Errorf("map-40 quest status = %d after objective, want DONE(%d)", st, gamedata.QuestStatusDone)
	}

	// The map-41 quest never moved off zero (kills happened on map 40).
	if p := questProgress(s, u.ID, elsewhere.ID); p != 0 {
		t.Errorf("map-41 quest advanced to %d on map-40 kills", p)
	}
}
