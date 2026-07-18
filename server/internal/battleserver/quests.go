package battleserver

import (
	"strconv"

	"tanatserver/internal/amf"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// PvE quest progress in battle: a Hunt mob kill advances the killer's accepted KILL/COLLECT
// quests for that map. The Ctrl channel owns the quest catalog, accept/turn-in and rewards
// (ctrlserver/quests.go); the Battle server only feeds progress, because the objective source --
// mobs dying -- lives here. Progress is map-scoped: any Hunt kill on the quest's map counts (the
// journal text tells the player which map + creature; per-mob targeting is a future refinement).
// Штурм creeps (homeless) and Arena carry no PvE quests and are excluded.

// creditQuestKillLocked advances the killer's quests for one slain Hunt mob and pushes the
// changed states over the shared MPD hub so the square/journal reflect them (live if the client
// still holds its Ctrl socket, otherwise on the next quest|update when they return to the city).
// Caller holds the conn lock; the store mutation itself is atomic under the store lock.
func (s *Server) creditQuestKillLocked(killer *conn, ms *mobState) {
	if killer == nil || ms == nil || !ms.homed {
		return // only homed Hunt mobs (not Штурм creeps) advance quests
	}
	hs := killer.huntState
	if hs == nil {
		return
	}
	mapID := hs.m.ID
	if _, ok := gamedata.HuntMapByID(mapID); !ok {
		return // killer is not in a Hunt (Штурм/Arena map) -> no PvE quests here
	}
	changed := s.Store.AddQuestKill(killer.selfPlayerID, mapID)
	if len(changed) == 0 {
		return
	}
	s.pushQuestStates(killer.selfPlayerID, changed)
}

// pushQuestStates delivers quest states to a user's MPD socket as quest|update_mpd. A kill only
// ever yields IN_PROGRESS or DONE states, so "time" (the cooldown field) is always -1 here.
func (s *Server) pushQuestStates(userID int32, states []session.QuestState) {
	if s.MPD == nil || len(states) == 0 {
		return
	}
	quests := amf.NewArray()
	for _, qs := range states {
		prog := amf.NewArray()
		prog.Set(strconv.Itoa(int(gamedata.QuestProgressID())), qs.Progress)
		quests.Set(strconv.Itoa(int(qs.QuestID)), amf.NewArray().
			Set("status", qs.Status).
			Set("time", int32(-1)).
			Set("progress", prog))
	}
	s.MPD.Push(userID, "quest|update", amf.NewArray().Set("quests", quests))
}
