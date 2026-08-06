package session

// FightLogEntry is one hero's row in a just-ended match's scoreboard: the wire shape
// ctrlserver's fight|log handler serves and the client's FightLogArgParser/GameEndMenu
// consume (BattleEndData). Built once, at match end, by battleserver (see
// battleserver/rating.go's settleMatchLocked) and handed to the Store because ctrlserver
// and battleserver share nothing else -- Store is the one thing both sides already hold.
type FightLogEntry struct {
	AvatarID int32
	Nick     string
	// Team is this hero's absolute in-battle team (1/2 for «Штурм», ArenaTeamA/B for
	// «Арена»), matching conn.playerTeam().
	Team int32
	// Kills/Assists/Deaths are this match's hero-kill tally (huntState.frags/
	// assists/deaths) -- never creep/structure kills, matching the client's own
	// AvatarKills naming.
	Kills, Assists, Deaths int32
	// Level is the in-battle level (0-based; the client's own display convention adds 1).
	Level int32
	// Money is the hero's current persistent wallet at match end (same "money" shape the
	// shop/item-tree screens already report), not match-earned income alone.
	Money int32
	// OldRating/NewRating are the persistent PvP rating before/after this match's Elo
	// settlement. Equal (no arrow shown client-side) for a bot row, or for any match that
	// skipped rating (see settleMatchLocked).
	OldRating, NewRating int32
}

// SetFightLog publishes a finished match's full scoreboard under battleID -- the
// CONNECT-issued id (see battleserver conn.battleID) the client that just received
// BATTLE_END will ask for by that exact number (fight|log's "fight_id"). Every
// participant's own connection got a DIFFERENT battleID at CONNECT time, so
// settleMatchLocked calls this once per participant, all with the same entries map.
func (s *Store) SetFightLog(battleID int32, entries map[int32]FightLogEntry) {
	if battleID == 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.fightLogs == nil {
		s.fightLogs = map[int32]map[int32]FightLogEntry{}
	}
	s.fightLogs[battleID] = entries
	// Bound the map: battleID only ever increments and is never otherwise consumed (a
	// client may ask for its scoreboard more than once, or never), so without this it
	// grows for the life of the process. Evicting the oldest handful once it's clearly
	// past "recent matches" is enough -- this is a scoreboard cache, not a match history.
	const maxFightLogs = 500
	if len(s.fightLogs) > maxFightLogs {
		s.evictOldestFightLogsLocked(maxFightLogs / 2)
	}
}

// evictOldestFightLogsLocked drops the n smallest battleIDs (battleID is a monotonically
// increasing counter, so smallest = oldest). Caller holds s.mu.
func (s *Store) evictOldestFightLogsLocked(n int) {
	ids := make([]int32, 0, len(s.fightLogs))
	for id := range s.fightLogs {
		ids = append(ids, id)
	}
	for i := 0; i < n && len(ids) > 0; i++ {
		minIdx := 0
		for j := 1; j < len(ids); j++ {
			if ids[j] < ids[minIdx] {
				minIdx = j
			}
		}
		delete(s.fightLogs, ids[minIdx])
		ids[minIdx] = ids[len(ids)-1]
		ids = ids[:len(ids)-1]
	}
}

// FightLog returns the scoreboard published for battleID, if any.
func (s *Store) FightLog(battleID int32) (map[int32]FightLogEntry, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	entries, ok := s.fightLogs[battleID]
	return entries, ok
}
