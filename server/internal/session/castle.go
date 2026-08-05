package session

// CASTLE OWNERSHIP + BATTLE HISTORY + REWARD PAYOUT for "Битва за замок". Ownership
// is a persistent castle-id -> clan-id mapping (a castle a clan holds until it loses
// a siege); history is an append-only per-castle log the castle|history screen reads.
// Both are Store-level (not gamedata) because gamedata.Castles() is a static seed and
// this is state that changes at runtime, exactly like clan headers.

import (
	"database/sql"
	"log"

	"tanatserver/internal/gamedata"
)

// castleOwnerRec is the in-memory ownership record for one castle.
type castleOwnerRec struct {
	ClanID   int32
	ClanName string
}

// CastleBattleRecord is one finished siege, oldest first in the per-castle log
// (castle|history reads it most-recent-first -- callers reverse when building the
// wire response).
type CastleBattleRecord struct {
	CastleID       int32
	WinnerClanID   int32
	WinnerClanName string
	EndedAt        int64
}

// CastleOwner returns the clan currently holding castleID, or (0, "", false) if it
// is unowned/neutral.
func (s *Store) CastleOwner(castleID int32) (clanID int32, clanName string, ok bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rec, found := s.castleOwner[castleID]
	if !found || rec.ClanID == 0 {
		return 0, "", false
	}
	return rec.ClanID, rec.ClanName, true
}

// CastleHistory returns castleID's finished-battle log, oldest first.
func (s *Store) CastleHistory(castleID int32) []CastleBattleRecord {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]CastleBattleRecord(nil), s.castleLog[castleID]...)
}

// SettleCastleBattle records a siege's outcome: the winning clan becomes the new
// owner (or keeps ownership, if the defender won), a history entry is appended, and
// every winning fighter is credited the castle's Reward (money/diamonds/exp, with
// level-ups) in the SAME commit -- mirrors CompleteQuest's single-saveLocked pattern
// so a crash can't leave ownership transferred with rewards uncredited, or vice versa.
// winnerClanID may be 0 (winner had no clan, e.g. a solo test fight): ownership then
// simply doesn't change, but the fighters are still paid.
func (s *Store) SettleCastleBattle(castleID, winnerClanID int32, winnerFighterIDs []int32, r gamedata.CastleReward, endedAt int64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if winnerClanID != 0 {
		name := ""
		if c := s.clansByID[winnerClanID]; c != nil {
			name = c.Name
		}
		s.castleOwner[castleID] = castleOwnerRec{ClanID: winnerClanID, ClanName: name}
		s.persistCastleOwnerLocked(castleID, winnerClanID, name)
		s.castleLog[castleID] = append(s.castleLog[castleID], CastleBattleRecord{
			CastleID: castleID, WinnerClanID: winnerClanID, WinnerClanName: name, EndedAt: endedAt,
		})
		s.persistCastleHistoryLocked(castleID, winnerClanID, name, endedAt)
	}

	for _, uid := range winnerFighterIDs {
		u := s.usersByID[uid]
		if u == nil || u.Hero == nil {
			continue
		}
		h := u.Hero
		if r.Money > 0 {
			h.Money += r.Money
		}
		if r.Diamonds > 0 {
			h.DiamondMoney += r.Diamonds
		}
		if r.Exp > 0 {
			h.Exp += r.Exp
			for h.NextExp > 0 && h.Exp >= h.NextExp {
				h.Exp -= h.NextExp
				h.Level++
				h.NextExp = heroExpNextLevel(h.Level)
			}
		}
		s.saveUserLocked(u)
	}
	log.Printf("session: castle %d settled, winner clan=%d fighters_paid=%d", castleID, winnerClanID, len(winnerFighterIDs))
}

// ---- persistence ----

func (s *Store) persistCastleOwnerLocked(castleID, clanID int32, clanName string) {
	if s.db == nil {
		return
	}
	if _, err := s.db.Exec(
		`INSERT INTO castle_owners(castle_id, clan_id, clan_name) VALUES(?,?,?)
		 ON CONFLICT(castle_id) DO UPDATE SET clan_id=excluded.clan_id, clan_name=excluded.clan_name`,
		castleID, clanID, clanName); err != nil {
		log.Printf("session: persist castle owner %d failed: %v", castleID, err)
	}
}

func (s *Store) persistCastleHistoryLocked(castleID, winnerClanID int32, winnerClanName string, endedAt int64) {
	if s.db == nil {
		return
	}
	if _, err := s.db.Exec(
		`INSERT INTO castle_history(castle_id, winner_clan_id, winner_clan_name, ended_at) VALUES(?,?,?,?)`,
		castleID, winnerClanID, winnerClanName, endedAt); err != nil {
		log.Printf("session: persist castle history %d failed: %v", castleID, err)
	}
}

// loadCastlesLocked restores ownership + history after a restart. Called from
// loadAllLocked once the clans are in memory (ownership doesn't strictly need them,
// but keeping the load order consistent with clan-dependent state is simplest).
func (s *Store) loadCastlesLocked() error {
	if s.db == nil {
		return nil
	}
	if err := s.loadChild(`SELECT castle_id, clan_id, clan_name FROM castle_owners`, func(rows *sql.Rows) error {
		var castleID, clanID int32
		var clanName string
		if err := rows.Scan(&castleID, &clanID, &clanName); err != nil {
			return err
		}
		if clanID != 0 {
			s.castleOwner[castleID] = castleOwnerRec{ClanID: clanID, ClanName: clanName}
		}
		return nil
	}); err != nil {
		return err
	}
	return s.loadChild(`SELECT castle_id, winner_clan_id, winner_clan_name, ended_at FROM castle_history ORDER BY id ASC`,
		func(rows *sql.Rows) error {
			var rec CastleBattleRecord
			if err := rows.Scan(&rec.CastleID, &rec.WinnerClanID, &rec.WinnerClanName, &rec.EndedAt); err != nil {
				return err
			}
			s.castleLog[rec.CastleID] = append(s.castleLog[rec.CastleID], rec)
			return nil
		})
}
