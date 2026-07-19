package session

// One-time migration from the retired JSON account file into SQLite. The old
// store marshalled the whole account graph to accounts.json (see the pre-SQLite
// persist.go); this reads that file, bcrypt-hashes each stored plaintext
// password (the legacy format kept them in the clear), and write-throughs every
// account into the database. It runs only against a brand-new DB, and the JSON
// is archived to accounts.json.imported afterwards so it never re-imports.

import (
	"encoding/json"
	"os"
)

// legacySnapshot / legacyUser mirror the exact shape the old JSON store wrote
// (Go default field names, snapshot tags). They exist ONLY to read the archive;
// the live User struct no longer carries a plaintext Password.
type legacySnapshot struct {
	NextUserID int32        `json:"next_user_id"`
	Users      []legacyUser `json:"users"`
}

type legacyUser struct {
	ID       int32
	Email    string
	Password string // legacy plaintext -> hashed on import
	Username string
	HasHero  bool
	Hero     *Hero
	Friends  []int32
	Ignores  []int32
}

// importLegacyJSONLocked reads path, converts each account (hashing its
// password), and persists ALL of them in ONE transaction, only populating memory
// after that commit succeeds. Returning an error means nothing was written (the
// tx rolled back) -- the caller then leaves accounts.json in place so the
// one-shot migration can retry instead of archiving the only copy over lost data.
// Caller holds s.mu (or, as at startup, runs before any concurrency).
func (s *Store) importLegacyJSONLocked(path string) (int, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	var snap legacySnapshot
	if err := json.Unmarshal(b, &snap); err != nil {
		return 0, err
	}

	built := make([]*User, 0, len(snap.Users))
	for i := range snap.Users {
		lu := &snap.Users[i]
		u := &User{
			ID:       lu.ID,
			Email:    lu.Email,
			Username: lu.Username,
			PassHash: hashPassword(lu.Password),
			Hero:     lu.Hero,
			Friends:  lu.Friends,
			Ignores:  lu.Ignores,
		}
		if u.Hero != nil {
			u.HasHero = true
			u.Hero.ID = u.ID
			u.Hero.sanitize()
		}
		built = append(built, u)
	}

	// All-or-nothing: a constraint violation or a crash mid-import leaves the DB
	// empty, so the next boot re-runs the whole import.
	if err := s.persistUsersLocked(built...); err != nil {
		return 0, err
	}

	for _, u := range built {
		s.usersByEmail[u.Email] = u
		s.usersByID[u.ID] = u
		if u.ID >= s.nextUserID {
			s.nextUserID = u.ID + 1
		}
	}
	if snap.NextUserID > s.nextUserID {
		s.nextUserID = snap.NextUserID
	}
	s.persistNextUserIDLocked()
	return len(built), nil
}
