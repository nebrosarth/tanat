package session

import (
	"encoding/json"
	"log"
	"os"
	"path/filepath"
)

// PLACEHOLDER PERSISTENCE: accounts (email/password/hero) are written to a JSON
// file so heroes survive restarts. Passwords are stored in the clear because
// auth is already a placeholder (any password logs in); swap for a real hashed
// credential store before this is exposed beyond a local test client. Sessions
// are intentionally NOT persisted -- clients re-login and get a fresh key.

type snapshot struct {
	NextUserID int32   `json:"next_user_id"`
	Users      []*User `json:"users"`
}

// NewPersistentStore creates a store backed by the JSON file at path, loading
// any existing accounts. A missing file is fine (starts empty). A malformed
// file is logged and ignored (starts empty) rather than losing the server.
func NewPersistentStore(path string) *Store {
	s := NewStore()
	s.path = path
	if err := s.load(); err != nil {
		log.Printf("session: could not load %s: %v (starting empty)", path, err)
	}
	return s
}

func (s *Store) load() error {
	b, err := os.ReadFile(s.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	var snap snapshot
	if err := json.Unmarshal(b, &snap); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.nextUserID = snap.NextUserID
	if s.nextUserID < 1 {
		s.nextUserID = 1
	}
	for _, u := range snap.Users {
		if u.Hero != nil {
			u.HasHero = true
			u.Hero.sanitize()
		}
		s.usersByEmail[u.Email] = u
		s.usersByID[u.ID] = u
	}
	log.Printf("session: loaded %d account(s) from %s", len(snap.Users), s.path)
	return nil
}

// saveLocked writes the current accounts to disk. Caller holds s.mu. Writes to a
// temp file then renames so a crash mid-write can't corrupt the store.
func (s *Store) saveLocked() {
	if s.path == "" {
		return
	}
	snap := snapshot{NextUserID: s.nextUserID, Users: make([]*User, 0, len(s.usersByID))}
	for _, u := range s.usersByID {
		snap.Users = append(snap.Users, u)
	}
	b, err := json.MarshalIndent(snap, "", "  ")
	if err != nil {
		log.Printf("session: marshal error: %v", err)
		return
	}
	if dir := filepath.Dir(s.path); dir != "" {
		_ = os.MkdirAll(dir, 0o755)
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		log.Printf("session: write error: %v", err)
		return
	}
	if err := os.Rename(tmp, s.path); err != nil {
		log.Printf("session: rename error: %v", err)
	}
}
