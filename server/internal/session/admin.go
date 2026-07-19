package session

// ADMIN STORE HELPERS. Everything the admin panel needs to read and mutate the
// account/hero store lives here, on top of the same write-through SQLite backing
// used by the game handlers. Every mutation persists immediately (saveUserLocked)
// or deletes through the CASCADE schema, so an admin edit survives a restart.

import (
	"log"
	"sort"
)

// AdminPlayer is a flat, JSON-friendly view of one account for the admin player
// list. Hero fields are zero when the account has no hero yet.
type AdminPlayer struct {
	ID        int32  `json:"id"`
	Email     string `json:"email"`
	Username  string `json:"username"`
	HasHero   bool   `json:"has_hero"`
	Banned    bool   `json:"banned"`
	Online    bool   `json:"online"`
	CreatedAt int64  `json:"created_at"`
	Race      int32  `json:"race"`
	Level     int32  `json:"level"`
	Exp       int32  `json:"exp"`
	NextExp   int32  `json:"next_exp"`
	Money     int32  `json:"money"`
	Diamonds  int32  `json:"diamonds"`
}

// ListPlayers returns every account as an AdminPlayer, sorted by id. Online is
// approximated by whether the account currently holds a Ctrl session key.
func (s *Store) ListPlayers() []AdminPlayer {
	s.mu.Lock()
	defer s.mu.Unlock()
	online := make(map[int32]bool, len(s.sessions))
	for _, sess := range s.sessions {
		online[sess.UserID] = true
	}
	out := make([]AdminPlayer, 0, len(s.usersByID))
	for _, u := range s.usersByID {
		p := AdminPlayer{
			ID:        u.ID,
			Email:     u.Email,
			Username:  u.Username,
			HasHero:   u.HasHero,
			Banned:    u.Banned,
			Online:    online[u.ID],
			CreatedAt: u.CreatedAt,
		}
		if u.Hero != nil {
			p.Race = u.Hero.Race
			p.Level = u.Hero.Level
			p.Exp = u.Hero.Exp
			p.NextExp = u.Hero.NextExp
			p.Money = u.Hero.Money
			p.Diamonds = u.Hero.DiamondMoney
		}
		out = append(out, p)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// SetHeroMoney sets an account's persistent money/diamond wallet to absolute
// values (negatives clamped to 0). Returns false if the account has no hero.
func (s *Store) SetHeroMoney(userID, money, diamonds int32) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, ok := s.usersByID[userID]
	if !ok || u.Hero == nil {
		return false
	}
	if money < 0 {
		money = 0
	}
	if diamonds < 0 {
		diamonds = 0
	}
	u.Hero.Money = money
	u.Hero.DiamondMoney = diamonds
	s.saveUserLocked(u)
	return true
}

// SetHeroProgress sets an account's level/exp/next-exp (level clamped to >=1,
// exp to >=0; a non-positive nextExp leaves the existing threshold untouched).
// Returns false if the account has no hero.
func (s *Store) SetHeroProgress(userID, level, exp, nextExp int32) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, ok := s.usersByID[userID]
	if !ok || u.Hero == nil {
		return false
	}
	if level < 1 {
		level = 1
	}
	if exp < 0 {
		exp = 0
	}
	u.Hero.Level = level
	u.Hero.Exp = exp
	if nextExp > 0 {
		u.Hero.NextExp = nextExp
	}
	s.saveUserLocked(u)
	return true
}

// SetBanned flags/unflags an account. Banning also drops any live Ctrl session so
// the account can't keep acting on an already-issued key (the Battle/MPD sockets
// are separate, but they can't re-authenticate once the session is gone, and the
// login gate refuses the next reconnect). Returns false for an unknown account.
func (s *Store) SetBanned(userID int32, banned bool) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, ok := s.usersByID[userID]
	if !ok {
		return false
	}
	u.Banned = banned
	if banned {
		for k, sess := range s.sessions {
			if sess.UserID == userID {
				delete(s.sessions, k)
			}
		}
	}
	s.saveUserLocked(u)
	return true
}

// IsBanned reports whether an account is currently banned (read under the lock,
// so it is consistent with a concurrent SetBanned).
func (s *Store) IsBanned(userID int32) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, ok := s.usersByID[userID]
	return ok && u.Banned
}

// InvalidateSession revokes a single session key (used by the login gate to undo
// the session a banned account was just issued). A no-op for an unknown key.
func (s *Store) InvalidateSession(key string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.sessions, key)
}

// DeleteAccount permanently removes an account: one DELETE FROM users cascades to
// the hero and every child table, and all in-memory state keyed by the id is
// dropped (including this id from any other account's friend/ignore lists, which
// have no cascading FK). Returns false for an unknown account or a failed DB
// delete (in which case the in-memory graph is left untouched).
func (s *Store) DeleteAccount(userID int32) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, ok := s.usersByID[userID]
	if !ok {
		return false
	}
	if s.db != nil {
		if _, err := s.db.Exec(`DELETE FROM users WHERE id=?`, userID); err != nil {
			log.Printf("session: delete account %d failed: %v", userID, err)
			return false
		}
	}
	delete(s.usersByEmail, u.Email)
	delete(s.usersByID, userID)
	delete(s.pending, userID)
	delete(s.lobbyArea, userID)
	delete(s.groups, userID)
	delete(s.friendReqs, userID)
	for k, sess := range s.sessions {
		if sess.UserID == userID {
			delete(s.sessions, k)
		}
	}
	// Scrub the deleted id from other accounts' social lists (friend_id/ignore_id
	// carry no FK, so the CASCADE above doesn't reach them) and re-persist the ones
	// that changed.
	for _, other := range s.usersByID {
		changed := false
		if nf := removeID(other.Friends, userID); len(nf) != len(other.Friends) {
			other.Friends = nf
			changed = true
		}
		if ni := removeID(other.Ignores, userID); len(ni) != len(other.Ignores) {
			other.Ignores = ni
			changed = true
		}
		if changed {
			s.saveUserLocked(other)
		}
	}
	return true
}

// ---- quest state (admin overrides) ----

// AdminHeroQuests returns a raw copy of a hero's persisted quest states. Unlike the
// game-facing HeroQuests it does NOT drop REPLAY quests whose cooldown has elapsed, so
// the operator sees exactly what is stored. ok=false if the account has no hero.
func (s *Store) AdminHeroQuests(userID int32) ([]QuestState, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, ok := s.usersByID[userID]
	if !ok || u.Hero == nil {
		return nil, false
	}
	out := make([]QuestState, len(u.Hero.Quests))
	copy(out, u.Hero.Quests)
	return out, true
}

// AdminSetQuestState upserts one quest's state for a hero to explicit values, bypassing
// the normal accept/complete state machine (an operator override). It never pays a
// reward -- it only writes Status/Progress/CooldownUntil. The caller is responsible for
// validating questID against the catalog and for clamping Progress to the objective.
// Negative Progress is floored to 0. Returns false if the account has no hero.
func (s *Store) AdminSetQuestState(userID, questID, status, progress int32, cooldownUntil int64) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, ok := s.usersByID[userID]
	if !ok || u.Hero == nil {
		return false
	}
	if progress < 0 {
		progress = 0
	}
	h := u.Hero
	if cur := findQuestLocked(h, questID); cur != nil {
		cur.Status = status
		cur.Progress = progress
		cur.CooldownUntil = cooldownUntil
		s.saveUserLocked(u)
		return true
	}
	h.Quests = append(h.Quests, QuestState{QuestID: questID, Status: status, Progress: progress, CooldownUntil: cooldownUntil})
	s.saveUserLocked(u)
	return true
}

// AdminRemoveQuest deletes a hero's state for questID unconditionally -- unlike
// CancelQuest, which refuses to drop a CLOSED/cooling record. This is an operator reset:
// removing a CLOSED one-time quest re-arms it for the player to take (and be rewarded)
// again, which is exactly the intended "reset this quest" behaviour here. Returns false
// if the hero holds no state for that quest.
func (s *Store) AdminRemoveQuest(userID, questID int32) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, ok := s.usersByID[userID]
	if !ok || u.Hero == nil {
		return false
	}
	h := u.Hero
	for i := range h.Quests {
		if h.Quests[i].QuestID == questID {
			h.Quests = append(h.Quests[:i], h.Quests[i+1:]...)
			s.saveUserLocked(u)
			return true
		}
	}
	return false
}

// AdminGrantWearable mints count fresh unequipped wearable instances of articleID into a
// hero's Owned bag WITHOUT charging gold (an operator gift, unlike BuyWearables). Each
// gets a stable instance id so it can later be dressed/sold. The caller validates
// articleID against the wearable catalog. Returns the minted instances; ok=false if the
// account has no hero or count<=0.
func (s *Store) AdminGrantWearable(userID, articleID, count int32) ([]OwnedItem, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, ok := s.usersByID[userID]
	if !ok || u.Hero == nil || count <= 0 {
		return nil, false
	}
	h := u.Hero
	added := make([]OwnedItem, 0, count)
	for i := int32(0); i < count; i++ {
		it := OwnedItem{ID: mintItemIDLocked(h), ArticleID: articleID}
		h.Owned = append(h.Owned, it)
		added = append(added, it)
	}
	s.saveUserLocked(u)
	return added, true
}

// GetMeta reads a value from the persistent meta key/value table. ok is false for
// a missing key or an in-memory (no-DB) store.
func (s *Store) GetMeta(key string) (value string, ok bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.db == nil {
		return "", false
	}
	if err := s.db.QueryRow(`SELECT value FROM meta WHERE key=?`, key).Scan(&value); err != nil {
		return "", false
	}
	return value, true
}

// SetMeta upserts a value into the meta table (a no-op, nil error, for an
// in-memory store).
func (s *Store) SetMeta(key, value string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.db == nil {
		return nil
	}
	_, err := s.db.Exec(
		`INSERT INTO meta(key, value) VALUES(?, ?)
		 ON CONFLICT(key) DO UPDATE SET value=excluded.value`, key, value)
	return err
}
