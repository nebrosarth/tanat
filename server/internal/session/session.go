// Package session provides a minimal in-memory account/session store.
//
// PLACEHOLDER AUTH: the original server presumably validated passwords
// against a real accounts database. We don't have that database, so for now
// any email/password pair logs in, auto-registering the account on first
// use. Swap Store for a real persistence layer + password check before this
// touches anything but a local test client.
package session

import (
	"crypto/rand"
	"encoding/hex"
	"sync"
)

type User struct {
	ID       int32
	Email    string
	Password string
	Username string
	HasHero  bool
	Hero     *Hero
}

type Hero struct {
	ID        int32
	Race      int32
	Gender    bool
	Face      int32
	Hair      int32
	DistMark  int32
	SkinColor int32
	HairColor int32

	// Lobby progression/economy. The hero is the account's persistent character
	// (1:1 with User); money and level carry across matches, and the permanent
	// "hero set" equipment lives here too (see DressedItem).
	Money        int32
	DiamondMoney int32
	Level        int32
	Exp          int32
	NextExp      int32
	Dressed      []DressedItem
}

// DressedItem is one permanently-equipped hero-set item (gives stats across all
// matches). Wire shape from HeroDataListArgParser: {id, artikul_id, cnt, slot}.
type DressedItem struct {
	ID        int32
	ArticleID int32
	Count     int32
	Slot      int32
}

type Session struct {
	Key    string
	UserID int32
}

// PendingBattle is the handoff between the Ctrl and Battle channels when a
// game mode launches: hunt|ready (Ctrl) stores it, and the Battle server
// consumes it when the same user reconnects with the issued password. Held in
// memory only -- a server restart just drops the player back into the lobby.
type PendingBattle struct {
	MapID    int32
	AvatarID int32
	Passwd   string
	Scene    string
	// Room is the shared-world id the player was matched into (open instance per
	// map). The Battle server routes every launch with the same Room into one
	// huntInstance. Zero/unset means a private solo world (the Battle server keys it
	// by the negative user id).
	Room int32
}

type Store struct {
	mu           sync.Mutex
	usersByEmail map[string]*User
	usersByID    map[int32]*User
	sessions     map[string]*Session
	pending      map[int32]PendingBattle
	lobbyArea    map[int32]int32 // userID -> central-square area the client last requested
	nextUserID   int32
	path         string // JSON persistence file; "" = in-memory only
}

func NewStore() *Store {
	return &Store{
		usersByEmail: map[string]*User{},
		usersByID:    map[int32]*User{},
		sessions:     map[string]*Session{},
		pending:      map[int32]PendingBattle{},
		lobbyArea:    map[int32]int32{},
		nextUserID:   1,
	}
}

// LoginOrRegister finds the account for email, creating it if this is the
// first time we've seen it, and issues a fresh session key.
func (s *Store) LoginOrRegister(email, password string) (*User, string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	u, ok := s.usersByEmail[email]
	if !ok {
		u = &User{
			ID:       s.nextUserID,
			Email:    email,
			Password: password,
			Username: email,
		}
		s.nextUserID++
		s.usersByEmail[email] = u
		s.usersByID[u.ID] = u
		s.saveLocked()
	}
	key := newToken()
	s.sessions[key] = &Session{Key: key, UserID: u.ID}
	return u, key
}

// ByID looks a user up by id. The Battle channel authenticates with the numeric
// user id (CONNECT's clientId), not a session key, so it uses this.
func (s *Store) ByID(id int32) (*User, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, ok := s.usersByID[id]
	return u, ok
}

// AddHeroMoney credits (or debits) a user's persistent hero money by delta and
// returns the new virtual-money / diamond totals. ok=false if the user or hero is
// missing. Used by the Battle channel to pay out mob-kill coin bounties. Money is
// a single integer decomposed client-side into gold/silver/bronze (1 bronze = 1
// unit), so a "6 bronze" reward is simply delta=6.
func (s *Store) AddHeroMoney(userID, delta int32) (money, diamonds int32, ok bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, found := s.usersByID[userID]
	if !found || u.Hero == nil {
		return 0, 0, false
	}
	u.Hero.Money += delta
	if u.Hero.Money < 0 {
		u.Hero.Money = 0
	}
	return u.Hero.Money, u.Hero.DiamondMoney, true
}

func (s *Store) BySessKey(key string) (*User, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	sess, ok := s.sessions[key]
	if !ok {
		return nil, false
	}
	u, ok := s.usersByID[sess.UserID]
	return u, ok
}

// CreateHero's Hero.ID intentionally equals the owning User.ID: the real
// client's SelfHero.Hero looks a hero up by mUserData.UserId directly (see
// TanatKernel.SelfHero.Hero), so hero and user ids must match 1:1.
//
// Customization values are clamped to >= 0: the client sends -1 for options
// the player never touched, but treats the served values as raw indices into
// its customization lists (HeroMgr.GetCustomizeColor does list[value] with no
// negative guard) — serving -1 back crashes hero rendering with an
// ArgumentOutOfRangeException.
func (s *Store) CreateHero(u *User, race int32, gender bool, face, hair, distMark, skinColor, hairColor int32) *Hero {
	s.mu.Lock()
	defer s.mu.Unlock()
	h := &Hero{
		ID:        u.ID,
		Race:      race,
		Gender:    gender,
		Face:      clampIdx(face),
		Hair:      clampIdx(hair),
		DistMark:  clampIdx(distMark),
		SkinColor: clampIdx(skinColor),
		HairColor: clampIdx(hairColor),
		// Starter lobby defaults so the client's UI populates sensibly.
		Money:        1000,
		DiamondMoney: 100,
		Level:        1,
		Exp:          0,
		NextExp:      100,
	}
	u.Hero = h
	u.HasHero = true
	s.saveLocked()
	return h
}

// SetPendingBattle records that userID has been issued a battle launch (mode
// entry): the Battle server will match the CONNECT password against it.
func (s *Store) SetPendingBattle(userID int32, pb PendingBattle) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pending[userID] = pb
}

// SetLobbyArea records which central-square area (1 = human city, 2 = elf city) the
// client last asked to enter via common|area_conf, so the Battle server can render
// the matching scene's walkability/spawn regardless of the hero's race (a player can
// visit the other race's square).
func (s *Store) SetLobbyArea(userID, area int32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lobbyArea[userID] = area
}

// LobbyArea returns the central-square area the user last entered, or 0 if none was
// recorded yet (the caller then falls back to the hero's home square by race).
func (s *Store) LobbyArea(userID int32) int32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lobbyArea[userID]
}

// TakePendingBattle consumes the pending launch for userID, if any. It is
// removed on read so a later plain reconnect (empty lobby password) drops the
// user back into the central square instead of re-entering a stale battle.
func (s *Store) TakePendingBattle(userID int32) (PendingBattle, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	pb, ok := s.pending[userID]
	if ok {
		delete(s.pending, userID)
	}
	return pb, ok
}

// Save persists the store to disk (no-op when in-memory). Callers that mutate a
// User/Hero outside the Store's own methods (e.g. spending money) should call it.
func (s *Store) Save() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.saveLocked()
}

// clampIdx normalizes a hero customization index ("-1" = untouched slider).
func clampIdx(v int32) int32 {
	if v < 0 {
		return 0
	}
	return v
}

// sanitize repairs a hero loaded from disk (heroes stored before the clamp was
// added may carry the client's -1 markers).
func (h *Hero) sanitize() {
	h.Face = clampIdx(h.Face)
	h.Hair = clampIdx(h.Hair)
	h.DistMark = clampIdx(h.DistMark)
	h.SkinColor = clampIdx(h.SkinColor)
	h.HairColor = clampIdx(h.HairColor)
}

func newToken() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}
