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
	"database/sql"
	"encoding/hex"
	"log"
	"strings"
	"sync"

	"golang.org/x/crypto/bcrypt"

	"tanatserver/internal/gamedata"
)

type User struct {
	ID    int32
	Email string
	// PassHash is the bcrypt hash of the account password (never the plaintext,
	// never sent to the client). Empty only for a malformed row; see checkPassword.
	PassHash string
	// CreatedAt is the unix time the account first registered (0 = unknown, e.g.
	// migrated from the pre-SQLite JSON store). Not sent to the client.
	CreatedAt int64
	Username  string
	HasHero   bool
	Hero      *Hero
	// Banned, when true, refuses the account at login (the Ctrl handler replies with
	// the client's BANNED code). Set/cleared from the admin panel; persisted.
	Banned bool
	// Friends/Ignores are the account's persistent social lists (white/black list in
	// the client's get_bw_list). Friendship is mutual (both carry each other); ignore
	// is one-directional.
	Friends []int32
	Ignores []int32
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
	// Bag is the hero's persistent consumable inventory (potions), merged by
	// article id (one stack per distinct item). Survives logout/restart, unlike
	// the ephemeral hunt-instance battle state.
	Bag []BagItem
	// Owned is the hero's unequipped WEARABLE gear (bought in the city shop, not
	// yet dressed). Unlike Bag potions these are DISCRETE instances (one row each,
	// never merged) with a STABLE per-hero instance ID, because user|dress and
	// store|sell address a specific owned instance by that id. Dressing moves an
	// instance from Owned into Dressed (keeping its id); undressing moves it back.
	Owned []OwnedItem
	// NextItemID mints the stable wearable instance ids (starts at
	// heroItemInstanceBase, only ever increments, never reused).
	NextItemID int32
	// Quests is the hero's persistent PvE quest state (accepted/in-progress/done/closed +
	// kill progress + REPLAY cooldowns). See session/quests.go for the state machine.
	Quests []QuestState
}

// BagItem is one persisted consumable stack. ArticleID is the gamedata item
// catalog id (shared between the Ctrl-channel bag's artikul_id and the Battle
// channel's inventory "proto").
type BagItem struct {
	ArticleID int32
	Count     int32
}

// DressedItem is one permanently-equipped hero-set item (gives stats across all
// matches). Wire shape from HeroDataListArgParser: {id, artikul_id, cnt, slot}.
type DressedItem struct {
	ID        int32
	ArticleID int32
	Count     int32
	Slot      int32
}

// OwnedItem is one unequipped wearable the hero owns (a discrete instance with a
// stable id). It has no slot until dressed; on the user|bag wire it appears as
// {id, artikul_id, cnt:1, used:0} alongside the potion stacks.
type OwnedItem struct {
	ID        int32
	ArticleID int32
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
	lobbyArea    map[int32]int32          // userID -> central-square area the client last requested
	groups       map[int32]*Group         // userID -> the party it belongs to (shared pointer)
	friendReqs   map[int32]map[int32]bool // target userID -> set of users with a pending friend request to them
	nextUserID   int32
	path         string  // SQLite database file; "" = in-memory only
	db           *sql.DB // nil for an in-memory store (NewStore)
}

func NewStore() *Store {
	return &Store{
		usersByEmail: map[string]*User{},
		usersByID:    map[int32]*User{},
		sessions:     map[string]*Session{},
		pending:      map[int32]PendingBattle{},
		lobbyArea:    map[int32]int32{},
		groups:       map[int32]*Group{},
		friendReqs:   map[int32]map[int32]bool{},
		nextUserID:   1,
	}
}

// LoginOrRegister authenticates email/password, auto-registering the account on
// first use, and issues a fresh session key. ok is false (with a nil user, no
// session issued) when the email already exists but the password does not match
// its bcrypt hash -- the Ctrl login handler turns that into a WRONG_PASS reply.
// The first login for an email sets its password.
func (s *Store) LoginOrRegister(email, password string) (u *User, key string, ok bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	u, exists := s.usersByEmail[email]
	if !exists {
		u = &User{
			ID:        s.nextUserID,
			Email:     email,
			PassHash:  hashPassword(password),
			Username:  email,
			CreatedAt: nowUnix(),
		}
		s.nextUserID++
		s.usersByEmail[email] = u
		s.usersByID[u.ID] = u
		s.saveUserLocked(u)
		s.persistNextUserIDLocked()
	} else if !u.checkPassword(password) {
		return nil, "", false
	}
	key = newToken()
	s.sessions[key] = &Session{Key: key, UserID: u.ID}
	return u, key, true
}

// hashPassword returns the bcrypt hash of a plaintext password. Any bcrypt error
// (which, thanks to bcryptInput's cap, no longer includes the >72-byte case) is
// logged and yields an empty hash; checkPassword then FAILS CLOSED on that empty
// hash so a hashing failure can never produce a passwordless account.
func hashPassword(password string) string {
	h, err := bcrypt.GenerateFromPassword(bcryptInput(password), bcrypt.DefaultCost)
	if err != nil {
		log.Printf("session: bcrypt hash failed: %v", err)
		return ""
	}
	return string(h)
}

// checkPassword reports whether password matches the account's stored hash. An
// empty stored hash (a malformed/never-hashed row) fails closed -- it must never
// authenticate an arbitrary password.
func (u *User) checkPassword(password string) bool {
	if u.PassHash == "" {
		return false
	}
	return bcrypt.CompareHashAndPassword([]byte(u.PassHash), bcryptInput(password)) == nil
}

// bcryptInput caps the password at bcrypt's hard 72-byte limit. Without this a
// long password -- e.g. a ~37-character Cyrillic one, 2 bytes/char in UTF-8 --
// makes GenerateFromPassword return ErrPasswordTooLong. Capping (bcrypt's own
// historical behavior) makes hashing deterministic; the same cap on verify keeps
// hash and check consistent.
func bcryptInput(password string) []byte {
	b := []byte(password)
	if len(b) > 72 {
		b = b[:72]
	}
	return b
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
	s.saveUserLocked(u)
	return u.Hero.Money, u.Hero.DiamondMoney, true
}

// HeroMoney returns a user's current persistent money/diamond totals WITHOUT
// changing them. Used by the Battle server to seed the in-battle balance (the
// initial SET_MONEY the item tree's affordability check reads). ok=false if the
// user or hero is missing.
func (s *Store) HeroMoney(userID int32) (money, diamonds int32, ok bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, found := s.usersByID[userID]
	if !found || u.Hero == nil {
		return 0, 0, false
	}
	return u.Hero.Money, u.Hero.DiamondMoney, true
}

// SpendHeroMoney atomically debits amount from a user's persistent hero money IF
// they can afford it, returning the new totals. ok=false with no change when the
// user/hero is missing, amount is negative, or the balance is insufficient --
// the caller (Battle BUY) uses this single call as the affordability check so a
// concurrent debit can't slip a purchase past a stale balance. Used to buy the
// in-battle avatar item tree.
func (s *Store) SpendHeroMoney(userID, amount int32) (money, diamonds int32, ok bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, found := s.usersByID[userID]
	if !found || u.Hero == nil {
		return 0, 0, false
	}
	if amount < 0 || u.Hero.Money < amount {
		return u.Hero.Money, u.Hero.DiamondMoney, false
	}
	u.Hero.Money -= amount
	s.saveUserLocked(u)
	return u.Hero.Money, u.Hero.DiamondMoney, true
}

// heroExpNextLevel is the persistent character-level curve: level N needs
// 100*N exp to advance to N+1. Unlike the ephemeral hunt-instance level (which
// scales in-battle power and resets every session), this is the account's
// permanent progression shown in the lobby (HeroGameInfo level/exp/next_exp).
func heroExpNextLevel(level int32) int32 { return 100 * level }

// AddHeroExp grants the account's persistent character experience and
// processes level-ups against heroExpNextLevel, saving the result. Only a
// FRACTION of in-hunt XP is meant to be passed in here (see the Battle
// server's heroExpShare) -- the hunt-instance level/XP itself is intentionally
// NOT persisted, per design: battle power scaling stays a per-session thing,
// while a slice of the XP earned still grows the permanent character.
func (s *Store) AddHeroExp(userID, delta int32) (level, exp, nextExp int32, ok bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, found := s.usersByID[userID]
	if !found || u.Hero == nil || delta <= 0 {
		if found && u.Hero != nil {
			return u.Hero.Level, u.Hero.Exp, u.Hero.NextExp, true
		}
		return 0, 0, 0, false
	}
	h := u.Hero
	h.Exp += delta
	for h.NextExp > 0 && h.Exp >= h.NextExp {
		h.Exp -= h.NextExp
		h.Level++
		h.NextExp = heroExpNextLevel(h.Level)
	}
	s.saveUserLocked(u)
	return h.Level, h.Exp, h.NextExp, true
}

// AddBagItem credits count of articleID to the hero's persistent consumable
// bag, merging into an existing stack of the same article, and saves.
func (s *Store) AddBagItem(userID, articleID, count int32) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, found := s.usersByID[userID]
	if !found || u.Hero == nil || count <= 0 {
		return false
	}
	h := u.Hero
	for i := range h.Bag {
		if h.Bag[i].ArticleID == articleID {
			h.Bag[i].Count += count
			s.saveUserLocked(u)
			return true
		}
	}
	h.Bag = append(h.Bag, BagItem{ArticleID: articleID, Count: count})
	s.saveUserLocked(u)
	return true
}

// RemoveBagItem debits count of articleID from the hero's bag (dropping the
// stack once it reaches zero), and saves. ok=false if the hero has no such
// stack (nothing to remove).
func (s *Store) RemoveBagItem(userID, articleID, count int32) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, found := s.usersByID[userID]
	if !found || u.Hero == nil {
		return false
	}
	h := u.Hero
	for i := range h.Bag {
		if h.Bag[i].ArticleID == articleID {
			h.Bag[i].Count -= count
			if h.Bag[i].Count <= 0 {
				h.Bag = append(h.Bag[:i], h.Bag[i+1:]...)
			}
			s.saveUserLocked(u)
			return true
		}
	}
	return false
}

// HeroBag returns a copy of the hero's persistent bag contents (nil if no
// account/hero). Used by the Ctrl-channel lobby bag screen and the Battle
// channel's hunt-entry inventory sync.
func (s *Store) HeroBag(userID int32) []BagItem {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, found := s.usersByID[userID]
	if !found || u.Hero == nil {
		return nil
	}
	out := make([]BagItem, len(u.Hero.Bag))
	copy(out, u.Hero.Bag)
	return out
}

// ByUsername resolves a display nick to its user (exact match, then a
// case-insensitive fallback). Used by private chat to route a message to the nick the
// sender typed in "[nick]". ok=false if no account carries that name.
func (s *Store) ByUsername(name string) (*User, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, u := range s.usersByID {
		if u.Username == name {
			return u, true
		}
	}
	for _, u := range s.usersByID {
		if strings.EqualFold(u.Username, name) {
			return u, true
		}
	}
	return nil, false
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
		Level:     1,
		Exp:       0,
		NextExp:   100,
	}
	// Starter wallet is admin-tunable (defaults 1000 bronze / 100 diamond).
	h.Money, h.DiamondMoney = gamedata.NewHeroWallet()
	u.Hero = h
	u.HasHero = true
	s.saveUserLocked(u)
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

// Save persists every account to the database (no-op when in-memory). Callers
// that mutate a User/Hero outside the Store's own methods (e.g. tests poking
// Hero.Money directly) should call it.
func (s *Store) Save() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.saveAllLocked()
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
