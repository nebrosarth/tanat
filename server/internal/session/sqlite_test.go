package session

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"tanatserver/internal/gamedata"
)

// TestAuthRejectsWrongPassword: once an account exists, a mismatched password is
// refused (ok=false, no session), while the correct one still logs in. The very
// first login for an email registers it and sets that password.
func TestAuthRejectsWrongPassword(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.db")
	s := NewPersistentStore(path)
	defer s.Close()

	u1, key1, ok := s.LoginOrRegister("hero@test.io", "correct-horse")
	if !ok || u1 == nil || key1 == "" {
		t.Fatal("first login should register and succeed")
	}

	if _, _, ok := s.LoginOrRegister("hero@test.io", "wrong"); ok {
		t.Error("login with wrong password must be rejected")
	}

	u2, key2, ok := s.LoginOrRegister("hero@test.io", "correct-horse")
	if !ok || u2.ID != u1.ID || key2 == "" {
		t.Errorf("login with correct password must succeed for the same account (ok=%v id=%d)", ok, u2.ID)
	}
}

// TestPasswordStoredHashed: the DB never holds the plaintext password, and the
// stored hash verifies with bcrypt.
func TestPasswordStoredHashed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.db")
	s := NewPersistentStore(path)
	defer s.Close()
	s.LoginOrRegister("hero@test.io", "s3cr3t-plaintext")

	var hash string
	if err := s.db.QueryRow(`SELECT password_hash FROM users WHERE email=?`, "hero@test.io").Scan(&hash); err != nil {
		t.Fatalf("query hash: %v", err)
	}
	if hash == "" || hash == "s3cr3t-plaintext" {
		t.Fatalf("password not hashed in DB: %q", hash)
	}
	// A reloaded store still verifies the same password.
	s2 := NewPersistentStore(path)
	defer s2.Close()
	if _, _, ok := s2.LoginOrRegister("hero@test.io", "s3cr3t-plaintext"); !ok {
		t.Error("reloaded account should verify its original password")
	}
	if _, _, ok := s2.LoginOrRegister("hero@test.io", "nope"); ok {
		t.Error("reloaded account should reject a wrong password")
	}
}

// TestFullAggregateRoundTrip exercises every child table (bag, owned, dressed,
// quests, friends, ignores) across a reload -- the load path had a dispatch bug
// that only a child-table round-trip catches.
func TestFullAggregateRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.db")
	s1 := NewPersistentStore(path)
	defer s1.Close()

	ua, _, _ := s1.LoginOrRegister("a@test.io", "pw")
	ub, _, _ := s1.LoginOrRegister("b@test.io", "pw")
	s1.CreateHero(ua, 1, true, 1, 2, 3, 4, 5)
	s1.CreateHero(ub, 2, false, 0, 0, 0, 0, 0)

	// Bag (two stacks), owned + dressed gear, a quest, and social lists.
	s1.AddBagItem(ua.ID, 5000, 4)
	s1.AddBagItem(ua.ID, 5008, 2)
	_, _, added, ok := s1.BuyWearables(ua.ID, 0, []int32{7001, 7002})
	if !ok || len(added) != 2 {
		t.Fatalf("BuyWearables failed: %+v ok=%v", added, ok)
	}
	if _, ok := s1.DressWearable(ua.ID, added[0].ID, 3); !ok {
		t.Fatal("DressWearable failed")
	}
	// A quest in progress.
	var qid int32
	for _, q := range gamedata.Quests() {
		qid = q.ID
		break
	}
	if qid != 0 {
		if _, ok := s1.AcceptQuest(ua.ID, qid); !ok {
			t.Logf("AcceptQuest(%d) did not start (ok=false) -- quest coverage skipped", qid)
			qid = 0
		}
	}
	// Mutual friendship + a one-way ignore.
	if now, ok := s1.AddFriendRequest(ua.ID, ub.ID); !ok || now {
		t.Fatalf("AddFriendRequest a->b: now=%v ok=%v", now, ok)
	}
	if now, ok := s1.AddFriendRequest(ub.ID, ua.ID); !ok || !now {
		t.Fatalf("AddFriendRequest b->a should complete the pair: now=%v ok=%v", now, ok)
	}
	s1.AddIgnore(ua.ID, 999)

	// Reload and verify everything survived.
	s2 := NewPersistentStore(path)
	defer s2.Close()
	got := s2.usersByEmail["a@test.io"]
	if got == nil || got.Hero == nil {
		t.Fatal("account a not reloaded")
	}
	h := got.Hero
	if h.Face != 1 || h.Hair != 2 || h.DistMark != 3 || h.SkinColor != 4 || h.HairColor != 5 {
		t.Errorf("hero customization not restored: %+v", h)
	}
	if len(h.Bag) != 2 {
		t.Errorf("bag = %+v, want 2 stacks", h.Bag)
	}
	// One wearable dressed (slot 3), one still owned.
	if len(h.Dressed) != 1 || h.Dressed[0].Slot != 3 {
		t.Errorf("dressed = %+v, want 1 item in slot 3", h.Dressed)
	}
	if len(h.Owned) != 1 {
		t.Errorf("owned = %+v, want 1 leftover", h.Owned)
	}
	// NextItemID must persist so future instance ids don't collide.
	if h.NextItemID <= heroItemInstanceBase {
		t.Errorf("NextItemID = %d, want > %d (mint counter not persisted)", h.NextItemID, heroItemInstanceBase)
	}
	if qid != 0 && len(h.Quests) != 1 {
		t.Errorf("quests = %+v, want 1", h.Quests)
	}
	if !s2.AreFriends(ua.ID, ub.ID) || !s2.AreFriends(ub.ID, ua.ID) {
		t.Error("mutual friendship did not survive reload")
	}
	if ign := s2.IgnoreIDs(ua.ID); len(ign) != 1 || ign[0] != 999 {
		t.Errorf("ignores = %v, want [999]", ign)
	}
}

// TestLegacyJSONMigration: a fresh DB next to an old accounts.json inherits its
// accounts (passwords hashed), then archives the JSON so it never re-imports.
func TestLegacyJSONMigration(t *testing.T) {
	dir := t.TempDir()
	// Write a legacy accounts.json in the exact old on-disk shape.
	legacy := legacySnapshot{
		NextUserID: 3,
		Users: []legacyUser{
			{
				ID: 1, Email: "old@test.io", Password: "legacypw", Username: "OldTimer", HasHero: true,
				Hero:    &Hero{ID: 1, Race: 1, Money: 7777, DiamondMoney: 50, Level: 4, Exp: 10, NextExp: 400, Bag: []BagItem{{ArticleID: 5000, Count: 9}}},
				Friends: []int32{2},
			},
			{ID: 2, Email: "friend@test.io", Password: "pw2", Username: "Buddy", Friends: []int32{1}},
		},
	}
	b, _ := json.MarshalIndent(legacy, "", "  ")
	if err := os.WriteFile(filepath.Join(dir, "accounts.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}

	dbPath := filepath.Join(dir, "tanat.db")
	s := NewPersistentStore(dbPath)
	defer s.Close()

	old := s.usersByEmail["old@test.io"]
	if old == nil || old.Hero == nil {
		t.Fatal("legacy account not imported")
	}
	if old.Hero.Money != 7777 || old.Hero.Level != 4 {
		t.Errorf("imported hero wrong: money=%d level=%d", old.Hero.Money, old.Hero.Level)
	}
	if len(old.Hero.Bag) != 1 || old.Hero.Bag[0].ArticleID != 5000 || old.Hero.Bag[0].Count != 9 {
		t.Errorf("imported bag wrong: %+v", old.Hero.Bag)
	}
	if !s.AreFriends(1, 2) {
		t.Error("imported friendship missing")
	}
	// Plaintext password became a verifying bcrypt hash.
	if old.PassHash == "" || old.PassHash == "legacypw" {
		t.Errorf("legacy password not hashed: %q", old.PassHash)
	}
	if _, _, ok := s.LoginOrRegister("old@test.io", "legacypw"); !ok {
		t.Error("migrated account should log in with its legacy password")
	}
	// nextUserID advanced past the imported ids.
	if s.nextUserID < 3 {
		t.Errorf("nextUserID = %d, want >= 3", s.nextUserID)
	}
	// The JSON is archived, not left to re-import.
	if _, err := os.Stat(filepath.Join(dir, "accounts.json")); !os.IsNotExist(err) {
		t.Error("accounts.json should have been renamed after import")
	}
	if _, err := os.Stat(filepath.Join(dir, "accounts.json.imported")); err != nil {
		t.Error("accounts.json.imported archive missing")
	}

	// Reopening must NOT re-import (JSON is gone) and must keep the accounts.
	s.Close()
	s2 := NewPersistentStore(dbPath)
	defer s2.Close()
	if s2.usersByEmail["old@test.io"] == nil {
		t.Error("accounts lost on second open")
	}
}

// TestLongPasswordFailsClosed: a >72-byte password (e.g. a long Cyrillic one)
// must still hash to a real, verifying hash -- NOT an empty hash that would let
// any password in. And an empty stored hash must fail closed.
func TestLongPasswordFailsClosed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.db")
	s := NewPersistentStore(path)
	defer s.Close()

	long := strings.Repeat("пароль", 20) // ~240 bytes UTF-8, well over bcrypt's 72
	if _, _, ok := s.LoginOrRegister("cyr@test.io", long); !ok {
		t.Fatal("registration with a long password should succeed")
	}
	// The stored hash must not be empty (empty == fail-open before the fix).
	var hash string
	if err := s.db.QueryRow(`SELECT password_hash FROM users WHERE email=?`, "cyr@test.io").Scan(&hash); err != nil {
		t.Fatalf("query hash: %v", err)
	}
	if hash == "" {
		t.Fatal("long-password account stored an EMPTY hash (fail-open account takeover risk)")
	}
	// A different password must be rejected (this is the takeover the fix closes).
	if _, _, ok := s.LoginOrRegister("cyr@test.io", "totally-different-password"); ok {
		t.Error("a long-password account must not accept an arbitrary password")
	}
	// The correct long password still verifies.
	if _, _, ok := s.LoginOrRegister("cyr@test.io", long); !ok {
		t.Error("correct long password should verify")
	}
	// An empty stored hash fails closed.
	if (&User{PassHash: ""}).checkPassword("anything") {
		t.Error("an empty stored hash must fail closed, not accept any password")
	}
}

// TestNextUserIDSurvivesGap: an imported legacy NextUserID ahead of max(id)+1
// (a gap) must survive a restart, so ids in the gap are never re-handed out.
func TestNextUserIDSurvivesGap(t *testing.T) {
	dir := t.TempDir()
	legacy := legacySnapshot{
		NextUserID: 50, // ahead of max id (10) -> a gap that must not be reused
		Users:      []legacyUser{{ID: 10, Email: "x@test.io", Password: "pw", Username: "x"}},
	}
	b, _ := json.Marshal(legacy)
	os.WriteFile(filepath.Join(dir, "accounts.json"), b, 0o644)

	dbPath := filepath.Join(dir, "tanat.db")
	s := NewPersistentStore(dbPath)
	if s.nextUserID < 50 {
		t.Fatalf("after import nextUserID = %d, want >= 50", s.nextUserID)
	}
	s.Close()

	// Reopen: the high-water mark must be recovered from meta, not recomputed to 11.
	s2 := NewPersistentStore(dbPath)
	defer s2.Close()
	if s2.nextUserID < 50 {
		t.Errorf("after reopen nextUserID = %d, want >= 50 (gap not persisted)", s2.nextUserID)
	}
	u, _, _ := s2.LoginOrRegister("new@test.io", "pw")
	if u.ID < 50 {
		t.Errorf("new account got id %d, reusing a gap id (< 50)", u.ID)
	}
}

// TestMigrationAtomicOnBadData: a constraint-violating legacy record (two users
// sharing an email) must roll the WHOLE import back -- no partial accounts, and
// accounts.json preserved (not archived) so it can be retried.
func TestMigrationAtomicOnBadData(t *testing.T) {
	dir := t.TempDir()
	legacy := legacySnapshot{
		NextUserID: 3,
		Users: []legacyUser{
			{ID: 1, Email: "dup@test.io", Password: "pw", Username: "a"},
			{ID: 2, Email: "dup@test.io", Password: "pw", Username: "b"}, // UNIQUE(email) violation
		},
	}
	b, _ := json.Marshal(legacy)
	os.WriteFile(filepath.Join(dir, "accounts.json"), b, 0o644)

	dbPath := filepath.Join(dir, "tanat.db")
	s := NewPersistentStore(dbPath)
	defer s.Close()

	if len(s.usersByID) != 0 {
		t.Errorf("failed import left %d partial account(s) in memory, want 0", len(s.usersByID))
	}
	var n int
	s.db.QueryRow(`SELECT count(*) FROM users`).Scan(&n)
	if n != 0 {
		t.Errorf("failed import left %d partial row(s) in DB, want 0 (not atomic)", n)
	}
	if _, err := os.Stat(filepath.Join(dir, "accounts.json")); err != nil {
		t.Error("accounts.json must be preserved when import fails (so it can retry)")
	}
	if _, err := os.Stat(filepath.Join(dir, "accounts.json.imported")); !os.IsNotExist(err) {
		t.Error("a failed import must not archive the json")
	}
}
