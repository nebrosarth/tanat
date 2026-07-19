package session

import (
	"database/sql"
	"path/filepath"
	"testing"

	_ "modernc.org/sqlite"
)

// withHero registers an account and gives it a hero, returning its id.
func withHero(t *testing.T, s *Store, email string) int32 {
	t.Helper()
	u, _, ok := s.LoginOrRegister(email, "pw")
	if !ok {
		t.Fatalf("register %s failed", email)
	}
	s.CreateHero(u, 1, false, 0, 0, 0, 0, 0)
	return u.ID
}

func TestAdminSetMoneyAndProgress(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.db")
	s := NewPersistentStore(path)
	defer s.Close()

	id := withHero(t, s, "a@x")
	if !s.SetHeroMoney(id, 5000, 42) {
		t.Fatal("SetHeroMoney failed")
	}
	if !s.SetHeroProgress(id, 12, 340, 1000) {
		t.Fatal("SetHeroProgress failed")
	}
	// Negatives clamp to zero.
	s.SetHeroMoney(id, -1, -1)
	if m, d, _ := s.HeroMoney(id); m != 0 || d != 0 {
		t.Errorf("negative money not clamped: %d,%d", m, d)
	}

	// Persisted: reopen and confirm.
	s.SetHeroMoney(id, 5000, 42)
	s.Close()
	s2 := NewPersistentStore(path)
	defer s2.Close()
	players := s2.ListPlayers()
	if len(players) != 1 {
		t.Fatalf("want 1 player, got %d", len(players))
	}
	p := players[0]
	if p.Money != 5000 || p.Diamonds != 42 || p.Level != 12 || p.Exp != 340 {
		t.Errorf("progress not persisted: %+v", p)
	}
}

func TestAdminQuestStateAndGrant(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.db")
	s := NewPersistentStore(path)
	defer s.Close()

	id := withHero(t, s, "q@x")

	// Upsert a quest to an explicit in-progress state, then bump it.
	if !s.AdminSetQuestState(id, 501, 1 /*in-progress*/, 3, 0) {
		t.Fatal("AdminSetQuestState (insert) failed")
	}
	if !s.AdminSetQuestState(id, 501, 2 /*done*/, 10, 0) {
		t.Fatal("AdminSetQuestState (update) failed")
	}
	qs, ok := s.AdminHeroQuests(id)
	if !ok || len(qs) != 1 || qs[0].QuestID != 501 || qs[0].Status != 2 || qs[0].Progress != 10 {
		t.Fatalf("quest state wrong: ok=%v %+v", ok, qs)
	}
	// Negative progress floors to 0.
	s.AdminSetQuestState(id, 501, 1, -5, 0)
	if qs, _ := s.AdminHeroQuests(id); qs[0].Progress != 0 {
		t.Errorf("negative progress not floored: %d", qs[0].Progress)
	}

	// Grant three wearable instances (no charge) -- distinct stable ids.
	added, ok := s.AdminGrantWearable(id, 80001, 3)
	if !ok || len(added) != 3 {
		t.Fatalf("AdminGrantWearable failed: ok=%v n=%d", ok, len(added))
	}
	if added[0].ID == added[1].ID || added[1].ID == added[2].ID {
		t.Errorf("granted instances share ids: %+v", added)
	}
	if m, _, _ := s.HeroMoney(id); m == 0 {
		t.Error("grant should not have zeroed money")
	}

	// Persist across reopen: quest + owned survive.
	s.Close()
	s2 := NewPersistentStore(path)
	defer s2.Close()
	if qs, _ := s2.AdminHeroQuests(id); len(qs) != 1 || qs[0].QuestID != 501 {
		t.Errorf("quest not persisted: %+v", qs)
	}
	if owned := s2.HeroOwned(id); len(owned) != 3 {
		t.Errorf("granted wearables not persisted: %d", len(owned))
	}

	// Unconditional remove.
	if !s2.AdminRemoveQuest(id, 501) {
		t.Fatal("AdminRemoveQuest failed")
	}
	if qs, _ := s2.AdminHeroQuests(id); len(qs) != 0 {
		t.Errorf("quest not removed: %+v", qs)
	}
	if s2.AdminRemoveQuest(id, 501) {
		t.Error("removing an absent quest should return false")
	}
}

func TestAdminBanPersists(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.db")
	s := NewPersistentStore(path)
	defer s.Close()

	id := withHero(t, s, "b@x")
	if s.IsBanned(id) {
		t.Fatal("should start unbanned")
	}
	if !s.SetBanned(id, true) {
		t.Fatal("SetBanned failed")
	}
	if !s.IsBanned(id) {
		t.Fatal("IsBanned should be true")
	}
	s.Close()

	s2 := NewPersistentStore(path)
	defer s2.Close()
	if !s2.IsBanned(id) {
		t.Fatal("ban did not persist across reopen")
	}
}

func TestAdminDeleteAccount(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.db")
	s := NewPersistentStore(path)
	defer s.Close()

	keep := withHero(t, s, "keep@x")
	del := withHero(t, s, "del@x")
	// Make them mutual friends so the delete must scrub the survivor's list.
	if u := s.usersByID[keep]; u != nil {
		u.Friends = append(u.Friends, del)
	}
	if u := s.usersByID[del]; u != nil {
		u.Friends = append(u.Friends, keep)
	}

	if !s.DeleteAccount(del) {
		t.Fatal("DeleteAccount failed")
	}
	if _, ok := s.ByID(del); ok {
		t.Fatal("deleted account still present")
	}
	if u, _ := s.ByID(keep); u != nil {
		for _, f := range u.Friends {
			if f == del {
				t.Fatal("deleted id not scrubbed from survivor friend list")
			}
		}
	}
	// Persisted: gone after reopen, survivor remains.
	s.Close()
	s2 := NewPersistentStore(path)
	defer s2.Close()
	if _, ok := s2.ByID(del); ok {
		t.Fatal("delete did not persist")
	}
	if _, ok := s2.ByID(keep); !ok {
		t.Fatal("survivor lost")
	}
}

func TestAdminMetaRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.db")
	s := NewPersistentStore(path)
	defer s.Close()

	if _, ok := s.GetMeta("nope"); ok {
		t.Fatal("missing key should not be ok")
	}
	if err := s.SetMeta("k", `{"x":1}`); err != nil {
		t.Fatal(err)
	}
	v, ok := s.GetMeta("k")
	if !ok || v != `{"x":1}` {
		t.Fatalf("GetMeta = %q,%v", v, ok)
	}
	// Survives reopen.
	s.Close()
	s2 := NewPersistentStore(path)
	defer s2.Close()
	if v, ok := s2.GetMeta("k"); !ok || v != `{"x":1}` {
		t.Fatalf("meta not persisted: %q,%v", v, ok)
	}
}

// TestSchemaMigrationAddsBannedColumn exercises the on-boot migration against a
// database created by an OLDER build (whose users table lacks the banned column,
// exactly like the already-deployed tanat.db). openDB must add the column so load
// and SetBanned work.
func TestSchemaMigrationAddsBannedColumn(t *testing.T) {
	path := filepath.Join(t.TempDir(), "legacy.db")

	// Build a legacy DB by hand: the pre-banned users table + one row.
	raw, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := raw.Exec(`CREATE TABLE users (
		id INTEGER PRIMARY KEY, email TEXT NOT NULL UNIQUE,
		password_hash TEXT NOT NULL DEFAULT '', username TEXT NOT NULL DEFAULT '',
		created_at INTEGER NOT NULL DEFAULT 0)`); err != nil {
		t.Fatal(err)
	}
	if _, err := raw.Exec(`INSERT INTO users(id,email,username) VALUES(1,'old@x','old')`); err != nil {
		t.Fatal(err)
	}
	raw.Close()

	// Opening through the store must migrate (add banned) and load cleanly.
	s := NewPersistentStore(path)
	defer s.Close()
	if _, ok := s.ByID(1); !ok {
		t.Fatal("legacy account did not load after migration")
	}
	if s.IsBanned(1) {
		t.Fatal("migrated account should default to unbanned")
	}
	if !s.SetBanned(1, true) {
		t.Fatal("SetBanned failed on migrated schema")
	}
	s.Close()
	s2 := NewPersistentStore(path)
	defer s2.Close()
	if !s2.IsBanned(1) {
		t.Fatal("ban on migrated schema did not persist")
	}
}
