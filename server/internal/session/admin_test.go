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

func TestGlobalBuffAddRefreshPersist(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.db")
	s := NewPersistentStore(path)
	defer s.Close()

	id := withHero(t, s, "buff@x")
	// Add a buff, then refresh the SAME article -> one entry, later expiry (no stacking).
	e1, ok := s.AddGlobalBuff(id, 70000, 3600)
	if !ok {
		t.Fatal("AddGlobalBuff failed")
	}
	e2, _ := s.AddGlobalBuff(id, 70000, 7200)
	if e2 <= e1 {
		t.Errorf("refresh did not extend expiry: %d -> %d", e1, e2)
	}
	if b := s.HeroActiveBuffs(id); len(b) != 1 {
		t.Fatalf("want 1 buff after refresh, got %d", len(b))
	}
	// A second distinct article -> two active buffs.
	s.AddGlobalBuff(id, 71000, 3600)
	if b := s.HeroActiveBuffs(id); len(b) != 2 {
		t.Fatalf("want 2 buffs, got %d", len(b))
	}
	// Non-positive duration is rejected.
	if _, ok := s.AddGlobalBuff(id, 72000, 0); ok {
		t.Error("zero-duration buff should be rejected")
	}

	// Persist across reopen.
	s.Close()
	s2 := NewPersistentStore(path)
	defer s2.Close()
	if b := s2.HeroActiveBuffs(id); len(b) != 2 {
		t.Errorf("buffs not persisted: %d", len(b))
	}
}

func TestAdminRemoveItems(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.db")
	s := NewPersistentStore(path)
	defer s.Close()

	id := withHero(t, s, "rm@x")

	// --- bag stack: partial then full removal ---
	s.AddBagItem(id, 70000, 5)
	if rem, ok := s.AdminRemoveBagArticle(id, 70000, 2); !ok || rem != 3 {
		t.Fatalf("partial bag remove: rem=%d ok=%v", rem, ok)
	}
	if rem, ok := s.AdminRemoveBagArticle(id, 70000, 0); !ok || rem != 0 {
		t.Fatalf("full bag remove (count 0): rem=%d ok=%v", rem, ok)
	}
	if len(s.HeroBag(id)) != 0 {
		t.Fatalf("bag not empty after removal: %+v", s.HeroBag(id))
	}
	if _, ok := s.AdminRemoveBagArticle(id, 70000, 1); ok {
		t.Error("removing an absent bag stack should be ok=false")
	}

	// --- owned wearable instance ---
	owned, _ := s.AdminGrantWearable(id, 80001, 2)
	if !s.AdminRemoveOwned(id, owned[0].ID) {
		t.Fatal("AdminRemoveOwned failed")
	}
	if o := s.HeroOwned(id); len(o) != 1 || o[0].ID != owned[1].ID {
		t.Fatalf("wrong owned removed: %+v", o)
	}
	if s.AdminRemoveOwned(id, 999999) {
		t.Error("removing an absent instance should be false")
	}

	// --- dressed item: dress the survivor, then destroy it (not undress-to-owned) ---
	if _, ok := s.DressWearable(id, owned[1].ID, 1); !ok {
		t.Fatal("DressWearable failed")
	}
	if !s.AdminRemoveDressed(id, 1) {
		t.Fatal("AdminRemoveDressed failed")
	}
	if len(s.HeroDressed(id)) != 0 {
		t.Fatalf("dressed not removed: %+v", s.HeroDressed(id))
	}
	if len(s.HeroOwned(id)) != 0 {
		t.Errorf("destroyed dressed item leaked back into owned: %+v", s.HeroOwned(id))
	}
	if s.AdminRemoveDressed(id, 1) {
		t.Error("removing from an empty slot should be false")
	}

	// --- persistence: all gone after reopen ---
	s.Close()
	s2 := NewPersistentStore(path)
	defer s2.Close()
	if len(s2.HeroBag(id)) != 0 || len(s2.HeroOwned(id)) != 0 || len(s2.HeroDressed(id)) != 0 {
		t.Errorf("removals not persisted: bag=%d owned=%d dressed=%d",
			len(s2.HeroBag(id)), len(s2.HeroOwned(id)), len(s2.HeroDressed(id)))
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
