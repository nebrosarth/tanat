package session

import (
	"path/filepath"
	"testing"
)

// TestPersistenceRoundTrip verifies accounts and heroes survive a reload from
// the JSON file, and that HasHero is reconstructed.
func TestPersistenceRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "accounts.json")

	s1 := NewPersistentStore(path)
	u, _ := s1.LoginOrRegister("a@b.c", "pw")
	s1.CreateHero(u, 2 /*elf*/, false, 3, 4, 5, 6, 7)
	u.Hero.Money = 4242 // simulate a money change
	s1.Save()

	s2 := NewPersistentStore(path)
	got, ok := s2.usersByEmail["a@b.c"]
	if !ok {
		t.Fatal("account not loaded after reload")
	}
	if got.ID != u.ID {
		t.Errorf("reloaded id = %d, want %d", got.ID, u.ID)
	}
	if !got.HasHero || got.Hero == nil {
		t.Fatal("hero not loaded / HasHero not reconstructed")
	}
	if got.Hero.Race != 2 {
		t.Errorf("reloaded race = %d, want 2", got.Hero.Race)
	}
	if got.Hero.Money != 4242 {
		t.Errorf("reloaded money = %d, want 4242", got.Hero.Money)
	}
	// nextUserID must continue past the loaded users.
	u2, _ := s2.LoginOrRegister("b@b.c", "pw")
	if u2.ID <= got.ID {
		t.Errorf("new user id %d not greater than loaded %d", u2.ID, got.ID)
	}
}

// TestInMemoryStoreDoesNotPersist confirms NewStore() (path "") writes nothing.
func TestInMemoryStoreDoesNotPersist(t *testing.T) {
	s := NewStore()
	u, _ := s.LoginOrRegister("x@y.z", "pw")
	s.CreateHero(u, 1, true, 0, 0, 0, 0, 0)
	s.Save() // must be a no-op, not panic
}

// TestHeroCustomizationClamped: the client sends -1 for untouched customize
// sliders, but crashes (HeroMgr.GetCustomizeColor list[-1]) if the server
// serves them back. Both creation and load must clamp to 0.
func TestHeroCustomizationClamped(t *testing.T) {
	path := filepath.Join(t.TempDir(), "accounts.json")
	s1 := NewPersistentStore(path)
	u, _ := s1.LoginOrRegister("a@b.c", "pw")
	h := s1.CreateHero(u, 2, true, -1, -1, -1, -1, -1)
	if h.Face != 0 || h.Hair != 0 || h.SkinColor != 0 || h.HairColor != 0 || h.DistMark != 0 {
		t.Errorf("CreateHero did not clamp -1 values: %+v", h)
	}

	// Simulate a legacy file with -1 values already stored.
	h.SkinColor = -1
	h.HairColor = -1
	s1.Save()
	s2 := NewPersistentStore(path)
	got := s2.usersByEmail["a@b.c"]
	if got.Hero.SkinColor != 0 || got.Hero.HairColor != 0 {
		t.Errorf("load did not sanitize legacy -1 values: %+v", got.Hero)
	}
}
