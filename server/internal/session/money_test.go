package session

import "testing"

// TestAddHeroMoney covers the mob-kill coin-bounty credit path: it adds/clamps
// hero money and reports the new totals, and fails safe for unknown users.
func TestAddHeroMoney(t *testing.T) {
	s := NewStore()
	u, _ := s.LoginOrRegister("hunter@test.io", "pw")
	s.CreateHero(u, 0, false, 0, 0, 0, 0, 0)
	start := u.Hero.Money

	money, diamonds, ok := s.AddHeroMoney(u.ID, 6)
	if !ok {
		t.Fatal("AddHeroMoney returned ok=false for a valid hero")
	}
	if money != start+6 {
		t.Errorf("money = %d, want %d", money, start+6)
	}
	if diamonds != u.Hero.DiamondMoney {
		t.Errorf("diamonds = %d, want %d", diamonds, u.Hero.DiamondMoney)
	}
	if u.Hero.Money != start+6 {
		t.Errorf("hero.Money not persisted: %d", u.Hero.Money)
	}

	// Debits clamp at zero, never negative.
	if m, _, _ := s.AddHeroMoney(u.ID, -1_000_000); m != 0 {
		t.Errorf("money should clamp to 0, got %d", m)
	}

	// Unknown user -> ok=false, no panic.
	if _, _, ok := s.AddHeroMoney(9999, 5); ok {
		t.Error("AddHeroMoney should return ok=false for an unknown user")
	}
}
