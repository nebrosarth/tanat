package session

import "testing"

// TestCreateHeroDefaultRating: a freshly created hero starts at RatingDefault, not 0 --
// zero would read as "unrated"/lowest-possible on the client's profile and roster
// screens, which is wrong for a brand new account.
func TestCreateHeroDefaultRating(t *testing.T) {
	s := NewStore()
	u, _, _ := s.LoginOrRegister("newbie@test.io", "pw")
	s.CreateHero(u, 0, false, 0, 0, 0, 0, 0)
	if u.Hero.Rating != RatingDefault {
		t.Errorf("new hero rating = %d, want %d", u.Hero.Rating, RatingDefault)
	}
}

// TestApplyHeroRatingDelta covers the settlement path battleserver's rating.go drives:
// old/new totals, clamping at 0, and failing safe for an unknown user.
func TestApplyHeroRatingDelta(t *testing.T) {
	s := NewStore()
	u, _, _ := s.LoginOrRegister("duelist@test.io", "pw")
	s.CreateHero(u, 0, false, 0, 0, 0, 0, 0)

	old, cur, ok := s.ApplyHeroRatingDelta(u.ID, 25)
	if !ok {
		t.Fatal("ApplyHeroRatingDelta returned ok=false for a valid hero")
	}
	if old != RatingDefault {
		t.Errorf("old rating = %d, want %d", old, RatingDefault)
	}
	if cur != RatingDefault+25 {
		t.Errorf("new rating = %d, want %d", cur, RatingDefault+25)
	}
	if u.Hero.Rating != RatingDefault+25 {
		t.Errorf("hero.Rating not persisted: %d", u.Hero.Rating)
	}
	if r, ok := s.HeroRating(u.ID); !ok || r != RatingDefault+25 {
		t.Errorf("HeroRating = %d,%v want %d,true", r, ok, RatingDefault+25)
	}

	// A big enough loss clamps at 0, never negative.
	if _, cur, _ := s.ApplyHeroRatingDelta(u.ID, -1_000_000); cur != 0 {
		t.Errorf("rating should clamp to 0, got %d", cur)
	}

	// Unknown user -> ok=false, no panic.
	if _, _, ok := s.ApplyHeroRatingDelta(9999, 10); ok {
		t.Error("ApplyHeroRatingDelta should return ok=false for an unknown user")
	}
}

// TestFightLogRoundTrip: SetFightLog/FightLog is a plain per-battleID cache -- what goes
// in for one id comes back out unchanged, and a never-set id reports ok=false rather than
// an empty-but-present map (the caller needs to tell "no data yet" from "empty match").
func TestFightLogRoundTrip(t *testing.T) {
	s := NewStore()
	entries := map[int32]FightLogEntry{
		1: {AvatarID: 1, Nick: "Alice", Team: 1, Kills: 3, Deaths: 1, NewRating: 1032, OldRating: 1000},
		2: {AvatarID: 2, Nick: "Bob", Team: 2, Kills: 1, Deaths: 3, NewRating: 968, OldRating: 1000},
	}
	s.SetFightLog(555, entries)

	got, ok := s.FightLog(555)
	if !ok {
		t.Fatal("FightLog(555) ok=false after SetFightLog")
	}
	if len(got) != 2 || got[1].Nick != "Alice" || got[2].Kills != 1 {
		t.Errorf("FightLog(555) = %+v, want the entries set", got)
	}

	if _, ok := s.FightLog(999); ok {
		t.Error("FightLog for an id that was never set should report ok=false")
	}
}
