package gamedata

import "testing"

// TestCastleRegistry sanity-checks the seeded castle list the castle screens read.
func TestCastleRegistry(t *testing.T) {
	cs := Castles()
	if len(cs) == 0 {
		t.Fatal("castle registry is empty")
	}
	seen := map[int32]bool{}
	for _, c := range cs {
		if c.ID == 0 {
			t.Errorf("castle %q has id 0 (reserved)", c.Name)
		}
		if seen[c.ID] {
			t.Errorf("duplicate castle id %d", c.ID)
		}
		seen[c.ID] = true
		if c.Name == "" {
			t.Errorf("castle %d has no name", c.ID)
		}
		if c.LevelMin > c.LevelMax {
			t.Errorf("castle %d level range inverted: %d..%d", c.ID, c.LevelMin, c.LevelMax)
		}
		if c.FightersMin <= 0 {
			t.Errorf("castle %d fighters_min = %d, want > 0", c.ID, c.FightersMin)
		}
		got, ok := CastleByID(c.ID)
		if !ok || got.ID != c.ID {
			t.Errorf("CastleByID(%d) failed", c.ID)
		}
	}
	if _, ok := CastleByID(-1); ok {
		t.Error("CastleByID(-1) should not resolve")
	}
}
