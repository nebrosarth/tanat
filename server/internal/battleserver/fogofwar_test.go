package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

// TestMobFogOfWar: mobs are shown only while the player is within reveal range,
// hidden again past hide range, and toggle as the player moves.
func TestMobFogOfWar(t *testing.T) {
	s, c, _, sx, sy := newNavConn(t)
	hs := c.huntState
	now := float64(s.battleTime())
	mob := gamedata.Mobs()[2] // skeleton

	far := &mobState{id: 3000, mobIdx: 2, mob: mob, x: sx + float32(mobHideRadius) + 5, y: sy, hp: mob.Health, homed: true}
	near := &mobState{id: 3001, mobIdx: 2, mob: mob, x: sx + 5, y: sy, hp: mob.Health, homed: true}
	hs.mobs[far.id] = far
	hs.mobs[near.id] = near

	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	s.mobInterestLocked(c, far, now)
	s.mobInterestLocked(c, near, now)
	if far.shown {
		t.Fatal("distant mob should be hidden by fog of war")
	}
	if !near.shown {
		t.Fatal("nearby mob should be revealed")
	}

	// Approach the far mob -> it reveals.
	c.x, c.y, c.snapT = far.x-3, far.y, float32(now)
	s.mobInterestLocked(c, far, now)
	if !far.shown {
		t.Fatal("mob should reveal once the player approaches")
	}

	// Retreat from the near mob -> it hides again (and drops aggro).
	near.aggro = true
	c.x, c.y, c.snapT = near.x+float32(mobHideRadius)+5, near.y, float32(now)
	s.mobInterestLocked(c, near, now)
	if near.shown {
		t.Fatal("mob should hide once the player leaves")
	}
	if near.aggro {
		t.Fatal("hidden mob should drop aggro")
	}
}

// TestMobShadeFogRing: a revealed mob is rendered translucent (shade fx) while it
// sits in the outer reveal ring, snaps to full opacity as the player closes in,
// re-shades on retreat, and drops the shade when it hides entirely.
func TestMobShadeFogRing(t *testing.T) {
	s, c, _, sx, sy := newNavConn(t)
	hs := c.huntState
	now := float64(s.battleTime())
	mob := gamedata.Mobs()[2] // skeleton

	// Distance 26: within reveal (28) but out past the shade radius (24).
	m := &mobState{id: 4000, mobIdx: 2, mob: mob, x: sx + 26, y: sy, hp: mob.Health, homed: true}
	hs.mobs[m.id] = m

	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	// Appears shaded: revealed at 26 (< 28) but that's beyond the shade radius.
	s.mobInterestLocked(c, m, now)
	if !m.shown {
		t.Fatal("mob within reveal range should be shown")
	}
	if !m.shaded || m.shadeFxUID == 0 {
		t.Fatal("distant-but-visible mob should appear shaded (fog ring)")
	}

	// Close in past the unshade radius (22) -> full opacity.
	c.x, c.y, c.snapT = m.x-20, m.y, float32(now)
	s.mobInterestLocked(c, m, now)
	if m.shaded || m.shadeFxUID != 0 {
		t.Fatal("nearby mob should clear its shade")
	}

	// Retreat back into the ring -> shaded again.
	c.x, c.y, c.snapT = m.x-26, m.y, float32(now)
	s.mobInterestLocked(c, m, now)
	if !m.shaded {
		t.Fatal("mob should re-shade when the player retreats into the outer ring")
	}

	// Retreat past the hide radius -> hidden and un-shaded.
	c.x, c.y, c.snapT = m.x-float32(mobHideRadius)-5, m.y, float32(now)
	s.mobInterestLocked(c, m, now)
	if m.shown {
		t.Fatal("mob past hide radius should be hidden")
	}
	if m.shaded || m.shadeFxUID != 0 {
		t.Fatal("hidden mob should carry no shade state")
	}
}
