package battleserver

import (
	"math"
	"testing"

	"tanatserver/internal/gamedata"
)

// TestMobLeashWalksHomeAndHeals: a wounded mob whose target has run past the leash
// range gives up, walks back to its spawn point, and regenerates to full HP along
// the way (classic MMO reset). The mob stays within the fog-hide radius throughout,
// so this exercises the visible walk-home path rather than the fog snap-reset.
func TestMobLeashWalksHomeAndHeals(t *testing.T) {
	s, c, nav, sx, sy := newNavConn(t)
	idx := mobIndexByPrefab(t, "Mob_Skeleton_1H_Melee_01")
	mob := gamedata.Mobs()[idx]

	// Home = spawn. Find a walkable point ~4u off spawn on a clear straight line
	// (this map has walls, so probe rather than assume a direction). The mob starts
	// there, dragged off its home; walking back is a short clear path.
	var dx, dy float32
	found := false
	for a := 0; a < 16 && !found; a++ {
		ang := float64(a) * math.Pi / 8
		tx := sx + 4*float32(math.Cos(ang))
		ty := sy + 4*float32(math.Sin(ang))
		cx, cy := nav.Clip(float64(sx), float64(sy), float64(tx), float64(ty))
		if math.Hypot(cx-float64(tx), cy-float64(ty)) < 0.1 {
			dx, dy, found = tx-sx, ty-sy, true
		}
	}
	if !found {
		t.Skip("no clear-line spot near spawn on this map")
	}
	// Unit vector spawn->mob, so the player can be placed on the far side.
	ux, uy := dx/4, dy/4
	m := &mobState{
		id: 2200, mobIdx: idx, mob: mob,
		x: sx + dx, y: sy + dy, spawnX: sx, spawnY: sy,
		hp: mob.Health * 0.3, shown: true, aggro: true,
	}

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	// Player 27u from spawn on the OPPOSITE side of the mob: ~31u from the mob
	// (>leash 22) yet <hide 34, and >aggro 9 for the whole walk home. Player coords
	// need not be walkable -- only the mob's motion is nav-clamped.
	c.x, c.y, c.snapT = sx-27*ux, sy-27*uy, float32(0)
	c.huntState.mobs[m.id] = m
	c.huntState.tr.add(m.id)

	// First tick: target is 24u away (> leash) -> mob gives up and enters returning.
	s.tickMobsLocked(c, 0.2)
	if !m.returning {
		t.Fatalf("mob past leash range should start returning home (aggro=%v)", m.aggro)
	}
	if m.aggro {
		t.Fatal("a returning mob should have dropped aggro")
	}

	// Drive the walk home and let the arrival top-off finish the heal.
	for i := 2; i < 80 && m.returning; i++ {
		s.tickMobsLocked(c, float64(i)*0.2)
	}

	if m.returning {
		t.Fatal("mob never finished returning home")
	}
	if got, want := m.hp, m.maxHealth(); got != want {
		t.Errorf("returned-home HP = %g, want full %g", got, want)
	}
	if math.Hypot(float64(m.x-m.spawnX), float64(m.y-m.spawnY)) > mobHomeEpsilon {
		t.Errorf("mob stopped at (%.2f,%.2f), want spawn (%.2f,%.2f)", m.x, m.y, m.spawnX, m.spawnY)
	}
	if m.vx != 0 || m.vy != 0 {
		t.Errorf("home mob should be stopped, got v=(%.2f,%.2f)", m.vx, m.vy)
	}
}

// TestMobLeashIgnoresPullUntilHome: a mob walking home IGNORES a player who chases it
// back into aggro range (WoW-style evade) -- it must not re-aggro mid-retreat -- and
// re-engages FRESH only once it has reset at spawn. A hit during the walk likewise
// deals damage but does not turn it around.
func TestMobLeashIgnoresPullUntilHome(t *testing.T) {
	s, c, nav, sx, sy := newNavConn(t)
	idx := mobIndexByPrefab(t, "Mob_Skeleton_1H_Melee_01")
	mob := gamedata.Mobs()[idx]

	// A clear straight line off spawn so the walk home isn't wall-blocked.
	var ux, uy float32
	found := false
	for a := 0; a < 16 && !found; a++ {
		ang := float64(a) * math.Pi / 8
		tx := sx + 6*float32(math.Cos(ang))
		ty := sy + 6*float32(math.Sin(ang))
		cx, cy := nav.Clip(float64(sx), float64(sy), float64(tx), float64(ty))
		if math.Hypot(cx-float64(tx), cy-float64(ty)) < 0.1 {
			ux, uy, found = float32(math.Cos(ang)), float32(math.Sin(ang)), true
		}
	}
	if !found {
		t.Skip("no clear-line spot near spawn on this map")
	}

	m := &mobState{
		id: 2201, mobIdx: idx, mob: mob,
		x: sx + 5*ux, y: sy + 5*uy, spawnX: sx, spawnY: sy,
		hp: mob.Health * 0.5, shown: true, aggro: true,
	}
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	c.huntState.mobs[m.id] = m
	c.huntState.tr.add(m.id)

	// Player far past the leash on the far side -> the mob gives up and starts home.
	c.x, c.y, c.snapT = sx-27*ux, sy-27*uy, float32(0)
	s.tickMobsLocked(c, 0.2)
	if !m.returning {
		t.Fatalf("mob past leash range should be returning (aggro=%v)", m.aggro)
	}

	// The player now chases and stays glued 2u from the mob (inside aggro range) the
	// whole way home, and even lands a hit -- the mob must keep returning, never aggro.
	homed := false
	for i := 2; i < 80; i++ {
		c.x, c.y, c.snapT = m.x+2*ux, m.y+2*uy, float32(0) // glued, within aggro range
		s.hitMobLocked(c, m, 1, c.objID)                   // poke it: damage yes, aggro no
		if m.aggro {
			t.Fatalf("tick %d: a hit on a returning mob must not re-aggro it", i)
		}
		s.tickMobsLocked(c, float64(i)*0.2)
		if m.returning && m.aggro {
			t.Fatalf("tick %d: a returning mob must not aggro a chasing player", i)
		}
		if !m.returning {
			homed = true
			break
		}
	}
	if !homed {
		t.Fatal("mob never finished returning home")
	}
	// Reset complete AND a player is adjacent -> it aggros FRESH on the next tick.
	c.x, c.y, c.snapT = m.x+2*ux, m.y+2*uy, float32(0)
	s.tickMobsLocked(c, 80*0.2)
	if !m.aggro {
		t.Fatal("a mob that reached home with an adjacent player must aggro fresh")
	}
}

// TestMobFogHideResetsToSpawn: a wounded, displaced mob abandoned past the fog-hide
// radius is snapped home at full HP so it re-reveals pristine (covers a mob dropped
// faster than the walk-home can finish -- e.g. a blink/teleport away).
func TestMobFogHideResetsToSpawn(t *testing.T) {
	s, c, _, sx, sy := newNavConn(t)
	idx := mobIndexByPrefab(t, "Mob_Skeleton_1H_Melee_01")
	mob := gamedata.Mobs()[idx]
	m := &mobState{
		id: 2202, mobIdx: idx, mob: mob,
		x: sx + 5, y: sy, spawnX: sx + 20, spawnY: sy,
		hp: mob.Health * 0.2, shown: true, aggro: true,
		// A Hunt mob: it owns its spawn, so the abandon path may send it home.
		homed: true,
	}

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	// Player is well past the hide radius from the mob -> fog hides it this tick.
	c.x, c.y, c.snapT = m.x+float32(mobHideRadius)+5, m.y, float32(0)
	c.huntState.mobs[m.id] = m
	c.huntState.tr.add(m.id)

	s.mobInterestLocked(c, m, 0.2)

	if m.shown {
		t.Fatal("mob past hide radius should be hidden")
	}
	if got, want := m.hp, m.maxHealth(); got != want {
		t.Errorf("abandoned mob HP = %g, want full %g", got, want)
	}
	if m.x != m.spawnX || m.y != m.spawnY {
		t.Errorf("abandoned mob at (%.2f,%.2f), want spawn (%.2f,%.2f)", m.x, m.y, m.spawnX, m.spawnY)
	}
	if m.aggro || m.returning {
		t.Errorf("abandoned mob should be idle: aggro=%v returning=%v", m.aggro, m.returning)
	}
}
