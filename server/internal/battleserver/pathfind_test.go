package battleserver

import (
	"math"
	"net"
	"testing"
	"time"

	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// newNavConn builds a minimal hunt conn positioned at the map_4_0 spawn with the
// scene's real walkability grid installed, plus a drain goroutine on the socket
// so pushed syncs never block. Caller gets the conn, the server and the nav.
func newNavConn(t *testing.T) (*Server, *conn, gamedata.Nav, float32, float32) {
	return newNavConnAvatar(t, 13)
}

// newNavConnAvatar is newNavConn for a specific avatar id (skill-kit dependent
// tests: toggles, summons, passives).
func newNavConnAvatar(t *testing.T, avatarID int32) (*Server, *conn, gamedata.Nav, float32, float32) {
	t.Helper()
	m, ok := gamedata.HuntMapByID(40) // map_4_0 crypt: the nav-backed Hunt scene
	if !ok || m.Nav == nil {
		t.Fatal("map_4_0 (id 40) has no nav grid")
	}
	s := New(session.NewStore())
	srv, cli := net.Pipe()
	t.Cleanup(func() { srv.Close(); cli.Close() })
	go func() {
		buf := make([]byte, 4096)
		for {
			if _, err := cli.Read(buf); err != nil {
				return
			}
		}
	}()

	av, ok := gamedata.AvatarByID(avatarID)
	if !ok {
		t.Fatalf("avatar id %d missing", avatarID)
	}
	sx, sy := m.Spawn()
	c := &conn{Conn: srv, nav: m.Nav}
	c.objID = 1000
	c.x, c.y, c.snapT = float32(sx), float32(sy), s.battleTime()
	hs := &huntState{
		av:      av,
		kit:     gamedata.SkillsFor(av),
		m:       m,
		mobs:    map[int32]*mobState{},
		summons: map[int32]*summonState{},
		hp:      100, mana: 200,
	}
	for i := range hs.skillLevel {
		hs.skillLevel[i] = 1
	}
	hs.tr.add(c.objID)
	c.huntState = hs
	t.Cleanup(func() { c.mvMu.Lock(); hs.closed = true; c.mvMu.Unlock() })
	return s, c, m.Nav, float32(sx), float32(sy)
}

// TestMoveClearLineSingleLeg: a straight, unobstructed move from spawn is one leg
// ending at the goal (unchanged from the pre-pathfinding behaviour).
func TestMoveClearLineSingleLeg(t *testing.T) {
	s, c, nav, sx, sy := newNavConn(t)

	// Find a nearby goal reachable in a straight line from spawn.
	var gx, gy float32
	found := false
	for r := float32(2); r <= 6 && !found; r++ {
		for a := 0; a < 16; a++ {
			ang := float64(a) * math.Pi / 8
			tx := sx + r*float32(math.Cos(ang))
			ty := sy + r*float32(math.Sin(ang))
			cx, cy := nav.Clip(float64(sx), float64(sy), float64(tx), float64(ty))
			if math.Hypot(cx-float64(tx), cy-float64(ty)) < 0.1 { // straight line clear
				gx, gy, found = tx, ty, true
				break
			}
		}
	}
	if !found {
		t.Skip("no clear-line goal near spawn on this map")
	}

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	c.moveToLocked(s, gx, gy)
	if len(c.path) != 1 {
		t.Fatalf("clear-line move should be a single leg, got %d waypoints: %v", len(c.path), c.path)
	}
	if !c.hasDest {
		t.Fatal("move did not set hasDest")
	}
	if math.Hypot(float64(c.destX-gx), float64(c.destY-gy)) > 0.1 {
		t.Fatalf("dest = (%.2f,%.2f), want ~(%.2f,%.2f)", c.destX, c.destY, gx, gy)
	}
	// First leg's velocity must point at the goal.
	if c.vx == 0 && c.vy == 0 {
		t.Fatal("no leg velocity set")
	}
}

// TestMoveAroundWallMultiLeg: a move to a point the straight line can't reach is
// routed into multiple legs that match nav.Path, ending at the routed goal.
func TestMoveAroundWallMultiLeg(t *testing.T) {
	s, c, nav, sx, sy := newNavConn(t)

	// Search for a goal that requires routing: nav.Path returns >1 waypoint.
	var gx, gy float32
	var want []gamedata.Vec2
	found := false
	for r := float32(6); r <= 40 && !found; r += 2 {
		for a := 0; a < 24 && !found; a++ {
			ang := float64(a) * math.Pi / 12
			tx := sx + r*float32(math.Cos(ang))
			ty := sy + r*float32(math.Sin(ang))
			if !nav.Walkable(float64(tx), float64(ty)) {
				continue
			}
			route := nav.Path(float64(sx), float64(sy), float64(tx), float64(ty))
			if len(route) > 1 {
				gx, gy, want, found = tx, ty, route, true
			}
		}
	}
	if !found {
		t.Skip("no wall-routed goal found near spawn on this map")
	}

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	c.moveToLocked(s, gx, gy)
	if len(c.path) != len(want) {
		t.Fatalf("routed move: got %d legs, want %d (%v vs %v)", len(c.path), len(want), c.path, want)
	}
	last := c.path[len(c.path)-1]
	if math.Hypot(last.X-float64(gx), last.Y-float64(gy)) > 0.1 {
		t.Fatalf("route ends at %v, want goal (%.2f,%.2f)", last, gx, gy)
	}
	if math.Hypot(float64(c.destX)-last.X, float64(c.destY)-last.Y) > 1e-6 {
		t.Fatalf("destX/destY (%.2f,%.2f) != final waypoint %v", c.destX, c.destY, last)
	}
	// First leg heads toward the first waypoint, NOT straight at the goal (proves
	// it is routing around the wall).
	wp0 := c.path[0]
	legAng := math.Atan2(float64(c.vy), float64(c.vx))
	wpAng := math.Atan2(wp0.Y-float64(sy), wp0.X-float64(sx))
	if d := math.Abs(legAng - wpAng); d > 0.05 && math.Abs(d-2*math.Pi) > 0.05 {
		t.Fatalf("first leg velocity (ang %.3f) does not aim at first waypoint (ang %.3f)", legAng, wpAng)
	}
}

// TestWalkMultiLegArrives drives the real leg/arrival timer chain: it issues a
// routed move and waits for the avatar to walk every leg to the destination,
// asserting it actually turns (more than one heading) and ends at the goal. This
// exercises the AfterFunc state machine (leg -> arrival -> next leg -> halt)
// under mvMu, not just the initial route setup.
func TestWalkMultiLegArrives(t *testing.T) {
	s, c, nav, sx, sy := newNavConn(t)

	// Find a routed goal (>=2 legs) whose total walk length is modest so the test
	// finishes in a few seconds at move speed ~4.
	pathLen := func(fx, fy float64, r []gamedata.Vec2) float64 {
		ax, ay, sum := fx, fy, 0.0
		for _, wp := range r {
			sum += math.Hypot(wp.X-ax, wp.Y-ay)
			ax, ay = wp.X, wp.Y
		}
		return sum
	}
	var gx, gy float32
	var want []gamedata.Vec2
	found := false
	for r := float32(6); r <= 40 && !found; r += 1 {
		for a := 0; a < 48 && !found; a++ {
			ang := float64(a) * math.Pi / 24
			tx := sx + r*float32(math.Cos(ang))
			ty := sy + r*float32(math.Sin(ang))
			if !nav.Walkable(float64(tx), float64(ty)) {
				continue
			}
			route := nav.Path(float64(sx), float64(sy), float64(tx), float64(ty))
			if len(route) >= 2 {
				if l := pathLen(float64(sx), float64(sy), route); l >= 6 && l <= 25 {
					gx, gy, want, found = tx, ty, route, true
				}
			}
		}
	}
	if !found {
		t.Skip("no short routed goal found near spawn on this map")
	}

	c.mvMu.Lock()
	c.moveToLocked(s, gx, gy)
	dx, dy := c.destX, c.destY
	c.mvMu.Unlock()

	// Poll the walker to completion, recording distinct headings as proof it
	// turned at a waypoint rather than walking straight.
	headings := map[int]bool{}
	deadline := time.Now().Add(10 * time.Second)
	arrived := false
	for time.Now().Before(deadline) {
		c.mvMu.Lock()
		moving := c.hasDest
		vx, vy := c.vx, c.vy
		c.mvMu.Unlock()
		if vx != 0 || vy != 0 {
			headings[int(math.Round(math.Atan2(float64(vy), float64(vx))*8))] = true
		}
		if !moving {
			arrived = true
			break
		}
		time.Sleep(40 * time.Millisecond)
	}
	if !arrived {
		t.Fatal("avatar never finished the routed walk (legs did not chain to completion)")
	}
	c.mvMu.Lock()
	ex, ey := c.x, c.y
	c.mvMu.Unlock()
	if d := math.Hypot(float64(ex-dx), float64(ey-dy)); d > 0.2 {
		t.Fatalf("avatar stopped %.2f from the destination (%.1f,%.1f), got (%.1f,%.1f)", d, dx, dy, ex, ey)
	}
	if len(headings) < 2 {
		t.Errorf("avatar walked a single heading (%d) — it did not turn at a waypoint; route was %v", len(headings), want)
	}
}

// TestDashStraightLine: a dash-style straight move to a wall-blocked point takes
// a single clipped leg (a straight lunge), NOT a routed multi-leg path around the
// wall — the regression the review flagged for dashLocked.
func TestDashStraightLine(t *testing.T) {
	s, c, nav, sx, sy := newNavConn(t)

	// Find a goal that moveToLocked WOULD route (multi-leg).
	var gx, gy float32
	found := false
	for r := float32(6); r <= 40 && !found; r += 2 {
		for a := 0; a < 24 && !found; a++ {
			ang := float64(a) * math.Pi / 12
			tx := sx + r*float32(math.Cos(ang))
			ty := sy + r*float32(math.Sin(ang))
			if nav.Walkable(float64(tx), float64(ty)) && len(nav.Path(float64(sx), float64(sy), float64(tx), float64(ty))) > 1 {
				gx, gy, found = tx, ty, true
			}
		}
	}
	if !found {
		t.Skip("no wall-routed goal near spawn on this map")
	}

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	c.moveStraightLocked(s, gx, gy)
	if len(c.path) != 1 {
		t.Fatalf("straight move produced %d legs, want 1 (a dash must not route around walls): %v", len(c.path), c.path)
	}
	// The single leg is clipped to walkable ground.
	last := c.path[0]
	if !nav.Walkable(last.X, last.Y) {
		t.Fatalf("clipped dash endpoint %v is not walkable", last)
	}
}

// TestAimAlongThrottlesRecompute: aimAlong must not re-run A* on every tick while
// mid-route (the throttle behind the "unreachable target every tick" fix).
func TestAimAlongThrottlesRecompute(t *testing.T) {
	_, c, nav, sx, sy := newNavConn(t)

	var mx, my float32
	found := false
	for r := float32(6); r <= 40 && !found; r += 2 {
		for a := 0; a < 24 && !found; a++ {
			ang := float64(a) * math.Pi / 12
			tx := sx + r*float32(math.Cos(ang))
			ty := sy + r*float32(math.Sin(ang))
			if nav.Walkable(float64(tx), float64(ty)) && !c.mobHasLoSLocked(float32(tx), float32(ty), sx, sy) &&
				len(nav.Path(float64(tx), float64(ty), float64(sx), float64(sy))) > 1 {
				mx, my, found = tx, ty, true
			}
		}
	}
	if !found {
		t.Skip("no wall-blocked mob position near spawn on this map")
	}

	var ps pathState
	c.aimAlong(&ps, mx, my, sx, sy, true, 10.0)
	if len(ps.pts) == 0 {
		t.Skip("route did not resolve")
	}
	at1 := ps.at
	// A tick later (0.2s), same target, mob barely moved: must NOT recompute.
	c.aimAlong(&ps, mx+0.05, my, sx, sy, true, 10.2)
	if ps.at != at1 {
		t.Errorf("aimAlong recomputed within the throttle window (at %.2f -> %.2f)", at1, ps.at)
	}
	// After the 1s staleness window: must recompute.
	c.aimAlong(&ps, mx+0.05, my, sx, sy, true, 11.3)
	if ps.at == at1 {
		t.Error("aimAlong did not recompute after the staleness window elapsed")
	}
}

// TestChaseMoveThrottles: chaseMoveLocked must not re-run A* (and churn the
// arrival timer + POSITION syncs) on every 200-250ms combat re-arm while the
// chase goal is unchanged — but must re-path at once on >1m goal drift, and at
// most once per second when the walker went idle short of the goal.
func TestChaseMoveThrottles(t *testing.T) {
	s, c, nav, sx, sy := newNavConn(t)

	// A walkable chase goal ~10m from spawn.
	var gx, gy float32
	found := false
	for a := 0; a < 24 && !found; a++ {
		ang := float64(a) * math.Pi / 12
		tx := sx + 10*float32(math.Cos(ang))
		ty := sy + 10*float32(math.Sin(ang))
		if nav.Walkable(float64(tx), float64(ty)) {
			gx, gy, found = tx, ty, true
		}
	}
	if !found {
		t.Skip("no walkable goal 10m from spawn")
	}

	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	// New chase session: paths immediately.
	c.resetChaseLocked()
	c.chaseMoveLocked(s, gx, gy)
	if !c.hasDest {
		t.Fatal("chase did not start a move")
	}
	gen := c.moveGen // bumped by stopArrivalLocked inside every moveToLocked

	// Same goal next tick: throttled, no re-path.
	c.chaseMoveLocked(s, gx, gy)
	if c.moveGen != gen {
		t.Fatal("chaseMoveLocked re-pathed an unchanged goal while the walker was busy")
	}
	// Sub-tolerance nudge (<1m): still throttled.
	c.chaseMoveLocked(s, gx+0.4, gy)
	if c.moveGen != gen {
		t.Fatal("chaseMoveLocked re-pathed a sub-tolerance (<1m) goal nudge")
	}
	// Real drift (>1m): re-paths at once.
	c.chaseMoveLocked(s, gx+2, gy)
	if c.moveGen == gen {
		t.Fatal("chaseMoveLocked ignored a >1m goal drift")
	}
	gen = c.moveGen

	// Walker idle short of the goal (e.g. clipped/failed route): the retry is
	// gated to once per second, then allowed.
	c.stopArrivalLocked()
	c.hasDest = false
	c.chaseMoveLocked(s, gx+2, gy)
	if c.moveGen != gen+1 { // stopArrivalLocked above bumped gen once itself
		t.Fatal("idle chase retried within the 1s gate")
	}
	c.chaseRepathAt = float64(s.battleTime()) - 2.0
	c.hasDest = false
	c.chaseMoveLocked(s, gx+2, gy)
	if c.moveGen == gen+1 {
		t.Fatal("idle chase did not retry after the 1s gate elapsed")
	}
}

// TestClampInvalidatesMobRoute: when the wall clamp clips a route-following mob
// (separation pushed it off the string-pulled line), the cached route must be
// invalidated so aimAlong re-paths from the clipped spot in the SAME tick,
// instead of steering at an occluded waypoint until the 1s staleness window.
func TestClampInvalidatesMobRoute(t *testing.T) {
	s, c, nav, sx, sy := newNavConn(t)
	hs := c.huntState

	// Find a mob spot that (a) is wall-blocked from the player (so aimAlong
	// keeps a route), (b) is within the leash, and (c) has an unwalkable
	// neighbour cell to steer into (so the clamp trips).
	var mx, my float32
	var wx, wy float64 // centre of the unwalkable neighbour
	found := false
	for r := float32(6); r <= 20 && !found; r += 1 {
		for a := 0; a < 24 && !found; a++ {
			ang := float64(a) * math.Pi / 12
			tx := sx + r*float32(math.Cos(ang))
			ty := sy + r*float32(math.Sin(ang))
			if !nav.Walkable(float64(tx), float64(ty)) || c.mobHasLoSLocked(tx, ty, sx, sy) {
				continue
			}
			if len(nav.Path(float64(tx), float64(ty), float64(sx), float64(sy))) <= 1 {
				continue
			}
			for _, d := range [][2]float64{{1, 0}, {-1, 0}, {0, 1}, {0, -1}} {
				nx, ny := float64(tx)+d[0], float64(ty)+d[1]
				if !nav.Walkable(nx, ny) {
					mx, my, wx, wy, found = tx, ty, nx, ny, true
					break
				}
			}
		}
	}
	if !found {
		t.Skip("no wall-blocked mob spot with an adjacent wall near spawn")
	}

	mob := &mobState{id: 6000, mobIdx: 2, mob: gamedata.Mobs()[2],
		x: mx, y: my, hp: gamedata.Mobs()[2].Health, aggro: true, shown: true}
	hs.mobs[mob.id] = mob

	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	// Prime a cached route toward the player, as a previous tick would have.
	c.aimAlong(&mob.pf, mob.x, mob.y, sx, sy, true, 10.0)
	if len(mob.pf.pts) == 0 {
		t.Skip("route did not resolve from the mob spot")
	}

	// Steer into the wall (as separation could): next dead-reckon step lands in
	// the unwalkable neighbour, so the clamp must trip and invalidate the route.
	dt := float32(tickInterval.Seconds())
	mob.vx = (float32(wx) - mob.x) / dt
	mob.vy = (float32(wy) - mob.y) / dt
	s.tickMobsLocked(c, 10.2)

	if !nav.Walkable(float64(mob.x), float64(mob.y)) {
		t.Fatalf("clamp failed: mob ended up on unwalkable ground (%.2f,%.2f)", mob.x, mob.y)
	}
	// The stale primed route (at=10.0) must be gone: either re-pathed this tick
	// (fresh at) or dropped. Steering at the old cached waypoint is the bug.
	if mob.pf.at == 10.0 && len(mob.pf.pts) > 0 && mob.pf.idx < len(mob.pf.pts) {
		t.Fatal("clamped mob kept steering its stale cached route (no same-tick re-path)")
	}
}

// TestMobRouteAroundWall: a mob whose straight line to the player is wall-blocked
// steers toward an A* waypoint (not straight through the wall).
func TestMobRouteAroundWall(t *testing.T) {
	_, c, nav, sx, sy := newNavConn(t)

	// Place a mob so that a wall sits between it and the player at spawn.
	var mx, my float32
	found := false
	for r := float32(6); r <= 40 && !found; r += 2 {
		for a := 0; a < 24 && !found; a++ {
			ang := float64(a) * math.Pi / 12
			tx := sx + r*float32(math.Cos(ang))
			ty := sy + r*float32(math.Sin(ang))
			if !nav.Walkable(float64(tx), float64(ty)) {
				continue
			}
			if !c.mobHasLoSLocked(float32(tx), float32(ty), sx, sy) {
				route := nav.Path(float64(tx), float64(ty), float64(sx), float64(sy))
				if len(route) > 1 {
					mx, my, found = tx, ty, true
				}
			}
		}
	}
	if !found {
		t.Skip("no wall-blocked mob position found near spawn on this map")
	}

	var ps pathState
	blocked := !c.mobHasLoSLocked(mx, my, sx, sy)
	if !blocked {
		t.Fatal("test setup: mob should be blocked from the player")
	}
	gx, gy := c.aimAlong(&ps, mx, my, sx, sy, blocked, 1.0)
	// The steer point must be a real waypoint (reachable in a straight line from
	// the mob), unlike the player which is behind a wall.
	if !nav.Walkable(float64(gx), float64(gy)) {
		t.Fatalf("mob steer point (%.2f,%.2f) is not walkable", gx, gy)
	}
	cx, cy := nav.Clip(float64(mx), float64(my), float64(gx), float64(gy))
	if math.Hypot(cx-float64(gx), cy-float64(gy)) > 0.6 {
		t.Fatalf("mob steer point is not reachable in a straight line (it would cut a wall)")
	}
	// And it must differ from steering straight at the (blocked) player.
	if math.Abs(float64(gx)-float64(sx)) < 0.1 && math.Abs(float64(gy)-float64(sy)) < 0.1 {
		t.Fatal("mob aimed straight at the player through the wall")
	}
}
