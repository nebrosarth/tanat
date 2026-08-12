package battleserver

import (
	"math"
	"testing"

	"tanatserver/internal/gamedata"
)

type countingOpenNav struct {
	pathCalls int
}

func (n *countingOpenNav) Walkable(float64, float64) bool { return true }
func (n *countingOpenNav) Spawn() (float64, float64)      { return 0, 0 }
func (n *countingOpenNav) Clip(_, _, tx, ty float64) (float64, float64) {
	return tx, ty
}
func (n *countingOpenNav) Path(_, _, tx, ty float64) []gamedata.Vec2 {
	n.pathCalls++
	return []gamedata.Vec2{{X: tx, Y: ty}}
}

func setupStructureDetourGeometry(t *testing.T) (*Server, *conn, *mobState, *botBrain, *countingOpenNav, float64) {
	t.Helper()
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	t.Cleanup(cleanup)
	nav := &countingOpenNav{}
	bot.nav = nav
	bot.x, bot.y = 20, 0
	bot.snapT = float32(s.battleTime())
	threat := structOfSide(inst, gamedata.DotaCreepTower, dotaTeamElf)
	if threat == nil {
		t.Fatal("setup: no enemy tower")
	}
	threat.x, threat.y = 0, 0
	threat.hitAt = float64(s.battleTime()) + 10
	threat.hitTarget = bot.objID
	b := &botBrain{c: bot}
	return s, bot, threat, b, nav, float64(s.battleTime())
}

func assertRouteOutsideCircle(t *testing.T, route []gamedata.Vec2, fx, fy float32, threat *mobState, radius float64) {
	t.Helper()
	ax, ay := float64(fx), float64(fy)
	for i, point := range route {
		if botSegmentIntersectsCircle(ax, ay, point.X, point.Y, float64(threat.x), float64(threat.y), radius) {
			t.Fatalf("route segment %d [(%.2f,%.2f)->(%.2f,%.2f)] enters danger circle r=%.2f", i, ax, ay, point.X, point.Y, radius)
		}
		ax, ay = point.X, point.Y
	}
}

func TestBotStructureDetourEastBotWestHomeTowerBetween(t *testing.T) {
	s, bot, threat, b, nav, now := setupStructureDetourGeometry(t)
	hx, hy := float32(-20), float32(0)

	bot.lock()
	defer bot.unlock()
	b.retreating = true
	waypoint, ok := s.botRetreatDetourWaypointLocked(b, now, hx, hy)
	if !ok {
		t.Fatal("no detour selected for east-bot/west-home/tower-between geometry")
	}
	if len(b.structureDetour) == 0 {
		t.Fatal("detour selected no remaining waypoint")
	}
	if b.structureDetourSide == 0 {
		t.Fatal("detour selected no stable side")
	}
	if waypoint.x == 0 && waypoint.y == 0 {
		t.Fatal("detour waypoint was origin")
	}
	radius := s.botStructureDangerRadiusLocked(bot, threat)
	assertRouteOutsideCircle(t, []gamedata.Vec2{{X: float64(waypoint.x), Y: float64(waypoint.y)}}, bot.x, bot.y, threat, radius)
	expandedRadius := radius + botStructureDetourClear
	for i, wp := range b.structureDetour {
		if math.Hypot(float64(wp.x-threat.x), float64(wp.y-threat.y)) < expandedRadius-0.01 {
			t.Fatalf("waypoint %d (%.2f,%.2f) is inside expanded danger circle r=%.2f", i, wp.x, wp.y, expandedRadius)
		}
	}
	fullRoute := make([]gamedata.Vec2, 0, len(b.structureDetour)+1)
	for _, wp := range b.structureDetour {
		fullRoute = append(fullRoute, gamedata.Vec2{X: float64(wp.x), Y: float64(wp.y)})
	}
	fullRoute = append(fullRoute, gamedata.Vec2{X: float64(hx), Y: float64(hy)})
	assertRouteOutsideCircle(t, fullRoute, bot.x, bot.y, threat, radius)
	if got := float64(waypoint.x-threat.x)*float64(bot.x-threat.x) + float64(waypoint.y-threat.y)*float64(bot.y-threat.y); got <= 0 {
		t.Fatalf("first detour waypoint crossed to the opposite side: dot=%.2f", got)
	}
	if nav.pathCalls == 0 {
		t.Fatal("detour never consulted navigation")
	}
	t.Logf("east-bot/west-home detour: side=%d waypoint=(%.2f,%.2f) remaining=%d nav_path_calls=%d", b.structureDetourSide, waypoint.x, waypoint.y, len(b.structureDetour), nav.pathCalls)
}

func TestBotCommittedFocusFirstMoveStaysCurrentSideAndOutward(t *testing.T) {
	s, bot, threat, b, _, now := setupStructureDetourGeometry(t)

	bot.lock()
	defer bot.unlock()
	s.botEscapeStructureFocusLocked(b, threat, now)
	if !bot.hasDest {
		t.Fatal("committed focus did not issue a first move")
	}
	if bot.destX <= threat.x {
		t.Fatalf("first committed-focus destination x=%.2f crossed west through tower at x=%.2f", bot.destX, threat.x)
	}
	if float64(bot.destX-threat.x)*float64(bot.x-threat.x)+float64(bot.destY-threat.y)*float64(bot.y-threat.y) <= 0 {
		t.Fatalf("first committed-focus move is not outward/current-side: bot=(%.2f,%.2f) dest=(%.2f,%.2f)", bot.x, bot.y, bot.destX, bot.destY)
	}
	assertRouteOutsideCircle(t, bot.path, bot.x, bot.y, threat, s.botStructureDangerRadiusLocked(bot, threat))
}

func TestBotCommittedFocusEscapesWhenStartingInsideDangerCircle(t *testing.T) {
	s, bot, threat, b, _, now := setupStructureDetourGeometry(t)
	bot.x = 5
	bot.y = 0
	bot.snapT = float32(now)

	bot.lock()
	defer bot.unlock()
	s.botEscapeStructureFocusLocked(b, threat, now)
	if !bot.hasDest {
		t.Fatal("committed focus did not issue an escape move from inside the danger circle")
	}
	startDistance := math.Hypot(float64(bot.x-threat.x), float64(bot.y-threat.y))
	destinationDistance := math.Hypot(float64(bot.destX-threat.x), float64(bot.destY-threat.y))
	if destinationDistance <= startDistance {
		t.Fatalf("inside-circle escape did not move outward: start=%.2f destination=%.2f", startDistance, destinationDistance)
	}
	if !s.botEscapeRouteClearOfStructureLocked(bot, bot.x, bot.y, bot.destX, bot.destY, threat, true) {
		t.Fatalf("inside-circle escape route is not monotonically outward: start=(%.2f,%.2f) destination=(%.2f,%.2f)", bot.x, bot.y, bot.destX, bot.destY)
	}
}

func TestBotCommittedFocusStableSideAcrossRepeatedTicks(t *testing.T) {
	s, bot, threat, b, _, now := setupStructureDetourGeometry(t)
	bot.lock()
	defer bot.unlock()

	s.botEscapeStructureFocusLocked(b, threat, now)
	firstX, firstY := bot.destX, bot.destY
	firstSide := math.Copysign(1, float64(firstX-threat.x))
	for i := 1; i <= 8; i++ {
		s.botHoldStructureAvoidanceLocked(b, now+float64(i)*0.2)
		if got := math.Copysign(1, float64(bot.destX-threat.x)); got != firstSide {
			t.Fatalf("tick %d changed escape side: initial=%v current=%v dest=(%.2f,%.2f)", i, firstSide, got, bot.destX, bot.destY)
		}
		if math.Hypot(float64(bot.destX-firstX), float64(bot.destY-firstY)) > 0.01 {
			t.Fatalf("tick %d oscillated escape destination: first=(%.2f,%.2f) current=(%.2f,%.2f)", i, firstX, firstY, bot.destX, bot.destY)
		}
	}
}

func TestBotCommittedFocusLatchesDestinationUntilThreatChangesOrPointIsReached(t *testing.T) {
	s, bot, threat, b, _, now := setupStructureDetourGeometry(t)
	bot.lock()
	defer bot.unlock()
	s.botEscapeStructureFocusLocked(b, threat, now)
	firstX, firstY := bot.destX, bot.destY
	firstPath := append([]gamedata.Vec2(nil), bot.path...)
	for i := 1; i <= 5; i++ {
		s.botHoldStructureAvoidanceLocked(b, now+float64(i)*0.2)
		if bot.destX != firstX || bot.destY != firstY || len(bot.path) != len(firstPath) {
			t.Fatalf("same committed threat churned movement at tick %d: first=(%.2f,%.2f) current=(%.2f,%.2f)", i, firstX, firstY, bot.destX, bot.destY)
		}
	}
	other := *threat
	other.id++
	other.x, other.y = threat.x, threat.y+20
	s.botEscapeStructureFocusLocked(b, &other, now+2)
	if b.structureAvoidTarget != other.id {
		t.Fatalf("new threat did not replace latched target: got %d, want %d", b.structureAvoidTarget, other.id)
	}
	if bot.destX == firstX && bot.destY == firstY {
		t.Fatal("new threat retained the old committed destination")
	}
	oldX, oldY := bot.destX, bot.destY
	bot.x, bot.y, bot.snapT = oldX, oldY, float32(now+3)
	s.botEscapeStructureFocusLocked(b, &other, now+3)
	if bot.destX == oldX && bot.destY == oldY {
		t.Fatal("reached committed destination was not recomputed")
	}
}

func TestBotClearHomeRouteRemainsDirect(t *testing.T) {
	s, bot, _, b, nav, now := setupStructureDetourGeometry(t)
	// Remove the firing state: a clear route must not acquire detour state.
	for _, m := range bot.huntState.mobs {
		if m.structure && m.enemyOf(bot.playerTeam()) {
			m.hitAt, m.hitTarget, m.projLaunchAt, m.projTarget, m.dtarget, m.nextSwing = 0, 0, 0, 0, 0, 0
		}
	}
	bot.lock()
	defer bot.unlock()
	if wp, ok := s.botRetreatDetourWaypointLocked(b, now, -20, 0); ok {
		t.Fatalf("clear home route selected detour waypoint %+v", wp)
	}
	s.botMoveTowardLocked(b, -20, 0, now)
	if len(bot.path) != 1 {
		t.Fatalf("clear home route produced %d segments, want direct: %v", len(bot.path), bot.path)
	}
	if nav.pathCalls != 2 {
		t.Fatalf("clear home route made %d nav Path calls, want one route check plus one direct move", nav.pathCalls)
	}
}

func TestBotStructureDetourPathCallsAreBounded(t *testing.T) {
	s, bot, _, b, nav, now := setupStructureDetourGeometry(t)
	bot.lock()
	defer bot.unlock()
	if _, ok := s.botRetreatDetourWaypointLocked(b, now, -20, 0); !ok {
		t.Fatal("detour setup did not resolve")
	}
	// The 48 perimeter samples may each validate their start/end leg, plus at
	// most eight middle legs. Keep enough headroom for filtering differences
	// while preventing a return to nav validation for every candidate pair.
	const maxPathCalls = 128
	t.Logf("structure detour candidate nav_path_calls=%d (ceiling=%d)", nav.pathCalls, maxPathCalls)
	if nav.pathCalls > maxPathCalls {
		t.Fatalf("structure detour made %d nav Path calls; likely pathological candidate search (ceiling %d)", nav.pathCalls, maxPathCalls)
	}
}
