package battleserver

import (
	"math"
	"testing"

	"tanatserver/internal/gamedata"
)

// The gap these tests exist to close: mobsep_test.go proves the separation HELPER works,
// and proved it for a year while «Штурм» never called it. A helper test cannot see a
// missing call. Everything here drives the real tick and asserts on where the units end
// up, so deleting the call from dotaMoveTowardLocked fails a test instead of a player.

// spreadCreeps puts n melee creeps of `team` on top of each other at lane waypoint `idx`,
// marching forward, and returns them.
func spreadCreeps(t *testing.T, inst *huntInstance, team int32, idx int, n int, now float64) []*mobState {
	t.Helper()
	lane := inst.dota.m.Lanes[0]
	wp := lane[idx]
	mi := inst.dota.m.HumanCreepMelee
	out := make([]*mobState, 0, n)
	for i := 0; i < n; i++ {
		m := &mobState{
			id: int32(61000 + i), mobIdx: mi, mob: gamedata.Mobs()[mi],
			// Within a tenth of a metre of one another: this IS the reported symptom.
			x: float32(wp.X) + float32(i)*0.05, y: float32(wp.Y),
			hp: 500, maxHP: 500, dmgMin: 5, dmgMax: 8,
			team: team, lane: lane, laneIdx: idx + 1, laneFwd: true, lastSync: now,
			active: true, shown: true,
		}
		inst.mobs[m.id] = m
		out = append(out, m)
	}
	return out
}

func minPairDist(ms []*mobState) float64 {
	best := math.Inf(1)
	for i := 0; i < len(ms); i++ {
		for j := i + 1; j < len(ms); j++ {
			if d := math.Hypot(float64(ms[i].x-ms[j].x), float64(ms[i].y-ms[j].y)); d < best {
				best = d
			}
		}
	}
	return best
}

// TestStormCreepsDoNotStackOnOnePoint is the user's report, verbatim: «Крипы ... в одну
// точку не вставали». A wave files down one lane toward one goal, so without a push it
// converges by construction -- four creeps spawned on one spot stay one spot forever.
func TestStormCreepsDoNotStackOnOnePoint(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())

	c.lock()
	creeps := spreadCreeps(t, inst, dotaPlayerTeam, 1, 4, now)
	c.unlock()

	start := minPairDist(creeps)
	if start > 0.2 {
		t.Fatalf("test setup is wrong: creeps start %.2fu apart, so nothing is being proven", start)
	}
	for i := 0; i < 40; i++ {
		now += tickInterval.Seconds()
		c.lock()
		s.dotaTickLocked(c, now)
		c.unlock()
	}
	got := minPairDist(creeps)
	// One body radius of clearance: they are beside each other rather than inside.
	if got < 1.0 {
		t.Errorf("after 8s of marching the two closest creeps are still %.2fu apart (started %.2fu): "+
			"the wave is walking as one point", got, start)
	}
}

// TestStormCreepsStillReachTheirGoal is the other half: separation must not turn a wave
// into a cloud that wanders off. A push that beats the march is not collision, it is a
// scatter -- and it would silently stall every lane.
func TestStormCreepsStillReachTheirGoal(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	lane := inst.dota.m.Lanes[0]

	c.lock()
	creeps := spreadCreeps(t, inst, dotaPlayerTeam, 1, 4, now)
	c.unlock()

	goal := lane[2]
	before := math.Hypot(float64(creeps[0].x)-goal.X, float64(creeps[0].y)-goal.Y)
	for i := 0; i < 40; i++ {
		now += tickInterval.Seconds()
		c.lock()
		s.dotaTickLocked(c, now)
		c.unlock()
	}
	for _, m := range creeps {
		// Each creep advances its own lane cursor on arrival, so measure progress as
		// "closed distance to the waypoint it was sent to, or already passed it".
		after := math.Hypot(float64(m.x)-goal.X, float64(m.y)-goal.Y)
		if m.laneIdx <= 2 && after >= before {
			t.Errorf("creep %d made no headway toward its waypoint (%.1fu -> %.1fu): separation is "+
				"overpowering the march", m.id, before, after)
		}
	}
}

// TestStormCreepsSeparateOnWalkableGroundOnly: the push is lateral, so it can aim a creep
// at ground the straight heading never touched -- and nothing clips a creep's step. This
// is the one way separation could newly walk a wave into rock.
func TestStormCreepsSeparateOnWalkableGroundOnly(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	if c.nav == nil {
		t.Fatal("«Штурм» instance has no nav: the walkability guard below proves nothing")
	}

	c.lock()
	creeps := spreadCreeps(t, inst, dotaPlayerTeam, 1, 6, now)
	c.unlock()

	for i := 0; i < 60; i++ {
		now += tickInterval.Seconds()
		c.lock()
		s.dotaTickLocked(c, now)
		c.unlock()
		for _, m := range creeps {
			if !c.nav.Walkable(float64(m.x), float64(m.y)) {
				t.Fatalf("creep %d was pushed into geometry at (%.1f,%.1f) on tick %d",
					m.id, m.x, m.y, i)
			}
		}
	}
}

// TestStormCreepInReachStaysPlanted guards the seam this fix had to thread. Hunt sidesteps
// while it swings (mobai.go, in-range arm); «Штурм» must NOT -- a creep that slides during
// its strike is the bug the client's blended attack clip made visible, and it was reported
// and fixed once already. Separation is allowed on the approach only.
func TestStormCreepInReachStaysPlanted(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())

	mi := inst.dota.m.HumanCreepMelee
	ei := inst.dota.m.ElfCreepMelee
	lane := inst.dota.m.Lanes[0]
	wp := lane[1]

	// Two allies crowding each other, both already in reach of one enemy: the crowd would
	// make Hunt sidestep. Here they must plant.
	a := &mobState{id: 62001, mobIdx: mi, mob: gamedata.Mobs()[mi],
		x: float32(wp.X), y: float32(wp.Y), hp: 500, maxHP: 500, dmgMin: 5, dmgMax: 8,
		team: dotaPlayerTeam, lane: lane, laneIdx: 2, laneFwd: true, lastSync: now,
		active: true, shown: true}
	b := &mobState{id: 62002, mobIdx: mi, mob: gamedata.Mobs()[mi],
		x: float32(wp.X) + 0.1, y: float32(wp.Y), hp: 500, maxHP: 500, dmgMin: 5, dmgMax: 8,
		team: dotaPlayerTeam, lane: lane, laneIdx: 2, laneFwd: true, lastSync: now,
		active: true, shown: true}
	enemy := &mobState{id: 62003, mobIdx: ei, mob: gamedata.Mobs()[ei],
		x: float32(wp.X) + 1.5, y: float32(wp.Y), hp: 5000, maxHP: 5000,
		team: dotaEnemyTeam, lastSync: now, active: true, shown: true}

	c.lock()
	inst.mobs[a.id], inst.mobs[b.id], inst.mobs[enemy.id] = a, b, enemy
	now += tickInterval.Seconds()
	s.dotaTickLocked(c, now)
	c.unlock()

	if a.swingDoneAt == 0 && b.swingDoneAt == 0 {
		t.Fatal("neither creep engaged the enemy in its reach: the test proves nothing about swinging")
	}
	for _, m := range []*mobState{a, b} {
		if m.swingDoneAt > 0 && (m.vx != 0 || m.vy != 0) {
			t.Errorf("creep %d is moving (v=%.2f,%.2f) while its strike clip plays: separation leaked "+
				"into the swing arm, which is exactly the «крипы двигаются во время замаха» bug",
				m.id, m.vx, m.vy)
		}
	}
}

// TestStormMarchDominatesSeparation pins the balance directly, where an integration test
// can't: the push must only TURN a marching creep, never replace its heading. Drive one
// step with a neighbour crowding from the side and assert the creep still advances toward
// its goal. (The integration tests miss this because a scattering pack's push decays to
// zero and the goal-fallback masks a march term that was dropped.)
func TestStormMarchDominatesSeparation(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	mi := inst.dota.m.HumanCreepMelee

	m := &mobState{id: 70001, mobIdx: mi, mob: gamedata.Mobs()[mi],
		x: 0, y: 0, team: dotaPlayerTeam, lastSync: now, active: true, shown: true}
	// A packmate crowding from +y, perpendicular to the +x travel: the push is straight
	// sideways, so if it ever beats the march the forward velocity collapses.
	nb := &mobState{id: 70002, mobIdx: mi, mob: gamedata.Mobs()[mi],
		x: 0, y: 0.6, team: dotaPlayerTeam, lastSync: now, active: true, shown: true}
	inst.mobs[m.id], inst.mobs[nb.id] = m, nb

	c.lock()
	s.dotaMoveTowardLocked(c, m, 100, 0, now) // goal far down +x
	c.unlock()

	sp := float32(mobSpeed(m, now))
	if m.vx <= sp*0.5 {
		t.Errorf("forward velocity collapsed to %.2f of speed %.2f: separation is overpowering the "+
			"march instead of merely deflecting it", m.vx, sp)
	}
}

// TestStormEngagementUnsticks is the hole the adversarial pass found on the real tick that
// the march-only fix left open: two enemy waves that MEET on a lane both enter reach, both
// stopMobLocked+attack, and weld onto one point forever (a natural engagement froze at a
// 0.24u gap). The in-range sidestep must part them -- and it must do so WITHOUT any creep
// moving on the frame it strikes (task #7).
func TestStormEngagementUnsticks(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())

	mi := inst.dota.m.HumanCreepMelee
	ei := inst.dota.m.ElfCreepMelee
	lane := inst.dota.m.Lanes[0]
	wp := lane[3] // out in open lane, clear of any structure's reach

	// Three allies stacked on one point, and one enemy for them all to swing at: exactly
	// the pile-up the report described, formed the way the game forms it.
	allies := make([]*mobState, 0, 3)
	for i := 0; i < 3; i++ {
		a := &mobState{id: int32(65000 + i), mobIdx: mi, mob: gamedata.Mobs()[mi],
			x: float32(wp.X) + float32(i)*0.05, y: float32(wp.Y),
			hp: 500, maxHP: 500, dmgMin: 5, dmgMax: 8,
			team: dotaPlayerTeam, lane: lane, laneIdx: 4, laneFwd: true, lastSync: now,
			active: true, shown: true}
		inst.mobs[a.id] = a
		allies = append(allies, a)
	}
	enemy := &mobState{id: 65100, mobIdx: ei, mob: gamedata.Mobs()[ei],
		x: float32(wp.X) + 0.9, y: float32(wp.Y), hp: 100000, maxHP: 100000,
		team: dotaEnemyTeam, lastSync: now, active: true, shown: true}
	inst.mobs[enemy.id] = enemy

	start := minPairDist(allies)
	var movedMidSwing int
	for i := 0; i < 120; i++ { // 24s: the free-tick duty cycle is only ~4%, so unsticking is slow
		now += tickInterval.Seconds()
		c.lock()
		s.dotaTickLocked(c, now)
		// The load-bearing invariant: nobody moves while its own strike clip plays.
		for _, a := range allies {
			if a.swingDoneAt > 0 && (a.vx != 0 || a.vy != 0) {
				movedMidSwing++
			}
		}
		c.unlock()
	}
	if movedMidSwing > 0 {
		t.Errorf("a creep moved on %d swing frames: the in-range sidestep leaked into the strike, "+
			"regressing «крипы двигаются во время замаха»", movedMidSwing)
	}
	got := minPairDist(allies)
	if got < 1.0 {
		t.Errorf("engagement stayed welded: closest allies %.2fu apart after 24s (started %.2fu). "+
			"The pile forms in the in-range arm, which the march-only push never reaches", got, start)
	}
}

// TestStormSeparationDoesNotFloodPositionSyncs. Separation perturbs the heading a little
// every tick, so the obvious wiring -- re-sync whenever it changed -- puts one POSITION
// per creep per tick on a 2011 client, forever, for every creep on the board. The
// throttle is part of the fix, not a nicety.
//
// Counted as velocity changes rather than SYNC packets on the wire: those two are the
// same event (a creep's velocity is only ever assigned in the branch that broadcasts it),
// but the wire also carries the fog pass revealing structures, which would drown the
// signal being measured here.
func TestStormSeparationDoesNotFloodPositionSyncs(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())

	c.lock()
	creeps := spreadCreeps(t, inst, dotaPlayerTeam, 1, 6, now)
	c.unlock()

	const ticks = 50 // 10 seconds
	var syncs int
	for i := 0; i < ticks; i++ {
		was := make(map[int32][2]float32, len(creeps))
		for _, m := range creeps {
			was[m.id] = [2]float32{m.vx, m.vy}
		}
		now += tickInterval.Seconds()
		c.lock()
		s.dotaTickLocked(c, now)
		c.unlock()
		for _, m := range creeps {
			if was[m.id] != [2]float32{m.vx, m.vy} {
				syncs++
			}
		}
	}

	// The staleness bound is 0.7s, so 6 permanently-crowded creeps over 10s justify about
	// 86. Un-throttled this scenario is 6*50 = 300, and it does not decay: the lanes are
	// ~4.5u wide, so a wave can never actually satisfy its own spacing and keeps jostling.
	if max := 6 * 10 * 2; syncs > max {
		t.Errorf("%d position re-syncs for 6 creeps over 10s (cap %d): separation is re-syncing "+
			"nearly every tick, which is what the 0.7s bound exists to stop", syncs, max)
	}
}

// TestSummonHoldsStillThroughItsSwing is the pet's half of the same seam. Giving a held
// pet a shuffle is what lets it climb out of a body it was standing in -- but a pet that
// shuffles WHILE it strikes is the «двигается во время замаха» bug wearing a different
// hat, and unlike a creep the summon tick has no freeze of its own to fall back on.
func TestSummonHoldsStillThroughItsSwing(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	px, py := c.posAtLocked(float32(now))

	mi := inst.dota.m.HumanCreepMelee
	ei := inst.dota.m.ElfCreepMelee
	// An enemy inside the pet's reach, so it commits to a swing...
	enemy := &mobState{id: 64001, mobIdx: ei, mob: gamedata.Mobs()[ei],
		x: px + 1.2, y: py, hp: 9000, maxHP: 9000,
		team: dotaEnemyTeam, lastSync: now, active: true, shown: true}
	// ...and three allies piled onto the pet, all to ONE side so the push is directional
	// and unmistakably past the deadband (symmetric bodies would cancel to a zero sum and
	// prove nothing). Allied, or the pet would fight one of them instead of the enemy under
	// test. If the mid-swing gate were removed, THIS is the crowd that would shove the pet
	// off its spot during its strike.
	blockers := make([]*mobState, 0, 3)
	for i, d := range [][2]float32{{0.1, 0.0}, {0.15, 0.1}, {0.15, -0.1}} {
		b := &mobState{id: int32(64010 + i), mobIdx: mi, mob: gamedata.Mobs()[mi],
			x: px + d[0], y: py + d[1], hp: 500, maxHP: 500,
			team: dotaPlayerTeam, lastSync: now, active: true, shown: true}
		inst.mobs[b.id] = b
		blockers = append(blockers, b)
	}
	dino := &summonState{
		id: 7200, hp: 300, maxHP: 300, dmg: 25, until: now + 600,
		x: px, y: py, pet: true, slot: 3, ordTarget: enemy.id,
	}

	c.lock()
	inst.mobs[enemy.id] = enemy
	c.huntState.summons[dino.id] = dino
	now += tickInterval.Seconds()
	s.tickSummonsLocked(c, now)
	swingEnds := dino.swingDoneAt
	c.unlock()

	if swingEnds == 0 {
		t.Fatal("the pet never swung at the enemy in its reach: nothing about swinging is being proven")
	}
	if dino.vx != 0 || dino.vy != 0 {
		t.Fatalf("the pet is sliding (v=%.2f,%.2f) on its swing frame while a body overlaps it: "+
			"separation is steering it through its own strike clip", dino.vx, dino.vy)
	}
	// Still true partway through the swing, not just on the frame it started.
	c.lock()
	s.tickSummonsLocked(c, (now+swingEnds)/2)
	c.unlock()
	if dino.vx != 0 || dino.vy != 0 {
		t.Fatalf("the pet shuffled off mid-swing (v=%.2f,%.2f)", dino.vx, dino.vy)
	}
}

// TestDinoSeparatesFromOtherBodies is the second half of the user's report: «динозавр
// гримлока тоже не имеет коллизии». A pet is not in inst.mobs, so nothing that scanned
// only mobs could ever push it or be pushed by it.
func TestDinoSeparatesFromOtherBodies(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())

	// A pet ordered to walk to a point that a creep is already standing on. Without
	// collision it walks straight in and stops inside the creep.
	px, py := c.posAtLocked(float32(now))
	dino := &summonState{
		id: 7100, hp: 300, maxHP: 300, dmg: 25, until: now + 600,
		x: px, y: py, pet: true, slot: 3,
		ordX: px + 6, ordY: py, ordMove: true,
	}
	mi := inst.dota.m.HumanCreepMelee
	blocker := &mobState{id: 63001, mobIdx: mi, mob: gamedata.Mobs()[mi],
		x: px + 6, y: py, hp: 500, maxHP: 500,
		team: dotaPlayerTeam, lastSync: now, active: true, shown: true}

	c.lock()
	c.huntState.summons[dino.id] = dino
	inst.mobs[blocker.id] = blocker
	for i := 0; i < 60; i++ {
		now += tickInterval.Seconds()
		s.tickSummonsLocked(c, now)
	}
	c.unlock()

	d := math.Hypot(float64(dino.x-blocker.x), float64(dino.y-blocker.y))
	// Bodies are 0.5 (summon) and 0.6 (creep): anything under ~1.0 is standing inside.
	if d < 1.0 {
		t.Errorf("the dinosaur came to rest %.2fu from a creep's centre -- inside its body. It walked "+
			"the ordered point straight through another unit; summons still have no collision", d)
	}
	// It must still have gone SOMEWHERE near the order, not refused to move.
	if math.Hypot(float64(dino.x-px), float64(dino.y-py)) < 3.0 {
		t.Errorf("the dinosaur barely left its start (%.1f,%.1f -> %.1f,%.1f): collision is blocking "+
			"the order instead of steering around it", px, py, dino.x, dino.y)
	}
}
