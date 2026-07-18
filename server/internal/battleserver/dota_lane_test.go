package battleserver

import (
	"fmt"
	"math"
	"testing"

	"tanatserver/internal/gamedata"
)

// creepsOf returns the live creeps (non-structure mobs) in the instance.
func creepsOf(inst *huntInstance) []*mobState {
	var out []*mobState
	for _, m := range inst.mobs {
		if !m.structure && !m.dead {
			out = append(out, m)
		}
	}
	return out
}

// barracksOf returns one side's barracks, in map order.
func barracksOf(side gamedata.DotaSide) []gamedata.DotaStructure {
	var out []gamedata.DotaStructure
	for _, sc := range gamedata.DotaMaps()[0].Structures {
		if sc.Role == gamedata.DotaCreepTower && sc.Side == side {
			out = append(out, sc)
		}
	}
	return out
}

// TestCreepWaveCoversEveryLane is the reported bug: every creep marched the same line.
// The cause was that only one lane was ever baked -- but baking three would have fixed
// nothing on its own, because the spawner handed Lanes[0] to every creep regardless.
// This drives the real spawn path over a side's whole barracks and asserts each lane
// actually receives troops.
func TestCreepWaveCoversEveryLane(t *testing.T) {
	s, c, inst, cleanup := newDotaConn(t, "Avtr_HK_Grimlok")
	defer cleanup()
	now := float64(s.battleTime())

	dm := gamedata.DotaMaps()[0]
	if len(dm.Lanes) < 2 {
		t.Fatalf("map has %d lanes: this test cannot say anything", len(dm.Lanes))
	}
	bars := barracksOf(gamedata.DotaSideHuman)

	c.lock()
	for _, bar := range bars {
		s.dotaSpawnCreepWaveLocked(c, bar, now)
	}
	c.unlock()

	perLane := map[int]int{}
	for _, m := range creepsOf(inst) {
		if len(m.lane) == 0 {
			t.Errorf("creep %d marches no lane at all", m.id)
			continue
		}
		// Identify the lane by its geometry, not by an index we handed out: that way the
		// test still means something if the assignment is restructured.
		matched := -1
		for li, lane := range dm.Lanes {
			if len(lane) == len(m.lane) && lane[0] == m.lane[0] && lane[len(lane)-1] == m.lane[len(lane)-1] {
				if sameLane(lane, m.lane) {
					matched = li
					break
				}
			}
		}
		if matched < 0 {
			t.Errorf("creep %d marches a lane that is not one of the map's", m.id)
			continue
		}
		perLane[matched]++
	}
	for li := range dm.Lanes {
		if perLane[li] == 0 {
			t.Errorf("lane %d got no creeps from a full round of waves: creeps still bunch onto a subset of lanes", li)
		}
		if perLane[li] != gamedata.CreepsPerWave {
			t.Errorf("lane %d got %d creeps, want %d: the barracks are not 1:1 with the lanes", li, perLane[li], gamedata.CreepsPerWave)
		}
	}
	if want := len(bars) * gamedata.CreepsPerWave; len(creepsOf(inst)) != want {
		t.Errorf("round spawned %d creeps, want %d (%d barracks x %d)", len(creepsOf(inst)), want, len(bars), gamedata.CreepsPerWave)
	}
}

// TestEachBarracksOwnsOneLane: the whole 1:1 design rests on the map really placing one
// barracks per lane, and on that placement being UNAMBIGUOUS. Both are claims about
// geometry, so measure them. The margin matters more than the distance: the springs are
// 5.7-15.6u from a lane, overlapping the barracks' own 3.8-9.7u, so what separates a
// barracks is not that it is close but that its lane is close and every OTHER lane is
// far. If a lane edit ever erodes that, the assignment silently starts guessing.
func TestEachBarracksOwnsOneLane(t *testing.T) {
	dm := gamedata.DotaMaps()[0]
	for _, side := range []gamedata.DotaSide{gamedata.DotaSideHuman, gamedata.DotaSideElf} {
		seen := map[int]int32{}
		bars := barracksOf(side)
		if len(bars) != len(dm.Lanes) {
			t.Errorf("side %d has %d barracks for %d lanes: not 1:1", side, len(bars), len(dm.Lanes))
		}
		for _, bar := range bars {
			li := dm.LaneFor(bar)
			if li < 0 {
				t.Errorf("barracks %d claims no lane: its creeps would have nowhere to march", bar.ID)
				continue
			}
			if other, dup := seen[li]; dup {
				t.Errorf("barracks %d and %d both claim lane %d: one lane would get double waves and another none", other, bar.ID, li)
			}
			seen[li] = bar.ID

			own, next := math.Inf(1), math.Inf(1)
			for j, lane := range dm.Lanes {
				d := distToLane(lane, bar.X, bar.Z)
				if j == li {
					own = d
				} else if d < next {
					next = d
				}
			}
			if next-own < 15 {
				t.Errorf("barracks %d is %.1fu from lane %d but only %.1fu from the next: the assignment is a coin flip, not geometry",
					bar.ID, own, li, next)
			}
		}
	}
}

func distToLane(lane []gamedata.Vec2, x, z float64) float64 {
	best := math.Inf(1)
	for i := 0; i+1 < len(lane); i++ {
		if d := distToSeg(lane[i], lane[i+1], x, z); d < best {
			best = d
		}
	}
	return best
}

// TestFirstWaveWaitsForTheGracePeriod guards a trap rather than a feature. The cadence
// check is `now < d.nextWave[bar.id]` over a MAP, so a barracks with no seeded entry
// reads 0 and is therefore always due -- seed the wrong role and waves do not stop, they
// start instantly, skipping the grace period players get to reach their lane. A missing
// entry has to fail loudly here because it cannot fail anywhere else.
func TestFirstWaveWaitsForTheGracePeriod(t *testing.T) {
	s, c, inst, cleanup := newDotaConn(t, "Avtr_HK_Grimlok")
	defer cleanup()
	start := float64(s.battleTime())

	c.lock()
	s.dotaSpawnWavesLocked(c, start+gamedata.CreepFirstWave-0.5)
	c.unlock()
	if n := len(creepsOf(inst)); n != 0 {
		t.Fatalf("%d creeps marched %gs before the first wave was due: the barracks' cadence is unseeded and reads 0",
			n, 0.5)
	}

	c.lock()
	s.dotaSpawnWavesLocked(c, start+gamedata.CreepFirstWave+0.1)
	c.unlock()
	if len(creepsOf(inst)) == 0 {
		t.Fatal("no creeps once the first wave came due")
	}
}

// TestDeadBarracksSilencesItsLane is the point of moving the waves onto the barracks:
// «Казарма» is a real objective. Kill one and its lane stops; the others keep coming.
func TestDeadBarracksSilencesItsLane(t *testing.T) {
	s, c, inst, cleanup := newDotaConn(t, "Avtr_HK_Grimlok")
	defer cleanup()
	dm := gamedata.DotaMaps()[0]
	bars := barracksOf(gamedata.DotaSideHuman)
	dead := bars[0]
	deadLane := dm.LaneFor(dead)

	c.lock()
	inst.mobs[dotaStructIDBase+dead.ID].dead = true
	now := float64(s.battleTime()) + gamedata.CreepFirstWave + 0.1
	s.dotaSpawnWavesLocked(c, now)
	c.unlock()

	for _, m := range creepsOf(inst) {
		if m.team != dotaPlayerTeam {
			continue // the elf barracks are untouched and must keep marching
		}
		for li, lane := range dm.Lanes {
			if li == deadLane && len(m.lane) == len(lane) && sameLane(lane, m.lane) {
				t.Fatalf("creep %d marches lane %d, whose barracks is rubble", m.id, li)
			}
		}
	}
	live := 0
	for _, m := range creepsOf(inst) {
		if m.team == dotaPlayerTeam {
			live++
		}
	}
	if want := (len(bars) - 1) * gamedata.CreepsPerWave; live != want {
		t.Errorf("%d allied creeps after one barracks fell, want %d: the surviving lanes must keep coming", live, want)
	}
}

func sameLane(a, b []gamedata.Vec2) bool {
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestWaveCarriesAnArcherOnEveryLane: the melee/ranged mix used to be picked with the
// same modulus that now also picks the lane. Left alone, that made every archer in a
// wave land on the same lane and the other two march as pure melee.
func TestWaveCarriesAnArcherOnEveryLane(t *testing.T) {
	dm := gamedata.DotaMaps()[0]
	bars := barracksOf(gamedata.DotaSideHuman)
	_, ranged := dm.CreepMobIdx(gamedata.DotaSideHuman)

	// Several consecutive waves: the archer's slot rotates with parity, so one wave
	// proves less than it looks.
	for wave := 0; wave < 3; wave++ {
		s, c, inst, cleanup := newDotaConn(t, "Avtr_HK_Grimlok")
		now := float64(s.battleTime())
		c.lock()
		inst.dota.waveParity = wave
		for _, bar := range bars {
			s.dotaSpawnCreepWaveLocked(c, bar, now)
		}
		c.unlock()

		archers := map[string]int{}
		for _, m := range creepsOf(inst) {
			if m.mobIdx == ranged {
				archers[laneKey(m.lane)]++
			}
		}
		if len(archers) != len(dm.Lanes) {
			t.Errorf("wave %d: archers reached %d of %d lanes", wave, len(archers), len(dm.Lanes))
		}
		cleanup()
	}
}

func laneKey(lane []gamedata.Vec2) string {
	if len(lane) < 2 {
		return ""
	}
	return fmt.Sprintf("%.1f,%.1f", lane[1].X, lane[1].Y)
}

// TestLaneEntrySkipsBacktrack: the lanes run altar to altar, but creeps come out of a
// generator off to one side of them. Entering at index 0 marches a creep back into its
// OWN altar -- 30u+ in the wrong direction, in full view, before it turns round. The
// first waypoint must be neither the home altar nor somewhere across the map.
func TestLaneEntrySkipsBacktrack(t *testing.T) {
	s, c, inst, cleanup := newDotaConn(t, "Avtr_HK_Grimlok")
	defer cleanup()
	now := float64(s.battleTime())

	for _, side := range []gamedata.DotaSide{gamedata.DotaSideHuman, gamedata.DotaSideElf} {
		for _, bar := range barracksOf(side) {
			c.lock()
			s.dotaSpawnCreepWaveLocked(c, bar, now)
			c.unlock()
		}
	}
	if len(creepsOf(inst)) == 0 {
		t.Fatal("no creeps spawned: the test proves nothing")
	}
	for _, m := range creepsOf(inst) {
		if m.laneIdx < 0 || m.laneIdx >= len(m.lane) {
			t.Errorf("creep %d entered at index %d, outside its %d-waypoint lane", m.id, m.laneIdx, len(m.lane))
			continue
		}
		home := 0 // the end of the lane this creep marches AWAY from: its own altar
		if !m.laneFwd {
			home = len(m.lane) - 1
		}
		if m.laneIdx == home {
			t.Errorf("creep %d (fwd=%v) enters at waypoint %d, its own altar: it marches backwards out of the gate",
				m.id, m.laneFwd, m.laneIdx)
		}
		wp := m.lane[m.laneIdx]
		if d := math.Hypot(float64(m.x)-wp.X, float64(m.y)-wp.Y); d > 70 {
			t.Errorf("creep %d enters its lane at waypoint %d, %.0fu away across open map", m.id, m.laneIdx, d)
		}
	}
}

// TestDotaPlayerMovementIsClipped: «Штурм» used to run with no walkability at all --
// server.go said so outright ("nav stays nil") -- so players walked through the map's
// rock. The instance must now carry map_1_0's real collision, and it must be the
// collision of THIS map: a grid in a mismatched frame passes a nil check and still
// lets you through walls, so the check is against ground truth, not against non-nil.
func TestDotaPlayerMovementIsClipped(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_HK_Grimlok")
	defer cleanup()
	_ = s
	if inst.nav == nil {
		t.Fatal("«Штурм» instance has no nav: players walk through the map")
	}
	dm := gamedata.DotaMaps()[0]
	if inst.nav.Walkable(dm.SpawnHuman.X, dm.SpawnHuman.Y) != true {
		t.Error("the human spawn is not walkable: a player would be stuck at birth")
	}
	// Inside poly[9], the river-bank rock the old centre lane used to cut through.
	if inst.nav.Walkable(-24, -6.4) {
		t.Error("(-24,-6.4) is inside a rock and the grid calls it open: wrong frame or wrong polygon")
	}
	// A click into that rock must stop short of it rather than teleport through.
	x, y := inst.nav.Clip(-60, -7, 0, -6)
	if inst.nav.Walkable(x, y) != true {
		t.Errorf("Clip landed the player at (%.1f,%.1f), which is not walkable", x, y)
	}
	if x >= -24 {
		t.Errorf("Clip walked the player to (%.1f,%.1f), through the rock at x=-24", x, y)
	}
}

// TestCreepsMarchFreeGround walks a full wave down its lane at march speed and asserts
// no creep ever stands inside the map's collision. The lanes are checked against the
// polygon in gamedata; this checks the thing that actually moves.
func TestCreepsMarchFreeGround(t *testing.T) {
	s, c, inst, cleanup := newDotaConn(t, "Avtr_HK_Grimlok")
	defer cleanup()
	dm := gamedata.DotaMaps()[0]
	if dm.Nav == nil {
		t.Fatal("map_1_0 has no nav: nothing to check against")
	}
	now := float64(s.battleTime())
	c.lock()
	for _, bar := range barracksOf(gamedata.DotaSideHuman) {
		s.dotaSpawnCreepWaveLocked(c, bar, now)
	}
	// Nothing to fight: the player is parked at his own base and every enemy structure
	// is far away, so the creeps do nothing but march, which is what we want to watch.
	for step := 0; step < 1200; step++ {
		now += 0.5
		for _, m := range creepsOf(inst) {
			s.dotaMarchLaneLocked(c, m, now)
			m.x += m.vx * 0.5
			m.y += m.vy * 0.5
			m.lastSync = now
			if !dm.Nav.Walkable(float64(m.x), float64(m.y)) {
				c.unlock()
				t.Fatalf("creep %d marched into geometry at (%.1f,%.1f), lane waypoint %d", m.id, m.x, m.y, m.laneIdx)
			}
		}
	}
	c.unlock()
}
