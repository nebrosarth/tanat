package battleserver

import (
	"math"
	"net"
	"testing"

	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// arenaConn builds one player conn wired into an arena instance on the given team, at
// (x,y). Mirrors newDotaCaptureConn but for «Арена»: no mobs, an explicit team.
func arenaConn(t *testing.T, s *Server, inst *huntInstance, objID int32, team int32, x, y float32) *conn {
	t.Helper()
	srv, cli := net.Pipe()
	t.Cleanup(func() { srv.Close(); cli.Close() })
	r := battleproto.NewReader(cli)
	go func() {
		for {
			if _, err := r.Read(); err != nil {
				return
			}
		}
	}()
	av := avatarByPrefab(t, "Avtr_Tank_Velial")
	c := &conn{Conn: srv}
	c.objID = objID
	c.selfPlayerID = objID
	c.x, c.y, c.snapT = x, y, s.battleTime()
	hs := &huntState{
		av: av, kit: gamedata.SkillsFor(av),
		mobs: inst.mobs, summons: map[int32]*summonState{},
		hp: av.Health, mana: av.Mana, team: team,
		worldReady: true,
	}
	hs.tr.add(objID)
	hs.inst = inst
	c.huntState = hs
	c.inst = inst
	c.nav = inst.nav
	c.lk = &inst.mu
	inst.members[objID] = c
	return c
}

func newArenaWorld(t *testing.T) (*Server, *huntInstance) {
	t.Helper()
	s := New(session.NewStore())
	m := gamedata.ArenaMaps()[0]
	inst := newArenaInstance(s, m.ID, m.ID)
	return s, inst
}

// TestArenaTeamsAreHostile: two players on opposing sides are enemies to each other and
// legal auto-attack targets; two on the same side are not (no friendly fire).
func TestArenaTeamsAreHostile(t *testing.T) {
	s, inst := newArenaWorld(t)
	a := arenaConn(t, s, inst, 1000, gamedata.ArenaTeamA, 0, 0)
	b := arenaConn(t, s, inst, 1001, gamedata.ArenaTeamB, 2, 0)
	ally := arenaConn(t, s, inst, 1002, gamedata.ArenaTeamA, -2, 0)

	if !arenaEnemies(a, b) {
		t.Error("opposing-team players are not enemies")
	}
	if arenaEnemies(a, ally) {
		t.Error("same-team players are enemies: friendly fire is on")
	}
	if a.playerTeam() == b.playerTeam() {
		t.Error("the two sides share a team value")
	}
}

// TestArenaPlayerCanKillEnemy is the core of the mode: A auto-attacks B, B's HP falls,
// and when it reaches zero B dies with A credited the frag.
func TestArenaPlayerCanKillEnemy(t *testing.T) {
	s, inst := newArenaWorld(t)
	a := arenaConn(t, s, inst, 1000, gamedata.ArenaTeamA, 0, 0)
	b := arenaConn(t, s, inst, 1001, gamedata.ArenaTeamB, 1, 0)
	now := float64(s.battleTime())

	// B is standing in A's reach. Land blows directly (the timer just schedules these)
	// until B drops, so the test doesn't depend on wall-clock AfterFunc timing.
	inst.mu.Lock()
	b.huntState.hp = 120
	before := b.huntState.hp
	landed := false
	for i := 0; i < 200 && b.huntState.deadUntil == 0; i++ {
		s.hitPlayerFromLocked(b, a.objID, 20, now, nil, a)
		landed = true
	}
	dead := b.huntState.deadUntil > 0
	frags := inst.arena.frags[gamedata.ArenaTeamA]
	aFrags := a.huntState.frags
	inst.mu.Unlock()

	if !landed || b.huntState.hp >= before && !dead {
		t.Fatalf("B took no damage from A (hp %.0f -> %.0f)", before, b.huntState.hp)
	}
	if !dead {
		t.Fatal("B never died despite repeated lethal blows")
	}
	if frags != 1 {
		t.Errorf("team A frag total = %d, want 1", frags)
	}
	if aFrags != 1 {
		t.Errorf("killer's personal frag count = %d, want 1", aFrags)
	}
}

// TestArenaFragLimitEndsMatch: reaching the frag limit broadcasts BATTLE_END with the
// winning team, exactly once.
func TestArenaFragLimitEndsMatch(t *testing.T) {
	s, inst := newArenaWorld(t)
	a := arenaConn(t, s, inst, 1000, gamedata.ArenaTeamA, 0, 0)
	b := arenaConn(t, s, inst, 1001, gamedata.ArenaTeamB, 1, 0)
	now := float64(s.battleTime())
	inst.arena.fragLimit = 3

	inst.mu.Lock()
	for i := 0; i < 3; i++ {
		// Credit a kill directly; victim identity doesn't matter for the count.
		s.arenaCreditKillLocked(a, b, now)
	}
	ended := inst.arena.ended
	winner := inst.arena.winner
	inst.mu.Unlock()

	if !ended {
		t.Fatal("match did not end at the frag limit")
	}
	if winner != gamedata.ArenaTeamA {
		t.Errorf("winner = %d, want team A (%d)", winner, gamedata.ArenaTeamA)
	}
	// A further kill after the end must not move the winner or re-broadcast.
	inst.mu.Lock()
	s.arenaCreditKillLocked(a, b, now)
	still := inst.arena.winner
	inst.mu.Unlock()
	if still != gamedata.ArenaTeamA {
		t.Errorf("a post-match kill changed the winner to %d", still)
	}
}

// TestArenaTeamAssignmentAlternates: joining players are split across the two sides so a
// full lobby is balanced, and a solo player just occupies side A.
func TestArenaTeamAssignmentAlternates(t *testing.T) {
	_, inst := newArenaWorld(t)
	got := []int32{}
	for i := 0; i < 4; i++ {
		got = append(got, inst.arena.assignTeam())
	}
	want := []int32{gamedata.ArenaTeamA, gamedata.ArenaTeamB, gamedata.ArenaTeamA, gamedata.ArenaTeamB}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("join %d assigned team %d, want %d", i, got[i], want[i])
		}
	}
}

// TestArenaInitialSpawnsSeparateSides: the two sides' first spawns are far apart, so a
// 1v1 doesn't start with the players on top of each other.
func TestArenaInitialSpawnsSeparateSides(t *testing.T) {
	_, inst := newArenaWorld(t)
	ax, ay := arenaInitialSpawn(inst.arena, gamedata.ArenaTeamA)
	bx, by := arenaInitialSpawn(inst.arena, gamedata.ArenaTeamB)
	d := (ax-bx)*(ax-bx) + (ay-by)*(ay-by)
	if d < 30*30 {
		t.Errorf("the two sides spawn %.1fu apart, want >= 30 so a 1v1 doesn't start face-to-face",
			d)
	}
}

// TestArenaRespawnAvoidsEnemy: a killed player respawns at the marker farthest from the
// living enemy, not on top of them.
func TestArenaRespawnAvoidsEnemy(t *testing.T) {
	s, inst := newArenaWorld(t)
	// Enemy sitting exactly on spawn marker 0.
	sp0 := inst.arena.m.Spawns[0]
	enemy := arenaConn(t, s, inst, 1001, gamedata.ArenaTeamB, float32(sp0.X), float32(sp0.Y))
	victim := arenaConn(t, s, inst, 1000, gamedata.ArenaTeamA, 0, 0)
	now := float64(s.battleTime())

	inst.mu.Lock()
	rx, ry := s.arenaSpawnLocked(inst, victim, now)
	inst.mu.Unlock()

	// The chosen point must be the marker FARTHEST from the enemy, not the nearest.
	// (A float32==float64 equality check on the coordinates silently never matched, so
	// this measures distance instead: the picker maximises distance-to-nearest-enemy.)
	chosen := math.Hypot(float64(rx)-sp0.X, float64(ry)-sp0.Y)
	maxD := 0.0
	for _, sp := range inst.arena.m.Spawns {
		if d := math.Hypot(sp.X-sp0.X, sp.Y-sp0.Y); d > maxD {
			maxD = d
		}
	}
	if math.Abs(chosen-maxD) > 0.5 {
		t.Errorf("respawn landed %.1fu from the enemy, but the farthest marker is %.1fu away: "+
			"the picker isn't avoiding the enemy", chosen, maxD)
	}
	_ = enemy
}
