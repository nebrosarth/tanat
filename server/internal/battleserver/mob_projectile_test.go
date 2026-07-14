package battleserver

import (
	"net"
	"sync"
	"testing"
	"time"

	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// findMob returns the index of the first non-boss (no skills) mob matching ranged.
func findMob(t *testing.T, ranged bool) (int, gamedata.Mob) {
	t.Helper()
	for i, m := range gamedata.Mobs() {
		if len(m.Skills) > 0 {
			continue // skip bosses -- they cast instead of basic-swinging
		}
		if (m.AttackRange > 0) == ranged {
			return i, m
		}
	}
	t.Fatalf("no non-boss mob with ranged=%v found", ranged)
	return 0, gamedata.Mob{}
}

// setupSwingConn builds a bare hunt conn with a packet-capturing reader and one
// mob planted in melee/aggro range of the avatar at the origin, ready to swing.
func setupSwingConn(t *testing.T, mobIdx int, mob gamedata.Mob, dist float32) (*Server, *conn, *mobState, *[]battleproto.Packet, *sync.Mutex) {
	t.Helper()
	s := New(session.NewStore())
	srv, cli := net.Pipe()
	t.Cleanup(func() { srv.Close(); cli.Close() })

	var mu sync.Mutex
	pkts := &[]battleproto.Packet{}
	r := battleproto.NewReader(cli)
	go func() {
		for {
			p, err := r.Read()
			if err != nil {
				return
			}
			mu.Lock()
			*pkts = append(*pkts, p)
			mu.Unlock()
		}
	}()

	av, _ := gamedata.AvatarByID(13) // a real avatar => sane body radius for reach math
	c := &conn{Conn: srv}
	c.objID = 1000
	c.x, c.y, c.snapT = 0, 0, s.battleTime()
	hs := &huntState{
		av:      av,
		mobs:    map[int32]*mobState{},
		summons: map[int32]*summonState{},
		hp:      500, mana: 200,
	}
	hs.tr.add(c.objID)
	m := &mobState{id: 2000, mobIdx: mobIdx, mob: mob, hp: mob.Health,
		x: dist, y: 0, spawnX: dist, spawnY: 0, aggro: true}
	hs.mobs[m.id] = m
	c.huntState = hs
	t.Cleanup(func() { c.mvMu.Lock(); hs.closed = true; c.mvMu.Unlock() })
	return s, c, m, pkts, &mu
}

func countProjectiles(t *testing.T, pkts *[]battleproto.Packet, mu *sync.Mutex, source int32) int {
	t.Helper()
	time.Sleep(50 * time.Millisecond) // let the pipe reader drain
	mu.Lock()
	defer mu.Unlock()
	n := 0
	for _, p := range *pkts {
		if p.Cmd != battleproto.CmdSetProjectile {
			continue
		}
		if src, _ := p.Args.GetInt("source"); src == source {
			n++
		}
	}
	return n
}

// TestRangedMobReleasesArrowAtAnimationEnd: a ranged mob (archer / caster / shooter
// plant) fires a SET_PROJECTILE so the client flies the arrow prefab -- but the arrow
// must leave the bow at the END of the draw animation (the ACTION_DONE moment), not
// partway through it. So the swing commits with NO projectile and hitAt unset; the
// SET_PROJECTILE is emitted when the animation ends (projLaunchAt == swingDoneAt), and
// the hit is then scheduled for the arrow's ARRIVAL, after the release and before the
// next swing.
func TestRangedMobReleasesArrowAtAnimationEnd(t *testing.T) {
	idx, mob := findMob(t, true)
	s, c, m, pkts, mu := setupSwingConn(t, idx, mob, 5) // shoots from range

	t0 := float64(s.battleTime())
	c.mvMu.Lock()
	s.tickMobsLocked(c, t0)
	swung := m.projLaunchAt > 0
	launchAt := m.projLaunchAt
	animEnd := m.swingDoneAt
	nextSwing := m.nextSwing
	hitBeforeRelease := m.hitAt
	c.mvMu.Unlock()

	if !swung {
		t.Fatal("ranged mob never committed a swing -- test would be vacuous")
	}
	// No projectile and no committed hit at swing start: the arrow waits for the draw.
	if got := countProjectiles(t, pkts, mu, m.id); got != 0 {
		t.Fatalf("arrow released at swing start (%d SET_PROJECTILE); it must wait for the animation end", got)
	}
	if hitBeforeRelease != 0 {
		t.Fatalf("ranged hit committed before the arrow flew (hitAt=%g, want 0)", hitBeforeRelease)
	}
	// The release lands in the LATE part of the draw -- past mid-swing, no later than
	// the animation end -- and still leaves room for the arrow's flight before the next
	// swing. (Not on the first frame, which was the bug.)
	midSwing := t0 + 0.5*(nextSwing-t0)
	if launchAt < midSwing || launchAt > animEnd+1e-9 {
		t.Fatalf("arrow release %g outside the late-draw window [%g, %g]", launchAt, midSwing, animEnd)
	}
	if launchAt > nextSwing-minArrowFlight {
		t.Fatalf("arrow release %g leaves no room for a visible flight before next swing %g", launchAt, nextSwing)
	}

	// A tick at the release point fires the arrow and schedules the hit on arrival,
	// strictly after the release and before the next swing.
	c.mvMu.Lock()
	s.tickMobsLocked(c, launchAt+1e-3)
	nowRel := launchAt + 1e-3
	hitAt := m.hitAt
	c.mvMu.Unlock()
	if got := countProjectiles(t, pkts, mu, m.id); got != 1 {
		t.Fatalf("expected exactly 1 SET_PROJECTILE at the release point, got %d", got)
	}
	if !(hitAt > nowRel && hitAt <= nextSwing) {
		t.Fatalf("hit at %g not scheduled between release %g and next swing %g", hitAt, nowRel, nextSwing)
	}
}

// TestMeleeMobSwingHasNoProjectile: a melee mob (AttackRange 0) strikes in place and
// must NOT emit a projectile (no arrow prefab, and a phantom bolt would look wrong).
func TestMeleeMobSwingHasNoProjectile(t *testing.T) {
	idx, mob := findMob(t, false)
	s, c, m, pkts, mu := setupSwingConn(t, idx, mob, 1) // point-blank melee reach

	c.mvMu.Lock()
	s.tickMobsLocked(c, float64(s.battleTime()))
	swung := m.hitAt > 0
	c.mvMu.Unlock()

	if !swung {
		t.Fatal("melee mob never committed a swing -- test would be vacuous")
	}
	if got := countProjectiles(t, pkts, mu, m.id); got != 0 {
		t.Fatalf("melee mob emitted %d SET_PROJECTILE, want 0", got)
	}
}
