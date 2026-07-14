package battleserver

import (
	"net"
	"sync"
	"testing"

	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// avatarByPrefab finds a roster avatar by prefab (test helper; avoids hard-coding ids).
func avatarByPrefab(t *testing.T, prefab string) gamedata.Avatar {
	t.Helper()
	for _, a := range gamedata.Avatars() {
		if a.Prefab == prefab {
			return a
		}
	}
	t.Fatalf("avatar %s not in roster", prefab)
	return gamedata.Avatar{}
}

// newHuntConn builds a minimal solo hunt connection wired to a drained pipe, so
// *Locked helpers can push packets without blocking. Returns the server, conn and
// a cleanup that closes the pipe.
func newHuntConn(t *testing.T, prefab string) (*Server, *conn, func()) {
	t.Helper()
	s := New(session.NewStore())
	srv, cli := net.Pipe()

	var mu sync.Mutex
	r := battleproto.NewReader(cli)
	go func() {
		for {
			if _, err := r.Read(); err != nil {
				return
			}
			mu.Lock()
			mu.Unlock()
		}
	}()

	av := avatarByPrefab(t, prefab)
	c := &conn{Conn: srv}
	c.objID = 1000
	c.x, c.y, c.snapT = 0, 0, s.battleTime()
	hs := &huntState{
		av: av, kit: gamedata.SkillsFor(av),
		mobs: map[int32]*mobState{}, summons: map[int32]*summonState{},
		hp: 100, mana: 200,
	}
	hs.tr.add(c.objID)
	c.huntState = hs
	cleanup := func() {
		c.mvMu.Lock()
		hs.closed = true
		c.mvMu.Unlock()
		srv.Close()
		cli.Close()
	}
	return s, c, cleanup
}

// TestZamaranReviveOnDeath verifies «Возрождение»: a learned OpRevive passive
// resurrects the avatar in place instead of dying, restores hpAdd HP, and then
// enters its internal cooldown (a second death within the cooldown is fatal).
func TestZamaranReviveOnDeath(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	hs := c.huntState
	// Slot-4 ult learned at rank 1; registration (done at world-build) points the
	// revive slot at it.
	hs.skillLevel[3] = 1
	hs.reviveSlot = 4

	now := float64(s.battleTime())
	c.mvMu.Lock()
	hs.hp = 0 // lethal blow just landed
	s.playerDieLocked(c, 42, now)
	dead1 := hs.deadUntil
	hp1 := hs.hp
	ready := hs.reviveReadyAt
	c.mvMu.Unlock()

	if dead1 != 0 {
		t.Fatalf("revive should have prevented death, but deadUntil=%g", dead1)
	}
	if hp1 != 150 { // rank-1 hpAdd, powerMul=1.0 at level 0
		t.Fatalf("revive HP = %g, want 150 (rank-1 hpAdd)", hp1)
	}
	if ready <= now {
		t.Fatalf("revive cooldown not armed: reviveReadyAt=%g now=%g", ready, now)
	}

	// A second death while the revive is on cooldown must be fatal.
	c.mvMu.Lock()
	hs.hp = 0
	s.playerDieLocked(c, 42, now+1)
	dead2 := hs.deadUntil
	c.mvMu.Unlock()
	if dead2 == 0 {
		t.Fatal("second death within revive cooldown should have been fatal")
	}
}

// TestBlackDragonWingsAura verifies «Крылья тьмы»: the learned passive aura pulses
// an attack-speed slow onto nearby enemies with no toggle and no mana upkeep.
func TestBlackDragonWingsAura(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_BlackDragon")
	defer cleanup()
	hs := c.huntState
	hs.skillLevel[2] = 1 // «Крылья тьмы» learned at rank 1

	idx := mobIndexByPrefab(t, "Mob_Skeleton_1H_Melee_01")
	m := &mobState{
		id: 2400, mobIdx: idx, mob: gamedata.Mobs()[idx],
		x: 2, y: 0, hp: 100, shown: true,
	}
	now := float64(s.battleTime())
	c.mvMu.Lock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	s.tickPassiveAurasLocked(c, now)
	slowUntil := m.st.atkSlowUntil
	factor := m.st.atkSlowFactor
	c.mvMu.Unlock()

	if slowUntil <= now {
		t.Fatalf("passive aura did not attack-slow the nearby mob (atkSlowUntil=%g now=%g)", slowUntil, now)
	}
	if factor != 0.8 { // rank-1 speedCoef
		t.Fatalf("attack-slow factor = %g, want 0.8", factor)
	}
}

// TestWilfangCloakBlocksCC verifies «Защитный покров»: the learned OpImmune passive
// blocks an incoming CC once, then goes on cooldown for Dur seconds before it can
// block again.
func TestWilfangCloakBlocksCC(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Wilfang")
	defer cleanup()
	hs := c.huntState
	hs.skillLevel[2] = 1 // «Защитный покров» learned at rank 1
	hs.ccImmuneSlot = 3

	now := float64(s.battleTime())
	c.mvMu.Lock()
	first := s.ccImmuneBlockLocked(c, now)
	ready := hs.ccImmuneReadyAt
	second := s.ccImmuneBlockLocked(c, now) // still on cooldown
	third := s.ccImmuneBlockLocked(c, now+13)
	c.mvMu.Unlock()

	if !first {
		t.Fatal("first CC should have been blocked by the immunity passive")
	}
	if ready <= now {
		t.Fatalf("immunity cooldown not armed: ccImmuneReadyAt=%g now=%g", ready, now)
	}
	if second {
		t.Fatal("immunity should be spent (on cooldown) for the second CC")
	}
	if !third {
		t.Fatal("immunity should be available again after its cooldown elapsed")
	}
}
