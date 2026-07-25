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

// TestBlackDragonBloodOfDragonThorns verifies «Кровь дракона» (slot 3, NOT the prior
// wiki-based "Крылья тьмы" rewrite): a MELEE attacker that strikes the dragon is slowed
// (move AND attack speed) and takes a magic DoT, all for 3s. Reverted per client-locale
// audit pass 5 -- the previous ambient proximity-aura encoding was the wrong skill.
func TestBlackDragonBloodOfDragonThorns(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_BlackDragon")
	defer cleanup()
	hs := c.huntState
	hs.skillLevel[2] = 1 // «Кровь дракона» learned at rank 1
	hs.hp = 1000

	// Register the on-damaged proc the way world-build does.
	for i, sk := range hs.kit.Skills {
		if sk.Type != "PASSIVE" {
			continue
		}
		for _, op := range sk.Ops {
			if op.Kind == gamedata.OpProc && op.OnDamaged {
				hs.defenseProcs = append(hs.defenseProcs, procState{
					slot: i + 1, chance: op.Chance, ops: op.Ops, meleeOnly: op.MeleeOnly,
				})
			}
		}
	}

	idx := mobIndexByPrefab(t, "Mob_Skeleton_1H_Melee_01") // melee (AttackRange == 0)
	attacker := &mobState{
		id: 2400, mobIdx: idx, mob: gamedata.Mobs()[idx],
		x: 1, y: 0, hp: 100, maxHP: 100, shown: true,
	}
	now := float64(s.battleTime())
	c.mvMu.Lock()
	hs.mobs[attacker.id] = attacker
	hs.tr.add(attacker.id)
	s.hitPlayerLocked(c, attacker, 10, now)
	slowUntil := attacker.st.slowUntil
	atkSlowUntil := attacker.st.atkSlowUntil
	c.mvMu.Unlock()

	if slowUntil <= now {
		t.Fatalf("melee attacker was not move-slowed by the thorns (slowUntil=%g now=%g)", slowUntil, now)
	}
	if atkSlowUntil <= now {
		t.Fatalf("melee attacker was not attack-slowed by the thorns (atkSlowUntil=%g now=%g)", atkSlowUntil, now)
	}
	if len(attacker.st.dots) == 0 {
		t.Fatal("melee attacker did not take the thorns DoT")
	}
}

// TestCcImmunePassiveBlocksCC verifies the OpImmune primitive (Wilfang's «Защитный
// покров» was the pass-4 skill it modelled; a client-locale re-audit in pass 5 reverted
// Wilfang's slot 3 to «Ядовитый укус» instead, so this test now exercises the primitive
// directly against a synthetic kit rather than any specific avatar's authored data --
// OpImmune remains a valid, reusable engine mechanic, just unassigned for now, same as
// it was documented "latent" even while it belonged to Wilfang): the learned OpImmune
// passive blocks an incoming CC once, then goes on cooldown for Dur seconds before it
// can block again.
func TestCcImmunePassiveBlocksCC(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Wilfang")
	defer cleanup()
	hs := c.huntState
	hs.kit = &gamedata.AvatarSkills{Skills: [4]gamedata.Skill{
		{}, {},
		{Ops: []gamedata.Op{{Kind: gamedata.OpImmune, Dur: gamedata.PerLevel{13, 13, 13, 13}, On: "self"}}},
		{},
	}}
	hs.skillLevel[2] = 1 // synthetic immunity passive learned at rank 1
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
