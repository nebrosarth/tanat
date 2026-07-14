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

// TestMobGiveUpClosesInFlightSwing reproduces the "mob walks home still swinging"
// bug: a mob killed the player mid-swing (swingDoneAt still in the future) drops
// straight into the leash/give-up branch, where resetInFlight would zero swingDoneAt
// WITHOUT ever telling the client the attack ended -- so the attack clip loops all
// the way back to spawn. The give-up path must broadcast the ACTION_DONE first.
func TestMobGiveUpClosesInFlightSwing(t *testing.T) {
	s := New(session.NewStore())
	srv, cli := net.Pipe()
	defer srv.Close()
	defer cli.Close()

	var mu sync.Mutex
	var pkts []battleproto.Packet
	r := battleproto.NewReader(cli)
	go func() {
		for {
			p, err := r.Read()
			if err != nil {
				return
			}
			mu.Lock()
			pkts = append(pkts, p)
			mu.Unlock()
		}
	}()

	idx := mobIndexByPrefab(t, "Mob_Skeleton_1H_Melee_01")
	mob := gamedata.Mobs()[idx]

	av, _ := gamedata.AvatarByID(13)
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
	defer func() { c.mvMu.Lock(); hs.closed = true; c.mvMu.Unlock() }()

	now := float64(s.battleTime())
	// Mob is shown and aggro, mid-swing with the swing NOT yet due to close
	// (swingDoneAt far in the future) -- exactly the state a player kill leaves it in.
	m := &mobState{
		id: 2300, mobIdx: idx, mob: mob,
		x: 1, y: 0, spawnX: 1, spawnY: 0,
		hp: mob.Health, shown: true, aggro: true,
		swingDoneAt: now + 5,
	}
	c.mvMu.Lock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	// The player is dead: mobTargetLocked skips dead members, so the mob has no
	// target (obj 0) and gives up this tick -- the return-home transition.
	hs.deadUntil = now + respawnDelay

	s.tickMobsLocked(c, now+0.2)
	sd := m.swingDoneAt
	ret := m.returning
	c.mvMu.Unlock()

	if !ret {
		t.Fatal("mob with no target should have started returning home")
	}
	if sd != 0 {
		t.Fatalf("give-up should have closed the swing (swingDoneAt=%g, want 0)", sd)
	}

	time.Sleep(50 * time.Millisecond) // let the reader drain
	mu.Lock()
	defer mu.Unlock()
	sawDone := false
	for _, p := range pkts {
		if p.Cmd != battleproto.CmdActionDone {
			continue
		}
		if id, _ := p.Args.GetInt("id"); id != m.id {
			continue
		}
		if act, _ := p.Args.GetInt("action"); act == mobAttackProtoID(m.mobIdx) {
			sawDone = true
		}
	}
	if !sawDone {
		t.Fatal("no ACTION_DONE for the mob's attack -- the swing clip loops as it walks home")
	}
}
