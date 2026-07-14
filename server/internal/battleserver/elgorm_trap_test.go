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

// TestElgormDefiledGroundAnchoredAtPoint: «Оскверненная почва» (slot 3) uses a
// SELF-mode ground fx that would parent to (and trail) the caster. The server must
// instead spawn an invisible stationary anchor at the cast point and own the fx to
// that, so the hazard stays put. Asserts an anchor object is created (CREATE_OBJECT
// with the anchor proto) and the trap fx's EFFECT_START owner is the anchor, not the
// avatar.
func TestElgormDefiledGroundAnchoredAtPoint(t *testing.T) {
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

	el, ok := gamedata.AvatarByID(31) // Elgorm
	if !ok {
		t.Fatal("Elgorm (id 31) missing")
	}
	c := &conn{Conn: srv}
	c.objID = 1000
	c.x, c.y, c.snapT = 0, 0, s.battleTime()
	kit := gamedata.SkillsFor(el)
	hs := &huntState{
		av: el, kit: kit,
		mobs: map[int32]*mobState{}, summons: map[int32]*summonState{},
		hp: 500, mana: 300,
	}
	hs.tr.add(c.objID)
	c.huntState = hs
	defer func() { c.mvMu.Lock(); hs.closed = true; c.mvMu.Unlock() }()

	trapOp := kit.Skills[2].Ops[0]
	if trapOp.Kind != gamedata.OpTrap {
		t.Fatalf("expected Elgorm slot 3 op[0] to be OpTrap, got %q", trapOp.Kind)
	}

	now := float64(s.battleTime())
	c.mvMu.Lock()
	// Cast the trap aimed at a ground point away from the caster (0,0).
	s.applyOpsLocked(c, []gamedata.Op{trapOp}, opCtx{slot: 3, level: 1, hasPos: true, px: 5, py: 0}, now)
	if len(hs.traps) != 1 {
		c.mvMu.Unlock()
		t.Fatalf("expected 1 armed trap, got %d", len(hs.traps))
	}
	anchor := hs.traps[0].anchor
	c.mvMu.Unlock()

	if anchor == 0 {
		t.Fatal("trap has no anchor object -- the SELF-mode fx would follow the avatar")
	}

	time.Sleep(50 * time.Millisecond) // let the reader drain the pushes
	mu.Lock()
	defer mu.Unlock()
	var sawCreate, sawFx bool
	for _, p := range pkts {
		switch p.Cmd {
		case battleproto.CmdCreateObject:
			if proto, _ := p.Args.GetInt("proto"); proto == trapAnchorProtoID {
				if id, _ := p.Args.GetInt("id"); id == anchor {
					sawCreate = true
				}
			}
		case battleproto.CmdEffectStart:
			if fx, _ := p.Args.GetString("fx"); fx == "ElgormSkill3Effect1" {
				owner, _ := p.Args.GetInt("owner")
				if owner == c.objID {
					t.Fatalf("trap fx owner=%d is the avatar -- a SELF fx would follow it", owner)
				}
				if owner != anchor {
					t.Fatalf("trap fx owner=%d, want the stationary anchor %d", owner, anchor)
				}
				sawFx = true
			}
		}
	}
	if !sawCreate {
		t.Fatalf("no CREATE_OBJECT for the trap anchor (proto %d, id %d)", trapAnchorProtoID, anchor)
	}
	if !sawFx {
		t.Fatal("no EFFECT_START for the trap fx ElgormSkill3Effect1")
	}
}

// TestElgormDefiledGroundAnchorRemovedOnExpiry: when the hazard's lifetime elapses
// with no trigger, its anchor object is deleted (DELETE_OBJECT) so it doesn't leak.
func TestElgormDefiledGroundAnchorRemovedOnExpiry(t *testing.T) {
	s := New(session.NewStore())
	srv, cli := net.Pipe()
	defer srv.Close()
	defer cli.Close()
	go func() { // drain
		buf := make([]byte, 4096)
		for {
			if _, err := cli.Read(buf); err != nil {
				return
			}
		}
	}()

	el, _ := gamedata.AvatarByID(31)
	c := &conn{Conn: srv}
	c.objID = 1000
	c.x, c.y, c.snapT = 0, 0, s.battleTime()
	kit := gamedata.SkillsFor(el)
	hs := &huntState{
		av: el, kit: kit,
		mobs: map[int32]*mobState{}, summons: map[int32]*summonState{},
		hp: 500, mana: 300,
	}
	hs.tr.add(c.objID)
	c.huntState = hs
	defer func() { c.mvMu.Lock(); hs.closed = true; c.mvMu.Unlock() }()

	trapOp := kit.Skills[2].Ops[0]
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	s.applyOpsLocked(c, []gamedata.Op{trapOp}, opCtx{slot: 3, level: 1, hasPos: true, px: 5, py: 0}, now)
	anchor := hs.traps[0].anchor
	if anchor == 0 || hs.tr.index(anchor) < 0 {
		t.Fatalf("anchor %d not tracked after cast", anchor)
	}

	// Tick past the trap lifetime with no mob in range -> expiry deletes the anchor.
	life := trapOp.Lifetime.At(1)
	s.tickTrapsLocked(c, now+life+1)
	if len(hs.traps) != 0 {
		t.Fatalf("trap survived past its lifetime (%d left)", len(hs.traps))
	}
	if hs.tr.index(anchor) >= 0 {
		t.Fatalf("anchor %d still tracked after the trap expired -- it leaked", anchor)
	}
}
