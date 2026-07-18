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

// newTitanidCapture builds a Titanid conn whose pushed battle packets are captured.
func newTitanidCapture(t *testing.T) (*Server, *conn, *huntState, *sync.Mutex, *[]battleproto.Packet) {
	t.Helper()
	s := New(session.NewStore())
	srv, cli := net.Pipe()
	t.Cleanup(func() { srv.Close(); cli.Close() })

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

	ti, ok := gamedata.AvatarByID(14) // Titanid
	if !ok {
		t.Fatal("Titanid (id 14) missing")
	}
	c := &conn{Conn: srv}
	c.objID = 1000
	c.x, c.y, c.snapT = 0, 0, s.battleTime()
	hs := &huntState{
		av: ti, kit: gamedata.SkillsFor(ti),
		mobs: map[int32]*mobState{}, summons: map[int32]*summonState{},
		hp: 2000, mana: 300,
	}
	for i := range hs.skillLevel {
		hs.skillLevel[i] = 1
	}
	hs.tr.add(c.objID)
	c.huntState = hs
	t.Cleanup(func() { c.mvMu.Lock(); hs.closed = true; c.mvMu.Unlock() })
	return s, c, hs, &mu, &pkts
}

// TestTitanidQuakeFxAnchoredAtPoint: «Землетрясение» (slot 1) is a SELF-baked ground
// fx, so its payload must be pinned to a stationary Dummy anchor at the cast point
// (CREATE_OBJECT of the anchor proto + EFFECT_START owned by the anchor, not the
// avatar) or it would trail the moving Titanid.
func TestTitanidQuakeFxAnchoredAtPoint(t *testing.T) {
	s, c, hs, mu, pkts := newTitanidCapture(t)

	now := float64(s.battleTime())
	c.mvMu.Lock()
	s.firePayloadLocked(c, payload{slot: 1, level: 1, hasPos: true, px: 5, py: 0}, now)
	var anchor int32
	if len(hs.anchorEnds) == 1 {
		anchor = hs.anchorEnds[0].id
	}
	c.mvMu.Unlock()

	if anchor == 0 {
		t.Fatal("skill 1 payload fx was not anchored to a Dummy -- it would follow the avatar")
	}

	time.Sleep(50 * time.Millisecond)
	mu.Lock()
	defer mu.Unlock()
	var sawCreate, sawFx bool
	for _, p := range *pkts {
		switch p.Cmd {
		case battleproto.CmdCreateObject:
			if proto, _ := p.Args.GetInt("proto"); proto == trapAnchorProtoID {
				if id, _ := p.Args.GetInt("id"); id == anchor {
					sawCreate = true
				}
			}
		case battleproto.CmdEffectStart:
			if fx, _ := p.Args.GetString("fx"); fx == "TitanidSkill1Effect1" {
				owner, _ := p.Args.GetInt("owner")
				if owner == c.objID {
					t.Fatalf("quake fx owner=%d is the avatar -- a SELF fx would follow it", owner)
				}
				if owner != anchor {
					t.Fatalf("quake fx owner=%d, want the stationary anchor %d", owner, anchor)
				}
				sawFx = true
			}
		}
	}
	if !sawCreate {
		t.Fatalf("no CREATE_OBJECT for the quake anchor (proto %d, id %d)", trapAnchorProtoID, anchor)
	}
	if !sawFx {
		t.Fatal("no EFFECT_START for the quake fx TitanidSkill1Effect1")
	}
}

// TestStoneSkinRegistersAsOnDamaged: «Каменная кожа» (slot 3) must register as an
// ON-DAMAGED proc (defenseProcs), never as an on-hit proc -- Titanid hardens when he
// is struck, not when he strikes. Mirrors the world-build registration split.
func TestStoneSkinRegistersAsOnDamaged(t *testing.T) {
	ti, _ := gamedata.AvatarByID(14)
	kit := gamedata.SkillsFor(ti)

	var procs, defense []procState
	for i, sk := range kit.Skills {
		if sk.Type != "PASSIVE" {
			continue
		}
		for _, op := range sk.Ops {
			if op.Kind == gamedata.OpProc {
				pr := procState{slot: i + 1, chance: op.Chance, ops: op.Ops}
				if op.OnDamaged {
					defense = append(defense, pr)
				} else {
					procs = append(procs, pr)
				}
			}
		}
	}
	if len(defense) != 1 || defense[0].slot != 3 {
		t.Fatalf("Stone Skin (slot 3) not registered as the on-damaged proc: defense=%+v", defense)
	}
	for _, p := range procs {
		if p.slot == 3 {
			t.Error("Stone Skin registered as an on-HIT proc -- it must trigger on taking damage")
		}
	}
}

// TestStoneSkinStacksArmorWhenStruck: taking a (survived) hit rolls the on-damaged
// proc, adding phys_armor; a second hit stacks more. Dealing a hit (the on-attack
// path) does not.
func TestStoneSkinStacksArmorWhenStruck(t *testing.T) {
	s, c, hs, _, _ := newTitanidCapture(t)
	kit := hs.kit

	// Register Stone Skin the way world-build does.
	for i, sk := range kit.Skills {
		if sk.Type != "PASSIVE" {
			continue
		}
		for _, op := range sk.Ops {
			if op.Kind == gamedata.OpProc && op.OnDamaged {
				hs.defenseProcs = append(hs.defenseProcs, procState{slot: i + 1, chance: op.Chance, ops: op.Ops})
			}
		}
	}

	mob := &mobState{id: 2000, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 5000, x: 1, y: 0}
	hs.tr.add(mob.id)
	hs.mobs[mob.id] = mob

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())

	if a := hs.st.modSum(now, "phys_armor"); a != 0 {
		t.Fatalf("precondition: phys_armor mod = %.0f, want 0", a)
	}

	// Take a hit -> armor stacks.
	s.hitPlayerLocked(c, mob, 40, now)
	first := hs.st.modSum(now, "phys_armor")
	if first <= 0 {
		t.Fatalf("Stone Skin did not stack armor on taking damage: phys_armor = %.0f", first)
	}

	// Take another -> stacks further.
	s.hitPlayerLocked(c, mob, 40, now)
	if second := hs.st.modSum(now, "phys_armor"); second <= first {
		t.Errorf("Stone Skin did not accumulate stacks: after 2 hits %.0f, after 1 hit %.0f", second, first)
	}
}
