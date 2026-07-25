package battleserver

import (
	"net"
	"testing"

	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// newVigilansConn builds a bare Vigilans combat conn with its passives registered as the
// world-state build would (hs.procs), so runProcsLocked drives the on-hit passives.
func newVigilansConn(t *testing.T) (*Server, *conn) {
	t.Helper()
	s := New(session.NewStore())
	srv, cli := net.Pipe()
	t.Cleanup(func() { srv.Close(); cli.Close() })
	go func() { // drain so pushes never block the pipe
		buf := make([]byte, 4096)
		for {
			if _, err := cli.Read(buf); err != nil {
				return
			}
		}
	}()
	vig, ok := gamedata.AvatarByID(20) // Avtr_HK_Vigilans
	if !ok {
		t.Fatal("Vigilans (id 20) missing")
	}
	c := &conn{Conn: srv}
	c.objID = 1000
	c.x, c.y, c.snapT = 0, 0, s.battleTime()
	kit := gamedata.SkillsFor(vig)
	hs := &huntState{
		av: vig, kit: kit,
		mobs: map[int32]*mobState{}, summons: map[int32]*summonState{},
		hp: 500, mana: 200,
	}
	hs.tr.add(c.objID)
	// Learn all four slots so the passives are live (rank 1).
	for i := range hs.skillLevel {
		hs.skillLevel[i] = 1
	}
	// Register on-hit procs exactly like the world-state build (slots 2 & 3 are passives).
	for i, sk := range kit.Skills {
		for _, op := range sk.Ops {
			if op.Kind == gamedata.OpProc && !op.OnDamaged {
				hs.procs = append(hs.procs, procState{slot: i + 1, chance: op.Chance, ops: op.Ops})
			}
		}
	}
	c.huntState = hs
	t.Cleanup(func() { c.mvMu.Lock(); hs.closed = true; c.mvMu.Unlock() })
	return s, c
}

// TestVigilansIsolatedBonusDamage: «Свидание со смертью» (slot 2) adds bonus damage on
// EVERY hit against an ISOLATED target, but NOT when the target stands next to an ally.
func TestVigilansIsolatedBonusDamage(t *testing.T) {
	s, c := newVigilansConn(t)
	hs := c.huntState
	now := float64(s.battleTime())

	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	// The slot-2 proc alone (drop slot 3's random stun to keep this deterministic).
	hs.procs = hs.procs[:0]
	slot2 := hs.kit.Skills[1].Ops
	hs.procs = append(hs.procs, procState{slot: 2, chance: slot2[0].Chance, ops: slot2[0].Ops})

	// Lone target: bonus applies on every hit.
	lone := &mobState{id: 2000, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 100000, x: 40, y: 0}
	hs.mobs[lone.id] = lone
	before := lone.hp
	s.runProcsLocked(c, lone, now)
	if lone.hp >= before {
		t.Fatalf("isolated target took no bonus damage (hp %v -> %v)", before, lone.hp)
	}

	// Packed target: an ally within the isolation radius suppresses the bonus.
	tgt := &mobState{id: 2001, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 100000, x: -40, y: 0}
	ally := &mobState{id: 2002, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 100000, x: -38, y: 0} // ~2 units away
	hs.mobs[tgt.id] = tgt
	hs.mobs[ally.id] = ally
	beforeP := tgt.hp
	s.runProcsLocked(c, tgt, now)
	if tgt.hp != beforeP {
		t.Errorf("target beside an ally still took bonus damage (hp %v -> %v); it must be gated on isolation", beforeP, tgt.hp)
	}
}
