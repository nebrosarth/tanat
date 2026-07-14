package battleserver

import (
	"net"
	"testing"

	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// setupVigilansForResume builds a Vigilans conn with a single high-HP mob in cast
// and aggro range, returning the server, conn, huntState and mob. Caller drives the
// skill/resume paths under mvMu and closes the hunt when done.
func setupVigilansForResume(t *testing.T) (*Server, *conn, *huntState, *mobState) {
	t.Helper()
	s := New(session.NewStore())
	srv, cli := net.Pipe()
	t.Cleanup(func() { srv.Close(); cli.Close() })
	go func() {
		buf := make([]byte, 4096)
		for {
			if _, err := cli.Read(buf); err != nil {
				return
			}
		}
	}()

	vig, ok := gamedata.AvatarByID(20) // Vigilans: slot 1 is an ENEMY-targeted cast
	if !ok {
		t.Fatal("Vigilans (id 20) missing")
	}
	c := &conn{Conn: srv}
	c.objID = 1000
	c.x, c.y, c.snapT = 0, 0, s.battleTime()
	hs := &huntState{
		av:      vig,
		kit:     gamedata.SkillsFor(vig),
		mobs:    map[int32]*mobState{},
		summons: map[int32]*summonState{},
		hp:      500, mana: 200,
	}
	hs.skillLevel[0] = 1
	hs.tr.add(c.objID)
	mob := &mobState{id: 2000, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 5000}
	mob.x, mob.y = 5, 0 // within Vigilans S1 range (10) and aggro range (9)
	hs.tr.add(mob.id)
	hs.mobs[mob.id] = mob
	c.huntState = hs
	return s, c, hs, mob
}

// TestAbilityCastTagsResumeTarget: an ENEMY-targeted cast records its victim on the
// action so the tick can roll the avatar back into auto-attack on it.
func TestAbilityCastTagsResumeTarget(t *testing.T) {
	s, c, hs, mob := setupVigilansForResume(t)
	c.mvMu.Lock()
	s.startSkillOrderLocked(c, 1, mob.id, mob.x, mob.y, true) // in range -> execCast
	var ad *actionDone
	for i := range hs.actionDones {
		if hs.actionDones[i].order {
			ad = &hs.actionDones[i]
		}
	}
	hs.closed = true
	c.mvMu.Unlock()

	if ad == nil {
		t.Fatal("cast scheduled no order action-done")
	}
	if ad.resumeTarget != mob.id {
		t.Errorf("action resumeTarget = %d, want the cast victim %d", ad.resumeTarget, mob.id)
	}
}

// TestResumeAutoAttackReengages: resumeAutoAttackLocked re-engages the preferred
// target, falls back to the nearest mob, and stays out of the way when the avatar
// is already busy or has nothing to hit.
func TestResumeAutoAttackReengages(t *testing.T) {
	s, c, hs, mob := setupVigilansForResume(t)
	c.mvMu.Lock()
	defer func() { hs.closed = true; c.mvMu.Unlock() }()
	now := float64(s.battleTime())

	// Preferred target: re-engages exactly it.
	s.resumeAutoAttackLocked(c, now, mob.id)
	if hs.attackTarget != mob.id {
		t.Fatalf("resume did not re-engage preferred target: attackTarget=%d want %d", hs.attackTarget, mob.id)
	}

	// Already attacking -> no-op (does not switch away).
	s.stopAttackLocked(c, true)
	hs.attackTarget = 999
	s.resumeAutoAttackLocked(c, now, mob.id)
	if hs.attackTarget != 999 {
		t.Errorf("resume overrode an active auto-attack (attackTarget=%d, want left at 999)", hs.attackTarget)
	}

	// Idle, no preferred (0): falls back to the nearest mob.
	hs.attackTarget = 0
	s.resumeAutoAttackLocked(c, now, 0)
	if hs.attackTarget != mob.id {
		t.Errorf("resume did not fall back to the nearest mob: attackTarget=%d want %d", hs.attackTarget, mob.id)
	}

	// Mid manual-move -> leave the player's movement alone.
	s.stopAttackLocked(c, true)
	hs.attackTarget = 0
	c.hasDest = true
	s.resumeAutoAttackLocked(c, now, mob.id)
	if hs.attackTarget != 0 {
		t.Errorf("resume hijacked a manual move (attackTarget=%d, want 0)", hs.attackTarget)
	}
	c.hasDest = false

	// No enemy in range: nothing to do.
	mob.x, mob.y = 500, 500 // shove it far outside aggro range
	hs.attackTarget = 0
	s.resumeAutoAttackLocked(c, now, 0)
	if hs.attackTarget != 0 {
		t.Errorf("resume acquired an out-of-range mob (attackTarget=%d, want 0)", hs.attackTarget)
	}
}

// TestServerDrivenAutoAttackChainsOnKill: when auto-attack is server-driven (a
// post-cast resume, so the client won't self-retarget), killing the target rolls the
// avatar straight onto the next nearby mob -- and keeps doing so. A client-driven
// attack instead stops on kill and leaves retargeting to the client's DEFENCE loop.
func TestServerDrivenAutoAttackChainsOnKill(t *testing.T) {
	s, c, hs, mob := setupVigilansForResume(t)
	c.mvMu.Lock()
	defer func() { hs.closed = true; c.mvMu.Unlock() }()
	now := float64(s.battleTime())

	mob.hp = 10
	mob.x, mob.y = 1, 0 // in melee reach
	mob2 := &mobState{id: 2001, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 10, x: 2, y: 0}
	mob3 := &mobState{id: 2002, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 10, x: 3, y: 0}
	hs.tr.add(mob2.id)
	hs.tr.add(mob3.id)
	hs.mobs[mob2.id] = mob2
	hs.mobs[mob3.id] = mob3

	// Server-driven resume onto mob1.
	s.resumeAutoAttackLocked(c, now, mob.id)
	if hs.attackTarget != mob.id || !hs.autoResumed {
		t.Fatalf("precondition: attackTarget=%d autoResumed=%v, want %d/true", hs.attackTarget, hs.autoResumed, mob.id)
	}

	// Kill mob1 -> the server chains onto the next mob rather than going idle.
	s.hitMobLocked(c, mob, 999, c.objID)
	if hs.attackTarget == 0 || hs.attackTarget == mob.id {
		t.Fatalf("server-driven kill did not chain to the next mob: attackTarget=%d", hs.attackTarget)
	}
	chained := hs.attackTarget

	// Kill that one too -> chain to the last mob (proves it keeps going, not a one-shot).
	s.hitMobLocked(c, hs.mobs[chained], 999, c.objID)
	if hs.attackTarget == 0 || hs.attackTarget == chained {
		t.Fatalf("chain stopped after one retarget: attackTarget=%d", hs.attackTarget)
	}

	// Contrast: a client-driven attack stops on kill (client retargets, not the server).
	last := hs.mobs[hs.attackTarget]
	hs.autoResumed = false // as a client-issued attack would leave it
	s.hitMobLocked(c, last, 999, c.objID)
	if hs.attackTarget != 0 {
		t.Errorf("client-driven kill should stop server-side (attackTarget=%d, want 0)", hs.attackTarget)
	}
}
