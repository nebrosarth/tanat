package battleserver

import (
	"net"
	"testing"
	"time"

	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// setupArcherForHit builds a ranged avatar (Miriam) with one high-HP mob in reach and
// a drained socket, for exercising the basic-attack hit schedulers directly.
func setupArcherForHit(t *testing.T) (*Server, *conn, *huntState, *mobState) {
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

	miriam, ok := gamedata.AvatarByID(25) // Miriam: ranged, fires a basic-attack projectile
	if !ok {
		t.Fatal("Miriam (id 25) missing")
	}
	c := &conn{Conn: srv}
	c.objID = 1000
	c.x, c.y, c.snapT = 0, 0, s.battleTime()
	hs := &huntState{
		av: miriam, kit: gamedata.SkillsFor(miriam),
		mobs: map[int32]*mobState{}, summons: map[int32]*summonState{},
		hp: 500, mana: 200, hasProjectile: true,
	}
	hs.tr.add(c.objID)
	mob := &mobState{id: 2000, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 5000, x: 3, y: 0}
	hs.tr.add(mob.id)
	hs.mobs[mob.id] = mob
	c.huntState = hs
	return s, c, hs, mob
}

func mobHP(c *conn, mob *mobState) float64 {
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	return mob.hp
}

// TestProjectileHitCommitsAfterCancel locks the fix: once the bolt is in flight, its
// hit must land even if the player cancels/retargets the attack (bumping attackSeq)
// before it arrives -- a visibly-connecting arrow that deals no damage is the bug.
func TestProjectileHitCommitsAfterCancel(t *testing.T) {
	s, c, hs, mob := setupArcherForHit(t)
	start := mob.hp

	c.mvMu.Lock()
	hs.attackSeq = 7
	s.scheduleProjectileHitLocked(c, mob.id, 10*time.Millisecond)
	hs.attackSeq++ // player cancels the attack while the bolt is mid-flight
	c.mvMu.Unlock()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) && mobHP(c, mob) >= start {
		time.Sleep(5 * time.Millisecond)
	}
	if got := mobHP(c, mob); got >= start {
		t.Fatalf("committed projectile hit dealt no damage after cancel: mob hp %.0f (start %.0f)", got, start)
	}

	c.mvMu.Lock()
	hs.closed = true
	c.mvMu.Unlock()
}

// TestMeleeHitDroppedOnCancel is the contrast: a melee swing hit stays gated on the
// attack being live, so cancelling before it connects deals no damage.
func TestMeleeHitDroppedOnCancel(t *testing.T) {
	s, c, hs, mob := setupArcherForHit(t)
	hs.hasProjectile = false // treat as a melee swing for this check
	start := mob.hp

	c.mvMu.Lock()
	hs.attackSeq = 7
	s.scheduleHitLocked(c, 7, mob.id, 10*time.Millisecond)
	hs.attackSeq++ // cancelled before the swing connects
	c.mvMu.Unlock()

	time.Sleep(150 * time.Millisecond) // long enough for the timer to have fired
	if got := mobHP(c, mob); got < start {
		t.Errorf("melee hit landed despite cancel: mob hp %.0f (start %.0f)", got, start)
	}

	c.mvMu.Lock()
	hs.closed = true
	c.mvMu.Unlock()
}
