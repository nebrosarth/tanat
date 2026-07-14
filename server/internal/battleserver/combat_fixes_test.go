package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

// TestLineSkillHitsMobsInFront: a line/rift skill (Velial slot 2, AoEWidth 3)
// aimed at a point BEHIND a mob still hits that mob (it lies in the swath from
// the caster toward the aim point), and misses a mob off to the side.
func TestLineSkillHitsMobsInFront(t *testing.T) {
	s, c, _, sx, sy := newNavConn(t) // avatar 13 = Velial
	hs := c.huntState

	front := &mobState{id: 4000, mobIdx: 2, mob: gamedata.Mobs()[2], x: sx + 3, y: sy}      // in the swath
	side := &mobState{id: 4001, mobIdx: 2, mob: gamedata.Mobs()[2], x: sx + 3, y: sy + 5}    // off to the side
	hs.mobs[front.id] = front
	hs.mobs[side.id] = side

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	// Aim at the floor 8m ahead — past the front mob.
	ctx := opCtx{slot: 2, level: 1, hasPos: true, px: sx + 8, py: sy}
	targets := s.damageTargetsLocked(c, ctx, 0)

	hitFront, hitSide := false, false
	for _, m := range targets {
		if m.id == front.id {
			hitFront = true
		}
		if m.id == side.id {
			hitSide = true
		}
	}
	if !hitFront {
		t.Fatal("line skill missed a mob standing between the caster and a far aim point")
	}
	if hitSide {
		t.Fatal("line skill hit a mob outside the swath width")
	}
}

// TestMobHitLandsAndRootsMidSwing: a committed swing hit lands even after the
// target walked out of attack range, and a mob mid-swing holds position.
func TestMobHitLandsAndRootsMidSwing(t *testing.T) {
	s, c, _, sx, sy := newNavConn(t)
	hs := c.huntState
	hs.hp = 1000

	// Mob out of attack range now, but with a hit committed from when it WAS in
	// range: the hit must still land.
	committed := &mobState{id: 5000, mobIdx: 2, mob: gamedata.Mobs()[2],
		x: sx + 5, y: sy, aggro: true, shown: true}
	committed.hitAt = 0.1
	committed.hitDmg = 50
	committed.hitTarget = c.objID

	// Mob mid-swing (animation still playing) must not chase.
	swinging := &mobState{id: 5001, mobIdx: 2, mob: gamedata.Mobs()[2],
		x: sx + 5, y: sy, vx: 3, vy: 0, aggro: true, shown: true}
	swinging.swingDoneAt = 2.0

	hs.mobs[committed.id] = committed
	hs.mobs[swinging.id] = swinging

	c.mvMu.Lock()
	s.tickMobsLocked(c, 1.0) // now=1.0 > hitAt, < swingDoneAt
	c.mvMu.Unlock()

	if hs.hp >= 1000 {
		t.Fatalf("committed hit did not land after the target moved out of range (hp %.0f)", hs.hp)
	}
	if swinging.vx != 0 || swinging.vy != 0 {
		t.Fatalf("mob moved (v=%.1f,%.1f) while its attack animation was playing", swinging.vx, swinging.vy)
	}
}
