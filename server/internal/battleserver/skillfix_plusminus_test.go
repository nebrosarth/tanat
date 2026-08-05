package battleserver

import (
	"math"
	"testing"

	"tanatserver/internal/gamedata"
)

// TestPlusMinusElectricShockDamageIsPurePosition: two mobs standing at the SAME distance
// from Plus-Minus must take the SAME damage, whether one or five other enemies are also
// caught in the ring -- «в зависимости от их положения», nothing else. The engine used to
// rank hit targets by ORDER (Op.PerTargetGrowth), so the very same mob at the very same
// spot took different damage depending on who else got hit that cast.
func TestPlusMinusElectricShockDamageIsPurePosition(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_PlusMinus")
	defer cleanup()
	hs := c.huntState

	near := mkMob(t, 8400, 1, 0) // close to the epicenter
	far := mkMob(t, 8401, 4, 0)  // near the ring's own edge (Radius=5)
	near.hp, near.maxHP = 100000, 100000
	far.hp, far.maxHP = 100000, 100000
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.mobs[near.id] = near
	hs.mobs[far.id] = far
	hs.tr.add(near.id)
	hs.tr.add(far.id)

	now := float64(s.battleTime())
	s.applyOpsLocked(c, hs.skillDef(1).Ops, opCtx{slot: 1, level: 1}, now)

	nearDmg := 100000 - near.hp
	farDmg := 100000 - far.hp
	if farDmg <= nearDmg {
		t.Fatalf("the farther mob (dmg=%g) did not take more than the closer one (dmg=%g)", farDmg, nearDmg)
	}
	// Linear interpolation from Value=80 at distance 0 to EdgeValue=140 at distance
	// Radius=5, both ends carrying the same spell-power add-on (PerSP=EdgeValueSP=1) --
	// computed here rather than hardcoded so the test does not depend on the test
	// avatar's baseline spell power.
	sp := hs.spellPowerLocked(now)
	wantNear := (80+sp)*0.8 + (140+sp)*0.2 // frac = 1/5
	wantFar := (80+sp)*0.2 + (140+sp)*0.8  // frac = 4/5
	if math.Abs(nearDmg-wantNear) > 0.01 {
		t.Errorf("mob at distance 1 took %g, want %g", nearDmg, wantNear)
	}
	if math.Abs(farDmg-wantFar) > 0.01 {
		t.Errorf("mob at distance 4 took %g, want %g", farDmg, wantFar)
	}

	// Now repeat with a THIRD mob also in range: the near/far mobs' damage must be
	// IDENTICAL to the two-mob case above -- position decides it, not headcount or rank.
	third := mkMob(t, 8402, 1, 0) // same spot as `near`
	third.hp, third.maxHP = 100000, 100000
	hs.mobs[third.id] = third
	hs.tr.add(third.id)
	near.hp, far.hp = 100000, 100000 // reset
	s.applyOpsLocked(c, hs.skillDef(1).Ops, opCtx{slot: 1, level: 1}, now)
	if got := 100000 - near.hp; got != nearDmg {
		t.Errorf("adding a third target changed the near mob's damage: %g -> %g", nearDmg, got)
	}
	if got := 100000 - far.hp; got != farDmg {
		t.Errorf("adding a third target changed the far mob's damage: %g -> %g", farDmg, got)
	}
}

// TestPlusMinusShortCircuitIsAChannel: «Короткое замыкание» is a CHANNELLED beam --
// PlusMinusSkill2's holder plays Cast02 with mLoopAnimation and its payload is a
// SELF_TO_TARGET Lightning that has to be sustained. Authored as an instant hit plus a
// fire-and-forget root, the caster could cast and stroll away while the victim stayed
// pinned and the beam kept playing.
func TestPlusMinusShortCircuitIsAChannel(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_PlusMinus")
	defer cleanup()
	hs := c.huntState

	m := mkMob(t, 7900, 4, 0)
	m.hp, m.maxHP = 100000, 100000 // survive the opening hit; we are testing the grip
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)

	now := float64(s.battleTime())
	hs.noteCastFxLocked(2, 5150, true) // stand in for the live beam/cast fx
	s.applyOpsLocked(c, hs.skillDef(2).Ops, opCtx{slot: 2, level: 1, target: m}, now)

	if len(hs.channels) != 1 {
		t.Fatalf("«Короткое замыкание» is not a channel: %d channels", len(hs.channels))
	}
	ch := hs.channels[0]
	if !ch.holdsTarget {
		t.Fatal("the channel must be marked as holding its victim, or breaking it leaves them pinned")
	}
	if len(ch.fxUIDs) == 0 {
		t.Fatal("the channel adopted no fx: breaking it could not stop the looping beam")
	}
	if m.st.rootUntil <= now || m.st.silenceUntil <= now {
		t.Fatalf("victim not held: root=%g silence=%g", m.st.rootUntil, m.st.silenceUntil)
	}

	c.hasDest = true // the caster walks
	s.tickChannelsLocked(c, now+0.2)
	if len(hs.channels) != 0 {
		t.Fatal("moving must break the channel")
	}
	if m.st.rootUntil > now+0.2 || m.st.silenceUntil > now+0.2 {
		t.Fatalf("the broken beam left the victim held: root=%g silence=%g", m.st.rootUntil, m.st.silenceUntil)
	}
}

// TestChannelBreaksWhenTargetDies: nothing is being channelled into a corpse. Without
// this the beam kept firing at a dead body for the rest of its nominal duration.
func TestChannelBreaksWhenTargetDies(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_PlusMinus")
	defer cleanup()
	hs := c.huntState

	m := mkMob(t, 7910, 4, 0)
	m.hp, m.maxHP = 100000, 100000
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)

	now := float64(s.battleTime())
	s.applyOpsLocked(c, hs.skillDef(2).Ops, opCtx{slot: 2, level: 1, target: m}, now)
	if len(hs.channels) != 1 {
		t.Fatalf("channel not started: %d", len(hs.channels))
	}
	m.dead = true
	s.tickChannelsLocked(c, now+0.2)
	if len(hs.channels) != 0 {
		t.Fatal("the channel outlived its target")
	}
}

// TestPlusMinusShortCircuitSpreadsItsDamage: the beam deals its damage gradually, so an
// interrupted channel deals proportionally less. Two invariants that are easy to break by
// hand-editing the table: the pulses must still SUM to the card's total at every rank, and
// the spell-power term must stay 1× -- a Scale:"magic" op with no explicit PerSP takes the
// caster's whole spell power on EVERY application, so the per-pulse share has to be 1/count
// and the count has to be rank-invariant.
func TestPlusMinusShortCircuitSpreadsItsDamage(t *testing.T) {
	kit := gamedata.SkillsFor(avatarByPrefab(t, "Avtr_Dsb_PlusMinus"))
	sk := kit.Skills[1]
	for _, op := range sk.Ops {
		if op.Kind == gamedata.OpDamage {
			t.Fatal("the damage must live INSIDE the channel, or it all lands up front")
		}
	}
	var ch *gamedata.Op
	for i := range sk.Ops {
		if sk.Ops[i].Kind == gamedata.OpChannel {
			ch = &sk.Ops[i]
		}
	}
	if ch == nil {
		t.Fatal("«Короткое замыкание» is not a channel")
	}
	wantTotal := []float64{75, 125, 175, 225, 275}
	for rank := 1; rank <= 5; rank++ {
		n := ch.Count.At(rank)
		if n != ch.Count.At(1) {
			t.Fatalf("pulse count changes with rank (%g at 1, %g at %d): the spell-power term would scale with rank", ch.Count.At(1), n, rank)
		}
		// The cadence -- not the count -- is what stretches to fill the per-rank duration.
		if got, want := n*ch.Intervals.At(rank), ch.Dur.At(rank); got < want-0.01 || got > want+0.51 {
			t.Errorf("rank %d: %g pulses × %gs spans %gs, but the hold lasts %gs",
				rank, n, ch.Intervals.At(rank), got, want)
		}
		for _, in := range ch.Ops {
			if in.Kind != gamedata.OpDamage {
				continue
			}
			if total := in.Value.At(rank) * n; total != wantTotal[rank-1] {
				t.Errorf("rank %d: %g pulses × %g = %g damage, want the card's %g",
					rank, n, in.Value.At(rank), total, wantTotal[rank-1])
			}
			if sp := in.PerSP * n; sp < 0.999 || sp > 1.001 {
				t.Errorf("rank %d: spell power counted %g×, want exactly 1×", rank, sp)
			}
		}
	}
}

// TestPlusMinusBallLandsOnTheGroundAndDetonatesLater: «создает НА ЗЕМЛЕ шаровую молнию,
// которая взрывается ЧЕРЕЗ 3 СЕКУНДЫ». Both client prefabs are baked SELF, so owning
// either to the caster puts it on the avatar -- which is what happened, with the two
// effects additionally swapped: the explosion played at cast and the ball grew 3s later.
func TestPlusMinusBallLandsOnTheGroundAndDetonatesLater(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_PlusMinus")
	defer cleanup()
	hs := c.huntState

	def := hs.skillDef(4)
	if def.PayloadFx != "PlusMinusSkill4BombEffect" {
		t.Errorf("the planted object is %q, want the BALL (…BombEffect)", def.PayloadFx)
	}
	if def.ImpactFx != "PlusMinusSkill4Effect" {
		t.Errorf("the detonation is %q, want …Skill4Effect", def.ImpactFx)
	}
	if def.PayloadFlight != 3 {
		t.Errorf("fuse = %gs, want the card's 3s", def.PayloadFlight)
	}
	if def.PayloadDelay != 0 {
		t.Errorf("PayloadDelay = %g: the 3s is a fuse, not a wind-up -- the ball must be visible for all of it", def.PayloadDelay)
	}

	m := mkMob(t, 7920, 6, 0)
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	startHP := m.hp

	now := float64(s.battleTime())
	s.firePayloadLocked(c, payload{slot: 4, level: 1, px: 6, py: 0, hasPos: true}, now)

	if m.hp != startHP {
		t.Fatalf("the ball damaged on being PLANTED: hp %g -> %g", startHP, m.hp)
	}
	if len(hs.payloads) != 1 {
		t.Fatalf("no detonation scheduled: %d pending", len(hs.payloads))
	}
	det := hs.payloads[0]
	if det.anchor == 0 {
		t.Fatal("the ball has no ground anchor: it would ride the caster instead of sitting where it was cast")
	}
	if got := det.at - now; got != 3 {
		t.Fatalf("detonation in %.3gs, want 3s", got)
	}

	s.firePayloadLocked(c, det, det.at)
	if m.hp >= startHP {
		t.Fatalf("the detonation dealt nothing: hp stayed at %g", m.hp)
	}
}
