package battleserver

import (
	"math"
	"testing"

	"tanatserver/internal/gamedata"
)

// TestBlackDragonWingBeatTicksFiveTimesPerSecond: explicitly requested over the card's own
// literal "каждую секунду" wording -- 5 pulses/sec instead of 1, same total dps and the
// same total spell-power contribution per second (a magic-scale op with no PerSP would
// otherwise add the FULL spell power on every one of the five pulses).
func TestBlackDragonWingBeatTicksFiveTimesPerSecond(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_BlackDragon")
	defer cleanup()
	hs := c.huntState

	def := hs.skillDef(4)
	var ch *gamedata.Op
	for i := range def.Ops {
		if def.Ops[i].Kind == gamedata.OpChannel {
			ch = &def.Ops[i]
		}
	}
	if ch == nil {
		t.Fatal("«Взмах погибели» is not a channel")
	}
	if ch.Interval != 0.2 {
		t.Fatalf("interval = %g, want 0.2 (5 pulses/sec)", ch.Interval)
	}
	var dmgOp *gamedata.Op
	for i := range ch.Ops {
		if ch.Ops[i].Kind == gamedata.OpDamage {
			dmgOp = &ch.Ops[i]
		}
	}
	if dmgOp == nil {
		t.Fatal("no damage op inside the channel")
	}
	wantDPS := []float64{13, 16, 19, 22} // PVP balance redesign values (was 30, 42, 54, 68)
	for rank := 1; rank <= 4; rank++ {
		perPulse := dmgOp.Value.At(rank)
		if total := perPulse * 5; math.Abs(total-wantDPS[rank-1]) > 0.01 {
			t.Errorf("rank %d: 5 pulses × %g = %g, want the card's %g dps", rank, perPulse, total, wantDPS[rank-1])
		}
	}
	if sp := dmgOp.PerSP * 5; math.Abs(sp-1) > 0.001 {
		t.Errorf("5 pulses carry %g× spell power total, want exactly 1×/sec", sp)
	}

	m := mkMob(t, 8801, 1, 0)
	m.hp, m.maxHP = 100000, 100000
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)

	now := float64(s.battleTime())
	s.applyOpsLocked(c, def.Ops, opCtx{slot: 4, level: 1}, now)
	ticks := 0
	for i := 1; i <= 5 && len(hs.channels) > 0; i++ {
		before := m.hp
		s.tickChannelsLocked(c, now+0.2*float64(i))
		if m.hp < before {
			ticks++
		}
	}
	if ticks < 4 {
		t.Fatalf("only %d damage ticks landed in the first second, want ~5", ticks)
	}
}

// TestBlackDragonWingBeatSurvivesMovementAndStun: «Взмах погибели» is a self-buff rage --
// the dragon flaps and pulses AoE damage around him for his own 8s window, not something
// that requires him to hold still or go uninterrupted (unlike Abominator's «Пожирание»
// drain, the model self/unit channels break for by default).
func TestBlackDragonWingBeatSurvivesMovementAndStun(t *testing.T) {
	if !channelSustainsThroughDisruption("Avtr_DPS_BlackDragon", 4) {
		t.Fatal("«Взмах погибели» is not marked as surviving movement/stun")
	}
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_BlackDragon")
	defer cleanup()
	hs := c.huntState

	m := mkMob(t, 8800, 1, 0)
	m.hp, m.maxHP = 100000, 100000
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)

	now := float64(s.battleTime())
	s.applyOpsLocked(c, hs.skillDef(4).Ops, opCtx{slot: 4, level: 1}, now)
	if len(hs.channels) != 1 {
		t.Fatalf("«Взмах погибели» did not start a channel: %d channels", len(hs.channels))
	}

	// Both a stun AND movement, at once -- either alone breaks an ordinary self channel.
	hs.st.stunUntil = now + 5
	c.hasDest = true
	s.tickChannelsLocked(c, now+0.2)
	if len(hs.channels) == 0 {
		t.Fatal("the wing-beat broke on stun+movement; it should keep pulsing")
	}

	// Death still ends it -- this flag only exempts stun/movement, nothing else.
	hs.deadUntil = now + 10
	s.tickChannelsLocked(c, now+0.4)
	if len(hs.channels) != 0 {
		t.Fatal("death did not end the channel")
	}
}

// TestAbominatorDrainStillBreaksOnMovement guards the default the new flag carves an
// exception out of: an ordinary self/unit channel still breaks on movement.
func TestAbominatorDrainStillBreaksOnMovement(t *testing.T) {
	if channelSustainsThroughDisruption("Avtr_HK_Abominator", 4) {
		t.Fatal("Abominator's drain must not be exempted from the break rule")
	}
}
