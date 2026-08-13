package battleserver

import "testing"

// TestTangrenSteelStormAllowsMovementButLocksAbilities models the Juggernaut
// Blade Fury contract: the spin is a five-second movement channel, but another
// ability cannot be started until the spin ends.
func TestTangrenSteelStormAllowsMovementButLocksAbilities(t *testing.T) {
	s, c, _, _, _ := newNavConnAvatar(t, 18) // Tangren
	hs := c.huntState

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())
	hs.skillLevel[0] = 1
	hs.skillLevel[2] = 1 // Healing Totem, used as the blocked follow-up ability.

	if !s.startSkillOrderLocked(c, 1, -1, 0, 0, true) {
		t.Fatal("Tangren's first skill was not accepted")
	}
	if len(hs.channels) != 1 {
		t.Fatalf("Tangren's first skill created %d channels, want 1", len(hs.channels))
	}
	if !channelSustainsThroughDisruption(hs.av.Prefab, 1) ||
		!channelAllowsMovement(hs.av.Prefab, 1) {
		t.Fatal("Tangren's first skill is not configured as a movement channel")
	}
	if hs.castLockUntil > now {
		t.Fatalf("Tangren's spin still has a movement lock until %g", hs.castLockUntil)
	}
	if c.movementBlockedLocked(now) {
		t.Fatal("Tangren's spin still blocks movement")
	}

	// Simulate the bot/player taking a movement order during the spin. The
	// channel must continue ticking while the route is active.
	c.hasDest = true
	s.tickChannelsLocked(c, now+0.2)
	if len(hs.channels) != 1 {
		t.Fatal("Tangren's spin was interrupted by movement")
	}

	manaBefore := hs.mana
	cdBefore := hs.cooldownUntil[2]
	if s.startSkillOrderLocked(c, 3, -1, 0, 0, false) {
		t.Fatal("Tangren was allowed to start another ability during the spin")
	}
	if hs.mana != manaBefore {
		t.Fatalf("blocked ability changed mana from %g to %g", manaBefore, hs.mana)
	}
	if hs.cooldownUntil[2] != cdBefore {
		t.Fatalf("blocked ability changed cooldown from %g to %g", cdBefore, hs.cooldownUntil[2])
	}

}
