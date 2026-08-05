package battleserver

import "testing"

// TestSharliLavaShacklesLandOnTheGround: SharliSkill3Effect's ground decal
// (VFX_Avtr_DPS_Sharli_skill3_prop01) is baked SELF and carries no VisualEffectTargets of
// its own, so owned to the caster it parented to Sharli instead of the clicked point.
func TestSharliLavaShacklesLandOnTheGround(t *testing.T) {
	if !payloadFxUsesAnchor("Avtr_DPS_Sharli", 3) {
		t.Fatal("«Лавовые оковы» ground decal is still owned to the caster")
	}
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Sharli")
	defer cleanup()
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())
	s.firePayloadLocked(c, payload{slot: 3, level: 1, px: 5, py: 5, hasPos: true}, now)

	hs := c.huntState
	if len(hs.anchorEnds) != 1 {
		t.Fatalf("no anchor scheduled: %d pending", len(hs.anchorEnds))
	}
}

// TestSharliPhoenixReleasesEarly: the phoenix (a FlyEffect prop covering a fixed 30 units
// at 20 units/sec off the prefab -- a real 1.5s flight) used to spawn at PayloadDelay=0.8
// on a 0.9s cast, so it, its damage, and its sound all landed at the very end of the swing
// instead of the start.
func TestSharliPhoenixReleasesEarly(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Sharli")
	defer cleanup()
	hs := c.huntState

	def := hs.skillDef(4)
	if def.PayloadDelay >= def.CastFxDur {
		t.Fatalf("PayloadDelay=%g is not early relative to the %gs cast", def.PayloadDelay, def.CastFxDur)
	}
	if def.PayloadDelay > 0.3 {
		t.Errorf("PayloadDelay=%g is not an early release", def.PayloadDelay)
	}

	m := mkMob(t, 8500, 5, 0)
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	startHP := m.hp

	now := float64(s.battleTime())
	s.firePayloadLocked(c, payload{slot: 4, level: 1, px: 5, py: 0, hasPos: true}, now)
	// Damage lands with the payload, not deferred another 1.5s for the bird's own flight --
	// it is a line sweep like Elgorm/Velial, not a projectile with a flight-based hit.
	if m.hp >= startHP {
		t.Fatal("«Огненный феникс» dealt no damage when its payload fired")
	}
}
