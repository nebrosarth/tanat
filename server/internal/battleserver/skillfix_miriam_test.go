package battleserver

import "testing"

// TestMiriamEnchantedArrowsIsAToggle: «Зачарованные стрелы» is a stance the client
// describes as «ПРИ ДЕЙСТВИИ НАВЫКА» with no duration anywhere in six locale fields, only
// a per-shot mana cost -- a toggle, not a timed active.
func TestMiriamEnchantedArrowsIsAToggle(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Miriam")
	defer cleanup()
	hs := c.huntState
	hs.skillLevel[2] = 1
	hs.mana = 1000

	def := hs.skillDef(3)
	if def.Type != "TOGGLE" {
		t.Fatalf("«Зачарованные стрелы» Type = %q, want TOGGLE", def.Type)
	}

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	s.toggleSkillLocked(c, 3)
	if !hs.toggleOn[2] {
		t.Fatal("toggling on did not set toggleOn")
	}
	if hs.manaShotSlot != 3 || hs.manaShotCost != 6 || hs.manaShotDmg != 25 {
		t.Fatalf("mana-shot window not armed by the toggle: slot=%d cost=%g dmg=%g", hs.manaShotSlot, hs.manaShotCost, hs.manaShotDmg)
	}
	now := float64(s.battleTime())
	if hs.manaShotUntil < now+1e8 {
		t.Fatal("mana-shot window has a short horizon: it would expire like the old timed active")
	}

	s.toggleSkillLocked(c, 3) // toggle off
	if hs.toggleOn[2] {
		t.Fatal("second click did not toggle off")
	}
	if hs.manaShotSlot != 0 || hs.manaShotUntil != 0 {
		t.Fatalf("toggle-off did not clear the mana-shot window: slot=%d until=%g", hs.manaShotSlot, hs.manaShotUntil)
	}
}

// TestMiriamKillingVolleyLandsAtTheClickedPoint: MiriamSkill4Effect's gfx is baked SELF
// (a PrefabTimeSpawn dropping arrows), so it rained down on Miriam instead of the clicked
// area.
func TestMiriamKillingVolleyLandsAtTheClickedPoint(t *testing.T) {
	if !payloadFxUsesAnchor("Avtr_DPS_Miriam", 4) {
		t.Fatal("«Убийственный залп» arrow-rain spawner is still owned to the caster")
	}
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Miriam")
	defer cleanup()
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())
	s.firePayloadLocked(c, payload{slot: 4, level: 1, px: 6, py: 6, hasPos: true}, now)

	hs := c.huntState
	if len(hs.anchorEnds) != 1 {
		t.Fatalf("no anchor scheduled for the arrow rain: %d pending", len(hs.anchorEnds))
	}
}

// TestMiriamKillingVolleyFirstHitWaitsForTheFirstArrow: damage used to land the instant the
// channel started, before the arrow spawner (m_SpawnTime=1.0) had even produced its first
// arrow.
func TestMiriamKillingVolleyFirstHitWaitsForTheFirstArrow(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Miriam")
	defer cleanup()
	hs := c.huntState

	m := mkMob(t, 8700, 3, 3)
	m.hp, m.maxHP = 100000, 100000
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	startHP := m.hp

	now := float64(s.battleTime())
	s.applyOpsLocked(c, hs.skillDef(4).Ops, opCtx{slot: 4, level: 1, px: 3, py: 3, hasPos: true}, now)
	if len(hs.channels) != 1 {
		t.Fatalf("«Убийственный залп» did not start a channel: %d channels", len(hs.channels))
	}
	if m.hp != startHP {
		t.Fatal("damage landed the instant the channel was cast, before any arrow existed")
	}
	if got := hs.channels[0].nextPulse - now; got < 0.99 || got > 1.01 {
		t.Fatalf("first pulse scheduled %.3gs out, want ~1.0s (the spawner's own first arrow)", got)
	}
	s.tickChannelsLocked(c, now+1.0)
	if m.hp >= startHP {
		t.Fatal("no damage landed once the first arrow would have spawned")
	}
}
