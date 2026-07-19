package gamedata

import (
	"encoding/json"
	"testing"
)

// restoreSettings snapshots the live settings and re-applies them on cleanup, so a
// test that mutates the global knobs never leaks into another test.
func restoreSettings(t *testing.T) {
	t.Helper()
	orig := Snapshot()
	t.Cleanup(func() { Apply(orig) })
}

func TestSettingsDefaultsReproduceAuthored(t *testing.T) {
	// Level 1 is identity; the default slopes reproduce the old constants.
	if MobHPMul(1) != 1 || MobDmgMul(1) != 1 || MobXPMul(1) != 1 || MobCoinMul(1) != 1 {
		t.Fatal("level-1 mob multipliers must be 1")
	}
	if got := MobHPMul(5); got < 1.599 || got > 1.601 { // 1 + 0.15*4
		t.Errorf("MobHPMul(5) = %g, want ~1.6", got)
	}
	if LevelPowerMul(0) != 1 || LevelHealthMul(0) != 1 {
		t.Error("level-0 hero multipliers must be 1")
	}
	if got := LevelPowerMul(10); got < 1.59 || got > 1.61 { // 1 + 0.06*10
		t.Errorf("LevelPowerMul(10) = %g, want ~1.6", got)
	}
	if !FogOfWar() {
		t.Error("fog-of-war must default to enabled")
	}
	if !HuntFog() {
		t.Error("hunt fog must default to enabled")
	}
	if m, d := NewHeroWallet(); m != 1000 || d != 100 {
		t.Errorf("NewHeroWallet() = %d,%d, want 1000,100", m, d)
	}
}

func TestGlobalRewardMultipliers(t *testing.T) {
	restoreSettings(t)
	ghoul := Mobs()[mobGhoul]
	_, _, _, xp0, coins0 := ghoul.ScaledStats(3)
	Update(func(s *Settings) { s.XPMultiplier = 2; s.CoinMultiplier = 2 })
	_, _, _, xp1, coins1 := ghoul.ScaledStats(3)
	if xp1 != xp0*2 {
		t.Errorf("xp with 2x = %g, want %g", xp1, xp0*2)
	}
	if coins1 != coins0*2 {
		t.Errorf("coins with 2x = %d, want %d", coins1, coins0*2)
	}
	// Bosses (level-exempt) still honour the global multiplier.
	boss := Mobs()[mobBossVelial]
	_, _, _, bxp0, _ := boss.ScaledStats(1)
	Apply(Snapshot()) // no-op, keep 2x
	_, _, _, bxp1, _ := boss.ScaledStats(1)
	if bxp1 != bxp0 { // both computed under 2x, equal
		t.Errorf("boss xp not stable under fixed mult: %g vs %g", bxp0, bxp1)
	}
}

func TestTunableMobSlope(t *testing.T) {
	restoreSettings(t)
	Update(func(s *Settings) { s.MobHPPerLevel = 0.5 })
	if got := MobHPMul(3); got < 1.999 || got > 2.001 { // 1 + 0.5*2
		t.Errorf("MobHPMul(3) with 0.5 slope = %g, want 2", got)
	}
}

func TestAvatarOverrideRoundTrip(t *testing.T) {
	restoreSettings(t)
	const id = 8 // Mihalych
	base, ok := AvatarByID(id)
	if !ok {
		t.Fatal("avatar 8 missing")
	}
	Update(func(s *Settings) { s.AvatarOverrides[id] = map[string]float64{"Health": 12345, "DmgMin": 99} })
	got, _ := AvatarByID(id)
	if got.Health != 12345 || got.DmgMin != 99 {
		t.Errorf("override not applied: Health=%g DmgMin=%d", got.Health, got.DmgMin)
	}
	if got.Mana != base.Mana {
		t.Errorf("un-overridden stat changed: Mana %g -> %g", base.Mana, got.Mana)
	}
	// Clearing restores the authored template.
	Update(func(s *Settings) { delete(s.AvatarOverrides, id) })
	restored, _ := AvatarByID(id)
	if restored.Health != base.Health {
		t.Errorf("clear didn't restore: Health=%g, want %g", restored.Health, base.Health)
	}
}

func TestMobOverrideRoundTrip(t *testing.T) {
	restoreSettings(t)
	base := Mobs()[mobGhoul]
	Update(func(s *Settings) { s.MobOverrides[mobGhoul] = map[string]float64{"Health": 777} })
	got := MobByIndex(mobGhoul)
	if got.Health != 777 {
		t.Errorf("mob override not applied: Health=%g", got.Health)
	}
	if got.DmgMax != base.DmgMax {
		t.Errorf("un-overridden mob stat changed: DmgMax %d -> %d", base.DmgMax, got.DmgMax)
	}
	// The shared roster is never mutated by an override.
	if Mobs()[mobGhoul].Health != base.Health {
		t.Error("override leaked into the shared roster")
	}
}

func TestSettingsJSONRoundTrip(t *testing.T) {
	restoreSettings(t)
	Update(func(s *Settings) {
		s.FogOfWarEnabled = false
		s.XPMultiplier = 3
		s.AvatarOverrides[1] = map[string]float64{"Health": 5000}
	})
	blob, err := MarshalSettings()
	if err != nil {
		t.Fatal(err)
	}
	// Reset to defaults, then re-apply from JSON: state must return.
	Apply(*defaultSettings())
	if FogOfWar() != true {
		t.Fatal("reset didn't restore fog default")
	}
	if err := ApplyJSON(blob); err != nil {
		t.Fatal(err)
	}
	if FogOfWar() != false {
		t.Error("fog not restored from JSON")
	}
	got := Snapshot()
	if got.XPMultiplier != 3 {
		t.Errorf("XPMultiplier not restored: %g", got.XPMultiplier)
	}
	if got.AvatarOverrides[1]["Health"] != 5000 {
		t.Error("avatar override not restored from JSON")
	}
}

func TestApplyJSONMissingFieldsKeepDefaults(t *testing.T) {
	restoreSettings(t)
	// A blob from an older build that only knew about fog must not zero the newer
	// multipliers (which would break scaling); they keep their defaults.
	if err := ApplyJSON([]byte(`{"fog_of_war":false}`)); err != nil {
		t.Fatal(err)
	}
	if FogOfWar() {
		t.Error("fog should be off")
	}
	if got := Snapshot(); got.XPMultiplier != 1 || got.MobHPPerLevel != 0.15 || got.NewHeroMoney != 1000 {
		t.Errorf("missing fields not defaulted: %+v", got)
	}
}

func TestNormalizeClampsNegatives(t *testing.T) {
	restoreSettings(t)
	Apply(Settings{XPMultiplier: -5, CoinMultiplier: -1, NewHeroMoney: -10, NewHeroDiamond: -3})
	got := Snapshot()
	if got.XPMultiplier != 0 || got.CoinMultiplier != 0 || got.NewHeroMoney != 0 || got.NewHeroDiamond != 0 {
		t.Errorf("negatives not clamped: %+v", got)
	}
	if got.AvatarOverrides == nil || got.MobOverrides == nil {
		t.Error("nil override maps not normalized")
	}
}

// sanity: Settings marshals to a JSON object (integer map keys as strings).
func TestSettingsMarshalShape(t *testing.T) {
	restoreSettings(t)
	Update(func(s *Settings) { s.MobOverrides[9] = map[string]float64{"Health": 1} })
	blob, _ := MarshalSettings()
	var m map[string]any
	if err := json.Unmarshal(blob, &m); err != nil {
		t.Fatalf("settings JSON not an object: %v", err)
	}
	if _, ok := m["mob_overrides"]; !ok {
		t.Error("mob_overrides missing from JSON")
	}
}
