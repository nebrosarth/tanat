package gamedata

import (
	"bufio"
	"os"
	"strings"
	"testing"
)

// loadValidFx reads the client's baked VisualEffectsMgr registry names (the
// only fx strings EFFECT_START can reference). testdata/valid_fx.txt is dumped
// from Tanat_Data/mainData; the empty string is always allowed (no fx).
func loadValidFx(t *testing.T) map[string]bool {
	t.Helper()
	f, err := os.Open("testdata/valid_fx.txt")
	if err != nil {
		t.Fatalf("open valid_fx.txt: %v", err)
	}
	defer f.Close()
	set := map[string]bool{"": true}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if line := strings.TrimSpace(sc.Text()); line != "" {
			set[line] = true
		}
	}
	return set
}

var validSkillType = map[string]bool{"ACTIVE": true, "TOGGLE": true, "PASSIVE": true}
var validScale = map[string]bool{"": true, "phys": true, "magic": true, "pure": true}
var validStat = map[string]bool{
	"dmg_pct": true, "phys_armor": true, "magic_armor": true, "attack_speed_pct": true,
	"move_speed_pct": true, "lifesteal_pct": true, "crit_pct": true, "dodge_pct": true,
	"spell_power": true, "hp_regen": true, "mana_regen": true, "max_hp": true,
	"thorns_pct": true, "armor_pct": true,
	"attack_range": true, "crit_dmg_pct": true,
}

// TestEveryAvatarHasAuthoredKit fails if any playable avatar is missing from the
// generated skill data (it would silently fall back to uniform nukes).
func TestEveryAvatarHasAuthoredKit(t *testing.T) {
	for _, a := range Avatars() {
		if _, ok := skillsByPrefab[a.Prefab]; !ok {
			t.Errorf("avatar %s (%s) has no authored skill kit", a.Prefab, a.ShortName)
		}
	}
	if len(skillsByPrefab) != len(Avatars()) {
		t.Errorf("skill kits = %d, roster = %d", len(skillsByPrefab), len(Avatars()))
	}
}

// TestProjectileAvatarsAreRanged guards the fix for ranged DPS (e.g. Miriam)
// walking to point-blank: any avatar whose kit fires a basic-attack projectile
// must have a ranged AttackRange, never the melee class-template default.
func TestProjectileAvatarsAreRanged(t *testing.T) {
	for _, a := range Avatars() {
		if !SkillsFor(a).AttackProjectile {
			continue
		}
		if a.AttackRange < rangedBasicAttackRange {
			t.Errorf("%s fires a basic-attack projectile but AttackRange=%.1f is melee (want >= %.1f)",
				a.ShortName, a.AttackRange, rangedBasicAttackRange)
		}
	}
	// Sanity: Miriam (id 25) specifically, the reported case.
	if m, ok := AvatarByID(25); !ok {
		t.Fatal("Miriam (id 25) missing from roster")
	} else if m.AttackRange < rangedBasicAttackRange {
		t.Errorf("Miriam AttackRange=%.1f, want ranged >= %.1f", m.AttackRange, rangedBasicAttackRange)
	}
}

// TestElgormSkullReleasesDuringThrow locks the visual fix: Elgorm's «Блуждающий
// ужас» skull must leave his hand no later than the end of the throw animation
// (PayloadDelay <= CastFxDur), not pop out after it finished.
func TestElgormSkullReleasesDuringThrow(t *testing.T) {
	a, ok := AvatarByID(31) // Elgorm
	if !ok {
		t.Fatal("Elgorm (id 31) missing from roster")
	}
	sk := SkillsFor(a).Skills[0]
	if sk.PayloadDelay > sk.CastFxDur {
		t.Errorf("skull PayloadDelay=%.2f > throw CastFxDur=%.2f: skull pops out after the animation ends",
			sk.PayloadDelay, sk.CastFxDur)
	}
}

// TestSkillKitInvariants checks structural and wire-safety invariants across
// every authored skill, including that every fx name exists in the client's
// registry and passive skills carry no mana/cooldown.
func TestSkillKitInvariants(t *testing.T) {
	fx := loadValidFx(t)
	var checkOp func(prefab string, slot int, op Op)
	checkOp = func(prefab string, slot int, op Op) {
		if !validScale[op.Scale] {
			t.Errorf("%s s%d: bad scale %q", prefab, slot, op.Scale)
		}
		if op.Kind == OpBuffStat && !validStat[op.Stat] {
			t.Errorf("%s s%d: bad buffstat %q", prefab, slot, op.Stat)
		}
		if op.Kind == OpSummon && op.Unit == "" {
			t.Errorf("%s s%d: summon without unit", prefab, slot)
		}
		if op.TrapFx != "" && !fx[op.TrapFx] {
			t.Errorf("%s s%d: unknown trap_fx %q", prefab, slot, op.TrapFx)
		}
		if op.TriggerFx != "" && !fx[op.TriggerFx] {
			t.Errorf("%s s%d: unknown trigger_fx %q", prefab, slot, op.TriggerFx)
		}
		for _, sub := range op.Ops {
			checkOp(prefab, slot, sub)
		}
	}
	for prefab, kit := range skillsByPrefab {
		for i, sk := range kit.Skills {
			if sk.Slot != i+1 {
				t.Errorf("%s: skill index %d has slot %d", prefab, i, sk.Slot)
			}
			if !validSkillType[sk.Type] {
				t.Errorf("%s s%d: bad type %q", prefab, sk.Slot, sk.Type)
			}
			for _, name := range []string{sk.CastFx, sk.PayloadFx, sk.BuffFx} {
				if !fx[name] {
					t.Errorf("%s s%d: unknown fx %q", prefab, sk.Slot, name)
				}
			}
			if sk.Type == "PASSIVE" {
				for _, v := range sk.ManaCost {
					if v != 0 {
						t.Errorf("%s s%d: passive with mana cost %d", prefab, sk.Slot, v)
					}
				}
			}
			if len(sk.Ops) == 0 {
				t.Errorf("%s s%d: no ops", prefab, sk.Slot)
			}
			for _, op := range sk.Ops {
				checkOp(prefab, sk.Slot, op)
			}
		}
	}
}

// TestBossSkillFxValid fails if any boss ability references an effect the client's
// VisualEffectsMgr doesn't ship -- EFFECT_START silently renders nothing for an
// unknown name, so an invalid Fx makes the whole ability invisible in-game (the
// original bug: the bosses used invented "VFX_Boss_*" names that don't exist).
func TestBossSkillFxValid(t *testing.T) {
	fx := loadValidFx(t)
	seen := 0
	for _, m := range Mobs() {
		for _, sk := range m.Skills {
			seen++
			if sk.Fx == "" {
				t.Errorf("boss %s skill %q has an empty Fx (nothing would show)", m.Prefab, sk.Name)
				continue
			}
			if !fx[sk.Fx] {
				t.Errorf("boss %s skill %q: unknown Fx %q (not in the client registry -> invisible)",
					m.Prefab, sk.Name, sk.Fx)
			}
		}
	}
	if seen == 0 {
		t.Fatal("no boss skills found -- test would be vacuous")
	}
}

// TestAvroraKit pins Avrora's hand-authored healer kit: she is registered, is a
// SUPPORT avatar, her skill icons/locale keys resolve to the Avrora bases, and
// every skill both heals the caster (OpHeal) and matches its recovered name.
func TestAvroraKit(t *testing.T) {
	a, ok := AvatarByID(40)
	if !ok {
		t.Fatal("Avrora (id 40) not in roster")
	}
	if a.Prefab != "Avtr_Sp_Avrora" || a.Type != AvatarTypeSupport {
		t.Fatalf("Avrora prefab/type = %q/%d, want Avtr_Sp_Avrora/SUPPORT", a.Prefab, a.Type)
	}
	// No asset-name fixups: skill icons and locale keys must derive from the prefab.
	if got := a.SkillIcon(4); got != "Avtr_Sp_Avrora_skill4" {
		t.Errorf("SkillIcon(4) = %q, want Avtr_Sp_Avrora_skill4", got)
	}
	if got := a.Name(); got != "IDS_Avtr_Sp_Avrora_Name" {
		t.Errorf("Name() = %q", got)
	}
	if got := a.SkillTitle(1); got != "IDS_AvroraSkill1_Name" {
		t.Errorf("SkillTitle(1) = %q", got)
	}
	kit, ok := skillsByPrefab["Avtr_Sp_Avrora"]
	if !ok {
		t.Fatal("Avrora has no authored kit")
	}
	wantNames := [4]string{"Милость", "Освященное место", "Воссоединение", "Молитва"}
	for i, sk := range kit.Skills {
		if sk.NameRu != wantNames[i] {
			t.Errorf("skill %d name = %q, want %q", i+1, sk.NameRu, wantNames[i])
		}
		hasHeal := false
		for _, op := range sk.Ops {
			if op.Kind == OpHeal || op.Kind == OpHot {
				hasHeal = true
			}
		}
		if !hasHeal {
			t.Errorf("skill %d (%s) has no heal op -- Avrora is a healer", i+1, sk.NameRu)
		}
		// fx are intentionally empty until the client registers her VFX.
		if sk.CastFx != "" || sk.PayloadFx != "" || sk.BuffFx != "" {
			t.Errorf("skill %d has a non-empty fx (unregistered in the 1.11 client)", i+1)
		}
	}
}

// TestArianaAssetNames pins Ariana's three-way naming split: her model bundle
// is "Arianna" (two n) but her portrait icons and locale keys are "Ariana"
// (one n), while her skill icons genuinely use the two-n prefab. A regression on
// the portrait icon makes her select-window button load a missing texture and
// render blank -- she looks absent from the roster.
func TestArianaAssetNames(t *testing.T) {
	a, ok := AvatarByID(38)
	if !ok {
		t.Fatal("Ariana (id 38) not in roster")
	}
	if a.Prefab != "Avtr_Sp_Arianna" {
		t.Fatalf("Prefab = %q, want Avtr_Sp_Arianna", a.Prefab)
	}
	if got := a.Icon(); got != "Gui/Avatars/Icons/Avtr_Sp_Ariana" {
		t.Errorf("Icon() = %q, want Gui/Avatars/Icons/Avtr_Sp_Ariana (one n)", got)
	}
	if got := a.Name(); got != "IDS_Avtr_Sp_Ariana_Name" {
		t.Errorf("Name() = %q, want IDS_Avtr_Sp_Ariana_Name (one n)", got)
	}
	if got := a.SkillIcon(1); got != "Avtr_Sp_Arianna_skill1" {
		t.Errorf("SkillIcon(1) = %q, want Avtr_Sp_Arianna_skill1 (two n)", got)
	}
}

// TestPerLevelAt clamps to the 1..4 range.
func TestPerLevelAt(t *testing.T) {
	p := PerLevel{10, 20, 30, 40}
	for _, tc := range []struct {
		lvl  int
		want float64
	}{{0, 10}, {1, 10}, {2, 20}, {4, 40}, {9, 40}} {
		if got := p.At(tc.lvl); got != tc.want {
			t.Errorf("At(%d) = %g, want %g", tc.lvl, got, tc.want)
		}
	}
}
