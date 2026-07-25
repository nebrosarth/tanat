package gamedata

import (
	"strings"
	"testing"
)

// Fidelity tests for the 2026-07-18 avatar-skill audit fixes: they pin the DATA-level
// corrections so a regression (a skill sliding back to the wrong mechanic) fails loudly.

// skillOf returns a prefab's slot (1-based).
func skillOf(t *testing.T, prefab string, slot int) Skill {
	t.Helper()
	ks, ok := skillsByPrefab[prefab]
	if !ok {
		t.Fatalf("no skills for %s", prefab)
	}
	return ks.Skills[slot-1]
}

// hasOp reports whether a skill (or its nested proc/aura/channel ops) contains an op of Kind
// satisfying pred.
func anyOp(ops []Op, pred func(Op) bool) bool {
	for _, op := range ops {
		if pred(op) {
			return true
		}
		if anyOp(op.Ops, pred) {
			return true
		}
	}
	return false
}

// TestOnDamagedProcsFlagged: every passive that must trigger on being STRUCK carries an
// OpProc with OnDamaged=true, and no OTHER passive OpProc is flagged (so the on-hit procs
// still fire on attack).
func TestOnDamagedProcsFlagged(t *testing.T) {
	want := map[string]int{ // prefab -> slot that must be OnDamaged
		"Avtr_Tank_Titanid":    3, // «Каменная кожа»
		"Avtr_Tank_Gektor":     2, // «Реванш»
		"Avtr_HK_Dutnik":       3, // «Детонация»
		"Avtr_DPS_Nerlag":      3, // «Прилив крови»
		"Avtr_DPS_BlackDragon": 3, // «Кровь дракона»
		"Avtr_Dsb_Edilia":      3, // «Пыльца забвения»: slow fires on the STRIKING mob
		"Avtr_HK_Tangren":      2, // «Контратака»: pass-7, chance-gated own-attack counter
		"Avtr_Sp_Neirofim":     2, // «Обращение энергии»: pass-8, SkillOnly reactive heal+mana+nova
	}
	for prefab, slot := range want {
		sk := skillOf(t, prefab, slot)
		if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpProc && o.OnDamaged }) {
			t.Errorf("%s slot %d («%s») must have an OnDamaged proc", prefab, slot, sk.NameRu)
		}
	}
	// Nothing else should be OnDamaged.
	for _, ks := range skillsByPrefab {
		for i, sk := range ks.Skills {
			for _, op := range sk.Ops {
				if op.Kind == OpProc && op.OnDamaged {
					if w, ok := want[ks.Prefab]; !ok || w != i+1 {
						t.Errorf("%s slot %d has an unexpected OnDamaged proc", ks.Prefab, i+1)
					}
				}
			}
		}
	}
}

// TestNerlagBloodHealsWhenStruck: «Прилив крови» is an on-damaged proc whose OpHeal has
// Value2>0 (heals for the damage just taken), NOT a lifesteal-on-attack buff.
func TestNerlagBloodHealsWhenStruck(t *testing.T) {
	sk := skillOf(t, "Avtr_DPS_Nerlag", 3)
	healByDamage := anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpHeal && o.Value2.At(1) > 0 })
	if !healByDamage {
		t.Error("Nerlag «Прилив крови» must heal for the damage taken (OpHeal Value2>0)")
	}
	if anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpBuffStat && o.Stat == "lifesteal_pct" }) {
		t.Error("Nerlag «Прилив крови» must NOT be lifesteal-on-attack anymore")
	}
}

// TestSkillStealthGranted: the self-cloak skills carry an OpStealth (real invisibility), not
// just a cosmetic InvisibilityEffect.
func TestSkillStealthGranted(t *testing.T) {
	for _, tc := range []struct {
		prefab string
		slot   int
	}{
		{"Avtr_DPS_Lirvein", 1}, // «Единение с ветром»
		{"Avtr_Dsb_Wilfang", 2}, // «Засада»
	} {
		sk := skillOf(t, tc.prefab, tc.slot)
		if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpStealth && o.Dur.At(1) > 0 }) {
			t.Errorf("%s slot %d («%s») must grant real stealth (OpStealth)", tc.prefab, tc.slot, sk.NameRu)
		}
	}
}

// TestKionaCloakIsDualTargetBuffDebuff: «Лесной покров» is a TARGETED cast that can land
// on either an enemy or an ally (client: "вешает накидку на врага ИЛИ союзника") -- an
// ally gets +base-attack, an enemy gets -base-attack, and either side shares damage taken
// as healing to nearby allies (OpDamageShare). Not the untargeted self-shield it used to be.
func TestKionaCloakIsDualTargetBuffDebuff(t *testing.T) {
	sk := skillOf(t, "Avtr_Sp_Kiona", 2)
	if !skillHasFlag(sk.Target, "ENEMY") || !skillHasFlag(sk.Target, "FRIEND") {
		t.Errorf("Kiona «Лесной покров» must be dual enemy/ally targetable, got Target=%q", sk.Target)
	}
	if !anyOp(sk.Ops, func(o Op) bool {
		return o.Kind == OpBuffStat && o.TargetSide == "ally" && o.Value.At(1) > 0
	}) {
		t.Error("Kiona «Лесной покров» must buff an ally's base attack")
	}
	if !anyOp(sk.Ops, func(o Op) bool {
		return o.Kind == OpBuffStat && o.TargetSide == "enemy" && o.Value.At(1) < 0
	}) {
		t.Error("Kiona «Лесной покров» must debuff an enemy's base attack")
	}
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpDamageShare }) {
		t.Error("Kiona «Лесной покров» must share the marked target's damage taken as ally healing")
	}
}

// skillHasFlag reports whether a '+'-joined SkillTarget mask names flag, matched whole.
func skillHasFlag(mask, flag string) bool {
	for _, f := range strings.Split(mask, "+") {
		if strings.TrimSpace(f) == flag {
			return true
		}
	}
	return false
}

// TestNeirofimDevoursMagic: the CLIENT «Пожирание магии» is an on-hit mana-devour (drains a
// % of Neirofim's OWN max mana from the target, dealing damage equal to what was taken) --
// NOT the magic_armor buff the stale wiki suggested.
func TestNeirofimDevoursMagic(t *testing.T) {
	sk := skillOf(t, "Avtr_Sp_Neirofim", 3)
	if !anyOp(sk.Ops, func(o Op) bool {
		return o.Kind == OpManaBurnHit && o.Apply == "own_mana" && o.Value2.At(1) > 0
	}) {
		t.Error("Neirofim «Пожирание магии» must be an on-hit mana-devour (OpManaBurnHit own_mana)")
	}
}

// TestCcConventionRootSilence: the roll/short-circuit skills that «обездвиживают и не дают
// использовать способности» use OpRoot+OpSilence, not a full OpStun.
func TestCcConventionRootSilence(t *testing.T) {
	for _, tc := range []struct {
		prefab string
		slot   int
	}{
		{"Avtr_Dsb_PlusMinus", 2}, // «Короткое замыкание»
	} {
		sk := skillOf(t, tc.prefab, tc.slot)
		if anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpStun }) {
			t.Errorf("%s slot %d («%s») should be root+silence, not a full stun", tc.prefab, tc.slot, sk.NameRu)
		}
		if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpRoot }) || !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpSilence }) {
			t.Errorf("%s slot %d («%s») must have both OpRoot and OpSilence", tc.prefab, tc.slot, sk.NameRu)
		}
	}
}

// TestGellarPresencePassive: «Ужасающее присутствие» is a PASSIVE aura with no mana upkeep.
func TestGellarPresencePassive(t *testing.T) {
	sk := skillOf(t, "Avtr_DPS_Gellar", 3)
	if sk.Type != "PASSIVE" {
		t.Errorf("Gellar «Ужасающее присутствие» Type = %q, want PASSIVE", sk.Type)
	}
	for _, op := range sk.Ops {
		if op.Kind == OpAura && op.TickCost.At(1) != 0 {
			t.Errorf("passive aura must have 0 TickCost, got %v", op.TickCost.At(1))
		}
	}
}

// --- 2026-07-18, second-pass LOW batch reconciled against the CLIENT locale ---

// TestMihalychRoarStuns: client IDS_MihalychSkill4 «Звериный рев» stuns nearby enemies
// «на 2 секунды», THEN slows them -- the AoE stun was missing before this pass.
func TestMihalychRoarStuns(t *testing.T) {
	sk := skillOf(t, "Avtr_HK_Mihalych", 4)
	stun := false
	for _, op := range sk.Ops {
		if op.Kind == OpStun && op.Dur.At(1) >= 2 && op.Radius > 0 {
			stun = true
		}
	}
	if !stun {
		t.Error("Mihalych «Звериный рев» must AoE-stun (~2s) nearby enemies")
	}
	// The damage stays an instant burst (the client text is stun+slow, not a DoT).
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpDamage }) {
		t.Error("Mihalych «Звериный рев» must still deal its burst damage")
	}
}

// TestTangrenDanceBounces: client IDS_TangrenSkill4 «Танец смерти» chains between five
// random enemies (OpBounce, damage per hop), not a stationary channel; the bounce needs
// a first target to resolve its origin, so the skill is TARGET-cast.
func TestTangrenDanceBounces(t *testing.T) {
	sk := skillOf(t, "Avtr_HK_Tangren", 4)
	if len(sk.Ops) == 0 || sk.Ops[0].Kind != OpBounce {
		t.Fatalf("Tangren «Танец смерти» op[0] must be OpBounce, got %+v", sk.Ops)
	}
	b := sk.Ops[0]
	if b.Count.At(1) != 5 {
		t.Errorf("bounce Count = %v, want 5 jumps", b.Count.At(1))
	}
	if !anyOp(b.Ops, func(o Op) bool { return o.Kind == OpDamage }) {
		t.Error("each «Танец смерти» hop must deal damage")
	}
	if sk.Targeting != "TARGET" {
		t.Errorf("Tangren «Танец смерти» Targeting = %q, want TARGET (bounce needs a first target)", sk.Targeting)
	}
}

// TestZamaranFlameRootIsChance: client IDS_ZamaranSkill2 «Пламя войны» roots «с
// вероятностью 20%», so its aura OpRoot carries a partial Chance instead of firing
// every tick.
func TestZamaranFlameRootIsChance(t *testing.T) {
	sk := skillOf(t, "Avtr_Tank_Zamaran", 2)
	hasChanceRoot := anyOp(sk.Ops, func(o Op) bool {
		c := o.Chance.At(1)
		return o.Kind == OpRoot && c > 0 && c < 1
	})
	if !hasChanceRoot {
		t.Error("Zamaran «Пламя войны» root must be a per-tick Chance (0<Chance<1), not guaranteed")
	}
}

// --- deferred batch: ally-targeting + execute + consecutive-hit ---

// TestAllySkillsHealOrBuffAllies: the friendly-support skills carry a heal/hot/shield/buff
// op flagged On in {"ally","allies"} so it lands on party members, not just the caster.
func TestAllySkillsHealOrBuffAllies(t *testing.T) {
	for _, tc := range []struct {
		prefab string
		slot   int
	}{
		{"Avtr_Sp_Arianna", 1},    // «Щит хранителя» -> ally shield
		{"Avtr_Sp_Arianna", 3},    // «Исцеление» -> allies heal
		{"Avtr_Sp_Arianna", 4},    // «Касание спасителя» -> ally shield+hot
		{"Avtr_HK_Tangren", 3},    // «Целительный тотем» -> allies hot
		{"Avtr_Dsb_Edilia", 1},    // «Касание природы» -> allies heal
		{"Avtr_Dsb_Edilia", 4},    // «Дерево жизни» -> allies heal (channel)
		{"Avtr_Sp_Kiona", 1},      // «Лечебная волна» -> chain-heals living allies
		{"Avtr_DPS_Sandariel", 2}, // «Прыжок» -> allies speed
	} {
		sk := skillOf(t, tc.prefab, tc.slot)
		// OpChainHeal (Kiona's «Лечебная волна») always targets living party members via
		// c.members() -- it carries no On field, unlike the plain ally-heal/buff ops.
		if !anyOp(sk.Ops, func(o Op) bool { return o.On == "ally" || o.On == "allies" || o.Kind == OpChainHeal }) {
			t.Errorf("%s slot %d («%s») must carry an ally-targeting op (On ally/allies, or OpChainHeal)", tc.prefab, tc.slot, sk.NameRu)
		}
	}
}

// TestGektorExecuteThreshold: «Казнь» is a real OpExecute with a kill threshold (Value2),
// not the old soft missing-HP damage approximation.
func TestGektorExecuteThreshold(t *testing.T) {
	sk := skillOf(t, "Avtr_Tank_Gektor", 4)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpExecute && o.Value2.At(1) > 0 }) {
		t.Error("Gektor «Казнь» must be an OpExecute with a HP threshold (Value2>0)")
	}
}

// TestMihalychTrepkaConsecutive: «Трепка» is a per-target consecutive-hit damage stack.
func TestMihalychTrepkaConsecutive(t *testing.T) {
	sk := skillOf(t, "Avtr_HK_Mihalych", 2)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpConsecutiveHit && o.Value.At(1) > 0 }) {
		t.Error("Mihalych «Трепка» must be an OpConsecutiveHit with a per-stack bonus")
	}
}

// --- deferred batch #2: on-kill attack stacks + Lirvein attack-speed streak ---

// TestGellarSoulsOnKillStack: «Порабощение» banks a persistent soul per kill (+2 attack,
// capped at {charges}, halved on death) -- a Dur-0 OpOnKillStack, not a flat dmg buff.
func TestGellarSoulsOnKillStack(t *testing.T) {
	sk := skillOf(t, "Avtr_DPS_Gellar", 2)
	ok := anyOp(sk.Ops, func(o Op) bool {
		return o.Kind == OpOnKillStack && o.Value.At(1) == 2 && o.Value2.At(1) > 0 && o.HalveOnDeath && o.Dur.At(1) == 0
	})
	if !ok {
		t.Error("Gellar «Порабощение» must be a persistent OpOnKillStack (+2/soul, capped, halve-on-death, Dur 0)")
	}
	if anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpBuffStat }) {
		t.Error("Gellar «Порабощение» must no longer be a flat dmg_pct buff")
	}
}

// TestHekataCultKillWindow: «Культ жнеца» keeps its +30% attack buff AND opens a timed
// kill-window (Dur>0 OpOnKillStack) that adds flat attack per kill during the buff.
func TestHekataCultKillWindow(t *testing.T) {
	sk := skillOf(t, "Avtr_Dsb_Hekata", 2)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpBuffStat && o.Stat == "dmg_pct" && o.Value.At(1) == 1.3 }) {
		t.Error("Hekata «Культ жнеца» must still grant its flat +30% attack buff")
	}
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpOnKillStack && o.Dur.At(1) > 0 && o.Value.At(1) > 0 }) {
		t.Error("Hekata «Культ жнеца» must open a kill-window (OpOnKillStack, Dur>0)")
	}
}

// TestLirveinRelentlessStreak: «Неумолимость» is a streak-based attack-speed passive
// (per-hit speed Value, capped at Value2), not the old always-on chance-proc haste.
func TestLirveinRelentlessStreak(t *testing.T) {
	sk := skillOf(t, "Avtr_DPS_Lirvein", 3)
	ok := anyOp(sk.Ops, func(o Op) bool {
		return o.Kind == OpAttackSpeedStreak && o.Value.At(1) > 0 && o.Value2.At(1) >= o.Value.At(1)
	})
	if !ok {
		t.Error("Lirvein «Неумолимость» must be an OpAttackSpeedStreak (per-hit speed, capped by Value2)")
	}
	if anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpProc }) {
		t.Error("Lirvein «Неумолимость» must no longer be a chance-proc buff")
	}
}

// --- deferred batch #3: attack-power-scaled on-hit + shield-explode ---

// TestGektorSmiteAttackScaled: «Разящий удар» is an every-swing (Chance 1) proc that adds
// a share of base attack (OpAttackDamage) plus a slow -- not a random flat-damage proc.
func TestGektorSmiteAttackScaled(t *testing.T) {
	sk := skillOf(t, "Avtr_Tank_Gektor", 3)
	var proc *Op
	for i := range sk.Ops {
		if sk.Ops[i].Kind == OpProc {
			proc = &sk.Ops[i]
		}
	}
	if proc == nil {
		t.Fatal("Gektor «Разящий удар» must be an on-hit proc")
	}
	if proc.Chance.At(1) != 1 {
		t.Errorf("«Разящий удар» must fire on every attack (Chance 1), got %g", proc.Chance.At(1))
	}
	if !anyOp(proc.Ops, func(o Op) bool { return o.Kind == OpAttackDamage && o.Value.At(1) > 0 }) {
		t.Error("«Разящий удар» must add attack-power-scaled damage (OpAttackDamage)")
	}
	if !anyOp(proc.Ops, func(o Op) bool { return o.Kind == OpSlow }) {
		t.Error("«Разящий удар» must still slow the target")
	}
	if anyOp(proc.Ops, func(o Op) bool { return o.Kind == OpDamage }) {
		t.Error("«Разящий удар» must no longer deal a flat OpDamage")
	}
}

// --- deferred batch #4: mob-mana skills, dual casts, Frost chill, Rognar/Gellar/Hekata ---

// TestManaSkillsWired: the mana-interaction skills carry the right ops.
func TestManaSkillsWired(t *testing.T) {
	if !anyOp(skillOf(t, "Avtr_Sp_Neirofim", 1).Ops, func(o Op) bool { return o.Kind == OpManaScaledDamage }) {
		t.Error("Neirofim «Паралич воли» must be OpManaScaledDamage")
	}
	if !anyOp(skillOf(t, "Avtr_Sp_Neirofim", 3).Ops, func(o Op) bool { return o.Kind == OpManaBurnHit && o.Apply == "own_mana" }) {
		t.Error("Neirofim «Пожирание магии» must drain own-mana-% (OpManaBurnHit own_mana)")
	}
	if !anyOp(skillOf(t, "Avtr_Sp_Neirofim", 4).Ops, func(o Op) bool { return o.Kind == OpSilenceAll }) {
		t.Error("Neirofim «Молчание» must be OpSilenceAll")
	}
	if !anyOp(skillOf(t, "Avtr_DPS_BlackDragon", 2).Ops, func(o Op) bool { return o.Kind == OpManaBurnHit }) {
		t.Error("BlackDragon «Выжигание маны» must burn mana (OpManaBurnHit)")
	}
	if !anyOp(skillOf(t, "Avtr_Sp_Inshari", 3).Ops, func(o Op) bool { return o.Kind == OpManaBurnHit && o.Apply == "restore" }) {
		t.Error("Inshari «Изъятие сущности» must siphon target mana to itself (OpManaBurnHit restore)")
	}
}

// TestDualCastSkills: the friend-or-foe skills are FRIEND-castable and carry BOTH an enemy
// op and an ally op split by TargetSide.
func TestDualCastSkills(t *testing.T) {
	for _, tc := range []struct {
		prefab string
		slot   int
	}{
		{"Avtr_Sp_Kiona", 4},   // «Страж леса»
		{"Avtr_Dsb_Frost", 3},  // «Гробница холода»
		{"Avtr_Dsb_Hekata", 3}, // «Выбор скверны»
	} {
		sk := skillOf(t, tc.prefab, tc.slot)
		if !strings.Contains(sk.Target, "FRIEND") {
			t.Errorf("%s slot %d must be FRIEND-castable", tc.prefab, tc.slot)
		}
		if !anyOp(sk.Ops, func(o Op) bool { return o.TargetSide == "enemy" }) {
			t.Errorf("%s slot %d must carry an enemy-side op", tc.prefab, tc.slot)
		}
		if !anyOp(sk.Ops, func(o Op) bool { return o.TargetSide == "ally" }) {
			t.Errorf("%s slot %d must carry an ally-side op", tc.prefab, tc.slot)
		}
	}
}

// TestFrostChillCombo: «Стужа», «Ледяной град» and «Гробница холода» all apply OpChill.
func TestFrostChillCombo(t *testing.T) {
	for _, slot := range []int{1, 2, 3} {
		sk := skillOf(t, "Avtr_Dsb_Frost", slot)
		if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpChill }) {
			t.Errorf("Frost slot %d («%s») must apply озноб (OpChill)", slot, sk.NameRu)
		}
	}
}

// TestHekataAshDebuffsEnemies: «Пепельный смерч» weakens enemies' attack via On:"enemies"
// (a real AoE hostile debuff, no longer mis-folding onto the caster).
func TestHekataAshDebuffsEnemies(t *testing.T) {
	sk := skillOf(t, "Avtr_Dsb_Hekata", 4)
	if !anyOp(sk.Ops, func(o Op) bool {
		return o.Kind == OpBuffStat && o.On == "enemies" && o.Stat == "dmg_pct" && o.Value.At(1) < 1
	}) {
		t.Error("Hekata «Пепельный смерч» must debuff enemy attack via On:\"enemies\"")
	}
}

// TestRognarRemakes: s1 «Окропление» empowers the next hit; s4 «Канал смерти» links damage.
func TestRognarRemakes(t *testing.T) {
	if !anyOp(skillOf(t, "Avtr_Tank_Rognar", 1).Ops, func(o Op) bool { return o.Kind == OpEmpowerNextHit }) {
		t.Error("Rognar «Окропление» must be OpEmpowerNextHit")
	}
	if !anyOp(skillOf(t, "Avtr_Tank_Rognar", 4).Ops, func(o Op) bool { return o.Kind == OpDeathLink && o.Value2.At(1) > 0 }) {
		t.Error("Rognar «Канал смерти» must carry an OpDeathLink redirect")
	}
}

// TestGellarArmySouls: «Армия душ» halves souls on cast and scales its waves with soul count.
func TestGellarArmySouls(t *testing.T) {
	sk := skillOf(t, "Avtr_DPS_Gellar", 4)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpConsumeSouls }) {
		t.Error("Gellar «Армия душ» must spend (halve) souls on cast (OpConsumeSouls)")
	}
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpDamage && o.PerSoul.At(1) > 0 }) {
		t.Error("Gellar «Армия душ» damage must scale with soul count (PerSoul)")
	}
}

// TestRognarBoneShieldExplodes: «Костяной щит» keeps its −phys shield + aura DPS and now
// carries a real OpShieldExplode (min<max blast), replacing the thorns proxy.
func TestRognarBoneShieldExplodes(t *testing.T) {
	sk := skillOf(t, "Avtr_Tank_Rognar", 2)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpAura }) {
		t.Error("Rognar «Костяной щит» must keep its aura DPS")
	}
	// pass-8: armor_pct (fed through the shared 50/(a+50) mitigation curve) could never
	// actually net the client's flat "снижающий получаемый физический урон на 50%" --
	// replaced with a direct dmg_reduction_pct multiplier on incoming damage.
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpBuffStat && o.Stat == "dmg_reduction_pct" }) {
		t.Error("Rognar «Костяной щит» must keep its flat damage-reduction buff")
	}
	if !anyOp(sk.Ops, func(o Op) bool {
		return o.Kind == OpShieldExplode && o.Value.At(1) > 0 && o.Value2.At(1) > o.Value.At(1)
	}) {
		t.Error("Rognar «Костяной щит» must carry an OpShieldExplode (min<max blast)")
	}
	if anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpBuffStat && o.Stat == "thorns_pct" }) {
		t.Error("Rognar «Костяной щит» must no longer use a thorns proxy for the explosion")
	}
}

// --- pass-7 audit (2026-07-20): Teridin, Zamaran, Veritas, Hekata, Tangren -- the last
// 6 avatars that had never had a full parallel client-locale sweep (Mihalych came back
// clean and needed no fix).

// TestTeridinSniperShotIsSingleHit: «Снайперский выстрел» must land ONE hit -- the prior
// OpChannel{Dur:2,Interval:2} wrapper had Dur==Interval, so the pulse and the channel's
// (strict) eviction landed on the same tick and a second, undescribed hit fired ~2s later.
func TestTeridinSniperShotIsSingleHit(t *testing.T) {
	sk := skillOf(t, "Avtr_HK_Teridin", 4)
	if anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpChannel }) {
		t.Error("Teridin «Снайперский выстрел» must not be wrapped in an OpChannel (client describes exactly one shot)")
	}
	if len(sk.Ops) != 1 || sk.Ops[0].Kind != OpDamage {
		t.Errorf("Teridin «Снайперский выстрел» must be a single top-level OpDamage, got %+v", sk.Ops)
	}
}

// TestZamaranChargeStrikesOnArrival: «Таран»'s slow+AoE damage land upon arrival at the
// dash destination ("Добежав до точки..."), not at the moment of cast.
func TestZamaranChargeStrikesOnArrival(t *testing.T) {
	sk := skillOf(t, "Avtr_Tank_Zamaran", 1)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpDash && o.StrikeOnArrival }) {
		t.Error("Zamaran «Таран» dash must carry StrikeOnArrival so damage/slow land on arrival")
	}
}

// TestVeritasMetamorphosisIsFlatDamage: «Метаморфоза»'s damage bonus is a flat additive
// quantity scaled by spell power (same construction as the HP term and Skill1's damage),
// not a 40-70% total-damage multiplier.
func TestVeritasMetamorphosisIsFlatDamage(t *testing.T) {
	sk := skillOf(t, "Avtr_Tank_Veritas", 4)
	if anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpBuffStat && o.Stat == "dmg_pct" }) {
		t.Error("Veritas «Метаморфоза» must not grant a dmg_pct multiplier")
	}
	if !anyOp(sk.Ops, func(o Op) bool {
		return o.Kind == OpBuffStat && o.Stat == "dmg_flat" && o.PerSP > 0
	}) {
		t.Error("Veritas «Метаморфоза» must grant a flat, SP-scaled dmg_flat bonus")
	}
	if !anyOp(sk.Ops, func(o Op) bool {
		return o.Kind == OpBuffStat && o.Stat == "max_hp" && o.PerSP > 0
	}) {
		t.Error("Veritas «Метаморфоза» HP bonus must also scale with spell power")
	}
}

// TestVeritasBlessingRegenScalesWithSP: «Благословение жизни»'s HP-regen term scales with
// the caster's own spell power (client: "{hpInc}+{damageSP} единиц в секунду"); the armor
// term stays flat.
func TestVeritasBlessingRegenScalesWithSP(t *testing.T) {
	sk := skillOf(t, "Avtr_Tank_Veritas", 3)
	if !anyOp(sk.Ops, func(o Op) bool {
		return o.Kind == OpBuffStat && o.Stat == "hp_regen" && o.PerSP > 0
	}) {
		t.Error("Veritas «Благословение жизни» hp_regen must carry PerSP>0")
	}
	if anyOp(sk.Ops, func(o Op) bool {
		return o.Kind == OpBuffStat && o.Stat == "phys_armor" && o.PerSP > 0
	}) {
		t.Error("Veritas «Благословение жизни» phys_armor must stay flat (no PerSP)")
	}
}

// TestHekataSulfurCloudSilencesNotStuns: «Серное облако» — "лишается возможности
// применять способности" is a silence (can still move/basic-attack), not a full stun, and
// its duration matches the DoT's flat 5s, not the old 2-3s stun window.
func TestHekataSulfurCloudSilencesNotStuns(t *testing.T) {
	sk := skillOf(t, "Avtr_Dsb_Hekata", 1)
	if anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpStun }) {
		t.Error("Hekata «Серное облако» must not stun (client says silence, not stun)")
	}
	if !anyOp(sk.Ops, func(o Op) bool {
		return o.Kind == OpSilence && o.Dur.At(1) == 5 && o.Dur.At(4) == 5
	}) {
		t.Error("Hekata «Серное облако» must silence for a flat 5s at every rank")
	}
}

// TestHekataCorruptionEnemyOnlySlows: «Выбор скверны» — the enemy branch must ONLY slow;
// the prior "solo aid" self-speed buff on the enemy TargetSide was never described.
func TestHekataCorruptionEnemyOnlySlows(t *testing.T) {
	sk := skillOf(t, "Avtr_Dsb_Hekata", 3)
	for _, op := range sk.Ops {
		if op.TargetSide == "enemy" && !(op.Kind == OpSlow) {
			t.Errorf("Hekata «Выбор скверны» enemy-side op must only be OpSlow, got %+v", op)
		}
	}
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpSlow && o.TargetSide == "enemy" }) {
		t.Error("Hekata «Выбор скверны» must still slow the enemy branch")
	}
	if !anyOp(sk.Ops, func(o Op) bool {
		return o.Kind == OpBuffStat && o.Stat == "move_speed_pct" && o.TargetSide == "ally"
	}) {
		t.Error("Hekata «Выбор скверны» must still speed up the ally branch")
	}
}

// TestTangrenCounterattackIsChanceProc: «Контратака» is a chance-gated ON-DAMAGED proc
// that counters with Tangren's own attack, not a deterministic thorns_pct reflect (whose
// Value array never even matched the surviving probability TipArgs).
func TestTangrenCounterattackIsChanceProc(t *testing.T) {
	sk := skillOf(t, "Avtr_HK_Tangren", 2)
	if anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpBuffStat && o.Stat == "thorns_pct" }) {
		t.Error("Tangren «Контратака» must no longer use a deterministic thorns_pct reflect")
	}
	if !anyOp(sk.Ops, func(o Op) bool {
		return o.Kind == OpProc && o.OnDamaged && o.Chance.At(1) > 0 && o.Chance.At(1) < 1
	}) {
		t.Error("Tangren «Контратака» must be a chance-gated (0<Chance<1) OnDamaged proc")
	}
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpAttackDamage }) {
		t.Error("Tangren «Контратака» must counter with the caster's own attack (OpAttackDamage)")
	}
}

// --- pass-8 audit (2026-07-21): second-opinion re-audit of 10 avatars (2 never fully
// swept -- Neirofim, Morlokay -- plus 8 from the oldest full-rigor pass, pass 4) using the
// same client-locale methodology plus adversarial verify. Dutnik came back clean.

// TestNeirofimEnergyReturnIsSkillTriggeredNova: «Обращение энергии» must be a reactive
// OnDamaged+SkillOnly proc (fires only when a mob/boss SKILL hits Neirofim, not a basic
// attack) that heals+restores mana+blasts an equal magic nova, not a permanent regen-rate
// buff with no trigger and no damage component at all.
func TestNeirofimEnergyReturnIsSkillTriggeredNova(t *testing.T) {
	sk := skillOf(t, "Avtr_Sp_Neirofim", 2)
	var proc *Op
	for i := range sk.Ops {
		if sk.Ops[i].Kind == OpProc {
			proc = &sk.Ops[i]
		}
	}
	if proc == nil || !proc.OnDamaged || !proc.SkillOnly {
		t.Fatalf("Neirofim «Обращение энергии» must be an OnDamaged+SkillOnly proc, got %+v", sk.Ops)
	}
	if !anyOp(proc.Ops, func(o Op) bool { return o.Kind == OpHeal }) {
		t.Error("Neirofim «Обращение энергии» must heal self")
	}
	if !anyOp(proc.Ops, func(o Op) bool { return o.Kind == OpManaRestore }) {
		t.Error("Neirofim «Обращение энергии» must restore mana")
	}
	if !anyOp(proc.Ops, func(o Op) bool { return o.Kind == OpDamage && o.Radius > 0 }) {
		t.Error("Neirofim «Обращение энергии» must deal an AoE magic nova")
	}
}

// TestGektorRevengeIsBasicAttackOnly: «Реванш» must gate on BasicAttackOnly -- the client
// scopes the trigger to «урона от базовой атаки», excluding mob/boss SKILL damage.
func TestGektorRevengeIsBasicAttackOnly(t *testing.T) {
	sk := skillOf(t, "Avtr_Tank_Gektor", 2)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpProc && o.OnDamaged && o.BasicAttackOnly }) {
		t.Error("Gektor «Реванш» must be a BasicAttackOnly OnDamaged proc")
	}
}

// TestRognarGraveColdIsRandomized: «Могильный холод» hits «двух СЛУЧАЙНЫХ целях», so both
// the DoT and slow must set Randomize (nearest-first was the prior, wrong encoding).
func TestRognarGraveColdIsRandomized(t *testing.T) {
	sk := skillOf(t, "Avtr_Tank_Rognar", 3)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpDot && o.MaxTargets == 2 && o.Randomize }) {
		t.Error("Rognar «Могильный холод» DoT must be MaxTargets:2 + Randomize")
	}
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpSlow && o.MaxTargets == 2 && o.Randomize }) {
		t.Error("Rognar «Могильный холод» slow must be MaxTargets:2 + Randomize")
	}
}

// TestLirveinOnKillRewardIsFlatDamage: the «Изощренный бросок» on-kill attack bonus must
// match its own damageMod/damageModSP TipArgs -- a flat, SP-scaled bonus, not a constant
// ×1.3 dmg_pct multiplier at every rank.
func TestLirveinOnKillRewardIsFlatDamage(t *testing.T) {
	sk := skillOf(t, "Avtr_DPS_Lirvein", 2)
	var onKill *Op
	for i := range sk.Ops {
		if sk.Ops[i].Kind == OpOnKill {
			onKill = &sk.Ops[i]
		}
	}
	if onKill == nil {
		t.Fatal("Lirvein «Изощренный бросок» must carry an OpOnKill")
	}
	if anyOp(onKill.Ops, func(o Op) bool { return o.Kind == OpBuffStat && o.Stat == "dmg_pct" }) {
		t.Error("Lirvein's on-kill reward must not be a dmg_pct multiplier")
	}
	if !anyOp(onKill.Ops, func(o Op) bool {
		return o.Kind == OpBuffStat && o.Stat == "dmg_flat" && o.PerSP > 0 && o.Value.At(1) == 15
	}) {
		t.Error("Lirvein's on-kill reward must be a flat, SP-scaled dmg_flat bonus (15 at rank 1)")
	}
}

// TestSigilionUnbreakableScalesWithSP: «Несокрушимость» must carry PerSP -- its own
// tooltip already promises a per-spell-power HP addition (damageSP), which the engine
// silently dropped without PerSP set.
func TestSigilionUnbreakableScalesWithSP(t *testing.T) {
	sk := skillOf(t, "Avtr_Tank_Sigilion", 3)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpBuffStat && o.Stat == "max_hp" && o.PerSP > 0 }) {
		t.Error("Sigilion «Несокрушимость» max_hp buff must carry PerSP>0")
	}
}

// TestShinDalarPoisonScalesWithSP / TestShinDalarWoundOpeningScalesWithSP: both skills'
// own TipArgs promise a per-spell-power addition (damageSP/damagePoisonSP) that the engine
// silently dropped without PerSP set on the op.
func TestShinDalarPoisonScalesWithSP(t *testing.T) {
	sk := skillOf(t, "Avtr_HK_ShinDalar", 3)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpDot && o.PerSP > 0 }) {
		t.Error("ShinDalar «Отравление» DoT must carry PerSP>0")
	}
}

func TestShinDalarWoundOpeningScalesWithSP(t *testing.T) {
	sk := skillOf(t, "Avtr_HK_ShinDalar", 4)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpConsumeDots && o.PerSP > 0 }) {
		t.Error("ShinDalar «Вскрытие ран» OpConsumeDots must carry PerSP>0")
	}
}

// TestAbominatorFleshThrowRefundsOnHit: «Бросок плоти» self-damage cost must be
// RefundIfHit (refunded when the throw connects, per «может восстановить, ударив цель»),
// with no fabricated lifesteal_pct buff (no locale field for this skill mentions it).
func TestAbominatorFleshThrowRefundsOnHit(t *testing.T) {
	sk := skillOf(t, "Avtr_HK_Abominator", 1)
	if !anyOp(sk.Ops, func(o Op) bool {
		return o.Kind == OpDamage && o.Apply == "self" && o.RefundIfHit
	}) {
		t.Error("Abominator «Бросок плоти» self-cost must be RefundIfHit")
	}
	if anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpBuffStat && o.Stat == "lifesteal_pct" }) {
		t.Error("Abominator «Бросок плоти» must not grant a fabricated lifesteal_pct buff")
	}
}

// TestAbominatorDevouringDrainsMaxHP: «Пожирание» must reduce the victim's MAX HP each
// channel tick (client: «уменьшается количество текущего И МАКСИМАЛЬНОГО здоровья»), not
// only deal ordinary damage via OpLifestealHit.
func TestAbominatorDevouringDrainsMaxHP(t *testing.T) {
	sk := skillOf(t, "Avtr_HK_Abominator", 2)
	var channel *Op
	for i := range sk.Ops {
		if sk.Ops[i].Kind == OpChannel {
			channel = &sk.Ops[i]
		}
	}
	if channel == nil {
		t.Fatal("Abominator «Пожирание» must carry an OpChannel")
	}
	if !anyOp(channel.Ops, func(o Op) bool { return o.Kind == OpDrainMaxHP && o.PerSP > 0 }) {
		t.Error("Abominator «Пожирание» channel must carry an SP-scaled OpDrainMaxHP")
	}
}

// TestAriannaSaviorsTouchNoDoubleRegen: «Касание спасителя» — «его И себя» get the
// IDENTICAL regen bonus; the prior encoding stacked an extra self-only hp_regen buff on
// top of Ariana's own OpHot, healing her at roughly double the ally's rate.
func TestAriannaSaviorsTouchNoDoubleRegen(t *testing.T) {
	sk := skillOf(t, "Avtr_Sp_Arianna", 4)
	if anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpBuffStat && o.Stat == "hp_regen" }) {
		t.Error("Arianna «Касание спасителя» must not stack an extra self-only hp_regen buff")
	}
	hots := 0
	for _, op := range sk.Ops {
		if op.Kind == OpHot {
			hots++
			if op.PerSP <= 0 {
				t.Error("Arianna «Касание спасителя» OpHot must carry PerSP>0")
			}
		}
	}
	if hots != 2 {
		t.Errorf("Arianna «Касание спасителя» must have exactly 2 OpHot (ally + self), got %d", hots)
	}
}

// TestMorlokayShackleIsEscalating: «Кабала» channel damage must ramp by Growth (+20/sec),
// not stay flat at the arithmetic mean of the described ramp.
func TestMorlokayShackleIsEscalating(t *testing.T) {
	sk := skillOf(t, "Avtr_Dsb_Morlokay", 2)
	var channel *Op
	for i := range sk.Ops {
		if sk.Ops[i].Kind == OpChannel {
			channel = &sk.Ops[i]
		}
	}
	if channel == nil {
		t.Fatal("Morlokay «Кабала» must carry an OpChannel")
	}
	if !anyOp(channel.Ops, func(o Op) bool {
		return o.Kind == OpDamage && o.Growth.At(1) == 20 && o.Value.At(1) == 25
	}) {
		t.Error("Morlokay «Кабала» tick must start at 25 and ramp by Growth 20/sec")
	}
}

// TestMorlokayArcaneBurstIsGuaranteed: «Всполох магии» fires on EVERY basic attack per all
// 6 locale fields (no chance/probability wording anywhere), not the prior fabricated 25-40%
// roll.
func TestMorlokayArcaneBurstIsGuaranteed(t *testing.T) {
	sk := skillOf(t, "Avtr_Dsb_Morlokay", 3)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpProc && o.Chance.At(1) == 1 }) {
		t.Error("Morlokay «Всполох магии» must be a guaranteed (Chance:1) proc")
	}
}

// --- pass-9 audit (2026-07-21, second-opinion re-audit of pass-5's 10 avatars) ---

// TestOnKillProcsFlagged: only Gayal's zombie-raise proc fires on enemy DEATH rather than
// on landing a hit; nothing else should carry Op.OnKill.
func TestOnKillProcsFlagged(t *testing.T) {
	want := map[string]int{"Avtr_DPS_Gayal": 2} // «Аура погибших»
	for prefab, slot := range want {
		sk := skillOf(t, prefab, slot)
		if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpProc && o.OnKill }) {
			t.Errorf("%s slot %d («%s») must have an OnKill proc", prefab, slot, sk.NameRu)
		}
	}
	for prefab, ks := range skillsByPrefab {
		for i, sk := range ks.Skills {
			if want[prefab] == i+1 {
				continue
			}
			if anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpProc && o.OnKill }) {
				t.Errorf("%s slot %d unexpectedly has an OnKill proc", prefab, i+1)
			}
		}
	}
}

// TestOnAnyDamageProcsFlagged: only Anhel's clone-summon proc fires on ANY damage the
// avatar deals (client's generic «При нанесении урона»), not just a basic-attack landing.
func TestOnAnyDamageProcsFlagged(t *testing.T) {
	want := map[string]int{"Avtr_Psh_Anhel": 3} // «Зов фантомов»
	for prefab, slot := range want {
		sk := skillOf(t, prefab, slot)
		if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpProc && o.OnAnyDamage }) {
			t.Errorf("%s slot %d («%s») must have an OnAnyDamage proc", prefab, slot, sk.NameRu)
		}
	}
	for prefab, ks := range skillsByPrefab {
		for i, sk := range ks.Skills {
			if want[prefab] == i+1 {
				continue
			}
			if anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpProc && o.OnAnyDamage }) {
				t.Errorf("%s slot %d unexpectedly has an OnAnyDamage proc", prefab, i+1)
			}
		}
	}
}

// TestCerberFerocitySelfBuffIsFlatDamage: «Свирепость» grants a flat, SP-scaled damage
// bonus (dmg_flat), not the prior undocumented %-multiplier (dmg_pct) with no SP term.
func TestCerberFerocitySelfBuffIsFlatDamage(t *testing.T) {
	sk := skillOf(t, "Avtr_DPS_Cerber", 2)
	if anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpBuffStat && o.Stat == "dmg_pct" }) {
		t.Error("Cerber «Свирепость» must not use dmg_pct (client describes a flat bonus)")
	}
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpBuffStat && o.Stat == "dmg_flat" && o.PerSP > 0 }) {
		t.Error("Cerber «Свирепость» must grant an SP-scaled dmg_flat bonus")
	}
}

// TestCerberBloodyFeastCapScalesWithSP: the heal-on-kill cap must scale with spell power
// per the tooltip's «не более {*healMax}+{*@damageSP}».
func TestCerberBloodyFeastCapScalesWithSP(t *testing.T) {
	sk := skillOf(t, "Avtr_DPS_Cerber", 3)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpHealOnKill && o.PerSP > 0 }) {
		t.Error("Cerber «Кровавый пир» heal cap must carry PerSP>0")
	}
}

// TestCerberDeathMarkDotScalesWithSP: the silence DoT must carry PerSP per the tooltip.
func TestCerberDeathMarkDotScalesWithSP(t *testing.T) {
	sk := skillOf(t, "Avtr_DPS_Cerber", 4)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpDot && o.PerSP > 0 }) {
		t.Error("Cerber «Знак смертника» DoT must carry PerSP>0")
	}
}

// TestTitanidEarthquakeGrowthScalesWithSP: each wave's damage INCREMENT (not just the base
// hit) must scale with spell power, per the tooltip's separate «damageIncSP» term.
func TestTitanidEarthquakeGrowthScalesWithSP(t *testing.T) {
	sk := skillOf(t, "Avtr_Tank_Titanid", 1)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpDamage && o.GrowthPerSP > 0 }) {
		t.Error("Titanid «Землетрясение» wave growth must carry GrowthPerSP>0")
	}
}

// TestTitanidShockwaveSplashExcludesPrimaryAndHalvesSP: the splash damage/slow must exclude
// the primary target («все ДРУГИЕ враги») and the splash must scale at half the SP rate the
// tooltip already declares (damageAoeSP=0.5).
func TestTitanidShockwaveSplashExcludesPrimaryAndHalvesSP(t *testing.T) {
	sk := skillOf(t, "Avtr_Tank_Titanid", 2)
	foundSplash, foundSlow := false, false
	for _, op := range sk.Ops {
		if op.Kind == OpDamage && op.Radius > 0 {
			foundSplash = true
			if !op.ExcludeCenterTarget {
				t.Error("Titanid «Ударная волна» splash must ExcludeCenterTarget")
			}
			if op.PerSP != 0.5 {
				t.Errorf("Titanid «Ударная волна» splash PerSP = %g, want 0.5", op.PerSP)
			}
		}
		if op.Kind == OpSlow {
			foundSlow = true
			if !op.ExcludeCenterTarget {
				t.Error("Titanid «Ударная волна» slow must ExcludeCenterTarget")
			}
		}
	}
	if !foundSplash || !foundSlow {
		t.Fatal("Titanid «Ударная волна» must have both a splash OpDamage and an OpSlow")
	}
}

// TestTitanidGigantismScalesWithSP: the attack bonus must carry its own SP-scaled additive
// term (client's «{*damageMod}+{*@damageSP}»), on top of the existing dmg_pct.
func TestTitanidGigantismScalesWithSP(t *testing.T) {
	sk := skillOf(t, "Avtr_Tank_Titanid", 4)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpBuffStat && o.Stat == "dmg_flat" && o.PerSP > 0 }) {
		t.Error("Titanid «Гигантизм» must grant an SP-scaled dmg_flat bonus")
	}
}

// TestGayalDeadAuraZombieScalesWithSP: the raised zombie's HP/Dmg must scale with spell
// power per the tooltip's «{*healthMod}+{*@damageHealthSP}» / «{*dmgMod}+{*@damageSP}».
func TestGayalDeadAuraZombieScalesWithSP(t *testing.T) {
	sk := skillOf(t, "Avtr_DPS_Gayal", 2)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpSummon && o.HpPerSP > 0 && o.DmgPerSP > 0 }) {
		t.Error("Gayal «Аура погибших» zombie must carry HpPerSP>0 and DmgPerSP>0")
	}
}

// TestAstarotDarkServantGrantsRealStealth: «Слуга тьмы» must carry a real OpStealth --
// the client's own text opens with «становится невидимым» (becomes invisible).
func TestAstarotDarkServantGrantsRealStealth(t *testing.T) {
	sk := skillOf(t, "Avtr_HK_Astarot", 2)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpStealth }) {
		t.Error("Astarot «Слуга тьмы» must grant real invisibility (OpStealth)")
	}
}

// TestAstarotDevilTrickManaCostIsFlat: the client locale states a flat "Стоимость 135 маны"
// for «Бесовский трюк» across all ranks (its Cooldown is already flat 15s at every rank,
// corroborating the mana cost is meant to be flat too), not a {45,50,55,60} per-rank climb.
func TestAstarotDevilTrickManaCostIsFlat(t *testing.T) {
	sk := skillOf(t, "Avtr_HK_Astarot", 1)
	for i, cost := range sk.ManaCost {
		if cost != 135 {
			t.Errorf("Astarot «Бесовский трюк» rank %d: ManaCost = %d, want flat 135", i+1, cost)
		}
	}
}

// TestBlackDragonFrenzyPenaltyIsFlatConstant: the self-penalty must be a flat, rank-CONSTANT
// +20% damage taken (dmg_reduction_pct going negative to amplify), not a rank-scaling armor
// debuff that drifts away from 20% at every rank.
func TestBlackDragonFrenzyPenaltyIsFlatConstant(t *testing.T) {
	sk := skillOf(t, "Avtr_DPS_BlackDragon", 1)
	if anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpBuffStat && o.Stat == "phys_armor" }) {
		t.Error("BlackDragon «Неистовство» must not use a phys_armor self-debuff")
	}
	found := false
	for _, op := range sk.Ops {
		if op.Kind == OpBuffStat && op.Stat == "dmg_reduction_pct" {
			found = true
			for i := 0; i < 4; i++ {
				if op.Value.At(i) != -0.2 {
					t.Errorf("BlackDragon «Неистовство» dmg_reduction_pct rank %d = %g, want -0.2 (flat)", i, op.Value.At(i))
				}
			}
		}
	}
	if !found {
		t.Error("BlackDragon «Неистовство» must carry a dmg_reduction_pct self-debuff")
	}
}

// TestBlackDragonManaBurnAndBloodScaleWithSP: both «Выжигание маны» (mana burn) and
// «Кровь дракона» (reactive DoT) must carry PerSP per their tooltips.
func TestBlackDragonManaBurnAndBloodScaleWithSP(t *testing.T) {
	s2 := skillOf(t, "Avtr_DPS_BlackDragon", 2)
	if !anyOp(s2.Ops, func(o Op) bool { return o.Kind == OpManaBurnHit && o.PerSP > 0 }) {
		t.Error("BlackDragon «Выжигание маны» must carry PerSP>0")
	}
	s3 := skillOf(t, "Avtr_DPS_BlackDragon", 3)
	if !anyOp(s3.Ops, func(o Op) bool { return o.Kind == OpDot && o.PerSP > 0 }) {
		t.Error("BlackDragon «Кровь дракона» DoT must carry PerSP>0")
	}
}

// TestWilfangPoisonBiteTickIsInertExplosionScalesWithSP: the periodic tick must deal NO
// damage (the client never describes one), while the death explosion carries its own,
// separate SP scaling (Op.ExplodeSP) decoupled from the (now zero) tick PerSP.
func TestWilfangPoisonBiteTickIsInertExplosionScalesWithSP(t *testing.T) {
	sk := skillOf(t, "Avtr_Dsb_Wilfang", 3)
	found := false
	for _, op := range sk.Ops {
		for _, nested := range op.Ops {
			if nested.Kind != OpDot {
				continue
			}
			found = true
			if nested.Value.At(0) != 0 {
				t.Errorf("Wilfang «Ядовитый укус» tick Value = %g, want 0 (client never describes tick damage)", nested.Value.At(0))
			}
			if nested.PerSP != 0 {
				t.Errorf("Wilfang «Ядовитый укус» tick PerSP = %g, want 0 (moved to ExplodeSP)", nested.PerSP)
			}
			if nested.ExplodeSP <= 0 {
				t.Error("Wilfang «Ядовитый укус» explosion must carry its own ExplodeSP>0")
			}
		}
	}
	if !found {
		t.Fatal("Wilfang «Ядовитый укус» must carry a nested OpDot")
	}
}

// TestPlusMinusElectricShockGrowsWithDistance: the ring nova must deal MORE damage to
// farther enemies (Op.PerTargetGrowth), not a single flat number for everyone in radius.
func TestPlusMinusElectricShockGrowsWithDistance(t *testing.T) {
	sk := skillOf(t, "Avtr_Dsb_PlusMinus", 1)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpDamage && o.PerTargetGrowth.At(0) > 0 }) {
		t.Error("PlusMinus «Электрошок» must carry PerTargetGrowth>0 (farther enemy = more damage)")
	}
}

// TestPlusMinusSuperconductivityExcludesStruckMob: the chain must exclude the already-struck
// mob it centers on («четырем СОСЕДНИМ целям»), not re-hit it as one of its own 4 targets.
func TestPlusMinusSuperconductivityExcludesStruckMob(t *testing.T) {
	sk := skillOf(t, "Avtr_Dsb_PlusMinus", 3)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpDamage && o.ExcludeCenterTarget }) {
		t.Error("PlusMinus «Сверхпроводимость» chain must ExcludeCenterTarget")
	}
}

// TestPlusMinusBallLightningManaBurnScalesWithSP: the mana-burn component must carry PerSP
// per the tooltip's «+{*damageManaSP} единиц за каждую единицу силы заклинаний».
func TestPlusMinusBallLightningManaBurnScalesWithSP(t *testing.T) {
	sk := skillOf(t, "Avtr_Dsb_PlusMinus", 4)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpManaBurnArea && o.PerSP > 0 }) {
		t.Error("PlusMinus «Шаровая молния» mana burn must carry PerSP>0")
	}
}

// TestSharliHeatOfSoulHasBuffIcon: the client ships a dedicated _BuffSelf_LongDesc tooltip
// for this passive, and the engine's buff-icon loop is Op-kind agnostic (Velial/Sandariel
// precedent), so there is no reason to suppress the icon.
func TestSharliHeatOfSoulHasBuffIcon(t *testing.T) {
	sk := skillOf(t, "Avtr_DPS_Sharli", 2)
	if !sk.BuffIcon {
		t.Error("Sharli «Жар души» must have BuffIcon:true")
	}
}

// TestGellarArmySoulsIsCappedAndRandom: per-tick damage must hit a capped, RANDOM subset of
// enemies («СЛУЧАЙНЫМ врагам»), not every enemy in radius like s1's «всем врагам».
func TestGellarArmySoulsIsCappedAndRandom(t *testing.T) {
	sk := skillOf(t, "Avtr_DPS_Gellar", 4)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpDamage && o.MaxTargets > 0 && o.Randomize }) {
		t.Error("Gellar «Армия душ» must cap targets with MaxTargets+Randomize")
	}
}

// --- pass-10 audit (2026-07-22, re-audit of pass-6's 10 avatars via a local Ollama harness,
// cross-checked against 3 models + manually re-derived by hand) ---

// TestFrostGraveOfColdDotScalesWithSP: the DoT half of «Гробница холода» must carry PerSP -
// the OpHeal on the same skill already did, an asymmetry the client's damageHealSP/damageSP
// pair (both present) rules out.
func TestFrostGraveOfColdDotScalesWithSP(t *testing.T) {
	sk := skillOf(t, "Avtr_Dsb_Frost", 3)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpDot && o.TargetSide == "enemy" && o.PerSP > 0 }) {
		t.Error("Frost «Гробница холода» enemy DoT must carry PerSP>0")
	}
}

// TestKionaForestGuardianBothSidesScaleWithSP: «Страж леса» damages an enemy OR heals an
// ally (dual cast); the client's single {*damageSP} placeholder applies to both branches.
func TestKionaForestGuardianBothSidesScaleWithSP(t *testing.T) {
	sk := skillOf(t, "Avtr_Sp_Kiona", 4)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpDot && o.PerSP > 0 }) {
		t.Error("Kiona «Страж леса» enemy DoT must carry PerSP>0")
	}
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpHot && o.PerSP > 0 }) {
		t.Error("Kiona «Страж леса» ally HoT must carry PerSP>0")
	}
}

// TestGrimlokDarkSideHealthBoostScalesWithSP: the client's «{*healthBoost}+{*@damageSP}»
// promises SP scaling on the flat HP grant; OpBuffStat already supports PerSP generically
// (Titanid s4 precedent).
func TestGrimlokDarkSideHealthBoostScalesWithSP(t *testing.T) {
	sk := skillOf(t, "Avtr_HK_Grimlok", 4)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpBuffStat && o.Stat == "max_hp" && o.PerSP > 0 }) {
		t.Error("Grimlok «Темная сторона» max_hp buff must carry PerSP>0")
	}
}

// TestVelialCrueltyDamageScalesWithSP: the missing-HP-scaled proc damage needs PerSP - its
// Scale is "phys", not "magic", so it does NOT get skillDamageLocked's implicit magic-Scale
// SP fallback the way a Scale:"magic" op would.
func TestVelialCrueltyDamageScalesWithSP(t *testing.T) {
	sk := skillOf(t, "Avtr_Tank_Velial", 3)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpDamage && len(o.CasterMissingHP) > 0 && o.PerSP > 0 }) {
		t.Error("Velial «Жестокость» missing-HP proc damage must carry PerSP>0")
	}
}

// TestEdiliaForgetfulPollenSlowIsBasicAttackOnly: client's «При получении удара от базовой
// атаки» scopes the retaliation slow to basic attacks - it must not fire off arbitrary skill
// damage landing on Edilia.
func TestEdiliaForgetfulPollenSlowIsBasicAttackOnly(t *testing.T) {
	sk := skillOf(t, "Avtr_Dsb_Edilia", 3)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpProc && o.OnDamaged && o.BasicAttackOnly }) {
		t.Error("Edilia «Пыльца забвения» retaliation-slow proc must carry BasicAttackOnly:true")
	}
}

// TestNerlagMeatGrinderScalesWithLiveAttack: client's «{aoeDamageCoef}% от базовой атаки»
// ties each hit to Nerlag's CURRENT base attack (gear/buffs), not a flat number baked in at
// authoring time - the flat Value must be gone in favor of PctOfAttack.
func TestNerlagMeatGrinderScalesWithLiveAttack(t *testing.T) {
	sk := skillOf(t, "Avtr_DPS_Nerlag", 2)
	found := false
	var walk func(ops []Op)
	walk = func(ops []Op) {
		for _, o := range ops {
			if o.Kind == OpDamage {
				if len(o.Value) > 0 {
					t.Error("Nerlag «Мясорубка» damage must not also carry a flat Value alongside PctOfAttack")
				}
				if len(o.PctOfAttack) > 0 && o.PctOfAttack.At(1) > 0 {
					found = true
				}
			}
			walk(o.Ops)
		}
	}
	walk(sk.Ops)
	if !found {
		t.Error("Nerlag «Мясорубка» damage must carry PctOfAttack>0")
	}
}

// --- pass-11 audit (2026-07-23, re-audit of pass-7's 6 avatars: Teridin/Zamaran/Veritas/
// Hekata/Tangren/Mihalych, via the local Ollama harness with gpt-oss:20b) ---

// TestMihalychBearTrapDamageScalesWithSP: the chain-pull damage is Scale:"phys", so it does
// NOT get skillDamageLocked's implicit magic-Scale SP fallback - it needs an explicit PerSP.
func TestMihalychBearTrapDamageScalesWithSP(t *testing.T) {
	sk := skillOf(t, "Avtr_HK_Mihalych", 1)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpDamage && o.Scale == "phys" && o.PerSP > 0 }) {
		t.Error("Mihalych «Медвежий капкан» damage must carry PerSP>0")
	}
}

// TestMihalychThrashingConsecutiveHitScalesWithSP: the client's per-hit damage increment
// itself carries an SP-scaled term, not just a flat per-rank number.
func TestMihalychThrashingConsecutiveHitScalesWithSP(t *testing.T) {
	sk := skillOf(t, "Avtr_HK_Mihalych", 2)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpConsecutiveHit && o.PerSP > 0 }) {
		t.Error("Mihalych «Трепка» consecutive-hit bonus must carry PerSP>0")
	}
}

// TestHekataCultOfTheReaperKillStackScalesWithSP: the on-kill attack-stack increment must
// carry PerSP - OpOnKillStack previously had zero spell-power support at all.
func TestHekataCultOfTheReaperKillStackScalesWithSP(t *testing.T) {
	sk := skillOf(t, "Avtr_Dsb_Hekata", 2)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpOnKillStack && o.PerSP > 0 }) {
		t.Error("Hekata «Культ жнеца» kill-stack must carry PerSP>0")
	}
}

// TestVeritasMetamorphosisBuffsViewRadius: the client explicitly lists view radius among
// Metamorphosis's temporary bonuses (alongside HP/damage/move speed); it must have its own
// buff op now that view_radius_pct is a real, synced stat.
func TestVeritasMetamorphosisBuffsViewRadius(t *testing.T) {
	sk := skillOf(t, "Avtr_Tank_Veritas", 4)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpBuffStat && o.Stat == "view_radius_pct" }) {
		t.Error("Veritas «Метаморфоза» must buff view_radius_pct")
	}
}

// TestZamaranChargePushesEnemiesAside: the client explicitly separates "shoves enemies
// aside" (during the charge) from the arrival-point slow+damage - PushAside models the
// first half, which used to be entirely unmodeled.
func TestZamaranChargePushesEnemiesAside(t *testing.T) {
	sk := skillOf(t, "Avtr_Tank_Zamaran", 1)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpDash && o.PushAside > 0 }) {
		t.Error("Zamaran «Таран» dash must carry PushAside>0")
	}
}

// TestZamaranMoltenMetalArmorDebuffIsPercentage: the client frames the armor reduction as a
// PERCENTAGE ({coef%}%, matching the existing coef TipArgs), not flat armor points.
func TestZamaranMoltenMetalArmorDebuffIsPercentage(t *testing.T) {
	sk := skillOf(t, "Avtr_Tank_Zamaran", 3)
	if !anyOp(sk.Ops, func(o Op) bool {
		return o.Kind == OpBuffStat && o.Stat == "armor_pct" && o.Value.At(1) < 1 && o.Value.At(1) > 0
	}) {
		t.Error("Zamaran «Расплавленный металл» armor debuff must be a armor_pct multiplier <1, not flat phys_armor")
	}
}

// --- pass-12 audit (2026-07-23, third-opinion re-audit of pass-8's 10 avatars: Rognar/
// Gektor/Lirvein/Sigilion/ShinDalar via qwen3-coder, Dutnik/Abominator/Arianna/Neirofim/
// Morlokay via gpt-oss:20b - first pass to use BOTH retained local models in parallel) ---

// TestAriannaGuardianShieldGrantsCCImmune: the client explicitly promises full immunity to
// stun/slow while shielded, not just an absorb pool.
func TestAriannaGuardianShieldGrantsCCImmune(t *testing.T) {
	sk := skillOf(t, "Avtr_Sp_Arianna", 1)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpShield && o.GrantsCCImmune }) {
		t.Error("Arianna «Щит хранителя» shield must carry GrantsCCImmune")
	}
}

// TestMorlokayVoodooCurseDotScalesWithSP: the DoT tick is missing PerSP despite the client's
// own damageSP TipArg promising spell-power scaling.
func TestMorlokayVoodooCurseDotScalesWithSP(t *testing.T) {
	sk := skillOf(t, "Avtr_Dsb_Morlokay", 1)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpDot && o.PerSP > 0 }) {
		t.Error("Morlokay «Проклятие Вуду» DoT must carry PerSP>0")
	}
}

// TestMorlokayBondageChannelRampScalesWithSP: the client's damageIncSP term means the RAMP
// itself (not just the base tick) must scale with spell power.
func TestMorlokayBondageChannelRampScalesWithSP(t *testing.T) {
	sk := skillOf(t, "Avtr_Dsb_Morlokay", 2)
	var found bool
	var walk func(ops []Op)
	walk = func(ops []Op) {
		for _, o := range ops {
			if o.Kind == OpDamage && o.Growth.At(1) > 0 && o.GrowthPerSP > 0 {
				found = true
			}
			walk(o.Ops)
		}
	}
	walk(sk.Ops)
	if !found {
		t.Error("Morlokay «Кабала» channel ramp must carry GrowthPerSP>0")
	}
}

// TestMorlokayStormTotemDamageScalesWithSP: the totem's zap is missing DmgPerSP despite the
// client's own damageSP TipArg.
func TestMorlokayStormTotemDamageScalesWithSP(t *testing.T) {
	sk := skillOf(t, "Avtr_Dsb_Morlokay", 4)
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpSummon && o.DmgPerSP > 0 }) {
		t.Error("Morlokay «Грозовой тотем» summon must carry DmgPerSP>0")
	}
}
