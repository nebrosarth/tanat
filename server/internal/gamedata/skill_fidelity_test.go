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
		"Avtr_Tank_Titanid": 3, // «Каменная кожа»
		"Avtr_Tank_Gektor":  2, // «Реванш»
		"Avtr_HK_Dutnik":    3, // «Детонация»
		"Avtr_DPS_Nerlag":   3, // «Прилив крови»
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

// TestKionaCloakIsAllyShield: «Лесной покров» is a self/ally absorb shield, not an
// enemy-targeted damage debuff.
func TestKionaCloakIsAllyShield(t *testing.T) {
	sk := skillOf(t, "Avtr_Sp_Kiona", 2)
	if sk.Target == "ENEMY+NOT_BUILDING" {
		t.Error("Kiona «Лесной покров» must not target an enemy")
	}
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpShield }) {
		t.Error("Kiona «Лесной покров» must apply a shield (OpShield)")
	}
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
		{"Avtr_Dsb_Wilfang", 1},   // «Сокрушительный рывок»
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
		{"Avtr_Sp_Arianna", 1},   // «Щит хранителя» -> ally shield
		{"Avtr_Sp_Arianna", 3},   // «Исцеление» -> allies heal
		{"Avtr_Sp_Arianna", 4},   // «Касание спасителя» -> ally shield+hot
		{"Avtr_HK_Tangren", 3},   // «Целительный тотем» -> allies hot
		{"Avtr_Dsb_Edilia", 1},   // «Касание природы» -> allies heal
		{"Avtr_Dsb_Edilia", 4},   // «Дерево жизни» -> allies heal (channel)
		{"Avtr_Sp_Kiona", 1},     // «Лечебная волна» -> allies heal
		{"Avtr_DPS_Sandariel", 2}, // «Прыжок» -> allies speed
	} {
		sk := skillOf(t, tc.prefab, tc.slot)
		if !anyOp(sk.Ops, func(o Op) bool { return o.On == "ally" || o.On == "allies" }) {
			t.Errorf("%s slot %d («%s») must carry an ally-targeting op (On ally/allies)", tc.prefab, tc.slot, sk.NameRu)
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
	if !anyOp(sk.Ops, func(o Op) bool { return o.Kind == OpBuffStat && o.Stat == "armor_pct" }) {
		t.Error("Rognar «Костяной щит» must keep its damage-reduction buff")
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
