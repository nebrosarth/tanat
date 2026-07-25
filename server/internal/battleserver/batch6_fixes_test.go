package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

// Engine/behavioral tests for the 2026-07-21 pass-8 audit batch: a second-opinion
// re-sweep of 10 avatars (Neirofim/Morlokay -- never fully swept before -- plus 8 from
// the oldest full-rigor pass) that found 19 more findings. Dutnik came back clean.
// Structural (data-shape) assertions for the same fixes live in
// gamedata/skill_fidelity_test.go; these confirm the actual runtime behavior.

// TestNeirofimEnergyReturnFiresOnSkillDamageOnly: «Обращение энергии» must react only to
// mob/boss SKILL damage (SkillOnly), healing+restoring mana+blasting a nova -- a basic
// attack must not trigger it at all.
func TestNeirofimEnergyReturnFiresOnSkillDamageOnly(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Sp_Neirofim")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	realOp := hs.kit.Skills[1].Ops[0]
	if realOp.Kind != gamedata.OpProc || !realOp.OnDamaged || !realOp.SkillOnly {
		t.Fatal("Neirofim «Обращение энергии» is no longer a SkillOnly OnDamaged proc; test needs updating")
	}
	hs.skillLevel[1] = 1 // «Обращение энергии» learned
	// newHuntConn builds a bare huntState without running the world-build registration
	// loop (see TestZamaranReviveOnDeath's comment) -- register the real op manually.
	hs.defenseProcs = []procState{{slot: 2, chance: realOp.Chance, ops: realOp.Ops, skillOnly: true}}

	attacker := mkMob(t, 5400, 1, 0)
	attacker.hp, attacker.maxHP = 100000, 100000

	c.mvMu.Lock()
	hs.mobs[attacker.id] = attacker
	hs.tr.add(attacker.id)
	hs.hp, hs.mana = 1, 0

	// Basic-attack damage: must NOT trigger the proc at all.
	hs.lastDamageWasSkill = false
	s.runDefenseProcsLocked(c, attacker, 10, now)
	hpAfterBasic, manaAfterBasic, atkHPAfterBasic := hs.hp, hs.mana, attacker.hp
	if hpAfterBasic != 1 || manaAfterBasic != 0 || atkHPAfterBasic != 100000 {
		c.mvMu.Unlock()
		t.Fatalf("basic-attack damage must not trigger the proc: hp=%g mana=%g attackerHP=%g", hpAfterBasic, manaAfterBasic, atkHPAfterBasic)
	}

	// Skill damage: must heal, restore mana, and nova the attacker (standing at the
	// blast center since ctx.target resolves the AoE origin for an OnDamaged proc).
	hs.lastDamageWasSkill = true
	s.runDefenseProcsLocked(c, attacker, 10, now)
	hpAfter, manaAfter, atkHPAfter := hs.hp, hs.mana, attacker.hp
	c.mvMu.Unlock()

	if hpAfter <= hpAfterBasic {
		t.Errorf("skill damage must heal Neirofim, hp stayed %g", hpAfter)
	}
	if manaAfter <= manaAfterBasic {
		t.Errorf("skill damage must restore mana, mana stayed %g", manaAfter)
	}
	if atkHPAfter >= atkHPAfterBasic {
		t.Errorf("skill damage must nova the attacker, attacker hp stayed %g", atkHPAfter)
	}
}

// TestGektorRevengeDoesNotFireOnSkillDamage: «Реванш» is scoped to «урона от базовой
// атаки» -- BasicAttackOnly must block it when the incoming hit was a mob/boss SKILL.
func TestGektorRevengeDoesNotFireOnSkillDamage(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Gektor")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	realOp := hs.kit.Skills[1].Ops[0]
	if realOp.Kind != gamedata.OpProc || !realOp.OnDamaged || !realOp.BasicAttackOnly {
		t.Fatal("Gektor «Реванш» is no longer a BasicAttackOnly OnDamaged proc; test needs updating")
	}

	attacker := mkMob(t, 5500, 1, 0)
	attacker.hp, attacker.maxHP = 100000, 100000

	c.mvMu.Lock()
	hs.mobs[attacker.id] = attacker
	hs.tr.add(attacker.id)
	// Force a deterministic roll (real Chance is only 17-20%; covered by the gamedata
	// structural test) while still exercising the real nested Ops.
	hs.defenseProcs = []procState{{slot: 2, chance: gamedata.PerLevel{1, 1, 1, 1}, ops: realOp.Ops, basicAttackOnly: true}}
	hs.skillLevel[1] = 1

	startHP := attacker.hp
	hs.lastDamageWasSkill = true
	s.runDefenseProcsLocked(c, attacker, 10, now)
	hpAfterSkill := attacker.hp

	hs.lastDamageWasSkill = false
	s.runDefenseProcsLocked(c, attacker, 10, now)
	hpAfterBasic := attacker.hp
	c.mvMu.Unlock()

	if hpAfterSkill != startHP {
		t.Errorf("Gektor «Реванш» must not fire on mob SKILL damage, attacker hp dropped to %g", hpAfterSkill)
	}
	if hpAfterBasic == hpAfterSkill {
		t.Error("Gektor «Реванш» must fire on basic-attack damage")
	}
}

// TestRognarBoneShieldReducesIncomingDamage: dmg_reduction_pct must cut incoming damage
// by roughly its own fraction -- armor_pct (the prior encoding) could never actually net
// a real 50% reduction through the shared mitigation curve.
func TestRognarBoneShieldReducesIncomingDamage(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Rognar")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	m := mkMob(t, 5600, 1, 0)
	c.mvMu.Lock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)

	startHP := hs.maxHPLocked(now)
	hs.hp = startHP
	s.hitPlayerFromLocked(c, m.id, 100, now, m, nil)
	dmgWithout := startHP - hs.hp

	hs.hp = startHP
	hs.st.mods = append(hs.st.mods, statMod{stat: "dmg_reduction_pct", value: 0.5})
	s.hitPlayerFromLocked(c, m.id, 100, now, m, nil)
	dmgWith := startHP - hs.hp
	c.mvMu.Unlock()

	if dmgWithout <= 0 {
		t.Fatal("baseline hit must deal some damage")
	}
	if dmgWith >= dmgWithout {
		t.Errorf("dmg_reduction_pct must cut incoming damage: without=%g with=%g", dmgWithout, dmgWith)
	}
	if diff := dmgWith - dmgWithout*0.5; diff > 0.5 || diff < -0.5 {
		t.Errorf("dmg_reduction_pct=0.5 should roughly halve the hit: without=%g with=%g (want ~%g)", dmgWithout, dmgWith, dmgWithout*0.5)
	}
}

// TestSigilionCounterReflectIsMeleeOnly: «Удар наотмашь» must reflect only melee-attack
// damage («При получении урона от ближней атаки») -- a ranged mob's hit must not reflect.
func TestSigilionCounterReflectIsMeleeOnly(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Sigilion")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())
	hs.hp = 1000
	hs.st.mods = append(hs.st.mods, statMod{stat: "thorns_pct", value: 0.5})

	meleeMob := mkMob(t, 5700, 1, 0)
	meleeMob.hp, meleeMob.maxHP = 100000, 100000
	rangedIdx := mobIndexByPrefab(t, "Mob_Skeleton_Range_01")
	rangedMob := &mobState{id: 5701, mobIdx: rangedIdx, mob: gamedata.Mobs()[rangedIdx], x: 1, y: 0, hp: 100000, maxHP: 100000, shown: true}

	c.mvMu.Lock()
	hs.mobs[meleeMob.id] = meleeMob
	hs.mobs[rangedMob.id] = rangedMob
	hs.tr.add(meleeMob.id)
	hs.tr.add(rangedMob.id)

	rangedStartHP := rangedMob.hp
	s.hitPlayerFromLocked(c, rangedMob.id, 100, now, rangedMob, nil)
	rangedHPAfter := rangedMob.hp

	meleeStartHP := meleeMob.hp
	s.hitPlayerFromLocked(c, meleeMob.id, 100, now, meleeMob, nil)
	meleeHPAfter := meleeMob.hp
	c.mvMu.Unlock()

	if rangedHPAfter != rangedStartHP {
		t.Errorf("a ranged attacker must not be reflected, hp dropped to %g", rangedHPAfter)
	}
	if meleeHPAfter == meleeStartHP {
		t.Error("a melee attacker must be reflected")
	}
}

// TestAbominatorFleshThrowRefundsSelfCostOnHit: the self-damage cost on «Бросок плоти»
// must be refunded (skipped entirely) when the throw's target is alive, and only actually
// apply when there's no live target to hit.
func TestAbominatorFleshThrowRefundsSelfCostOnHit(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_HK_Abominator")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	var selfOp gamedata.Op
	found := false
	for _, op := range hs.kit.Skills[0].Ops {
		if op.Kind == gamedata.OpDamage && op.Apply == "self" {
			selfOp, found = op, true
		}
	}
	if !found {
		t.Fatal("Abominator «Бросок плоти» must have a self-damage op")
	}
	if !selfOp.RefundIfHit {
		t.Fatal("Abominator's self-damage op must be RefundIfHit")
	}

	m := mkMob(t, 5800, 1, 0)

	c.mvMu.Lock()
	startHP := hs.maxHPLocked(now)
	hs.hp = startHP
	s.applyOpsLocked(c, []gamedata.Op{selfOp}, opCtx{slot: 1, level: 1, target: m}, now)
	hpWithTarget := hs.hp

	hs.hp = startHP
	s.applyOpsLocked(c, []gamedata.Op{selfOp}, opCtx{slot: 1, level: 1, target: nil}, now)
	hpNoTarget := hs.hp
	c.mvMu.Unlock()

	if hpWithTarget != startHP {
		t.Errorf("self-cost must be refunded when the throw connects, hp %g -> %g", startHP, hpWithTarget)
	}
	if hpNoTarget >= startHP {
		t.Errorf("self-cost must still apply with no live target, hp stayed %g", hpNoTarget)
	}
}

// TestAbominatorDevouringReducesTargetMaxHP: «Пожирание»'s channel must permanently
// shrink the victim's MAX hp each tick (not just deal ordinary damage), clamping current
// hp down to match.
func TestAbominatorDevouringReducesTargetMaxHP(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_HK_Abominator")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	var drainOp gamedata.Op
	for _, op := range hs.kit.Skills[1].Ops {
		if op.Kind != gamedata.OpChannel {
			continue
		}
		for _, nested := range op.Ops {
			if nested.Kind == gamedata.OpDrainMaxHP {
				drainOp = nested
			}
		}
	}
	if drainOp.Kind != gamedata.OpDrainMaxHP {
		t.Fatal("Abominator «Пожирание» channel must carry an OpDrainMaxHP")
	}

	m := mkMob(t, 5900, 1, 0)
	m.maxHP, m.hp = 1000, 1000

	c.mvMu.Lock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	s.applyOpsLocked(c, []gamedata.Op{drainOp}, opCtx{slot: 2, level: 1, target: m}, now)
	c.mvMu.Unlock()

	if m.maxHP >= 1000 {
		t.Errorf("target max HP must be reduced, stayed %g", m.maxHP)
	}
	if m.hp > m.maxHP {
		t.Errorf("target current HP must be clamped down to the new max: hp=%g maxHP=%g", m.hp, m.maxHP)
	}
}

// TestMorlokayTotemPicksRandomTargets: the stationary «Грозовой тотем» must zap a RANDOM
// enemy in range each swing («случайным врагам вокруг»), not always the nearest one --
// otherwise a single mob parked next to it could tank every bolt forever.
func TestMorlokayTotemPicksRandomTargets(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Morlokay")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	mobs := []*mobState{mkMob(t, 6001, 1, 0), mkMob(t, 6002, -1, 0), mkMob(t, 6003, 0, 1), mkMob(t, 6004, 0, -1)}
	c.mvMu.Lock()
	for _, m := range mobs {
		m.hp, m.maxHP = 100000, 100000
		hs.mobs[m.id] = m
		hs.tr.add(m.id)
	}
	sm := &summonState{id: 6100, hp: 500, maxHP: 500, until: now + 1000, x: 0, y: 0, stationary: true, dmg: 10}
	hs.summons[sm.id] = sm

	hit := map[int32]bool{}
	for i := 0; i < 60; i++ {
		before := make(map[int32]float64, len(mobs))
		for _, m := range mobs {
			before[m.id] = m.hp
		}
		sm.nextSwing = 0 // eligible to swing this tick regardless of attack speed
		s.tickSummonsLocked(c, now+float64(i)*1.1)
		for _, m := range mobs {
			if m.hp < before[m.id] {
				hit[m.id] = true
			}
		}
	}
	c.mvMu.Unlock()

	if len(hit) < 2 {
		t.Errorf("totem should hit more than one distinct target across 60 swings with 4 equidistant candidates, got %d: %v", len(hit), hit)
	}
}
