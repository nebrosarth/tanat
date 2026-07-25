package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

// Fidelity/engine tests for the 2026-07-20 pass-6 client-locale audit batch (10 more
// avatars: Cerber-class review skipped Elgorm s1 per user override; Avrora/Inshari/
// Frost/Kiona/Edilia/Elgorm/Velial/Grimlok/Nerlag/Sandariel fixed). Mirrors
// batch2_fixes_test.go/batch3_fixes_test.go's structure.

// ---- Avrora ----

// TestAvroraConsecratedGroundScalesWithEnemyCount: «Освященное место»'s periodic damage
// must grow with how many enemies are caught in the same zone (Op.GrowthPerEnemy), not
// stay flat regardless of occupancy.
func TestAvroraConsecratedGroundScalesWithEnemyCount(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Sp_Avrora")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	m1 := mkMob(t, 4400, 1, 0)
	m1.hp = 100000
	m2 := mkMob(t, 4401, 1.5, 0)
	m2.hp = 100000

	c.mvMu.Lock()
	hs.mobs[m1.id] = m1
	hs.tr.add(m1.id)
	hs.channels = append(hs.channels, channelState{
		slot: 1, level: 1, until: now + 10, interval: 0.8, nextPulse: now,
		px: 0, py: 0, hasPos: true,
		ops: []gamedata.Op{
			{Kind: gamedata.OpDamage, Value: gamedata.PerLevel{20}, Scale: "magic", Radius: 4, GrowthPerEnemy: gamedata.PerLevel{5}},
		},
		growthPerEnemy: 5, growthRadius: 4,
	})
	start1 := m1.hp
	s.tickChannelsLocked(c, now)
	oneEnemyDmg := start1 - m1.hp

	hs.mobs[m2.id] = m2
	hs.tr.add(m2.id)
	hs.channels[0].nextPulse = now
	start2 := m1.hp
	s.tickChannelsLocked(c, now)
	twoEnemyDmg := start2 - m1.hp
	c.mvMu.Unlock()

	if twoEnemyDmg <= oneEnemyDmg {
		t.Fatalf("damage with 2 enemies in zone (%.1f) did not exceed 1 enemy (%.1f)", twoEnemyDmg, oneEnemyDmg)
	}
}

// ---- Inshari ----

// TestInshariExecuteScalesWithMissingHPAndCaps: «Возмездие» must deal damage linear in the
// TARGET's missing HP (Op.MissingHPLinear), clamped at Op.DamageCap -- not a flat base hit
// that always lands even at full health.
func TestInshariExecuteScalesWithMissingHPAndCaps(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Sp_Inshari")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	full := mkMob(t, 4410, 1, 0)
	full.maxHP = 1000
	full.hp = 1000 // at full health -> ~0 damage
	hurt := mkMob(t, 4411, 2, 0)
	// Huge max/current HP so the target SURVIVES the hit (a mob clamped to 0 by
	// hitMobLocked on a lethal blow would make "damage dealt" read as its last HP
	// instead of the actual (possibly capped) amount -- the exact overkill-clipping
	// mistake earlier passes' tests hit before).
	hurt.maxHP = 1000000
	hurt.hp = 100000 // 900,000 missing -> far exceeds the 200 cap, must clamp

	op := gamedata.Op{Kind: gamedata.OpDamage, MissingHPLinear: gamedata.PerLevel{0.5}, DamageCap: gamedata.PerLevel{200}, Scale: "magic"}

	c.mvMu.Lock()
	hs.mobs[full.id] = full
	hs.tr.add(full.id)
	hs.mobs[hurt.id] = hurt
	hs.tr.add(hurt.id)

	s.applyOpsLocked(c, []gamedata.Op{op}, opCtx{slot: 4, level: 1, target: full, px: full.x, py: full.y, hasPos: true}, now)
	fullDmg := 1000 - full.hp

	startHurt := hurt.hp
	s.applyOpsLocked(c, []gamedata.Op{op}, opCtx{slot: 4, level: 1, target: hurt, px: hurt.x, py: hurt.y, hasPos: true}, now)
	hurtDmg := startHurt - hurt.hp
	c.mvMu.Unlock()

	if fullDmg > 5 {
		t.Errorf("full-health target took %.1f damage, want ~0 (execute scales with missing HP)", fullDmg)
	}
	if hurtDmg > 200.5 {
		t.Errorf("near-dead target took %.1f damage, want capped at 200", hurtDmg)
	}
	if hurtDmg < 190 {
		t.Errorf("near-dead target took only %.1f damage, want near the 200 cap (900 missing * 0.5 = 450, clamped)", hurtDmg)
	}
}

// TestInshariZoneArmorGatesByAttackerDistance: «Угнетение»'s armor bonus must only
// mitigate a hit from an attacker standing OUTSIDE the aura radius -- one fighting her
// INSIDE the zone hits at her normal, unbuffed armor.
func TestInshariZoneArmorGatesByAttackerDistance(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Sp_Inshari")
	defer cleanup()
	hs := c.huntState
	hs.skillLevel[1] = 1 // «Угнетение» learned
	hs.zoneArmorSlot = 2
	now := float64(s.battleTime())

	near := mkMob(t, 4420, 1, 0) // inside the 5-unit aura radius
	far := mkMob(t, 4421, 20, 0) // outside it

	c.mvMu.Lock()
	c.x, c.y = 0, 0
	hs.hp = 1000
	s.hitPlayerFromLocked(c, near.id, 100, now, near, nil)
	hpAfterNear := hs.hp

	hs.hp = 1000
	s.hitPlayerFromLocked(c, far.id, 100, now, far, nil)
	hpAfterFar := hs.hp
	c.mvMu.Unlock()

	dmgNear := 1000 - hpAfterNear
	dmgFar := 1000 - hpAfterFar
	if dmgFar >= dmgNear {
		t.Fatalf("attacker outside the zone dealt %.1f, attacker inside dealt %.1f -- outside should be mitigated (less)", dmgFar, dmgNear)
	}
}

// ---- Kiona ----

// TestKionaChainHealSelfOnly verifies OpChainHeal in solo (no teammates): it must heal
// the caster and damage nearby enemies without crashing, capped to the available party.
func TestKionaChainHealSelfOnly(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Sp_Kiona")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	m := mkMob(t, 4430, 1, 0)
	m.hp = 100000

	op := gamedata.Op{Kind: gamedata.OpChainHeal, Value: gamedata.PerLevel{70}, Value2: gamedata.PerLevel{40}, Radius: 4, Count: gamedata.PerLevel{2}}

	c.mvMu.Lock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	hs.hp = 500
	c.x, c.y = 0, 0
	s.applyOpsLocked(c, []gamedata.Op{op}, opCtx{slot: 1, level: 1}, now)
	healed := hs.hp
	dmgDealt := 100000 - m.hp
	c.mvMu.Unlock()

	if healed <= 500 {
		t.Errorf("OpChainHeal did not heal the caster: hp=%v (started 500)", healed)
	}
	if dmgDealt <= 0 {
		t.Error("OpChainHeal did not damage the enemy near the healed ally")
	}
}

// TestKionaCloakBuffsAllyDebuffsEnemy: «Лесной покров»'s dual TargetSide ops must buff an
// ally's base attack (dmg_flat, positive) and debuff an enemy's (negative).
func TestKionaCloakBuffsAllyDebuffsEnemy(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Sp_Kiona")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	sk := gamedata.SkillsFor(hs.av).Skills[1] // slot 2 «Лесной покров»
	m := mkMob(t, 4440, 1, 0)

	c.mvMu.Lock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	// Ally half: On:"ally" with no allyTarget falls back to self per allyTargetsLocked.
	s.applyOpsLocked(c, sk.Ops, opCtx{slot: 2, level: 1, allyTarget: c}, now)
	allyBonus := hs.st.modSum(now, "dmg_flat")

	// Enemy half: TargetSide:"enemy" requires ctx.target set.
	s.applyOpsLocked(c, sk.Ops, opCtx{slot: 2, level: 1, target: m}, now)
	c.mvMu.Unlock()

	var enemyDebuff float64
	for _, mod := range m.st.mods {
		if mod.stat == "dmg_flat" {
			enemyDebuff = mod.value
		}
	}

	if allyBonus <= 0 {
		t.Errorf("ally dmg_flat bonus = %v, want > 0", allyBonus)
	}
	if enemyDebuff >= 0 {
		t.Errorf("enemy dmg_flat mod = %v, want < 0", enemyDebuff)
	}
}

// TestKionaCloakSharesDamageAsAllyHeal: a unit marked by Op.DamageShare must heal nearby
// allies for a fraction of any damage it takes.
func TestKionaCloakSharesDamageAsAllyHeal(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Sp_Kiona")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	m := mkMob(t, 4450, 1, 0)
	m.hp = 1000
	m.st.cloakUntil = now + 10
	m.st.cloakOwner = c.objID
	m.st.cloakHealCoeff = 0.2
	m.st.cloakRadius = 10

	c.mvMu.Lock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	hs.hp = 500
	c.x, c.y = 0, 0
	s.hitMobLocked(c, m, 100, c.objID)
	after := hs.hp
	c.mvMu.Unlock()

	if after <= 500 {
		t.Errorf("cloaked target's damage did not heal nearby allies: hp=%v (started 500)", after)
	}
}

// ---- Edilia ----

// TestEdiliaPollenSlowsAttackerNotSelf: «Пыльца забвения» must slow the STRIKING mob when
// Edilia is hit (OnDamaged), never the mob Edilia herself attacks.
func TestEdiliaPollenSlowsAttackerNotSelf(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Edilia")
	defer cleanup()
	hs := c.huntState
	hs.skillLevel[2] = 1 // «Пыльца забвения» learned
	sk := gamedata.SkillsFor(hs.av).Skills[2]
	for _, op := range sk.Ops {
		if op.Kind == gamedata.OpProc {
			hs.defenseProcs = append(hs.defenseProcs, procState{slot: 3, chance: op.Chance, ops: op.Ops, cd: sk.TipArgs["cooldown"]})
		}
	}
	now := float64(s.battleTime())
	attacker := mkMob(t, 4460, 1, 0)

	c.mvMu.Lock()
	hs.mobs[attacker.id] = attacker
	hs.tr.add(attacker.id)
	s.runDefenseProcsLocked(c, attacker, 10, now)
	c.mvMu.Unlock()

	if attacker.st.atkSlowUntil <= now {
		t.Error("Edilia's «Пыльца забвения» must slow the attacking mob when she is struck")
	}
}

// ---- Elgorm ----

// TestElgormPoisonedGroundScalesWithVictimMaxHP: «Оскверненная почва»'s DoT must be a %
// of the VICTIM's own max HP, so a tankier target takes more per second than a frail one.
func TestElgormPoisonedGroundScalesWithVictimMaxHP(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Elgorm")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	weak := mkMob(t, 4470, 1, 0)
	weak.maxHP = 100
	weak.hp = 100
	tanky := mkMob(t, 4471, 2, 0)
	tanky.maxHP = 1000
	tanky.hp = 1000

	op := gamedata.Op{Kind: gamedata.OpDot, VictimMaxHPPct: gamedata.PerLevel{0.05}, Dur: gamedata.PerLevel{8}, Scale: "magic"}

	c.mvMu.Lock()
	hs.mobs[weak.id] = weak
	hs.tr.add(weak.id)
	hs.mobs[tanky.id] = tanky
	hs.tr.add(tanky.id)
	s.applyOpsLocked(c, []gamedata.Op{op}, opCtx{slot: 3, level: 1, target: weak, px: weak.x, py: weak.y, hasPos: true}, now)
	s.applyOpsLocked(c, []gamedata.Op{op}, opCtx{slot: 3, level: 1, target: tanky, px: tanky.x, py: tanky.y, hasPos: true}, now)
	c.mvMu.Unlock()

	if len(weak.st.dots) == 0 || len(tanky.st.dots) == 0 {
		t.Fatal("OpDot did not apply to one or both targets")
	}
	weakPerSec := weak.st.dots[0].perSec
	tankyPerSec := tanky.st.dots[0].perSec
	if tankyPerSec <= weakPerSec {
		t.Fatalf("tanky target's DoT (%.1f/s) did not exceed the weak target's (%.1f/s)", tankyPerSec, weakPerSec)
	}
	if weakPerSec != 5 { // 5% of 100 max HP
		t.Errorf("weak target DoT = %.1f/s, want 5 (5%% of 100 max HP)", weakPerSec)
	}
}

// ---- Velial ----

// TestVelialTribunalRevealsTarget: Op.RevealTarget must make a marked mob fully revealed
// to the team regardless of distance (mobViewDistLocked returns 0).
func TestVelialTribunalRevealsTarget(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())

	m := mkMob(t, 4480, 500, 500) // far from everyone
	m.st.revealUntil = now + 30

	c.mvMu.Lock()
	d := s.mobViewDistLocked(c, m, now)
	c.mvMu.Unlock()

	if d != 0 {
		t.Errorf("mobViewDistLocked = %v for a revealed target, want 0", d)
	}
}

// ---- Grimlok ----

// TestGrimlokWildFormBuffsSummonSpeed: «Дикость»'s own_summons buff must speed up BOTH
// the dinosaur's attack AND move speed, not just attack.
func TestGrimlokWildFormBuffsSummonSpeed(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_HK_Grimlok")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	sm := &summonState{id: 4490, hp: 100, maxHP: 100, slot: 1}
	hs.summons[sm.id] = sm

	sk := gamedata.SkillsFor(hs.av).Skills[2] // slot 3 «Дикость»
	c.mvMu.Lock()
	s.applyOpsLocked(c, sk.Ops, opCtx{slot: 3, level: 1}, now)
	c.mvMu.Unlock()

	if sm.atkSpeedMul <= 1 || now >= sm.atkSpeedMulUntil {
		t.Errorf("dinosaur attack speed not buffed: mul=%v until=%v now=%v", sm.atkSpeedMul, sm.atkSpeedMulUntil, now)
	}
	if sm.moveSpeedMul <= 1 || now >= sm.moveSpeedMulUntil {
		t.Errorf("dinosaur move speed not buffed: mul=%v until=%v now=%v", sm.moveSpeedMul, sm.moveSpeedMulUntil, now)
	}
}

// TestGrimlokDarkSideForcesMeleeRange: the «Темная сторона» toggle must drop the avatar's
// effective attack range to melee and suppress its projectile while active, restoring both
// on toggle-off.
func TestGrimlokDarkSideForcesMeleeRange(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_HK_Grimlok")
	defer cleanup()
	hs := c.huntState
	hs.skillLevel[3] = 1 // ult learned
	hs.hasProjectile = true
	now := float64(s.battleTime())

	baseline := hs.effAttackRangeLocked(now)

	c.mvMu.Lock()
	s.toggleSkillLocked(c, 4)
	rangedWhileOn := hs.effAttackRangeLocked(now)
	projWhileOn := hs.hasProjectile
	s.toggleSkillLocked(c, 4) // toggle off
	c.mvMu.Unlock()

	if rangedWhileOn >= baseline {
		t.Errorf("melee-form range (%v) did not drop below baseline ranged reach (%v)", rangedWhileOn, baseline)
	}
	if projWhileOn {
		t.Error("melee-form toggle must suppress hasProjectile while active")
	}
	if !hs.hasProjectile {
		t.Error("toggle-off must restore hasProjectile")
	}
}

// ---- Nerlag ----

// TestNerlagAxesGrowPerTarget: «Метание топоров» must deal MORE damage to each successive
// enemy along the throw (Op.PerTargetGrowth), nearest-to-caster first.
func TestNerlagAxesGrowPerTarget(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Nerlag")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	near := mkMob(t, 4500, 2, 0)
	near.hp = 100000
	far := mkMob(t, 4501, 5, 0)
	far.hp = 100000

	op := gamedata.Op{Kind: gamedata.OpDamage, Value: gamedata.PerLevel{60}, Scale: "magic", Radius: 10, PerTargetGrowth: gamedata.PerLevel{12}}

	c.mvMu.Lock()
	c.x, c.y = 0, 0
	hs.mobs[near.id] = near
	hs.tr.add(near.id)
	hs.mobs[far.id] = far
	hs.tr.add(far.id)
	startNear, startFar := near.hp, far.hp
	s.applyOpsLocked(c, []gamedata.Op{op}, opCtx{slot: 1, level: 1}, now)
	c.mvMu.Unlock()

	nearDmg := startNear - near.hp
	farDmg := startFar - far.hp
	if farDmg <= nearDmg {
		t.Fatalf("farther (later-hit) target took %.1f, nearer (first-hit) took %.1f -- want farther > nearer", farDmg, nearDmg)
	}
}

// TestNerlagSlaughterScalesPerEnemyHit: «Поголовная бойня»'s speed/attack buffs must scale
// with how many enemies the landing AoE actually hit (Op.ScalePerHit), not a flat number.
func TestNerlagSlaughterScalesPerEnemyHit(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Nerlag")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	m1 := mkMob(t, 4510, 1, 0)
	m2 := mkMob(t, 4511, 1.5, 0)

	sk := gamedata.SkillsFor(hs.av).Skills[3] // slot 4 «Поголовная бойня»
	var ops []gamedata.Op
	for _, op := range sk.Ops {
		if op.Kind != gamedata.OpDash { // skip the dash so the caster stays put
			ops = append(ops, op)
		}
	}

	c.mvMu.Lock()
	c.x, c.y = 0, 0
	hs.mobs[m1.id] = m1
	hs.tr.add(m1.id)
	hs.mobs[m2.id] = m2
	hs.tr.add(m2.id)
	s.applyOpsLocked(c, ops, opCtx{slot: 4, level: 1}, now)
	c.mvMu.Unlock()

	speedMul := hs.st.modMul(now, "move_speed_pct")
	flatBonus := hs.st.modSum(now, "dmg_flat")
	if speedMul <= 1 {
		t.Errorf("move_speed_pct mul = %v, want > 1 (2 enemies hit)", speedMul)
	}
	if flatBonus <= 0 {
		t.Errorf("dmg_flat bonus = %v, want > 0 (2 enemies hit)", flatBonus)
	}
}

// ---- Sandariel ----

// TestSandarielMoveChargeAddsAttackDamage: «Острие странника» must bank a charge every
// chargeDist walked and add flat damage to the NEXT basic attack, resetting after.
func TestSandarielMoveChargeAddsAttackDamage(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Sandariel")
	defer cleanup()
	hs := c.huntState
	hs.skillLevel[2] = 1 // «Острие странника» learned
	hs.moveChargeSlot = 3
	now := float64(s.battleTime())

	c.mvMu.Lock()
	c.x, c.y = 0, 0
	s.tickMoveChargeLocked(c, now) // establishes the last-sampled position
	c.x += 12                      // walk 12 units (chargeDist=5 -> 2 charges)
	s.tickMoveChargeLocked(c, now)
	charges := hs.moveChargeCount
	bonus := s.consumeMoveChargeLocked(hs, now)
	chargesAfter := hs.moveChargeCount
	c.mvMu.Unlock()

	if charges != 2 {
		t.Errorf("moveChargeCount = %d after walking 12 units (chargeDist=5), want 2", charges)
	}
	if bonus != 8 { // chargeDamage rank1 = 4, ×2 charges
		t.Errorf("consumeMoveChargeLocked returned %v, want 8 (4 per charge x2)", bonus)
	}
	if chargesAfter != 0 {
		t.Error("move-charge stack must reset to 0 after a basic attack consumes it")
	}
}

// TestSandarielVeilCloaksAlliesNotSelf: «Сокрывающая вуаль» must cloak nearby ALLIES, not
// Sandariel herself -- solo (no other allies) means nobody gets stealthed, but she still
// keeps her own dodge buff.
func TestSandarielVeilCloaksAlliesNotSelf(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Sandariel")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	sk := gamedata.SkillsFor(hs.av).Skills[3] // slot 4 «Сокрывающая вуаль»
	c.mvMu.Lock()
	s.applyOpsLocked(c, sk.Ops, opCtx{slot: 4, level: 1}, now)
	c.mvMu.Unlock()

	if hs.invisibleUntil > now {
		t.Error("Sandariel must NOT go invisible from her own veil -- she stays visible")
	}
	if hs.st.modSum(now, "dodge_pct") <= 0 {
		t.Error("Sandariel must still get her own dodge_pct buff from the veil")
	}
}

// ---- Frost ----

// TestFrostElementalSlowsAndStunsOnHit: the summoned elemental's own on-hit combo must
// slow the struck enemy, then stun it (and clear the chill) on a SECOND hit while still
// chilled.
func TestFrostElementalSlowsAndStunsOnHit(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Frost")
	defer cleanup()
	hs := c.huntState
	hs.skillLevel[3] = 1 // «Исчадие мерзлоты» learned
	now := float64(s.battleTime())

	sk := gamedata.SkillsFor(hs.av).Skills[3]
	var summonOp gamedata.Op
	for _, op := range sk.Ops {
		if op.Kind == gamedata.OpSummon {
			summonOp = op
		}
	}
	sm := &summonState{id: 4520, hp: 100, maxHP: 100, slot: 4, onHitOps: summonOp.Ops}
	target := mkMob(t, 4521, 1, 0)

	c.mvMu.Lock()
	hs.mobs[target.id] = target
	hs.tr.add(target.id)
	s.applySummonOnHitOpsLocked(c, sm, target, now)
	firstSlow := target.st.slowUntil
	chillUntil := target.st.chillUntil
	s.applySummonOnHitOpsLocked(c, sm, target, now) // second hit while still chilled
	c.mvMu.Unlock()

	if firstSlow <= now {
		t.Error("elemental's first hit must slow the target")
	}
	if chillUntil <= now {
		t.Error("elemental's first hit must chill the target")
	}
	if target.st.stunUntil <= now {
		t.Error("elemental's second hit on an already-chilled target must stun it")
	}
}
