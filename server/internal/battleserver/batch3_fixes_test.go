package battleserver

import (
	"testing"
	"time"

	"tanatserver/internal/gamedata"
)

// Fidelity/engine tests for the 2026-07-20 pass-5 client-locale audit batch (10 more
// avatars: Cerber clean, Titanid/Gayal/Astarot/Anhel/BlackDragon/Wilfang/PlusMinus/
// Sharli/Gellar fixed). Mirrors batch2_fixes_test.go's structure.

// ---- Titanid ----

// TestTitanidQuakeWavesEscalate: «Землетрясение»'s 3-wave channel must ramp damage AND
// stun duration per pulse (Op.Growth on both nested ops) and widen its AoE
// (Op.RadiusGrowth), not repeat a flat averaged hit.
func TestTitanidQuakeWavesEscalate(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Titanid")
	defer cleanup()
	hs := c.huntState

	m := mkMob(t, 3300, 1, 0)
	m.hp = 100000 // high HP so neither wave overkills it (would clip the comparison)
	now := float64(s.battleTime())
	c.mvMu.Lock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	hs.channels = append(hs.channels, channelState{
		slot: 1, level: 1, until: now + 10, interval: 0.8, nextPulse: now,
		ops: []gamedata.Op{
			{Kind: gamedata.OpDamage, Value: gamedata.PerLevel{40}, Scale: "magic", Radius: 5, RadiusGrowth: 1},
			{Kind: gamedata.OpStun, Dur: gamedata.PerLevel{1.0}},
		},
		growth: 10, stunGrowth: 0.2, radiusGrowth: 1,
	})
	startHP := m.hp
	s.tickChannelsLocked(c, now)
	hp1 := m.hp
	stunUntil1 := m.st.stunUntil
	// Force the second pulse to fire immediately.
	hs.channels[0].nextPulse = now
	s.tickChannelsLocked(c, now)
	hp2 := m.hp
	stunUntil2 := m.st.stunUntil
	c.mvMu.Unlock()

	firstHit := startHP - hp1
	secondHit := hp1 - hp2
	if secondHit <= firstHit {
		t.Fatalf("wave 2 (%.1f dmg) did not exceed wave 1 (%.1f dmg)", secondHit, firstHit)
	}
	if stunUntil2 <= stunUntil1 {
		t.Fatalf("wave 2 stun (until=%g) did not exceed wave 1 stun (until=%g)", stunUntil2, stunUntil1)
	}
}

// TestTitanidShockwaveIsSlowNotRootSilence: «Ударная волна»'s splash must slow nearby
// enemies (client: "замедляются на 25%"), not root+silence (the prior wiki-based fix).
func TestTitanidShockwaveIsSlowNotRootSilence(t *testing.T) {
	ti, _ := gamedata.AvatarByID(14)
	sk := gamedata.SkillsFor(ti).Skills[1]
	hasSlow, hasRoot := false, false
	for _, op := range sk.Ops {
		if op.Kind == gamedata.OpSlow && op.Radius > 0 {
			hasSlow = true
		}
		if op.Kind == gamedata.OpRoot || op.Kind == gamedata.OpSilence {
			hasRoot = true
		}
	}
	if !hasSlow {
		t.Error("Titanid «Ударная волна» must AoE-slow nearby enemies")
	}
	if hasRoot {
		t.Error("Titanid «Ударная волна» must NOT root/silence -- client only mentions a slow")
	}
}

// TestTitanidStoneSkinArmorCapped: repeated «Каменная кожа» procs must stop stacking
// once StackCap is reached, per rank ({12,16,20,24}).
func TestTitanidStoneSkinArmorCapped(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Titanid")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())
	op := gamedata.Op{Kind: gamedata.OpBuffStat, Value: gamedata.PerLevel{3}, Dur: gamedata.PerLevel{5}, Stat: "phys_armor", On: "self", StackCap: gamedata.PerLevel{12}}

	c.mvMu.Lock()
	for i := 0; i < 10; i++ { // far more procs than the cap allows
		s.applyOpsLocked(c, []gamedata.Op{op}, opCtx{slot: 3, level: 1}, now)
	}
	total := hs.st.modSum(now, "phys_armor")
	c.mvMu.Unlock()

	if total != 12 {
		t.Fatalf("stacked phys_armor = %g, want capped at 12", total)
	}
}

// ---- Gayal ----

// TestGayalHitStackBurstsAndResets: «Меч жажды» must bank a stack per landing basic
// attack, and burst bonus damage + reset stacks at the cap (5).
func TestGayalHitStackBurstsAndResets(t *testing.T) {
	s, c, hs, mob := setupArcherForHit(t)
	gayal, ok := gamedata.AvatarByID(24)
	if !ok {
		t.Fatal("Gayal (id 24) missing")
	}
	c.mvMu.Lock()
	hs.av = gayal
	hs.kit = gamedata.SkillsFor(gayal)
	hs.hasProjectile = false
	mob.hp = 100000
	mob.maxHP = 100000
	now := float64(s.battleTime())
	s.applyOpsLocked(c, []gamedata.Op{
		{Kind: gamedata.OpHitStack, Value: gamedata.PerLevel{0.05}, Value2: gamedata.PerLevel{0.03},
			Count: gamedata.PerLevel{5}, Dur: gamedata.PerLevel{30}, StackBurstDamage: gamedata.PerLevel{200}},
	}, opCtx{slot: 1, level: 1}, now)
	if hs.hitStackCap != 5 || hs.hitStackUntil <= now {
		t.Fatalf("OpHitStack did not arm correctly: cap=%d until=%g now=%g", hs.hitStackCap, hs.hitStackUntil, now)
	}
	c.mvMu.Unlock()

	// Land 5 basic attacks (synchronously, via the same helper the swing scheduler uses).
	var lastBurst float64
	for i := 0; i < 5; i++ {
		c.mvMu.Lock()
		hpBefore := mob.hp
		lastBurst = s.applyHitStackLocked(c, float64(s.battleTime()))
		mob.hp = hpBefore // applyHitStackLocked doesn't itself deal damage; it only returns the bonus
		c.mvMu.Unlock()
	}
	if lastBurst <= 0 {
		t.Fatal("5th stack did not burst bonus damage")
	}
	c.mvMu.Lock()
	count := hs.hitStackCount
	c.mvMu.Unlock()
	if count != 0 {
		t.Fatalf("stacks did not reset after the burst: count=%d", count)
	}

	// The scheduler itself must also route through applyHitStackLocked on a real swing.
	c.mvMu.Lock()
	hs.attackSeq = 1
	c.mvMu.Unlock()
	s.scheduleProjectileHitLocked(c, mob.id, time.Millisecond)
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		c.mvMu.Lock()
		stacked := hs.hitStackCount > 0
		c.mvMu.Unlock()
		if stacked {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("a real basic-attack swing never banked a hit-stack stack")
}

// TestGayalAuraGivesAllAlliesLifesteal: «Аура погибших»'s lifesteal must reach nearby
// ALLIES, not just Gayal himself, and the zombie-raise chance must be 15% (not 5%).
func TestGayalAuraGivesAllAlliesLifesteal(t *testing.T) {
	gayal, _ := gamedata.AvatarByID(24)
	sk := gamedata.SkillsFor(gayal).Skills[1]
	var chance float64
	var auraOnAllies bool
	for _, op := range sk.Ops {
		if op.Kind == gamedata.OpProc {
			chance = op.Chance.At(1)
		}
		if op.Kind == gamedata.OpAura {
			for _, nested := range op.Ops {
				if nested.Stat == "lifesteal_pct" && (nested.On == "allies" || nested.On == "ally") {
					auraOnAllies = true
				}
			}
		}
	}
	if chance != 0.15 {
		t.Errorf("Gayal «Аура погибших» zombie chance = %g, want 0.15", chance)
	}
	if !auraOnAllies {
		t.Error("Gayal «Аура погибших» lifesteal must reach nearby allies (OpAura On:allies), not just self")
	}
}

// TestGayalLifeDrainScalesWithOwnMaxHP: «Поглощение жизни» must deal/heal a % of Gayal's
// OWN live max HP each tick, not a flat per-rank number.
func TestGayalLifeDrainScalesWithOwnMaxHP(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Gayal")
	defer cleanup()
	hs := c.huntState
	m := mkMob(t, 3400, 1, 0)
	now := float64(s.battleTime())

	c.mvMu.Lock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	hs.av.Health = 1000 // live max HP = 1000
	hs.hp = 500         // damaged, so the heal is actually visible
	startHP := m.hp
	s.applyOpsLocked(c, []gamedata.Op{
		{Kind: gamedata.OpDamage, SelfMaxHPPct: gamedata.PerLevel{0.05}, Scale: "magic", Radius: 4},
		{Kind: gamedata.OpHeal, SelfMaxHPPct: gamedata.PerLevel{0.05}},
	}, opCtx{slot: 4, level: 1, target: m}, now)
	dealt := startHP - m.hp
	healed := hs.hp
	c.mvMu.Unlock()

	if dealt != 50 { // 5% of 1000
		t.Fatalf("SelfMaxHPPct damage = %g, want 50 (5%% of 1000 max HP)", dealt)
	}
	if healed != 550 { // 500 + 5% of 1000
		t.Fatalf("SelfMaxHPPct heal = %g, want 550 (500 + 50)", healed)
	}
}

// ---- Astarot ----

// TestAstarotBackstabIsProcNotBuff: «Удар в спину» must be a 20%-chance on-attack proc
// dealing bonus damage, not the prior wiki-based always-on crit buff.
func TestAstarotBackstabIsProcNotBuff(t *testing.T) {
	as, _ := gamedata.AvatarByID(7)
	sk := gamedata.SkillsFor(as).Skills[2]
	if sk.NameRu != "Удар в спину" {
		t.Errorf("Astarot slot 3 NameRu = %q, want «Удар в спину»", sk.NameRu)
	}
	var proc *gamedata.Op
	for i := range sk.Ops {
		if sk.Ops[i].Kind == gamedata.OpProc {
			proc = &sk.Ops[i]
		}
		if sk.Ops[i].Stat == "crit_pct" || sk.Ops[i].Stat == "crit_dmg_pct" {
			t.Error("Astarot «Удар в спину» must not be a crit buff")
		}
	}
	if proc == nil || proc.Chance.At(1) != 0.2 {
		t.Fatal("Astarot «Удар в спину» must be a 20% on-attack proc")
	}
}

// ---- Anhel ----

// TestAnhelSpeedBuffReachesOwnSummons: «Гнев океана» must also speed up Anhel's own
// live clones, not just herself.
func TestAnhelSpeedBuffReachesOwnSummons(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Psh_Anhel")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())
	sm := &summonState{id: 300001, hp: 100, maxHP: 100, until: now + 30}

	c.mvMu.Lock()
	hs.summons[sm.id] = sm
	s.applyOpsLocked(c, []gamedata.Op{
		{Kind: gamedata.OpBuffStat, Value: gamedata.PerLevel{1.25}, Dur: gamedata.PerLevel{10}, Stat: "attack_speed_pct", On: "own_summons"},
	}, opCtx{slot: 2, level: 1}, now)
	mul := summonAttackSpeedMulLocked(sm, now)
	c.mvMu.Unlock()

	if mul != 1.25 {
		t.Fatalf("summon attack-speed multiplier = %g, want 1.25", mul)
	}
}

// TestAnhelClonesScaleWithOwnerStats: «Зов фантомов»/«Стражи глубин» clones must carry
// 1/3 of Anhel's OWN live HP and attack, not a flat per-rank table.
func TestAnhelClonesScaleWithOwnerStats(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Psh_Anhel")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	c.mvMu.Lock()
	hs.hp, hs.av.Health = 900, 900 // live max HP = 900 -> clone should get 300
	hs.summonProtos = map[string]int32{"Avtr_Psh_Anhel": 1}
	s.summonLocked(c, gamedata.Op{
		Unit: "Avtr_Psh_Anhel", Count: gamedata.PerLevel{1}, Lifetime: gamedata.PerLevel{15},
		HpPctOfOwner: 1.0 / 3, DmgPctOfAttack: 1.0 / 3,
	}, opCtx{slot: 3, level: 1}, now)
	var clone *summonState
	for _, sm := range hs.summons {
		clone = sm
	}
	c.mvMu.Unlock()

	if clone == nil {
		t.Fatal("no clone was spawned")
	}
	if clone.maxHP != 300 {
		t.Fatalf("clone maxHP = %g, want 300 (1/3 of 900)", clone.maxHP)
	}
}

// ---- BlackDragon ----

// TestBlackDragonCleaveHitsNearbyEnemies: while «Неистовство» is active, a basic attack
// must also strike OTHER enemies near the primary target, not just the one target.
func TestBlackDragonCleaveHitsNearbyEnemies(t *testing.T) {
	s, c, hs, primary := setupArcherForHit(t)
	dragon, ok := gamedata.AvatarByID(23)
	if !ok {
		t.Fatal("BlackDragon (id 23) missing")
	}
	secondary := mkMob(t, 2001, 3.5, 0) // near the primary target
	secondary.hp = 5000

	c.mvMu.Lock()
	hs.av = dragon
	hs.kit = gamedata.SkillsFor(dragon)
	hs.hasProjectile = false
	hs.mobs[secondary.id] = secondary
	hs.tr.add(secondary.id)
	now := float64(s.battleTime())
	s.applyOpsLocked(c, []gamedata.Op{
		{Kind: gamedata.OpAttackCleave, Dur: gamedata.PerLevel{20}, Radius: 4},
	}, opCtx{slot: 1, level: 1}, now)
	startHP := secondary.hp
	hs.attackSeq = 1
	c.mvMu.Unlock()

	_ = primary
	s.scheduleProjectileHitLocked(c, primary.id, time.Millisecond)
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		c.mvMu.Lock()
		hit := secondary.hp < startHP
		c.mvMu.Unlock()
		if hit {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("cleave never hit the nearby secondary enemy")
}

// ---- Wilfang ----

// TestWilfangAmbushDamageOnBreakNotOnCast: «Засада»'s AoE damage must land when
// invisibility BREAKS (move/attack/cast), not at the moment of casting.
func TestWilfangAmbushDamageOnBreakNotOnCast(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Wilfang")
	defer cleanup()
	hs := c.huntState
	m := mkMob(t, 3500, 1, 0)
	now := float64(s.battleTime())

	c.mvMu.Lock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	startHP := m.hp
	s.applyOpsLocked(c, []gamedata.Op{
		{Kind: gamedata.OpStealth, Dur: gamedata.PerLevel{6}, BreakOnMove: true, Ops: []gamedata.Op{
			{Kind: gamedata.OpDamage, Value: gamedata.PerLevel{75}, Scale: "magic", Radius: 3},
		}},
	}, opCtx{slot: 2, level: 1}, now)
	stillFull := m.hp == startHP // no damage yet -- the burst is armed, not fired
	s.breakInvisibilityLocked(c, now+0.5)
	brokeHP := m.hp
	c.mvMu.Unlock()

	if !stillFull {
		t.Fatal("Wilfang «Засада» damaged the target at CAST time -- it must wait for the stealth break")
	}
	if brokeHP >= startHP {
		t.Fatal("breaking stealth did not detonate the ambush damage")
	}
}

// TestWilfangPoisonExplodesOnDeath: «Ядовитый укус» -- a target that dies while still
// poisoned must explode, splashing nearby enemies.
func TestWilfangPoisonExplodesOnDeath(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Wilfang")
	defer cleanup()
	hs := c.huntState
	victim := mkMob(t, 3600, 1, 0)
	bystander := mkMob(t, 3601, 2, 0)
	now := float64(s.battleTime())

	c.mvMu.Lock()
	hs.mobs[victim.id] = victim
	hs.mobs[bystander.id] = bystander
	hs.tr.add(victim.id)
	hs.tr.add(bystander.id)
	s.applyOpsLocked(c, []gamedata.Op{
		{Kind: gamedata.OpDot, Value: gamedata.PerLevel{10}, Dur: gamedata.PerLevel{7}, Scale: "magic",
			ExplodeOnDeath: true, ExplodeDamage: gamedata.PerLevel{200}, ExplodeRadius: 4},
	}, opCtx{slot: 3, level: 1, target: victim}, now)
	bystanderStartHP := bystander.hp
	// Kill the poisoned victim outright (any source, not just Wilfang's own hit).
	s.hitMobLocked(c, victim, victim.hp+9999, c.objID)
	splashed := bystander.hp < bystanderStartHP
	c.mvMu.Unlock()

	if !splashed {
		t.Fatal("poisoned target's death did not splash the nearby bystander")
	}
}

// ---- PlusMinus ----

// TestPlusMinusChainDecaysPerTarget: «Сверхпроводимость» must hit at most 4 targets,
// each successive (nearest-first) hit 20% weaker than the last.
func TestPlusMinusChainDecaysPerTarget(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_PlusMinus")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())
	mobs := []*mobState{
		mkMob(t, 3700, 1, 0),
		mkMob(t, 3701, 2, 0),
		mkMob(t, 3702, 3, 0),
		mkMob(t, 3703, 4, 0),
		mkMob(t, 3704, 5, 0), // 5th enemy -- must be EXCLUDED by MaxTargets
	}
	starts := make([]float64, len(mobs))
	c.mvMu.Lock()
	for i, m := range mobs {
		m.hp = 100000 // high HP so no hop overkills its target (would clip the comparison)
		hs.mobs[m.id] = m
		hs.tr.add(m.id)
		starts[i] = m.hp
	}
	s.applyOpsLocked(c, []gamedata.Op{
		{Kind: gamedata.OpDamage, Value: gamedata.PerLevel{100}, Scale: "magic", Radius: 6, MaxTargets: 4, PerTargetDecay: 0.2},
	}, opCtx{slot: 3, level: 1}, now)
	c.mvMu.Unlock()

	if mobs[4].hp != starts[4] {
		t.Fatal("chain hit a 5th target -- MaxTargets should cap it at 4")
	}
	var last float64 = -1
	for i := 0; i < 4; i++ {
		dealt := starts[i] - mobs[i].hp
		if dealt <= 0 {
			t.Fatalf("target %d was not hit at all", i)
		}
		if last >= 0 && dealt >= last {
			t.Fatalf("hop %d (dealt %g) did not decay below hop %d (dealt %g)", i, dealt, i-1, last)
		}
		last = dealt
	}
}

// TestPlusMinusManaBurnArea: «Шаровая молния» must burn mana from every enemy in the
// blast, not just deal damage+slow.
func TestPlusMinusManaBurnArea(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_PlusMinus")
	defer cleanup()
	hs := c.huntState
	m := mkMob(t, 3800, 1, 0)
	m.maxMana, m.mana = 200, 200
	now := float64(s.battleTime())

	c.mvMu.Lock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	s.applyOpsLocked(c, []gamedata.Op{
		{Kind: gamedata.OpManaBurnArea, Value: gamedata.PerLevel{150}, Radius: 4},
	}, opCtx{slot: 4, level: 1}, now)
	c.mvMu.Unlock()

	if m.mana != 50 {
		t.Fatalf("mob mana = %g, want 50 (200 - 150 burned)", m.mana)
	}
}

// ---- Sharli ----

// TestSharliManaScaledAttackSpeed: «Жар души» must speed up attacks continuously as
// current mana drains, not apply a fixed flat buff.
func TestSharliManaScaledAttackSpeed(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Sharli")
	defer cleanup()
	hs := c.huntState
	hs.skillLevel[1] = 1 // «Жар души» learned at rank 1
	hs.manaSpeedSlot = 2
	hs.av.Mana = 200

	c.mvMu.Lock()
	hs.mana = 200 // full mana -- no bonus
	full := s.attackPeriodLocked(hs)
	hs.mana = 0 // empty mana -- max bonus (+2% * 10 = +20%)
	empty := s.attackPeriodLocked(hs)
	c.mvMu.Unlock()

	if empty <= full {
		t.Fatalf("attack period at 0 mana (%g) did not exceed full mana (%g)", empty, full)
	}
	wantEmpty := full * 1.2
	if diff := empty - wantEmpty; diff > 0.01 || diff < -0.01 {
		t.Fatalf("attack period at 0 mana = %g, want ~%g (full ×1.2)", empty, wantEmpty)
	}
}

// ---- Gellar ----

// TestGellarSoulDamageMatchesTooltipAndScalesWithSP: the tooltip must match the actual
// per-tick damage, and the per-soul bonus must scale with spell power too.
func TestGellarSoulDamageMatchesTooltipAndScalesWithSP(t *testing.T) {
	gel, _ := gamedata.AvatarByID(29)
	sk := gamedata.SkillsFor(gel).Skills[3]
	var dmgOp *gamedata.Op
	for i := range sk.Ops {
		if sk.Ops[i].Kind == gamedata.OpChannel {
			for j := range sk.Ops[i].Ops {
				if sk.Ops[i].Ops[j].Kind == gamedata.OpDamage {
					dmgOp = &sk.Ops[i].Ops[j]
				}
			}
		}
	}
	if dmgOp == nil {
		t.Fatal("Gellar «Армия душ» has no channel damage op")
	}
	if dmgOp.Value.At(1) != sk.TipArgs["damage"].At(1) {
		t.Errorf("channel damage (%g) does not match the tooltip (%g)", dmgOp.Value.At(1), sk.TipArgs["damage"].At(1))
	}
	if dmgOp.PerSoulSP <= 0 {
		t.Error("Gellar «Армия душ» per-soul bonus must scale with spell power (PerSoulSP)")
	}
}
