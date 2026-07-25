package battleserver

import (
	"math"
	"testing"
	"time"

	"tanatserver/internal/gamedata"
)

// TestDotDecayFadesToZero: Op.DecayTo on an OpDot (Rognar's «Могильный холод») makes
// the per-tick damage fade linearly from Value at cast to DecayTo by expiry.
func TestDotDecayFadesToZero(t *testing.T) {
	o := overTime{perSec: 10, perSecEnd: 0, startAt: 0, until: 10}
	if v := o.currentPerSec(0); v != 10 {
		t.Errorf("at start currentPerSec = %g, want 10", v)
	}
	if v := o.currentPerSec(5); math.Abs(v-5) > 0.01 {
		t.Errorf("at midpoint currentPerSec = %g, want ~5", v)
	}
	if v := o.currentPerSec(10); v != 0 {
		t.Errorf("at expiry currentPerSec = %g, want 0", v)
	}
	// A flat (non-decaying) stream -- perSecEnd == perSec -- must stay constant.
	flat := overTime{perSec: 10, perSecEnd: 10, startAt: 0, until: 10}
	if v := flat.currentPerSec(9); v != 10 {
		t.Errorf("flat overTime decayed: currentPerSec(9) = %g, want 10", v)
	}
}

// TestSlowDecayFadesToNoSlow: Op.DecayTo on an OpSlow fades the move-speed penalty
// back to 1.0 (no slow) by expiry instead of holding a flat factor.
func TestSlowDecayFadesToNoSlow(t *testing.T) {
	st := &unitStatus{slowUntil: 10, slowFactor: 0.5, slowFactorEnd: 1.0, slowStart: 0}
	if f := st.moveFactor(0); math.Abs(f-0.5) > 0.01 {
		t.Errorf("moveFactor at start = %g, want ~0.5", f)
	}
	if f := st.moveFactor(9); f <= 0.8 {
		t.Errorf("moveFactor near expiry = %g, want close to 1.0 (barely slowed)", f)
	}
}

// TestRognarDeathChannelIsDualCast: «Канал смерти» must accept an ally target
// (FRIEND in the mask) with an enemy/ally split via TargetSide, and no longer carries
// the undescribed extra DoT a prior pass added.
func TestRognarDeathChannelIsDualCast(t *testing.T) {
	rognar, _ := gamedata.AvatarByID(1)
	s4 := gamedata.SkillsFor(rognar).Skills[3]
	if s4.NameRu != "Канал смерти" {
		t.Fatalf("Rognar slot 4 is %q, not «Канал смерти»", s4.NameRu)
	}
	if !skillHasTargetFlag(s4, "FRIEND") {
		t.Fatal("«Канал смерти» must be castable on a friendly avatar (client: «дружественной или враждебной целью»)")
	}
	var sawEnemyDamage, sawAllyHeal, sawDot bool
	for _, op := range s4.Ops {
		switch {
		case op.Kind == gamedata.OpDamage && op.TargetSide == "enemy":
			sawEnemyDamage = true
		case op.Kind == gamedata.OpHeal && op.TargetSide == "ally":
			sawAllyHeal = true
		case op.Kind == gamedata.OpDot:
			sawDot = true
		}
	}
	if !sawEnemyDamage {
		t.Error("missing an enemy-side OpDamage")
	}
	if !sawAllyHeal {
		t.Error("missing an ally-side OpHeal")
	}
	if sawDot {
		t.Error("«Канал смерти» should not carry an extra DoT -- the client only describes one-time damage/heal + the deathlink redirect")
	}
}

// TestSigilionRecoilCostsHP: while «Мощь берсерка» is armed, a basic-attack swing
// costs HP proportional to the LIVE dmg_pct bonus ("ранит себя на 50% от увеличенного
// урона").
func TestSigilionRecoilCostsHP(t *testing.T) {
	s, c, hs, mob := setupArcherForHit(t)
	sigilion, ok := gamedata.AvatarByID(10)
	if !ok {
		t.Fatal("Sigilion (id 10) missing")
	}
	c.mvMu.Lock()
	hs.av = sigilion
	hs.kit = gamedata.SkillsFor(sigilion)
	hs.hasProjectile = false
	hs.hp = 1000
	startHP := hs.hp
	// Arm the +80% damage buff and the matching recoil window directly (mirrors what
	// casting «Мощь берсерка» at rank 4 does).
	now := float64(s.battleTime())
	ctx := opCtx{slot: 4, level: 4}
	s.applyOpsLocked(c, []gamedata.Op{
		{Kind: gamedata.OpBuffStat, Value: gamedata.PerLevel{1.8}, Dur: gamedata.PerLevel{20}, Stat: "dmg_pct", On: "self"},
		{Kind: gamedata.OpSelfRecoil, Value: gamedata.PerLevel{0.5}, Dur: gamedata.PerLevel{20}},
	}, ctx, now)
	if hs.recoilUntil <= now {
		t.Fatal("OpSelfRecoil did not arm hs.recoilUntil")
	}
	hs.attackSeq = 1
	s.scheduleHitLocked(c, 1, mob.id, time.Millisecond)
	c.mvMu.Unlock()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		c.mvMu.Lock()
		dropped := hs.hp < startHP
		c.mvMu.Unlock()
		if dropped {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	if hs.hp >= startHP {
		t.Fatalf("recoil never cost any HP: hp stayed at %g", hs.hp)
	}
}

// TestMiriamManaShotConsumesManaForBonusDamage: while «Зачарованные стрелы» is armed,
// a swing that can afford the mana cost consumes it (extra flat damage rides the same
// hit, unobservable in isolation from the random roll, so mana spend is the signal).
func TestMiriamManaShotConsumesManaForBonusDamage(t *testing.T) {
	s, c, hs, mob := setupArcherForHit(t)
	c.mvMu.Lock()
	hs.mana = 100
	now := float64(s.battleTime())
	s.applyOpsLocked(c, []gamedata.Op{
		{Kind: gamedata.OpAttackManaBonus, Value: gamedata.PerLevel{75}, Value2: gamedata.PerLevel{12}, Dur: gamedata.PerLevel{13}},
	}, opCtx{slot: 3, level: 4}, now)
	if hs.manaShotUntil <= now || hs.manaShotCost != 12 || hs.manaShotDmg != 75 {
		t.Fatalf("OpAttackManaBonus did not arm correctly: until=%g cost=%g dmg=%g", hs.manaShotUntil, hs.manaShotCost, hs.manaShotDmg)
	}
	startMana := hs.mana
	hs.attackSeq = 1
	s.scheduleProjectileHitLocked(c, mob.id, time.Millisecond)
	c.mvMu.Unlock()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		c.mvMu.Lock()
		spent := hs.mana < startMana
		c.mvMu.Unlock()
		if spent {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	if hs.mana != startMana-12 {
		t.Fatalf("mana after mana-shot swing = %g, want %g", hs.mana, startMana-12)
	}
}

// TestEinzenhaimCastMarkPunishesBossCast: a boss marked by «Изгнание колдовства» takes
// the extra damage the instant it casts a skill, credited to the marker.
func TestEinzenhaimCastMarkPunishesBossCast(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Einzenhaim")
	defer cleanup()
	hs := c.huntState

	boss := mkMob(t, 4200, 3, 0)
	boss.mob.Skills = []gamedata.BossSkill{{Name: "test", Range: 20, Cooldown: 10, CastTime: 0.1, Dmg: 10, Radius: 2}}
	boss.skillReady = []float64{0}
	boss.homed = true

	c.mvMu.Lock()
	hs.mobs[boss.id] = boss
	hs.tr.add(boss.id)
	now := float64(s.battleTime())
	s.applyOpsLocked(c, []gamedata.Op{
		{Kind: gamedata.OpCastMark, Value: gamedata.PerLevel{90}, Dur: gamedata.PerLevel{8}, Radius: 4},
	}, opCtx{slot: 4, level: 4, target: boss}, now)
	if boss.st.castMarkUntil <= now || boss.st.castMarkDmg != 90 || boss.st.castMarkOwner != c.objID {
		t.Fatalf("OpCastMark did not mark the boss: until=%g dmg=%g owner=%d", boss.st.castMarkUntil, boss.st.castMarkDmg, boss.st.castMarkOwner)
	}
	startHP := boss.hp
	fired := s.tryBossSkillLocked(c, boss, []*conn{c}, now)
	c.mvMu.Unlock()

	if !fired {
		t.Fatal("boss should have cast its skill (nothing gates it in this setup)")
	}
	if boss.hp >= startHP {
		t.Fatalf("marked boss took no punish damage on casting: hp stayed at %g", boss.hp)
	}
}

// TestLirveinKillMarkPaysOffLateKill: «Изощренный бросок» marks a surviving target for
// 10s -- a LATER kill by any source still credits the ORIGINAL caster with the
// cooldown reset + attack buff, not just an instant kill by the throw itself.
func TestLirveinKillMarkPaysOffLateKill(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Lirvein")
	defer cleanup()
	hs := c.huntState
	hs.cooldownUntil[1] = 999999 // pretend slot 2 is on a long cooldown

	m := mkMob(t, 4300, 2, 0)
	c.mvMu.Lock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	now := float64(s.battleTime())
	// The throw's own OpOnKill: target survives the initial hit, so this only marks it.
	s.applyOpsLocked(c, []gamedata.Op{
		{Kind: gamedata.OpOnKill, Dur: gamedata.PerLevel{10}, Ops: []gamedata.Op{
			{Kind: gamedata.OpCooldownReset},
		}},
	}, opCtx{slot: 2, level: 4, target: m}, now)
	if m.st.killMarkUntil <= now || m.st.killMarkOwner != c.objID {
		t.Fatalf("surviving target should have been marked: until=%g owner=%d", m.st.killMarkUntil, m.st.killMarkOwner)
	}
	// A later kill (e.g. a basic attack) still triggers the cooldown reset.
	s.hitMobLocked(c, m, m.hp+1, c.objID)
	c.mvMu.Unlock()

	if hs.cooldownUntil[1] != 0 {
		t.Fatalf("late kill within the mark window should have reset cooldowns, slot2 cooldownUntil=%g", hs.cooldownUntil[1])
	}
}

// TestAriannaAuraBuffsNearbyAlly: «Аура стойкости» must buff every nearby FRIENDLY
// avatar, not just Arianna herself.
func TestAriannaAuraBuffsNearbyAlly(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Sp_Arianna")
	defer cleanup()
	hs := c.huntState
	hs.skillLevel[1] = 1 // «Аура стойкости» learned
	hs.worldReady = true

	allyHS := &huntState{av: hs.av, kit: hs.kit, team: hs.team, hp: 500, mana: 100,
		mobs: hs.mobs, summons: map[int32]*summonState{}, worldReady: true}
	ally := &conn{Conn: c.Conn, objID: c.objID + 1, huntState: allyHS, x: 1, y: 0}

	c.mvMu.Lock()
	c.inst = &huntInstance{members: map[int32]*conn{c.objID: c, ally.objID: ally}}
	ally.inst = c.inst
	now := float64(s.battleTime())
	s.tickPassiveAurasLocked(c, now)
	c.mvMu.Unlock()

	found := false
	for _, m := range allyHS.st.mods {
		if m.stat == "phys_armor" {
			found = true
		}
	}
	if !found {
		t.Fatal("«Аура стойкости» did not buff the nearby ally's phys_armor")
	}
}

// TestAbominatorCleansePassiveShedsDebuffs: «Окоченение» sheds every hostile effect
// currently on the caster. Latent in live play (nothing applies avatar-vs-avatar CC
// yet) but the primitive itself must behave correctly when invoked.
func TestAbominatorCleansePassiveShedsDebuffs(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_HK_Abominator")
	defer cleanup()
	hs := c.huntState
	hs.skillLevel[2] = 1 // «Окоченение» learned
	hs.cleanseSlot = 3

	c.mvMu.Lock()
	now := float64(s.battleTime())
	hs.st.stunUntil = now + 5
	hs.st.rootUntil = now + 5
	hs.st.slowUntil = now + 5
	hs.st.dots = []overTime{{perSec: 10, until: now + 5}}
	ok := s.cleansePlayerDebuffsLocked(c, now)
	c.mvMu.Unlock()

	if !ok {
		t.Fatal("cleansePlayerDebuffsLocked reported no cleanse despite a learned OpCleanseOnHit")
	}
	if hs.st.stunUntil != 0 || hs.st.rootUntil != 0 || hs.st.slowUntil != 0 || len(hs.st.dots) != 0 {
		t.Fatalf("debuffs not fully shed: stun=%g root=%g slow=%g dots=%d",
			hs.st.stunUntil, hs.st.rootUntil, hs.st.slowUntil, len(hs.st.dots))
	}
}

// TestChannelGrowthRampsDamage: Op.Growth (Miriam's «Убийственный залп») makes each
// channel pulse deal more damage than the last.
func TestChannelGrowthRampsDamage(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Miriam")
	defer cleanup()
	hs := c.huntState
	m := mkMob(t, 4400, 0, 0)
	m.hp = 100000

	c.mvMu.Lock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	now := float64(s.battleTime())
	hs.channels = []channelState{{
		slot: 4, level: 4, until: now + 10, interval: 1, nextPulse: now,
		target: m.id, ops: []gamedata.Op{{Kind: gamedata.OpDamage, Value: gamedata.PerLevel{72}, Scale: "magic"}},
		growth: 16,
	}}
	hp0 := m.hp
	s.tickChannelsLocked(c, now) // pulse 1: dmgBonus = 16*0 = 0
	hp1 := m.hp
	hs.channels[0].nextPulse = now // force pulse 2 to fire immediately
	s.tickChannelsLocked(c, now)
	hp2 := m.hp
	c.mvMu.Unlock()

	firstHit := hp0 - hp1
	secondHit := hp1 - hp2
	if secondHit <= firstHit {
		t.Fatalf("channel damage did not ramp: first hit %g, second hit %g", firstHit, secondHit)
	}
	if math.Abs(secondHit-firstHit-16) > 0.5 {
		t.Fatalf("ramp step = %g, want ~16 (Op.Growth)", secondHit-firstHit)
	}
}
