package battleserver

import (
	"net"
	"testing"

	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// twoMemberInstance builds a shared huntInstance with two world-ready members, each on a
// drained pipe so *Locked helpers can push HP/stat syncs without blocking.
func twoMemberInstance(t *testing.T, aPrefab, bPrefab string, ax, ay, bx, by float32) (*Server, *conn, *conn) {
	t.Helper()
	s := New(session.NewStore())
	inst := &huntInstance{id: 1, members: map[int32]*conn{}}
	mk := func(obj int32, prefab string, x, y float32) *conn {
		av := avatarByPrefab(t, prefab)
		srv, cli := net.Pipe()
		t.Cleanup(func() { srv.Close(); cli.Close() })
		go func() {
			b := make([]byte, 4096)
			for {
				if _, err := cli.Read(b); err != nil {
					return
				}
			}
		}()
		hs := &huntState{
			av: av, kit: gamedata.SkillsFor(av), inst: inst, worldReady: true,
			mobs: map[int32]*mobState{}, summons: map[int32]*summonState{}, hp: 100, mana: 300,
		}
		for i := range hs.skillLevel {
			hs.skillLevel[i] = 1
		}
		c := &conn{Conn: srv, inst: inst, huntState: hs}
		c.objID, c.x, c.y, c.snapT = obj, x, y, 0
		hs.tr.add(obj)
		inst.members[obj] = c
		t.Cleanup(func() { c.mvMu.Lock(); hs.closed = true; c.mvMu.Unlock() })
		return c
	}
	return s, mk(1000, aPrefab, ax, ay), mk(1001, bPrefab, bx, by)
}

// TestAllyTargetsSoloIsSelf: with no instance, both On:"ally" and On:"allies" resolve to
// just the caster (self is an ally), so ally-heal/shield skills stay usable in solo.
func TestAllyTargetsSoloIsSelf(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Sp_Arianna")
	defer cleanup()
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	for _, on := range []string{"ally", "allies"} {
		got := s.allyTargetsLocked(c, opCtx{slot: 3, level: 1}, gamedata.Op{On: on, Radius: 5})
		if len(got) != 1 || got[0] != c {
			t.Errorf("solo On:%q -> %d targets, want [self]", on, len(got))
		}
	}
}

// TestAllyTargetsRadius: On:"allies" catches every party member inside the radius of the
// AoE centre (self included) and excludes those outside it.
func TestAllyTargetsRadius(t *testing.T) {
	s, a, b := twoMemberInstance(t, "Avtr_Sp_Arianna", "Avtr_Sp_Kiona", 0, 0, 3, 0)
	// A third member far outside the radius.
	far := avatarByPrefab(t, "Avtr_HK_Tangren")
	fc := &conn{inst: a.inst, huntState: &huntState{av: far, kit: gamedata.SkillsFor(far), inst: a.inst, worldReady: true, mobs: map[int32]*mobState{}}}
	fc.objID, fc.x, fc.y = 1002, 40, 0
	a.inst.members[1002] = fc

	a.mvMu.Lock()
	defer a.mvMu.Unlock()
	got := s.allyTargetsLocked(a, opCtx{slot: 3, level: 1}, gamedata.Op{On: "allies", Radius: 5})
	set := map[*conn]bool{}
	for _, m := range got {
		set[m] = true
	}
	if !set[a] || !set[b] {
		t.Errorf("On:allies should include the caster and the nearby ally (got %d)", len(got))
	}
	if set[fc] {
		t.Error("On:allies must exclude a member outside the radius")
	}
}

// TestAllyHealCrossMember: an On:"allies" heal cast by one member restores a WOUNDED
// nearby party member's HP (real ally-targeting, not a self-only heal).
func TestAllyHealCrossMember(t *testing.T) {
	s, caster, ally := twoMemberInstance(t, "Avtr_Sp_Arianna", "Avtr_Sp_Kiona", 0, 0, 2, 0)
	ally.huntState.hp = 40

	caster.mvMu.Lock()
	defer caster.mvMu.Unlock()
	now := float64(s.battleTime())
	op := gamedata.Op{Kind: gamedata.OpHeal, Value: gamedata.PerLevel{150, 150, 150, 150}, On: "allies", Radius: 5}
	s.applyOpsLocked(caster, []gamedata.Op{op}, opCtx{slot: 3, level: 1}, now)

	if ally.huntState.hp <= 40 {
		t.Errorf("nearby ally HP = %.0f, want healed above 40", ally.huntState.hp)
	}
}

// TestAllyShieldSingleTarget: On:"ally" puts the absorb shield on the aimed friend
// (ctx.allyTarget), not the caster.
func TestAllyShieldSingleTarget(t *testing.T) {
	s, caster, ally := twoMemberInstance(t, "Avtr_Sp_Arianna", "Avtr_Sp_Kiona", 0, 0, 2, 0)
	caster.mvMu.Lock()
	defer caster.mvMu.Unlock()
	now := float64(s.battleTime())
	op := gamedata.Op{Kind: gamedata.OpShield, Value: gamedata.PerLevel{200, 200, 200, 200}, Dur: gamedata.PerLevel{5, 5, 5, 5}, On: "ally"}
	s.applyOpsLocked(caster, []gamedata.Op{op}, opCtx{slot: 1, level: 1, allyTarget: ally}, now)

	if ally.huntState.st.shield <= 0 {
		t.Error("On:ally shield must land on the aimed friend")
	}
	if caster.huntState.st.shield != 0 {
		t.Error("On:ally shield must NOT land on the caster when a friend was aimed")
	}
}

// TestExecuteThreshold: «Казнь» (OpExecute) instant-kills a target at/below the HP
// threshold (Value2) and merely damages one above it.
func TestExecuteThreshold(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Gektor")
	defer cleanup()
	hs := c.huntState
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())

	op := gamedata.Op{Kind: gamedata.OpExecute, Value: gamedata.PerLevel{150, 150, 150, 150},
		Value2: gamedata.PerLevel{120, 120, 120, 120}, Scale: "magic", PerSP: 1}

	low := &mobState{id: 3000, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 100, x: 1, y: 0, shown: true}
	hs.tr.add(low.id)
	hs.mobs[low.id] = low
	s.applyOpsLocked(c, []gamedata.Op{op}, opCtx{slot: 4, level: 1, target: low}, now)
	if !low.dead {
		t.Errorf("execute must kill a target (%.0f HP) at/below the threshold (120)", low.hp)
	}

	high := &mobState{id: 3001, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 5000, x: 1, y: 0, shown: true}
	hs.tr.add(high.id)
	hs.mobs[high.id] = high
	s.applyOpsLocked(c, []gamedata.Op{op}, opCtx{slot: 4, level: 1, target: high}, now)
	if high.dead {
		t.Error("execute must NOT instant-kill a target above the threshold")
	}
	if high.hp >= 5000 {
		t.Error("execute must still deal its Value damage above the threshold")
	}
}

// TestConsecutiveHitStreak: «Трепка» bonus is 0 on the first hit of a target, grows by
// Value each subsequent same-target hit, and resets to 0 when the target changes.
func TestConsecutiveHitStreak(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_HK_Mihalych")
	defer cleanup()
	hs := c.huntState
	// World-build sets consecutiveHitSlot + learns skills; mirror both on this bare conn.
	for i := range hs.skillLevel {
		hs.skillLevel[i] = 1
	}
	for i, sk := range hs.kit.Skills {
		for _, op := range sk.Ops {
			if op.Kind == gamedata.OpConsecutiveHit {
				hs.consecutiveHitSlot = i + 1
			}
		}
	}
	if hs.consecutiveHitSlot == 0 {
		t.Fatal("Mihalych should carry an OpConsecutiveHit passive")
	}
	per := 15.0 // rank-1 «Трепка» bonus per stack

	if b := s.consecutiveHitBonusLocked(hs, 500); b != 0 {
		t.Errorf("first hit bonus = %.0f, want 0", b)
	}
	if b := s.consecutiveHitBonusLocked(hs, 500); b != per {
		t.Errorf("second same-target hit bonus = %.0f, want %.0f", b, per)
	}
	if b := s.consecutiveHitBonusLocked(hs, 500); b != 2*per {
		t.Errorf("third same-target hit bonus = %.0f, want %.0f", b, 2*per)
	}
	if b := s.consecutiveHitBonusLocked(hs, 999); b != 0 {
		t.Errorf("switching targets must reset the streak, got %.0f", b)
	}
}

// TestSoulStacksGrowAttack: Gellar's «Порабощение» banks +2 base attack per kill up to the
// rank cap, and a death halves the banked souls.
func TestSoulStacksGrowAttack(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Gellar")
	defer cleanup()
	hs := c.huntState
	hs.skillLevel[1] = 1 // learn s2; register the slot as world-build would
	hs.soulSlot = 2
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())

	if b := s.killAttackBonusLocked(hs, now); b != 0 {
		t.Fatalf("no souls yet, attack bonus = %.0f want 0", b)
	}
	for i := 0; i < 3; i++ {
		s.applyOnKillStacksLocked(c, now)
	}
	if hs.soulStacks != 3 {
		t.Fatalf("soulStacks = %d want 3", hs.soulStacks)
	}
	if b := s.killAttackBonusLocked(hs, now); b != 6 { // 3 souls × +2
		t.Fatalf("soul attack bonus = %.0f want 6", b)
	}
	// The soul count is capped at rank-1 charges = 10.
	for i := 0; i < 20; i++ {
		s.applyOnKillStacksLocked(c, now)
	}
	if hs.soulStacks != 10 {
		t.Fatalf("soulStacks capped = %d want 10", hs.soulStacks)
	}
	// A death sheds half the souls.
	hs.hp = 0
	s.playerDieLocked(c, 99, now)
	if hs.soulStacks != 5 {
		t.Errorf("after death soulStacks = %d want 5 (half)", hs.soulStacks)
	}
}

// TestKillWindowStacksAttack: Hekata's «Культ жнеца» banks flat attack per kill only while
// its cast-opened window is live; the bonus lapses once the window closes.
func TestKillWindowStacksAttack(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Hekata")
	defer cleanup()
	hs := c.huntState
	hs.skillLevel[1] = 1 // s2 rank 1
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())

	// Hekata has no persistent soul slot: a kill with no window open banks nothing.
	s.applyOnKillStacksLocked(c, now)
	if b := s.killAttackBonusLocked(hs, now); b != 0 {
		t.Fatalf("kill outside a window must add no attack, got %.0f", b)
	}
	// Casting «Культ жнеца» opens the kill-window.
	s.applyOpsLocked(c, hs.skillDef(2).Ops, opCtx{slot: 2, level: 1}, now)
	if hs.killWindowUntil <= now {
		t.Fatalf("window not opened: killWindowUntil=%g now=%g", hs.killWindowUntil, now)
	}
	// Two kills during the window → +4 attack each (rank-1 damagePerKill).
	s.applyOnKillStacksLocked(c, now)
	s.applyOnKillStacksLocked(c, now)
	if hs.killWindowStacks != 2 {
		t.Fatalf("killWindowStacks = %d want 2", hs.killWindowStacks)
	}
	if b := s.killAttackBonusLocked(hs, now); b != 8 { // 2 × +4
		t.Fatalf("window attack bonus = %.0f want 8", b)
	}
	// Once the window closes the bonus is gone.
	if b := s.killAttackBonusLocked(hs, hs.killWindowUntil+1); b != 0 {
		t.Errorf("attack bonus after window = %.0f want 0", b)
	}
}

// TestLirveinStreakHastensSwing: «Неумолимость» shortens the swing interval as the
// same-target streak grows, capped at speedModMax; an avatar without the passive is
// unaffected.
func TestLirveinStreakHastensSwing(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Lirvein")
	defer cleanup()
	hs := c.huntState
	hs.skillLevel[2] = 1 // s3 rank 1
	hs.attackSpeedStreakSlot = 3
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	base := s.swingIntervalLocked(hs)
	if s.attackSpeedStreakBonusLocked(hs) != 0 {
		t.Fatalf("a fresh streak must add no haste")
	}
	// A growing streak adds attack speed and shortens the interval.
	hs.hitStreak = 2
	if s.attackSpeedStreakBonusLocked(hs) <= 0 {
		t.Fatalf("streak 2 should add attack speed")
	}
	if got := s.swingIntervalLocked(hs); got >= base {
		t.Errorf("streak should shorten the swing interval: base=%v streak=%v", base, got)
	}
	// The per-hit speed is capped at rank-1 speedModMax = 0.1.
	hs.hitStreak = 1000
	if b := s.attackSpeedStreakBonusLocked(hs); b != 0.1 {
		t.Errorf("streak haste must cap at 0.1, got %g", b)
	}
	// Without the passive, no streak haste at all (scoped: other avatars' cadence is fixed).
	hs.attackSpeedStreakSlot = 0
	if s.attackSpeedStreakBonusLocked(hs) != 0 {
		t.Error("an avatar without «Неумолимость» must get no streak haste")
	}
}

// TestGektorAttackDamageBonus: OpAttackDamage scales with the caster's base attack -- a 0.5
// coefficient deals ~twice what a 0.25 coefficient does to the same target.
func TestGektorAttackDamageBonus(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Gektor")
	defer cleanup()
	hs := c.huntState
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())
	if hs.baseAttackLocked(now) <= 0 {
		t.Fatal("Gektor base attack should be positive")
	}
	mk := func(id int32) *mobState {
		m := &mobState{id: id, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 100000, x: 1, y: 0, shown: true}
		hs.tr.add(m.id)
		hs.mobs[m.id] = m
		return m
	}
	half, quarter := mk(4000), mk(4001)
	s.applyOpsLocked(c, []gamedata.Op{{Kind: gamedata.OpAttackDamage, Value: gamedata.PerLevel{0.5, 0.5, 0.5, 0.5}, Scale: "phys"}}, opCtx{slot: 3, level: 1, target: half}, now)
	s.applyOpsLocked(c, []gamedata.Op{{Kind: gamedata.OpAttackDamage, Value: gamedata.PerLevel{0.25, 0.25, 0.25, 0.25}, Scale: "phys"}}, opCtx{slot: 3, level: 1, target: quarter}, now)
	dHalf := 100000 - half.hp
	dQuarter := 100000 - quarter.hp
	if dHalf <= 0 || dQuarter <= 0 {
		t.Fatalf("both targets should take attack-scaled damage: half=%.1f quarter=%.1f", dHalf, dQuarter)
	}
	if r := dHalf / dQuarter; r < 1.8 || r > 2.2 {
		t.Errorf("attack-scaled damage ratio = %.2f, want ~2.0 (0.5 vs 0.25 coef)", r)
	}
}

// TestBoneShieldExplodesOnThirdHit: Rognar's «Костяной щит» absorbs two hits, then the
// third detonates it -- the toggle switches off and a nearby enemy takes the blast.
func TestBoneShieldExplodesOnThirdHit(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Rognar")
	defer cleanup()
	hs := c.huntState
	hs.skillLevel[1] = 1 // s2 rank 1
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())

	m := &mobState{id: 4100, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 100000, x: 1, y: 0, shown: true}
	hs.tr.add(m.id)
	hs.mobs[m.id] = m

	s.toggleSkillLocked(c, 2) // raise the bone shield
	if hs.shieldExplodeSlot != 2 || hs.shieldHitsLeft != shieldExplodeHits {
		t.Fatalf("shield not armed: slot=%d hitsLeft=%d", hs.shieldExplodeSlot, hs.shieldHitsLeft)
	}
	// Two hits: shield holds, no blast.
	s.hitPlayerFromLocked(c, m.id, 5, now, m, nil)
	s.hitPlayerFromLocked(c, m.id, 5, now, m, nil)
	if hs.shieldExplodeSlot == 0 {
		t.Fatalf("shield must still be up after two hits")
	}
	if m.hp != 100000 {
		t.Fatalf("enemy must be unharmed before the shield pops, hp=%.0f", m.hp)
	}
	// Third hit detonates.
	s.hitPlayerFromLocked(c, m.id, 5, now, m, nil)
	if hs.shieldExplodeSlot != 0 || hs.toggleOn[1] {
		t.Errorf("shield should be spent and toggle off after the third hit (slot=%d on=%v)", hs.shieldExplodeSlot, hs.toggleOn[1])
	}
	if m.hp >= 100000 {
		t.Errorf("the explosion must damage the nearby enemy, hp=%.0f", m.hp)
	}
}

// --- deferred batch #4: mob mana + dual casts + Frost chill + Rognar/Gellar mechanics ---

func mkManaMob(t *testing.T, hs *huntState, id int32, maxMana float64) *mobState {
	t.Helper()
	m := &mobState{id: id, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 100000, maxMana: maxMana, mana: maxMana, x: 1, y: 0, shown: true}
	hs.tr.add(m.id)
	hs.mobs[m.id] = m
	return m
}

// TestMobManaHelpers: only ranged/boss mobs carry mana; drainManaLocked clamps at 0.
func TestMobManaHelpers(t *testing.T) {
	melee := &mobState{maxMana: 0}
	if melee.hasMana() || melee.drainManaLocked(50) != 0 {
		t.Error("a melee mob must have no mana and drain nothing")
	}
	ranged := &mobState{maxMana: 150, mana: 100}
	if !ranged.hasMana() {
		t.Error("a ranged mob must have mana")
	}
	if got := ranged.drainManaLocked(30); got != 30 || ranged.mana != 70 {
		t.Errorf("drain 30 -> took %.0f, mana %.0f", got, ranged.mana)
	}
	if got := ranged.drainManaLocked(1000); got != 70 || ranged.mana != 0 {
		t.Errorf("over-drain must clamp at 0: took %.0f, mana %.0f", got, ranged.mana)
	}
}

// TestNeirofimDevourDrainsMana: «Пожирание магии» drains a % of Neirofim's own max mana
// from the target and deals damage equal to what was taken.
func TestNeirofimDevourDrainsMana(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Sp_Neirofim")
	defer cleanup()
	hs := c.huntState
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())
	m := mkManaMob(t, hs, 5000, 200)
	op := gamedata.Op{Kind: gamedata.OpManaBurnHit, Value: gamedata.PerLevel{0.05, 0.05, 0.05, 0.05}, Value2: gamedata.PerLevel{1, 1, 1, 1}, Apply: "own_mana"}
	s.applyOpsLocked(c, []gamedata.Op{op}, opCtx{slot: 3, level: 1, target: m}, now)
	if m.mana >= 200 {
		t.Errorf("target mana should drop, got %.0f", m.mana)
	}
	if m.hp >= 100000 {
		t.Errorf("devour should damage the target, hp %.0f", m.hp)
	}
}

// TestManaScaledDamageMissingMana: Neirofim s1 hits a mana-DRY target harder, and slows a
// mana-FULL target more.
func TestManaScaledDamageMissingMana(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Sp_Neirofim")
	defer cleanup()
	hs := c.huntState
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())
	op := gamedata.Op{Kind: gamedata.OpManaScaledDamage, Value: gamedata.PerLevel{70, 70, 70, 70}, Value2: gamedata.PerLevel{0.5, 0.5, 0.5, 0.5}, Dur: gamedata.PerLevel{3, 3, 3, 3}, Scale: "magic"}
	full := mkManaMob(t, hs, 5001, 200)
	dry := mkManaMob(t, hs, 5002, 200)
	dry.mana = 0
	s.applyOpsLocked(c, []gamedata.Op{op}, opCtx{slot: 1, level: 1, target: full}, now)
	s.applyOpsLocked(c, []gamedata.Op{op}, opCtx{slot: 1, level: 1, target: dry}, now)
	if (100000 - dry.hp) <= (100000 - full.hp) {
		t.Errorf("a mana-dry target must take MORE damage (%.0f) than a full one (%.0f)", 100000-dry.hp, 100000-full.hp)
	}
	if full.st.slowFactor >= dry.st.slowFactor {
		t.Errorf("a mana-full target must be slowed more (full %.2f vs dry %.2f)", full.st.slowFactor, dry.st.slowFactor)
	}
}

// TestSilenceAllSilencesAndDrains: «Молчание» silences every mob and drains only the nearby.
func TestSilenceAllSilencesAndDrains(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Sp_Neirofim")
	defer cleanup()
	hs := c.huntState
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())
	near := mkManaMob(t, hs, 5006, 200)
	far := mkManaMob(t, hs, 5007, 200)
	far.x = 100
	op := gamedata.Op{Kind: gamedata.OpSilenceAll, Dur: gamedata.PerLevel{5, 5, 5, 5}, Value: gamedata.PerLevel{50, 50, 50, 50}, Radius: 12}
	s.applyOpsLocked(c, []gamedata.Op{op}, opCtx{slot: 4, level: 1, px: 0, py: 0, hasPos: true}, now)
	if !near.st.silenced(now) || !far.st.silenced(now) {
		t.Error("silence-all must silence EVERY hostile mob on the map")
	}
	if near.mana >= 200 {
		t.Error("a nearby mob must lose mana")
	}
	if far.mana != 200 {
		t.Error("a mob outside the drain radius must keep its mana")
	}
}

// TestChillComboStuns: a second OpChill on a chilled mob stuns it and clears the chill.
func TestChillComboStuns(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Frost")
	defer cleanup()
	hs := c.huntState
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())
	m := &mobState{id: 5003, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 100000, x: 1, shown: true}
	hs.tr.add(m.id)
	hs.mobs[m.id] = m
	op := gamedata.Op{Kind: gamedata.OpChill, Dur: gamedata.PerLevel{40, 40, 40, 40}, Value2: gamedata.PerLevel{1, 1, 1, 1}}
	ctx := opCtx{slot: 1, level: 1, target: m, px: m.x, py: m.y, hasPos: true}
	s.applyOpsLocked(c, []gamedata.Op{op}, ctx, now)
	if now >= m.st.chillUntil {
		t.Fatal("first chill must open the chill window")
	}
	if m.st.stunned(now) {
		t.Error("first chill must NOT stun")
	}
	s.applyOpsLocked(c, []gamedata.Op{op}, ctx, now)
	if !m.st.stunned(now) {
		t.Error("re-chilling a chilled mob must stun it")
	}
	if m.st.chillUntil != 0 {
		t.Error("the combo stun must clear the chill")
	}
}

// TestBloodlettingEmpowersNextHit: Rognar s1 spends HP and stores a next-hit magic bonus.
func TestBloodlettingEmpowersNextHit(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Rognar")
	defer cleanup()
	hs := c.huntState
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())
	hs.hp = 1000
	op := gamedata.Op{Kind: gamedata.OpEmpowerNextHit, Value: gamedata.PerLevel{2, 2, 2, 2}, Value2: gamedata.PerLevel{0.1, 0.1, 0.1, 0.1}}
	s.applyOpsLocked(c, []gamedata.Op{op}, opCtx{slot: 1, level: 1}, now)
	if hs.hp != 900 {
		t.Errorf("must spend 10%% of 1000 HP, hp=%.0f", hs.hp)
	}
	if hs.nextHitBonus != 200 {
		t.Errorf("next-hit bonus = coef 2 × 100 HP spent = 200, got %.0f", hs.nextHitBonus)
	}
}

// TestDeathLinkRedirects: Rognar's «Канал смерти» forwards a share of an incoming blow to
// the linked mob.
func TestDeathLinkRedirects(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Rognar")
	defer cleanup()
	hs := c.huntState
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())
	m := &mobState{id: 5004, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 100000, x: 1, shown: true}
	hs.tr.add(m.id)
	hs.mobs[m.id] = m
	hs.hp = 100000
	hs.deathLinkObj, hs.deathLinkUntil, hs.deathLinkFrac = m.id, now+10, 0.3
	s.hitPlayerFromLocked(c, 999, 100, now, nil, nil)
	if m.hp >= 100000 {
		t.Errorf("the linked mob must take a share of the blow, hp=%.0f", m.hp)
	}
}

// TestDualCastGate: an "enemy" op fires only on a foe aim, an "ally" op only on a friend aim.
func TestDualCastGate(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Sp_Kiona")
	defer cleanup()
	hs := c.huntState
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())
	m := &mobState{id: 5005, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 100000, x: 1, shown: true}
	hs.tr.add(m.id)
	hs.mobs[m.id] = m
	ops := []gamedata.Op{
		{Kind: gamedata.OpDamage, Value: gamedata.PerLevel{100, 100, 100, 100}, Scale: "magic", TargetSide: "enemy"},
		{Kind: gamedata.OpHeal, Value: gamedata.PerLevel{100, 100, 100, 100}, On: "ally", TargetSide: "ally"},
	}
	// Aim an enemy: only the damage fires.
	hs.hp = 500
	s.applyOpsLocked(c, ops, opCtx{slot: 4, level: 1, target: m, px: m.x, py: m.y, hasPos: true}, now)
	if m.hp >= 100000 {
		t.Error("the enemy op must fire on an enemy aim")
	}
	if hs.hp != 500 {
		t.Errorf("the ally heal must NOT fire on an enemy aim, hp=%.0f", hs.hp)
	}
	// Aim a friend (self): only the heal fires.
	hs.hp = 500
	before := m.hp
	s.applyOpsLocked(c, ops, opCtx{slot: 4, level: 1, allyTarget: c}, now)
	if hs.hp <= 500 {
		t.Error("the ally heal must fire on a friend aim")
	}
	if m.hp != before {
		t.Error("the enemy op must NOT fire on a friend aim")
	}
}

// TestGellarArmyScalesWithSouls: «Армия душ» damage grows with banked souls (PerSoul).
func TestGellarArmyScalesWithSouls(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Gellar")
	defer cleanup()
	hs := c.huntState
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	op := gamedata.Op{Kind: gamedata.OpDamage, Value: gamedata.PerLevel{20, 20, 20, 20}, PerSoul: gamedata.PerLevel{5, 5, 5, 5}, Scale: "magic"}
	hs.soulStacks = 0
	d0 := s.skillDamageLocked(c, op, opCtx{slot: 4, level: 1}, nil)
	hs.soulStacks = 4
	d4 := s.skillDamageLocked(c, op, opCtx{slot: 4, level: 1}, nil)
	if d4 <= d0 {
		t.Errorf("more souls must deal more damage: 0 souls=%.0f, 4 souls=%.0f", d0, d4)
	}
	if d4-d0 != 4*5*hs.powerMul() {
		t.Errorf("soul bonus = 4 souls × 5 = 20, got %.0f", d4-d0)
	}
}
