package battleserver

import (
	"math"
	"testing"

	"tanatserver/internal/gamedata"
)

// TestArmorMitigationCurve pins the shared armor->damage-multiplier curve used by both
// the player and mob damage paths: positive armor reduces, 0 is a no-op, and negative
// armor ("armor broken", e.g. by Velial's ult) amplifies -- bounded strictly below 2x
// and continuous at 0.
func TestArmorMitigationCurve(t *testing.T) {
	cases := []struct {
		armor, want float64
	}{
		{0, 1.0},             // no armor -> damage lands in full
		{50, 0.5},            // armor == K halves the damage
		{150, 0.25},          // 150/(150+50) mitigated -> quarter damage
		{-50, 1.5},           // broken -50 -> +50% damage
		{-24, 1 + 24.0/74.0}, // Velial rank-1 break on a 0-armor target
	}
	for _, tc := range cases {
		if got := armorMitigation(tc.armor); math.Abs(got-tc.want) > 1e-9 {
			t.Errorf("armorMitigation(%g) = %g, want %g", tc.armor, got, tc.want)
		}
	}
	// Monotonic decreasing in armor.
	prev := math.Inf(1)
	for a := -49.0; a <= 200; a += 1 {
		m := armorMitigation(a)
		if m >= prev {
			t.Fatalf("armorMitigation not strictly decreasing at armor=%g (%g >= %g)", a, m, prev)
		}
		prev = m
	}
	// Negative armor amplifies but never reaches 2x, no matter how deep the break.
	for _, a := range []float64{-1, -50, -200, -1000, -1e6} {
		if m := armorMitigation(a); m <= 1 || m >= 2 {
			t.Errorf("armorMitigation(%g) = %g, want in (1,2)", a, m)
		}
	}
}

// TestMobPhysArmorReducesDamage: a roster mob with base physical armor takes LESS than
// the raw damage, scaled exactly by armorMitigation(base). (Zero-armor trash is covered
// implicitly -- armorMitigation(0)==1, so those hits are unchanged.)
func TestMobPhysArmorReducesDamage(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Cerber")
	defer cleanup()
	hs := c.huntState

	const idx = 6 // Cerber crypt boss -- an armored roster entry
	base := gamedata.Mobs()[idx].PhysArmor
	if base <= 0 {
		t.Fatalf("precondition: roster mob %d should carry armor, got %g", idx, base)
	}
	m := &mobState{id: 3000, mobIdx: idx, mob: gamedata.Mobs()[idx], hp: 100000, shown: true}

	const raw = 1000.0
	c.mvMu.Lock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	before := m.hp
	s.hitMobLocked(c, m, raw, c.objID)
	got := before - m.hp
	c.mvMu.Unlock()

	want := raw * armorMitigation(base)
	if math.Abs(got-want) > 1e-6 {
		t.Fatalf("armored hit removed %.4f hp, want %.4f (raw %.0f, armor %g)", got, want, raw, base)
	}
	if got >= raw {
		t.Fatalf("armor did not mitigate: removed %.4f of %.0f raw", got, raw)
	}
}

// TestVelialUltBreaksMobArmor is the headline: Velial's ult «Трибунал» applies a negative
// phys_armor debuff to its target, so the SAME raw hit strips MORE HP off a debuffed mob
// than an intact one. Also data-pins the ult's debuff op and the exact broken-armor math.
func TestVelialUltBreaksMobArmor(t *testing.T) {
	// Data pin: the ult (slot 4) must carry a negative phys_armor debuff On the target.
	vel := avatarByPrefab(t, "Avtr_Tank_Velial")
	ult := gamedata.SkillsFor(vel).Skills[3]
	var debuff float64
	found := false
	for _, op := range ult.Ops {
		if op.Kind == gamedata.OpBuffStat && op.Stat == "phys_armor" && op.On == "target" {
			debuff, found = op.Value.At(1), true // rank 1
		}
	}
	if !found || debuff >= 0 {
		t.Fatalf("Velial ult should apply a negative phys_armor debuff to the target (found=%v value=%g)", found, debuff)
	}

	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	hs := c.huntState

	const idx = 6 // armored target (Cerber boss)
	base := gamedata.Mobs()[idx].PhysArmor
	mk := func(id int32) *mobState {
		return &mobState{id: id, mobIdx: idx, mob: gamedata.Mobs()[idx], hp: 100000, x: 3, y: 0, shown: true}
	}
	ctl := mk(4000) // control: armor intact
	brk := mk(4001) // armor broken by the ult

	const raw = 1000.0
	c.mvMu.Lock()
	hs.mobs[ctl.id] = ctl
	hs.mobs[brk.id] = brk
	hs.tr.add(ctl.id)
	hs.tr.add(brk.id)
	now := float64(s.battleTime())

	// Cast the ult on brk only (rank 1) -- appends the phys_armor debuff to brk.st.
	s.applyOpsLocked(c, ult.Ops, opCtx{slot: 4, level: 1, target: brk}, now)
	brokenArmor := brk.physArmor(now)

	ctlBefore, brkBefore := ctl.hp, brk.hp
	s.hitMobLocked(c, ctl, raw, c.objID)
	s.hitMobLocked(c, brk, raw, c.objID)
	ctlDrop := ctlBefore - ctl.hp
	brkDrop := brkBefore - brk.hp
	c.mvMu.Unlock()

	if math.Abs(brokenArmor-(base+debuff)) > 1e-6 {
		t.Fatalf("after ult, mob armor = %g, want %g (base %g + debuff %g)", brokenArmor, base+debuff, base, debuff)
	}
	if brkDrop <= ctlDrop {
		t.Fatalf("Velial ult did not increase damage taken: broken %.3f vs intact %.3f", brkDrop, ctlDrop)
	}
	if want := raw * armorMitigation(base+debuff); math.Abs(brkDrop-want) > 1e-6 {
		t.Fatalf("broken-armor hit removed %.4f, want %.4f (armor %g)", brkDrop, want, base+debuff)
	}
}

// TestArmorPenetrationChipsMobArmor: an attacker's phys_armor_pen (an AntiPhysArmor
// potion) reduces the mob's effective armor toward 0 before mitigation, so the hit lands
// harder -- but penetration alone cannot push armor negative (that is the debuff's job).
func TestArmorPenetrationChipsMobArmor(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Cerber")
	defer cleanup()
	hs := c.huntState

	const idx = 6
	base := gamedata.Mobs()[idx].PhysArmor
	const pen = 10.0
	if pen >= base {
		t.Fatalf("test assumes pen (%g) < base armor (%g) so armor stays positive", pen, base)
	}
	m := &mobState{id: 5000, mobIdx: idx, mob: gamedata.Mobs()[idx], hp: 100000, shown: true}

	const raw = 1000.0
	c.mvMu.Lock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	now := float64(s.battleTime())
	hs.st.mods = append(hs.st.mods, statMod{stat: "phys_armor_pen", value: pen, until: now + 60})
	before := m.hp
	s.hitMobLocked(c, m, raw, c.objID)
	got := before - m.hp
	c.mvMu.Unlock()

	want := raw * armorMitigation(base-pen)
	if math.Abs(got-want) > 1e-6 {
		t.Fatalf("penetrated hit removed %.4f hp, want %.4f (raw %.0f, armor %g-%g)", got, want, raw, base, pen)
	}
	if noPen := raw * armorMitigation(base); got <= noPen {
		t.Fatalf("armor penetration did not help: removed %.3f, un-penetrated would remove %.3f", got, noPen)
	}
}
