package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

// skillsForPrefab looks up an avatar's authored kit by its prefab name.
func skillsForPrefab(t *testing.T, prefab string) *gamedata.AvatarSkills {
	t.Helper()
	for _, av := range gamedata.Avatars() {
		if av.Prefab == prefab {
			return gamedata.SkillsFor(av)
		}
	}
	t.Fatalf("no avatar with prefab %q", prefab)
	return nil
}

// TestDashCleavesAreSwaths pins the data: each confirmed dash-cleave (its wiki text
// says the damage lands ALONG the dash path -- "на пути" / "в радиус действия рывка")
// has a dash op AND an authored AoEWidth, so damageTargetsLocked routes it through
// the line-swath path instead of a destination circle.
func TestDashCleavesAreSwaths(t *testing.T) {
	cases := []struct {
		prefab string
		slot   int
	}{
		{"Avtr_HK_ShinDalar", 1}, // Смертоносный рывок (reported)
		{"Avtr_DPS_Gayal", 3},    // Туманное перемещение ("каждому врагу... на пути")
		{"Avtr_Dsb_Wilfang", 1},  // Сокрушительный рывок ("распихивая попавшихся на пути")
	}
	for _, tc := range cases {
		kit := skillsForPrefab(t, tc.prefab)
		sk := kit.Skills[tc.slot-1]
		if sk.AoEWidth <= 0 {
			t.Errorf("%s slot %d: AoEWidth=%d, want >0 (dash damage must be a swath along the path, not a circle at the point)", tc.prefab, tc.slot, sk.AoEWidth)
		}
		hasDash := false
		for _, op := range sk.Ops {
			if op.Kind == gamedata.OpDash {
				hasDash = true
			}
		}
		if !hasDash {
			t.Errorf("%s slot %d: expected an OpDash (it is a dash-cleave)", tc.prefab, tc.slot)
		}
	}
}

// TestLandingSlamStaysCircle guards the other side: Zamaran's "Таран" damages only
// AFTER reaching the point (wiki: "Добежав до точки... наносит урон"), so it must
// stay a destination circle (AoEWidth==0), NOT be converted to a swath.
func TestLandingSlamStaysCircle(t *testing.T) {
	kit := skillsForPrefab(t, "Avtr_Tank_Zamaran")
	if w := kit.Skills[0].AoEWidth; w != 0 {
		t.Errorf("Zamaran slot 1 AoEWidth=%d, want 0 -- it is a landing slam, not a path-cleave", w)
	}
}

// TestShinDalarDashHitsAlongPath is the behavioural proof: the slot-1 dash-cleave
// hits a mob standing MIDWAY along the dash line (which the old destination circle
// would have missed) while sparing a mob off to the side and one behind the caster.
func TestShinDalarDashHitsAlongPath(t *testing.T) {
	s, c, _, sx, sy := newNavConnAvatar(t, 16) // Shin Dalar
	hs := c.huntState

	midway := &mobState{id: 5000, mobIdx: 0, mob: gamedata.Mobs()[0], x: sx + 4, y: sy}   // on the dash line, far from the endpoint
	side := &mobState{id: 5001, mobIdx: 0, mob: gamedata.Mobs()[0], x: sx + 4, y: sy + 5} // off to the side
	behind := &mobState{id: 5002, mobIdx: 0, mob: gamedata.Mobs()[0], x: sx - 3, y: sy}   // behind the caster
	for _, m := range []*mobState{midway, side, behind} {
		hs.mobs[m.id] = m
	}

	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	// Aim 8m ahead (+x). The endpoint circle of the old (radius-3) implementation
	// would have centred on (sx+8, sy) -- 4m from the midway mob, so it missed it.
	ctx := opCtx{slot: 1, level: 1, hasPos: true, px: sx + 8, py: sy}
	var dmgOp gamedata.Op
	for _, op := range hs.skillDef(1).Ops {
		if op.Kind == gamedata.OpDamage {
			dmgOp = op
		}
	}
	hit := map[int32]bool{}
	for _, m := range s.damageTargetsLocked(c, ctx, dmgOp.Radius) {
		hit[m.id] = true
	}
	if !hit[midway.id] {
		t.Error("dash-cleave missed a mob standing midway along the dash line (the old destination circle would have too)")
	}
	if hit[side.id] {
		t.Error("dash-cleave hit a mob well off to the side -- swath is too wide / not a line")
	}
	if hit[behind.id] {
		t.Error("dash-cleave hit a mob behind the caster -- swath must start at the caster")
	}
}
