package gamedata

import "testing"

// TestSkillRankShape: slots 1-3 normalize to 5 ranks, the ult (slot 4) to 4, for
// every avatar in the roster (authored 4-rank kits get a 5th rank via normalizeKit).
func TestSkillRankShape(t *testing.T) {
	for _, a := range Avatars() {
		kit := SkillsFor(a)
		for i := 0; i < 4; i++ {
			sk := kit.Skills[i]
			want := 5
			if i == 3 {
				want = 4
			}
			if got := sk.MaxRank(); got != want {
				t.Errorf("%s slot %d MaxRank = %d, want %d", a.Prefab, i+1, got, want)
			}
			if len(sk.ManaCost) != want || len(sk.Cooldown) != want {
				t.Errorf("%s slot %d cost/cd lengths = %d/%d, want %d",
					a.Prefab, i+1, len(sk.ManaCost), len(sk.Cooldown), want)
			}
		}
	}
}

// TestPerLevelAtClampsToLen: At() clamps to the array length (rank 5 of a 5-array
// is the 5th value; rank 6 clamps to the 5th; an empty array yields 0).
func TestPerLevelAtClampsToLen(t *testing.T) {
	p := PerLevel{10, 20, 30, 40, 50}
	if p.At(5) != 50 || p.At(6) != 50 || p.At(0) != 10 {
		t.Errorf("At clamp wrong: At(5)=%g At(6)=%g At(0)=%g", p.At(5), p.At(6), p.At(0))
	}
	if (PerLevel{}).At(3) != 0 {
		t.Error("empty PerLevel must yield 0")
	}
}

// TestExtendPerLevelContinuesTrend: normalizeKit's extrapolation continues the last
// delta (flat stays flat; growth continues).
func TestExtendPerLevelContinuesTrend(t *testing.T) {
	if got := extendSeq(PerLevel{70, 100, 135, 175}, 5); got.At(5) != 215 {
		t.Errorf("growth extrapolation rank5 = %g, want 215 (175 + last delta 40)", got.At(5))
	}
	if got := extendSeq(PerLevel{2, 2, 2, 2}, 5); got.At(5) != 2 {
		t.Errorf("flat extrapolation rank5 = %g, want 2", got.At(5))
	}
	if got := extendSeq([]int{100, 100, 100, 100}, 5); got[4] != 100 {
		t.Errorf("flat int extrapolation rank5 = %d, want 100", got[4])
	}
}

// TestAuthoredFileSkillValues pins the stats.txt skill numbers authored under the
// 5-rank model for the file-specified avatars.
func TestAuthoredFileSkillValues(t *testing.T) {
	pm := skillsByPrefab["Avtr_Dsb_PlusMinus"]
	// Электрошок: cost 100 flat, cd 16 flat, rank-5 tooltip max 400.
	if pm.Skills[0].ManaCost[4] != 100 || pm.Skills[0].Cooldown[4] != 16 {
		t.Errorf("PlusMinus Электрошок rank5 cost/cd = %d/%d, want 100/16",
			pm.Skills[0].ManaCost[4], pm.Skills[0].Cooldown[4])
	}
	if pm.Skills[0].TipArgs["dmgMax"].At(5) != 400 {
		t.Errorf("PlusMinus Электрошок rank5 dmgMax = %g, want 400", pm.Skills[0].TipArgs["dmgMax"].At(5))
	}
	// Короткое замыкание: cost 100/110/120/130/140, cd 20 flat.
	if pm.Skills[1].ManaCost[4] != 140 || pm.Skills[1].Cooldown[0] != 20 {
		t.Errorf("PlusMinus Короткое замыкание cost5/cd1 = %d/%d, want 140/20",
			pm.Skills[1].ManaCost[4], pm.Skills[1].Cooldown[0])
	}
	// Сверхпроводимость: proc chance to 60% at rank 5.
	if pm.Skills[2].Ops[0].Chance.At(5) != 0.6 {
		t.Errorf("PlusMinus Сверхпроводимость rank5 chance = %g, want 0.6", pm.Skills[2].Ops[0].Chance.At(5))
	}
	// Шаровая молния (ult, 4 ranks): dmg 300..600, cost 200..320, cd 60.
	if pm.Skills[3].MaxRank() != 4 || pm.Skills[3].Ops[0].Value.At(4) != 600 || pm.Skills[3].Cooldown[0] != 60 {
		t.Errorf("PlusMinus ult rank4 dmg/cd = %g/%d (maxRank %d), want 600/60/4",
			pm.Skills[3].Ops[0].Value.At(4), pm.Skills[3].Cooldown[0], pm.Skills[3].MaxRank())
	}

	// Ariana Исцеление: 90/150/210/270/330, cost 100..180, cd 20.
	ar := skillsByPrefab["Avtr_Sp_Arianna"].Skills[2]
	if ar.Ops[0].Value.At(1) != 90 || ar.Ops[0].Value.At(5) != 330 || ar.ManaCost[4] != 180 || ar.Cooldown[0] != 20 {
		t.Errorf("Ariana Исцеление = %g..%g cost5 %d cd %d, want 90..330 / 180 / 20",
			ar.Ops[0].Value.At(1), ar.Ops[0].Value.At(5), ar.ManaCost[4], ar.Cooldown[0])
	}

	// Tangren Целительный тотем: 4/8/12/16/20 per tick.
	tg := skillsByPrefab["Avtr_HK_Tangren"].Skills[2]
	if tg.Ops[0].Value.At(1) != 4 || tg.Ops[0].Value.At(5) != 20 {
		t.Errorf("Tangren тотем = %g..%g, want 4..20", tg.Ops[0].Value.At(1), tg.Ops[0].Value.At(5))
	}

	// Velial Разлом (slot 2): cd 20 flat.
	ve := skillsByPrefab["Avtr_Tank_Velial"].Skills[1]
	if ve.Cooldown[0] != 20 || ve.Cooldown[4] != 20 {
		t.Errorf("Velial Разлом cd = %v, want flat 20", ve.Cooldown)
	}
}
