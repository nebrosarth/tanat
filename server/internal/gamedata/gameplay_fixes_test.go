package gamedata

import "testing"

// avatarByPrefabT looks up a roster avatar by prefab for the tests below.
func avatarByPrefabT(t *testing.T, prefab string) Avatar {
	t.Helper()
	for _, a := range Avatars() {
		if a.Prefab == prefab {
			return a
		}
	}
	t.Fatalf("avatar %s not in roster", prefab)
	return Avatar{}
}

// TestSandarielSkill3ShowsBuffIcon: «Острие странника» (passive) must carry BuffIcon so
// it renders a permanent status-effect icon like Velial's «Воля к победе» -- it did not,
// so the dmg buff applied invisibly ("3 скилл не отображается как статус эффект").
func TestSandarielSkill3ShowsBuffIcon(t *testing.T) {
	sk := SkillsFor(avatarByPrefabT(t, "Avtr_DPS_Sandariel")).Skills[2]
	if sk.Type != "PASSIVE" {
		t.Fatalf("Sandariel skill 3 should be PASSIVE, got %q", sk.Type)
	}
	if !sk.BuffIcon {
		t.Error("Sandariel skill 3 must set BuffIcon: true so it shows as a status effect")
	}
	// The working reference must not have regressed.
	if !SkillsFor(avatarByPrefabT(t, "Avtr_Tank_Velial")).Skills[2].BuffIcon {
		t.Error("Velial skill 3 buff icon regressed")
	}
}

// TestSandarielSkill2IsAOE: «Прыжок» must land an AoE (Radius>0 damage op + an AoERadius
// cursor), not a single-target/no-target leap ("2 скилл должен быть не таргетным, а AOE").
func TestSandarielSkill2IsAOE(t *testing.T) {
	sk := SkillsFor(avatarByPrefabT(t, "Avtr_DPS_Sandariel")).Skills[1]
	if sk.AoERadius <= 0 {
		t.Errorf("Sandariel skill 2 AoERadius = %d, want > 0 (AOE cursor)", sk.AoERadius)
	}
	aoeDmg := false
	for _, op := range sk.Ops {
		if op.Kind == OpDamage && op.Radius > 0 {
			aoeDmg = true
		}
	}
	if !aoeDmg {
		t.Error("Sandariel skill 2 must have an AoE damage op (OpDamage with Radius > 0)")
	}
}

// TestIsBossCoversBothLadders: IsBoss recognises the crypt AND jungle bosses (the latter
// ship no BossSkill list, so the battle server's len(Skills)>0 test would miss them) and
// rejects trash/creeps. Drives the boss spawn-leash.
func TestIsBossCoversBothLadders(t *testing.T) {
	for _, b := range []int{mobBossElgorm, mobBossVelial, mobBossCerber, mobBossHekata,
		mobBossGrimlok, mobBossTitanid, mobBossFairy, mobBossAnhel} {
		if !IsBoss(b) {
			t.Errorf("IsBoss(%d) = false, want true (boss)", b)
		}
	}
	for _, m := range []int{mobGhoul, mobSkeleton, mobDemon, mobHumanCreepMelee, mobSpider} {
		if IsBoss(m) {
			t.Errorf("IsBoss(%d) = true, want false (not a boss)", m)
		}
	}
}

// TestHuntLevelRampIsSmooth: mob levels must ramp gradually, not snap between the region
// anchor bands ("плавное нарастание, а не по 5 сразу"). Proof: some packs carry a level
// that is NOT one of the authored anchor bands -- i.e. interpolation filled the gaps.
func TestHuntLevelRampIsSmooth(t *testing.T) {
	check := func(name string, pack []MobSpawn, bands map[int]bool) {
		present := map[int]bool{}
		for _, sp := range pack {
			if sp.Level > 0 {
				present[sp.Level] = true
			}
		}
		intermediate := 0
		for l := range present {
			if !bands[l] {
				intermediate++
			}
		}
		if intermediate == 0 {
			t.Errorf("%s: no intermediate (non-band) levels -- the ramp still steps by whole bands", name)
		}
	}
	// Crypt anchor bands: {1,3,5,8,10,12,14,16,18,20}. Jungle: {1,4,5,7,8,9,10,12,13}.
	check("crypt map_4_0", dungeonPack40, map[int]bool{1: true, 3: true, 5: true, 8: true, 10: true, 12: true, 14: true, 16: true, 18: true, 20: true})
	check("jungle map_4_2", junglePack, map[int]bool{1: true, 4: true, 5: true, 7: true, 8: true, 9: true, 10: true, 12: true, 13: true})
}
