package gamedata

import "testing"

// TestAvatarXPCurve pins the 20-level progression to the original stats.txt
// anchors (the per-level COST at the listed transitions).
func TestAvatarXPCurve(t *testing.T) {
	if len(AvatarXPLevels) != 20 {
		t.Fatalf("AvatarXPLevels = %d entries, want 20 (levels 1..20)", len(AvatarXPLevels))
	}
	if AvatarXPLevels[0] != 0 {
		t.Errorf("level 1 threshold = %g, want 0", AvatarXPLevels[0])
	}
	// cost[d] = XP to go from displayed level d to d+1 = AvatarXPLevels[d]-AvatarXPLevels[d-1].
	cost := func(d int) float64 { return AvatarXPLevels[d] - AvatarXPLevels[d-1] }
	for _, tc := range []struct {
		from int
		want float64
	}{{1, 200}, {4, 500}, {8, 900}, {14, 1500}, {16, 1800}, {19, 2000}} {
		if got := cost(tc.from); got != tc.want {
			t.Errorf("cost %d->%d = %g, want %g (stats.txt)", tc.from, tc.from+1, got, tc.want)
		}
	}
	// Monotonic, strictly increasing.
	for i := 1; i < len(AvatarXPLevels); i++ {
		if AvatarXPLevels[i] <= AvatarXPLevels[i-1] {
			t.Errorf("XP curve not increasing at %d: %g <= %g", i, AvatarXPLevels[i], AvatarXPLevels[i-1])
		}
	}
}

// TestLevelScaling checks the per-level multipliers: exactly 1.0 at level 0
// (so level-1 combat is unchanged) and monotonic up to ~2.1x at level 20.
func TestLevelScaling(t *testing.T) {
	if LevelPowerMul(0) != 1.0 || LevelHealthMul(0) != 1.0 {
		t.Fatalf("level-0 multipliers must be 1.0, got power=%g hp=%g", LevelPowerMul(0), LevelHealthMul(0))
	}
	if LevelPowerMul(-3) != 1.0 {
		t.Errorf("negative level must clamp to 1.0")
	}
	if m := LevelPowerMul(19); m < 2.0 || m > 2.3 {
		t.Errorf("level-20 power mul = %g, want ~2.1", m)
	}
	for l := int32(1); l < 20; l++ {
		if LevelPowerMul(l) <= LevelPowerMul(l-1) {
			t.Errorf("power mul not increasing at level %d", l)
		}
	}
}

// TestVelialAvatarStats pins Velial's tank/bruiser stat line, rebalanced to the
// known real-avatar scale (top of the ~420-500 HP band, not the old 900 outlier).
func TestVelialAvatarStats(t *testing.T) {
	a, ok := AvatarByID(13)
	if !ok || a.Prefab != "Avtr_Tank_Velial" {
		t.Fatal("Velial (id 13) missing or wrong prefab")
	}
	if a.Type != AvatarTypeWarrior {
		t.Errorf("Velial type = %d, want WARRIOR", a.Type)
	}
	if a.Health != 520 || a.Mana != 200 || a.DmgMin != 42 || a.DmgMax != 48 {
		t.Errorf("Velial stats drifted: HP=%g Mana=%g Dmg=%d-%d, want 520/200/42-48", a.Health, a.Mana, a.DmgMin, a.DmgMax)
	}
	if a.AttackSpeed != 0.85 {
		t.Errorf("Velial AttackSpeed = %g, want 0.85 (heavy bruiser)", a.AttackSpeed)
	}
	// Velial must not tower over the known real avatars (Sigilion 500 is the tank
	// anchor); a small bruiser margin is fine, but nowhere near the old 900.
	if sig, _ := AvatarByID(10); a.Health <= sig.Health || a.Health > sig.Health+80 {
		t.Errorf("Velial HP %g should sit just above Sigilion's %g (top of the known band)", a.Health, sig.Health)
	}
	if a.PhysArmor != 6 || a.SpellPower != 0 {
		t.Errorf("Velial should be a physical bruiser: physArmor=%g spellPower=%g", a.PhysArmor, a.SpellPower)
	}
}

// TestStatsTxtMobValues pins the per-mob numbers base_balance/stats.txt states
// for the crypt starters (lines 79-83).
func TestStatsTxtMobValues(t *testing.T) {
	// "Скелет мечник" (base swordsman) = starter tier: 80 HP / 10 XP / 6 coins.
	sk := Mobs()[mobSkeleton]
	if sk.Health != 80 || sk.XP != 10 || sk.Coins != 6 {
		t.Errorf("Скелет мечник = %g HP / %g XP / %d coins, want 80/10/6", sk.Health, sk.XP, sk.Coins)
	}
	// "Зомби солдат": 150 HP and a HIGH attack rate (0.7s interval). AttackSpeed is
	// attacks/sec, so a 0.7s interval means ~1.43 (fast), NOT 0.7 (which is slow).
	zs := Mobs()[mobZombieSoldier]
	if zs.Health != 150 {
		t.Errorf("Зомби солдат HP = %g, want 150", zs.Health)
	}
	if zs.AttackSpeed < 1.4 {
		t.Errorf("Зомби солдат AttackSpeed = %g (interval %.2gs), want ~1.43 (0.7s interval, fast)",
			zs.AttackSpeed, 1/zs.AttackSpeed)
	}
	// "Зомби крушитель" bounty stays 8 bronze.
	if c := Mobs()[mobZombie].Coins; c != 8 {
		t.Errorf("Зомби крушитель coins = %d, want 8", c)
	}
}

// TestStatsTxtBossRewards pins the boss coin bounties from stats.txt (Elgorm 96,
// Velial 214, Hekata 374) and the interpolated Cerber (294, not in the sheet).
func TestStatsTxtBossRewards(t *testing.T) {
	for _, tc := range []struct {
		idx  int
		name string
		want int32
	}{
		{mobBossElgorm, "Elgorm", 96},
		{mobBossVelial, "Velial", 214},
		{mobBossCerber, "Cerber", 294},
		{mobBossHekata, "Hekata", 374},
	} {
		if got := Mobs()[tc.idx].Coins; got != tc.want {
			t.Errorf("%s coins = %d, want %d (stats.txt)", tc.name, got, tc.want)
		}
	}
}

// TestReleaseAvatarStats pins the "после релиза" per-avatar stat lines from
// stats.txt for the four heroes the user opted to override (the rest keep the
// class templates in statsFor).
func TestReleaseAvatarStats(t *testing.T) {
	for _, tc := range []struct {
		id                          int32
		name                        string
		hp, mana, hReg, mReg, aSpd  float64
		dmgMin, dmgMax, phys, magic int32
		checkArmor                  bool
	}{
		// Teridin: HP 495, regen 1/1, atk 40-46, mana 209, phys 5, magic 15.
		{17, "Teridin", 495, 209, 1, 1, 0, 40, 46, 5, 15, true},
		// Mihalych: 472 HP, 1/1 regen, 214 mana, 39 dmg, 1s interval (AtkSpd 1.0).
		{8, "Mihalych", 472, 214, 1, 1, 1.0, 39, 39, 0, 0, false},
		// Astarot: 423 HP, 1 HP/s, 184 mana, 36 dmg, 1.3s interval (AtkSpd 0.77).
		{7, "Astarot", 423, 184, 1, 0, 0.77, 36, 36, 0, 0, false},
		// Sigilion: 500 HP, 1/1 regen, 184 mana, 41 dmg.
		{10, "Sigilion", 500, 184, 1, 1, 0, 41, 41, 0, 0, false},
	} {
		a, ok := AvatarByID(tc.id)
		if !ok {
			t.Errorf("%s (id %d) not in roster", tc.name, tc.id)
			continue
		}
		if a.Health != tc.hp || a.Mana != tc.mana {
			t.Errorf("%s HP/Mana = %g/%g, want %g/%g", tc.name, a.Health, a.Mana, tc.hp, tc.mana)
		}
		if a.HealthRegen != tc.hReg {
			t.Errorf("%s HealthRegen = %g, want %g", tc.name, a.HealthRegen, tc.hReg)
		}
		if tc.mReg != 0 && a.ManaRegen != tc.mReg {
			t.Errorf("%s ManaRegen = %g, want %g", tc.name, a.ManaRegen, tc.mReg)
		}
		if tc.aSpd != 0 && a.AttackSpeed != tc.aSpd {
			t.Errorf("%s AttackSpeed = %g, want %g", tc.name, a.AttackSpeed, tc.aSpd)
		}
		if a.DmgMin != tc.dmgMin || a.DmgMax != tc.dmgMax {
			t.Errorf("%s dmg = %d-%d, want %d-%d", tc.name, a.DmgMin, a.DmgMax, tc.dmgMin, tc.dmgMax)
		}
		if tc.checkArmor && (a.PhysArmor != float64(tc.phys) || a.MagicArmor != float64(tc.magic)) {
			t.Errorf("%s armor = %g/%g, want %d/%d", tc.name, a.PhysArmor, a.MagicArmor, tc.phys, tc.magic)
		}
	}
}

// TestCryptMobRoster pins the Ghoul starter mob and the demon elite.
func TestCryptMobRoster(t *testing.T) {
	ghoul := Mobs()[mobGhoul]
	if ghoul.Prefab != "Mob_ZombieCrawl_01" { // = Elgorm's summon model
		t.Errorf("Ghoul prefab = %q, want Mob_ZombieCrawl_01 (Elgorm's summon)", ghoul.Prefab)
	}
	if ghoul.Health != 80 || ghoul.DmgMin != 8 || ghoul.DmgMax != 8 || ghoul.XP != 10 {
		t.Errorf("Ghoul = HP %g dmg %d-%d XP %g, want 80/8-8/10", ghoul.Health, ghoul.DmgMin, ghoul.DmgMax, ghoul.XP)
	}
	if Mobs()[mobDemon].Health <= Mobs()[mobSkeleton].Health {
		t.Errorf("demon must be tougher than skeleton")
	}
}

// TestBossLadder checks the four bosses form a strictly increasing HP/XP/coin
// ladder (Elgorm < Velial < Cerber < Hekata) with Hekata at the Titanid anchor.
func TestBossLadder(t *testing.T) {
	order := []int{mobBossElgorm, mobBossVelial, mobBossCerber, mobBossHekata}
	for i := 1; i < len(order); i++ {
		if Mobs()[order[i]].Health <= Mobs()[order[i-1]].Health {
			t.Errorf("boss HP ladder broken at %d: %g <= %g",
				i, Mobs()[order[i]].Health, Mobs()[order[i-1]].Health)
		}
		if Mobs()[order[i]].XP <= Mobs()[order[i-1]].XP {
			t.Errorf("boss XP ladder broken at %d", i)
		}
		if Mobs()[order[i]].Coins <= Mobs()[order[i-1]].Coins {
			t.Errorf("boss coin ladder broken at %d", i)
		}
	}
	if hp := Mobs()[mobBossHekata].Health; hp != 12000 {
		t.Errorf("Hekata (final boss) HP = %g, want 12000 (Titanid anchor)", hp)
	}
	// Bosses are exempt from mob level scaling: ScaledStats returns them verbatim.
	b := Mobs()[mobBossHekata]
	if hp, _, _, xp, coins := b.ScaledStats(20); hp != b.Health || xp != b.XP || coins != b.Coins {
		t.Errorf("boss should be unscaled: got hp %g xp %g coins %d", hp, xp, coins)
	}
}

// TestMobLevelScaling checks the mob per-level multipliers and the reward spec
// the user pinned (Ghoul & Skeleton-archer: 10 XP + 6 coins at level 1).
func TestMobLevelScaling(t *testing.T) {
	if MobHPMul(1) != 1 || MobDmgMul(1) != 1 || MobXPMul(1) != 1 || MobCoinMul(1) != 1 {
		t.Fatal("mob level 1 must be identity for every multiplier")
	}
	if MobHPMul(0) != 1 || MobXPMul(0) != 1 {
		t.Error("level < 1 must clamp to level 1")
	}
	if MobHPMul(10) <= MobHPMul(5) || MobDmgMul(20) <= MobDmgMul(10) || MobXPMul(20) <= MobXPMul(5) {
		t.Error("mob HP/damage/XP must scale up with level")
	}
	// XP scales GENTLY now (+40%/level, was a steep N x): mid-map trash isn't absurdly
	// rich but deep grinding still pays. Coins are gentler still (+15%/level).
	if got := MobXPMul(5); got < 2.5 || got > 2.7 {
		t.Errorf("MobXPMul(5) = %g, want ~2.6 (gentle, not the old 5x)", got)
	}
	if MobCoinMul(10) >= MobXPMul(10) {
		t.Errorf("coin scaling (%g) must be gentler than XP scaling (%g)", MobCoinMul(10), MobXPMul(10))
	}

	ghoul, archer := Mobs()[mobGhoul], Mobs()[mobSkeletonArcher]
	if ghoul.XP != 10 || ghoul.Coins != 6 {
		t.Errorf("ghoul reward = %g XP / %d coins, want 10/6", ghoul.XP, ghoul.Coins)
	}
	if archer.XP != 10 || archer.Coins != 6 {
		t.Errorf("archer reward = %g XP / %d coins, want 10/6", archer.XP, archer.Coins)
	}
	if archer.Health != 40 || archer.Mana != 150 {
		t.Errorf("skeleton archer L1 = %g HP / %g mana, want 40/150", archer.Health, archer.Mana)
	}
	// The level-1 STARTER ghoul is the user-pinned 80 HP / 6 bronze / 10 XP (identity).
	if hp, _, _, xp, coins := ghoul.ScaledStats(1); hp != 80 || coins != 6 || xp != 10 {
		t.Errorf("L1 ghoul: hp %g xp %g coins %d, want 80/10/6", hp, xp, coins)
	}
	// A reinforced (level-2, 92 HP) ghoul yields 7 bronze -- not the old steep 12.
	if hp, _, _, _, coins := ghoul.ScaledStats(2); hp != 92 || coins != 7 {
		t.Errorf("L2 ghoul: %g HP / %d coins, want 92/7", hp, coins)
	}
	// XP is gentle now: a level-10 ghoul gives ~46 XP (was 100 under the N x curve)
	// and only ~14 coins.
	if hp, _, _, xp, coins := ghoul.ScaledStats(10); xp < 45 || xp > 47 || coins != 14 || hp <= ghoul.Health {
		t.Errorf("L10 ghoul: hp %g xp %g coins %d, want xp ~46 coins 14 hp>%g", hp, xp, coins, ghoul.Health)
	}
	// The XP-inflation the user flagged is gone: a base-16 skeleton (sniper) at its
	// level-5 placement gives ~42 XP, not the old 80.
	if _, _, _, xp, _ := Mobs()[mobSkeletonSniper].ScaledStats(5); xp < 38 || xp > 45 {
		t.Errorf("L5 sniper XP = %g, want ~42 (was 80)", xp)
	}
}

// TestMobLevelGradient checks mob level rises along the route (start low, deep high).
func TestMobLevelGradient(t *testing.T) {
	var maxLvl, ghoulMax int
	for _, sp := range dungeonPack40 {
		if sp.Level > maxLvl {
			maxLvl = sp.Level
		}
		if sp.Mob == mobGhoul && sp.Level > ghoulMax {
			ghoulMax = sp.Level
		}
	}
	if ghoulMax > 2 {
		t.Errorf("starter ghouls should stay low level, got max %d", ghoulMax)
	}
	if maxLvl < 18 {
		t.Errorf("deepest mobs should approach level 20, got max %d", maxLvl)
	}
}

// isDemonMob reports whether idx is any demon-family index (common or elite,
// melee or ranged) -- used wherever a check must count the WHOLE demon family,
// not just the original two indices.
func isDemonMob(idx int) bool {
	for _, d := range demonFamily {
		if idx == d {
			return true
		}
	}
	return false
}

// TestDungeonPlacement checks the map_4_0 layout: the pathfind anchor stays an
// offset, demons guard Velial's lair and the Cerber->Hekata connector, and ghouls
// open the dungeon.
func TestDungeonPlacement(t *testing.T) {
	if dungeonPack40[0].Abs || dungeonPack40[0].Mob != mobGhoul {
		t.Errorf("dungeonPack40[0] must be an offset ghoul (pathfind-test anchor)")
	}
	var ghouls, velialDemons, connectorDemons int
	for _, sp := range dungeonPack40 {
		if sp.Mob == mobGhoul {
			ghouls++
		}
		if !isDemonMob(sp.Mob) || !sp.Abs {
			continue
		}
		// Velial's lair chamber. The boss now keeps a wide (26m) mob-free ring like a
		// respawn point, so demons form a RING around the lair rather than hugging the
		// boss -- the box spans the whole chamber (X 455..520, Z 275..345), not the
		// centre. Velial sits at ~(486,306); nothing spawns within 26m of him.
		if sp.DX >= 455 && sp.DX <= 520 && sp.DY >= 275 && sp.DY <= 345 {
			velialDemons++
		}
		// Cerber->Hekata connector (the mid-left walkable region).
		if sp.DX >= 110 && sp.DX <= 200 && sp.DY >= 240 && sp.DY <= 320 {
			connectorDemons++
		}
	}
	if ghouls < 4 {
		t.Errorf("expected several starter ghouls, got %d", ghouls)
	}
	if velialDemons < 3 {
		t.Errorf("expected a demon pack at Velial's lair, got %d", velialDemons)
	}
	if connectorDemons < 3 {
		t.Errorf("expected demons in the Cerber->Hekata connector, got %d", connectorDemons)
	}
}

// TestFullBestiaryPlaced is the guard for "use every skeleton/ghoul/zombie/demon
// variant": every mob index from mobGhoul through mobDemonRangeElite must appear
// at least once in dungeonPack40, so no shipped-and-named creature sits unused.
func TestFullBestiaryPlaced(t *testing.T) {
	bestiary := []struct {
		idx  int
		name string
	}{
		{mobGhoul, "Голодный гуль"}, {mobGhoulPossessed, "Одержимый гуль"},
		{mobSkeleton, "Скелет мечник"}, {mobSkeletonHewer, "Скелет рубака"},
		{mobSkeletonWarrior, "Скелет воитель"}, {mobSkeletonBerserk, "Скелет берсерк"},
		{mobSkeletonArcher, "Скелет лучник"}, {mobSkeletonSniper, "Скелет снайпер"},
		{mobSkeletonBurning, "Горящий скелет"},
		{mobZombie, "Зомби крушитель"}, {mobZombieBigElite, "Зомби губитель"},
		{mobZombieSoldier, "Зомби солдат"}, {mobZombieSoldierElite, "Зомби ратоборец"},
		{mobDemon, "Демон страж"}, {mobDemonMeleeElite, "Демон захватчик"},
		{mobDemonRange, "Демон воитель"}, {mobDemonRangeElite, "Демон надзиратель"},
	}
	seen := map[int]int{}
	for _, sp := range dungeonPack40 {
		seen[sp.Mob]++
	}
	for _, b := range bestiary {
		if seen[b.idx] == 0 {
			t.Errorf("%s (index %d) is never placed on map_4_0", b.name, b.idx)
		}
	}
}

// TestBestiaryLocaleAndTiering checks every new family member resolves to its
// real shipped locale key (matching the game's own naming convention: base name,
// then "_g" for the elite tier) and that elites outclass their common counterpart.
func TestBestiaryLocaleAndTiering(t *testing.T) {
	type pair struct {
		common, elite       int
		commonKey, eliteKey string
	}
	for _, p := range []pair{
		{mobSkeleton, mobSkeletonWarrior, "IDS_Mob_Skeleton1H_Melee_Name", "IDS_Mob_Skeleton1H_Melee_g_Name"},
		{mobSkeletonHewer, mobSkeletonBerserk, "IDS_Mob_Skeleton1HSh_Melee_Name", "IDS_Mob_Skeleton1HSh_Melee_g_Name"},
		{mobSkeletonArcher, mobSkeletonSniper, "IDS_Mob_Skeleton_Range_Name", "IDS_Mob_Skeleton_Range_g_Name"},
		{mobGhoul, mobGhoulPossessed, "IDS_Mob_ZombieCrawl_Name", "IDS_Mob_ZombieCrawl_g_Name"},
		{mobZombie, mobZombieBigElite, "IDS_Mob_ZombieBig_Name", "IDS_Mob_ZombieBig_g_Name"},
		{mobZombieSoldier, mobZombieSoldierElite, "IDS_Mob_ZombieSword_Name", "IDS_Mob_ZombieSword_g_Name"},
		{mobDemon, mobDemonMeleeElite, "IDS_Mob_Demon_Melee_Name", "IDS_Mob_Demon_Melee_g_Name"},
		{mobDemonRange, mobDemonRangeElite, "IDS_Mob_Demon_Range_Name", "IDS_Mob_Demon_Range_g_Name"},
	} {
		c, e := Mobs()[p.common], Mobs()[p.elite]
		if c.NameKey != p.commonKey {
			t.Errorf("common mob %d NameKey = %q, want %q", p.common, c.NameKey, p.commonKey)
		}
		if e.NameKey != p.eliteKey {
			t.Errorf("elite mob %d NameKey = %q, want %q", p.elite, e.NameKey, p.eliteKey)
		}
		if e.Health <= c.Health {
			t.Errorf("%s: elite HP %g not tougher than common %g", p.eliteKey, e.Health, c.Health)
		}
	}
}

// TestBossBeatabilityLadder is the design guard: a simplified Velial DPS model
// shows each boss is beatable in a sane window (~40-90s) at its INTENDED level
// (Elgorm 5, Velial 10, Cerber 15, Hekata 20), and that the same boss is a hard
// slog if you show up badly under-levelled -- i.e. leveling actually matters.
func TestBossBeatabilityLadder(t *testing.T) {
	v, _ := AvatarByID(13) // Velial avatar
	kit := SkillsFor(v)
	// Velial DPS at a 0-based level: scaled basic + the two damage skills at rank 4.
	dps := func(level int32) float64 {
		mul := LevelPowerMul(level)
		basic := (float64(v.DmgMin) + float64(v.DmgMax)) / 2 * v.AttackSpeed * mul
		s1 := kit.Skills[0].Ops[0].Value.At(4) * mul / float64(kit.Skills[0].Cooldown[3]) // Удар изверга
		s2 := kit.Skills[1].Ops[0].Value.At(4) * mul / float64(kit.Skills[1].Cooldown[3]) // Разлом
		return basic + s1 + s2
	}
	type enc struct {
		boss    int
		dispLvl int32 // intended displayed level
	}
	for _, e := range []enc{
		{mobBossElgorm, 5}, {mobBossVelial, 10}, {mobBossCerber, 15}, {mobBossHekata, 20},
	} {
		hp := Mobs()[e.boss].Health
		ttk := hp / dps(e.dispLvl-1) // 0-based level = displayed-1
		if ttk < 40 || ttk > 95 {
			t.Errorf("boss %d at level %d: TTK %.0fs out of the 40-95s band (hp %g dps %.1f)",
				e.boss, e.dispLvl, ttk, hp, dps(e.dispLvl-1))
		}
		// Leveling never makes a fight slower (monotonic power curve).
		if slow := hp / dps(0); slow < ttk {
			t.Errorf("boss %d: level-1 TTK %.0fs shorter than level-%d TTK %.0fs (curve inverted)",
				e.boss, slow, e.dispLvl, ttk)
		}
	}
	// The final boss must be a hard slog for a fresh level-1 hero -- that gap is the
	// whole point of the level gate (survivability aside, DPS alone shows it).
	hekataHP := Mobs()[mobBossHekata].Health
	if slow, fast := hekataHP/dps(0), hekataHP/dps(19); slow < fast*1.7 {
		t.Errorf("Hekata level-1 TTK %.0fs not far worse than level-20 TTK %.0fs", slow, fast)
	}
}
