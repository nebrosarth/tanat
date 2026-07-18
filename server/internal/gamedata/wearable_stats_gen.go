// Code generated from the official wiki «Геройские комплекты»
// (tanat/_decompiled/wiki_archive/hero_sets_gerojskie_komplekty.txt). DO NOT EDIT by hand:
// these are the exact per-item stat VALUES the site published for each Set piece, keyed by
// (Color, Tier, Slot, Stat). Race-agnostic -- the wiki gives one stat block per set and both
// races share it (Elf/Human wearables of the same tier+color+slot get identical stats).
//
// Units match the client's baked LongDesc placeholders (verified against resources.assets):
// every stat renders RAW except {CritChance%}, which the client multiplies by 100 -- so
// CritChance is stored here as a FRACTION (wiki %% / 100), matching the battle engine's
// crit_pct probability; all other stats are stored exactly as the wiki prints them.
//
// Not covered here (fall back to the synthetic wearableStatValue): weapon pieces (the wiki
// lists no weapons), the Violet S5 «Каратель» set (wiki: «Скоро будет добавлен»), and the
// Grey S1 belt (the wiki's Воитель belt lists HealthRegen+PhysArmor, but that item's baked
// placeholders are Mana+MagicArmor like every other belt, so the wiki row has nowhere to go).

package gamedata

// wikiStatKey identifies one authored stat: a Set piece's (Color, Tier, Slot) and the stat name.
type wikiStatKey struct {
	Color string
	Tier  int32
	Slot  string
	Stat  string
}

// wikiWearableStats holds the 391 authored Set-piece stat values from the wiki.
var wikiWearableStats = map[wikiStatKey]float64{
	// Grey S0 Body
	{"Grey", 0, "Body", "HealthRegen"}: 0.04,
	{"Grey", 0, "Body", "PhysArmor"}: 0.5,
	// Grey S0 Shoulders
	{"Grey", 0, "Shoulders", "Health"}: 19,
	{"Grey", 0, "Shoulders", "MagicArmor"}: 0.3,
	// Grey S0 Hands
	{"Grey", 0, "Hands", "Health"}: 10,
	{"Grey", 0, "Hands", "AttackSpeed"}: 0.01,
	{"Grey", 0, "Hands", "CritChance"}: 0.004,
	// Grey S0 Belt
	{"Grey", 0, "Belt", "HealthRegen"}: 0.04,
	{"Grey", 0, "Belt", "PhysArmor"}: 0.5,
	// Grey S0 Legs
	{"Grey", 0, "Legs", "Mana"}: 4,
	{"Grey", 0, "Legs", "ManaRegen"}: 0.04,
	{"Grey", 0, "Legs", "AntiPhysArmor"}: 1,
	// Grey S0 Foots
	{"Grey", 0, "Foots", "Health"}: 10,
	{"Grey", 0, "Foots", "HealthRegen"}: 0.04,
	{"Grey", 0, "Foots", "PhysArmor"}: 0.5,
	// Grey S1 Helm
	{"Grey", 1, "Helm", "Mana"}: 9,
	{"Grey", 1, "Helm", "ManaRegen"}: 0.04,
	{"Grey", 1, "Helm", "AntiMagicArmor"}: 0.008,
	// Grey S1 Body
	{"Grey", 1, "Body", "HealthRegen"}: 0.04,
	{"Grey", 1, "Body", "PhysArmor"}: 0.6,
	// Grey S1 Shoulders
	{"Grey", 1, "Shoulders", "Health"}: 23,
	{"Grey", 1, "Shoulders", "MagicArmor"}: 0.5,
	// Grey S1 Hands
	{"Grey", 1, "Hands", "Health"}: 12,
	{"Grey", 1, "Hands", "AttackSpeed"}: 0.01,
	{"Grey", 1, "Hands", "CritChance"}: 0.005,
	// Grey S1 Legs
	{"Grey", 1, "Legs", "Mana"}: 5,
	{"Grey", 1, "Legs", "ManaRegen"}: 0.04,
	{"Grey", 1, "Legs", "AntiPhysArmor"}: 1.6,
	// Grey S1 Foots
	{"Grey", 1, "Foots", "Health"}: 12,
	{"Grey", 1, "Foots", "HealthRegen"}: 0.04,
	{"Grey", 1, "Foots", "PhysArmor"}: 0.6,
	// Grey S2 Helm
	{"Grey", 2, "Helm", "Mana"}: 17,
	{"Grey", 2, "Helm", "ManaRegen"}: 0.09,
	{"Grey", 2, "Helm", "AntiMagicArmor"}: 1.5,
	// Grey S2 Body
	{"Grey", 2, "Body", "HealthRegen"}: 0.09,
	{"Grey", 2, "Body", "PhysArmor"}: 1.1,
	// Grey S2 Shoulders
	{"Grey", 2, "Shoulders", "Health"}: 42,
	{"Grey", 2, "Shoulders", "MagicArmor"}: 0.8,
	// Grey S2 Hands
	{"Grey", 2, "Hands", "Health"}: 21,
	{"Grey", 2, "Hands", "AttackSpeed"}: 0.03,
	{"Grey", 2, "Hands", "CritChance"}: 0.009,
	// Grey S2 Belt
	{"Grey", 2, "Belt", "Mana"}: 9,
	{"Grey", 2, "Belt", "MagicArmor"}: 0.8,
	// Grey S2 Legs
	{"Grey", 2, "Legs", "Mana"}: 9,
	{"Grey", 2, "Legs", "ManaRegen"}: 0.09,
	{"Grey", 2, "Legs", "AntiPhysArmor"}: 3,
	// Grey S2 Foots
	{"Grey", 2, "Foots", "Health"}: 21,
	{"Grey", 2, "Foots", "HealthRegen"}: 0.09,
	{"Grey", 2, "Foots", "PhysArmor"}: 1.1,
	// Grey S3 Helm
	{"Grey", 3, "Helm", "Mana"}: 18,
	{"Grey", 3, "Helm", "ManaRegen"}: 0.1,
	{"Grey", 3, "Helm", "AntiMagicArmor"}: 1.8,
	// Grey S3 Body
	{"Grey", 3, "Body", "HealthRegen"}: 0.1,
	{"Grey", 3, "Body", "PhysArmor"}: 1.4,
	// Grey S3 Shoulders
	{"Grey", 3, "Shoulders", "Health"}: 45,
	{"Grey", 3, "Shoulders", "MagicArmor"}: 1,
	// Grey S3 Hands
	{"Grey", 3, "Hands", "Health"}: 23,
	{"Grey", 3, "Hands", "AttackSpeed"}: 0.03,
	{"Grey", 3, "Hands", "CritChance"}: 0.011,
	// Grey S3 Belt
	{"Grey", 3, "Belt", "Mana"}: 9,
	{"Grey", 3, "Belt", "MagicArmor"}: 1,
	// Grey S3 Legs
	{"Grey", 3, "Legs", "Mana"}: 9,
	{"Grey", 3, "Legs", "ManaRegen"}: 0.1,
	{"Grey", 3, "Legs", "AntiPhysArmor"}: 3.6,
	// Grey S3 Foots
	{"Grey", 3, "Foots", "Health"}: 23,
	{"Grey", 3, "Foots", "HealthRegen"}: 0.1,
	{"Grey", 3, "Foots", "PhysArmor"}: 1.3,
	// Grey S4 Helm
	{"Grey", 4, "Helm", "Mana"}: 26,
	{"Grey", 4, "Helm", "ManaRegen"}: 0.13,
	{"Grey", 4, "Helm", "AntiMagicArmor"}: 2.8,
	// Grey S4 Body
	{"Grey", 4, "Body", "HealthRegen"}: 0.13,
	{"Grey", 4, "Body", "PhysArmor"}: 2.1,
	// Grey S4 Shoulders
	{"Grey", 4, "Shoulders", "Health"}: 64,
	{"Grey", 4, "Shoulders", "MagicArmor"}: 1.6,
	// Grey S4 Hands
	{"Grey", 4, "Hands", "Health"}: 32,
	{"Grey", 4, "Hands", "AttackSpeed"}: 0.05,
	{"Grey", 4, "Hands", "CritChance"}: 0.02,
	// Grey S4 Belt
	{"Grey", 4, "Belt", "Mana"}: 13,
	{"Grey", 4, "Belt", "MagicArmor"}: 1.6,
	// Grey S4 Legs
	{"Grey", 4, "Legs", "Mana"}: 13,
	{"Grey", 4, "Legs", "ManaRegen"}: 0.13,
	{"Grey", 4, "Legs", "AntiPhysArmor"}: 5.6,
	// Grey S4 Foots
	{"Grey", 4, "Foots", "Health"}: 32,
	{"Grey", 4, "Foots", "HealthRegen"}: 0.13,
	{"Grey", 4, "Foots", "PhysArmor"}: 2.1,
	// Grey S5 Helm
	{"Grey", 5, "Helm", "Mana"}: 34,
	{"Grey", 5, "Helm", "ManaRegen"}: 0.17,
	{"Grey", 5, "Helm", "AntiMagicArmor"}: 4,
	// Grey S5 Body
	{"Grey", 5, "Body", "HealthRegen"}: 0.17,
	{"Grey", 5, "Body", "PhysArmor"}: 3,
	// Grey S5 Shoulders
	{"Grey", 5, "Shoulders", "Health"}: 84,
	{"Grey", 5, "Shoulders", "MagicArmor"}: 2.3,
	// Grey S5 Hands
	{"Grey", 5, "Hands", "Health"}: 42,
	{"Grey", 5, "Hands", "AttackSpeed"}: 0.072,
	{"Grey", 5, "Hands", "CritChance"}: 0.03,
	// Grey S5 Belt
	{"Grey", 5, "Belt", "Mana"}: 17,
	{"Grey", 5, "Belt", "MagicArmor"}: 2.3,
	// Grey S5 Legs
	{"Grey", 5, "Legs", "Mana"}: 17,
	{"Grey", 5, "Legs", "ManaRegen"}: 0.17,
	{"Grey", 5, "Legs", "AntiPhysArmor"}: 8,
	// Grey S5 Foots
	{"Grey", 5, "Foots", "Health"}: 42,
	{"Grey", 5, "Foots", "HealthRegen"}: 0.17,
	{"Grey", 5, "Foots", "PhysArmor"}: 3,
	// Grey S6 Helm
	{"Grey", 6, "Helm", "Mana"}: 41,
	{"Grey", 6, "Helm", "ManaRegen"}: 0.21,
	{"Grey", 6, "Helm", "AntiMagicArmor"}: 5.4,
	// Grey S6 Body
	{"Grey", 6, "Body", "HealthRegen"}: 0.21,
	{"Grey", 6, "Body", "PhysArmor"}: 4.1,
	// Grey S6 Shoulders
	{"Grey", 6, "Shoulders", "Health"}: 104,
	{"Grey", 6, "Shoulders", "MagicArmor"}: 3,
	// Grey S6 Hands
	{"Grey", 6, "Hands", "Health"}: 52,
	{"Grey", 6, "Hands", "AttackSpeed"}: 0.097,
	{"Grey", 6, "Hands", "CritChance"}: 0.03,
	// Grey S6 Belt
	{"Grey", 6, "Belt", "Mana"}: 21,
	{"Grey", 6, "Belt", "MagicArmor"}: 3,
	// Grey S6 Legs
	{"Grey", 6, "Legs", "Mana"}: 21,
	{"Grey", 6, "Legs", "ManaRegen"}: 0.21,
	{"Grey", 6, "Legs", "AntiPhysArmor"}: 10.8,
	// Grey S6 Foots
	{"Grey", 6, "Foots", "Health"}: 52,
	{"Grey", 6, "Foots", "HealthRegen"}: 0.21,
	{"Grey", 6, "Foots", "PhysArmor"}: 4.1,
	// Grey S7 Helm
	{"Grey", 7, "Helm", "Mana"}: 54,
	{"Grey", 7, "Helm", "ManaRegen"}: 0.27,
	{"Grey", 7, "Helm", "AntiMagicArmor"}: 7,
	// Grey S7 Body
	{"Grey", 7, "Body", "HealthRegen"}: 0.27,
	{"Grey", 7, "Body", "PhysArmor"}: 5.3,
	// Grey S7 Shoulders
	{"Grey", 7, "Shoulders", "Health"}: 135,
	{"Grey", 7, "Shoulders", "MagicArmor"}: 3.9,
	// Grey S7 Hands
	{"Grey", 7, "Hands", "Health"}: 67,
	{"Grey", 7, "Hands", "AttackSpeed"}: 0.126,
	{"Grey", 7, "Hands", "CritChance"}: 0.04,
	// Grey S7 Belt
	{"Grey", 7, "Belt", "Mana"}: 27,
	{"Grey", 7, "Belt", "MagicArmor"}: 3.9,
	// Grey S7 Legs
	{"Grey", 7, "Legs", "Mana"}: 27,
	{"Grey", 7, "Legs", "ManaRegen"}: 0.27,
	{"Grey", 7, "Legs", "AntiPhysArmor"}: 1.4,
	// Grey S7 Foots
	{"Grey", 7, "Foots", "Health"}: 67,
	{"Grey", 7, "Foots", "HealthRegen"}: 0.27,
	{"Grey", 7, "Foots", "PhysArmor"}: 5.3,
	// Green S1 Helm
	{"Green", 1, "Helm", "Mana"}: 18,
	{"Green", 1, "Helm", "ManaRegen"}: 0.09,
	{"Green", 1, "Helm", "AntiMagicArmor"}: 1.6,
	// Green S1 Body
	{"Green", 1, "Body", "HealthRegen"}: 0.09,
	{"Green", 1, "Body", "PhysArmor"}: 1.2,
	// Green S1 Shoulders
	{"Green", 1, "Shoulders", "Health"}: 45,
	{"Green", 1, "Shoulders", "MagicArmor"}: 0.9,
	// Green S1 Hands
	{"Green", 1, "Hands", "Health"}: 23,
	{"Green", 1, "Hands", "AttackSpeed"}: 0.03,
	{"Green", 1, "Hands", "CritChance"}: 0.01,
	// Green S1 Belt
	{"Green", 1, "Belt", "Mana"}: 9,
	{"Green", 1, "Belt", "MagicArmor"}: 0.9,
	// Green S1 Legs
	{"Green", 1, "Legs", "Mana"}: 9,
	{"Green", 1, "Legs", "ManaRegen"}: 0.09,
	{"Green", 1, "Legs", "AntiPhysArmor"}: 3,
	// Green S1 Foots
	{"Green", 1, "Foots", "Health"}: 23,
	{"Green", 1, "Foots", "HealthRegen"}: 0.09,
	{"Green", 1, "Foots", "PhysArmor"}: 1.2,
	// Green S2 Helm
	{"Green", 2, "Helm", "Mana"}: 34,
	{"Green", 2, "Helm", "ManaRegen"}: 0.18,
	{"Green", 2, "Helm", "AntiMagicArmor"}: 3,
	// Green S2 Body
	{"Green", 2, "Body", "HealthRegen"}: 0.18,
	{"Green", 2, "Body", "PhysArmor"}: 2.2,
	// Green S2 Shoulders
	{"Green", 2, "Shoulders", "Health"}: 84,
	{"Green", 2, "Shoulders", "MagicArmor"}: 1.6,
	// Green S2 Hands
	{"Green", 2, "Hands", "Health"}: 42,
	{"Green", 2, "Hands", "AttackSpeed"}: 0.05,
	{"Green", 2, "Hands", "CritChance"}: 0.018,
	// Green S2 Belt
	{"Green", 2, "Belt", "Mana"}: 17,
	{"Green", 2, "Belt", "MagicArmor"}: 1.6,
	// Green S2 Legs
	{"Green", 2, "Legs", "Mana"}: 17,
	{"Green", 2, "Legs", "ManaRegen"}: 0.18,
	{"Green", 2, "Legs", "AntiPhysArmor"}: 6,
	// Green S2 Foots
	{"Green", 2, "Foots", "Health"}: 42,
	{"Green", 2, "Foots", "HealthRegen"}: 0.18,
	{"Green", 2, "Foots", "PhysArmor"}: 2.2,
	// Green S3 Helm
	{"Green", 3, "Helm", "Mana"}: 36,
	{"Green", 3, "Helm", "ManaRegen"}: 0.18,
	{"Green", 3, "Helm", "AntiMagicArmor"}: 3.6,
	// Green S3 Body
	{"Green", 3, "Body", "HealthRegen"}: 0.18,
	{"Green", 3, "Body", "PhysArmor"}: 2.7,
	// Green S3 Shoulders
	{"Green", 3, "Shoulders", "Health"}: 90,
	{"Green", 3, "Shoulders", "MagicArmor"}: 2,
	// Green S3 Hands
	{"Green", 3, "Hands", "Health"}: 45,
	{"Green", 3, "Hands", "AttackSpeed"}: 0.06,
	{"Green", 3, "Hands", "CritChance"}: 0.022,
	// Green S3 Belt
	{"Green", 3, "Belt", "Mana"}: 18,
	{"Green", 3, "Belt", "MagicArmor"}: 2,
	// Green S3 Legs
	{"Green", 3, "Legs", "Mana"}: 18,
	{"Green", 3, "Legs", "ManaRegen"}: 0.18,
	{"Green", 3, "Legs", "AntiPhysArmor"}: 7.2,
	// Green S3 Foots
	{"Green", 3, "Foots", "Health"}: 45,
	{"Green", 3, "Foots", "HealthRegen"}: 0.18,
	{"Green", 3, "Foots", "PhysArmor"}: 2.7,
	// Green S4 Helm
	{"Green", 4, "Helm", "Mana"}: 36,
	{"Green", 4, "Helm", "ManaRegen"}: 0.18,
	{"Green", 4, "Helm", "AntiMagicArmor"}: 3.6,
	// Green S4 Body
	{"Green", 4, "Body", "HealthRegen"}: 0.22,
	{"Green", 4, "Body", "PhysArmor"}: 3.7,
	// Green S4 Shoulders
	{"Green", 4, "Shoulders", "Health"}: 112,
	{"Green", 4, "Shoulders", "MagicArmor"}: 2.8,
	// Green S4 Hands
	{"Green", 4, "Hands", "Health"}: 56,
	{"Green", 4, "Hands", "AttackSpeed"}: 0.088,
	{"Green", 4, "Hands", "CritChance"}: 0.03,
	// Green S4 Belt
	{"Green", 4, "Belt", "Mana"}: 27,
	{"Green", 4, "Belt", "MagicArmor"}: 2.8,
	// Green S4 Legs
	{"Green", 4, "Legs", "Mana"}: 22,
	{"Green", 4, "Legs", "ManaRegen"}: 0.22,
	{"Green", 4, "Legs", "AntiPhysArmor"}: 9.8,
	// Green S4 Foots
	{"Green", 4, "Foots", "Health"}: 56,
	{"Green", 4, "Foots", "HealthRegen"}: 0.22,
	{"Green", 4, "Foots", "PhysArmor"}: 3.7,
	// Green S5 Helm
	{"Green", 5, "Helm", "Mana"}: 54,
	{"Green", 5, "Helm", "ManaRegen"}: 0.27,
	{"Green", 5, "Helm", "AntiMagicArmor"}: 6.4,
	// Green S5 Body
	{"Green", 5, "Body", "HealthRegen"}: 0.27,
	{"Green", 5, "Body", "PhysArmor"}: 4.8,
	// Green S5 Shoulders
	{"Green", 5, "Shoulders", "Health"}: 134,
	{"Green", 5, "Shoulders", "MagicArmor"}: 3.6,
	// Green S5 Hands
	{"Green", 5, "Hands", "Health"}: 67,
	{"Green", 5, "Hands", "AttackSpeed"}: 0.115,
	{"Green", 5, "Hands", "CritChance"}: 0.04,
	// Green S5 Belt
	{"Green", 5, "Belt", "Mana"}: 27,
	{"Green", 5, "Belt", "MagicArmor"}: 3.6,
	// Green S5 Legs
	{"Green", 5, "Legs", "Mana"}: 27,
	{"Green", 5, "Legs", "ManaRegen"}: 0.27,
	{"Green", 5, "Legs", "AntiPhysArmor"}: 12.4,
	// Green S5 Foots
	{"Green", 5, "Foots", "Health"}: 67,
	{"Green", 5, "Foots", "HealthRegen"}: 0.27,
	{"Green", 5, "Foots", "PhysArmor"}: 4.8,
	// Green S6 Helm
	{"Green", 6, "Helm", "Mana"}: 59,
	{"Green", 6, "Helm", "ManaRegen"}: 0.3,
	{"Green", 6, "Helm", "AntiMagicArmor"}: 7.7,
	// Green S6 Body
	{"Green", 6, "Body", "HealthRegen"}: 0.3,
	{"Green", 6, "Body", "PhysArmor"}: 5.8,
	// Green S6 Shoulders
	{"Green", 6, "Shoulders", "Health"}: 148,
	{"Green", 6, "Shoulders", "MagicArmor"}: 4.4,
	// Green S6 Hands
	{"Green", 6, "Hands", "Health"}: 74,
	{"Green", 6, "Hands", "AttackSpeed"}: 0.139,
	{"Green", 6, "Hands", "CritChance"}: 0.05,
	// Green S6 Belt
	{"Green", 6, "Belt", "Mana"}: 30,
	{"Green", 6, "Belt", "MagicArmor"}: 4.4,
	// Green S6 Legs
	{"Green", 6, "Legs", "Mana"}: 30,
	{"Green", 6, "Legs", "ManaRegen"}: 0.3,
	{"Green", 6, "Legs", "AntiPhysArmor"}: 15.5,
	// Green S6 Foots
	{"Green", 6, "Foots", "Health"}: 74,
	{"Green", 6, "Foots", "HealthRegen"}: 0.3,
	{"Green", 6, "Foots", "PhysArmor"}: 5.8,
	// Green S7 Helm
	{"Green", 7, "Helm", "Mana"}: 69,
	{"Green", 7, "Helm", "ManaRegen"}: 0.35,
	{"Green", 7, "Helm", "AntiMagicArmor"}: 9,
	// Green S7 Body
	{"Green", 7, "Body", "HealthRegen"}: 0.35,
	{"Green", 7, "Body", "PhysArmor"}: 6.8,
	// Green S7 Shoulders
	{"Green", 7, "Shoulders", "Health"}: 173,
	{"Green", 7, "Shoulders", "MagicArmor"}: 5.1,
	// Green S7 Hands
	{"Green", 7, "Hands", "Health"}: 87,
	{"Green", 7, "Hands", "AttackSpeed"}: 0.162,
	{"Green", 7, "Hands", "CritChance"}: 0.06,
	// Green S7 Belt
	{"Green", 7, "Belt", "Mana"}: 35,
	{"Green", 7, "Belt", "MagicArmor"}: 5.1,
	// Green S7 Legs
	{"Green", 7, "Legs", "Mana"}: 35,
	{"Green", 7, "Legs", "ManaRegen"}: 0.35,
	{"Green", 7, "Legs", "AntiPhysArmor"}: 1.8,
	// Green S7 Foots
	{"Green", 7, "Foots", "Health"}: 87,
	{"Green", 7, "Foots", "HealthRegen"}: 0.35,
	{"Green", 7, "Foots", "PhysArmor"}: 6.8,
	// Blue S3 Helm
	{"Blue", 3, "Helm", "Mana"}: 42,
	{"Blue", 3, "Helm", "ManaRegen"}: 0.21,
	{"Blue", 3, "Helm", "AntiMagicArmor"}: 4.2,
	// Blue S3 Body
	{"Blue", 3, "Body", "HealthRegen"}: 0.21,
	{"Blue", 3, "Body", "PhysArmor"}: 3.2,
	// Blue S3 Shoulders
	{"Blue", 3, "Shoulders", "Health"}: 105,
	{"Blue", 3, "Shoulders", "MagicArmor"}: 2.4,
	// Blue S3 Hands
	{"Blue", 3, "Hands", "Health"}: 53,
	{"Blue", 3, "Hands", "AttackSpeed"}: 0.076,
	{"Blue", 3, "Hands", "CritChance"}: 0.03,
	// Blue S3 Belt
	{"Blue", 3, "Belt", "Mana"}: 21,
	{"Blue", 3, "Belt", "MagicArmor"}: 2.4,
	// Blue S3 Legs
	{"Blue", 3, "Legs", "Mana"}: 21,
	{"Blue", 3, "Legs", "ManaRegen"}: 0.21,
	{"Blue", 3, "Legs", "AntiPhysArmor"}: 8.4,
	// Blue S3 Foots
	{"Blue", 3, "Foots", "Health"}: 53,
	{"Blue", 3, "Foots", "HealthRegen"}: 0.21,
	{"Blue", 3, "Foots", "PhysArmor"}: 3.2,
	// Blue S4 Helm
	{"Blue", 4, "Helm", "Mana"}: 51,
	{"Blue", 4, "Helm", "ManaRegen"}: 0.26,
	{"Blue", 4, "Helm", "AntiMagicArmor"}: 5.6,
	// Blue S4 Body
	{"Blue", 4, "Body", "HealthRegen"}: 0.26,
	{"Blue", 4, "Body", "PhysArmor"}: 4.2,
	// Blue S4 Shoulders
	{"Blue", 4, "Shoulders", "Health"}: 128,
	{"Blue", 4, "Shoulders", "MagicArmor"}: 3.2,
	// Blue S4 Hands
	{"Blue", 4, "Hands", "Health"}: 64,
	{"Blue", 4, "Hands", "AttackSpeed"}: 0.101,
	{"Blue", 4, "Hands", "CritChance"}: 0.04,
	// Blue S4 Belt
	{"Blue", 4, "Belt", "Mana"}: 26,
	{"Blue", 4, "Belt", "MagicArmor"}: 3.2,
	// Blue S4 Legs
	{"Blue", 4, "Legs", "Mana"}: 26,
	{"Blue", 4, "Legs", "ManaRegen"}: 0.26,
	{"Blue", 4, "Legs", "AntiPhysArmor"}: 11.2,
	// Blue S4 Foots
	{"Blue", 4, "Foots", "Health"}: 64,
	{"Blue", 4, "Foots", "HealthRegen"}: 0.26,
	{"Blue", 4, "Foots", "PhysArmor"}: 4.2,
	// Blue S5 Helm
	{"Blue", 5, "Helm", "Mana"}: 60,
	{"Blue", 5, "Helm", "ManaRegen"}: 0.3,
	{"Blue", 5, "Helm", "AntiMagicArmor"}: 7.2,
	// Blue S5 Body
	{"Blue", 5, "Body", "HealthRegen"}: 0.3,
	{"Blue", 5, "Body", "PhysArmor"}: 5.4,
	// Blue S5 Shoulders
	{"Blue", 5, "Shoulders", "Health"}: 151,
	{"Blue", 5, "Shoulders", "MagicArmor"}: 4.1,
	// Blue S5 Hands
	{"Blue", 5, "Hands", "Health"}: 75,
	{"Blue", 5, "Hands", "AttackSpeed"}: 0.13,
	{"Blue", 5, "Hands", "CritChance"}: 0.05,
	// Blue S5 Belt
	{"Blue", 5, "Belt", "Mana"}: 30,
	{"Blue", 5, "Belt", "MagicArmor"}: 4.1,
	// Blue S5 Legs
	{"Blue", 5, "Legs", "Mana"}: 30,
	{"Blue", 5, "Legs", "ManaRegen"}: 0.3,
	{"Blue", 5, "Legs", "AntiPhysArmor"}: 14.4,
	// Blue S5 Foots
	{"Blue", 5, "Foots", "Health"}: 75,
	{"Blue", 5, "Foots", "HealthRegen"}: 0.3,
	{"Blue", 5, "Foots", "PhysArmor"}: 5.4,
	// Blue S6 Helm
	{"Blue", 6, "Helm", "Mana"}: 65,
	{"Blue", 6, "Helm", "ManaRegen"}: 0.32,
	{"Blue", 6, "Helm", "AntiMagicArmor"}: 8.5,
	// Blue S6 Body
	{"Blue", 6, "Body", "HealthRegen"}: 0.32,
	{"Blue", 6, "Body", "PhysArmor"}: 6.3,
	// Blue S6 Shoulders
	{"Blue", 6, "Shoulders", "Health"}: 162,
	{"Blue", 6, "Shoulders", "MagicArmor"}: 4.8,
	// Blue S6 Hands
	{"Blue", 6, "Hands", "Health"}: 81,
	{"Blue", 6, "Hands", "AttackSpeed"}: 0.152,
	{"Blue", 6, "Hands", "CritChance"}: 0.05,
	// Blue S6 Belt
	{"Blue", 6, "Belt", "Mana"}: 32,
	{"Blue", 6, "Belt", "MagicArmor"}: 4.8,
	// Blue S6 Legs
	{"Blue", 6, "Legs", "Mana"}: 32,
	{"Blue", 6, "Legs", "ManaRegen"}: 0.32,
	{"Blue", 6, "Legs", "AntiPhysArmor"}: 16.9,
	// Blue S6 Foots
	{"Blue", 6, "Foots", "Health"}: 81,
	{"Blue", 6, "Foots", "HealthRegen"}: 0.32,
	{"Blue", 6, "Foots", "PhysArmor"}: 6.3,
	// Blue S7 Helm
	{"Blue", 7, "Helm", "Mana"}: 77,
	{"Blue", 7, "Helm", "ManaRegen"}: 0.39,
	{"Blue", 7, "Helm", "AntiMagicArmor"}: 10,
	// Blue S7 Body
	{"Blue", 7, "Body", "HealthRegen"}: 0.39,
	{"Blue", 7, "Body", "PhysArmor"}: 7.5,
	// Blue S7 Shoulders
	{"Blue", 7, "Shoulders", "Health"}: 193,
	{"Blue", 7, "Shoulders", "MagicArmor"}: 5.6,
	// Blue S7 Hands
	{"Blue", 7, "Hands", "Health"}: 96,
	{"Blue", 7, "Hands", "AttackSpeed"}: 0.18,
	{"Blue", 7, "Hands", "CritChance"}: 0.06,
	// Blue S7 Belt
	{"Blue", 7, "Belt", "Mana"}: 39,
	{"Blue", 7, "Belt", "MagicArmor"}: 4.8,
	// Blue S7 Legs
	{"Blue", 7, "Legs", "Mana"}: 39,
	{"Blue", 7, "Legs", "ManaRegen"}: 0.39,
	{"Blue", 7, "Legs", "AntiPhysArmor"}: 20,
	// Blue S7 Foots
	{"Blue", 7, "Foots", "Health"}: 96,
	{"Blue", 7, "Foots", "HealthRegen"}: 0.39,
	{"Blue", 7, "Foots", "PhysArmor"}: 7.5,
	// Violet S6 Helm
	{"Violet", 6, "Helm", "Mana"}: 76,
	{"Violet", 6, "Helm", "ManaRegen"}: 0.38,
	{"Violet", 6, "Helm", "AntiMagicArmor"}: 9.9,
	// Violet S6 Body
	{"Violet", 6, "Body", "HealthRegen"}: 0.38,
	{"Violet", 6, "Body", "PhysArmor"}: 7.4,
	// Violet S6 Shoulders
	{"Violet", 6, "Shoulders", "Health"}: 190,
	{"Violet", 6, "Shoulders", "MagicArmor"}: 5.6,
	// Violet S6 Hands
	{"Violet", 6, "Hands", "Health"}: 95,
	{"Violet", 6, "Hands", "AttackSpeed"}: 0.178,
	{"Violet", 6, "Hands", "CritChance"}: 0.06,
	// Violet S6 Belt
	{"Violet", 6, "Belt", "Mana"}: 38,
	{"Violet", 6, "Belt", "MagicArmor"}: 5.6,
	// Violet S6 Legs
	{"Violet", 6, "Legs", "Mana"}: 38,
	{"Violet", 6, "Legs", "ManaRegen"}: 0.38,
	{"Violet", 6, "Legs", "AntiPhysArmor"}: 19.8,
	// Violet S6 Foots
	{"Violet", 6, "Foots", "Health"}: 95,
	{"Violet", 6, "Foots", "HealthRegen"}: 0.38,
	{"Violet", 6, "Foots", "PhysArmor"}: 7.4,
	// Violet S7 Helm
	{"Violet", 7, "Helm", "Mana"}: 88,
	{"Violet", 7, "Helm", "ManaRegen"}: 0.44,
	{"Violet", 7, "Helm", "AntiMagicArmor"}: 11.4,
	// Violet S7 Body
	{"Violet", 7, "Body", "HealthRegen"}: 0.44,
	{"Violet", 7, "Body", "PhysArmor"}: 8.6,
	// Violet S7 Shoulders
	{"Violet", 7, "Shoulders", "Health"}: 219,
	{"Violet", 7, "Shoulders", "MagicArmor"}: 6.4,
	// Violet S7 Hands
	{"Violet", 7, "Hands", "Health"}: 110,
	{"Violet", 7, "Hands", "AttackSpeed"}: 0.205,
	{"Violet", 7, "Hands", "CritChance"}: 0.07,
	// Violet S7 Belt
	{"Violet", 7, "Belt", "Mana"}: 44,
	{"Violet", 7, "Belt", "MagicArmor"}: 6.4,
	// Violet S7 Legs
	{"Violet", 7, "Legs", "Mana"}: 44,
	{"Violet", 7, "Legs", "ManaRegen"}: 0.44,
	{"Violet", 7, "Legs", "AntiPhysArmor"}: 22.8,
	// Violet S7 Foots
	{"Violet", 7, "Foots", "Health"}: 110,
	{"Violet", 7, "Foots", "HealthRegen"}: 0.44,
	{"Violet", 7, "Foots", "PhysArmor"}: 8.6,
}
