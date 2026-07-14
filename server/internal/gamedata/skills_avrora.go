package gamedata

// Avrora («Аврора») — a post-1.11 SUPPORT healer recovered from the 1.19 drop
// (model bundle Avtr_Sp_Avrora.unity3d). Her skill names and descriptions come
// from an in-game screenshot, transcribed in
// tanat/_decompiled/wiki_archive/avatar_skills_roster.md (## avrora):
//
//	1. Милость          — heal a chosen ally, damaging every enemy around them.
//	2. Освященное место — consecrate an area: allies heal, enemies take damage;
//	                       the more allies inside, the stronger the effect.
//	3. Воссоединение    — pull a chosen ally to her, restoring part of their HP.
//	4. Молитва          — instantly heal part of every ally's HP across the map.
//
// This kit is HAND-AUTHORED (not part of the generated skillspecs), so it lives
// in its own file to keep skills_gen.go pristine.
//
// Solo-PvE adaptation: Avrora is a pure healer whose four abilities all target
// ALLIES. The Hunt battle engine (effects.go) has no allied player objects, so
// every OpHeal/OpHot resolves to the caster and OpPull grabs an enemy. Each skill
// is therefore mapped to its closest self/enemy form -- self-heal plus the enemy
// half of the ability. Faithful ally-targeting needs multiplayer party battles,
// which are not implemented yet.
//
// All fx fields are intentionally EMPTY. Avrora's VFX prefabs are not in the 1.11
// client's baked VisualEffectsMgr registry, so triggering them by name would do
// nothing (fxStartLocked no-ops on ""). Fill them in once the client is repacked
// to register her effects.
func init() {
	registerSkills(&AvatarSkills{
		Prefab: "Avtr_Sp_Avrora", AttackProjectile: false,
		Skills: [4]Skill{
			{
				// 1 — Милость: mend the caster and smite the enemies around her
				// (the "damage enemies around the healed ally" half, ally = self).
				Slot: 1, NameRu: "Милость", Type: "ACTIVE",
				Target: "SELF", Targeting: "SELF", Distance: 0, AoERadius: 5, AoEWidth: 0,
				ManaCost: []int{30, 35, 40, 45}, Cooldown: []int{10, 9, 9, 8},
				PayloadDelay: 0.3,
				Ops: []Op{
					{Kind: OpHeal, Value: PerLevel{60, 80, 100, 120}, PerSP: 1},
					{Kind: OpDamage, Value: PerLevel{40, 55, 70, 85}, Scale: "magic", Radius: 5},
				},
				TipArgs: map[string]PerLevel{
					"hpRestore": PerLevel{60, 80, 100, 120},
					"damageSP":  PerLevel{1, 1, 1, 1},
				},
			},
			{
				// 2 — Освященное место: a consecrated spot that burns enemies over
				// time (channel) while mending the caster (the "allies heal" half).
				Slot: 2, NameRu: "Освященное место", Type: "ACTIVE",
				Target: "POINT", Targeting: "", Distance: 10, AoERadius: 4, AoEWidth: 0,
				ManaCost: []int{45, 50, 55, 60}, Cooldown: []int{14, 13, 12, 11},
				PayloadDelay: 0.3,
				Ops: []Op{
					{Kind: OpHot, Value: PerLevel{14, 18, 22, 26}, Dur: PerLevel{5, 5, 5, 5}},
					{Kind: OpChannel, Dur: PerLevel{5, 5, 5, 5}, Interval: 1, Ops: []Op{
						{Kind: OpDamage, Value: PerLevel{22, 30, 38, 46}, Scale: "magic", Radius: 4},
					}},
				},
				TipArgs: map[string]PerLevel{
					"duration": PerLevel{5, 5, 5, 5},
					"damageSP": PerLevel{1, 1, 1, 1},
				},
			},
			{
				// 3 — Воссоединение: the ally-reunite pull. With no allies to grab it
				// yanks the targeted foe to the caster and restores her health.
				Slot: 3, NameRu: "Воссоединение", Type: "ACTIVE",
				Target: "ENEMY+NOT_BUILDING", Targeting: "TARGET", Distance: 10, AoERadius: 0, AoEWidth: 0,
				ManaCost: []int{40, 45, 50, 55}, Cooldown: []int{14, 13, 12, 11},
				PayloadDelay: 0.2,
				Ops: []Op{
					{Kind: OpPull},
					{Kind: OpHeal, Value: PerLevel{50, 70, 90, 110}, PerSP: 1},
					{Kind: OpDamage, Value: PerLevel{30, 40, 50, 60}, Scale: "magic"},
				},
				TipArgs: map[string]PerLevel{
					"hpRestore": PerLevel{50, 70, 90, 110},
					"damageSP":  PerLevel{1, 1, 1, 1},
				},
			},
			{
				// 4 — Молитва: instant heal to every ally on the map (= the caster).
				Slot: 4, NameRu: "Молитва", Type: "ACTIVE",
				Target: "SELF", Targeting: "SELF", Distance: 0, AoERadius: 0, AoEWidth: 0,
				ManaCost: []int{70, 80, 90, 100}, Cooldown: []int{45, 42, 39, 36},
				PayloadDelay: 0.2,
				Ops: []Op{
					{Kind: OpHeal, Value: PerLevel{150, 200, 250, 300}, PerSP: 1},
				},
				TipArgs: map[string]PerLevel{
					"hpRestore": PerLevel{150, 200, 250, 300},
					"damageSP":  PerLevel{1, 1, 1, 1},
				},
			},
		},
	})
}
