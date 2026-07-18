// Package gamedata holds the static game content the server hands to the
// client: the battle-avatar roster and the game-mode map list. Everything here
// references assets that actually ship inside the 1.11 client:
//
//   - Prefab names come from data/resources.xml (bundles in
//     data/Characters/Avatars/<Prefab>.unity3d, mobs in data/Characters/Mobs).
//   - Portrait icons are Unity Resources paths baked into Tanat_Data/mainData:
//     gui/avatars/icons/<prefab lowercase>_01.._03; the client appends the
//     _01/_02/_03 suffix itself (SelectAvatarWindow does GetImage(mImg+"_03")).
//   - Skill icons: gui/icons/skills/<base lowercase>_skill1..4 (+_1 variants).
//     The lobby UI prepends "Gui/Icons/skills/" to avatars.amf skills[].icon
//     itself, so that field carries the BARE texture name.
//   - ALL display strings are locale keys resolved by GuiSystem.GetLocaleText
//     against the client's embedded Russian locale (TextAsset configs/locale in
//     resources.assets; extracted copy kept in the session scratchpad). A key
//     missing from the locale renders as the literal "EMPTY!".
//   - Hunt scenes map_4_0/map_4_1/map_4_2 exist in data/scenes (MapType.HUNT=4).
package gamedata

import (
	"fmt"
	"math"
	"sort"
)

// AvatarType mirrors SelectAvatarWindow.AvatarType. The select window only
// builds button columns for WARRIOR..SUPPORT (NONE=0 and DISCOUNT are dropped
// with "Unexpected avatar type"), so avatars.amf "type" MUST be one of these.
const (
	AvatarTypeWarrior int32 = 1
	AvatarTypeKiller  int32 = 2
	AvatarTypeMage    int32 = 3
	AvatarTypeSupport int32 = 4
)

// Avatar is one playable battle avatar. ID is the Ctrl-side avatar id used in
// avatar|list, hunt|ready {avatar_id} and PLAYER_REG.
type Avatar struct {
	ID     int32
	Prefab string // Unity prefab (= bundle name in data/Characters/Avatars)
	Type   int32  // AvatarType* (column in the select window)
	// ShortName is the hero id used by the skill locale keys
	// (IDS_<ShortName>Skill<N>_Name / _LobbyDesc), e.g. "Mihalych".
	ShortName string
	// skillIconBase overrides the skill-icon texture base for heroes whose
	// icons shipped under a different class prefix than their prefab (verified
	// against the resource table in Tanat_Data/mainData: Astarot's skill icons
	// are avtr_psh_astarot_skillN). Empty = Prefab.
	skillIconBase string
	// localeBase overrides the IDS_<base>_Name/_ShortDesc/_LongDesc key base for
	// heroes whose model bundle name differs from their locale name (verified:
	// Ariana's model bundle is Avtr_Sp_Arianna but her name key is
	// IDS_Avtr_Sp_Ariana_Name -- one "n"). Empty = Prefab.
	localeBase string
	// iconBase overrides the Gui/Avatars/Icons/<base>_NN portrait base for heroes
	// whose portrait textures shipped under a different name than their prefab
	// (verified against data/Icons/manifest.tsv: Ariana's prefab is
	// Avtr_Sp_Arianna but her portraits are Avtr_Sp_Ariana_01.._04 -- one "n").
	// Empty = Prefab.
	iconBase string

	Health      float64
	Mana        float64
	HealthRegen float64
	ManaRegen   float64
	AttackSpeed float64
	AttackRange float64
	// AttackWindup is the fraction of a basic-attack swing (0..1) the caster spends
	// winding up before the projectile is released. 0 = a snap shot that leaves at the
	// start of the swing (melee heroes, who never fire one). Every ranged hero looses
	// at the END of the draw/throw so it flies and lands late; this is set uniformly
	// by rangedAttackWindup in registerSkills for AttackProjectile avatars (0.65, or
	// 0.5 for Plus-Minus, the one mid-animation exception) -- not hand-set here.
	AttackWindup float64
	DmgMin       int32
	DmgMax       int32
	PhysArmor    float64
	MagicArmor   float64
	SpellPower   float64
	// CollisionRadius is the unit's body radius in world units -- the logical
	// hitbox the client's own AI reads as SyncType.RADIUS (reach = attackRange +
	// attacker.radius + target.radius). Roughly the red ring drawn under the
	// unit. 0 => DefaultAvatarRadius.
	CollisionRadius float64
}

// DefaultAvatarRadius / DefaultMobRadius are the body radii used when a unit
// leaves CollisionRadius unset -- chosen so the previous flat reach constants
// (~2.0 melee) reproduce for an average human-sized fighter.
const (
	DefaultAvatarRadius = 0.8
	DefaultMobRadius    = 0.6
)

// Radius returns the avatar's body radius (CollisionRadius or the default).
func (a Avatar) Radius() float64 {
	if a.CollisionRadius > 0 {
		return a.CollisionRadius
	}
	return DefaultAvatarRadius
}

// localeName is the base string for the IDS_<base>_* display-string keys.
func (a Avatar) localeName() string {
	if a.localeBase != "" {
		return a.localeBase
	}
	return a.Prefab
}

// Name returns the locale key of the avatar's display name
// (e.g. IDS_Avtr_HK_Mihalych_Name = "Михалыч").
func (a Avatar) Name() string { return "IDS_" + a.localeName() + "_Name" }

// Short returns the locale key of the avatar's short description.
func (a Avatar) Short() string { return "IDS_" + a.localeName() + "_ShortDesc" }

// Long returns the locale key of the avatar's long description.
func (a Avatar) Long() string { return "IDS_" + a.localeName() + "_LongDesc" }

// Icon returns the portrait Resources path without the _NN suffix (the client
// appends _01/_02/_03 itself).
func (a Avatar) Icon() string {
	base := a.iconBase
	if base == "" {
		base = a.Prefab
	}
	return "Gui/Avatars/Icons/" + base
}

// SkillTitle returns the locale key of the n-th (1-based) skill's name.
func (a Avatar) SkillTitle(n int) string {
	return fmt.Sprintf("IDS_%sSkill%d_Name", a.ShortName, n)
}

// SkillDesc returns the locale key of the n-th skill's lobby description. The
// _LobbyDesc variant is plain prose; _Short/_Long/_CommonDesc contain
// unsubstituted {*damage} placeholders that would render literally.
func (a Avatar) SkillDesc(n int) string {
	return fmt.Sprintf("IDS_%sSkill%d_LobbyDesc", a.ShortName, n)
}

// SkillIcon returns the BARE texture name of the n-th skill icon for
// avatars.amf (the lobby UI prepends "Gui/Icons/skills/" itself).
func (a Avatar) SkillIcon(n int) string {
	base := a.skillIconBase
	if base == "" {
		base = a.Prefab
	}
	return fmt.Sprintf("%s_skill%d", base, n)
}

// SkillIconPath returns the FULL Resources path of the n-th skill icon for the
// battle-side effector prototypes (MainInfoWindow loads the Desc icon verbatim
// and the upgrade button loads icon+"_1").
func (a Avatar) SkillIconPath(n int) string {
	return "Gui/Icons/skills/" + a.SkillIcon(n)
}

// statsFor returns the per-class baseline combat stats. Every hero of a given
// AvatarType shares this template -- the exact numbers only tune Hunt PvE
// pacing, so the 39-avatar roster stays maintainable instead of hand-tuning
// each stat line. Melee classes get short AttackRange; casters get 6.0.
func statsFor(t int32) Avatar {
	switch t {
	case AvatarTypeWarrior:
		return Avatar{Health: 840, Mana: 200, HealthRegen: 1.3, ManaRegen: 0.6,
			AttackSpeed: 0.9, AttackRange: 2.5, DmgMin: 42, DmgMax: 54, PhysArmor: 5, MagicArmor: 4}
	case AvatarTypeKiller:
		return Avatar{Health: 580, Mana: 260, HealthRegen: 0.9, ManaRegen: 0.8,
			AttackSpeed: 1.2, AttackRange: 2.5, DmgMin: 50, DmgMax: 62, PhysArmor: 3, MagicArmor: 3}
	case AvatarTypeMage:
		return Avatar{Health: 540, Mana: 400, HealthRegen: 0.8, ManaRegen: 1.3,
			AttackSpeed: 1.0, AttackRange: 6.0, DmgMin: 44, DmgMax: 56, PhysArmor: 2, MagicArmor: 5, SpellPower: 12}
	case AvatarTypeSupport:
		return Avatar{Health: 520, Mana: 420, HealthRegen: 0.9, ManaRegen: 1.4,
			AttackSpeed: 1.0, AttackRange: 6.0, DmgMin: 34, DmgMax: 44, PhysArmor: 2, MagicArmor: 4, SpellPower: 10}
	default:
		return Avatar{Health: 600, Mana: 250, HealthRegen: 1.0, ManaRegen: 0.8,
			AttackSpeed: 1.0, AttackRange: 2.5, DmgMin: 40, DmgMax: 50, PhysArmor: 3, MagicArmor: 3}
	}
}

// mk builds a roster avatar of the given class with template stats. ShortName
// is the token used by the IDS_<ShortName>Skill<N>_* skill locale keys (usually
// the prefab's last segment; a few heroes differ -- see the fixups below).
func mk(id int32, prefab, short string, t int32) Avatar {
	a := statsFor(t)
	a.ID = id
	a.Prefab = prefab
	a.ShortName = short
	a.Type = t
	return a
}

// avatars is the full playable roster: every avatar bundle in
// data/Characters/Avatars that also has an IDS_<Prefab>_Name locale entry (42
// bundles minus the 3 NPC models SpiderQueen/Alchemist/Priest that lack names).
// Taladar is omitted -- it has a locale name but no shipped bundle (unfinished).
// Types (WARRIOR/KILLER/MAGE/SUPPORT) come from each hero's _ShortDesc class
// wording; a few DPS/Dsb heroes cross their prefix (e.g. Cerber/Nerlag are
// bruiser WARRIORs, Wilfang a melee KILLER, Edilia a healer SUPPORT).
var avatars = buildAvatars()

func buildAvatars() []Avatar {
	list := []Avatar{
		// WARRIOR
		mk(1, "Avtr_Tank_Rognar", "Rognar", AvatarTypeWarrior),
		mk(2, "Avtr_Tank_Gektor", "Gektor", AvatarTypeWarrior),
		mk(5, "Avtr_DPS_Cerber", "Cerber", AvatarTypeWarrior),
		mk(9, "Avtr_Tank_Veritas", "Veritas", AvatarTypeWarrior),
		mk(10, "Avtr_Tank_Sigilion", "Sigilion", AvatarTypeWarrior),
		mk(11, "Avtr_Tank_Zamaran", "Zamaran", AvatarTypeWarrior),
		mk(12, "Avtr_Tank_Urg", "Urg", AvatarTypeWarrior),
		mk(13, "Avtr_Tank_Velial", "Velial", AvatarTypeWarrior),
		mk(14, "Avtr_Tank_Titanid", "Titanid", AvatarTypeWarrior),
		mk(15, "Avtr_DPS_Nerlag", "Nerlag", AvatarTypeWarrior),
		// Gayal is a melee lifesteal bruiser -- classed WARRIOR (like Cerber) so
		// the KILLER column stays within the select window's 15-per-class limit.
		mk(24, "Avtr_DPS_Gayal", "Gayal", AvatarTypeWarrior),
		// KILLER
		mk(4, "Avtr_DPS_Lirvein", "Lirvein", AvatarTypeKiller),
		mk(7, "Avtr_HK_Astarot", "Astarot", AvatarTypeKiller),
		mk(8, "Avtr_HK_Mihalych", "Mihalych", AvatarTypeKiller),
		mk(16, "Avtr_HK_ShinDalar", "ShinDalar", AvatarTypeKiller),
		mk(17, "Avtr_HK_Teridin", "Teridin", AvatarTypeKiller),
		mk(18, "Avtr_HK_Tangren", "Tangren", AvatarTypeKiller),
		mk(19, "Avtr_HK_Dutnik", "Dutnik", AvatarTypeKiller),
		mk(20, "Avtr_HK_Vigilans", "Vigilans", AvatarTypeKiller),
		mk(21, "Avtr_HK_Grimlok", "Grimlok", AvatarTypeKiller),
		mk(22, "Avtr_HK_Abominator", "Abominator", AvatarTypeKiller),
		mk(23, "Avtr_DPS_BlackDragon", "BlackDragon", AvatarTypeKiller),
		mk(25, "Avtr_DPS_Miriam", "Miriam", AvatarTypeKiller),
		mk(26, "Avtr_DPS_Einzenhaim", "Einzenhaim", AvatarTypeKiller),
		mk(27, "Avtr_DPS_Sandariel", "Sandariel", AvatarTypeKiller),
		mk(28, "Avtr_Dsb_Wilfang", "Wilfang", AvatarTypeKiller),
		// MAGE
		mk(3, "Avtr_DPS_Sharli", "Sharli", AvatarTypeMage),
		mk(29, "Avtr_DPS_Gellar", "Gellar", AvatarTypeMage),
		mk(30, "Avtr_Psh_Anhel", "Anhel", AvatarTypeMage),
		mk(31, "Avtr_Dsb_Elgorm", "Elgorm", AvatarTypeMage),
		mk(32, "Avtr_Dsb_Hekata", "Hekata", AvatarTypeMage),
		mk(33, "Avtr_Dsb_Morlokay", "Morlokay", AvatarTypeMage),
		mk(34, "Avtr_Dsb_Frost", "Frost", AvatarTypeMage),
		mk(35, "Avtr_Dsb_PlusMinus", "PlusMinus", AvatarTypeMage),
		// SUPPORT
		mk(6, "Avtr_Sp_Kiona", "Kiona", AvatarTypeSupport),
		mk(36, "Avtr_Sp_Neirofim", "Neirofim", AvatarTypeSupport),
		mk(37, "Avtr_Sp_Inshari", "Inshari", AvatarTypeSupport),
		mk(38, "Avtr_Sp_Arianna", "ArianaMey", AvatarTypeSupport),
		mk(39, "Avtr_Dsb_Edilia", "Edilia", AvatarTypeSupport),
		// Avrora («Аврора») — a post-1.11 healer, NOT in the stock 1.11 client.
		// Her model bundle (data/Characters/Avatars/Avtr_Sp_Avrora.unity3d) and its
		// resources.xml entry are copied in from the 1.19 drop; her skill kit is
		// hand-authored in skills_avrora.go. She still lacks BAKED client assets
		// (IDS_Avtr_Sp_Avrora_* locale keys, portrait, skill icons, VFX registry),
		// so until the client is repacked her name/skills render "EMPTY!" and her
		// portrait/skill icons are blank -- she is mechanically playable regardless.
		mk(40, "Avtr_Sp_Avrora", "Avrora", AvatarTypeSupport),
	}
	// Per-avatar asset-name exceptions, verified against data/Characters/Avatars,
	// data/resources.xml and the embedded locale.
	for i := range list {
		switch list[i].Prefab {
		case "Avtr_DPS_Einzenhaim":
			// Requested ranged: attack from range instead of the Killer template's 2.5
			// melee reach. His client model has NO basic-attack projectile pool
			// (VisualEffectOptions.mProjectiles is empty), so there is no arrow visual --
			// the swing plays and the hit lands instantly at range (a hitscan, like a
			// caster). A real flying projectile would require repacking his bundle. He
			// stays AttackProjectile:false, so no SET_PROJECTILE is emitted.
			list[i].AttackRange = rangedBasicAttackRange
		case "Avtr_HK_Astarot":
			// GUI skill icons shipped under the Psh_ class prefix.
			list[i].skillIconBase = "Avtr_Psh_Astarot"
			// "После релиза" stat line (base_balance/stats.txt): 423 HP, 1 HP/s,
			// 184 mana, 1.3s attack interval (=> AttackSpeed 1/1.3 ~= 0.77), 36 dmg.
			// ManaRegen not stated -> keep the Killer template value.
			list[i].Health = 423
			list[i].Mana = 184
			list[i].HealthRegen = 1.0
			list[i].AttackSpeed = 0.77
			list[i].DmgMin = 36
			list[i].DmgMax = 36
		case "Avtr_Sp_Neirofim":
			// Model bundle is "Neirofim" (with an i) but the GUI skill icons
			// shipped as Avtr_Sp_Nerofim_skill<n> (no i), so the generated
			// Neirofim path resolves to no texture and the icons stay blank.
			list[i].skillIconBase = "Avtr_Sp_Nerofim"
		case "Avtr_HK_Teridin":
			// GUI skill icons shipped under the DPS_ class prefix (the HK_ set is
			// incomplete: only Skill1/Skill4 exist, and with a capital "Skill").
			list[i].skillIconBase = "Avtr_DPS_Teridin"
			// Original stats.txt unit line: HP 495, regen 1, atk 40-46, mana 209,
			// mana-regen 1, phys armor 5, magic armor 15 (the same anchor Velial was
			// tuned from). AttackSpeed not stated -> keep the Killer template value.
			list[i].Health = 495
			list[i].Mana = 209
			list[i].HealthRegen = 1.0
			list[i].ManaRegen = 1.0
			list[i].DmgMin = 40
			list[i].DmgMax = 46
			list[i].PhysArmor = 5
			list[i].MagicArmor = 15
		case "Avtr_HK_Tangren":
			// GUI skill icons shipped under the Dsb_ class prefix.
			list[i].skillIconBase = "Avtr_Dsb_Tangren"
		case "Avtr_Sp_Arianna":
			// Model bundle is "Arianna" (two n); name key AND portrait icons are
			// "Ariana" (one n) -- IDS_Avtr_Sp_Ariana_Name + Avtr_Sp_Ariana_01..04.
			// Without the icon fixup her select-window button loads a missing
			// texture and renders as a blank slot (she looks absent from the
			// roster). Her SKILL icons genuinely use two n's, so skillIconBase
			// stays on the prefab.
			list[i].localeBase = "Avtr_Sp_Ariana"
			list[i].iconBase = "Avtr_Sp_Ariana"
		case "Avtr_HK_Mihalych":
			// "После релиза" stat line (base_balance/stats.txt): 472 HP, 1 HP/s,
			// 214 mana, 1 mana/s, 39 dmg, 1s attack interval (=> AttackSpeed 1.0).
			// (The L2 line 508/236/42 in stats.txt is just per-level scaling.)
			list[i].Health = 472
			list[i].Mana = 214
			list[i].HealthRegen = 1.0
			list[i].ManaRegen = 1.0
			list[i].AttackSpeed = 1.0
			list[i].DmgMin = 39
			list[i].DmgMax = 39
		case "Avtr_Tank_Sigilion":
			// "После релиза" stat line (base_balance/stats.txt): 500 HP, 1 HP/s,
			// 184 mana, 1 mana/s, 41 dmg. AttackSpeed not stated -> keep the Warrior
			// template value.
			list[i].Health = 500
			list[i].Mana = 184
			list[i].HealthRegen = 1.0
			list[i].ManaRegen = 1.0
			list[i].DmgMin = 41
			list[i].DmgMax = 41
		case "Avtr_Tank_Velial":
			// Velial's base stats are set relative to the KNOWN real-avatar lines
			// (stats.txt: Astarot 423, Mihalych 472, Teridin 495, Sigilion 500 HP,
			// ~36-46 dmg, 1/s regen, ~184-214 mana). His old 900 HP / 44-54 dmg were
			// a wild outlier; as a sustain Tank/bruiser he sits at the TOP of that
			// range (520 HP, a step above Sigilion's 500), hits a touch harder than the
			// single-target killers (42-48, cf. Teridin's 40-46) but slower (0.85), and
			// is tankier (phys 6). His kit (skills_gen.go) is unchanged: Удар изверга /
			// Разлом / Воля к победе / Трибунал.
			list[i].Health = 520
			list[i].Mana = 200
			list[i].HealthRegen = 1.0
			list[i].ManaRegen = 1.0
			list[i].AttackSpeed = 0.85
			list[i].AttackRange = 2.5
			list[i].DmgMin = 42
			list[i].DmgMax = 48
			list[i].PhysArmor = 6
			list[i].MagicArmor = 4
			list[i].SpellPower = 0
			list[i].CollisionRadius = 0.9
		}
	}
	return list
}

// Avatars returns the full roster in a stable order.
func Avatars() []Avatar { return avatars }

// AvatarByID finds a roster avatar by its Ctrl-side id.
func AvatarByID(id int32) (Avatar, bool) {
	for _, a := range avatars {
		if a.ID == id {
			return a, true
		}
	}
	return Avatar{}, false
}

// Mob is one hostile creature type. NameKey must exist in the client locale;
// Prefab must exist in data/Characters/Mobs (bundle = prefab name).
type Mob struct {
	NameKey     string
	Prefab      string
	Icon        string // enemy-card portrait base path; the client appends "_03"
	Health      float64
	Mana        float64 // authored for flavour; the battle engine does not use mob mana
	DmgMin      int32
	DmgMax      int32
	AttackSpeed float64
	Speed       float64
	XP          float64 // experience granted to the killer (at level 1; scales with MobSpawn.Level)
	Coins       int32   // bronze-coin bounty granted to the killer (at level 1; scales too)

	// PhysArmor is the mob's flat physical armor: incoming physical damage is scaled
	// by armorMitigation(armor) = 50/(armor+50), so e.g. 50 armor halves it. It is a
	// PERCENTAGE curve, so armor is NOT level-scaled (a value is one fixed % of
	// mitigation at every placement level). 0 = no armor, damage lands in full (the
	// default for fragile trash -- unchanged behaviour). Deliberately non-zero on
	// bosses and heavy/armored creatures (golems, big undead, armored demons), which
	// is exactly what an armor-break debuff (Velial's ult «Трибунал», which appends a
	// negative phys_armor status mod) is meant to strip -- breaking it past 0 flips
	// the curve to AMPLIFY damage. Attacker armor penetration (AntiPhysArmor potions,
	// phys_armor_pen) chips positive armor toward 0. Mobs have no magic armor yet.
	PhysArmor float64

	// Stationary mobs never chase or reposition -- they hold their spawn point and
	// only attack when the target enters range (turret/plant behaviour). Speed is
	// unused for them.
	Stationary bool
	// AttackRange overrides the default melee reach (0 = use mobAttackRange). Ranged
	// mobs (e.g. the FlowerHunter) get a larger reach so a stationary shooter can
	// actually hit at distance.
	AttackRange float64

	// CollisionRadius is the mob's body radius (see Avatar.CollisionRadius); big
	// bosses get a much larger ring than small mobs. 0 => DefaultMobRadius.
	CollisionRadius float64

	// Skills are special abilities the mob casts on cooldown between basic attacks
	// (used by bosses). Each plays an attack animation, spawns its VFX prefab, and
	// lands its effect after CastTime.
	Skills []BossSkill
}

// Radius returns the mob's body radius (CollisionRadius or the default).
func (m Mob) Radius() float64 {
	if m.CollisionRadius > 0 {
		return m.CollisionRadius
	}
	return DefaultMobRadius
}

// BossSkill is one boss ability: a windup that spawns a VFX prefab and then deals
// damage. It reuses the client's effector VFX channel (the Fx prefab ships in the
// boss's own bundle) and the standard damage pipeline.
type BossSkill struct {
	Name     string  // for logs
	Fx       string  // VFX prefab spawned on cast (from the boss bundle)
	OnTarget bool    // true: VFX plays at the target's position; false: on the boss
	Cooldown float64 // seconds between casts
	CastTime float64 // wind-up before the hit lands (boss is rooted meanwhile)
	Range    float64 // max distance to the target to start the cast
	Radius   float64 // >0 => AoE around the cast point (dodgeable by moving); 0 => guaranteed single-target
	Dmg      float64 // damage dealt on impact
}

// summonUnits carries the object-card strings for summon prefabs the mob roster does
// NOT hold. Summons normally reuse a mob model and borrow that mob's name and icon, so
// this table only exists for the two whose prefab is an avatar-side asset with no mob
// entry to borrow from -- and which therefore came up with a blank name AND a blank
// card, the same defect the «Штурм» creeps had, arrived at from the other direction.
//
// The elemental's real strings are shipped and were simply never wired up. The totem's
// are not: the client has no unit name for it (IDS_MorlokaySkill4_Name is the SKILL's
// name, which reads correctly on the unit too) and no totem art at object-card
// resolution (only item-sized totem icons, which lack the "_03" variant ObjectInfo
// demands), so its icon is a deliberate stand-in. Leaving it blank is NOT the cheaper
// option: ObjectInfo appends "_03" before loading, so "" makes the client hunt for a
// texture named "_03" and log a failure on every single selection.
var summonUnits = map[string]struct{ NameKey, Icon string }{
	"Avtr_Dsb_Frost_Elemental":        {"IDS_Mob_FrostElemental_Name", "Gui/Mobs/Icons/summon_elemental_frost"},
	"Avtr_Dsb_Morlokay_Skill4_prop01": {"IDS_MorlokaySkill4_Name", "Gui/Mobs/Icons/neitral_creep_tree"},
}

// UnitDesc returns the name key and object-card icon for a summonable unit prefab: the
// mob roster first (most summons are mob models), then the summon-only table. ok is
// false for a prefab the client ships no strings for -- a caller must never paper over
// that with empty strings, which the client renders as the literal text "EMPTY!" and a
// failed texture load rather than as "no card".
func UnitDesc(prefab string) (nameKey, icon string, ok bool) {
	for _, m := range mobs {
		if m.Prefab == prefab {
			return m.NameKey, m.Icon, true
		}
	}
	if d, found := summonUnits[prefab]; found {
		return d.NameKey, d.Icon, true
	}
	return "", "", false
}

// mobs: creatures for the map_4_* hunt scenes, with the display names the
// original locale shipped for them. Indices are referenced by MobSpawn.Mob.
// Move speeds are set slightly ABOVE the avatar's base run speed (lobbyMoveSpeed
// = 4.0 u/s) so a fleeing player can't simply outrun the pack; heavier mobs are
// a touch slower than lighter ones but all still edge out the avatar.
var mobs = []Mob{
	// 0,1 -- forest theme (open-air hunt maps).
	{NameKey: "IDS_MobMeleePanter_Name", Prefab: "Mob_Panter_Melee",
		Health: 140, DmgMin: 10, DmgMax: 16, AttackSpeed: 1.0, Speed: 4.5, XP: 30, CollisionRadius: 0.7},
	// The FlowerHunter bundle ships its prefabs as Mob_FlowerHunter_Range_01/02
	// (there is no bare "Mob_FlowerHunter" GameObject); the client's AssetLoader
	// keys on the exact prefab name, so the bundle name alone resolves to null and
	// the mob spawns invisible/untargetable. Use a real prefab name.
	{NameKey: "IDS_MobMeleeTree_Name", Prefab: "Mob_FlowerHunter_Range_01", Icon: "Gui/Mobs/Icons/mob_flower",
		Health: 220, DmgMin: 14, DmgMax: 22, AttackSpeed: 0.7, Speed: 4.2, XP: 45,
		Stationary: true, AttackRange: 8.0, CollisionRadius: 0.9}, // rooted ranged plant: holds ground, shoots
	// 2,3 -- dungeon theme (map_4_0 crypt): undead + demons. Prefab names verified
	// against data/resources.xml; name keys verified against the REAL baked locale
	// (extracted from Tanat_Data/resources.assets, TextAsset "locale" pathid 2266).
	// "Скелет мечник" = the base level-1 swordsman skeleton. Per stats.txt it is a
	// starter-tier mob alongside the ghoul: 80 HP, 10 XP, 6 bronze (was 170/8/5).
	{NameKey: "IDS_Mob_Skeleton1H_Melee_Name", Prefab: "Mob_Skeleton_1H_Melee_01", Icon: "Gui/Mobs/Icons/mob_skeleton",
		Health: 80, Mana: 0, DmgMin: 14, DmgMax: 20, AttackSpeed: 0.9, Speed: 4.4, XP: 10, Coins: 6, CollisionRadius: 0.6},
	// Index 3 -- the ELITE demon (Mob_Demon_Melee01, real name "Демон страж" =
	// Demon Guardian -- an apt fit, since he's the one guarding Velial's lair and
	// the Cerber->Hekata connector). Base = LEVEL 1; always placed deep (level
	// 10-18), so MobHPMul lifts it to elite-tier HP there (~470 at L10, ~710 at L18).
	{NameKey: "IDS_Mob_Demon_Melee_Name", Prefab: "Mob_Demon_Melee01", Icon: "Gui/Mobs/Icons/Mob_Demon",
		Health: 200, Mana: 0, DmgMin: 22, DmgMax: 32, AttackSpeed: 0.85, Speed: 4.3, XP: 18, Coins: 8, PhysArmor: 8, CollisionRadius: 0.95}, // armored guardian
	// 4-7 -- map_4_0 dungeon BOSSES, tuned as a difficulty LADDER the player climbs
	// as they level: Elgorm (beatable ~L5) < Velial (~L10) < Cerber (~L15) <
	// Hekata (~L20, the final boss). HP scales toward the original Titanid-boss
	// anchor (12000 HP, stats.txt) for Hekata; boss damage is set so an
	// under-levelled hero (less HP from LevelHealthMul) gets bursted down. Names
	// resolve from IDS_Boss_*_Name; prefabs load from data/Characters/Bosses.
	// Casters (Elgorm, Hekata) keep a longer AttackRange; the bruisers hit close.
	// Bosses are EXEMPT from mob level scaling (hand-tuned): their Health/damage/XP
	// and coin bounty are final. Coin bounties come straight from stats.txt (Elgorm
	// 96, Velial 214, Hekata 374); Cerber is a server-only intermediate boss (not in
	// stats.txt), set to 294 to keep the bounty ladder monotonic between Velial and
	// Hekata (linear at his level-15 slot: 214 + (374-214)/2).
	{NameKey: "IDS_Boss_Elgorm_Name", Prefab: "Boss_Elgorm", Icon: "Gui/Mobs/Icons/Boss_Elgorm",
		Health: 6000, DmgMin: 26, DmgMax: 40, AttackSpeed: 0.8, Speed: 4.0, XP: 600, Coins: 96, AttackRange: 7.0,
		CollisionRadius: 1.8, PhysArmor: 12, Skills: elgormSkills},
	{NameKey: "IDS_Boss_Velial_Name", Prefab: "Boss_Velial", Icon: "Gui/Mobs/Icons/Boss_Velial",
		Health: 8000, DmgMin: 42, DmgMax: 60, AttackSpeed: 0.9, Speed: 4.3, XP: 1500, Coins: 214,
		CollisionRadius: 2.2, PhysArmor: 18, Skills: velialSkills},
	{NameKey: "IDS_Boss_Cerber_Name", Prefab: "Boss_Cerber", Icon: "Gui/Mobs/Icons/Boss_Cerber",
		Health: 10000, DmgMin: 48, DmgMax: 66, AttackSpeed: 1.2, Speed: 4.9, XP: 3000, Coins: 294,
		CollisionRadius: 2.0, PhysArmor: 22, Skills: cerberSkills},
	{NameKey: "IDS_Boss_Hekata_Name", Prefab: "Boss_Hekata", Icon: "Gui/Mobs/Icons/Boss_Hekata",
		Health: 12000, DmgMin: 40, DmgMax: 60, AttackSpeed: 0.9, Speed: 4.0, XP: 5000, Coins: 374, AttackRange: 7.0,
		CollisionRadius: 1.9, PhysArmor: 20, Skills: hekataSkills},
	// 8-11 -- extra crypt creatures (undead + demons) for map_4_0. Stats below are
	// LEVEL 1; placement level (MobSpawn.Level) scales them. The Ghoul is the
	// trivial starter at the dungeon mouth: it reuses Elgorm's summon model
	// (Mob_ZombieCrawl_01), whose REAL shipped name is exactly "Голодный гуль" --
	// no locale patch needed, the key already bakes this text.
	{NameKey: "IDS_Mob_ZombieCrawl_Name", Prefab: "Mob_ZombieCrawl_01", Icon: "Gui/Mobs/Icons/Mob_Zombie",
		Health: 80, Mana: 0, DmgMin: 8, DmgMax: 8, AttackSpeed: 1.0, Speed: 4.2, XP: 10, Coins: 6, CollisionRadius: 0.5},
	// Skeleton archer: a fragile mobile ranged undead (chases, then shoots). Per the
	// original data: 40 HP / 150 mana at level 1, 10 XP + 6 coins.
	{NameKey: "IDS_Mob_Skeleton_Range_Name", Prefab: "Mob_Skeleton_Range_01", Icon: "Gui/Mobs/Icons/mob_skeleton",
		Health: 40, Mana: 150, DmgMin: 12, DmgMax: 18, AttackSpeed: 0.8, Speed: 4.2, XP: 10, Coins: 6, AttackRange: 9.0, CollisionRadius: 0.6},
	// Zombie brute: slow but tanky, a speed bump in the corridors. Real shipped name
	// for Mob_ZombieBig_01 is "Зомби крушитель" -- no locale patch needed either.
	{NameKey: "IDS_Mob_ZombieBig_Name", Prefab: "Mob_ZombieBig_01", Icon: "Gui/Mobs/Icons/Mob_Zombie",
		Health: 150, Mana: 0, DmgMin: 16, DmgMax: 24, AttackSpeed: 0.6, Speed: 3.4, XP: 14, Coins: 8, PhysArmor: 8, CollisionRadius: 1.0}, // heavy brute
	// Ranged demon: the elite ranged guardian, paired with the melee demons. Real
	// shipped name for Mob_Demon_Range is "Демон воитель" (was wrongly reusing the
	// melee demon's key -- fixed).
	{NameKey: "IDS_Mob_Demon_Range_Name", Prefab: "Mob_Demon_Range", Icon: "Gui/Mobs/Icons/Mob_Demon",
		Health: 170, Mana: 0, DmgMin: 26, DmgMax: 36, AttackSpeed: 0.8, Speed: 4.2, XP: 20, Coins: 9, AttackRange: 10.0, PhysArmor: 5, CollisionRadius: 0.95},
	// 12-22 -- the rest of each family the client actually ships a name+model for
	// (verified in the real baked locale), so the crypt uses every skeleton/
	// ghoul/zombie/demon variant instead of just one representative per family.
	// Each family = a "common" tier (roughly the existing baseline) + an elite
	// "_g" ("гневный"/"grand" -- tougher reskin) tier the locale itself names.
	//
	// Skeletons: 1H (мечник/воитель), 1HSh (рубака/берсерк), ranged
	// (лучник/снайпер), plus the one-off "Горящий скелет". No locale key ships
	// for the 2H_Melee model family, so it stays unused rather than mislabeled.
	{NameKey: "IDS_Mob_Skeleton1HSh_Melee_Name", Prefab: "Mob_Skeleton_1HSh_Melee_01", Icon: "Gui/Mobs/Icons/mob_skeleton",
		Health: 210, Mana: 0, DmgMin: 12, DmgMax: 18, AttackSpeed: 0.8, Speed: 4.2, XP: 9, Coins: 6, PhysArmor: 5, CollisionRadius: 0.6}, // shield-bearer
	{NameKey: "IDS_Mob_Skeleton1H_Melee_g_Name", Prefab: "Mob_Skeleton_1H_Melee_02", Icon: "Gui/Mobs/Icons/mob_skeleton",
		Health: 260, Mana: 0, DmgMin: 22, DmgMax: 30, AttackSpeed: 0.9, Speed: 4.5, XP: 14, Coins: 7, CollisionRadius: 0.65},
	{NameKey: "IDS_Mob_Skeleton1HSh_Melee_g_Name", Prefab: "Mob_Skeleton_1HSh_Melee_02", Icon: "Gui/Mobs/Icons/mob_skeleton",
		Health: 280, Mana: 0, DmgMin: 24, DmgMax: 34, AttackSpeed: 1.0, Speed: 4.6, XP: 15, Coins: 7, PhysArmor: 7, CollisionRadius: 0.65}, // elite shield-bearer
	{NameKey: "IDS_Mob_Skeleton_Range_g_Name", Prefab: "Mob_Skeleton_Range_02", Icon: "Gui/Mobs/Icons/mob_skeleton",
		Health: 60, Mana: 200, DmgMin: 20, DmgMax: 28, AttackSpeed: 0.8, Speed: 4.2, XP: 16, Coins: 7, AttackRange: 11.0, CollisionRadius: 0.6},
	{NameKey: "IDS_Mob_Skeleton_1H_Melee_05_Name", Prefab: "Mob_Skeleton_1H_Melee_05", Icon: "Gui/Mobs/Icons/mob_skeleton",
		Health: 120, Mana: 0, DmgMin: 22, DmgMax: 30, AttackSpeed: 1.0, Speed: 4.5, XP: 12, Coins: 7, CollisionRadius: 0.6},
	// Possessed ghoul: no second ZombieCrawl prefab ships, so the elite reuses the
	// same model (common in these games -- same mesh, different name/stats/tier).
	{NameKey: "IDS_Mob_ZombieCrawl_g_Name", Prefab: "Mob_ZombieCrawl_01", Icon: "Gui/Mobs/Icons/Mob_Zombie",
		Health: 140, Mana: 0, DmgMin: 14, DmgMax: 14, AttackSpeed: 1.1, Speed: 4.5, XP: 18, Coins: 8, CollisionRadius: 0.5},
	{NameKey: "IDS_Mob_ZombieBig_g_Name", Prefab: "Mob_ZombieBig_02", Icon: "Gui/Mobs/Icons/Mob_Zombie",
		Health: 240, Mana: 0, DmgMin: 26, DmgMax: 38, AttackSpeed: 0.6, Speed: 3.4, XP: 22, Coins: 9, PhysArmor: 10, CollisionRadius: 1.0}, // elite heavy brute
	// "Зомби солдат": a fast-hitting mid-weight brute. Per stats.txt: 150 HP with a
	// HIGH attack rate (0.7s between attacks) and heavy damage. AttackSpeed is
	// attacks-per-second (interval = 1/rate), so 0.7s interval => 1.43 (NOT 0.7,
	// which would be a slow 1.43s interval). Movement stays slow (3.8).
	{NameKey: "IDS_Mob_ZombieSword_Name", Prefab: "Mob_ZombieSword_01", Icon: "Gui/Mobs/Icons/Mob_Zombie",
		Health: 150, Mana: 0, DmgMin: 18, DmgMax: 26, AttackSpeed: 1.43, Speed: 3.8, XP: 12, Coins: 7, CollisionRadius: 0.8},
	{NameKey: "IDS_Mob_ZombieSword_g_Name", Prefab: "Mob_ZombieSword_02", Icon: "Gui/Mobs/Icons/Mob_Zombie",
		Health: 220, Mana: 0, DmgMin: 30, DmgMax: 42, AttackSpeed: 0.8, Speed: 3.9, XP: 20, Coins: 8, CollisionRadius: 0.85},
	// Elite melee demon (a distinct second Demon_Melee model ships); elite ranged
	// demon reuses the one Demon_Range model, same as the ghoul case above.
	{NameKey: "IDS_Mob_Demon_Melee_g_Name", Prefab: "Mob_Demon_Melee02", Icon: "Gui/Mobs/Icons/Mob_Demon",
		Health: 320, Mana: 0, DmgMin: 34, DmgMax: 48, AttackSpeed: 0.9, Speed: 4.4, XP: 28, Coins: 11, PhysArmor: 10, CollisionRadius: 1.0}, // elite armored demon
	{NameKey: "IDS_Mob_Demon_Range_g_Name", Prefab: "Mob_Demon_Range", Icon: "Gui/Mobs/Icons/Mob_Demon",
		Health: 270, Mana: 0, DmgMin: 38, DmgMax: 52, AttackSpeed: 0.85, Speed: 4.3, XP: 30, Coins: 12, AttackRange: 11.0, PhysArmor: 6, CollisionRadius: 0.95},
	// 23-26 -- map_4_2 («Заповедные джунгли») BOSS LADDER. Placed at fixed arenas
	// (MobSpawn.Abs, absolute world X,Z measured in-game) as a rising difficulty
	// ladder the party clears in order: Grimlok < Fairy < Titanid < Anhel (final).
	// (Roster order below is by id; the ladder RUNGS -- HP/dmg/XP/coins -- are assigned
	// per that power order, so Fairy sits between Grimlok and Titanid.)
	// v1 = BASIC ATTACK only (no Skills) -> hand-tuned via authored stats, placed at
	// Level 1 so ScaledStats is the identity (bosses stay unscaled). Names resolve
	// from the STOCK baked locale (verified in Tanat_Data/resources.assets): the
	// three hero-model bosses use their avatar name keys, Fairy the real boss key.
	// Grimlok/Titanid/Anhel REUSE playable AVATAR prefabs as boss models -- valid in
	// data/Characters/Avatars, and the client already instantiates avatar prefabs as
	// world objects (the lobby renders other players this way), so they render + play
	// their attack animation; confirm in-game. Fairy is a genuine Boss_* prefab.
	// Fairy/Anhel are ranged (AttackRange set): the client model's own attack spawns
	// its shipped projectile (Boss_Fairy_pojectile / Anhel VFX), same as ranged mobs.
	{NameKey: "IDS_Avtr_HK_Grimlok_Name", Prefab: "Avtr_HK_Grimlok", Icon: "Gui/Avatars/Icons/Avtr_HK_Grimlok",
		Health: 5000, DmgMin: 24, DmgMax: 38, AttackSpeed: 0.9, Speed: 4.2, XP: 500, Coins: 90, PhysArmor: 12, CollisionRadius: 1.8},
	{NameKey: "IDS_Avtr_Tank_Titanid_Name", Prefab: "Avtr_Tank_Titanid", Icon: "Gui/Avatars/Icons/Avtr_Tank_Titanid",
		Health: 9000, DmgMin: 40, DmgMax: 56, AttackSpeed: 0.85, Speed: 4.0, XP: 2000, Coins: 240, PhysArmor: 25, CollisionRadius: 2.0}, // stone tank
	{NameKey: "IDS_Boss_Fairy_Name", Prefab: "Boss_Fairy", Icon: "Gui/Mobs/Icons/Boss_Fairy",
		Health: 7000, DmgMin: 34, DmgMax: 50, AttackSpeed: 0.9, Speed: 4.3, XP: 1000, Coins: 150, AttackRange: 8.0, PhysArmor: 10, CollisionRadius: 1.6},
	{NameKey: "IDS_Avtr_Psh_Anhel_Name", Prefab: "Avtr_Psh_Anhel", Icon: "Gui/Avatars/Icons/Avtr_Psh_Anhel",
		Health: 12000, DmgMin: 44, DmgMax: 62, AttackSpeed: 0.9, Speed: 4.0, XP: 4000, Coins: 360, AttackRange: 7.0, PhysArmor: 22, CollisionRadius: 1.9},
	// 27-39 -- map_4_2 («Заповедные джунгли») TRASH ROSTER for the procedural pack
	// generator (buildJunglePack42). Jungle theme: spiders, tribesmen (natives),
	// dinosaurs, gorillas, and GOLEMS. Base = LEVEL 1; placement level (MobSpawn.Level)
	// scales HP/dmg/XP/coins along the route. Prefabs verified in data/Characters/Mobs
	// (valid_prefabs.txt); NameKeys verified in the stock baked locale (resources.assets).
	// Icons are best-effort (the GUI icon atlas isn't in resources.assets, same as the
	// crypt mobs) -- a missing one renders a blank enemy card, never a crash.
	// Spiders: small, fast, fragile swarmers at the mouth.
	{NameKey: "IDS_Mob_Spider_01_Name", Prefab: "Mob_Spider_01", Icon: "Gui/Mobs/Icons/mob_spider_g1",
		Health: 90, DmgMin: 8, DmgMax: 12, AttackSpeed: 1.1, Speed: 4.8, XP: 12, Coins: 6, CollisionRadius: 0.6},
	{NameKey: "IDS_Mob_Spider_02_Name", Prefab: "Mob_Spider_02", Icon: "Gui/Mobs/Icons/mob_spider_g2",
		Health: 130, DmgMin: 12, DmgMax: 16, AttackSpeed: 1.0, Speed: 4.7, XP: 16, Coins: 7, CollisionRadius: 0.7},
	// Tribesmen: native humanoids -- melee spearmen, ranged blowgunners, undead.
	{NameKey: "IDS_Mob_Tribesman_Melee_01_Name", Prefab: "Mob_Tribesman_Melee_01", Icon: "Gui/Mobs/Icons/mob_tribesman",
		Health: 150, DmgMin: 14, DmgMax: 20, AttackSpeed: 0.9, Speed: 4.4, XP: 14, Coins: 7, CollisionRadius: 0.7},
	{NameKey: "IDS_Mob_Tribesman_Range_01_Name", Prefab: "Mob_Tribesman_Range_01", Icon: "Gui/Mobs/Icons/mob_tribesman",
		Health: 90, DmgMin: 14, DmgMax: 20, AttackSpeed: 0.8, Speed: 4.3, XP: 14, Coins: 7, AttackRange: 9.0, CollisionRadius: 0.6},
	{NameKey: "IDS_Mob_Tribesman_Zombie_02_Name", Prefab: "Mob_Tribesman_Zombie_02", Icon: "Gui/Mobs/Icons/mob_tribesman",
		Health: 180, DmgMin: 16, DmgMax: 22, AttackSpeed: 0.8, Speed: 3.9, XP: 15, Coins: 7, CollisionRadius: 0.7},
	{NameKey: "IDS_Mob_TribesmanBig_Melee_01_Name", Prefab: "Mob_TribesmanBig_Melee_01", Icon: "Gui/Mobs/Icons/mob_tribesman",
		Health: 300, DmgMin: 26, DmgMax: 36, AttackSpeed: 0.7, Speed: 3.9, XP: 24, Coins: 10, PhysArmor: 8, CollisionRadius: 0.9}, // heavy native
	// Dinosaurs: fast melee raptors + a ranged spitter.
	{NameKey: "IDS_Mob_Dinosaur_Melee_01_Name", Prefab: "Mob_Dinosaur_Melee_01", Icon: "Gui/Mobs/Icons/mob_dinosaur",
		Health: 220, DmgMin: 20, DmgMax: 28, AttackSpeed: 0.9, Speed: 4.6, XP: 20, Coins: 8, CollisionRadius: 0.9},
	{NameKey: "IDS_Mob_Dinosaur_Melee_02_Name", Prefab: "Mob_Dinosaur_Melee_02", Icon: "Gui/Mobs/Icons/mob_dinosaur",
		Health: 300, DmgMin: 28, DmgMax: 38, AttackSpeed: 0.9, Speed: 4.7, XP: 26, Coins: 10, CollisionRadius: 1.0},
	{NameKey: "IDS_Mob_Dinosaur_Range_01_Name", Prefab: "Mob_Dinosaur_Range_01", Icon: "Gui/Mobs/Icons/mob_dinosaur",
		Health: 160, DmgMin: 22, DmgMax: 30, AttackSpeed: 0.8, Speed: 4.4, XP: 22, Coins: 9, AttackRange: 10.0, CollisionRadius: 0.9},
	// Gorillas: heavy bruisers; the GorillaBoss model is reused as an ELITE mob here.
	{NameKey: "IDS_Mob_Gorilla_Melee_01_Name", Prefab: "Mob_Gorilla_Melee_01", Icon: "Gui/Mobs/Icons/mob_gorilla",
		Health: 260, DmgMin: 24, DmgMax: 34, AttackSpeed: 0.85, Speed: 4.5, XP: 24, Coins: 10, CollisionRadius: 1.0},
	{NameKey: "IDS_Mob_GorillaBoss_Melee_01_Name", Prefab: "Mob_GorillaBoss_Melee_01", Icon: "Gui/Mobs/Icons/mob_gorilla",
		Health: 420, DmgMin: 34, DmgMax: 46, AttackSpeed: 0.8, Speed: 4.4, XP: 34, Coins: 13, PhysArmor: 10, CollisionRadius: 1.2}, // elite bruiser
	// Golems: tanky, slow stone guardians -- GATED to the Titanid trail (golemAllowed).
	{NameKey: "IDS_Mob_Golem_Melee_01_Name", Prefab: "Mob_Golem_Melee_01", Icon: "Gui/Mobs/Icons/mob_golem",
		Health: 500, DmgMin: 30, DmgMax: 44, AttackSpeed: 0.6, Speed: 3.4, XP: 40, Coins: 14, PhysArmor: 14, CollisionRadius: 1.3}, // stone golem
	{NameKey: "IDS_Mob_Golem_Melee_02_Name", Prefab: "Mob_Golem_Melee_02", Icon: "Gui/Mobs/Icons/mob_golem",
		Health: 650, DmgMin: 38, DmgMax: 52, AttackSpeed: 0.6, Speed: 3.3, XP: 48, Coins: 16, PhysArmor: 16, CollisionRadius: 1.4}, // greater stone golem
	// «Штурм» racial troops (mobHumanCreepMelee..mobElfCreepRange). Prefabs verified in
	// data/Characters/Creeps (H_Creep*/Elf_Creep* bundles expose Mnst_Human_Creep1..3 /
	// Mnst_Elf_Creep1..3). Creep1 = melee footman, Creep2 = ranged (ships a projectile);
	// Creep3 is the siege unit (Катапультозавр / Осадный медведь) and is not rostered.
	// Modest HP so a hero clears a wave but a lone hero can't solo a whole lane of
	// cannons+creeps instantly. XP/coins low (lane farm, not boss bounty). Team is set
	// per battle by the DOTA instance, not here.
	//
	// These four were the ONLY roster entries whose name AND icon were both invented:
	// the IDS_Mnst_* keys were never in the locale, and the icons never lived under
	// Gui/Mobs/Icons. The client resolves both by name against baked tables and cannot
	// be handed a key that does not exist -- an unknown locale id renders the literal
	// text "EMPTY!" and a missing texture renders nothing -- so «Штурм» creeps came up
	// nameless and blank. The real keys and the real Gui/Creeps/Icons folder are below;
	// side naming is the client's own: Sobor = Human, Apostate = Elf.
	{NameKey: "IDS_DotaCreepMeleeSobor_Name", Prefab: "Mnst_Human_Creep1_prop01", Icon: "Gui/Creeps/Icons/Mnst_Sobor_Creep1",
		Health: 200, DmgMin: 14, DmgMax: 20, AttackSpeed: 0.9, Speed: 4.0, XP: 12, Coins: 4, CollisionRadius: 0.6},
	{NameKey: "IDS_DotaCreepRangeSobor_Name", Prefab: "Mnst_Human_Creep2_prop01", Icon: "Gui/Creeps/Icons/Mnst_Sobor_Creep2",
		Health: 130, DmgMin: 16, DmgMax: 22, AttackSpeed: 0.8, Speed: 4.0, XP: 14, Coins: 5, AttackRange: 9.0, CollisionRadius: 0.55},
	{NameKey: "IDS_DotaCreepMeleeApostate_Name", Prefab: "Mnst_Elf_Creep1_prop01", Icon: "Gui/Creeps/Icons/Mnst_Apostate_Creep1",
		Health: 200, DmgMin: 14, DmgMax: 20, AttackSpeed: 0.9, Speed: 4.0, XP: 12, Coins: 4, CollisionRadius: 0.6},
	{NameKey: "IDS_DotaCreepRangeApostate_Name", Prefab: "Mnst_Elf_Creep2_prop01", Icon: "Gui/Creeps/Icons/Mnst_Apostate_Creep2",
		Health: 130, DmgMin: 16, DmgMax: 22, AttackSpeed: 0.8, Speed: 4.0, XP: 14, Coins: 5, AttackRange: 9.0, CollisionRadius: 0.55},
}

// Boss ability kits, one entry per distinctive VFX prefab in each boss bundle
// (verified in data/resources.xml). Casters get a ranged bolt + an AoE nuke;
// bruisers get a point-blank cleave + a lobbed AoE.
var (
	// Boss ability damage rises with the ladder (Elgorm < Velial < Cerber <
	// Hekata). AoE skills (Radius>0) are dodgeable by moving out of the ring, so
	// the single-target hits are the reliable-threat floor each fight leans on.
	// Boss-skill Fx MUST be a name the client's VisualEffectsMgr actually ships
	// (testdata/valid_fx.txt), or EFFECT_START silently renders nothing. The bosses
	// have no dedicated VFX registry, so each skill borrows its theme avatar's
	// matching effect: a @target/@point payload for OnTarget skills (plays at the
	// victim/cast point) and a @self payload for the point-blank ones (plays on the
	// boss). Validated by TestBossSkillFxValid.
	elgormSkills = []BossSkill{
		{Name: "Death Bolt", Fx: "ElgormSkill1Effect2", OnTarget: true,
			Cooldown: 6, CastTime: 0.6, Range: 9, Radius: 0, Dmg: 150},
		{Name: "Dead Hands", Fx: "ElgormSkill4Effect1", OnTarget: true,
			Cooldown: 11, CastTime: 1.0, Range: 8, Radius: 4.0, Dmg: 130},
	}
	velialSkills = []BossSkill{
		{Name: "Cleave", Fx: "VelialSkill1Effect", OnTarget: false,
			Cooldown: 5, CastTime: 0.5, Range: 3.5, Radius: 3.5, Dmg: 200},
		{Name: "Fire Skulls", Fx: "VelialSkill4Effect", OnTarget: true,
			Cooldown: 9, CastTime: 0.9, Range: 9, Radius: 3.5, Dmg: 190},
	}
	cerberSkills = []BossSkill{
		{Name: "Rend", Fx: "CerberSkill2Effect", OnTarget: false,
			Cooldown: 4, CastTime: 0.4, Range: 3.5, Radius: 3.0, Dmg: 230},
		{Name: "Chain Lash", Fx: "CerberSkill4Effect1", OnTarget: true,
			Cooldown: 7, CastTime: 0.5, Range: 10, Radius: 0, Dmg: 200},
	}
	hekataSkills = []BossSkill{
		{Name: "Hand Fire", Fx: "HekataSkill3Effect2", OnTarget: true,
			Cooldown: 5, CastTime: 0.5, Range: 9, Radius: 0, Dmg: 210},
		{Name: "Soul Burn", Fx: "HekataSkill4Effect", OnTarget: true,
			Cooldown: 10, CastTime: 1.2, Range: 10, Radius: 5.0, Dmg: 320},
	}
)

// Mob roster indices, referenced by the per-map spawn packs. Indices 0-2
// (panther/flowerhunter/skeleton) are pinned -- several battle tests key off
// them by number. New crypt creatures are appended at the end.
const (
	mobPanther = iota
	mobFlowerHunter
	mobSkeleton
	mobDemon
	mobBossElgorm
	mobBossVelial
	mobBossCerber
	mobBossHekata
	mobGhoul
	mobSkeletonArcher
	mobZombie
	mobDemonRange
	mobSkeletonHewer
	mobSkeletonWarrior
	mobSkeletonBerserk
	mobSkeletonSniper
	mobSkeletonBurning
	mobGhoulPossessed
	mobZombieBigElite
	mobZombieSoldier
	mobZombieSoldierElite
	mobDemonMeleeElite
	mobDemonRangeElite
	// map_4_2 jungle boss ladder (appended; several tests key mobs 0-2 by number).
	mobBossGrimlok
	mobBossTitanid
	mobBossFairy
	mobBossAnhel
	// map_4_2 jungle trash roster (procedural packs, buildJunglePack42).
	mobSpider
	mobSpiderElite
	mobTribesman
	mobTribesmanRange
	mobTribesmanZombie
	mobTribesmanBig
	mobDino
	mobDinoElite
	mobDinoRange
	mobGorilla
	mobGorillaElite
	mobGolem
	mobGolemElite
	// «Штурм» (DOTA, map_1_0) racial troops ("crips"): melee + ranged per race.
	// Rendered from the Mnst_Human_Creep* / Mnst_Elf_Creep* prefabs (data/Characters/
	// Creeps). They march a lane and fight; team is assigned per-battle, not here.
	mobHumanCreepMelee
	mobHumanCreepRange
	mobElfCreepMelee
	mobElfCreepRange
)

// demonFamily lists every demon-type mob index (used to test demon coverage in
// a region regardless of common/elite tier).
var demonFamily = []int{mobDemon, mobDemonRange, mobDemonMeleeElite, mobDemonRangeElite}

// Mobs returns the mob roster; index is the per-map spawn reference.
func Mobs() []Mob { return mobs }

// MobSpawn places one mob (by roster index) at an offset from the map's spawn
// point, so the layout follows wherever the measured spawn lands.
type MobSpawn struct {
	Mob int // index into Mobs()
	DX  float64
	DY  float64
	// Abs makes DX,DY absolute world coordinates instead of an offset from the
	// spawn point -- used for bosses pinned to fixed arenas across the map.
	Abs bool
	// Level is this placement's mob level (deeper = higher), scaling a regular
	// mob's HP/damage/XP/coins via MobHPMul/MobDmgMul/MobXPMul/MobCoinMul. 0/1 = level 1.
	// Ignored for bosses (hand-tuned). This is what makes mob level rise along the
	// route as the player progresses.
	Level int
}

// HuntMap is one PvE hunt location (MapType.HUNT = 4). Scene must match a
// SceneConfig.mSceneName inside the client (bundle data/scenes/<Scene>.unity3d).
type HuntMap struct {
	ID         int32
	Name       string
	Scene      string
	LevelMin   int32
	LevelMax   int32
	Desc       string
	WinDesc    string
	MinPlayers int32
	MaxPlayers int32

	// SpawnX/SpawnY: fallback spawn in the scene's world coordinates (the scene's
	// Reborn point, read from the bundle via UnityPy), used only when Nav is nil.
	// When Nav is set the spawn comes from Nav.Spawn() instead.
	SpawnX float64
	SpawnY float64

	// SpawnAt, when non-nil, is the battle-start point and OVERRIDES both Nav.Spawn()
	// and SpawnX/SpawnY. Used when the nav grid's seed marker is not the intended
	// player start: the real start is confirmed in-game (walk to the spot, read the
	// CLICK target coords) and pinned here, while Nav still drives walkability/clipping.
	SpawnAt *Vec2

	// Nav is the walkability oracle for this scene (nil = unrestricted movement).
	// When set it drives both the spawn point and movement clipping.
	Nav Nav

	Spawns []MobSpawn

	// Reborn are the scene's respawn checkpoints (the Reborn_point markers). Death
	// returns the player to the LAST one they walked near, not always the start:
	// approaching a checkpoint activates it and deactivates the previous. Empty =
	// always respawn at Spawn().
	Reborn []Vec2
}

// Spawn returns the avatar's start position in scene world coordinates: the
// nav-mesh interior point when walkability data exists, else the fallback.
func (m HuntMap) Spawn() (float64, float64) {
	if m.SpawnAt != nil {
		return m.SpawnAt.X, m.SpawnAt.Y
	}
	if m.Nav != nil {
		return m.Nav.Spawn()
	}
	return m.SpawnX, m.SpawnY
}

// MapTypeHunt mirrors TanatKernel.MapType.HUNT.
const MapTypeHunt int32 = 4

// dungeonPack40 is the map_4_0 (crypt) mob layout, laid out along the route the
// player takes as they level: Голодные гули at the mouth, undead trash through
// the corridors, then the four bosses as a difficulty ladder with demon packs
// guarding Velial's lair and the Cerber->Hekata connector. Every position is on
// real walkable floor from the authored PassibilityData grid (verified by
// TestNavGrid40MobsWalkable). All mobs -- trash and bosses alike -- respawn 5
// minutes after death (mobRespawnDelay), so the location can be farmed.
//
// dungeonPack40[0] stays a spawn-relative offset (>9.5m, clear line of sight):
// TestPathOnRealMap routes to it, and the offset rule keeps it from aggroing at
// battle start. Everything else is absolute world X,Z (reachable by pathfinding).
// dungeonRegion is a themed zone anchor: every generated PACK gets the LEVEL and
// creature POOL of whichever region anchor is nearest its centre, so each area is
// populated with the right creatures + level (level rises with progression depth,
// ghouls at the mouth ... demons at Hekata).
type dungeonRegion struct {
	x, y  float64
	level int
	pool  []int
}

// nearestRegion returns the region whose centre is closest to (wx,wy) and its index
// in regions (first-min on a tie), so callers address the region without recovering
// the index by float-equality re-scan.
func nearestRegion(regions []dungeonRegion, wx, wy float64) (dungeonRegion, int) {
	best := math.Inf(1)
	r := regions[0]
	ri := 0
	for i, rg := range regions {
		if d := math.Hypot(wx-rg.x, wy-rg.y); d < best {
			best, r, ri = d, rg, i
		}
	}
	return r, ri
}

func dungeonRegions() []dungeonRegion {
	return []dungeonRegion{
		// Pools mix races (ghoul/skeleton/zombie/demon) so a pack is a MOTLEY band, not
		// a monoculture -- except demons stay TERRITORIAL: concentrated at Velial's lair
		// and the Cerber->Hekata half (per the original demon-placement spec). The base
		// ghoul is confined to the two level-1 entrance regions (starter 80 HP / 6 coins).
		{493, 64, 1, []int{mobGhoul, mobGhoul, mobGhoulPossessed}},                                                        // spawn mouth (ghouls)
		{455, 117, 1, []int{mobGhoul, mobSkeleton, mobGhoulPossessed, mobSkeletonArcher}},                                 // first corridor (ghoul+skel)
		{384, 140, 3, []int{mobSkeleton, mobSkeletonHewer, mobZombieSoldier, mobSkeletonArcher, mobSkeletonBurning}},      // toward Elgorm
		{304, 121, 5, []int{mobSkeletonWarrior, mobSkeletonBerserk, mobZombie, mobSkeletonSniper, mobZombieSoldier}},      // Elgorm's hall
		{389, 328, 8, []int{mobZombie, mobSkeleton, mobZombieBigElite, mobSkeletonArcher, mobZombieSoldierElite}},         // mid (CP3)
		{486, 306, 10, []int{mobDemon, mobDemonMeleeElite, mobZombieSoldierElite, mobDemonRange, mobZombieBigElite}},      // Velial's lair (demons)
		{306, 444, 12, []int{mobZombieBigElite, mobDemon, mobSkeletonBerserk, mobZombieSoldierElite, mobDemonMeleeElite}}, // toward Cerber (CP4)
		{239, 434, 14, []int{mobDemon, mobDemonMeleeElite, mobDemonRange, mobZombieBigElite}},                             // Cerber's gate (demons)
		{170, 300, 16, []int{mobDemon, mobDemonMeleeElite, mobDemonRange, mobDemonRangeElite, mobZombieSoldierElite}},     // Cerber->Hekata connector (demons)
		{124, 346, 18, []int{mobDemonMeleeElite, mobSkeletonSniper, mobDemon, mobSkeletonWarrior, mobDemonRangeElite}},    // CP5
		{120, 212, 20, []int{mobDemonMeleeElite, mobDemonRangeElite, mobDemon, mobSkeletonWarrior, mobZombieBigElite}},    // Hekata's chamber
	}
}

const (
	// Trash is placed as discrete PACKS, not a uniform carpet: a handful of mobs
	// clustered tight enough to share one aggro pull (mobAggroRange 9m in the battle
	// server), with clear ground between packs so the player fights one group at a
	// time instead of wading through a continuous swarm.
	dungeonPackSpacing   = 24.0 // min centre-to-centre distance between packs (Poisson-disk)
	dungeonPackRadius    = 4.0  // hard cap on a member's distance from the centre
	dungeonMemberSpacing = 2.4  // target gap between packmates (Vogel spiral, room to breathe)

	// dungeonMinClear is the smallest wall-clearance (in cells ~= metres, Chebyshev)
	// a pack CENTRE may have: below this the centre would sit against a wall or
	// railing. Centres are chosen from the highest-clearance cells first, so packs
	// land on the MIDDLE of corridors and rooms, not jammed into corners.
	dungeonMinClear = 2

	dungeonSpawnClear  = 26.0 // mob-free ring around the player's start tile (safe zone)
	dungeonRebornClear = 26.0 // and around every respawn checkpoint (no mobs on a res)
	// Bosses get an even wider mob-free ring than a player respawn point (was 7m, then
	// 26m = respawn clearance, now +50%): you engage a boss in a clean arena, with no
	// trash pack pulled in alongside it. Applies to every map that pins bosses (crypt
	// dungeonPack40 + jungle buildJunglePack42).
	dungeonBossClear = 39.0
)

// bossPlacement pins one boss to a fixed world arena (spawned Abs). Shared by every
// map that places bosses -- the crypt (dungeonBosses) and the invasion lair
// (invasionBosses41).
type bossPlacement struct {
	mob  int
	x, y float64
}

var dungeonBosses = []bossPlacement{
	{mobBossElgorm, 303.8, 121.3},
	{mobBossVelial, 486.3, 306.1},
	{mobBossCerber, 239.1, 434.6},
	{mobBossHekata, 119.9, 211.7},
}

// dungeonReborns40 are the map_4_0 respawn checkpoints (the scene's Reborn_point
// markers, world X,Z). Shared between buildDungeonPack40 -- which keeps a mob-free
// ring around each so you never resurrect into a pack -- and the HuntMap literal.
// (493,64) is the battle-start; the rest activate as the player advances.
var dungeonReborns40 = []Vec2{
	{X: 493.0, Y: 64.0},
	{X: 378.5, Y: 140.2},
	{X: 389.0, Y: 328.0},
	{X: 306.5, Y: 444.5},
	{X: 124.6, Y: 346.8},
}

// dungeonPackSize picks a pack's member count (1..5), skewed BIGGER the deeper the
// region: the entrance yields lone mobs, pairs and the odd trio; Hekata's chamber
// yields full packs of 5 ("по мере продвижения шанс больших групп больше"). The pack
// index cycles a per-depth-tier menu, so it's deterministic and spread across sizes.
func dungeonPackSize(level, pi int) int {
	var menu []int
	switch {
	case level <= 2: // entrance: 1..3
		menu = []int{1, 2, 1, 3, 2}
	case level <= 5: // early
		menu = []int{2, 1, 3, 2, 3}
	case level <= 10: // mid
		menu = []int{2, 3, 4, 3, 2}
	case level <= 15: // mid-deep
		menu = []int{3, 4, 3, 5, 4}
	default: // deep: 3..5
		menu = []int{4, 5, 4, 5, 3}
	}
	if pi < 0 {
		pi = -pi
	}
	return menu[pi%len(menu)]
}

// navClearance returns, for every cell of ng, the Chebyshev distance (in cells ~=
// metres) to the nearest non-walkable cell or map edge -- 0 on walls. Corridor and
// room CENTRES are the local maxima (the medial axis), so ranking pack centres by
// clearance keeps them in the MIDDLE of the walkable space, off walls and railings.
// Standard two-pass chamfer transform (all 8 neighbours cost 1); deterministic.
func navClearance(ng *NavGrid) []int {
	W, H := ng.W, ng.H
	const inf = 1 << 20
	d := make([]int, W*H)
	for j := 0; j < H; j++ {
		for i := 0; i < W; i++ {
			if ng.cellWalkable(i, j) {
				d[j*W+i] = inf
			}
		}
	}
	at := func(i, j int) int {
		if i < 0 || j < 0 || i >= W || j >= H {
			return 0 // outside the map counts as a wall
		}
		return d[j*W+i]
	}
	relax := func(i, j, v int) {
		if v+1 < d[j*W+i] {
			d[j*W+i] = v + 1
		}
	}
	for j := 0; j < H; j++ { // forward: from top-left neighbours
		for i := 0; i < W; i++ {
			if d[j*W+i] == 0 {
				continue
			}
			relax(i, j, at(i-1, j))
			relax(i, j, at(i-1, j-1))
			relax(i, j, at(i, j-1))
			relax(i, j, at(i+1, j-1))
		}
	}
	for j := H - 1; j >= 0; j-- { // backward: from bottom-right neighbours
		for i := W - 1; i >= 0; i-- {
			if d[j*W+i] == 0 {
				continue
			}
			relax(i, j, at(i+1, j))
			relax(i, j, at(i+1, j+1))
			relax(i, j, at(i, j+1))
			relax(i, j, at(i-1, j+1))
		}
	}
	return d
}

// dungeonPack40 is GENERATED from the map_4_0 walkability grid as spread-out PACKS
// seeded on the MEDIAL AXIS: cells are ranked by wall-clearance (navClearance) and
// taken high-to-low, greedily accepting one only if it clears the safe-zone/boss
// rings and sits >= dungeonPackSpacing from every centre already taken. Taking the
// highest-clearance cell in each 28m neighbourhood puts every pack in the MIDDLE of
// its corridor or room, never against a wall or railing. Each centre then gets 1..5
// mobs (skewing bigger with depth) spread on a spiral inside a clearance-capped
// radius, drawn from the nearest region's race-MIXED creature pool + level. The
// four bosses are pinned last. Every position is walkable by construction and
// reachable (the grid is the spawn-connected component), so TestNavGrid40MobsWalkable
// passes. Deterministic (fixed scan + stable sort + index cycling, no rand).
var dungeonPack40 = buildDungeonPack40()

func buildDungeonPack40() []MobSpawn {
	// The first entry MUST be a spawn-relative offset (>9.5m, clear line of sight):
	// TestPathOnRealMap routes to it and the offset rule keeps it from aggroing at
	// battle start. Kept just outside the doubled safe zone (27m, clear LoS down the
	// entrance corridor). Everything generated after it is absolute.
	return append([]MobSpawn{{Mob: mobGhoul, DX: 9.2, DY: 25.4, Level: 1}},
		buildPacks(navGrid40, dungeonRegions(), dungeonReborns40, dungeonBosses)...)
}

// buildPacks lays out discrete mob PACKS on ng's medial axis and pins the bosses,
// keeping a mob-free ring around the player start, every respawn checkpoint and every
// boss. Shared by the crypt (map_4_0, dungeonPack40) and the invasion lair (map_4_1,
// invasionPack41). Pass 1 seeds pack centres on the medial axis (navClearance); Pass 2
// builds a 1..5-member pack (skewing bigger with region depth) at each centre from the
// nearest region's creature pool + level; the bosses are pinned last. Every position is
// walkable by construction and reachable (the grid is the spawn-connected component), so
// the per-map mob-walkability tests pass. Deterministic (fixed scan + stable sort).
func buildPacks(ng *NavGrid, regions []dungeonRegion, reborns []Vec2, bosses []bossPlacement) []MobSpawn {
	sx, sy := ng.Spawn()

	// clearOf reports whether a point is far enough from the start, every respawn
	// checkpoint, and every boss to host a pack.
	clearOf := func(wx, wy float64) bool {
		if math.Hypot(wx-sx, wy-sy) < dungeonSpawnClear {
			return false
		}
		for _, r := range reborns {
			if math.Hypot(wx-r.X, wy-r.Y) < dungeonRebornClear {
				return false
			}
		}
		for _, b := range bosses {
			if math.Hypot(wx-b.x, wy-b.y) < dungeonBossClear {
				return false
			}
		}
		return true
	}
	var out []MobSpawn

	// Pass 1: seed pack centres on the medial axis (see navClearance). Collect every
	// walkable cell whose clearance clears dungeonMinClear and the rings, rank them
	// clearance-high-to-low, then greedily accept while >= dungeonPackSpacing apart.
	// Highest-clearance-first means each neighbourhood's centre is its local clearance
	// peak -- the middle of the corridor/room. Stable sort => deterministic ties.
	clr := navClearance(ng)
	type cand struct {
		x, y float64
		clr  int
	}
	var cands []cand
	for j := 0; j < ng.H; j++ {
		for i := 0; i < ng.W; i++ {
			cv := clr[j*ng.W+i]
			if cv < dungeonMinClear {
				continue
			}
			wx, wy := ng.cellCenterX(i), ng.cellCenterY(j)
			if !clearOf(wx, wy) {
				continue
			}
			cands = append(cands, cand{wx, wy, cv})
		}
	}
	sort.SliceStable(cands, func(a, b int) bool { return cands[a].clr > cands[b].clr })
	var centres [][3]float64 // x, y, clearance(m)
	for _, cd := range cands {
		ok := true
		for _, c := range centres {
			if math.Hypot(cd.x-c[0], cd.y-c[1]) < dungeonPackSpacing {
				ok = false
				break
			}
		}
		if ok {
			centres = append(centres, [3]float64{cd.x, cd.y, float64(cd.clr)})
		}
	}

	// Pass 2: build a pack at each centre. Size (1..5) skews bigger with region depth
	// (dungeonPackSize). Members spread on a Vogel/sunflower spiral ~dungeonMemberSpacing
	// apart -- room to breathe, not a huddle -- scaled down to fit the centre's wall
	// clearance so a tight corridor draws the pack in. Each member is round-then-tested
	// so its STORED coords are proven walkable; one that can't find ground is dropped. A
	// per-region counter cycles that region's (race-mixed) pool, so a pack is a motley
	// band and every listed creature gets used.
	regionCounter := make([]int, len(regions))
	for pi, c := range centres {
		cx, cy := c[0], c[1]
		// Clearance cap so the WHOLE pack stays off the walls, not just the centre.
		effRad := dungeonPackRadius
		if lim := c[2] - 0.6; lim < effRad {
			effRad = lim
		}
		if effRad < 1.0 {
			effRad = 1.0
		}
		rg, ri := nearestRegion(regions, cx, cy)
		size := dungeonPackSize(rg.level, pi)
		// Cap the count to what the local clearance can hold at ~1.8m spacing (the
		// battle server's mobSepRange): otherwise a big pack in a tight corridor would
		// shrink its spiral until bodies overlap. So a narrow corridor gets a SMALLER
		// group, not a squashed one; open halls still get the full 4-5.
		if fit := 1 + int((effRad/1.8)*(effRad/1.8)); size > fit {
			size = fit
		}
		// Vogel spiral: member k at radius spread*sqrt(k). Shrink `spread` if the
		// outermost member would exceed the clearance cap.
		spread := dungeonMemberSpacing
		if maxR := spread * math.Sqrt(float64(size-1)); maxR > effRad && maxR > 0 {
			spread *= effRad / maxR
		}
		for k := 0; k < size; k++ {
			var mx, my float64
			placed := false
			rad := spread * math.Sqrt(float64(k))
			for attempt := 0; attempt < 4; attempt++ {
				ang := float64(k)*2.399963 + float64(attempt)*0.7 // golden angle + nudge
				rx := math.Round((cx+rad*math.Cos(ang))*10) / 10
				ry := math.Round((cy+rad*math.Sin(ang))*10) / 10
				if ng.Walkable(rx, ry) {
					mx, my, placed = rx, ry, true
					break
				}
			}
			if !placed {
				continue
			}
			out = append(out, MobSpawn{
				Mob:   rg.pool[regionCounter[ri]%len(rg.pool)],
				DX:    mx,
				DY:    my,
				Abs:   true,
				Level: rg.level,
			})
			regionCounter[ri]++
		}
	}

	for _, b := range bosses {
		out = append(out, MobSpawn{Mob: b.mob, DX: b.x, DY: b.y, Abs: true})
	}
	return out
}

// invasionReborns41 are the map_4_1 («Логово вторжения») respawn checkpoints: the 5
// canonical Reborn_point markers from the scene bundle (world X,Z). (-140.5,3.1) is the
// battle-start (baked into navGrid41's spawn); the rest activate as the party advances.
// Shared between the pack generator (kept mob-free) and the HuntMap literal.
var invasionReborns41 = []Vec2{
	{X: -140.5, Y: 3.1},
	{X: 150.0, Y: 104.5},
	{X: 69.0, Y: -227.2},
	{X: -208.4, Y: -107.4},
	{X: -50.6, Y: -64.2},
}

// invasionBosses41 pins «Логово вторжения»'s four bosses to the cardinal positions the
// GROUP quest chain (IDS_Quest_Map_4_1_..._PVE_Group_*) names, at the correct minimap
// bearings: Elgorm in the NORTH (his hall behind the stone guardians, Stage 1), Velial
// in the CENTRE by the sacrificial well (Stage 2), Cerber in the EAST at the far end
// (Stage 3, "в самом конце"), Hekata in the SOUTH-WEST at her altar (Stage 4, final).
// Same four-boss ladder as the crypt, but this is the harder co-op version. Each anchor
// is the highest-wall-clearance walkable cell of navGrid41 in that minimap cardinal (a
// roomy arena), computed via the minimap->world transform (tools/navgrid/boss_anchors41).
var invasionBosses41 = []bossPlacement{
	{mobBossElgorm, -95.9, -168.8}, // north
	{mobBossVelial, -113.9, 84.2},  // centre
	{mobBossCerber, -263.9, 53.2},  // east, far end
	{mobBossHekata, 178.1, 109.2},  // south-west
}

// invasionRegions41 themes «Логово вторжения»'s roster by area, matching the quest, and
// is ELITE-ONLY (no common trash -- user spec: "боссов и только элитных мобов, демонов и
// нежить"). Undead elites (Одержимый гуль, скелеты-воители/берсерки/снайперы, Зомби
// губитель) hold the NORTH around the spawn and Elgorm's hall; demon legions -- захватчики
// (Demon_Melee02) and надзиратели (elite Demon_Range) -- plus elite zombies hold the
// CENTRE/EAST/SOUTH-WEST toward Velial, Cerber and Hekata (quest: "демоны... на юге и в
// центре"). Level climbs along the boss ladder (Elgorm ~5 ... Hekata ~20); base elite
// stats + that scaling make this a group-tier fight.
func invasionRegions41() []dungeonRegion {
	undead := []int{mobGhoulPossessed, mobSkeletonBerserk, mobSkeletonWarrior, mobSkeletonSniper, mobZombieBigElite}
	return []dungeonRegion{
		{-140.5, 3.1, 5, undead},   // spawn / north entrance (undead elites)
		{-95.9, -168.8, 6, undead}, // Elgorm's hall, north (undead elites)
		{-113.9, 84.2, 10, []int{mobDemonMeleeElite, mobDemonRangeElite, mobZombieSoldierElite, mobZombieBigElite}},  // Velial, centre (demon+zombie elites)
		{-263.9, 53.2, 15, []int{mobDemonMeleeElite, mobDemonRangeElite, mobSkeletonBerserk, mobZombieSoldierElite}}, // Cerber, east (demon+skeleton elites)
		{178.1, 109.2, 20, []int{mobDemonMeleeElite, mobDemonRangeElite, mobSkeletonWarrior, mobZombieBigElite}},     // Hekata, south-west (demon+skeleton elites)
	}
}

// invasionPack41 is the map_4_1 layout: elite-only demon/undead packs generated on the
// reconstructed navGrid41 (buildPacks) plus the four pinned bosses. Unlike the crypt it
// has no spawn-relative offset mob -- every position is absolute and walkable by
// construction (buildPacks rounds-then-tests), so TestMap41MobsSnapToWalkableFloor's
// clamp is a no-op here.
var invasionPack41 = buildPacks(navGrid41, invasionRegions41(), invasionReborns41, invasionBosses41)

var huntMaps = []HuntMap{
	{
		// map_4_0 = «Подземный город» (undead/demon crypt). Name/Desc/WinDesc are
		// client locale KEYS (SelectGameMenu runs each through GuiSystem.GetLocaleText;
		// a non-key renders as "EMPTY!"). Keys from the client locale (resources.assets).
		ID: 40, Name: "Map_4_0_Name", Scene: "map_4_0", LevelMin: 1, LevelMax: 50,
		Desc:       "Map_4_0_Desc",
		WinDesc:    "Map_4_0_WinDesc",
		MinPlayers: 1, MaxPlayers: 4,
		Nav:    navGrid40, // spawn (Reborn_point) + walkability from real floor meshes
		Spawns: dungeonPack40,
		// The 5 Reborn_point checkpoints from the map_4_0 scene bundle (world X,Z).
		// Shared with the pack generator, which keeps them mob-free.
		Reborn: dungeonReborns40,
	},
	{
		// map_4_1 = «Логово вторжения» (GROUP PvE; demon legions on the lower tiers -- the
		// co-op version of the same four-boss dungeon as the crypt). Walkability
		// RECONSTRUCTED from the client minimap: map_4_1 is the one hunt map with NO authored
		// PassibilityData polygon in any scene bundle (its Reborn markers fall inside no
		// bundle's polygon — it was never exported). navGrid41 is derived from
		// Map_4_1_2029.png (lit floor minus lava); the minimap→world transform is fit to the
		// 5 canonical Reborn_point checkpoints (all land on floor). ~F1 0.75-0.83 vs an
		// authored grid — tighten with a HUNT_CALIBRATE=1 in-game click pass. Roster =
		// Elgorm/Velial/Cerber/Hekata at the quest cardinals + ELITE-ONLY demon/undead packs
		// (invasionPack41). See navgrid_map41.go / tools/navgrid.
		ID: 41, Name: "Map_4_1_Name", Scene: "map_4_1", LevelMin: 1, LevelMax: 50,
		Desc:       "Map_4_1_Desc",
		WinDesc:    "Map_4_1_WinDesc",
		MinPlayers: 1, MaxPlayers: 4,
		Nav:    navGrid41, // minimap-reconstructed; spawn (-140.5,3.1) baked into the grid
		SpawnX: -140.5, SpawnY: 3.1,
		Spawns: invasionPack41,
		Reborn: invasionReborns41,
	},
	{
		// map_4_2 = «Заповедные джунгли» (dinos, tribesmen, a shaman boss). Walkability
		// from the authored polygon (shuffled into the map_4_1 bundle); all 6 of map_4_2's
		// Reborn markers fall inside it and form one connected component. Nav drives spawn.
		ID: 42, Name: "Map_4_2_Name", Scene: "map_4_2", LevelMin: 1, LevelMax: 50,
		Desc:       "Map_4_2_Desc",
		WinDesc:    "Map_4_2_WinDesc",
		MinPlayers: 1, MaxPlayers: 4,
		Nav: navGrid42,
		// Player start confirmed in-game (walk to the spot, read the CLICK target; it is
		// a real Reborn marker ~(35,30)); overrides navGrid42's seed marker (-335.7,102.6).
		// Nav still drives clipping. Shared with the generator's safe ring via jungleSpawn.
		SpawnAt: &jungleSpawn,
		Spawns:  junglePack,
	},
}

// HuntMaps returns all hunt locations in a stable order.
func HuntMaps() []HuntMap { return huntMaps }

// HuntMapByID finds a hunt map by id.
func HuntMapByID(id int32) (HuntMap, bool) {
	for _, m := range huntMaps {
		if m.ID == id {
			return m, true
		}
	}
	return HuntMap{}, false
}

// AvatarXPLevels are the cumulative XP thresholds per avatar level (index =
// 0-based level; the client's PExperiencer XP array must match — it drives the
// XP bar between thresholds). Level N (0-based) is reached at AvatarXPLevels[N].
// The 20 entries (displayed levels 1..20) reproduce the original game's XP curve
// from base_balance/stats.txt: the level-up COST is +100 XP per level up to
// L14→15 (200, 300, ... 1500), then steepens toward the 2000 cap at L19→20 —
// hitting every anchor the original listed (1→2 200, 4→5 500, 8→9 900,
// 14→15 1500, 16→17 1800, 19→20 2000). This is the progression that gates the
// map_4_0 bosses to levels 5/10/15/20 (see LevelPowerMul).
var AvatarXPLevels = []float64{
	0, 200, 500, 900, 1400, 2000, 2700, 3500, 4400, 5400,
	6500, 7700, 9000, 10400, 11900, 13550, 15350, 17220, 19155, 21155,
}

// Per-level power scaling. An avatar's basic-attack damage, skill damage/heals,
// spell power, max health and max mana all grow with its 0-based battle level,
// so a level-20 hero is ~2.1x a level-1 one. This is the curve that lets the
// map_4_0 bosses be beaten at their intended levels (Elgorm 5, Velial 10,
// Cerber 15, Hekata 20): an under-levelled player both hits softer and dies
// faster. The multiplier is exactly 1.0 at level 0 (battle start), so level-1
// combat — and every level-1 test — is unchanged.
const (
	levelCombatGrowth = 0.06 // per level: basic + skill damage/heals, spell power
	levelHealthGrowth = 0.06 // per level: max health + max mana
)

// LevelPowerMul is the damage/heal/spell-power multiplier at a 0-based level.
func LevelPowerMul(level int32) float64 {
	if level < 0 {
		level = 0
	}
	return 1 + levelCombatGrowth*float64(level)
}

// LevelHealthMul is the max-health/max-mana multiplier at a 0-based level.
func LevelHealthMul(level int32) float64 {
	if level < 0 {
		level = 0
	}
	return 1 + levelHealthGrowth*float64(level)
}

// Mob per-level scaling. Regular (non-boss) creatures are authored at level 1 and
// scaled up by their placement level (MobSpawn.Level), so the same creature grows
// tougher and more rewarding the deeper into the map it sits -- "по мере
// продвижения по карте уровень мобов растёт". Level 1 is identity, so the authored
// values ARE the level-1 stats. Bosses are exempt (hand-tuned, see mobs slice).
func MobHPMul(level int) float64  { return 1 + 0.15*float64(mobLvl(level)-1) } // +15%/level
func MobDmgMul(level int) float64 { return 1 + 0.10*float64(mobLvl(level)-1) } // +10%/level

// MobXPMul scales XP bounty. It used to be a steep N x (a level-N mob = N x base
// XP), which made mid-map trash absurdly rich -- a base-16 skeleton at level 5 gave
// 80 XP. Now it grows GENTLY at +40%/level: ~2.6x at level 5 (that skeleton -> ~42),
// still ~8.6x at level 20 so deep grinding keeps the climb to 20 feasible.
func MobXPMul(level int) float64 { return 1 + 0.40*float64(mobLvl(level)-1) }

// MobCoinMul scales COIN bounty GENTLY -- deeper mobs drop a little more bronze,
// tracking toughness (+15%/level, same as HP): a level-2 (reinforced, 92 HP) ghoul
// yields 7 bronze, not 12. Rounded to whole coins in ScaledStats.
func MobCoinMul(level int) float64 { return 1 + 0.15*float64(mobLvl(level)-1) }

func mobLvl(level int) int {
	if level < 1 {
		return 1
	}
	return level
}

// ScaledStats returns a mob's effective combat stats and rewards at a placement
// level. Bosses (which carry Skills) are hand-tuned and returned UNSCALED; every
// other creature scales its level-1 base by the mob multipliers.
func (m Mob) ScaledStats(level int) (maxHP, dmgMin, dmgMax, xp float64, coins int32) {
	if len(m.Skills) > 0 {
		return m.Health, float64(m.DmgMin), float64(m.DmgMax), m.XP, m.Coins
	}
	return m.Health * MobHPMul(level),
		float64(m.DmgMin) * MobDmgMul(level),
		float64(m.DmgMax) * MobDmgMul(level),
		m.XP * MobXPMul(level),
		int32(math.Round(float64(m.Coins) * MobCoinMul(level)))
}
