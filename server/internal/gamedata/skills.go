package gamedata

import "fmt"

// Skill definitions: the server-side "op kit" every avatar skill compiles to.
// The battleserver executes ops; the client only renders what we trigger
// (EFFECT_START fx names from the client's baked 352-entry VisualEffectsMgr
// registry, ADD_EFFECTOR buff icons, stat SYNCs). Authored specs live in
// skills_gen.go (generated from the session's skillspecs/*.json).

// OpKind enumerates the mechanics the battle engine can execute.
type OpKind string

const (
	OpDamage       OpKind = "damage"
	OpDot          OpKind = "dot"
	OpHeal         OpKind = "heal"
	OpHot          OpKind = "hot"
	OpManaRestore  OpKind = "manarestore"
	OpStun         OpKind = "stun"
	OpRoot         OpKind = "root"
	OpSlow         OpKind = "slow"
	OpAttackSlow   OpKind = "attackslow"
	OpSilence      OpKind = "silence"
	OpBuffStat     OpKind = "buffstat"
	OpShield       OpKind = "shield"
	OpLifestealHit OpKind = "lifesteal_hit"
	OpBlink        OpKind = "blink"
	OpDash         OpKind = "dash"
	OpPull         OpKind = "pull"
	OpSummon       OpKind = "summon"
	OpTrap         OpKind = "trap"
	OpChannel      OpKind = "channel"
	OpProc         OpKind = "proc"
	OpAura         OpKind = "aura"
	// OpBounce is a chaining projectile (Elgorm's «Блуждающий ужас» skull, à la
	// Dota's Paralyzing Cask): it strikes one enemy, then hops to the nearest
	// not-yet-hit enemy within Radius after Interval seconds, applying its nested
	// Ops (damage + stun) on each hit, up to Count total hits.
	OpBounce OpKind = "bounce"
	// OpConsumeDots deals Value bonus damage per DoT stack currently on the
	// target, then clears them (ShinDalar's "Вскрытие ран").
	OpConsumeDots OpKind = "consume_dots"
	// OpDrainMaxHP permanently reduces the target mob's max HP by Value(+PerSP), clamping
	// current HP down to match (Abominator's «Пожирание»: «уменьшается количество текущего
	// и максимального здоровья... каждую секунду», nested in the same channel tick as the
	// OpLifestealHit that heals the caster for the same amount). No auto-revert: mobState's
	// maxHealth() has no live/mod-aware lookup (unlike the player's maxHPLocked), so unlike
	// the caster's own temporary max_hp gain this drain lasts the rest of the mob's life --
	// an acceptable simplification since Hunt mobs are typically killed within one encounter.
	OpDrainMaxHP OpKind = "drain_max_hp"
	// OpRevive is a PASSIVE auto-resurrection (Zamaran's «Возрождение»): when the
	// caster would die, if this passive is learned and off its internal cooldown, it
	// resurrects in place instead of dying. Value = HP restored (scaled by powerMul,
	// capped at max HP); Dur = internal cooldown in seconds. Registered at world-build
	// (reviveSlot) and honored in playerDieLocked, not run through applyOpsLocked.
	OpRevive OpKind = "revive"
	// OpImmune is a PASSIVE that makes the caster immune to a crowd-control effect
	// (Wilfang's «Защитный покров» — immunity to root/stun/silence), consumed on the
	// first blocked hit and then unavailable for Dur seconds. Dur = recovery cooldown.
	// Registered at world-build (ccImmuneSlot) and honored by the player-CC gate
	// (ccImmuneBlockLocked); no mob applies player-facing CC today, so it is latent
	// until such a source exists. Not run through applyOpsLocked.
	OpImmune OpKind = "immune"
	// OpHealOnKill is a PASSIVE heal-on-kill (Cerber's «Кровавый пир»): whenever the
	// caster kills an enemy, heal for Value × the KILLED target's max HP, capped at
	// Value2. Registered at world-build (healOnKillSlot) and honored in the mob-death
	// branch (hitMobLocked). Not run through applyOpsLocked.
	OpHealOnKill OpKind = "heal_on_kill"
	// OpManaOnKill is a PASSIVE mana-on-kill siphon (Kiona's «Вдохновение»): whenever the
	// caster kills an enemy, restore Value (+ PerSP × SP) CURRENT mana, scaled by powerMul,
	// clamped to max. Registered at world-build (manaOnKillSlot) and honored in the mob-death
	// branch, mirroring OpHealOnKill. Not run through applyOpsLocked.
	OpManaOnKill OpKind = "mana_on_kill"
	// OpOnKill is a cast-time wrapper (Lirvein's «Изощренный бросок»): its nested Ops
	// run immediately if the cast's primary target died from this cast (ctx.target.dead).
	// If Dur > 0 and the target survived the cast, it instead MARKS the target for Dur
	// seconds («накладывая на цель эффект на N секунд»): if that target later dies from
	// ANY source before the mark expires, the mob-death branch still fires the nested
	// Ops for the original caster, mirroring a delayed kill credit.
	OpOnKill OpKind = "on_kill"
	// OpCooldownReset clears all of the caster's skill cooldowns (Lirvein's on-kill
	// reset). Typically nested under OpOnKill.
	OpCooldownReset OpKind = "cooldown_reset"
	// OpKnockback shoves the targeted enemies away from the caster by Value units
	// (the inverse of OpPull) — Dutnik's «Взрыв» detonation blast.
	OpKnockback OpKind = "knockback"
	// OpStealth cloaks the CASTER (Lirvein's «Единение с ветром», Sandariel's
	// «Сокрывающая вуаль», Astarot's «Слуга тьмы», Wilfang's «Засада»): it sets the
	// player's invisibleUntil for Dur seconds, reusing the same stealth the
	// Invisibility potion grants — mobs stop targeting a hidden avatar, and the
	// stealth breaks the instant the player attacks or casts (revealSelfLocked). The
	// engine had no skill-driven stealth before (only the potion), so these skills
	// previously shipped a cosmetic InvisibilityEffect VFX with no gameplay effect.
	OpStealth OpKind = "stealth"
	// OpExecute is a threshold finisher (Gektor's «Казнь»): if the target's current
	// HP is at or below Value2 (the execute threshold) it dies outright; otherwise it
	// takes Value damage (scaled like OpDamage). Resolved on the single locked target.
	OpExecute OpKind = "execute"
	// OpConsecutiveHit is a PASSIVE marker (Mihalych's «Трепка»): every basic attack that
	// lands on the SAME target as the previous one deals Value extra damage per stack, the
	// streak resetting when the avatar switches targets. Not run through applyOpsLocked --
	// the basic-attack path reads it via consecutiveHitBonusLocked.
	OpConsecutiveHit OpKind = "consecutive_hit"
	// OpOnKillStack grows the caster's flat base attack for every enemy it kills.
	// Two flavours, told apart by Dur:
	//   Dur == 0 -> a PERSISTENT passive (Gellar's «Порабощение» souls): every kill adds
	//     one stack (+Value attack each), capped at Value2 stacks; HalveOnDeath drops half
	//     the souls when the caster dies. Registered at world-build (soulSlot).
	//   Dur  > 0 -> an ACTIVE window (Hekata's «Культ жнеца»): casting it opens a Dur-second
	//     window during which each kill adds +Value flat attack (capped at Value2, 0 =
	//     uncapped); the bonus vanishes when the window ends. Run through applyOpsLocked to
	//     open/refresh the window; the kills are tallied in the mob-death branch.
	// The accumulated attack feeds the basic-attack `flat` channel (killAttackBonusLocked).
	OpOnKillStack OpKind = "on_kill_stack"
	// OpAttackSpeedStreak is a PASSIVE marker (Lirvein's «Неумолимость»): each consecutive
	// basic attack on the SAME target raises attack speed by Value (attacks/sec), capped at
	// Value2, resetting when the avatar switches targets. It shares the hitStreak counter
	// with OpConsecutiveHit and is read when the swing interval is (re)computed, not run
	// through applyOpsLocked. Registered at world-build (attackSpeedStreakSlot).
	OpAttackSpeedStreak OpKind = "attack_speed_streak"
	// OpAttackDamage deals bonus damage equal to Value × the caster's current base attack
	// to the struck target (Gektor's «Разящий удар»: «+{damageCoef%}% урона от своей базовой
	// атаки»). Meant to sit inside a Chance-1 on-hit OpProc so it lands on every swing, next
	// to an OpSlow. Scale flags phys/magic for the hit flavour.
	OpAttackDamage OpKind = "attack_damage"
	// OpShieldExplode marks a TOGGLE that blows up after taking a fixed number of hits
	// (Rognar's «Костяной щит»: «при получении трёх ударов щит взрывается»). While the toggle
	// is on, each incoming hit is counted; on the third the shield deals AoE magic damage to
	// enemies within Radius and switches off. Value/Value2 are the min/max blast (PerSP-
	// scaled); the magnitude decays from max toward min the longer the shield stood («чем
	// меньше времени — тем больше урон»). Handled by the toggle/incoming-hit gates, not run
	// through applyOpsLocked.
	OpShieldExplode OpKind = "shield_explode"
	// OpManaScaledDamage deals magic damage to the single target that scales with how much
	// mana it is MISSING (Neirofim's «Паралич воли»: «чем меньше маны, тем больше урон»):
	// Value base + Value2 × (target.maxMana - target.mana), plus a slow whose strength scales
	// with the target's REMAINING mana (strong when full, none when dry). Manaless (melee)
	// mobs still take the flat base (+PerSP) -- only the mana-dependent bonus and the slow,
	// which have nothing to scale from, are skipped for them.
	OpManaScaledDamage OpKind = "mana_scaled_damage"
	// OpManaBurnHit drains mana from the struck target on a basic attack (sit it inside a
	// Chance-1 on-hit OpProc). Value = amount; Apply "own_mana" makes it Value × the caster's
	// OWN max mana (Neirofim's «Пожирание магии» drains a % of his pool), otherwise a flat
	// burn (BlackDragon's «Выжигание маны»). Value2 = fraction of the drained mana dealt back
	// as magic damage (Neirofim=1, BlackDragon=0). Apply "restore" instead refunds the drained
	// mana to the caster (Inshari's «Изъятие сущности» siphon). Melee mobs have no mana → 0.
	OpManaBurnHit OpKind = "mana_burn_hit"
	// OpSilenceAll silences EVERY hostile mob in the instance for Dur (Neirofim's «Молчание»)
	// and drains Value mana from mobs within Radius of the cast.
	OpSilenceAll OpKind = "silence_all"
	// OpChill applies the Frost «озноб» debuff for Dur; casting it on an ALREADY-chilled
	// target instead STUNS it for Value2 seconds and clears the chill (the signature Frost
	// combo). Value is unused. Resolved on the op's area targets.
	OpChill OpKind = "chill"
	// OpEmpowerNextHit spends Apply-fraction of the caster's CURRENT HP and stores a bonus
	// magic hit of Value × the HP spent onto the next basic attack (Rognar's «Окропление
	// кровью»). Consumed by scheduleHitAfterLocked.
	OpEmpowerNextHit OpKind = "empower_next_hit"
	// OpConsumeSouls halves the caster's banked Gellar souls (soulStacks) — «При применении
	// теряет половину из накопленных душ» (Gellar's «Армия душ»).
	OpConsumeSouls OpKind = "consume_souls"
	// OpDeathLink links the caster to the target for Dur (Rognar's «Канал смерти»): while it
	// holds, Value2 fraction of every blow the caster takes is forwarded to the linked enemy
	// as magic damage (or heals a linked ally). The link's objID/until live on huntState.
	OpDeathLink OpKind = "death_link"
	// OpVisionWard plants a stationary friendly SCOUT object (Urg's «Росток» acorn) at the
	// cast point for Lifetime seconds — a pure vision utility, no damage or CC. The visible
	// prop is Unit (a loadable prefab); it carries a fog-of-war VIEW_RADIUS (Radius) so the
	// whole friendly team sees the patch of terrain around it («открывает небольшой участок
	// местности для всей дружественной команды»), and each world tick it REVEALS any
	// stealthed enemy within Radius — breaking their invisibility (balance patch 1.08's
	// «теперь способность даёт возможность видеть невидимых врагов»). TrapFx is an optional
	// persistent ground fx. Handled by the ward tick, spawned through applyOpsLocked.
	OpVisionWard OpKind = "vision_ward"

	// OpTreeForm turns a friendly avatar (self in solo) into a TREE — Urg's «Древесный
	// камуфляж». It cloaks the target as break-on-move stealth, plants the disguise prop
	// (Unit) at their feet, and ARMS the reveal burst (nested Ops — magic damage + silence)
	// that detonates on the surrounding enemies the moment they leave tree form: by moving
	// («При движении… выходит из формы дерева»), acting, or the stealth expiring. Dur is the
	// camouflage lifetime; the nested Ops carry their own Radius.
	OpTreeForm OpKind = "tree_form"

	// OpGrove grows a ring of trees around the caster — Urg's «Непроглядные дебри». Count
	// props (Unit) are planted on a circle of Radius and stand for Dur seconds; while they
	// stand the sibling OpSilence on the skill keeps enemies inside from casting («враги не
	// могут применять способности»). When the trees FALL (Dur elapses) the nested Ops (a
	// magic-damage burst) hit every enemy still inside («Когда деревья исчезнут, все враги
	// внутри получат магический урон»). Handled through applyOpsLocked; the fall-damage rides
	// the deferred-payload queue and the props expire via the anchor-end queue.
	OpGrove OpKind = "grove"

	// OpSelfRecoil arms a per-attack self-punish window (Sigilion's «Мощь берсерка»):
	// while it holds (Dur seconds), every basic-attack swing costs the caster
	// Value×(the attack's live dmg_pct-attributable bonus) as self-damage («ранит себя
	// на 50% от увеличенного урона от атак»). Value2/Chance unused. Consumed at hit-time
	// in scheduleHitAfterLocked (hs.recoilFrac/recoilUntil), not run through applyOpsLocked.
	OpSelfRecoil OpKind = "self_recoil"

	// OpAttackManaBonus arms a mana-fueled attack window (Miriam's «Зачарованные
	// стрелы»): while it holds (Dur), each basic-attack swing that can afford Value2
	// mana instead deals Value(+PerSP×SP) extra FLAT damage, consuming that mana — a
	// swing with insufficient mana lands with no bonus, no failure. Consumed at
	// hit-time in scheduleHitAfterLocked (hs.manaShot*), not run through applyOpsLocked.
	OpAttackManaBonus OpKind = "attack_mana_bonus"

	// OpCastMark tags every enemy caught in the op's area (Einzenhaim's «Изгнание
	// колдовства»): if the marked unit casts a skill within Dur seconds, it takes an
	// extra Value(+PerSP×SP) damage from the caster who marked it («и при применении
	// любого навыка в течение {duration} секунд получают {*castDamage} урона»). Only
	// bosses actually cast skills today, so the mark is honored in tryBossSkillLocked.
	OpCastMark OpKind = "cast_mark"

	// OpAttackCleave arms a Dur-second window (BlackDragon's «Неистовство»: «нанесение
	// урона нескольким целям») during which every basic-attack swing ALSO strikes every
	// other enemy within Radius of the primary target for the same damage, instead of
	// hitting only the one target. Consumed at hit-time in scheduleHitAfterLocked
	// (hs.cleaveUntil/cleaveRadius), not run through applyOpsLocked for the per-hit part.
	OpAttackCleave OpKind = "attack_cleave"

	// OpManaBurnArea drains Value mana from every enemy within Radius of the cast
	// (PlusMinus's «Шаровая молния»: «сжигая {manaDamage} единиц маны всем врагам» in the
	// blast). Unlike OpManaBurnHit (a single struck target on a basic attack), this is a
	// one-shot AoE burn with no damage-back component -- mirrors the radius-drain half of
	// OpSilenceAll without the silence.
	OpManaBurnArea OpKind = "mana_burn_area"

	// OpManaScaledAttackSpeed is a PASSIVE marker (Sharli's «Жар души»): attack speed
	// rises by Value PERCENT for every 10% of the caster's CURRENT mana pool that is
	// missing («увеличение базовой скорости атаки на {additionalSpeed} за каждые 10%
	// отсутствующей маны») -- a live, continuously-reevaluated bonus (full mana = none,
	// empty mana = 10× Value%), not a fixed buff. Not run through applyOpsLocked;
	// registered at world-build (manaSpeedSlot) and read each time the swing interval is
	// (re)computed (attackPeriodLocked).
	OpManaScaledAttackSpeed OpKind = "mana_scaled_attack_speed"

	// OpCleanseOnHit is a PASSIVE that sheds every hostile effect on the caster the
	// instant a negative ability lands on them (Abominator's «Окоченение»: «При
	// применении негативной способности на Абоминатора, он скидывает все враждебные
	// воздействия»). Registered at world-build (cleanseSlot) and honored by the
	// player-facing-debuff-apply gate (cleansePlayerDebuffsLocked) — LATENT today like
	// OpImmune, since no mob or avatar currently applies CC to a player avatar. Not run
	// through applyOpsLocked.
	OpCleanseOnHit OpKind = "cleanse_on_hit"

	// OpHitStack arms a Dur-second stacking window on cast (Gayal's «Меч жажды»: «каждый
	// удар увеличивает похищение жизни на 5% и скорость атаки... до 5 ударов»): every basic
	// attack that lands while it holds adds one stack, each worth +Value lifesteal_pct and
	// +Value2 attack_speed_pct, up to Count stacks; on reaching Count the landing swing also
	// deals StackBurstDamage(+PerSP×SP) bonus magic damage and the stacks reset to 0 (the
	// window itself keeps running, so a fresh climb to Count can burst again before it
	// expires). Consumed at hit-time in scheduleHitAfterLocked (hs.hitStack*), not run
	// through applyOpsLocked for the per-hit part -- only the initial arming is.
	OpHitStack OpKind = "hit_stack"

	// OpMoveChargeAttack is a PASSIVE marker (Sandariel's «Острие странника»): banks a
	// charge every Value2 units the avatar walks (up to Count charges), each charge
	// adding Value flat damage to the caster's NEXT basic attack; the stack resets to 0
	// the instant that attack lands. Registered at world-build (huntState.moveChargeSlot);
	// distance is accumulated in the per-tick upkeep, consumed in scheduleHitAfterLocked.
	OpMoveChargeAttack OpKind = "move_charge_attack"

	// OpChainHeal is Kiona's «Лечебная волна»: hops between up to Count living allies
	// (self first), healing each for Value(+PerSP×SP) and dealing Value2 magic damage to
	// enemies within Radius of whichever ally is currently being healed -- "восстанавливающую
	// здоровья каждой цели... враги, находящиеся вблизи исцеляемой цели, получают урона
	// при каждом скачке".
	OpChainHeal OpKind = "chain_heal"

	// OpDamageShare arms Kiona's «Лесной покров» damage-redirect: while live, Value
	// fraction of any damage the marked unit (enemy mob OR ally player -- whichever this
	// cast resolved) takes is ALSO dealt as healing to every living ally within Radius of
	// it, credited to this op's caster. Honored in hitMobFlagsLocked/hitPlayerFromLocked
	// via the shared unitStatus.cloak* fields.
	OpDamageShare OpKind = "damage_share"

	// OpRevealTarget arms Velial's «Трибунал»: the marked target stays fully revealed to
	// the caster's whole team for the duration, bypassing the normal distance-based fog/
	// hide radius (mobViewDistLocked) -- "цель становится видна в невидимости для всех
	// членов союзной команды". Mobs have no self-stealth today, so this is currently only
	// observable as an always-visible/un-shaded mark; the primitive is real and tested
	// independent of that (mirrors OpImmune/OpCleanseOnHit's "latent but genuine" pattern).
	OpRevealTarget OpKind = "reveal_target"

	// OpZoneArmor is a TOGGLE-only marker (Inshari's «Угнетение»): while the toggle is on,
	// Value is an armor multiplier that applies ONLY against an attacker standing OUTSIDE
	// Radius of the caster -- "враги, находящиеся вне зоны действия, наносят пониженный
	// урон" (an attacker fighting her INSIDE the aura hits at her normal, unbuffed armor).
	// Armed via huntState.zoneArmorSlot in toggleSkillLocked/toggleOffLocked, read live in
	// hitPlayerFromLocked (replaces the generic, unconditional "armor_pct" stat for this
	// specific skill).
	OpZoneArmor OpKind = "zone_armor"

	// OpMeleeForm is a TOGGLE-only marker (Grimlok's «Темная сторона»): while the toggle is
	// on, the avatar's effective auto-attack range drops to meleeReach and its projectile
	// pool is suppressed ("изменяя атаку с дальней на ближнюю"), restored the instant the
	// toggle turns off. Armed via huntState.meleeFormSlot.
	OpMeleeForm OpKind = "melee_form"
)

// PerLevel holds one value per skill RANK. Slots 1-3 carry 5 ranks, the ult
// (slot 4) carries 4 (see normalizeKits / the levels gating in the battleserver).
// A shorter array is fine -- At() clamps and un-authored ranks reuse the last.
type PerLevel []float64

// At returns the value for a 1-based skill rank, clamped to [1,len]. An empty
// PerLevel yields 0 (a skill that does not use this field).
func (p PerLevel) At(level int) float64 {
	if len(p) == 0 {
		return 0
	}
	if level < 1 {
		level = 1
	}
	if level > len(p) {
		level = len(p)
	}
	return p[level-1]
}

// Op is one atomic effect. Which fields matter depends on Kind (see the
// authoring contract in the session scratchpad skill_spec.md).
type Op struct {
	Kind   OpKind
	Value  PerLevel // damage / heal / factor / dash speed ...
	Value2 PerLevel // secondary (lifesteal fraction, ...)
	Dur    PerLevel // seconds; 0 on a passive buffstat = permanent
	Chance PerLevel // proc chance 0..1

	Scale          string // "phys" | "magic" | "pure" | ""
	Radius         float64
	PerSP          float64 // damage/heal added per point of spell power (from bonus_per_sp/per_sp)
	// PctOfAttack multiplies the caster's LIVE base attack power (hs.baseAttackLocked) by a
	// per-rank coefficient onto an OpDamage, so the hit tracks gear/buffs instead of a flat
	// per-rank number baked in at authoring time (Nerlag's «Мясорубка»: client's
	// «{aoeDamageCoef}% от базовой атаки»). Additive with Value, not a replacement for it.
	// Empty/zero = no attack-power term (back-compat default).
	PctOfAttack    PerLevel
	// PushAside, on an OpDash, shoves every enemy along the charge's straight-line path
	// this distance away from that line at the moment the dash starts (Zamaran's «Таран»:
	// client's «отпихивая всех врагов в стороны» -- distinct from the arrival-point
	// slow+damage AoE, which already exists as separate ops). 0 = no push (back-compat
	// default). No numeric value is given in the locale text, so this reuses knockbackMobLocked's
	// own modest default distance.
	PushAside      float64
	Apply          string  // "self" targets the caster instead of enemies (health-cost damage)
	Stat           string  // buffstat: dmg_pct, phys_armor, ... (see statMod)
	On             string  // buffstat: "self" | "target"
	BonusMissingHP PerLevel
	// RefundIfHit, on an OpDamage with Apply=="self", skips the self-inflicted cost entirely
	// when the cast's paired enemy-facing hit actually connected (ctx.target alive at
	// application time) -- Abominator's «Бросок плоти»: «теряет здоровье, которое он может
	// восстановить, ударив цель». false = the self-cost always applies (back-compat default).
	RefundIfHit bool
	// GrantsCCImmune, on an OpShield, makes the shielded unit (self or the ally On:"ally"
	// targets) fully immune to stun/root/silence/slow for as long as the shield lasts
	// (Arianna's «Щит хранителя»: «щит дает полную неуязвимость... к оглушению и
	// замедлению»). Reuses the shield's own Dur as the immunity window (huntState.
	// tempCCImmuneUntil, checked in ccImmuneBlockLocked alongside the permanent passive).
	// false = an ordinary shield with no CC immunity (back-compat default).
	GrantsCCImmune bool
	// CasterMissingHP adds flat bonus damage = value × the CASTER's missing-HP fraction
	// (Velial's «Воля к победе»: fights harder the closer to death). Added after all
	// multipliers, so it is NOT scaled by power/attack buffs -- it matches the observed
	// in-game values directly (~100 × missing at ranks 4-5).
	CasterMissingHP PerLevel
	// MaxTargets, when >0, caps a damaging/CC op to the N nearest enemies in its area
	// (Rognar's «Могильный холод» hits only two). 0 = no cap (hit everything in range).
	MaxTargets int
	// Randomize shuffles the in-radius candidate list before a MaxTargets cap truncates it,
	// so the N hit are a random subset instead of the N nearest (Rognar's «Могильный холод»:
	// «волны холода на двух СЛУЧАЙНЫХ целях» -- explicitly random, not nearest-first).
	// Ignored when MaxTargets==0. false = nearest-first (back-compat default).
	Randomize bool
	// ExcludeCenterTarget drops the AoE's own center target (ctx.target, when the center is
	// a struck/aimed unit rather than a point) from this op's candidate list before
	// MaxTargets/PerTargetDecay apply (Titanid's «Ударная волна»: «все ДРУГИЕ враги вокруг
	// него» explicitly excludes the primary target from the splash/slow; PlusMinus's
	// «Сверхпроводимость»: the chain's «четырем СОСЕДНИМ целям» excludes the already-struck
	// mob it centers on). false = the center target may be re-selected as one of its own
	// splash targets (back-compat default).
	ExcludeCenterTarget bool
	// PerTargetDecay reduces an OpDamage's magnitude by this fraction for each successive
	// target in MaxTargets' nearest-first order (PlusMinus's «Сверхпроводимость»: a
	// 4-target chain, «каждый следующий удар на 20% слабее предыдущего»). Requires
	// MaxTargets > 0 (the ordering it decays along). 0 = flat damage to every target
	// (back-compat default).
	PerTargetDecay float64
	// HalveOnDeath drops half of a persistent OpOnKillStack's accumulated stacks when the
	// caster dies (Gellar's souls — «При смерти теряет половину из накопленных душ»).
	HalveOnDeath bool
	// TargetSide gates a friend-or-foe DUAL cast: "enemy" ops fire only when the aimed unit
	// is an enemy (ctx.target set), "ally" ops only when it is a friend (ctx.allyTarget set).
	// Empty = fire regardless. Lets one skill do X to a foe and Y to a friend (Kiona's «Страж
	// леса», Frost's «Гробница холода», Hekata's «Выбор скверны»).
	TargetSide string
	// PerSoul adds Value×soulStacks bonus damage to a damage op (Gellar's «Армия душ» scales
	// with banked souls). 0 = no soul scaling.
	PerSoul PerLevel
	// PerSoulSP adds PerSoulSP×spellPower×soulStacks on top of PerSoul (Gellar's «Армия
	// душ»: client's «{*damagePerSoul}+{*@damageSoulSP}» -- the per-soul term is ALSO
	// SP-scaled, on top of the flat op's own PerSP). 0 = no SP scaling on the soul term.
	PerSoulSP float64

	// SelfMaxHPPct makes an OpDamage or OpHeal's magnitude a fraction of the CASTER's own
	// live max HP instead of a flat per-rank table (Gayal's «Поглощение жизни»: «5% от
	// своего максимального здоровья» dealt and healed each channel tick). 0 = use Value as
	// authored (back-compat default).
	SelfMaxHPPct PerLevel

	// StackBurstDamage is the bonus magic damage OpHitStack's swing deals when its stacks
	// reach Count (Gayal's «Меч жажды»: «наносит {damage}+{@damageSP} урона» on the 5th
	// hit). PerSP scales it with spell power, matching every other magic burst.
	StackBurstDamage PerLevel

	// StackCap caps the TOTAL live value an OpBuffStat can accumulate on the same target
	// from repeated procs of the SAME skill (Titanid's «Каменная кожа»: armor keeps
	// hardening on every hit, «но не более чем на {armourMax}»). Empty/0 = uncapped
	// (back-compat default). Checked in addPlayerModLocked against the sum of currently
	// live mods sharing this op's stat and cast source; a new stack is clamped to the
	// remaining headroom (0 headroom = the stack is dropped, not appended).
	StackCap PerLevel

	// DecayTo makes a DoT/Slow's magnitude decay LINEARLY from Value at application down to
	// this value by the time its Dur expires (Rognar's «Могильный холод»: «замедление и
	// урон постепенно спадает за время действия»). Empty = flat, no decay (back-compat
	// default). On OpDot it is the terminal per-second damage; on OpSlow the terminal move
	// factor (1 = no slow left). Read once at application into overTime/unitStatus, then
	// interpolated at tick/read time -- see overTime.currentPerSec and moveFactor.
	DecayTo PerLevel

	// Growth adds Value×(pulse index, 0-based) to an OpChannel's nested op on every tick:
	// on a nested OpDamage it ramps damage (Miriam's «Убийственный залп»: «каждую секунду
	// урон... дополнительно увеличивается»); on a nested OpStun it ramps the stun duration
	// (Titanid's «Землетрясение»: each of the 3 waves stuns 0.2s longer than the last). 0 =
	// flat, no ramp (back-compat default). Read once into channelState.growth/stunGrowth at
	// OpChannel creation, applied per-pulse in tickChannelsLocked.
	Growth PerLevel
	// RadiusGrowth widens an OpChannel's nested op's AoE by this many units per pulse
	// (Titanid's «Землетрясение»: «каждая следующая волна шире, чем предыдущая»). 0 = flat
	// radius (back-compat default). Flat, not per-rank, since the client text gives no
	// per-rank figure, only "wider each wave". Read once into channelState.radiusGrowth,
	// applied per-pulse as opCtx.radiusBonus.
	RadiusGrowth float64
	// GrowthPerSP adds GrowthPerSP×(pulse index, 0-based)×spellPower on top of Growth's
	// flat per-pulse ramp (Titanid's «Землетрясение»: client's «{aoeDamageInc}+{damageIncSP}
	// больше урона» -- the wave-to-wave INCREMENT itself is also SP-scaled, not just the
	// base hit). 0 = the ramp has no SP term (back-compat default).
	GrowthPerSP float64

	// GrowthPerEnemy adds Value×(live enemy count within this op's own Radius) bonus
	// damage to an OpChannel's nested OpDamage every pulse (Avrora's «Освященное место»:
	// «чем больше союзников вокруг, тем сильнее эффект» -- the solo engine has no allies
	// to count, so this counts enemies caught in the same zone instead, the nearest
	// available analog). 0 = flat damage regardless of occupancy (back-compat default).
	GrowthPerEnemy PerLevel

	// MissingHPLinear/DamageCap implement a true execute (Inshari's «Возмездие»): damage =
	// MissingHPLinear × (victim's missing HP), clamped to DamageCap(+PerSP×spellPower).
	// MissingHPLinear==0 leaves OpDamage using Value as authored (back-compat default) --
	// this REPLACES the base Value formula rather than modifying it, same as SelfMaxHPPct.
	MissingHPLinear PerLevel
	DamageCap       PerLevel

	// DmgPctOfAttack, on an OpSummon, makes each spawned unit's Dmg a fraction of the
	// OWNER's live base attack at spawn time (Lirvein's «Вендетта» daggers: «30% от
	// базовой атаки») instead of a static per-rank Dmg table. 0 = use Dmg as authored.
	DmgPctOfAttack float64
	// LastDouble doubles the FINAL spawned unit's Dmg (Lirvein's «Вендетта»: «последний
	// кинжал нанесёт удвоенный урон»).
	LastDouble bool
	// HpPctOfOwner, on an OpSummon, makes each spawned unit's HP a fraction of the OWNER's
	// live max HP at spawn time (Anhel's clones: «треть жизни и силы атаки самого Анхеля»)
	// instead of a static per-rank HP table. 0 = use HP as authored.
	HpPctOfOwner float64
	// HpPerSP/DmgPerSP add the caster's live spellPower×value on top of an OpSummon's HP/Dmg
	// tables (Gayal's raised zombies: client's «{healthMod}+{damageHealthSP}» /
	// «{dmgMod}+{damageSP}» -- both stats carry their own SP-scaled additive term, mirroring
	// how OpDamage/OpHeal already read PerSP). 0 = no SP scaling on that stat (back-compat
	// default).
	HpPerSP, DmgPerSP float64

	// TargetIsolated gates an op so it fires ONLY when the aimed enemy is ALONE — no other
	// living enemy of its own side within TriggerRadius of it (Vigilans's «Свидание со
	// смертью»: «Если цель атаки не находится рядом с союзниками, то каждый удар наносит
	// увеличенный урон»). Sat inside a Chance-1 on-hit OpProc so the bonus rides every basic
	// attack, but only against a target caught away from its pack. TriggerRadius = the
	// "рядом" distance (a sensible default is used when 0).
	TargetIsolated bool

	// ExplodeOnDeath arms an OpDot's target to detonate if it dies while still poisoned
	// (Wilfang's «Ядовитый укус»: «если цель умрёт, находясь под этим эффектом,
	// произойдёт взрыв, наносящий магический урон по области»): an AoE magic burst of
	// ExplodeDamage(+PerSP×SP) centred on the corpse, radius ExplodeRadius, credited to
	// this op's caster. false = a plain DoT with no death detonation (back-compat default).
	ExplodeOnDeath bool
	ExplodeDamage  PerLevel
	ExplodeRadius  float64
	// ExplodeSP scales ONLY the death explosion's damage by spellPower, decoupled from this
	// op's own PerSP (which, on an ExplodeOnDeath OpDot, instead scales the periodic TICK
	// damage). Wilfang's «Ядовитый укус» needs the two to differ: the tick itself deals no
	// damage the client ever describes (Value=0, PerSP=0), while the death burst is the
	// skill's one real, SP-scaled number. 0 = the explosion has no SP term of its own
	// (back-compat default).
	ExplodeSP float64

	// MeleeOnly restricts an OnDamaged OpProc to fire only when the STRIKING mob is a
	// MELEE attacker (AttackRange <= 0) -- BlackDragon's «Кровь дракона»: «При ближней
	// атаке по дракону...». A ranged mob's hit does not trigger it. Ignored when false
	// (fires on any attacker, the back-compat default).
	MeleeOnly bool
	// BasicAttackOnly restricts an OnDamaged OpProc to fire only when the incoming hit was
	// a plain basic attack, NOT a mob/boss SKILL cast (Gektor's «Реванш»: «При получении
	// урона от базовой атаки...» -- an explicit scope, unlike Titanid/Nerlag's unscoped
	// OnDamaged procs). Ignored when false (fires regardless of source, back-compat default).
	BasicAttackOnly bool
	// SkillOnly is the opposite restriction: the proc fires only when the incoming hit came
	// from a mob/boss SKILL cast, not a basic attack (Neirofim's «Обращение энергии»:
	// «Каждое направленное на Нейрофима заклинание...» -- only spells trigger it).
	SkillOnly bool

	// OnDamaged marks a PASSIVE OpProc that fires when the avatar is STRUCK rather than
	// when it hits (Titanid's «Каменная кожа» hardens on being hit; Gektor's «Реванш»
	// counter-novas the attacker; Dutnik's «Детонация» cooks off ammo when damaged). The
	// world-build proc split reads this instead of a hard-coded prefab list. When true and
	// Value>0, the nested heal/damage ops may also read the size of the hit that triggered
	// them (Nerlag's «Прилив крови» heals for the damage just taken).
	OnDamaged bool
	// OnKill marks a PASSIVE OpProc that fires whenever the avatar's side KILLS an enemy
	// (Gayal's «Аура погибших»: «При смерти врагов... 15% шанс, что из них восстанут
	// упыри» -- gated on the victim's DEATH, not on landing a hit). Registered into
	// hs.killProcs and rolled once per kill from the mob-death branch, centred on the
	// slain mob's position, distinct from the on-hit (hs.procs) and on-struck
	// (hs.defenseProcs) proc lists.
	OnKill bool
	// OnAnyDamage widens a normal on-hit OpProc (still rolled from basic-attack landings
	// via hs.procs) so it ALSO rolls once per cast when the caster's own ACTIVE skill
	// damage lands (Anhel's «Зов фантомов»: client's generic «При нанесении урона» --
	// "upon DEALING damage" -- unlike every other proc's attack/strike-specific wording,
	// implying any damage she deals, not just her basic attack). Registered into a second
	// list, hs.anyDamageProcs, alongside the normal hs.procs entry.
	OnAnyDamage bool

	// summon
	Unit                     string // loadable character prefab
	Count, Lifetime, HP, Dmg PerLevel
	// Stationary marks a summon that HOLDS its spawn point instead of seeking/escorting
	// (Morlokai's «Грозовой тотем»): it never moves, and attacks nearby enemies with a
	// ranged zap rather than walking into melee. A killable stationary turret.
	Stationary bool
	// SummonFx is a persistent VFX attached to (owned by) the spawned summon for its
	// lifetime (Frost's «Исчадие мерзлоты» ice aura belongs to the elemental, not the
	// caster). Started once on spawn parented to the unit, so it dies with the body.
	SummonFx string
	// Pet marks a persistent COMMANDED companion rather than a fire-and-forget swarm.
	// Grimlok's dinosaur is the one: it lives 180s (the others are 15-30s swarms of
	// 1-3), so it is a unit the player keeps and directs, not a burst of temporary
	// bodies. A pet: (1) is unique -- re-casting the skill replaces it instead of
	// stacking; (2) obeys the OWNER'S ORDERS (it attacks what the player ordered an
	// attack on and walks where the player ordered a move) instead of seeking the
	// nearest enemy on its own; (3) does not escort the owner -- with no order it
	// simply holds, because its position is the player's to decide.
	Pet bool

	// trap / channel / aura
	TriggerRadius float64
	Interval      float64
	TickCost      PerLevel
	TrapFx        string
	TriggerFx     string

	// DotFx is a persistent status VFX attached to a mob for the lifetime of a DoT
	// applied by this op (e.g. Shin Dalar's acid), owned by the mob so it stays on
	// it; ended once every DoT on that mob clears.
	DotFx string

	// OpDash modifiers for a "charge" lunge:
	//   NoClip          -- drive straight to the target THROUGH obstacles (a leap),
	//                      instead of stopping the lunge at the first wall.
	//   StrikeOnArrival -- defer the ops that FOLLOW this dash in the batch until the
	//                      dash lands, so damage/root/etc. hit on impact, not on cast.
	NoClip          bool
	StrikeOnArrival bool

	// BreakOnMove marks an OpStealth whose invisibility drops the moment the player issues a
	// MOVE order, not only on the next attack/cast (Wilfang's «Засада»: an ambush you must
	// hold in place -- stepping out of position reveals you). Left false on stealth that
	// persists through movement (Lirvein's «Единение с ветром»). Read at grant time into
	// huntState.stealthBreaksOnMove.
	BreakOnMove bool

	// PerTargetGrowth adds Value×(target index, 0-based, nearest-to-caster-first) flat bonus
	// damage to an OpDamage hitting multiple targets in one cast (Nerlag's «Метание топоров»:
	// «дополнительного урона каждому последующему врагу, которого они заденут» -- the
	// OPPOSITE of PerTargetDecay, growth instead of falloff). PerTargetGrowthSP adds the same
	// index×spellPower term on top (the client's «{*aoeDamageInc}+{*@damageIncSP}»). Both 0 =
	// flat damage to every target (back-compat default). Forces opTargetsLocked to sort
	// nearest-to-CASTER first (a beam/line hit's natural "first, second, third..." order).
	PerTargetGrowth   PerLevel
	PerTargetGrowthSP float64

	// VictimMaxHPPct makes an OpDot's per-second magnitude a fraction of the TARGET's own
	// live max HP instead of a flat per-rank table (Elgorm's «Оскверненная почва»: «получают
	// магический урон равный {damage%}% от своего максимального здоровья»). 0 = use Value as
	// authored (back-compat default). Sibling of SelfMaxHPPct, which instead reads the
	// CASTER's own max HP on OpDamage/OpHeal.
	VictimMaxHPPct PerLevel

	// ScalePerHit rescales an OpBuffStat's authored Value by however many enemies the SAME
	// cast's preceding OpDamage actually hit (Nerlag's «Поголовная бойня»: «...за каждого
	// врага, попавшего в область действия» -- the buff grows or shrinks with the landing
	// hit count, not a flat per-rank number). Value is read as the PER-HIT unit: for a "_pct"
	// stat it is the multiplier ONE hit would grant (e.g. 1.05 = +5%), combined as
	// 1+(Value-1)×N; for any other stat it is a flat per-hit amount, combined as Value×N.
	// N is ctx.hitCount, set by the OpDamage op earlier in the same ops list; 0 hits = no
	// buff at all. False (default) leaves Value untouched (back-compat).
	ScalePerHit bool

	Ops []Op // nested ops for trap/channel/proc/aura
}

// Skill is one authored skill slot.
type Skill struct {
	Slot   int
	NameRu string
	Type   string // "ACTIVE" | "TOGGLE" | "PASSIVE"

	Target    string // SkillTarget '+'-joined flags ("" = instant self-cast)
	Targeting string // SkillTargeting flags (preview only)
	Distance  int
	AoERadius int
	AoEWidth  int

	// ManaCost/Cooldown carry one entry per rank -- the array LENGTH defines the
	// skill's max rank (5 for slots 1-3, 4 for the ult), which the battleserver
	// turns into the client's `levels` gating array.
	ManaCost []int
	Cooldown []int

	CastFx       string
	CastFxDur    float64
	PayloadFx    string
	PayloadFxAt  string // "target" | "point" | "self"
	PayloadDelay float64
	BuffFx       string
	BuffFxOn     string // "self" | "target"
	// GrowFx is a per-level self VFX base name for a passive that enlarges the
	// model (Titanid's "Гигантизм"): the client's MorphEffect on the fx prefab
	// scales the parented avatar root, and each level uses a progressively larger
	// prefab GrowFx+level (e.g. "TitanidSkill4Effect"+"1".."4"). "" = no grow.
	GrowFx string

	BuffIcon        bool
	BuffDescVariant string // "BuffSelf" | "BuffTarget" (locale desc variant)

	Ops     []Op
	TipArgs map[string]PerLevel
}

// MaxDur returns the longest op duration at the given level (buff icon timer).
func (s Skill) MaxDur(level int) float64 {
	var d float64
	for _, op := range s.Ops {
		if v := op.Dur.At(level); v > d {
			d = v
		}
		if v := op.Lifetime.At(level); v > d {
			d = v
		}
	}
	return d
}

// AvatarSkills is the full authored kit of one avatar.
type AvatarSkills struct {
	Prefab           string
	AttackProjectile bool // has a working SET_PROJECTILE pool in its bundle
	Skills           [4]Skill
}

// skillsByPrefab is populated by skills_gen.go (generated data).
var skillsByPrefab = map[string]*AvatarSkills{}

// SkillsFor returns the authored kit for a prefab, or a uniform-nuke fallback
// so an avatar missing from the generated data stays playable.
func SkillsFor(a Avatar) *AvatarSkills {
	if ks, ok := skillsByPrefab[a.Prefab]; ok {
		return ks
	}
	return defaultSkills(a)
}

// defaultSkills mirrors the legacy Phase-B uniform nukes.
func defaultSkills(a Avatar) *AvatarSkills {
	ks := &AvatarSkills{Prefab: a.Prefab}
	for i := 0; i < 4; i++ {
		slot := i + 1
		mana := make([]int, 4)
		cd := make([]int, 4)
		dmg := make(PerLevel, 4)
		for l := 1; l <= 4; l++ {
			mana[l-1] = 20 + 10*slot + 5*(l-1)
			cd[l-1] = 4 + 2*slot
			dmg[l-1] = float64(50+25*slot) + 20*float64(l-1) + a.SpellPower
		}
		ks.Skills[i] = Skill{
			Slot:         slot,
			Type:         "ACTIVE",
			Target:       "ENEMY+NOT_BUILDING",
			Targeting:    "TARGET",
			Distance:     8,
			ManaCost:     mana,
			Cooldown:     cd,
			PayloadDelay: 0.3,
			Ops:          []Op{{Kind: OpDamage, Value: dmg, Scale: "magic"}},
		}
	}
	normalizeKit(ks)
	return ks
}

// rangedBasicAttackRange is the auto-attack reach an avatar needs to actually
// fight at range. Several ranged DPS (Miriam, Lirvein, Sandariel, Dutnik, Grimlok,
// Teridin) are Killer-class and inherit that template's melee AttackRange (2.5,
// see statsFor), yet their kit fires a basic-attack projectile -- so without an
// override they chase a mob to point-blank before "shooting". Matches the caster
// template's 6.0 so every projectile basic-attacker shares one ranged reach.
const rangedBasicAttackRange = 6.0

// registerSkills is called by generated data; duplicate registration is a
// programming error caught at init.
func registerSkills(ks *AvatarSkills) {
	if _, dup := skillsByPrefab[ks.Prefab]; dup {
		panic(fmt.Sprintf("gamedata: duplicate skills for %s", ks.Prefab))
	}
	normalizeKit(ks)
	skillsByPrefab[ks.Prefab] = ks
	// A projectile basic-attacker must fight at range: raise any avatar that
	// inherited a shorter melee AttackRange from its class template. This runs at
	// init() (after buildAvatars), so the roster slice is already built and every
	// consumer -- auto-attack reach, the SYNC'd attackRange stat, the tooltip
	// distance -- reads the corrected value.
	if ks.AttackProjectile {
		for i := range avatars {
			if avatars[i].Prefab != ks.Prefab {
				continue
			}
			if avatars[i].AttackRange < rangedBasicAttackRange {
				avatars[i].AttackRange = rangedBasicAttackRange
			}
			avatars[i].AttackWindup = rangedAttackWindup(ks.Prefab)
		}
	}
	fixupSkillData(ks)
}

// rangedAttackWindup is the fraction of a projectile basic-attack swing the caster
// spends winding up before the bolt is loosed. Every ranged hero looses at the END
// of the draw/throw animation (0.65) so the projectile flies and lands late, matching
// the swing -- without it the bolt snapped out at the very start. Plus-Minus is the
// sole exception: its bolt leaves in the MIDDLE of the animation.
func rangedAttackWindup(prefab string) float64 {
	if prefab == "Avtr_Dsb_PlusMinus" {
		return 0.5 // mid-animation release (exception)
	}
	return 0.65 // end-of-animation release (all other ranged heroes)
}

// fixupSkillData corrects authored skill fields (timings, geometry) that desync
// from the intended in-game behaviour. Hand-maintained (the generator/specs are
// gone), keyed by prefab.
func fixupSkillData(ks *AvatarSkills) {
	if ks.Prefab == "Avtr_Dsb_Elgorm" {
		// «Блуждающий ужас» (slot 1): the skull must leave Elgorm's hand at the throw
		// RELEASE -- early in the wind-up, not after it. Its authored PayloadDelay (1.0)
		// popped the skull out ~0.2s past the end of the 0.8s throw fx; 0.2 releases it
		// during the throw motion (~0.5s earlier than the previous CastFxDur-0.1 tuning).
		if ks.Skills[0].PayloadDelay > 0.2 {
			ks.Skills[0].PayloadDelay = 0.2
		}
		// «Блуждающий ужас» is a BOUNCING skull (Paralyzing Cask), not an instant AoE:
		// it strikes one enemy, then hops to a random nearby one, stunning + damaging
		// each. Rewrap the authored damage+stun as the per-hit ops of a single-target
		// OpBounce (Radius 0 on the damage so each hop hits ONE enemy, not a circle).
		if sk := &ks.Skills[0]; len(sk.Ops) >= 1 && sk.Ops[0].Kind != OpBounce {
			perHit := make([]Op, len(sk.Ops))
			copy(perHit, sk.Ops)
			perHit[0].Radius = 0 // single-target per hop
			// Count = total impacts = rank + 1, so the number of BOUNCES (hops after the
			// first strike) is exactly the skill's rank: 1/2/3/4/5 at ranks 1-5. This
			// overrides the authored TipArgs "steps" ({2,2,3,3} -> 1/1/2/2 bounces).
			sk.Ops = []Op{{
				Kind:     OpBounce,
				Count:    PerLevel{2, 3, 4, 5, 6},
				Radius:   6,   // hop to an enemy within this of the last hit
				Interval: 0.3, // seconds the skull spends jumping between enemies
				Ops:      perHit,
			}}
		}
		// «Стрелы Аркана» (slot 4): a BEAM, not a circle -- the arrows should strike
		// along a line from Elgorm toward the aim point (like Velial's «Разлом»), not
		// in a radius around it. AoEWidth>0 switches the damage resolver to the line
		// swath (damageTargetsLocked) and the client to the directional line cursor.
		// The server extends the swath to the FULL skill range (a stationary line skill
		// projects the whole Distance in the aim direction, matching the client's
		// SkillLineZone.SelfNoClamp -- see damageTargetsLocked), so the beam no longer
		// stops at the exact click point.
		ks.Skills[3].AoEWidth = 3
		ks.Skills[3].AoERadius = 0
		// Sync the channel's damage ticks with the client arrow rain. The payload fx
		// (ProjectileBurst on VFX_Avtr_Dsb_Elgorm_skill4_prop01) fires an arrow every
		// mInterval=0.46s starting after mDelay=0.2s; over the 4s cast animation that is
		// 9 arrows. Pulsing the channel at the same 0.46s cadence lands 9 damage ticks
		// in step with the 9 arrows (the previous 1.0s interval gave only ~5, and tick
		// drift on the 0.2s server tick cut it lower still -- fixed in tickChannelsLocked).
		for i := range ks.Skills[3].Ops {
			if ks.Skills[3].Ops[i].Kind == OpChannel {
				ks.Skills[3].Ops[i].Interval = 0.46
			}
		}
	}
}

// MaxRank is a skill's highest attainable rank = the length of its per-rank
// arrays after normalizeKit (5 for slots 1-3, 4 for the ult).
func (s Skill) MaxRank() int {
	if n := len(s.Cooldown); n > 0 {
		return n
	}
	if n := len(s.ManaCost); n > 0 {
		return n
	}
	return 4
}

// normalizeKit enforces the rank shape the real game uses: skills in slots 1-3
// can be leveled to rank 5, the ult (slot 4) to rank 4. Authored arrays that are
// short are extended by continuing their own last delta (so a flat array stays
// flat and a growing one keeps growing); over-long ones are truncated. This lets
// a skill be authored with only 4 ranks and still gain a sensible 5th, while the
// file-specified avatars carry their exact 5-rank tables.
func normalizeKit(ks *AvatarSkills) {
	for i := range ks.Skills {
		n := 5
		if i == 3 { // slot 4 = ult
			n = 4
		}
		sk := &ks.Skills[i]
		sk.ManaCost = extendSeq(sk.ManaCost, n)
		sk.Cooldown = extendSeq(sk.Cooldown, n)
		for j := range sk.Ops {
			normalizeOp(&sk.Ops[j], n)
		}
		for k, v := range sk.TipArgs {
			sk.TipArgs[k] = extendSeq(v, n)
		}
	}
}

func normalizeOp(op *Op, n int) {
	op.Value = extendSeq(op.Value, n)
	op.Value2 = extendSeq(op.Value2, n)
	op.Dur = extendSeq(op.Dur, n)
	op.Chance = extendSeq(op.Chance, n)
	op.BonusMissingHP = extendSeq(op.BonusMissingHP, n)
	op.CasterMissingHP = extendSeq(op.CasterMissingHP, n)
	op.Count = extendSeq(op.Count, n)
	op.Lifetime = extendSeq(op.Lifetime, n)
	op.HP = extendSeq(op.HP, n)
	op.Dmg = extendSeq(op.Dmg, n)
	op.TickCost = extendSeq(op.TickCost, n)
	op.DecayTo = extendSeq(op.DecayTo, n)
	op.Growth = extendSeq(op.Growth, n)
	op.StackCap = extendSeq(op.StackCap, n)
	op.StackBurstDamage = extendSeq(op.StackBurstDamage, n)
	op.SelfMaxHPPct = extendSeq(op.SelfMaxHPPct, n)
	op.ExplodeDamage = extendSeq(op.ExplodeDamage, n)
	op.PerTargetGrowth = extendSeq(op.PerTargetGrowth, n)
	op.VictimMaxHPPct = extendSeq(op.VictimMaxHPPct, n)
	op.GrowthPerEnemy = extendSeq(op.GrowthPerEnemy, n)
	op.MissingHPLinear = extendSeq(op.MissingHPLinear, n)
	op.DamageCap = extendSeq(op.DamageCap, n)
	for i := range op.Ops {
		normalizeOp(&op.Ops[i], n)
	}
}

// extendSeq resizes a per-rank sequence to exactly n ranks. An empty sequence (an
// unused field) stays empty; growth continues the last delta; an over-long sequence
// is truncated. The slice-type parameter S keeps the concrete type end to end
// (PerLevel in -> PerLevel out, []int in -> []int out) so callers need no casts.
func extendSeq[S ~[]E, E int | float64](p S, n int) S {
	if len(p) == 0 {
		return p
	}
	for len(p) < n {
		k := len(p)
		next := p[k-1]
		if k >= 2 {
			next = p[k-1] + (p[k-1] - p[k-2])
		}
		p = append(p, next)
	}
	if len(p) > n {
		p = p[:n]
	}
	return p
}
