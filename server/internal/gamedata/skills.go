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
	// OpOnKill is a cast-time wrapper (Lirvein's «Изощренный бросок»): its nested Ops
	// run only if the cast's primary target died from this cast (ctx.target.dead).
	OpOnKill OpKind = "on_kill"
	// OpCooldownReset clears all of the caster's skill cooldowns (Lirvein's on-kill
	// reset). Typically nested under OpOnKill.
	OpCooldownReset OpKind = "cooldown_reset"
	// OpKnockback shoves the targeted enemies away from the caster by Value units
	// (the inverse of OpPull) — Dutnik's «Взрыв» detonation blast.
	OpKnockback OpKind = "knockback"
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
	Apply          string  // "self" targets the caster instead of enemies (health-cost damage)
	Stat           string  // buffstat: dmg_pct, phys_armor, ... (see statMod)
	On             string  // buffstat: "self" | "target"
	BonusMissingHP PerLevel
	// CasterMissingHP adds flat bonus damage = value × the CASTER's missing-HP fraction
	// (Velial's «Воля к победе»: fights harder the closer to death). Added after all
	// multipliers, so it is NOT scaled by power/attack buffs -- it matches the observed
	// in-game values directly (~100 × missing at ranks 4-5).
	CasterMissingHP PerLevel
	// MaxTargets, when >0, caps a damaging/CC op to the N nearest enemies in its area
	// (Rognar's «Могильный холод» hits only two). 0 = no cap (hit everything in range).
	MaxTargets int

	// summon
	Unit                     string // loadable character prefab
	Count, Lifetime, HP, Dmg PerLevel
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
