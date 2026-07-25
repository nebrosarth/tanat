package battleserver

import "tanatserver/internal/gamedata"

// Status-effect state shared by the player avatar, mobs and summons. All
// timestamps are battle-time seconds (Server.battleTime). Mutations happen
// under conn.mvMu; the combat tick expires entries and reverses their
// client-visible side effects (stat SYNCs, EFFECT_END, REMOVE_EFFECTOR).

// statMod is one timed (or permanent, until=0) stat modifier.
//
// Stats (matching the authoring contract):
//
//	dmg_pct           attack damage ×value
//	phys_armor        flat physical armor
//	magic_armor       flat magic armor
//	armor_pct         both armors ×value
//	attack_speed_pct  attack speed ×value
//	move_speed_pct    move speed ×value
//	lifesteal_pct     heal for value fraction of dealt attack damage
//	crit_pct          crit chance +value (crit = ×1.5, RECEIVE_HIT flags 2)
//	dodge_pct         dodge chance +value (RECEIVE_HIT flags 1, no damage)
//	spell_power       flat spell power
//	hp_regen          flat hp/sec
//	mana_regen        flat mana/sec
//	max_hp            flat max health
//	thorns_pct        reflect value fraction of taken damage
type statMod struct {
	stat  string
	value float64
	until float64 // 0 = permanent (passive)

	// client-visible attachments reversed on expiry
	buffEffID int32  // ADD_EFFECTOR id shown on the self buff bar (0 = none)
	fxUID     int32  // looped EFFECT_START uid (0 = none)
	src       string // debug tag (avatar skill slot etc.)
}

// overTime is one DoT or HoT stream, ticking once per second.
type overTime struct {
	perSec   float64
	until    float64
	nextTick float64
	srcObj   int32 // damager object id for RECEIVE_HIT (DoT)
	fxUID    int32

	// perSecEnd/startAt implement Op.DecayTo: the tick magnitude interpolates linearly
	// from perSec (at startAt) to perSecEnd (at until). Left equal to perSec (the
	// default when no decay was authored) this is a no-op fast path -- see currentPerSec.
	perSecEnd float64
	startAt   float64
}

// currentPerSec returns this stream's tick magnitude at `now`, linearly interpolated
// between perSec (at startAt) and perSecEnd (at until) when they differ (Op.DecayTo);
// a flat (non-decaying) stream has perSecEnd == perSec and this returns perSec as-is.
func (o overTime) currentPerSec(now float64) float64 {
	if o.perSecEnd == o.perSec {
		return o.perSec
	}
	span := o.until - o.startAt
	if span <= 0 {
		return o.perSec
	}
	frac := (now - o.startAt) / span
	if frac < 0 {
		frac = 0
	} else if frac > 1 {
		frac = 1
	}
	return o.perSec + (o.perSecEnd-o.perSec)*frac
}

// unitStatus aggregates every timed effect on one unit.
type unitStatus struct {
	stunUntil    float64
	rootUntil    float64
	silenceUntil float64

	slowUntil  float64
	slowFactor float64 // move speed ×factor while slowed (e.g. 0.85)
	// slowFactorEnd/slowStart implement Op.DecayTo on a slow: the effective factor
	// interpolates linearly from slowFactor (at slowStart) to slowFactorEnd (at
	// slowUntil). Left equal to slowFactor (the no-decay default) this is a no-op --
	// see moveFactor.
	slowFactorEnd float64
	slowStart     float64

	atkSlowUntil  float64
	atkSlowFactor float64

	// killMarkUntil/killMarkOwner/killMarkOps/killMarkLevel implement Op.KillMark
	// (Lirvein's «Изощренный бросок»): if THIS unit dies from any source before
	// killMarkUntil, the mob-death branch fires killMarkOps for killMarkOwner instead
	// of requiring an instant kill by the marking cast.
	killMarkUntil float64
	killMarkOwner int32
	killMarkOps   []gamedata.Op
	killMarkLevel int
	killMarkSlot  int

	// castMarkUntil/castMarkDmg/castMarkOwner implement Op.CastMark (Einzenhaim's
	// «Изгнание колдовства»): if THIS unit casts a skill before castMarkUntil, it takes
	// castMarkDmg extra damage credited to castMarkOwner. Honored in tryBossSkillLocked
	// (only bosses actually cast).
	castMarkUntil float64
	castMarkDmg   float64
	castMarkOwner int32

	// poisonExplode* implement Op.ExplodeOnDeath (Wilfang's «Ядовитый укус»): if this unit
	// dies before poisonExplodeUntil, the mob-death branch detonates an AoE magic burst of
	// poisonExplodeDmg over poisonExplodeRadius, credited to poisonExplodeOwner.
	poisonExplodeUntil  float64
	poisonExplodeOwner  int32
	poisonExplodeDmg    float64
	poisonExplodeRadius float64

	// cloak* implement Op.DamageShare (Kiona's «Лесной покров»): while this unit (an enemy
	// mob OR an ally player -- unitStatus is shared by both) is marked, cloakHealCoeff
	// fraction of any damage IT takes also heals every living ally within cloakRadius of
	// it, credited to cloakOwner. Honored in hitMobFlagsLocked/hitPlayerFromLocked.
	cloakUntil     float64
	cloakOwner     int32
	cloakHealCoeff float64
	cloakRadius    float64

	// revealUntil implements Op.RevealTarget (Velial's «Трибунал»): while live, this mob
	// stays fully revealed to the marking caster's team regardless of the normal distance-
	// based fog/hide radius (mobViewDistLocked).
	revealUntil float64

	// chillUntil is the Frost «озноб» window: re-chilling an already-chilled unit stuns it
	// (OpChill). Purely a marker for the combo -- it carries no movement penalty of its own
	// (Frost's slow rides slowUntil/slowFactor).
	chillUntil float64
	chillFx    int32

	shield      float64 // absorb pool
	shieldUntil float64

	dots []overTime
	hots []overTime
	mods []statMod

	// looped status fx uids to EFFECT_END when the matching timer expires
	stunFx, rootFx, silenceFx, slowFx, atkSlowFx, shieldFx int32
	// dotFx is a single persistent "poisoned/acid" VFX shown while ANY DoT is
	// active on the unit (started by the first DoT carrying an Op.DotFx, ended when
	// the last DoT clears). One shared visual so re-procs don't stack copies.
	dotFx int32
	// anchorFxUntil is set when a ground-anchored player buff VFX (e.g. Vigilans'
	// ult barrier) is parented to THIS mob. If the ult then kills it, the corpse
	// must linger until this time so the SELF-mode barrier keeps its stationary
	// anchor instead of orphaning onto the caster when the body is deleted.
	anchorFxUntil float64
}

func (st *unitStatus) stunned(now float64) bool  { return now < st.stunUntil }
func (st *unitStatus) rooted(now float64) bool   { return now < st.rootUntil || st.stunned(now) }
func (st *unitStatus) silenced(now float64) bool { return now < st.silenceUntil || st.stunned(now) }

// moveFactor returns the current move-speed multiplier.
func (st *unitStatus) moveFactor(now float64) float64 {
	f := 1.0
	if now < st.slowUntil {
		factor := st.slowFactor
		if st.slowFactorEnd != st.slowFactor {
			span := st.slowUntil - st.slowStart
			if span > 0 {
				frac := (now - st.slowStart) / span
				if frac < 0 {
					frac = 0
				} else if frac > 1 {
					frac = 1
				}
				factor = st.slowFactor + (st.slowFactorEnd-st.slowFactor)*frac
			}
		}
		f *= factor
	}
	for _, m := range st.mods {
		if m.stat == "move_speed_pct" && (m.until == 0 || now < m.until) {
			f *= m.value
		}
	}
	if f < 0.05 {
		f = 0.05
	}
	return f
}

// attackFactor returns the current attack-speed multiplier.
func (st *unitStatus) attackFactor(now float64) float64 {
	f := 1.0
	if now < st.atkSlowUntil {
		f *= st.atkSlowFactor
	}
	for _, m := range st.mods {
		if m.stat == "attack_speed_pct" && (m.until == 0 || now < m.until) {
			f *= m.value
		}
	}
	if f < 0.1 {
		f = 0.1
	}
	return f
}

// modSum sums flat modifiers of one stat; modMul multiplies factor stats.
func (st *unitStatus) modSum(now float64, stat string) float64 {
	var v float64
	for _, m := range st.mods {
		if m.stat == stat && (m.until == 0 || now < m.until) {
			v += m.value
		}
	}
	return v
}

func (st *unitStatus) modMul(now float64, stat string) float64 {
	f := 1.0
	for _, m := range st.mods {
		if m.stat == stat && (m.until == 0 || now < m.until) {
			f *= m.value
		}
	}
	return f
}

// armorConst is the K in the armor curve armor/(armor+K): armor == K halves the
// damage (50% mitigation). Shared by the player and mob damage paths so both read
// the same tuning.
const armorConst = 50.0

// armorMitigation maps an armor value to an incoming-damage MULTIPLIER on the classic
// armor/(armor+K) curve (K = armorConst):
//
//	positive armor -> K/(armor+K), i.e. 1 - armor/(armor+K): reduce, in (0,1]
//	zero armor      -> 1.0 (no change; matches the pre-armor behaviour exactly)
//	negative armor  -> 1 + (-armor)/(-armor+K): AMPLIFY, bounded strictly below 2x
//
// Negative armor is "armor broken past 0" -- e.g. Velial's ult «Трибунал» appends a
// negative phys_armor status mod, so a stripped target takes amplified damage. The
// symmetric negative branch keeps the multiplier finite and continuous at 0 (a naive
// 1 - armor/(armor+K) blows up as armor -> -K); it never divides by zero and caps the
// amplification just under 2x no matter how deep the break stacks.
func armorMitigation(armor float64) float64 {
	if armor >= 0 {
		return armorConst / (armor + armorConst)
	}
	return 1 + (-armor)/(-armor+armorConst)
}

// absorb applies the shield pool to incoming damage, returning what remains.
func (st *unitStatus) absorb(now float64, dmg float64) float64 {
	if now >= st.shieldUntil || st.shield <= 0 {
		return dmg
	}
	if st.shield >= dmg {
		st.shield -= dmg
		return 0
	}
	dmg -= st.shield
	st.shield = 0
	return dmg
}
