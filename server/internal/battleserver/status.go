package battleserver

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
}

// unitStatus aggregates every timed effect on one unit.
type unitStatus struct {
	stunUntil    float64
	rootUntil    float64
	silenceUntil float64

	slowUntil  float64
	slowFactor float64 // move speed ×factor while slowed (e.g. 0.85)

	atkSlowUntil  float64
	atkSlowFactor float64

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
		f *= st.slowFactor
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
