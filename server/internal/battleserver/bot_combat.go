package battleserver

// Hero-vs-hero combat decisions: whether a fight is worth starting, who to focus, which
// ability to spend on it, and healing/shielding a hurt ally. Everything here leans on the
// PVP ability-vs-hero engine fix (pvp_hero_targets.go) -- an ability aimed at an enemy
// hero's position now actually damages/CCs them, so "use abilities when advantageous"
// means something real here, not just an animation.

import (
	"math"

	"tanatserver/internal/gamedata"
)

// botEngageRadius is how far a bot will notice and commit to an opportunistic hero fight.
const botEngageRadius = 16.0

// botFightRadius is the radius around the FIGHT LOCATION (not the bot itself) used to
// count allies/enemies for the favourability check -- wider than the engage radius since a
// teammate a little further back still arrives in time to matter.
const botFightRadius = 24.0

// botNoDiveRadius keeps a chase from continuing into an enemy tower/cannon's kill zone.
// DotaGunRange/DotaTowerRange are both ~10-11u; a few extra units of margin means a bot
// breaks off BEFORE it's actually taking structure damage, not after. Without this, a
// fleeing hero kited a bot straight through the enemy's own cannons and all the way to
// their fountain -- the numbers-favourability check alone never catches this, since it
// only counts nearby HEROES, not the structures doing the actual killing.
const botNoDiveRadius = 15.0

// botEnemyStructureDangerLocked reports whether (x,y) is within botNoDiveRadius of any
// living enemy structure (gun/tower/altar) -- i.e. whether continuing a fight AT that spot
// means tower-diving, not hero-fighting.
func (s *Server) botEnemyStructureDangerLocked(c *conn, x, y float32) bool {
	for _, m := range c.huntState.mobs {
		if m.dead || !m.structure || !m.enemyOf(c.playerTeam()) {
			continue
		}
		if dist2(x, y, m.x, m.y) <= botNoDiveRadius*botNoDiveRadius {
			return true
		}
	}
	return false
}

// botCombatTickLocked looks for a nearby enemy hero worth fighting and, if one is found,
// drives that engagement for this tick (ability, then attack order). Reports whether it
// issued an order, so the caller (botTickLocked) skips the phase-specific logic below it.
func (s *Server) botCombatTickLocked(b *botBrain, now float64) bool {
	c, hs := b.c, b.c.huntState
	// A hurting ally (or self) always gets a heal/shield first, fight or no fight --
	// this is what makes Ariana/Neirofim actually play their support role. Checked BEFORE
	// the retreat gate below: retreating exits combat handling entirely (returns false),
	// so a heal ready on the very tick a bot crosses the retreat threshold used to never
	// fire at all -- the bot fled instead of spending its one instant self-save.
	if s.botConsiderHealLocked(b, now) {
		return true
	}
	if s.botShouldRetreatLocked(b, now) {
		return false
	}
	enemies := botLivingEnemyHeroes(c, now)
	if len(enemies) == 0 {
		b.engageTarget = 0
		s.stopPvpAttackLocked(c, false)
		return false
	}
	target := s.botPickEngageTargetLocked(b, enemies, now)
	if target == nil {
		wasEngaged := b.engageTarget != 0 || hs.pvpTarget != 0
		b.engageTarget = 0
		// The brain just decided this fight is no longer worth it (out of range,
		// unfavourable numbers, or -- see botEnemyStructureDangerLocked -- about to walk
		// into the enemy's own tower/cannon range). Without this, hs.pvpTarget stays set
		// and armPvpAttackTimer's own chase-and-rearm loop keeps running on its own
		// cadence regardless of what the brain thinks now: reported live, this is how a
		// bot ended up chasing a fleeing hero straight through the enemy's cannons and
		// onto their fountain. armPvpAttackTimer only ever stops itself when the victim
		// actually dies/leaves/goes invisible -- it has no notion of "too deep," so the
		// brain has to be the one pulling the plug every think tick it reconsiders.
		s.stopPvpAttackLocked(c, false)
		if wasEngaged {
			tx, ty := s.botDisengagePointLocked(b, now)
			s.botMoveTowardLocked(b, tx, ty, now)
			return true
		}
		return false
	}
	b.engageTarget = target.objID
	s.botConsiderOffensiveAbilityLocked(b, target, now)
	if hs.pvpTarget != target.objID {
		s.startPvpAttackLocked(c, target)
	}
	return true
}

// botPickEngageTargetLocked chooses which nearby enemy hero to fight, or nil if no fight
// is currently favourable. Sticky on the already-committed target so it doesn't flip-flop
// between two similarly-close enemies every think tick.
func (s *Server) botPickEngageTargetLocked(b *botBrain, enemies []*conn, now float64) *conn {
	c, hs := b.c, b.c.huntState
	cx, cy := c.posAtLocked(float32(now))

	var nearest *conn
	nearestD := math.Inf(1)
	for _, e := range enemies {
		ex, ey := e.posAtLocked(float32(now))
		if d := math.Hypot(float64(ex-cx), float64(ey-cy)); d <= botEngageRadius && d < nearestD {
			nearestD, nearest = d, e
		}
	}
	if b.engageTarget != 0 {
		if cur := c.pvpMember(b.engageTarget); cur != nil && cur.huntState.deadUntil == 0 {
			ux, uy := cur.posAtLocked(float32(now))
			if math.Hypot(float64(ux-cx), float64(uy-cy)) <= botEngageRadius*1.4 {
				nearest = cur // keep pressing the committed target over a marginally-closer one
			}
		}
	}
	if nearest == nil {
		return nil
	}
	tx, ty := nearest.posAtLocked(float32(now))
	if s.botEnemyStructureDangerLocked(c, tx, ty) {
		return nil // the fight (or the chase toward it) is inside the enemy's own tower/cannon range -- not worth pressing
	}

	// Favourability: count living allies vs enemies near the FIGHT LOCATION. A lone
	// laner does not start a fight next to two enemy teammates, and won't press a fight
	// while alone and already hurt even if the numbers are even.
	allyN, enemyN := 1, 1 // self + target
	for _, mem := range c.inst.members {
		if mem == c || mem == nearest || mem.huntState == nil || mem.huntState.deadUntil > 0 {
			continue
		}
		mx, my := mem.posAtLocked(float32(now))
		if math.Hypot(float64(mx-tx), float64(my-ty)) > botFightRadius {
			continue
		}
		if mem.playerTeam() == c.playerTeam() {
			allyN++
		} else {
			enemyN++
		}
	}
	if allyN < enemyN {
		return nil
	}
	if allyN == 1 && botHPFrac(hs, now) < 0.5 {
		return nil
	}
	return nearest
}

// botOffensiveOpPriority scores a ready ability for use against an enemy hero: hard CC
// first (a stunned/rooted/silenced target can't fight back or escape), then damage,
// everything else last.
func botOffensiveOpPriority(def gamedata.Skill) int {
	switch {
	case botSkillHasOp(def, gamedata.OpStun):
		return 4
	case botSkillHasOp(def, gamedata.OpRoot), botSkillHasOp(def, gamedata.OpSilence):
		return 3
	case botSkillHasOp(def, gamedata.OpDamage), botSkillHasOp(def, gamedata.OpExecute),
		botSkillHasOp(def, gamedata.OpManaScaledDamage), botSkillHasOp(def, gamedata.OpSlow):
		return 2
	case botSkillHasOp(def, gamedata.OpHeal), botSkillHasOp(def, gamedata.OpShield), botSkillHasOp(def, gamedata.OpHot):
		return 0 // handled separately by botConsiderHealLocked, never picked here
	default:
		return 1
	}
}

// botConsiderOffensiveAbilityLocked casts the highest-priority ready ability at target's
// current position. A point-targeted cast at the hero's own coordinates resolves the hero
// itself (see pvp_hero_targets.go), so this works uniformly for AoE, line and single-target
// kits alike without needing to special-case each skill's exact Target mask.
func (s *Server) botConsiderOffensiveAbilityLocked(b *botBrain, target *conn, now float64) {
	c, hs := b.c, b.c.huntState
	bestSlot, bestP := 0, -1
	for slot := 1; slot <= 4; slot++ {
		if !s.botAbilityReadyLocked(hs, slot, now) {
			continue
		}
		def := hs.skillDef(slot)
		if p := botOffensiveOpPriority(def); p > bestP {
			bestP, bestSlot = p, slot
		}
	}
	if bestSlot == 0 || bestP <= 0 {
		return
	}
	def := hs.skillDef(bestSlot)
	if def.Target == "" || def.Target == "SELF" {
		s.startSkillOrderLocked(c, bestSlot, 0, 0, 0, false)
		return
	}
	tx, ty := target.posAtLocked(float32(now))
	s.startSkillOrderLocked(c, bestSlot, 0, tx, ty, true)
}

// botHealNeedFrac is the HP fraction below which a hero (self or ally) is worth spending a
// heal/shield on.
const botHealNeedFrac = 0.65

// botSkillTargetsAlliesLocked reports whether a heal/hot/shield slot can actually be aimed
// at (or land on) a hurt ally: an explicit FRIEND-flagged single-target cast, a
// self-centered cast (Target==""/"SELF" -- e.g. Tangren's "Целительный тотем", an
// AoE-on-self totem with On:"allies"), or a ground-targeted AoE (Target=="POINT" -- e.g.
// Ariana's "Исцеление", aimed at the hurt ally's position). Without the latter two, an
// entire class of real, already-balanced heals was structurally invisible to the bot: it
// only ever considered the FRIEND-flagged case, so a hero whose sole heal is a self-AoE
// totem or a ground-target AoE never cast it once, in any match, regardless of how hurt it
// or its allies were.
func botSkillTargetsAlliesLocked(def gamedata.Skill) bool {
	return skillHasTargetFlag(def, "FRIEND") || def.Target == "" || def.Target == "SELF" || def.Target == "POINT"
}

// botFindHealTargetLocked returns the most hurt nearby ally (self included) below
// botHealNeedFrac and a ready ally-reaching heal/shield/hot slot for it, or (nil, 0).
func (s *Server) botFindHealTargetLocked(b *botBrain, now float64) (*conn, int) {
	c, hs := b.c, b.c.huntState
	healSlot := 0
	for slot := 1; slot <= 4; slot++ {
		if !s.botAbilityReadyLocked(hs, slot, now) {
			continue
		}
		def := hs.skillDef(slot)
		if !botSkillTargetsAlliesLocked(def) {
			continue
		}
		if botSkillHasOp(def, gamedata.OpHeal) || botSkillHasOp(def, gamedata.OpHot) || botSkillHasOp(def, gamedata.OpShield) {
			healSlot = slot
			break
		}
	}
	if healSlot == 0 {
		return nil, 0
	}
	dist := float64(hs.skillDef(healSlot).Distance)
	if dist <= 0 {
		dist = 8
	}
	cx, cy := c.posAtLocked(float32(now))

	var worst *conn
	worstFrac := botHealNeedFrac
	if f := botHPFrac(hs, now); f < worstFrac {
		worst, worstFrac = c, f
	}
	for _, mem := range botLivingAllies(c) {
		f := botHPFrac(mem.huntState, now)
		if f >= worstFrac {
			continue
		}
		mx, my := mem.posAtLocked(float32(now))
		if math.Hypot(float64(mx-cx), float64(my-cy)) > dist+6 {
			continue // too far to reasonably reach with this cast right now
		}
		worst, worstFrac = mem, f
	}
	return worst, healSlot
}

// botConsiderHealLocked casts a support ability on the most hurt nearby ally (or self) if
// one is found. Reports whether it acted.
func (s *Server) botConsiderHealLocked(b *botBrain, now float64) bool {
	target, slot := s.botFindHealTargetLocked(b, now)
	if target == nil {
		return false
	}
	c := b.c
	def := c.huntState.skillDef(slot)
	tx, ty := target.posAtLocked(float32(now))
	switch def.Target {
	case "", "SELF":
		// Self-centered (or self-only) cast: startSkillOrderLocked's own self-cast branch
		// fires unconditionally on def.Target=="" || "SELF" and ignores target/position
		// entirely, so this only needs to happen at all -- who it ends up helping (e.g.
		// Tangren's totem healing whichever allies stand in its radius) is the skill's own
		// business, not this call's.
		s.startSkillOrderLocked(c, slot, 0, 0, 0, false)
	case "POINT":
		// Ground-targeted AoE heal (e.g. Ariana's "Исцеление"): must NOT carry the ally's
		// objID -- startSkillOrderLocked only resolves an object target through the
		// FRIEND-flagged ally path, and a bare POINT skill doesn't carry that flag, so
		// passing target.objID here would silently fizzle the order (falls through to
		// orderDoneLocked with nothing cast). Aim at the hurt ally's position instead.
		s.startSkillOrderLocked(c, slot, 0, tx, ty, true)
	default:
		s.startSkillOrderLocked(c, slot, target.objID, tx, ty, true)
	}
	return true
}
