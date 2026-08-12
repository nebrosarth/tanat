package battleserver

import (
	"math"
	"testing"

	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
)

// TestPvpSpellDamageHitsEnemyHero: an AoE OpDamage centered on the caster must land on a
// living enemy hero standing in range. Before pvp_hero_targets.go, opTargetsLocked only
// ever scanned inst.mobs (creeps/structures) -- a caster's own spells never touched an
// enemy hero's HP at all; only the separate auto-attack path did.
func TestPvpSpellDamageHitsEnemyHero(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, human.x+2, human.y)
	elf.huntState.hp = 500

	human.lock()
	defer human.unlock()
	now := float64(s.battleTime())
	ops := []gamedata.Op{{Kind: gamedata.OpDamage, Value: gamedata.PerLevel{120}, Radius: 5}}
	s.applyOpsLocked(human, ops, opCtx{slot: 1, level: 1}, now)

	if elf.huntState.hp >= 500 {
		t.Fatalf("enemy hero hp = %g, want less than 500 -- AoE spell damage never landed on the hero", elf.huntState.hp)
	}
}

// TestPvpKnockbackMovesEnemyHero: an OpKnockback caught in the same AoE as OpDamage/OpRoot
// (Miriam's «Выстрел бури») must actually shove the enemy hero's real position, not a
// disposable shadow -- reported live: the shot damaged and rooted an enemy hero fine, but
// never visibly pushed them (knockbackMobLocked mutated a throwaway shadow's x/y).
func TestPvpKnockbackMovesEnemyHero(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, human.x+2, human.y)
	preX, preY := elf.x, elf.y

	human.lock()
	defer human.unlock()
	now := float64(s.battleTime())
	ops := []gamedata.Op{{Kind: gamedata.OpKnockback, Value: gamedata.PerLevel{4}, Radius: 5}}
	s.applyOpsLocked(human, ops, opCtx{slot: 1, level: 1}, now)

	moved := math.Hypot(float64(elf.x-preX), float64(elf.y-preY))
	if moved < 0.5 {
		t.Fatalf("enemy hero moved %.3f units after OpKnockback, want a real shove away from the caster", moved)
	}
}

// TestPvpPullMovesEnemyHeroCloser: OpPull must yank the real enemy hero toward the caster,
// the inverse of the knockback bug above.
func TestPvpPullMovesEnemyHeroCloser(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, human.x+4, human.y)
	now := float64(s.battleTime())
	preDist := math.Hypot(float64(elf.x-human.x), float64(elf.y-human.y))

	human.lock()
	defer human.unlock()
	ops := []gamedata.Op{{Kind: gamedata.OpPull, Radius: 6}}
	s.applyOpsLocked(human, ops, opCtx{slot: 1, level: 1, target: dotaHeroShadowLocked(human, elf, now)}, now)

	postDist := math.Hypot(float64(elf.x-human.x), float64(elf.y-human.y))
	if postDist >= preDist {
		t.Fatalf("enemy hero distance from caster = %.2f after OpPull, want less than the starting %.2f", postDist, preDist)
	}
}

// TestPvpInvisibleHeroTargetRequiresVision pins the server-side target gate used by
// direct, AoE and line skills: an invisible hero is not represented by a shadow until
// the caster's vision reaches it. A DotaGun is true sight; an ordinary structure is
// not, even though structures themselves remain visible through the fog.
func TestPvpInvisibleHeroTargetRequiresVision(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, human.x+250, human.y+250)
	elf.huntState.invisibleUntil = now + 60

	human.lock()
	defer human.unlock()
	if got := dotaHeroShadowLocked(human, elf, now); got != nil {
		t.Fatal("ordinary hero vision exposed an invisible enemy to PvP targeting")
	}

	var gun, ordinary *mobState
	for _, m := range inst.mobs {
		if !m.structure || m.teamVal() != dotaTeamHuman {
			continue
		}
		if m.dotaRole == gamedata.DotaGun && gun == nil {
			gun = m
		}
		if m.dotaRole != gamedata.DotaGun && ordinary == nil {
			ordinary = m
		}
		m.dead = true
	}
	if gun == nil || ordinary == nil {
		t.Fatal("precondition: missing human-side gun or ordinary structure")
	}
	ordinary.dead = false
	ordinary.x, ordinary.y = elf.x, elf.y
	if got := dotaHeroShadowLocked(human, elf, now); got != nil {
		t.Fatal("an ordinary structure incorrectly granted true sight for PvP targeting")
	}

	gun.dead = false
	gun.x, gun.y = elf.x, elf.y
	if got := dotaHeroShadowLocked(human, elf, now); got == nil {
		t.Fatal("a nearby DotaGun did not grant true sight for PvP targeting")
	}
}

// TestPvpChillComboStunsEnemyHero: Frost's chill-then-rechill combo (OpChill) must persist
// its "already chilled" marker on the REAL hero across two separate casts, then stun on the
// second -- a hero shadow rebuilt fresh per cast can't hold that marker itself.
func TestPvpChillComboStunsEnemyHero(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, human.x+2, human.y)

	human.lock()
	defer human.unlock()
	now := float64(s.battleTime())
	ops := []gamedata.Op{{Kind: gamedata.OpChill, Value2: gamedata.PerLevel{2}, Dur: gamedata.PerLevel{4}, Radius: 5}}
	s.applyOpsLocked(human, ops, opCtx{slot: 1, level: 1}, now)
	if elf.huntState.st.chillUntil <= now {
		t.Fatal("first OpChill did not mark the enemy hero as chilled")
	}

	s.applyOpsLocked(human, ops, opCtx{slot: 1, level: 1}, now+0.1)
	if !elf.huntState.st.stunned(now + 0.1) {
		t.Fatal("re-chilling an already-chilled enemy hero did not stun them")
	}
}

// TestPvpSpellStunAppliesToEnemyHero: OpStun must actually restrict the enemy hero it
// lands on -- stunned() gates casting/auto-attack-resume, and (via handleMove's new guard)
// movement too, so the write has to hit the REAL huntState.st, not a disposable shadow.
func TestPvpSpellStunAppliesToEnemyHero(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, human.x+2, human.y)

	human.lock()
	defer human.unlock()
	now := float64(s.battleTime())
	ops := []gamedata.Op{{Kind: gamedata.OpStun, Dur: gamedata.PerLevel{2}, Radius: 5}}
	s.applyOpsLocked(human, ops, opCtx{slot: 1, level: 1}, now)

	if !elf.huntState.st.stunned(now) {
		t.Fatal("enemy hero is not stunned after an in-range OpStun -- CC never reached the real huntState.st")
	}
	if !elf.huntState.st.rooted(now) {
		t.Fatal("a stunned hero must also read as rooted (stunned() implies rooted())")
	}
}

// TestPvpSpellSlowAndSilenceApplyToEnemyHero pins the remaining two CC ops on a hero
// target: OpSlow and OpSilence, both routed through their hero-side helpers.
func TestPvpSpellSlowAndSilenceApplyToEnemyHero(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, human.x+2, human.y)

	human.lock()
	defer human.unlock()
	now := float64(s.battleTime())
	ops := []gamedata.Op{
		{Kind: gamedata.OpSlow, Value: gamedata.PerLevel{0.5}, Dur: gamedata.PerLevel{3}, Radius: 5},
		{Kind: gamedata.OpSilence, Dur: gamedata.PerLevel{3}, Radius: 5},
	}
	s.applyOpsLocked(human, ops, opCtx{slot: 1, level: 1}, now)

	if got := elf.huntState.st.moveFactor(now); got >= 0.99 {
		t.Fatalf("enemy hero moveFactor = %g after OpSlow, want ~0.5", got)
	}
	if !elf.huntState.st.silenced(now) {
		t.Fatal("enemy hero is not silenced after an in-range OpSilence")
	}
	if elf.huntState.st.stunned(now) {
		t.Fatal("OpSilence on a hero must not also stun it -- a hero keeps auto-attacking while silenced")
	}
}

// TestRootedHeroChaseDoesNotMove: a rooted hero must not be walked anywhere by SERVER-
// DRIVEN movement either (a PvP/mob auto-attack chase re-arming on its own cadence, or an
// approach-then-cast retry) -- only handleMove (the player's own click) was gated before.
// Reported live: Miriam's «Выстрел бури» knocked an enemy hero back and damaged them, but
// the root never visibly held -- because the victim, still auto-attacking someone back,
// kept getting walked toward that target by chaseMoveLocked every ~250ms regardless.
func TestRootedHeroChaseDoesNotMove(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, human.x+2, human.y)

	elf.lock()
	now := float64(s.battleTime())
	startX, startY := elf.x, elf.y
	// The CC must stop an already-running leg too; testing only a fresh chase
	// would miss the arrival-timer path that used to keep walking on the client.
	elf.moveToLocked(s, startX+50, startY)
	if !elf.hasDest {
		elf.unlock()
		t.Fatal("setup: hero did not start a movement leg")
	}
	elf.huntState.st.rootUntil = now + 5
	elf.chaseMoveLocked(s, startX+50, startY) // e.g. armPvpAttackTimer's own re-arm chasing a target
	rootStopped := !elf.hasDest && elf.vx == 0 && elf.vy == 0
	// Stun uses the same movement gate and must also reject a brand-new route.
	elf.huntState.st.rootUntil = 0
	elf.huntState.st.stunUntil = now + 5
	elf.moveToLocked(s, startX+50, startY)
	stunStopped := !elf.hasDest && elf.vx == 0 && elf.vy == 0
	elf.unlock()

	if !rootStopped {
		t.Fatal("rooted hero kept an existing movement leg after chaseMoveLocked -- server-driven movement ignored the root")
	}
	if !stunStopped {
		t.Fatal("stunned hero accepted a new movement leg -- server movement gate ignored the stun")
	}
	if math.Hypot(float64(elf.x-startX), float64(elf.y-startY)) > 0.01 {
		t.Fatalf("rooted hero moved to (%.1f,%.1f), want to stay at (%.1f,%.1f)", elf.x, elf.y, startX, startY)
	}
}

// TestRootedBotDecisionDoesNotReissueMovement covers the bot-only movement path.
// Unlike a real player's MOVE_PLAYER handler, botMoveTowardLocked calls moveToLocked
// directly, so a rooted retreating bot could previously start walking again on its next
// think tick even though the root helper had already stopped its current leg.
func TestRootedBotDecisionDoesNotReissueMovement(t *testing.T) {
	s, caster, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	victim := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, caster.x+2, caster.y)
	b := &botBrain{c: victim, lane: 0, phase: botPhaseLane}
	now := float64(s.battleTime())

	victim.lock()
	// Put the bot on an active retreat leg before the enemy spell lands.
	s.botMoveTowardLocked(b, victim.x+40, victim.y, now)
	if !victim.hasDest {
		t.Fatal("setup: bot did not start a movement leg")
	}
	victim.unlock()

	// This is the actual OpRoot target path used by PlusMinus skill 2.
	caster.lock()
	s.dotaRootHeroLocked(caster, victim, now+0.1, 3)
	caster.unlock()

	// The next brain decision must not overwrite the root with a new retreat order.
	victim.lock()
	defer victim.unlock()
	s.botMoveTowardLocked(b, victim.x+80, victim.y, now+0.2)
	if victim.hasDest || victim.vx != 0 || victim.vy != 0 {
		t.Fatalf("rooted bot kept moving: hasDest=%v velocity=(%.2f,%.2f)", victim.hasDest, victim.vx, victim.vy)
	}
}

// TestMoveCancelsPvpAttack: clicking to move away while auto-attacking an enemy hero must
// cancel the PvP attack chain, not leave it armed to keep chasing/re-engaging on its own
// schedule. Reported live: a player fighting a hero (auto-attack, possibly with a skill
// thrown in) could click to flee repeatedly and the avatar kept turning back to fight --
// handleMove cancelled hs.attackTarget (mob attacks) but never hs.pvpTarget, so the
// armPvpAttackTimer chain from the original engage stayed alive and its own ~250ms retry
// kept calling chaseMoveLocked back toward the enemy, overriding every move order.
func TestMoveCancelsPvpAttack(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, human.x+2, human.y)

	human.lock()
	s.startPvpAttackLocked(human, elf)
	startedOK := human.huntState.pvpTarget == elf.objID
	human.unlock()
	if !startedOK {
		t.Fatal("setup: PvP attack did not start")
	}

	s.handleMove(human, fakeMovePacket(human.x-20, human.y))

	human.lock()
	defer human.unlock()
	if human.huntState.pvpTarget != 0 {
		t.Fatalf("pvpTarget = %d after a move order, want 0 (still armed to chase back to the enemy)",
			human.huntState.pvpTarget)
	}
}

// TestPvpRootedHeroCannotMove pins handleMove's new guard: a rooted player's MOVE_PLAYER
// is rejected and the avatar is frozen in place, instead of silently walking away from a
// root/stun that CC ops can now actually apply.
func TestPvpRootedHeroCannotMove(t *testing.T) {
	s, human, _, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())

	human.lock()
	human.huntState.st.rootUntil = now + 5
	startX, startY := human.x, human.y
	human.unlock()

	s.handleMove(human, fakeMovePacket(startX+10, startY))

	human.lock()
	defer human.unlock()
	if human.hasDest || human.x != startX || human.y != startY {
		t.Fatalf("rooted player moved: hasDest=%v pos=(%.1f,%.1f), want frozen at (%.1f,%.1f)",
			human.hasDest, human.x, human.y, startX, startY)
	}
}

// TestDotaHeroKillPaysBounty: a hero kill in «Штурм» must pay the killer XP + gold -- before
// dotaCreditHeroKillLocked, arenaCreditKillLocked's inst.arena==nil guard made every «Штурм»
// hero kill pay nothing at all.
func TestDotaHeroKillPaysBounty(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, human.x+1, human.y)
	elf.huntState.hp = 10
	now := float64(s.battleTime())

	human.lock()
	defer human.unlock()
	startXP := human.huntState.xp
	s.hitPlayerFromLocked(elf, human.objID, 1000, now, nil, human)

	if elf.huntState.deadUntil == 0 {
		t.Fatal("enemy hero did not die to the lethal blow")
	}
	if human.huntState.xp <= startXP {
		t.Fatalf("killer xp = %g after a hero kill, want more than %g -- «Штурм» hero kills must pay a bounty", human.huntState.xp, startXP)
	}
}

// TestDotaKillXPSplitsAmongNearbyTeammates: a creep kill's XP divides evenly among every
// living teammate within dotaXPShareRadius of the kill, not just the killer -- the
// mechanical reason stacking the whole team on one lane is inefficient.
func TestDotaKillXPSplitsAmongNearbyTeammates(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	ally := dotaPlayerConn(t, s, inst, 1001, dotaTeamHuman, human.x+3, human.y)

	m := &mobState{id: 65000, mobIdx: inst.dota.m.HumanCreepMelee, mob: gamedata.Mobs()[inst.dota.m.HumanCreepMelee],
		x: human.x, y: human.y, hp: 1, maxHP: 500, xp: 100, coins: 10, team: dotaTeamElf}

	human.lock()
	defer human.unlock()
	inst.mobs[m.id] = m
	startHuman, startAlly := human.huntState.xp, ally.huntState.xp
	s.hitMobLocked(human, m, 10, human.objID)

	gotHuman, gotAlly := human.huntState.xp-startHuman, ally.huntState.xp-startAlly
	if gotHuman <= 0 || gotAlly <= 0 {
		t.Fatalf("xp gained: killer=%g ally=%g, want both positive (split, not all-or-nothing)", gotHuman, gotAlly)
	}
	if diff := gotHuman - gotAlly; diff > 0.01 || diff < -0.01 {
		t.Fatalf("xp gained: killer=%g ally=%g, want an even split (both in range of the kill)", gotHuman, gotAlly)
	}
	if want := 100.0 / 2; gotHuman < want-0.01 || gotHuman > want+0.01 {
		t.Fatalf("killer xp share = %g, want %g (100 xp split 2 ways)", gotHuman, want)
	}
}

// TestDotaCreepKillGrantsProximityXP covers the no-last-hit case: when a friendly creep
// kills an enemy creep, nearby heroes still split XP even though no hero receives gold.
func TestDotaCreepKillGrantsProximityXP(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	ally := dotaPlayerConn(t, s, inst, 1001, dotaTeamHuman, human.x+3, human.y)
	attacker := &mobState{id: 65010, team: dotaTeamHuman, hp: 100, maxHP: 100}
	victim := &mobState{id: 65011, team: dotaTeamElf, hp: 1, maxHP: 100, xp: 100,
		x: human.x, y: human.y, shown: true, active: true}

	human.lock()
	defer human.unlock()
	inst.mobs[attacker.id] = attacker
	inst.mobs[victim.id] = victim
	startHuman, startAlly := human.huntState.xp, ally.huntState.xp
	s.dotaDamageLocked(human, victim, 10, attacker.id, float64(s.battleTime()))

	gotHuman := human.huntState.xp - startHuman
	gotAlly := ally.huntState.xp - startAlly
	if gotHuman < 49.99 || gotHuman > 50.01 || gotAlly < 49.99 || gotAlly > 50.01 {
		t.Fatalf("creep-death proximity XP = human %.2f ally %.2f, want 50/50", gotHuman, gotAlly)
	}
}

// fakeMovePacket builds a MOVE_PLAYER-shaped packet for handleMove tests.
func fakeMovePacket(x, y float32) battleproto.Packet {
	return battleproto.Packet{
		Cmd: battleproto.CmdMovePlayer, Status: true,
		Args: amf.NewArray().Set("targetPos", amf.NewArray().Set("x", float64(x)).Set("y", float64(y))),
	}
}
