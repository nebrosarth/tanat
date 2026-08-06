package battleserver

import (
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

// fakeMovePacket builds a MOVE_PLAYER-shaped packet for handleMove tests.
func fakeMovePacket(x, y float32) battleproto.Packet {
	return battleproto.Packet{
		Cmd: battleproto.CmdMovePlayer, Status: true,
		Args: amf.NewArray().Set("targetPos", amf.NewArray().Set("x", float64(x)).Set("y", float64(y))),
	}
}
