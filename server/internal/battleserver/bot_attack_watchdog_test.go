package battleserver

import (
	"testing"
	"time"
)

func TestBotAttackWatchdogReleasesInRangeOrderWithoutSwing(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_DPS_BlackDragon")
	defer cleanup()
	bot.huntState.team = dotaTeamHuman
	creep := teleportTestCreep(inst, 83001, dotaTeamElf, bot.x+1, bot.y)
	inst.mobs[creep.id] = creep
	brain := &botBrain{c: bot}
	inst.bots[bot.objID] = brain
	now := 3.0
	bot.huntState.attackTarget = creep.id
	brain.attackTargetSince = now - 2

	bot.lock()
	cleared := s.botAttackWatchdogLocked(brain, now)
	remaining := bot.huntState.attackTarget
	bot.unlock()

	if !cleared {
		t.Fatal("stale in-range attack order was not released")
	}
	if remaining != 0 {
		t.Fatalf("stale attack target remained set: %d", remaining)
	}
}

func TestBotAttackWatchdogReleasesSwingThatNeverLanded(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_HK_Teridin")
	defer cleanup()
	bot.huntState.team = dotaTeamHuman
	creep := teleportTestCreep(inst, 83006, dotaTeamElf, bot.x+1, bot.y)
	inst.mobs[creep.id] = creep
	brain := &botBrain{c: bot}
	inst.bots[bot.objID] = brain
	now := 8.0
	bot.huntState.attackTarget = creep.id
	brain.attackTargetSince = now - 5
	brain.attackLastSwingAt = now - 4
	brain.attackLastHitAt = 0

	bot.lock()
	cleared := s.botAttackWatchdogLocked(brain, now)
	remaining := bot.huntState.attackTarget
	bot.unlock()

	if !cleared || remaining != 0 {
		t.Fatalf("unconfirmed swing was not released: cleared=%v target=%d", cleared, remaining)
	}
}

func TestBotAttackWatchdogReleasesDeadOrRemovedTarget(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_HK_Abominator")
	defer cleanup()
	bot.huntState.team = dotaTeamHuman
	creep := teleportTestCreep(inst, 83008, dotaTeamElf, bot.x+1, bot.y)
	inst.mobs[creep.id] = creep
	brain := &botBrain{c: bot, farmTarget: creep.id, farmTargetScore: 10}
	inst.bots[bot.objID] = brain
	bot.huntState.attackTarget = creep.id
	bot.huntState.attackSeq = 1
	creep.dead = true

	now := float64(s.battleTime()) + 5
	bot.lock()
	cleared := s.botAttackWatchdogLocked(brain, now)
	remaining := bot.huntState.attackTarget
	bot.unlock()

	if !cleared || remaining != 0 || brain.farmTarget != 0 {
		t.Fatalf("dead creep target was not released: cleared=%v attackTarget=%d farmTarget=%d", cleared, remaining, brain.farmTarget)
	}
}

func TestBotAttackWatchdogKeepsOutOfRangeChase(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_DPS_BlackDragon")
	defer cleanup()
	bot.huntState.team = dotaTeamHuman
	creep := teleportTestCreep(inst, 83002, dotaTeamElf, bot.x+20, bot.y)
	inst.mobs[creep.id] = creep
	brain := &botBrain{c: bot}
	inst.bots[bot.objID] = brain
	now := float64(s.battleTime()) + 20
	bot.huntState.attackTarget = creep.id
	brain.attackTargetSince = now - 10
	bot.hasDest = true
	brain.attackLastProgressAt = now
	brain.attackLastX, brain.attackLastY = bot.x, bot.y

	bot.lock()
	cleared := s.botAttackWatchdogLocked(brain, now)
	remaining := bot.huntState.attackTarget
	bot.unlock()

	if cleared || remaining != creep.id {
		t.Fatalf("out-of-range chase was treated as a stale attack: cleared=%v target=%d", cleared, remaining)
	}
}

func TestBotAttackWatchdogReleasesStalledOutOfRangeOrder(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_DPS_BlackDragon")
	defer cleanup()
	bot.huntState.team = dotaTeamHuman
	creep := teleportTestCreep(inst, 83005, dotaTeamElf, bot.x+20, bot.y)
	inst.mobs[creep.id] = creep
	brain := &botBrain{c: bot}
	inst.bots[bot.objID] = brain
	now := float64(s.battleTime()) + 20
	bot.huntState.attackTarget = creep.id
	brain.attackTargetSince = now - 10
	brain.attackLastSwingAt = now - 9
	brain.attackLastProgressAt = now - 9
	brain.attackLastX, brain.attackLastY = bot.x, bot.y

	bot.lock()
	cleared := s.botAttackWatchdogLocked(brain, now)
	remaining := bot.huntState.attackTarget
	bot.unlock()

	if !cleared || remaining != 0 {
		t.Fatalf("stalled out-of-range attack was not released: cleared=%v target=%d", cleared, remaining)
	}
}

func TestBotAttackTimerRecordsRealSwing(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_DPS_BlackDragon")
	defer cleanup()
	bot.huntState.team = dotaTeamHuman
	creep := teleportTestCreep(inst, 83003, dotaTeamElf, bot.x+1, bot.y)
	inst.mobs[creep.id] = creep
	brain := &botBrain{c: bot}
	inst.bots[bot.objID] = brain
	bot.huntState.attackTarget = creep.id
	bot.huntState.attackSeq = 1

	s.armAttackTimer(bot, 1, 0, 2*time.Second)
	swung := false
	for i := 0; i < 200; i++ {
		time.Sleep(10 * time.Millisecond)
		bot.lock()
		swung = brain.attackLastSwingAt > 0
		bot.unlock()
		if swung {
			break
		}
	}
	bot.lock()
	s.stopAttackLocked(bot, true)
	bot.unlock()
	if !swung {
		t.Fatal("attack timer committed no observable swing for an in-range creep")
	}
}

func TestBotMeleeSwingCommitsDamageWhenCreepMovesDuringWindup(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_DPS_BlackDragon")
	defer cleanup()
	bot.huntState.team = dotaTeamHuman
	creep := teleportTestCreep(inst, 83004, dotaTeamElf, bot.x+1, bot.y)
	creep.hp = 1000
	creep.maxHP = 1000
	inst.mobs[creep.id] = creep
	brain := &botBrain{c: bot}
	inst.bots[bot.objID] = brain
	bot.huntState.attackTarget = creep.id
	bot.huntState.attackSeq = 1

	s.armAttackTimer(bot, 1, 0, 2*time.Second)
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		bot.lock()
		started := brain.attackLastSwingAt > 0
		if started {
			// Move the creep beyond the hit-time reach and invalidate the attack
			// sequence after the swing has committed. The bot's melee hit must
			// still land.
			creep.x = bot.x + 20
			bot.huntState.attackSeq++
		}
		dealt := creep.hp < creep.maxHP
		bot.unlock()
		if dealt {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("committed bot melee swing dealt no damage; creep hp=%.1f", creep.hp)
}

func TestBotRangedSwingCommitsDamageToMovingCreep(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_HK_Teridin")
	defer cleanup()
	bot.huntState.team = dotaTeamHuman
	creep := teleportTestCreep(inst, 83007, dotaTeamElf, bot.x+4, bot.y)
	creep.hp, creep.maxHP = 10000, 10000
	inst.mobs[creep.id] = creep
	brain := &botBrain{c: bot}
	inst.bots[bot.objID] = brain

	bot.lock()
	s.startAttackLocked(bot, creep)
	bot.unlock()
	deadline := time.Now().Add(4 * time.Second)
	for time.Now().Before(deadline) {
		bot.lock()
		dealt := creep.hp < creep.maxHP
		bot.unlock()
		if dealt {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("ranged bot attack dealt no damage; creep hp=%.1f", creep.hp)
}
