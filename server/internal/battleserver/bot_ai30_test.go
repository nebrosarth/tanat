package battleserver

import "testing"

func ai30TestBrain(bot *conn, version int) *botBrain {
	return &botBrain{
		c: bot, lane: 0, phase: botPhaseLane, aiVersion: version, aiVersionSet: true,
		macroAssignment: botMacroAssignment{Mode: botMacroLane, Lane: 0, FarmLane: 0, FarmLaneSet: true},
	}
}

func TestAI30ActivelyApproachesCoveredWave(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	bot.x += 35
	bot.snapT = float32(now)
	target := teleportTestCreep(inst, 93001, dotaTeamElf, bot.x+7, bot.y)
	cover := teleportTestCreep(inst, 93002, dotaTeamHuman, target.x-1, target.y)
	inst.mobs[target.id], inst.mobs[cover.id] = target, cover
	b := ai30TestBrain(bot, 30)
	inst.bots[bot.objID] = b

	bot.lock()
	acted := s.botMoveToFarmTargetLocked(b, target, now)
	decision, hasDest := b.farmDecision, bot.hasDest
	bot.unlock()
	if !acted || decision != "active_wave_clear" || !hasDest {
		t.Fatalf("AI-30 covered wave action: acted=%v decision=%q hasDest=%v", acted, decision, hasDest)
	}
}

func TestAI20KeepsRearAnchorForSameCoveredWave(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	bot.x += 35
	bot.snapT = float32(now)
	target := teleportTestCreep(inst, 93011, dotaTeamElf, bot.x+7, bot.y)
	cover := teleportTestCreep(inst, 93012, dotaTeamHuman, target.x-1, target.y)
	inst.mobs[target.id], inst.mobs[cover.id] = target, cover
	b := ai30TestBrain(bot, 20)
	inst.bots[bot.objID] = b

	bot.lock()
	acted := s.botMoveToFarmTargetLocked(b, target, now)
	decision := b.farmDecision
	bot.unlock()
	if !acted || decision != "wave_anchor" {
		t.Fatalf("AI-20 covered wave action: acted=%v decision=%q, want rear anchor", acted, decision)
	}
}

func TestAI30DoesNotApproachUncoveredOrPressuredWave(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	bot.x += 35
	bot.snapT = float32(now)
	target := teleportTestCreep(inst, 93003, dotaTeamElf, bot.x+7, bot.y)
	inst.mobs[target.id] = target
	b := ai30TestBrain(bot, 30)
	inst.bots[bot.objID] = b

	bot.lock()
	_, _, uncovered := s.botAI30FarmAttackPointLocked(b, target, now)
	cover := teleportTestCreep(inst, 93004, dotaTeamHuman, target.x-1, target.y)
	inst.mobs[cover.id] = cover
	bot.huntState.hp = bot.huntState.maxHPLocked(now) * 0.6
	_, _, lowHP := s.botAI30FarmAttackPointLocked(b, target, now)
	bot.unlock()
	if uncovered || lowHP {
		t.Fatalf("AI-30 accepted unsafe farm approach: uncovered=%v lowHP=%v", uncovered, lowHP)
	}
}

func TestAI30AttacksCreepAlreadyInReachWithoutCover(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	target := teleportTestCreep(inst, 93009, dotaTeamElf, bot.x+1, bot.y)
	inst.mobs[target.id] = target
	b := ai30TestBrain(bot, 30)
	inst.bots[bot.objID] = b

	bot.lock()
	acted := s.botMoveToFarmTargetLocked(b, target, now)
	attackTarget := bot.huntState.attackTarget
	bot.unlock()
	if !acted || attackTarget != target.id {
		t.Fatalf("AI-30 did not attack uncovered creep already in reach: acted=%v target=%d", acted, attackTarget)
	}
}

func TestAI30KeepsCreepAttackInReachWhenCoverDisappears(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	bot.x += 35
	bot.snapT = float32(now)
	target := teleportTestCreep(inst, 93013, dotaTeamElf, bot.x+1, bot.y)
	cover := teleportTestCreep(inst, 93014, dotaTeamHuman, target.x-1, target.y)
	inst.mobs[target.id], inst.mobs[cover.id] = target, cover
	b := ai30TestBrain(bot, 30)
	inst.bots[bot.objID] = b

	bot.lock()
	s.startAttackLocked(bot, target)
	cover.dead = true
	s.botTickLocked(b, now)
	attackTarget := bot.huntState.attackTarget
	bot.unlock()
	if attackTarget != target.id {
		t.Fatalf("AI-30 stopped safe in-range creep attack after allied cover disappeared: got %d want %d", attackTarget, target.id)
	}
}

func TestAI30PrefersWoundedHighValueHero(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	bot.x += 35
	bot.snapT = float32(now)
	near := dotaPlayerConn(t, s, inst, 93005, dotaTeamElf, bot.x+2, bot.y)
	wounded := dotaPlayerConn(t, s, inst, 93006, dotaTeamElf, bot.x+5, bot.y)
	wounded.huntState.av = avatarByPrefab(t, "Avtr_Sp_Arianna")
	wounded.huntState.hp = wounded.huntState.maxHPLocked(now) * 0.2
	b := ai30TestBrain(bot, 30)
	inst.bots[bot.objID] = b

	bot.lock()
	got := s.botAI30PreferredTargetLocked(b, []*conn{near, wounded}, now)
	bot.unlock()
	if got != wounded {
		t.Fatalf("AI-30 target=%v, want wounded support %d", got, wounded.objID)
	}
}

func TestAI30ReadyPowerUsesOnlyUsableAbilities(t *testing.T) {
	s, bot, _, cleanup := newDotaConn(t, "Avtr_HK_Abominator")
	defer cleanup()
	now := float64(s.battleTime())
	hs := bot.huntState

	bot.lock()
	slot := 0
	for i := 1; i <= 4; i++ {
		if def := hs.skillDef(i); def.Type == "ACTIVE" && botOffensiveOpPriority(def) > 0 {
			slot = i
			break
		}
	}
	if slot == 0 {
		bot.unlock()
		t.Fatal("setup: avatar has no offensive active ability")
	}
	hs.skillLevel[slot-1] = 1
	hs.mana = hs.maxManaLocked(now)
	ready := s.botAI30ReadyPowerLocked(bot, now)
	hs.cooldownUntil[slot-1] = now + 10
	cooldown := s.botAI30ReadyPowerLocked(bot, now)
	bot.unlock()
	if ready <= 0 || cooldown >= ready {
		t.Fatalf("AI-30 ready power: ready=%g cooldown=%g", ready, cooldown)
	}
}

func TestAI30BreaksOptionalChaseOnLowHP(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	bot.x += 35
	bot.snapT = float32(now)
	enemy := dotaPlayerConn(t, s, inst, 93007, dotaTeamElf, bot.x+3, bot.y)
	b := ai30TestBrain(bot, 30)
	inst.bots[bot.objID] = b

	bot.lock()
	s.startPvpAttackLocked(bot, enemy)
	b.engageTarget = enemy.objID
	bot.huntState.hp = bot.huntState.maxHPLocked(now) * 0.5
	acted := s.botCombatTickLocked(b, now)
	target, hasDest := bot.huntState.pvpTarget, bot.hasDest
	bot.unlock()
	if !acted || target != 0 || !hasDest {
		t.Fatalf("AI-30 low-HP chase: acted=%v pvpTarget=%d hasDest=%v", acted, target, hasDest)
	}
}

func TestAI30BreaksChaseWhenTargetLeavesVision(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	bot.x += 35
	bot.snapT = float32(now)
	enemy := dotaPlayerConn(t, s, inst, 93008, dotaTeamElf, bot.x+3, bot.y)
	b := ai30TestBrain(bot, 30)
	inst.bots[bot.objID] = b

	bot.lock()
	s.startPvpAttackLocked(bot, enemy)
	b.engageTarget = enemy.objID
	enemy.x, enemy.y, enemy.snapT = bot.x+250, bot.y+250, float32(now)
	acted := s.botCombatTickLocked(b, now)
	target := bot.huntState.pvpTarget
	bot.unlock()
	if !acted || target != 0 {
		t.Fatalf("AI-30 hidden chase: acted=%v pvpTarget=%d", acted, target)
	}
}
