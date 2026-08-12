package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

func ai11OwnBarracks(inst *huntInstance) *mobState {
	for _, m := range botSortedMobs(inst) {
		if m.structure && !m.altar && m.team == dotaTeamHuman && m.dotaRole == gamedata.DotaCreepTower && !m.dead {
			return m
		}
	}
	return nil
}

func TestAI11UrgentBarracksDefensePreemptsTeleportAndKeepsChargeConsumed(t *testing.T) {
	s, bot, inst, b, it, now, cleanup := newTeleportTestBot(t)
	defer cleanup()

	barracks := ai11OwnBarracks(inst)
	if barracks == nil {
		t.Fatal("setup: no allied barracks")
	}
	target := teleportTestCreep(inst, 81100, bot.playerTeam(), 0, 0)
	installTeleportTarget(inst, bot, target)
	bot.lock()
	if !s.botMaybeStartTeleportLocked(b, now) {
		bot.unlock()
		t.Fatal("setup: teleport channel did not start")
	}
	bot.unlock()
	inst.mobs[barracks.id] = barracks
	laneIndex := botNearestLaneToPointLocked(inst.dota, barracks.x, barracks.y)
	threat := teleportTestCreep(inst, 81101, dotaTeamElf, barracks.x-2, barracks.y)
	threat.lane = inst.dota.m.Lanes[laneIndex]
	threat.laneIdx = len(threat.lane) / 2
	threat.dtarget = barracks.id
	inst.mobs[threat.id] = threat
	b.macroAssignment = botMacroAssignment{Mode: botMacroBase, Role: "defender", ObjectiveID: barracks.id}

	bot.lock()
	s.botTickTeleportLocked(b, now+0.5)
	bot.unlock()
	if b.pendingTeleport != nil {
		t.Fatal("urgent barracks assignment left an incompatible teleport channel pending")
	}
	if got := bot.huntState.bag[it.ArticleID]; got != botTeleportCharges-1 {
		t.Fatalf("preemption charge count = %d, want %d", got, botTeleportCharges-1)
	}
}

func TestAI11ApproachingBarracksWaveDoesNotPullDefenderOffFarm(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	barracks := ai11OwnBarracks(inst)
	if barracks == nil {
		t.Fatal("setup: no allied barracks")
	}
	laneIndex := botNearestLaneToPointLocked(inst.dota, barracks.x, barracks.y)
	lane := inst.dota.m.Lanes[laneIndex]
	for i := 0; i < 3; i++ {
		creep := teleportTestCreep(inst, int32(81110+i), dotaTeamElf, barracks.x-70-float32(i), barracks.y)
		creep.lane = lane
		creep.laneFwd = false
		creep.laneIdx = len(lane) / 3
		inst.mobs[creep.id] = creep
	}
	threat, severity := s.botBarracksThreatLocked(inst, dotaTeamHuman, now)
	if threat != nil || severity != 0 {
		t.Fatalf("distant barracks wave threat = (%v,%d), want no direct threat", threat, severity)
	}

	s.spawnDotaBotsLocked(inst, 8)
	inst.dota.teamPlans = map[int32]botTeamPlan{}
	plan := s.botPlanTeamLocked(inst, dotaTeamHuman, now)
	defenders := 0
	for _, assignment := range plan.Assignments {
		if assignment.Mode == botMacroBase && (assignment.Role == "defender" || assignment.Role == "cover") {
			defenders++
		}
	}
	if plan.Mode == botMacroBase || defenders != 0 {
		t.Fatalf("distant barracks wave pulled %d defenders into base plan=%+v", defenders, plan)
	}
}

func TestAI11FarmApproachesNearbyWaveWithoutGoldLastHitPriority(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	b := &botBrain{c: bot, lane: 0, phase: botPhaseLane}
	full := teleportTestCreep(inst, 81122, dotaTeamElf, bot.x+24, bot.y)
	inst.mobs[full.id] = full

	bot.lock()
	s.botLaneTickLocked(b, now)
	if !bot.hasDest {
		bot.unlock()
		t.Fatal("laning tick did not move toward the nearby live farm wave")
	}
	weak := teleportTestCreep(inst, 81121, dotaTeamElf, bot.x+1, bot.y)
	weak.hp, weak.maxHP = 1, 1
	inst.mobs[weak.id] = weak
	bot.hasDest = false
	b.nextThinkAt = 0
	s.botLaneTickLocked(b, now+0.2)
	bot.unlock()
	if bot.huntState.attackTarget != weak.id || b.farmDecision != "wave_clear" {
		t.Fatalf("in-range farm choice = target %d/decision %q, want creep %d/wave_clear", bot.huntState.attackTarget, b.farmDecision, weak.id)
	}
}

func TestAI11ReadyWaveClearAbilityPrecedesSingleLowHPCreep(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_DPS_Gellar")
	defer cleanup()
	now := float64(s.battleTime())
	hs := bot.huntState
	slot := 0
	for candidate := 1; candidate <= 4; candidate++ {
		def := hs.skillDef(candidate)
		if def.AoERadius > 0 && botSkillHasOp(def, gamedata.OpDamage) {
			slot = candidate
			break
		}
	}
	if slot == 0 {
		t.Fatal("setup: Velial has no authored wave-clear damage ability")
	}
	hs.skillLevel[slot-1] = 1
	hs.mana = hs.maxManaLocked(now)
	last := teleportTestCreep(inst, 81123, dotaTeamElf, bot.x+1, bot.y)
	last.hp, last.maxHP = 1, 1
	clusterA := teleportTestCreep(inst, 81124, dotaTeamElf, bot.x+2, bot.y)
	clusterB := teleportTestCreep(inst, 81125, dotaTeamElf, bot.x+3, bot.y)
	inst.mobs[last.id], inst.mobs[clusterA.id], inst.mobs[clusterB.id] = last, clusterA, clusterB
	b := &botBrain{c: bot, lane: 0, phase: botPhaseLane}

	bot.lock()
	s.botLaneTickLocked(b, now)
	bot.unlock()
	if hs.attackTarget == last.id {
		t.Fatalf("XP-first lane tick selected a gold last hit before ready AoE: attackTarget=%d", hs.attackTarget)
	}
	if hs.cooldownUntil[slot-1] <= now {
		t.Fatalf("XP-first lane tick did not cast ready wave-clear slot %d: cooldownUntil=%.2f now=%.2f", slot, hs.cooldownUntil[slot-1], now)
	}
	if b.farmDecision != "wave_clear_ability" {
		t.Fatalf("XP-first lane decision = %q, want wave_clear_ability", b.farmDecision)
	}
}

func TestAI11AssignmentRetainsEligibleRespondersWhenCloserCandidateAppears(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	barracks := ai11OwnBarracks(inst)
	if barracks == nil {
		t.Fatal("setup: no allied barracks")
	}
	laneIndex := botNearestLaneToPointLocked(inst.dota, barracks.x, barracks.y)
	threat := teleportTestCreep(inst, 81131, dotaTeamElf, barracks.x-2, barracks.y)
	threat.lane = inst.dota.m.Lanes[laneIndex]
	threat.laneIdx = len(threat.lane) / 2
	threat.dtarget = barracks.id
	inst.mobs[threat.id] = threat
	s.spawnDotaBotsLocked(inst, 8)
	first := s.botPlanTeamLocked(inst, dotaTeamHuman, now)
	inst.dota.teamPlans[dotaTeamHuman] = first
	var selected, replacement *botBrain
	for _, brain := range inst.bots {
		if brain.c.playerTeam() != dotaTeamHuman {
			continue
		}
		assignment := first.Assignments[brain.c.objID]
		if assignment.Mode == botMacroBase && assignment.Role == "defender" {
			if selected == nil {
				selected = brain
			}
		} else if replacement == nil {
			replacement = brain
		}
	}
	if selected == nil || replacement == nil {
		t.Fatal("setup: need a selected defender and an eligible replacement")
	}
	replacement.c.x, replacement.c.y = barracks.x, barracks.y
	second := s.botPlanTeamLocked(inst, dotaTeamHuman, now+0.2)
	if got := second.Assignments[selected.c.objID]; got.Role != "defender" || got.Mode != botMacroBase {
		t.Fatalf("eligible prior responder was churned: %+v", got)
	}
}
