package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

func TestAI10FarmDirectorChoosesSafeWaveTargetAndOwnsLane(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	lane := inst.dota.m.Lanes[0]
	bot.x, bot.y, bot.snapT = float32(lane[len(lane)/2].X), float32(lane[len(lane)/2].Y), float32(now)
	weak := teleportTestCreep(inst, 71001, dotaTeamElf, bot.x+1, bot.y)
	weak.hp, weak.maxHP = 1, weak.mob.Health
	weak.lane, weak.laneIdx = lane, len(lane)/2
	full := teleportTestCreep(inst, 71002, dotaTeamElf, bot.x+2, bot.y)
	full.hp, full.maxHP = full.mob.Health, full.mob.Health
	full.lane, full.laneIdx = lane, len(lane)/2
	inst.mobs[weak.id], inst.mobs[full.id] = weak, full
	b := &botBrain{c: bot, lane: 0, phase: botPhaseLane}

	bot.lock()
	defer bot.unlock()
	target := s.botFindLaneTargetLocked(b, now, botLaneEngageRadius, false)
	if target != weak {
		t.Fatalf("farm target = %v, want nearest safe wave creep %d", target, weak.id)
	}
	if b.farmDecision != "wave_clear" || b.farmLane != 0 {
		t.Fatalf("farm telemetry state = decision=%q lane=%d, want wave_clear/lane 0", b.farmDecision, b.farmLane)
	}
}

func TestAI10SkillAndItemPriorityAreRoleAware(t *testing.T) {
	s, bot, _, cleanup := newDotaConn(t, "Avtr_Sp_Arianna")
	defer cleanup()
	b := &botBrain{c: bot, lane: 0}
	bot.lock()
	defer bot.unlock()
	hs := bot.huntState
	slot := botChooseSkillToLevelLocked(hs)
	if slot == 0 {
		t.Fatal("support bot found no learnable skill")
	}
	chosen := hs.skillDef(slot)
	if !botSkillHasOpDeep(chosen, gamedata.OpHeal) && !botSkillHasOpDeep(chosen, gamedata.OpShield) && !botSkillHasOpDeep(chosen, gamedata.OpHot) {
		t.Fatalf("support first skill slot = %d, want a recovery/support operation", slot)
	}

	s.Store.CreateBotHero(bot.selfPlayerID, "ai10-item-bot")
	s.botBuyItemsLocked(b, float64(s.battleTime()))
	if len(hs.ownedTreeItems) == 0 {
		t.Fatal("support bot did not buy an affordable frontier item")
	}
	for article := range hs.ownedTreeItems {
		it, ok := gamedata.AvatarItemByArticle(article)
		if !ok {
			t.Fatalf("owned unknown avatar item %d", article)
		}
		if !botItemRecoveryUtility(it) && it.TreeID != gamedata.AvatarTreeSupport && it.TreeID != gamedata.AvatarTreeMagic {
			t.Fatalf("support bot bought non-role item %+v", it)
		}
	}
}

func TestAI10BarracksThreatAssignsDefenderSeparatelyFromAltar(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	barracks := structOfSide(inst, gamedata.DotaCreepTower, dotaTeamHuman)
	if barracks == nil {
		t.Fatal("setup: no allied barracks")
	}
	laneIndex := botNearestLaneToPointLocked(inst.dota, barracks.x, barracks.y)
	creep := teleportTestCreep(inst, 71003, dotaTeamElf, barracks.x-2, barracks.y)
	creep.lane = inst.dota.m.Lanes[laneIndex]
	creep.laneIdx = len(creep.lane) / 2
	creep.dtarget = barracks.id
	inst.mobs[creep.id] = creep
	b := &botBrain{c: bot, lane: 1, phase: botPhaseLane}
	inst.bots[bot.objID] = b
	if inst.dota.teamPlans == nil {
		inst.dota.teamPlans = map[int32]botTeamPlan{}
	}

	bot.lock()
	defer bot.unlock()
	threat, severity := s.botBarracksThreatLocked(inst, dotaTeamHuman, now)
	if threat != barracks || severity <= 0 {
		t.Fatalf("barracks threat = (%v,%d), want allied barracks and positive severity", threat, severity)
	}
	plan := s.botPlanTeamLocked(inst, dotaTeamHuman, now)
	if plan.Mode != botMacroBase || plan.ObjectiveID != barracks.id || plan.Objective == "own_altar" {
		t.Fatalf("barracks plan = %+v, want base defense of barracks", plan)
	}
	assignment := plan.Assignments[bot.objID]
	if assignment.Mode != botMacroBase || assignment.Role != "defender" || assignment.ObjectiveID != barracks.id {
		t.Fatalf("barracks defender assignment = %+v", assignment)
	}
}

func TestAI10GunThreatAssignsAFocusedDefender(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	gun := structOfSide(inst, gamedata.DotaGun, dotaTeamHuman)
	if gun == nil {
		t.Fatal("setup: no allied gun")
	}
	attacker := teleportTestCreep(inst, 71004, dotaTeamElf, gun.x+1, gun.y)
	attacker.dtarget = gun.id
	inst.mobs[attacker.id] = attacker
	s.spawnDotaBotsLocked(inst, 8)

	plan := s.botPlanTeamLocked(inst, dotaTeamHuman, now)
	if plan.Mode != botMacroBase || plan.ObjectiveID != gun.id || plan.Reason != "own_gun_under_live_threat" {
		t.Fatalf("gun defense plan = %+v, want focused live gun defense for %d", plan, gun.id)
	}
	defenders := 0
	for _, assignment := range plan.Assignments {
		if assignment.Mode == botMacroBase {
			defenders++
		}
	}
	if defenders == 0 {
		t.Fatalf("gun defense plan assigned no defender: %+v", plan.Assignments)
	}
	attacker.x, attacker.y, attacker.dtarget = gun.x+80, gun.y, 0
	if severity := s.botDefenseStructureThreatSeverityLocked(inst, dotaTeamHuman, gun, now); severity != 0 {
		t.Fatalf("distant gun threat severity = %d, want no direct defense trigger", severity)
	}
}

func TestAI10FocusTargetIsDeterministicAndSkipsRetreatingBot(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	low := dotaPlayerConn(t, s, inst, 72002, dotaTeamElf, bot.x+3, bot.y)
	high := dotaPlayerConn(t, s, inst, 72001, dotaTeamElf, bot.x+3, bot.y)
	low.huntState.hp = low.huntState.maxHPLocked(now) * 0.2
	high.huntState.hp = high.huntState.maxHPLocked(now) * 0.8
	inst.bots[low.objID] = &botBrain{c: low, retreating: true, retreatMode: botRetreatModeDisengage}
	for i := 0; i < 10; i++ {
		if got := botSelectTeamFocusTargetLocked(inst, dotaTeamHuman, now); got != high {
			t.Fatalf("focus target iteration %d = %v, want non-retreating enemy %d", i, got, high.objID)
		}
	}
}

func TestAI10FocusPrefersActiveAttackerOverRolePriority(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	victim := dotaPlayerConn(t, s, inst, 72003, dotaTeamHuman, bot.x+3, bot.y)
	victim.huntState.hp = victim.huntState.maxHPLocked(now) * 0.35
	attacker := dotaPlayerConn(t, s, inst, 72004, dotaTeamElf, bot.x+4, bot.y)
	attacker.huntState.pvpTarget = victim.objID
	attacker.huntState.attackTarget = victim.objID
	support := dotaPlayerConn(t, s, inst, 72005, dotaTeamElf, bot.x+5, bot.y)
	support.huntState.av = avatarByPrefab(t, "Avtr_Sp_Arianna")
	support.huntState.hp = support.huntState.maxHPLocked(now)
	if got := botSelectTeamFocusTargetLocked(inst, dotaTeamHuman, now); got != attacker {
		t.Fatalf("focus target = %v, want active attacker %d over support %d", got, attacker.objID, support.objID)
	}
}

func TestAI10RiskDirectorAndTeleportRejectBadValue(t *testing.T) {
	if got := botTacticalRiskValue(20, 8, 15); got >= 0 {
		t.Fatalf("risk value = %.1f, want negative rejection", got)
	}
	s, bot, _, b, it, now, cleanup := newTeleportTestBot(t)
	defer cleanup()
	bot.lock()
	defer bot.unlock()
	if s.botTeleportMateriallyFasterLocked(bot, bot.x+1, bot.y, 1, it) {
		t.Fatal("teleport accepted a one-unit walk that is not materially faster")
	}
	if b.pendingTeleport != nil {
		t.Fatal("teleport rejection left a pending channel")
	}
	_ = now
}

func TestAI10TelemetryIncludesFarmAndFocusFields(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	inst.dota.startedAt = now - 120
	bot.huntState.xp = 300
	b := &botBrain{c: bot, lane: 0, phase: botPhaseLane, farmDecision: "wave_clear", farmTarget: 7001, farmTargetScore: 99, farmLane: 0, farmCatchUp: true, farmLastHits: 0, farmWaveClears: 2}
	inst.bots[bot.objID] = b
	inst.dota.teamPlans = map[int32]botTeamPlan{dotaTeamHuman: {Team: dotaTeamHuman, FocusTarget: 72001}}
	dir := t.TempDir()
	rec := newTelemetryRecorder(dir, inst.dota.m.ID)
	if rec == nil {
		t.Fatal("telemetry recorder unavailable")
	}
	bot.lock()
	s.telemetrySnapshotLocked(rec, bot, 1, now)
	bot.unlock()
	rec.close()
	lines := readTelemetryLines(t, dir)
	if len(lines) != 1 {
		t.Fatalf("telemetry lines = %d, want 1", len(lines))
	}
	line := lines[0]
	for key, want := range map[string]any{"xp": float64(300), "xp_per_minute": float64(150), "farm_decision": "wave_clear", "farm_target": float64(7001), "farm_lane": float64(0), "focus_target": float64(72001)} {
		if line[key] != want {
			t.Errorf("telemetry %s = %v, want %v", key, line[key], want)
		}
	}
}
