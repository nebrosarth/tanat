package battleserver

import (
	"testing"

	"tanatserver/internal/battleproto"
)

func TestBaseResponsePreservesPrimaryAndCoverOrderAcrossPremiseJitter(t *testing.T) {
	for _, mode := range []string{botMacroBase, botMacroAltar, botMacroPush} {
		t.Run(mode, func(t *testing.T) {
			s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
			defer cleanup()
			now := float64(s.battleTime()) + 20
			bots := []*botBrain{
				macroAddBot(t, s, inst, botIDBase+910, dotaTeamHuman, 20, 0, 0),
				macroAddBot(t, s, inst, botIDBase+920, dotaTeamHuman, 30, 0, 0),
				macroAddBot(t, s, inst, botIDBase+930, dotaTeamHuman, 40, 0, 0),
			}
			for _, b := range bots {
				b.c.snapT = float32(now)
			}
			makeAssignment := func(b *botBrain) botMacroAssignment {
				return botMacroAssignment{Mode: mode, Lane: 0, BaselineLane: b.lane, ObjectiveID: 777, Coverage: true}
			}
			plan := botTeamPlan{Mode: mode, Lane: 0, ObjectiveID: 777, Assignments: map[int32]botMacroAssignment{}}
			for _, b := range bots {
				plan.Assignments[b.c.objID] = makeAssignment(b)
			}

			inst.mu.Lock()
			if !s.botAssignMacroRespondersLocked(&plan, botTeamPlan{}, false, bots, 0, 0, 3, 1, now) {
				inst.mu.Unlock()
				t.Fatal("initial responder selection failed")
			}
			primary, cover1, cover2 := bots[0], bots[1], bots[2]
			wantPrimaryRole := map[string]string{botMacroBase: "defender", botMacroAltar: "assault", botMacroPush: "push"}[mode]
			if plan.Assignments[primary.c.objID].Role != wantPrimaryRole ||
				plan.Assignments[cover1.c.objID].Role != "cover" || plan.Assignments[cover2.c.objID].Role != "cover" {
				inst.mu.Unlock()
				t.Fatalf("initial roles = %+v, want primary %q then covers", plan.Assignments, wantPrimaryRole)
			}

			previous := plan
			newBot := macroAddBot(t, s, inst, botIDBase+900, dotaTeamHuman, 1, 0, 0)
			newBot.c.snapT = float32(now)
			desired := botTeamPlan{Mode: mode, Lane: 0, ObjectiveID: 777, Assignments: map[int32]botMacroAssignment{}}
			for id, assignment := range previous.Assignments {
				desired.Assignments[id] = assignment
			}
			desired.Assignments[newBot.c.objID] = botMacroAssignment{
				Mode: botMacroLane, Lane: 0, BaselineLane: newBot.lane, ObjectiveID: 777, Coverage: true,
			}
			if !s.botAssignMacroRespondersLocked(&desired, previous, true, append(bots, newBot), 0, 0, 3, 1, now+100) {
				inst.mu.Unlock()
				t.Fatal("selection with a new closer candidate failed")
			}
			if desired.Assignments[primary.c.objID].Role != wantPrimaryRole ||
				desired.Assignments[cover1.c.objID].Role != "cover" ||
				desired.Assignments[cover2.c.objID].Role != "cover" ||
				desired.Assignments[newBot.c.objID].Role != "" {
				inst.mu.Unlock()
				t.Fatalf("new closer candidate reordered responders: primary=%+v cover1=%+v cover2=%+v new=%+v",
					desired.Assignments[primary.c.objID], desired.Assignments[cover1.c.objID], desired.Assignments[cover2.c.objID], desired.Assignments[newBot.c.objID])
			}

			primary.c.huntState.deadUntil = now + 10
			replaced := desired
			if !s.botAssignMacroRespondersLocked(&replaced, desired, true, append(bots, newBot), 0, 0, 3, 1, now+0.2) {
				inst.mu.Unlock()
				t.Fatal("replacement selection failed")
			}
			inst.mu.Unlock()
			if replaced.Assignments[cover1.c.objID].Role != wantPrimaryRole ||
				replaced.Assignments[cover2.c.objID].Role != "cover" ||
				replaced.Assignments[newBot.c.objID].Role != "cover" {
				t.Fatalf("ineligible primary replacement changed cover order: cover1=%+v cover2=%+v new=%+v",
					replaced.Assignments[cover1.c.objID], replaced.Assignments[cover2.c.objID], replaced.Assignments[newBot.c.objID])
			}
		})
	}
}

func TestBasePressureSwitchesToDirectStructureThreatAndReleasesWithoutTimeLatch(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	altar := altarOf(inst, dotaTeamHuman)
	if altar == nil {
		t.Fatal("setup: missing own altar")
	}
	now := float64(s.battleTime()) + 20
	pressure := macroEnemyCreep(inst, 89200, altar.x+41, altar.y)
	macroAddBot(t, s, inst, botIDBase+895, dotaTeamHuman, 300, 0, 0)

	inst.mu.Lock()
	entered := s.botPlanTeamLocked(inst, dotaTeamHuman, now)
	if entered.Mode != botMacroBase {
		inst.mu.Unlock()
		t.Fatalf("threat at 41 selected mode %q, want base", entered.Mode)
	}
	keyAtEntry := botTeamPlanKey(entered)
	laterSameState := s.botPlanTeamLocked(inst, dotaTeamHuman, now+1000)
	if botTeamPlanKey(laterSameState) != keyAtEntry {
		inst.mu.Unlock()
		t.Fatal("equivalent live pressure changed plan with elapsed time")
	}
	inst.dota.teamPlans[dotaTeamHuman] = laterSameState
	pressure.x = altar.x + 50
	nearStructure := s.botPlanTeamLocked(inst, dotaTeamHuman, now+1001)
	if nearStructure.Mode != botMacroBase {
		inst.mu.Unlock()
		t.Fatalf("threat near another base structure did not retain focused defense: mode=%q objective=%d reason=%q", nearStructure.Mode, nearStructure.ObjectiveID, nearStructure.Reason)
	}
	pressure.x = altar.x + 1000
	released := s.botPlanTeamLocked(inst, dotaTeamHuman, now+1002)
	if released.Mode == botMacroBase {
		inst.mu.Unlock()
		t.Fatal("threat outside all direct structure radii retained base defense")
	}
	pressure.x = altar.x + 1000
	altar.dead = true
	dead := s.botPlanTeamLocked(inst, dotaTeamHuman, now+1003)
	inst.mu.Unlock()
	if dead.Mode == botMacroBase {
		t.Fatal("dead altar retained base defense despite nearby threat")
	}
}

func baseAnchorClone(src *mobState, id int32, x, y float32) *mobState {
	clone := *src
	clone.id, clone.x, clone.y = id, x, y
	clone.dead = false
	clone.altar = false
	clone.dtarget, clone.hitTarget, clone.projTarget = 0, 0, 0
	return &clone
}

func TestBaseDefenseAnchorSelectionRejectsUnsafeCandidatesAndPrefersClearance(t *testing.T) {
	s, bot, inst, b, objective, _, now, _, cleanup := stickyBaseTeleportFixture(t)
	defer cleanup()
	near := baseAnchorClone(objective, dotaStructIDBase+901, 8, 0)
	rear := baseAnchorClone(objective, dotaStructIDBase+902, 30, 0)
	objective.x, objective.y = 0, 0
	pressure := teleportTestCreep(inst, 89210, dotaTeamElf, 0, 20)
	setTeleportTestMobs(inst, bot, objective, near, rear, pressure)

	bot.lock()
	nearX, nearY, nearOK := s.botTeleportDestinationLocked(bot, near)
	rearX, rearY, rearOK := s.botTeleportDestinationLocked(bot, rear)
	altarX, altarY, altarOK := s.botTeleportDestinationLocked(bot, objective)
	if !nearOK || !rearOK || !altarOK {
		bot.unlock()
		t.Fatalf("setup destinations: near=%v rear=%v altar=%v", nearOK, rearOK, altarOK)
	}
	nearClearance := s.botTeleportEnemyClearanceLocked(bot, nearX, nearY, now)
	rearClearance := s.botTeleportEnemyClearanceLocked(bot, rearX, rearY, now)
	altarClearance := s.botTeleportEnemyClearanceLocked(bot, altarX, altarY, now)
	if rearClearance <= altarClearance || rearClearance <= nearClearance {
		bot.unlock()
		t.Fatalf("rear clearance %.2f did not materially beat altar %.2f/near %.2f", rearClearance, altarClearance, nearClearance)
	}
	target, _, _, ok := s.botBaseDefenseTeleportTargetLocked(b, now)
	bot.unlock()
	if !ok || target != rear {
		t.Fatalf("directly pressured altar/near anchor selected target=%v, want safer rear %d", target, rear.id)
	}

	invalid := baseAnchorClone(objective, dotaStructIDBase+903, 12, 0)
	invalid.dotaPrefab = ""
	dead := baseAnchorClone(objective, dotaStructIDBase+904, 14, 0)
	dead.dead = true
	far := baseAnchorClone(objective, dotaStructIDBase+905, 100, 0)
	unsafe := baseAnchorClone(objective, dotaStructIDBase+906, 10, 0)
	enemies := []*mobState{
		teleportTestCreep(inst, 89211, dotaTeamElf, unsafe.x, unsafe.y),
		teleportTestCreep(inst, 89212, dotaTeamElf, unsafe.x, unsafe.y),
		teleportTestCreep(inst, 89213, dotaTeamElf, unsafe.x, unsafe.y),
	}
	setTeleportTestMobs(inst, bot, objective, rear, invalid, dead, far, unsafe, pressure, enemies[0], enemies[1], enemies[2])
	bot.lock()
	if s.botTeleportTargetValidLocked(b, invalid) || s.botTeleportTargetValidLocked(b, dead) {
		bot.unlock()
		t.Fatal("invalid or dead anchor passed target eligibility")
	}
	if _, _, ok := s.botTeleportDestinationLocked(bot, unsafe); ok {
		bot.unlock()
		t.Fatal("unsafe anchor retained a destination")
	}
	target, _, _, ok = s.botBaseDefenseTeleportTargetLocked(b, now)
	bot.unlock()
	if !ok || target != rear {
		t.Fatalf("candidate filtering selected target=%v, want useful rear %d", target, rear.id)
	}
}

func TestBaseDefenseAnchorTieBreaksDeterministicallyByID(t *testing.T) {
	s, bot, inst, b, objective, _, now, _, cleanup := stickyBaseTeleportFixture(t)
	defer cleanup()
	objective.x, objective.y = 0, 0
	first := baseAnchorClone(objective, dotaStructIDBase+920, 20, 0)
	second := baseAnchorClone(objective, dotaStructIDBase+921, -20, 0)
	enemy1 := teleportTestCreep(inst, 89220, dotaTeamElf, 0, 0)
	enemy2 := teleportTestCreep(inst, 89221, dotaTeamElf, 0, 0)
	enemy3 := teleportTestCreep(inst, 89222, dotaTeamElf, 0, 0)
	setTeleportTestMobs(inst, bot, objective, first, second, enemy1, enemy2, enemy3)

	bot.lock()
	target, _, _, ok := s.botBaseDefenseTeleportTargetLocked(b, now)
	if !ok || target != first {
		bot.unlock()
		t.Fatalf("equal-clearance anchor tie selected %v, want lower ID anchor %d", target, first.id)
	}
	target, _, _, ok = s.botBaseDefenseTeleportTargetLocked(b, now+1000)
	bot.unlock()
	if !ok || target != first {
		t.Fatalf("reordered equal-clearance tie selected %v, want stable lower ID anchor %d", target, first.id)
	}
}

func TestBaseDefenseRedeployAcceptsRoleOnlyChangeButCancelsWhenResponderLeaves(t *testing.T) {
	for _, initialRole := range []string{"defender", "cover"} {
		t.Run(initialRole, func(t *testing.T) {
			s, bot, inst, b, objective, it, now, _, cleanup := stickyBaseTeleportFixture(t)
			defer cleanup()
			wire := captureBotTeleportVisual(t, bot)
			b.macroAssignment.Role = initialRole
			plan := inst.dota.teamPlans[dotaTeamHuman]
			plan.Assignments[bot.objID] = b.macroAssignment
			inst.dota.teamPlans[dotaTeamHuman] = plan

			bot.lock()
			if !s.botMaybeStartTeleportLocked(b, now) {
				bot.unlock()
				t.Fatal("redeploy-eligible role did not start channel")
			}
			order := *b.pendingTeleport
			if order.targetKind != "base_structure" || order.macroObjective != objective.id {
				bot.unlock()
				t.Fatalf("order=%+v, want base structure for objective %d", order, objective.id)
			}
			roleOnly := "defender"
			if initialRole == "defender" {
				roleOnly = "cover"
			}
			b.macroAssignment.Role = roleOnly
			plan = inst.dota.teamPlans[dotaTeamHuman]
			plan.Assignments[bot.objID] = b.macroAssignment
			inst.dota.teamPlans[dotaTeamHuman] = plan
			s.botTickTeleportLocked(b, now+0.3)
			if b.pendingTeleport == nil {
				bot.unlock()
				t.Fatal("role-only responder change cancelled active channel")
			}
			if bot.huntState.bag[it.ArticleID] != botTeleportCharges-1 || b.laneRedeployPointValid {
				bot.unlock()
				t.Fatalf("role-only change altered charge/hold: charges=%d hold=%v", bot.huntState.bag[it.ArticleID], b.laneRedeployPointValid)
			}

			plan.Mode = botMacroLane
			delete(plan.Assignments, bot.objID)
			inst.dota.teamPlans[dotaTeamHuman] = plan
			b.macroAssignment = botMacroAssignment{Mode: botMacroLane, Lane: b.lane, BaselineLane: b.lane, Coverage: true}
			s.botTickTeleportLocked(b, now+0.6)
			bot.unlock()
			if b.pendingTeleport != nil {
				t.Fatal("responder removal left active channel pending")
			}
			if bot.huntState.bag[it.ArticleID] != botTeleportCharges-1 || b.laneRedeployPointValid {
				t.Fatalf("cancellation changed charge/hold: charges=%d hold=%v", bot.huntState.bag[it.ArticleID], b.laneRedeployPointValid)
			}
			if got := len(botTeleportVisualEffectPackets(wire.packets(t), battleproto.CmdEffectEnd)); got != 2 {
				t.Fatalf("responder removal EFFECT_END count=%d, want 2", got)
			}
		})
	}
}

func TestBaseDefenseAnchorSelectionHasNoTimeDependentTie(t *testing.T) {
	s, bot, inst, b, objective, _, now, _, cleanup := stickyBaseTeleportFixture(t)
	defer cleanup()
	objective.x, objective.y = 0, 0
	left := baseAnchorClone(objective, dotaStructIDBase+940, -10, 0)
	right := baseAnchorClone(objective, dotaStructIDBase+941, 10, 0)
	setTeleportTestMobs(inst, bot, objective, left, right)
	bot.lock()
	first, _, _, ok := s.botBaseDefenseTeleportTargetLocked(b, now)
	second, _, _, okLater := s.botBaseDefenseTeleportTargetLocked(b, now+1000)
	bot.unlock()
	if !ok || !okLater || first != second {
		t.Fatalf("equivalent anchor state changed with time: first=%v/%v second=%v/%v", first, ok, second, okLater)
	}
}
