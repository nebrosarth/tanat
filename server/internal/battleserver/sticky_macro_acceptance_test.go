package battleserver

import (
	"math"
	"sort"
	"testing"

	"tanatserver/internal/gamedata"
)

func stickyPlanResponders(plan botTeamPlan) []int32 {
	ids := make([]int32, 0)
	for id, assignment := range plan.Assignments {
		if assignment.Mode == botMacroPush || assignment.Mode == botMacroCover ||
			assignment.Mode == botMacroBase || assignment.Mode == botMacroAltar {
			ids = append(ids, id)
		}
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return ids
}

func stickyLaneFixture(t *testing.T, count int32) (*Server, *huntInstance, []*botBrain, float64, func()) {
	t.Helper()
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	now := float64(s.battleTime()) + 20
	p := inst.dota.m.Lanes[0][len(inst.dota.m.Lanes[0])/2]
	bots := make([]*botBrain, 0, count)
	for i := int32(0); i < count; i++ {
		b := macroAddBot(t, s, inst, botIDBase+700+i, dotaTeamHuman, float32(p.X), float32(p.Y), 0)
		b.c.snapT = float32(now)
		bots = append(bots, b)
	}
	creep := &mobState{
		id: 87001, mobIdx: inst.dota.m.HumanCreepMelee, mob: gamedata.Mobs()[inst.dota.m.HumanCreepMelee],
		x: float32(p.X), y: float32(p.Y), hp: 500, maxHP: 500, team: dotaTeamHuman,
		lane: inst.dota.m.Lanes[0], laneIdx: len(inst.dota.m.Lanes[0]) - 1,
	}
	inst.mobs[creep.id] = creep
	return s, inst, bots, now, cleanup
}

func TestStickyMacroPlanKeepsOneKeyAndEligibleRespondersAcrossJitter(t *testing.T) {
	s, inst, bots, now, cleanup := stickyLaneFixture(t, 4)
	defer cleanup()
	rec := &telemetryRecorder{ch: make(chan any, 32)}
	inst.dota.telemetry = rec
	inst.mu.Lock()
	s.botPlanTeamsLocked(inst, now)
	stable := inst.dota.teamPlans[dotaTeamHuman]
	wantKey := botTeamPlanKey(stable)
	wantResponders := stickyPlanResponders(stable)
	if stable.Mode != botMacroPush || len(wantResponders) != 2 {
		inst.mu.Unlock()
		t.Fatalf("setup plan = %+v responders=%v, want push with two responders", stable, wantResponders)
	}
	for tick := 1; tick <= 24; tick++ {
		for i, b := range bots {
			p := inst.dota.m.Lanes[0][len(inst.dota.m.Lanes[0])/2]
			jitter := float32(((tick+i)%5)-2) * 1.25
			macroSetPosition(b.c, float32(p.X)+jitter, float32(p.Y)-jitter, now+float64(tick))
		}
		s.botPlanTeamsLocked(inst, now+float64(tick))
		got := inst.dota.teamPlans[dotaTeamHuman]
		if got.Mode != stable.Mode || got.Lane != stable.Lane || got.ObjectiveID != stable.ObjectiveID {
			inst.mu.Unlock()
			t.Fatalf("sub-material jitter switched live plan at tick %d: got=%+v want=%+v", tick, got, stable)
		}
		if key := botTeamPlanKey(got); key != wantKey {
			inst.mu.Unlock()
			t.Fatalf("sub-material jitter changed plan key at tick %d: %q -> %q", tick, wantKey, key)
		}
		if gotResponders := stickyPlanResponders(got); !sameInt32s(gotResponders, wantResponders) {
			inst.mu.Unlock()
			t.Fatalf("sub-material jitter changed responders at tick %d: got=%v want=%v", tick, gotResponders, wantResponders)
		}
	}
	inst.mu.Unlock()
	if got := len(rec.ch); got != 2 {
		t.Fatalf("unchanged live premise emitted %d team-plan telemetry events, want initial human+elf only", got)
	}
	for _, b := range bots {
		if a := stable.Assignments[b.c.objID]; a.BaselineLane != b.lane {
			t.Fatalf("bot %d baseline lane changed in plan overlay: %+v", b.c.objID, a)
		}
	}
}

func TestFullMobilizationPreparesBeforeAttack(t *testing.T) {
	s, inst, bots, now, cleanup := stickyLaneFixture(t, 3)
	defer cleanup()
	objective, _ := s.botMacroLaneObjectiveLocked(inst, dotaTeamHuman, 0)
	if objective == nil {
		t.Fatal("setup: missing enemy lane objective")
	}
	objective.hp = objective.maxHealth() * 0.50
	rx, ry, ok := botMobilizationRallyPointLocked(inst, dotaTeamHuman, objective)
	if !ok {
		t.Fatal("setup: no mobilization rally point")
	}
	for i, brain := range bots {
		brain.c.huntState.hp = brain.c.huntState.maxHPLocked(now)
		brain.c.huntState.skillLevel[3] = 1
		if i == 1 {
			brain.c.huntState.hp *= 0.40
		}
		macroSetPosition(brain.c, rx, ry, now)
	}
	if s.botMobilizationReadyLocked(inst, dotaTeamHuman, objective, bots, now) {
		t.Fatal("mobilization became ready while one member was below safe HP")
	}
	plan := botTeamPlan{
		Team: dotaTeamHuman, Mode: botMacroPush, Lane: 0, ObjectiveID: objective.id,
		Reason: botMacroReasonMobilizationPreparation, Assignments: make(map[int32]botMacroAssignment),
	}
	s.botAssignMobilizationPreparationLocked(&plan, bots, inst, now)
	for i, brain := range bots {
		a := plan.Assignments[brain.c.objID]
		if i == 1 {
			if a.Mode != botMacroRecover || !brain.retreating {
				t.Fatalf("injured bot preparation assignment=%+v retreating=%v, want recovery", a, brain.retreating)
			}
		} else if a.Mode != botMacroPush || a.Aggressive {
			t.Fatalf("healthy bot preparation assignment=%+v, want non-aggressive muster", a)
		}
	}
	bots[1].c.huntState.hp = bots[1].c.huntState.maxHPLocked(now)
	botClearRetreatLocked(bots[1])
	for _, brain := range bots {
		macroSetPosition(brain.c, rx, ry, now)
	}
	if !s.botMobilizationReadyLocked(inst, dotaTeamHuman, objective, bots, now) {
		t.Fatal("mobilization remained unready after all members recovered and assembled")
	}
	bots[0].c.huntState.deadUntil = now + 5
	if s.botMobilizationReadyLocked(inst, dotaTeamHuman, objective, bots, now) {
		t.Fatal("mobilization launched while one roster member was dead")
	}
	bots[0].c.huntState.deadUntil = 0
	bots[0].c.huntState.cooldownUntil[3] = now + 5
	if s.botMobilizationReadyLocked(inst, dotaTeamHuman, objective, bots, now) {
		t.Fatal("mobilization launched while one ultimate was on cooldown")
	}
	bots[0].c.huntState.cooldownUntil[3] = now
	if !s.botMobilizationReadyLocked(inst, dotaTeamHuman, objective, bots, now) {
		t.Fatal("mobilization stayed blocked after every ultimate became ready")
	}
}

func TestFullMobilizationDoesNotStartBeforeEveryUltimateIsLearned(t *testing.T) {
	_, _, bots, now, cleanup := stickyLaneFixture(t, 3)
	defer cleanup()
	for _, brain := range bots {
		brain.c.huntState.skillLevel[3] = 1
	}
	bots[1].c.huntState.skillLevel[3] = 0
	if botMobilizationUltimatesLearnedLocked(bots) {
		t.Fatal("full mobilization considered attainable with an unlearned ultimate")
	}
	if botMobilizationUltimatesReadyLocked(bots, now) {
		t.Fatal("unlearned ultimate was reported ready")
	}
	bots[1].c.huntState.skillLevel[3] = 1
	if !botMobilizationUltimatesLearnedLocked(bots) || !botMobilizationUltimatesReadyLocked(bots, now) {
		t.Fatal("all learned, off-cooldown ultimates were not accepted")
	}
}

func TestFullMobilizationPlanWaitsForEveryUltimateCooldown(t *testing.T) {
	s, inst, bots, now, cleanup := stickyLaneFixture(t, 3)
	defer cleanup()
	objective, _ := s.botMacroLaneObjectiveLocked(inst, dotaTeamHuman, 0)
	if objective == nil {
		t.Fatal("setup: missing enemy lane objective")
	}
	objective.hp = objective.maxHealth() * 0.50
	rx, ry, ok := botMobilizationRallyPointLocked(inst, dotaTeamHuman, objective)
	if !ok {
		t.Fatal("setup: no mobilization rally point")
	}
	for _, brain := range bots {
		brain.c.huntState.hp = brain.c.huntState.maxHPLocked(now)
		brain.c.huntState.skillLevel[3] = 1
		macroSetPosition(brain.c, rx, ry, now)
	}
	bots[0].c.huntState.cooldownUntil[3] = now + 10

	inst.mu.Lock()
	plan := s.botPlanTeamLocked(inst, dotaTeamHuman, now)
	inst.mu.Unlock()
	if botMobilizationReason(plan.Reason) {
		t.Fatalf("plan entered mobilization while one ultimate was on cooldown: reason=%q plan=%+v", plan.Reason, plan)
	}

	bots[0].c.huntState.cooldownUntil[3] = now
	inst.mu.Lock()
	plan = s.botPlanTeamLocked(inst, dotaTeamHuman, now+1)
	inst.mu.Unlock()
	if plan.Reason != botMacroReasonFullMobilization {
		t.Fatalf("plan did not launch after every ultimate became ready: reason=%q plan=%+v", plan.Reason, plan)
	}
}

func TestFullMobilizationContinuesToNextObjectiveAfterCheckpointDestroy(t *testing.T) {
	s, inst, bots, now, cleanup := stickyLaneFixture(t, 3)
	defer cleanup()
	objective, _ := s.botMacroLaneObjectiveLocked(inst, dotaTeamHuman, 0)
	if objective == nil {
		t.Fatal("setup: missing first enemy lane objective")
	}
	objective.hp = objective.maxHealth() * 0.50
	rx, ry, ok := botMobilizationRallyPointLocked(inst, dotaTeamHuman, objective)
	if !ok {
		t.Fatal("setup: no mobilization rally point")
	}
	for _, brain := range bots {
		brain.c.huntState.hp = brain.c.huntState.maxHPLocked(now)
		brain.c.huntState.skillLevel[3] = 1
		brain.c.huntState.cooldownUntil[3] = now
		macroSetPosition(brain.c, rx, ry, now)
	}

	inst.mu.Lock()
	s.botPlanTeamsLocked(inst, now)
	before := inst.dota.teamPlans[dotaTeamHuman]
	if before.Mode != botMacroPush || before.Reason != botMacroReasonFullMobilization {
		inst.mu.Unlock()
		t.Fatalf("setup plan=%+v, want active full mobilization", before)
	}
	oldObjective := before.ObjectiveID
	oldLane := before.Lane
	inst.mobs[oldObjective].dead = true
	s.botPlanTeamsLocked(inst, now+0.2)
	after := inst.dota.teamPlans[dotaTeamHuman]
	inst.mu.Unlock()

	if after.Mode != botMacroPush || after.Reason != botMacroReasonFullMobilization {
		t.Fatalf("destroying checkpoint interrupted mobilization: before=%+v after=%+v", before, after)
	}
	if after.Lane != oldLane || after.ObjectiveID == oldObjective || after.ObjectiveID == 0 {
		t.Fatalf("mobilization did not advance along the same lane: before=%+v after=%+v", before, after)
	}
}

func TestObjectiveConversionUsesVisibleEnemyPowerOnly(t *testing.T) {
	s, inst, bots, now, cleanup := stickyLaneFixture(t, 3)
	defer cleanup()
	objective, _ := s.botMacroLaneObjectiveLocked(inst, dotaTeamHuman, 0)
	if objective == nil {
		t.Fatal("setup: missing enemy lane objective")
	}
	objective.hp = objective.maxHealth() * 0.50
	for _, brain := range bots {
		brain.c.huntState.hp = brain.c.huntState.maxHPLocked(now)
		macroSetPosition(brain.c, objective.x, objective.y, now)
	}
	// Put several enemy heroes inside the macro approach radius but outside the
	// team's current vision. An omniscient calculation would reject the push;
	// the real-information calculation must not count them as visible pressure.
	for i := int32(0); i < 4; i++ {
		dotaPlayerConn(t, s, inst, botIDBase+900+i, dotaTeamElf, objective.x+100, objective.y)
	}
	if !s.botObjectiveConversionReadyLocked(inst, dotaTeamHuman, objective, now) {
		t.Fatal("hidden enemy heroes incorrectly blocked objective conversion")
	}
}

func TestFullMobilizationGroupDoesNotAttackBeforeRally(t *testing.T) {
	s, inst, bots, now, cleanup := stickyLaneFixture(t, 1)
	defer cleanup()
	objective, _ := s.botMacroLaneObjectiveLocked(inst, dotaTeamHuman, 0)
	if objective == nil {
		t.Fatal("setup: missing enemy lane objective")
	}
	rx, ry, ok := botMobilizationRallyPointLocked(inst, dotaTeamHuman, objective)
	if !ok {
		t.Fatal("setup: no mobilization rally point")
	}
	b := bots[0]
	b.c.huntState.hp = b.c.huntState.maxHPLocked(now)
	macroSetPosition(b.c, rx, ry, now)
	b.c.huntState.attackTarget = objective.id // stale order from the previous phase
	b.macroAssignment = botMacroAssignment{
		Mode: botMacroPush, Lane: 0, ObjectiveID: objective.id,
		Reason: botMacroReasonMobilizationPreparation,
	}

	inst.mu.Lock()
	s.botGroupTickLocked(b, now)
	inst.mu.Unlock()
	if b.c.huntState.attackTarget != 0 {
		t.Fatalf("preparation kept stale attack target %d", b.c.huntState.attackTarget)
	}
	if b.c.huntState.attackActionActive {
		t.Fatal("preparation kept an active attack action")
	}
}

func TestFullMobilizationAttacksOnlyAfterRally(t *testing.T) {
	s, inst, bots, now, cleanup := stickyLaneFixture(t, 1)
	defer cleanup()
	objective, _ := s.botMacroLaneObjectiveLocked(inst, dotaTeamHuman, 0)
	if objective == nil {
		t.Fatal("setup: missing enemy lane objective")
	}
	b := bots[0]
	b.c.huntState.hp = b.c.huntState.maxHPLocked(now)
	macroSetPosition(b.c, objective.x, objective.y, now)
	b.macroAssignment = botMacroAssignment{
		Mode: botMacroPush, Lane: 0, ObjectiveID: objective.id,
		Reason: botMacroReasonFullMobilization, Aggressive: true,
	}

	inst.mu.Lock()
	s.botGroupTickLocked(b, now)
	inst.mu.Unlock()
	if b.c.huntState.attackTarget != objective.id {
		t.Fatalf("ready mobilization did not focus objective: attack=%d objective=%d",
			b.c.huntState.attackTarget, objective.id)
	}
}

func TestObjectiveStagingDoesNotResumeStaleCreepAttack(t *testing.T) {
	s, inst, bots, now, cleanup := stickyLaneFixture(t, 1)
	defer cleanup()
	objective, _ := s.botMacroLaneObjectiveLocked(inst, dotaTeamHuman, 0)
	if objective == nil {
		t.Fatal("setup: missing enemy lane objective")
	}
	rx, ry, ok := botMobilizationRallyPointLocked(inst, dotaTeamHuman, objective)
	if !ok {
		t.Fatal("setup: no objective staging point")
	}
	b := bots[0]
	b.c.huntState.hp = b.c.huntState.maxHPLocked(now)
	macroSetPosition(b.c, rx, ry, now)
	b.c.huntState.attackTarget = 61401 // stale lane order from before the objective plan
	b.c.huntState.attackActionActive = true
	b.macroAssignment = botMacroAssignment{
		Mode: botMacroPush, Lane: 0, ObjectiveID: objective.id,
		Reason: botMacroReasonObjectiveStaging, Aggressive: true,
	}

	inst.mu.Lock()
	s.botGroupTickLocked(b, now)
	inst.mu.Unlock()
	if b.c.huntState.attackTarget != 0 || b.c.huntState.attackActionActive {
		t.Fatalf("objective staging kept stale creep attack: target=%d active=%v",
			b.c.huntState.attackTarget, b.c.huntState.attackActionActive)
	}
}

func TestStickyMacroMaterialLossThreatAndResponderLossAreImmediate(t *testing.T) {
	t.Run("objective loss switches altar plan immediately", func(t *testing.T) {
		s, _, inst, bots, now, cleanup := stickyAltarFixture(t)
		defer cleanup()
		inst.mu.Lock()
		s.botPlanTeamsLocked(inst, now)
		before := inst.dota.teamPlans[dotaTeamHuman]
		if before.Mode != botMacroAltar {
			inst.mu.Unlock()
			t.Fatalf("setup plan = %+v, want altar", before)
		}
		altar := inst.mobs[before.ObjectiveID]
		altar.dead = true
		s.botPlanTeamsLocked(inst, now+0.2)
		after := inst.dota.teamPlans[dotaTeamHuman]
		inst.mu.Unlock()
		if after.Mode == botMacroAltar || after.ObjectiveID == before.ObjectiveID {
			t.Fatalf("dead objective remained authoritative: before=%+v after=%+v", before, after)
		}
		if baseline := stickyBaselineCoverage(after, bots); baseline == 0 {
			t.Fatal("objective loss consumed every baseline lane-coverage assignment")
		}
	})

	t.Run("base threat overrides ordinary lane plan immediately", func(t *testing.T) {
		s, inst, bots, now, cleanup := stickyLaneFixture(t, 4)
		defer cleanup()
		inst.mu.Lock()
		s.botPlanTeamsLocked(inst, now)
		if got := inst.dota.teamPlans[dotaTeamHuman].Mode; got != botMacroPush {
			inst.mu.Unlock()
			t.Fatalf("setup mode=%q, want push", got)
		}
		own := altarOf(inst, dotaTeamHuman)
		if own == nil {
			inst.mu.Unlock()
			t.Fatal("setup: missing own altar")
		}
		macroEnemyCreep(inst, 87002, own.x+35, own.y)
		s.botPlanTeamsLocked(inst, now+0.2)
		plan := inst.dota.teamPlans[dotaTeamHuman]
		inst.mu.Unlock()
		if plan.Mode != botMacroBase || plan.ObjectiveID != own.id {
			t.Fatalf("live base threat plan=%+v, want base objective %d", plan, own.id)
		}
		if baseline := stickyBaselineCoverage(plan, bots); baseline == 0 {
			t.Fatal("base response consumed every baseline lane-coverage assignment")
		}
	})

	for _, mode := range []struct {
		name string
		dead bool
	}{
		{name: "retreating responder", dead: false},
		{name: "dead responder", dead: true},
	} {
		t.Run(mode.name, func(t *testing.T) {
			s, inst, bots, now, cleanup := stickyLaneFixture(t, 4)
			defer cleanup()
			inst.mu.Lock()
			s.botPlanTeamsLocked(inst, now)
			before := inst.dota.teamPlans[dotaTeamHuman]
			responders := stickyPlanResponders(before)
			if len(responders) != 2 {
				inst.mu.Unlock()
				t.Fatalf("setup responders=%v, want two", responders)
			}
			lostID := responders[0]
			if mode.dead {
				inst.bots[lostID].c.huntState.deadUntil = now + 10
			} else {
				inst.bots[lostID].retreating = true
			}
			s.botPlanTeamsLocked(inst, now+0.2)
			after := inst.dota.teamPlans[dotaTeamHuman]
			inst.mu.Unlock()
			got := stickyPlanResponders(after)
			if sameInt32s(got, responders) || len(got) != 2 || got[0] == lostID {
				t.Fatalf("lost responder was not deterministically replaced: before=%v after=%v", responders, got)
			}
			if a := after.Assignments[lostID]; a.Mode != botMacroRecover {
				t.Fatalf("lost responder assignment=%+v, want recover", a)
			}
			if stickyBaselineCoverage(after, bots) == 0 {
				t.Fatal("responder replacement consumed every baseline lane-coverage assignment")
			}
		})
	}
}

func stickyAltarFixture(t *testing.T) (*Server, *conn, *huntInstance, []*botBrain, float64, func()) {
	t.Helper()
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	now := float64(s.battleTime()) + 20
	altar := altarOf(inst, dotaTeamElf)
	if altar == nil {
		cleanup()
		t.Fatal("setup: missing enemy altar")
	}
	macroOpenAltar(inst, altar)
	bots := make([]*botBrain, 0, 4)
	for i := int32(0); i < 4; i++ {
		b := macroAddBot(t, s, inst, botIDBase+740+i, dotaTeamHuman, altar.x+float32(i+1), altar.y, int(i%3))
		b.c.snapT = float32(now)
		bots = append(bots, b)
	}
	bot.snapT = float32(now)
	return s, bot, inst, bots, now, cleanup
}

func stickyBaselineCoverage(plan botTeamPlan, bots []*botBrain) int {
	n := 0
	for _, b := range bots {
		a := plan.Assignments[b.c.objID]
		if a.Mode == botMacroLane && a.Coverage && a.BaselineLane == b.lane {
			n++
		}
	}
	return n
}

func sameInt32s(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func stickyBaseTeleportFixture(t *testing.T) (*Server, *conn, *huntInstance, *botBrain, *mobState, gamedata.Item, float64, *mobState, func()) {
	t.Helper()
	s, bot, inst, b, it, now, cleanup := newTeleportTestBot(t)
	now += 20
	objective := altarOf(inst, bot.playerTeam())
	if objective == nil {
		cleanup()
		t.Fatal("setup: missing own altar")
	}
	pressure := macroEnemyCreep(inst, 88001, objective.x+35, objective.y)
	assignment := botMacroAssignment{
		Mode: botMacroBase, Lane: 0, BaselineLane: 0, ObjectiveID: objective.id,
		Role: "defender", Reason: "own_altar_under_live_pressure",
	}
	b.macroAssignment = assignment
	inst.dota.teamPlans[dotaTeamHuman] = botTeamPlan{
		Team: dotaTeamHuman, Mode: botMacroBase, Lane: 0, ObjectiveID: objective.id,
		Objective: "own_altar", Reason: assignment.Reason,
		Assignments: map[int32]botMacroAssignment{bot.objID: assignment},
	}
	bot.x, bot.y, bot.vx, bot.vy, bot.snapT = objective.x-300, objective.y, 0, 0, float32(now)
	return s, bot, inst, b, objective, it, now, pressure, cleanup
}

func TestBaseDefenseTeleportRequiresLivePremiseAndStartsOnce(t *testing.T) {
	s, bot, _, b, objective, it, now, _, cleanup := stickyBaseTeleportFixture(t)
	defer cleanup()
	bot.lock()
	target, destX, destY, ok := s.botBaseDefenseTeleportTargetLocked(b, now)
	if !ok || target == nil || !target.structure || target.dead || target.team != bot.playerTeam() ||
		dist2(target.x, target.y, objective.x, objective.y) > float32(botBaseDefenseTeleportRadius*botBaseDefenseTeleportRadius) {
		bot.unlock()
		t.Fatalf("base target=(%v,%v,%v,%v), want safe living allied structure near objective %d", target, destX, destY, ok, objective.id)
	}
	walk := s.botTeleportWalkDistanceLocked(bot, destX, destY, now)
	if !s.botTeleportMateriallyFasterLocked(bot, destX, destY, walk, it) {
		bot.unlock()
		t.Fatalf("valid base redeploy was not materially faster: walk=%.1f speed=%.1f", walk, bot.moveSpeedLocked(s))
	}
	if !s.botMaybeStartTeleportLocked(b, now) {
		bot.unlock()
		t.Fatal("valid base defender did not start teleport")
	}
	if got := bot.huntState.bag[it.ArticleID]; got != botTeleportCharges-1 {
		bot.unlock()
		t.Fatalf("base start consumed %d charges, want exactly one from %d", bot.huntState.bag[it.ArticleID], botTeleportCharges)
	}
	if b.pendingTeleport == nil || b.pendingTeleport.targetKind != "base_structure" || b.pendingTeleport.target != target.id {
		bot.unlock()
		t.Fatalf("pending base redeploy=%+v, want target %d/base_structure", b.pendingTeleport, target.id)
	}
	complete := b.pendingTeleport.complete
	s.botTickTeleportLocked(b, complete)
	bot.unlock()
	if b.pendingTeleport != nil || b.macroAssignment.Mode != botMacroBase || b.macroAssignment.Role != "defender" {
		t.Fatalf("successful base redeploy state pending=%v assignment=%+v", b.pendingTeleport, b.macroAssignment)
	}
	if b.laneRedeployPointValid {
		t.Fatal("base redeploy incorrectly created a lane hold")
	}
	if got := bot.huntState.bag[it.ArticleID]; got != botTeleportCharges-1 {
		t.Fatalf("successful base redeploy consumed again: charges=%d", got)
	}
}

func TestBaseDefenseTeleportRejectsInvalidStarts(t *testing.T) {
	cases := []struct {
		name  string
		setup func(*Server, *conn, *huntInstance, *botBrain, *mobState, float64)
	}{
		{name: "not far", setup: func(_ *Server, bot *conn, _ *huntInstance, _ *botBrain, objective *mobState, now float64) {
			bot.x, bot.y, bot.snapT = objective.x+20, objective.y, float32(now)
		}},
		{name: "below safe health", setup: func(_ *Server, bot *conn, _ *huntInstance, _ *botBrain, _ *mobState, now float64) {
			bot.huntState.hp = bot.huntState.maxHPLocked(now) * 0.50
		}},
		{name: "recent damage", setup: func(_ *Server, bot *conn, _ *huntInstance, b *botBrain, _ *mobState, now float64) {
			bot.huntState.hp = bot.huntState.maxHPLocked(now) * 0.60
			b.hpHistory[0] = hpSample{t: now - 0.5, frac: 0.90}
		}},
		{name: "origin contact", setup: func(_ *Server, bot *conn, inst *huntInstance, _ *botBrain, _ *mobState, now float64) {
			macroEnemyCreep(inst, 88002, bot.x, bot.y)
			bot.snapT = float32(now)
		}},
		{name: "assignment changed", setup: func(_ *Server, _ *conn, _ *huntInstance, b *botBrain, _ *mobState, _ float64) {
			b.macroAssignment.Mode = botMacroLane
		}},
		{name: "objective changed", setup: func(_ *Server, _ *conn, _ *huntInstance, b *botBrain, _ *mobState, _ float64) {
			b.macroAssignment.ObjectiveID++
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s, bot, inst, b, objective, it, now, _, cleanup := stickyBaseTeleportFixture(t)
			defer cleanup()
			tc.setup(s, bot, inst, b, objective, now)
			bot.lock()
			started := s.botMaybeStartTeleportLocked(b, now)
			bot.unlock()
			if started || b.pendingTeleport != nil {
				t.Fatalf("invalid base start entered channel: started=%v pending=%+v", started, b.pendingTeleport)
			}
			if got := bot.huntState.bag[it.ArticleID]; got != botTeleportCharges {
				t.Fatalf("invalid base start changed charges: got=%d", got)
			}
		})
	}
}

func TestBaseDefenseTeleportCancelsWithExactReasonsWithoutRefundOrArrival(t *testing.T) {
	cases := []struct {
		name   string
		want   string
		mutate func(*Server, *conn, *huntInstance, *botBrain, *mobState, float64)
	}{
		{name: "live assignment", want: "base_assignment_changed", mutate: func(_ *Server, _ *conn, inst *huntInstance, b *botBrain, _ *mobState, _ float64) {
			plan := inst.dota.teamPlans[dotaTeamHuman]
			plan.Mode = botMacroLane
			plan.Assignments[b.c.objID] = botMacroAssignment{Mode: botMacroLane, Lane: b.lane, BaselineLane: b.lane, Coverage: true}
			inst.dota.teamPlans[dotaTeamHuman] = plan
		}},
		{name: "objective", want: "base_objective_changed", mutate: func(_ *Server, _ *conn, _ *huntInstance, b *botBrain, _ *mobState, _ float64) {
			b.macroAssignment.ObjectiveID++
		}},
		{name: "premise", want: "base_premise_lost", mutate: func(_ *Server, _ *conn, _ *huntInstance, _ *botBrain, pressure *mobState, _ float64) {
			pressure.x, pressure.y = 1000, 1000
		}},
		{name: "target distance", want: "base_target_far", mutate: func(_ *Server, _ *conn, inst *huntInstance, b *botBrain, objective *mobState, _ float64) {
			var target *mobState
			for _, m := range inst.mobs {
				if m.structure && !m.altar && !m.dead && m.team == b.c.playerTeam() &&
					dist2(m.x, m.y, objective.x, objective.y) <= float32(botBaseDefenseTeleportRadius*botBaseDefenseTeleportRadius) {
					target = m
					break
				}
			}
			if target == nil {
				panic("setup: no non-altar allied structure near own altar")
			}
			order := b.pendingTeleport
			order.target = target.id
			target.x, target.y = objective.x+100, objective.y
		}},
		{name: "renewed damage", want: "base_recent_damage", mutate: func(_ *Server, bot *conn, _ *huntInstance, b *botBrain, _ *mobState, now float64) {
			bot.huntState.hp = bot.huntState.maxHPLocked(now) * 0.60
			b.hpHistory[0] = hpSample{t: now - 0.5, frac: 0.90}
		}},
		{name: "origin contact", want: "base_origin_contact", mutate: func(_ *Server, bot *conn, inst *huntInstance, _ *botBrain, _ *mobState, now float64) {
			macroEnemyCreep(inst, 88003, bot.x, bot.y)
			bot.snapT = float32(now)
		}},
		{name: "dead target", want: "base_target_invalid", mutate: func(_ *Server, _ *conn, inst *huntInstance, b *botBrain, _ *mobState, _ float64) {
			objective := inst.mobs[b.macroAssignment.ObjectiveID]
			for _, m := range inst.mobs {
				if m.structure && !m.altar && !m.dead && m.team == b.c.playerTeam() &&
					dist2(m.x, m.y, objective.x, objective.y) <= float32(botBaseDefenseTeleportRadius*botBaseDefenseTeleportRadius) {
					b.pendingTeleport.target = m.id
					m.dead = true
					return
				}
			}
			panic("setup: no non-altar allied structure near own altar")
		}},
		{name: "unsafe target", want: "destination_unsafe", mutate: func(_ *Server, _ *conn, inst *huntInstance, b *botBrain, _ *mobState, _ float64) {
			order := b.pendingTeleport
			for i := int32(0); i < 3; i++ {
				inst.mobs[88010+i] = &mobState{id: 88010 + i, mobIdx: inst.dota.m.ElfCreepMelee, mob: gamedata.Mobs()[inst.dota.m.ElfCreepMelee],
					x: order.targetX, y: order.targetY, hp: 500, maxHP: 500, team: dotaTeamElf}
			}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s, bot, inst, b, objective, _, now, pressure, cleanup := stickyBaseTeleportFixture(t)
			defer cleanup()
			rec := &telemetryRecorder{ch: make(chan any, 8)}
			inst.dota.telemetry = rec
			bot.lock()
			if !s.botMaybeStartTeleportLocked(b, now) {
				bot.unlock()
				t.Fatal("setup: valid base channel did not start")
			}
			originX, originY := bot.x, bot.y
			tc.mutate(s, bot, inst, b, pressure, now)
			s.botTickTeleportLocked(b, now+0.3)
			bot.unlock()
			if b.pendingTeleport != nil {
				t.Fatal("cancelled base channel remained pending")
			}
			if got := bot.huntState.bag[teleportArticleForTest(t)]; got != botTeleportCharges-1 {
				t.Fatalf("base cancellation refunded/changed charge: got=%d", got)
			}
			if math.Hypot(float64(bot.x-originX), float64(bot.y-originY)) > 0.01 {
				t.Fatalf("cancelled base channel arrived/moved: origin=(%.1f,%.1f) now=(%.1f,%.1f)", originX, originY, bot.x, bot.y)
			}
			var cancel telemetryBotTeleport
			for i := 0; i < 2; i++ {
				raw := <-rec.ch
				if ev, ok := raw.(telemetryBotTeleport); ok && ev.Type == "bot_teleport_cancel" {
					cancel = ev
				}
			}
			if cancel.Type != "bot_teleport_cancel" || cancel.CancelReason != tc.want {
				t.Fatalf("cancel telemetry=%+v, want exact reason %q", cancel, tc.want)
			}
			_ = objective
		})
	}
}

func teleportArticleForTest(t *testing.T) int32 {
	t.Helper()
	it, ok := botTeleportScroll()
	if !ok {
		t.Fatal("setup: teleport scroll missing")
	}
	return it.ArticleID
}

func TestUnitLandedStructureAndAltarTelemetryPrecedeOneMatchEndAndFreezeAtEnd(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("TANAT_BOT_TELEMETRY", dir)
	s, attackerConn, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	altar := altarOf(inst, dotaTeamElf)
	if altar == nil {
		t.Fatal("setup: missing enemy altar")
	}
	macroOpenAltar(inst, altar)
	structure := structOfSide(inst, gamedata.DotaGun, dotaTeamElf)
	if structure == nil {
		t.Fatal("setup: missing live enemy structure after altar guards opened")
	}
	hold := structOfSide(inst, gamedata.DotaGenerator, dotaTeamHuman)
	if hold == nil {
		t.Fatal("setup: missing live allied structure for freeze hold")
	}
	attacker := teleportTestCreep(inst, 89001, dotaTeamHuman, 0, 0)
	inst.mobs[attacker.id] = attacker
	attackerConn.huntState.mobs = inst.mobs
	attackerConn.huntState.channels = []channelState{{until: 100, target: hold.id, holdsTarget: true}}
	now := float64(s.battleTime()) + 20
	matchEndAt := now + 1
	hold.st.rootUntil, hold.st.silenceUntil, hold.st.atkSlowUntil = matchEndAt+10, matchEndAt+10, matchEndAt+10
	s.dotaDamageLocked(attackerConn, structure, structure.maxHealth()*100, attacker.id, now)
	s.dotaDamageLocked(attackerConn, altar, altar.maxHealth()*100, attacker.id, matchEndAt)
	if !inst.dota.ended {
		t.Fatal("unit-landed altar kill did not end match")
	}
	lines := readTelemetryLines(t, dir)
	var types []string
	var destroys []map[string]any
	var end map[string]any
	for _, line := range lines {
		switch line["type"] {
		case "structure_destroy":
			types = append(types, "structure_destroy")
			destroys = append(destroys, line)
		case "match_end":
			types = append(types, "match_end")
			end = line
		}
	}
	if !sameStrings(types, []string{"structure_destroy", "structure_destroy", "match_end"}) {
		t.Fatalf("terminal telemetry order=%v, want two destroys then one match_end", types)
	}
	if len(destroys) != 2 || destroys[0]["structure_id"] != float64(structure.id) || destroys[1]["structure_id"] != float64(altar.id) {
		t.Fatalf("structure_destroy lines=%v, want unit structure %d then altar %d", destroys, structure.id, altar.id)
	}
	if end == nil || end["duration"] != end["t"] {
		t.Fatalf("match_end duration/t=%v, want equal match-end duration", end)
	}
	if hold.st.rootUntil > matchEndAt || hold.st.silenceUntil > matchEndAt || hold.st.atkSlowUntil > matchEndAt {
		t.Fatalf("freeze left held target beyond match end: root=%g silence=%g slow=%g end=%g", hold.st.rootUntil, hold.st.silenceUntil, hold.st.atkSlowUntil, matchEndAt)
	}
}

func sameStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
