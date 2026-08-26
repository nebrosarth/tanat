package battleserver

import (
	"cmp"
	"math"
	"reflect"
	"slices"
	"testing"

	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
)

func externalAssaultReset(seed int64, maxSteps uint32) AssaultResetV1 {
	return AssaultResetV1{Seed: seed, MaxSteps: maxSteps}
}

func TestAssaultSortEntitiesPreservesObservationOrderContract(t *testing.T) {
	got := []assaultEntity{
		{id: 41, kind: 2, distance: 4}, {id: 13, kind: 1, distance: 8},
		{id: 31, kind: 3, distance: 2}, {id: 12, kind: 1, distance: 2},
		{id: 42, kind: 2, distance: 2}, {id: 32, kind: 3, distance: 2},
		{id: 11, kind: 1, distance: 2}, {id: 43, kind: 2, distance: 2},
	}
	want := slices.Clone(got)
	slices.SortFunc(want, func(a, b assaultEntity) int {
		if pi, pj := assaultEntityPriority(a.kind), assaultEntityPriority(b.kind); pi != pj {
			return cmp.Compare(pi, pj)
		}
		if a.distance != b.distance {
			return cmp.Compare(a.distance, b.distance)
		}
		return cmp.Compare(a.id, b.id)
	})
	assaultSortEntities(got)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("sorted entities=%v, want %v", got, want)
	}
}

func TestAssaultTopEntitiesPreservesTruncatedObservationOrder(t *testing.T) {
	entities := make([]assaultEntity, 0, AssaultMaxEntities+40)
	for i := 0; i < AssaultMaxEntities+40; i++ {
		entities = append(entities, assaultEntity{
			// Deliberately scramble ids and priorities to cover heap replacement,
			// equal-distance ties and the category ordering simultaneously.
			id:       int32((i*37)%211 + 1),
			kind:     []int{2, 3, 1}[i%3],
			distance: float64((i * 11) % 29),
		})
	}
	want := slices.Clone(entities)
	slices.SortFunc(want, func(a, b assaultEntity) int {
		if assaultEntityLess(a, b) {
			return -1
		}
		if assaultEntityLess(b, a) {
			return 1
		}
		return 0
	})
	got := assaultTopEntities(slices.Clone(entities), new([]assaultEntity))
	if !reflect.DeepEqual(got, want[:AssaultMaxEntities]) {
		t.Fatalf("top entities=%v, want %v", got, want[:AssaultMaxEntities])
	}
}

func TestAssaultRewardV2ZeroSumAndTeamSpirit(t *testing.T) {
	env := NewAssaultEnv()
	defer env.Close()
	if _, err := env.Reset(externalAssaultReset(17, 20)); err != nil {
		t.Fatal(err)
	}
	env.inst.mu.Lock()
	now := env.clock.Now()
	env.heroes[0].huntState.xp += 100
	result := env.resultLocked(nil, now)
	env.inst.mu.Unlock()
	want := [AssaultHeroCount]float64{0.168, 0.008, 0.008, 0.008, 0.008, -0.04, -0.04, -0.04, -0.04, -0.04}
	var human, elf float64
	for i, got := range result.Reward {
		if math.Abs(float64(got)-want[i]) > 1e-6 {
			t.Fatalf("reward[%d]=%.6f, want %.6f", i, got, want[i])
		}
		if i < AssaultHeroCount/2 {
			human += float64(got)
		} else {
			elf += float64(got)
		}
	}
	if math.Abs(human+elf) > 1e-6 {
		t.Fatalf("team rewards are not zero-sum: human=%g elf=%g", human, elf)
	}
}

func TestAssaultExecutedActionTelemetryIsDeterministicAndFailClosed(t *testing.T) {
	tests := []struct {
		name       string
		action     HeroActionV1
		navigation bool
		valid      uint8
		reason     uint8
	}{
		{name: "accepted wait", action: HeroActionV1{Kind: AssaultActionWait}, valid: 1, reason: AssaultRejectionReasonNone},
		{name: "accepted exact navigation offset", action: HeroActionV1{Kind: AssaultActionMove, Direction: 80}, navigation: true, valid: 1, reason: AssaultRejectionReasonNone},
		{name: "invalid navigation direction", action: HeroActionV1{Kind: AssaultActionMove, Direction: 81}, navigation: true, valid: 0, reason: AssaultRejectionReasonInvalid},
		{name: "invalid navigation anchor", action: HeroActionV1{Kind: AssaultActionMove, Direction: 40, Distance: 15}, navigation: true, valid: 0, reason: AssaultRejectionReasonInvalid},
		{name: "masked teleport", action: HeroActionV1{Kind: AssaultActionTeleport}, valid: 0, reason: AssaultRejectionReasonMasked},
		{name: "invalid kind", action: HeroActionV1{Kind: 255}, valid: 0, reason: AssaultRejectionReasonInvalid},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			env := NewAssaultEnv()
			defer env.Close()
			env.ConfigureNavigationActions(test.navigation)
			if _, err := env.Reset(externalAssaultReset(700, 10)); err != nil {
				t.Fatal(err)
			}
			actions := [AssaultHeroCount]HeroActionV1{}
			actions[0] = test.action
			result, err := env.Step(actions)
			if err != nil {
				t.Fatal(err)
			}
			if result.ExecutedValid[0] != test.valid || result.RejectionReason[0] != test.reason {
				t.Fatalf("valid/reason=%d/%d, want %d/%d", result.ExecutedValid[0], result.RejectionReason[0], test.valid, test.reason)
			}
			if test.valid == 0 && result.ExecutedActions[0] != (HeroActionV1{}) {
				t.Fatalf("rejected action was reported as executed: %+v", result.ExecutedActions[0])
			}
			if test.valid == 1 && result.ExecutedActions[0] != test.action {
				t.Fatalf("executed action=%+v, want exact submitted action %+v", result.ExecutedActions[0], test.action)
			}
		})
	}
}

func TestAssaultAI42HoldPreservesAndIdleCancelsMovement(t *testing.T) {
	env := NewAssaultEnv()
	defer env.Close()
	if _, err := env.Reset(externalAssaultReset(701, 10)); err != nil {
		t.Fatal(err)
	}
	hero := env.heroes[0]
	env.inst.mu.Lock()
	hero.hasDest = true
	hero.destX, hero.destY = hero.x+100, hero.y+100
	env.inst.mu.Unlock()
	var actions [AssaultHeroCount]HeroActionV1
	var controls [AssaultHeroCount]AssaultControlV1
	controls[0] = AssaultControlHold
	holdResult, err := env.StepControlled(actions, controls)
	if err != nil {
		t.Fatal(err)
	}
	env.inst.mu.Lock()
	preserved := hero.hasDest
	env.inst.mu.Unlock()
	if !preserved {
		t.Fatal("HOLD cancelled the active movement order")
	}
	if holdResult.ActiveOrder[0] != 1 {
		t.Fatal("HOLD response did not expose the active movement order")
	}
	controls[0] = AssaultControlIdle
	idleResult, err := env.StepControlled(actions, controls)
	if err != nil {
		t.Fatal(err)
	}
	env.inst.mu.Lock()
	cancelled := !hero.hasDest && hero.arrival == nil && len(hero.path) == 0
	env.inst.mu.Unlock()
	if !cancelled {
		t.Fatal("IDLE did not cancel the active movement order")
	}
	if idleResult.ActiveOrder[0] != 0 {
		t.Fatal("IDLE response retained a cancelled movement order")
	}
}

func TestAssaultStrategicRewardTimeWeightAndPostZeroSumDraw(t *testing.T) {
	if got := assaultShapingTimeWeight(600); math.Abs(got-0.6) > 1e-12 {
		t.Fatalf("shaping weight at 10 minutes = %g, want 0.6", got)
	}
	env := NewAssaultEnv()
	env.ConfigureWrongLaneCurriculum(true, false)
	env.ConfigureNavigationActions(true)
	env.ConfigureStrategicReward(true)
	defer env.Close()
	initial, err := env.Reset(externalAssaultReset(41, 10))
	if err != nil {
		t.Fatal(err)
	}
	if initial.RewardHash != AssaultRewardHashV5 {
		t.Fatalf("strategic reward hash = %x", initial.RewardHash[:4])
	}
	env.inst.mu.Lock()
	env.step = env.maxSteps
	result := env.resultLocked(nil, env.clock.Now())
	repeated := env.resultLocked(nil, env.clock.Now())
	env.inst.mu.Unlock()
	for index, reward := range result.Reward {
		if math.Abs(float64(reward)+2) > 1e-6 {
			t.Fatalf("draw reward[%d] = %g, want -2", index, reward)
		}
		if math.Abs(float64(repeated.Reward[index])) > 1e-6 {
			t.Fatalf("repeated terminal reward[%d] = %g, want 0", index, repeated.Reward[index])
		}
	}
}

func TestAssaultTanatLastHitCalibrationPreservesOpenAIFiveCorrection(t *testing.T) {
	if assaultOpenAIFiveCreepKillCorrection != -0.16 {
		t.Fatalf("OpenAI Five last-hit correction = %g, want -0.16",
			assaultOpenAIFiveCreepKillCorrection)
	}
	dm, ok := gamedata.DotaMapByID(101)
	if !ok {
		t.Fatal("classic Assault map is missing")
	}
	meleeIdx, rangedIdx := dm.CreepMobIdx(gamedata.DotaSideHuman)
	melee, ranged := gamedata.MobByIndex(meleeIdx), gamedata.MobByIndex(rangedIdx)
	lastHitReward := func(m gamedata.Mob) float64 {
		return m.XP*assaultOpenAIFiveXPGainReward +
			float64(m.Coins)*assaultTanatMoneyGainReward +
			assaultOpenAIFiveCreepKillCorrection + assaultTanatCreepLastHitBonus
	}
	if got := lastHitReward(melee); math.Abs(got-0.384) > 1e-12 {
		t.Fatalf("melee last-hit reward = %g, want 0.384", got)
	}
	if got := lastHitReward(ranged); math.Abs(got-0.448) > 1e-12 {
		t.Fatalf("ranged last-hit reward = %g, want 0.448", got)
	}
	if got := (3*lastHitReward(melee) + lastHitReward(ranged)) / 4; math.Abs(got-0.4) > 1e-12 {
		t.Fatalf("standard-wave mean last-hit reward = %g, want 0.4", got)
	}
}

func TestAssaultWrongLaneCurriculumAssignmentObservationAndReward(t *testing.T) {
	env := NewAssaultEnv()
	env.EnableWrongLaneCurriculum(true)
	defer env.Close()
	reset := externalAssaultReset(410041, 20)
	reset.Controllers = [AssaultHeroCount]AssaultControllerV1{
		AssaultControllerAI40, AssaultControllerAI40, AssaultControllerAI40,
		AssaultControllerAI40, AssaultControllerAI40, AssaultControllerAI40,
		AssaultControllerAI40, AssaultControllerAI40, AssaultControllerAI40,
		AssaultControllerAI40,
	}
	initial, err := env.Reset(reset)
	if err != nil {
		t.Fatal(err)
	}
	if initial.RewardHash != AssaultRewardHashV3 ||
		env.laneAssignmentUntil < assaultLaneMinSeconds ||
		env.laneAssignmentUntil > assaultLaneMaxSeconds {
		t.Fatalf("wrong-lane contract not initialized: hash=%x until=%g",
			initial.RewardHash[:4], env.laneAssignmentUntil)
	}
	for side := 0; side < 2; side++ {
		var counts [3]int
		for slot := side * 5; slot < (side+1)*5; slot++ {
			lane := int(env.laneAssignment[slot])
			counts[lane]++
			obs := initial.Observations[slot].Global
			if obs[8] != 1 || obs[9+lane] != 1 || obs[12] != 0 || obs[13] != 0 {
				t.Fatalf("slot %d lane observation=%v", slot, obs[8:14])
			}
		}
		if counts != [3]int{2, 1, 2} {
			t.Fatalf("side %d lane split=%v, want 2-1-2", side+1, counts)
		}
	}

	var penalized, correct StepResultV1
	env.inst.mu.Lock()
	now := env.clock.Now()
	assigned := int(env.laneAssignment[0])
	wrongLane := 0
	if assigned == 0 {
		wrongLane = 2
	}
	wrongPath := env.inst.dota.m.Lanes[wrongLane]
	wrongPoint := wrongPath[len(wrongPath)/2]
	hero := env.heroes[0]
	hero.stopArrivalLocked()
	hero.hasDest = false
	hero.x, hero.y, hero.vx, hero.vy, hero.snapT =
		float32(wrongPoint.X), float32(wrongPoint.Y), 0, 0, float32(now)
	env.step = 1
	penalized = env.resultLocked(nil, now)
	assignedPath := env.inst.dota.m.Lanes[assigned]
	assignedPoint := assignedPath[len(assignedPath)/2]
	hero.x, hero.y, hero.snapT = float32(assignedPoint.X), float32(assignedPoint.Y), float32(now)
	env.step = 2
	correct = env.resultLocked(nil, now)
	env.inst.mu.Unlock()

	if penalized.Observations[0].Global[12] != 1 {
		t.Fatal("hero outside its assigned corridor is not marked wrong-lane")
	}
	want := [AssaultHeroCount]float64{
		-0.0252, -0.0012, -0.0012, -0.0012, -0.0012,
		0.006, 0.006, 0.006, 0.006, 0.006,
	}
	for i, got := range penalized.Reward {
		if math.Abs(float64(got)-want[i]) > 1e-6 {
			t.Fatalf("wrong-lane reward[%d]=%.6f, want %.6f", i, got, want[i])
		}
	}

	if correct.Observations[0].Global[12] != 0 {
		t.Fatal("hero on its assigned lane is marked wrong-lane")
	}
	for i, got := range correct.Reward {
		if math.Abs(float64(got)) > 1e-6 {
			t.Fatalf("on-lane reward[%d]=%.6f, want zero", i, got)
		}
	}
}

func TestRepeatedAttackOrderKeepsCurrentSession(t *testing.T) {
	env := NewAssaultEnv()
	defer env.Close()
	if _, err := env.Reset(externalAssaultReset(410042, 20)); err != nil {
		t.Fatal(err)
	}
	env.inst.mu.Lock()
	defer env.inst.mu.Unlock()

	c := env.heroes[0]
	var target *mobState
	for _, mob := range env.inst.mobs {
		if !mob.dead && mob.enemyOf(c.playerTeam()) {
			target = mob
			break
		}
	}
	if target == nil {
		t.Fatal("assault reset produced no enemy attack target")
	}
	env.server.startAttackLocked(c, target)
	c.policyAttackRetargetAt = float64(env.server.battleTime())
	seq, moveGen := c.huntState.attackSeq, c.moveGen
	env.server.startAttackLocked(c, target)
	if c.huntState.attackSeq != seq || c.moveGen != moveGen {
		t.Fatalf("repeated mob attack restarted session: seq %d->%d moveGen %d->%d",
			seq, c.huntState.attackSeq, moveGen, c.moveGen)
	}
	if assaultPolicyAttackTargetAllowed(c, target.id+1,
		c.policyAttackRetargetAt+assaultAttackRetargetInterval/2) {
		t.Fatal("policy was allowed to thrash to a different live target during the hold")
	}
	if !assaultPolicyAttackTargetAllowed(c, target.id+1,
		c.policyAttackRetargetAt+assaultAttackRetargetInterval+0.1) {
		t.Fatal("policy could not retarget after the target hold")
	}

	victim := env.heroes[AssaultHeroCount/2]
	env.server.startPvpAttackLocked(c, victim)
	seq, moveGen = c.huntState.attackSeq, c.moveGen
	env.server.startPvpAttackLocked(c, victim)
	if c.huntState.attackSeq != seq || c.moveGen != moveGen {
		t.Fatalf("repeated PvP attack restarted session: seq %d->%d moveGen %d->%d",
			seq, c.huntState.attackSeq, moveGen, c.moveGen)
	}
}

func TestAssaultNavigationLocalGrid(t *testing.T) {
	tests := []struct {
		index uint8
		x, y  float32
	}{
		{0, -12, -12}, {4, 0, -12}, {40, 0, 0}, {76, 0, 12}, {80, 12, 12},
	}
	for _, test := range tests {
		x, y := assaultLocalOffset(test.index)
		if x != test.x || y != test.y {
			t.Fatalf("grid[%d]=(%g,%g), want (%g,%g)",
				test.index, x, y, test.x, test.y)
		}
	}
}

func TestAssaultNavigationAnchorsAreReachableAndRepeatedOrderReusesRoute(t *testing.T) {
	env := NewAssaultEnv()
	env.EnableWrongLaneCurriculum(true)
	env.ConfigureNavigationActions(true)
	defer env.Close()
	reset := externalAssaultReset(410043, 20)
	reset.Controllers[0] = AssaultControllerAI40
	reset.Controllers[5] = AssaultControllerAI40
	if _, err := env.Reset(reset); err != nil {
		t.Fatal(err)
	}
	env.inst.mu.Lock()
	defer env.inst.mu.Unlock()
	for _, slot := range []int{0, 5} {
		c := env.heroes[slot]
		sx, sy := c.posAtLocked(float32(env.server.battleTime()))
		for anchor := 1; anchor < AssaultNavigationAnchors; anchor++ {
			x, y, ok := env.assaultNavigationAnchorLocked(c, anchor)
			if !ok {
				t.Fatalf("slot %d anchor %d did not resolve", slot, anchor)
			}
			if route := c.nav.Path(float64(sx), float64(sy), float64(x), float64(y)); len(route) == 0 {
				t.Fatalf("slot %d anchor %d=(%.1f,%.1f) is unreachable", slot, anchor, x, y)
			}
		}
	}
	action := HeroActionV1{Kind: AssaultActionMove, Distance: 6}
	if !env.applyActionLocked(0, action) {
		t.Fatal("first global anchor action was rejected")
	}
	c := env.heroes[0]
	moveGen, destX, destY := c.moveGen, c.destX, c.destY
	if !env.applyActionLocked(0, action) {
		t.Fatal("repeated global anchor action was rejected")
	}
	if c.moveGen != moveGen || c.destX != destX || c.destY != destY {
		t.Fatalf("repeated anchor rebuilt route: gen %d->%d dest (%.1f,%.1f)->(%.1f,%.1f)",
			moveGen, c.moveGen, destX, destY, c.destX, c.destY)
	}
}

func TestAssaultObjectivePotentialUsesEnemyDamageOnly(t *testing.T) {
	env := NewAssaultEnv()
	defer env.Close()
	if _, err := env.Reset(externalAssaultReset(23, 20)); err != nil {
		t.Fatal(err)
	}
	env.inst.mu.Lock()
	defer env.inst.mu.Unlock()
	var enemyGun *mobState
	for _, mob := range env.inst.mobs {
		if mob.structure && mob.teamVal() == dotaTeamElf && mob.dotaRole == gamedata.DotaGun {
			enemyGun = mob
			break
		}
	}
	if enemyGun == nil {
		t.Fatal("missing enemy gun")
	}
	enemyGun.hp = enemyGun.maxHealth() / 2
	if got := env.assaultObjectivePotentialLocked(dotaTeamHuman); math.Abs(got-0.75) > 1e-6 {
		t.Fatalf("human objective potential=%g, want 0.75", got)
	}
	if got := env.assaultObjectivePotentialLocked(dotaTeamElf); got != 0 {
		t.Fatalf("elf potential includes own structure damage: %g", got)
	}
	enemyGun.dead = true
	delete(env.inst.mobs, enemyGun.id)
	if got := env.assaultObjectivePotentialLocked(dotaTeamHuman); math.Abs(got-2.25) > 1e-6 {
		t.Fatalf("removed enemy gun potential=%g, want persistent 2.25", got)
	}
}

func TestAssaultEnvResetProducesTenMaskedObservations(t *testing.T) {
	env := NewAssaultEnv()
	defer env.Close()
	result, err := env.Reset(externalAssaultReset(7, 20))
	if err != nil {
		t.Fatal(err)
	}
	if result.SchemaHash != AssaultSchemaHashV1 || result.RewardHash != AssaultRewardHashV2 || result.Done || result.Step != 0 {
		t.Fatalf("reset result = hash:%x done:%v step:%d", result.SchemaHash[:4], result.Done, result.Step)
	}
	for i, obs := range result.Observations {
		if obs.Hero[0] == 0 || obs.ActionMask.Kinds[AssaultActionWait] != 1 {
			t.Fatalf("hero %d observation not initialized", i)
		}
		if i < 5 && obs.Hero[1] != float32(dotaTeamHuman) {
			t.Fatalf("hero %d team=%v, want human", i, obs.Hero[1])
		}
		if i >= 5 && obs.Hero[1] != float32(dotaTeamElf) {
			t.Fatalf("hero %d team=%v, want elf", i, obs.Hero[1])
		}
	}
}

func TestAssaultEnvExposesAuthoredAbilitySemantics(t *testing.T) {
	env := NewAssaultEnv()
	defer env.Close()
	result, err := env.Reset(externalAssaultReset(71, 20))
	if err != nil {
		t.Fatal(err)
	}
	for heroIndex, obs := range result.Observations {
		heroID := obs.Hero[0]
		for slot, ability := range obs.Abilities {
			if ability[0] != float32(slot+1)/4 {
				t.Fatalf("hero %d ability %d slot feature=%g", heroIndex, slot, ability[0])
			}
			if ability[1] != heroID {
				t.Fatalf("hero %d ability %d hero feature=%g, want %g", heroIndex, slot, ability[1], heroID)
			}
			if ability[3] <= 0 || ability[10]+ability[11]+ability[12] != 1 {
				t.Fatalf("hero %d ability %d has no authored rank/type: %v", heroIndex, slot, ability)
			}
		}
	}
}

func TestAssaultEnvManualStepsAreDeterministic(t *testing.T) {
	run := func() []StepResultV1 {
		env := NewAssaultEnv()
		defer env.Close()
		if _, err := env.Reset(externalAssaultReset(12345, 100)); err != nil {
			t.Fatal(err)
		}
		var actions [AssaultHeroCount]HeroActionV1
		out := make([]StepResultV1, 0, 45)
		for i := 0; i < 45; i++ { // crosses the first creep-wave boundary at 8s
			result, err := env.Step(actions)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, result)
		}
		return out
	}
	a, b := run(), run()
	if !reflect.DeepEqual(a, b) {
		for i := range a {
			if !reflect.DeepEqual(a[i], b[i]) {
				t.Fatalf("same seed/actions diverged at step %d", i+1)
			}
		}
		t.Fatal("same seed/actions diverged")
	}
}

func TestAssaultEnvAI20MatchIsDeterministic(t *testing.T) {
	run := func() []StepResultV1 {
		env := NewAssaultEnv()
		defer env.Close()
		cfg := externalAssaultReset(8128, 350)
		for i := range cfg.Controllers {
			cfg.Controllers[i] = AssaultControllerAI20
		}
		result, err := env.Reset(cfg)
		if err != nil {
			t.Fatal(err)
		}
		results := make([]StepResultV1, 0, 350)
		for i := 0; i < 350 && !result.Done; i++ {
			result, err = env.Step([AssaultHeroCount]HeroActionV1{})
			if err != nil {
				t.Fatal(err)
			}
			results = append(results, result)
		}
		return results
	}
	a, b := run(), run()
	for i := range a {
		if !reflect.DeepEqual(a[i], b[i]) {
			for hero := 0; hero < AssaultHeroCount; hero++ {
				if !reflect.DeepEqual(a[i].Observations[hero], b[i].Observations[hero]) || a[i].Reward[hero] != b[i].Reward[hero] {
					for entity := 0; entity < AssaultMaxEntities; entity++ {
						if !reflect.DeepEqual(a[i].Observations[hero].Entities[entity], b[i].Observations[hero].Entities[entity]) ||
							a[i].Observations[hero].EntityMask[entity] != b[i].Observations[hero].EntityMask[entity] {
							t.Fatalf("AI-20 trajectories diverged at step %d hero %d entity %d: %v/%v",
								i+1, hero, entity, a[i].Observations[hero].Entities[entity], b[i].Observations[hero].Entities[entity])
						}
					}
					t.Fatalf("AI-20 trajectories diverged at step %d hero %d: reward=%g/%g hero=%v/%v",
						i+1, hero, a[i].Reward[hero], b[i].Reward[hero], a[i].Observations[hero].Hero, b[i].Observations[hero].Hero)
				}
			}
			t.Fatalf("AI-20 trajectories with the same seed diverged at step %d header", i+1)
		}
	}
}

func TestAssaultEnvAI30ControllerSelectsScriptedTeacher(t *testing.T) {
	env := NewAssaultEnv()
	defer env.Close()
	cfg := externalAssaultReset(41, 20)
	for i := range cfg.Controllers {
		cfg.Controllers[i] = AssaultControllerAI30
	}
	if _, err := env.Reset(cfg); err != nil {
		t.Fatal(err)
	}
	env.inst.mu.Lock()
	defer env.inst.mu.Unlock()
	if env.inst.dota.botAIVersionByTeam[dotaTeamHuman] != 30 ||
		env.inst.dota.botAIVersionByTeam[dotaTeamElf] != 30 {
		t.Fatalf("headless team profiles=%v, want AI-30 mirror", env.inst.dota.botAIVersionByTeam)
	}
	for i, brain := range env.brains {
		if botAIVersionForBrain(brain) != 30 {
			t.Fatalf("headless brain %d version=%d, want 30", i, botAIVersionForBrain(brain))
		}
	}
}

func newAssaultTeacherEnv(t *testing.T, controllers [AssaultHeroCount]AssaultControllerV1) (*AssaultEnv, StepResultV1) {
	t.Helper()
	env := NewAssaultEnv()
	env.ConfigureTeacherActions(true)
	result, err := env.Reset(AssaultResetV1{Seed: 410044, MaxSteps: 20, Controllers: controllers})
	if err != nil {
		env.Close()
		t.Fatal(err)
	}
	t.Cleanup(env.Close)
	return env, result
}

func assaultTeacherSetMoveLocked(env *AssaultEnv, slot int, dx float32) {
	c := env.heroes[slot]
	c.hasDest = true
	c.destX, c.destY = c.x+dx, c.y
}

func TestAssaultTeacherActionHoldCancelWaitSequence(t *testing.T) {
	var controllers [AssaultHeroCount]AssaultControllerV1
	for i := range controllers {
		controllers[i] = AssaultControllerAI30
	}
	env, initial := newAssaultTeacherEnv(t, controllers)
	if initial.TeacherStatus[0] != AssaultTeacherStatusWait {
		t.Fatalf("initial teacher status=%d, want WAIT", initial.TeacherStatus[0])
	}
	env.inst.mu.Lock()
	defer env.inst.mu.Unlock()
	assaultTeacherSetMoveLocked(env, 0, 12)
	first := env.resultLocked(nil, env.clock.Now())
	if first.TeacherStatus[0] != AssaultTeacherStatusAction || first.TeacherIntent[0].Kind != AssaultActionMove {
		t.Fatalf("new teacher order status/action=%d/%+v, want ACTION/move", first.TeacherStatus[0], first.TeacherIntent[0])
	}
	if first.TeacherIntent[0].Target != 0 {
		t.Fatalf("move teacher action leaked target slot %d", first.TeacherIntent[0].Target)
	}
	hold := env.resultLocked(nil, env.clock.Now())
	if hold.TeacherStatus[0] != AssaultTeacherStatusHold || hold.TeacherIntent[0] != (HeroActionV1{}) {
		t.Fatalf("continued teacher order status/action=%d/%+v, want HOLD/zero", hold.TeacherStatus[0], hold.TeacherIntent[0])
	}
	env.heroes[0].hasDest = false
	cancel := env.resultLocked(nil, env.clock.Now())
	if cancel.TeacherStatus[0] != AssaultTeacherStatusCancel || cancel.TeacherIntent[0] != (HeroActionV1{}) {
		t.Fatalf("cancelled teacher order status/action=%d/%+v, want CANCEL/zero", cancel.TeacherStatus[0], cancel.TeacherIntent[0])
	}
	wait := env.resultLocked(nil, env.clock.Now())
	if wait.TeacherStatus[0] != AssaultTeacherStatusWait {
		t.Fatalf("idle teacher status=%d, want WAIT", wait.TeacherStatus[0])
	}
}

func TestAssaultTeacherFrameBindsPreDecisionObservationAndTransition(t *testing.T) {
	var controllers [AssaultHeroCount]AssaultControllerV1
	for i := range controllers {
		controllers[i] = AssaultControllerAI30
	}
	env, _ := newAssaultTeacherEnv(t, controllers)
	env.inst.mu.Lock()
	defer env.inst.mu.Unlock()

	const marker = float32(0.4242)
	observation := env.observationLocked(0, env.clock.Now())
	observation.Hero[AssaultHeroFeatureSize-1] = marker
	action := HeroActionV1{Kind: AssaultActionMove, Direction: 40}
	frame := assaultTeacherFrame{}
	frame.present[0] = true
	frame.observations[0] = observation
	frame.projections[0] = assaultTeacherProjection{active: true, representable: true, action: action}

	// Mutate the live state after the decision snapshot. The wire row must use
	// the frame rather than silently rebuilding a post-decision observation or
	// advancing the teacher state a second time to HOLD.
	env.heroes[0].x += 100
	result := env.resultLockedWithTeacher(nil, env.clock.Now(), &frame, false)
	if got := result.Observations[0].Hero[AssaultHeroFeatureSize-1]; got != marker {
		t.Fatalf("teacher observation marker=%g, want pre-decision %g", got, marker)
	}
	if result.TeacherStatus[0] != AssaultTeacherStatusAction || result.TeacherIntent[0] != action {
		t.Fatalf("teacher frame status/action=%d/%+v, want ACTION/%+v", result.TeacherStatus[0], result.TeacherIntent[0], action)
	}
	if env.teacherState[0] != (assaultTeacherState{active: true, action: action}) {
		t.Fatalf("result encoding did not finalize teacher state exactly once: %+v", env.teacherState[0])
	}
}

func TestAssaultTeacherFrameTerminalRetiresEarlierTickDecision(t *testing.T) {
	var controllers [AssaultHeroCount]AssaultControllerV1
	for i := range controllers {
		controllers[i] = AssaultControllerAI30
	}
	env, _ := newAssaultTeacherEnv(t, controllers)
	env.inst.mu.Lock()
	defer env.inst.mu.Unlock()

	action := HeroActionV1{Kind: AssaultActionMove, Direction: 40}
	env.teacherState[0] = assaultTeacherState{active: true, action: action}
	frame := assaultTeacherFrame{}
	frame.present[0] = true
	frame.observations[0] = env.observationLocked(0, env.clock.Now())
	frame.projections[0] = assaultTeacherProjection{active: true, representable: true, action: action}
	env.step = env.maxSteps

	result := env.resultLockedWithTeacher(nil, env.clock.Now(), &frame, false)
	if !result.Done || result.TeacherStatus[0] != AssaultTeacherStatusCancel {
		t.Fatalf("terminal teacher frame done/status=%v/%d, want true/CANCEL", result.Done, result.TeacherStatus[0])
	}
	if env.teacherState[0] != (assaultTeacherState{}) {
		t.Fatalf("terminal teacher frame retained lineage %+v", env.teacherState[0])
	}
}

func TestAssaultTeacherReplacementPrefersActionAndRetiresOldLineage(t *testing.T) {
	var controllers [AssaultHeroCount]AssaultControllerV1
	for i := range controllers {
		controllers[i] = AssaultControllerAI30
	}
	env, _ := newAssaultTeacherEnv(t, controllers)
	env.inst.mu.Lock()
	defer env.inst.mu.Unlock()
	assaultTeacherSetMoveLocked(env, 0, 12)
	first := env.resultLocked(nil, env.clock.Now())
	old := first.TeacherIntent[0]
	assaultTeacherSetMoveLocked(env, 0, -12)
	replacement := env.resultLocked(nil, env.clock.Now())
	if replacement.TeacherStatus[0] != AssaultTeacherStatusAction {
		t.Fatalf("replacement status=%d, want ACTION", replacement.TeacherStatus[0])
	}
	if replacement.TeacherIntent[0] == old {
		t.Fatalf("replacement retained old canonical action %+v", old)
	}
	if replacement.TeacherActions[0] != replacement.TeacherIntent[0] || replacement.TeacherValid[0] != 1 {
		t.Fatalf("replacement legacy teacher mirror=%+v/%d, want action/1", replacement.TeacherActions[0], replacement.TeacherValid[0])
	}
	hold := env.resultLocked(nil, env.clock.Now())
	if hold.TeacherStatus[0] != AssaultTeacherStatusHold {
		t.Fatalf("replacement continuation status=%d, want HOLD", hold.TeacherStatus[0])
	}
}

func TestAssaultTeacherProjectedActionChangeStartsNewLineage(t *testing.T) {
	var controllers [AssaultHeroCount]AssaultControllerV1
	for i := range controllers {
		controllers[i] = AssaultControllerAI30
	}
	env, _ := newAssaultTeacherEnv(t, controllers)
	env.inst.mu.Lock()
	defer env.inst.mu.Unlock()
	assaultTeacherSetMoveLocked(env, 0, 12)
	first := env.resultLocked(nil, env.clock.Now())
	if first.TeacherStatus[0] != AssaultTeacherStatusAction {
		t.Fatalf("first status=%d, want ACTION", first.TeacherStatus[0])
	}
	c := env.heroes[0]
	now := env.clock.Now()
	c.x, c.vx, c.snapT = c.destX-3, 0, float32(now)
	changed := env.resultLocked(nil, now)
	if changed.TeacherStatus[0] != AssaultTeacherStatusAction {
		t.Fatalf("changed actor-visible move status=%d, want ACTION", changed.TeacherStatus[0])
	}
	if changed.TeacherIntent[0] == first.TeacherIntent[0] {
		t.Fatalf("projected move did not change: %+v", changed.TeacherIntent[0])
	}
}

func TestAssaultTeacherCancelSurvivesUnavailableGap(t *testing.T) {
	var controllers [AssaultHeroCount]AssaultControllerV1
	for i := range controllers {
		controllers[i] = AssaultControllerAI30
	}
	env, _ := newAssaultTeacherEnv(t, controllers)
	env.inst.mu.Lock()
	defer env.inst.mu.Unlock()
	assaultTeacherSetMoveLocked(env, 0, 12)
	first := env.resultLocked(nil, env.clock.Now())
	if first.TeacherStatus[0] != AssaultTeacherStatusAction {
		t.Fatalf("first status=%d, want ACTION", first.TeacherStatus[0])
	}
	masked := first.Observations[0]
	masked.ActionMask.Kinds[AssaultActionMove] = 0
	if _, status := env.assaultTeacherTransitionLocked(0, &masked, false, env.clock.Now()); status != AssaultTeacherStatusUnavailable {
		t.Fatalf("masked continuation status=%d, want UNAVAILABLE", status)
	}
	env.heroes[0].hasDest = false
	if _, status := env.assaultTeacherTransitionLocked(0, &first.Observations[0], false, env.clock.Now()); status != AssaultTeacherStatusCancel {
		t.Fatalf("disappearance after unavailable status=%d, want CANCEL", status)
	}
}

func TestAssaultV13SkillNavigationRejectsIgnoredAnchor(t *testing.T) {
	if !assaultNavigationActionFieldsValid(HeroActionV1{
		Kind: AssaultActionMove, Distance: AssaultNavigationAnchors - 1,
	}) {
		t.Fatal("movement anchor was rejected")
	}
	if !assaultNavigationActionFieldsValid(HeroActionV1{
		Kind: AssaultActionSkill1, Direction: AssaultNavigationOffsets - 1,
	}) {
		t.Fatal("skill local offset was rejected")
	}
	if assaultNavigationActionFieldsValid(HeroActionV1{
		Kind: AssaultActionSkill1, Distance: 1,
	}) {
		t.Fatal("skill anchor was accepted even though execution ignores it")
	}
}

func TestAssaultTeacherUnavailableDoesNotLeakFoggedTarget(t *testing.T) {
	var controllers [AssaultHeroCount]AssaultControllerV1
	for i := range controllers {
		controllers[i] = AssaultControllerAI30
	}
	env, _ := newAssaultTeacherEnv(t, controllers)
	env.inst.mu.Lock()
	defer env.inst.mu.Unlock()
	// The opposing hero is outside the initial team vision radius. The
	// authoritative order exists, but no actor-visible target slot can represent it.
	env.heroes[0].huntState.attackTarget = env.heroes[5].objID
	result := env.resultLocked(nil, env.clock.Now())
	if result.TeacherStatus[0] != AssaultTeacherStatusUnavailable {
		t.Fatalf("fogged teacher status=%d, want UNAVAILABLE", result.TeacherStatus[0])
	}
	if result.TeacherIntent[0] != (HeroActionV1{}) || result.TeacherActions[0] != (HeroActionV1{}) {
		t.Fatalf("fogged teacher payload leaked target: intent=%+v legacy=%+v", result.TeacherIntent[0], result.TeacherActions[0])
	}
	if result.TeacherStatus[1] != AssaultTeacherStatusWait {
		t.Fatalf("unrelated teacher slot status=%d, want WAIT", result.TeacherStatus[1])
	}
	env.heroes[0].huntState.attackTarget = 0
	if next := env.resultLocked(nil, env.clock.Now()); next.TeacherStatus[0] != AssaultTeacherStatusWait {
		t.Fatalf("unrepresentable one-shot order left status=%d, want WAIT", next.TeacherStatus[0])
	}
}

func TestAssaultTeacherStateResetsOnDeathRespawnAndResetReplay(t *testing.T) {
	run := func(t *testing.T) []uint8 {
		var controllers [AssaultHeroCount]AssaultControllerV1
		for i := range controllers {
			controllers[i] = AssaultControllerAI30
		}
		env, _ := newAssaultTeacherEnv(t, controllers)
		env.inst.mu.Lock()
		defer env.inst.mu.Unlock()
		assaultTeacherSetMoveLocked(env, 0, 12)
		first := env.resultLocked(nil, env.clock.Now())
		now := env.clock.Now()
		env.heroes[0].huntState.deadUntil = now + 1
		env.heroes[0].hasDest = false // mirrors playerDieLocked's movement cancellation
		death := env.resultLocked(nil, now)
		dead := env.resultLocked(nil, now)
		env.heroes[0].huntState.deadUntil = 0
		respawn := env.resultLocked(nil, now)
		return []uint8{first.TeacherStatus[0], death.TeacherStatus[0], dead.TeacherStatus[0], respawn.TeacherStatus[0]}
	}
	a := run(t)
	b := run(t)
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("same seed teacher reset/replay diverged: %v/%v", a, b)
	}
	want := []uint8{AssaultTeacherStatusAction, AssaultTeacherStatusCancel, AssaultTeacherStatusWait, AssaultTeacherStatusWait}
	if !reflect.DeepEqual(a, want) {
		t.Fatalf("death/respawn teacher statuses=%v, want %v", a, want)
	}

	var controllers [AssaultHeroCount]AssaultControllerV1
	for i := range controllers {
		controllers[i] = AssaultControllerAI30
	}
	env, _ := newAssaultTeacherEnv(t, controllers)
	env.inst.mu.Lock()
	assaultTeacherSetMoveLocked(env, 0, 12)
	_ = env.resultLocked(nil, env.clock.Now())
	env.inst.mu.Unlock()
	if _, err := env.Reset(AssaultResetV1{Seed: 410044, MaxSteps: 20, Controllers: controllers}); err != nil {
		t.Fatal(err)
	}
	if env.teacherState != [AssaultHeroCount]assaultTeacherState{} {
		t.Fatal("Reset retained teacher lineage")
	}
}

func TestAssaultTeacherRealDeathCleanupDoesNotRestoreStalePvpOrder(t *testing.T) {
	var controllers [AssaultHeroCount]AssaultControllerV1
	for i := range controllers {
		controllers[i] = AssaultControllerAI30
	}
	env, _ := newAssaultTeacherEnv(t, controllers)
	env.inst.mu.Lock()
	defer env.inst.mu.Unlock()
	c := env.heroes[0]
	c.huntState.pvpTarget = env.heroes[AssaultHeroCount/2].objID
	now := env.clock.Now()
	env.server.playerDieLocked(c, 0, now)
	if c.huntState.pvpTarget != 0 {
		t.Fatalf("real death retained stale pvpTarget=%d", c.huntState.pvpTarget)
	}
	c.huntState.deadUntil = 0
	result := env.resultLocked(nil, now)
	if result.TeacherStatus[0] != AssaultTeacherStatusWait {
		t.Fatalf("respawn after real death status=%d, want WAIT", result.TeacherStatus[0])
	}
}

func TestAssaultTeacherUnavailableIsOnlyForNonAI30TeacherSlots(t *testing.T) {
	var controllers [AssaultHeroCount]AssaultControllerV1
	controllers[0] = AssaultControllerAI30
	controllers[1] = AssaultControllerAI20
	controllers[2] = AssaultControllerAI40
	controllers[3] = AssaultControllerExternal
	_, result := newAssaultTeacherEnv(t, controllers)
	if result.TeacherStatus[0] != AssaultTeacherStatusWait {
		t.Fatalf("AI-30 teacher status=%d, want WAIT", result.TeacherStatus[0])
	}
	for _, slot := range []int{1, 2, 3} {
		if result.TeacherStatus[slot] != AssaultTeacherStatusUnavailable {
			t.Fatalf("non-AI30 slot %d teacher status=%d, want UNAVAILABLE", slot, result.TeacherStatus[slot])
		}
		if result.TeacherIntent[slot] != (HeroActionV1{}) {
			t.Fatalf("non-AI30 slot %d teacher payload=%+v, want zero", slot, result.TeacherIntent[slot])
		}
	}
}

func TestAssaultDAggerInterventionUsesAI30ForOneTickAndRestoresController(t *testing.T) {
	var controllers [AssaultHeroCount]AssaultControllerV1
	for i := range controllers {
		controllers[i] = AssaultControllerAI40
	}
	env := NewAssaultEnv()
	defer env.Close()
	env.ConfigureTeacherActions(true)
	if _, err := env.Reset(AssaultResetV1{Seed: 150042, MaxSteps: 20, Controllers: controllers}); err != nil {
		t.Fatal(err)
	}
	var actions [AssaultHeroCount]HeroActionV1
	var controls [AssaultHeroCount]AssaultControlV1
	var interventions [AssaultHeroCount]uint8
	interventions[0] = 1
	result, err := env.StepIntervened(actions, controls, interventions)
	if err != nil {
		t.Fatal(err)
	}
	if result.TeacherStatus[0] == AssaultTeacherStatusUnavailable {
		t.Fatal("intervened AI-40 slot did not receive an AI-30 teacher transition")
	}
	if env.controllers[0] != AssaultControllerAI40 || botAIVersionForBrain(env.brains[0]) != 40 {
		t.Fatalf("intervention leaked controller/profile: controller=%d AI=%d",
			env.controllers[0], botAIVersionForBrain(env.brains[0]))
	}
	if env.inst.dota.botAIVersionByTeam[dotaTeamHuman] != 40 {
		t.Fatal("intervention leaked the temporary team orchestrator profile")
	}
	interventions[1] = 2
	if _, err := env.StepIntervened(actions, controls, interventions); err == nil {
		t.Fatal("invalid intervention byte was accepted")
	}
}

func TestAssaultDAggerPreservesTeacherLineageBetweenPeriodicQueries(t *testing.T) {
	var controllers [AssaultHeroCount]AssaultControllerV1
	for i := range controllers {
		controllers[i] = AssaultControllerAI40
	}
	env := NewAssaultEnv()
	defer env.Close()
	env.ConfigureTeacherActions(true)
	if _, err := env.Reset(AssaultResetV1{Seed: 150043, MaxSteps: 20, Controllers: controllers}); err != nil {
		t.Fatal(err)
	}
	want := assaultTeacherState{
		active: true,
		action: HeroActionV1{Kind: AssaultActionMove, Direction: 41},
	}
	env.inst.mu.Lock()
	env.teacherState[0] = want
	env.inst.mu.Unlock()

	var actions [AssaultHeroCount]HeroActionV1
	var controls [AssaultHeroCount]AssaultControlV1
	var interventions [AssaultHeroCount]uint8
	result, err := env.StepIntervened(actions, controls, interventions)
	if err != nil {
		t.Fatal(err)
	}
	if result.TeacherStatus[0] != AssaultTeacherStatusUnavailable {
		t.Fatalf("non-query status=%d, want UNAVAILABLE", result.TeacherStatus[0])
	}
	if env.teacherState[0] != want {
		t.Fatalf("DAgger non-query reset teacher lineage: got %+v, want %+v", env.teacherState[0], want)
	}

	if _, err := env.StepControlled(actions, controls); err != nil {
		t.Fatal(err)
	}
	if env.teacherState[0] != (assaultTeacherState{}) {
		t.Fatalf("ordinary controlled step retained DAgger lineage: %+v", env.teacherState[0])
	}
}

func TestAssaultEnvSkipsExternalObservationsForScriptedSlots(t *testing.T) {
	env := NewAssaultEnv()
	defer env.Close()
	cfg := externalAssaultReset(43, 20)
	for i := range cfg.Controllers {
		cfg.Controllers[i] = AssaultControllerAI30
	}
	cfg.Controllers[0] = AssaultControllerAI40
	result, err := env.Reset(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if result.Observations[0] == (AssaultObservationV1{}) {
		t.Fatal("external AI-40 slot has no protocol observation")
	}
	for i := 1; i < AssaultHeroCount; i++ {
		if result.Observations[i] != (AssaultObservationV1{}) {
			t.Fatalf("scripted slot %d unexpectedly built an external observation", i)
		}
	}
}

func TestAssaultTangrenChannelMaskAllowsMovementAndBlocksAbilities(t *testing.T) {
	env := NewAssaultEnv()
	defer env.Close()
	cfg := externalAssaultReset(150043, 20)
	cfg.Roster[0] = 18 // Tangren
	if _, err := env.Reset(cfg); err != nil {
		t.Fatal(err)
	}

	env.inst.mu.Lock()
	defer env.inst.mu.Unlock()
	c := env.heroes[0]
	hs := c.huntState
	hs.skillLevel[0] = 1
	hs.skillLevel[2] = 1
	if !env.server.startSkillOrderLocked(c, 1, -1, 0, 0, true) {
		t.Fatal("Tangren's first skill was not accepted")
	}
	result := env.resultLocked(nil, env.clock.Now())
	mask := result.Observations[0].ActionMask
	if mask.Kinds[AssaultActionMove] != 1 {
		t.Fatal("Tangren's movement channel masked movement")
	}
	for kind := AssaultActionSkill1; kind <= AssaultActionSkill4; kind++ {
		if mask.Kinds[kind] != 0 {
			t.Fatalf("Tangren's movement channel exposed ability kind %d", kind)
		}
	}
}

func TestAssaultEnvAI40MirrorUsesOnlyExternalSharedPolicy(t *testing.T) {
	env := NewAssaultEnv()
	defer env.Close()
	cfg := externalAssaultReset(42, 20)
	for i := range cfg.Controllers {
		cfg.Controllers[i] = AssaultControllerAI40
	}
	if _, err := env.Reset(cfg); err != nil {
		t.Fatal(err)
	}
	env.inst.mu.Lock()
	defer env.inst.mu.Unlock()
	if env.inst.dota.botAIVersionByTeam[dotaTeamHuman] != 40 ||
		env.inst.dota.botAIVersionByTeam[dotaTeamElf] != 40 {
		t.Fatalf("headless team profiles=%v, want AI-40 mirror", env.inst.dota.botAIVersionByTeam)
	}
	if len(env.inst.bots) != 0 {
		t.Fatalf("AI-40 self-play retained %d live sidecar/script brains", len(env.inst.bots))
	}
	for i, brain := range env.brains {
		if brain == nil || botAIVersionForBrain(brain) != 40 || env.controllers[i] != AssaultControllerAI40 {
			t.Fatalf("AI-40 slot %d brain/controller mismatch", i)
		}
	}
}

func TestAssaultEnvParticipantsSuppressClientWireEncoding(t *testing.T) {
	env := NewAssaultEnv()
	t.Cleanup(env.Close)
	if _, err := env.Reset(AssaultResetV1{Seed: 41, MaxSteps: 2}); err != nil {
		t.Fatal(err)
	}
	for i, hero := range env.heroes {
		if hero == nil || !hero.headless {
			t.Fatalf("hero %d is not marked headless", i)
		}
		if _, ok := hero.Conn.(headlessBotConn); !ok {
			t.Fatalf("hero %d conn = %T, want headlessBotConn", i, hero.Conn)
		}
		if hero.r != nil {
			t.Fatalf("hero %d unexpectedly owns a Battle packet reader", i)
		}
		if err := hero.send(battleproto.Packet{}); err != nil {
			t.Fatalf("hero %d no-op send: %v", i, err)
		}
	}
}

func BenchmarkAssaultEnvAI40Step(b *testing.B) {
	env := NewAssaultEnv()
	defer env.Close()
	reset := AssaultResetV1{Seed: 73, MaxSteps: uint32(b.N + 1)}
	for i := range reset.Controllers {
		reset.Controllers[i] = AssaultControllerAI40
	}
	if _, err := env.Reset(reset); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := env.Step([AssaultHeroCount]HeroActionV1{}); err != nil {
			b.Fatal(err)
		}
	}
}

func TestAssaultEnvAI30MatchIsDeterministic(t *testing.T) {
	run := func() []StepResultV1 {
		env := NewAssaultEnv()
		defer env.Close()
		cfg := externalAssaultReset(8140, 400)
		for i := range cfg.Controllers {
			cfg.Controllers[i] = AssaultControllerAI30
		}
		result, err := env.Reset(cfg)
		if err != nil {
			t.Fatal(err)
		}
		results := make([]StepResultV1, 0, 400)
		for i := 0; i < 400 && !result.Done; i++ {
			result, err = env.Step([AssaultHeroCount]HeroActionV1{})
			if err != nil {
				t.Fatal(err)
			}
			results = append(results, result)
		}
		return results
	}
	a, b := run(), run()
	if !reflect.DeepEqual(a, b) {
		t.Fatal("AI-30 trajectories with the same seed diverged")
	}
}

func TestAssaultEnvRejectsUnknownController(t *testing.T) {
	env := NewAssaultEnv()
	defer env.Close()
	cfg := externalAssaultReset(43, 20)
	cfg.Controllers[3] = 99
	if _, err := env.Reset(cfg); err == nil {
		t.Fatal("unsupported controller was accepted")
	}
}

func TestAssaultEnvMixedAI20AI30KeepsPerHeroProfiles(t *testing.T) {
	env := NewAssaultEnv()
	defer env.Close()
	cfg := externalAssaultReset(47, 20)
	cfg.Controllers[0] = AssaultControllerAI30
	cfg.Controllers[1] = AssaultControllerAI20
	if _, err := env.Reset(cfg); err != nil {
		t.Fatal(err)
	}
	env.inst.mu.Lock()
	defer env.inst.mu.Unlock()
	if botAIVersionForBrain(env.brains[0]) != 30 || botAIVersionForBrain(env.brains[1]) != 20 {
		t.Fatalf("mixed profiles = %d/%d, want 30/20",
			botAIVersionForBrain(env.brains[0]), botAIVersionForBrain(env.brains[1]))
	}
	if env.inst.dota.botAIVersionByTeam[dotaTeamHuman] != 30 {
		t.Fatal("mixed scripted team did not select AI-30 orchestrator")
	}
}

func TestAssaultEnvMaxStepsAndInvalidAction(t *testing.T) {
	env := NewAssaultEnv()
	defer env.Close()
	if _, err := env.Reset(externalAssaultReset(1, 1)); err != nil {
		t.Fatal(err)
	}
	var actions [AssaultHeroCount]HeroActionV1
	actions[0] = HeroActionV1{Kind: AssaultActionTeleport}
	result, err := env.Step(actions)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Done || result.Step != 1 || result.Invalid[0] != 1 {
		t.Fatalf("step result done=%v step=%d invalid[0]=%d", result.Done, result.Step, result.Invalid[0])
	}
}

func TestManualBattleClockRunsEqualDeadlinesInInsertionOrder(t *testing.T) {
	c := newManualBattleClock()
	var got []int
	c.After(200000000, func() { got = append(got, 1) })
	c.After(200000000, func() { got = append(got, 2) })
	c.Advance(200000000)
	if !reflect.DeepEqual(got, []int{1, 2}) || c.Now() != 0.2 {
		t.Fatalf("events=%v now=%v", got, c.Now())
	}
}
