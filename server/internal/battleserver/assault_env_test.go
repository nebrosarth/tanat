package battleserver

import (
	"math"
	"reflect"
	"testing"

	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
)

func externalAssaultReset(seed int64, maxSteps uint32) AssaultResetV1 {
	return AssaultResetV1{Seed: seed, MaxSteps: maxSteps}
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
		cfg := externalAssaultReset(8140, 120)
		for i := range cfg.Controllers {
			cfg.Controllers[i] = AssaultControllerAI30
		}
		result, err := env.Reset(cfg)
		if err != nil {
			t.Fatal(err)
		}
		results := make([]StepResultV1, 0, 120)
		for i := 0; i < 120 && !result.Done; i++ {
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
