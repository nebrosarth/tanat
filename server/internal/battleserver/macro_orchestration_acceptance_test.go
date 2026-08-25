package battleserver

import (
	"reflect"
	"testing"
	"time"

	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
)

func macroAddBot(t *testing.T, s *Server, inst *huntInstance, id, team int32, x, y float32, lane int) *botBrain {
	t.Helper()
	c := dotaPlayerConn(t, s, inst, id, team, x, y)
	b := newBotBrain(c, len(inst.bots), lane)
	b.lane = lane
	inst.bots[id] = b
	return b
}

func macroSetPosition(c *conn, x, y float32, now float64) {
	c.x, c.y, c.vx, c.vy, c.snapT = x, y, 0, 0, float32(now)
}

func macroEnemyCreep(inst *huntInstance, id int32, x, y float32) *mobState {
	idx := inst.dota.m.ElfCreepMelee
	m := &mobState{
		id: id, mobIdx: idx, mob: gamedata.Mobs()[idx],
		x: x, y: y, hp: 500, maxHP: 500, team: dotaTeamElf,
	}
	inst.mobs[id] = m
	return m
}

func macroOpenAltar(inst *huntInstance, altar *mobState) {
	for _, id := range inst.dota.altarGuards[altar.id] {
		inst.mobs[id].dead = true
	}
}

func macroBattleEndCount(pkts *[]battleproto.Packet, mu interface {
	Lock()
	Unlock()
}) int {
	time.Sleep(50 * time.Millisecond)
	mu.Lock()
	defer mu.Unlock()
	n := 0
	for _, p := range *pkts {
		if p.Cmd == battleproto.CmdBattleEnd {
			n++
		}
	}
	return n
}

func TestDotaAltarKillFinalizesSynchronouslyAndIdempotently(t *testing.T) {
	tests := []struct {
		name string
		kill func(*Server, *conn, *huntInstance, *mobState, float64)
	}{
		{
			name: "hero damage",
			kill: func(s *Server, c *conn, inst *huntInstance, altar *mobState, now float64) {
				s.hitMobLocked(c, altar, altar.hp*100+1, c.objID)
				if !inst.dota.ended || inst.dota.winner != dotaTeamHuman {
					t.Fatalf("hero altar kill returned before finalization: ended=%v winner=%d", inst.dota.ended, inst.dota.winner)
				}
			},
		},
		{
			name: "unit damage twin",
			kill: func(s *Server, c *conn, inst *huntInstance, altar *mobState, now float64) {
				attacker := &mobState{id: 61001, team: dotaTeamHuman}
				inst.mobs[attacker.id] = attacker
				s.dotaDamageLocked(c, altar, altar.hp*100+1, attacker.id, now)
				if !inst.dota.ended || inst.dota.winner != dotaTeamHuman {
					t.Fatalf("unit altar kill returned before finalization: ended=%v winner=%d", inst.dota.ended, inst.dota.winner)
				}
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s, c, inst, pkts, mu := newDotaCaptureConn(t)
			altar := altarOf(inst, dotaTeamElf)
			if altar == nil {
				t.Fatal("setup: missing enemy altar")
			}
			macroOpenAltar(inst, altar)
			altar.hp = altar.maxHealth()
			now := float64(s.battleTime())

			c.lock()
			tc.kill(s, c, inst, altar, now)
			// A second finalization attempt, including the opposite winner, must be a no-op.
			s.dotaEndLocked(c, dotaTeamElf, now+1)
			s.dotaDamageLocked(c, altar, 1000, c.objID, now+1)
			c.unlock()

			if altar.dead != true || !inst.dota.ended || inst.dota.winner != dotaTeamHuman {
				t.Fatalf("final state = dead=%v ended=%v winner=%d", altar.dead, inst.dota.ended, inst.dota.winner)
			}
			if got := macroBattleEndCount(pkts, mu); got != 1 {
				t.Fatalf("BATTLE_END count = %d, want exactly one", got)
			}
		})
	}
}

func TestDotaFinalizationFreezesActionsAndEndedTicker(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	clock := newManualBattleClock()
	s.clock = clock
	bot := macroAddBot(t, s, inst, botIDBase+1, dotaTeamHuman, c.x+2, c.y, 0)
	enemy := dotaPlayerConn(t, s, inst, botIDBase+2, dotaTeamElf, c.x+3, c.y)
	target := structOfSide(inst, gamedata.DotaGun, dotaTeamElf)
	if target == nil {
		t.Fatal("setup: missing enemy structure")
	}
	now := float64(s.battleTime())

	inst.mu.Lock()
	c.huntState.attackTarget = target.id
	c.huntState.pvpTarget = enemy.objID
	c.huntState.order = &pendingCast{slot: 1, target: target.id}
	c.huntState.channels = []channelState{{slot: 1, until: now + 10}}
	c.huntState.payloads = []payload{{at: now + 10}}
	c.huntState.actionDones = []actionDone{{at: now + 10}}
	c.huntState.castLockUntil = now + 10
	c.huntState.dashUntil = now + 10
	c.arrival = time.NewTimer(time.Hour)
	c.hasDest, c.vx, c.vy = true, 3, 4
	c.path = []gamedata.Vec2{{X: float64(c.x + 20), Y: float64(c.y)}}
	bot.pendingTeleport = &botTeleportOrder{target: target.id, targetKind: "structure", targetAnchor: 0}

	target.dtarget = c.objID
	target.vx, target.vy = 3, 4
	target.hitAt, target.hitDmg, target.hitTarget = now+5, 10, c.objID
	target.projLaunchAt, target.projTarget, target.projFlying = now+4, c.objID, true
	target.swingDoneAt, target.nextSwing = now+5, now+5
	target.skillHitAt, target.skillDmg, target.skillRadius = now+5, 10, 4
	target.pf = pathState{pts: []gamedata.Vec2{{X: 1, Y: 1}}}

	s.dotaEndLocked(c, dotaTeamHuman, now)
	inst.mu.Unlock()

	if c.huntState.attackTarget != 0 || c.huntState.pvpTarget != 0 || c.huntState.order != nil ||
		len(c.huntState.channels) != 0 || len(c.huntState.payloads) != 0 || len(c.huntState.actionDones) != 0 ||
		c.huntState.castLockUntil != 0 || c.huntState.dashUntil != 0 || c.arrival != nil || c.hasDest || c.vx != 0 || c.vy != 0 {
		t.Fatalf("avatar action state survived freeze: hs=%+v dest=%v arrival=%v velocity=(%v,%v)",
			c.huntState, c.hasDest, c.arrival != nil, c.vx, c.vy)
	}
	if bot.pendingTeleport != nil || bot.macroAssignment.Mode != botMacroRecover {
		t.Fatalf("bot freeze state = pending=%v assignment=%+v", bot.pendingTeleport != nil, bot.macroAssignment)
	}
	if target.dtarget != 0 || target.vx != 0 || target.vy != 0 || target.hitAt != 0 || target.hitDmg != 0 ||
		target.hitTarget != 0 || target.projLaunchAt != 0 || target.projTarget != 0 || target.projFlying ||
		target.swingDoneAt != 0 || target.nextSwing != 0 || target.skillHitAt != 0 || target.skillDmg != 0 ||
		target.skillRadius != 0 || len(target.pf.pts) != 0 {
		t.Fatalf("mob action state survived freeze: %+v", target)
	}

	phase, nextThink, x, y := bot.phase, bot.nextThinkAt, target.x, target.y
	driver := &manualDotaBots{server: s, inst: inst, clock: clock}
	for tick := 0; tick < 2; tick++ {
		driver.step()
	}
	inst.mu.Lock()
	if bot.phase != phase || bot.nextThinkAt != nextThink || target.x != x || target.y != y {
		inst.mu.Unlock()
		t.Fatalf("ended ticker progressed state: phase=%d/%d nextThink=%v/%v mob=(%v,%v)/(%v,%v)",
			bot.phase, phase, bot.nextThinkAt, nextThink, target.x, target.y, x, y)
	}
	inst.closed = true
	inst.mu.Unlock()
}

func TestDotaTeamPlansAreDeterministicAndTelemetryIsEdgeTriggered(t *testing.T) {
	build := func(order []int32) (botTeamPlan, map[int32]int) {
		s, c, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
		t.Cleanup(cleanup)
		now := float64(s.battleTime()) + 10
		p := inst.dota.m.Lanes[0][len(inst.dota.m.Lanes[0])/2]
		for _, id := range order {
			b := macroAddBot(t, s, inst, id, dotaTeamHuman, float32(p.X), float32(p.Y), 0)
			b.c.snapT = float32(now)
		}
		// The human remains at its fountain; it must not contribute to active coverage.
		c.snapT = float32(now)
		inst.mu.Lock()
		plan := s.botPlanTeamLocked(inst, dotaTeamHuman, now)
		inst.mu.Unlock()
		lanes := map[int32]int{}
		for id, b := range inst.bots {
			lanes[id] = b.lane
		}
		return plan, lanes
	}

	planA, lanesA := build([]int32{botIDBase + 1, botIDBase + 2, botIDBase + 3, botIDBase + 4})
	planB, lanesB := build([]int32{botIDBase + 4, botIDBase + 2, botIDBase + 1, botIDBase + 3})
	if !reflect.DeepEqual(planA, planB) || !reflect.DeepEqual(lanesA, lanesB) {
		t.Fatalf("member insertion order changed plan: A=%+v/%v B=%+v/%v", planA, lanesA, planB, lanesB)
	}

	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	for i := int32(1); i <= 4; i++ {
		macroAddBot(t, s, inst, botIDBase+i, dotaTeamHuman, 0, 0, int(i%3))
	}
	baseline := map[int32]int{}
	for id, b := range inst.bots {
		baseline[id] = b.lane
	}
	rec := &telemetryRecorder{ch: make(chan any, 16)}
	inst.dota.telemetry = rec
	inst.mu.Lock()
	var keyAtTen string
	for _, now := range []float64{10, 100} {
		for _, b := range inst.bots {
			b.c.snapT = float32(now)
		}
		s.botPlanTeamsLocked(inst, now)
		key := botTeamPlanKey(inst.dota.teamPlans[dotaTeamHuman])
		if keyAtTen == "" {
			keyAtTen = key
		} else if key != keyAtTen {
			inst.mu.Unlock()
			t.Fatalf("equivalent live state changed plan key with match time: %q -> %q", keyAtTen, key)
		}
	}
	inst.mu.Unlock()
	for id, want := range baseline {
		if got := inst.bots[id].lane; got != want {
			t.Fatalf("baseline lane for bot %d changed from %d to %d", id, want, got)
		}
	}
	if got := len(rec.ch); got != 2 {
		t.Fatalf("unchanged two-team plan emitted %d telemetry events, want initial two only", got)
	}
}

func TestDotaBasePressureUsesLiveBoundedDefendersAndClears(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	altar := altarOf(inst, dotaTeamHuman)
	if altar == nil {
		t.Fatal("setup: missing own altar")
	}
	now := float64(s.battleTime()) + 20
	bots := []*botBrain{
		macroAddBot(t, s, inst, botIDBase+1, dotaTeamHuman, altar.x+1, altar.y, 0),
		macroAddBot(t, s, inst, botIDBase+2, dotaTeamHuman, altar.x+10, altar.y, 1),
		macroAddBot(t, s, inst, botIDBase+3, dotaTeamHuman, altar.x+20, altar.y, 2),
		macroAddBot(t, s, inst, botIDBase+4, dotaTeamHuman, altar.x+30, altar.y, 0),
		macroAddBot(t, s, inst, botIDBase+5, dotaTeamHuman, altar.x+40, altar.y, 1),
	}
	baseline := map[int32]int{}
	for _, b := range bots {
		b.c.snapT = float32(now)
		baseline[b.c.objID] = b.lane
	}
	bots[0].c.huntState.hp = bots[0].c.huntState.maxHPLocked(now) * 0.40
	for i := 0; i < 3; i++ {
		macroEnemyCreep(inst, botIDBase+100+int32(i), altar.x+float32(i), altar.y)
	}

	inst.mu.Lock()
	plan := s.botPlanTeamLocked(inst, dotaTeamHuman, now)
	inst.mu.Unlock()
	if plan.Mode != botMacroBase {
		t.Fatalf("live altar pressure plan mode=%q, want base", plan.Mode)
	}
	responders := 0
	for id, a := range plan.Assignments {
		if a.Mode == botMacroBase {
			responders++
			if id == bots[0].c.objID {
				t.Fatal("unhealthy nearest bot selected as defender")
			}
		}
	}
	if responders == 0 || responders > 3 {
		t.Fatalf("base responders=%d, want bounded 1..3", responders)
	}
	for _, b := range bots[1:4] {
		if plan.Assignments[b.c.objID].Mode != botMacroBase {
			t.Fatalf("healthy nearer bot %d was not selected: %+v", b.c.objID, plan.Assignments[b.c.objID])
		}
	}
	if plan.Assignments[bots[4].c.objID].Mode == botMacroBase {
		t.Fatal("farther healthy bot was selected before nearer healthy defenders")
	}
	for id, want := range baseline {
		if got := inst.bots[id].lane; got != want {
			t.Fatalf("base overlay mutated bot %d baseline lane from %d to %d", id, want, got)
		}
	}

	for _, m := range inst.mobs {
		if !m.structure && m.team == dotaTeamElf {
			m.x, m.y = altar.x+1000, altar.y+1000
		}
	}
	inst.mu.Lock()
	cleared := s.botPlanTeamLocked(inst, dotaTeamHuman, now)
	inst.mu.Unlock()
	if cleared.Mode == botMacroBase {
		t.Fatal("base defense remained latched after live pressure left")
	}

	altar.hp = altar.maxHealth() * 0.40
	inst.mu.Lock()
	damagedOnly := s.botPlanTeamLocked(inst, dotaTeamHuman, now)
	inst.mu.Unlock()
	if damagedOnly.Mode == botMacroBase {
		t.Fatal("damaged altar alone latched base defense")
	}
}

func TestDotaBasePressureCounterPushKeepsTwoDefenders(t *testing.T) {
	type fixture struct {
		s       *Server
		human   *conn
		inst    *huntInstance
		now     float64
		cleanup func()
	}
	newFixture := func(t *testing.T, botCount int, viable, severe bool, order []int32) fixture {
		t.Helper()
		s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
		now := float64(s.battleTime()) + 20
		altar := altarOf(inst, dotaTeamHuman)
		if altar == nil {
			cleanup()
			t.Fatal("setup: missing own altar")
		}
		for i, id := range order {
			if i >= botCount {
				break
			}
			b := macroAddBot(t, s, inst, id, dotaTeamHuman, altar.x+20, altar.y, 0)
			b.c.snapT = float32(now)
		}
		if !viable {
			for _, m := range inst.mobs {
				if !m.structure && m.team == dotaTeamHuman {
					m.dead = true
				}
			}
		}
		if viable {
			lane := inst.dota.m.Lanes[0]
			front := &mobState{id: 63000, mobIdx: inst.dota.m.HumanCreepMelee, mob: gamedata.Mobs()[inst.dota.m.HumanCreepMelee],
				x: float32(lane[len(lane)-1].X), y: float32(lane[len(lane)-1].Y), hp: 500, maxHP: 500,
				team: dotaTeamHuman, lane: lane, laneIdx: len(lane) - 1}
			inst.mobs[front.id] = front
		}
		pressureCount := 1
		if severe {
			pressureCount = 3
		}
		for i := 0; i < pressureCount; i++ {
			macroEnemyCreep(inst, 63100+int32(i), altar.x+float32(i), altar.y)
		}
		human.snapT = float32(now)
		hx, hy := botHomeLocked(human)
		macroSetPosition(human, hx, hy, now) // AFK human stays excluded from lane coverage.
		return fixture{s: s, human: human, inst: inst, now: now, cleanup: cleanup}
	}
	planFor := func(t *testing.T, f fixture) botTeamPlan {
		t.Helper()
		f.inst.mu.Lock()
		plan := f.s.botPlanTeamLocked(f.inst, dotaTeamHuman, f.now)
		f.inst.mu.Unlock()
		return plan
	}
	countRoles := func(plan botTeamPlan) (defenders, counterPush int) {
		for _, assignment := range plan.Assignments {
			if assignment.Mode == botMacroBase && (assignment.Role == "defender" || assignment.Role == "cover") {
				defenders++
			}
			if assignment.Mode == botMacroPush && assignment.Role == botMacroCounterPushRole {
				counterPush++
			}
		}
		return defenders, counterPush
	}

	t.Run("selection and deterministic order", func(t *testing.T) {
		order := []int32{botIDBase + 100, botIDBase + 200, botIDBase + 300, botIDBase + 400}
		first := newFixture(t, 4, true, false, order)
		defer first.cleanup()
		plan := planFor(t, first)
		defenders, counterPush := countRoles(plan)
		if plan.Mode != botMacroBase || defenders != 2 || counterPush != 1 {
			t.Fatalf("base assignments=%+v, want two defenders and one counter-push", plan.Assignments)
		}
		var counterID int32
		for id, assignment := range plan.Assignments {
			if assignment.Role == botMacroCounterPushRole {
				counterID = id
				if assignment.Mode != botMacroPush || assignment.Lane != 0 || assignment.ObjectiveID == 0 || !assignment.Aggressive {
					t.Fatalf("counter-push assignment=%+v, want existing push mode on viable lane", assignment)
				}
			}
		}
		if counterID == 0 {
			t.Fatal("no counter-push candidate selected")
		}

		second := newFixture(t, 4, true, false, []int32{order[3], order[1], order[0], order[2]})
		defer second.cleanup()
		if got := botTeamPlanKey(planFor(t, second)); got != botTeamPlanKey(plan) {
			t.Fatalf("member insertion order changed counter-push plan: first=%q second=%q", botTeamPlanKey(plan), got)
		}
	})

	t.Run("severe pressure keeps three defenders", func(t *testing.T) {
		f := newFixture(t, 4, true, true, []int32{botIDBase + 500, botIDBase + 600, botIDBase + 700, botIDBase + 800})
		defer f.cleanup()
		plan := planFor(t, f)
		defenders, counterPush := countRoles(plan)
		if defenders != 3 || counterPush != 0 {
			t.Fatalf("severe-pressure assignments=%+v, want three defenders and no counter-push", plan.Assignments)
		}
	})

	for _, tc := range []struct {
		name    string
		bots    int
		viable  bool
		wantDef int
	}{
		{name: "no viable lane", bots: 4, viable: false, wantDef: 1},
		{name: "fewer than three eligible bots", bots: 2, viable: true, wantDef: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newFixture(t, tc.bots, tc.viable, false, []int32{botIDBase + 900, botIDBase + 901, botIDBase + 902, botIDBase + 903})
			defer f.cleanup()
			if tc.name == "no viable lane" {
				for _, m := range f.inst.mobs {
					if m.structure && m.team == dotaTeamElf {
						m.dead = true
					}
				}
			}
			plan := planFor(t, f)
			defenders, counterPush := countRoles(plan)
			if counterPush != 0 || defenders != tc.wantDef {
				t.Fatalf("assignments=%+v, want %d current defenders and no counter-push", plan.Assignments, tc.wantDef)
			}
		})
	}

	t.Run("AFK human does not make a lane viable", func(t *testing.T) {
		f := newFixture(t, 3, false, false, []int32{botIDBase + 1000, botIDBase + 1001, botIDBase + 1002})
		defer f.cleanup()
		altar := altarOf(f.inst, dotaTeamHuman)
		for _, b := range f.inst.bots {
			macroSetPosition(b.c, altar.x+1000, altar.y+1000, f.now)
		}
		objective, _ := f.s.botMacroLaneObjectiveLocked(f.inst, dotaTeamHuman, 0)
		if objective == nil {
			t.Fatal("setup: missing enemy lane objective")
		}
		objective.hp = objective.maxHealth() * 0.50
		if botHumanMacroActiveLocked(f.inst, f.human, f.now) {
			t.Fatal("setup human at fountain was counted as macro-active")
		}
		if _, counterPush := countRoles(planFor(t, f)); counterPush != 0 {
			t.Fatal("AFK human at fountain created a counter-push lane")
		}
		lane := f.inst.dota.m.Lanes[0]
		macroSetPosition(f.human, float32(lane[len(lane)/2].X), float32(lane[len(lane)/2].Y), f.now)
		if !botHumanMacroActiveLocked(f.inst, f.human, f.now) {
			t.Fatal("active human leaving fountain was not counted as macro-active")
		}
		if _, counterPush := countRoles(planFor(t, f)); counterPush != 1 {
			t.Fatal("active human lane coverage did not make the damaged lane viable")
		}
	})
}

func TestDotaMacroModesBoundRespondersAndKeepRecoveryNonAggressive(t *testing.T) {
	t.Run("push", func(t *testing.T) {
		s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
		defer cleanup()
		now := float64(s.battleTime()) + 20
		p := inst.dota.m.Lanes[0][len(inst.dota.m.Lanes[0])/2]
		bots := make([]*botBrain, 0, 7)
		baseline := map[int32]int{}
		for i := int32(0); i < 7; i++ {
			b := macroAddBot(t, s, inst, botIDBase+10+i, dotaTeamHuman, float32(p.X), float32(p.Y), 0)
			b.c.snapT = float32(now)
			baseline[b.c.objID] = b.lane
			bots = append(bots, b)
		}
		bots[5].c.huntState.deadUntil = now + 10
		bots[6].retreating, bots[6].retreatMode = true, botRetreatModeDisengage
		ownCreep := &mobState{id: 62001, mobIdx: inst.dota.m.HumanCreepMelee, mob: gamedata.Mobs()[inst.dota.m.HumanCreepMelee],
			x: float32(p.X), y: float32(p.Y), hp: 500, maxHP: 500, team: dotaTeamHuman,
			lane: inst.dota.m.Lanes[0], laneIdx: len(inst.dota.m.Lanes[0]) - 1}
		inst.mobs[ownCreep.id] = ownCreep

		inst.mu.Lock()
		plan := s.botPlanTeamLocked(inst, dotaTeamHuman, now)
		inst.mu.Unlock()
		if plan.Mode != botMacroPush {
			t.Fatalf("push setup selected mode=%q reason=%q", plan.Mode, plan.Reason)
		}
		responders := 0
		baselineCoverage := 0
		for id, b := range inst.bots {
			a := plan.Assignments[id]
			if a.Mode == botMacroPush || a.Mode == botMacroCover {
				responders++
			}
			if id == bots[5].c.objID || id == bots[6].c.objID {
				if a.Mode != botMacroRecover || a.Aggressive {
					t.Fatalf("bot %d recovery assignment=%+v", id, a)
				}
			}
			if a.Mode == botMacroLane && (!a.Coverage || a.BaselineLane != b.lane) {
				t.Fatalf("baseline assignment for bot %d=%+v", id, a)
			}
			if a.Mode == botMacroLane && a.Coverage {
				baselineCoverage++
			}
			if b.lane != baseline[id] {
				t.Fatalf("push overlay mutated bot %d baseline lane", id)
			}
		}
		if responders == 0 || responders > 3 {
			t.Fatalf("push responders=%d, want bounded 1..3", responders)
		}
		if baselineCoverage == 0 {
			t.Fatal("push responders consumed every bot; remaining bots must keep baseline coverage")
		}
	})

	t.Run("rally", func(t *testing.T) {
		s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
		defer cleanup()
		now := float64(s.battleTime()) + 20
		p := inst.dota.m.Lanes[0][len(inst.dota.m.Lanes[0])/2]
		b := macroAddBot(t, s, inst, botIDBase+30, dotaTeamHuman, float32(p.X), float32(p.Y), 0)
		b.c.snapT = float32(now)
		inst.mobs[62002] = &mobState{id: 62002, mobIdx: inst.dota.m.HumanCreepMelee, mob: gamedata.Mobs()[inst.dota.m.HumanCreepMelee],
			x: float32(p.X), y: float32(p.Y), hp: 500, maxHP: 500, team: dotaTeamHuman,
			lane: inst.dota.m.Lanes[0], laneIdx: len(inst.dota.m.Lanes[0]) - 1}
		inst.mu.Lock()
		plan := s.botPlanTeamLocked(inst, dotaTeamHuman, now)
		inst.mu.Unlock()
		if plan.Mode != botMacroRally {
			t.Fatalf("rally setup selected mode=%q reason=%q", plan.Mode, plan.Reason)
		}
		responders := 0
		for _, a := range plan.Assignments {
			if a.Mode == botMacroRally || a.Mode == botMacroCover {
				responders++
			}
		}
		if responders > 1 {
			t.Fatalf("rally responders=%d, want at most one", responders)
		}
	})

	t.Run("altar", func(t *testing.T) {
		s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
		defer cleanup()
		now := float64(s.battleTime()) + 20
		altar := altarOf(inst, dotaTeamElf)
		macroOpenAltar(inst, altar)
		for i := int32(0); i < 5; i++ {
			b := macroAddBot(t, s, inst, botIDBase+40+i, dotaTeamHuman, altar.x+float32(i+1), altar.y, int(i%3))
			b.c.snapT = float32(now)
		}
		inst.mu.Lock()
		plan := s.botPlanTeamLocked(inst, dotaTeamHuman, now)
		inst.mu.Unlock()
		if plan.Mode != botMacroAltar {
			t.Fatalf("altar setup selected mode=%q reason=%q", plan.Mode, plan.Reason)
		}
		responders := 0
		baseline := 0
		for _, a := range plan.Assignments {
			if a.Mode == botMacroAltar || a.Mode == botMacroCover {
				responders++
			}
			if a.Mode == botMacroLane && a.Coverage {
				baseline++
			}
		}
		if responders == 0 || responders > 3 || baseline == 0 {
			t.Fatalf("altar responders=%d baseline=%d, want bounded responders and remaining lane coverage", responders, baseline)
		}
	})
}

func TestDotaAFKHumanAtFountainIsExcludedFromMacroActivity(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime()) + 10
	hx, hy := botHomeLocked(human)
	macroSetPosition(human, hx, hy, now)
	if botHumanMacroActiveLocked(inst, human, now) {
		t.Fatal("human at fountain counted as macro-active")
	}
	p := inst.dota.m.Lanes[0][0]
	macroSetPosition(human, float32(p.X), float32(p.Y), now)
	if !botHumanMacroActiveLocked(inst, human, now) {
		t.Fatal("human who left fountain was not counted as macro-active")
	}
}
