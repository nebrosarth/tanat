package battleserver

import (
	"math"
	"sort"
	"testing"

	"tanatserver/internal/gamedata"
)

// TestSpawnDotaBotsFillsBalancedTeams: spawning bots up to a target headcount keeps the
// two sides within 1 of each other and assigns lanes in the 2/1/2 pattern per side.
// newDotaConn's solo member never calls assignSide() (it defaults to Human without
// advancing dotaState.nextSide) -- exactly the "explicit/pre-assigned side" case a real
// matchmade player hits, which is what desynced the fill in the first place (a 10-headcount
// fill split 6/4 before balancedDotaSideLocked started counting actual membership instead
// of trusting the alternator).
func TestSpawnDotaBotsFillsBalancedTeams(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()

	human.lock()
	defer human.unlock()
	s.spawnDotaBotsLocked(inst, 10)

	if got := len(inst.members); got != 10 {
		t.Fatalf("members = %d, want 10 (1 real + 9 bots)", got)
	}
	if got := len(inst.bots); got != 9 {
		t.Fatalf("bot brains = %d, want 9", got)
	}
	humanN, elfN := 0, 0
	for _, mem := range inst.members {
		if mem.playerTeam() == dotaTeamHuman {
			humanN++
		} else {
			elfN++
		}
	}
	if humanN+elfN != 10 {
		t.Fatalf("human+elf = %d, want 10", humanN+elfN)
	}
	if diff := humanN - elfN; diff > 1 || diff < -1 {
		t.Fatalf("sides = human %d / elf %d, want within 1 of each other (5/5 for 10)", humanN, elfN)
	}
	for _, b := range inst.bots {
		if b.lane < 0 || b.lane >= len(inst.dota.m.Lanes) {
			t.Fatalf("bot lane = %d, out of range [0,%d)", b.lane, len(inst.dota.m.Lanes))
		}
	}
	// The real player is on Velial, one of the 10 roster avatars -- a full 10-headcount
	// fill (1 real + 9 bots against a 10-avatar roster) is exactly the scenario that
	// reported a bot on the ENEMY team playing the same hero as the real player's pick.
	seen := map[int32]int32{} // avatar id -> which objID has it
	for id, mem := range inst.members {
		if other, dup := seen[mem.huntState.av.ID]; dup {
			t.Fatalf("avatar %d (%s) played by both %d and %d -- a full-roster fill must never duplicate a hero",
				mem.huntState.av.ID, mem.huntState.av.Prefab, other, id)
		}
		seen[mem.huntState.av.ID] = id
	}
}

// TestSpawnDotaBotsSkipsPlayersAvatar: reported live -- a real player who picked one of
// the bot roster's own 10 avatars found a bot on the ENEMY team playing the exact same
// hero. A small fill (well short of exhausting the 10-avatar roster) must never hand any
// bot the real player's already-picked avatar.
func TestSpawnDotaBotsSkipsPlayersAvatar(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()

	human.lock()
	defer human.unlock()
	s.spawnDotaBotsLocked(inst, 5)

	for id, mem := range inst.members {
		if mem == human {
			continue
		}
		if mem.huntState.av.ID == human.huntState.av.ID {
			t.Fatalf("bot %d was assigned %s, the same avatar the real player already picked",
				id, mem.huntState.av.Prefab)
		}
	}
}

// TestSpawnDotaBotsBalancesAroundExplicitElfPlayer: the same desync bug, mirrored --
// a real player pre-assigned to the ELF side (dotaPlayerConn's explicit team, exactly how
// a matchmaker-assigned PendingBattle.Team lands) must still get an evenly split fill, not
// have the fill treat them as if nextSide already accounted for them.
func TestSpawnDotaBotsBalancesAroundExplicitElfPlayer(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	// Replace the default Human solo member with an explicit Elf one, the same way
	// sendHuntWorldState assigns a matchmade player's side straight from
	// PendingBattle.Team without ever touching dotaState.assignSide().
	delete(inst.members, human.objID)
	elf := dotaPlayerConn(t, s, inst, 2000, dotaTeamElf, human.x, human.y)

	elf.lock()
	defer elf.unlock()
	s.spawnDotaBotsLocked(inst, 10)

	humanN, elfN := 0, 0
	for _, mem := range inst.members {
		if mem.playerTeam() == dotaTeamHuman {
			humanN++
		} else {
			elfN++
		}
	}
	if humanN+elfN != 10 {
		t.Fatalf("human+elf = %d, want 10", humanN+elfN)
	}
	if diff := humanN - elfN; diff > 1 || diff < -1 {
		t.Fatalf("sides = human %d / elf %d, want within 1 of each other (5/5 for 10)", humanN, elfN)
	}
}

// TestBotLastHitsLowHPCreep: a bot with an enemy creep in range that will die to its next
// swing must attack it, not walk past it toward the lane point.
func TestBotLastHitsLowHPCreep(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	b := &botBrain{c: human, slot: 0, lane: 0, phase: botPhaseLane}

	m := &mobState{id: 65100, mobIdx: inst.dota.m.ElfCreepMelee, mob: gamedata.Mobs()[inst.dota.m.ElfCreepMelee],
		x: human.x + 1, y: human.y, hp: 1, maxHP: 200, team: dotaTeamElf, shown: true, active: true}

	human.lock()
	defer human.unlock()
	inst.mobs[m.id] = m
	now := float64(s.battleTime())
	s.botLaneTickLocked(b, now)

	if human.huntState.attackTarget != m.id {
		t.Fatalf("bot did not attack the killable creep in range (attackTarget=%d, want %d)",
			human.huntState.attackTarget, m.id)
	}
}

// TestBotRetreatsAtLowHP: a bot below the retreat HP threshold must head home, not engage.
func TestBotRetreatsAtLowHP(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	b := &botBrain{c: human, slot: 0, lane: 0, phase: botPhaseLane}

	m := &mobState{id: 65101, mobIdx: inst.dota.m.ElfCreepMelee, mob: gamedata.Mobs()[inst.dota.m.ElfCreepMelee],
		x: human.x + 1, y: human.y, hp: 1, maxHP: 200, team: dotaTeamElf, shown: true, active: true}

	human.lock()
	defer human.unlock()
	inst.mobs[m.id] = m
	human.huntState.hp = 1 // far below botRetreatHPFrac of max
	now := float64(s.battleTime())
	s.botLaneTickLocked(b, now)

	if human.huntState.attackTarget != 0 {
		t.Fatal("a critically low-HP bot attacked instead of retreating")
	}
	if !b.retreating {
		t.Fatal("bot did not latch retreating at critical HP")
	}
	hx, hy := botHomeLocked(human)
	if !human.hasDest && (human.x != hx || human.y != hy) {
		t.Fatalf("retreating bot neither moved toward home (%.1f,%.1f) nor is already there (at %.1f,%.1f)",
			hx, hy, human.x, human.y)
	}
}

// TestBotCrashRetreatDoesNotBypassRecoveryStateMachine pins the telemetry regression:
// once burst detection latched retreating, it used to return true forever and prevent the
// normal think path from healing or clearing the latch at safe HP.
func TestBotCrashRetreatDoesNotBypassRecoveryStateMachine(t *testing.T) {
	s, human, _, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	b := &botBrain{c: human, retreating: true}

	human.lock()
	defer human.unlock()
	now := float64(s.battleTime())
	if s.botCheckHPCrashLocked(b, now) {
		t.Fatal("an already-latched retreat must not bypass the regular recovery think path")
	}
	human.huntState.hp = human.huntState.maxHPLocked(now) * botSafeHPFrac
	if s.botShouldRetreatLocked(b, now) || b.retreating {
		t.Fatal("retreat latch did not clear after reaching safe HP")
	}
}

func TestBotPredictiveRetreatUsesDamageTrendAndHeroPressure(t *testing.T) {
	for _, tc := range []struct {
		name              string
		current, previous float64
		enemyDistance     float32
		want              bool
	}{
		{name: "losing close fight", current: 0.50, previous: 0.62, enemyDistance: 10, want: true},
		{name: "same damage without hero pressure", current: 0.50, previous: 0.62, enemyDistance: 40, want: false},
		{name: "harmless chip", current: 0.65, previous: 0.68, enemyDistance: 10, want: false},
		{name: "high hp remains committed", current: 0.80, previous: 1.00, enemyDistance: 10, want: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
			defer cleanup()
			b := &botBrain{c: bot, lane: 0, phase: botPhaseLane}
			now := float64(s.battleTime()) + 10
			bot.x, bot.y, bot.vx, bot.vy, bot.snapT = 0, 0, 0, 0, float32(now)
			dotaPlayerConn(t, s, inst, 65980, dotaTeamElf, tc.enemyDistance, 0)
			bot.huntState.hp = bot.huntState.maxHPLocked(now) * tc.current
			b.hpHistory[0] = hpSample{t: now - 0.8, frac: tc.previous}

			bot.lock()
			got := s.botShouldRetreatLocked(b, now)
			bot.unlock()
			if got != tc.want {
				t.Fatalf("predictive retreat = %v, want %v", got, tc.want)
			}
		})
	}

	t.Run("safe hp clears predictive latch", func(t *testing.T) {
		s, bot, _, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
		defer cleanup()
		b := &botBrain{c: bot}
		now := float64(s.battleTime()) + 10
		botSetRetreatModeLocked(b, botRetreatModeDisengage, now)
		bot.huntState.hp = bot.huntState.maxHPLocked(now) * botSafeHPFrac
		bot.lock()
		got := s.botShouldRetreatLocked(b, now+botDisengageMinHold-0.01)
		bot.unlock()
		if !got || !b.retreating {
			t.Fatal("predictive retreat cleared before its minimum hold expired")
		}

		bot.lock()
		got = s.botShouldRetreatLocked(b, now+botDisengageMinHold+0.01)
		bot.unlock()
		if got || b.retreating {
			t.Fatal("safe HP did not clear predictive retreat latch after its hold")
		}
	})
}

func TestBotPredictiveRetreatDisengagesBehindFriendlyWave(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	b := &botBrain{c: bot, lane: 0, phase: botPhaseLane}
	now := float64(s.battleTime()) + 10
	bot.x, bot.y, bot.snapT = 0, 0, float32(now)

	lane := inst.dota.m.Lanes[b.lane]
	wave := &mobState{
		id: 65152, team: dotaTeamHuman, lane: lane, laneIdx: 4,
		x: float32(lane[4].X), y: float32(lane[4].Y), hp: 100, maxHP: 100,
	}
	inst.mobs[wave.id] = wave
	dotaPlayerConn(t, s, inst, 65982, dotaTeamElf, 10, 0)
	bot.huntState.hp = bot.huntState.maxHPLocked(now) * 0.50
	b.hpHistory[0] = hpSample{t: now - 0.8, frac: 0.62}

	bot.lock()
	defer bot.unlock()
	if !s.botShouldRetreatLocked(b, now) {
		t.Fatal("predictive damage trend did not enter retreat")
	}
	if b.retreatMode != botRetreatModeDisengage {
		t.Fatalf("predictive retreat mode = %d, want disengage", b.retreatMode)
	}

	hx, hy := botHomeLocked(bot)
	fx, fy, ok := botLaneFrontLocked(bot, lane)
	if !ok {
		t.Fatal("setup: friendly wave was not recognized as the lane front")
	}
	rx, ry := s.botRetreatPointLocked(b, now)
	if rx == hx && ry == hy {
		t.Fatalf("predictive disengage destination = fountain (%.1f,%.1f), want behind-wave step", rx, ry)
	}
	if dist2(rx, ry, hx, hy) >= dist2(fx, fy, hx, hy) {
		t.Fatalf("disengage destination (%.1f,%.1f) is not behind friendly wave (%.1f,%.1f) toward home (%.1f,%.1f)",
			rx, ry, fx, fy, hx, hy)
	}
}

func TestBotRecoveryRetreatStateMachine(t *testing.T) {
	t.Run("low hp enters recovery at fountain", func(t *testing.T) {
		s, bot, _, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
		defer cleanup()
		b := &botBrain{c: bot, lane: 0}
		now := float64(s.battleTime())
		bot.huntState.hp = bot.huntState.maxHPLocked(now) * 0.30

		bot.lock()
		defer bot.unlock()
		if !s.botShouldRetreatLocked(b, now) {
			t.Fatal("30% HP did not enter recovery retreat")
		}
		if b.retreatMode != botRetreatModeRecovery {
			t.Fatalf("low-HP retreat mode = %d, want recovery", b.retreatMode)
		}
		hx, hy := botHomeLocked(bot)
		rx, ry := s.botRetreatPointLocked(b, now)
		if rx != hx || ry != hy {
			t.Fatalf("recovery destination = (%.1f,%.1f), want fountain (%.1f,%.1f)", rx, ry, hx, hy)
		}
	})

	t.Run("crash enters recovery at fountain", func(t *testing.T) {
		s, bot, _, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
		defer cleanup()
		b := &botBrain{c: bot}
		now := float64(s.battleTime())
		bot.huntState.hp = bot.huntState.maxHPLocked(now) * 0.65
		b.hpHistory[0] = hpSample{t: now - 0.8, frac: 0.95}

		bot.lock()
		defer bot.unlock()
		if !s.botCheckHPCrashLocked(b, now) {
			t.Fatal("25% HP crash did not enter recovery retreat")
		}
		if b.retreatMode != botRetreatModeRecovery {
			t.Fatalf("crash retreat mode = %d, want recovery", b.retreatMode)
		}
		hx, hy := botHomeLocked(bot)
		rx, ry := s.botRetreatPointLocked(b, now)
		if rx != hx || ry != hy {
			t.Fatalf("crash recovery destination = (%.1f,%.1f), want fountain (%.1f,%.1f)", rx, ry, hx, hy)
		}
	})

	t.Run("recovery clears at 55 percent", func(t *testing.T) {
		s, bot, _, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
		defer cleanup()
		b := &botBrain{c: bot, retreating: true, retreatMode: botRetreatModeRecovery}
		now := float64(s.battleTime())
		bot.huntState.hp = bot.huntState.maxHPLocked(now) * botSafeHPFrac

		bot.lock()
		defer bot.unlock()
		if s.botShouldRetreatLocked(b, now) || b.retreating {
			t.Fatal("recovery retreat did not clear at safe 55% HP")
		}
	})

	t.Run("death resets any retreat mode", func(t *testing.T) {
		s, bot, _, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
		defer cleanup()
		b := &botBrain{c: bot, retreating: true, retreatMode: botRetreatModeDisengage, retreatHoldUntil: 99}
		now := float64(s.battleTime())
		bot.huntState.deadUntil = now + 5

		bot.lock()
		defer bot.unlock()
		if s.botShouldRetreatLocked(b, now) || b.retreating {
			t.Fatal("dead bot retained retreating state")
		}
		if b.retreatMode != botRetreatModeRecovery || b.retreatHoldUntil != 0 {
			t.Fatalf("dead bot retreat state = mode %d hold %.2f, want recovery/0", b.retreatMode, b.retreatHoldUntil)
		}
	})
}

func TestBotPredictiveRetreatRecognizesRecentDamageInsideEnemyGunZone(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	gun := structOfSide(inst, gamedata.DotaGun, dotaTeamElf)
	if gun == nil {
		t.Fatal("setup: missing enemy gun")
	}
	now := float64(s.battleTime()) + 10
	bot.x, bot.y, bot.vx, bot.vy, bot.snapT = gun.x, gun.y, 0, 0, float32(now)
	bot.huntState.hp = bot.huntState.maxHPLocked(now) * 0.50
	b := &botBrain{c: bot, lane: 0, phase: botPhaseGroup}
	b.hpHistory[0] = hpSample{t: now - 0.8, frac: 0.62}

	bot.lock()
	defer bot.unlock()
	if !s.botShouldRetreatLocked(b, now) || !b.retreating {
		t.Fatal("recent damage inside an enemy gun zone did not trigger predictive retreat")
	}
}

func TestBotPredictiveRetreatContinuesUnderPressure(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	b := &botBrain{c: bot}
	now := float64(s.battleTime()) + 10
	bot.x, bot.y, bot.snapT = 0, 0, float32(now)
	dotaPlayerConn(t, s, inst, 65983, dotaTeamElf, 10, 0)
	botSetRetreatModeLocked(b, botRetreatModeDisengage, now)
	bot.huntState.hp = bot.huntState.maxHPLocked(now) * 0.50

	bot.lock()
	defer bot.unlock()
	if !s.botShouldRetreatLocked(b, now+botDisengageMinHold+1) || !b.retreating {
		t.Fatal("predictive retreat cleared despite continued nearby enemy pressure")
	}
}

func TestBotDisengageEscalatesBeforeCriticalHPWhenPressurePersists(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	b := &botBrain{c: bot, retreating: true, retreatMode: botRetreatModeDisengage}
	now := float64(s.battleTime()) + 10
	bot.x, bot.y, bot.snapT = 0, 0, float32(now)
	dotaPlayerConn(t, s, inst, 65984, dotaTeamElf, 10, 0)
	bot.huntState.hp = bot.huntState.maxHPLocked(now) * 0.44
	b.retreatHoldUntil = now - 1

	bot.lock()
	defer bot.unlock()
	if !s.botShouldRetreatLocked(b, now) {
		t.Fatal("persistent pressure did not keep the bot retreating")
	}
	if b.retreatMode != botRetreatModeRecovery {
		t.Fatalf("persistent pressure mode = %d, want recovery before critical HP", b.retreatMode)
	}
	hx, hy := botHomeLocked(bot)
	rx, ry := s.botRetreatPointLocked(b, now)
	if rx != hx || ry != hy {
		t.Fatalf("persistent-pressure destination=(%.1f,%.1f), want fountain=(%.1f,%.1f)", rx, ry, hx, hy)
	}
}

func TestBotDisengageEscalatesToCriticalRecovery(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	b := &botBrain{
		c:                bot,
		lane:             0,
		retreating:       true,
		retreatMode:      botRetreatModeDisengage,
		retreatHoldUntil: 99,
	}
	now := float64(s.battleTime())
	lane := inst.dota.m.Lanes[b.lane]
	wave := &mobState{
		id: 65153, team: dotaTeamHuman, lane: lane, laneIdx: 4,
		x: float32(lane[4].X), y: float32(lane[4].Y), hp: 100, maxHP: 100,
	}
	inst.mobs[wave.id] = wave
	bot.huntState.hp = bot.huntState.maxHPLocked(now) * botRetreatHPFrac

	bot.lock()
	defer bot.unlock()
	behindWaveX, behindWaveY := s.botRetreatPointLocked(b, now)
	if !s.botShouldRetreatLocked(b, now) {
		t.Fatal("critical HP cleared an active disengage instead of retaining retreat")
	}
	if b.retreatMode != botRetreatModeRecovery {
		t.Fatalf("critical disengage mode = %d, want recovery", b.retreatMode)
	}
	if b.retreatHoldUntil != 0 {
		t.Fatalf("critical recovery retained disengage hold until %.2f, want 0", b.retreatHoldUntil)
	}

	hx, hy := botHomeLocked(bot)
	rx, ry := s.botRetreatPointLocked(b, now)
	if rx != hx || ry != hy {
		t.Fatalf("critical recovery destination = (%.1f,%.1f), want fountain/detour recovery at fountain (%.1f,%.1f)",
			rx, ry, hx, hy)
	}
	if rx == behindWaveX && ry == behindWaveY {
		t.Fatalf("critical recovery kept behind-wave disengage destination (%.1f,%.1f)", rx, ry)
	}
}

// TestBotRecoveryAlwaysHeadsToFountain ensures a friendly wave cannot trap a damaged bot
// on the lane: lane fronts have no fountain regeneration and were the source of 200s+
// retreats in telemetry.
func TestBotRecoveryAlwaysHeadsToFountain(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	b := &botBrain{c: human, lane: 0, retreating: true}
	lane := inst.dota.m.Lanes[0]
	wave := &mobState{id: 65150, team: dotaTeamHuman, lane: lane, laneIdx: 2, x: 50, y: 50, hp: 100}

	human.lock()
	defer human.unlock()
	inst.mobs[wave.id] = wave
	hx, hy := botHomeLocked(human)
	rx, ry := s.botRetreatPointLocked(b, float64(s.battleTime()))
	if rx != hx || ry != hy {
		t.Fatalf("recovery point = (%.1f,%.1f), want fountain (%.1f,%.1f)", rx, ry, hx, hy)
	}
}

// TestBotRebalancesAwayFromAFKHuman verifies that a real player still at the fountain
// after the grace period stops reserving the first lane-pattern slot. Leaving base later
// restores that slot and rebalances the bots once more.
func TestBotRebalancesAwayFromAFKHuman(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()

	human.lock()
	defer human.unlock()
	s.spawnDotaBotsLocked(inst, 10)
	now := float64(s.battleTime())
	inst.dota.startedAt = now - botHumanLaneGrace - 1
	inst.dota.nextLaneRebalanceAt = 0
	s.botRebalanceLanesLocked(inst, now)

	var teamBots []*botBrain
	for _, b := range inst.bots {
		if b.c.playerTeam() == human.playerTeam() {
			teamBots = append(teamBots, b)
		}
	}
	sort.Slice(teamBots, func(i, j int) bool { return teamBots[i].slot < teamBots[j].slot })
	for i, b := range teamBots {
		if want := assignBotLane(i); b.lane != want {
			t.Fatalf("AFK-human bot %d lane = %d, want %d", i, b.lane, want)
		}
	}

	human.x += botHumanLeftBaseRadius + 5
	human.snapT = float32(now + 1)
	inst.dota.nextLaneRebalanceAt = 0
	s.botRebalanceLanesLocked(inst, now+1)
	if !inst.dota.laneActiveHumans[human.objID] {
		t.Fatal("human was not marked lane-active after leaving base")
	}
	for i, b := range teamBots {
		if want := assignBotLane(i + 1); b.lane != want {
			t.Fatalf("active-human bot %d lane = %d, want %d", i, b.lane, want)
		}
	}
}

// TestBotRetreatCancelsOwnAttack: a bot that was already auto-attacking (mob or PvP) when
// it drops to retreat HP must actually leave -- not have its own still-armed attack chase
// walk it right back into the fight. Reported live: a low-HP bot tried to retreat and its
// own auto-attack kept pulling it back. handleMove cancels this for a real player's click,
// but a bot's move decision never goes through handleMove at all.
func TestBotRetreatCancelsOwnAttack(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, human.x+2, human.y)
	b := &botBrain{c: human, slot: 0, lane: 0, phase: botPhaseLane}

	human.lock()
	defer human.unlock()
	// Pull the bot away from its own spawn/home first -- newDotaConn spawns exactly on
	// home, which would make the retreat move a same-point no-op and defeat the point of
	// asserting hasDest below.
	human.x, human.snapT = human.x+40, s.battleTime()
	s.startPvpAttackLocked(human, elf)
	if human.huntState.pvpTarget != elf.objID {
		t.Fatal("setup: PvP attack did not start")
	}
	human.huntState.hp = 1 // far below botRetreatHPFrac of max
	now := float64(s.battleTime())
	s.botLaneTickLocked(b, now)

	if human.huntState.pvpTarget != 0 {
		t.Fatalf("pvpTarget = %d after the bot decided to retreat, want 0 (still armed to chase back)",
			human.huntState.pvpTarget)
	}
	if !human.hasDest {
		t.Fatal("retreating bot has no destination -- its own attack chase blocked the move")
	}
}

// TestBotAvoidsOutnumberedFight: a bot must not engage a lone enemy hero who has two
// teammates standing right next to them.
func TestBotAvoidsOutnumberedFight(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	b := &botBrain{c: human, slot: 0, lane: 0, phase: botPhaseLane}

	elf1 := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, human.x+3, human.y)
	elf2 := dotaPlayerConn(t, s, inst, 1002, dotaTeamElf, human.x+3.5, human.y+0.5)
	_ = elf2

	human.lock()
	defer human.unlock()
	now := float64(s.battleTime())
	acted := s.botCombatTickLocked(b, now)

	if acted || human.huntState.pvpTarget != 0 {
		t.Fatalf("bot engaged a 1-vs-2 fight (pvpTarget=%d, acted=%v)", human.huntState.pvpTarget, acted)
	}
	_ = elf1
}

// TestBotEngagesFavorableFight: a bot facing a single, similarly-escorted enemy hero (a
// fair 1v1) must commit to the fight.
func TestBotEngagesFavorableFight(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	b := &botBrain{c: human, slot: 0, lane: 0, phase: botPhaseLane}
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, human.x+3, human.y)

	human.lock()
	defer human.unlock()
	now := float64(s.battleTime())
	acted := s.botCombatTickLocked(b, now)

	if !acted || human.huntState.pvpTarget != elf.objID {
		t.Fatalf("bot did not engage a fair 1v1 (acted=%v pvpTarget=%d, want %d)",
			acted, human.huntState.pvpTarget, elf.objID)
	}
}

// TestBotDoesNotEngageNearEnemyStructure: a bot must not pick a fight with an enemy hero
// standing inside their own tower/cannon's range -- fighting there means fighting the
// structure too, not just the hero.
func TestBotDoesNotEngageNearEnemyStructure(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	b := &botBrain{c: human, slot: 0, lane: 0, phase: botPhaseLane}
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, human.x+3, human.y)
	gun := &mobState{id: 65200, structure: true, team: dotaTeamElf, x: elf.x, y: elf.y, hp: 100, maxHP: 100}

	human.lock()
	defer human.unlock()
	inst.mobs[gun.id] = gun
	now := float64(s.battleTime())
	acted := s.botCombatTickLocked(b, now)

	if acted || human.huntState.pvpTarget != 0 {
		t.Fatalf("bot engaged a hero standing in its own structure's range (pvpTarget=%d, acted=%v)",
			human.huntState.pvpTarget, acted)
	}
}

// TestBotBreaksOffChaseIntoEnemyStructureRange: a bot already mid-chase must actually
// disengage once press the fight would mean tower-diving, not just stop picking new
// fights. Reported live: a bot kept chasing a fleeing hero straight through the enemy's
// own cannons and onto their fountain -- armPvpAttackTimer has no notion of "too deep", so
// the brain dropping the target has to actually cancel the attack, not just stop re-
// affirming it.
func TestBotBreaksOffChaseIntoEnemyStructureRange(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	b := &botBrain{c: human, slot: 0, lane: 0, phase: botPhaseLane}
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, human.x+3, human.y)

	human.lock()
	defer human.unlock()
	s.startPvpAttackLocked(human, elf)
	b.engageTarget = elf.objID
	if human.huntState.pvpTarget != elf.objID {
		t.Fatal("setup: PvP attack did not start")
	}
	// The chase has now carried the fleeing target next to their own cannon.
	gun := &mobState{id: 65201, structure: true, team: dotaTeamElf, x: elf.x, y: elf.y, hp: 100, maxHP: 100}
	inst.mobs[gun.id] = gun
	now := float64(s.battleTime())
	acted := s.botCombatTickLocked(b, now)

	if !acted || human.huntState.pvpTarget != 0 {
		t.Fatalf("bot did not issue a tactical disengage from structure range (pvpTarget=%d, acted=%v)",
			human.huntState.pvpTarget, acted)
	}
}

// TestBotSpendsSkillPointOnAvailability: a bot with a banked point and a learnable slot
// must spend it the same tick, through the real UPGRADE_SKILL validation.
func TestBotSpendsSkillPointOnAvailability(t *testing.T) {
	s, human, _, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	b := &botBrain{c: human, slot: 0, lane: 0, phase: botPhaseLane}

	human.lock()
	defer human.unlock()
	human.huntState.points = 1
	s.botSpendSkillPointLocked(b)

	total := int32(0)
	for _, r := range human.huntState.skillLevel {
		total += r
	}
	if total != 1 {
		t.Fatalf("skill ranks total = %d after spending 1 point, want 1", total)
	}
	if human.huntState.points != 0 {
		t.Fatalf("points left = %d, want 0", human.huntState.points)
	}
}

// TestBotBuysAffordableItem: a bot with enough gold for a root item in its preferred tree
// must buy it.
func TestBotBuysAffordableItem(t *testing.T) {
	s, human, _, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	b := &botBrain{c: human, slot: 0, lane: 0, phase: botPhaseLane}

	store := s.Store
	store.CreateBotHero(human.selfPlayerID, "test")
	money, _, _ := store.HeroMoney(human.selfPlayerID)
	if money <= 0 {
		t.Fatalf("test hero starting money = %d, want positive", money)
	}

	human.lock()
	defer human.unlock()
	now := float64(s.battleTime())
	s.botBuyItemsLocked(b, now)

	if len(human.huntState.ownedTreeItems) == 0 {
		t.Fatal("bot did not buy any item despite having starting gold")
	}
}

// TestBotHealsHurtAlly: a support-type bot with a ready FRIEND heal and a badly hurt
// nearby ally must cast on the ally, not itself, and the ally's HP must actually rise.
func TestBotHealsHurtAlly(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Sp_Arianna")
	defer cleanup()
	b := &botBrain{c: human, slot: 0, lane: 0, phase: botPhaseLane}
	ally := dotaPlayerConn(t, s, inst, 1001, dotaTeamHuman, human.x+2, human.y)

	human.lock()
	defer human.unlock()
	hs := human.huntState
	healSlot := 0
	for slot := 1; slot <= 4; slot++ {
		def := hs.skillDef(slot)
		if skillHasTargetFlag(def, "FRIEND") &&
			(botSkillHasOp(def, gamedata.OpHeal) || botSkillHasOp(def, gamedata.OpHot) || botSkillHasOp(def, gamedata.OpShield)) {
			healSlot = slot
			break
		}
	}
	if healSlot == 0 {
		t.Fatal("Arianna's kit has no FRIEND heal/hot/shield skill -- wrong test avatar")
	}
	now := float64(s.battleTime())
	hs.skillLevel[healSlot-1] = 1
	hs.mana = hs.maxManaLocked(now)
	ally.huntState.hp = 1

	acted := s.botConsiderHealLocked(b, now)
	if !acted {
		t.Fatal("bot did not cast its ready heal on a critically hurt ally")
	}
	// The cast's payload may be delayed (a cast wind-up, e.g. Arianna's PayloadDelay=0.3)
	// rather than landing synchronously -- run the queued payload forward past it.
	s.runDuePayloadsLocked(human, now+1.0)
	// The chosen skill may be an instant heal, a heal-over-time stream, or a shield --
	// any of the three is a real support action; only silence is a failure.
	ah := ally.huntState
	if ah.hp <= 1 && len(ah.st.hots) == 0 && ah.st.shield <= 0 {
		t.Fatalf("ally shows no sign of being healed/shielded after the bot's cast (hp=%g hots=%d shield=%g)",
			ah.hp, len(ah.st.hots), ah.st.shield)
	}
}

// TestBotCommittedStructureFocusOnlySeesActiveGunOrTowerShots distinguishes a
// structure's current dtarget from a committed shot. A generator/idle structure,
// or a gun/tower whose committed shot is aimed at a creep, must not make this bot
// flee; a live tower hitscan or cannon projectile aimed at the bot must.
func TestBotCommittedStructureFocusOnlySeesActiveGunOrTowerShots(t *testing.T) {
	tests := []struct {
		name       string
		role       gamedata.DotaRole
		configure  func(*mobState, float64, int32)
		wantThreat bool
	}{
		{
			name: "idle gun target only",
			role: gamedata.DotaGun,
			configure: func(m *mobState, _ float64, botID int32) {
				m.dtarget = botID
			},
		},
		{
			name: "tower committed on creep",
			role: gamedata.DotaCreepTower,
			configure: func(m *mobState, now float64, _ int32) {
				m.hitAt, m.hitTarget = now+1, 61001
			},
		},
		{
			name: "tower hitscan on bot",
			role: gamedata.DotaCreepTower,
			configure: func(m *mobState, now float64, botID int32) {
				m.hitAt, m.hitTarget = now+1, botID
			},
			wantThreat: true,
		},
		{
			name: "gun projectile on bot",
			role: gamedata.DotaGun,
			configure: func(m *mobState, now float64, botID int32) {
				m.projLaunchAt, m.projTarget = now+1, botID
			},
			wantThreat: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
			defer cleanup()
			structure := structOfSide(inst, tc.role, dotaTeamElf)
			if structure == nil || structure.mob.AttackRange <= 0 {
				t.Fatalf("setup: no active %v structure", tc.role)
			}
			now := float64(s.battleTime())
			tc.configure(structure, now, bot.objID)

			bot.lock()
			defer bot.unlock()
			got := s.botCommittedStructureFocusLocked(bot, now)
			if (got != nil) != tc.wantThreat {
				t.Fatalf("committed structure focus = %v, want threat=%v", got, tc.wantThreat)
			}
			if tc.wantThreat && got != structure {
				t.Fatalf("committed structure focus = %v, want structure %d", got, structure.id)
			}
		})
	}
}

// TestBotEscapesCommittedHitscanAndProjectileFocus verifies the full healthy-bot
// escape lifecycle for both active shot representations: cancel current attacks and
// orders, move outside structure reach, hold that choice for two seconds without
// entering the low-HP retreat state, then resume ordinary decisions after the shot
// commitment expires.
func TestBotEscapesCommittedHitscanAndProjectileFocus(t *testing.T) {
	tests := []struct {
		name      string
		role      gamedata.DotaRole
		configure func(*mobState, float64, int32)
	}{
		{
			name: "hitscan tower",
			role: gamedata.DotaCreepTower,
			configure: func(m *mobState, now float64, botID int32) {
				m.hitAt, m.hitTarget = now+0.4, botID
			},
		},
		{
			name: "projectile gun",
			role: gamedata.DotaGun,
			configure: func(m *mobState, now float64, botID int32) {
				m.projLaunchAt, m.projTarget = now+0.4, botID
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
			defer cleanup()
			structure := structOfSide(inst, tc.role, dotaTeamElf)
			if structure == nil || structure.mob.AttackRange <= 0 {
				t.Fatalf("setup: no active %v structure", tc.role)
			}
			now := float64(s.battleTime())
			bot.x, bot.y, bot.snapT = 0, 0, float32(now)
			structure.x, structure.y = 10, 0
			tc.configure(structure, now, bot.objID)

			// Seed all three order forms the escape path must cancel. The ids do not
			// need live targets: the cancellation boundary is the bot's own state.
			bot.huntState.attackTarget = 61001
			bot.huntState.pvpTarget = 61002
			bot.huntState.order = &pendingCast{slot: 1}
			b := &botBrain{c: bot, lane: 0}

			bot.lock()
			s.botTickLocked(b, now)
			if got := s.botCommittedStructureFocusLocked(bot, now); got != structure {
				t.Fatalf("setup focus = %v, want structure %d", got, structure.id)
			}
			if bot.huntState.attackTarget != 0 || bot.huntState.pvpTarget != 0 || bot.huntState.order != nil {
				t.Fatalf("committed focus left attack/order state armed: attack=%d pvp=%d order=%v",
					bot.huntState.attackTarget, bot.huntState.pvpTarget, bot.huntState.order)
			}
			if !bot.hasDest {
				t.Fatal("committed focus did not issue a safe movement destination")
			}
			if s.botEnemyStructureDangerLocked(bot, bot.destX, bot.destY) {
				t.Fatalf("escape destination (%.1f,%.1f) remains inside enemy structure danger radius", bot.destX, bot.destY)
			}
			if b.retreating {
				t.Fatal("healthy committed-structure escape incorrectly entered fountain retreat")
			}
			if got, want := b.structureAvoidUntil, now+botStructureAvoidHold; math.Abs(got-want) > 0.01 {
				t.Fatalf("structure hold expires at %.3f, want %.3f", got, want)
			}
			if b.structureAvoidTarget != structure.id {
				t.Fatalf("structure hold target = %d, want %d", b.structureAvoidTarget, structure.id)
			}

			// The committed shot has now resolved. During the remaining hold the bot
			// must keep the tactical escape, not re-enter its normal lane logic.
			s.botTickLocked(b, now+1)
			if b.structureAvoidTarget != structure.id || !bot.hasDest {
				t.Fatal("bot resumed normal decisions before the two-second structure hold expired")
			}

			s.botTickLocked(b, now+botStructureAvoidHold+0.1)
			if b.structureAvoidTarget != 0 {
				t.Fatalf("structure hold target = %d after expiry, want cleared", b.structureAvoidTarget)
			}
			if b.nextThinkAt <= now+botStructureAvoidHold {
				t.Fatalf("normal bot decisions did not resume after structure hold: nextThinkAt=%.3f", b.nextThinkAt)
			}
			bot.unlock()
		})
	}
}
