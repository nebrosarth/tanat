package battleserver

import (
	"math"
	"strings"
	"testing"

	"tanatserver/internal/gamedata"
)

func newTeleportTestBot(t *testing.T) (*Server, *conn, *huntInstance, *botBrain, gamedata.Item, float64, func()) {
	t.Helper()
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	bot.objID = botIDBase + 99
	bot.selfPlayerID = bot.objID
	s.Store.CreateBotHero(bot.selfPlayerID, "teleport-test-bot")
	bot.nav = nil // Keep path length deterministic; target safety is tested separately.
	bot.huntState.team = dotaTeamHuman
	bot.huntState.respawnX, bot.huntState.respawnY = botHomeLocked(bot)
	bot.x, bot.y = bot.huntState.respawnX, bot.huntState.respawnY
	bot.vx, bot.vy = 0, 0
	bot.snapT = s.battleTime()

	bot.lock()
	s.seedBotTeleportScrollLocked(bot)
	bot.unlock()

	it, ok := botTeleportScroll()
	if !ok {
		cleanup()
		t.Fatal("setup: authored teleport scroll is missing or duplicated")
	}
	b := &botBrain{c: bot, lane: 0, phase: botPhaseLane}
	return s, bot, inst, b, it, float64(s.battleTime()), cleanup
}

func teleportTestCreep(inst *huntInstance, id int32, team int32, x, y float32) *mobState {
	lane := inst.dota.m.Lanes[0]
	idx := inst.dota.m.HumanCreepMelee
	if team == dotaTeamElf {
		idx = inst.dota.m.ElfCreepMelee
	}
	mob := gamedata.Mobs()[idx]
	return &mobState{
		id: id, mobIdx: idx, mob: mob, x: x, y: y,
		hp: mob.Health, maxHP: mob.Health, team: team,
		lane: lane, laneIdx: len(lane) / 2, active: true, shown: true,
	}
}

func setTeleportTestMobs(inst *huntInstance, bot *conn, mobs ...*mobState) {
	set := make(map[int32]*mobState, len(mobs))
	for _, m := range mobs {
		set[m.id] = m
	}
	inst.mobs = set
	inst.dota.instMobs = set
	bot.huntState.mobs = set
}

func TestBotTeleportScrollAuthoredTierOneDefinition(t *testing.T) {
	it, ok := botTeleportScroll()
	if !ok {
		t.Fatal("botTeleportScroll did not resolve exactly one item")
	}
	if it.Tier != 1 {
		t.Fatalf("tier = %d, want 1", it.Tier)
	}
	if it.NameKey != "IDS_Scroll_Teleport_Grey_Name" || it.DescKey != "IDS_Scroll_Teleport_Grey_LongDesc" {
		t.Fatalf("locale keys = %q/%q, want original grey scroll keys", it.NameKey, it.DescKey)
	}
	if it.Icon != "neutral/potion/scroll_teleport_grey" {
		t.Fatalf("icon = %q, want original grey teleport scroll icon", it.Icon)
	}
	if it.Preparation != 10 || it.Cooldown != 80 {
		t.Fatalf("timing = preparation %.1f/cooldown %.1f, want 10/80", it.Preparation, it.Cooldown)
	}
	proto := itemProtoDesc(it)
	for _, want := range []string{it.NameKey, it.DescKey, it.Icon, "<Article value=\"" + itoa(int(it.ArticleID)) + "\"/>"} {
		if !strings.Contains(proto, want) {
			t.Errorf("item prototype missing %q: %s", want, proto)
		}
	}
}

func TestBotTeleportScrollSeedsTenBattleLocalCharges(t *testing.T) {
	s, bot, _, _, it, _, cleanup := newTeleportTestBot(t)
	defer cleanup()

	if got := s.Store.HeroBag(bot.selfPlayerID); len(got) != 0 {
		t.Fatalf("session.Store hero bag = %#v, want no teleport scroll", got)
	}
	bot.lock()
	defer bot.unlock()
	if got := bot.huntState.bag[it.ArticleID]; got != botTeleportCharges {
		t.Fatalf("battle-local scroll count = %d, want %d", got, botTeleportCharges)
	}
	if got := bot.huntState.bagItemID[it.ArticleID]; got == 0 {
		t.Fatal("battle-local scroll has no inventory wire id")
	}
	if len(bot.huntState.bag) != 1 {
		t.Fatalf("seeded battle bag has %d articles, want exactly one scroll article", len(bot.huntState.bag))
	}
}

func TestBotTeleportTargetEligibility(t *testing.T) {
	s, bot, inst, b, _, _, cleanup := newTeleportTestBot(t)
	defer cleanup()
	_ = s

	bot.lock()
	defer bot.unlock()
	validCreep := teleportTestCreep(inst, 61001, bot.playerTeam(), 0, 0)
	var bakedStructure *mobState
	for _, m := range inst.mobs {
		if m.structure && m.team == bot.playerTeam() {
			bakedStructure = m
			break
		}
	}
	if bakedStructure == nil {
		t.Fatal("setup: no baked same-team Dota structure")
	}
	if bakedStructure.dotaPrefab == "" {
		t.Fatalf("setup: seeded Dota structure %d has empty prefab", bakedStructure.id)
	}
	tests := []struct {
		name string
		mob  *mobState
		want bool
	}{
		{name: "living same-team Dota lane creep", mob: validCreep, want: true},
		{name: "baked same-team structure", mob: bakedStructure, want: true},
		{name: "enemy creep", mob: func() *mobState { m := *validCreep; m.team = dotaTeamElf; return &m }(), want: false},
		{name: "dead allied creep", mob: func() *mobState { m := *validCreep; m.dead = true; return &m }(), want: false},
		{name: "generic allied mob", mob: &mobState{id: 61002, mob: gamedata.Mobs()[0], team: bot.playerTeam(), hp: 100, maxHP: 100}, want: false},
		{name: "stationary boss", mob: &mobState{id: 61003, structure: true, boss: true, team: bot.playerTeam(), hp: 100, maxHP: 100}, want: false},
		{name: "reserved-range structure without baked prefab", mob: &mobState{id: dotaStructIDBase + 999, structure: true, team: bot.playerTeam(), hp: 100, maxHP: 100, dotaPrefab: ""}, want: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := s.botTeleportTargetValidLocked(b, tc.mob); got != tc.want {
				t.Fatalf("eligible = %v, want %v for %+v", got, tc.want, tc.mob)
			}
		})
	}
}

func TestMobilizationPreparationDoesNotTeleportHealthyBotAwayFromRally(t *testing.T) {
	s, bot, inst, b, it, now, cleanup := newTeleportTestBot(t)
	defer cleanup()
	b.macroAssignment = botMacroAssignment{Mode: botMacroPush, Reason: botMacroReasonMobilizationPreparation}
	target := teleportTestCreep(inst, 61501, bot.playerTeam(), bot.x+240, bot.y)
	installTeleportTarget(inst, bot, target)

	bot.lock()
	started := s.botMaybeStartTeleportLocked(b, now)
	bot.unlock()
	if started || b.pendingTeleport != nil {
		t.Fatal("healthy preparation bot started a non-rally teleport")
	}
	if got := bot.huntState.bag[it.ArticleID]; got != botTeleportCharges {
		t.Fatalf("preparation teleport changed scroll count: got %d want %d", got, botTeleportCharges)
	}
}

func installTeleportTarget(inst *huntInstance, bot *conn, target *mobState, extras ...*mobState) {
	setTeleportTestMobs(inst, bot, append([]*mobState{target}, extras...)...)
}

func TestBotTeleportRejectsShortAndUnsafeDestinations(t *testing.T) {
	t.Run("short travel", func(t *testing.T) {
		s, bot, inst, b, it, now, cleanup := newTeleportTestBot(t)
		defer cleanup()
		bot.lock()
		defer bot.unlock()
		target := teleportTestCreep(inst, 62001, bot.playerTeam(), bot.x+1, bot.y)
		installTeleportTarget(inst, bot, target)
		if s.botMaybeStartTeleportLocked(b, now) {
			t.Fatal("started teleport for a materially short walk")
		}
		if b.pendingTeleport != nil || bot.huntState.bag[it.ArticleID] != botTeleportCharges {
			t.Fatalf("short travel changed channel state: pending=%v charges=%d", b.pendingTeleport, bot.huntState.bag[it.ArticleID])
		}
	})

	t.Run("unsafe destination", func(t *testing.T) {
		s, bot, inst, b, it, now, cleanup := newTeleportTestBot(t)
		defer cleanup()
		bot.lock()
		defer bot.unlock()
		target := teleportTestCreep(inst, 62002, bot.playerTeam(), 0, 0)
		enemies := []*mobState{
			{id: 62003, mob: gamedata.Mobs()[0], team: dotaTeamElf, x: 0, y: 0, hp: 100, maxHP: 100},
			{id: 62004, mob: gamedata.Mobs()[0], team: dotaTeamElf, x: 0, y: 0, hp: 100, maxHP: 100},
			{id: 62005, mob: gamedata.Mobs()[0], team: dotaTeamElf, x: 0, y: 0, hp: 100, maxHP: 100},
		}
		installTeleportTarget(inst, bot, target, enemies...)
		if _, _, ok := s.botTeleportDestinationLocked(bot, target); ok {
			t.Fatal("unsafe destination was accepted")
		}
		if s.botMaybeStartTeleportLocked(b, now) {
			t.Fatal("started teleport to an unsafe destination")
		}
		if b.pendingTeleport != nil || bot.huntState.bag[it.ArticleID] != botTeleportCharges {
			t.Fatalf("unsafe target changed channel state: pending=%v charges=%d", b.pendingTeleport, bot.huntState.bag[it.ArticleID])
		}
	})
}

func startTeleportTestChannel(t *testing.T) (*Server, *conn, *huntInstance, *botBrain, *mobState, gamedata.Item, float64, func()) {
	s, bot, inst, b, it, now, cleanup := newTeleportTestBot(t)
	target := teleportTestCreep(inst, 63001, bot.playerTeam(), 0, 0)
	installTeleportTarget(inst, bot, target)
	bot.lock()
	started := s.botMaybeStartTeleportLocked(b, now)
	bot.unlock()
	if !started {
		cleanup()
		t.Fatal("setup: teleport channel did not start")
	}
	return s, bot, inst, b, target, it, now, cleanup
}

func TestBotTeleportChannelBlocksActionsUntilPreparationCompletes(t *testing.T) {
	s, bot, _, b, target, it, now, cleanup := startTeleportTestChannel(t)
	defer cleanup()
	bot.lock()
	defer bot.unlock()
	homeX, homeY := botHomeLocked(bot)
	if got := bot.huntState.bag[it.ArticleID]; got != botTeleportCharges-1 {
		t.Fatalf("channel start charges = %d, want %d", got, botTeleportCharges-1)
	}
	if got := bot.huntState.itemCooldownUntil[it.ArticleID]; got != now+it.Cooldown {
		t.Fatalf("cooldown = %.3f, want %.3f", got, now+it.Cooldown)
	}
	if got := b.pendingTeleport.complete; got != now+it.Preparation {
		t.Fatalf("completion = %.3f, want %.3f", got, now+it.Preparation)
	}

	bot.huntState.attackTarget = target.id
	bot.huntState.pvpTarget = 12345
	bot.huntState.order = &pendingCast{slot: 1, target: target.id}
	bot.hasDest = true
	bot.vx, bot.vy = 0, 0
	s.botTickTeleportLocked(b, now+it.Preparation-0.01)
	if b.pendingTeleport == nil {
		t.Fatal("preparation tick completed the channel early")
	}
	if bot.huntState.attackTarget != 0 || bot.huntState.pvpTarget != 0 || bot.huntState.order != nil || bot.hasDest || bot.vx != 0 || bot.vy != 0 {
		t.Fatalf("channel did not suppress actions: attack=%d pvp=%d order=%v hasDest=%v velocity=(%.1f,%.1f)",
			bot.huntState.attackTarget, bot.huntState.pvpTarget, bot.huntState.order, bot.hasDest, bot.vx, bot.vy)
	}
	if bot.x != homeX || bot.y != homeY {
		t.Fatalf("bot relocated during preparation: (%.2f,%.2f), want home (%.2f,%.2f)", bot.x, bot.y, homeX, homeY)
	}

	s.botTickTeleportLocked(b, now+it.Preparation)
	if b.pendingTeleport != nil {
		t.Fatal("successful channel remained pending")
	}
	distance := math.Hypot(float64(bot.x-target.x), float64(bot.y-target.y))
	need := target.mob.Radius() + bot.huntState.av.Radius()
	if distance <= need {
		t.Fatalf("destination is inside target: distance %.3f, body radii %.3f", distance, need)
	}
	wantGap := need + 0.75
	if math.Abs(distance-wantGap) > 0.001 {
		t.Fatalf("destination gap = %.3f, want %.3f", distance, wantGap)
	}
	if bot.huntState.bag[it.ArticleID] != botTeleportCharges-1 {
		t.Fatalf("successful channel consumed more than once: charges=%d", bot.huntState.bag[it.ArticleID])
	}
	s.botTickTeleportLocked(b, now+it.Preparation+1)
	if bot.huntState.bag[it.ArticleID] != botTeleportCharges-1 {
		t.Fatalf("post-success tick consumed again: charges=%d", bot.huntState.bag[it.ArticleID])
	}
}

func TestBotTeleportCancellationDoesNotRefund(t *testing.T) {
	for _, tc := range []struct {
		name   string
		cancel func(*conn, *mobState, float64)
	}{
		{name: "invalid target", cancel: func(_ *conn, target *mobState, _ float64) { target.dead = true }},
		{name: "bot death", cancel: func(bot *conn, _ *mobState, now float64) { bot.huntState.deadUntil = now + 1 }},
		{name: "bot stun", cancel: func(bot *conn, _ *mobState, now float64) { bot.huntState.st.stunUntil = now + 1 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, bot, _, b, target, it, now, cleanup := startTeleportTestChannel(t)
			defer cleanup()
			bot.lock()
			defer bot.unlock()
			tc.cancel(bot, target, now)
			s.botTickTeleportLocked(b, now+0.5)
			if b.pendingTeleport != nil {
				t.Fatal("cancelled channel remained pending")
			}
			if got := bot.huntState.bag[it.ArticleID]; got != botTeleportCharges-1 {
				t.Fatalf("cancellation refunded/changed charges: got %d, want %d", got, botTeleportCharges-1)
			}
			if got := bot.huntState.itemCooldownUntil[it.ArticleID]; got != now+it.Cooldown {
				t.Fatalf("cancellation changed cooldown: got %.3f, want %.3f", got, now+it.Cooldown)
			}
		})
	}
}

func TestRecoveryTeleportCommitsBelowHardRetreatFloor(t *testing.T) {
	s, bot, _, b, target, _, now, cleanup := startTeleportTestChannel(t)
	defer cleanup()

	bot.lock()
	defer bot.unlock()
	b.retreating = true
	b.retreatMode = botRetreatModeRecovery
	b.pendingTeleport.targetKind = "recovery_structure"
	// Make the in-flight channel look like a real emergency recovery: the
	// current sample is below the hard floor and the previous sample records a
	// fresh hit. The target itself remains valid, so only the emergency rule is
	// being exercised.
	bot.huntState.hp = bot.huntState.maxHPLocked(now+0.5) * (botRetreatHPFrac - 0.05)
	b.hpHistIdx = 0
	b.hpHistory[0] = hpSample{t: now, frac: botRetreatHPFrac + 0.20}
	s.botTickTeleportLocked(b, now+0.5)
	if b.pendingTeleport == nil {
		t.Fatalf("emergency recovery channel was cancelled under damage; target=%d", target.id)
	}
}

func TestRecoveryTeleportStartsBelowHardRetreatFloorDuringRecentCreepDamage(t *testing.T) {
	s, bot, _, b, it, now, cleanup := recoveryTeleportAcceptanceFixture(t)
	defer cleanup()

	bot.lock()
	defer bot.unlock()
	maxHP := bot.huntState.maxHPLocked(now)
	bot.huntState.hp = maxHP * (botRetreatHPFrac - 0.05)
	// Simulate a recent creep hit. The emergency floor is allowed to override
	// this one condition, but the ordinary hero-contact and enemy-structure
	// safety gates remain active.
	b.hpHistIdx = 0
	b.hpHistory[0] = hpSample{t: now - 0.5, frac: 0.55}
	if s.botRecoveryTeleportDamageEndedLocked(b, now) {
		t.Fatal("setup: recent damage was not visible to recovery teleport gate")
	}
	if !s.botMaybeStartTeleportLocked(b, now) {
		t.Fatal("critical recovery bot did not start an escape channel during recent creep damage")
	}
	if b.pendingTeleport == nil || b.pendingTeleport.targetKind != "recovery_structure" {
		t.Fatalf("pending recovery channel = %+v, want recovery_structure", b.pendingTeleport)
	}
	if got := bot.huntState.bag[it.ArticleID]; got != botTeleportCharges-1 {
		t.Fatalf("recovery channel charges = %d, want %d", got, botTeleportCharges-1)
	}
}

func TestBotTeleportTacticalEligibility(t *testing.T) {
	t.Run("empty structure is rejected at any match time", func(t *testing.T) {
		for _, tc := range []struct {
			name      string
			startedAt float64
			now       float64
		}{
			{name: "match start", startedAt: 0, now: 0},
			{name: "much later", startedAt: 1000, now: 2000},
		} {
			t.Run(tc.name, func(t *testing.T) {
				s, bot, inst, b, it, _, cleanup := newTeleportTestBot(t)
				defer cleanup()
				inst.dota.startedAt = tc.startedAt
				target := bakedBotTeleportCannon(t, inst, bot)
				installTeleportTarget(inst, bot, target)

				bot.lock()
				defer bot.unlock()
				if s.botMaybeStartTeleportLocked(b, tc.now) {
					t.Fatal("teleport started to an empty allied structure")
				}
				if got := bot.huntState.bag[it.ArticleID]; got != botTeleportCharges {
					t.Fatalf("empty-structure rejection consumed charges: got %d, want %d", got, botTeleportCharges)
				}
				if got := bot.huntState.itemCooldownUntil[it.ArticleID]; got != 0 {
					t.Fatalf("empty-structure rejection set cooldown %.3f", got)
				}
			})
		}
	})

	t.Run("advanced creep is accepted immediately", func(t *testing.T) {
		s, bot, inst, b, it, now, cleanup := newTeleportTestBot(t)
		defer cleanup()
		inst.dota.startedAt = now
		target := teleportTestCreep(inst, 65001, bot.playerTeam(), 0, 0)
		installTeleportTarget(inst, bot, target)

		bot.lock()
		defer bot.unlock()
		if !s.botMaybeStartTeleportLocked(b, now) {
			t.Fatal("materially advanced allied creep was rejected at match time zero")
		}
		if got := bot.huntState.bag[it.ArticleID]; got != botTeleportCharges-1 {
			t.Fatalf("immediate creep teleport charges = %d, want %d", got, botTeleportCharges-1)
		}
	})

	t.Run("structure usefulness follows the 24-unit allied-wave radius", func(t *testing.T) {
		s, bot, inst, b, _, now, cleanup := newTeleportTestBot(t)
		defer cleanup()
		structure := bakedBotTeleportCannon(t, inst, bot)
		bot.lock()
		defer bot.unlock()
		destX, destY, ok := s.botTeleportDestinationLocked(bot, structure)
		if !ok {
			t.Fatal("setup: structure destination is unsafe")
		}
		for _, tc := range []struct {
			name string
			dist float32
			want bool
		}{
			{name: "within radius", dist: botTeleportStructureSupportRadius - 0.1, want: true},
			{name: "beyond radius", dist: botTeleportStructureSupportRadius + 0.1, want: false},
		} {
			t.Run(tc.name, func(t *testing.T) {
				wave := teleportTestCreep(inst, 65002, bot.playerTeam(), structure.x+tc.dist, structure.y)
				installTeleportTarget(inst, bot, structure, wave)
				if got := s.botTeleportStructureUsefulLocked(b, structure, 0, destX, destY, now); got != tc.want {
					t.Fatalf("structure usefulness at %.1f units = %v, want %v", tc.dist, got, tc.want)
				}
			})
		}
	})

	t.Run("enemy lane pressure is accepted until destination becomes unsafe", func(t *testing.T) {
		for _, count := range []int{1, 2, 3} {
			t.Run(itoa(count)+" enemy creeps", func(t *testing.T) {
				s, bot, inst, b, it, now, cleanup := newTeleportTestBot(t)
				defer cleanup()
				structure := bakedBotTeleportCannon(t, inst, bot)
				enemies := make([]*mobState, count)
				for i := range enemies {
					enemies[i] = teleportTestCreep(inst, int32(65010+i), dotaTeamElf, structure.x, structure.y)
				}
				installTeleportTarget(inst, bot, structure, enemies...)

				bot.lock()
				defer bot.unlock()
				started := s.botMaybeStartTeleportLocked(b, now)
				if count < 3 && !started {
					t.Fatal("safe pressured structure was rejected")
				}
				if count == 3 && started {
					t.Fatal("unsafe three-creep pressure was accepted")
				}
				wantCharges := botTeleportCharges
				if count < 3 {
					wantCharges--
				}
				if got := bot.huntState.bag[it.ArticleID]; got != wantCharges {
					t.Fatalf("pressure case charges = %d, want %d", got, wantCharges)
				}
			})
		}
	})

	t.Run("visible enemy hero pressure remains unsafe", func(t *testing.T) {
		s, bot, inst, b, it, now, cleanup := newTeleportTestBot(t)
		defer cleanup()
		structure := bakedBotTeleportCannon(t, inst, bot)
		dotaPlayerConn(t, s, inst, 65020, dotaTeamElf, structure.x, structure.y)
		installTeleportTarget(inst, bot, structure)

		bot.lock()
		defer bot.unlock()
		if s.botMaybeStartTeleportLocked(b, now) {
			t.Fatal("teleport entered a structure area occupied by an enemy hero")
		}
		if got := bot.huntState.bag[it.ArticleID]; got != botTeleportCharges {
			t.Fatalf("enemy-hero rejection consumed charges: got %d, want %d", got, botTeleportCharges)
		}
	})
}

func TestBotTeleportCreepReliabilitySelection(t *testing.T) {
	t.Run("durable trailing creep beats pressured front", func(t *testing.T) {
		s, bot, inst, b, _, now, cleanup := newTeleportTestBot(t)
		defer cleanup()
		front := teleportTestCreep(inst, 65100, bot.playerTeam(), 0, 0)
		trailing := teleportTestCreep(inst, 65101, bot.playerTeam(), -40, 0)
		front.laneIdx += 2
		trailing.laneIdx++
		front.hp = front.maxHealth() * 0.80
		pressure := teleportTestCreep(inst, 65102, dotaTeamElf, front.x, front.y)
		installTeleportTarget(inst, bot, front, trailing, pressure)

		bot.lock()
		target, _, _, _, ok := s.botTeleportTargetLocked(b, now)
		bot.unlock()
		if !ok || target != trailing {
			t.Fatalf("selected target = %v, want durable trailing creep %d", target, trailing.id)
		}
	})

	t.Run("approaching hero outside landing radius rejects creep", func(t *testing.T) {
		s, bot, inst, b, _, now, cleanup := newTeleportTestBot(t)
		defer cleanup()
		target := teleportTestCreep(inst, 65110, bot.playerTeam(), 0, 0)
		installTeleportTarget(inst, bot, target)
		dotaPlayerConn(t, s, inst, 65111, dotaTeamElf, 25, 0)

		bot.lock()
		_, _, _, _, ok := s.botTeleportTargetLocked(b, now)
		bot.unlock()
		if ok {
			t.Fatal("creep inside hero forecast radius remained eligible")
		}
	})

	t.Run("committed distant hero focus rejects creep", func(t *testing.T) {
		s, bot, inst, b, _, now, cleanup := newTeleportTestBot(t)
		defer cleanup()
		target := teleportTestCreep(inst, 65120, bot.playerTeam(), 0, 0)
		installTeleportTarget(inst, bot, target)
		enemy := dotaPlayerConn(t, s, inst, 65121, dotaTeamElf, 100, 100)
		enemy.huntState.attackTarget = target.id

		bot.lock()
		_, _, _, _, ok := s.botTeleportTargetLocked(b, now)
		bot.unlock()
		if ok {
			t.Fatal("creep under committed hero focus remained eligible")
		}
	})

	t.Run("isolated full health advanced creep remains reliable", func(t *testing.T) {
		s, bot, inst, _, _, now, cleanup := newTeleportTestBot(t)
		defer cleanup()
		target := teleportTestCreep(inst, 65130, bot.playerTeam(), 0, 0)
		installTeleportTarget(inst, bot, target)
		bot.lock()
		_, reliable := s.botTeleportCreepReliabilityLocked(bot, target, now)
		bot.unlock()
		if !reliable {
			t.Fatal("isolated uncontested full-health creep was rejected")
		}
	})
}

func TestBotTeleportChannelCancelsWhenCreepBecomesRisky(t *testing.T) {
	s, bot, inst, b, it, now, cleanup := newTeleportTestBot(t)
	defer cleanup()
	target := teleportTestCreep(inst, 65140, bot.playerTeam(), 0, 0)
	installTeleportTarget(inst, bot, target)

	bot.lock()
	if !s.botMaybeStartTeleportLocked(b, now) {
		bot.unlock()
		t.Fatal("setup: safe creep teleport did not start")
	}
	if got := bot.huntState.bag[it.ArticleID]; got != botTeleportCharges-1 {
		bot.unlock()
		t.Fatalf("started channel charges = %d, want %d", got, botTeleportCharges-1)
	}
	bot.unlock()

	dotaPlayerConn(t, s, inst, 65141, dotaTeamElf, target.x+25, target.y)
	rec := &telemetryRecorder{ch: make(chan any, 4)}
	inst.dota.telemetry = rec
	bot.lock()
	s.botTickTeleportLocked(b, now+0.2)
	bot.unlock()
	inst.dota.telemetry = nil

	if b.pendingTeleport != nil {
		t.Fatal("risky creep did not cancel the pending teleport")
	}
	if got := bot.huntState.bag[it.ArticleID]; got != botTeleportCharges-1 {
		t.Fatalf("risk cancellation changed consumed charge: got %d", got)
	}
	select {
	case raw := <-rec.ch:
		ev, ok := raw.(telemetryBotTeleport)
		if !ok || ev.Type != "bot_teleport_cancel" || ev.CancelReason != "target_became_risky" {
			t.Fatalf("cancel telemetry = %#v, want target_became_risky", raw)
		}
	default:
		t.Fatal("risk cancellation emitted no telemetry")
	}
}

func TestBotTeleportStructureFallbackHoldsLandingPointButCreepDoesNot(t *testing.T) {
	for _, tc := range []struct {
		name      string
		structure bool
	}{
		{name: "structure", structure: true},
		{name: "creep", structure: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, bot, inst, b, _, now, cleanup := newTeleportTestBot(t)
			defer cleanup()
			var target *mobState
			if tc.structure {
				target = bakedBotTeleportCannon(t, inst, bot)
			} else {
				target = teleportTestCreep(inst, 65002, bot.playerTeam(), 0, 0)
			}
			if target == nil {
				t.Fatal("setup: no teleport target")
			}
			if tc.structure {
				pressure := teleportTestCreep(inst, 65003, dotaTeamElf, target.x, target.y)
				installTeleportTarget(inst, bot, target, pressure)
			} else {
				installTeleportTarget(inst, bot, target)
			}
			bot.lock()
			if !s.botMaybeStartTeleportLocked(b, now) {
				bot.unlock()
				t.Fatal("teleport did not start")
			}
			complete := b.pendingTeleport.complete
			s.botTickTeleportLocked(b, complete)
			landingX, landingY := bot.x, bot.y
			if tc.structure {
				pressure := inst.mobs[65003]
				pressure.x, pressure.y = landingX+40, landingY
				gotX, gotY := s.botLanePoint(b, complete+1)
				if math.Hypot(float64(gotX-landingX), float64(gotY-landingY)) > 0.01 {
					t.Fatalf("structure fallback released landing hold early: got (%.2f,%.2f), landing (%.2f,%.2f)", gotX, gotY, landingX, landingY)
				}
				wave := teleportTestCreep(inst, 65004, bot.playerTeam(), landingX+5, landingY)
				setTeleportTestMobs(inst, bot, target, pressure, wave)
				gotX, gotY = s.botLanePoint(b, complete+2)
				if math.Hypot(float64(gotX-wave.x), float64(gotY-wave.y)) > 0.01 {
					t.Fatalf("structure fallback did not release when wave caught up: got (%.2f,%.2f), wave (%.2f,%.2f)", gotX, gotY, wave.x, wave.y)
				}
				b.laneRedeployPointValid = true
				b.laneRedeployUntil = complete + botTeleportStructureHold
				gotX, gotY = s.botLanePoint(b, b.laneRedeployUntil+0.01)
				if math.Hypot(float64(gotX-wave.x), float64(gotY-wave.y)) > 0.01 {
					t.Fatalf("structure fallback timeout did not release hold: got (%.2f,%.2f), wave (%.2f,%.2f)", gotX, gotY, wave.x, wave.y)
				}
			} else if b.laneRedeployPointValid {
				t.Fatal("creep teleport incorrectly latched a lane redeploy hold")
			}
			bot.unlock()
		})
	}
}
