package battleserver

import (
	"math"
	"testing"

	"tanatserver/internal/gamedata"
)

func recoveryTeleportAcceptanceFixture(t *testing.T) (*Server, *conn, *huntInstance, *botBrain, gamedata.Item, float64, func()) {
	t.Helper()
	s, bot, inst, b, it, now, cleanup := newTeleportTestBot(t)
	b.retreating = true
	b.retreatMode = botRetreatModeRecovery
	now = float64(s.battleTime()) + 10
	// The centre of the north lane is safely away from the fountain and leaves enough
	// walking distance for the tactical savings gate to be meaningful.
	p := inst.dota.m.Lanes[0][4]
	bot.x, bot.y, bot.vx, bot.vy, bot.snapT = float32(p.X), float32(p.Y), 0, 0, float32(now)
	return s, bot, inst, b, it, now, cleanup
}

func TestBotDisengageLifecycleReleasesAfterDamageTrendEndsNearHero(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	b := &botBrain{c: bot, retreating: true, retreatMode: botRetreatModeDisengage}
	now := float64(s.battleTime()) + 10
	b.retreatHoldUntil = now
	bot.huntState.hp = bot.huntState.maxHPLocked(now) * 0.60
	nearby := dotaPlayerConn(t, s, inst, 66080, dotaTeamElf, bot.x+4, bot.y)
	nearby.huntState.pvpTarget = bot.objID

	bot.lock()
	got := s.botShouldRetreatLocked(b, now+botDisengageMinHold+0.1)
	bot.unlock()
	if got || b.retreating {
		t.Fatalf("safe-HP disengage did not release after hold with ended damage trend: got=%v retreating=%v", got, b.retreating)
	}
}

func TestBotDisengageLifecycleKeepsBelowSafeHPAndEscalatesAtCriticalHP(t *testing.T) {
	for _, tc := range []struct {
		name string
		hp   float64
		mode botRetreatMode
	}{
		{name: "below safe remains disengage", hp: 0.50, mode: botRetreatModeDisengage},
		{name: "critical escalates recovery", hp: botRetreatHPFrac, mode: botRetreatModeRecovery},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, bot, _, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
			defer cleanup()
			now := float64(s.battleTime()) + 10
			b := &botBrain{c: bot, retreating: true, retreatMode: botRetreatModeDisengage, retreatHoldUntil: now}
			bot.huntState.hp = bot.huntState.maxHPLocked(now) * tc.hp

			bot.lock()
			got := s.botShouldRetreatLocked(b, now+botDisengageMinHold+0.1)
			bot.unlock()
			if !got || !b.retreating || b.retreatMode != tc.mode {
				t.Fatalf("retreat state = got=%v active=%v mode=%d, want active mode=%d", got, b.retreating, b.retreatMode, tc.mode)
			}
		})
	}
}

func TestBotRecoveryTeleportSelectsLivingOwnAltarAndConsumesOne(t *testing.T) {
	t.Run("living own altar", func(t *testing.T) {
		s, bot, inst, b, it, now, cleanup := recoveryTeleportAcceptanceFixture(t)
		defer cleanup()
		own := altarOf(inst, bot.playerTeam())
		if own == nil {
			t.Fatal("setup: missing own altar")
		}
		bot.lock()
		started := s.botMaybeStartTeleportLocked(b, now)
		bot.unlock()
		if !started {
			t.Fatal("safe recovery bot did not start a materially useful altar recall")
		}
		if b.pendingTeleport == nil || b.pendingTeleport.target != own.id || b.pendingTeleport.targetKind != "recovery_structure" {
			t.Fatalf("pending recovery recall = %+v, want own living altar %d", b.pendingTeleport, own.id)
		}
		if got := bot.huntState.bag[it.ArticleID]; got != botTeleportCharges-1 {
			t.Fatalf("recovery recall charges = %d, want exactly one consumed from %d", got, botTeleportCharges)
		}
	})

	t.Run("dead own altar is rejected", func(t *testing.T) {
		s, bot, inst, b, it, now, cleanup := recoveryTeleportAcceptanceFixture(t)
		defer cleanup()
		own := altarOf(inst, bot.playerTeam())
		if own == nil {
			t.Fatal("setup: missing own altar")
		}
		own.dead = true
		bot.lock()
		started := s.botMaybeStartTeleportLocked(b, now)
		bot.unlock()
		if started || b.pendingTeleport != nil {
			t.Fatalf("recovery recall selected dead own altar: started=%v pending=%+v", started, b.pendingTeleport)
		}
		if got := bot.huntState.bag[it.ArticleID]; got != botTeleportCharges {
			t.Fatalf("dead-altar rejection changed charges: got %d, want %d", got, botTeleportCharges)
		}
	})
}

func TestBotRecoveryTeleportRejectsUnsafeOrInsufficientStarts(t *testing.T) {
	tests := []struct {
		name  string
		setup func(*Server, *conn, *huntInstance, *botBrain, float64)
	}{
		{
			name: "recent damage",
			setup: func(_ *Server, bot *conn, _ *huntInstance, b *botBrain, now float64) {
				bot.huntState.hp = bot.huntState.maxHPLocked(now) * 0.50
				b.hpHistory[0] = hpSample{t: now - 0.5, frac: 0.80}
			},
		},
		{
			name: "enemy hero at origin",
			setup: func(s *Server, bot *conn, inst *huntInstance, _ *botBrain, _ float64) {
				enemy := dotaPlayerConn(t, s, inst, 66081, dotaTeamElf, bot.x+4, bot.y)
				enemy.huntState.pvpTarget = bot.objID
			},
		},
		{
			name: "enemy structure danger at origin",
			setup: func(_ *Server, bot *conn, inst *huntInstance, _ *botBrain, _ float64) {
				threat := structOfSide(inst, gamedata.DotaCreepTower, dotaTeamElf)
				if threat == nil {
					t.Fatalf("setup: missing enemy tower")
				}
				threat.x, threat.y = bot.x, bot.y
			},
		},
		{
			name: "insufficient time savings",
			setup: func(_ *Server, bot *conn, _ *huntInstance, _ *botBrain, _ float64) {
				hx, hy := botHomeLocked(bot)
				bot.x, bot.y = hx+9, hy
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s, bot, inst, b, it, now, cleanup := recoveryTeleportAcceptanceFixture(t)
			defer cleanup()
			tc.setup(s, bot, inst, b, now)
			bot.lock()
			started := s.botMaybeStartTeleportLocked(b, now)
			bot.unlock()
			if started || b.pendingTeleport != nil {
				t.Fatalf("unsafe recovery case started channel: started=%v pending=%+v", started, b.pendingTeleport)
			}
			if got := bot.huntState.bag[it.ArticleID]; got != botTeleportCharges {
				t.Fatalf("rejected recovery case changed charges: got %d, want %d", got, botTeleportCharges)
			}
		})
	}
}

func TestBotRecoveryTeleportCancelsOnNewDamageOrHeroWithoutRefundOrArrival(t *testing.T) {
	for _, tc := range []struct {
		name string
		arm  func(*Server, *conn, *huntInstance, *botBrain, float64)
	}{
		{
			name: "new damage",
			arm: func(_ *Server, bot *conn, _ *huntInstance, b *botBrain, now float64) {
				bot.huntState.hp = bot.huntState.maxHPLocked(now+0.3) * 0.50
				b.hpHistory[0] = hpSample{t: now - 0.2, frac: 0.80}
			},
		},
		{
			name: "hero pressure at origin",
			arm: func(s *Server, bot *conn, inst *huntInstance, _ *botBrain, _ float64) {
				enemy := dotaPlayerConn(t, s, inst, 66082, dotaTeamElf, bot.x+4, bot.y)
				enemy.huntState.attackTarget = bot.objID
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, bot, inst, b, it, now, cleanup := recoveryTeleportAcceptanceFixture(t)
			defer cleanup()
			bot.lock()
			if !s.botMaybeStartTeleportLocked(b, now) {
				bot.unlock()
				t.Fatal("setup: recovery channel did not start")
			}
			originX, originY := bot.x, bot.y
			bot.unlock()
			tc.arm(s, bot, inst, b, now)

			bot.lock()
			s.botTickTeleportLocked(b, now+0.3)
			bot.unlock()
			if b.pendingTeleport != nil {
				t.Fatal("recovery channel remained pending after renewed risk")
			}
			if got := bot.huntState.bag[it.ArticleID]; got != botTeleportCharges-1 {
				t.Fatalf("cancellation refunded/changed charge: got %d, want %d", got, botTeleportCharges-1)
			}
			hx, hy := botHomeLocked(bot)
			if math.Hypot(float64(bot.x-hx), float64(bot.y-hy)) < 20 {
				t.Fatalf("cancelled recovery channel arrived or moved home: position=(%.1f,%.1f) home=(%.1f,%.1f) origin=(%.1f,%.1f)", bot.x, bot.y, hx, hy, originX, originY)
			}
		})
	}
}

func TestBotRecoveryTeleportArrivalKeepsRecoveryAndTargetsFountainWithoutLaneHold(t *testing.T) {
	s, bot, inst, b, _, now, cleanup := recoveryTeleportAcceptanceFixture(t)
	defer cleanup()
	own := altarOf(inst, bot.playerTeam())
	if own == nil {
		t.Fatal("setup: missing own altar")
	}

	bot.lock()
	if !s.botMaybeStartTeleportLocked(b, now) {
		bot.unlock()
		t.Fatal("setup: recovery channel did not start")
	}
	complete := b.pendingTeleport.complete
	s.botTickTeleportLocked(b, complete)
	bot.unlock()
	if b.pendingTeleport != nil {
		t.Fatal("successful recovery channel remained pending")
	}
	if b.retreatMode != botRetreatModeRecovery || !b.retreating {
		t.Fatalf("successful recovery recall changed retreat state: retreating=%v mode=%d", b.retreating, b.retreatMode)
	}
	if b.laneRedeployPointValid {
		t.Fatal("recovery altar recall created an outbound laneRedeploy hold")
	}
	if math.Hypot(float64(bot.x-own.x), float64(bot.y-own.y)) > float64(own.mob.Radius()+bot.huntState.av.Radius()+2) {
		t.Fatalf("arrival is not at own altar: bot=(%.1f,%.1f) altar=(%.1f,%.1f)", bot.x, bot.y, own.x, own.y)
	}
	hx, hy := botHomeLocked(bot)
	rx, ry := s.botRetreatPointLocked(b, complete+1)
	if rx != hx || ry != hy {
		t.Fatalf("post-recall recovery target=(%.1f,%.1f), want fountain=(%.1f,%.1f)", rx, ry, hx, hy)
	}
}

func TestBotStrategicCentroidsExcludeRetreatingBotsButHealingSeesThem(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Sp_Arianna")
	defer cleanup()
	now := float64(s.battleTime()) + 200
	bot.x, bot.y, bot.snapT = 33, 185, float32(now)
	strategic := dotaPlayerConn(t, s, inst, 66083, dotaTeamHuman, 33, 185)
	retreating := dotaPlayerConn(t, s, inst, 66084, dotaTeamHuman, 27.8, -201.8)
	strategicBrain := &botBrain{c: strategic}
	retreatingBrain := &botBrain{c: retreating, retreating: true, retreatMode: botRetreatModeRecovery}
	inst.bots[bot.objID] = &botBrain{c: bot}
	inst.bots[strategic.objID] = strategicBrain
	inst.bots[retreating.objID] = retreatingBrain

	bot.lock()
	s.botUpdatePhaseLocked(inst.bots[bot.objID], now)
	push := s.botPushLaneLocked(inst.bots[bot.objID], now)
	redirect := s.botRedirectLaneLocked(inst.bots[bot.objID], now)
	bot.unlock()
	if inst.bots[bot.objID].phase == botPhaseGroup {
		t.Fatalf("phase counted retreating teammate: phase=%d, want roam or lane with only one strategic ally", inst.bots[bot.objID].phase)
	}
	if push != 0 || redirect != 0 {
		t.Fatalf("strategic centroids included retreating bot: push lane=%d redirect lane=%d, want north lane 0", push, redirect)
	}

	retreating.huntState.hp = 1
	retreating.x, retreating.y, retreating.snapT = bot.x+2, bot.y, float32(now)
	bot.huntState.hp = bot.huntState.maxHPLocked(now)
	healSlot := 0
	for slot := 1; slot <= 4; slot++ {
		def := bot.huntState.skillDef(slot)
		if skillHasTargetFlag(def, "FRIEND") &&
			(botSkillHasOp(def, gamedata.OpHeal) || botSkillHasOp(def, gamedata.OpHot) || botSkillHasOp(def, gamedata.OpShield)) {
			healSlot = slot
			break
		}
	}
	if healSlot == 0 {
		t.Fatal("setup: Arianna has no FRIEND heal/hot/shield skill")
	}
	bot.huntState.skillLevel[healSlot-1] = 1
	bot.huntState.mana = bot.huntState.maxManaLocked(now)
	bot.lock()
	worst, slot := s.botFindHealTargetLocked(inst.bots[bot.objID], now)
	bot.unlock()
	if worst != retreating || slot == 0 {
		t.Fatalf("healing logic ignored retreating ally: target=%v slot=%d, want ally=%d and a heal slot", worst, slot, retreating.objID)
	}
}
