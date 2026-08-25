package battleserver

import (
	"math"
	"os"
	"sort"
	"testing"
	"time"
)

const (
	assaultLongMatchBots = 10
	assaultLongMatchIdle = 30 * time.Second
)

type assaultBotActivity struct {
	bot             *botBrain
	lastX           float32
	lastY           float32
	lastXP          float64
	lastLevel       int32
	lastActivityAt  float64
	maxIdleSeconds  float64
	activeSamples   int
	eligibleSamples int
	sawMovement     bool
	sawAction       bool
	sawProgress     bool
}

// TestTenBotAssaultRunsFiveMinutesWithoutAFK drives five simulated minutes by
// advancing the authoritative battle clock. It never waits for wall-clock
// timers, so it is safe to run as part of the ordinary integration suite.
func TestTenBotAssaultRunsFiveMinutesWithoutAFK(t *testing.T) {
	if testing.Short() {
		t.Skip("long bot match test is disabled in -short mode")
	}
	if os.Getenv("TANAT_RUN_5M_BOT_TEST") != "1" {
		t.Skip("set TANAT_RUN_5M_BOT_TEST=1 to run the five-minute Assault test")
	}

	duration := 5 * time.Minute
	driver := newManualDotaBots(t, 190010, assaultLongMatchBots)
	server, instance := driver.server, driver.inst

	instance.mu.Lock()
	server.spawnDotaBotsLocked(instance, assaultLongMatchBots)
	if len(instance.bots) != assaultLongMatchBots {
		instance.mu.Unlock()
		t.Fatalf("expected %d bots, spawned %d", assaultLongMatchBots, len(instance.bots))
	}

	activities := make(map[int32]*assaultBotActivity, assaultLongMatchBots)
	for id, bot := range instance.bots {
		x, y := bot.c.posAtLocked(float32(server.battleTime()))
		activities[id] = &assaultBotActivity{
			bot:            bot,
			lastX:          x,
			lastY:          y,
			lastXP:         float64(bot.c.huntState.xp),
			lastLevel:      bot.c.huntState.level,
			lastActivityAt: 0,
		}
	}
	instance.mu.Unlock()

	for tick := 0; tick < int(duration/AssaultTick); tick++ {
		ended := driver.step()

		instance.mu.Lock()
		matchTime := float64(server.battleTime()) - instance.dota.startedAt
		if !ended {
			for id, activity := range activities {
				if activity.bot == nil || activity.bot.c == nil || activity.bot.c.huntState == nil {
					continue
				}
				if activity.bot.c.huntState.closed {
					instance.mu.Unlock()
					t.Fatalf("bot %d closed its connection at %.1fs", id, matchTime)
				}
				moved, action, progress, eligible := assaultBotActivityLocked(
					activity.bot,
					float32(server.battleTime()),
					activity.lastX,
					activity.lastY,
					activity.lastXP,
					activity.lastLevel,
				)
				if activity.lastActivityAt == 0 {
					activity.lastActivityAt = matchTime
				}
				if moved {
					activity.sawMovement = true
				}
				if action {
					activity.sawAction = true
				}
				if progress {
					activity.sawProgress = true
				}
				if eligible {
					activity.eligibleSamples++
					if moved || action || progress {
						activity.activeSamples++
						idle := matchTime - activity.lastActivityAt
						if idle > activity.maxIdleSeconds {
							activity.maxIdleSeconds = idle
						}
						activity.lastActivityAt = matchTime
					} else if idle := matchTime - activity.lastActivityAt; idle > activity.maxIdleSeconds {
						activity.maxIdleSeconds = idle
					}
				} else {
					// Death, stun, root, or an active cast is not an AFK interval.
					activity.lastActivityAt = matchTime
				}
				activity.lastX, activity.lastY = botPositionLocked(activity.bot, float32(server.battleTime()))
				activity.lastXP = float64(activity.bot.c.huntState.xp)
				activity.lastLevel = activity.bot.c.huntState.level
			}
		}
		instance.mu.Unlock()

		if ended {
			t.Fatalf("Assault ended before the five-minute activity window at %.1fs", matchTime)
		}
	}

	instance.mu.Lock()
	finalMatchTime := float64(server.battleTime()) - instance.dota.startedAt
	ended := instance.dota != nil && instance.dota.ended
	for id, activity := range activities {
		if activity.bot == nil || activity.bot.c == nil || activity.bot.c.huntState == nil {
			instance.mu.Unlock()
			t.Fatalf("bot %d disappeared before the activity window ended", id)
		}
		if activity.bot.c.huntState.closed {
			instance.mu.Unlock()
			t.Fatalf("bot %d closed its connection at %.1fs", id, finalMatchTime)
		}
		if idle := finalMatchTime - activity.lastActivityAt; idle > activity.maxIdleSeconds {
			activity.maxIdleSeconds = idle
		}
		if !activity.sawMovement && !activity.sawAction && !activity.sawProgress {
			instance.mu.Unlock()
			t.Fatalf("bot %d had no movement, combat/cast action, or progress during %.1fs", id, finalMatchTime)
		}
		if activity.maxIdleSeconds > assaultLongMatchIdle.Seconds() {
			instance.mu.Unlock()
			t.Fatalf("bot %d was inactive for %.1fs (limit %.1fs)", id, activity.maxIdleSeconds, assaultLongMatchIdle.Seconds())
		}
	}
	ordered := make([]*assaultBotActivity, 0, len(activities))
	for _, activity := range activities {
		ordered = append(ordered, activity)
	}
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].bot.slot < ordered[j].bot.slot })
	t.Logf("Assault final score after %.1fs:", finalMatchTime)
	for _, activity := range ordered {
		hs := activity.bot.c.huntState
		t.Logf("%s avatar=%s side=%d level=%d xp=%.0f kills=%d deaths=%d assists=%d max_idle=%.1fs",
			activity.bot.c.name,
			hs.av.Prefab,
			hs.team,
			hs.level+1,
			hs.xp,
			hs.frags,
			hs.deaths,
			hs.assists,
			activity.maxIdleSeconds,
		)
	}
	instance.mu.Unlock()

	if ended {
		t.Fatalf("Assault ended before the five-minute activity window at %.1fs", finalMatchTime)
	}
}

// TestTenBotAssaultRunsFiveMinutesAtX20 is retained as a compatibility alias
// for developer scripts. The simulation is now manual, therefore X20 does not
// alter correctness or execution time.
func TestTenBotAssaultRunsFiveMinutesAtX20(t *testing.T) {
	if os.Getenv("TANAT_RUN_X20_BOT_TEST") != "1" {
		t.Skip("set TANAT_RUN_X20_BOT_TEST=1 to run the compatibility alias")
	}
	t.Setenv("TANAT_RUN_5M_BOT_TEST", "1")
	TestTenBotAssaultRunsFiveMinutesWithoutAFK(t)
}

func assaultBotActivityLocked(bot *botBrain, now, previousX, previousY float32, previousXP float64, previousLevel int32) (moved, action, progress, eligible bool) {
	if bot == nil || bot.c == nil || bot.c.huntState == nil {
		return false, false, false, false
	}

	hs := bot.c.huntState
	x, y := botPositionLocked(bot, now)
	moved = math.Hypot(float64(x-previousX), float64(y-previousY)) > 0.05 ||
		bot.c.hasDest || math.Abs(float64(bot.c.vx)) > 0.01 || math.Abs(float64(bot.c.vy)) > 0.01
	action = hs.attackTarget != 0 || hs.attackActionActive || hs.pvpTarget != 0 ||
		hs.order != nil || len(hs.payloads) > 0 || len(hs.channels) > 0 || bot.pendingTeleport != nil
	progress = float64(hs.xp) > previousXP+0.01 || hs.level != previousLevel

	nowSeconds := float64(now)
	incapacitated := hs.deadUntil > nowSeconds || hs.st.stunned(nowSeconds) || hs.st.rooted(nowSeconds) ||
		hs.castLockUntil > nowSeconds || hs.dashUntil > nowSeconds
	eligible = !incapacitated
	return moved, action, progress, eligible
}

func botPositionLocked(bot *botBrain, now float32) (float32, float32) {
	if bot == nil || bot.c == nil {
		return 0, 0
	}
	return bot.c.posAtLocked(now)
}
