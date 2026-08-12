package battleserver

import (
	"math"
	"testing"

	"tanatserver/internal/gamedata"
)

func retreatTestSkill(slot int, target string, distance int, ops ...gamedata.Op) gamedata.Skill {
	return gamedata.Skill{
		Slot: slot, Type: "ACTIVE", Target: target, Distance: distance,
		ManaCost: []int{0}, Cooldown: []int{1}, Ops: ops,
	}
}

func installRetreatTestKit(bot *conn, skills ...gamedata.Skill) {
	kit := &gamedata.AvatarSkills{}
	for i := range kit.Skills {
		kit.Skills[i] = retreatTestSkill(i+1, "ENEMY", 8, gamedata.Op{Kind: gamedata.OpDamage, Value: gamedata.PerLevel{1}})
	}
	for _, skill := range skills {
		kit.Skills[skill.Slot-1] = skill
		bot.huntState.skillLevel[skill.Slot-1] = 1
	}
	bot.huntState.kit = kit
}

func TestBotRetreatUtilitySelectionAndSafety(t *testing.T) {
	t.Run("heal remains first", func(t *testing.T) {
		s, bot, _, cleanup := newDotaConn(t, "Avtr_HK_Astarot")
		defer cleanup()
		now := float64(s.battleTime())
		installRetreatTestKit(bot,
			retreatTestSkill(1, "SELF", 0, gamedata.Op{Kind: gamedata.OpHeal, Value: gamedata.PerLevel{10}}),
			retreatTestSkill(2, "SELF", 0, gamedata.Op{Kind: gamedata.OpStealth, Dur: gamedata.PerLevel{3}}),
		)
		bot.lock()
		defer bot.unlock()
		bot.huntState.hp = 1
		if !s.botConsiderRetreatUtilityLocked(&botBrain{c: bot}, now) {
			t.Fatal("low-health bot did not use its heal")
		}
		if bot.huntState.cooldownUntil[0] <= 0 || bot.huntState.cooldownUntil[1] != 0 {
			t.Fatalf("heal priority cooldowns = %v, want slot 1 only", bot.huntState.cooldownUntil)
		}
	})

	t.Run("point dash is homeward and opens distance", func(t *testing.T) {
		s, bot, inst, cleanup := newDotaConn(t, "Avtr_HK_Astarot")
		defer cleanup()
		now := float64(s.battleTime())
		hx, hy := botHomeLocked(bot)
		bot.x, bot.y, bot.snapT = hx+10, hy, float32(now)
		pursuer := dotaPlayerConn(t, s, inst, 66001, dotaTeamElf, bot.x+4, bot.y)
		pursuer.huntState.pvpTarget = bot.objID
		installRetreatTestKit(bot, retreatTestSkill(1, "POINT", 20,
			gamedata.Op{Kind: gamedata.OpDash, Value: gamedata.PerLevel{20}},
			gamedata.Op{Kind: gamedata.OpDamage, Value: gamedata.PerLevel{10}}))
		bot.lock()
		defer bot.unlock()
		before := math.Hypot(float64(bot.x-pursuer.x), float64(bot.y-pursuer.y))
		tx, ty, ok := botRetreatPointDashLocked(&botBrain{c: bot}, gamedata.Skill{Target: "POINT", Distance: 20}, now, pursuer)
		if !ok || math.Hypot(float64(tx-pursuer.x), float64(ty-pursuer.y)) <= before {
			t.Fatalf("homeward dash does not increase pursuer distance: before=%.2f after=(%.2f,%.2f)", before, tx, ty)
		}
		if !s.botConsiderRetreatUtilityLocked(&botBrain{c: bot}, now) || bot.huntState.cooldownUntil[0] <= 0 {
			t.Fatal("Astarot-like point dash+damage was not accepted")
		}
	})

	t.Run("damage plus CC controls nearby pursuer without chase", func(t *testing.T) {
		s, bot, inst, cleanup := newDotaConn(t, "Avtr_HK_Astarot")
		defer cleanup()
		now := float64(s.battleTime())
		enemy := dotaPlayerConn(t, s, inst, 66002, dotaTeamElf, bot.x+3, bot.y)
		enemy.huntState.pvpTarget = bot.objID
		installRetreatTestKit(bot, retreatTestSkill(1, "ENEMY", 8,
			gamedata.Op{Kind: gamedata.OpDamage, Value: gamedata.PerLevel{5}},
			gamedata.Op{Kind: gamedata.OpStun, Dur: gamedata.PerLevel{1}}))
		bot.lock()
		defer bot.unlock()
		if !s.botConsiderRetreatUtilityLocked(&botBrain{c: bot}, now) || bot.huntState.cooldownUntil[0] <= 0 {
			t.Fatal("nearby damage+CC pursuer control was skipped")
		}
		if bot.huntState.pvpTarget != 0 {
			t.Fatalf("retreat utility armed a chase: pvpTarget=%d", bot.huntState.pvpTarget)
		}
	})

	t.Run("damage plus CC skips out-of-range pursuer", func(t *testing.T) {
		s, bot, inst, cleanup := newDotaConn(t, "Avtr_HK_Astarot")
		defer cleanup()
		now := float64(s.battleTime())
		enemy := dotaPlayerConn(t, s, inst, 66003, dotaTeamElf, bot.x+12, bot.y)
		enemy.huntState.pvpTarget = bot.objID
		installRetreatTestKit(bot, retreatTestSkill(1, "ENEMY", 8,
			gamedata.Op{Kind: gamedata.OpDamage, Value: gamedata.PerLevel{5}},
			gamedata.Op{Kind: gamedata.OpStun, Dur: gamedata.PerLevel{1}}))
		bot.lock()
		defer bot.unlock()
		if s.botConsiderRetreatUtilityLocked(&botBrain{c: bot}, now) {
			t.Fatal("out-of-range damage+CC pursuer was accepted")
		}
	})

	t.Run("blocking self channel is never an escape utility", func(t *testing.T) {
		s, bot, _, cleanup := newDotaConn(t, "Avtr_HK_Tangren")
		defer cleanup()
		now := float64(s.battleTime())
		installRetreatTestKit(bot, retreatTestSkill(1, "SELF", 0,
			gamedata.Op{Kind: gamedata.OpBuffStat, On: "self", Stat: "magic_armor", Value: gamedata.PerLevel{1000}},
			gamedata.Op{Kind: gamedata.OpChannel, Dur: gamedata.PerLevel{5}, Interval: 1, Ops: []gamedata.Op{
				{Kind: gamedata.OpDamage, Value: gamedata.PerLevel{10}},
			}},
		))
		bot.lock()
		defer bot.unlock()
		bot.huntState.hp = 1
		if s.botConsiderRetreatUtilityLocked(&botBrain{c: bot}, now) {
			t.Fatal("blocking self channel was selected as retreat utility")
		}
		if bot.huntState.cooldownUntil[0] != 0 {
			t.Fatalf("blocking self channel was cast, cooldowns=%v", bot.huntState.cooldownUntil)
		}
	})

	for _, tc := range []struct {
		name  string
		skill gamedata.Skill
	}{
		{"pure damage", retreatTestSkill(1, "ENEMY", 8, gamedata.Op{Kind: gamedata.OpDamage, Value: gamedata.PerLevel{5}})},
		{"self damage", retreatTestSkill(1, "SELF", 0, gamedata.Op{Kind: gamedata.OpDamage, Apply: "self", Value: gamedata.PerLevel{5}})},
		{"attack speed only", retreatTestSkill(1, "SELF", 0, gamedata.Op{Kind: gamedata.OpBuffStat, On: "self", Stat: "attack_speed_pct", Value: gamedata.PerLevel{1}})},
	} {
		t.Run(tc.name+" is skipped", func(t *testing.T) {
			s, bot, _, cleanup := newDotaConn(t, "Avtr_HK_Astarot")
			defer cleanup()
			now := float64(s.battleTime())
			installRetreatTestKit(bot, tc.skill)
			bot.lock()
			defer bot.unlock()
			if s.botConsiderRetreatUtilityLocked(&botBrain{c: bot}, now) {
				t.Fatal("unsafe retreat utility was accepted")
			}
		})
	}

	for _, tc := range []struct {
		name string
		op   gamedata.Op
	}{
		{"move speed", gamedata.Op{Kind: gamedata.OpBuffStat, On: "self", Stat: "move_speed_pct", Value: gamedata.PerLevel{1}}},
		{"stealth", gamedata.Op{Kind: gamedata.OpStealth, Dur: gamedata.PerLevel{3}}},
		{"shield", gamedata.Op{Kind: gamedata.OpShield, Dur: gamedata.PerLevel{3}, Value: gamedata.PerLevel{10}}},
	} {
		t.Run(tc.name+" is accepted", func(t *testing.T) {
			s, bot, _, cleanup := newDotaConn(t, "Avtr_HK_Astarot")
			defer cleanup()
			now := float64(s.battleTime())
			installRetreatTestKit(bot, retreatTestSkill(1, "SELF", 0, tc.op))
			bot.lock()
			defer bot.unlock()
			if !s.botConsiderRetreatUtilityLocked(&botBrain{c: bot}, now) || bot.huntState.cooldownUntil[0] <= 0 {
				t.Fatal("safe retreat utility was not accepted")
			}
		})
	}
}
