package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

func TestBotLaneLastHitRespectsTowerDangerWhenRequireCover(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	tower := structOfSide(inst, gamedata.DotaCreepTower, dotaTeamElf)
	if tower == nil || tower.mob.AttackRange <= 0 {
		t.Fatal("setup: enemy shooting barracks tower is missing")
	}
	bot.x, bot.y, bot.snapT = tower.x+float32(tower.mob.AttackRange/2), tower.y, float32(now)
	last := teleportTestCreep(inst, 74001, dotaTeamElf, bot.x+1, bot.y)
	last.hp, last.maxHP = 1, last.mob.Health
	inst.mobs[last.id] = last
	b := &botBrain{c: bot, lane: 0, phase: botPhaseLane}

	bot.lock()
	defer bot.unlock()
	if got := s.botFindLaneTargetLocked(b, now, botLaneEngageRadius, true); got != nil {
		t.Fatalf("covered last-hit target = %d, want tower-danger rejection", got.id)
	}
	if got := s.botFindLaneTargetLocked(b, now, botLaneEngageRadius, false); got != last {
		t.Fatalf("grouped last-hit target = %v, want creep %d", got, last.id)
	}
}

func TestBotCommittedStructureFocusUsesLowestThreatID(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	now := float64(s.battleTime())
	var threats []*mobState
	for _, m := range botSortedMobs(inst) {
		if m != nil && !m.dead && m.structure && m.enemyOf(bot.playerTeam()) && m.mob.AttackRange > 0 {
			threats = append(threats, m)
			if len(threats) == 2 {
				break
			}
		}
	}
	if len(threats) != 2 {
		t.Fatal("setup: need two enemy shooting structures")
	}
	for _, threat := range threats {
		threat.hitAt = now + 1
		threat.hitTarget = bot.objID
	}
	want := threats[0]
	if got := s.botCommittedStructureFocusLocked(bot, now); got != want {
		t.Fatalf("committed structure focus = %v, want lowest threat ID %d", got, want.id)
	}
}

func TestBotLaneFrontEqualIndexUsesLowestID(t *testing.T) {
	_, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	lane := inst.dota.m.Lanes[0]
	lowID := teleportTestCreep(inst, 74002, dotaTeamHuman, bot.x, bot.y)
	highID := teleportTestCreep(inst, 74003, dotaTeamHuman, bot.x+1, bot.y)
	lowID.lane, lowID.laneIdx = lane, len(lane)/2
	highID.lane, highID.laneIdx = lane, len(lane)/2
	inst.mobs[lowID.id], inst.mobs[highID.id] = lowID, highID

	fx, fy, ok := botLaneFrontLocked(bot, lane)
	if !ok || fx != lowID.x || fy != lowID.y {
		t.Fatalf("lane front = (%.1f,%.1f,%v), want lower ID creep %d at (%.1f,%.1f)", fx, fy, ok, lowID.id, lowID.x, lowID.y)
	}
	fx, fy, ok = botLaneFrontLockedForTeam(inst, dotaTeamHuman, 0)
	if !ok || fx != lowID.x || fy != lowID.y {
		t.Fatalf("team lane front = (%.1f,%.1f,%v), want lower ID creep %d at (%.1f,%.1f)", fx, fy, ok, lowID.id, lowID.x, lowID.y)
	}

	elfLow := teleportTestCreep(inst, 74004, dotaTeamElf, bot.x+2, bot.y)
	elfHigh := teleportTestCreep(inst, 74005, dotaTeamElf, bot.x+3, bot.y)
	elfLow.lane, elfLow.laneIdx = lane, len(lane)/2
	elfHigh.lane, elfHigh.laneIdx = lane, len(lane)/2
	inst.mobs[elfLow.id], inst.mobs[elfHigh.id] = elfLow, elfHigh
	fx, fy, ok = botLaneFrontLockedForTeam(inst, dotaTeamElf, 0)
	if !ok || fx != elfLow.x || fy != elfLow.y {
		t.Fatalf("elf team lane front = (%.1f,%.1f,%v), want lower ID creep %d at (%.1f,%.1f)", fx, fy, ok, elfLow.id, elfLow.x, elfLow.y)
	}
}

func TestBotTeamPlanKeyIncludesAssignmentObjectiveID(t *testing.T) {
	base := botTeamPlan{
		Mode: botMacroPush, Lane: 1, ObjectiveID: 50019,
		Assignments: map[int32]botMacroAssignment{
			1001: {Mode: botMacroPush, Lane: 1, BaselineLane: 0, ObjectiveID: 50019, Role: "push", Reason: "barracks"},
		},
	}
	changed := base
	changed.Assignments = map[int32]botMacroAssignment{
		1001: {Mode: botMacroPush, Lane: 1, BaselineLane: 0, ObjectiveID: 50017, Role: "push", Reason: "barracks"},
	}
	if botTeamPlanKey(base) == botTeamPlanKey(changed) {
		t.Fatal("team plan key ignored per-assignment objective change")
	}
}

func TestBotMatchStartWalletIsExactly100Gold(t *testing.T) {
	s, player, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	const slot = 123
	id := botIDBase + slot
	s.Store.CreateBotHero(id, "carry-over-bot")
	s.Store.SetHeroMoney(id, 777, 42)

	inst.mu.Lock()
	bot := s.newBotConnLocked(inst, slot, gamedata.DotaSideHuman, player.huntState.av)
	inst.mu.Unlock()
	t.Cleanup(func() { bot.Conn.Close() })

	money, diamonds, ok := s.Store.HeroMoney(id)
	wantMoney := int32(100) * session.BronzePerGold
	if !ok || money != wantMoney || diamonds != 0 {
		t.Fatalf("bot start wallet = (%d,%d,%v), want (%d,0,true)", money, diamonds, ok, wantMoney)
	}
}
