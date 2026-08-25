package battleserver

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"
	"time"
)

const (
	unbalancedAssaultBots = 9
)

// TestFiveBotAssaultFourVsFiveFirstFourMinutesFarm is the short farm probe for
// the numerical-advantage macro. It runs against a manually advanced clock.
func TestFiveBotAssaultFourVsFiveFirstFourMinutesFarm(t *testing.T) {
	if testing.Short() {
		t.Skip("4v5 accelerated Assault test is disabled in -short mode")
	}
	if os.Getenv("TANAT_RUN_4V5_BOT_TEST") != "1" {
		t.Skip("set TANAT_RUN_4V5_BOT_TEST=1 to run the 4v5 acceptance test")
	}
	runUnbalancedAssault(t, 4, false, "TANAT_4V5_TELEMETRY_DIR")
}

// TestFiveBotAssaultFourVsFiveWithinThirtyMinutes is the end-to-end goal gate:
// the side with five bots must destroy the opposing altar within 30 simulated
// minutes. It runs against a manually advanced clock.
func TestFiveBotAssaultFourVsFiveWithinThirtyMinutes(t *testing.T) {
	if testing.Short() {
		t.Skip("5v4 accelerated Assault test is disabled in -short mode")
	}
	if os.Getenv("TANAT_RUN_5V4_BOT_TEST") != "1" {
		t.Skip("set TANAT_RUN_5V4_BOT_TEST=1 to run the 5v4 acceptance test")
	}
	runUnbalancedAssault(t, 30, true, "TANAT_5V4_TELEMETRY_DIR")
}

func runUnbalancedAssault(t *testing.T, minutes int, requireFiveBotWin bool, telemetryEnv string) {
	telemetryDir := os.Getenv(telemetryEnv)
	if telemetryDir == "" {
		telemetryDir = t.TempDir()
	}
	t.Setenv("TANAT_BOT_TELEMETRY", telemetryDir)

	driver := newManualDotaBots(t, 190012, unbalancedAssaultBots)
	s, inst := driver.server, driver.inst

	inst.mu.Lock()
	if len(inst.bots) != unbalancedAssaultBots {
		inst.mu.Unlock()
		t.Fatalf("spawned %d bots, want %d", len(inst.bots), unbalancedAssaultBots)
	}
	ours, enemy := 0, 0
	for _, member := range inst.members {
		if member.playerTeam() == dotaTeamHuman {
			ours++
		} else {
			enemy++
		}
	}
	if (ours != 5 || enemy != 4) && (ours != 4 || enemy != 5) {
		inst.mu.Unlock()
		t.Fatalf("unexpected bot sides: %d vs %d, want 5 vs 4", ours, enemy)
	}
	largerTeam := dotaTeamHuman
	if enemy > ours {
		largerTeam = dotaTeamElf
	}
	start := inst.dota.startedAt
	inst.mu.Unlock()

	for tick := 0; tick < int((time.Duration(minutes)*time.Minute)/AssaultTick); tick++ {
		ended := driver.step()
		inst.mu.Lock()
		if ended {
			elapsed := float64(s.battleTime()) - start
			winner := inst.dota.winner
			logUnbalancedAssaultScore(t, inst, elapsed, winner)
			misses := inst.dota.earlyCreepXPMisses
			inst.mu.Unlock()
			if requireFiveBotWin && winner != largerTeam {
				t.Fatalf("5v4 Assault ended in %.1f simulated minutes, winner team=%d, want five-bot team=%d (early creep XP misses=%d)", elapsed/60, winner, largerTeam, misses)
			}
			if !requireFiveBotWin && misses > 0 {
				t.Fatalf("first 4 simulated minutes had %d creep XP misses", misses)
			}
			return
		}
		inst.mu.Unlock()
	}

	inst.mu.Lock()
	elapsed := float64(s.battleTime()) - start
	structures := unbalancedAssaultStructureSummary(inst)
	logUnbalancedAssaultScore(t, inst, elapsed, 0)
	logUnbalancedAssaultPlans(t, inst)
	misses := inst.dota.earlyCreepXPMisses
	ended := inst.dota.ended
	winner := inst.dota.winner
	inst.mu.Unlock()
	if requireFiveBotWin {
		t.Fatalf("5v4 Assault did not finish within %d simulated minutes: ended=%v winner=%d five-bot-team=%d early_creep_xp_misses=%d; structures: %s", minutes, ended, winner, largerTeam, misses, structures)
	}
	if misses > 0 {
		t.Fatalf("4v5 Assault did not satisfy first-4-minute farm metric: %d creep XP misses; structures: %s", misses, structures)
	}
	t.Logf("4v5 Assault first-%d-minute probe completed (elapsed %.1f simulated minutes); structures: %s",
		minutes, elapsed/60, structures)
}

func closeUnbalancedAssaultTestInstance(inst *huntInstance) {
	if inst == nil {
		return
	}
	inst.mu.Lock()
	inst.closed = true
	members := make([]*conn, 0, len(inst.members))
	for _, member := range inst.members {
		if member == nil {
			continue
		}
		if member.huntState != nil {
			member.huntState.closed = true
			member.huntState.attackSeq++
			member.stopArrivalLocked()
		}
		members = append(members, member)
	}
	if inst.dota != nil && inst.dota.telemetry != nil {
		inst.dota.telemetry.close()
		inst.dota.telemetry = nil
	}
	inst.mu.Unlock()
	for _, member := range members {
		if member.Conn != nil {
			_ = member.Conn.Close()
		}
	}
}

func logUnbalancedAssaultScore(t *testing.T, inst *huntInstance, elapsed float64, winner int32) {
	t.Logf("4v5 Assault result: elapsed=%.1f simulated minutes winner=%d early_creep_xp_misses=%d", elapsed/60, winner, inst.dota.earlyCreepXPMisses)
	bots := make([]*botBrain, 0, len(inst.bots))
	for _, brain := range inst.bots {
		bots = append(bots, brain)
	}
	sort.Slice(bots, func(i, j int) bool { return bots[i].slot < bots[j].slot })
	for _, brain := range bots {
		if brain == nil || brain.c == nil || brain.c.huntState == nil {
			continue
		}
		hs := brain.c.huntState
		t.Logf("%s side=%d level=%d xp=%.0f kills=%d deaths=%d assists=%d farm_xp=%d",
			brain.c.name, brain.c.playerTeam(), hs.level+1, hs.xp, hs.frags, hs.deaths, hs.assists, brain.farmXPEvents)
	}
}

func unbalancedAssaultStructureSummary(inst *huntInstance) string {
	if inst == nil || inst.dota == nil {
		return "no dota state"
	}
	parts := make([]string, 0)
	for _, structure := range botSortedMobs(inst) {
		if structure == nil || !structure.structure || structure.dead {
			continue
		}
		parts = append(parts, fmt.Sprintf("team%d:%s#%d=%.0f", structure.team, botMacroObjectiveName(structure), structure.id, structure.hp))
	}
	return strings.Join(parts, ",")
}

func logUnbalancedAssaultPlans(t *testing.T, inst *huntInstance) {
	for _, team := range []int32{dotaTeamHuman, dotaTeamElf} {
		plan := inst.dota.teamPlans[team]
		t.Logf("team %d final plan: mode=%s lane=%d objective=%d reason=%s", team, plan.Mode, plan.Lane, plan.ObjectiveID, plan.Reason)
		ids := make([]int32, 0, len(plan.Assignments))
		for id := range plan.Assignments {
			ids = append(ids, id)
		}
		sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
		for _, id := range ids {
			a := plan.Assignments[id]
			t.Logf("team %d bot %d assignment: mode=%s role=%s lane=%d farm=%d objective=%d reason=%s", team, id, a.Mode, a.Role, a.Lane, a.FarmLane, a.ObjectiveID, a.Reason)
		}
	}
}
