package battleserver

import "testing"

func TestParseBotAIVersion(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want int
	}{
		{name: "empty defaults to latest", raw: "", want: botAIVersionDefault},
		{name: "plain number", raw: "10", want: 10},
		{name: "AI label", raw: "AI-20", want: 20},
		{name: "scripted teacher", raw: "AI-30", want: 30},
		{name: "neural profile", raw: "AI-40", want: 40},
		{name: "lowercase label", raw: "ai18", want: 18},
		{name: "invalid defaults to latest", raw: "experimental", want: botAIVersionDefault},
		{name: "unsupported defaults to latest", raw: "9", want: botAIVersionDefault},
		{name: "unassigned version", raw: "AI-21", want: botAIVersionDefault},
		{name: "legacy profile", raw: "AI-0", want: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := parseBotAIVersion(tt.raw); got != tt.want {
				t.Fatalf("parseBotAIVersion(%q) = %d, want %d", tt.raw, got, tt.want)
			}
		})
	}
}

func TestAI0SkipsTeamOrchestrator(t *testing.T) {
	t.Setenv(botAIEnvTeam1, "AI-0")
	t.Setenv(botAIEnvTeam2, "AI-20")
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	legacy := macroAddBot(t, s, inst, botIDBase+711, dotaTeamHuman, 0, 0, 0)
	modern := macroAddBot(t, s, inst, botIDBase+712, dotaTeamElf, 0, 0, 0)

	s.botPlanTeamsLocked(inst, 0)
	if _, ok := inst.dota.teamPlans[dotaTeamHuman]; ok {
		t.Fatal("AI-0 team unexpectedly received a team plan")
	}
	if legacy.macroAssignment.Reason != "" {
		t.Fatalf("AI-0 brain was assigned by orchestrator: %+v", legacy.macroAssignment)
	}
	if _, ok := inst.dota.teamPlans[dotaTeamElf]; !ok {
		t.Fatal("AI-20 team did not receive a team plan")
	}
	if modern.macroAssignment.Reason == "" {
		t.Fatal("AI-20 brain did not receive an orchestrator assignment")
	}
}

func TestBotAIVersionIsSelectedPerTeamAtMatchCreation(t *testing.T) {
	t.Setenv(botAIEnvTeam1, "AI-10")
	t.Setenv(botAIEnvTeam2, "20")
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()

	human := macroAddBot(t, s, inst, botIDBase+701, dotaTeamHuman, 0, 0, 0)
	elf := macroAddBot(t, s, inst, botIDBase+702, dotaTeamElf, 0, 0, 0)
	if got := human.aiVersion; got != 10 {
		t.Fatalf("team 1 bot AI version = %d, want 10", got)
	}
	if got := elf.aiVersion; got != 20 {
		t.Fatalf("team 2 bot AI version = %d, want 20", got)
	}
	if got := s.botPlanTeamLocked(inst, dotaTeamHuman, 0).AIVersion; got != 10 {
		t.Fatalf("team 1 plan AI version = %d, want 10", got)
	}
	if got := s.botPlanTeamLocked(inst, dotaTeamElf, 0).AIVersion; got != 20 {
		t.Fatalf("team 2 plan AI version = %d, want 20", got)
	}
}

func TestBotAIProfilesAreCumulative(t *testing.T) {
	if botAIProfileForVersion(0).UsesTeamOrchestrator() {
		t.Fatal("AI-0 unexpectedly enabled the team orchestrator")
	}
	if !botAIProfileForVersion(20).UsesTeamOrchestrator() {
		t.Fatal("AI-20 unexpectedly disabled the team orchestrator")
	}
	if botAIProfileForVersion(40).UsesTeamOrchestrator() {
		t.Fatal("AI-40 unexpectedly enabled scripted team orchestrator")
	}
	if profile := botAIProfileForVersion(30); profile.Version() != 30 || !profile.UsesTeamOrchestrator() ||
		!profile.UsesFarmSafeWave() || !profile.UsesFarmDebt() {
		t.Fatalf("AI-30 did not inherit the complete scripted policy: %+v", profile)
	}
	legacy := botAIProfileForVersion(10)
	modern := botAIProfileForVersion(20)
	if legacy.UsesFarmLanePlan() {
		t.Fatal("AI-10 unexpectedly enabled live farm-lane planning")
	}
	if !modern.UsesFarmLanePlan() || !modern.UsesFarmDebt() {
		t.Fatal("AI-20 did not enable the latest farm features")
	}
	if botAIProfileForVersion(18).UsesFarmStability() {
		t.Fatal("AI-18 unexpectedly enabled AI-19 farm stability")
	}
}
