package battleserver

import (
	"os"
	"strconv"
	"strings"
)

// Bot AI versions are cumulative profiles. AI-10 is the original combat,
// economy, and macro baseline; AI-20 is the current profile. Version-specific
// policy lives in bot_ai10.go and bot_ai20.go; this file only owns selection
// and match-scoped lookup.
const (
	botAIVersionMin     = 0
	botAIVersionMax     = 20
	botAIVersionDefault = botAIVersionMax
)

type botAIPolicy interface {
	Version() int
	UsesTeamOrchestrator() bool
	UsesFarmSafeWave() bool
	UsesFarmLanePlan() bool
	UsesPlanHysteresis() bool
	UsesFarmRescue() bool
	UsesFarmRotation() bool
	UsesFarmStability() bool
	UsesFarmDebt() bool
}

// Environment variables are sampled when the Dota instance is created, not
// on every bot tick. This makes an A/B match reproducible even if the process
// environment is changed while the server is running.
const (
	botAIEnvTeam1 = "TANAT_DOTA_BOT_AI_TEAM1"
	botAIEnvTeam2 = "TANAT_DOTA_BOT_AI_TEAM2"
)

func parseBotAIVersion(raw string) int {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return botAIVersionDefault
	}
	upper := strings.ToUpper(raw)
	upper = strings.TrimPrefix(upper, "AI-")
	upper = strings.TrimPrefix(upper, "AI")
	v, err := strconv.Atoi(upper)
	if err != nil || (v != 0 && (v < 10 || v > botAIVersionMax)) {
		return botAIVersionDefault
	}
	return v
}

func botAIVersionFromEnv(name string) int {
	return parseBotAIVersion(os.Getenv(name))
}

func botAIVersionForTeam(team int32) int {
	switch team {
	case dotaTeamHuman:
		return botAIVersionFromEnv(botAIEnvTeam1)
	case dotaTeamElf:
		return botAIVersionFromEnv(botAIEnvTeam2)
	default:
		return botAIVersionDefault
	}
}

func normalizeBotAIVersion(v int) int {
	if v != 0 && (v < 10 || v > botAIVersionMax) {
		return botAIVersionDefault
	}
	return v
}

func botAIProfileForVersion(version int) botAIPolicy {
	version = normalizeBotAIVersion(version)
	if version == 0 {
		return botAI0Profile{}
	}
	if version == 10 {
		return botAI10Profile{}
	}
	return botAI20Profile{version: version}
}

func botAIVersionForBrain(b *botBrain) int {
	if b == nil || !b.aiVersionSet {
		return botAIVersionDefault
	}
	return normalizeBotAIVersion(b.aiVersion)
}

func botAIProfileForBrain(b *botBrain) botAIPolicy {
	return botAIProfileForVersion(botAIVersionForBrain(b))
}

func botTeamAIVersionLocked(inst *huntInstance, team int32) int {
	if inst != nil && inst.dota != nil {
		if version, ok := inst.dota.botAIVersionByTeam[team]; ok {
			return normalizeBotAIVersion(version)
		}
	}
	return botAIVersionDefault
}

func botPlanAIVersionLocked(inst *huntInstance, plan *botTeamPlan) int {
	if plan == nil {
		return botAIVersionDefault
	}
	if plan.AIVersionSet {
		return plan.AIVersion
	}
	return botTeamAIVersionLocked(inst, plan.Team)
}

func botAIProfileForPlanLocked(inst *huntInstance, plan *botTeamPlan) botAIPolicy {
	return botAIProfileForVersion(botPlanAIVersionLocked(inst, plan))
}
