package battleserver

// botAI10Profile contains the deliberately small AI-10 policy surface. The
// simulation engine, combat rules, and movement primitives stay shared; only
// strategic improvements introduced after AI-10 are disabled here.
type botAI10Profile struct{}

func (botAI10Profile) Version() int { return 10 }

func (botAI10Profile) UsesTeamOrchestrator() bool { return true }

func (botAI10Profile) UsesFarmSafeWave() bool   { return false }
func (botAI10Profile) UsesFarmLanePlan() bool   { return false }
func (botAI10Profile) UsesPlanHysteresis() bool { return false }
func (botAI10Profile) UsesFarmRescue() bool     { return false }
func (botAI10Profile) UsesFarmRotation() bool   { return false }
func (botAI10Profile) UsesFarmStability() bool  { return false }
func (botAI10Profile) UsesFarmDebt() bool       { return false }
