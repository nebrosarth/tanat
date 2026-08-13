package battleserver

// botAI20Profile is the current cumulative policy. Keeping the milestone
// thresholds here makes the version-specific behaviour discoverable in one
// file instead of scattering AI-number comparisons through the orchestrator.
type botAI20Profile struct {
	version int
}

func (p botAI20Profile) Version() int { return p.version }

func (p botAI20Profile) UsesTeamOrchestrator() bool { return true }

func (p botAI20Profile) UsesFarmSafeWave() bool   { return p.version >= 12 }
func (p botAI20Profile) UsesFarmLanePlan() bool   { return p.version >= 13 }
func (p botAI20Profile) UsesPlanHysteresis() bool { return p.version >= 14 }
func (p botAI20Profile) UsesFarmRescue() bool     { return p.version >= 15 }
func (p botAI20Profile) UsesFarmRotation() bool   { return p.version >= 18 }
func (p botAI20Profile) UsesFarmStability() bool  { return p.version >= 19 }
func (p botAI20Profile) UsesFarmDebt() bool       { return p.version >= 20 }
