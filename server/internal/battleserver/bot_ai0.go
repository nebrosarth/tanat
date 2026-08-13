package battleserver

// botAI0Profile is the pre-orchestrator legacy profile. It deliberately does
// not expose any team-policy feature: each bot keeps its authored lane and
// lets botUpdatePhaseLocked/botLaneTickLocked/botRoamTickLocked make local
// decisions from its own world state.
type botAI0Profile struct{}

func (botAI0Profile) Version() int               { return 0 }
func (botAI0Profile) UsesTeamOrchestrator() bool { return false }
func (botAI0Profile) UsesFarmSafeWave() bool     { return false }
func (botAI0Profile) UsesFarmLanePlan() bool     { return false }
func (botAI0Profile) UsesPlanHysteresis() bool   { return false }
func (botAI0Profile) UsesFarmRescue() bool       { return false }
func (botAI0Profile) UsesFarmRotation() bool     { return false }
func (botAI0Profile) UsesFarmStability() bool    { return false }
func (botAI0Profile) UsesFarmDebt() bool         { return false }
