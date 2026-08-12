package battleserver

import (
	"os"
	"strconv"
	"strings"
	"time"
)

// dotaSimulationSpeed is a development-only wall-clock multiplier for the
// headless Assault runner. The default is exactly one, so ordinary client
// matches and all existing tests retain their current timing.
//
// At X20, one 200ms world step is scheduled every 10ms of wall time. The
// authoritative battle clock still advances by 200ms per step; delayed attack,
// movement, projectile, channel, and corpse callbacks use the same conversion.
func dotaSimulationSpeed() float64 {
	raw := strings.TrimSpace(os.Getenv("TANAT_DOTA_SIM_SPEED"))
	if raw == "" {
		return 1
	}
	speed, err := strconv.ParseFloat(raw, 64)
	if err != nil || speed <= 0 {
		return 1
	}
	return speed
}

func dotaSimulationWallDuration(d time.Duration) time.Duration {
	if d <= 0 {
		return d
	}
	wall := time.Duration(float64(d) / dotaSimulationSpeed())
	if wall < time.Millisecond {
		return time.Millisecond
	}
	return wall
}

func dotaSimulationAfter(d time.Duration, fn func()) *time.Timer {
	return time.AfterFunc(dotaSimulationWallDuration(d), fn)
}

func dotaSimulationTickerInterval() time.Duration {
	return dotaSimulationWallDuration(tickInterval)
}
