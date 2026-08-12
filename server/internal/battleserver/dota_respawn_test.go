package battleserver

import "testing"

func TestDotaHeroRespawnDelayUsesCurrentLevelTable(t *testing.T) {
	tests := []struct {
		internalLevel int32
		wantSeconds   float64
	}{
		{internalLevel: 0, wantSeconds: 12},  // displayed level 1
		{internalLevel: 1, wantSeconds: 15},  // displayed level 2
		{internalLevel: 4, wantSeconds: 24},  // displayed level 5
		{internalLevel: 5, wantSeconds: 26},  // displayed level 6
		{internalLevel: 10, wantSeconds: 36}, // displayed level 11
		{internalLevel: 11, wantSeconds: 44}, // displayed level 12
		{internalLevel: 17, wantSeconds: 65}, // displayed level 18
		{internalLevel: 19, wantSeconds: 75}, // displayed level 20
	}
	for _, tt := range tests {
		hs := &huntState{level: tt.internalLevel}
		if got := dotaHeroRespawnDelay(hs); got != tt.wantSeconds {
			t.Errorf("dotaHeroRespawnDelay(internal level %d) = %.0f, want %.0f", tt.internalLevel, got, tt.wantSeconds)
		}
	}
}

func TestDotaHeroRespawnDelayClampsToLevelTwenty(t *testing.T) {
	hs := &huntState{level: 99}
	if got := dotaHeroRespawnDelay(hs); got != 75 {
		t.Fatalf("dotaHeroRespawnDelay(level 100 display) = %.0f, want level-20 cap 75", got)
	}
}

func TestPlayerDeathSchedulesTableRespawnTime(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Velial")
	defer cleanup()

	for _, tc := range []struct {
		internalLevel int32
		wantSeconds   float64
	}{
		{internalLevel: 0, wantSeconds: 12},
		{internalLevel: 11, wantSeconds: 44},
		{internalLevel: 19, wantSeconds: 75},
	} {
		now := float64(s.battleTime())
		c.mvMu.Lock()
		c.huntState.level = tc.internalLevel
		c.huntState.hp = 0
		s.playerDieLocked(c, 42, now)
		got := c.huntState.deadUntil - now
		// Reset the state for the next table row; this test targets the timer,
		// not the full death/respawn lifecycle.
		c.huntState.deadUntil = 0
		c.huntState.hp = c.huntState.maxHPLocked(now)
		c.mvMu.Unlock()
		if got != tc.wantSeconds {
			t.Errorf("player death at internal level %d scheduled %.0fs, want %.0fs", tc.internalLevel, got, tc.wantSeconds)
		}
	}
}
