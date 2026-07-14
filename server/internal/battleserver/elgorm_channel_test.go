package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

// TestElgormArrowChannelBreaksOnMove: Elgorm's «Стрелы Аркана» (slot 4) is a
// caster-sustained ground channel, so it ends the moment the caster moves --
// unlike a fire-and-forget ground channel (Titanid's quake, covered by
// TestGroundChannelSurvivesMovement) which keeps erupting.
func TestElgormArrowChannelBreaksOnMove(t *testing.T) {
	s, c, _, sx, sy := newNavConnAvatar(t, 31) // Elgorm
	hs := c.huntState
	kit := gamedata.SkillsFor(hs.av)
	chOp := kit.Skills[3].Ops[0]
	if chOp.Kind != gamedata.OpChannel {
		t.Fatalf("expected Elgorm slot 4 op[0] to be OpChannel, got %q", chOp.Kind)
	}

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())

	if !channelInterruptible(hs.av.Prefab, 4) {
		t.Fatal("Elgorm slot 4 must be an interruptible channel")
	}
	hs.channels = []channelState{{
		slot: 4, level: 1, until: now + 10, interval: 1, nextPulse: now + 5,
		target: 0, px: sx + 3, py: sy, hasPos: true, ops: chOp.Ops,
		interruptible: channelInterruptible(hs.av.Prefab, 4),
	}}

	// The caster presses walk -> the sustained channel breaks (removed).
	c.hasDest = true
	s.tickChannelsLocked(c, now)
	if len(hs.channels) != 0 {
		t.Fatal("Elgorm's arrow channel should break when the caster moves")
	}
}

// TestNewCastBreaksInterruptibleChannel: a fresh cast supersedes a sustained
// channel (breakInterruptibleChannelsLocked) but leaves a fire-and-forget ground
// channel running.
func TestNewCastBreaksInterruptibleChannel(t *testing.T) {
	s, c, _, sx, sy := newNavConnAvatar(t, 31)
	hs := c.huntState
	now := float64(s.battleTime())

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.channels = []channelState{
		{slot: 4, level: 1, until: now + 10, hasPos: true, px: sx, py: sy, interruptible: true},
		{slot: 1, level: 1, until: now + 10, hasPos: true, px: sx + 2, py: sy, interruptible: false},
	}
	s.breakInterruptibleChannelsLocked(c)
	if len(hs.channels) != 1 {
		t.Fatalf("expected only the fire-and-forget channel to survive, got %d", len(hs.channels))
	}
	if hs.channels[0].interruptible {
		t.Fatal("the surviving channel should be the non-interruptible one")
	}
}
