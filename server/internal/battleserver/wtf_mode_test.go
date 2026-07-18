package battleserver

import (
	"net"
	"testing"

	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// TestWTFModeFreeCastsNoCooldown pins WTF MODE: an avatar skill cast spends no mana
// and leaves no cooldown, so it can be recast immediately. Without WTF the same
// skill would deduct its ManaCost and block on Cooldown. Mobs are unaffected (they
// have no skill mana/cooldown), so this only needs to prove the avatar path.
func TestWTFModeFreeCastsNoCooldown(t *testing.T) {
	s := New(session.NewStore())
	srv, cli := net.Pipe()
	defer srv.Close()
	defer cli.Close()
	go func() {
		buf := make([]byte, 4096)
		for {
			if _, err := cli.Read(buf); err != nil {
				return
			}
		}
	}()

	velial, ok := gamedata.AvatarByID(13)
	if !ok {
		t.Fatal("Velial (id 13) missing")
	}
	kit := gamedata.SkillsFor(velial)
	// Pick a slot whose skill actually costs mana, so "no deduction" is meaningful.
	slot := 1
	if kit.Skills[0].ManaCost[0] <= 0 {
		t.Skip("Velial S1 has no mana cost; test needs a costed skill")
	}

	c := &conn{Conn: srv}
	c.objID = 1000
	c.x, c.y, c.snapT = 100, 100, s.battleTime()
	hs := &huntState{
		av:      velial,
		kit:     kit,
		mobs:    map[int32]*mobState{},
		summons: map[int32]*summonState{},
		hp:      100, mana: 200,
	}
	hs.skillLevel[slot-1] = 1
	hs.tr.add(c.objID)
	c.huntState = hs
	defer func() { c.mvMu.Lock(); hs.closed = true; c.mvMu.Unlock() }()

	debugWTFMode = true
	defer func() { debugWTFMode = false }()

	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	startMana := hs.mana
	s.execCastLocked(c, slot, nil, 0, 0, false, 0)
	if hs.mana != startMana {
		t.Errorf("WTF cast spent mana: %g -> %g (want unchanged)", startMana, hs.mana)
	}
	now := float64(s.battleTime())
	if hs.cooldownUntil[slot-1] > now {
		t.Errorf("WTF cast set a cooldown until %g (now %g) — skill should be instantly ready", hs.cooldownUntil[slot-1], now)
	}

	// Immediate recast: still free, still not blocked by cooldown.
	s.execCastLocked(c, slot, nil, 0, 0, false, 0)
	if hs.mana != startMana {
		t.Errorf("second WTF cast spent mana: now %g (want %g)", hs.mana, startMana)
	}

	// Sanity: with WTF OFF the same skill DOES cost mana and set a cooldown.
	debugWTFMode = false
	hs.cooldownUntil[slot-1] = 0
	hs.mana = 200
	s.execCastLocked(c, slot, nil, 0, 0, false, 0)
	if hs.mana >= 200 {
		t.Errorf("non-WTF cast did not spend mana (still %g) — the wrapper leaked into the off state", hs.mana)
	}
	if hs.cooldownUntil[slot-1] <= float64(s.battleTime()) {
		t.Error("non-WTF cast set no cooldown — the wrapper leaked into the off state")
	}
}
