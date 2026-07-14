package battleserver

import (
	"net"
	"testing"

	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// TestShinDalarKitShape pins the two data-level fixes: the astral jump (slot 2)
// moves the avatar via OpDash (OpBlink never visibly teleported -- the client only
// slides toward a synced position), and the poison passive (slot 3) carries an acid
// DotFx so the DoT paints a visual on the victim.
func TestShinDalarKitShape(t *testing.T) {
	shin, ok := gamedata.AvatarByID(16)
	if !ok {
		t.Fatal("Shin Dalar (id 16) missing")
	}
	kit := gamedata.SkillsFor(shin)

	var hasDash bool
	for _, op := range kit.Skills[1].Ops { // slot 2
		if op.Kind == gamedata.OpDash {
			hasDash = true
		}
		if op.Kind == gamedata.OpBlink {
			t.Error("slot 2 still uses OpBlink -- the client won't teleport, it slides")
		}
	}
	if !hasDash {
		t.Error("slot 2 astral jump must move the avatar via OpDash")
	}

	// Slot 3 poison proc -> OpDot must carry the acid DotFx.
	var acid string
	for _, op := range kit.Skills[2].Ops { // slot 3, an OpProc wrapping the DoT
		for _, inner := range op.Ops {
			if inner.Kind == gamedata.OpDot {
				acid = inner.DotFx
			}
		}
	}
	if acid == "" {
		t.Error("slot 3 poison DoT has no DotFx -- no acid visual on enemies")
	}
}

// TestShinDalarPoisonAcidAndSmoothDrain: applying the poison DoT attaches the acid
// visual to the victim (st.dotFx set), the health then bleeds down in small per-tick
// slices (not one lurch a second), and the acid is dropped once the DoT expires.
// Uses a raw drain (never blocks/parses) and asserts server state -- decoding the
// packet stream while ticking the mob AI desyncs a framed reader and hangs the pipe.
func TestShinDalarPoisonAcidAndSmoothDrain(t *testing.T) {
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

	shin, _ := gamedata.AvatarByID(16) // Shin Dalar
	c := &conn{Conn: srv}
	c.objID = 1000
	now := float64(s.battleTime())
	hs := &huntState{
		av:      shin,
		kit:     gamedata.SkillsFor(shin),
		mobs:    map[int32]*mobState{},
		summons: map[int32]*summonState{},
		hp:      500, mana: 100,
	}
	hs.tr.add(c.objID)
	mob := &mobState{id: 2000, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 1000, aggro: true, shown: true}
	hs.tr.add(mob.id)
	hs.mobs[mob.id] = mob
	c.huntState = hs
	defer func() { c.mvMu.Lock(); hs.closed = true; c.mvMu.Unlock() }()

	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	// Apply a DoT: 100/sec for 1s, with the acid visual.
	dot := gamedata.Op{Kind: gamedata.OpDot, Value: gamedata.PerLevel{100, 100, 100, 100},
		Dur: gamedata.PerLevel{1, 1, 1, 1}, Scale: "magic", DotFx: "ShinDalarSkill3Target"}
	s.applyOpsLocked(c, []gamedata.Op{dot}, opCtx{slot: 3, level: 1, target: mob}, now)

	if mob.st.dotFx == 0 {
		t.Fatal("poison did not attach the acid visual (st.dotFx) to the enemy")
	}

	startHP := mob.hp
	// First fine tick (0.2s in): only ~one slice (~20), NOT a full second (100).
	s.tickMobsLocked(c, now+dotTickInterval)
	firstDrop := startHP - mob.hp
	if firstDrop <= 0 {
		t.Fatal("DoT dealt no damage on the first tick")
	}
	if firstDrop > 40 { // one 0.2s slice of 100/s = 20; a 1s lurch would be 100
		t.Errorf("first tick drained %.0f hp -- poison is lurching, not bleeding smoothly", firstDrop)
	}

	// Drain across the rest of the second in fine steps, then a tick past 1s expires it.
	for tck := 2.0; tck <= 6; tck++ {
		s.tickMobsLocked(c, now+tck*dotTickInterval)
	}
	s.tickMobsLocked(c, now+1.5)

	if totalDrop := startHP - mob.hp; totalDrop < 80 || totalDrop > 120 { // ~100 over 1s
		t.Errorf("total poison damage %.0f, want ~100 (perSec*duration)", totalDrop)
	}
	if len(mob.st.dots) != 0 {
		t.Errorf("DoT did not expire: %d stacks remain", len(mob.st.dots))
	}
	if mob.st.dotFx != 0 {
		t.Error("acid visual was not cleared when the poison expired")
	}
}
