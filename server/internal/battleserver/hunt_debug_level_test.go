package battleserver

import (
	"testing"

	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// TestDebugStartLevelSpawnsHigher pins the testing aid that spawns a hunt avatar
// at a chosen level (debugStartLevelOverride / TANAT_HUNT_START_LEVEL). At start
// level 5 the SET_AVATAR must report the 0-based level 4 with 5 skill points, the
// server huntState must sit at that level with the matching XP floor and full
// pools, and -- the point of the feature -- the level-5-gated ult must now be
// rankable, whereas by default (level 1) it is not.
func TestDebugStartLevelSpawnsHigher(t *testing.T) {
	const startLevel = 5

	scCh := make(chan *conn, 1)
	testHookNewConn = func(sc *conn) {
		select {
		case scCh <- sc:
		default:
		}
	}
	defer func() { testHookNewConn = nil }()

	debugStartLevelOverride = startLevel
	defer func() { debugStartLevelOverride = 0 }()

	c, r := enterHunt(t, session.NewStore(), 20) // Vigilans
	got := readWorld(t, c, r)

	// --- SET_AVATAR: 0-based level and the accumulated skill points ---
	if len(got[battleproto.CmdSetAvatar]) == 0 {
		t.Fatal("no SET_AVATAR in world state")
	}
	sa := got[battleproto.CmdSetAvatar][0]
	if lvl, _ := sa.Args.GetInt("level"); lvl != startLevel-1 {
		t.Errorf("SET_AVATAR level = %d, want %d (0-based level 5)", lvl, startLevel-1)
	}
	if pts, _ := sa.Args.GetInt("points"); pts != startLevel {
		t.Errorf("SET_AVATAR points = %d, want %d (1 + one per level gained)", pts, startLevel)
	}

	// --- Server-side huntState: level, XP floor, full pools, gate satisfied ---
	sc := <-scCh
	sc.mvMu.Lock()
	defer sc.mvMu.Unlock()
	hs := sc.huntState
	if hs == nil {
		t.Fatal("no huntState on server conn")
	}
	if hs.level != startLevel-1 {
		t.Errorf("hs.level = %d, want %d", hs.level, startLevel-1)
	}
	if hs.points != startLevel {
		t.Errorf("hs.points = %d, want %d", hs.points, startLevel)
	}
	if want := gamedata.AvatarXPLevels[startLevel-1]; hs.xp != want {
		t.Errorf("hs.xp = %g, want %g (floor of the start level)", hs.xp, want)
	}
	if maxHP := hs.maxHPLocked(0); hs.hp != maxHP {
		t.Errorf("hs.hp = %g, want full %g at the start level", hs.hp, maxHP)
	}

	// Skills still start UNLEARNED (rank 0) -- the flag grants level, not ranks.
	for i, lv := range hs.skillLevel {
		if lv != 0 {
			t.Errorf("skill slot %d starts at rank %d, want 0 (level flag must not pre-learn skills)", i+1, lv)
		}
	}
	// But the ult's rank-1 gate (level 5) is now met: level >= skillReqLevel(4,0).
	if int(hs.level) < skillReqLevel(4, 0) {
		t.Errorf("at start level %d the ult is still gated (level %d < req %d) — the flag should unlock it",
			startLevel, hs.level, skillReqLevel(4, 0))
	}
}
