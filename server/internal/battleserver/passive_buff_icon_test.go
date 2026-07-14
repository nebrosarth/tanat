package battleserver

import (
	"math"
	"net"
	"testing"

	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// buildHuntConn is like newHuntConn but does NOT start a drain goroutine, so a
// readWorld pass on the same reader is the sole consumer (used to capture the
// exact packets a *Locked helper pushes).
func buildHuntConn(t *testing.T, prefab string) (*Server, *conn, net.Conn, *battleproto.Reader) {
	t.Helper()
	s := New(session.NewStore())
	srv, cli := net.Pipe()
	av := avatarByPrefab(t, prefab)
	c := &conn{Conn: srv}
	c.objID = 1000
	c.x, c.y, c.snapT = 0, 0, s.battleTime()
	hs := &huntState{
		av: av, kit: gamedata.SkillsFor(av),
		mobs: map[int32]*mobState{}, summons: map[int32]*summonState{},
		hp: 100, mana: 200,
	}
	hs.tr.add(c.objID)
	c.huntState = hs
	return s, c, cli, battleproto.NewReader(cli)
}

// TestVelialPassiveBuffIcon verifies «Воля к победе» (and every BuffIcon passive):
// once learned, the avatar carries a permanent BUFF-type effector on its status
// bar -- instantiated with the skill's level/tip args and NO "duration" (so the
// client shows it forever with no countdown). Before this fix a passive registered
// its buff prototype but never created the effector, so no icon appeared.
func TestVelialPassiveBuffIcon(t *testing.T) {
	s, c, cli, r := buildHuntConn(t, "Avtr_Tank_Velial")
	defer cli.Close()
	defer c.Conn.Close()

	hs := c.huntState
	hs.skillLevel[2] = 5 // learn «Воля к победе» at rank 5
	now := float64(s.battleTime())

	// sendEffectorsLocked mirrors the world-build/respawn effector set; run it in a
	// goroutine because net.Pipe writes block until readWorld consumes them.
	go func() {
		c.mvMu.Lock()
		s.sendEffectorsLocked(c, now)
		c.mvMu.Unlock()
	}()
	got := readWorld(t, cli, r)

	want := buffProtoID(hs.av, 3)
	var buff *battleproto.Packet
	for i := range got[battleproto.CmdAddEffector] {
		p := got[battleproto.CmdAddEffector][i]
		if proto, _ := p.Args.GetInt("proto"); proto == want {
			buff = &p
		}
	}
	if buff == nil {
		t.Fatal("no ADD_EFFECTOR for the passive's BUFF proto -- the status-effect icon is missing")
	}
	if owner, _ := buff.Args.GetInt("owner"); owner != c.objID {
		t.Errorf("buff effector owner = %d, want the avatar %d", owner, c.objID)
	}
	if _, hasDur := buff.Args.GetFloat("duration"); hasDur {
		t.Error("permanent passive buff must not carry a duration (would draw a countdown/expire visually)")
	}
	args, ok := buff.Args.GetArray("args")
	if !ok {
		t.Fatal("buff effector missing args map")
	}
	if lvl, _ := args.GetInt("level"); lvl != 5 {
		t.Errorf("buff effector level arg = %d, want 5 (tooltip must track the learned rank)", lvl)
	}
}

// TestVelialBuffCounterTracksMissingHP verifies the number drawn beside the icon:
// the buff effector carries a "counter" arg equal to the current bonus damage
// (coeff × missing-HP fraction), and it is re-sent (REM + ADD) with an updated value
// when Velial's HP changes enough to move the displayed integer.
func TestVelialBuffCounterTracksMissingHP(t *testing.T) {
	s, c, cli, r := buildHuntConn(t, "Avtr_Tank_Velial")
	defer cli.Close()
	defer c.Conn.Close()

	hs := c.huntState
	hs.skillLevel[2] = 5 // «Воля к победе» rank 5, coeff 100
	now := float64(s.battleTime())
	maxHP := hs.maxHPLocked(now)

	// Start at 45% missing HP -> bonus 45.
	hs.hp = maxHP * (1 - 0.45)
	go func() {
		c.mvMu.Lock()
		s.sendEffectorsLocked(c, now)
		c.mvMu.Unlock()
	}()
	got := readWorld(t, cli, r)

	want := buffProtoID(hs.av, 3)
	counterOf := func(pkts map[battleproto.CmdID][]battleproto.Packet) (int32, bool) {
		for _, p := range pkts[battleproto.CmdAddEffector] {
			if proto, _ := p.Args.GetInt("proto"); proto != want {
				continue
			}
			args, ok := p.Args.GetArray("args")
			if !ok {
				return 0, false
			}
			return args.GetInt("counter")
		}
		return 0, false
	}
	cnt, ok := counterOf(got)
	if !ok {
		t.Fatal("buff effector has no counter arg -- no number would show beside the icon")
	}
	if cnt != 45 {
		t.Errorf("initial counter = %d, want 45 (100 * 0.45 missing)", cnt)
	}

	// Drop to 70% missing -> bonus 70; the tick refresh must re-send with the new number.
	c.mvMu.Lock()
	oldEff := hs.passiveBuffEff[2]
	hs.hp = maxHP * (1 - 0.70)
	c.mvMu.Unlock()
	go func() {
		c.mvMu.Lock()
		s.refreshPassiveBuffCountersLocked(c, now)
		c.mvMu.Unlock()
	}()
	got2 := readWorld(t, cli, r)

	// The old icon is removed and a fresh one added with counter 70.
	remmed := false
	for _, p := range got2[battleproto.CmdRemEffector] {
		if id, _ := p.Args.GetInt("id"); id == oldEff {
			remmed = true
		}
	}
	if !remmed {
		t.Error("stale buff effector was not removed before re-adding with the new counter")
	}
	cnt2, ok := counterOf(got2)
	if !ok {
		t.Fatal("refreshed buff effector has no counter arg")
	}
	if cnt2 != 70 {
		t.Errorf("refreshed counter = %d, want 70 (100 * 0.70 missing)", cnt2)
	}
	c.mvMu.Lock()
	stored := hs.passiveBuffCount[2]
	c.mvMu.Unlock()
	if stored != 70 {
		t.Errorf("passiveBuffCount[2] = %d, want 70", stored)
	}

	// A tiny HP wiggle that does NOT change the rounded bonus must NOT re-send.
	c.mvMu.Lock()
	hs.hp = maxHP*(1-0.70) - 1 // sub-integer change
	beforeEff := hs.passiveBuffEff[2]
	c.mvMu.Unlock()
	go func() {
		c.mvMu.Lock()
		s.refreshPassiveBuffCountersLocked(c, now)
		c.mvMu.Unlock()
	}()
	got3 := readWorld(t, cli, r)
	if len(got3[battleproto.CmdAddEffector]) != 0 {
		t.Errorf("counter re-sent on a sub-integer HP change (%d ADD_EFFECTOR), should be a no-op",
			len(got3[battleproto.CmdAddEffector]))
	}
	c.mvMu.Lock()
	unchanged := hs.passiveBuffEff[2] == beforeEff
	// sanity: the expected bonus really is still 70 after the wiggle
	expBonus := int32(math.Round(100 * (1 - hs.hp/maxHP)))
	c.mvMu.Unlock()
	if !unchanged {
		t.Error("buff effector id changed on a no-op refresh")
	}
	if expBonus != 70 {
		t.Fatalf("test setup off: expected bonus %d after wiggle, want 70", expBonus)
	}
}

// TestUnlearnedPassiveNoBuffIcon guards the gate: a rank-0 (unlearned) passive must
// NOT show a buff icon -- the effector appears only once a point is spent.
func TestUnlearnedPassiveNoBuffIcon(t *testing.T) {
	s, c, cli, r := buildHuntConn(t, "Avtr_Tank_Velial")
	defer cli.Close()
	defer c.Conn.Close()

	hs := c.huntState
	hs.skillLevel[2] = 0 // «Воля к победе» not yet learned
	now := float64(s.battleTime())

	go func() {
		c.mvMu.Lock()
		s.sendEffectorsLocked(c, now)
		c.mvMu.Unlock()
	}()
	got := readWorld(t, cli, r)

	want := buffProtoID(hs.av, 3)
	for _, p := range got[battleproto.CmdAddEffector] {
		if proto, _ := p.Args.GetInt("proto"); proto == want {
			t.Fatal("an unlearned passive must not create its buff-bar icon")
		}
	}
	if hs.passiveBuffEff[2] != 0 {
		t.Errorf("passiveBuffEff[2] = %d, want 0 for an unlearned passive", hs.passiveBuffEff[2])
	}
}
