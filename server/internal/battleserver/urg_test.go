package battleserver

import (
	"net"
	"sync"
	"testing"
	"time"

	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// newUrgConn builds a bare, world-ready Urg conn (team 1) with a draining client reader,
// returning the server, conn, and a mutex-guarded slice of everything pushed to the client.
func newUrgConn(t *testing.T) (*Server, *conn, *sync.Mutex, *[]battleproto.Packet) {
	t.Helper()
	s := New(session.NewStore())
	srv, cli := net.Pipe()
	t.Cleanup(func() { srv.Close(); cli.Close() })

	var mu sync.Mutex
	pkts := &[]battleproto.Packet{}
	r := battleproto.NewReader(cli)
	go func() {
		for {
			p, err := r.Read()
			if err != nil {
				return
			}
			mu.Lock()
			*pkts = append(*pkts, p)
			mu.Unlock()
		}
	}()

	urg, ok := gamedata.AvatarByID(12) // Urg
	if !ok {
		t.Fatal("Urg (id 12) missing")
	}
	c := &conn{Conn: srv}
	c.objID = 1000
	c.x, c.y, c.snapT = 0, 0, s.battleTime()
	hs := &huntState{
		av: urg, kit: gamedata.SkillsFor(urg), team: 1, worldReady: true,
		mobs: map[int32]*mobState{}, summons: map[int32]*summonState{},
		hp: 500, mana: 300,
	}
	for i := range hs.skillLevel {
		hs.skillLevel[i] = 1
	}
	hs.tr.add(c.objID)
	c.huntState = hs
	t.Cleanup(func() { c.mvMu.Lock(); hs.closed = true; c.mvMu.Unlock() })
	return s, c, &mu, pkts
}

// TestUrgTreeFormRevealBurst pins «Древесный камуфляж» (slot 1): the cast turns Urg into a
// tree — break-on-move stealth + a standing disguise prop — and does NO damage on cast; the
// magic-damage + silence burst fires only when the form ENDS (here, a reveal). This is the
// client's "При движении, он выходит из формы дерева, нанося магический урон и запрещая
// применение способностей", not the old instant burst.
func TestUrgTreeFormRevealBurst(t *testing.T) {
	s, c, mu, pkts := newUrgConn(t)
	hs := c.huntState

	// An enemy standing inside the reveal-burst radius (4) of the caster at (0,0).
	mob := gamedata.Mobs()[2]
	victim := &mobState{id: 5001, mobIdx: 2, mob: mob, x: 2, y: 0, hp: 800, homed: true}
	hs.mobs[victim.id] = victim

	skill1 := hs.kit.Skills[0]
	if skill1.Ops[0].Kind != gamedata.OpTreeForm {
		t.Fatalf("Urg slot 1 op[0] = %q, want tree_form", skill1.Ops[0].Kind)
	}

	now := float64(s.battleTime())
	c.mvMu.Lock()
	s.applyOpsLocked(c, skill1.Ops, opCtx{slot: 1, level: 1, hasPos: false}, now)
	c.mvMu.Unlock()

	// On cast: cloaked (break-on-move), disguise prop planted, burst ARMED but not fired.
	if now >= hs.invisibleUntil {
		t.Errorf("tree form did not cloak the avatar (invisibleUntil=%g, now=%g)", hs.invisibleUntil, now)
	}
	if !hs.stealthBreaksOnMove {
		t.Error("tree camouflage must break on move (client: «При движении… выходит из формы»)")
	}
	if hs.treeFormObj == 0 {
		t.Error("no disguise prop was planted")
	}
	if hs.treeFormBurst == nil {
		t.Error("reveal burst was not armed")
	}
	if victim.hp != 800 || victim.st.silenceUntil != 0 {
		t.Errorf("burst hit on CAST (hp=%g silence=%g); it must wait for the form to end", victim.hp, victim.st.silenceUntil)
	}

	// Leaving the form (any reveal path) detonates the burst around the avatar.
	c.mvMu.Lock()
	s.breakInvisibilityLocked(c, now)
	c.mvMu.Unlock()

	if hs.treeFormBurst != nil {
		t.Error("burst should have fired and disarmed on reveal")
	}
	if hs.treeFormObj != 0 {
		t.Error("disguise prop should be removed on reveal")
	}
	if victim.hp >= 800 {
		t.Errorf("reveal burst dealt no damage (hp still %g)", victim.hp)
	}
	if victim.st.silenceUntil <= now {
		t.Errorf("reveal burst did not silence the enemy (silenceUntil=%g)", victim.st.silenceUntil)
	}

	// The disguise prop must have been created (on cast) then deleted (on reveal).
	time.Sleep(40 * time.Millisecond)
	mu.Lock()
	var sawCreate, sawDelete bool
	for _, p := range *pkts {
		switch p.Cmd {
		case battleproto.CmdCreateObject:
			sawCreate = true
		case battleproto.CmdDeleteObject:
			sawDelete = true
		}
	}
	mu.Unlock()
	if !sawCreate {
		t.Error("no CREATE_OBJECT for the disguise tree prop")
	}
	if !sawDelete {
		t.Error("no DELETE_OBJECT for the disguise tree prop on reveal")
	}
}

// TestUrgGroveSilenceThenDamage pins «Непроглядные дебри» (slot 4): the cast SILENCES enemies
// inside the ring immediately, plants a ring of tree props, and defers the magic burst until
// the trees FALL. This is the client's "внутри которого враги не могут применять способности.
// Когда деревья исчезнут, все враги внутри получат магический урон".
func TestUrgGroveSilenceThenDamage(t *testing.T) {
	s, c, mu, pkts := newUrgConn(t)
	hs := c.huntState

	mob := gamedata.Mobs()[2]
	victim := &mobState{id: 6001, mobIdx: 2, mob: mob, x: 1, y: 0, hp: 900, homed: true}
	hs.mobs[victim.id] = victim

	skill4 := hs.kit.Skills[3]
	if skill4.Ops[0].Kind != gamedata.OpSilence || skill4.Ops[1].Kind != gamedata.OpGrove {
		t.Fatalf("Urg slot 4 ops = [%q,%q], want [silence,grove]", skill4.Ops[0].Kind, skill4.Ops[1].Kind)
	}
	grove := skill4.Ops[1]
	dur := grove.Dur.At(1)
	wantTrees := int(grove.Count.At(1))

	now := float64(s.battleTime())
	c.mvMu.Lock()
	s.applyOpsLocked(c, skill4.Ops, opCtx{slot: 4, level: 1, hasPos: false}, now)
	nPayloads := len(hs.payloads)
	nAnchors := len(hs.anchorEnds)
	c.mvMu.Unlock()

	// Immediate: enemy inside the ring is silenced; the burst has NOT landed yet.
	if victim.st.silenceUntil <= now {
		t.Errorf("grove did not silence the enemy inside it (silenceUntil=%g)", victim.st.silenceUntil)
	}
	if victim.hp != 900 {
		t.Errorf("grove dealt damage on CAST (hp=%g); the burst must wait for the trees to fall", victim.hp)
	}
	if nPayloads != 1 {
		t.Errorf("expected 1 deferred fall-damage payload, got %d", nPayloads)
	}
	if nAnchors != wantTrees {
		t.Errorf("expected %d tree props queued to fall, got %d", wantTrees, nAnchors)
	}

	// The trees fall: the deferred burst hits everyone still inside, and the props delete.
	c.mvMu.Lock()
	s.runDuePayloadsLocked(c, now+dur+0.5)
	s.tickAnchorEndsLocked(c, now+dur+0.5)
	c.mvMu.Unlock()

	if victim.hp >= 900 {
		t.Errorf("fall-damage burst dealt no damage (hp still %g)", victim.hp)
	}

	time.Sleep(40 * time.Millisecond)
	mu.Lock()
	creates, deletes := 0, 0
	for _, p := range *pkts {
		switch p.Cmd {
		case battleproto.CmdCreateObject:
			creates++
		case battleproto.CmdDeleteObject:
			deletes++
		}
	}
	mu.Unlock()
	if creates < wantTrees {
		t.Errorf("saw %d CREATE_OBJECT, want >= %d tree props", creates, wantTrees)
	}
	if deletes < wantTrees {
		t.Errorf("saw %d DELETE_OBJECT, want >= %d fallen trees", deletes, wantTrees)
	}
}
