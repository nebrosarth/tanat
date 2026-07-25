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

// TestUrgVisionWardSpawnsRevealsAndExpires drives Urg's «Росток» (slot 2): casting it
// plants the visible scout-tree prop at the point (CREATE_OBJECT with the ward proto),
// synced with the caster's TEAM and a fog VIEW_RADIUS so a friendly client reveals the
// area, with the ground fx pinned to the (stationary) prop; ticking past its lifetime
// deletes the prop.
func TestUrgVisionWardSpawnsRevealsAndExpires(t *testing.T) {
	s := New(session.NewStore())
	srv, cli := net.Pipe()
	defer srv.Close()
	defer cli.Close()

	var mu sync.Mutex
	var pkts []battleproto.Packet
	r := battleproto.NewReader(cli)
	go func() {
		for {
			p, err := r.Read()
			if err != nil {
				return
			}
			mu.Lock()
			pkts = append(pkts, p)
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
	kit := gamedata.SkillsFor(urg)
	hs := &huntState{
		av: urg, kit: kit,
		mobs: map[int32]*mobState{}, summons: map[int32]*summonState{},
		hp: 500, mana: 300,
	}
	hs.tr.add(c.objID)
	c.huntState = hs
	defer func() { c.mvMu.Lock(); hs.closed = true; c.mvMu.Unlock() }()

	wardOp := kit.Skills[1].Ops[0]
	if wardOp.Kind != gamedata.OpVisionWard {
		t.Fatalf("expected Urg slot 2 op[0] to be OpVisionWard, got %q", wardOp.Kind)
	}
	wantProto, ok := wardProtoIDFor(wardOp.Unit)
	if !ok {
		t.Fatalf("ward prop %q has no registered proto", wardOp.Unit)
	}
	// Regression: the ward proto must not collide with any other fixed proto, or the
	// client renders the wrong model (901 = the loot chest -> a chest instead of a tree).
	if wantProto == dropChestProtoID || wantProto == trapAnchorProtoID {
		t.Fatalf("ward proto %d collides with another fixed proto (chest=%d, anchor=%d)", wantProto, dropChestProtoID, trapAnchorProtoID)
	}

	now := float64(s.battleTime())
	c.mvMu.Lock()
	// Plant the ward at a ground point away from the caster (0,0).
	s.applyOpsLocked(c, []gamedata.Op{wardOp}, opCtx{slot: 2, level: 1, hasPos: true, px: 6, py: 0}, now)
	if len(hs.wards) != 1 {
		c.mvMu.Unlock()
		t.Fatalf("expected 1 planted ward, got %d", len(hs.wards))
	}
	w := hs.wards[0]
	c.mvMu.Unlock()

	if w.obj == 0 {
		t.Fatal("ward has no prop object id")
	}
	if w.radius != wardOp.Radius {
		t.Errorf("ward radius = %v, want %v", w.radius, wardOp.Radius)
	}
	if want := now + wardOp.Lifetime.At(1); w.until != want {
		t.Errorf("ward until = %v, want %v", w.until, want)
	}

	time.Sleep(50 * time.Millisecond) // let the reader drain the pushes
	mu.Lock()
	var sawCreate, sawFogSync, sawFx bool
	for _, p := range pkts {
		switch p.Cmd {
		case battleproto.CmdCreateObject:
			if proto, _ := p.Args.GetInt("proto"); proto == wantProto {
				if id, _ := p.Args.GetInt("id"); id == w.obj {
					sawCreate = true
				}
			}
		case battleproto.CmdSync:
			if b, ok := p.Args.Assoc["data"].([]byte); ok {
				if syncTypeMask(t, b)&(syncViewRadius|syncTeam) == (syncViewRadius | syncTeam) {
					sawFogSync = true
				}
			}
		case battleproto.CmdEffectStart:
			if fx, _ := p.Args.GetString("fx"); fx == wardOp.TrapFx {
				if owner, _ := p.Args.GetInt("owner"); owner != w.obj {
					t.Errorf("ward fx owner=%d, want the stationary prop %d (a SELF fx would trail the avatar)", owner, w.obj)
				}
				sawFx = true
			}
		}
	}
	mu.Unlock()
	if !sawCreate {
		t.Errorf("no CREATE_OBJECT for the ward prop (proto %d, id %d)", wantProto, w.obj)
	}
	if !sawFogSync {
		t.Error("ward SYNC did not carry VIEW_RADIUS+TEAM (a friendly client would reveal no fog)")
	}
	if !sawFx {
		t.Errorf("no EFFECT_START for the ward ground fx %q", wardOp.TrapFx)
	}

	// Tick past the ward's lifetime: it must expire and delete its prop.
	c.mvMu.Lock()
	s.tickWardsLocked(c, w.until+1)
	remaining := len(hs.wards)
	c.mvMu.Unlock()
	if remaining != 0 {
		t.Fatalf("ward survived past its lifetime (%d left)", remaining)
	}

	time.Sleep(50 * time.Millisecond)
	mu.Lock()
	sawDelete := false
	for _, p := range pkts {
		if p.Cmd == battleproto.CmdDeleteObject {
			if id, _ := p.Args.GetInt("id"); id == w.obj {
				sawDelete = true
			}
		}
	}
	mu.Unlock()
	if !sawDelete {
		t.Errorf("no DELETE_OBJECT for the expired ward prop %d", w.obj)
	}
}

// TestVisionWardRevealsMobs: a mob too far from the player to be rendered becomes shown
// (and unshaded) once a vision ward is planted next to it, and hides again after the ward
// expires. This is «Росток» «открывая участок местности» — showing the mobs on it.
func TestVisionWardRevealsMobs(t *testing.T) {
	s, c, _, sx, sy := newNavConn(t)
	hs := c.huntState
	now := float64(s.battleTime())
	mob := gamedata.Mobs()[2] // skeleton

	// A mob well past the hide radius: with only the player nearby it stays hidden.
	far := &mobState{id: 3200, mobIdx: 2, mob: mob, x: sx + float32(mobHideRadius) + 20, y: sy, hp: mob.Health, homed: true}
	hs.mobs[far.id] = far

	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	s.mobInterestLocked(c, far, now)
	if far.shown {
		t.Fatal("distant mob should be hidden before any ward is planted")
	}

	// Plant a ward right on the mob: it must reveal and fully un-shade it.
	hs.wards = append(hs.wards, wardState{obj: 950001, x: far.x, y: far.y, radius: 35, until: now + 30})
	s.mobInterestLocked(c, far, now)
	if !far.shown {
		t.Fatal("ward should reveal a mob standing inside its radius")
	}
	if far.shaded {
		t.Error("a mob at the ward centre should be fully revealed (not shaded)")
	}

	// Expire the ward: the mob hides again (no player near).
	hs.wards = hs.wards[:0]
	s.mobInterestLocked(c, far, now)
	if far.shown {
		t.Fatal("mob should hide again once the ward is gone")
	}
}

// TestVisionWardRevealsStealthedEnemy: a ward breaks the invisibility of an ENEMY-team
// member standing inside its radius (patch 1.08's «видеть невидимых врагов»), while a
// friendly stealthed member and an out-of-range enemy stay hidden.
func TestVisionWardRevealsStealthedEnemy(t *testing.T) {
	s := New(session.NewStore())
	inst := &huntInstance{s: s, members: map[int32]*conn{}}

	// mkMember builds a bare instance member on team `team` at (x,y), stealthed until +10.
	mkMember := func(objID, team int32, x, y float32) *conn {
		srvC, cliC := net.Pipe()
		t.Cleanup(func() { srvC.Close(); cliC.Close() })
		// drain so a push never blocks the pipe
		go func() { r := battleproto.NewReader(cliC); for { if _, err := r.Read(); err != nil { return } } }()
		c := &conn{Conn: srvC, inst: inst}
		c.objID = objID
		c.x, c.y, c.snapT = x, y, s.battleTime()
		c.huntState = &huntState{
			team: team, worldReady: true,
			mobs: map[int32]*mobState{}, summons: map[int32]*summonState{},
		}
		inst.members[objID] = c
		return c
	}

	now := float64(s.battleTime())
	caster := mkMember(1, 1, 0, 0)            // team 1 (owns the ward)
	nearEnemy := mkMember(2, -1, 5, 0)        // enemy inside radius
	farEnemy := mkMember(3, -1, 100, 0)       // enemy outside radius
	friend := mkMember(4, 1, 5, 0)            // ally inside radius (must NOT be revealed)
	for _, m := range []*conn{nearEnemy, farEnemy, friend} {
		m.huntState.invisibleUntil = now + 10
	}

	w := wardState{obj: 400001, x: 5, y: 0, radius: 35, until: now + 30}
	caster.lock()
	s.revealStealthedEnemiesLocked(caster, w, now)
	caster.unlock()

	if nearEnemy.huntState.invisibleUntil != 0 {
		t.Errorf("in-range enemy still stealthed (invisibleUntil=%g)", nearEnemy.huntState.invisibleUntil)
	}
	if farEnemy.huntState.invisibleUntil == 0 {
		t.Error("out-of-range enemy was revealed but should stay hidden")
	}
	if friend.huntState.invisibleUntil == 0 {
		t.Error("a friendly stealthed member must not be revealed by an ally's ward")
	}
}
