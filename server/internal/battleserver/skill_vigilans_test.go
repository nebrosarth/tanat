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

// TestVigilansUltDashesAndDropsStationaryBarrier pins the two reported bugs on
// Vigilans' ult "Супружеский долг" (slot 4):
//
//  1. It must DASH to the target (a visible lunge the client animates), not BLINK
//     (an instant server-side teleport the client never mirrors). With a blink the
//     avatar stood still on screen while the server placed her next to the target
//     and let her attack from range -- a position desync. A dash leaves the avatar
//     at her start position server-side and issues a real move leg toward the mob.
//
//  2. The barrier VFX (BuffFx) must NOT be parented to the avatar (which dragged
//     it around). The prefab is a SELF-mode VFX that renders only while parented to
//     an owner GameObject, so it is anchored to the ROOTED target (owner = the mob
//     id): the ult roots the enemy for the same duration, so the barrier sits still
//     on the trapped enemy. Owner = the avatar (follows) or -1 (invisible) are both
//     wrong.
func TestVigilansUltDashesAndDropsStationaryBarrier(t *testing.T) {
	s := New(session.NewStore())
	srv, cli := net.Pipe()
	defer srv.Close()
	defer cli.Close()

	// Concurrent reader: net.Pipe writes block until read, so the server's pushes
	// during applyOpsLocked would deadlock without a live consumer. Decode packets
	// into a guarded slice.
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

	vig, ok := gamedata.AvatarByID(20) // Avtr_HK_Vigilans
	if !ok {
		t.Fatal("Vigilans (id 20) missing")
	}
	c := &conn{Conn: srv}
	c.objID = 1000
	c.x, c.y, c.snapT = 0, 0, s.battleTime() // start at the origin
	kit := gamedata.SkillsFor(vig)
	hs := &huntState{
		av:      vig,
		kit:     kit,
		mobs:    map[int32]*mobState{},
		summons: map[int32]*summonState{},
		hp:      500, mana: 200,
	}
	hs.tr.add(c.objID)
	mob := &mobState{id: 2000, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 400}
	mob.x, mob.y = 6, 0 // several units away: dash target
	hs.tr.add(mob.id)
	hs.mobs[mob.id] = mob
	c.huntState = hs
	defer func() { c.mvMu.Lock(); hs.closed = true; c.mvMu.Unlock() }()

	now := float64(s.battleTime())
	c.mvMu.Lock()
	ctx := opCtx{slot: 4, level: 1, target: mob}
	s.applyOpsLocked(c, kit.Skills[3].Ops, ctx, now) // the whole ult batch
	// Snapshot the movement state under the lock.
	dashUntil := hs.dashUntil
	hasDest := c.hasDest
	destX := c.destX
	selfX, selfY := c.x, c.y
	hpAfterCast := mob.hp
	// --- Fix 3: damage lands on ARRIVAL, not on cast. The ops after the dash are
	// deferred to a payload at dashUntil; run it to simulate the lunge landing.
	s.runDuePayloadsLocked(c, dashUntil+0.01)
	hpAfterArrival := mob.hp
	c.mvMu.Unlock()

	// --- Fix 1: dash, not blink ---
	if !hasDest || dashUntil <= now {
		t.Fatalf("ult did not start a dash (hasDest=%v dashUntil=%.2f now=%.2f) — a blink would teleport with no move leg",
			hasDest, dashUntil, now)
	}
	if selfX != 0 || selfY != 0 {
		t.Fatalf("avatar was teleported to (%.1f,%.1f) — a dash must leave her at the start and walk a move leg", selfX, selfY)
	}
	if destX < 3 { // moving toward the mob at x=6, not standing still
		t.Fatalf("dash destination x=%.1f does not head to the target at x=6", destX)
	}

	// --- Fix 3: no damage on cast, damage on arrival ---
	if hpAfterCast != 400 {
		t.Errorf("mob took %.0f damage on CAST, want 0 — the ult must strike only on arrival", 400-hpAfterCast)
	}
	if hpAfterArrival >= 400 {
		t.Errorf("mob took no damage on arrival (hp=%.0f) — the deferred strike never landed", hpAfterArrival)
	}

	// --- Fix 2: barrier anchored to the ROOTED target (owner = the mob), NOT the
	// avatar. The prefab's barrier gfx is SELF-mode (parents to and follows its owner
	// GameObject; a targetPos is ignored), confirmed on the client -- owner=caster
	// made it trail Vigilans. The rooted enemy is the only stationary anchor.
	time.Sleep(50 * time.Millisecond) // let the reader drain any trailing push
	mu.Lock()
	defer mu.Unlock()
	var found bool
	for _, p := range pkts {
		if p.Cmd != battleproto.CmdEffectStart {
			continue
		}
		if fx, _ := p.Args.GetString("fx"); fx != "VigilansSkill4Effect" {
			continue
		}
		found = true
		owner, _ := p.Args.GetInt("owner")
		if owner == c.objID {
			t.Fatalf("barrier EFFECT_START owner=%d is the avatar — a SELF-mode gfx would follow her", owner)
		}
		if owner != mob.id {
			t.Fatalf("barrier EFFECT_START owner=%d, want the rooted target %d", owner, mob.id)
		}
	}
	if !found {
		t.Fatal("ult emitted no VigilansSkill4Effect barrier EFFECT_START")
	}
}

// TestVigilansUltDeferredStrikeSurvivesTickLoop reproduces the LIVE server flow the
// direct-applyOpsLocked test above missed: the ult's first payload is fired from
// INSIDE runDuePayloadsLocked (the combat tick), where the OpDash appends the
// deferred strike to hs.payloads. A naive runDuePayloadsLocked that overwrites the
// queue with a not-yet-due snapshot drops that append -- so the charge dashes but
// never damages/roots/barriers. This pins that the deferred strike survives and lands.
func TestVigilansUltDeferredStrikeSurvivesTickLoop(t *testing.T) {
	s := New(session.NewStore())
	srv, cli := net.Pipe()
	defer srv.Close()
	defer cli.Close()
	go func() { // raw drain so net.Pipe writes never block
		buf := make([]byte, 4096)
		for {
			if _, err := cli.Read(buf); err != nil {
				return
			}
		}
	}()

	vig, _ := gamedata.AvatarByID(20)
	c := &conn{Conn: srv}
	c.objID = 1000
	c.x, c.y, c.snapT = 0, 0, s.battleTime()
	kit := gamedata.SkillsFor(vig)
	hs := &huntState{
		av: vig, kit: kit,
		mobs: map[int32]*mobState{}, summons: map[int32]*summonState{},
		hp: 500, mana: 200,
	}
	hs.tr.add(c.objID)
	mob := &mobState{id: 2000, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 400}
	mob.x, mob.y = 6, 0
	hs.tr.add(mob.id)
	hs.mobs[mob.id] = mob
	c.huntState = hs
	defer func() { c.mvMu.Lock(); hs.closed = true; c.mvMu.Unlock() }()

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())

	// Queue the ult's first payload exactly as execCastLocked does (PayloadDelay), then
	// let the tick's runDuePayloadsLocked fire it -- this is the path that dropped the
	// deferred strike.
	hs.payloads = append(hs.payloads, payload{at: now, slot: 4, level: 1, target: mob.id})
	s.runDuePayloadsLocked(c, now)

	if mob.hp != 400 {
		t.Errorf("mob took %.0f damage before arrival, want 0", 400-mob.hp)
	}
	if len(hs.payloads) != 1 {
		t.Fatalf("deferred strike was dropped by the tick loop: %d payloads queued, want 1", len(hs.payloads))
	}

	dashUntil := hs.dashUntil
	s.runDuePayloadsLocked(c, dashUntil+0.01) // arrival tick
	if mob.hp >= 400 {
		t.Errorf("deferred strike never landed after arrival (hp=%.0f)", mob.hp)
	}
	if mob.st.rootUntil <= now {
		t.Error("deferred strike did not root the target")
	}
}

// TestVigilansUltResumesAutoAttackAfterCharge: the action-done's auto-attack resume
// fires mid-dash (hasDest set) and bails, so the charge carries its own resume in the
// deferred strike -- once the lunge lands the avatar rolls back into auto-attacking
// the target it just slammed.
func TestVigilansUltResumesAutoAttackAfterCharge(t *testing.T) {
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

	vig, _ := gamedata.AvatarByID(20)
	c := &conn{Conn: srv}
	c.objID = 1000
	c.x, c.y, c.snapT = 0, 0, s.battleTime()
	hs := &huntState{
		av: vig, kit: gamedata.SkillsFor(vig),
		mobs: map[int32]*mobState{}, summons: map[int32]*summonState{},
		hp: 500, mana: 200,
	}
	hs.tr.add(c.objID)
	mob := &mobState{id: 2000, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 400}
	mob.x, mob.y = 1, 0 // where she lands: within melee reach after the charge
	hs.tr.add(mob.id)
	hs.mobs[mob.id] = mob
	c.huntState = hs
	defer func() { c.mvMu.Lock(); hs.closed = true; c.mvMu.Unlock() }()

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())

	// The deferred strike payload as the dash schedules it, carrying resume.
	dashUntil := now + 0.5
	hs.dashUntil = dashUntil
	hs.payloads = append(hs.payloads, payload{
		at: dashUntil, slot: 4, level: 1, target: mob.id,
		ops:    []gamedata.Op{{Kind: gamedata.OpDamage, Value: gamedata.PerLevel{170, 230, 290, 350}, Scale: "magic", PerSP: 1}},
		resume: true,
	})
	// Simulate the arrival timer having cleared the dash: she is now standing still.
	c.hasDest = false

	if hs.attackTarget != 0 {
		t.Fatalf("precondition: attackTarget=%d, want 0", hs.attackTarget)
	}
	s.runDuePayloadsLocked(c, dashUntil+0.01)

	if hs.attackTarget != mob.id {
		t.Errorf("avatar did not resume auto-attack after the charge: attackTarget=%d, want %d", hs.attackTarget, mob.id)
	}
}

// TestVigilansUltBarrierRecordsCorpseAnchor: the ult records a corpse-anchor time on
// the struck target so, if the ult kills it, hitMobLocked holds the body until the
// barrier expires (the SELF-mode VFX would otherwise orphan onto the caster when the
// corpse is deleted at 4s -- the intermittent "barrier follows me"). Also pins that
// the barrier op runs BEFORE the damage op, so the anchor is set on a LIVE mob.
func TestVigilansUltBarrierRecordsCorpseAnchor(t *testing.T) {
	// Data pin: OpBuffStat (the barrier) must precede OpDamage in the ult.
	vig, _ := gamedata.AvatarByID(20)
	ult := gamedata.SkillsFor(vig).Skills[3].Ops
	buffAt, dmgAt := -1, -1
	for i, op := range ult {
		if op.Kind == gamedata.OpBuffStat && buffAt < 0 {
			buffAt = i
		}
		if op.Kind == gamedata.OpDamage && dmgAt < 0 {
			dmgAt = i
		}
	}
	if buffAt < 0 || dmgAt < 0 || buffAt > dmgAt {
		t.Fatalf("ult op order wrong: barrier(OpBuffStat)@%d must come before damage@%d", buffAt, dmgAt)
	}

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

	c := &conn{Conn: srv}
	c.objID = 1000
	c.x, c.y, c.snapT = 0, 0, s.battleTime()
	hs := &huntState{
		av: vig, kit: gamedata.SkillsFor(vig),
		mobs: map[int32]*mobState{}, summons: map[int32]*summonState{},
		hp: 500, mana: 200,
	}
	hs.tr.add(c.objID)
	mob := &mobState{id: 2000, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 5000} // survives, so st is readable after
	mob.x, mob.y = 6, 0
	hs.tr.add(mob.id)
	hs.mobs[mob.id] = mob
	c.huntState = hs
	defer func() { c.mvMu.Lock(); hs.closed = true; c.mvMu.Unlock() }()

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())
	s.applyOpsLocked(c, hs.kit.Skills[3].Ops, opCtx{slot: 4, level: 1, target: mob}, now)
	s.runDuePayloadsLocked(c, hs.dashUntil+0.01) // arrival: root + barrier + damage

	if mob.st.anchorFxUntil <= now {
		t.Errorf("barrier did not record a corpse-anchor on the target (anchorFxUntil=%.2f, now=%.2f)", mob.st.anchorFxUntil, now)
	}
}
