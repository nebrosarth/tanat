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

// newCaptureConn builds a solo hunt conn (avatar by prefab) whose pushed battle
// packets are captured for inspection. Mirrors newTitanidCapture but parameterized.
func newCaptureConn(t *testing.T, prefab string) (*Server, *conn, *huntState, *sync.Mutex, *[]battleproto.Packet) {
	t.Helper()
	s := New(session.NewStore())
	srv, cli := net.Pipe()
	t.Cleanup(func() { srv.Close(); cli.Close() })

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

	av := avatarByPrefab(t, prefab)
	c := &conn{Conn: srv}
	c.objID = 1000
	c.x, c.y, c.snapT = 0, 0, s.battleTime()
	hs := &huntState{
		av: av, kit: gamedata.SkillsFor(av),
		mobs: map[int32]*mobState{}, summons: map[int32]*summonState{},
		hp: 2000, mana: 300,
	}
	for i := range hs.skillLevel {
		hs.skillLevel[i] = 1 // learn everything, incl. the level-gated ult (slot 4)
	}
	hs.tr.add(c.objID)
	c.huntState = hs
	t.Cleanup(func() { c.mvMu.Lock(); hs.closed = true; c.mvMu.Unlock() })
	return s, c, hs, &mu, &pkts
}

// TestVelialUltTargetBuffFxData pins the data contract behind the fix: Velial's ult
// «Трибунал» (slot 4) authors a target-mode BuffFx and a top-level target stat-buff,
// so targetBuffTTL reports its real lifetime. Without both, the debuff would be
// invisible on the struck mob (the bug this fixes).
func TestVelialUltTargetBuffFxData(t *testing.T) {
	vel := avatarByPrefab(t, "Avtr_Tank_Velial")
	ult := gamedata.SkillsFor(vel).Skills[3]

	if ult.BuffFxOn != "target" {
		t.Fatalf("Velial ult BuffFxOn = %q, want \"target\"", ult.BuffFxOn)
	}
	if ult.BuffFx == "" {
		t.Fatal("Velial ult has no BuffFx -- nothing to show on the debuffed mob")
	}
	if ttl := targetBuffTTL(ult, 1); ttl != 30 {
		t.Fatalf("targetBuffTTL(Velial ult, rank 1) = %g, want 30 (the armor-break duration)", ttl)
	}

	// A skill whose only stat-buff is on SELF (not the target) must report 0, so no
	// target BuffFx is ever pinned on a mob for it.
	for _, sk := range gamedata.SkillsFor(vel).Skills {
		if sk.BuffFxOn == "self" {
			if ttl := targetBuffTTL(sk, 1); ttl != 0 {
				t.Fatalf("targetBuffTTL for self-buff %q = %g, want 0", sk.NameRu, ttl)
			}
			break
		}
	}
}

// TestVelialUltShowsDebuffFxOnMob is the headline for the reported bug ("на крипе не
// отображается дебаф от 4 скилла Велиала"): casting the ult on a crypt mob must emit
// an EFFECT_START of VelialSkill4Effect PARENTED to that mob (owner=mob id), and
// schedule its removal after the 30s armor break -- while also actually breaking the
// armor (visual and mechanic land together).
func TestVelialUltShowsDebuffFxOnMob(t *testing.T) {
	s, c, hs, mu, pkts := newCaptureConn(t, "Avtr_Tank_Velial")

	const idx = 6 // Cerber crypt boss -- an armored roster entry
	base := gamedata.Mobs()[idx].PhysArmor
	if base <= 0 {
		t.Fatalf("precondition: roster mob %d should carry armor, got %g", idx, base)
	}
	mob := &mobState{id: 2000, mobIdx: idx, mob: gamedata.Mobs()[idx], hp: 100000, x: 2, y: 0, shown: true}

	c.mvMu.Lock()
	hs.mobs[mob.id] = mob
	hs.tr.add(mob.id)
	now := float64(s.battleTime())
	nEndsBefore := len(hs.fxEnds)
	// Fire the ult payload directly on the mob (skips the approach/cast wind-up).
	s.firePayloadLocked(c, payload{slot: 4, level: 1, target: mob.id}, now)
	brokenArmor := mob.physArmor(now)

	// The BuffFx must be scheduled to end after the debuff's own 30s TTL.
	var scheduledEnd bool
	for _, e := range hs.fxEnds[nEndsBefore:] {
		if e.at >= now+30-1e-6 {
			scheduledEnd = true
		}
	}
	c.mvMu.Unlock()

	// Mechanic: the armor is actually broken (so the visual isn't a lie).
	if brokenArmor >= base {
		t.Fatalf("ult did not break armor: %g >= base %g", brokenArmor, base)
	}
	if !scheduledEnd {
		t.Fatalf("Velial ult BuffFx was not scheduled to end after its 30s TTL (fxEnds=%v)", hs.fxEnds)
	}

	// Visual: an EFFECT_START of VelialSkill4Effect owned by the mob reached the client.
	time.Sleep(50 * time.Millisecond)
	mu.Lock()
	defer mu.Unlock()
	var sawFx bool
	for _, p := range *pkts {
		if p.Cmd != battleproto.CmdEffectStart {
			continue
		}
		if fx, _ := p.Args.GetString("fx"); fx == "VelialSkill4Effect" {
			owner, _ := p.Args.GetInt("owner")
			if owner != mob.id {
				t.Fatalf("VelialSkill4Effect owner=%d, want the debuffed mob %d (must parent to the target, not the caster)", owner, mob.id)
			}
			sawFx = true
		}
	}
	if !sawFx {
		t.Fatal("no EFFECT_START for VelialSkill4Effect on the mob -- the ult debuff is invisible (the reported bug)")
	}
}
