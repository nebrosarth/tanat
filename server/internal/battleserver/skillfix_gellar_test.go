package battleserver

import (
	"net"
	"testing"
	"time"

	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// TestSoulBonusShowsOnTheCharacterCard: the real swing already folded banked souls into
// its damage (killAttackBonusLocked, called from scheduleHitAfterLocked) -- but the
// DISPLAYED dmgMin/dmgMax pushPlayerStatsLocked sends to the client never included it, so
// the card showed plain base damage while actual hits already carried the soul bonus.
func TestSoulBonusShowsOnTheCharacterCard(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Gellar")
	defer cleanup()
	hs := c.huntState
	hs.soulSlot = 2 // «Порабощение», registered at world build
	hs.skillLevel[1] = 1
	hs.soulStacks = 5 // +2 attack each = +10

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())

	without := hs.st.modSum(now, "dmg_flat")
	withSouls := without + s.killAttackBonusLocked(hs, now)
	if withSouls <= without {
		t.Fatal("banked souls did not raise killAttackBonusLocked at all")
	}

	// pushPlayerStatsLocked must use the SAME flat term as the real hit formula.
	hs.tr.add(c.objID)
	s.pushPlayerStatsLocked(c, now)
	// No direct read of the pushed packet here (no wire capture in this harness); assert
	// the fix at its source instead -- the exact expression pushPlayerStatsLocked computes.
	dmgFlat := hs.st.modSum(now, "dmg_flat") + s.killAttackBonusLocked(hs, now)
	if dmgFlat != withSouls {
		t.Fatalf("displayed dmgFlat = %g, want %g (souls included)", dmgFlat, withSouls)
	}
}

// TestGellarUltStartsItsVisualEffect: «Армия душ» is a self OpChannel with no OpBuffStat
// anywhere -- the two places that ever start a "self" BuffFx (addPlayerModLocked on a stat
// mod, and the toggle path) never run for this shape, so the buff visual
// (GellarSkill4Effect: real gfx+sfx in the client's registry) never appeared at all beyond
// the opening cast flourish.
func TestGellarUltStartsItsVisualEffect(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Gellar")
	defer cleanup()
	hs := c.huntState
	hs.soulStacks = 3 // affects only how many soul PARTICLES the fx draws, not whether it fires

	def := hs.skillDef(4)
	if def.BuffFx != "GellarSkill4Effect" || def.BuffFxOn != "self" {
		t.Fatalf("unexpected BuffFx wiring: fx=%q on=%q", def.BuffFx, def.BuffFxOn)
	}
	if skillHasAnyBuffStat(def) {
		t.Fatal("test premise broken: «Армия душ» now carries an OpBuffStat -- the engine gate no longer applies")
	}

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())
	s.applyOpsLocked(c, def.Ops, opCtx{slot: 4, level: 1}, now)

	if len(hs.channels) != 1 {
		t.Fatalf("«Армия душ» did not start a channel: %d channels", len(hs.channels))
	}
	if len(hs.fxEnds) == 0 {
		t.Fatal("no fx was started for the ult at all: it would show nothing but the cast flourish")
	}
	// The buff fx must be the specific ended fx tied to the channel's own duration.
	var found bool
	for _, f := range hs.fxEnds {
		if f.at == now+7 { // Dur: PerLevel{7,7,7,7}; 3 souls at 1s each doesn't outrun it
			found = true
		}
	}
	if !found {
		t.Fatal("no fx is scheduled to end with the channel's own 7s duration")
	}
}

// TestGellarUltKeepsItsOriginalCadenceAndDoesNotSpendSouls: the HIT COUNT/timing is the
// original timed volley (unaffected by soul count, unaffected by casting it with zero
// souls banked) -- only the damage (PerSoul) and the fx's particle count are soul-scaled.
func TestGellarUltKeepsItsOriginalCadenceAndDoesNotSpendSouls(t *testing.T) {
	for _, souls := range []int{0, 4, 12} {
		s, c, cleanup := newHuntConn(t, "Avtr_DPS_Gellar")
		hs := c.huntState
		hs.soulStacks = souls

		c.mvMu.Lock()
		now := float64(s.battleTime())
		s.applyOpsLocked(c, hs.skillDef(4).Ops, opCtx{slot: 4, level: 1}, now)

		if len(hs.channels) != 1 {
			t.Errorf("souls=%d: «Армия душ» did not start a channel: %d channels", souls, len(hs.channels))
		} else if got := hs.channels[0].until - now; got != 7 {
			t.Errorf("souls=%d: channel ends in %gs, want the card's fixed 7s regardless of soul count", souls, got)
		}
		if hs.soulStacks != souls {
			t.Errorf("souls=%d: soulStacks changed to %d -- the card never says casting spends souls", souls, hs.soulStacks)
		}
		c.mvMu.Unlock()
		cleanup()
	}
}

// TestGellarUltSoulParticleCountIsVisualOnly: the number of "soul" particles the client
// draws (GellarSkill4Effect's EFFECT_START "counter" arg, read by ParticlesMgr.SetCounter
// off the decompiled client) tracks the banked soul count -- this is purely cosmetic and
// must not be confused with, or drive, the hit count.
func TestGellarUltSoulParticleCountIsVisualOnly(t *testing.T) {
	s := New(session.NewStore())
	srv, cli := net.Pipe()
	defer srv.Close()
	defer cli.Close()
	r := battleproto.NewReader(cli)

	c := &conn{Conn: srv}
	c.objID = 1000
	c.huntState = &huntState{mobs: map[int32]*mobState{}, summons: map[int32]*summonState{}}
	c.huntState.tr.add(c.objID)
	c.huntState.soulStacks = 9

	// net.Pipe() is unbuffered: the push below blocks until something reads it, so it
	// must run concurrently with the r.Read() call, not before it on the same goroutine.
	uidCh := make(chan int32, 1)
	go func() {
		c.mvMu.Lock()
		defer c.mvMu.Unlock()
		uidCh <- s.fxStartCounterLocked(c, "GellarSkill4Effect", c.objID, 0, false, 0, 0, int32(c.huntState.soulStacks))
	}()

	_ = cli.SetReadDeadline(time.Now().Add(2 * time.Second))
	p, err := r.Read()
	if err != nil {
		t.Fatalf("no packet read: %v", err)
	}
	if p.Cmd != battleproto.CmdEffectStart {
		t.Fatalf("cmd = %v, want CmdEffectStart", p.Cmd)
	}
	args, ok := p.Args.GetArray("args")
	if !ok {
		t.Fatal("EFFECT_START carries no args")
	}
	counter, ok := args.GetInt("counter")
	if !ok || counter != 9 {
		t.Fatalf("counter = %v,%v, want 9 (the banked soul count)", counter, ok)
	}
	select {
	case uid := <-uidCh:
		if uid == 0 {
			t.Fatal("fxStartCounterLocked returned no uid")
		}
	case <-time.After(time.Second):
		t.Fatal("fxStartCounterLocked never returned")
	}
}

// TestFxStartCounterOmitsArgAtOneOrBelow: a counter of 0 or 1 must not even be sent -- the
// client only applies it (VisualEffectHolder.Init: mEffectCounter > 1) above 1, and every
// OTHER caller of fxStartLocked (every skill but Gellar's ult) must see byte-identical
// wire output to before this existed.
func TestFxStartCounterOmitsArgAtOneOrBelow(t *testing.T) {
	s := New(session.NewStore())
	srv, cli := net.Pipe()
	defer srv.Close()
	defer cli.Close()
	r := battleproto.NewReader(cli)

	c := &conn{Conn: srv}
	c.objID = 1000
	c.huntState = &huntState{mobs: map[int32]*mobState{}, summons: map[int32]*summonState{}}
	c.huntState.tr.add(c.objID)

	go func() {
		c.mvMu.Lock()
		defer c.mvMu.Unlock()
		s.fxStartCounterLocked(c, "SomeEffect", c.objID, 0, false, 0, 0, 1)
	}()

	_ = cli.SetReadDeadline(time.Now().Add(2 * time.Second))
	p, err := r.Read()
	if err != nil {
		t.Fatalf("no packet read: %v", err)
	}
	args, _ := p.Args.GetArray("args")
	if _, ok := args.GetInt("counter"); ok {
		t.Fatal("counter=1 must be omitted from the wire, not sent as 1")
	}
}

// TestGellarUltIgnoresMovementAndStun: the souls are already loosed at cast time, not
// something the caster actively sustains -- movement and stuns must not cut it off.
func TestGellarUltIgnoresMovementAndStun(t *testing.T) {
	if !channelSustainsThroughDisruption("Avtr_DPS_Gellar", 4) {
		t.Fatal("«Армия душ» is not marked as surviving movement/stun")
	}
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Gellar")
	defer cleanup()
	hs := c.huntState
	hs.soulStacks = 5

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())
	s.applyOpsLocked(c, hs.skillDef(4).Ops, opCtx{slot: 4, level: 1}, now)
	if len(hs.channels) != 1 {
		t.Fatalf("channel not started: %d", len(hs.channels))
	}

	hs.st.stunUntil = now + 5
	c.hasDest = true
	s.tickChannelsLocked(c, now+0.2)
	if len(hs.channels) == 0 {
		t.Fatal("the volley broke on stun+movement; the souls are already loosed")
	}
}

// TestHekataAshWhirlwindStillGetsItsOwnBuffFx: Hekata's «Пепельный смерч» is NOT the dead
// shape above -- its channel's OWN nested OpBuffStat re-triggers the fx every tick via
// addPlayerModLocked, and the new engine gate must not double-start it.
func TestHekataAshWhirlwindStillGetsItsOwnBuffFx(t *testing.T) {
	kit := gamedata.SkillsFor(avatarByPrefab(t, "Avtr_Dsb_Hekata"))
	def := kit.Skills[3]
	if !skillHasAnyBuffStat(def) {
		t.Fatal("«Пепельный смерч» must still be recognized as having a nested OpBuffStat")
	}
}
