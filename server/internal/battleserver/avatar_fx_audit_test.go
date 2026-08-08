package battleserver

import (
	"sync"
	"testing"
	"time"

	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
)

// This file pins the 31-finding client-fx-registry audit's fixes: the SAME bug class
// as the Einzenhaim/Morlokay/Edilia/Kiona/Miriam/Sharli/Frost fixes (see
// trap_anchor.go), found for 19 more heroes by systematically re-running the same
// methodology (fxreg.py against the baked client registry) over the rest of the
// roster.

// TestPayloadFxUsesAnchorAudit pins the new payloadFxUsesAnchor entries (ArianaMey
// slot 3, Cerber slot 1): both SELF-baked ground fx that trailed the caster instead of
// staying at the cast point.
func TestPayloadFxUsesAnchorAudit(t *testing.T) {
	cases := []struct {
		prefab string
		slot   int
		want   bool
	}{
		{"Avtr_Sp_Arianna", 3, true},
		{"Avtr_DPS_Cerber", 1, true},
		{"Avtr_Tank_Rognar", 2, false}, // negative control
	}
	for _, c := range cases {
		if got := payloadFxUsesAnchor(c.prefab, c.slot); got != c.want {
			t.Errorf("payloadFxUsesAnchor(%s, %d) = %v, want %v", c.prefab, c.slot, got, c.want)
		}
	}
}

// TestPayloadTargetFxOwnedToTargetAudit pins the new payloadTargetFxOwnedToTarget
// entries found across the audit: each is a SELF-baked payload fx sent with
// PayloadFxAt="target" that rendered on the CASTER instead of the struck enemy.
func TestPayloadTargetFxOwnedToTargetAudit(t *testing.T) {
	cases := []struct {
		prefab string
		slot   int
	}{
		{"Avtr_Sp_Arianna", 4},
		{"Avtr_DPS_Cerber", 4},
		{"Avtr_Tank_Gektor", 4},
		{"Avtr_Sp_Inshari", 4},
		{"Avtr_Sp_Neirofim", 1},
		{"Avtr_Tank_Rognar", 1},
		{"Avtr_Tank_Rognar", 4},
		{"Avtr_DPS_Sandariel", 1},
		{"Avtr_HK_ShinDalar", 4},
		{"Avtr_HK_Tangren", 4},
	}
	for _, c := range cases {
		if !payloadTargetFxOwnedToTarget(c.prefab, c.slot) {
			t.Errorf("payloadTargetFxOwnedToTarget(%s, %d) = false, want true", c.prefab, c.slot)
		}
	}
	if payloadTargetFxOwnedToTarget("Avtr_Tank_Rognar", 3) {
		t.Error("payloadTargetFxOwnedToTarget(Rognar, 3) = true, want false (negative control)")
	}
}

// TestTrapUsesAnchorAudit pins the new trapUsesAnchor entries (Dutnik's two mines,
// Mihalych's trap): SELF-baked ground fx that followed the caster instead of sitting
// at the planted trap location.
func TestTrapUsesAnchorAudit(t *testing.T) {
	for _, c := range []struct {
		prefab string
		slot   int
	}{
		{"Avtr_HK_Dutnik", 1},
		{"Avtr_HK_Dutnik", 2},
		{"Avtr_HK_Mihalych", 3},
	} {
		if !trapUsesAnchor(c.prefab, c.slot) {
			t.Errorf("trapUsesAnchor(%s, %d) = false, want true", c.prefab, c.slot)
		}
	}
	if trapUsesAnchor("Avtr_HK_Dutnik", 3) {
		t.Error("trapUsesAnchor(Dutnik, 3) = true, want false (negative control)")
	}
}

// TestAstarotUsesDashNotBlink mirrors TestShinDalarKitShape: OpBlink never visibly
// teleports the client avatar (SmoothErrorCorrector caps catch-up speed while
// standing), so Astarot's «Бесовский трюк» (slot 1) must move him via OpDash.
func TestAstarotUsesDashNotBlink(t *testing.T) {
	av, ok := gamedata.AvatarByID(7) // Astarot
	if !ok {
		t.Fatal("Astarot (id 7) missing")
	}
	kit := gamedata.SkillsFor(av)
	var hasDash bool
	for _, op := range kit.Skills[0].Ops { // slot 1
		if op.Kind == gamedata.OpDash {
			hasDash = true
		}
		if op.Kind == gamedata.OpBlink {
			t.Error("Astarot slot 1 still uses OpBlink -- the client won't teleport, it slides")
		}
	}
	if !hasDash {
		t.Error("Astarot slot 1 «Бесовский трюк» must move the avatar via OpDash")
	}
}

// TestAbominatorFlingHasFlightSpeed pins PayloadFlightSpeed on slot 1: the client's
// own prop is a real SmoothMove(bySpeed=true, speed=40) flying chunk, matching the
// skill's own Distance:10, so the stun/damage/slow must wait for it to actually arrive.
func TestAbominatorFlingHasFlightSpeed(t *testing.T) {
	av, ok := gamedata.AvatarByID(22) // Abominator
	if !ok {
		t.Fatal("Abominator (id 22) missing")
	}
	sk := gamedata.SkillsFor(av).Skills[0] // slot 1
	if sk.PayloadFlightSpeed != 40 {
		t.Errorf("Abominator slot 1 PayloadFlightSpeed = %g, want 40", sk.PayloadFlightSpeed)
	}
}

// TestNerlagAxeThrowUsesThrowMode: both NerlagSkill1Effect gfx sub-effects are baked
// SELF_TO_TARGET (need a real target OBJECT to fly to), so PayloadFxAt must be "throw"
// (which anchors an object at the click point), not "point" (which always sends
// target=0). PayloadFlight must be a real nonzero number, or the axes never travel.
func TestNerlagAxeThrowUsesThrowMode(t *testing.T) {
	av, ok := gamedata.AvatarByID(15) // Nerlag
	if !ok {
		t.Fatal("Nerlag (id 15) missing")
	}
	sk := gamedata.SkillsFor(av).Skills[0] // slot 1
	if sk.PayloadFxAt != "throw" {
		t.Errorf("Nerlag slot 1 PayloadFxAt = %q, want \"throw\"", sk.PayloadFxAt)
	}
	if sk.PayloadFlight <= 0 {
		t.Errorf("Nerlag slot 1 PayloadFlight = %g, want > 0", sk.PayloadFlight)
	}
}

// TestNerlagBloodRushHasFx pins the slot-3 passive's fx wiring: the client's baked
// registry has a dedicated one-shot blood-burst (NerlagSkill3Effect, verified SELF-mode
// off the registry) that was simply never referenced.
func TestNerlagBloodRushHasFx(t *testing.T) {
	av, ok := gamedata.AvatarByID(15)
	if !ok {
		t.Fatal("Nerlag (id 15) missing")
	}
	sk := gamedata.SkillsFor(av).Skills[2] // slot 3
	if sk.PayloadFx == "" || sk.PayloadFxAt != "self" {
		t.Errorf("Nerlag slot 3 PayloadFx/PayloadFxAt = %q/%q, want a non-empty fx at \"self\"", sk.PayloadFx, sk.PayloadFxAt)
	}
}

// TestShinDalarDashCleaveHasNoDeadPayloadFx: the registry's "ShinDalarSkill1Effect"
// carries zero gfx/sfx -- referencing it renders nothing, so it must be cleared rather
// than left pointing at an empty entry.
func TestShinDalarDashCleaveHasNoDeadPayloadFx(t *testing.T) {
	av, ok := gamedata.AvatarByID(16)
	if !ok {
		t.Fatal("ShinDalar (id 16) missing")
	}
	sk := gamedata.SkillsFor(av).Skills[0] // slot 1
	if sk.PayloadFx != "" {
		t.Errorf("ShinDalar slot 1 PayloadFx = %q, want empty (registry entry has no gfx/sfx)", sk.PayloadFx)
	}
}

// TestRognarGraveColdIsSelfAnchored: a self-cast (Target:"") that picks 2 RANDOM
// enemies inside OpDot/OpSlow's own resolution never has a single ctx.target for a
// "target"-mode placement to hang on. The registry agrees: RognarSkill3Effect2 (the
// BuffFx) is SELF-mode, an aura on Rognar himself, not a per-target debuff prop.
func TestRognarGraveColdIsSelfAnchored(t *testing.T) {
	av, ok := gamedata.AvatarByID(1) // Rognar
	if !ok {
		t.Fatal("Rognar (id 1) missing")
	}
	sk := gamedata.SkillsFor(av).Skills[2] // slot 3
	if sk.PayloadFxAt != "self" {
		t.Errorf("Rognar slot 3 PayloadFxAt = %q, want \"self\"", sk.PayloadFxAt)
	}
	if sk.BuffFxOn != "self" {
		t.Errorf("Rognar slot 3 BuffFxOn = %q, want \"self\"", sk.BuffFxOn)
	}
}

// TestUrgTreeFormGrantsTargetBuffTTL: targetBuffTTL must recognize a top-level
// OpTreeForm (Urg's «Древесный камуфляж») as an encase-like state worth a target
// BuffFx, the same way it already treats OpStun/OpRoot -- otherwise the tree-
// transformation visual never starts on the disguised ally.
func TestUrgTreeFormGrantsTargetBuffTTL(t *testing.T) {
	av, ok := gamedata.AvatarByID(12) // Urg
	if !ok {
		t.Fatal("Urg (id 12) missing")
	}
	sk := gamedata.SkillsFor(av).Skills[0] // slot 1
	if sk.BuffFxOn != "target" || sk.BuffFx == "" {
		t.Fatalf("Urg slot 1 BuffFxOn/BuffFx = %q/%q, want a target-mode buff fx", sk.BuffFxOn, sk.BuffFx)
	}
	if ttl := targetBuffTTL(sk, 1); ttl != 8 {
		t.Errorf("targetBuffTTL(Urg slot 1, rank 1) = %g, want 8 (OpTreeForm's own Dur)", ttl)
	}
}

// TestActiveSelfBuffTTLCoversNonBuffStatSkills: Rognar's «Канал смерти» (OpDeathLink)
// and Urg's «Непроглядные дебри» (OpSilence+OpGrove) both declare a self-mode BuffFx
// with no OpBuffStat/OpChannel anywhere in their ops -- the only two paths that
// normally start one. activeSelfBuffTTL is the fallback duration for exactly this
// shape; it must read a real number from whichever op actually carries one.
func TestActiveSelfBuffTTLCoversNonBuffStatSkills(t *testing.T) {
	rognar, ok := gamedata.AvatarByID(1)
	if !ok {
		t.Fatal("Rognar (id 1) missing")
	}
	s4 := gamedata.SkillsFor(rognar).Skills[3]
	if s4.BuffFxOn != "self" || s4.BuffFx == "" {
		t.Fatalf("Rognar slot 4 BuffFxOn/BuffFx = %q/%q, want a self-mode buff fx", s4.BuffFxOn, s4.BuffFx)
	}
	if skillHasAnyBuffStat(s4) || skillHasChannel(s4) {
		t.Fatal("Rognar slot 4 unexpectedly has OpBuffStat/OpChannel -- the addPlayerModLocked/channel path would already cover it")
	}
	if ttl := activeSelfBuffTTL(s4, 1); ttl != 10 {
		t.Errorf("activeSelfBuffTTL(Rognar slot 4, rank 1) = %g, want 10 (OpDeathLink's own Dur)", ttl)
	}

	urg, ok := gamedata.AvatarByID(12)
	if !ok {
		t.Fatal("Urg (id 12) missing")
	}
	s4u := gamedata.SkillsFor(urg).Skills[3]
	if s4u.BuffFxOn != "self" || s4u.BuffFx == "" {
		t.Fatalf("Urg slot 4 BuffFxOn/BuffFx = %q/%q, want a self-mode buff fx", s4u.BuffFxOn, s4u.BuffFx)
	}
	if skillHasAnyBuffStat(s4u) || skillHasChannel(s4u) {
		t.Fatal("Urg slot 4 unexpectedly has OpBuffStat/OpChannel")
	}
	if ttl := activeSelfBuffTTL(s4u, 1); ttl <= 0 {
		t.Errorf("activeSelfBuffTTL(Urg slot 4, rank 1) = %g, want > 0 (OpSilence/OpGrove's own Dur)", ttl)
	}
}

// awaitEffectStart polls *pkts briefly (the capture goroutine reads asynchronously)
// and reports whether any EFFECT_START matched `want` fired, running check(p) on it.
func awaitEffectStartFx(t *testing.T, mu *sync.Mutex, pkts *[]battleproto.Packet, fx string, check func(p battleproto.Packet)) bool {
	t.Helper()
	time.Sleep(50 * time.Millisecond)
	mu.Lock()
	defer mu.Unlock()
	found := false
	for _, p := range *pkts {
		if p.Cmd != battleproto.CmdEffectStart {
			continue
		}
		got, _ := p.Args.GetString("fx")
		if got != fx {
			continue
		}
		found = true
		if check != nil {
			check(p)
		}
	}
	return found
}

// TestExecCastStartsSelfBuffFxWithoutBuffStat drives the real cast pipeline for
// Rognar's «Канал смерти» end to end and checks the fx actually gets pushed: without
// the execCastLocked fallback, RognarSkill4Effect1 (the 10s death-link visual) was
// configured but never started.
func TestExecCastStartsSelfBuffFxWithoutBuffStat(t *testing.T) {
	s, c, hs, mu, pkts := newCaptureConn(t, "Avtr_Tank_Rognar")
	c.mvMu.Lock()
	victim := mkMob(t, 2000, 5, 0)
	hs.mobs[victim.id] = victim
	hs.tr.add(victim.id)
	s.execCastLocked(c, 4, victim, victim.x, victim.y, true, 0)
	c.mvMu.Unlock()

	if !awaitEffectStartFx(t, mu, pkts, "RognarSkill4Effect1", nil) {
		t.Error("RognarSkill4Effect1 (self-mode BuffFx) was never started by the cast")
	}
}

// TestSelfPayloadFxResolvesAgainstCaster pins the general firePayloadLocked "self"
// case fix: target=c.objID (not 0), so a registry entry that ALSO carries a
// TARGET/SELF_TO_TARGET sub-effect (Inshari's «Тёмное правосудие», an untargeted
// radius nova) has something to resolve its endpoint against instead of rendering at
// the world origin.
func TestSelfPayloadFxResolvesAgainstCaster(t *testing.T) {
	s, c, _, mu, pkts := newCaptureConn(t, "Avtr_Sp_Inshari")
	c.mvMu.Lock()
	c.x, c.y, c.snapT = 12, 7, s.battleTime()
	// Fire the payload directly (bypasses PayloadDelay's queueing -- the placement fix
	// under test lives in firePayloadLocked's "self" case, not in the cast/delay path).
	s.firePayloadLocked(c, payload{slot: 1, level: 1}, float64(s.battleTime()))
	c.mvMu.Unlock()

	sawOwner := false
	if !awaitEffectStartFx(t, mu, pkts, "InshariSkill1Effect2", func(p battleproto.Packet) {
		owner, _ := p.Args.GetInt("owner")
		args, hasArgs := p.Args.GetArray("args")
		if !hasArgs {
			return
		}
		target, hasTarget := args.GetInt("target")
		if owner == c.objID && hasTarget && target == c.objID {
			sawOwner = true
		}
	}) {
		t.Fatal("InshariSkill1Effect2 was never started by the self-cast")
	}
	if !sawOwner {
		t.Error("InshariSkill1Effect2 owner/target did not both resolve to the caster's own objID")
	}
}

// TestGektorReprisalFiresSelfFxFromDefenseProc: a defense (OnDamaged) proc never casts,
// so firePayloadLocked's "self" case is unreachable -- runDefenseProcsLocked must start
// GektorSkill2Effect itself.
func TestGektorReprisalFiresSelfFxFromDefenseProc(t *testing.T) {
	s, c, hs, mu, pkts := newCaptureConn(t, "Avtr_Tank_Gektor")
	realOp := hs.kit.Skills[1].Ops[0] // slot 2 "Реванш"
	if realOp.Kind != gamedata.OpProc || !realOp.OnDamaged {
		t.Fatal("Gektor «Реванш» is no longer an OnDamaged proc; test needs updating")
	}
	hs.defenseProcs = []procState{{slot: 2, chance: gamedata.PerLevel{1, 1, 1, 1}, ops: realOp.Ops, basicAttackOnly: true}}

	c.mvMu.Lock()
	attacker := mkMob(t, 3000, 1, 0)
	hs.mobs[attacker.id] = attacker
	hs.tr.add(attacker.id)
	hs.lastDamageWasSkill = false
	s.runDefenseProcsLocked(c, attacker, 10, float64(s.battleTime()))
	c.mvMu.Unlock()

	if !awaitEffectStartFx(t, mu, pkts, "GektorSkill2Effect", nil) {
		t.Error("GektorSkill2Effect (Реванш) never fired from runDefenseProcsLocked")
	}
}

// TestGektorRendingStrikeFiresTargetFxFromOnHitProc: an on-hit passive (runProcsLocked)
// never casts either, so the BuffFxOn=="target" fx must be started explicitly there.
func TestGektorRendingStrikeFiresTargetFxFromOnHitProc(t *testing.T) {
	s, c, hs, mu, pkts := newCaptureConn(t, "Avtr_Tank_Gektor")
	realOp := hs.kit.Skills[2].Ops[0] // slot 3 "Разящий удар"
	if realOp.Kind != gamedata.OpProc {
		t.Fatal("Gektor «Разящий удар» is no longer an OpProc; test needs updating")
	}
	hs.procs = []procState{{slot: 3, chance: gamedata.PerLevel{1, 1, 1, 1}, ops: realOp.Ops}}

	c.mvMu.Lock()
	victim := mkMob(t, 4000, 1, 0)
	hs.mobs[victim.id] = victim
	hs.tr.add(victim.id)
	s.runProcsLocked(c, victim, float64(s.battleTime()))
	c.mvMu.Unlock()

	sawOwner := false
	if !awaitEffectStartFx(t, mu, pkts, "GektorSkill3Effect", func(p battleproto.Packet) {
		owner, _ := p.Args.GetInt("owner")
		if owner == victim.id {
			sawOwner = true
		}
	}) {
		t.Fatal("GektorSkill3Effect (Разящий удар) never fired from runProcsLocked")
	}
	if !sawOwner {
		t.Error("GektorSkill3Effect owner did not resolve to the struck victim")
	}
}

// TestTitanidQuakeFiresEscalatingWaveFx: channelWavePulseFx maps Titanid's 3-wave
// «Землетрясение» channel pulses to TitanidSkill1Effect2/3 (Effect1 already plays once
// at cast time as the skill's own PayloadFx) -- previously dead registry entries.
func TestTitanidQuakeFiresEscalatingWaveFx(t *testing.T) {
	if fx := channelWavePulseFx("Avtr_Tank_Titanid", 1, 0); fx != "" {
		t.Errorf("channelWavePulseFx(Titanid, slot1, pulse0) = %q, want empty (covered by the cast-time payload)", fx)
	}
	if fx := channelWavePulseFx("Avtr_Tank_Titanid", 1, 1); fx != "TitanidSkill1Effect2" {
		t.Errorf("channelWavePulseFx(Titanid, slot1, pulse1) = %q, want TitanidSkill1Effect2", fx)
	}
	if fx := channelWavePulseFx("Avtr_Tank_Titanid", 1, 2); fx != "TitanidSkill1Effect3" {
		t.Errorf("channelWavePulseFx(Titanid, slot1, pulse2) = %q, want TitanidSkill1Effect3", fx)
	}
	if fx := channelWavePulseFx("Avtr_Tank_Rognar", 1, 1); fx != "" {
		t.Errorf("channelWavePulseFx(Rognar, slot1, pulse1) = %q, want empty (negative control)", fx)
	}
}

// TestTitanidQuakeTickFiresWaveFx drives tickChannelsLocked through 2 pulses and
// checks both TitanidSkill1Effect2 and TitanidSkill1Effect3 actually get pushed.
func TestTitanidQuakeTickFiresWaveFx(t *testing.T) {
	s, c, hs, mu, pkts := newCaptureConn(t, "Avtr_Tank_Titanid")
	c.mvMu.Lock()
	now := s.battleTime()
	hs.channels = []channelState{{
		slot: 1, level: 1, until: float64(now) + 10, interval: 0.8,
		hasPos: true, px: 5, py: 5,
		ops: []gamedata.Op{{Kind: gamedata.OpDamage, Value: gamedata.PerLevel{1, 1, 1, 1}, Radius: 5}},
	}}
	for i := 0; i < 3; i++ { // pulseCount 0 (cast-time wave, no tick fx), 1 (Effect2), 2 (Effect3)
		hs.channels[0].nextPulse = float64(s.battleTime())
		s.tickChannelsLocked(c, float64(s.battleTime()))
	}
	c.mvMu.Unlock()

	time.Sleep(50 * time.Millisecond)
	mu.Lock()
	seen := map[string]bool{}
	for _, p := range *pkts {
		if p.Cmd != battleproto.CmdEffectStart {
			continue
		}
		if fx, _ := p.Args.GetString("fx"); fx != "" {
			seen[fx] = true
		}
	}
	mu.Unlock()
	if !seen["TitanidSkill1Effect2"] {
		t.Error("TitanidSkill1Effect2 never fired on the channel's 2nd pulse")
	}
	if !seen["TitanidSkill1Effect3"] {
		t.Error("TitanidSkill1Effect3 never fired on the channel's 3rd pulse")
	}
}

// TestZamaranSlamNoLongerDuplicatesAtCastTime: PayloadFx is cleared on the primary
// payload (ZamaranSkill1 is the SAME registry entry as CastFx, so leaving both wired
// fired the one-shot slam burst twice near cast time).
func TestZamaranSlamNoLongerDuplicatesAtCastTime(t *testing.T) {
	av, ok := gamedata.AvatarByID(11) // Zamaran
	if !ok {
		t.Fatal("Zamaran (id 11) missing")
	}
	sk := gamedata.SkillsFor(av).Skills[0] // slot 1
	if sk.PayloadFx != "" {
		t.Errorf("Zamaran slot 1 PayloadFx = %q, want empty (dashArrivalFx fires it at arrival instead)", sk.PayloadFx)
	}
	if sk.CastFx != "ZamaranSkill1" {
		t.Errorf("Zamaran slot 1 CastFx = %q, want \"ZamaranSkill1\"", sk.CastFx)
	}
}

// TestDashArrivalFxFiresAtLanding: dashArrivalFx names ZamaranSkill1 for slot 1, and
// firePayloadLocked's resume (StrikeOnArrival) branch fires it owned to the caster --
// who by then stands at the real landing point, which is what the fx's SELFPOS mode
// needs (it captures the owner's position once, at EFFECT_START).
func TestDashArrivalFxFiresAtLanding(t *testing.T) {
	if fx := dashArrivalFx("Avtr_Tank_Zamaran", 1); fx != "ZamaranSkill1" {
		t.Fatalf("dashArrivalFx(Zamaran, slot1) = %q, want ZamaranSkill1", fx)
	}
	if fx := dashArrivalFx("Avtr_Tank_Rognar", 1); fx != "" {
		t.Errorf("dashArrivalFx(Rognar, slot1) = %q, want empty (negative control)", fx)
	}

	s, c, hs, mu, pkts := newCaptureConn(t, "Avtr_Tank_Zamaran")
	c.mvMu.Lock()
	now := float64(s.battleTime())
	hs.payloads = []payload{{
		at: now, slot: 1, level: 1, resume: true,
		ops: []gamedata.Op{{Kind: gamedata.OpDamage, Value: gamedata.PerLevel{1, 1, 1, 1}, Radius: 4}},
	}}
	s.runDuePayloadsLocked(c, now)
	c.mvMu.Unlock()

	sawOwner := false
	if !awaitEffectStartFx(t, mu, pkts, "ZamaranSkill1", func(p battleproto.Packet) {
		owner, _ := p.Args.GetInt("owner")
		if owner == c.objID {
			sawOwner = true
		}
	}) {
		t.Fatal("ZamaranSkill1 never fired on the dash's StrikeOnArrival payload")
	}
	if !sawOwner {
		t.Error("ZamaranSkill1 owner did not resolve to the caster at the arrival point")
	}
}
