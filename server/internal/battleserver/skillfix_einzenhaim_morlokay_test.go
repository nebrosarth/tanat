package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

// This file locks the fixes made after a live pass on Einzenhaim and Morlokai. Every
// expectation below is anchored in the CLIENT's own shipped data (the baked
// VisualEffectsMgr registry in Tanat_Data/mainData and the prefabs in
// data/Characters/Avatars/*.unity3d), not in taste -- see the comments in skills_gen.go.

// TestChannelHonorsAuthoredIntervalAndCount: a channel with Count fires exactly that
// many pulses, at the AUTHORED cadence. The tick used to floor the interval at 0.4s,
// which silently rewrote Einzenhaim's 3-5 shot volley into 2 hits at every rank.
func TestChannelHonorsAuthoredIntervalAndCount(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Einzenhaim")
	defer cleanup()
	hs := c.huntState

	m := mkMob(t, 7100, 2, 0)
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)

	now := float64(s.battleTime())
	s.applyOpsLocked(c, []gamedata.Op{
		{Kind: gamedata.OpChannel, Count: gamedata.PerLevel{3}, Dur: gamedata.PerLevel{0.65}, Interval: 0.3,
			Ops: []gamedata.Op{{Kind: gamedata.OpDamage, Value: gamedata.PerLevel{10}, Scale: "pure"}}},
	}, opCtx{slot: 2, level: 1, target: m}, now)
	startHP := m.mob.Health
	if len(hs.channels) != 1 {
		t.Fatalf("channel not started: %d channels", len(hs.channels))
	}
	// Shot 1 lands AT the cast, not a tick later: a counted volley is "N shots starting
	// now", and the 0.2s tick grid would otherwise lag it behind the client's muzzle flash.
	if m.hp != startHP-10 {
		t.Fatalf("first shot did not land immediately: hp %g, want %g", m.hp, startHP-10)
	}
	if got := hs.channels[0].pulsesLeft; got != 2 {
		t.Fatalf("pulsesLeft = %d, want 2 (3 shots, one already fired)", got)
	}

	// Drive the same 0.2s grid the real tick runs on, well past the nominal duration.
	for i := 1; i <= 20 && len(hs.channels) > 0; i++ {
		s.tickChannelsLocked(c, now+0.2*float64(i))
	}
	if len(hs.channels) != 0 {
		t.Fatalf("channel still alive after 4s: pulsesLeft=%d", hs.channels[0].pulsesLeft)
	}
	// Exactly 3 shots × 10 pure damage -- no more, no fewer.
	if want := startHP - 30; m.hp != want {
		t.Fatalf("mob hp = %g, want %g (3 shots × 10)", m.hp, want)
	}
}

// TestChannelBreakEndsVisualsAndReleasesHold: walking away from a channel must stop the
// SHOW as well as the damage. The client's holders for these skills set mLoopAnimation
// and a looping VisualEffect, so they run until an EFFECT_END arrives; and a channel that
// is a GRIP (Morlokai's «Кабала» roots and silences its victim) must let the victim go.
func TestChannelBreakEndsVisualsAndReleasesHold(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Morlokay")
	defer cleanup()
	hs := c.huntState

	m := mkMob(t, 7200, 2, 0)
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)

	now := float64(s.battleTime())
	hs.noteCastFxLocked(2, 4242, true) // pretend the cast fx is live
	s.applyOpsLocked(c, hs.skillDef(2).Ops, opCtx{slot: 2, level: 1, target: m}, now)
	if len(hs.channels) != 1 {
		t.Fatalf("«Кабала» did not start a channel: %d channels", len(hs.channels))
	}
	ch := hs.channels[0]
	if !ch.holdsTarget {
		t.Fatal("a channel cast alongside root+silence must be marked as holding its target")
	}
	if len(ch.fxUIDs) == 0 || ch.fxUIDs[0] != 4242 {
		t.Fatalf("channel did not adopt the cast's fx uids: %v", ch.fxUIDs)
	}
	if m.st.rootUntil <= now || m.st.silenceUntil <= now {
		t.Fatalf("victim not held: root=%g silence=%g now=%g", m.st.rootUntil, m.st.silenceUntil, now)
	}

	c.hasDest = true // the caster walks off
	s.tickChannelsLocked(c, now+0.2)
	if len(hs.channels) != 0 {
		t.Fatal("moving must break a unit channel")
	}
	if m.st.rootUntil > now+0.2 || m.st.silenceUntil > now+0.2 {
		t.Fatalf("broken hold left the victim pinned: root=%g silence=%g", m.st.rootUntil, m.st.silenceUntil)
	}
}

// TestEinzenhaimRecoilThrowsCasterBack: «В момент выстрела аватар отлетает назад из-за
// отдачи» -- stated in all three locale variants, and nothing in the client does it.
func TestEinzenhaimRecoilThrowsCasterBack(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Einzenhaim")
	defer cleanup()
	hs := c.huntState

	m := mkMob(t, 7300, 6, 0) // target straight ahead on +x
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)

	now := float64(s.battleTime())
	s.applyOpsLocked(c, []gamedata.Op{
		{Kind: gamedata.OpKnockback, Apply: "self", Value: gamedata.PerLevel{4}},
	}, opCtx{slot: 3, level: 1, target: m}, now)

	if len(c.path) == 0 {
		t.Fatal("recoil issued no movement at all")
	}
	dest := c.path[len(c.path)-1]
	if dest.X >= 0 {
		t.Fatalf("recoil moved the caster toward the target (x=%g), want backwards (x<0)", dest.X)
	}
	if hs.dashUntil <= now {
		t.Fatal("recoil must run at dash speed so the client glides it, not walk speed")
	}
}

// TestEinzenhaimSkill3HasRecoilOp guards the spec itself: the recoil is part of the
// skill, not an engine capability nobody uses. One shot, one kick, the card's damage
// undivided -- a two-barrel split was tried and reverted (nothing in the client times the
// second barrel).
func TestEinzenhaimSkill3HasRecoilOp(t *testing.T) {
	kit := gamedata.SkillsFor(avatarByPrefab(t, "Avtr_DPS_Einzenhaim"))
	sk := kit.Skills[2]
	var recoil, dmg int
	for _, op := range sk.Ops {
		switch {
		case op.Kind == gamedata.OpKnockback && op.Apply == "self" && op.Value.At(1) > 0:
			recoil++
		case op.Kind == gamedata.OpDamage:
			dmg++
			if op.Value.At(1) != 85 {
				t.Errorf("rank-1 damage = %g, want the card's 85", op.Value.At(1))
			}
		}
	}
	if recoil != 1 {
		t.Fatalf("«Выстрел с отдачей» has %d self-knockback ops, want exactly 1", recoil)
	}
	if dmg != 1 {
		t.Fatalf("«Выстрел с отдачей» has %d damage ops, want exactly 1 (single shot)", dmg)
	}
}

// TestEinzenhaimBottleFliesBeforeItHurts: the ult is a THROW. The client's bottle prefab
// hangs off a _Parent carrying TimeDestroy(0.65), so the payload may not land before it
// does: on the throw tick nothing is damaged, and a second payload is queued for the
// arrival, carrying the anchor the impact effect plays on.
func TestEinzenhaimBottleFliesBeforeItHurts(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_DPS_Einzenhaim")
	defer cleanup()
	hs := c.huntState

	def := hs.skillDef(4)
	if def.PayloadFxAt != "throw" || def.PayloadFlight <= 0 {
		t.Fatalf("«Изгнание колдовства» is not modelled as a throw: at=%q flight=%g", def.PayloadFxAt, def.PayloadFlight)
	}
	if def.ImpactFx == "" || def.LingerFx == "" {
		t.Fatalf("the landing effects are unused: impact=%q linger=%q", def.ImpactFx, def.LingerFx)
	}

	m := mkMob(t, 7400, 3, 0)
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	startHP := m.hp

	now := float64(s.battleTime())
	s.firePayloadLocked(c, payload{slot: 4, level: 1, px: 3, py: 0, hasPos: true}, now)

	if m.hp != startHP {
		t.Fatalf("damage landed at the moment of the THROW: hp %g -> %g", startHP, m.hp)
	}
	if len(hs.payloads) != 1 {
		t.Fatalf("no arrival payload queued: %d pending", len(hs.payloads))
	}
	arr := hs.payloads[0]
	if arr.anchor == 0 {
		t.Fatal("arrival payload carries no anchor: the explosion would play on the caster")
	}
	if got := arr.at - now; got < def.PayloadFlight-1e-6 {
		t.Fatalf("arrival scheduled %.3gs out, want the bottle's own %.3gs flight", got, def.PayloadFlight)
	}

	// Now let it land: the ops run, and the channel adopts the lingering ground effect.
	s.firePayloadLocked(c, arr, arr.at)
	if len(hs.channels) == 0 {
		t.Fatal("arrival did not start the spray channel")
	}
}

// TestMorlokaySkill3HasInternalCooldown: an ungated Chance:1 proc fired the AoE burst --
// and its sound -- on literally every swing.
func TestMorlokaySkill3HasInternalCooldown(t *testing.T) {
	kit := gamedata.SkillsFor(avatarByPrefab(t, "Avtr_Dsb_Morlokay"))
	cd := kit.Skills[2].TipArgs["cooldown"]
	if len(cd) == 0 || cd.At(1) <= 0 {
		t.Fatal("«Всполох магии» has no internal cooldown: it procs on every basic attack")
	}
	// And it must actually gate the proc, not just sit in the tooltip.
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Morlokay")
	defer cleanup()
	hs := c.huntState
	hs.skillLevel[2] = 1

	// Procs are registered during world build, which a bare test conn never runs; mirror
	// that one line here so the ENGINE gate is what is under test.
	sk := kit.Skills[2]
	for _, op := range sk.Ops {
		if op.Kind == gamedata.OpProc {
			hs.procs = append(hs.procs, procState{slot: 3, chance: op.Chance, ops: op.Ops, cd: sk.TipArgs["cooldown"]})
		}
	}
	if len(hs.procs) != 1 {
		t.Fatalf("slot-3 passive has %d proc ops, want 1", len(hs.procs))
	}
	pr := &hs.procs[0]
	if pr.cd.At(1) <= 0 {
		t.Fatalf("registered proc carries no cooldown: %v", pr.cd)
	}

	m := mkMob(t, 7500, 1, 0)
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)

	now := float64(s.battleTime())
	s.runProcsLocked(c, m, now)
	afterFirst := m.hp
	if afterFirst >= m.mob.Health {
		t.Fatal("the passive did not fire at all on the first hit")
	}
	s.runProcsLocked(c, m, now+0.5) // a second swing, well inside the cooldown
	if m.hp != afterFirst {
		t.Fatalf("passive fired again %gs later, inside its %gs cooldown", 0.5, pr.cd.At(1))
	}
}

// TestMorlokayTotemShootsAProjectile: the totem's prefab
// (Avtr_Dsb_Morlokay_Skill4_prop01) declares mProjectiles = [Avtr_Dsb_Morlokay_projectile_prop02]
// with damageType LIGHTNING2, so the client CAN draw the bolt -- it was simply never told
// to. Until SET_PROJECTILE is sent, the victim just loses health with nothing on screen.
func TestMorlokayTotemShootsAProjectile(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Morlokay")
	defer cleanup()
	hs := c.huntState

	m := mkMob(t, 7600, 4, 0)
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)

	now := float64(s.battleTime())
	def := hs.skillDef(4)
	// The summon prototype is registered at world build, which a bare test conn skips.
	hs.summonProtos = map[string]int32{def.Ops[0].Unit: 800}
	s.applyOpsLocked(c, def.Ops, opCtx{slot: 4, level: 1, px: 0, py: 0, hasPos: true}, now)
	if len(hs.summons) != 1 {
		t.Fatalf("«Грозовой тотем» summoned %d units, want 1", len(hs.summons))
	}
	var sm *summonState
	for _, v := range hs.summons {
		sm = v
	}
	if !sm.ranged {
		t.Fatal("the totem is not marked ranged: it would resolve its zap silently, with no lightning and no sound")
	}

	startHP := m.hp
	s.strikeSummonLocked(c, sm, m, now)
	if sm.swingHitAt <= now || sm.swingTarget != m.id {
		t.Fatalf("the swing did not arm a wind-up: at=%g target=%d", sm.swingHitAt, sm.swingTarget)
	}
	if m.hp != startHP || sm.projHitAt != 0 {
		t.Fatal("the bolt was loosed at the START of the swing, not at the end of it")
	}
	// The wind-up completes: the bolt is loosed, but nothing is hurt until it lands.
	s.resolveSummonSwingLocked(c, sm, sm.swingHitAt)
	if m.hp != startHP {
		t.Fatal("a ranged summon must not damage on release: the bolt has to arrive first")
	}
	if sm.projHitAt <= now || sm.projTarget != m.id {
		t.Fatalf("no bolt in flight: hitAt=%g target=%d", sm.projHitAt, sm.projTarget)
	}
	s.resolveSummonBoltLocked(c, sm, sm.projHitAt)
	if m.hp >= startHP {
		t.Fatalf("the bolt arrived but dealt nothing: hp stayed at %g", m.hp)
	}
}
