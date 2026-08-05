package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

// TestFrostBoltDamagesOnArrival: «Стужа» is a real flying bolt --
// VFX_Avtr_Dsb_Frost_skill1_prop01 carries a SmoothMove with mBySpeed=true and mSpeed=35,
// so the client flies it at 35 units/sec. The damage used to land the moment the bolt was
// spawned, i.e. before it got there.
func TestFrostBoltDamagesOnArrival(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Frost")
	defer cleanup()
	hs := c.huntState

	if sp := hs.skillDef(1).PayloadFlightSpeed; sp != 35 {
		t.Fatalf("bolt speed = %g, want the prefab's 35 units/sec", sp)
	}

	m := mkMob(t, 8000, 7, 0) // near the skill's 8-unit reach
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)
	startHP := m.hp

	now := float64(s.battleTime())
	s.firePayloadLocked(c, payload{slot: 1, level: 1, target: m.id}, now)
	if m.hp != startHP {
		t.Fatalf("damage landed as the bolt was loosed: hp %g -> %g", startHP, m.hp)
	}
	if len(hs.payloads) != 1 {
		t.Fatalf("no arrival scheduled: %d pending", len(hs.payloads))
	}
	arr := hs.payloads[0]
	if got, want := arr.at-now, 7.0/35.0; got < want-0.01 || got > want+0.01 {
		t.Fatalf("arrival in %.3gs, want distance/speed = %.3gs", got, want)
	}
	s.firePayloadLocked(c, arr, arr.at)
	if m.hp >= startHP {
		t.Fatalf("the bolt arrived but dealt nothing: hp stayed at %g", m.hp)
	}
}

// TestFrostTombEncasesTheTargetNotTheCaster: «Гробница холода» freezes ONE unit, «врага или
// союзника» -- and the ice block is its BuffFx. It never rendered at all («не создаётся
// prop») because the TTL was measured only from OpBuffStat On:"target", and this skill's
// only stat-buff is on the ALLY half, so the TTL came out zero.
func TestFrostTombEncasesTheTargetNotTheCaster(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Frost")
	defer cleanup()
	hs := c.huntState

	def := hs.skillDef(3)
	// prop02 is the block ITSELF (FreezeEffect + a looping VisualEffect + the IceCube);
	// prop01 is only the 0.3s formation flash. All three props carry an IceCube mesh, so
	// the names cannot be trusted -- this pins the reading down.
	if def.BuffFx != "FrostSkill3Effect2" || def.BuffFxOn != "target" {
		t.Fatalf("the ice block is not pinned to the target: fx=%q on=%q", def.BuffFx, def.BuffFxOn)
	}
	if def.TargetFxEnd != "FrostSkill3Effect3" {
		t.Errorf("the block never shatters: TargetFxEnd=%q", def.TargetFxEnd)
	}
	// The formation flash is baked SELF, so it must be OWNED to the victim -- owned to the
	// caster it froze Frost instead of whoever she aimed at.
	if def.PayloadFx != "FrostSkill3Effect1" || def.PayloadFxAt != "target" {
		t.Errorf("the ice-forming flash is %q at %q, want FrostSkill3Effect1 at \"target\"", def.PayloadFx, def.PayloadFxAt)
	}
	if !payloadTargetFxOwnedToTarget("Avtr_Dsb_Frost", 3) {
		t.Error("the formation flash is still owned to the caster")
	}
	// The block must stand for the encasement, which is the skill's own stun.
	if ttl := targetBuffTTL(def, 1); ttl != 3 {
		t.Fatalf("ice block TTL = %gs, want the 3s encasement", ttl)
	}

	m := mkMob(t, 8100, 3, 0)
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)

	now := float64(s.battleTime())
	s.firePayloadLocked(c, payload{slot: 3, level: 1, target: m.id}, now)
	if m.st.riderFx == 0 {
		t.Fatal("no ice block was placed on the encased mob")
	}
	// ...and it is scheduled to turn into the shattering effect, not merely vanish.
	var shatters bool
	for _, f := range hs.fxEnds {
		if f.uid == m.st.riderFx && f.then == "FrostSkill3Effect3" && f.thenOwner == m.id {
			shatters = true
		}
	}
	if !shatters {
		t.Fatal("the block was not scheduled to shatter on the target")
	}
}

// TestFrostHailIsASelfCentredChannel settles the other shape question, and settles it from
// the PREFAB rather than from the phrasing. VFX_Avtr_Dsb_Frost_skill2_prop02 is a
// PrefabTimeSpawn dropping a shard every 0.07s with m_SpawnRadius=6.5 and
// m_UseParentTransorm=false, and the prefab is baked SELF -- so the hail rains around
// whatever owns it, and it travels with that owner. Owned to the mage, that is «вокруг
// мага», exactly as the card says. The looping Cast02 pose makes it a channel.
func TestFrostHailIsASelfCentredChannel(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Frost")
	defer cleanup()
	hs := c.huntState

	sk := hs.skillDef(2)
	if sk.Targeting != "SELF" {
		t.Errorf("targeting = %q, want SELF -- the hail is centred on the mage, not placed", sk.Targeting)
	}
	if sk.PayloadFx != "FrostSkill2Effect" || sk.PayloadFxAt != "self" {
		t.Errorf("the hail cloud is %q at %q, want FrostSkill2Effect at \"self\"", sk.PayloadFx, sk.PayloadFxAt)
	}
	for _, op := range sk.Ops {
		if op.Kind != gamedata.OpChannel {
			continue
		}
		for _, in := range op.Ops {
			if in.Radius != 6.5 {
				t.Errorf("%s radius %g, want the prefab's own 6.5 spawn radius", in.Kind, in.Radius)
			}
		}
	}

	m := mkMob(t, 8300, 2, 0)
	m.hp, m.maxHP = 100000, 100000
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)

	now := float64(s.battleTime())
	hs.noteCastFxLocked(2, 6060, true) // stand in for the looping Cast02 pose
	s.applyOpsLocked(c, sk.Ops, opCtx{slot: 2, level: 1}, now)
	if len(hs.channels) != 1 {
		t.Fatalf("the hail is not a channel: %d channels", len(hs.channels))
	}
	if hs.channels[0].hasPos {
		t.Fatal("the hail is pinned to a ground point: it must follow the mage")
	}
	if len(hs.channels[0].fxUIDs) == 0 {
		t.Fatal("the channel adopted no fx: breaking it could not stop the looping pose")
	}

	// Walking must end it -- pose, hail and damage together.
	c.hasDest = true
	s.tickChannelsLocked(c, now+0.2)
	if len(hs.channels) != 0 {
		t.Fatal("moving did not break the hail channel: the cast animation would keep looping")
	}
}

// TestFrostTombIsSingleTargetNotAnArea settles the shape question from the client text:
// «Заковывает врага или союзника в ледяную глыбу... ЦЕЛЬ не может атаковать или
// двигаться» -- one unit the player picks. Not a ring around the caster, not a placed
// area, and not a channel.
func TestFrostTombIsSingleTargetNotAnArea(t *testing.T) {
	kit := gamedata.SkillsFor(avatarByPrefab(t, "Avtr_Dsb_Frost"))
	sk := kit.Skills[2]
	if sk.Targeting != "TARGET" {
		t.Errorf("targeting = %q, want TARGET (one picked unit)", sk.Targeting)
	}
	if sk.AoERadius != 0 {
		t.Errorf("AoERadius = %d, want 0 -- it is not an area skill", sk.AoERadius)
	}
	for _, op := range sk.Ops {
		if op.Kind == gamedata.OpChannel {
			t.Error("the tomb is not channelled: the ice stands on its own for its duration")
		}
		if op.Radius != 0 {
			t.Errorf("%s has Radius %g: the tomb hits only its one target", op.Kind, op.Radius)
		}
	}
}

// TestFrostElementalStrikesAtTheEndOfItsSwing: a summon used to connect the instant its
// swing started, so the victim lost health while the model was still winding up. The
// elemental is also a SHOOTER -- its prefab declares a real Projectile fired from
// Bone_proj -- so the bolt is loosed at that same late point and the hit lands on arrival.
func TestFrostElementalStrikesAtTheEndOfItsSwing(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Frost")
	defer cleanup()
	hs := c.huntState

	m := mkMob(t, 8200, 1, 0) // inside the summon's melee reach, so it swings rather than walks
	m.hp, m.maxHP = 100000, 100000
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)

	sm := &summonState{id: 8201, x: 0, y: 0, hp: 500, maxHP: 500, dmg: 45, until: 1e9,
		ranged: summonRangedPrefabs["Avtr_Dsb_Frost_Elemental"]}
	if !sm.ranged {
		t.Fatal("the ice elemental is not marked ranged, but its prefab carries a projectile")
	}
	hs.summons[sm.id] = sm

	now := 100.0
	s.tickSummonsLocked(c, now)
	if sm.swingHitAt == 0 {
		t.Fatal("the swing armed no wind-up")
	}
	startHP := m.hp
	s.tickSummonsLocked(c, now+0.1) // early in the swing
	if m.hp != startHP {
		t.Fatal("the blow landed at the start of the swing")
	}
	// Late in the swing the bolt is loosed; it still has to fly before anything is hurt.
	s.tickSummonsLocked(c, now+summonStrikeFrac+0.01)
	if m.hp != startHP {
		t.Fatal("the bolt dealt damage on release instead of on arrival")
	}
	if sm.projHitAt == 0 {
		t.Fatal("no bolt was loosed at the end of the swing")
	}
	s.tickSummonsLocked(c, sm.projHitAt+0.01)
	if m.hp >= startHP {
		t.Fatalf("the bolt arrived but dealt nothing: hp stayed at %g", m.hp)
	}
}
