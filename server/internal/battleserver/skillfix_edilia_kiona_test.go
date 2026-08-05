package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

// TestEdiliaBlocksFirstBasicAttackThenRecharges: «При получении удара от базовой атаки,
// Эдилия не получает урона» -- a GUARANTEED negation of the first basic attack, then the
// internal cooldown. It used to be a 10-22% dodge roll, so the block almost never happened
// while the paired proc still greyed the button.
func TestEdiliaBlocksFirstBasicAttackThenRecharges(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Edilia")
	defer cleanup()
	hs := c.huntState
	hs.skillLevel[2] = 1
	hs.blockHitSlot = 3 // registered at world build; a bare test conn skips that

	m := mkMob(t, 7700, 1, 0)
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)

	now := float64(s.battleTime())
	startHP := hs.hp
	s.hitPlayerFromLocked(c, m.id, 40, now, m, nil)
	if hs.hp != startHP {
		t.Fatalf("the first basic attack was not blocked: hp %g -> %g", startHP, hs.hp)
	}
	if hs.blockHitReadyAt <= now {
		t.Fatal("blocking must start the internal cooldown")
	}
	// «атакующий теряет скорость атаки на 50% в течение 1 секунды»
	if m.st.atkSlowUntil <= now {
		t.Fatal("the attacker was not slowed by the block")
	}

	// The NEXT attack, inside the cooldown, lands in full.
	s.hitPlayerFromLocked(c, m.id, 40, now+0.5, m, nil)
	if hs.hp >= startHP {
		t.Fatalf("a second attack inside the cooldown was also blocked: hp still %g", hs.hp)
	}
}

// TestEdiliaBlockCoversRangedAttacks: the block lives in the shared incoming-damage path,
// so an arriving projectile is negated exactly like a melee swing -- the reported «не
// срабатывает на дальние атаки мобов».
func TestEdiliaBlockCoversRangedAttacks(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Edilia")
	defer cleanup()
	hs := c.huntState
	hs.skillLevel[2] = 1
	hs.blockHitSlot = 3

	ranged := mkMob(t, 7710, 9, 0)
	ranged.mob.AttackRange = 8 // a shooter, not a melee mob
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.mobs[ranged.id] = ranged
	hs.tr.add(ranged.id)

	now := float64(s.battleTime())
	startHP := hs.hp
	s.hitPlayerFromLocked(c, ranged.id, 40, now, ranged, nil)
	if hs.hp != startHP {
		t.Fatalf("a ranged basic attack got through the block: hp %g -> %g", startHP, hs.hp)
	}
}

// TestEdiliaBlockIgnoresMobSkills: «от базовой атаки» -- a boss skill still connects.
func TestEdiliaBlockIgnoresMobSkills(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Edilia")
	defer cleanup()
	hs := c.huntState
	hs.skillLevel[2] = 1
	hs.blockHitSlot = 3

	m := mkMob(t, 7720, 1, 0)
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)

	now := float64(s.battleTime())
	startHP := hs.hp
	hs.lastDamageWasSkill = true
	s.hitPlayerFromLocked(c, m.id, 40, now, m, nil)
	if hs.hp >= startHP {
		t.Fatal("a mob SKILL must not be blocked by «Пыльца забвения»")
	}
	if hs.blockHitReadyAt > now {
		t.Fatal("a skill hit must not consume the block's cooldown either")
	}
}

// TestEdiliaTreeCursorMatchesTheGrove: the client's ground cursor is drawn from
// AoERadius, so it has to agree with the grove prefab it plants -- and with the area the
// ops actually cover, or the cursor promises reach the damage never delivers.
func TestEdiliaTreeCursorMatchesTheGrove(t *testing.T) {
	kit := gamedata.SkillsFor(avatarByPrefab(t, "Avtr_Dsb_Edilia"))
	ult := kit.Skills[3]
	if ult.AoERadius < 6 {
		t.Fatalf("«Дерево жизни» cursor radius = %d, too small for the measured grove", ult.AoERadius)
	}
	var checked int
	for _, op := range ult.Ops {
		if op.Kind != gamedata.OpChannel {
			continue
		}
		for _, in := range op.Ops {
			if in.Kind != gamedata.OpDamage && in.Kind != gamedata.OpHeal {
				continue
			}
			checked++
			if int(in.Radius) != ult.AoERadius {
				t.Errorf("%s radius %g does not match the cursor's %d", in.Kind, in.Radius, ult.AoERadius)
			}
		}
	}
	if checked == 0 {
		t.Fatal("the tree channel has no damage/heal op to check")
	}
}

// TestKionaOwlRidesItsTarget: KionaSkill4Effect's gfx is baked SELF, so it parents to
// whatever object owns it. Owned to the caster, the guardian owl followed KIONA around
// instead of watching over the unit «Страж леса» was cast on.
func TestKionaOwlRidesItsTarget(t *testing.T) {
	if !payloadTargetFxOwnedToTarget("Avtr_Sp_Kiona", 4) {
		t.Fatal("Kiona's «Страж леса» payload fx is still owned to the caster")
	}
	kit := gamedata.SkillsFor(avatarByPrefab(t, "Avtr_Sp_Kiona"))
	ult := kit.Skills[3]
	if ult.PayloadFxAt != "target" {
		t.Fatalf("«Страж леса» payload fx placement = %q, want \"target\"", ult.PayloadFxAt)
	}
	// And the owl must outlive a 3s default: it accompanies a 10s effect.
	if d := skillOverTimeDur(ult, 1); d < 10 {
		t.Fatalf("longest over-time effect = %gs, expected the owl's full 10s watch", d)
	}
}

// TestKionaOwlDiesWithItsTarget: the owl is parented to the unit it watches, so a corpse
// (or a deleted body) must not leave it circling. The fx is remembered on the body and
// ended in the death branch, well before its own 10s window would have closed.
func TestKionaOwlDiesWithItsTarget(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Sp_Kiona")
	defer cleanup()
	hs := c.huntState

	m := mkMob(t, 7800, 3, 0)
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.mobs[m.id] = m
	hs.tr.add(m.id)

	now := float64(s.battleTime())
	s.firePayloadLocked(c, payload{slot: 4, level: 1, target: m.id}, now)
	if m.st.riderFx == 0 {
		t.Fatal("the owl was not recorded on the body it rides -- its death could not end it")
	}

	s.hitMobLocked(c, m, m.hp+1000, c.objID) // kill it
	if !m.dead {
		t.Fatal("test setup: the mob should be dead")
	}
	if m.st.riderFx != 0 {
		t.Fatal("the owl outlived its target")
	}
}

// TestMorlokayCurseCircleStaysAtTheCastPoint: MorlokaySkill1Effect1's decal is baked SELF
// and its VisualEffect loops forever, while the registry marks the gfx stop_on_done=false
// -- so EFFECT_END cannot stop it. Owned to the caster it followed Morlokai around and
// never went away. It has to hang off an anchor at the cast point, which is also the only
// thing whose deletion can take it with it.
func TestMorlokayCurseCircleStaysAtTheCastPoint(t *testing.T) {
	if !payloadFxUsesAnchor("Avtr_Dsb_Morlokay", 1) {
		t.Fatal("«Проклятие Вуду» ground decal is still owned to the caster")
	}
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Morlokay")
	defer cleanup()
	hs := c.huntState

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())
	s.firePayloadLocked(c, payload{slot: 1, level: 1, px: 5, py: 5, hasPos: true}, now)

	if len(hs.anchorEnds) != 1 {
		t.Fatalf("no anchor scheduled for removal: %d pending", len(hs.anchorEnds))
	}
	if at := hs.anchorEnds[0].at; at <= now {
		t.Fatalf("anchor removal scheduled at %g, not in the future (now %g)", at, now)
	}
}
