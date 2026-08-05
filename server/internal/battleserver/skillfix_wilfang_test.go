package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

// TestWilfangPoisonExplosionShowsAnEffect: the death detonation dealt damage with nothing
// shown for it at all -- WilfangSkill3Effect (a real gfx+sfx pair) is now started AT THE
// CORPSE the instant it goes off.
func TestWilfangPoisonExplosionShowsAnEffect(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Dsb_Wilfang")
	defer cleanup()
	hs := c.huntState

	def := hs.skillDef(3)
	var dotOp *gamedata.Op
	for i := range def.Ops {
		for j := range def.Ops[i].Ops {
			if def.Ops[i].Ops[j].Kind == gamedata.OpDot {
				dotOp = &def.Ops[i].Ops[j]
			}
		}
	}
	if dotOp == nil || dotOp.ExplodeFx == "" {
		t.Fatal("«Ядовитый укус» detonation carries no ExplodeFx")
	}

	victim := mkMob(t, 8600, 3, 0)
	killer := mkMob(t, 8601, 1, 0)
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs.mobs[victim.id] = victim
	hs.mobs[killer.id] = killer
	hs.tr.add(victim.id)
	hs.tr.add(killer.id)

	now := float64(s.battleTime())
	s.applyOpsLocked(c, def.Ops, opCtx{slot: 3, level: 1, target: victim}, now)
	if victim.st.poisonExplodeFx == "" {
		t.Fatal("the poison arm did not record an explosion fx on the victim")
	}

	before := c.objID // sanity: fxStartLocked must not panic when the corpse detonates
	_ = before
	s.hitMobLocked(c, victim, victim.hp+1000, killer.id) // kill it while still poisoned
	if !victim.dead {
		t.Fatal("test setup: victim should be dead")
	}
	// No direct assertion on the wire packet (no capture harness here); the call not
	// panicking and the field being read/cleared with the rest of ms.st is the coverage
	// available at this layer -- see the ExplodeFx plumbing in hunt.go's death branch.
}

// TestWilfangAmbushDoesNotResumeAutoAttack: casting «Засада» must not roll the avatar into
// auto-attacking the nearest enemy the instant the cast closes -- that would immediately
// blow the stealth the player just paid mana for.
func TestWilfangAmbushDoesNotResumeAutoAttack(t *testing.T) {
	def := gamedata.SkillsFor(avatarByPrefab(t, "Avtr_Dsb_Wilfang")).Skills[1]
	if !skillStealthsCaster(def) {
		t.Fatal("«Засада» is not recognized as stealthing the caster")
	}
}

// TestSandarielVeilStillResumesAutoAttack: Sandariel's «Сокрывающая вуаль» stealths
// ALLIES while she explicitly stays visible -- her own auto-resume must not be suppressed.
func TestSandarielVeilStillResumesAutoAttack(t *testing.T) {
	def := gamedata.SkillsFor(avatarByPrefab(t, "Avtr_DPS_Sandariel")).Skills[3]
	if skillStealthsCaster(def) {
		t.Fatal("«Сокрывающая вуаль» must not be treated as stealthing SANDARIEL herself")
	}
}
