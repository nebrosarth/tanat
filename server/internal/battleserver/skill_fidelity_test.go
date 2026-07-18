package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

// TestOpStealthGrantsInvisibility: OpStealth (Lirvein/Wilfang cloak skills) sets the same
// invisibleUntil the Invisibility potion uses, so mobs stop targeting the hidden avatar.
func TestOpStealthGrantsInvisibility(t *testing.T) {
	s, c, _, cleanup := newHuntConnWithHero(t, "Avtr_DPS_Lirvein")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	c.lock()
	s.applyOpsLocked(c, []gamedata.Op{{Kind: gamedata.OpStealth, Dur: gamedata.PerLevel{4, 4, 4, 4}}},
		opCtx{slot: 1, level: 1}, now)
	c.unlock()

	if hs.invisibleUntil <= now {
		t.Fatalf("OpStealth did not grant invisibility: invisibleUntil=%.1f now=%.1f", hs.invisibleUntil, now)
	}
	if hs.invisFxUID == 0 {
		t.Error("OpStealth did not start the shade fx")
	}
}

// TestOnDamagedHealByDamage: an on-damaged proc whose OpHeal has Value2>0 heals for the
// size of the hit that triggered it (Nerlag «Прилив крови»).
func TestOnDamagedHealByDamage(t *testing.T) {
	s, c, _, cleanup := newHuntConnWithHero(t, "Avtr_DPS_Nerlag")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	c.lock()
	defer c.unlock()
	// Register a 100%-chance on-damaged heal-for-damage proc directly (slot 3 = «Прилив крови»).
	hs.skillLevel[2] = 1
	hs.defenseProcs = []procState{{
		slot:   3,
		chance: gamedata.PerLevel{1, 1, 1, 1},
		ops:    []gamedata.Op{{Kind: gamedata.OpHeal, Value2: gamedata.PerLevel{1, 1, 1, 1}}},
	}}
	// Drop to half HP, then take a 200-damage blow: the proc should heal that 200 back.
	full := hs.maxHPLocked(now)
	hs.hp = full - 400
	before := hs.hp
	s.runDefenseProcsLocked(c, nil, 200, now)
	if got := hs.hp - before; got < 199 || got > 201 {
		t.Errorf("on-damaged heal = %.1f, want ~200 (heal for the damage taken)", got)
	}
}

// TestPerOpChanceGate: applyOpsLocked's per-op Chance gate fires a non-proc op only a
// fraction of the time when 0<Chance<1, but ALWAYS for Chance>=1 and for chance-less
// ops (so existing skills are untouched). Backs Zamaran's «Пламя войны» 20% root.
func TestPerOpChanceGate(t *testing.T) {
	s, c, cleanup := newHuntConn(t, "Avtr_Tank_Zamaran")
	defer cleanup()
	hs := c.huntState
	now := float64(s.battleTime())

	mob := &mobState{id: 5000, mobIdx: 0, mob: gamedata.Mobs()[0], hp: 5000, maxHP: 5000, x: 1, y: 0, shown: true}
	hs.tr.add(mob.id)
	hs.mobs[mob.id] = mob

	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	const trials = 400
	roll := func(chance gamedata.PerLevel) int {
		hits := 0
		for i := 0; i < trials; i++ {
			mob.st.rootUntil = 0
			s.applyOpsLocked(c, []gamedata.Op{{Kind: gamedata.OpRoot, Dur: gamedata.PerLevel{1, 1, 1, 1}, Chance: chance}},
				opCtx{slot: 2, level: 1, target: mob}, now)
			if mob.st.rootUntil > now {
				hits++
			}
		}
		return hits
	}

	// A chance-less op must ALWAYS fire -- the gate must not touch existing skills.
	if h := roll(gamedata.PerLevel{}); h != trials {
		t.Errorf("unset Chance: applied %d/%d, want %d (gate must ignore chance-less ops)", h, trials, trials)
	}
	// Chance 1.0 always fires.
	if h := roll(gamedata.PerLevel{1, 1, 1, 1}); h != trials {
		t.Errorf("Chance 1.0: applied %d/%d, want %d", h, trials, trials)
	}
	// Chance 0.2 fires a partial fraction (expected ~80/400; never 0 or all).
	if h := roll(gamedata.PerLevel{0.2, 0.2, 0.2, 0.2}); h < 20 || h > 240 {
		t.Errorf("Chance 0.2: applied %d/%d, want a partial fraction (~80)", h, trials)
	}
}
