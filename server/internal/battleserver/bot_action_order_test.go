package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

func TestStartSkillOrderLockedRejectsSilencedBotWithoutArmingCast(t *testing.T) {
	s, bot, _, cleanup := newDotaConn(t, "Avtr_Sp_Arianna")
	defer cleanup()

	bot.lock()
	defer bot.unlock()
	hs := bot.huntState
	now := float64(s.battleTime())
	slot := 0
	for candidate := 1; candidate <= 4; candidate++ {
		if hs.skillDef(candidate).Type == "ACTIVE" {
			slot = candidate
			break
		}
	}
	if slot == 0 {
		t.Fatal("setup: Arianna has no active skill")
	}
	hs.skillLevel[slot-1] = 1
	hs.mana = hs.maxManaLocked(now)
	hs.st.silenceUntil = now + 10

	manaBefore := hs.mana
	cooldownBefore := hs.cooldownUntil[slot-1]
	payloadsBefore := len(hs.payloads)
	accepted := s.startSkillOrderLocked(bot, slot, 0, bot.x, bot.y, true)

	if accepted {
		t.Fatal("silenced bot skill order was reported as accepted")
	}
	if hs.order != nil {
		t.Fatalf("silenced bot armed an approach-cast order: %+v", hs.order)
	}
	if hs.mana != manaBefore || hs.cooldownUntil[slot-1] != cooldownBefore || len(hs.payloads) != payloadsBefore {
		t.Fatalf("silenced bot armed a cast: mana %.2f->%.2f cooldown %.2f->%.2f payloads %d->%d",
			manaBefore, hs.mana, cooldownBefore, hs.cooldownUntil[slot-1], payloadsBefore, len(hs.payloads))
	}
}

func TestBotLaneTickPreservesPendingOutOfRangeHealOrder(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Sp_Arianna")
	defer cleanup()
	b := &botBrain{c: bot, lane: 0, phase: botPhaseLane}

	bot.lock()
	defer bot.unlock()
	hs := bot.huntState
	now := float64(s.battleTime())
	healSlot := 0
	for slot := 1; slot <= 4; slot++ {
		def := hs.skillDef(slot)
		if def.Type == "ACTIVE" && botSkillTargetsAlliesLocked(def) &&
			(botSkillHasOp(def, gamedata.OpHeal) || botSkillHasOp(def, gamedata.OpHot) || botSkillHasOp(def, gamedata.OpShield)) {
			healSlot = slot
			break
		}
	}
	if healSlot == 0 {
		t.Fatal("setup: Arianna has no active ally-reaching heal/hot/shield skill")
	}
	hs.skillLevel[healSlot-1] = 1
	hs.mana = hs.maxManaLocked(now)
	castRange := hs.skillDef(healSlot).Distance
	if castRange <= 0 {
		castRange = 8
	}
	ally := dotaPlayerConn(t, s, inst, 1001, dotaTeamHuman, bot.x+float32(castRange)+1, bot.y)
	ally.huntState.hp = 1

	creep := &mobState{
		id: 65300, mobIdx: inst.dota.m.ElfCreepMelee, mob: gamedata.Mobs()[inst.dota.m.ElfCreepMelee],
		x: bot.x + 1, y: bot.y, hp: 1, maxHP: 200, team: dotaTeamElf, shown: true, active: true,
	}
	inst.mobs[creep.id] = creep

	s.botLaneTickLocked(b, now)

	if hs.order == nil {
		t.Fatal("out-of-range heal was not retained as a pending approach order")
	}
	if hs.order.slot != healSlot {
		t.Fatalf("pending order slot = %d, want heal slot %d", hs.order.slot, healSlot)
	}
	if !bot.hasDest {
		t.Fatal("pending heal did not arm movement toward the hurt ally")
	}
	if hs.attackTarget != 0 || hs.pvpTarget != 0 {
		t.Fatalf("same lane tick overwrote heal approach with attack (mob=%d pvp=%d)", hs.attackTarget, hs.pvpTarget)
	}
	if hs.cooldownUntil[healSlot-1] != 0 {
		t.Fatalf("out-of-range heal cast immediately instead of remaining pending (cooldownUntil=%.2f)", hs.cooldownUntil[healSlot-1])
	}
}
