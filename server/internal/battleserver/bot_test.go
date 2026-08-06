package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

// TestSpawnDotaBotsFillsBalancedTeams: spawning bots up to a target headcount keeps the
// two sides within 1 of each other and assigns lanes in the 2/1/2 pattern per side.
// newDotaConn's solo member never calls assignSide() (it defaults to Human without
// advancing dotaState.nextSide) -- exactly the "explicit/pre-assigned side" case a real
// matchmade player hits, which is what desynced the fill in the first place (a 10-headcount
// fill split 6/4 before balancedDotaSideLocked started counting actual membership instead
// of trusting the alternator).
func TestSpawnDotaBotsFillsBalancedTeams(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()

	human.lock()
	defer human.unlock()
	s.spawnDotaBotsLocked(inst, 10)

	if got := len(inst.members); got != 10 {
		t.Fatalf("members = %d, want 10 (1 real + 9 bots)", got)
	}
	if got := len(inst.bots); got != 9 {
		t.Fatalf("bot brains = %d, want 9", got)
	}
	humanN, elfN := 0, 0
	for _, mem := range inst.members {
		if mem.playerTeam() == dotaTeamHuman {
			humanN++
		} else {
			elfN++
		}
	}
	if humanN+elfN != 10 {
		t.Fatalf("human+elf = %d, want 10", humanN+elfN)
	}
	if diff := humanN - elfN; diff > 1 || diff < -1 {
		t.Fatalf("sides = human %d / elf %d, want within 1 of each other (5/5 for 10)", humanN, elfN)
	}
	for _, b := range inst.bots {
		if b.lane < 0 || b.lane >= len(inst.dota.m.Lanes) {
			t.Fatalf("bot lane = %d, out of range [0,%d)", b.lane, len(inst.dota.m.Lanes))
		}
	}
}

// TestSpawnDotaBotsBalancesAroundExplicitElfPlayer: the same desync bug, mirrored --
// a real player pre-assigned to the ELF side (dotaPlayerConn's explicit team, exactly how
// a matchmaker-assigned PendingBattle.Team lands) must still get an evenly split fill, not
// have the fill treat them as if nextSide already accounted for them.
func TestSpawnDotaBotsBalancesAroundExplicitElfPlayer(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	// Replace the default Human solo member with an explicit Elf one, the same way
	// sendHuntWorldState assigns a matchmade player's side straight from
	// PendingBattle.Team without ever touching dotaState.assignSide().
	delete(inst.members, human.objID)
	elf := dotaPlayerConn(t, s, inst, 2000, dotaTeamElf, human.x, human.y)

	elf.lock()
	defer elf.unlock()
	s.spawnDotaBotsLocked(inst, 10)

	humanN, elfN := 0, 0
	for _, mem := range inst.members {
		if mem.playerTeam() == dotaTeamHuman {
			humanN++
		} else {
			elfN++
		}
	}
	if humanN+elfN != 10 {
		t.Fatalf("human+elf = %d, want 10", humanN+elfN)
	}
	if diff := humanN - elfN; diff > 1 || diff < -1 {
		t.Fatalf("sides = human %d / elf %d, want within 1 of each other (5/5 for 10)", humanN, elfN)
	}
}

// TestBotLastHitsLowHPCreep: a bot with an enemy creep in range that will die to its next
// swing must attack it, not walk past it toward the lane point.
func TestBotLastHitsLowHPCreep(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	b := &botBrain{c: human, slot: 0, lane: 0, phase: botPhaseLane}

	m := &mobState{id: 65100, mobIdx: inst.dota.m.ElfCreepMelee, mob: gamedata.Mobs()[inst.dota.m.ElfCreepMelee],
		x: human.x + 1, y: human.y, hp: 1, maxHP: 200, team: dotaTeamElf, shown: true, active: true}

	human.lock()
	defer human.unlock()
	inst.mobs[m.id] = m
	now := float64(s.battleTime())
	s.botLaneTickLocked(b, now)

	if human.huntState.attackTarget != m.id {
		t.Fatalf("bot did not attack the killable creep in range (attackTarget=%d, want %d)",
			human.huntState.attackTarget, m.id)
	}
}

// TestBotRetreatsAtLowHP: a bot below the retreat HP threshold must head home, not engage.
func TestBotRetreatsAtLowHP(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	b := &botBrain{c: human, slot: 0, lane: 0, phase: botPhaseLane}

	m := &mobState{id: 65101, mobIdx: inst.dota.m.ElfCreepMelee, mob: gamedata.Mobs()[inst.dota.m.ElfCreepMelee],
		x: human.x + 1, y: human.y, hp: 1, maxHP: 200, team: dotaTeamElf, shown: true, active: true}

	human.lock()
	defer human.unlock()
	inst.mobs[m.id] = m
	human.huntState.hp = 1 // far below botRetreatHPFrac of max
	now := float64(s.battleTime())
	s.botLaneTickLocked(b, now)

	if human.huntState.attackTarget != 0 {
		t.Fatal("a critically low-HP bot attacked instead of retreating")
	}
	if !b.retreating {
		t.Fatal("bot did not latch retreating at critical HP")
	}
	hx, hy := botHomeLocked(human)
	if !human.hasDest && (human.x != hx || human.y != hy) {
		t.Fatalf("retreating bot neither moved toward home (%.1f,%.1f) nor is already there (at %.1f,%.1f)",
			hx, hy, human.x, human.y)
	}
}

// TestBotRetreatCancelsOwnAttack: a bot that was already auto-attacking (mob or PvP) when
// it drops to retreat HP must actually leave -- not have its own still-armed attack chase
// walk it right back into the fight. Reported live: a low-HP bot tried to retreat and its
// own auto-attack kept pulling it back. handleMove cancels this for a real player's click,
// but a bot's move decision never goes through handleMove at all.
func TestBotRetreatCancelsOwnAttack(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, human.x+2, human.y)
	b := &botBrain{c: human, slot: 0, lane: 0, phase: botPhaseLane}

	human.lock()
	defer human.unlock()
	// Pull the bot away from its own spawn/home first -- newDotaConn spawns exactly on
	// home, which would make the retreat move a same-point no-op and defeat the point of
	// asserting hasDest below.
	human.x, human.snapT = human.x+40, s.battleTime()
	s.startPvpAttackLocked(human, elf)
	if human.huntState.pvpTarget != elf.objID {
		t.Fatal("setup: PvP attack did not start")
	}
	human.huntState.hp = 1 // far below botRetreatHPFrac of max
	now := float64(s.battleTime())
	s.botLaneTickLocked(b, now)

	if human.huntState.pvpTarget != 0 {
		t.Fatalf("pvpTarget = %d after the bot decided to retreat, want 0 (still armed to chase back)",
			human.huntState.pvpTarget)
	}
	if !human.hasDest {
		t.Fatal("retreating bot has no destination -- its own attack chase blocked the move")
	}
}

// TestBotAvoidsOutnumberedFight: a bot must not engage a lone enemy hero who has two
// teammates standing right next to them.
func TestBotAvoidsOutnumberedFight(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	b := &botBrain{c: human, slot: 0, lane: 0, phase: botPhaseLane}

	elf1 := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, human.x+3, human.y)
	elf2 := dotaPlayerConn(t, s, inst, 1002, dotaTeamElf, human.x+3.5, human.y+0.5)
	_ = elf2

	human.lock()
	defer human.unlock()
	now := float64(s.battleTime())
	acted := s.botCombatTickLocked(b, now)

	if acted || human.huntState.pvpTarget != 0 {
		t.Fatalf("bot engaged a 1-vs-2 fight (pvpTarget=%d, acted=%v)", human.huntState.pvpTarget, acted)
	}
	_ = elf1
}

// TestBotEngagesFavorableFight: a bot facing a single, similarly-escorted enemy hero (a
// fair 1v1) must commit to the fight.
func TestBotEngagesFavorableFight(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	b := &botBrain{c: human, slot: 0, lane: 0, phase: botPhaseLane}
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, human.x+3, human.y)

	human.lock()
	defer human.unlock()
	now := float64(s.battleTime())
	acted := s.botCombatTickLocked(b, now)

	if !acted || human.huntState.pvpTarget != elf.objID {
		t.Fatalf("bot did not engage a fair 1v1 (acted=%v pvpTarget=%d, want %d)",
			acted, human.huntState.pvpTarget, elf.objID)
	}
}

// TestBotDoesNotEngageNearEnemyStructure: a bot must not pick a fight with an enemy hero
// standing inside their own tower/cannon's range -- fighting there means fighting the
// structure too, not just the hero.
func TestBotDoesNotEngageNearEnemyStructure(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	b := &botBrain{c: human, slot: 0, lane: 0, phase: botPhaseLane}
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, human.x+3, human.y)
	gun := &mobState{id: 65200, structure: true, team: dotaTeamElf, x: elf.x, y: elf.y, hp: 100, maxHP: 100}

	human.lock()
	defer human.unlock()
	inst.mobs[gun.id] = gun
	now := float64(s.battleTime())
	acted := s.botCombatTickLocked(b, now)

	if acted || human.huntState.pvpTarget != 0 {
		t.Fatalf("bot engaged a hero standing in its own structure's range (pvpTarget=%d, acted=%v)",
			human.huntState.pvpTarget, acted)
	}
}

// TestBotBreaksOffChaseIntoEnemyStructureRange: a bot already mid-chase must actually
// disengage once press the fight would mean tower-diving, not just stop picking new
// fights. Reported live: a bot kept chasing a fleeing hero straight through the enemy's
// own cannons and onto their fountain -- armPvpAttackTimer has no notion of "too deep", so
// the brain dropping the target has to actually cancel the attack, not just stop re-
// affirming it.
func TestBotBreaksOffChaseIntoEnemyStructureRange(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	b := &botBrain{c: human, slot: 0, lane: 0, phase: botPhaseLane}
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, human.x+3, human.y)

	human.lock()
	defer human.unlock()
	s.startPvpAttackLocked(human, elf)
	b.engageTarget = elf.objID
	if human.huntState.pvpTarget != elf.objID {
		t.Fatal("setup: PvP attack did not start")
	}
	// The chase has now carried the fleeing target next to their own cannon.
	gun := &mobState{id: 65201, structure: true, team: dotaTeamElf, x: elf.x, y: elf.y, hp: 100, maxHP: 100}
	inst.mobs[gun.id] = gun
	now := float64(s.battleTime())
	acted := s.botCombatTickLocked(b, now)

	if acted || human.huntState.pvpTarget != 0 {
		t.Fatalf("bot kept chasing into its own target's structure range (pvpTarget=%d, acted=%v)",
			human.huntState.pvpTarget, acted)
	}
}

// TestBotSpendsSkillPointOnAvailability: a bot with a banked point and a learnable slot
// must spend it the same tick, through the real UPGRADE_SKILL validation.
func TestBotSpendsSkillPointOnAvailability(t *testing.T) {
	s, human, _, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	b := &botBrain{c: human, slot: 0, lane: 0, phase: botPhaseLane}

	human.lock()
	defer human.unlock()
	human.huntState.points = 1
	s.botSpendSkillPointLocked(b)

	total := int32(0)
	for _, r := range human.huntState.skillLevel {
		total += r
	}
	if total != 1 {
		t.Fatalf("skill ranks total = %d after spending 1 point, want 1", total)
	}
	if human.huntState.points != 0 {
		t.Fatalf("points left = %d, want 0", human.huntState.points)
	}
}

// TestBotBuysAffordableItem: a bot with enough gold for a root item in its preferred tree
// must buy it.
func TestBotBuysAffordableItem(t *testing.T) {
	s, human, _, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	b := &botBrain{c: human, slot: 0, lane: 0, phase: botPhaseLane}

	store := s.Store
	store.CreateBotHero(human.selfPlayerID, "test")
	money, _, _ := store.HeroMoney(human.selfPlayerID)
	if money <= 0 {
		t.Fatalf("test hero starting money = %d, want positive", money)
	}

	human.lock()
	defer human.unlock()
	now := float64(s.battleTime())
	s.botBuyItemsLocked(b, now)

	if len(human.huntState.ownedTreeItems) == 0 {
		t.Fatal("bot did not buy any item despite having starting gold")
	}
}

// TestBotHealsHurtAlly: a support-type bot with a ready FRIEND heal and a badly hurt
// nearby ally must cast on the ally, not itself, and the ally's HP must actually rise.
func TestBotHealsHurtAlly(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Sp_Arianna")
	defer cleanup()
	b := &botBrain{c: human, slot: 0, lane: 0, phase: botPhaseLane}
	ally := dotaPlayerConn(t, s, inst, 1001, dotaTeamHuman, human.x+2, human.y)

	human.lock()
	defer human.unlock()
	hs := human.huntState
	healSlot := 0
	for slot := 1; slot <= 4; slot++ {
		def := hs.skillDef(slot)
		if skillHasTargetFlag(def, "FRIEND") &&
			(botSkillHasOp(def, gamedata.OpHeal) || botSkillHasOp(def, gamedata.OpHot) || botSkillHasOp(def, gamedata.OpShield)) {
			healSlot = slot
			break
		}
	}
	if healSlot == 0 {
		t.Fatal("Arianna's kit has no FRIEND heal/hot/shield skill -- wrong test avatar")
	}
	now := float64(s.battleTime())
	hs.skillLevel[healSlot-1] = 1
	hs.mana = hs.maxManaLocked(now)
	ally.huntState.hp = 1

	acted := s.botConsiderHealLocked(b, now)
	if !acted {
		t.Fatal("bot did not cast its ready heal on a critically hurt ally")
	}
	// The cast's payload may be delayed (a cast wind-up, e.g. Arianna's PayloadDelay=0.3)
	// rather than landing synchronously -- run the queued payload forward past it.
	s.runDuePayloadsLocked(human, now+1.0)
	// The chosen skill may be an instant heal, a heal-over-time stream, or a shield --
	// any of the three is a real support action; only silence is a failure.
	ah := ally.huntState
	if ah.hp <= 1 && len(ah.st.hots) == 0 && ah.st.shield <= 0 {
		t.Fatalf("ally shows no sign of being healed/shielded after the bot's cast (hp=%g hots=%d shield=%g)",
			ah.hp, len(ah.st.hots), ah.st.shield)
	}
}
