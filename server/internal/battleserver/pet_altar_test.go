package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

// grimlokDinoOp returns the real «Вызов динозавра» summon op off the live roster, so
// these tests break if the Pet flag is ever dropped from the data rather than passing
// against a hand-made op that proves nothing about the shipped skill.
func grimlokDinoOp(t *testing.T) gamedata.Op {
	t.Helper()
	for _, a := range gamedata.Avatars() {
		for _, sk := range gamedata.SkillsFor(a).Skills {
			for _, op := range sk.Ops {
				if op.Kind == gamedata.OpSummon && op.Unit == "Mob_Dinosaur_Melee_01" {
					if !op.Pet {
						t.Fatalf("%s/%s summons the dinosaur but is not marked Pet: it will escort and pick its own fights", a.Prefab, sk.NameRu)
					}
					return op
				}
			}
		}
	}
	t.Fatal("no dinosaur summon op in the roster")
	return gamedata.Op{}
}

// TestResummonReplacesThePet: the dinosaur lives 180s, so without a uniqueness rule the
// player just spams the button and fields a pack.
func TestResummonReplacesThePet(t *testing.T) {
	s, c, _ := newHuntConn(t, "Avtr_HK_Grimlok")
	c.huntState.summonProtos = map[string]int32{"Mob_Dinosaur_Melee_01": 801}
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	op := grimlokDinoOp(t)
	ctx := opCtx{slot: 1, level: 1}

	s.summonLocked(c, op, ctx, now)
	if len(c.huntState.summons) != 1 {
		t.Fatalf("first cast summoned %d pets, want 1", len(c.huntState.summons))
	}
	var first int32
	for id := range c.huntState.summons {
		first = id
	}

	s.summonLocked(c, op, ctx, now+1)
	if n := len(c.huntState.summons); n != 1 {
		t.Fatalf("after re-summon there are %d dinosaurs, want 1 (the old one must die)", n)
	}
	if _, ok := c.huntState.summons[first]; ok {
		t.Error("re-summon kept the OLD dinosaur")
	}
}

// TestSwarmSummonsStillStack is the contrast: a fire-and-forget swarm is not a pet and
// must be unaffected by the uniqueness rule.
func TestSwarmSummonsStillStack(t *testing.T) {
	s, c, _ := newHuntConn(t, "Avtr_Dsb_Elgorm")
	c.huntState.summonProtos = map[string]int32{"Mob_ZombieCrawl_01": 800}
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	op := gamedata.Op{Kind: gamedata.OpSummon, Count: gamedata.PerLevel{2, 2, 2, 2},
		Lifetime: gamedata.PerLevel{20, 20, 20, 20}, HP: gamedata.PerLevel{100, 100, 100, 100},
		Dmg: gamedata.PerLevel{10, 10, 10, 10}, Unit: "Mob_ZombieCrawl_01"}
	ctx := opCtx{slot: 2, level: 1}

	s.summonLocked(c, op, ctx, now)
	s.summonLocked(c, op, ctx, now+1)
	if n := len(c.huntState.summons); n != 4 {
		t.Errorf("swarm summons = %d after two casts of 2, want 4: the pet rule leaked onto a swarm", n)
	}
}

// TestPetObeysAttackOrder: the pet fights what the player ordered, and NOTHING else --
// the nearer enemy it is standing next to must be ignored.
func TestPetObeysAttackOrder(t *testing.T) {
	s, c, _ := newHuntConn(t, "Avtr_HK_Grimlok")
	c.huntState.summonProtos = map[string]int32{"Mob_Dinosaur_Melee_01": 801}
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	idx := mobIndexByPrefab(t, "Mob_Skeleton_1H_Melee_01")
	mk := func(id int32, x float32) *mobState {
		m := &mobState{id: id, mobIdx: idx, mob: gamedata.Mobs()[idx], x: x, y: c.y, hp: 500, maxHP: 500}
		c.huntState.mobs[id] = m
		return m
	}
	near := mk(2700, c.x+2)   // right next to the pet
	ordered := mk(2701, c.x+7) // further away, but the one the player clicked

	s.summonLocked(c, grimlokDinoOp(t), opCtx{slot: 1, level: 1}, now)
	var sm *summonState
	for _, x := range c.huntState.summons {
		sm = x
	}
	sm.x, sm.y = c.x, c.y

	// No order yet: it must NOT pick a fight of its own, even standing on top of one.
	if got := s.petTargetLocked(c, sm, now); got != nil {
		t.Fatalf("an unordered pet acquired %d by itself", got.id)
	}

	s.orderPetsAttackLocked(c, ordered.id)
	got := s.petTargetLocked(c, sm, now)
	if got == nil {
		t.Fatal("pet ignored the attack order")
	}
	if got.id == near.id {
		t.Fatal("pet went for the NEAREST enemy instead of the ordered one")
	}
	if got.id != ordered.id {
		t.Fatalf("pet acquired %d, want the ordered %d", got.id, ordered.id)
	}
}

// TestPetBreaksOffOnMoveOrder: a move click means disengage and go, exactly as it does
// for the avatar itself.
func TestPetBreaksOffOnMoveOrder(t *testing.T) {
	s, c, _ := newHuntConn(t, "Avtr_HK_Grimlok")
	c.huntState.summonProtos = map[string]int32{"Mob_Dinosaur_Melee_01": 801}
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	idx := mobIndexByPrefab(t, "Mob_Skeleton_1H_Melee_01")
	enemy := &mobState{id: 2702, mobIdx: idx, mob: gamedata.Mobs()[idx], x: c.x + 3, y: c.y, hp: 500, maxHP: 500}
	c.huntState.mobs[enemy.id] = enemy

	s.summonLocked(c, grimlokDinoOp(t), opCtx{slot: 1, level: 1}, now)
	var sm *summonState
	for _, x := range c.huntState.summons {
		sm = x
	}
	s.orderPetsAttackLocked(c, enemy.id)
	if sm.ordTarget != enemy.id {
		t.Fatal("precondition: pet is not on the attack order")
	}

	s.orderPetsMoveLocked(c, c.x+20, c.y+20)
	if sm.ordTarget != 0 {
		t.Error("a move order did not break the pet's fight")
	}
	if !sm.ordMove || sm.ordX != c.x+20 || sm.ordY != c.y+20 {
		t.Errorf("pet did not take the move order: ordMove=%v at (%.1f,%.1f)", sm.ordMove, sm.ordX, sm.ordY)
	}
	if got := s.petTargetLocked(c, sm, now); got != nil {
		t.Errorf("pet still has target %d after a move order", got.id)
	}
}

// TestPetReaimsAfterAKill: its target dying must not leave it standing there -- it rolls
// onto the next enemy the way the player's own auto-attack chains.
func TestPetReaimsAfterAKill(t *testing.T) {
	s, c, _ := newHuntConn(t, "Avtr_HK_Grimlok")
	c.huntState.summonProtos = map[string]int32{"Mob_Dinosaur_Melee_01": 801}
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	idx := mobIndexByPrefab(t, "Mob_Skeleton_1H_Melee_01")
	mk := func(id int32, x float32) *mobState {
		m := &mobState{id: id, mobIdx: idx, mob: gamedata.Mobs()[idx], x: x, y: c.y, hp: 500, maxHP: 500}
		c.huntState.mobs[id] = m
		return m
	}
	first := mk(2703, c.x+2)
	second := mk(2704, c.x+4)

	s.summonLocked(c, grimlokDinoOp(t), opCtx{slot: 1, level: 1}, now)
	var sm *summonState
	for _, x := range c.huntState.summons {
		sm = x
	}
	sm.x, sm.y = c.x, c.y
	s.orderPetsAttackLocked(c, first.id)

	first.dead = true
	got := s.petTargetLocked(c, sm, now)
	if got == nil {
		t.Fatal("pet froze after its target died instead of rolling onto the next enemy")
	}
	if got.id != second.id {
		t.Fatalf("pet re-aimed at %d, want the remaining enemy %d", got.id, second.id)
	}
	if sm.ordTarget != second.id {
		t.Error("pet re-aimed but did not latch the new order")
	}
}

// TestGuardedAltarIsNotAttackable: the push rule is a TARGETING rule, not only a damage
// one. Guns up -> refused and flagged PHYS_IMM; guns down -> open.
func TestGuardedAltarIsNotAttackable(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	var altar *mobState
	for _, m := range inst.mobs {
		if m.altar && m.team == dotaEnemyTeam {
			altar = m
			break
		}
	}
	if altar == nil {
		t.Fatal("precondition: no enemy altar")
	}
	if !inst.dota.m.AltarGuardedByGuns() {
		t.Skip("this map does not gate its altar on guns")
	}

	if !c.altarShieldedLocked(altar) {
		t.Fatal("precondition: the altar is not shielded with its guns still up")
	}
	if got := altarPhysImm(inst.dota, altar); got != 1 {
		t.Errorf("shielded altar PHYS_IMM = %d, want 1 (the client keeps auto-attacking it)", got)
	}

	// Damage is refused...
	hp0 := altar.hp
	s.hitMobLocked(c, altar, 500, c.objID)
	if altar.hp != hp0 {
		t.Errorf("shielded altar took damage: %g -> %g", hp0, altar.hp)
	}
	// ...and so is auto-acquisition.
	c.x, c.y, c.snapT = altar.x+1, altar.y, float32(now)
	if got := s.nearestAttackableMobLocked(c, now, mobAggroRange); got != nil && got.id == altar.id {
		t.Error("auto-attack acquired a shielded altar")
	}

	// Drop every gun on that side: the altar opens up.
	for _, m := range inst.mobs {
		if m.structure && m.dotaRole == gamedata.DotaGun && m.team == altar.team {
			m.dead = true
		}
	}
	if c.altarShieldedLocked(altar) {
		t.Fatal("altar still shielded with all its guns dead -- the rule never releases")
	}
	if got := altarPhysImm(inst.dota, altar); got != 0 {
		t.Errorf("open altar PHYS_IMM = %d, want 0", got)
	}
	s.hitMobLocked(c, altar, 500, c.objID)
	if altar.hp >= hp0 {
		t.Errorf("open altar took no damage: %g -> %g", hp0, altar.hp)
	}
}

// TestNoDropsInShturm: «Штурм» is a match, not a farm. The Hunt control proves the gate
// is a mode rule and not a switch that turned loot off everywhere.
func TestNoDropsInShturm(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	boss := &mobState{mob: gamedata.Mob{Skills: []gamedata.BossSkill{{}}}} // always drops in Hunt
	for i := 0; i < 50; i++ {
		if s.rollDropLocked(c, boss) {
			t.Fatal("«Штурм» rolled a drop")
		}
	}
	trash := &mobState{}
	for i := 0; i < 2000; i++ {
		if s.rollDropLocked(c, trash) {
			t.Fatal("«Штурм» rolled a trash drop")
		}
	}
	_ = inst

	// Hunt control.
	sh, ch, _ := newHuntConn(t, "Avtr_Tank_Velial")
	ch.mvMu.Lock()
	defer ch.mvMu.Unlock()
	if !sh.rollDropLocked(ch, boss) {
		t.Error("a Hunt boss stopped dropping: the gate is not mode-scoped")
	}
}

// TestPetTickIgnoresEnemiesAndDoesNotEscort drives the REAL tick, not petTargetLocked
// directly. It pins the two halves of "the pet acts on clicks": with no order it neither
// picks a fight with the enemy it is standing on, nor walks back to its owner.
func TestPetTickIgnoresEnemiesAndDoesNotEscort(t *testing.T) {
	s, c, _ := newHuntConn(t, "Avtr_HK_Grimlok")
	c.huntState.summonProtos = map[string]int32{"Mob_Dinosaur_Melee_01": 801}
	now := float64(s.battleTime())
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	s.summonLocked(c, grimlokDinoOp(t), opCtx{slot: 1, level: 1}, now)
	var sm *summonState
	for _, x := range c.huntState.summons {
		sm = x
	}
	// Park it well away from the owner, with an enemy right on top of it and no order.
	sm.x, sm.y = c.x+25, c.y
	sm.nextSwing = 0
	px, py := sm.x, sm.y

	idx := mobIndexByPrefab(t, "Mob_Skeleton_1H_Melee_01")
	enemy := &mobState{id: 2705, mobIdx: idx, mob: gamedata.Mobs()[idx],
		x: sm.x + 1, y: sm.y, hp: 500, maxHP: 500}
	c.huntState.mobs[enemy.id] = enemy

	s.tickSummonsLocked(c, now)

	if enemy.hp != 500 {
		t.Errorf("an UNORDERED pet attacked the enemy beside it (hp %g): it picks its own fights", enemy.hp)
	}
	if sm.swingDoneAt != 0 {
		t.Error("an unordered pet started a swing")
	}
	if sm.vx != 0 || sm.vy != 0 {
		t.Errorf("an unordered pet is moving (v=%.2f,%.2f) -- it is escorting its owner", sm.vx, sm.vy)
	}
	if sm.x != px || sm.y != py {
		t.Errorf("unordered pet drifted from (%.1f,%.1f) to (%.1f,%.1f)", px, py, sm.x, sm.y)
	}

	// Positive control: ordered onto that same enemy, it fights through the same tick.
	s.orderPetsAttackLocked(c, enemy.id)
	s.tickSummonsLocked(c, now+0.1)
	if enemy.hp >= 500 {
		t.Errorf("an ORDERED pet did not attack (hp %g) -- the test above proves nothing", enemy.hp)
	}
}

// TestDoActionOnGuardedAltarIsRefused covers the CLICK path. The client's own
// auto-acquire honours PHYS_IMM, but SelfPlayer.PerformAttack does not consult it, so an
// explicit order still reaches the server and must be turned away here.
func TestDoActionOnGuardedAltarIsRefused(t *testing.T) {
	s, c, inst, _, _ := newDotaCaptureConn(t)
	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	var altar, gun *mobState
	for _, m := range inst.mobs {
		if m.altar && m.team == dotaEnemyTeam {
			altar = m
		}
		if m.structure && m.dotaRole == gamedata.DotaGun && m.team == dotaEnemyTeam && !m.dead {
			gun = m
		}
	}
	if altar == nil || gun == nil {
		t.Fatal("precondition: need an enemy altar and at least one enemy gun")
	}
	if !inst.dota.m.AltarGuardedByGuns() {
		t.Skip("this map does not gate its altar on guns")
	}

	act := attackProtoID(c.huntState.av)
	s.doActionLocked(c, -1, act, altar.id, 0, 0, false)
	if c.huntState.attackTarget == altar.id {
		t.Error("DO_ACTION on a shielded altar was accepted: the avatar walks over and swings for nothing")
	}

	// Positive control: the gun guarding it IS a legal order.
	s.doActionLocked(c, -1, act, gun.id, 0, 0, false)
	if c.huntState.attackTarget != gun.id {
		t.Fatalf("DO_ACTION on the gun was refused too (target=%d): the guard rejects everything",
			c.huntState.attackTarget)
	}
}
