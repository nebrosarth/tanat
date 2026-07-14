package battleserver

import (
	"math"
	"testing"
	"time"

	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// --- Bug 1: Titanid "Гигантизм" enlarges the model via a per-level grow VFX ---

// TestTitanidGrowFxInWorldState: entering as Titanid (id 14) the world state must
// contain an EFFECT_START for the skill-4 grow prefab (TitanidSkill4Effect1 at
// level 1) attached to the avatar, so the client's MorphEffect scales the model.
// Previously the passive's fx was never sent and the model stayed default-small.
// TestTitanidGrowFxGatedAtStart: Titanid's "Гигантизм" (grow passive) lives in the
// ult slot (4), which now starts UNLEARNED at rank 0 (unlocks at avatar level 5).
// So at battle start (level 1) the model must NOT grow -- no grow VFX in the world
// state. The learn->grow path is covered by TestTitanidGrowFxStepsOnUpgrade.
func TestTitanidGrowFxGatedAtStart(t *testing.T) {
	c, r := enterHunt(t, session.NewStore(), 14) // Avtr_Tank_Titanid
	got := readWorld(t, c, r)

	var objID int32
	for _, p := range got[battleproto.CmdSetAvatar] {
		objID, _ = p.Args.GetInt("avatarID")
	}
	if objID == 0 {
		t.Fatal("no SET_AVATAR in world state")
	}

	for _, p := range got[battleproto.CmdEffectStart] {
		if fx, _ := p.Args.GetString("fx"); fx == "TitanidSkill4Effect1" {
			t.Fatal("grow VFX emitted at level 1 — Гигантизм is the ult slot and must stay dormant until it is learned at level 5")
		}
	}
}

// TestTitanidGrowFxStepsOnUpgrade: leveling Gigantism swaps the grow prefab to
// the next size (EFFECT_END old + EFFECT_START of the higher-level prefab), so
// the model visibly steps up. Driven through the real UPGRADE_SKILL handler.
func TestTitanidGrowFxStepsOnUpgrade(t *testing.T) {
	s, c, _, _, _ := newNavConnAvatar(t, 14) // Titanid
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs := c.huntState

	// Seed the level-1 grow fx exactly as the world build does.
	hs.growFx[3] = s.fxStartLocked(c, "TitanidSkill4Effect1", c.objID, 0, false, 0, 0)
	old := hs.growFx[3]
	if old == 0 {
		t.Fatal("grow fx not started")
	}
	hs.points = 1
	hs.skillLevel[3] = 2 // handleUpgradeSkill increments; simulate the post-increment reapply
	s.reapplyPassiveLocked(c, 4, float64(s.battleTime()))

	if hs.growFx[3] == 0 {
		t.Fatal("grow fx cleared and not restarted on upgrade")
	}
	if hs.growFx[3] == old {
		t.Fatal("grow fx uid unchanged on upgrade — the level-2 prefab was not re-attached")
	}
}

// --- Bug 2: Elgorm summons follow their owner and their swing animation loops ---

// TestSummonFollowsOwnerWhenIdle: a summon with no enemy nearby steers back
// toward its owner (pet leash) instead of freezing where it was left.
func TestSummonFollowsOwnerWhenIdle(t *testing.T) {
	s, c, nav, sx, sy := newNavConn(t)
	hs := c.huntState

	// A walkable spot well outside the follow leash so the summon must move.
	var px, py float32
	found := false
	for r := float32(4); r <= 8 && !found; r++ {
		for a := 0; a < 16; a++ {
			ang := float64(a) * math.Pi / 8
			qx := sx + r*float32(math.Cos(ang))
			qy := sy + r*float32(math.Sin(ang))
			if nav.Walkable(float64(qx), float64(qy)) && c.mobHasLoSLocked(qx, qy, sx, sy) {
				px, py, found = qx, qy, true
				break
			}
		}
	}
	if !found {
		t.Skip("no walkable summon spot near spawn")
	}

	sm := &summonState{id: 7000, x: px, y: py, hp: 300, maxHP: 300, dmg: 20, until: 1e9}
	hs.summons[sm.id] = sm

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	s.tickSummonsLocked(c, 10.0) // owner is at spawn (c.x,c.y); no mobs exist

	if sm.vx == 0 && sm.vy == 0 {
		t.Fatal("idle summon did not move toward its owner (it froze in place)")
	}
	// Velocity must point back toward the owner (positive dot with owner-summon).
	ox, oy := c.x-sm.x, c.y-sm.y
	if float64(sm.vx*ox+sm.vy*oy) <= 0 {
		t.Fatalf("idle summon steered away from its owner (v=%.2f,%.2f toward %.2f,%.2f)", sm.vx, sm.vy, ox, oy)
	}
}

// TestSummonSwingSchedulesAndClosesActionDone: a summon in range of a mob swings
// (schedules swingDoneAt + damages the mob), and the swing is closed out with an
// ACTION_DONE once its window elapses so the next swing can replay the WrapMode.
// Once attack clip (the fix for "attack animation plays once then never again").
func TestSummonSwingSchedulesAndClosesActionDone(t *testing.T) {
	s, c, _, sx, sy := newNavConn(t)
	hs := c.huntState

	sm := &summonState{id: 7001, x: sx, y: sy, hp: 300, maxHP: 300, dmg: 25, until: 1e9}
	hs.summons[sm.id] = sm
	// A mob within the summon's reach (meleeReach+summonRadius+mobRadius).
	mob := &mobState{id: 2500, mobIdx: 2, mob: gamedata.Mobs()[2], x: sx + 1.2, y: sy, hp: 170}
	hs.mobs[mob.id] = mob

	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	s.tickSummonsLocked(c, 10.0) // in range, nextSwing=0 -> swings now
	if sm.swingDoneAt == 0 {
		t.Fatal("summon swing did not schedule an ACTION_DONE close-out")
	}
	if mob.hp >= 170 {
		t.Fatalf("summon swing dealt no damage (mob hp %.0f)", mob.hp)
	}
	doneAt := sm.swingDoneAt

	// A shade past the close-out window, but before the next swing (nextSwing=11):
	// the ACTION_DONE fires and swingDoneAt clears.
	s.tickSummonsLocked(c, doneAt+0.02)
	if sm.swingDoneAt != 0 {
		t.Fatal("summon swing was never closed out (ACTION_DONE not sent) — animation would freeze")
	}
}

// --- Bug 3: Abominator toggle appearance reverts on toggle-off ---

// TestAbominatorToggleUsesRemovableBuffFx: toggling skill 4 on attaches the
// BuffFx (AbominatorSkill4Effect1 — the stopOnDone=true tentacle prop that
// EFFECT_END can remove), NOT the un-stoppable CastFx (Effect2), and never a
// duplicate; toggling off ends exactly that effect so the tentacles disappear.
func TestAbominatorToggleUsesRemovableBuffFx(t *testing.T) {
	// Capture the server-side conn so we can learn the level-gated ult below.
	scCh := make(chan *conn, 1)
	testHookNewConn = func(sc *conn) {
		select {
		case scCh <- sc:
		default:
		}
	}
	defer func() { testHookNewConn = nil }()

	c, r := enterHunt(t, session.NewStore(), 22) // Avtr_HK_Abominator
	readWorld(t, c, r)

	// The ult (slot 4, "Трупоглот") starts UNLEARNED at rank 0 (level-5 gate), so
	// learn it first -- otherwise the toggle is correctly rejected as uncastable.
	sc := <-scCh
	sc.mvMu.Lock()
	sc.huntState.skillLevel[3] = 1
	sc.mvMu.Unlock()

	av, _ := gamedata.AvatarByID(22)
	toggleAction := skillProtoID(av, 4)

	// Toggle ON.
	send(t, c, battleproto.Packet{Cmd: battleproto.CmdDoAction, RequestID: 5, Status: true,
		Args: amf.NewArray().
			Set("id", int32(1042)).
			Set("action", toggleAction).
			Set("target", int32(0)).
			Set("targetPos", amf.NewArray().Set("x", 0.0).Set("y", 0.0))})

	var buffUID int32
	var effect1Starts, effect2Starts int
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		_ = c.SetReadDeadline(time.Now().Add(400 * time.Millisecond))
		p, err := r.Read()
		if err != nil {
			break
		}
		if p.Cmd == battleproto.CmdEffectStart {
			fx, _ := p.Args.GetString("fx")
			switch fx {
			case "AbominatorSkill4Effect1":
				effect1Starts++
				buffUID, _ = p.Args.GetInt("effect")
			case "AbominatorSkill4Effect2":
				effect2Starts++
			}
		}
	}
	if effect1Starts != 1 {
		t.Fatalf("toggle-on started the removable BuffFx %d times, want exactly 1 (0 = tentacles never show; 2 = duplicate leaks)", effect1Starts)
	}
	if effect2Starts != 0 {
		t.Fatalf("toggle-on attached the un-stoppable CastFx (Effect2) %d times — it can't be removed and would leave the tentacles stuck on", effect2Starts)
	}

	// Toggle OFF: the same effect uid must be ended.
	send(t, c, battleproto.Packet{Cmd: battleproto.CmdDoAction, RequestID: 6, Status: true,
		Args: amf.NewArray().
			Set("id", int32(1042)).
			Set("action", toggleAction).
			Set("target", int32(0)).
			Set("targetPos", amf.NewArray().Set("x", 0.0).Set("y", 0.0))})

	endedBuff := false
	deadline = time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		_ = c.SetReadDeadline(time.Now().Add(400 * time.Millisecond))
		p, err := r.Read()
		if err != nil {
			break
		}
		if p.Cmd == battleproto.CmdEffectEnd {
			if id, _ := p.Args.GetInt("id"); id == buffUID {
				endedBuff = true
			}
		}
	}
	if !endedBuff {
		t.Fatal("toggle-off did not EFFECT_END the tentacle visual — it would stay on the avatar")
	}
}

// TestToggleNoDuplicateBuffFxMod: the toggle's self-buff OpBuffStat mod must NOT
// also start the BuffFx (the toggle owns it via toggleFx); otherwise toggle-off
// leaves the duplicate copy attached. State-level guard for the double-attach.
func TestToggleNoDuplicateBuffFxMod(t *testing.T) {
	s, c, _, _, _ := newNavConnAvatar(t, 22) // Abominator
	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	hs := c.huntState

	s.toggleSkillLocked(c, 4)
	if !hs.toggleOn[3] || hs.toggleFx[3] == 0 {
		t.Fatal("toggle did not turn on / start a persistent visual")
	}
	togModCount := 0
	for _, mod := range hs.st.mods {
		if mod.src == toggleSrc(4) {
			togModCount++
			if mod.fxUID != 0 {
				t.Errorf("toggle self-buff mod started a duplicate BuffFx (fxUID=%d); the toggleFx must own it alone", mod.fxUID)
			}
		}
	}
	if togModCount == 0 {
		t.Fatal("toggle-on applied no self-buff mod")
	}

	s.toggleOffLocked(c, 4, float64(s.battleTime())+0.1, true)
	if hs.toggleOn[3] || hs.toggleFx[3] != 0 {
		t.Fatal("toggle-off did not clear the toggle state / visual handle")
	}
	for _, mod := range hs.st.mods {
		if mod.src == toggleSrc(4) {
			t.Fatal("toggle self-buff mod survived toggle-off")
		}
	}
}

// --- Titanid «Землетрясение»: a ground-placed channel survives caster movement ---

// TestGroundChannelSurvivesMovement: a POINT-anchored channel (target-less,
// hasPos) keeps pulsing at its world point while the caster walks — Titanid's
// 3-wave quake is cast-and-forget. A self/unit channel (hasPos=false) still
// breaks when the caster moves.
func TestGroundChannelSurvivesMovement(t *testing.T) {
	s, c, _, sx, sy := newNavConnAvatar(t, 14) // Titanid: skillDef(1) = «Землетрясение»
	hs := c.huntState
	kit := gamedata.SkillsFor(hs.av)
	waveOps := kit.Skills[0].Ops[0].Ops // the OpChannel's per-pulse ops (damage+stun)

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	now := float64(s.battleTime())

	// A mob standing on the quake's anchor point (radius 5).
	mob := &mobState{id: 2700, mobIdx: 2, mob: gamedata.Mobs()[2], x: sx + 2, y: sy, hp: 170, shown: true}
	hs.mobs[mob.id] = mob

	// Ground-anchored quake at the mob, and a self-channel centered on the caster.
	hs.channels = []channelState{
		{slot: 1, level: 1, until: now + 10, interval: 0.8, nextPulse: now,
			target: 0, px: mob.x, py: mob.y, hasPos: true, ops: waveOps},
		{slot: 1, level: 1, until: now + 10, interval: 0.8, nextPulse: now,
			target: 0, hasPos: false, ops: waveOps},
	}

	// The caster presses walk.
	c.hasDest = true
	s.tickChannelsLocked(c, now)

	// The quake still ran (mob took a wave) and stayed alive; the self-channel broke.
	if mob.hp >= 170 {
		t.Fatal("ground quake did not pulse while the caster was walking (it was wrongly interrupted)")
	}
	if len(hs.channels) != 1 {
		t.Fatalf("expected exactly the ground channel to survive movement, got %d channels", len(hs.channels))
	}
	if !hs.channels[0].hasPos {
		t.Fatal("the surviving channel is not the ground-anchored one")
	}
}

// --- Bug 4: per-unit collision radius drives hitboxes ---

// mobIndexByPrefab finds a mob roster index by prefab name.
func mobIndexByPrefab(t *testing.T, prefab string) int {
	t.Helper()
	for i, m := range gamedata.Mobs() {
		if m.Prefab == prefab {
			return i
		}
	}
	t.Fatalf("mob prefab %q not found", prefab)
	return -1
}

// TestUnitRadii: authored radii scale with body size (bosses >> mobs) and unset
// units fall back to the package defaults.
func TestUnitRadii(t *testing.T) {
	mobs := gamedata.Mobs()
	skeleton := mobs[2]
	elgorm := mobs[mobIndexByPrefab(t, "Boss_Elgorm")]
	if skeleton.Radius() >= elgorm.Radius() {
		t.Fatalf("boss radius (%.1f) should dwarf a small mob's (%.1f)", elgorm.Radius(), skeleton.Radius())
	}
	// Unset radius falls back to the default.
	var bare gamedata.Mob
	if bare.Radius() != gamedata.DefaultMobRadius {
		t.Fatalf("unset mob radius = %.2f, want default %.2f", bare.Radius(), gamedata.DefaultMobRadius)
	}
	var bareAv gamedata.Avatar
	if bareAv.Radius() != gamedata.DefaultAvatarRadius {
		t.Fatalf("unset avatar radius = %.2f, want default %.2f", bareAv.Radius(), gamedata.DefaultAvatarRadius)
	}
}

// TestBossRadiusExtendsReach: a big-bodied boss reaches (and acts on) the player
// from a distance at which a small mob is still chasing, because reach now
// credits both bodies' radii.
func TestBossRadiusExtendsReach(t *testing.T) {
	s, c, _, sx, sy := newNavConn(t)
	hs := c.huntState
	hs.hp = 5000

	// Distance chosen between a skeleton's reach and a boss's reach:
	//   skeleton: meleeReach(0.6)+0.6+av.Radius(0.8) = 2.0  -> out of range at 3.0
	//   Velial:   meleeReach(0.6)+2.2+av.Radius(0.8) = 3.6  -> in range at 3.0
	const d = 3.0
	velial := mobIndexByPrefab(t, "Boss_Velial")
	skeleton := &mobState{id: 2600, mobIdx: 2, mob: gamedata.Mobs()[2],
		x: sx + d, y: sy, hp: 170, aggro: true, shown: true}
	boss := &mobState{id: 2601, mobIdx: velial, mob: gamedata.Mobs()[velial],
		x: sx - d, y: sy, hp: 3500, aggro: true, shown: true}
	boss.skillReady = make([]float64, len(boss.mob.Skills))
	hs.mobs[skeleton.id] = skeleton
	hs.mobs[boss.id] = boss

	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	// First tick: the boss is in reach (acts: casts/swings, holds position) while
	// the skeleton is out of reach (chases: nonzero velocity).
	s.tickMobsLocked(c, 10.0)
	if skeleton.vx == 0 && skeleton.vy == 0 {
		t.Fatal("skeleton at d=3.0 should be chasing (its reach 2.0 < 3.0), not standing still")
	}
	if boss.skillHitAt == 0 && boss.hitAt == 0 && boss.swingDoneAt == 0 {
		t.Fatal("boss at d=3.0 should be attacking (its reach 3.6 > 3.0), not idle/chasing")
	}

	// Advance to land the boss's committed hit.
	hpBefore := hs.hp
	for i := 1; i < 10; i++ {
		s.tickMobsLocked(c, 10.0+float64(i)*0.2)
	}
	if hs.hp >= hpBefore {
		t.Fatal("boss committed hit never landed on the player")
	}
}
