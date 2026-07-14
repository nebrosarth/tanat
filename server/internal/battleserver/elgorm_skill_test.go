package battleserver

import (
	"math"
	"strings"
	"testing"

	"tanatserver/internal/gamedata"
)

// TestElgormGuardSpawnsAtPoint: «Гвардия мертвых» (a POINT summon) spawns its guls
// tightly around the clicked point, NOT around the caster.
func TestElgormGuardSpawnsAtPoint(t *testing.T) {
	s, c, _, sx, sy := newNavConnAvatar(t, 31) // Elgorm
	hs := c.huntState
	hs.summonProtos = map[string]int32{"Mob_ZombieCrawl_01": 800}

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	c.x, c.y, c.snapT = sx, sy, float32(0)
	now := float64(s.battleTime())

	px, py := sx+8, sy // aim point, 8u from the caster
	op := gamedata.Op{
		Kind: gamedata.OpSummon, Count: gamedata.PerLevel{3, 3, 3, 3},
		Lifetime: gamedata.PerLevel{20, 20, 20, 20}, HP: gamedata.PerLevel{120, 120, 120, 120},
		Dmg: gamedata.PerLevel{10, 10, 10, 10}, Unit: "Mob_ZombieCrawl_01",
	}
	s.summonLocked(c, op, opCtx{slot: 2, level: 1, hasPos: true, px: px, py: py}, now)

	if len(hs.summons) != 3 {
		t.Fatalf("expected 3 guls, got %d", len(hs.summons))
	}
	for _, sm := range hs.summons {
		dPoint := math.Hypot(float64(sm.x-px), float64(sm.y-py))
		dCaster := math.Hypot(float64(sm.x-sx), float64(sm.y-sy))
		if dPoint > summonSpawnRadius+0.01 {
			t.Errorf("gul at (%.1f,%.1f) is %.1f from the cast point, want <= %.1f", sm.x, sm.y, dPoint, summonSpawnRadius)
		}
		if dCaster < 5 {
			t.Errorf("gul spawned %.1f from the caster -- it should be out at the point (~8u away)", dCaster)
		}
	}
}

// TestSummonProtoHasIcon: a summon whose prefab is a mob model borrows that mob's
// name + enemy-card icon (the "guls have no icon" fix).
func TestSummonProtoHasIcon(t *testing.T) {
	desc := summonUnitProtoDesc("Mob_ZombieCrawl_01", 120)
	if strings.Contains(desc, `<Icon value=""/>`) {
		t.Error("summon proto still has an empty icon")
	}
	if !strings.Contains(desc, `Gui/Mobs/Icons/Mob_Zombie`) {
		t.Errorf("summon proto missing the ghoul icon, got: %s", desc)
	}
	if !strings.Contains(desc, `IDS_Mob_ZombieCrawl_Name`) {
		t.Errorf("summon proto missing the ghoul name, got: %s", desc)
	}
}

// TestElgormArrowsAreBeam: «Стрелы Аркана» must resolve as a line/beam (AoEWidth>0,
// no circle radius), like Velial's «Разлом», not a radius around the aim point.
func TestElgormArrowsAreBeam(t *testing.T) {
	a, ok := gamedata.AvatarByID(31)
	if !ok {
		t.Fatal("Elgorm missing")
	}
	sk := gamedata.SkillsFor(a).Skills[3]
	if sk.AoEWidth <= 0 {
		t.Errorf("Elgorm skill 4 AoEWidth=%v, want a positive beam width", sk.AoEWidth)
	}
	if sk.AoERadius != 0 {
		t.Errorf("Elgorm skill 4 AoERadius=%v, want 0 (beam, not circle)", sk.AoERadius)
	}
}

// TestElgormSkullBouncesBetweenEnemies: «Блуждающий ужас» is a chaining skull -- it
// strikes the cast target (always first, on impact not on cast), then hops to a
// RANDOM other enemy in range, up to `steps` hits (2 at rank 1). With three enemies
// and a 2-hit budget exactly one of the other two is struck (whichever the random
// hop picks); the third is left untouched.
func TestElgormSkullBouncesBetweenEnemies(t *testing.T) {
	s, c, _, sx, sy := newNavConnAvatar(t, 31) // Elgorm
	hs := c.huntState
	bounceOp := gamedata.SkillsFor(hs.av).Skills[0].Ops[0]
	if bounceOp.Kind != gamedata.OpBounce {
		t.Fatalf("skill 1 op[0] should be OpBounce, got %q", bounceOp.Kind)
	}

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	c.x, c.y, c.snapT = sx, sy, float32(0)
	now := float64(s.battleTime())

	skel := gamedata.Mobs()[2]
	mk := func(id int32, dx float32) *mobState {
		m := &mobState{id: id, mobIdx: 2, mob: skel, x: sx + dx, y: sy, hp: 500, maxHP: 500, shown: true}
		hs.mobs[id] = m
		hs.tr.add(id)
		return m
	}
	a := mk(2900, 2)  // cast target -- always the first hit
	b := mk(2901, 4)  // one of the two possible hop targets (both in range of A)
	cc := mk(2902, 5) // the other; the random hop lands on exactly one of b/cc

	// Cast (rank 1 -> steps = 2 total hits): the skull is THROWN but no damage lands
	// yet -- it must fly to the first enemy first (damage on impact, not on cast).
	s.applyOpsLocked(c, []gamedata.Op{bounceOp},
		opCtx{slot: 1, level: 1, target: a, px: a.x, py: a.y, hasPos: true}, now)
	if a.hp < 500 || a.st.stunned(now) {
		t.Fatalf("first target took damage on cast -- it should land on impact: hp=%.0f stunned=%v", a.hp, a.st.stunned(now))
	}

	// The skull reaches the cast target after its flight (Distance/skullMoverSpeed);
	// the hop has NOT happened yet, so both other enemies are still untouched.
	s.tickBouncesLocked(c, now+0.2)
	if a.hp >= 500 || !a.st.stunned(now+0.2) {
		t.Fatalf("skull did not strike the cast target on arrival: hp=%.0f stunned=%v", a.hp, a.st.stunned(now+0.2))
	}
	if b.hp < 500 || cc.hp < 500 {
		t.Fatal("a hop landed before the skull left the cast target")
	}

	// Run the single hop out; it goes to a random one of {b, cc}.
	for tick := 0; tick < 40 && len(hs.bounces) > 0; tick++ {
		s.tickBouncesLocked(c, now+0.25+0.05*float64(tick+1))
	}
	hopB, hopC := b.hp < 500, cc.hp < 500
	if hopB == hopC {
		t.Errorf("exactly one of the other two enemies should be hit (2-hit budget): B=%v C=%v", hopB, hopC)
	}
	if len(hs.bounces) != 0 {
		t.Errorf("bounce chain should be finished, %d left", len(hs.bounces))
	}
}

// TestSkullFlightMatchesClientMover: the server's skull flight time is the client
// SmoothMove formula Distance/mSpeed (mSpeed=15, from VFX_Avtr_Dsb_Elgorm_skill1_prop01),
// floored by skullMinFlight so a coincident target still shows a throw.
func TestSkullFlightMatchesClientMover(t *testing.T) {
	if got := skullFlightTime(0, 0, 30, 0); math.Abs(got-30.0/15.0) > 1e-9 {
		t.Errorf("flight for 30u = %v, want %v (Distance/15)", got, 30.0/15.0)
	}
	if got := skullFlightTime(0, 0, 3, 4); math.Abs(got-5.0/15.0) > 1e-9 {
		t.Errorf("flight for 5u (3-4-5) = %v, want %v", got, 5.0/15.0)
	}
	if got := skullFlightTime(0, 0, 0, 0); got != skullMinFlight {
		t.Errorf("coincident-target flight = %v, want the %v floor", got, skullMinFlight)
	}
}

// TestElgormSkullPingPongsBetweenTwoEnemies: with only two enemies and a 4-hit
// budget (rank 3 -> 3 bounces), the skull must bounce A->B->A->B rather than
// fizzle after one hop -- it may revisit enemies, only avoiding the mob it is
// currently standing on.
func TestElgormSkullPingPongsBetweenTwoEnemies(t *testing.T) {
	s, c, _, sx, sy := newNavConnAvatar(t, 31) // Elgorm
	hs := c.huntState
	bounceOp := gamedata.SkillsFor(hs.av).Skills[0].Ops[0]
	if bounceOp.Kind != gamedata.OpBounce {
		t.Fatalf("skill 1 op[0] should be OpBounce, got %q", bounceOp.Kind)
	}
	// Count = total impacts = rank + 1, so rank 3 gives 3 bounces (4 impacts total).
	if got := int(bounceOp.Count.At(3)); got != 4 {
		t.Fatalf("rank 3 should be a 4-impact chain (3 bounces), got %d", got)
	}

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	c.x, c.y, c.snapT = sx, sy, float32(0)
	now := float64(s.battleTime())

	skel := gamedata.Mobs()[2]
	mk := func(id int32, dx float32) *mobState {
		m := &mobState{id: id, mobIdx: 2, mob: skel, x: sx + dx, y: sy, hp: 5000, maxHP: 5000, shown: true}
		hs.mobs[id] = m
		hs.tr.add(id)
		return m
	}
	a := mk(3100, 2) // cast target
	b := mk(3101, 4) // the only other enemy

	// Rank 3 -> 4 impacts: A (cast) -> B -> A -> B. Only two mobs exist, so the
	// chain must land by revisiting each in turn.
	s.applyOpsLocked(c, []gamedata.Op{bounceOp},
		opCtx{slot: 1, level: 3, target: a, px: a.x, py: a.y, hasPos: true}, now)

	hits := 0
	prevA, prevB := a.hp, b.hp
	for tick := 0; tick < 60 && len(hs.bounces) > 0; tick++ {
		s.tickBouncesLocked(c, now+0.05*float64(tick+1))
		if a.hp < prevA {
			hits++
			prevA = a.hp
		}
		if b.hp < prevB {
			hits++
			prevB = b.hp
		}
	}
	if hits != 4 {
		t.Errorf("skull landed %d hits, want 4 (A->B->A->B ping-pong)", hits)
	}
	if a.hp >= 5000 || b.hp >= 5000 {
		t.Errorf("both enemies should be damaged: A=%.0f B=%.0f", a.hp, b.hp)
	}
	if a.hp != b.hp {
		t.Errorf("A and B were each struck twice so should share the same hp: A=%.0f B=%.0f", a.hp, b.hp)
	}
}

// TestElgormArrowsHitAlongLineNotRadius: the beam strikes a mob in front of the
// caster along the aim line, but NOT one off to the side that a radius-around-point
// AoE would have caught.
func TestElgormArrowsHitAlongLineNotRadius(t *testing.T) {
	s, c, _, sx, sy := newNavConnAvatar(t, 31)
	hs := c.huntState
	kit := gamedata.SkillsFor(hs.av)
	innerOps := kit.Skills[3].Ops[0].Ops // the channel's per-pulse ops (the damage)

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	c.x, c.y, c.snapT = sx, sy, float32(0)
	now := float64(s.battleTime())

	skel := gamedata.Mobs()[2]
	// onLine: 4u ahead of the caster, on the aim line -> inside the beam.
	onLine := &mobState{id: 2800, mobIdx: 2, mob: skel, x: sx + 4, y: sy, hp: 500, maxHP: 500, shown: true}
	// offSide: near the aim point but 2.5u to the side -> outside the 1.5u half-width
	// beam, though inside the old 3u radius-around-point.
	offSide := &mobState{id: 2801, mobIdx: 2, mob: skel, x: sx + 8, y: sy + 2.5, hp: 500, maxHP: 500, shown: true}
	hs.mobs[onLine.id] = onLine
	hs.mobs[offSide.id] = offSide
	hs.tr.add(onLine.id)
	hs.tr.add(offSide.id)

	// One channel pulse aimed at the point 8u ahead.
	s.applyOpsLocked(c, innerOps, opCtx{slot: 4, level: 1, hasPos: true, px: sx + 8, py: sy}, now)

	if onLine.hp >= 500 {
		t.Error("mob on the beam line took no damage")
	}
	if offSide.hp < 500 {
		t.Error("mob off to the side was hit -- the beam is still resolving as a radius")
	}
}

// TestElgormArrowsBeamReachesFullRange: «Стрелы Аркана» is a STATIONARY line skill,
// so the beam projects the FULL skill range in the aim direction (like Velial's
// «Разлом» and the client's SkillLineZone), NOT only up to the exact click point. A
// mob standing BEYOND a near click but within range, on the aim line, is still hit.
func TestElgormArrowsBeamReachesFullRange(t *testing.T) {
	s, c, _, sx, sy := newNavConnAvatar(t, 31)
	hs := c.huntState
	kit := gamedata.SkillsFor(hs.av)
	innerOps := kit.Skills[3].Ops[0].Ops // the channel's per-pulse damage
	dist := kit.Skills[3].Distance

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	c.x, c.y, c.snapT = sx, sy, float32(0)
	now := float64(s.battleTime())

	skel := gamedata.Mobs()[2]
	// far: past a near click point but within the skill range, on the aim line.
	far := &mobState{id: 2810, mobIdx: 2, mob: skel, x: sx + float32(dist) - 1, y: sy, hp: 500, maxHP: 500, shown: true}
	hs.mobs[far.id] = far
	hs.tr.add(far.id)

	// Aim at a point only 2u ahead (a near click); the far mob is ~Distance-1 ahead.
	s.applyOpsLocked(c, innerOps, opCtx{slot: 4, level: 1, hasPos: true, px: sx + 2, py: sy}, now)

	if far.hp >= 500 {
		t.Errorf("beam capped at the click point -- a mob at range %d on the aim line took no damage (want full-range beam)", dist-1)
	}
}

// TestElgormArrowsFxSweepsFullRange: the arrow-rain payload fx (a caster->targetPos
// ProjectileBurst) is aimed at the FULL skill range in the click direction, not at
// the exact click point, so the arrows visually cover the whole beam. A dash-cleave
// line skill keeps the click point (its lane is only the travelled path).
func TestElgormArrowsFxSweepsFullRange(t *testing.T) {
	a, ok := gamedata.AvatarByID(31)
	if !ok {
		t.Fatal("Elgorm missing")
	}
	sk := gamedata.SkillsFor(a).Skills[3] // «Стрелы Аркана», a stationary beam
	// Caster at origin, a near click 2u ahead on +x -> endpoint at full range on +x.
	fx, fy := lineFxEndpoint(0, 0, 2, 0, sk)
	if math.Abs(float64(fx)-float64(sk.Distance)) > 1e-3 || math.Abs(float64(fy)) > 1e-3 {
		t.Errorf("arrow fx endpoint = (%.2f,%.2f), want (%d,0) at full range", fx, fy, sk.Distance)
	}

	// A dash-cleave line skill (Shin Dalar slot 1) must NOT be extended -- the fx stays
	// at the click point (the dash lane is only the path travelled).
	shin, ok := gamedata.AvatarByID(16)
	if !ok {
		t.Fatal("Shin Dalar (id 16) missing")
	}
	dash := gamedata.SkillsFor(shin).Skills[0]
	dx, dy := lineFxEndpoint(0, 0, 3, 0, dash)
	if dx != 3 || dy != 0 {
		t.Errorf("dash-cleave fx endpoint = (%.2f,%.2f), want the click point (3,0)", dx, dy)
	}
}

// TestElgormArrowsNineTicks: the arrow-rain channel lands 9 damage ticks over the
// cast, synced to the client's ProjectileBurst (mDelay=0.2, mInterval=0.46 -> 9
// arrows in the 4s window). Regression guard for the tick-drift bug (recomputing
// nextPulse from the coarse 0.2s tick spread the 0.46s cadence to ~0.6s -> ~7 ticks).
func TestElgormArrowsNineTicks(t *testing.T) {
	s, c, _, sx, sy := newNavConnAvatar(t, 31)
	hs := c.huntState
	chOp := gamedata.SkillsFor(hs.av).Skills[3].Ops[0]
	if chOp.Kind != gamedata.OpChannel {
		t.Fatalf("expected Elgorm slot 4 op[0] to be OpChannel, got %q", chOp.Kind)
	}

	c.mvMu.Lock()
	defer c.mvMu.Unlock()
	c.x, c.y, c.snapT = sx, sy, float32(0)
	now := float64(s.battleTime())

	skel := gamedata.Mobs()[2]
	// A tanky mob straight ahead on the beam line so every pulse lands on it.
	victim := &mobState{id: 2820, mobIdx: 2, mob: skel, x: sx + 4, y: sy, hp: 100000, maxHP: 100000, shown: true}
	hs.mobs[victim.id] = victim
	hs.tr.add(victim.id)

	// Create the channel (aimed straight ahead). Caster stays put (no move/stun) so
	// the interruptible channel runs to completion.
	s.applyOpsLocked(c, []gamedata.Op{chOp}, opCtx{slot: 4, level: 1, hasPos: true, px: sx + 8, py: sy}, now)

	// Drive the tick over the whole cast window on the real 0.2s grid and count the
	// pulses that actually dealt damage.
	ticks := 0
	prev := victim.hp
	for i := 0; i <= 22; i++ { // 0 .. 4.4s, a hair past the 4s channel
		s.tickChannelsLocked(c, now+float64(i)*tickInterval.Seconds())
		if victim.hp < prev {
			ticks++
			prev = victim.hp
		}
	}
	if ticks != 9 {
		t.Errorf("arrow rain dealt %d damage ticks over the cast, want 9 (synced to the 9 client arrows)", ticks)
	}
}
