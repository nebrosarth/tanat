package battleserver

import (
	"math"
	"testing"

	"tanatserver/internal/gamedata"
)

func sepOf(hs *huntState, members map[int32]*conn, m *mobState) (float32, float32) {
	return hs.bodySeparation(members, 0, m.id, m.x, m.y, m.mob.Radius())
}

// TestBodySeparation verifies packmates push apart, exact overlaps still part,
// and a lone mob gets no spurious steering.
func TestBodySeparation(t *testing.T) {
	hs := &huntState{mobs: map[int32]*mobState{}}
	// active: separation only steers SIMULATED mobs (bodySeparation skips inactive
	// ones, which are always far past its range anyway, and whose coordinates are
	// stale). Note active, not shown -- a «Штурм» creep is never culled but is always
	// simulated, and gating on the render flag would silently disable separation for
	// that whole mode.
	a := &mobState{id: 1, x: 5, y: 5, shown: true, active: true}
	b := &mobState{id: 2, x: 5.5, y: 5, shown: true, active: true} // 0.5 apart, inside range
	hs.mobs[1], hs.mobs[2] = a, b

	ax, _ := sepOf(hs, nil, a)
	bx, _ := sepOf(hs, nil, b)
	if ax >= 0 {
		t.Fatalf("mob a should be pushed -x away from b, got %.3f", ax)
	}
	if bx <= 0 {
		t.Fatalf("mob b should be pushed +x away from a, got %.3f", bx)
	}
	// Closer overlap => stronger push.
	b.x = 5.2
	ax2, _ := sepOf(hs, nil, a)
	if math.Abs(float64(ax2)) <= math.Abs(float64(ax)) {
		t.Fatalf("closer overlap should push harder: %.3f vs %.3f", ax2, ax)
	}

	// Perfectly overlapping mobs must still separate (deterministic by id).
	b.x, b.y = 5, 5
	sx, sy := sepOf(hs, nil, b)
	if sx == 0 && sy == 0 {
		t.Fatal("perfectly overlapping mobs produced zero separation")
	}

	// A mob far from all others gets no push.
	lone := &mobState{id: 9, x: 80, y: 80}
	lx, ly := sepOf(hs, nil, lone)
	if lx != 0 || ly != 0 {
		t.Fatalf("lone mob got spurious separation (%.3f,%.3f)", lx, ly)
	}
}

// TestBodySeparationScalesWithRadius is the «Штурм» half of the rule. Hunt's bodies are
// all roughly one size, so a flat range served it; «Штурм» puts a 0.6m creep next to a
// 3.0m altar, and a flat 1.8m would let the creep stand a metre INSIDE the altar it is
// hitting. The range must come from the pair, and it must never shrink below the number
// Hunt was tuned on.
func TestBodySeparationScalesWithRadius(t *testing.T) {
	big := gamedata.Mob{CollisionRadius: 3.0, Stationary: true} // altar-sized
	hs := &huntState{mobs: map[int32]*mobState{}}
	altar := &mobState{id: 1, mob: big, x: 0, y: 0, active: true}
	hs.mobs[1] = altar

	// 2.5m from the altar's centre: outside a flat 1.8m, but well inside its body.
	creep := &mobState{id: 2, mob: gamedata.Mob{CollisionRadius: 0.6}, x: 2.5, y: 0, active: true}
	hs.mobs[2] = creep
	cx, _ := sepOf(hs, nil, creep)
	if cx <= 0 {
		t.Fatalf("a creep standing inside the altar's 3.0m body got push %.3f: the range is "+
			"still flat, not the sum of the two radii", cx)
	}

	// ...but the push must not outrun the attack reach, or creeps could never break it.
	// dotaReach adds both radii to the 2.2m melee reach; separation adds only sepMargin.
	rng := math.Max(mobSepRange, big.Radius()+creep.mob.Radius()+sepMargin)
	if reach := dotaMeleeReach + big.Radius() + creep.mob.Radius(); rng >= reach {
		t.Fatalf("separation range %.2f >= attack reach %.2f: a creep would be pushed out of "+
			"its own reach and could never destroy the altar", rng, reach)
	}

	// The floor: two small bodies keep Hunt's tuned spacing rather than tightening to
	// their radii (0.6+0.6+0.35 = 1.55 < 1.8).
	small := &mobState{id: 3, mob: gamedata.Mob{CollisionRadius: 0.6}, x: 100, y: 0, active: true}
	other := &mobState{id: 4, mob: gamedata.Mob{CollisionRadius: 0.6}, x: 101.7, y: 0, active: true}
	hs2 := &huntState{mobs: map[int32]*mobState{3: small, 4: other}}
	if sx, _ := sepOf(hs2, nil, small); sx == 0 {
		t.Errorf("two 0.6m mobs 1.7m apart got no push: the mobSepRange floor is gone, so Hunt's "+
			"packs now bunch tighter than they were tuned for (got %.3f)", sx)
	}
}

// TestBodySeparationPushIsBounded: the push must be capped at unit length, because
// mobSepWeight scales it against a march term of exactly 1. Every crowding neighbour adds
// up to 1 to the raw sum, so a body boxed in on one side produced a push several times the
// march and oscillated at full speed. Bounded, it can only ever turn the heading.
func TestBodySeparationPushIsBounded(t *testing.T) {
	hs := &huntState{mobs: map[int32]*mobState{}}
	center := &mobState{id: 1, mob: gamedata.Mob{CollisionRadius: 0.6}, x: 0, y: 0, active: true}
	hs.mobs[1] = center
	// Eight bodies packed tight, all on the +x side, so their pushes reinforce rather than
	// cancel and the raw sum is well over 1.
	for i := 0; i < 8; i++ {
		b := &mobState{id: int32(100 + i), mob: gamedata.Mob{CollisionRadius: 0.6},
			x: 0.3, y: float32(i-4) * 0.05, active: true}
		hs.mobs[b.id] = b
	}
	sx, sy := sepOf(hs, nil, center)
	if mag := math.Hypot(float64(sx), float64(sy)); mag > 1.0001 {
		t.Errorf("push magnitude %.3f exceeds 1: unclamped it outvotes the unit march term and the "+
			"creep oscillates at full speed instead of settling beside its neighbours", mag)
	}
}

// TestBodySeparationSeesSummons pins the bug the user actually reported for the dino:
// summons live in their OWNER's map, never in inst.mobs, so a scan of hs.mobs alone
// cannot see them. Nothing pushed the dinosaur and the dinosaur pushed nothing.
func TestBodySeparationSeesSummons(t *testing.T) {
	const now = 100.0
	owner := &conn{huntState: &huntState{summons: map[int32]*summonState{}}}
	dino := &summonState{id: 7001, x: 10, y: 10, hp: 100, until: now + 60}
	owner.huntState.summons[dino.id] = dino
	members := map[int32]*conn{1: owner}

	hs := &huntState{mobs: map[int32]*mobState{}}
	creep := &mobState{id: 5, mob: gamedata.Mob{CollisionRadius: 0.6}, x: 10.5, y: 10, active: true}
	hs.mobs[5] = creep

	// The mob must be pushed off the summon...
	if cx, _ := hs.bodySeparation(members, now, creep.id, creep.x, creep.y, creep.mob.Radius()); cx <= 0 {
		t.Errorf("a creep standing on the dinosaur got push %.3f: mobs still cannot see summons", cx)
	}
	// ...and the summon off the mob. Both directions, or one body just absorbs the other.
	if dx, _ := hs.bodySeparation(members, now, dino.id, dino.x, dino.y, summonRadius); dx >= 0 {
		t.Errorf("the dinosaur standing on a creep got push %.3f: summons still have no collision", dx)
	}
	// A summon must not push itself.
	solo := &huntState{mobs: map[int32]*mobState{}}
	if sx, sy := solo.bodySeparation(members, now, dino.id, dino.x, dino.y, summonRadius); sx != 0 || sy != 0 {
		t.Errorf("the dinosaur pushed itself (%.3f,%.3f)", sx, sy)
	}
	// An expired summon is not a body: liveness matches tickSummonsLocked's own test,
	// not the lazily-set dead flag (its tick has not run yet).
	dino.until = now - 1
	if sx, sy := hs.bodySeparation(members, now, creep.id, creep.x, creep.y, creep.mob.Radius()); sx != 0 || sy != 0 {
		t.Errorf("an expired summon still pushes (%.3f,%.3f): it is about to vanish", sx, sy)
	}
}
