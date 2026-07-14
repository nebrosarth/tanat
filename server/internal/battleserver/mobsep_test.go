package battleserver

import (
	"math"
	"testing"
)

// TestMobSeparation verifies packmates push apart, exact overlaps still part,
// and a lone mob gets no spurious steering.
func TestMobSeparation(t *testing.T) {
	hs := &huntState{mobs: map[int32]*mobState{}}
	// shown: separation only steers mobs the client can see (mobSeparation skips
	// hidden mobs, which are always far past mobSepRange anyway).
	a := &mobState{id: 1, x: 5, y: 5, shown: true}
	b := &mobState{id: 2, x: 5.5, y: 5, shown: true} // 0.5 apart, well inside mobSepRange
	hs.mobs[1], hs.mobs[2] = a, b

	ax, _ := hs.mobSeparation(a)
	bx, _ := hs.mobSeparation(b)
	if ax >= 0 {
		t.Fatalf("mob a should be pushed -x away from b, got %.3f", ax)
	}
	if bx <= 0 {
		t.Fatalf("mob b should be pushed +x away from a, got %.3f", bx)
	}
	// Closer overlap => stronger push.
	b.x = 5.2
	ax2, _ := hs.mobSeparation(a)
	if math.Abs(float64(ax2)) <= math.Abs(float64(ax)) {
		t.Fatalf("closer overlap should push harder: %.3f vs %.3f", ax2, ax)
	}

	// Perfectly overlapping mobs must still separate (deterministic by id).
	b.x, b.y = 5, 5
	sx, sy := hs.mobSeparation(b)
	if sx == 0 && sy == 0 {
		t.Fatal("perfectly overlapping mobs produced zero separation")
	}

	// A mob far from all others gets no push.
	lone := &mobState{id: 9, x: 80, y: 80}
	lx, ly := hs.mobSeparation(lone)
	if lx != 0 || ly != 0 {
		t.Fatalf("lone mob got spurious separation (%.3f,%.3f)", lx, ly)
	}
}
