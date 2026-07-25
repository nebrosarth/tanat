package gamedata

import "testing"

// TestUrgSproutIsVisionWard pins «Росток» (Urg slot 2) as a PURE vision utility: it must
// be an OpVisionWard that plants the visible scout-tree prop and carries NO offensive op.
// The old encoding was an OpTrap with an inner root+DoT that contradicted the benign wiki
// description («открывает небольшой участок местности… видеть невидимых врагов»).
func TestUrgSproutIsVisionWard(t *testing.T) {
	urg, ok := AvatarByID(12) // Urg
	if !ok {
		t.Fatal("Urg (id 12) missing")
	}
	sk := SkillsFor(urg).Skills[1] // slot 2
	if sk.NameRu != "Росток" {
		t.Fatalf("slot 2 name = %q, want Росток", sk.NameRu)
	}
	if len(sk.Ops) != 1 {
		t.Fatalf("slot 2 has %d ops, want exactly 1 (the ward)", len(sk.Ops))
	}
	w := sk.Ops[0]
	if w.Kind != OpVisionWard {
		t.Errorf("slot 2 op kind = %q, want vision_ward", w.Kind)
	}
	if w.Unit != "Avtr_Tank_Urg_Totem_prop01" {
		t.Errorf("ward prop = %q, want the totem scout prefab (distinct from slot 1's disguise tree)", w.Unit)
	}
	if w.Lifetime.At(1) < 900 {
		t.Errorf("ward lifetime = %v, want 15 minutes (900s per client)", w.Lifetime.At(1))
	}
	if w.Radius <= 0 {
		t.Errorf("ward radius = %v, want a positive vision radius", w.Radius)
	}
	if w.Lifetime.At(1) <= 0 {
		t.Errorf("ward lifetime = %v, want a positive duration", w.Lifetime.At(1))
	}
	// A scouting ward deals no damage and applies no CC.
	for _, bad := range []OpKind{OpDamage, OpDot, OpRoot, OpStun, OpSlow, OpSilence} {
		if w.Kind == bad {
			t.Errorf("ward must not carry offensive op %q", bad)
		}
	}
	for _, inner := range w.Ops {
		t.Errorf("ward must have no nested ops, found %q", inner.Kind)
	}
}
