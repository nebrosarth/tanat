package gamedata

import "testing"

// TestVigilansPassives pins the two Vigilans passives against the client locale:
//   slot 2 «Свидание со смертью» — «Если цель НЕ рядом с союзниками, то КАЖДЫЙ удар
//     наносит увеличенный урон»: a Chance-1 on-hit proc whose damage is isolation-gated
//     (NOT the old 50% coin-flip).
//   slot 3 «Любовные узы» — «При атаке есть {8/12/16/20}% шанс оглушить на 0.7 сек».
func TestVigilansPassives(t *testing.T) {
	vig, ok := AvatarByID(20)
	if !ok {
		t.Fatal("Vigilans (id 20) missing")
	}
	sk := SkillsFor(vig).Skills

	// --- slot 2: isolation-gated per-hit bonus ---
	s2 := sk[1]
	if s2.Type != "PASSIVE" {
		t.Errorf("slot 2 type = %q, want PASSIVE", s2.Type)
	}
	if len(s2.Ops) != 1 || s2.Ops[0].Kind != OpProc {
		t.Fatalf("slot 2 should be a single OpProc, got %+v", s2.Ops)
	}
	proc := s2.Ops[0]
	if got := proc.Chance.At(1); got != 1 {
		t.Errorf("slot 2 proc chance = %v, want 1 (every hit, not a coin-flip)", got)
	}
	if len(proc.Ops) != 1 || proc.Ops[0].Kind != OpDamage {
		t.Fatalf("slot 2 proc should wrap one OpDamage, got %+v", proc.Ops)
	}
	dmg := proc.Ops[0]
	if !dmg.TargetIsolated {
		t.Error("slot 2 bonus damage must be gated TargetIsolated (fires only vs a lone target)")
	}
	if dmg.TriggerRadius <= 0 {
		t.Error("slot 2 isolation gate needs a positive TriggerRadius")
	}
	if dmg.Value.At(1) <= 0 {
		t.Error("slot 2 bonus damage should be positive")
	}

	// --- slot 3: per-hit stun chance ---
	s3 := sk[2]
	if s3.Type != "PASSIVE" {
		t.Errorf("slot 3 type = %q, want PASSIVE", s3.Type)
	}
	if len(s3.Ops) != 1 || s3.Ops[0].Kind != OpProc {
		t.Fatalf("slot 3 should be a single OpProc, got %+v", s3.Ops)
	}
	sp := s3.Ops[0]
	wantCh := []float64{0.08, 0.12, 0.16, 0.2}
	for i, w := range wantCh {
		if got := sp.Chance.At(i + 1); got != w {
			t.Errorf("slot 3 stun chance rank %d = %v, want %v", i+1, got, w)
		}
	}
	if len(sp.Ops) != 1 || sp.Ops[0].Kind != OpStun {
		t.Fatalf("slot 3 proc should wrap one OpStun, got %+v", sp.Ops)
	}
	if got := sp.Ops[0].Dur.At(1); got != 0.7 {
		t.Errorf("slot 3 stun duration = %v, want 0.7", got)
	}
}
