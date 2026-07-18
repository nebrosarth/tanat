package gamedata

import "testing"

// TestAltarGuardedByBaseCannons pins the «Штурм» altar-guard derivation: each altar is guarded
// by exactly the two BASE cannons flanking it (Human 5 & 6, Elf 20 & 21 on map_1_0), NOT by all
// 11 of the side's guns. This is the data behind the fix for "the altar stays invulnerable after
// I destroy the two cannons next to it": the guard set is the base cannons only, so razing them
// opens the altar even while the distant lane guns still stand.
func TestAltarGuardedByBaseCannons(t *testing.T) {
	m, ok := DotaMapByID(101)
	if !ok {
		t.Fatal("map_1_0 (id 101) not found")
	}
	want := map[int32][]int32{
		16: {5, 6},   // Human altar <- its two base cannons
		33: {20, 21}, // Elf altar   <- its two base cannons
	}
	for altarID, wantGuns := range want {
		got := m.AltarGuardGunIDs(altarID)
		set := map[int32]bool{}
		for _, g := range got {
			set[g] = true
		}
		if len(got) != len(wantGuns) {
			t.Errorf("altar %d guarded by %v, want exactly %v", altarID, got, wantGuns)
		}
		for _, g := range wantGuns {
			if !set[g] {
				t.Errorf("altar %d: base cannon %d missing from guard set %v", altarID, g, got)
			}
		}
	}
	// A non-altar id (a gun) has no guard set.
	if got := m.AltarGuardGunIDs(5); len(got) != 0 {
		t.Errorf("AltarGuardGunIDs(non-altar) = %v, want empty", got)
	}
	// Sanity: the side has far more guns than guard the altar (proves we are not just
	// returning "every gun"). map_1_0 gives each side 11 guns; only 2 guard the altar.
	sideGuns := 0
	for _, sc := range m.Structures {
		if sc.Role == DotaGun && sc.Side == DotaSideElf {
			sideGuns++
		}
	}
	if sideGuns <= len(want[33]) {
		t.Fatalf("expected many more Elf guns (%d) than the 2 that guard the altar", sideGuns)
	}
}
