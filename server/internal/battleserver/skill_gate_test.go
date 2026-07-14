package battleserver

import "testing"

// TestSkillGateAttribs pins the per-slot client `levels` gating arrays, start
// ranks, and the server-side level requirement, matching the real game: EVERY
// skill starts UNLEARNED at rank 0 (no free ranks). Skills 1-3 = 5 ranks, rank 1
// buyable from level 1; the ult (slot 4) = 4 ranks unlocking at avatar levels
// 5/10/15/20 (0-based 4/9/14/19).
func TestSkillGateAttribs(t *testing.T) {
	// Slots 1-3: no free rank -> leading 0 (buyable from the start), then levels 1..4.
	for _, slot := range []int{1, 2, 3} {
		if got := skillLevelsAttr(slot, 5); got != "0;1;2;3;4" {
			t.Errorf("slot %d levels = %q, want 0;1;2;3;4", slot, got)
		}
		if skillStartRank(slot) != 0 {
			t.Errorf("slot %d start rank = %d, want 0 (nothing free)", slot, skillStartRank(slot))
		}
	}
	// Ult (slot 4): gated 5/10/15/20 (0-based 4/9/14/19), starts unlearned at 0.
	if got := skillLevelsAttr(4, 4); got != "4;9;14;19" {
		t.Errorf("ult levels = %q, want 4;9;14;19", got)
	}
	if skillStartRank(4) != 0 {
		t.Errorf("ult start rank = %d, want 0", skillStartRank(4))
	}

	// Server-side gate (0-based avatar level required for the NEXT rank).
	if skillReqLevel(1, 0) != 0 { // slots 1-3 rank 1 buyable from level 0 (needs a point)
		t.Errorf("slot 1 rank-1 req = %d, want 0 (buyable from start, not free)", skillReqLevel(1, 0))
	}
	if skillReqLevel(1, 1) != 1 || skillReqLevel(1, 4) != 4 {
		t.Error("slot 1-3 rank r needs 0-based level r")
	}
	// Ult: rank 1 needs level 4 (=char level 5), rank 4 needs level 19 (=char 20).
	if skillReqLevel(4, 0) != 4 || skillReqLevel(4, 3) != 19 {
		t.Errorf("ult gate wrong: rank1 req %d (want 4), rank4 req %d (want 19)",
			skillReqLevel(4, 0), skillReqLevel(4, 3))
	}
}
