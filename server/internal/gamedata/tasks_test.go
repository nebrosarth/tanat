package gamedata

import "testing"

// TestPvpTaskCount pins the authored «Штурм» task set: 9 Group + 4 GroupVS + 4 VS mirrors +
// 3 Single + 1 Event = 21.
func TestPvpTaskCount(t *testing.T) {
	if got := len(PvpTasks()); got != 21 {
		t.Fatalf("PvpTasks() = %d, want 21", got)
	}
}

// TestPvpTaskIDsUniqueAndDisjoint is the no-collision invariant: task ids are unique, above the
// task base, and share no value with a PvE quest id OR any item article id. A collision would let
// a QUEST_TASK resolve to the wrong catalog entry (the client merges quests.amf + tasks.amf into
// one store), or a task id shadow an item.
func TestPvpTaskIDsUniqueAndDisjoint(t *testing.T) {
	used := map[int32]string{}
	for _, q := range Quests() {
		used[q.ID] = "quest " + q.Key
	}
	for _, it := range Items() {
		used[it.ArticleID] = "potion"
	}
	for _, it := range AvatarItems() {
		used[it.ArticleID] = "avatar-item"
	}
	for _, w := range Wearables() {
		used[w.ArticleID] = "wearable"
	}
	seen := map[int32]bool{}
	for _, tk := range PvpTasks() {
		if tk.ID <= pvpTaskIDBase {
			t.Errorf("task %s id %d not above base %d", tk.Key, tk.ID, pvpTaskIDBase)
		}
		if seen[tk.ID] {
			t.Errorf("duplicate task id %d (%s)", tk.ID, tk.Key)
		}
		seen[tk.ID] = true
		if owner, clash := used[tk.ID]; clash {
			t.Errorf("task %s id %d collides with %s", tk.Key, tk.ID, owner)
		}
	}
}

// TestPvpTaskLocaleKeysResolve is the "no EMPTY!" guard: every cited locale key must exist in the
// baked client locale, or the client renders the literal "EMPTY!" for that field. This is the
// reason battle tasks cite _GuiProgress for the in-progress line (they have no _DialogProgress).
func TestPvpTaskLocaleKeysResolve(t *testing.T) {
	keys := validLocaleKeys(t)
	check := func(tKey, field, k string) {
		if k == "" {
			t.Errorf("task %s: empty %s key", tKey, field)
			return
		}
		if !keys[k] {
			t.Errorf("task %s: %s key %q not in the baked locale -> would render EMPTY!", tKey, field, k)
		}
	}
	for _, tk := range PvpTasks() {
		check(tk.Key, "Name", tk.NameKey)
		check(tk.Key, "Task", tk.TaskKey)
		check(tk.Key, "Start", tk.StartKey)
		check(tk.Key, "Progress", tk.ProgKey)
		check(tk.Key, "Win", tk.WinKey)
		check(tk.Key, "Lose", tk.LoseKey)
		check(tk.Key, "Gui", tk.GuiKey)
	}
}

// TestPvpTaskVSPairing: every GroupVS attacker points at its defender mirror and vice-versa
// (partner = id +/-10), reciprocally, sharing the objective count and time limit. Non-VS tasks
// carry no partner.
func TestPvpTaskVSPairing(t *testing.T) {
	for _, tk := range PvpTasks() {
		isVS := tk.Group == "GroupVS" || tk.Group == "GroupVS_VS"
		if !isVS {
			if tk.VSPartner != 0 {
				t.Errorf("non-VS task %s has partner %d", tk.Key, tk.VSPartner)
			}
			continue
		}
		partner, ok := PvpTaskByID(tk.VSPartner)
		if !ok {
			t.Fatalf("task %s partner %d does not exist", tk.Key, tk.VSPartner)
		}
		if partner.VSPartner != tk.ID {
			t.Errorf("task %s <-> %s pairing is not reciprocal", tk.Key, partner.Key)
		}
		if partner.Count != tk.Count || partner.TimeLimit != tk.TimeLimit {
			t.Errorf("task %s and mirror %s disagree on count/time (%d/%d vs %d/%d)",
				tk.Key, partner.Key, tk.Count, tk.TimeLimit, partner.Count, partner.TimeLimit)
		}
	}
}

// TestPvpTaskActiveSubset: the solo-v1 assigned set is exactly the enemy-structure tasks, each
// with a real objective (cannon/barracks), a valid lane, a positive count and a reward.
func TestPvpTaskActiveSubset(t *testing.T) {
	active := ActivePvpTasks()
	if len(active) == 0 {
		t.Fatal("no active PvP tasks -- solo «Штурм» would assign nothing")
	}
	nActiveFlag := 0
	for _, tk := range PvpTasks() {
		if tk.Active {
			nActiveFlag++
		}
	}
	if nActiveFlag != len(active) {
		t.Errorf("ActivePvpTasks() = %d but %d tasks are flagged Active", len(active), nActiveFlag)
	}
	for _, tk := range active {
		if tk.Objective != PvpObjCannon && tk.Objective != PvpObjBarracks {
			t.Errorf("active task %s objective %d is not a solo-completable structure kill", tk.Key, tk.Objective)
		}
		if tk.Lane < pvpLaneAny || tk.Lane > 2 {
			t.Errorf("active task %s bad lane %d", tk.Key, tk.Lane)
		}
		if tk.Count < 1 {
			t.Errorf("active task %s count %d < 1", tk.Key, tk.Count)
		}
		if tk.Money <= 0 || tk.Exp <= 0 {
			t.Errorf("active task %s reward money=%d exp=%d, both must be > 0", tk.Key, tk.Money, tk.Exp)
		}
	}
}

// TestPvpTaskRewardsAndKind: every task carries a positive reward and a valid QuestType.
func TestPvpTaskRewardsAndKind(t *testing.T) {
	for _, tk := range PvpTasks() {
		if tk.Money <= 0 || tk.Exp <= 0 {
			t.Errorf("task %s reward money=%d exp=%d, both must be > 0", tk.Key, tk.Money, tk.Exp)
		}
		if tk.Kind != QuestKindKill && tk.Kind != QuestKindCollect {
			t.Errorf("task %s bad kind %d", tk.Key, tk.Kind)
		}
		if tk.Count < 1 {
			t.Errorf("task %s count %d < 1", tk.Key, tk.Count)
		}
	}
}

// TestPvpTaskStructByID: a structure task's lane resolves through the map geometry, so the Battle
// server can classify a destroyed structure. Spot-check the map's own structures round-trip.
func TestPvpTaskStructByID(t *testing.T) {
	m, ok := DotaMapByID(101)
	if !ok {
		t.Fatal("map_1_0 (id 101) not found")
	}
	for _, sc := range m.Structures {
		got, ok := m.StructByID(sc.ID)
		if !ok || got.ID != sc.ID || got.Role != sc.Role {
			t.Errorf("StructByID(%d) = %+v, %v", sc.ID, got, ok)
		}
	}
	if _, ok := m.StructByID(999999); ok {
		t.Error("StructByID(unknown) should be false")
	}
}
