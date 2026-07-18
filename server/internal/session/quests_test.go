package session

import (
	"testing"

	"tanatserver/internal/gamedata"
)

func questHero(t *testing.T) (*Store, *User) {
	t.Helper()
	s := NewStore()
	u, _ := s.LoginOrRegister("quester@test.io", "pw")
	s.CreateHero(u, 1, false, 0, 0, 0, 0, 0)
	return s, u
}

// findQuest returns the first catalog quest matching pred, failing the test if none.
func findQuest(t *testing.T, pred func(gamedata.Quest) bool) gamedata.Quest {
	t.Helper()
	for _, q := range gamedata.Quests() {
		if pred(q) {
			return q
		}
	}
	t.Fatal("no matching quest in catalog")
	return gamedata.Quest{}
}

// killIdx returns a mob roster index that credits quest q (its first authored target, or any
// mob for an AnyMob quest), so the lifecycle tests below can drive q to completion.
func killIdx(q gamedata.Quest) int {
	if q.AnyMob || len(q.Targets) == 0 {
		return 0
	}
	return q.Targets[0]
}

// TestQuestAcceptProgressComplete walks the whole one-time lifecycle: accept -> kills advance
// progress -> DONE at the objective count -> CompleteQuest transitions to CLOSED exactly once.
func TestQuestAcceptProgressComplete(t *testing.T) {
	s, u := questHero(t)
	q := findQuest(t, func(q gamedata.Quest) bool { return !q.Repeatable() && q.Count >= 3 })

	if _, ok := s.AcceptQuest(u.ID, q.ID); !ok {
		t.Fatal("AcceptQuest failed on a fresh quest")
	}
	// A kill on a DIFFERENT map must not advance it.
	otherMap := int32(41)
	if q.MapID == 41 {
		otherMap = 42
	}
	if got := s.AddQuestKill(u.ID, otherMap, killIdx(q)); len(got) != 0 {
		t.Errorf("kill on map %d advanced a map-%d quest", otherMap, q.MapID)
	}
	// Kills on the right map advance one step each until DONE.
	for i := int32(1); i < q.Count; i++ {
		changed := s.AddQuestKill(u.ID, q.MapID, killIdx(q))
		if len(changed) != 1 || changed[0].Progress != i {
			t.Fatalf("kill %d: progress = %+v", i, changed)
		}
		if changed[0].Status != gamedata.QuestStatusInProgress {
			t.Fatalf("kill %d flipped status early: %d", i, changed[0].Status)
		}
	}
	final := s.AddQuestKill(u.ID, q.MapID, killIdx(q))
	if len(final) != 1 || final[0].Status != gamedata.QuestStatusDone {
		t.Fatalf("objective count not DONE: %+v", final)
	}
	// Further kills do not overshoot a DONE quest.
	if got := s.AddQuestKill(u.ID, q.MapID, killIdx(q)); len(got) != 0 {
		t.Errorf("kill advanced an already-DONE quest: %+v", got)
	}
	// CompleteQuest fires once; a second call is rejected (no double reward).
	if _, ok := s.CompleteQuest(u.ID, q.ID); !ok {
		t.Fatal("CompleteQuest rejected a DONE quest")
	}
	if _, ok := s.CompleteQuest(u.ID, q.ID); ok {
		t.Error("CompleteQuest fired twice (double reward)")
	}
	// One-time quest is CLOSED and cannot be re-accepted.
	if _, ok := s.AcceptQuest(u.ID, q.ID); ok {
		t.Error("re-accepted a CLOSED one-time quest")
	}
}

// TestQuestCannotDoneBeforeObjective: CompleteQuest is refused while merely IN_PROGRESS.
func TestQuestCannotDoneBeforeObjective(t *testing.T) {
	s, u := questHero(t)
	q := findQuest(t, func(q gamedata.Quest) bool { return q.Count >= 2 })
	s.AcceptQuest(u.ID, q.ID)
	s.AddQuestKill(u.ID, q.MapID, killIdx(q)) // 1 of >=2 -> still IN_PROGRESS
	if _, ok := s.CompleteQuest(u.ID, q.ID); ok {
		t.Error("turned in a quest before its objective was met")
	}
}

// TestQuestCancel abandons a quest so it can be taken again.
func TestQuestCancel(t *testing.T) {
	s, u := questHero(t)
	q := findQuest(t, func(q gamedata.Quest) bool { return true })
	s.AcceptQuest(u.ID, q.ID)
	if !s.CancelQuest(u.ID, q.ID) {
		t.Fatal("CancelQuest failed on an accepted quest")
	}
	if s.CancelQuest(u.ID, q.ID) {
		t.Error("CancelQuest succeeded twice")
	}
	if len(s.HeroQuests(u.ID)) != 0 {
		t.Error("cancelled quest still present")
	}
	if _, ok := s.AcceptQuest(u.ID, q.ID); !ok {
		t.Error("cannot re-accept a cancelled quest")
	}
}

// TestQuestCancelCannotReArmClosedReward is the dupe-exploit regression: a CLOSED one-time quest
// must not be cancellable, because deleting its CLOSED record would let AcceptQuest re-arm it and
// re-farm the reward. Verifies cancel is refused, the CLOSED state survives, and re-accept still
// fails.
func TestQuestCancelCannotReArmClosedReward(t *testing.T) {
	s, u := questHero(t)
	q := findQuest(t, func(q gamedata.Quest) bool { return !q.Repeatable() })
	s.AcceptQuest(u.ID, q.ID)
	for i := int32(0); i < q.Count; i++ {
		s.AddQuestKill(u.ID, q.MapID, killIdx(q))
	}
	if _, ok := s.CompleteQuest(u.ID, q.ID); !ok {
		t.Fatal("could not complete the one-time quest")
	}
	moneyAfterFirst := u.Hero.Money
	// The exploit attempt: cancel the CLOSED quest, then try to re-accept and re-farm.
	if s.CancelQuest(u.ID, q.ID) {
		t.Fatal("EXPLOIT: cancelled a CLOSED one-time quest")
	}
	st := s.HeroQuests(u.ID)
	if len(st) != 1 || st[0].Status != gamedata.QuestStatusClosed {
		t.Fatalf("CLOSED record was disturbed: %+v", st)
	}
	if _, ok := s.AcceptQuest(u.ID, q.ID); ok {
		t.Fatal("EXPLOIT: re-accepted a CLOSED quest after cancel")
	}
	if u.Hero.Money != moneyAfterFirst {
		t.Errorf("reward re-farmed: money moved from %d to %d", moneyAfterFirst, u.Hero.Money)
	}
}

// TestQuestCancelCannotBypassReplayCooldown is the cooldown-bypass regression: a REPLAY quest
// still cooling must not be cancellable (else the next accept starts fresh with no cooldown).
// Once the cooldown elapses the record is re-offerable, so cancelling it then is harmless.
func TestQuestCancelCannotBypassReplayCooldown(t *testing.T) {
	s, u := questHero(t)
	q := findQuest(t, func(q gamedata.Quest) bool { return q.Repeatable() })
	fake := int64(5000)
	old := nowUnix
	nowUnix = func() int64 { return fake }
	defer func() { nowUnix = old }()

	s.AcceptQuest(u.ID, q.ID)
	for i := int32(0); i < q.Count; i++ {
		s.AddQuestKill(u.ID, q.MapID, killIdx(q))
	}
	s.CompleteQuest(u.ID, q.ID) // -> WAIT_COOLDOWN
	if s.CancelQuest(u.ID, q.ID) {
		t.Fatal("EXPLOIT: cancelled a REPLAY quest still on cooldown")
	}
	if _, ok := s.AcceptQuest(u.ID, q.ID); ok {
		t.Fatal("EXPLOIT: re-accepted a cooling REPLAY quest")
	}
	// An IN_PROGRESS quest, by contrast, is still freely abandonable.
	other := findQuest(t, func(c gamedata.Quest) bool { return c.ID != q.ID })
	s.AcceptQuest(u.ID, other.ID)
	if !s.CancelQuest(u.ID, other.ID) {
		t.Error("could not cancel an active IN_PROGRESS quest")
	}
}

// TestQuestReplayCooldown: a REPLAY quest parks on cooldown after turn-in, is filtered out of
// HeroQuests once the timer elapses, and can then be taken again.
func TestQuestReplayCooldown(t *testing.T) {
	s, u := questHero(t)
	q := findQuest(t, func(q gamedata.Quest) bool { return q.Repeatable() })

	fake := int64(1000)
	old := nowUnix
	nowUnix = func() int64 { return fake }
	defer func() { nowUnix = old }()

	s.AcceptQuest(u.ID, q.ID)
	for i := int32(0); i < q.Count; i++ {
		s.AddQuestKill(u.ID, q.MapID, killIdx(q))
	}
	if _, ok := s.CompleteQuest(u.ID, q.ID); !ok {
		t.Fatal("could not complete replay quest")
	}
	// While cooling: present as WAIT_COOLDOWN, and NOT re-acceptable.
	states := s.HeroQuests(u.ID)
	if len(states) != 1 || states[0].Status != gamedata.QuestStatusWaitCooldown {
		t.Fatalf("post-turn-in state = %+v, want WAIT_COOLDOWN", states)
	}
	if _, ok := s.AcceptQuest(u.ID, q.ID); ok {
		t.Error("re-accepted a replay quest still on cooldown")
	}
	// After the cooldown elapses: filtered from HeroQuests and re-acceptable.
	fake += int64(q.Cooldown) + 1
	if len(s.HeroQuests(u.ID)) != 0 {
		t.Error("expired-cooldown replay quest still reported")
	}
	if _, ok := s.AcceptQuest(u.ID, q.ID); !ok {
		t.Error("could not re-accept a replay quest after cooldown")
	}
	if st := s.HeroQuests(u.ID); len(st) != 1 || st[0].Status != gamedata.QuestStatusInProgress || st[0].Progress != 0 {
		t.Errorf("re-accepted replay quest state = %+v, want fresh IN_PROGRESS", st)
	}
}

// TestAddQuestKillGatesByCreature: a kill only advances a quest when the slain creature is one
// of the quest's targets -- the fix for «kill 10 ghouls» counting any mob.
func TestAddQuestKillGatesByCreature(t *testing.T) {
	s, u := questHero(t)
	q := findQuest(t, func(q gamedata.Quest) bool { return q.Key == "NPC1_PVE_Single_Stage1_3" }) // «10 ghouls»
	if q.AnyMob || len(q.Targets) == 0 {
		t.Fatal("the ghoul quest must target a specific creature")
	}
	if _, ok := s.AcceptQuest(u.ID, q.ID); !ok {
		t.Fatal("AcceptQuest failed")
	}
	right := q.Targets[0]
	wrong := right + 1 // a different roster index, not a ghoul
	// A wrong-creature kill (right map, wrong mob) must NOT advance it.
	if got := s.AddQuestKill(u.ID, q.MapID, wrong); len(got) != 0 {
		t.Fatalf("a non-target kill advanced the ghoul quest: %+v", got)
	}
	// The correct creature advances it.
	if got := s.AddQuestKill(u.ID, q.MapID, right); len(got) != 1 || got[0].Progress != 1 {
		t.Fatalf("a ghoul kill did not advance the quest: %+v", got)
	}
}

// TestQuestUnknownAndNoHero: guards fail safe.
func TestQuestUnknownAndNoHero(t *testing.T) {
	s, u := questHero(t)
	if _, ok := s.AcceptQuest(u.ID, 123456); ok {
		t.Error("accepted a non-existent quest")
	}
	if got := s.AddQuestKill(9999, 40, 0); got != nil {
		t.Error("AddQuestKill for unknown user returned progress")
	}
	if s.HeroQuests(9999) != nil {
		t.Error("HeroQuests for unknown user should be nil")
	}
}
