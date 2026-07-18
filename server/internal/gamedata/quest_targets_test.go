package gamedata

import "testing"

// spawnableOnMap returns the set of mob roster indices actually placed on a Hunt map, read
// from the generated spawn packs (the ground truth). Used to prove no quest targets a creature
// that never appears where the quest is fought.
func spawnableOnMap(mapID int32) map[int]bool {
	var pack []MobSpawn
	switch mapID {
	case 40:
		pack = dungeonPack40
	case 41:
		pack = invasionPack41
	case 42:
		pack = junglePack
	}
	set := map[int]bool{}
	for _, sp := range pack {
		set[sp.Mob] = true
	}
	return set
}

// TestEveryQuestHasTargeting: every baked quest must have an explicit kill-targeting entry, so
// the AnyMob fallback in init() never silently fires and every quest is intentionally scoped.
func TestEveryQuestHasTargeting(t *testing.T) {
	for _, q := range Quests() {
		if _, ok := questKillTargets[q.Key]; !ok {
			t.Errorf("quest %q has no entry in questKillTargets (would fall back to AnyMob)", q.Key)
		}
	}
	// And no stale entries for quests that no longer exist.
	known := map[string]bool{}
	for _, q := range Quests() {
		known[q.Key] = true
	}
	for key := range questKillTargets {
		if !known[key] {
			t.Errorf("questKillTargets has a stale entry %q (no such quest)", key)
		}
	}
}

// TestQuestTargetingWellFormed: AnyMob and Targets are mutually exclusive, a non-AnyMob quest
// names at least one target, and every target is a real roster index.
func TestQuestTargetingWellFormed(t *testing.T) {
	n := len(Mobs())
	for _, q := range Quests() {
		if q.AnyMob {
			if len(q.Targets) != 0 {
				t.Errorf("%s: AnyMob quest also lists Targets %v", q.Key, q.Targets)
			}
			continue
		}
		if len(q.Targets) == 0 {
			t.Errorf("%s: not AnyMob yet has no Targets (uncreditable)", q.Key)
		}
		for _, idx := range q.Targets {
			if idx < 0 || idx >= n {
				t.Errorf("%s: target index %d out of roster range [0,%d)", q.Key, idx, n)
			}
		}
	}
}

// TestQuestTargetsAreReachable: for every creature-targeted quest, at least one of its targets
// actually spawns on the quest's own map -- otherwise the quest could never be completed. This
// is the guarantee that gating kills by creature did not create an impossible quest.
func TestQuestTargetsAreReachable(t *testing.T) {
	for _, q := range Quests() {
		if q.AnyMob {
			continue
		}
		spawns := spawnableOnMap(q.MapID)
		reachable := false
		for _, idx := range q.Targets {
			if spawns[idx] {
				reachable = true
				break
			}
		}
		if !reachable {
			t.Errorf("%s (map %d): none of its targets %v spawn on that map -- impossible quest",
				q.Key, q.MapID, q.Targets)
		}
	}
}

// TestGhoulQuestRejectsOtherMobs is the reported bug, pinned: the starter «kill 10 ghouls» quest
// credits a ghoul kill but NOT a skeleton (or any other) kill.
func TestGhoulQuestRejectsOtherMobs(t *testing.T) {
	var ghoulQuest Quest
	found := false
	for _, q := range Quests() {
		if q.Key == "NPC1_PVE_Single_Stage1_3" {
			ghoulQuest, found = q, true
			break
		}
	}
	if !found {
		t.Fatal("the starter ghoul quest (NPC1_PVE_Single_Stage1_3) is missing from the catalog")
	}
	if ghoulQuest.AnyMob {
		t.Fatal("the ghoul quest must NOT be AnyMob -- that is the bug")
	}
	if !QuestCreditsKill(ghoulQuest, mobGhoul) {
		t.Error("killing a ghoul must credit the ghoul quest")
	}
	for _, other := range []int{mobSkeleton, mobSkeletonArcher, mobZombie, mobDemon, mobGhoulPossessed} {
		if QuestCreditsKill(ghoulQuest, other) {
			t.Errorf("killing mob %d must NOT credit the «kill 10 ghouls» quest", other)
		}
	}
}

// TestAnyMobQuestCreditsEverything: an «убить N любых существ» quest is credited by any creature.
func TestAnyMobQuestCreditsEverything(t *testing.T) {
	var anyQuest Quest
	for _, q := range Quests() {
		if q.Key == "NPC1_PVE_Single_Stage6_3" { // «Натиск»: 200 любых существ
			anyQuest = q
			break
		}
	}
	if !anyQuest.AnyMob {
		t.Fatal("the «200 любых существ» quest should be AnyMob")
	}
	for _, idx := range []int{mobGhoul, mobSkeleton, mobDemon, mobBossHekata} {
		if !QuestCreditsKill(anyQuest, idx) {
			t.Errorf("an AnyMob quest must credit mob %d", idx)
		}
	}
}
