package gamedata

import "testing"

// TestQuestCount pins the full baked PvE questline (map_4_0 crypt + map_4_1 invasion +
// map_4_2 jungle). If the locale extraction changes, regenerate quests_gen.go.
func TestQuestCount(t *testing.T) {
	if got := len(Quests()); got != 103 {
		t.Fatalf("quest count = %d, want 103", got)
	}
}

// TestQuestIDsUnique verifies ids are unique, in the quest namespace, and round-trip through
// QuestByID / IsQuestID.
func TestQuestIDsUnique(t *testing.T) {
	seen := map[int32]bool{}
	for _, q := range Quests() {
		if q.ID <= questIDBase {
			t.Errorf("quest %s id %d not above base %d", q.Key, q.ID, questIDBase)
		}
		if seen[q.ID] {
			t.Errorf("duplicate quest id %d", q.ID)
		}
		seen[q.ID] = true
		if !IsQuestID(q.ID) {
			t.Errorf("IsQuestID(%d) = false", q.ID)
		}
		if got, ok := QuestByID(q.ID); !ok || got.Key != q.Key {
			t.Errorf("QuestByID(%d) round-trip failed", q.ID)
		}
	}
	// No collision with item article ids (potions/tree/wearables all sit below 90000).
	for _, w := range Wearables() {
		if IsQuestID(w.ArticleID) {
			t.Errorf("wearable article %d collides with a quest id", w.ArticleID)
		}
	}
}

// TestQuestLocaleKeysResolve is the "no EMPTY!" guard: every locale key the catalog cites for
// every quest must exist in the baked client locale, or the client renders the literal text
// "EMPTY!" (LocaleState.GetText) for that field.
func TestQuestLocaleKeysResolve(t *testing.T) {
	keys := validLocaleKeys(t)
	check := func(qKey, field, k string) {
		if k == "" {
			t.Errorf("quest %s: empty %s key", qKey, field)
			return
		}
		if !keys[k] {
			t.Errorf("quest %s: %s key %q not in the baked locale -> would render EMPTY!", qKey, field, k)
		}
	}
	for _, q := range Quests() {
		check(q.Key, "Name", q.NameKey)
		check(q.Key, "Task", q.TaskKey)
		check(q.Key, "Start", q.StartKey)
		check(q.Key, "Progress", q.ProgKey)
		check(q.Key, "Win", q.WinKey)
		check(q.Key, "Gui", q.GuiKey)
	}
}

// TestQuestMapsValid checks every quest is bound to a real Hunt map and carries a sane
// objective/reward/kind/pve-type.
func TestQuestMapsValid(t *testing.T) {
	for _, q := range Quests() {
		if _, ok := HuntMapByID(q.MapID); !ok {
			t.Errorf("quest %s map %d is not a Hunt map", q.Key, q.MapID)
		}
		if q.Count < 1 {
			t.Errorf("quest %s count = %d, want >= 1", q.Key, q.Count)
		}
		if q.Kind != QuestKindKill && q.Kind != QuestKindCollect {
			t.Errorf("quest %s bad kind %d", q.Key, q.Kind)
		}
		if q.PveType < QuestPveSingle || q.PveType > QuestPveReplay {
			t.Errorf("quest %s bad pve type %d", q.Key, q.PveType)
		}
		if q.Money <= 0 || q.Exp <= 0 {
			t.Errorf("quest %s reward money=%d exp=%d, both must be > 0", q.Key, q.Money, q.Exp)
		}
		if q.NpcID != questGiverNpcID {
			t.Errorf("quest %s giver = %d, want %d", q.Key, q.NpcID, questGiverNpcID)
		}
	}
}

// TestQuestReplayCooldown: REPLAY quests re-offer after a cooldown; one-time quests carry none.
func TestQuestReplayCooldown(t *testing.T) {
	for _, q := range Quests() {
		if q.Repeatable() {
			if q.Cooldown <= 0 {
				t.Errorf("replay quest %s has no cooldown", q.Key)
			}
		} else if q.Cooldown != 0 {
			t.Errorf("one-time quest %s has cooldown %d", q.Key, q.Cooldown)
		}
	}
}

// TestQuestNpcsResolve: NPC name/desc keys are baked, icons are the real portraits, and the
// race filter yields the quest-giver plus lore NPCs (never the wrong race's skin).
func TestQuestNpcsResolve(t *testing.T) {
	keys := validLocaleKeys(t)
	validIcons := map[string]bool{"npc1_human": true, "npc1_elf": true, "npc2_human": true, "npc2_elf": true, "npc_event": true}
	for _, n := range questNpcs {
		if !keys[n.NameKey] {
			t.Errorf("npc %d name key %q not in locale", n.ID, n.NameKey)
		}
		if !keys[n.DescKey] {
			t.Errorf("npc %d desc key %q not in locale", n.ID, n.DescKey)
		}
		if !validIcons[n.Icon] {
			t.Errorf("npc %d icon %q is not a baked Gui/NPCMenu/Icons portrait", n.ID, n.Icon)
		}
	}
	for _, race := range []int32{1, 2} {
		got := QuestNpcsForRace(race)
		var giverQuests int
		for _, n := range got {
			if n.Race != 0 && n.Race != race {
				t.Errorf("race %d got wrong-race npc %d (race %d)", race, n.ID, n.Race)
			}
			if n.ID == questGiverNpcID {
				giverQuests = len(n.QuestIDs)
			}
		}
		if giverQuests != len(Quests()) {
			t.Errorf("race %d quest-giver offers %d quests, want %d", race, giverQuests, len(Quests()))
		}
	}
}

// TestQuestKeyForID sanity-checks the key builder used for logging.
func TestQuestKeyForID(t *testing.T) {
	q := Quests()[0]
	if got := questKeyForID(q.ID, "Name"); got != q.NameKey && got != "IDS_Quest_"+q.Key+"_Name" {
		t.Errorf("questKeyForID = %q", got)
	}
	if questKeyForID(-1, "Name") != "" {
		t.Error("questKeyForID(unknown) should be empty")
	}
}
