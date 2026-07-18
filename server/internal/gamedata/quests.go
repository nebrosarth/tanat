package gamedata

import "fmt"

// PvE quests ("квесты") -- the baked story questline the city NPC (NPC1) offers. A quest is
// accepted in the central square (quest|accept), progressed by killing mobs on its Hunt map
// (map_4_0/4_1/4_2), and turned in at the NPC for gold + experience (quest|done). All display
// text (title, journal, NPC dialog, progress label) comes 1:1 from the client's baked
// IDS_Quest_* locale (resolved in quests_gen.go); the server only cites the keys, never invents
// text. The mechanical data the original server held -- map, objective count, rewards -- is
// authored here: counts are parsed from the real journal text, so "Уничтожить 20 зомби" needs
// 20 kills. Progress is map-scoped (any Hunt kill on the quest's map advances it); per-mob
// targeting is a deliberate future refinement (see the Battle server's quest-credit hook).
//
// This is a SEPARATE system from the PvP battle-tasks (map_1_0 «Штурм», the tasks.amf /
// QUEST_TASK battle packet), which are intentionally not implemented in this vertical.

// questIDBase is the first quest id. Quest ids live in their own namespace (quests.amf +
// quest|update keys), disjoint from item article ids, but we keep them clearly above the item
// ranges (potions/tree/wearables) to avoid any confusion in logs.
const questIDBase int32 = 90000

// QuestType / QuestPvEType mirror the client enums (TanatKernel.QuestType, QuestPvEType).
const (
	QuestKindKill    int32 = 1 // QuestType.KILL
	QuestKindCollect int32 = 2 // QuestType.COLLECT

	QuestPveSingle int32 = 1 // QuestPvEType.SINGLE
	QuestPveGroup  int32 = 2 // QuestPvEType.GROUP
	QuestPveReplay int32 = 3 // QuestPvEType.REPLAY (repeatable after a cooldown)
)

// QuestStatus mirrors the client enum (TanatKernel.QuestStatus) as the server tracks a hero's
// per-quest state. WAIT_COOLDOWN is the post-turn-in resting state of a REPLAY quest; a one-time
// quest ends CLOSED and is never re-offered.
const (
	QuestStatusWaitCooldown int32 = 0
	QuestStatusInProgress   int32 = 1
	QuestStatusDone         int32 = 2
	QuestStatusClosed       int32 = 3
)

// questGiverNpcID is NPC1, the single quest-giver in both race squares (all 103 PvE quests are
// keyed NPC1_PVE_*). NPC2 and the neutral NPC carry lore but offer no quests.
const questGiverNpcID int32 = 1

// questProgressID is the fixed progress-slot id used for every quest's single objective. The
// AMF `progress` map and the per-hero cur-progress map are both keyed by it.
const questProgressID int32 = 1

// questReplayCooldown is how long a REPLAY quest waits before it can be taken again (client
// counts it down from the cooldown time we send). One-time quests use 0 (they CLOSE on turn-in).
const questReplayCooldown int32 = 3600 // 1 hour

// Quest is one baked PvE quest with its authored mechanics + reward.
type Quest struct {
	ID       int32
	Key      string // stable identity infix (IDS_Quest_<Key>_...)
	NpcID    int32  // giver NPC
	MapID    int32  // Hunt map fought on (40/41/42)
	Kind     int32  // QuestType (KILL/COLLECT) -- drives the client's category icon
	PveType  int32  // QuestPvEType (SINGLE/GROUP/REPLAY)
	Count    int32  // objective count (kills needed on MapID)
	Money    int32  // gold reward
	Exp      int32  // persistent-hero experience reward
	Cooldown int32  // seconds before repeatable (REPLAY only; 0 = one-time)

	// Locale keys (pre-resolved to non-empty baked values by the generator).
	NameKey  string
	TaskKey  string
	StartKey string
	ProgKey  string
	WinKey   string
	GuiKey   string
}

// Repeatable reports whether the quest re-offers after a cooldown (REPLAY) rather than closing.
func (q Quest) Repeatable() bool { return q.PveType == QuestPveReplay }

// questMapFactor scales rewards by dungeon depth: the jungle (42) pays more than the crypt (40).
func questMapFactor(mapID int32) float64 {
	switch mapID {
	case 41:
		return 1.4
	case 42:
		return 1.8
	}
	return 1.0 // 40 (crypt)
}

// questMoney / questExp author the gold + experience payout from the objective size, map depth
// and quest kind (a GROUP quest, meant for a party, pays a premium). Kept modest so quests
// supplement -- not replace -- ordinary mob farming.
func questMoney(d questDef) int32 {
	base := 15.0 + float64(d.Count)*2.0
	if d.PveType == QuestPveGroup {
		base *= 1.5
	}
	return int32(base * questMapFactor(d.MapID))
}

func questExp(d questDef) int32 {
	base := 25.0 + float64(d.Count)*3.0
	if d.PveType == QuestPveGroup {
		base *= 1.5
	}
	return int32(base * questMapFactor(d.MapID))
}

var (
	quests       []Quest
	questByID    map[int32]Quest
	questNpcs    []QuestNPC
	questsForMap map[int32][]Quest
)

// QuestNPC is one quest-hub NPC as npc|list serves it. Race scopes which square shows it (a
// Human hero never sees the Elf-skinned NPC1). Icon is the leaf under Gui/NPCMenu/Icons (a real
// baked portrait: npc1_human/npc1_elf/npc2_human/npc2_elf/npc_event).
type QuestNPC struct {
	ID       int32
	Race     int32 // 1 Human, 2 Elf, 0 = both (neutral)
	NameKey  string
	DescKey  string
	Icon     string
	NeedShow bool
	QuestIDs []int32 // quests this NPC offers (resolved to the race-appropriate set at serve time)
}

func init() {
	quests = make([]Quest, 0, len(questDefs))
	questByID = make(map[int32]Quest, len(questDefs))
	questsForMap = map[int32][]Quest{}
	for i, d := range questDefs {
		q := Quest{
			ID:       questIDBase + int32(i) + 1,
			Key:      d.Key,
			NpcID:    questGiverNpcID,
			MapID:    d.MapID,
			Kind:     d.Kind,
			PveType:  d.PveType,
			Count:    d.Count,
			Money:    questMoney(d),
			Exp:      questExp(d),
			NameKey:  d.NameKey,
			TaskKey:  d.TaskKey,
			StartKey: d.StartKey,
			ProgKey:  d.ProgKey,
			WinKey:   d.WinKey,
			GuiKey:   d.GuiKey,
		}
		if q.PveType == QuestPveReplay {
			q.Cooldown = questReplayCooldown
		}
		quests = append(quests, q)
		questByID[q.ID] = q
		questsForMap[q.MapID] = append(questsForMap[q.MapID], q)
	}

	// The quest-giver offers every PvE quest (all are NPC1_PVE_*). Same id in both squares,
	// race-skinned name/desc/icon.
	allIDs := make([]int32, len(quests))
	for i, q := range quests {
		allIDs[i] = q.ID
	}
	questNpcs = []QuestNPC{
		{ID: 1, Race: 1, NameKey: "IDS_NPC1_Human_Name", DescKey: "IDS_NPC1_Human_Desc", Icon: "npc1_human", NeedShow: true, QuestIDs: allIDs},
		{ID: 1, Race: 2, NameKey: "IDS_NPC1_Elf_Name", DescKey: "IDS_NPC1_Elf_Desc", Icon: "npc1_elf", NeedShow: true, QuestIDs: allIDs},
		// Lore NPCs with no quests -- shown so the square's NPC roster matches the baked scene.
		{ID: 2, Race: 1, NameKey: "IDS_NPC2_Human_Name", DescKey: "IDS_NPC2_Human_Desc", Icon: "npc2_human", NeedShow: true},
		{ID: 2, Race: 2, NameKey: "IDS_NPC2_Elf_Name", DescKey: "IDS_NPC2_Elf_Desc", Icon: "npc2_elf", NeedShow: true},
		{ID: 3, Race: 0, NameKey: "IDS_Npc_Neutral_01_Name", DescKey: "IDS_Npc_Neutral_01_Desc", Icon: "npc_event", NeedShow: true},
	}
}

// Quests returns the full baked PvE quest catalog (read-only).
func Quests() []Quest { return quests }

// QuestByID resolves a quest by its id.
func QuestByID(id int32) (Quest, bool) {
	q, ok := questByID[id]
	return q, ok
}

// IsQuestID reports whether id names a quest.
func IsQuestID(id int32) bool {
	_, ok := questByID[id]
	return ok
}

// QuestsOnMap returns the quests fought on a given Hunt map (nil if none). Used by the Battle
// server to know which accepted quests a kill on that map can advance.
func QuestsOnMap(mapID int32) []Quest { return questsForMap[mapID] }

// QuestGiverNpcID is NPC1, the id every quest is offered under.
func QuestGiverNpcID() int32 { return questGiverNpcID }

// QuestProgressID is the fixed progress-slot id (both the AMF progress map and the per-hero
// cur-progress are keyed by it).
func QuestProgressID() int32 { return questProgressID }

// QuestNpcsForRace returns the NPCs a hero of the given race code (1 Human, 2 Elf) sees in its
// square: the race-matching skinned NPCs plus the race-neutral ones.
func QuestNpcsForRace(raceCode int32) []QuestNPC {
	out := make([]QuestNPC, 0, len(questNpcs))
	for _, n := range questNpcs {
		if n.Race == 0 || n.Race == raceCode {
			out = append(out, n)
		}
	}
	return out
}

// questKeyForID builds a specific field key for a quest (used only in tests/logging).
func questKeyForID(id int32, field string) string {
	q, ok := questByID[id]
	if !ok {
		return ""
	}
	return fmt.Sprintf("IDS_Quest_%s_%s", q.Key, field)
}
