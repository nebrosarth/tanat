package gamedata

// PvP battle-tasks ("задания") for «Штурм» (map_1_0) -- the Battle-channel counterpart to the
// PvE quests. The client downloads /xml/tasks.amf into the SAME QuestStore as /xml/quests.amf
// (CtrlServerConnection.DownloadQuests(QuestsUrl) + DownloadQuests(TasksUrl) both feed mQuests),
// so a task is just a quest-definition in the identical QuestStore.Retrieve shape, living in a
// disjoint id range. What differs is the runtime: a task is never accepted in a city; the Battle
// server assigns it at match start and drives it live over the QUEST_TASK packet (BattleCmdId
// 543) -- SelfPvpQuestStore.OnTask reads {task,state,limit,time}. state>=0 is the current count,
// -1 UNINITED, -2 FAILED, -3 DONE (the client drops a DONE/FAILED task from the HUD). The client
// SILENTLY DROPS any QUEST_TASK whose id is absent from the merged catalog, so EVERY id the
// server could send must be authored here.
//
// As with items and quests, all display text is baked locale KEYS the client resolves via
// GetLocaleText (an unknown/empty key renders the literal "EMPTY!"); the server cites keys, never
// invents text. The objective/count/time-limit are parsed from the real Russian journal text
// (see the pvp-tasks-research memory): "Уничтожьте три вражеские казармы в течение 30 минут" is
// 3 barracks in 1800s. The reward gold/exp are authored here (the original server held them).
//
// Solo v1 reality: «Штурм» is one player (Human side) versus the AI Elf base -- there are no
// enemy AVATARS and no second team. So only the tasks a lone pusher can finish against enemy
// STRUCTURES are Active (assigned + tracked): destroy enemy cannons/barracks. The kill-enemy-
// players / survive / comparative / collection tasks need multi-team «Штурм» that does not exist
// yet, so they are authored (the id resolves, the HUD could show them) but not assigned -- the
// same "ship the completable core, defer the infra-bound rest" split the PvE vertical used.

// pvpTaskIDBase keeps PvP task ids in a disjoint 1000-wide window above the PvE quests (base
// 90000, ~90001..90103) and every item article range (potions 50000, tree 60000, wearables
// 80000). The low digits encode the family so the VS pairing is arithmetic (defender = attacker
// + 10) and ids are self-documenting; see the seed table below.
const pvpTaskIDBase int32 = 91000

// PvpObjective classifies what in-match event advances a task. Only PvpObjCannon/PvpObjBarracks
// are auto-tracked in solo v1 (the Battle server's kill hook); the rest are authored for the
// catalog but wait on multi-team «Штурм» or extra hooks (see the Active flag).
type PvpObjective int

const (
	PvpObjOther       PvpObjective = iota // authored, not auto-tracked in solo v1
	PvpObjCannon                          // destroy an enemy cannon (DotaGun)
	PvpObjBarracks                        // destroy an enemy barracks (DotaCreepTower)
	PvpObjKillPlayers                     // kill enemy avatars (needs a second team)
	PvpObjReachLevel                      // reach an avatar level
	PvpObjSurvive                         // hold a structure for the duration (VS defender)
	PvpObjCollect                         // cross-battle collection (PVP_Event soul shards)
)

// pvpLaneAny means a structure task counts a kill on any lane (no lane filter).
const pvpLaneAny = -1

// PvpTask is one baked «Штурм» battle-task: its wire id, the locale keys the client resolves, the
// authored objective/count/time-limit and reward, and whether solo v1 actually assigns it.
type PvpTask struct {
	ID        int32
	Key       string       // locale infix: IDS_Quest_<Key>_<field>
	Group     string       // family: Group / GroupVS / GroupVS_VS / Single / Event
	Kind      int32        // QuestType (KILL/COLLECT) -- the client's category icon
	Objective PvpObjective // what advances it (server-side)
	Lane      int          // structure tasks: 0 north / 1 centre / 2 south / -1 any
	Count     int32        // objective max (QUEST_TASK "limit")
	TimeLimit int32        // seconds to finish, 0 = the whole battle (no countdown)
	Money     int32        // gold reward on completion
	Exp       int32        // persistent-hero experience reward
	VSPartner int32        // the paired _VS task id (mirror objective), 0 if none
	Active    bool         // assigned + tracked in solo «Штурм» v1

	// Locale keys, resolved to the present-and-non-empty baked key for each field.
	NameKey  string
	TaskKey  string
	StartKey string
	ProgKey  string // in_progress_desc: the HUD progress line
	WinKey   string
	LoseKey  string
	GuiKey   string // progress[pid].desc
}

// Repeatable is false for battle-tasks: they live only for one match (there is no cooldown/
// re-accept model like the PvE REPLAY quests). Present for API symmetry with Quest.
func (PvpTask) Repeatable() bool { return false }

// pvpTaskSeed is the hand-authored design row; init() derives ids' locale keys and rewards.
type pvpTaskSeed struct {
	id        int32
	key       string
	group     string
	isEvent   bool // uses the PvE-style _DialogProgress/_DialogEnd suffixes, not _DialogSuccess/_Fail
	kind      int32
	obj       PvpObjective
	lane      int
	count     int32
	timeLimit int32
	vs        int32
	active    bool
}

// pvpTaskSeeds authors every «Штурм» task from the baked IDS_Quest_Map_1_0_PVP_* / IDS_Quest_
// PVP_Event_1 families. Objective/count/time-limit come from the real journal text. Only the
// enemy-structure tasks are Active (solo-completable); see the file header.
var pvpTaskSeeds = []pvpTaskSeed{
	// Group_Type1_* -- one-sided team objectives (id 91001..91009).
	{id: pvpTaskIDBase + 1, key: "Map_1_0_PVP_Group_Type1_1", group: "Group", kind: QuestKindKill, obj: PvpObjCannon, lane: 1, count: 1, timeLimit: 600, active: true}, // «На штурм!» first centre cannon in 10m
	{id: pvpTaskIDBase + 2, key: "Map_1_0_PVP_Group_Type1_2", group: "Group", kind: QuestKindKill, obj: PvpObjKillPlayers, lane: pvpLaneAny, count: 1},                 // «Больше крови» out-kill the enemy team
	{id: pvpTaskIDBase + 3, key: "Map_1_0_PVP_Group_Type1_3", group: "Group", kind: QuestKindKill, obj: PvpObjOther, lane: pvpLaneAny, count: 40},                      // «Рота смертников» 40 enemy melee soldiers
	{id: pvpTaskIDBase + 4, key: "Map_1_0_PVP_Group_Type1_4", group: "Group", kind: QuestKindKill, obj: PvpObjKillPlayers, lane: pvpLaneAny, count: 1},                 // «Первая кровь» team's first avatar kill
	{id: pvpTaskIDBase + 5, key: "Map_1_0_PVP_Group_Type1_5", group: "Group", kind: QuestKindKill, obj: PvpObjKillPlayers, lane: pvpLaneAny, count: 20},                // «Приказано убивать» 20 avatar kills
	{id: pvpTaskIDBase + 6, key: "Map_1_0_PVP_Group_Type1_6", group: "Group", kind: QuestKindKill, obj: PvpObjCannon, lane: pvpLaneAny, count: 3, active: true},        // «Трезубец войны» three first cannons, all lanes
	{id: pvpTaskIDBase + 7, key: "Map_1_0_PVP_Group_Type1_7", group: "Group", kind: QuestKindKill, obj: PvpObjCannon, lane: 0, count: 3},                               // «Линия истребления» all top-lane cannons
	{id: pvpTaskIDBase + 8, key: "Map_1_0_PVP_Group_Type1_8", group: "Group", kind: QuestKindKill, obj: PvpObjOther, lane: pvpLaneAny, count: 1},                       // «Вызов монстру» the Spider Queen boss
	{id: pvpTaskIDBase + 9, key: "Map_1_0_PVP_Group_Type1_9", group: "Group", kind: QuestKindKill, obj: PvpObjBarracks, lane: 2, count: 1, active: true},               // «На пороге крепости» south-lane barracks

	// GroupVS_Type1_* -- attacker half of a two-team race (id 91011..91014); partner = id+10.
	{id: pvpTaskIDBase + 11, key: "Map_1_0_PVP_GroupVS_Type1_1", group: "GroupVS", kind: QuestKindKill, obj: PvpObjCannon, lane: 0, count: 1, timeLimit: 600, vs: pvpTaskIDBase + 21},             // «Дерзкий прорыв»
	{id: pvpTaskIDBase + 12, key: "Map_1_0_PVP_GroupVS_Type1_2", group: "GroupVS", kind: QuestKindKill, obj: PvpObjCannon, lane: pvpLaneAny, count: 3, timeLimit: 900, vs: pvpTaskIDBase + 22},    // «Огонь из всех орудий»
	{id: pvpTaskIDBase + 13, key: "Map_1_0_PVP_GroupVS_Type1_3", group: "GroupVS", kind: QuestKindKill, obj: PvpObjBarracks, lane: pvpLaneAny, count: 3, timeLimit: 1800, vs: pvpTaskIDBase + 23}, // «В тылу врага»
	{id: pvpTaskIDBase + 14, key: "Map_1_0_PVP_GroupVS_Type1_4", group: "GroupVS", kind: QuestKindKill, obj: PvpObjCannon, lane: 1, count: 3, timeLimit: 1200, vs: pvpTaskIDBase + 24},            // «Подавление обороны» (middle lane authoritative)

	// GroupVS_Type1_*_VS -- defender mirror (id 91021..91024); partner = id-10.
	{id: pvpTaskIDBase + 21, key: "Map_1_0_PVP_GroupVS_Type1_1_VS", group: "GroupVS_VS", kind: QuestKindKill, obj: PvpObjSurvive, lane: 0, count: 1, timeLimit: 600, vs: pvpTaskIDBase + 11},           // «Ни шагу назад»
	{id: pvpTaskIDBase + 22, key: "Map_1_0_PVP_GroupVS_Type1_2_VS", group: "GroupVS_VS", kind: QuestKindKill, obj: PvpObjSurvive, lane: pvpLaneAny, count: 3, timeLimit: 900, vs: pvpTaskIDBase + 12},  // «По всем фронтам»
	{id: pvpTaskIDBase + 23, key: "Map_1_0_PVP_GroupVS_Type1_3_VS", group: "GroupVS_VS", kind: QuestKindKill, obj: PvpObjSurvive, lane: pvpLaneAny, count: 3, timeLimit: 1800, vs: pvpTaskIDBase + 13}, // «Своих не бросаем»
	{id: pvpTaskIDBase + 24, key: "Map_1_0_PVP_GroupVS_Type1_4_VS", group: "GroupVS_VS", kind: QuestKindKill, obj: PvpObjSurvive, lane: 1, count: 3, timeLimit: 1200, vs: pvpTaskIDBase + 14},          // «Сохранение огневой мощи»

	// Single_Type1_* -- personal objectives (id 91031..91033).
	{id: pvpTaskIDBase + 31, key: "Map_1_0_PVP_Single_Type1_1", group: "Single", kind: QuestKindKill, obj: PvpObjOther, lane: pvpLaneAny, count: 3},                        // «Сдерживание осады» 3 siege creeps
	{id: pvpTaskIDBase + 32, key: "Map_1_0_PVP_Single_Type1_2", group: "Single", kind: QuestKindKill, obj: PvpObjKillPlayers, lane: pvpLaneAny, count: 5},                  // «Личные счеты» 5 enemy avatars
	{id: pvpTaskIDBase + 33, key: "Map_1_0_PVP_Single_Type1_3", group: "Single", kind: QuestKindKill, obj: PvpObjReachLevel, lane: pvpLaneAny, count: 15, timeLimit: 1200}, // «Новые горизонты» reach level 15 in 20m

	// PVP_Event_1 -- a persistent cross-battle collection meta-quest (id 91900), not a per-match
	// task: collect soul shards from enemy-avatar kills across battles. Uses the PvE-style
	// _DialogProgress/_DialogEnd suffixes and QuestType.COLLECT.
	{id: pvpTaskIDBase + 900, key: "PVP_Event_1", group: "Event", isEvent: true, kind: QuestKindCollect, obj: PvpObjCollect, lane: pvpLaneAny, count: 400},
}

var (
	pvpTasks      []PvpTask
	pvpTaskByID   map[int32]PvpTask
	pvpTaskActive []PvpTask
)

// pvpTaskMoney/pvpTaskExp author a modest reward from the objective size, so a finished task
// supplements a «Штурм» push without dwarfing ordinary coin/xp income.
func pvpTaskMoney(count int32) int32 { return 25 + count*15 }
func pvpTaskExp(count int32) int32   { return 30 + count*20 }

// pvpTaskKey builds "IDS_Quest_<Key>_<field>".
func pvpTaskKey(infix, field string) string { return "IDS_Quest_" + infix + "_" + field }

func init() {
	pvpTasks = make([]PvpTask, 0, len(pvpTaskSeeds))
	pvpTaskByID = make(map[int32]PvpTask, len(pvpTaskSeeds))
	for _, sd := range pvpTaskSeeds {
		t := PvpTask{
			ID:        sd.id,
			Key:       sd.key,
			Group:     sd.group,
			Kind:      sd.kind,
			Objective: sd.obj,
			Lane:      sd.lane,
			Count:     sd.count,
			TimeLimit: sd.timeLimit,
			Money:     pvpTaskMoney(sd.count),
			Exp:       pvpTaskExp(sd.count),
			VSPartner: sd.vs,
			Active:    sd.active,
			NameKey:   pvpTaskKey(sd.key, "Name"),
			TaskKey:   pvpTaskKey(sd.key, "JournalDesc"),
			StartKey:  pvpTaskKey(sd.key, "DialogBegin"),
			GuiKey:    pvpTaskKey(sd.key, "GuiProgress"),
		}
		if sd.isEvent {
			// The event alone carries the PvE-style suffixes: a real in-progress line and a
			// single _DialogEnd for both the win text and the (never-shown) lose text.
			t.ProgKey = pvpTaskKey(sd.key, "DialogProgress")
			t.WinKey = pvpTaskKey(sd.key, "DialogEnd")
			t.LoseKey = pvpTaskKey(sd.key, "DialogEnd")
		} else {
			// Battle tasks have no _DialogProgress key: cite the GuiProgress line for the HUD's
			// in-progress text so it never renders "EMPTY!".
			t.ProgKey = pvpTaskKey(sd.key, "GuiProgress")
			t.WinKey = pvpTaskKey(sd.key, "DialogSuccess")
			t.LoseKey = pvpTaskKey(sd.key, "DialogFail")
		}
		pvpTasks = append(pvpTasks, t)
		pvpTaskByID[t.ID] = t
		if t.Active {
			pvpTaskActive = append(pvpTaskActive, t)
		}
	}
}

// PvpTasks returns the full baked «Штурм» task catalog (read-only) for /xml/tasks.amf.
func PvpTasks() []PvpTask { return pvpTasks }

// PvpTaskByID resolves a task by its id.
func PvpTaskByID(id int32) (PvpTask, bool) {
	t, ok := pvpTaskByID[id]
	return t, ok
}

// IsPvpTaskID reports whether id names a PvP task.
func IsPvpTaskID(id int32) bool {
	_, ok := pvpTaskByID[id]
	return ok
}

// ActivePvpTasks returns the tasks solo «Штурм» v1 assigns and tracks (enemy-structure
// objectives). The Battle server sends these at match start and advances them on kills.
func ActivePvpTasks() []PvpTask { return pvpTaskActive }
