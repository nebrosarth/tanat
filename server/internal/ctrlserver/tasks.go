package ctrlserver

import (
	"strconv"

	"tanatserver/internal/amf"
	"tanatserver/internal/gamedata"
)

// PvP battle-task catalog served at /xml/tasks.amf. The client downloads it into the SAME
// QuestStore as /xml/quests.amf (CtrlServerConnection.DownloadQuests feeds both into mQuests), so
// the wire shape is byte-for-byte the quest shape: a DENSE array of quest MixedArrays parsed by
// QuestStore.Retrieve. A task differs only at runtime -- it is never accepted in a city; the
// Battle server assigns it and drives it live over the QUEST_TASK packet (see battleserver/
// pvptasks.go). The client SILENTLY DROPS any QUEST_TASK whose id is absent from this catalog, so
// every task id the Battle server can send MUST appear here.
//
// All text fields are baked locale keys (GetLocaleText -> "EMPTY!" if absent); the objective
// count/time-limit and rewards are authored in gamedata/tasks.go. show_cur AND show_max are both
// true on every task: the client's progress plate only renders a real "State/Limit" bar when both
// are set (a show_max-only branch prints the literal boolean instead).

// taskCatalogEntry builds one PvP task object for /xml/tasks.amf.
func taskCatalogEntry(t gamedata.PvpTask) *amf.MixedArray {
	progress := amf.NewArray()
	progress.Set(strconv.Itoa(int(gamedata.QuestProgressID())), amf.NewArray().
		Set("desc", t.GuiKey).
		Set("max", t.Count))
	return amf.NewArray().
		Set("id", t.ID).
		Set("type", t.Kind).
		Set("pve_type", int32(0)). // battle tasks are not PvE-typed (QuestPvEType.NONE)
		Set("name", t.NameKey).
		Set("task_desc", t.TaskKey).
		Set("start_desc", t.StartKey).
		Set("in_progress_desc", t.ProgKey). // HUD progress line (GuiProgress for battle tasks)
		Set("win_desc", t.WinKey).
		Set("lose_desc", t.LoseKey). // shown on a FAILED (state -2) task, unlike PvE turn-ins
		Set("money", t.Money).
		Set("money_type", int32(1)). // Currency.gold
		Set("exp", t.Exp).
		Set("show_cur", true).
		Set("show_max", true).
		Set("progress", progress)
}

// handleTasksProto builds the full /xml/tasks.amf catalog (a dense array of task objects).
func (s *Server) handleTasksProto() *amf.MixedArray {
	root := amf.NewArray()
	for _, t := range gamedata.PvpTasks() {
		root.Add(taskCatalogEntry(t))
	}
	return root
}
