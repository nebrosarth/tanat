package ctrlserver

import (
	"strconv"
	"testing"

	"tanatserver/internal/amf"
	"tanatserver/internal/gamedata"
)

// TestTaskCatalogEntry: a task encodes its id/name/reward/progress and the whole /xml/tasks.amf
// dense array carries every authored PvP task.
func TestTaskCatalogEntry(t *testing.T) {
	tk := gamedata.PvpTasks()[0]
	e := taskCatalogEntry(tk)
	if id, _ := e.GetInt("id"); id != tk.ID {
		t.Errorf("entry id = %d, want %d", id, tk.ID)
	}
	if name, _ := e.GetString("name"); name != tk.NameKey {
		t.Errorf("entry name = %q, want %q", name, tk.NameKey)
	}
	// A battle task cites its GuiProgress key for the in-progress HUD line (no _DialogProgress).
	if ip, _ := e.GetString("in_progress_desc"); ip != tk.ProgKey {
		t.Errorf("in_progress_desc = %q, want %q", ip, tk.ProgKey)
	}
	if lose, _ := e.GetString("lose_desc"); lose != tk.LoseKey {
		t.Errorf("lose_desc = %q, want %q", lose, tk.LoseKey)
	}
	// show_cur AND show_max must both be true or the client renders no real progress bar.
	if sc, _ := e.GetBool("show_cur"); !sc {
		t.Error("show_cur must be true")
	}
	if sm, _ := e.GetBool("show_max"); !sm {
		t.Error("show_max must be true")
	}
	prog, ok := e.GetArray("progress")
	if !ok {
		t.Fatal("catalog entry missing progress")
	}
	slot, ok := prog.Assoc[strconv.Itoa(int(gamedata.QuestProgressID()))].(*amf.MixedArray)
	if !ok {
		t.Fatal("progress slot missing")
	}
	if mx, _ := slot.GetInt("max"); mx != tk.Count {
		t.Errorf("progress max = %d, want %d", mx, tk.Count)
	}

	if got := len(New().handleTasksProto().Dense); got != len(gamedata.PvpTasks()) {
		t.Errorf("tasks.amf has %d entries, want %d", got, len(gamedata.PvpTasks()))
	}
}
