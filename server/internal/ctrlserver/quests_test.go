package ctrlserver

import (
	"net"
	"net/http/httptest"
	"strconv"
	"testing"

	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/mpd"
)

func newQuestHero(t *testing.T, srv *Server, race int32) (int32, string) {
	t.Helper()
	u, key := srv.Store.LoginOrRegister("quest@example.com", "pw")
	srv.Store.CreateHero(u, race, false, 0, 0, 0, 0, 0)
	return u.ID, key
}

func questReq(sessKey string, action string, questID int32) ctrlproto.Request {
	p := amf.NewArray()
	if questID != 0 {
		p.Set("quest_id", questID)
	}
	return ctrlproto.Request{Object: "quest", Action: action, Params: p, SessKey: sessKey}
}

// questKillIdx returns a mob roster index that credits quest q (its first authored target, or
// any mob for an AnyMob quest).
func questKillIdx(q gamedata.Quest) int {
	if q.AnyMob || len(q.Targets) == 0 {
		return 0
	}
	return q.Targets[0]
}

// driveToDone accepts questID and advances it to DONE via the store kill path.
func driveToDone(t *testing.T, srv *Server, uid int32, q gamedata.Quest) {
	t.Helper()
	if _, ok := srv.Store.AcceptQuest(uid, q.ID); !ok {
		t.Fatal("AcceptQuest failed")
	}
	for i := int32(0); i < q.Count; i++ {
		srv.Store.AddQuestKill(uid, q.MapID, questKillIdx(q))
	}
}

// TestNpcListRaceScoped: npc|list returns the requesting hero's own-race NPCs, and the
// quest-giver (NPC1) carries the full quest list with a real portrait + locale keys.
func TestNpcListRaceScoped(t *testing.T) {
	srv := New()
	_, key := newQuestHero(t, srv, 2) // Elf
	resp := ctrlproto.NewResponse()
	srv.handleNpcList(questReq(key, "list", 0), resp)

	npcs, ok := subResp(t, resp, ctrlproto.CmdKey("npc", "list")).GetArray("npcs")
	if !ok || len(npcs.Assoc) == 0 {
		t.Fatal("npc|list returned no NPCs for an Elf hero")
	}
	giver, ok := npcs.Assoc[strconv.Itoa(int(gamedata.QuestGiverNpcID(2)))].(*amf.MixedArray)
	if !ok {
		t.Fatal("quest-giver NPC missing from npc|list")
	}
	if icon, _ := giver.GetString("icon"); icon != "npc1_elf" {
		t.Errorf("Elf quest-giver icon = %q, want npc1_elf", icon)
	}
	quests, ok := giver.GetArray("quests")
	if !ok || len(quests.Dense) != len(gamedata.Quests()) {
		t.Fatalf("quest-giver offers %d quests, want %d", len(quests.Dense), len(gamedata.Quests()))
	}
}

// TestQuestUpdateReflectsAcceptedState: an accepted quest shows up in quest|update as
// IN_PROGRESS with zero progress and no cooldown.
func TestQuestUpdateReflectsAcceptedState(t *testing.T) {
	srv := New()
	uid, key := newQuestHero(t, srv, 1)
	q := gamedata.Quests()[0]
	srv.Store.AcceptQuest(uid, q.ID)

	resp := ctrlproto.NewResponse()
	srv.handleQuestUpdate(questReq(key, "update", 0), resp)
	quests, ok := subResp(t, resp, ctrlproto.CmdKey("quest", "update")).GetArray("quests")
	if !ok {
		t.Fatal("quest|update missing 'quests'")
	}
	entry, ok := quests.Assoc[strconv.Itoa(int(q.ID))].(*amf.MixedArray)
	if !ok {
		t.Fatal("accepted quest not in quest|update")
	}
	if st, _ := entry.GetInt("status"); st != gamedata.QuestStatusInProgress {
		t.Errorf("status = %d, want IN_PROGRESS", st)
	}
	if ti, _ := entry.GetInt("time"); ti != -1 {
		t.Errorf("time = %d, want -1 for an active quest", ti)
	}
	prog, ok := entry.GetArray("progress")
	if !ok {
		t.Fatal("quest entry missing progress")
	}
	if cur, _ := prog.GetInt(strconv.Itoa(int(gamedata.QuestProgressID()))); cur != 0 {
		t.Errorf("fresh progress = %d, want 0", cur)
	}
}

// TestQuestAcceptGates: accept fails for an unknown id and for an already-active quest.
func TestQuestAcceptGates(t *testing.T) {
	srv := New()
	_, key := newQuestHero(t, srv, 1)
	q := gamedata.Quests()[0]

	if resp := ctrlproto.NewResponse(); func() bool {
		srv.handleQuestAccept(questReq(key, "accept", 999999), resp)
		st, _ := subResp(t, resp, ctrlproto.CmdKey("quest", "accept")).GetInt("status")
		return st == ctrlproto.StatusOK
	}() {
		t.Error("accepted an unknown quest id")
	}
	// First accept OK, second rejected (already active).
	resp1 := ctrlproto.NewResponse()
	srv.handleQuestAccept(questReq(key, "accept", q.ID), resp1)
	if st, _ := subResp(t, resp1, ctrlproto.CmdKey("quest", "accept")).GetInt("status"); st != ctrlproto.StatusOK {
		t.Fatal("first accept should succeed")
	}
	resp2 := ctrlproto.NewResponse()
	srv.handleQuestAccept(questReq(key, "accept", q.ID), resp2)
	if st, _ := subResp(t, resp2, ctrlproto.CmdKey("quest", "accept")).GetInt("status"); st == ctrlproto.StatusOK {
		t.Error("re-accepted an already-active quest")
	}
}

// TestQuestDonePaysRewardOnce: turning in a DONE quest acks, credits gold+exp, and cannot be
// repeated. A done on a not-yet-complete quest fails.
func TestQuestDonePaysRewardOnce(t *testing.T) {
	srv := New()
	uid, key := newQuestHero(t, srv, 1)
	q := gamedata.Quests()[0]

	// Not completable before the objective is met.
	srv.Store.AcceptQuest(uid, q.ID)
	respEarly := ctrlproto.NewResponse()
	srv.handleQuestDone(questReq(key, "done", q.ID), respEarly)
	if st, _ := subResp(t, respEarly, ctrlproto.CmdKey("quest", "done")).GetInt("status"); st == ctrlproto.StatusOK {
		t.Fatal("turned in an incomplete quest")
	}

	// Drive to DONE, then turn in.
	for i := int32(0); i < q.Count; i++ {
		srv.Store.AddQuestKill(uid, q.MapID, questKillIdx(q))
	}
	moneyBefore, _, _ := srv.Store.HeroMoney(uid)
	lvlBefore, expBefore, _, _ := srv.Store.AddHeroExp(uid, 0)

	resp := ctrlproto.NewResponse()
	srv.handleQuestDone(questReq(key, "done", q.ID), resp)
	if st, _ := subResp(t, resp, ctrlproto.CmdKey("quest", "done")).GetInt("status"); st != ctrlproto.StatusOK {
		t.Fatal("quest|done failed on a DONE quest")
	}
	if m, _ := subResp(t, resp, ctrlproto.CmdKey("user", "money")).GetInt("money"); m != moneyBefore+q.Money {
		t.Errorf("reward money in response = %d, want %d", m, moneyBefore+q.Money)
	}
	moneyAfter, _, _ := srv.Store.HeroMoney(uid)
	if moneyAfter != moneyBefore+q.Money {
		t.Errorf("hero gold = %d after turn-in, want %d", moneyAfter, moneyBefore+q.Money)
	}
	lvlAfter, expAfter, _, _ := srv.Store.AddHeroExp(uid, 0)
	if lvlAfter == lvlBefore && expAfter == expBefore {
		t.Error("quest exp reward was not credited")
	}

	// Second turn-in pays nothing.
	resp2 := ctrlproto.NewResponse()
	srv.handleQuestDone(questReq(key, "done", q.ID), resp2)
	if st, _ := subResp(t, resp2, ctrlproto.CmdKey("quest", "done")).GetInt("status"); st == ctrlproto.StatusOK {
		t.Error("double turn-in succeeded (double reward)")
	}
	if m, _, _ := srv.Store.HeroMoney(uid); m != moneyAfter {
		t.Errorf("gold changed on a rejected second turn-in: %d", m)
	}
}

// TestQuestCatalogEntry: a served quest object carries the baked locale keys and the single
// objective progress slot {pid:{desc,max}}.
func TestQuestCatalogEntry(t *testing.T) {
	q := gamedata.Quests()[0]
	e := questCatalogEntry(q)
	if id, _ := e.GetInt("id"); id != q.ID {
		t.Errorf("entry id = %d, want %d", id, q.ID)
	}
	if name, _ := e.GetString("name"); name != q.NameKey {
		t.Errorf("entry name = %q, want %q", name, q.NameKey)
	}
	prog, ok := e.GetArray("progress")
	if !ok {
		t.Fatal("catalog entry missing progress")
	}
	slot, ok := prog.Assoc[strconv.Itoa(int(gamedata.QuestProgressID()))].(*amf.MixedArray)
	if !ok {
		t.Fatal("progress slot missing")
	}
	if mx, _ := slot.GetInt("max"); mx != q.Count {
		t.Errorf("progress max = %d, want %d", mx, q.Count)
	}
	if desc, _ := slot.GetString("desc"); desc != q.GuiKey {
		t.Errorf("progress desc = %q, want %q", desc, q.GuiKey)
	}
	// Full catalog matches the quest count.
	if got := len(srv0().handleQuestsProto().Dense); got != len(gamedata.Quests()) {
		t.Errorf("quests.amf has %d entries, want %d", got, len(gamedata.Quests()))
	}
}

func srv0() *Server { return New() }

// TestQuestAcceptPushesOverMPD drives an accept over HTTP with a live MPD socket and asserts the
// IN_PROGRESS state is delivered as quest|update_mpd (guards the push/encoding path end to end).
func TestQuestAcceptPushesOverMPD(t *testing.T) {
	srv := New()
	hub := mpd.NewHub(srv.Store)
	srv.MPD = hub
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	go hub.Serve(ln)

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	uid, key := newQuestHero(t, srv, 1)
	_, br := dialMPD(t, ln.Addr().String(), uid, key)
	q := gamedata.Quests()[0]

	postEnvelope(t, ts.URL+"/entry_point.php", amf.NewArray().
		Set("object", "quest").Set("action", "accept").
		Set("params", amf.NewArray().Set("quest_id", q.ID)).
		Set("sess_uid", int32(0)).Set("sess_key", key).Set("counter", int32(1)))

	args := readPushArgs(t, br, "quest|update")
	if args == nil {
		t.Fatal("no quest|update_mpd pushed on accept")
	}
	quests, ok := args.GetArray("quests")
	if !ok {
		t.Fatal("push missing 'quests'")
	}
	entry, ok := quests.Assoc[strconv.Itoa(int(q.ID))].(*amf.MixedArray)
	if !ok {
		t.Fatal("accepted quest not in the push")
	}
	if st, _ := entry.GetInt("status"); st != gamedata.QuestStatusInProgress {
		t.Errorf("pushed status = %d, want IN_PROGRESS", st)
	}
}
