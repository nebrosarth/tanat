package ctrlserver

import (
	"strconv"
	"time"

	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// PvE quests ("квесты") on the Ctrl channel. Verified wire facts (TanatKernel.QuestStore/
// SelfQuestStore/NpcStore + the *ArgParsers):
//   - /xml/quests.amf is a DENSE array of quest objects the client's QuestStore.Retrieve
//     parses at connect (fields: id/type/pve_type/name/task_desc/start_desc/in_progress_desc/
//     win_desc/money/money_type/exp/show_cur/show_max/progress{pid:{desc,max}}). All the text
//     fields are LOCALE KEYS -- GetLocaleText resolves them, so an unknown key renders "EMPTY!".
//   - npc|list returns {npcs:{"<id>":{name,desc,icon,need_show,quests:[id...]}}}; name/desc are
//     locale keys and icon is a leaf under Gui/NPCMenu/Icons.
//   - quest|update returns the requesting hero's ACCEPTED quest state {quests:{"<id>":{status,
//     time,progress:{"<pid>":cur}}}}; the same shape rides quest|update_mpd (under "arguments").
//   - quest|accept/cancel/done carry {quest_id}; none has a reply parser, so each just needs a
//     status:100 ack and the server drives the client via a quest|update_mpd push (accept ->
//     IN_PROGRESS; cancel -> a WAIT_COOLDOWN+time:-1 state the client REMOVES; done -> CLOSED or
//     a cooling REPLAY). "time" is the cooldown seconds remaining (>=0) or -1 for no cooldown.
//
// Progress is server-authoritative and advanced by the Battle server on Hunt mob kills (see
// battleserver/quests.go); quest|done pays the gold+exp bounty exactly once (CompleteQuest gates
// it). errQuest is any non-100 code (the client only treats status==100 as success).
const errQuest int32 = 1

// handleNpcList answers npc|list with the requesting hero's race-appropriate quest NPCs. An
// empty-but-valid npcs map is returned when there is no hero, so the client doesn't throw on a
// missing "npcs" key.
func (s *Server) handleNpcList(req ctrlproto.Request, resp *ctrlproto.Response) {
	npcs := amf.NewArray()
	if h := s.heroFromSession(req); h != nil {
		for _, n := range gamedata.QuestNpcsForRace(h.Race) {
			quests := amf.NewArray()
			for _, qid := range n.QuestIDs {
				quests.Add(qid)
			}
			npcs.Set(strconv.Itoa(int(n.ID)), amf.NewArray().
				Set("name", n.NameKey).
				Set("desc", n.DescKey).
				Set("icon", n.Icon).
				Set("need_show", n.NeedShow).
				Set("quests", quests))
		}
	}
	resp.Add("npc", "list", amf.NewArray().Set("npcs", npcs))
}

// handleQuestUpdate answers quest|update with the hero's accepted quest states.
func (s *Server) handleQuestUpdate(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	quests := amf.NewArray()
	if u != nil && u.Hero != nil {
		quests = questStatesToAMF(s.Store.HeroQuests(u.ID))
	}
	resp.Add("quest", "update", amf.NewArray().Set("quests", quests))
}

// handleQuestAccept serves quest|accept {quest_id}: start a real, offered quest. The client
// gates re-accepts too, but the server is the authority (AcceptQuest rejects already-active,
// CLOSED, or still-cooling quests). On success it acks and pushes the new IN_PROGRESS state.
func (s *Server) handleQuestAccept(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil || u.Hero == nil {
		resp.Fail("quest", "accept", errQuest)
		return
	}
	qid, _ := req.Params.GetInt("quest_id")
	if !gamedata.IsQuestID(qid) {
		resp.Fail("quest", "accept", errQuest)
		return
	}
	qs, ok := s.Store.AcceptQuest(u.ID, qid)
	if !ok {
		resp.Fail("quest", "accept", errQuest)
		return
	}
	resp.Ack("quest", "accept")
	s.pushQuestStates(u.ID, []session.QuestState{qs})
}

// handleQuestCancel serves quest|cancel {quest_id}: abandon an accepted quest. The removal is
// pushed as a WAIT_COOLDOWN state with time:-1, which the client's SelfQuestStore drops.
func (s *Server) handleQuestCancel(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil || u.Hero == nil {
		resp.Fail("quest", "cancel", errQuest)
		return
	}
	qid, _ := req.Params.GetInt("quest_id")
	if !s.Store.CancelQuest(u.ID, qid) {
		resp.Fail("quest", "cancel", errQuest)
		return
	}
	resp.Ack("quest", "cancel")
	// A WAIT_COOLDOWN state with no cooldown time makes the client remove the quest entirely.
	s.pushQuestStates(u.ID, []session.QuestState{{QuestID: qid, Status: gamedata.QuestStatusWaitCooldown}})
}

// handleQuestDone serves quest|done {quest_id}: turn in a completed quest. CompleteQuest is the
// single-fire gate (status must be DONE), so the gold+exp bounty is paid exactly once even under
// a double request. It then pushes the new balance and the quest's post-turn-in state.
func (s *Server) handleQuestDone(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil || u.Hero == nil {
		resp.Fail("quest", "done", errQuest)
		return
	}
	qid, _ := req.Params.GetInt("quest_id")
	rew, ok := s.Store.CompleteQuest(u.ID, qid)
	if !ok {
		resp.Fail("quest", "done", errQuest)
		return
	}
	// CompleteQuest already credited gold+exp atomically with the state transition; just report
	// the new balance (exp/level reflect on the next lobby game_info refresh).
	resp.Ack("quest", "done")
	resp.Add("user", "money", amf.NewArray().Set("money", rew.Money).Set("money_d", rew.Diamonds))
	// Push the post-turn-in state (CLOSED, or a cooling REPLAY) so the NPC marks refresh.
	if states := s.Store.HeroQuests(u.ID); states != nil {
		for _, qs := range states {
			if qs.QuestID == qid {
				s.pushQuestStates(u.ID, []session.QuestState{qs})
				return
			}
		}
	}
	// One-time quest that HeroQuests still reports as CLOSED, or a REPLAY already filtered out:
	// fall back to an explicit CLOSED push so the client stops showing it as ready-to-turn-in.
	s.pushQuestStates(u.ID, []session.QuestState{{QuestID: qid, Status: gamedata.QuestStatusClosed}})
}

// pushQuestStates delivers quest states to a user's MPD socket as quest|update_mpd (no-op if
// offline). Shared by the Ctrl handlers and, indirectly, the Battle kill-credit path.
func (s *Server) pushQuestStates(userID int32, states []session.QuestState) {
	if s.MPD == nil || len(states) == 0 {
		return
	}
	s.MPD.Push(userID, "quest|update", amf.NewArray().Set("quests", questStatesToAMF(states)))
}

// questStatesToAMF encodes hero quest states into the wire {quests:{"<id>":{status,time,
// progress}}} map. time is the REPLAY cooldown seconds remaining, or -1 (no cooldown / active).
func questStatesToAMF(states []session.QuestState) *amf.MixedArray {
	quests := amf.NewArray()
	now := time.Now().Unix()
	for _, qs := range states {
		prog := amf.NewArray()
		prog.Set(strconv.Itoa(int(gamedata.QuestProgressID())), qs.Progress)
		cd := int32(-1)
		if qs.Status == gamedata.QuestStatusWaitCooldown && qs.CooldownUntil > now {
			cd = int32(qs.CooldownUntil - now)
		}
		quests.Set(strconv.Itoa(int(qs.QuestID)), amf.NewArray().
			Set("status", qs.Status).
			Set("time", cd).
			Set("progress", prog))
	}
	return quests
}

// questCatalogEntry builds one quest object for /xml/quests.amf. All text fields are the baked
// locale keys the client resolves; progress carries the single objective slot {pid:{desc,max}}.
func questCatalogEntry(q gamedata.Quest) *amf.MixedArray {
	progress := amf.NewArray()
	progress.Set(strconv.Itoa(int(gamedata.QuestProgressID())), amf.NewArray().
		Set("desc", q.GuiKey).
		Set("max", q.Count))
	return amf.NewArray().
		Set("id", q.ID).
		Set("type", q.Kind).
		Set("pve_type", q.PveType).
		Set("name", q.NameKey).
		Set("task_desc", q.TaskKey).
		Set("start_desc", q.StartKey).
		Set("in_progress_desc", q.ProgKey).
		Set("win_desc", q.WinKey).
		Set("money", q.Money).
		Set("money_type", int32(1)). // Currency.gold
		Set("exp", q.Exp).
		Set("show_cur", true).
		Set("show_max", true).
		Set("progress", progress)
}

// handleQuestsProto builds the full /xml/quests.amf catalog (a dense array of quest objects).
func (s *Server) handleQuestsProto() *amf.MixedArray {
	root := amf.NewArray()
	for _, q := range gamedata.Quests() {
		root.Add(questCatalogEntry(q))
	}
	return root
}
