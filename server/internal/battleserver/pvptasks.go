package battleserver

import (
	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
)

// PvP battle-tasks in «Штурм» (map_1_0). Unlike PvE quests -- accepted in a city and progressed
// over the Ctrl channel -- a task lives entirely inside one battle: the server assigns it when
// the player enters the arena and drives it live over the QUEST_TASK packet (BattleCmdId 543).
// The client's SelfPvpQuestStore.OnTask reads {task,state,limit,time}: state>=0 is the running
// count, -3 DONE and -2 FAILED both remove the task from the HUD (the client also auto-pops the
// task's win/fail dialog). It DROPS any QUEST_TASK whose id is not in the merged quests+tasks
// catalog, so we only ever send ids authored in gamedata/tasks.go (served at /xml/tasks.amf).
//
// Solo v1 tracks only the enemy-STRUCTURE objectives (gamedata.ActivePvpTasks): a lone pusher
// destroying enemy cannons/barracks. Kill-enemy-avatar / survive / collection tasks need a second
// team that solo «Штурм» does not field, so they are catalogued but never assigned. Reward gold
// and experience are paid HERE, battle-authoritatively, when a task completes -- there is no Ctrl
// turn-in round-trip for a task -- mirroring the coin/XP award paths (awardCoinsLocked/AddHeroExp).

// PvpQuestState wire values the client's SelfPvpQuestStore recognises (TanatKernel.PvpQuestState).
// A state >= 0 is the current progress count; these two are terminal and drop the task from the
// HUD.
const (
	pvpStateFailed int32 = -2 // PvpQuestState.FAILED
	pvpStateDone   int32 = -3 // PvpQuestState.DONE
)

// pvpTaskState is one player's live progress on one assigned task. All fields are touched only
// under the conn lock (the shared instance mutex for «Штурм» members).
type pvpTaskState struct {
	def      gamedata.PvpTask
	count    int32
	endTime  float64 // absolute battle-time deadline (0 = no timer)
	done     bool
	failed   bool
	rewarded bool // single-fire reward guard
}

// pvpTaskRun is a player's set of assigned tasks for the current battle, hung on huntState.pvp.
type pvpTaskRun struct {
	tasks []*pvpTaskState
}

// sendQuestTask pushes one QUEST_TASK (543) to a single player. state is a PvpQuestState (>=0
// progress, -2 FAILED, -3 DONE); limit is the objective max; endTime is the ABSOLUTE battle-clock
// second at which a timed task expires (the client renders MM:SS of endTime-now, and draws no
// timer when endTime-now < 0), or 0 for an untimed task.
func (s *Server) sendQuestTask(c *conn, taskID, state, limit int32, endTime float64) {
	s.push(c, battleproto.CmdQuestTask, amf.NewArray().
		Set("task", taskID).
		Set("state", state).
		Set("limit", limit).
		Set("time", endTime))
}

// assignPvpTasksLocked hands this player the solo «Штурм» task set and sends the initial
// QUEST_TASK (state 0) for each. Called at the end of the world-state build (right after
// hs.worldReady), still under the conn lock, so the client's QUEST_TASK handler is subscribed and
// the merged task catalog (downloaded at login) is already populated. No-op outside «Штурм».
func (s *Server) assignPvpTasksLocked(c *conn) {
	if c.inst == nil || c.inst.dota == nil {
		return // «Штурм» only
	}
	hs := c.huntState
	if hs == nil {
		return
	}
	now := float64(s.battleTime())
	run := &pvpTaskRun{}
	for _, def := range gamedata.ActivePvpTasks() {
		st := &pvpTaskState{def: def}
		if def.TimeLimit > 0 {
			st.endTime = now + float64(def.TimeLimit)
		}
		run.tasks = append(run.tasks, st)
		s.sendQuestTask(c, def.ID, 0, def.Count, st.endTime)
	}
	hs.pvp = run
}

// creditPvpStructureKillLocked advances «Штурм» structure tasks for one destroyed enemy building,
// crediting EVERY allied player -- the objective ("Команда должна уничтожить...") belongs to the
// team, not whoever landed the final blow. It is the single crediting entry point, called from
// BOTH structure-death paths: the player-landed branch (hitMobFlagsLocked) AND the unit-vs-unit
// branch (dotaDamageLocked), where a friendly CREEP or tower routinely lands the last hit on an
// enemy cannon/barracks. A structure dies through exactly one of those paths, so credit fires once
// per building. No-op outside «Штурм» or for a non-structure victim.
func (s *Server) creditPvpStructureKillLocked(inst *huntInstance, ms *mobState) {
	if inst == nil || inst.dota == nil || ms == nil || !ms.structure {
		return
	}
	sc, ok := inst.dota.m.StructByID(ms.id - dotaStructIDBase)
	if !ok {
		return
	}
	lane := inst.dota.m.LaneFor(sc)
	for _, mem := range inst.members {
		s.advancePvpStructTasksLocked(mem, ms, lane)
	}
}

// advancePvpStructTasksLocked advances one player's tasks for a destroyed enemy structure on the
// given lane, flipping any that reach their count to DONE (state -3) and paying the reward exactly
// once. The building must be the player's ENEMY; a player's own structure never counts.
func (s *Server) advancePvpStructTasksLocked(c *conn, ms *mobState, lane int) {
	hs := c.huntState
	if hs == nil || hs.pvp == nil {
		return
	}
	if ms.team == c.playerTeam() {
		return // only ENEMY structures advance a task
	}
	for _, st := range hs.pvp.tasks {
		if st.done || st.failed || st.count >= st.def.Count {
			continue
		}
		if !pvpStructMatches(st.def, ms, lane) {
			continue
		}
		st.count++
		if st.count >= st.def.Count {
			st.done = true
			s.sendQuestTask(c, st.def.ID, pvpStateDone, st.def.Count, st.endTime)
			s.payPvpTaskRewardLocked(c, st)
		} else {
			s.sendQuestTask(c, st.def.ID, st.count, st.def.Count, st.endTime)
		}
	}
}

// pvpStructMatches reports whether a destroyed structure satisfies a task's objective: the right
// building role (cannon vs barracks) on the right lane (or any lane when the task is lane-agnostic;
// lane -1 also means "unclassified structure", which only an any-lane task accepts).
func pvpStructMatches(def gamedata.PvpTask, ms *mobState, lane int) bool {
	switch def.Objective {
	case gamedata.PvpObjCannon:
		if ms.dotaRole != gamedata.DotaGun {
			return false
		}
	case gamedata.PvpObjBarracks:
		if ms.dotaRole != gamedata.DotaCreepTower {
			return false
		}
	default:
		return false // non-structure objectives are not tracked in solo v1
	}
	return def.Lane < 0 || def.Lane == lane
}

// sweepPvpTaskTimersLocked fails any timed task whose deadline has passed with the objective
// unmet, pushing a FAILED (-2) so the client drops the plate and shows the fail dialog. Called
// every «Штурм» tick for every member. Caller holds the instance lock.
func (s *Server) sweepPvpTaskTimersLocked(rep *conn, now float64) {
	if rep.inst == nil {
		return
	}
	for _, mem := range rep.inst.members {
		hs := mem.huntState
		if hs == nil || hs.pvp == nil {
			continue
		}
		for _, st := range hs.pvp.tasks {
			if st.done || st.failed || st.endTime <= 0 {
				continue
			}
			if now >= st.endTime {
				st.failed = true
				s.sendQuestTask(mem, st.def.ID, pvpStateFailed, st.def.Count, st.endTime)
			}
		}
	}
}

// payPvpTaskRewardLocked credits a finished task's gold + experience exactly once. Gold reuses
// awardCoinsLocked (persists to the hero via AddHeroMoney and floats a "+N" over the avatar);
// experience is added to the persistent character (reflected on the next lobby game_info), the
// same split the PvE quest turn-in uses.
func (s *Server) payPvpTaskRewardLocked(c *conn, st *pvpTaskState) {
	if st.rewarded {
		return
	}
	st.rewarded = true
	if st.def.Money > 0 {
		s.awardCoinsLocked(c, c.objID, st.def.Money)
	}
	if st.def.Exp > 0 {
		s.Store.AddHeroExp(c.selfPlayerID, st.def.Exp)
	}
}
