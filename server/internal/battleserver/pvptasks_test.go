package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

// pvpTaskOf returns a player's live state for one task id (nil if unassigned).
func pvpTaskOf(c *conn, id int32) *pvpTaskState {
	if c.huntState == nil || c.huntState.pvp == nil {
		return nil
	}
	for _, st := range c.huntState.pvp.tasks {
		if st.def.ID == id {
			return st
		}
	}
	return nil
}

// enemyStructsByRole collects the enemy (Elf) structures of a role, split by lane so a test can
// pick a specific-lane building to destroy.
func enemyStructsByLane(inst *huntInstance, role gamedata.DotaRole) map[int][]*mobState {
	out := map[int][]*mobState{}
	for _, m := range inst.mobs {
		if !m.structure || m.team != dotaEnemyTeam || m.dotaRole != role {
			continue
		}
		sc, ok := inst.dota.m.StructByID(m.id - dotaStructIDBase)
		if !ok {
			continue
		}
		lane := inst.dota.m.LaneFor(sc)
		out[lane] = append(out[lane], m)
	}
	return out
}

// heroMoney reads a hero's current gold (AddHeroMoney with a zero delta is a read).
func heroMoney(t *testing.T, s *Server, uid int32) int32 {
	t.Helper()
	money, _, ok := s.Store.AddHeroMoney(uid, 0)
	if !ok {
		t.Fatal("no hero bound")
	}
	return money
}

// bindHero attaches a fresh hero to a battle conn so rewards can be credited.
func bindHero(t *testing.T, s *Server, c *conn, email string) int32 {
	t.Helper()
	u, _, _ := s.Store.LoginOrRegister(email, "pw")
	s.Store.CreateHero(u, 1, false, 0, 0, 0, 0, 0)
	c.selfPlayerID = u.ID
	return u.ID
}

// TestPvpTaskAssign: entering «Штурм» assigns exactly the active catalog tasks, each fresh
// (count 0, not done/failed), with a deadline only on the timed ones.
func TestPvpTaskAssign(t *testing.T) {
	s, c, _, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	bindHero(t, s, c, "assign@test.io")

	c.lock()
	s.assignPvpTasksLocked(c)
	c.unlock()

	active := gamedata.ActivePvpTasks()
	if c.huntState.pvp == nil || len(c.huntState.pvp.tasks) != len(active) {
		t.Fatalf("assigned %v tasks, want %d", c.huntState.pvp, len(active))
	}
	for _, def := range active {
		st := pvpTaskOf(c, def.ID)
		if st == nil {
			t.Fatalf("task %s not assigned", def.Key)
		}
		if st.count != 0 || st.done || st.failed || st.rewarded {
			t.Errorf("task %s not fresh: %+v", def.Key, st)
		}
		if def.TimeLimit > 0 && st.endTime <= 0 {
			t.Errorf("timed task %s has no deadline", def.Key)
		}
		if def.TimeLimit == 0 && st.endTime != 0 {
			t.Errorf("untimed task %s has a deadline %v", def.Key, st.endTime)
		}
	}
}

// TestPvpTaskStructureProgress drives the kill hook end to end: destroying enemy cannons/barracks
// advances the matching tasks, completes them at their count, pays the reward exactly once, and
// ignores non-matching kills (ally structure, creep, wrong lane).
func TestPvpTaskStructureProgress(t *testing.T) {
	s, c, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	uid := bindHero(t, s, c, "progress@test.io")

	c.lock()
	s.assignPvpTasksLocked(c)
	c.unlock()

	guns := enemyStructsByLane(inst, gamedata.DotaGun)
	centre := guns[1]
	if len(centre) == 0 {
		t.Fatal("no enemy centre-lane cannon in the seeded map")
	}
	money0 := heroMoney(t, s, uid)

	// Kill one centre cannon: completes 91001 (centre, count 1) AND advances 91006 (any, count 3).
	credit := func(m *mobState) {
		c.lock()
		s.creditPvpStructureKillLocked(inst, m)
		c.unlock()
	}
	credit(centre[0])

	g1 := pvpTaskOf(c, gamedata.ActivePvpTasks()[0].ID) // 91001
	if g1 == nil {
		t.Fatal("91001 not assigned")
	}
	t1, _ := gamedata.PvpTaskByID(91001)
	t6, _ := gamedata.PvpTaskByID(91006)
	if s1 := pvpTaskOf(c, 91001); s1 == nil || !s1.done {
		t.Fatalf("91001 (centre cannon) not done after a centre kill: %+v", s1)
	}
	if s6 := pvpTaskOf(c, 91006); s6 == nil || s6.count != 1 || s6.done {
		t.Fatalf("91006 (3 cannons) = %+v after one kill, want count 1 not done", s6)
	}
	money1 := heroMoney(t, s, uid)
	if money1 != money0+t1.Money {
		t.Errorf("money after 91001 done = %d, want %d (+%d)", money1, money0+t1.Money, t1.Money)
	}

	// A second centre-cannon kill must NOT re-pay 91001 (already done); it advances 91006 to 2.
	if len(centre) > 1 {
		credit(centre[1])
	} else {
		// fall back to any other enemy cannon
		for _, list := range guns {
			for _, m := range list {
				if m != centre[0] {
					credit(m)
					break
				}
			}
		}
	}
	if money2 := heroMoney(t, s, uid); money2 != money1 {
		t.Errorf("money moved on a re-kill of a done task: %d -> %d", money1, money2)
	}
	if s6 := pvpTaskOf(c, 91006); s6 == nil || s6.count != 2 {
		t.Fatalf("91006 count = %+v after two cannon kills, want 2", s6)
	}

	// Third distinct enemy cannon completes 91006 and pays its reward.
	var third *mobState
	for _, list := range guns {
		for _, m := range list {
			if m != centre[0] && (len(centre) < 2 || m != centre[1]) {
				third = m
				break
			}
		}
		if third != nil {
			break
		}
	}
	if third == nil {
		t.Fatal("need a third distinct enemy cannon")
	}
	credit(third)
	if s6 := pvpTaskOf(c, 91006); s6 == nil || !s6.done {
		t.Fatalf("91006 not done after 3 cannon kills: %+v", s6)
	}
	if money3 := heroMoney(t, s, uid); money3 != money1+t6.Money {
		t.Errorf("money after 91006 done = %d, want %d (+%d)", money3, money1+t6.Money, t6.Money)
	}

	// Destroying an enemy south-lane barracks completes 91009.
	barr := enemyStructsByLane(inst, gamedata.DotaCreepTower)
	if len(barr[2]) == 0 {
		t.Fatal("no enemy south-lane barracks")
	}
	credit(barr[2][0])
	if s9 := pvpTaskOf(c, 91009); s9 == nil || !s9.done {
		t.Fatalf("91009 (south barracks) not done: %+v", s9)
	}
}

// TestPvpTaskIgnoresNonObjectives: an ally structure, a creep, and an enemy barracks-on-the-wrong-
// lane must not advance the cannon/centre/south tasks.
func TestPvpTaskIgnoresNonObjectives(t *testing.T) {
	s, c, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	bindHero(t, s, c, "ignore@test.io")
	c.lock()
	s.assignPvpTasksLocked(c)
	c.unlock()

	credit := func(m *mobState) {
		c.lock()
		s.creditPvpStructureKillLocked(inst, m)
		c.unlock()
	}

	// An ALLY (player-side) cannon must never credit.
	var allyGun *mobState
	for _, m := range inst.mobs {
		if m.structure && m.team == dotaPlayerTeam && m.dotaRole == gamedata.DotaGun {
			allyGun = m
			break
		}
	}
	if allyGun == nil {
		t.Fatal("no ally cannon")
	}
	credit(allyGun)
	if s6 := pvpTaskOf(c, 91006); s6 == nil || s6.count != 0 {
		t.Errorf("ally cannon advanced 91006: %+v", s6)
	}

	// A plain creep (non-structure) must not credit any structure task.
	creep := &mobState{id: 9001, team: dotaEnemyTeam, structure: false}
	credit(creep)
	if s6 := pvpTaskOf(c, 91006); s6 == nil || s6.count != 0 {
		t.Errorf("creep advanced 91006: %+v", s6)
	}

	// An enemy NON-south barracks must not advance the south-lane task 91009.
	barr := enemyStructsByLane(inst, gamedata.DotaCreepTower)
	for lane, list := range barr {
		if lane == 2 || len(list) == 0 {
			continue
		}
		credit(list[0])
	}
	if s9 := pvpTaskOf(c, 91009); s9 == nil || s9.done || s9.count != 0 {
		t.Errorf("non-south barracks advanced 91009 (south): %+v", s9)
	}
}

// TestPvpTaskCreditsOnCreepKill is the review regression: a structure destroyed by the DOTA
// unit-vs-unit path (a friendly creep/tower landing the final blow, dotaDamageLocked) must credit
// the task exactly like a player-landed kill -- otherwise a team objective met by a creep would
// strand the task and a timed one would falsely FAIL.
func TestPvpTaskCreditsOnCreepKill(t *testing.T) {
	s, c, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	uid := bindHero(t, s, c, "creep@test.io")
	c.lock()
	s.assignPvpTasksLocked(c)
	c.unlock()

	guns := enemyStructsByLane(inst, gamedata.DotaGun)
	if len(guns[1]) == 0 {
		t.Fatal("no enemy centre-lane cannon")
	}
	centreGun := guns[1][0]
	money0 := heroMoney(t, s, uid)
	t1, _ := gamedata.PvpTaskByID(91001)

	// A friendly creep (objID 7777, no owning conn) lands a lethal blow on the enemy centre cannon
	// through the DOTA sim path -- the branch that previously bypassed task credit.
	now := float64(s.battleTime())
	c.lock()
	s.dotaDamageLocked(c, centreGun, 1e9, 7777, now) // overwhelming, past armor mitigation
	c.unlock()
	if !centreGun.dead {
		t.Fatal("cannon should be dead after a lethal creep hit")
	}
	if s1 := pvpTaskOf(c, 91001); s1 == nil || !s1.done {
		t.Fatalf("91001 not completed by a creep-landed cannon kill: %+v", s1)
	}
	if money1 := heroMoney(t, s, uid); money1 != money0+t1.Money {
		t.Errorf("reward not paid on creep kill: money %d, want %d", money1, money0+t1.Money)
	}
	// The timer sweep must NOT fail a task the creep already completed.
	c.lock()
	s.sweepPvpTaskTimersLocked(c, 1e12)
	c.unlock()
	if s1 := pvpTaskOf(c, 91001); s1.failed {
		t.Error("a creep-completed task was later flipped to FAILED by the sweep")
	}
}

// TestPvpTaskTimerFails: the sweep fails a timed task past its deadline (client drops it), but
// never touches a done or untimed task.
func TestPvpTaskTimerFails(t *testing.T) {
	s, c, _, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	bindHero(t, s, c, "timer@test.io")
	c.lock()
	s.assignPvpTasksLocked(c)
	c.unlock()

	timed := pvpTaskOf(c, 91001) // centre cannon, 600s
	if timed == nil || timed.endTime <= 0 {
		t.Fatal("91001 should be a timed task")
	}
	untimed := pvpTaskOf(c, 91006) // any cannon, no timer
	if untimed == nil || untimed.endTime != 0 {
		t.Fatal("91006 should be untimed")
	}

	// Sweep with a clock past the deadline: the timed task fails, the untimed one is untouched.
	c.lock()
	s.sweepPvpTaskTimersLocked(c, timed.endTime+1)
	c.unlock()
	if !timed.failed {
		t.Error("timed task not failed past its deadline")
	}
	if untimed.failed {
		t.Error("untimed task was failed by the sweep")
	}

	// A done task is never re-failed even if its (already-past) deadline is swept.
	untimed.done = true
	c.lock()
	s.sweepPvpTaskTimersLocked(c, 1e12)
	c.unlock()
	if untimed.failed {
		t.Error("a done task was flipped to failed")
	}
}

// TestPvpTaskShturmOnly: outside «Штурм» (a Hunt conn with no dota instance) assignment and the
// kill hook are inert.
func TestPvpTaskShturmOnly(t *testing.T) {
	s, c, _, _, _ := newNavConn(t) // a Hunt (map_4_0) member, inst.dota == nil
	bindHero(t, s, c, "hunt@test.io")

	c.mvMu.Lock()
	s.assignPvpTasksLocked(c)
	c.mvMu.Unlock()
	if c.huntState.pvp != nil {
		t.Error("PvP tasks assigned outside «Штурм»")
	}

	// The kill hook must no-op with no dota instance.
	ms := &mobState{id: dotaStructIDBase + 5, team: dotaEnemyTeam, structure: true, dotaRole: gamedata.DotaGun}
	c.mvMu.Lock()
	s.creditPvpStructureKillLocked(c.inst, ms)
	c.mvMu.Unlock()
	if c.huntState.pvp != nil {
		t.Error("kill hook created a pvp run outside «Штурм»")
	}
}
