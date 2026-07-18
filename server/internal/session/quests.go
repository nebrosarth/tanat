package session

import (
	"time"

	"tanatserver/internal/gamedata"
)

// Per-hero PvE quest state -- the store side of "квесты". A quest is accepted in the city
// (AcceptQuest -> IN_PROGRESS), advanced by killing mobs on its Hunt map (AddQuestKill bumps
// Progress, flipping to DONE at the objective count), and turned in at the NPC (CompleteQuest
// pays out via the Ctrl handler and CLOSES a one-time quest or parks a REPLAY quest on cooldown).
// State persists with the Hero (JSON), so a quest half-finished before logout resumes exactly.
// The mechanical thresholds (objective count, map, repeatable/cooldown) come from the gamedata
// catalog; this file owns only the mutable per-hero state and keeps every transition atomic
// under the store lock so concurrent kills can never double-count or lose a completion.

// QuestState is one hero's progress on one quest. Status uses gamedata.QuestStatus*; Progress is
// the current kill count toward the quest's objective; CooldownUntil is the unix time a REPLAY
// quest becomes re-offerable (0 = not on cooldown).
type QuestState struct {
	QuestID       int32
	Status        int32
	Progress      int32
	CooldownUntil int64
}

// nowUnix is overridable in tests; production uses the wall clock for REPLAY cooldowns.
var nowUnix = func() int64 { return time.Now().Unix() }

// findQuestLocked returns a pointer to the hero's state for questID (or nil). Caller holds s.mu.
func findQuestLocked(h *Hero, questID int32) *QuestState {
	for i := range h.Quests {
		if h.Quests[i].QuestID == questID {
			return &h.Quests[i]
		}
	}
	return nil
}

// HeroQuests returns a snapshot of the hero's quest states, DROPPING any REPLAY quest whose
// cooldown has elapsed (so it reads as fresh/available again). Returns nil if no account/hero.
func (s *Store) HeroQuests(userID int32) []QuestState {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, found := s.usersByID[userID]
	if !found || u.Hero == nil {
		return nil
	}
	now := nowUnix()
	out := make([]QuestState, 0, len(u.Hero.Quests))
	for _, qs := range u.Hero.Quests {
		if qs.Status == gamedata.QuestStatusWaitCooldown && qs.CooldownUntil > 0 && now >= qs.CooldownUntil {
			continue // cooldown finished -> omit, so the NPC re-offers it
		}
		out = append(out, qs)
	}
	return out
}

// AcceptQuest starts a quest for a hero. ok=false with no change when the quest is unknown, is
// already active (IN_PROGRESS/DONE), is a CLOSED one-time quest, or is a REPLAY quest still on
// cooldown. A REPLAY quest whose cooldown has elapsed is restarted fresh. Returns the new state.
func (s *Store) AcceptQuest(userID, questID int32) (QuestState, bool) {
	q, known := gamedata.QuestByID(questID)
	if !known {
		return QuestState{}, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	u, found := s.usersByID[userID]
	if !found || u.Hero == nil {
		return QuestState{}, false
	}
	h := u.Hero
	now := nowUnix()
	if cur := findQuestLocked(h, questID); cur != nil {
		switch cur.Status {
		case gamedata.QuestStatusInProgress, gamedata.QuestStatusDone:
			return *cur, false // already active
		case gamedata.QuestStatusClosed:
			return *cur, false // one-time quest, done for good
		case gamedata.QuestStatusWaitCooldown:
			if cur.CooldownUntil > 0 && now < cur.CooldownUntil {
				return *cur, false // still cooling down
			}
			// cooldown elapsed -> restart in place
			cur.Status = gamedata.QuestStatusInProgress
			cur.Progress = 0
			cur.CooldownUntil = 0
			s.saveLocked()
			return *cur, true
		}
	}
	qs := QuestState{QuestID: questID, Status: gamedata.QuestStatusInProgress}
	_ = q
	h.Quests = append(h.Quests, qs)
	s.saveLocked()
	return qs, true
}

// CancelQuest abandons an ACTIVE quest, removing its state so it can be taken again. It refuses
// to remove a CLOSED one-time quest -- that CLOSED record is the ONLY thing stopping AcceptQuest
// from re-arming an already-rewarded quest, so deleting it would turn every one-time quest into
// an unlimited gold/exp faucet (cancel -> accept -> re-farm). It likewise refuses a REPLAY quest
// still on cooldown, whose deletion would let the next accept skip the repeat throttle. Only
// IN_PROGRESS/DONE (pre-reward) and an already-elapsed cooldown record may be dropped. ok=false
// with no change otherwise.
func (s *Store) CancelQuest(userID, questID int32) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, found := s.usersByID[userID]
	if !found || u.Hero == nil {
		return false
	}
	h := u.Hero
	now := nowUnix()
	for i := range h.Quests {
		if h.Quests[i].QuestID != questID {
			continue
		}
		st := h.Quests[i].Status
		if st == gamedata.QuestStatusClosed ||
			(st == gamedata.QuestStatusWaitCooldown && h.Quests[i].CooldownUntil > 0 && now < h.Quests[i].CooldownUntil) {
			return false // a completed/cooling quest is not abandonable (would re-arm its reward)
		}
		h.Quests = append(h.Quests[:i], h.Quests[i+1:]...)
		s.saveLocked()
		return true
	}
	return false
}

// QuestReward is the outcome of a successful turn-in: the quest plus the hero's NEW balances and
// progression, so the Ctrl handler can push user|money / hero info without re-reading the store.
type QuestReward struct {
	Quest    gamedata.Quest
	Money    int32
	Diamonds int32
	Level    int32
	Exp      int32
	NextExp  int32
}

// CompleteQuest turns in a quest whose status is DONE: it transitions the quest (CLOSED for a
// one-time quest, WAIT_COOLDOWN+cooldown for a REPLAY) AND credits the gold+exp bounty (with
// level-ups) in ONE critical section committed by a single saveLocked, so the state change and
// the payout persist together -- a crash can't leave a quest CLOSED with its reward uncredited.
// It is the single-fire gate: ok=true happens exactly once per completion (a concurrent second
// call sees a non-DONE status and returns false with no payout), so the reward is paid once.
func (s *Store) CompleteQuest(userID, questID int32) (QuestReward, bool) {
	q, known := gamedata.QuestByID(questID)
	if !known {
		return QuestReward{}, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	u, found := s.usersByID[userID]
	if !found || u.Hero == nil {
		return QuestReward{Quest: q}, false
	}
	h := u.Hero
	cur := findQuestLocked(h, questID)
	if cur == nil || cur.Status != gamedata.QuestStatusDone {
		return QuestReward{Quest: q}, false
	}
	if q.Repeatable() {
		cur.Status = gamedata.QuestStatusWaitCooldown
		cur.Progress = 0
		cur.CooldownUntil = nowUnix() + int64(q.Cooldown)
	} else {
		cur.Status = gamedata.QuestStatusClosed
	}
	// Credit the bounty in the same locked commit as the transition (mirrors AddHeroMoney /
	// AddHeroExp, but atomic with the CLOSED/WAIT_COOLDOWN write).
	if q.Money > 0 {
		h.Money += q.Money
		if h.Money < 0 {
			h.Money = 0
		}
	}
	if q.Exp > 0 {
		h.Exp += q.Exp
		for h.NextExp > 0 && h.Exp >= h.NextExp {
			h.Exp -= h.NextExp
			h.Level++
			h.NextExp = heroExpNextLevel(h.Level)
		}
	}
	s.saveLocked()
	return QuestReward{Quest: q, Money: h.Money, Diamonds: h.DiamondMoney, Level: h.Level, Exp: h.Exp, NextExp: h.NextExp}, true
}

// AddQuestKill credits one Hunt kill on mapID toward every IN_PROGRESS quest the hero holds for
// that map, bumping Progress and flipping to DONE at the objective count. Returns the states that
// actually changed (for the quest|update_mpd push); nil if nothing advanced. All of it happens
// under the store lock, so simultaneous kills can neither race the threshold nor lose a tick.
func (s *Store) AddQuestKill(userID, mapID int32) []QuestState {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, found := s.usersByID[userID]
	if !found || u.Hero == nil {
		return nil
	}
	var changed []QuestState
	dirty := false
	for i := range u.Hero.Quests {
		qs := &u.Hero.Quests[i]
		if qs.Status != gamedata.QuestStatusInProgress {
			continue
		}
		q, ok := gamedata.QuestByID(qs.QuestID)
		if !ok || q.MapID != mapID {
			continue
		}
		if qs.Progress >= q.Count {
			continue // already at objective (awaiting turn-in)
		}
		qs.Progress++
		if qs.Progress >= q.Count {
			qs.Status = gamedata.QuestStatusDone
		}
		changed = append(changed, *qs)
		dirty = true
	}
	if dirty {
		s.saveLocked()
	}
	return changed
}
