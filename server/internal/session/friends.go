package session

import "log"

// Friends/ignore lists. A friendship is mutual (both accounts carry each other) and
// persists with the account; ignore is one-directional. Friend REQUESTS are transient
// (in-memory), like party invites: `from` asks `to`, and the pair only becomes friends
// when `to` accepts (or when the two have requested each other -- a mutual add).

// persistPairLocked persists both sides of a mutual friendship change in ONE
// transaction, so a crash/IO-failure can't leave the friendship asymmetric
// (one account listing the other but not vice-versa). Nil users are skipped.
func (s *Store) persistPairLocked(a, b *User) {
	if err := s.persistUsersLocked(a, b); err != nil {
		log.Printf("session: persist friend pair failed: %v", err)
	}
}

// AddFriendRequest records that `from` wants to befriend `to`. If `to` had already
// requested `from`, the pending pair is completed immediately (mutual add) and it
// returns nowFriends=true. Otherwise the request is stored pending and returns false.
// ok=false for a self-request, an unknown user, or an existing friendship.
func (s *Store) AddFriendRequest(from, to int32) (nowFriends, ok bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if from == to {
		return false, false
	}
	fu, tu := s.usersByID[from], s.usersByID[to]
	if fu == nil || tu == nil || containsID(fu.Friends, to) {
		return false, false
	}
	if s.friendReqs[from][to] { // `to` already asked `from` -> mutual
		s.clearReqLocked(from, to)
		s.addFriendPairLocked(fu, tu)
		s.persistPairLocked(fu, tu)
		return true, true
	}
	if s.friendReqs[to] == nil {
		s.friendReqs[to] = map[int32]bool{}
	}
	s.friendReqs[to][from] = true
	return false, true
}

// AcceptFriendRequest makes answerer and requester mutual friends (clearing any pending
// request between them). Lenient: it does not require the pending request to still
// exist, so an accept still works after a server restart dropped it.
func (s *Store) AcceptFriendRequest(answerer, requester int32) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	au, ru := s.usersByID[answerer], s.usersByID[requester]
	if au == nil || ru == nil || answerer == requester {
		return false
	}
	s.clearReqLocked(answerer, requester)
	s.addFriendPairLocked(au, ru)
	s.persistPairLocked(au, ru)
	return true
}

// DeclineFriendRequest drops a pending request from requester to answerer.
func (s *Store) DeclineFriendRequest(answerer, requester int32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.clearReqLocked(answerer, requester)
}

// RemoveFriend drops the mutual friendship between uid and target (both sides).
func (s *Store) RemoveFriend(uid, target int32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	u := s.usersByID[uid]
	t := s.usersByID[target]
	if u != nil {
		u.Friends = removeID(u.Friends, target)
	}
	if t != nil {
		t.Friends = removeID(t.Friends, uid)
	}
	s.persistPairLocked(u, t) // both nil-safe, one transaction
}

// AddIgnore adds target to uid's ignore list (one-directional).
func (s *Store) AddIgnore(uid, target int32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	u := s.usersByID[uid]
	if uid == target || u == nil || containsID(u.Ignores, target) {
		return
	}
	u.Ignores = append(u.Ignores, target)
	s.saveUserLocked(u)
}

// RemoveIgnore drops target from uid's ignore list.
func (s *Store) RemoveIgnore(uid, target int32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if u := s.usersByID[uid]; u != nil {
		u.Ignores = removeID(u.Ignores, target)
		s.saveUserLocked(u)
	}
}

// FriendIDs / IgnoreIDs return copies of uid's social lists.
func (s *Store) FriendIDs(uid int32) []int32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if u := s.usersByID[uid]; u != nil {
		return append([]int32(nil), u.Friends...)
	}
	return nil
}

func (s *Store) IgnoreIDs(uid int32) []int32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if u := s.usersByID[uid]; u != nil {
		return append([]int32(nil), u.Ignores...)
	}
	return nil
}

// AreFriends reports whether a and b are mutual friends.
func (s *Store) AreFriends(a, b int32) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	u := s.usersByID[a]
	return u != nil && containsID(u.Friends, b)
}

// clearReqLocked removes any pending request in either direction between x and y.
func (s *Store) clearReqLocked(x, y int32) {
	if s.friendReqs[x] != nil {
		delete(s.friendReqs[x], y)
	}
	if s.friendReqs[y] != nil {
		delete(s.friendReqs[y], x)
	}
}

// addFriendPairLocked adds each user to the other's friends list (idempotent).
func (s *Store) addFriendPairLocked(a, b *User) {
	if !containsID(a.Friends, b.ID) {
		a.Friends = append(a.Friends, b.ID)
	}
	if !containsID(b.Friends, a.ID) {
		b.Friends = append(b.Friends, a.ID)
	}
}

func containsID(ids []int32, id int32) bool {
	for _, x := range ids {
		if x == id {
			return true
		}
	}
	return false
}

func removeID(ids []int32, id int32) []int32 {
	out := ids[:0]
	for _, x := range ids {
		if x != id {
			out = append(out, x)
		}
	}
	return out
}
