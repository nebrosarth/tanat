package session

// Group is a party: a set of member user ids with one leader (who is also a
// member). There is NO explicit "create group" command in the protocol -- a group
// is born the first time an invite is accepted (JoinGroup), and it disbands when it
// collapses to a single member (the client shows no party for one person). All
// members index the SAME *Group via Store.groups, guarded by Store.mu.
type Group struct {
	Leader  int32
	Members []int32 // includes the leader; leader-first join order
}

func (g *Group) copy() *Group {
	m := make([]int32, len(g.Members))
	copy(m, g.Members)
	return &Group{Leader: g.Leader, Members: m}
}

// GroupOf returns a snapshot of the party uid belongs to, or nil if solo.
func (s *Store) GroupOf(uid int32) *Group {
	s.mu.Lock()
	defer s.mu.Unlock()
	if g := s.groups[uid]; g != nil {
		return g.copy()
	}
	return nil
}

// InGroup reports whether uid is currently in a party.
func (s *Store) InGroup(uid int32) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.groups[uid] != nil
}

// JoinGroup adds invitee to the party that owner belongs to, creating it (with owner
// as sole member + leader) on first use. Returns the members that were ALREADY in
// the party before invitee joined -- the set to notify with joined_to_group. ok is
// false if invitee is the same user, is already in a party, or (rare) collides.
func (s *Store) JoinGroup(owner, invitee int32) (prior []int32, ok bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if owner == invitee || s.groups[invitee] != nil {
		return nil, false
	}
	g := s.groups[owner]
	if g == nil {
		g = &Group{Leader: owner, Members: []int32{owner}}
		s.groups[owner] = g
	}
	prior = append([]int32(nil), g.Members...)
	g.Members = append(g.Members, invitee)
	s.groups[invitee] = g
	return prior, true
}

// LeaveGroup removes uid from its party. Returns the members remaining afterward,
// the (possibly promoted) leader, and disbanded=true when the party dropped to one
// member (whom we also evict, so the returned remaining is that lone member and the
// caller should tell them the group is gone). ok is false if uid wasn't in a party.
func (s *Store) LeaveGroup(uid int32) (remaining []int32, leader int32, disbanded, ok bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	g := s.groups[uid]
	if g == nil {
		return nil, 0, false, false
	}
	delete(s.groups, uid)
	filtered := g.Members[:0]
	for _, m := range g.Members {
		if m != uid {
			filtered = append(filtered, m)
		}
	}
	g.Members = filtered
	if g.Leader == uid && len(g.Members) > 0 {
		g.Leader = g.Members[0]
	}
	if len(g.Members) <= 1 {
		rem := append([]int32(nil), g.Members...)
		for _, m := range g.Members {
			delete(s.groups, m)
		}
		g.Members = nil
		return rem, g.Leader, true, true
	}
	return append([]int32(nil), g.Members...), g.Leader, false, true
}

// ChangeLeader makes newLeader the party leader, if uid is the current leader and
// newLeader is a member. Returns the members to notify. ok is false otherwise.
func (s *Store) ChangeLeader(uid, newLeader int32) (members []int32, ok bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	g := s.groups[uid]
	if g == nil || g.Leader != uid {
		return nil, false
	}
	found := false
	for _, m := range g.Members {
		if m == newLeader {
			found = true
			break
		}
	}
	if !found {
		return nil, false
	}
	g.Leader = newLeader
	return append([]int32(nil), g.Members...), true
}
