package ctrlserver

import (
	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
)

// Party / group. All commands use object="user". The HTTP request is the action
// (invite, answer, leave, kick, change-leader); the membership changes are pushed to
// the affected players over the MPD channel (base key "user|<name>", payload under
// "arguments"; the client appends "_mpd"). Player id == User id throughout.
//
// JoinAnswer enum delivered in the *_answer pushes.
const (
	joinNoPlace  int32 = 1
	joinAdded    int32 = 2
	joinDeclined int32 = 3
)

// handleGroupInvite: user|join_from_group_request {user_id(invitee), referred}. A
// leader invites a player. The HTTP reply reports whether the invitee is already in
// a party; if free, the invite is pushed to them so their client shows a YesNoDialog.
func (s *Server) handleGroupInvite(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	inviteeID, _ := req.Params.GetInt("user_id")
	inGroup := u != nil && s.Store.InGroup(inviteeID)
	resp.Add("user", "join_from_group_request", amf.NewArray().Set("user_in_group", inGroup))
	if u == nil || inGroup || inviteeID == u.ID || s.MPD == nil {
		return
	}
	s.MPD.Push(inviteeID, "user|join_from_group_request", amf.NewArray().
		Set("user_id", u.ID).Set("nick", u.Username))
}

// handleGroupInviteAnswer: user|join_from_group_answer {user_id(leaderId), answer}.
// The invitee accepts/declines. On accept, form/extend the leader's party and tell
// the prior members; either way tell the leader the outcome.
func (s *Server) handleGroupInviteAnswer(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	resp.Ack("user", "join_from_group_answer") // invitee re-requests group_list on the reply
	if u == nil || s.MPD == nil {
		return
	}
	leaderID, _ := req.Params.GetInt("user_id")
	if !req.Params.BoolOr("answer", false) {
		s.MPD.Push(leaderID, "user|join_from_group_answer", amf.NewArray().Set("answer", joinDeclined))
		return
	}
	prior, ok := s.Store.JoinGroup(leaderID, u.ID)
	if !ok {
		s.MPD.Push(leaderID, "user|join_from_group_answer", amf.NewArray().Set("answer", joinNoPlace))
		return
	}
	s.pushJoined(prior, u.ID)
	s.MPD.Push(leaderID, "user|join_from_group_answer", amf.NewArray().Set("answer", joinAdded))
}

// handleGroupJoinRequest: user|join_to_group_request {user_id(a member of target)}.
// A player asks to join the party that user_id is in; the request is pushed to that
// party's leader to accept.
func (s *Server) handleGroupJoinRequest(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	memberID, _ := req.Params.GetInt("user_id")
	g := s.Store.GroupOf(memberID)
	resp.Add("user", "join_to_group_request", amf.NewArray().Set("not_in_group", g == nil))
	if u == nil || g == nil || s.MPD == nil {
		return
	}
	s.MPD.Push(g.Leader, "user|join_to_group_request", amf.NewArray().
		Set("user_id", u.ID).Set("nick", u.Username))
}

// handleGroupJoinAnswer: user|join_to_group_answer {user_id(requester), answer}. A
// member accepts/declines an ask-to-join; on accept the requester joins that party.
func (s *Server) handleGroupJoinAnswer(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	resp.Ack("user", "join_to_group_answer")
	if u == nil || s.MPD == nil {
		return
	}
	requesterID, _ := req.Params.GetInt("user_id")
	if !req.Params.BoolOr("answer", false) {
		s.MPD.Push(requesterID, "user|join_to_group_answer", amf.NewArray().Set("answer", joinDeclined))
		return
	}
	prior, ok := s.Store.JoinGroup(u.ID, requesterID)
	if !ok {
		s.MPD.Push(requesterID, "user|join_to_group_answer", amf.NewArray().Set("answer", joinNoPlace))
		return
	}
	s.pushJoined(prior, requesterID)
	s.MPD.Push(requesterID, "user|join_to_group_answer", amf.NewArray().Set("answer", joinAdded))
}

// handleGroupLeave: user|leave_group. A member (or leader) leaves; remaining members
// are notified, and if the party collapses to one it is disbanded.
func (s *Server) handleGroupLeave(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	resp.Ack("user", "leave_group")
	if u == nil || s.MPD == nil {
		return
	}
	remaining, _, disbanded, ok := s.Store.LeaveGroup(u.ID)
	if !ok {
		return
	}
	s.pushLeft(remaining, u.ID, disbanded)
}

// handleGroupKick: user|remove_from_group {user_id(target)}. Only the leader kicks.
func (s *Server) handleGroupKick(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	resp.Ack("user", "remove_from_group")
	if u == nil || s.MPD == nil {
		return
	}
	target, _ := req.Params.GetInt("user_id")
	if g := s.Store.GroupOf(u.ID); g == nil || g.Leader != u.ID || target == u.ID {
		return
	}
	remaining, _, disbanded, ok := s.Store.LeaveGroup(target)
	if !ok {
		return
	}
	// The kicked player is told explicitly (distinct push from a voluntary leave).
	s.MPD.Push(target, "user|remove_from_group", amf.NewArray().Set("user_id", u.ID))
	s.pushLeft(remaining, target, disbanded)
}

// handleGroupChangeLeader: user|change_leader {user_id(new leader)}.
func (s *Server) handleGroupChangeLeader(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	resp.Ack("user", "change_leader")
	if u == nil || s.MPD == nil {
		return
	}
	newLeader, _ := req.Params.GetInt("user_id")
	members, ok := s.Store.ChangeLeader(u.ID, newLeader)
	if !ok {
		return
	}
	for _, m := range members {
		s.MPD.Push(m, "user|group_leader_changed", amf.NewArray().Set("leader", newLeader))
	}
}

// pushJoined tells every prior member that newID joined the party.
func (s *Server) pushJoined(prior []int32, newID int32) {
	nick := ""
	if mu, ok := s.Store.ByID(newID); ok {
		nick = mu.Username
	}
	for _, m := range prior {
		s.MPD.Push(m, "user|joined_to_group", amf.NewArray().Set("user", newID).Set("nick", nick))
	}
}

// pushLeft tells the remaining members that leaverID left; a disbanded party gets a
// group_deleted to its lone remaining member instead.
func (s *Server) pushLeft(remaining []int32, leaverID int32, disbanded bool) {
	for _, m := range remaining {
		if disbanded {
			s.MPD.Push(m, "user|group_deleted", amf.NewArray())
		} else {
			s.MPD.Push(m, "user|removed_from_group", amf.NewArray().Set("user", leaverID))
		}
	}
}

// NotifyOnline / NotifyOffline push a user's presence to their party co-members when
// their MPD socket connects / disconnects. Wired as the Hub's OnConnect/OnDisconnect.
func (s *Server) NotifyOnline(uid int32)  { s.notifyPresence(uid, "user|online") }
func (s *Server) NotifyOffline(uid int32) { s.notifyPresence(uid, "user|offline") }

func (s *Server) notifyPresence(uid int32, key string) {
	if s.MPD == nil {
		return
	}
	g := s.Store.GroupOf(uid)
	if g == nil {
		return
	}
	for _, m := range g.Members {
		if m != uid {
			s.MPD.Push(m, key, amf.NewArray().Set("user_id", uid))
		}
	}
}
