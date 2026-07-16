package ctrlserver

import (
	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
)

// Friends / ignore (the client's white/black list, PlayersListController + UsersListSender).
// The client:
//   - fetches the lists with user|get_bw_list {type} -> {white:[friends], black:[ignore]}
//   - adds with user|add_to_list {user_id, type}; type=1 friend, type=2 ignore
//   - accepts a friend request with user|friend_answer {user_id(requester), answer}
//   - removes with user|remove_from_list {user_id, type}
//   - searches with user|find {tag, nick} -> {result:[rows]}
// A friend add to an online player is a REQUEST: the target gets a user|friend_request
// MPD push (YesNoDialog); on accept both sides become friends and get a user|add_to_list
// push to refresh + show the "agreed/declined" note (PlayersListController.OnAddToListMpd).
//
// Wire ListType (TanatKernel.ListType): FRIEND=1, IGNORE=2.
const (
	listFriend int32 = 1
	listIgnore int32 = 2
)

// handleGetBwList answers user|get_bw_list -> {white:[friends], black:[ignore]}. Both
// lists are always returned regardless of the requested type; the client reads the one
// it needs (BwListArgParser only looks at the keys present).
func (s *Server) handleGetBwList(req ctrlproto.Request, resp *ctrlproto.Response) {
	white, black := amf.NewArray(), amf.NewArray()
	if u := s.userFromSession(req); u != nil {
		for _, fid := range s.Store.FriendIDs(u.ID) {
			white.Add(s.shortUserRow(fid))
		}
		for _, iid := range s.Store.IgnoreIDs(u.ID) {
			black.Add(s.shortUserRow(iid))
		}
	}
	resp.Add("user", "get_bw_list", amf.NewArray().Set("white", white).Set("black", black))
}

// shortUserRow builds the {id, level, nick, rating, tag, clan_id, online} row that
// BwListArgParser.ParseUser reads (online is only honored when >5 assoc keys are present,
// which this row satisfies). online: 0 OFFLINE, 1 CS (ShortUserInfo.Status).
func (s *Server) shortUserRow(uid int32) *amf.MixedArray {
	nick := ""
	var level int32
	if mu, ok := s.Store.ByID(uid); ok {
		nick = mu.Username
		if mu.Hero != nil {
			level = mu.Hero.Level
		}
	}
	online := int32(0)
	if s.MPD != nil && s.MPD.Online(uid) {
		online = 1
	}
	return amf.NewArray().
		Set("id", uid).
		Set("level", level).
		Set("nick", nick).
		Set("rating", int32(0)).
		Set("tag", "").
		Set("clan_id", int32(0)).
		Set("online", online)
}

// handleAddToList: user|add_to_list {user_id, type}. Friend adds are request/accept;
// ignore adds are immediate.
func (s *Server) handleAddToList(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	resp.Ack("user", "add_to_list")
	if u == nil {
		return
	}
	target, _ := req.Params.GetInt("user_id")
	if target == u.ID {
		return
	}
	if req.Params.IntOr("type", listFriend) == listIgnore {
		s.Store.AddIgnore(u.ID, target)
		s.pushListRefresh(u.ID, s.nickOf(target), true, listIgnore)
		return
	}
	nowFriends, ok := s.Store.AddFriendRequest(u.ID, target)
	if !ok || s.MPD == nil {
		return
	}
	if nowFriends {
		// The two had requested each other -> instant mutual friends; refresh both.
		s.pushListRefresh(target, u.Username, true, listFriend)
		s.pushListRefresh(u.ID, s.nickOf(target), true, listFriend)
		return
	}
	// Prompt the target with a friend request (YesNoDialog).
	s.MPD.Push(target, "user|friend_request", amf.NewArray().
		Set("user_id", u.ID).Set("nick", u.Username))
}

// handleFriendAnswer: user|friend_answer {user_id(requester), answer}. On accept the
// two become friends; the requester is told the outcome so their list refreshes.
func (s *Server) handleFriendAnswer(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	resp.Ack("user", "friend_answer")
	if u == nil {
		return
	}
	requester, _ := req.Params.GetInt("user_id")
	if req.Params.BoolOr("answer", false) {
		s.Store.AcceptFriendRequest(u.ID, requester)
		s.pushListRefresh(requester, u.Username, true, listFriend)
		return
	}
	s.Store.DeclineFriendRequest(u.ID, requester)
	s.pushListRefresh(requester, u.Username, false, listFriend)
}

// handleRemoveFromList: user|remove_from_list {user_id, type}. The response echoes type
// (RemoveFromListArgParser reads it off the packet) so the client refetches that list.
func (s *Server) handleRemoveFromList(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	target, _ := req.Params.GetInt("user_id")
	typ := req.Params.IntOr("type", listFriend)
	resp.Add("user", "remove_from_list", amf.NewArray().Set("type", typ))
	if u == nil {
		return
	}
	if typ == listIgnore {
		s.Store.RemoveIgnore(u.ID, target)
	} else {
		s.Store.RemoveFriend(u.ID, target)
	}
}

// handleUserFind: user|find {tag, nick} -> {result:[rows]}. Resolves an exact nick.
func (s *Server) handleUserFind(req ctrlproto.Request, resp *ctrlproto.Response) {
	result := amf.NewArray()
	if nick := req.Params.StringOr("nick", ""); nick != "" {
		if mu, ok := s.Store.ByUsername(nick); ok {
			result.Add(s.shortUserRow(mu.ID))
		}
	}
	resp.Add("user", "find", amf.NewArray().Set("result", result))
}

// pushListRefresh sends a user|add_to_list MPD push (AddToListMpdArgParser reads
// {nick, answer, type}); the client refetches the list and, for a friend answer, shows
// the agreed/declined note. No-op if the target is offline.
func (s *Server) pushListRefresh(target int32, nick string, answer bool, typ int32) {
	if s.MPD == nil {
		return
	}
	s.MPD.Push(target, "user|add_to_list", amf.NewArray().
		Set("nick", nick).Set("answer", answer).Set("type", typ))
}

// nickOf resolves a user id to its display nick ("" if unknown).
func (s *Server) nickOf(uid int32) string {
	if mu, ok := s.Store.ByID(uid); ok {
		return mu.Username
	}
	return ""
}
