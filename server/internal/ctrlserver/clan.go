package ctrlserver

import (
	"strconv"

	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
	"tanatserver/internal/session"
)

// Clan handlers. obj="clan". Direct command responses put their fields at the response
// ROOT (matching CreateArg/InfoArg/RemoveUserArg parsers); failures set the client's
// Clan.ErrorCode via resp.Fail. The four async events (info_mpd/remove_user_mpd/
// invite_mpd/invite_answer_mpd) are pushed over MPD -- Hub.Push nests the payload under
// "arguments" and the client appends the "_mpd" suffix, so the base key is passed here.
//
// Server-authoritative: the client pre-checks money/name/permissions but the store
// re-enforces everything (see session/clan.go); handlers only marshal wire shapes.

func (s *Server) handleClanCreate(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil {
		resp.Fail("clan", "create", session.ClanErrSystem)
		return
	}
	id, code := s.Store.CreateClan(u.ID, req.Params.StringOr("name", ""), req.Params.StringOr("tag", ""))
	if code != session.ClanOK {
		resp.Fail("clan", "create", code)
		return
	}
	resp.Add("clan", "create", amf.NewArray().Set("id", id))
	s.broadcastClanTag(u.ID) // reflect the new [tag] to nearby players
}

func (s *Server) handleClanInfo(req ctrlproto.Request, resp *ctrlproto.Response) {
	var view *session.ClanView
	var ok bool
	if id, has := clanParamInt(req.Params, "clan_id"); has && id > 0 {
		view, ok = s.Store.ClanByID(id)
	} else if uid, has := clanParamInt(req.Params, "user_id"); has && uid > 0 {
		view, ok = s.Store.ClanOfUser(uid)
	} else if tag := req.Params.StringOr("tag", ""); tag != "" {
		view, ok = s.Store.ClanByTag(tag)
	}
	if !ok || view == nil {
		resp.Fail("clan", "info", session.ClanErrWrongParams)
		return
	}
	// users is a DENSE array of member records (InfoArgParser iterates .Dense).
	users := amf.NewArray()
	for _, m := range view.Members {
		users.Add(amf.NewArray().
			Set("id", m.UserID).
			Set("nick", m.Nick).
			Set("location", s.memberLocation(m.UserID)).
			Set("role", m.Role))
	}
	resp.Add("clan", "info", amf.NewArray().
		Set("id", view.ID).
		Set("tag", view.Tag).
		Set("name", view.Name).
		Set("level", view.Level).
		Set("rating", view.Rating).
		Set("users", users))
}

func (s *Server) handleClanRemoveUser(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil {
		resp.Fail("clan", "remove_user", session.ClanErrYouNotMember)
		return
	}
	target, _ := clanParamInt(req.Params, "user_id")
	code := s.Store.RemoveClanMember(u.ID, target)
	if code != session.ClanOK {
		resp.Fail("clan", "remove_user", code)
		return
	}
	resp.Add("clan", "remove_user", amf.NewArray().Set("user_id", target))
	if s.MPD != nil {
		// type=1 REMOVE_USER: tells the removed player (kick or self-leave) to clear clan state.
		s.MPD.Push(target, "clan|remove_user", amf.NewArray().
			Set("user_id", target).Set("type", session.ClanRemoveReasonUser))
	}
	s.broadcastClanTag(target) // target is now clanless -> clears the [tag] for observers
}

func (s *Server) handleClanRemove(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil {
		resp.Fail("clan", "remove", session.ClanErrYouNotMember)
		return
	}
	code, members := s.Store.DisbandClan(u.ID)
	if code != session.ClanOK {
		resp.Fail("clan", "remove", code)
		return
	}
	resp.Ack("clan", "remove") // the disbanding HEAD clears state off this ack (OnClanRemoved)
	for _, m := range members {
		if s.MPD != nil && m != u.ID {
			// type=2 REMOVE_CLAN: tells each other member the clan is gone.
			s.MPD.Push(m, "clan|remove_user", amf.NewArray().
				Set("user_id", m).Set("type", session.ClanRemoveReasonClan))
		}
		s.broadcastClanTag(m)
	}
}

func (s *Server) handleClanChangeRole(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil {
		resp.Fail("clan", "change_role", session.ClanErrYouNotMember)
		return
	}
	rolesArr, ok := req.Params.GetArray("roles")
	if !ok {
		resp.Fail("clan", "change_role", session.ClanErrWrongParams)
		return
	}
	roles := map[int32]int32{}
	for k, v := range rolesArr.Assoc {
		uid, err := strconv.Atoi(k)
		if err != nil {
			continue
		}
		roles[int32(uid)] = amfInt(v)
	}
	if len(roles) == 0 {
		resp.Fail("clan", "change_role", session.ClanErrWrongParams)
		return
	}
	if code := s.Store.ChangeClanRoles(u.ID, roles); code != session.ClanOK {
		resp.Fail("clan", "change_role", code)
		return
	}
	resp.Ack("clan", "change_role")
}

func (s *Server) handleClanInvite(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil {
		resp.Fail("clan", "invite", session.ClanErrYouNotMember)
		return
	}
	target, found := s.Store.ByUsername(req.Params.StringOr("nick", ""))
	if !found || target.ID == u.ID {
		resp.Fail("clan", "invite", session.ClanErrWrongParams)
		return
	}
	if s.MPD == nil || !s.MPD.Online(target.ID) {
		resp.Fail("clan", "invite", session.ClanErrUserOffline)
		return
	}
	clanName, _, code := s.Store.RecordClanInvite(u.ID, target.ID)
	if code != session.ClanOK {
		resp.Fail("clan", "invite", code)
		return
	}
	resp.Ack("clan", "invite")
	// Push the invitation to the invitee (nick=inviter, clan_name) -> Yes/No dialog.
	s.MPD.Push(target.ID, "clan|invite", amf.NewArray().
		Set("nick", u.Username).Set("clan_name", clanName))
}

func (s *Server) handleClanInviteAnswer(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil {
		resp.Fail("clan", "invite_answer", session.ClanErrSystem)
		return
	}
	accept := req.Params.BoolOr("answer", false)
	_, inviterID, joined, code := s.Store.AnswerClanInvite(u.ID, accept)
	if code != session.ClanOK {
		resp.Fail("clan", "invite_answer", code)
		return
	}
	resp.Ack("clan", "invite_answer")
	if s.MPD != nil && inviterID != 0 {
		// Tell the inviter the outcome; on accept their client refreshes clan|info.
		s.MPD.Push(inviterID, "clan|invite_answer", amf.NewArray().
			Set("nick", u.Username).Set("answer", joined))
	}
	if joined {
		s.broadcastClanTag(u.ID)
	}
}

// broadcastClanTag pushes clan|info_mpd {id, user_id, tag} to every OTHER occupant of the
// affected player's lobby square so the floating [tag] over their avatar updates. id is
// the client's -1 "no clan" sentinel when the player is now clanless.
func (s *Server) broadcastClanTag(userID int32) {
	if s.MPD == nil {
		return
	}
	id, tag := s.Store.HeroClanInfo(userID)
	if id == 0 {
		id = -1
	}
	for _, m := range s.MPD.AreaMembers(s.MPD.AreaOf(userID)) {
		if m == userID {
			continue
		}
		s.MPD.Push(m, "clan|info", amf.NewArray().
			Set("id", id).Set("user_id", userID).Set("tag", tag))
	}
}

// clanInfoArray builds the {id, tag} clan_info sub-array the client's HeroGameInfo /
// hero-data parsers read; id is -1 (the client's "no clan" sentinel) when clanless.
func (s *Server) clanInfoArray(heroID int32) *amf.MixedArray {
	id, tag := s.Store.HeroClanInfo(heroID)
	if id == 0 {
		id = -1
	}
	return amf.NewArray().Set("id", id).Set("tag", tag)
}

// memberLocation returns the clan-roster "Location" cell: a zone locale key for an online
// member (the client's only online indicator in the roster), or "" so it shows Offline.
// The exact per-zone locale key is a deferred nicety; any online member shows the square.
func (s *Server) memberLocation(uid int32) string {
	if s.MPD != nil && s.MPD.Online(uid) {
		return "Central_Square_Text"
	}
	return ""
}

// clanParamInt reads an int param that the client may send as an int OR a decimal string
// (clan|info sends clan_id/user_id/tag as strings).
func clanParamInt(p *amf.MixedArray, key string) (int32, bool) {
	if v, ok := p.GetInt(key); ok {
		return v, true
	}
	if str, ok := p.GetString(key); ok && str != "" {
		if n, err := strconv.Atoi(str); err == nil {
			return int32(n), true
		}
	}
	return 0, false
}

// amfInt coerces a decoded AMF scalar (int32 or float64) to int32.
func amfInt(v interface{}) int32 {
	switch n := v.(type) {
	case int32:
		return n
	case float64:
		return int32(n)
	case int:
		return int32(n)
	}
	return 0
}
