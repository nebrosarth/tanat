package ctrlserver

import (
	"strconv"

	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
	"tanatserver/internal/gamedata"
)

// Castle (clan-siege) SCAFFOLD. obj="castle". This implements the out-of-battle screens
// -- browse castles, view a castle's roster/history, and enroll/desert as a fighter --
// backed by a static registry (gamedata.Castles) plus an in-memory enrollment roster.
//
// The LIVE siege is DEFERRED (per "leave contentious points for later"): the real-time
// battle, scheduling windows, tournament bracket, the pre-battle avatar-draft/ready phase,
// rewards payout, and castle-ownership transfer are all intentionally not implemented. The
// corresponding request commands (select_avatar, ready, desert_battle) simply fall through
// to the dispatch's generic ack, and the bracket (battle_info) returns an empty-but-valid
// stage list, so every castle screen opens without a crash.
//
// Wire-shape notes (from CastleListArgParser / CastleMembersArgParser / ...):
//   - list/info/history/fighters/battle_info RESPONSES put their collection at the ROOT.
//   - id-keyed collections (castles, members, queue, fighters) are ASSOCIATIVE arrays with
//     STRING keys the client int.TryParse's; stages is a DENSE array.
//   - times are RELATIVE: start_time = seconds until the next battle window.

// castleRoster is one castle's in-memory fighter enrollment (the live siege is deferred, so
// this only records who signed up and their assigned group number).
type castleRoster struct {
	nick  map[int32]string // uid -> display nick
	group map[int32]int32  // uid -> assigned fighter group / battle number
}

// castleRosterOf returns castleID's roster, lazily creating the map set. Caller must hold
// castleMu (use the with* helpers below).
func (s *Server) castleRosterLocked(castleID int32) *castleRoster {
	if s.castleRosters == nil {
		s.castleRosters = map[int32]*castleRoster{}
	}
	r := s.castleRosters[castleID]
	if r == nil {
		r = &castleRoster{nick: map[int32]string{}, group: map[int32]int32{}}
		s.castleRosters[castleID] = r
	}
	return r
}

// handleCastleList answers castle|list -> {castles: assoc(id->castle)}. CastleListArgParser
// requires the "castles" key present.
func (s *Server) handleCastleList(req ctrlproto.Request, resp *ctrlproto.Response) {
	castles := amf.NewArray()
	for _, c := range gamedata.Castles() {
		rewards := amf.NewArray().
			Set("money_d", c.Reward.Diamonds).
			Set("money", c.Reward.Money).
			Set("exp", c.Reward.Exp).
			Set("item", c.Reward.Item).
			Set("item_count", c.Reward.ItemCount)
		castles.Set(strconv.Itoa(int(c.ID)), amf.NewArray().
			Set("level_max", c.LevelMax).
			Set("level_min", c.LevelMin).
			Set("name", c.Name).
			Set("owner_name", c.OwnerName).
			Set("owner_id", c.OwnerID).
			Set("fighters_min", c.FightersMin).
			Set("start_time", c.StartInSec).
			Set("rewards", rewards))
	}
	resp.Add("castle", "list", amf.NewArray().Set("castles", castles))
}

// handleCastleInfo answers castle|info -> a MEMBERS payload (CastleMembersArgParser):
// roster + join-eligibility flags. NOTE the response is members, not a CastleInfo.
func (s *Server) handleCastleInfo(req ctrlproto.Request, resp *ctrlproto.Response) {
	castleID, _ := clanParamInt(req.Params, "castle_id")
	c, known := gamedata.CastleByID(castleID)
	if !known {
		resp.Fail("castle", "info", 7015) // WRONG_PARAMETERS
		return
	}
	u := s.userFromSession(req)
	rightLevel := false
	joined := false
	if u != nil && u.Hero != nil {
		rightLevel = u.Hero.Level >= c.LevelMin && u.Hero.Level <= c.LevelMax
		joined = s.castleContains(castleID, u.ID)
	}

	members := amf.NewArray()
	s.castleMu.Lock()
	if r := s.castleRosters[castleID]; r != nil {
		for uid, nick := range r.nick {
			members.Set(strconv.Itoa(int(uid)), nick)
		}
	}
	s.castleMu.Unlock()

	resp.Add("castle", "info", amf.NewArray().
		Set("members", members).
		Set("queue", amf.NewArray()). // no waiting queue in the scaffold
		Set("joined", joined).
		Set("editable", true). // real gate = clan role >= COMMANDER (deferred)
		Set("in_progress", false).
		Set("right_level", rightLevel).
		Set("ban_count", int32(0)))
}

// handleCastleHistory answers castle|history -> {battles: assoc}. No sieges have run yet,
// so this is an empty-but-valid list (real battle history is deferred with the live siege).
func (s *Server) handleCastleHistory(req ctrlproto.Request, resp *ctrlproto.Response) {
	resp.Add("castle", "history", amf.NewArray().Set("battles", amf.NewArray()))
}

// handleCastleFighters answers castle|fighters -> {fighters: assoc(uid->group), can_leave}.
func (s *Server) handleCastleFighters(req ctrlproto.Request, resp *ctrlproto.Response) {
	castleID, _ := clanParamInt(req.Params, "castle_id")
	fighters := amf.NewArray()
	s.castleMu.Lock()
	if r := s.castleRosters[castleID]; r != nil {
		for uid, grp := range r.group {
			fighters.Set(strconv.Itoa(int(uid)), grp)
		}
	}
	s.castleMu.Unlock()
	resp.Add("castle", "fighters", amf.NewArray().
		Set("fighters", fighters).
		Set("can_leave", true))
}

// handleCastleSetFighters answers castle|set_fighters -> bare Ack. This is the enroll action.
// Scaffold policy: a player enrolls ONLY THEMSELVES (multi-fighter assignment by a commander
// is deferred), with the group number they picked for their own slot (default 1).
func (s *Server) handleCastleSetFighters(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil || u.Hero == nil {
		resp.Fail("castle", "set_fighters", 7015)
		return
	}
	castleID, _ := clanParamInt(req.Params, "castle_id")
	if _, known := gamedata.CastleByID(castleID); !known {
		resp.Fail("castle", "set_fighters", 7015)
		return
	}
	group := int32(1)
	if fArr, ok := req.Params.GetArray("fighters"); ok {
		if g, ok2 := fArr.GetInt(strconv.Itoa(int(u.ID))); ok2 {
			group = g
		}
	}
	s.castleMu.Lock()
	r := s.castleRosterLocked(castleID)
	r.nick[u.ID] = u.Username
	r.group[u.ID] = group
	s.castleMu.Unlock()
	resp.Ack("castle", "set_fighters") // client notifies GUI_CASTLE_IN_BATTLE + re-fetches fighters
}

// handleCastleDesert answers castle|desert -> bare Ack: leave a castle's fighter roster.
func (s *Server) handleCastleDesert(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil {
		resp.Fail("castle", "desert", 7015)
		return
	}
	castleID, _ := clanParamInt(req.Params, "castle_id")
	s.castleMu.Lock()
	if r := s.castleRosters[castleID]; r != nil {
		delete(r.nick, u.ID)
		delete(r.group, u.ID)
	}
	s.castleMu.Unlock()
	resp.Ack("castle", "desert")
}

// handleCastleBattleInfo answers castle|battle_info -> {stages: dense[]}. The tournament
// bracket is deferred, so this returns an empty-but-valid stage list (renders an empty
// bracket, no crash).
func (s *Server) handleCastleBattleInfo(req ctrlproto.Request, resp *ctrlproto.Response) {
	resp.Add("castle", "battle_info", amf.NewArray().Set("stages", amf.NewArray()))
}

// castleContains reports whether uid is enrolled in castleID's roster.
func (s *Server) castleContains(castleID, uid int32) bool {
	s.castleMu.Lock()
	defer s.castleMu.Unlock()
	r := s.castleRosters[castleID]
	if r == nil {
		return false
	}
	_, ok := r.group[uid]
	return ok
}
