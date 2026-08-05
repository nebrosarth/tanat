package ctrlserver

import (
	"log"
	"strconv"
	"time"

	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// Castle (clan-siege), «Битва за замок». obj="castle". Out-of-battle screens (browse
// castles, view a castle's roster/history, enroll/desert as a fighter) are backed by a
// static registry (gamedata.Castles) plus an in-memory enrollment roster. The LIVE
// siege runs over the SAME queue/lobby handshake «Штурм» uses (fight|*, see dota.go),
// just entered from a scheduled window instead of fight|join:
//
//	(roster fills via castle|set_fighters, any time before the window fires)
//	scheduler fires -> castle|start_request_mpd  (match found: team rosters + timer)
//	castle|select_avatar {avatar}                -> ack + MPD broadcast select_avatar_mpd
//	castle|ready                                 -> MPD push ready_mpd then launch_mpd
//	castle|desert_battle                         -> leave the in-flight draft
//
// The battle itself IS «Штурм» (map10/map_1_0's two altars ARE a "destroy the enemy's
// single win structure" siege objective already -- see battleserver/castle.go), tagged
// with the castle id via session.PendingBattle.CastleID so the Battle server knows to
// transfer ownership + log history + pay CastleReward when it ends.
//
// What's still a deliberate simplification (not the real client's multi-clan bracket):
// only two sides ever form per window -- the owning clan (or, if unowned, whichever
// enrolled clan has the most fighters) defends, every other enrolled fighter attacks
// as one side -- and the tournament bracket (battle_info) stays an empty stage list.
// A fighter roster with nobody enrolled simply lets the window pass with no battle.
//
// Wire-shape notes (from CastleListArgParser / CastleMembersArgParser / ...):
//   - list/info/history/fighters/battle_info RESPONSES put their collection at the ROOT.
//   - id-keyed collections (castles, members, queue, fighters) are ASSOCIATIVE arrays with
//     STRING keys the client int.TryParse's; stages is a DENSE array.
//   - times are RELATIVE: start_time = seconds until the next battle window.

// castleRoster is one castle's in-memory fighter enrollment: who signed up (nick, for
// the fighters screen) and which squad they picked (group: main/reserve -- Castle_
// Request_Base_Text / _Reserve_Text -- NOT the in-battle attacker/defender side; that
// is derived from clan membership at battle-window time, see assignCastleTeams).
type castleRoster struct {
	nick  map[int32]string // uid -> display nick
	group map[int32]int32  // uid -> assigned fighter group / battle number
}

// castleRoomBase is the id floor for castle-siege shared-world rooms, clear of Hunt
// map ids and «Штурм»'s dotaRoomBase (200000+).
const castleRoomBase int32 = 300000

// castleSelectTimeoutSec is the avatar-select window's duration, pushed as both the
// start_request "time" field and the separate select_avatar_timer push (Castle, unlike
// «Штурм», has a dedicated timer_mpd command -- CtrlCmdId.castle.select_avatar_timer_mpd
// -- so both are sent; a client that ignores one still has the other).
const castleSelectTimeoutSec int32 = 30

// castleSelection is the in-flight draft choice held between the scheduler's
// start_request push and the arg-less castle|ready -- the castle twin of
// fightSelection (dota.go). team is 1 (attacker/challenger) or 2 (defender/owner),
// assigned by assignCastleTeams when the battle window fires.
type castleSelection struct {
	castleID int32
	avatarID int32
	room     int32
	team     int32
}

func (s *Server) setCastleSel(uid int32, sel castleSelection) {
	s.castleMu.Lock()
	defer s.castleMu.Unlock()
	if s.castleSel == nil {
		s.castleSel = map[int32]castleSelection{}
	}
	s.castleSel[uid] = sel
}

func (s *Server) getCastleSel(uid int32) (castleSelection, bool) {
	s.castleMu.Lock()
	defer s.castleMu.Unlock()
	sel, ok := s.castleSel[uid]
	return sel, ok
}

func (s *Server) clearCastleSel(uid int32) {
	s.castleMu.Lock()
	defer s.castleMu.Unlock()
	delete(s.castleSel, uid)
}

// castleInProgress reports whether castleID currently has an in-flight draft (any
// enrolled fighter still holding a castleSelection for it) -- doubles as the
// castle|info "in_progress" flag: it's naturally true from the moment the battle
// window fires until every drafted fighter has either readied (launched) or deserted.
func (s *Server) castleInProgress(castleID int32) bool {
	s.castleMu.Lock()
	defer s.castleMu.Unlock()
	for _, sel := range s.castleSel {
		if sel.castleID == castleID {
			return true
		}
	}
	return false
}

// castleCountdownRemaining returns c's live seconds-until-battle, lazily seeded from
// the registry's StartInSec on first read. The scheduler (StartCastleScheduler) counts
// this down and resets it to CycleSec each time the window fires; castle|list/info
// read it instead of the static gamedata value once the server has been running.
func (s *Server) castleCountdownRemaining(c gamedata.Castle) int32 {
	s.castleMu.Lock()
	defer s.castleMu.Unlock()
	if s.castleCountdown == nil {
		s.castleCountdown = map[int32]int32{}
	}
	if _, ok := s.castleCountdown[c.ID]; !ok {
		s.castleCountdown[c.ID] = c.StartInSec
	}
	return s.castleCountdown[c.ID]
}

// StartCastleScheduler launches the background goroutine that ticks every castle's
// countdown once a second and fires its battle window at zero. Idempotent (a second
// call while one is already running is a no-op). Call once from cmd/ctrlserver.
func (s *Server) StartCastleScheduler() {
	s.castleMu.Lock()
	if s.castleStop != nil {
		s.castleMu.Unlock()
		return
	}
	stop := make(chan struct{})
	s.castleStop = stop
	s.castleMu.Unlock()
	go func() {
		t := time.NewTicker(1 * time.Second)
		defer t.Stop()
		for {
			select {
			case <-stop:
				return
			case <-t.C:
				s.tickCastles(1)
			}
		}
	}()
}

// StopCastleScheduler halts the background goroutine. Safe to call even if it was
// never started.
func (s *Server) StopCastleScheduler() {
	s.castleMu.Lock()
	stop := s.castleStop
	s.castleStop = nil
	s.castleMu.Unlock()
	if stop != nil {
		close(stop)
	}
}

// tickCastles decrements every castle's countdown by elapsedSec, firing the battle
// window for any that reach zero. Exposed directly (not just via the 1s ticker) so
// tests can fast-forward a window without sleeping on real time.
func (s *Server) tickCastles(elapsedSec int32) {
	for _, c := range gamedata.Castles() {
		s.castleMu.Lock()
		if s.castleCountdown == nil {
			s.castleCountdown = map[int32]int32{}
		}
		if _, ok := s.castleCountdown[c.ID]; !ok {
			s.castleCountdown[c.ID] = c.StartInSec
		}
		s.castleCountdown[c.ID] -= elapsedSec
		fire := s.castleCountdown[c.ID] <= 0
		if fire {
			cycle := c.CycleSec
			if cycle <= 0 {
				cycle = c.StartInSec
			}
			s.castleCountdown[c.ID] = cycle
		}
		s.castleMu.Unlock()
		if fire {
			s.fireCastleBattleWindow(c)
		}
	}
}

// assignCastleTeams splits an enrolled roster into attacker(1)/defender(2) by clan:
// the castle's current owner clan defends; every other enrolled fighter attacks. An
// unowned castle instead makes the largest enrolled clan the defender (first-seen
// wins a tie) -- a stand-in for "whoever shows the most force claims the defense"
// absent a real multi-clan bracket. A fighter in no clan always attacks.
//
// Simplification note: the real client supports many simultaneous clan challengers
// via a tournament bracket (castle|battle_info, still an empty stage list here); this
// collapses every window to exactly two sides.
func (s *Server) assignCastleTeams(c gamedata.Castle, uids []int32) map[int32]int32 {
	defenderClan, _, owned := s.Store.CastleOwner(c.ID)
	if !owned {
		counts := map[int32]int32{}
		var order []int32
		for _, uid := range uids {
			clanID, _ := s.Store.HeroClanInfo(uid)
			if clanID == 0 {
				continue
			}
			if counts[clanID] == 0 {
				order = append(order, clanID)
			}
			counts[clanID]++
		}
		var best int32
		for _, clanID := range order {
			if counts[clanID] > best {
				best, defenderClan = counts[clanID], clanID
			}
		}
	}
	teams := map[int32]int32{}
	for _, uid := range uids {
		clanID, _ := s.Store.HeroClanInfo(uid)
		if defenderClan != 0 && clanID == defenderClan {
			teams[uid] = 2
		} else {
			teams[uid] = 1
		}
	}
	return teams
}

// fireCastleBattleWindow snapshots c's enrolled roster, assigns sides, and pushes
// castle|start_request (+ select_avatar_timer) to every enrolled fighter -- the
// castle twin of handleFightJoin's "match found" push. An empty roster just lets the
// window pass silently (no battle, no push): there is no queue to refund.
func (s *Server) fireCastleBattleWindow(c gamedata.Castle) {
	s.castleMu.Lock()
	var uids []int32
	if r := s.castleRosters[c.ID]; r != nil {
		for uid := range r.nick {
			uids = append(uids, uid)
		}
	}
	s.castleMu.Unlock()
	if len(uids) == 0 {
		log.Printf("ctrl: castle %d battle window passed with no enrolled fighters", c.ID)
		return
	}

	teams := s.assignCastleTeams(c, uids)

	s.castleMu.Lock()
	s.nextCastleRoom++
	room := castleRoomBase + s.nextCastleRoom
	if s.castleSel == nil {
		s.castleSel = map[int32]castleSelection{}
	}
	for _, uid := range uids {
		s.castleSel[uid] = castleSelection{castleID: c.ID, avatarID: -1, room: room, team: teams[uid]}
	}
	s.castleMu.Unlock()

	fighters := amf.NewArray()
	for _, uid := range uids {
		nick := ""
		if usr, ok := s.Store.ByID(uid); ok {
			nick = usr.Username
		}
		fighters.Set(strconv.Itoa(int(uid)), amf.NewArray().
			Set("nick", nick).Set("tag", "").Set("team", teams[uid]))
	}
	log.Printf("ctrl: castle %d battle window fired: room=%d fighters=%v", c.ID, room, uids)
	if s.MPD == nil {
		return
	}
	for _, uid := range uids {
		s.MPD.Push(uid, "castle|start_request", amf.NewArray().
			Set("castle_id", c.ID).
			Set("fighters", fighters).
			Set("map_id", c.MapID).
			Set("time", castleSelectTimeoutSec).
			Set("deny_for_map", amf.NewArray()).
			Set("add_stats", amf.NewArray()))
		s.MPD.Push(uid, "castle|select_avatar_timer", amf.NewArray().
			Set("castle_id", c.ID).Set("time", castleSelectTimeoutSec))
	}
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
		ownerID, ownerName := c.OwnerID, c.OwnerName
		if id, name, held := s.Store.CastleOwner(c.ID); held {
			ownerID, ownerName = id, name
		}
		rewards := amf.NewArray().
			Set("money_d", c.Reward.Diamonds).
			Set("money", c.Reward.Money).
			Set("exp", c.Reward.Exp).
			Set("item", c.Reward.Item).
			Set("item_count", c.Reward.ItemCount)
		castles.Set(strconv.Itoa(int(c.ID)), amf.NewArray().
			Set("level_max", c.LevelMax).
			Set("level_min", c.LevelMin).
			Set("name", c.NameKey).
			Set("owner_name", ownerName).
			Set("owner_id", ownerID).
			Set("fighters_min", c.FightersMin).
			Set("start_time", s.castleCountdownRemaining(c)).
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
		Set("in_progress", s.castleInProgress(castleID)).
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

// handleCastleSelectAvatar answers castle|select_avatar {avatar_id}: records the
// draft choice and broadcasts castle|select_avatar {user_id, avatar_id} so the roster
// tile updates -- the castle twin of handleFightSelectAvatar.
func (s *Server) handleCastleSelectAvatar(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil {
		resp.Fail("castle", "select_avatar", 6013) // WRONG_SESSION
		return
	}
	avatarID := req.Params.IntOr("avatar_id", -1)
	if avatarID == -1 {
		avatarID = randomAvatarID() // the client's "random" button sends -1
	}
	if _, ok := gamedata.AvatarByID(avatarID); !ok {
		resp.Fail("castle", "select_avatar", 8011)
		return
	}
	sel, ok := s.getCastleSel(u.ID)
	if !ok {
		// No in-flight draft for this user (window not fired, or already readied/
		// deserted): a clean error, not a crash.
		resp.Fail("castle", "select_avatar", 7015)
		return
	}
	sel.avatarID = avatarID
	s.setCastleSel(u.ID, sel)
	log.Printf("ctrl: castle|select_avatar user=%d castle=%d avatar=%d", u.ID, sel.castleID, avatarID)
	resp.Ack("castle", "select_avatar")
	if s.MPD != nil {
		s.MPD.Push(u.ID, "castle|select_avatar", amf.NewArray().
			Set("user_id", u.ID).Set("avatar_id", avatarID))
	}
}

// handleCastleReady answers castle|ready (no args): records the pending battle so the
// Battle server recognises the reconnect (tagged with CastleID so it knows to settle
// ownership/rewards on this room), then pushes castle|ready (UI lock) and
// castle|launch {ip, port[], passwd, scene, map_id} -- the castle twin of
// handleFightReady.
func (s *Server) handleCastleReady(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil {
		resp.Fail("castle", "ready", 6013)
		return
	}
	resp.Ack("castle", "ready")
	sel, ok := s.getCastleSel(u.ID)
	if !ok {
		log.Printf("ctrl: castle|ready user=%d with no selection -- ignoring", u.ID)
		return
	}
	c, known := gamedata.CastleByID(sel.castleID)
	if !known {
		return
	}
	avatarID := sel.avatarID
	if avatarID <= 0 {
		avatarID = randomAvatarID()
	}
	passwd := newBattlePasswd()
	room := sel.room
	s.Store.SetPendingBattle(u.ID, session.PendingBattle{
		MapID:    c.MapID,
		AvatarID: avatarID,
		Passwd:   passwd,
		Scene:    c.Scene,
		Room:     room,
		Team:     sel.team,
		CastleID: c.ID,
	})
	s.clearCastleSel(u.ID)
	log.Printf("ctrl: castle|ready user=%d castle=%d avatar=%d scene=%s room=%d -> launch",
		u.ID, c.ID, avatarID, c.Scene, room)
	if s.MPD == nil {
		return
	}
	s.MPD.Push(u.ID, "castle|ready", amf.NewArray().Set("user_id", u.ID))
	ip, ports := s.launchTarget(c.MapID, room)
	s.MPD.Push(u.ID, "castle|launch", amf.NewArray().
		Set("ip", ip).
		Set("port", ports).
		Set("passwd", passwd).
		Set("scene", c.Scene).
		Set("map_id", c.MapID))
}

// handleCastleDesertBattle answers castle|desert_battle (no args): leave the
// in-flight draft (distinct from castle|desert, which leaves the REGISTRATION
// roster before any window has fired).
func (s *Server) handleCastleDesertBattle(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil {
		resp.Fail("castle", "desert_battle", 6013)
		return
	}
	s.clearCastleSel(u.ID)
	log.Printf("ctrl: castle|desert_battle user=%d", u.ID)
	resp.Ack("castle", "desert_battle")
	if s.MPD != nil {
		s.MPD.Push(u.ID, "castle|desert_battle", amf.NewArray().Set("user_id", u.ID))
	}
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
