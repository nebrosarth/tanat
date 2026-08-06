package ctrlserver

import (
	"log"
	"strconv"

	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// «Штурм» (MapType.DOTA) matchmaking over the fight|* command path. Unlike Hunt
// (hunt|join -> hunt|ready, one shot over Ctrl), DOTA uses the queue/lobby handshake:
//
//	fight|join {map_id}            -> queue ack; then MPD push fight|start_select_avatar
//	fight|in_request              -> ack opens the avatar-select window
//	fight|select_avatar {avatar}  -> ack + MPD broadcast fight|select_avatar
//	fight|ready                   -> MPD push fight|ready then fight|launch {ip,port,passwd,scene,map_id}
//
// v1 is a SOLO instant-match: the "queue" resolves immediately to a one-player match
// on the player's side (team 1), so the whole flow runs for a single client with no
// real opponents -- the battle world fills the enemy side with structures + creeps
// (see battleserver/dota.go). The wire shape mirrors the real client exactly, so the
// same handlers extend to true N-player matchmaking later.

// dotaRoomBase is the id floor for per-match shared-world rooms, clear of Hunt/DOTA
// map ids (which double as their own open-instance rooms).
const dotaRoomBase int32 = 200000

// fightSelection is the in-flight DOTA choice held between fight|select_avatar and the
// arg-less fight|ready: the map, the chosen avatar, the shared-world room the matched
// players will all launch into, and the side the matchmaker put this player on.
type fightSelection struct {
	mapID    int32
	avatarID int32
	room     int32
	// team is the assigned PvP side for a «Штурм» match: 1 = Human/«Собор», 2 =
	// Elf/«Изгнанники». 0 for a co-op/«Арена» launch where the Battle server picks sides
	// itself. Carried into PendingBattle so the pre-battle roster and the battle agree.
	team int32
}

func (s *Server) setFightSel(uid int32, sel fightSelection) {
	s.fightMu.Lock()
	defer s.fightMu.Unlock()
	if s.fightSel == nil {
		s.fightSel = map[int32]fightSelection{}
	}
	s.fightSel[uid] = sel
}

func (s *Server) getFightSel(uid int32) (fightSelection, bool) {
	s.fightMu.Lock()
	defer s.fightMu.Unlock()
	sel, ok := s.fightSel[uid]
	return sel, ok
}

func (s *Server) clearFightSel(uid int32) {
	s.fightMu.Lock()
	defer s.fightMu.Unlock()
	delete(s.fightSel, uid)
}

// dotaRoomForMap is the fallback shared-world room for a DOTA map when no per-match
// room was assigned (e.g. a stray fight|ready). Distinct from Hunt map ids.
func dotaRoomForMap(mapID int32) int32 { return mapID }

// removeFromDotaQueueLocked drops uid from every map's waiting list. Caller holds
// fightMu.
func (s *Server) removeFromDotaQueueLocked(uid int32) {
	for m, q := range s.dotaQueue {
		out := q[:0]
		for _, id := range q {
			if id != uid {
				out = append(out, id)
			}
		}
		s.dotaQueue[m] = out
	}
}

// handleFightJoin answers fight|join {map_id}: the player enters the map's queue and
// gets a queue ack. Once DotaMatchSize players are waiting, a match forms -- a fresh
// shared room is assigned to all of them and each is pushed fight|start_select_avatar
// ("match found"). With DotaMatchSize=1 the match forms on the joiner's own request
// (the solo instant-match).
func (s *Server) handleFightJoin(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil {
		resp.Fail("fight", "join", 6013) // WRONG_SESSION
		return
	}
	mapID := req.Params.IntOr("map_id", -1)
	scene, ok := fightSceneForMap(mapID)
	if !ok {
		log.Printf("ctrl: fight|join unknown fight map %d from user %d", mapID, u.ID)
		resp.Fail("fight", "join", 8011)
		return
	}

	s.fightMu.Lock()
	if s.dotaQueue == nil {
		s.dotaQueue = map[int32][]int32{}
	}
	if s.fightSel == nil {
		s.fightSel = map[int32]fightSelection{}
	}
	s.removeFromDotaQueueLocked(u.ID) // a re-join replaces the old queue slot
	s.dotaQueue[mapID] = append(s.dotaQueue[mapID], u.ID)
	size := s.DotaMatchSize
	if size < 1 {
		size = 1
	}
	_, isDota := gamedata.DotaMapByID(mapID)
	waiting := append([]int32(nil), s.dotaQueue[mapID]...) // snapshot for queue-size pushes
	var match []int32
	var room int32
	teams := map[int32]int32{} // uid -> assigned side (0 = Battle server picks)
	if int32(len(s.dotaQueue[mapID])) >= size {
		match = append([]int32(nil), s.dotaQueue[mapID][:size]...)
		s.dotaQueue[mapID] = s.dotaQueue[mapID][size:]
		s.nextDotaRoom++
		room = dotaRoomBase + s.nextDotaRoom
		for i, uid := range match {
			// «Штурм» is true PvP: split the matched players across the two bases,
			// alternating Human/Elf so an even lobby is balanced (1v1, 2v2, ...). An odd
			// match gives the extra player to «Собор» (team 1). «Арена» keeps team 0 and
			// lets the Battle server assign sides (it has its own alternating counter).
			var team int32
			if isDota {
				team = int32(1 + i%2)
			}
			teams[uid] = team
			s.fightSel[uid] = fightSelection{mapID: mapID, avatarID: -1, room: room, team: team}
		}
	}
	s.fightMu.Unlock()

	log.Printf("ctrl: fight|join user=%d map=%d queued=%d matchSize=%d formed=%v",
		u.ID, mapID, len(waiting), size, match != nil)
	resp.Add("fight", "join", amf.NewArray().
		Set("map_id", mapID).
		Set("wait", int32(0)).
		Set("queue_size", int32(len(waiting))))

	if s.MPD == nil {
		return
	}
	if match == nil {
		// Still short of a full match: refresh everyone's queue counter.
		for _, uid := range waiting {
			s.MPD.Push(uid, "fight|queue_size", amf.NewArray().
				Set("map_id", mapID).Set("count", int32(len(waiting))))
		}
		return
	}
	// Match found: push start_select_avatar to every matched player with the shared
	// roster. Each fighter carries the side the matchmaker gave them, so the «match found»
	// screen splits into the two columns (team 1 vs the rest) the client renders -- a
	// «Штурм» PvP match shows «Собор» vs «Изгнанники», a co-op/«Арена» one shows one team.
	fighters := amf.NewArray()
	for _, uid := range match {
		nick := ""
		if usr, ok := s.Store.ByID(uid); ok {
			nick = usr.Username
		}
		team := teams[uid]
		if team == 0 {
			team = 1 // display default for co-op/«Арена»: the client groups by team-1-vs-rest
		}
		fighters.Set(strconv.Itoa(int(uid)), amf.NewArray().
			Set("nick", nick).Set("tag", "").Set("team", team))
	}
	for _, uid := range match {
		s.MPD.Push(uid, "fight|start_select_avatar", amf.NewArray().
			Set("fighters", fighters).
			Set("map_id", mapID).
			Set("time", int32(30)).
			Set("deny_for_map", amf.NewArray()).
			Set("add_stats", amf.NewArray()))
	}
	log.Printf("ctrl: fight match formed map=%d scene=%s room=%d players=%v", mapID, scene, room, match)
}

// fightSceneForMap resolves the scene bundle for a fight|join / fight|ready map id,
// accepting both «Штурм» (DOTA) and «Арена» (DM) maps -- the two modes matched through
// this same queue. Returns ("", false) for anything else. A CastleOnly map (map_6_0)
// is rejected here UNLESS castleTestMapExposed() -- see its doc comment in hunt.go.
func fightSceneForMap(mapID int32) (string, bool) {
	if dm, ok := gamedata.DotaMapByID(mapID); ok && (!dm.CastleOnly || castleTestMapExposed()) {
		return dm.Scene, true
	}
	if am, ok := gamedata.ArenaMapByID(mapID); ok {
		return am.Scene, true
	}
	return "", false
}

// handleFightInRequest answers fight|in_request: a plain success ack, which flips the
// client into the SelectAvatarWindow (FightHelper.OnInRequest -> SetBattleInfo, from
// the fighters it already stored on start_select_avatar).
func (s *Server) handleFightInRequest(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil {
		resp.Fail("fight", "in_request", 6013)
		return
	}
	log.Printf("ctrl: fight|in_request user=%d -> open avatar select", u.ID)
	resp.Ack("fight", "in_request")
	if s.MPD != nil {
		s.MPD.Push(u.ID, "fight|in_request", amf.NewArray().Set("user_id", u.ID))
	}
}

// handleFightSelectAvatar answers fight|select_avatar {avatar_id}: records the choice
// and broadcasts fight|select_avatar {user_id, avatar_id} so the roster tile updates
// (including on the chooser).
func (s *Server) handleFightSelectAvatar(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil {
		resp.Fail("fight", "select_avatar", 6013)
		return
	}
	avatarID := req.Params.IntOr("avatar_id", -1)
	if avatarID == -1 {
		avatarID = randomAvatarID() // the client's "random" button sends -1
	}
	if _, ok := gamedata.AvatarByID(avatarID); !ok {
		resp.Fail("fight", "select_avatar", 8011)
		return
	}
	sel, _ := s.getFightSel(u.ID)
	sel.avatarID = avatarID
	s.setFightSel(u.ID, sel)
	log.Printf("ctrl: fight|select_avatar user=%d avatar=%d", u.ID, avatarID)
	resp.Ack("fight", "select_avatar")
	if s.MPD != nil {
		s.MPD.Push(u.ID, "fight|select_avatar", amf.NewArray().
			Set("user_id", u.ID).Set("avatar_id", avatarID))
	}
}

// handleFightReady answers fight|ready (no args): records the pending battle so the
// Battle server recognises the reconnect, then pushes fight|ready (UI lock) and
// fight|launch {ip, port[], passwd, scene, map_id} -- the Battle-server handoff.
func (s *Server) handleFightReady(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil {
		resp.Fail("fight", "ready", 6013)
		return
	}
	resp.Ack("fight", "ready")
	sel, ok := s.getFightSel(u.ID)
	if !ok {
		log.Printf("ctrl: fight|ready user=%d with no selection -- ignoring", u.ID)
		return
	}
	scene, ok := fightSceneForMap(sel.mapID)
	if !ok {
		return
	}
	avatarID := sel.avatarID
	if avatarID <= 0 {
		avatarID = randomAvatarID()
	}
	passwd := newBattlePasswd()
	room := sel.room // the match's shared room; all matched players share one world
	if room <= 0 {
		room = dotaRoomForMap(sel.mapID)
	}
	s.Store.SetPendingBattle(u.ID, session.PendingBattle{
		MapID:    sel.mapID,
		AvatarID: avatarID,
		Passwd:   passwd,
		Scene:    scene,
		Room:     room,
		Team:     sel.team, // the matchmaker-assigned «Штурм» side (0 = Battle server picks)
	})
	s.clearFightSel(u.ID)
	log.Printf("ctrl: fight|ready user=%d map=%d avatar=%d scene=%s room=%d -> launch",
		u.ID, sel.mapID, avatarID, scene, room)
	if s.MPD == nil {
		return
	}
	s.MPD.Push(u.ID, "fight|ready", amf.NewArray().Set("user_id", u.ID))
	// Route the client to this match's dedicated Battle server (own clock, so the
	// in-battle timer counts from match start) when a launcher is configured; every
	// player of this room gets the same server.
	ip, ports := s.launchTarget(sel.mapID, room)
	s.MPD.Push(u.ID, "fight|launch", amf.NewArray().
		Set("ip", ip).
		Set("port", ports).
		Set("passwd", passwd).
		Set("scene", scene).
		Set("map_id", sel.mapID))
}

// handleFightLog answers fight|log {fight_id}: this match's end-of-battle scoreboard,
// requested once by the client right after it receives BATTLE_END (Battle.OnBattleEnd ->
// SendGetFightLog(BattleId, -1) in the decompiled client). fight_id is the REQUESTING
// CLIENT'S OWN battleId -- every participant of the same match was issued a DIFFERENT one
// at CONNECT (see battleserver's conn.battleID) -- so this is always a single map lookup,
// never a room/match id. Before this handler existed at all, fight|log fell through to the
// generic UNHANDLED ack (no "log" key whatsoever), which is why the end-of-match table
// came up empty: FightLogArgParser built zero rows from a response that never had any.
// Missing/expired data (server restarted mid-match, a stray/duplicate request, or a match
// that never called settleMatchLocked) answers with an empty heroes map rather than
// failing -- the client tolerates that as an empty scoreboard, same as before this existed.
func (s *Server) handleFightLog(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil {
		resp.Fail("fight", "log", 6013)
		return
	}
	fightID := req.Params.IntOr("fight_id", -1)
	heroes := amf.NewArray()
	n := 0
	if entries, ok := s.Store.FightLog(fightID); ok {
		for avatarID, e := range entries {
			heroes.Set(strconv.Itoa(int(avatarID)), amf.NewArray().
				Set("avatar", e.AvatarID).
				Set("nick", e.Nick).
				Set("team", e.Team).
				Set("AvatarKills", e.Kills).
				Set("Assists", e.Assists).
				Set("Deaths", e.Deaths).
				Set("level", e.Level).
				Set("money", e.Money).
				Set("rating", e.NewRating).
				Set("old_rating", e.OldRating))
			n++
		}
	}
	log.Printf("ctrl: fight|log user=%d fight_id=%d heroes=%d", u.ID, fightID, n)
	resp.Add("fight", "log", amf.NewArray().Set("log", amf.NewArray().Set("heroes", heroes)))
}

// handleFightDesert answers fight|desert {map_id}: leave the queue/lobby. For a solo
// instant-match this just drops the pending selection.
func (s *Server) handleFightDesert(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil {
		resp.Fail("fight", "desert", 6013)
		return
	}
	s.clearFightSel(u.ID)
	s.fightMu.Lock()
	s.removeFromDotaQueueLocked(u.ID)
	s.fightMu.Unlock()
	log.Printf("ctrl: fight|desert user=%d", u.ID)
	resp.Ack("fight", "desert")
	if s.MPD != nil {
		s.MPD.Push(u.ID, "fight|desert", amf.NewArray().Set("user_id", u.ID))
	}
}
