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
// arg-less fight|ready: the map, the chosen avatar, and the shared-world room the
// matched players will all launch into.
type fightSelection struct {
	mapID    int32
	avatarID int32
	room     int32
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
	dm, ok := gamedata.DotaMapByID(mapID)
	if !ok {
		log.Printf("ctrl: fight|join unknown DOTA map %d from user %d", mapID, u.ID)
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
	waiting := append([]int32(nil), s.dotaQueue[mapID]...) // snapshot for queue-size pushes
	var match []int32
	var room int32
	if int32(len(s.dotaQueue[mapID])) >= size {
		match = append([]int32(nil), s.dotaQueue[mapID][:size]...)
		s.dotaQueue[mapID] = s.dotaQueue[mapID][size:]
		s.nextDotaRoom++
		room = dotaRoomBase + s.nextDotaRoom
		for _, uid := range match {
			s.fightSel[uid] = fightSelection{mapID: mapID, avatarID: -1, room: room}
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
	// roster (all on team 1 -- co-op push against the enemy base in v1).
	fighters := amf.NewArray()
	for _, uid := range match {
		nick := ""
		if usr, ok := s.Store.ByID(uid); ok {
			nick = usr.Username
		}
		fighters.Set(strconv.Itoa(int(uid)), amf.NewArray().
			Set("nick", nick).Set("tag", "").Set("team", int32(1)))
	}
	for _, uid := range match {
		s.MPD.Push(uid, "fight|start_select_avatar", amf.NewArray().
			Set("fighters", fighters).
			Set("map_id", mapID).
			Set("time", int32(30)).
			Set("deny_for_map", amf.NewArray()).
			Set("add_stats", amf.NewArray()))
	}
	log.Printf("ctrl: «Штурм» match formed map=%d scene=%s room=%d players=%v", mapID, dm.Scene, room, match)
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
	dm, ok := gamedata.DotaMapByID(sel.mapID)
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
		Scene:    dm.Scene,
		Room:     room,
	})
	s.clearFightSel(u.ID)
	log.Printf("ctrl: fight|ready user=%d map=%d avatar=%d scene=%s room=%d -> launch",
		u.ID, sel.mapID, avatarID, dm.Scene, room)
	if s.MPD == nil {
		return
	}
	s.MPD.Push(u.ID, "fight|ready", amf.NewArray().Set("user_id", u.ID))
	ports := amf.NewArray()
	for _, p := range s.BattlePorts {
		ports.Add(p)
	}
	s.MPD.Push(u.ID, "fight|launch", amf.NewArray().
		Set("ip", s.BattleHost).
		Set("port", ports).
		Set("passwd", passwd).
		Set("scene", dm.Scene).
		Set("map_id", sel.mapID))
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
