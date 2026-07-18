package ctrlserver

import (
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"log"
	"strconv"

	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// This file implements the "Охота" (Hunt, MapType=4) matchmaking flow -- the
// only game mode that runs entirely over the Ctrl channel, with no MPD push
// connection (see FightHelper.OnHuntJoin/OnHuntReady and FightSender):
//
//	arena|get_maps_info  -> map list for the start-battle menu (MapInfoArgParser)
//	hunt|join {map_id}   -> opens the avatar-select lobby (HuntJoinArgParser)
//	  (avatar choice is local; no packet until ready)
//	hunt|ready {map_id, avatar_id} -> battle launch params (HuntReadyArgParser ->
//	  FightLaunchArgParser: ip/port/passwd/scene); the client counts down 5s and
//	  reconnects its Battle TCP socket with that password, loading the scene.
//	hunt|accept {map_id} -> group-invite confirmation, validation-only.

// handleArenaMapsInfoReal answers arena|get_maps_info with the real hunt map
// roster. FightHelper.FindMapById must succeed for every map we later accept in
// hunt|join, or the client NPEs in SetBattleInfo -- keep this list authoritative.
func (s *Server) handleArenaMapsInfoReal(req ctrlproto.Request, resp *ctrlproto.Response) {
	maps := amf.NewArray()
	for _, m := range gamedata.HuntMaps() {
		maps.Add(amf.NewArray().
			Set("id", m.ID).
			Set("type_id", gamedata.MapTypeHunt).
			Set("name", m.Name).
			Set("level_min", m.LevelMin).
			Set("level_max", m.LevelMax).
			Set("scene", m.Scene).
			Set("available", true).
			Set("desc", m.Desc).
			Set("win_desc", m.WinDesc).
			// Client-side naming quirk (MapInfoArgParser): "max_players" feeds
			// mMinPlayers and "map_max_players" feeds mMaxPlayers.
			Set("max_players", m.MinPlayers).
			Set("map_max_players", m.MaxPlayers))
	}
	// «Штурм» (DOTA) maps: FightHelper.FindMapById must resolve every DOTA map we
	// accept in fight|join, and its mType (=type_id) is what routes JoinBattle down
	// the fight|* (not hunt|*) path.
	for _, m := range gamedata.DotaMaps() {
		maps.Add(amf.NewArray().
			Set("id", m.ID).
			Set("type_id", gamedata.MapTypeDota).
			Set("name", m.Name).
			Set("level_min", m.LevelMin).
			Set("level_max", m.LevelMax).
			Set("scene", m.Scene).
			Set("available", true).
			Set("desc", m.Desc).
			Set("win_desc", m.WinDesc).
			Set("max_players", m.MinPlayers).
			Set("map_max_players", m.MaxPlayers))
	}
	// «Арена» (DM) maps: same fight|* path as DOTA (only HUNT routes down hunt|*),
	// so FindMapById must resolve these too or the client NPEs on join. type_id=DM
	// makes the client file them under the «Арена» tab.
	for _, m := range gamedata.ArenaMaps() {
		maps.Add(amf.NewArray().
			Set("id", m.ID).
			Set("type_id", gamedata.MapTypeDM).
			Set("name", m.Name).
			Set("level_min", m.LevelMin).
			Set("level_max", m.LevelMax).
			Set("scene", m.Scene).
			Set("available", true).
			Set("desc", m.Desc).
			Set("win_desc", m.WinDesc).
			Set("max_players", m.MinPlayers).
			Set("map_max_players", m.MaxPlayers))
	}
	resp.Add("arena", "get_maps_info", amf.NewArray().Set("maps_info", maps))
}

// handleMapTypeDescs answers arena|get_map_type_descs -> {descs:[{type_id,
// desc}]} (MapTypeDescsArgParser), the mode blurbs in the start-battle menu.
func (s *Server) handleMapTypeDescs(req ctrlproto.Request, resp *ctrlproto.Response) {
	descs := amf.NewArray()
	descs.Add(amf.NewArray().
		Set("type_id", gamedata.MapTypeHunt).
		Set("desc", "Охота — PvE-режим: истребляйте монстров, зарабатывайте опыт и трофеи."))
	descs.Add(amf.NewArray().
		Set("type_id", gamedata.MapTypeDota).
		Set("desc", "Штурм — командный захват: сокрушите оборону и уничтожьте вражеский алтарь."))
	descs.Add(amf.NewArray().
		Set("type_id", gamedata.MapTypeDM).
		Set("desc", "Арена — бой насмерть: сражайтесь с другими игроками до предела фрагов."))
	resp.Add("arena", "get_map_type_descs", amf.NewArray().Set("descs", descs))
}

// handleArenaGetMaps answers arena|get_maps {type[, map_id]} -> {type,
// leave_info, maps:{"<id>":{...}}} (MapListArgParser; assoc keyed by map id).
func (s *Server) handleArenaGetMaps(req ctrlproto.Request, resp *ctrlproto.Response) {
	mapType := req.Params.IntOr("type", -1)
	maps := amf.NewArray()
	if mapType == gamedata.MapTypeHunt {
		for _, m := range gamedata.HuntMaps() {
			maps.Set(strconv.Itoa(int(m.ID)), amf.NewArray().
				Set("name", m.Name).
				Set("scene", m.Scene).
				Set("available", true).
				Set("used", false).
				Set("level_min", m.LevelMin).
				Set("level_max", m.LevelMax).
				Set("desc", m.Desc).
				Set("win_desc", m.WinDesc).
				Set("max_players", m.MinPlayers).
				Set("map_max_players", m.MaxPlayers))
		}
	}
	if mapType == gamedata.MapTypeDota {
		for _, m := range gamedata.DotaMaps() {
			maps.Set(strconv.Itoa(int(m.ID)), amf.NewArray().
				Set("name", m.Name).
				Set("scene", m.Scene).
				Set("available", true).
				Set("used", false).
				Set("level_min", m.LevelMin).
				Set("level_max", m.LevelMax).
				Set("desc", m.Desc).
				Set("win_desc", m.WinDesc).
				Set("max_players", m.MinPlayers).
				Set("map_max_players", m.MaxPlayers))
		}
	}
	if mapType == gamedata.MapTypeDM {
		for _, m := range gamedata.ArenaMaps() {
			maps.Set(strconv.Itoa(int(m.ID)), amf.NewArray().
				Set("name", m.Name).
				Set("scene", m.Scene).
				Set("available", true).
				Set("used", false).
				Set("level_min", m.LevelMin).
				Set("level_max", m.LevelMax).
				Set("desc", m.Desc).
				Set("win_desc", m.WinDesc).
				Set("max_players", m.MinPlayers).
				Set("map_max_players", m.MaxPlayers))
		}
	}
	resp.Add("arena", "get_maps", amf.NewArray().
		Set("type", mapType).
		Set("leave_info", amf.NewArray().Set("banned", false).Set("time", int32(0))).
		Set("maps", maps))
}

// handleHuntJoin answers hunt|join {map_id}: it opens the client's
// avatar-select lobby (FightHelper.OnHuntJoin builds the fight data from this).
// deny_for_map would grey out specific avatars on this map; add_stats carries
// the hero-set stat bonuses (none yet).
func (s *Server) handleHuntJoin(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil {
		resp.Fail("hunt", "join", 6013) // WRONG_SESSION
		return
	}
	mapID := req.Params.IntOr("map_id", -1)
	if _, ok := gamedata.HuntMapByID(mapID); !ok {
		log.Printf("ctrl: hunt|join unknown map %d from user %d", mapID, u.ID)
		resp.Fail("hunt", "join", 8011)
		return
	}
	log.Printf("ctrl: hunt|join user=%d map=%d", u.ID, mapID)
	resp.Add("hunt", "join", amf.NewArray().
		Set("deny_for_map", amf.NewArray()).
		Set("map_id", mapID).
		Set("user_id", u.ID).
		Set("nick", u.Username).
		Set("tag", "").
		Set("add_stats", amf.NewArray()))

	// Party hunt: when a group LEADER joins a hunt, invite the rest of the party so
	// they get the avatar-select prompt and can enter the same instance together.
	s.inviteGroupToHunt(u, mapID)
}

// inviteGroupToHunt pushes fight|invite_mpd to the leader's party members when a
// group leader joins a hunt, so each member sees the "X invites you to hunt on map
// Y" dialog (FightView.FriendInvite -> YesNoDialog). Accepting re-sends hunt|join,
// which opens the member's own avatar-select window and, via huntRoomForMap, lands
// them in the leader's shared instance.
//
// The whole party-hunt flow is server-driven: the client has NO hunt-invite send
// command (the hunt enum is only join/ready/accept, and fight|invite_mpd is a
// push-only server->client message), so this push is the only initiator -- and it
// needs no client edit. The push key is "fight|invite" WITHOUT the _mpd suffix (the
// client's ParseCmd appends it); the payload rides under "arguments" as {map_id,
// nick}. leave is omitted (=false, a fresh invite; FightInviteMpdArgParser only
// reads it when present).
//
// No-op for a solo user, a non-leader (so a member accepting -- which re-runs
// hunt|join -- does NOT re-invite, avoiding a loop), or offline members (Push drops
// them). Members not standing in the central square have no FightView GUI bound and
// silently ignore the invite, which is the intended best-effort behaviour.
func (s *Server) inviteGroupToHunt(leader *session.User, mapID int32) {
	if s.MPD == nil {
		return
	}
	g := s.Store.GroupOf(leader.ID)
	if g == nil || g.Leader != leader.ID {
		return
	}
	for _, mid := range g.Members {
		if mid == leader.ID {
			continue
		}
		s.MPD.Push(mid, "fight|invite", amf.NewArray().
			Set("map_id", mapID).
			Set("nick", leader.Username))
	}
}

// handleHuntReady answers hunt|ready {map_id, avatar_id} with the battle launch
// parameters and records the pending battle so the Battle server recognises the
// user when it reconnects with the issued one-time password.
func (s *Server) handleHuntReady(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil {
		resp.Fail("hunt", "ready", 6013)
		return
	}
	mapID := req.Params.IntOr("map_id", -1)
	avatarID := req.Params.IntOr("avatar_id", -1)
	// The select window's "random" button sends avatar_id -1: the client does NOT
	// choose locally (SelectAvatarWindow.OnSelectButton passes the button's mId, which
	// is -1 for RANDOM_BUTTON), it relies on the server to pick. Resolve it here so the
	// launch succeeds -- the Battle server renders whatever avatar we store below.
	if avatarID == -1 {
		avatarID = randomAvatarID()
		log.Printf("ctrl: hunt|ready user=%d random avatar -> %d", u.ID, avatarID)
	}
	m, ok := gamedata.HuntMapByID(mapID)
	if !ok {
		log.Printf("ctrl: hunt|ready unknown map %d from user %d", mapID, u.ID)
		resp.Fail("hunt", "ready", 8011)
		return
	}
	if _, ok := gamedata.AvatarByID(avatarID); !ok {
		log.Printf("ctrl: hunt|ready unknown avatar %d from user %d", avatarID, u.ID)
		resp.Fail("hunt", "ready", 8011)
		return
	}

	passwd := newBattlePasswd()
	// Open instance per map: route every player who readies for the same map into
	// one shared world (room id = the map id, a stable positive value distinct from
	// the Battle server's negative solo-room ids). The Battle server creates the
	// world on the first arrival and disposes it once the last member leaves. Party
	// invites (a private room per group) can layer on top of this later.
	room := huntRoomForMap(mapID)
	s.Store.SetPendingBattle(u.ID, session.PendingBattle{
		MapID:    mapID,
		AvatarID: avatarID,
		Passwd:   passwd,
		Scene:    m.Scene,
		Room:     room,
	})
	log.Printf("ctrl: hunt|ready user=%d map=%d avatar=%d scene=%s room=%d -> launch",
		u.ID, mapID, avatarID, m.Scene, room)

	// Route the client to this room's dedicated Battle server (own clock, so the
	// in-battle timer counts from entry rather than server uptime) when a launcher
	// is configured; everyone on this map (room = map id) shares the one server.
	ip, ports := s.launchTarget(mapID, room)
	resp.Add("hunt", "ready", amf.NewArray().
		Set("params", amf.NewArray().
			Set("ip", ip).
			Set("port", ports).
			Set("passwd", passwd).
			Set("scene", m.Scene).
			Set("map_id", mapID)))
}

// handleAvatarsProto serves /xml/avatars.amf: the dense prototype list
// CtrlAvatarStore.Retrieve parses (id/name/icon/prefab/stats/skills). Without
// it the avatar-select window has no data and the ready button stays locked.
func (s *Server) handleAvatarsProto() *amf.MixedArray {
	root := amf.NewArray()
	for _, a := range gamedata.Avatars() {
		// Exactly 4 skills: the select window and avatar menu index fixed
		// 4-element buffers by mSkills.Count — a 5th entry crashes the client.
		skills := amf.NewArray()
		for n := 1; n <= 4; n++ {
			skills.Add(amf.NewArray().
				Set("title", a.SkillTitle(n)).
				Set("desc", a.SkillDesc(n)).
				Set("icon", a.SkillIcon(n)))
		}
		root.Add(amf.NewArray().
			Set("id", a.ID).
			Set("name", a.Name()).
			Set("short", a.Short()).
			Set("long", a.Long()).
			Set("icon", a.Icon()).
			Set("prefab", a.Prefab).
			Set("type", a.Type).
			Set("artifact", int32(-1)).
			Set("restr_logic", int32(-1)).
			Set("Health", a.Health).
			Set("HealthRegen", a.HealthRegen).
			Set("Mana", a.Mana).
			Set("ManaRegen", a.ManaRegen).
			Set("SpellPower", a.SpellPower).
			Set("AttackSpeed", a.AttackSpeed).
			Set("DamageMin", a.DmgMin).
			Set("DamageMax", a.DmgMax).
			Set("PhysArmor", a.PhysArmor).
			Set("MagicArmor", a.MagicArmor).
			Set("skills", skills))
	}
	return root
}

// handleAvatarListReal answers avatar|list with every roster avatar unlocked
// (available:true is what makes the select-window button clickable).
func (s *Server) handleAvatarListReal(req ctrlproto.Request, resp *ctrlproto.Response) {
	list := amf.NewArray()
	for _, a := range gamedata.Avatars() {
		list.Set(strconv.Itoa(int(a.ID)), amf.NewArray().
			Set("available", true).
			Set("level", int32(0)).
			Set("max_level", int32(5)).
			Set("inc_mastery_time", int32(0)).
			Set("inc_mastery_potion", int32(0)).
			Set("discount", false).
			Set("add_stats", amf.NewArray()).
			Set("relics", amf.NewArray()))
	}
	resp.Add("avatar", "list", amf.NewArray().
		Set("avatars", list).
		Set("extra_level", int32(0)).
		Set("extra_price", amf.NewArray()))
}

// huntRoomForMap maps a hunt map to the shared-world room id. One open instance
// per map: everyone who enters map N lands in room N. Kept as a function so a
// future capacity split (map_max_players) or party rooms can change the policy
// without touching the launch handler.
func huntRoomForMap(mapID int32) int32 { return mapID }

func newBattlePasswd() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// randomAvatarID returns a uniformly random roster avatar id, used to resolve the
// select-window "random" button (which sends avatar_id -1). Returns -1 only if the
// roster is empty (which would fail the launch, correctly).
func randomAvatarID() int32 {
	avs := gamedata.Avatars()
	if len(avs) == 0 {
		return -1
	}
	var b [4]byte
	_, _ = rand.Read(b[:])
	idx := binary.BigEndian.Uint32(b[:]) % uint32(len(avs))
	return avs[idx].ID
}
