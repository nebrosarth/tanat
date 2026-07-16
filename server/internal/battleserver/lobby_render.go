package battleserver

import (
	"log"
	"sync"

	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
)

// Lobby multiplayer: the central square (cs_human / cs_elf) is a shared hub where
// everyone standing in the same square sees the others walk around. It mirrors the
// hunt shared-world cross-render (avatars.go / broadcast.go) but carries NO combat:
// no mobs, no skills, no per-tick simulation. Presence + movement ride entirely on
// the Battle channel (PLAYER_REG + CREATE_OBJECT("Hero") + SET_AVATAR + POSITION
// SYNC), exactly the pipeline hunt already proves works; the client renders another
// player's customized body once the Ctrl channel answers the hero.get_data_list /
// user.game_info it auto-fires on bind (see ctrlserver/lobby.go).
//
// A lobbyInstance is one square. Unlike a huntInstance (keyed by an unbounded room
// id and disposed when empty), there are only ever two squares, so each instance is
// a permanent per-area hub: created lazily on first entry, reused forever, never
// disposed. Members join on entry and leave on disconnect.
type lobbyInstance struct {
	mu      sync.Mutex
	s       *Server
	area    int32
	members map[int32]*conn // keyed by avatar objID
}

// joinLobbyInstance places c into the shared square for its lobby area (creating the
// hub on first entry), repoints c's lock at the instance mutex so every occupant of
// one square serializes on the same lock, and returns the instance.
//
// Concurrency mirrors joinInstance: s.mu (the registry) and linst.mu are taken
// sequentially, never nested, so no lock-order cycle exists.
func (s *Server) joinLobbyInstance(area int32, c *conn) *lobbyInstance {
	s.mu.Lock()
	linst := s.linsts[area]
	if linst == nil {
		linst = &lobbyInstance{s: s, area: area, members: map[int32]*conn{}}
		s.linsts[area] = linst
	}
	s.mu.Unlock()

	linst.mu.Lock()
	linst.members[c.objID] = c
	c.linst = linst
	c.lk = &linst.mu
	n := len(linst.members)
	linst.mu.Unlock()
	// Mirror square occupancy into the MPD hub so area chat can fan out to this
	// square. Done outside linst.mu (the hub has its own lock).
	if s.MPD != nil {
		s.MPD.SetArea(c.selfPlayerID, area)
	}
	log.Printf("battle: %s joined lobby square area=%d (now %d players)", c.RemoteAddr(), area, n)
	return linst
}

// memberList snapshots the square's READY occupants -- those whose world state is
// fully built (lready). A just-joined occupant is in members (so it is a viewer
// target once ready) but excluded here until lready, so no broadcast fan-out can
// push a teammate packet to a client that is still loading its scene. Caller holds
// linst.mu. Mirrors huntInstance.memberList.
func (linst *lobbyInstance) memberList() []*conn {
	out := make([]*conn, 0, len(linst.members))
	for _, c := range linst.members {
		if c.lready {
			out = append(out, c)
		}
	}
	return out
}

// leaveLobbyInstanceLocked removes c from its square (disconnect) and tells the
// remaining occupants to drop c's avatar. The hub itself is never disposed (one per
// area, reused). Caller holds linst.mu (via c.lock() in closeHunt).
func (linst *lobbyInstance) leaveLobbyInstanceLocked(c *conn) {
	// Guard on identity, not just presence: if the same user reconnected (objID is
	// reused) while this stale conn was still registered, members[objID] now points
	// at the live replacement -- don't evict it or tell everyone to delete the avatar
	// that the new conn is actively rendering.
	if cur, ok := linst.members[c.objID]; !ok || cur != c {
		return
	}
	delete(linst.members, c.objID)
	if linst.s.MPD != nil {
		linst.s.MPD.ClearArea(c.selfPlayerID)
	}
	for _, other := range linst.members {
		if other == c {
			continue
		}
		linst.s.removeLobbyAvatarForLocked(other, c)
	}
}

// renderLobbyAvatarForLocked builds owner's Hero avatar on viewer's client: the
// shared "Hero" prototype (idempotent -- every hero re-registers the same id/desc),
// a PLAYER_REG on team 1, CREATE_OBJECT, the bind, and an initial position+speed
// SYNC so the run animation plays. No combat effectors/stats (the square has none).
// No-op if viewer already tracks owner. Caller holds the square lock.
func (s *Server) renderLobbyAvatarForLocked(viewer, owner *conn, now float64) {
	tr := viewer.renderTr()
	if tr == nil || viewer == owner {
		return
	}
	if tr.index(owner.objID) >= 0 {
		return
	}
	s.push(viewer, battleproto.CmdPrototypeInfo, amf.NewArray().
		Set("id", heroPrototypeID).Set("desc", heroProtoDesc))
	s.push(viewer, battleproto.CmdPlayerReg, amf.NewArray().
		Set("id", owner.selfPlayerID).Set("name", owner.name).
		Set("team", int32(1)).Set("avatar", int32(-1)))
	s.push(viewer, battleproto.CmdCreateObject, amf.NewArray().
		Set("id", owner.objID).Set("proto", heroPrototypeID))
	s.push(viewer, battleproto.CmdSetAvatar, amf.NewArray().
		Set("playerID", owner.selfPlayerID).Set("avatarID", owner.objID).
		Set("level", int32(1)).Set("points", int32(0)))

	idx := tr.add(owner.objID)
	bt := float32(now)
	ox, oy := owner.posAtLocked(bt)
	s.push(viewer, battleproto.CmdSync, amf.NewArray().Set("data",
		newSyncBlob(bt).addObject(owner.objID).
			position(idx, ox, oy, owner.vx, owner.vy, bt).
			setFloats(syncSpeed, idx, lobbyMoveSpeed).
			build(tr.count())))
}

// removeLobbyAvatarForLocked drops owner's avatar from viewer's client (owner left):
// tracker swap-remove + DELETE_OBJECT, via the shared untrack helper.
func (s *Server) removeLobbyAvatarForLocked(viewer, owner *conn) {
	s.untrackObjForMemberLocked(viewer, owner.objID, s.battleTime())
}

// introduceLobbyMemberLocked wires a freshly-ready occupant into the square: it
// shows every existing occupant to the newcomer and the newcomer to them. Caller
// holds the square lock. Mirrors introduceMemberLocked (minus mobs/summons).
//
// Hero appearance is seeded FIRST over MPD (hero.get_data_list_mpd): a post-load
// avatar bind (UpdateHeroesInfo _send=false) renders the body only from the client's
// pre-populated mHeroesWait, never by requesting it. The Battle cross-render that
// follows creates the avatar object and starts its movement. Pushing the hero data
// before the avatar chain gives the (separate) MPD socket a head start so the data is
// waiting when the SET_AVATAR is bound.
func (s *Server) introduceLobbyMemberLocked(c *conn, now float64) {
	var others []*conn
	for _, other := range c.members() {
		if other != c {
			others = append(others, other)
		}
	}
	if s.MPD != nil {
		existing := make([]int32, 0, len(others))
		for _, other := range others {
			existing = append(existing, other.selfPlayerID)
		}
		s.MPD.PushHeroData(c.selfPlayerID, existing) // existing occupants -> newcomer
		for _, other := range others {
			s.MPD.PushHeroData(other.selfPlayerID, []int32{c.selfPlayerID}) // newcomer -> each
		}
	}
	for _, other := range others {
		s.renderLobbyAvatarForLocked(c, other, now) // existing player -> newcomer's client
		s.renderLobbyAvatarForLocked(other, c, now) // newcomer -> existing player's client
	}
}
