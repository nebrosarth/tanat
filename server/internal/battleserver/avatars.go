package battleserver

import (
	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
)

// Cross-player rendering: a shared hunt world only looks shared if every member
// can see the others' avatars move and fight. Each member already renders its own
// avatar (the self-player world-state chain in sendHuntWorldState); this file
// makes each OTHER member's avatar appear on a viewer's client, tracked in that
// viewer's own object list, and keeps it in sync as they move / attack / die.

// renderAvatarForLocked builds owner's avatar on viewer's client: the avatar
// prototype (idempotent -- re-registering the same id/desc is harmless), a PLAYER_REG
// on owner's own team (playerTeam(): the co-op side in Hunt, but owner's real side in
// «Арена»/«Штурм», so a viewer sees a teammate as FRIEND and an enemy hero as ENEMY),
// CREATE_OBJECT, the bind, and an initial position+stats SYNC. No-op if viewer already
// tracks owner.
func (s *Server) renderAvatarForLocked(viewer, owner *conn, now float64) {
	s.renderAvatarForLockedWithStats(viewer, owner, now, false)
}

// renderAvatarWithStatsLocked is the fresh-registration path used when an avatar becomes
// visible to a client. Hard-reset callers keep renderAvatarForLocked's packet contract, while
// newly introduced/re-rendered avatars also receive the current PlayerStore status.
func (s *Server) renderAvatarWithStatsLocked(viewer, owner *conn, now float64) {
	s.renderAvatarForLockedWithStats(viewer, owner, now, true)
}

func (s *Server) renderAvatarForLockedWithStats(viewer, owner *conn, now float64, withStats bool) {
	vh, oh := viewer.huntState, owner.huntState
	if vh == nil || oh == nil || viewer == owner {
		return
	}
	if vh.tr.index(owner.objID) >= 0 {
		return
	}
	a := oh.av
	proto := avatarProtoID(a.ID)
	s.push(viewer, battleproto.CmdPrototypeInfo, amf.NewArray().
		Set("id", proto).Set("desc", avatarProtoDesc(a)))
	s.push(viewer, battleproto.CmdPlayerReg, amf.NewArray().
		Set("id", owner.selfPlayerID).Set("name", owner.name).
		Set("team", owner.playerTeam()).Set("avatar", a.ID))
	s.push(viewer, battleproto.CmdCreateObject, amf.NewArray().
		Set("id", owner.objID).Set("proto", proto))
	s.push(viewer, battleproto.CmdSetAvatar, amf.NewArray().
		Set("playerID", owner.selfPlayerID).Set("avatarID", owner.objID).
		Set("level", oh.level).Set("points", int32(0)))

	// Register the avatar's ATTACK effector on the viewer's client (proto + a bound
	// effector on the rendered object), mirroring the mob attack effector in
	// revealMobToMemberLocked. Without it the viewer can't resolve owner's
	// basic-attack ACTION (action = attackProtoID) and plays no swing/projectile.
	s.push(viewer, battleproto.CmdPrototypeInfo, amf.NewArray().
		Set("id", attackProtoID(a)).Set("desc", attackProtoDesc(a)))
	s.addAttackEffectorLocked(viewer, owner.objID, attackProtoID(a), now)

	idx := vh.tr.add(owner.objID)
	bt := float32(now)
	ox, oy := owner.posAtLocked(bt)
	maxHP := oh.maxHPLocked(now)
	maxMana := oh.maxManaLocked(now)
	hpFrac, manaFrac := float32(1), float32(1)
	if maxHP > 0 {
		hpFrac = float32(oh.hp / maxHP)
	}
	if maxMana > 0 {
		manaFrac = float32(oh.mana / maxMana)
	}
	s.push(viewer, battleproto.CmdSync, amf.NewArray().Set("data",
		newSyncBlob(bt).addObject(owner.objID).
			position(idx, ox, oy, owner.vx, owner.vy, bt).
			setFloats(syncHealth, idx, hpFrac).
			setFloats(syncMaxHealth, idx, float32(maxHP)).
			setFloats(syncMana, idx, manaFrac).
			setFloats(syncMaxMana, idx, float32(maxMana)).
			setFloats(syncSpeed, idx, lobbyMoveSpeed).
			// ATTACK_SPEED is load-bearing for the swing animation: the client scales the
			// attack clip's playback speed by mAttackSpeed (animState.speed = length *
			// attackSpeed). Without this the teammate's copy keeps mAttackSpeed=0 and the
			// swing clip is frozen at one frame -- the avatar moves but never visibly
			// attacks. Mirrors the mob reveal (mobai.go) and the self world-state (hunt.go).
			setFloats(syncAttackSpeed, idx, float32(a.AttackSpeed)).
			setFloats(syncAttackRange, idx, float32(a.AttackRange)).
			setFloats(syncDmgMin, idx, float32(a.DmgMin)).
			setFloats(syncDmgMax, idx, float32(a.DmgMax)).
			setFloats(syncRadius, idx, float32(a.Radius())).
			// Vision: WarFogObject.Update only spawns a reveal zone once mViewRadius > 0
			// AND friendliness is known (see the self-avatar's identical field in hunt.go's
			// world-state chain). This sync never carried it, so an ALLY's avatar rendered
			// on a teammate's client -- unlike the self avatar, every mob, every summon and
			// every structure, all of which already send it -- lit nothing: a teammate
			// standing in fog stayed fogged from every other member's point of view even
			// though their own model was right there.
			setFloats(syncViewRadius, idx, effectiveViewRadius(avatarViewRadius)).
			setInt(syncTeam, idx, owner.playerTeam()).
			build(vh.tr.count())))
	// ADD_TO_INVENTORY is owner-local, while EnemyInfoWindow reads the remote
	// Player.ActiveItems list. Replay the authoritative tree ownership after the
	// avatar is registered so late reveals show the same items as live purchases.
	s.replayAvatarItemsToViewerLocked(viewer, owner)
	if withStats {
		s.pushPlayerStatsToViewerLocked(viewer, owner)
	}
}

func playerStatsArgs(owner *conn) *amf.MixedArray {
	hs := owner.huntState
	return amf.NewArray().
		Set("id", owner.selfPlayerID).
		Set("level", hs.level).
		Set("kills", hs.frags).
		Set("deaths", hs.deaths).
		Set("killer", hs.lastKiller).
		Set("assists", hs.assists)
}

func (s *Server) pushPlayerStatsToViewerLocked(viewer, owner *conn) {
	if viewer == nil || owner == nil || viewer.huntState == nil || owner.huntState == nil {
		return
	}
	if viewer.huntState.tr.index(owner.objID) < 0 {
		return
	}
	s.push(viewer, battleproto.CmdPlayerStats, playerStatsArgs(owner))
}

// removeAvatarForLocked drops owner's avatar from viewer's client (owner left, or
// the world is tearing down): tracker swap-remove + DELETE_OBJECT.
func (s *Server) removeAvatarForLocked(viewer, owner *conn) {
	s.untrackObjForMemberLocked(viewer, owner.objID, s.battleTime())
}

// hardResetAvatarForLocked recreates an owner's already-rendered avatar on every
// remote viewer. A fresh object's first POSITION runs the client's render reset,
// avoiding SmoothErrorCorrector catch-up after a hard relocation. The tracker
// guard is intentional: an unseen avatar must stay unseen behind Dota fog.
func (s *Server) hardResetAvatarForLocked(owner *conn, now float64) {
	if owner == nil || owner.huntState == nil {
		return
	}
	for _, viewer := range owner.members() {
		if viewer == owner || viewer.huntState == nil || viewer.huntState.tr.index(owner.objID) < 0 {
			continue
		}
		s.removeAvatarForLocked(viewer, owner)
		s.renderAvatarForLocked(viewer, owner, now)
	}
}

// introduceMemberLocked wires a freshly-joined member into the shared world's
// rendering: it shows every existing member to the newcomer and the newcomer to
// them, and reveals the already-visible mobs (which the newcomer's own world
// state didn't include -- the fog pass only reveals on a not-shown -> shown
// transition, so a mob already shown to the party needs an explicit reveal here).
func (s *Server) introduceMemberLocked(c *conn, now float64) {
	// «Штурм» fog: an enemy avatar/summon/creep is left untracked here and picked up by
	// dotaVisionPassLocked (vision.go) the moment the newcomer's own side actually has
	// vision of it -- the same delay every other unit's fog gets, instead of the
	// newcomer seeing the whole enemy team the instant they connect. Arena has no
	// vision pass yet (inst.dota is nil there), so it keeps showing enemies immediately,
	// unchanged.
	dotaFog := c.inst != nil && c.inst.dota != nil
	for _, other := range c.members() {
		if other == c || (dotaFog && arenaEnemies(c, other)) {
			continue
		}
		s.renderAvatarWithStatsLocked(c, other, now) // existing player -> newcomer's client
		s.renderAvatarWithStatsLocked(other, c, now) // newcomer -> existing player's client
		// Show that player's live summons to the newcomer (the newcomer has none yet,
		// so this is one-directional). sm.alive is the shared liveness test, and the
		// lazily-set sm.dead flag alone would not do: a summon that took a lethal hit or
		// expired this tick isn't reaped until the owner's next tick, and revealing it
		// here would pop a 0-HP model onto the newcomer that vanishes (no death anim,
		// since it never saw the ON_KILL) a tick later.
		for _, sm := range other.huntState.summons {
			if sm.alive(now) {
				s.revealSummonToMemberLocked(c, sm, now)
			}
		}
	}
	for _, m := range c.huntState.mobs {
		if !m.shown || m.dead {
			continue
		}
		if dotaFog && m.enemyOf(c.playerTeam()) {
			continue // enemy creep: dotaVisionPassLocked reveals it once in range
		}
		s.revealMobToMemberLocked(c, m, now)
	}
}
