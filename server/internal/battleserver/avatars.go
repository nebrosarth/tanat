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
// prototype (idempotent -- re-registering the same id/desc is harmless), a
// PLAYER_REG on team 1 (allies -- this is co-op PvE), CREATE_OBJECT, the bind, and
// an initial position+stats SYNC. No-op if viewer already tracks owner.
func (s *Server) renderAvatarForLocked(viewer, owner *conn, now float64) {
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
		Set("team", int32(1)).Set("avatar", a.ID))
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
			setInt(syncTeam, idx, 1).
			build(vh.tr.count())))
}

// removeAvatarForLocked drops owner's avatar from viewer's client (owner left, or
// the world is tearing down): tracker swap-remove + DELETE_OBJECT.
func (s *Server) removeAvatarForLocked(viewer, owner *conn) {
	s.untrackObjForMemberLocked(viewer, owner.objID, s.battleTime())
}

// introduceMemberLocked wires a freshly-joined member into the shared world's
// rendering: it shows every existing member to the newcomer and the newcomer to
// them, and reveals the already-visible mobs (which the newcomer's own world
// state didn't include -- the fog pass only reveals on a not-shown -> shown
// transition, so a mob already shown to the party needs an explicit reveal here).
func (s *Server) introduceMemberLocked(c *conn, now float64) {
	for _, other := range c.members() {
		if other == c {
			continue
		}
		s.renderAvatarForLocked(c, other, now) // existing player -> newcomer's client
		s.renderAvatarForLocked(other, c, now) // newcomer -> existing player's client
		// Show that player's live summons to the newcomer (the newcomer has none yet,
		// so this is one-directional). Match tickSummonsLocked's ACTUAL liveness test
		// (hp>0 && not expired), not just the lazily-set sm.dead flag: a summon that
		// took a lethal hit or expired this tick isn't reaped until the owner's next
		// tick, and revealing it here would pop a 0-HP model onto the newcomer that
		// vanishes (no death anim, since it never saw the ON_KILL) a tick later.
		for _, sm := range other.huntState.summons {
			if !sm.dead && sm.hp > 0 && sm.until >= now {
				s.revealSummonToMemberLocked(c, sm, now)
			}
		}
	}
	for _, m := range c.huntState.mobs {
		if m.shown && !m.dead {
			s.revealMobToMemberLocked(c, m, now)
		}
	}
}
