package battleserver

// «Штурм» fog-of-war: reported live as "I see enemy heroes and creeps through the
// fog". The map bakes a real WarFogPlane (see the avatarViewRadius doc comment in
// hunt.go), but the plane only ever dims the GROUND texture -- nothing in the client
// hides a unit's own model when it stands on fogged ground (WarFogObject.Update just
// decides whether an object gets to PUNCH a hole in that texture; an ENEMY-team object
// never does, but nothing there ever sets its own renderer invisible either -- see
// WarFogObject.cs). So "seeing through fog" was never a client bug: the server was
// simply handing every member's client a CREATE_OBJECT for literally every enemy
// avatar, creep and summon the instant it existed, on both sides, unconditionally
// (avatars.go's renderAvatarForLocked/introduceMemberLocked, dota.go's
// spawnCreepWaveLocked, mobai.go's summonLocked all fanned out over the WHOLE
// instance, not the caster's own team). A real "is this on my object list at all"
// gate never existed for the PvP sides the way Hunt's mobInterestLocked has always
// gated its own mobs.
//
// This file is that gate for «Штурм» (and «Битва за замок», which reuses the same
// dotaTickLocked): once per tick, each side's living heroes/creeps/structures light a
// vision circle, and every ENEMY hero/creep/summon is revealed or hidden on that
// side's clients depending on whether it falls inside one. Structures are exempt --
// a «Штурм» base stays visible on the map the way Dota's own buildings do, and
// nothing here ever touches m.active: simulation already runs unconditionally
// (dotaTickLocked's own comment), and this pass only ever decides what a tracker
// shows, never what the world does.
//
// Arena has no equivalent pass yet (it never set inst.dota, so dotaTickLocked never
// runs for it) -- out of scope here; every reveal path this file's callers touch is
// explicitly guarded so Arena keeps its current "enemy is shown immediately" behaviour
// unchanged.

// dotaVisionHysteresis widens a source's own radius before an ALREADY-visible enemy
// unit is judged out of range. Without it, a unit sitting near the boundary of a
// moving source (an escorting creep, a walking hero) would flicker
// CREATE_OBJECT/DELETE_OBJECT every tick as its distance ticks back and forth across
// the same radius.
const dotaVisionHysteresis float32 = 4.0

// dotaVisionSource is one living unit granting its team a circle of vision.
type dotaVisionSource struct {
	x, y, r float32
}

// dotaVisionSourcesLocked collects every vision source `team` currently owns: its
// living heroes (avatarViewRadius, matching what their own avatar reveal sync
// carries), their living summons (summonViewRadius), living creeps (creepViewRadius)
// and living structures (structViewRadius, matching dotaRevealStructureLocked). A dead
// hero grants no vision -- it neither moves nor fights, so there is nothing to look
// from until it respawns.
func (s *Server) dotaVisionSourcesLocked(inst *huntInstance, team int32, now float64) []dotaVisionSource {
	var out []dotaVisionSource
	bt := float32(now)
	for _, mem := range inst.members {
		hs := mem.huntState
		if hs == nil || mem.playerTeam() != team || hs.deadUntil > 0 {
			continue
		}
		px, py := mem.posAtLocked(bt)
		out = append(out, dotaVisionSource{px, py, avatarViewRadius})
		for _, sm := range hs.summons {
			if sm.alive(now) {
				out = append(out, dotaVisionSource{sm.x, sm.y, summonViewRadius})
			}
		}
	}
	for _, m := range inst.mobs {
		if m.dead || m.teamVal() != team {
			continue
		}
		r := creepViewRadius
		if m.structure {
			r = structViewRadius
		}
		out = append(out, dotaVisionSource{m.x, m.y, r})
	}
	return out
}

// dotaVisibleLocked reports whether (x,y) sits inside any source's vision circle.
// already widens every circle by dotaVisionHysteresis first -- see its doc comment.
func dotaVisibleLocked(sources []dotaVisionSource, x, y float32, already bool) bool {
	for _, src := range sources {
		r := src.r
		if already {
			r += dotaVisionHysteresis
		}
		dx, dy := x-src.x, y-src.y
		if dx*dx+dy*dy <= r*r {
			return true
		}
	}
	return false
}

// dotaApplyTeamVisionLocked reveals or hides every ENEMY hero, summon and creep on
// viewerTeam's own clients, using sources (viewerTeam's own living units this tick).
func (s *Server) dotaApplyTeamVisionLocked(inst *huntInstance, viewerTeam int32, sources []dotaVisionSource, now float64) {
	bt := float32(now)
	for _, mem := range inst.members {
		if mem.huntState == nil || mem.playerTeam() != viewerTeam {
			continue
		}
		for _, other := range inst.members {
			if other == mem || other.huntState == nil || other.playerTeam() == viewerTeam {
				continue
			}
			ox, oy := other.posAtLocked(bt)
			tracked := mem.huntState.tr.index(other.objID) >= 0
			switch {
			case dotaVisibleLocked(sources, ox, oy, tracked) && !tracked:
				s.renderAvatarForLocked(mem, other, now)
			case !dotaVisibleLocked(sources, ox, oy, tracked) && tracked:
				s.removeAvatarForLocked(mem, other)
			}
			for _, sm := range other.huntState.summons {
				if !sm.alive(now) {
					continue
				}
				smTracked := mem.huntState.tr.index(sm.id) >= 0
				switch {
				case dotaVisibleLocked(sources, sm.x, sm.y, smTracked) && !smTracked:
					s.revealSummonToMemberLocked(mem, sm, now)
				case !dotaVisibleLocked(sources, sm.x, sm.y, smTracked) && smTracked:
					s.hideSummonFromMemberLocked(mem, sm, now)
				}
			}
		}
		for _, m := range inst.mobs {
			if m.structure || m.dead || m.teamVal() == viewerTeam {
				continue
			}
			tracked := mem.huntState.tr.index(m.id) >= 0
			switch {
			case dotaVisibleLocked(sources, m.x, m.y, tracked) && !tracked:
				s.revealMobToMemberLocked(mem, m, now)
			case !dotaVisibleLocked(sources, m.x, m.y, tracked) && tracked:
				s.untrackObjForMemberLocked(mem, m.id, bt)
			}
		}
	}
}

// dotaVisionPassLocked is «Штурм»'s per-tick fog-of-war reveal, run once per instance
// tick from dotaTickLocked -- the same cadence Hunt's mobInterestLocked uses for its
// own reveal/hide gate.
func (s *Server) dotaVisionPassLocked(inst *huntInstance, now float64) {
	human := s.dotaVisionSourcesLocked(inst, dotaTeamHuman, now)
	elf := s.dotaVisionSourcesLocked(inst, dotaTeamElf, now)
	s.dotaApplyTeamVisionLocked(inst, dotaTeamHuman, human, now)
	s.dotaApplyTeamVisionLocked(inst, dotaTeamElf, elf, now)
}

// dotaRevealCreepToOwnTeamLocked renders a freshly spawned «Штурм» creep on its own
// team's clients only. The enemy side sees it once dotaVisionPassLocked finds it
// within vision on a later tick -- the same delay Dota's own fog gives a wave nobody
// is watching yet. Mirrors revealMobLocked (shown=true + lastSync bookkeeping so
// introduceMemberLocked and the leash/despawn invariants still see a normal mob), just
// scoped to c.members() on m's own side instead of the whole instance.
func (s *Server) dotaRevealCreepToOwnTeamLocked(c *conn, m *mobState, now float64) {
	m.shown = true
	for _, mem := range c.members() {
		if mem.huntState == nil || mem.playerTeam() != m.teamVal() {
			continue
		}
		s.revealMobToMemberLocked(mem, m, now)
	}
	m.lastSync = now
}
