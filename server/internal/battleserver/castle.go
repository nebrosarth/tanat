package battleserver

// «Битва за замок» settlement: called once, from dotaEndLocked, when a «Штурм»-shaped
// match that was launched as a castle siege (dotaState.castleID != 0, set from
// session.PendingBattle.CastleID at avatar spawn -- see hunt.go) ends. Ownership
// transfer, the history-log entry, and the reward payout all land in ONE
// session.Store commit (session/castle.go's SettleCastleBattle), mirroring the
// "state change and payout together" guarantee CompleteQuest uses for quests -- a
// crash can't leave the castle transferred with nobody paid, or vice versa.
//
// The battle itself is mechanically identical to «Штурм» (same map10 structures,
// teams, and altar-fall win check); this file is only the castle-specific tail.

import (
	"log"
	"time"

	"tanatserver/internal/gamedata"
)

// castleCheckWinLocked ends a «Битва за замок» siege once every defending
// GA_ClanWars_Gun structure is dead: the attacker (team 1) wins. Called from
// dotaCheckWinLocked exactly when the instance's DotaMap has no altar at all (map_6_0
// has none -- the "Castle" object itself is confirmed static scenery with no combat
// stats, see gamedata/dota.go's map60 doc comment). There is no symmetric
// defender-win structure: the defending side holds simply by keeping any gun alive.
func (s *Server) castleCheckWinLocked(rep *conn, now float64) {
	for _, m := range rep.inst.mobs {
		if m.structure && m.dotaRole == gamedata.DotaGun && !m.dead {
			return
		}
	}
	s.dotaEndLocked(rep, dotaTeamHuman, now)
}

// settleCastleBattleLocked credits gamedata.Castle.Reward to every member on the
// winning team and transfers ownership to the majority clan on that side (a winning
// side with no clan members at all -- e.g. a solo test fight -- still pays its
// fighters but leaves ownership unchanged). Caller holds the instance lock (called
// from dotaEndLocked, itself always invoked under it).
func (s *Server) settleCastleBattleLocked(inst *huntInstance, castleID int32, winnerTeam int32) {
	c, ok := gamedata.CastleByID(castleID)
	if !ok {
		log.Printf("battle: castle %d settle: unknown castle id, no payout", castleID)
		return
	}

	var winnerFighters []int32
	clanVotes := map[int32]int32{} // clanID -> count of winning fighters in that clan
	for _, mem := range inst.members {
		if mem.huntState == nil || mem.huntState.team != winnerTeam {
			continue
		}
		uid := mem.selfPlayerID
		winnerFighters = append(winnerFighters, uid)
		if clanID, _ := s.Store.HeroClanInfo(uid); clanID != 0 {
			clanVotes[clanID]++
		}
	}

	var winnerClanID, best int32
	for clanID, n := range clanVotes {
		if n > best {
			best, winnerClanID = n, clanID
		}
	}

	s.Store.SettleCastleBattle(castleID, winnerClanID, winnerFighters, c.Reward, time.Now().Unix())
	log.Printf("battle: castle %d («%s») settled: winner team=%d clan=%d fighters_paid=%d",
		castleID, c.NameKey, winnerTeam, winnerClanID, len(winnerFighters))
}
