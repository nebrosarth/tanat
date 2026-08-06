package battleserver

// End-of-match settlement shared by every PvP mode: builds the scoreboard the client asks
// for right after BATTLE_END (fight|log, via session.Store.SetFightLog -- see
// session/fightlog.go for why that handoff exists) and, alongside it, applies a
// team-average Elo rating adjustment. Called once from dotaEndLocked (which also covers a
// «Битва за замок» siege, itself funnelled through dotaEndLocked -- see castle.go) and
// once from arenaEndLocked, the two choke points every PvP match ends through.

import (
	"math"

	"tanatserver/internal/session"
)

// ratingKFactor is the Elo K-factor: the maximum rating swing a single match can cause
// (at a 100%-vs-0%-expected blowout). 32 is the common default for a "point win/loss"
// game (chess federations use 16-32 depending on player experience); nothing here is
// tuned against real match data yet, so this is a reasonable, easily-revisited starting
// point rather than a derived constant.
const ratingKFactor = 32.0

// settleMatchLocked builds this match's fight-log rows for every participant and, if both
// sides are non-empty, adjusts every REAL (non-bot) participant's persistent rating.
// Caller holds inst.mu (both callers are already deep inside the world lock).
func (s *Server) settleMatchLocked(inst *huntInstance, winnerTeam int32, now float64) {
	var winners, losers []*conn
	hasBots := false
	for _, mem := range inst.members {
		if mem.huntState == nil {
			continue
		}
		if isBotConn(mem) {
			hasBots = true
		}
		if mem.playerTeam() == winnerTeam {
			winners = append(winners, mem)
		} else {
			losers = append(losers, mem)
		}
	}
	if len(winners) == 0 && len(losers) == 0 {
		return
	}

	// A one-sided "match" (e.g. a solo dev launch that pushed down an empty enemy base
	// with nobody -- not even a bot -- on the other side) has no opponent to compute an
	// Elo expected score against, so it settles the scoreboard but never touches rating.
	delta := int32(0)
	if len(winners) > 0 && len(losers) > 0 && (!hasBots || dotaRateBotMatches()) {
		delta = s.eloDeltaLocked(winners, losers)
	}

	entries := map[int32]session.FightLogEntry{}
	s.buildFightLogTeamLocked(entries, winners, delta, now)
	s.buildFightLogTeamLocked(entries, losers, -delta, now)

	for _, mem := range winners {
		if mem.battleID != 0 {
			s.Store.SetFightLog(mem.battleID, entries)
		}
	}
	for _, mem := range losers {
		if mem.battleID != 0 {
			s.Store.SetFightLog(mem.battleID, entries)
		}
	}
}

// eloDeltaLocked returns the rating points the winning side gains (and the losing side
// loses) this match: the standard Elo expected-score formula over each side's average
// rating, scaled by ratingKFactor and floored at 0 -- the winning side never LOSES rating
// for winning, even against a side that badly outrated it (Elo's formula alone can't
// produce a negative delta for the winner, but rounding at exactly 0 could without the
// floor spelling out the invariant explicitly).
func (s *Server) eloDeltaLocked(winners, losers []*conn) int32 {
	avgWin := s.avgRatingLocked(winners)
	avgLose := s.avgRatingLocked(losers)
	expectedWin := 1.0 / (1.0 + math.Pow(10, (avgLose-avgWin)/400.0))
	delta := int32(math.Round(ratingKFactor * (1.0 - expectedWin)))
	if delta < 0 {
		delta = 0
	}
	return delta
}

// avgRatingLocked is a side's average persistent rating. A bot (isBotConn) contributes
// session.RatingDefault -- it has no rating of its own worth reading -- so a bot-filled
// side is a neutral opponent rather than a free-points or punishing one relative to the
// default rating band.
func (s *Server) avgRatingLocked(side []*conn) float64 {
	if len(side) == 0 {
		return float64(session.RatingDefault)
	}
	sum := 0.0
	for _, mem := range side {
		if isBotConn(mem) {
			sum += float64(session.RatingDefault)
			continue
		}
		if r, ok := s.Store.HeroRating(mem.selfPlayerID); ok {
			sum += float64(r)
		} else {
			sum += float64(session.RatingDefault)
		}
	}
	return sum / float64(len(side))
}

// buildFightLogTeamLocked adds one session.FightLogEntry per member of side to entries,
// applying delta to every real member's persistent rating (a bot's row simply shows its
// current/default rating twice -- no change, since it has nothing worth adjusting). now is
// accepted (unused today) for a future per-match "Time" field, matching every other
// *Locked helper's signature convention in this package.
func (s *Server) buildFightLogTeamLocked(entries map[int32]session.FightLogEntry, side []*conn, delta int32, now float64) {
	for _, mem := range side {
		hs := mem.huntState
		old, cur := session.RatingDefault, session.RatingDefault
		if !isBotConn(mem) {
			if o, n, ok := s.Store.ApplyHeroRatingDelta(mem.selfPlayerID, delta); ok {
				old, cur = o, n
			}
		}
		money, _, _ := s.Store.HeroMoney(mem.selfPlayerID)
		entries[mem.selfPlayerID] = session.FightLogEntry{
			AvatarID:  mem.selfPlayerID,
			Nick:      mem.name,
			Team:      mem.playerTeam(),
			Kills:     hs.frags,
			Assists:   hs.assists,
			Deaths:    hs.deaths,
			Level:     hs.level,
			Money:     money,
			OldRating: old,
			NewRating: cur,
		}
	}
}
