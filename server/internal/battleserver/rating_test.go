package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// TestDotaHeroKillCreditsKillDeathAssist: a «Штурм» hero kill must update the scoreboard
// counters (huntState.frags/deaths/assists) the same way it already pays XP/gold -- the
// end-of-match table (fight|log) reads straight off these, so a kill that doesn't
// increment them is exactly the "table comes up empty" bug reported live, just for the
// K/D/A columns instead of the whole response.
func TestDotaHeroKillCreditsKillDeathAssist(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, human.x+1, human.y)
	// A second ally close enough to share the kill's XP -- see dotaXPShareRadius --
	// must be credited an assist, not a kill.
	ally := dotaPlayerConn(t, s, inst, 1002, dotaTeamHuman, human.x+2, human.y)
	elf.huntState.hp = 10
	now := float64(s.battleTime())

	human.lock()
	defer human.unlock()
	s.hitPlayerFromLocked(elf, human.objID, 1000, now, nil, human)

	if elf.huntState.deadUntil == 0 {
		t.Fatal("enemy hero did not die to the lethal blow")
	}
	if human.huntState.frags != 1 {
		t.Errorf("killer frags = %d, want 1", human.huntState.frags)
	}
	if elf.huntState.deaths != 1 {
		t.Errorf("victim deaths = %d, want 1", elf.huntState.deaths)
	}
	if ally.huntState.assists != 1 {
		t.Errorf("nearby ally assists = %d, want 1", ally.huntState.assists)
	}
	if human.huntState.assists != 0 {
		t.Errorf("killer's own assists = %d, want 0 (the kill itself is not also an assist)", human.huntState.assists)
	}
}

// TestSettleMatchAppliesEloRatingAndFightLog: dotaEndLocked's settlement applies a
// symmetric Elo rating swing to both real participants and publishes a fight-log entry,
// under EACH participant's own battleID (every connection gets a different one -- see
// conn.battleID), with the match's real K/D data.
func TestSettleMatchAppliesEloRatingAndFightLog(t *testing.T) {
	store := session.NewStore()
	s := New(store)
	winnerUser, _, _ := store.LoginOrRegister("winner@test.io", "pw")
	store.CreateHero(winnerUser, 0, false, 0, 0, 0, 0, 0)
	loserUser, _, _ := store.LoginOrRegister("loser@test.io", "pw")
	store.CreateHero(loserUser, 0, false, 0, 0, 0, 0, 0)

	dm := gamedata.DotaMaps()[0]
	inst := newDotaInstance(s, dm.ID, dm.ID)
	winner := dotaPlayerConn(t, s, inst, 1000, dotaTeamHuman, 0, 0)
	winner.selfPlayerID, winner.battleID = winnerUser.ID, 5001
	winner.huntState.frags, winner.huntState.assists = 3, 1
	loser := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, 0, 0)
	loser.selfPlayerID, loser.battleID = loserUser.ID, 5002
	loser.huntState.deaths = 2

	winner.lock()
	now := float64(s.battleTime())
	s.dotaEndLocked(winner, dotaTeamHuman, now)
	winner.unlock()

	newWinnerRating, _ := store.HeroRating(winnerUser.ID)
	newLoserRating, _ := store.HeroRating(loserUser.ID)
	if newWinnerRating <= session.RatingDefault {
		t.Errorf("winner rating = %d, want > %d (a win must never lower or leave rating unchanged)", newWinnerRating, session.RatingDefault)
	}
	if newLoserRating >= session.RatingDefault {
		t.Errorf("loser rating = %d, want < %d", newLoserRating, session.RatingDefault)
	}
	gained, lost := newWinnerRating-session.RatingDefault, session.RatingDefault-newLoserRating
	if gained != lost {
		t.Errorf("winner gained %d but loser lost %d, want a symmetric swing (equal starting ratings)", gained, lost)
	}

	entries, ok := store.FightLog(winner.battleID)
	if !ok {
		t.Fatal("no fight log published under the winner's own battleID")
	}
	we, ok := entries[winner.selfPlayerID]
	if !ok || we.Kills != 3 || we.Assists != 1 {
		t.Errorf("winner fight-log entry = %+v, want Kills=3 Assists=1", we)
	}
	le, ok := entries[loser.selfPlayerID]
	if !ok || le.Deaths != 2 {
		t.Errorf("loser fight-log entry = %+v, want Deaths=2", le)
	}
	if entriesForLoser, ok := store.FightLog(loser.battleID); !ok || len(entriesForLoser) != len(entries) {
		t.Error("the loser's own battleID did not resolve to the same shared scoreboard")
	}
}

// TestSettleMatchSkipsRatingForBotMatchWhenFlagOff: TANAT_RATE_BOT_MATCHES=0 must stop a
// bot-filled match's outcome from touching a real player's rating at all -- the whole
// point of the opt-out.
func TestSettleMatchSkipsRatingForBotMatchWhenFlagOff(t *testing.T) {
	t.Setenv("TANAT_RATE_BOT_MATCHES", "0")

	store := session.NewStore()
	s := New(store)
	winnerUser, _, _ := store.LoginOrRegister("nobotrating@test.io", "pw")
	store.CreateHero(winnerUser, 0, false, 0, 0, 0, 0, 0)

	dm := gamedata.DotaMaps()[0]
	inst := newDotaInstance(s, dm.ID, dm.ID)
	winner := dotaPlayerConn(t, s, inst, 1000, dotaTeamHuman, 0, 0)
	winner.selfPlayerID = winnerUser.ID
	dotaPlayerConn(t, s, inst, botIDBase, dotaTeamElf, 0, 0) // a bot opponent

	winner.lock()
	now := float64(s.battleTime())
	s.dotaEndLocked(winner, dotaTeamHuman, now)
	winner.unlock()

	if r, _ := store.HeroRating(winnerUser.ID); r != session.RatingDefault {
		t.Errorf("winner rating = %d after a bot-filled match with the opt-out flag off, want unchanged %d", r, session.RatingDefault)
	}
}

// TestSettleMatchRatesBotMatchByDefault: with TANAT_RATE_BOT_MATCHES unset (the default),
// a bot-filled match still settles rating for its real player -- bots are meant to make a
// short queue feel like a real match, progression included, unless explicitly opted out.
func TestSettleMatchRatesBotMatchByDefault(t *testing.T) {
	t.Setenv("TANAT_RATE_BOT_MATCHES", "1")

	store := session.NewStore()
	s := New(store)
	winnerUser, _, _ := store.LoginOrRegister("botratingon@test.io", "pw")
	store.CreateHero(winnerUser, 0, false, 0, 0, 0, 0, 0)

	dm := gamedata.DotaMaps()[0]
	inst := newDotaInstance(s, dm.ID, dm.ID)
	winner := dotaPlayerConn(t, s, inst, 1000, dotaTeamHuman, 0, 0)
	winner.selfPlayerID = winnerUser.ID
	dotaPlayerConn(t, s, inst, botIDBase, dotaTeamElf, 0, 0) // a bot opponent

	winner.lock()
	now := float64(s.battleTime())
	s.dotaEndLocked(winner, dotaTeamHuman, now)
	winner.unlock()

	if r, _ := store.HeroRating(winnerUser.ID); r <= session.RatingDefault {
		t.Errorf("winner rating = %d after beating a bot-filled team with rating ON, want > %d", r, session.RatingDefault)
	}
}
