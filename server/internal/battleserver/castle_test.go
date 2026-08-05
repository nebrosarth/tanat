package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// TestCastleSettlementTransfersOwnershipAndPaysRewards: when a «Штурм»-shaped match
// tagged with a castleID ends, the winning side's clan becomes the new owner and every
// winning fighter is credited the castle's CastleReward -- all in one Store commit
// (session.Store.SettleCastleBattle). The losing side is untouched.
func TestCastleSettlementTransfersOwnershipAndPaysRewards(t *testing.T) {
	store := session.NewStore()
	s := New(store)

	winU, _, ok := store.LoginOrRegister("winner@example.com", "pw")
	if !ok {
		t.Fatal("register winner")
	}
	store.CreateHero(winU, 1, false, 0, 0, 0, 0, 0)
	loseU, _, ok := store.LoginOrRegister("loser@example.com", "pw")
	if !ok {
		t.Fatal("register loser")
	}
	store.CreateHero(loseU, 1, false, 0, 0, 0, 0, 0)
	store.AddHeroMoney(winU.ID, session.ClanCreatePrice)
	store.AddHeroMoney(loseU.ID, session.ClanCreatePrice)
	winClanID, code := store.CreateClan(winU.ID, "WinnersClan", "WIN")
	if code != session.ClanOK {
		t.Fatalf("create winner clan: %d", code)
	}
	if _, code := store.CreateClan(loseU.ID, "LosersClan", "LOSE"); code != session.ClanOK {
		t.Fatalf("create loser clan: %d", code)
	}

	c := gamedata.Castles()[0]
	dm := gamedata.DotaMaps()[0]
	inst := newDotaInstance(s, dm.ID, dm.ID)
	inst.dota.castleID = c.ID

	winner := dotaPlayerConn(t, s, inst, 1000, dotaTeamElf, 0, 0)
	winner.selfPlayerID = winU.ID
	loser := dotaPlayerConn(t, s, inst, 1001, dotaTeamHuman, 0, 0)
	loser.selfPlayerID = loseU.ID

	winBefore, _, _ := store.HeroMoney(winU.ID)
	loseBefore, _, _ := store.HeroMoney(loseU.ID)

	s.settleCastleBattleLocked(inst, c.ID, dotaTeamElf)

	ownerID, ownerName, ok := store.CastleOwner(c.ID)
	if !ok || ownerID != winClanID {
		t.Fatalf("castle owner = %d (%q) ok=%v, want %d (WinnersClan)", ownerID, ownerName, ok, winClanID)
	}
	winMoney, _, _ := store.HeroMoney(winU.ID)
	if winMoney != winBefore+c.Reward.Money {
		t.Errorf("winner money = %d, want %d (before + the reward)", winMoney, winBefore+c.Reward.Money)
	}
	loseMoney, _, _ := store.HeroMoney(loseU.ID)
	if loseMoney != loseBefore {
		t.Errorf("loser money changed to %d, want unchanged %d (not on the winning side)", loseMoney, loseBefore)
	}

	hist := store.CastleHistory(c.ID)
	if len(hist) != 1 || hist[0].WinnerClanID != winClanID {
		t.Fatalf("castle history = %+v, want one entry for clan %d", hist, winClanID)
	}
}

// TestCastleSettlementUnclannedWinnerStillPaidNoOwnershipChange: a winning side with
// no clan members (e.g. a solo test fight) still gets paid, but ownership is left
// exactly as it was -- there is no clan to award it to.
func TestCastleSettlementUnclannedWinnerStillPaidNoOwnershipChange(t *testing.T) {
	store := session.NewStore()
	s := New(store)
	u, _, ok := store.LoginOrRegister("solo@example.com", "pw")
	if !ok {
		t.Fatal("register solo")
	}
	store.CreateHero(u, 1, false, 0, 0, 0, 0, 0)

	c := gamedata.Castles()[0]
	dm := gamedata.DotaMaps()[0]
	inst := newDotaInstance(s, dm.ID, dm.ID)
	inst.dota.castleID = c.ID

	winner := dotaPlayerConn(t, s, inst, 1000, dotaTeamHuman, 0, 0)
	winner.selfPlayerID = u.ID

	before, _, _ := store.HeroMoney(u.ID)
	s.settleCastleBattleLocked(inst, c.ID, dotaTeamHuman)

	if _, _, ok := store.CastleOwner(c.ID); ok {
		t.Fatal("castle should remain unowned: the winning fighter has no clan")
	}
	money, _, _ := store.HeroMoney(u.ID)
	if money != before+c.Reward.Money {
		t.Errorf("solo winner money = %d, want %d (before + the reward)", money, before+c.Reward.Money)
	}
}

// TestCastleAltarWinEndToEndSettles: dotaCheckWinLocked's normal altar-fall detection
// -- not a direct settleCastleBattleLocked call -- still triggers the castle payout
// when the instance is castle-tagged, via dotaEndLocked's hook.
func TestCastleAltarWinEndToEndSettles(t *testing.T) {
	store := session.NewStore()
	s := New(store)
	u, _, ok := store.LoginOrRegister("attacker@example.com", "pw")
	if !ok {
		t.Fatal("register attacker")
	}
	store.CreateHero(u, 1, false, 0, 0, 0, 0, 0)

	c := gamedata.Castles()[0]
	dm := gamedata.DotaMaps()[0]
	inst := newDotaInstance(s, dm.ID, dm.ID)
	inst.dota.castleID = c.ID

	human := dotaPlayerConn(t, s, inst, 1000, dotaTeamHuman, 0, 0)
	human.selfPlayerID = u.ID

	before, _, _ := store.HeroMoney(u.ID)
	now := float64(s.battleTime())
	altarOf(inst, dotaTeamElf).dead = true
	human.lock()
	s.dotaTickLocked(human, now)
	human.unlock()

	if !inst.dota.ended || inst.dota.winner != dotaTeamHuman {
		t.Fatalf("altar fall did not end the match for Human: ended=%v winner=%d", inst.dota.ended, inst.dota.winner)
	}
	money, _, _ := store.HeroMoney(u.ID)
	if money != before+c.Reward.Money {
		t.Errorf("attacker money after altar win = %d, want %d (settlement did not fire)", money, before+c.Reward.Money)
	}
}
