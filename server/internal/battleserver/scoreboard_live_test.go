package battleserver

import (
	"encoding/binary"
	"math"
	"testing"
	"time"

	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

func TestLiveDotaFightLogRefreshesRosterWithoutRatingSettlement(t *testing.T) {
	store := session.NewStore()
	s := New(store)
	winnerUser, _, _ := store.LoginOrRegister("live-winner@test.io", "pw")
	store.CreateHero(winnerUser, 0, false, 0, 0, 0, 0, 0)
	loserUser, _, _ := store.LoginOrRegister("live-loser@test.io", "pw")
	store.CreateHero(loserUser, 0, false, 0, 0, 0, 0, 0)

	dm := gamedata.DotaMaps()[0]
	inst := newDotaInstance(s, dm.ID, dm.ID)
	winner := dotaPlayerConn(t, s, inst, 1000, dotaTeamHuman, 0, 0)
	winner.selfPlayerID, winner.battleID, winner.name = winnerUser.ID, 6101, "Winner"
	loser := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, 0, 0)
	loser.selfPlayerID, loser.battleID, loser.name = loserUser.ID, 6102, "Loser"

	inst.mu.Lock()
	s.publishLiveFightLogLocked(inst)
	inst.mu.Unlock()

	entries, ok := store.FightLog(winner.battleID)
	if !ok || len(entries) != 2 {
		t.Fatalf("initial live roster = (%v, %d rows), want two rows", ok, len(entries))
	}
	if entries[winner.selfPlayerID].Kills != 0 || entries[loser.selfPlayerID].Deaths != 0 {
		t.Fatalf("initial live K/D = %+v, want zeroes", entries)
	}

	winner.huntState.frags = 2
	winner.huntState.assists = 1
	winner.huntState.level = 3
	loser.huntState.deaths = 1
	inst.mu.Lock()
	s.publishLiveFightLogLocked(inst)
	inst.mu.Unlock()

	entries, _ = store.FightLog(loser.battleID)
	if got := entries[winner.selfPlayerID]; got.Kills != 2 || got.Assists != 1 || got.Level != 3 {
		t.Errorf("refreshed winner row = %+v, want K=2 A=1 L=3", got)
	}
	if got := entries[loser.selfPlayerID]; got.Deaths != 1 {
		t.Errorf("refreshed loser row = %+v, want D=1", got)
	}
	if got, _ := store.HeroRating(winnerUser.ID); got != session.RatingDefault {
		t.Errorf("live publication changed winner rating to %d", got)
	}
	if got, _ := store.HeroRating(loserUser.ID); got != session.RatingDefault {
		t.Errorf("live publication changed loser rating to %d", got)
	}

	inst.mu.Lock()
	s.settleMatchLocked(inst, dotaTeamHuman, float64(s.battleTime()))
	inst.mu.Unlock()
	entries, _ = store.FightLog(winner.battleID)
	if got := entries[winner.selfPlayerID]; got.NewRating <= got.OldRating || got.Kills != 2 {
		t.Errorf("final winner row = %+v, want settled rating increase and K=2", got)
	}
	if got := entries[loser.selfPlayerID]; got.NewRating >= got.OldRating || got.Deaths != 1 {
		t.Errorf("final loser row = %+v, want settled rating decrease and D=1", got)
	}
}

func TestLiveFightLogCopiesInputAndOutput(t *testing.T) {
	store := session.NewStore()
	input := map[int32]session.FightLogEntry{
		7: {AvatarID: 13, Nick: "Copy", Kills: 1},
	}
	store.SetLiveFightLog(7101, input)
	input[7] = session.FightLogEntry{AvatarID: 99, Nick: "mutated input", Kills: 99}

	got, ok := store.FightLog(7101)
	if !ok || got[7].AvatarID != 13 || got[7].Nick != "Copy" || got[7].Kills != 1 {
		t.Fatalf("stored live snapshot changed through input map: ok=%v row=%+v", ok, got[7])
	}
	got[7] = session.FightLogEntry{AvatarID: 88, Nick: "mutated output", Kills: 88}
	gotAgain, _ := store.FightLog(7101)
	if gotAgain[7].AvatarID != 13 || gotAgain[7].Nick != "Copy" || gotAgain[7].Kills != 1 {
		t.Fatalf("stored live snapshot changed through output map: row=%+v", gotAgain[7])
	}
}

func TestPlayerStatsPacketUsesPlayerStoreContractAndRegisteredID(t *testing.T) {
	s, viewer, inst, packets, packetsMu := newDotaCaptureConn(t)
	target := dotaPlayerConn(t, s, inst, 1001, dotaTeamHuman, 0, 0)
	target.selfPlayerID = 7007
	target.huntState.level = 4
	target.huntState.frags = 3
	target.huntState.deaths = 2
	target.huntState.assists = 5
	target.huntState.lastKiller = 9009

	inst.mu.Lock()
	s.renderAvatarForLocked(viewer, target, float64(s.battleTime()))
	*packets = (*packets)[:0]
	s.broadcastPlayerStatsLocked(target)
	inst.mu.Unlock()

	var got *battleproto.Packet
	deadline := time.Now().Add(time.Second)
	for got == nil && time.Now().Before(deadline) {
		packetsMu.Lock()
		for i := range *packets {
			if (*packets)[i].Cmd == battleproto.CmdPlayerStats {
				p := (*packets)[i]
				got = &p
				break
			}
		}
		packetsMu.Unlock()
		if got == nil {
			time.Sleep(time.Millisecond)
		}
	}
	if got == nil || got.Args == nil {
		t.Fatal("rendering viewer received no PLAYER_STATS")
	}
	for _, key := range []string{"id", "level", "kills", "deaths", "killer", "assists"} {
		if _, ok := got.Args.Assoc[key]; !ok {
			t.Errorf("PLAYER_STATS missing %q", key)
		}
	}
	for _, key := range []string{"player_id", "last_killer"} {
		if _, ok := got.Args.Assoc[key]; ok {
			t.Errorf("PLAYER_STATS still contains obsolete client-incompatible key %q", key)
		}
	}
	checks := map[string]int32{
		"id": 7007, "level": 4, "kills": 3,
		"deaths": 2, "killer": 9009, "assists": 5,
	}
	for key, want := range checks {
		if gotValue, ok := got.Args.GetInt(key); !ok || gotValue != want {
			t.Errorf("PLAYER_STATS[%s] = %d (ok=%v), want %d", key, gotValue, ok, want)
		}
	}
}

func TestRespawnManaSyncReachesRenderedViewerWithFractionAndMax(t *testing.T) {
	s, viewer, inst, packets, packetsMu := newDotaCaptureConn(t)
	viewer.selfPlayerID = botIDBase
	target := dotaPlayerConn(t, s, inst, 1001, dotaTeamHuman, 0, 0)
	inst.mu.Lock()
	s.renderAvatarForLocked(viewer, target, float64(s.battleTime()))
	*packets = (*packets)[:0]
	target.huntState.deadUntil = float64(s.battleTime()) + 1
	target.huntState.mana = 0
	now := float64(s.battleTime())
	s.respawnPlayerLocked(target, now)
	inst.mu.Unlock()

	var data []byte
	packetsMu.Lock()
	for _, p := range *packets {
		if p.Cmd == battleproto.CmdSync && p.Args != nil {
			if candidate, ok := p.Args.Assoc["data"].([]byte); ok && syncTypeMask(t, candidate)&(syncMana|syncMaxMana) == syncMana|syncMaxMana {
				data = candidate
			}
		}
	}
	packetsMu.Unlock()
	if data == nil {
		t.Fatal("rendered viewer received no SYNC carrying mana and max-mana")
	}
	idx := viewer.huntState.tr.index(target.objID)
	frac, okFrac := syncFloatForIndex(data, syncMana, idx)
	max, okMax := syncFloatForIndex(data, syncMaxMana, idx)
	if !okFrac || !okMax {
		t.Fatalf("viewer mana sync missing target index %d: frac=%v max=%v", idx, okFrac, okMax)
	}
	wantMax := float32(target.huntState.maxManaLocked(now))
	if frac != 1 || math.Abs(float64(max-wantMax)) > 0.001 {
		t.Errorf("viewer mana sync = fraction %.3f max %.3f, want 1.000/%.3f", frac, max, wantMax)
	}
}

func TestRespawnManaSyncReachesBotViewer(t *testing.T) {
	s, viewer, inst, packets, packetsMu := newDotaCaptureConn(t)
	// Keep the real capture socket, but exercise the same reserved-ID path used by
	// server-controlled viewers.
	viewer.objID = botIDBase
	viewer.selfPlayerID = botIDBase
	target := dotaPlayerConn(t, s, inst, 1001, dotaTeamHuman, 0, 0)

	inst.mu.Lock()
	s.renderAvatarForLocked(viewer, target, float64(s.battleTime()))
	*packets = (*packets)[:0]
	now := float64(s.battleTime())
	target.huntState.deadUntil = now + 1
	target.huntState.mana = 0
	s.respawnPlayerLocked(target, now)
	inst.mu.Unlock()

	var data []byte
	packetsMu.Lock()
	for _, p := range *packets {
		if p.Cmd == battleproto.CmdSync && p.Args != nil {
			if candidate, ok := p.Args.Assoc["data"].([]byte); ok && syncTypeMask(t, candidate)&(syncMana|syncMaxMana) == syncMana|syncMaxMana {
				data = candidate
			}
		}
	}
	packetsMu.Unlock()
	if data == nil {
		t.Fatal("bot viewer received no SYNC carrying mana and max-mana")
	}
	idx := viewer.huntState.tr.index(target.objID)
	frac, okFrac := syncFloatForIndex(data, syncMana, idx)
	max, okMax := syncFloatForIndex(data, syncMaxMana, idx)
	wantMax := float32(target.huntState.maxManaLocked(now))
	if !okFrac || !okMax || frac != 1 || math.Abs(float64(max-wantMax)) > 0.001 {
		t.Errorf("bot viewer mana sync = fraction %.3f max %.3f (present=%v/%v), want 1.000/%.3f", frac, max, okFrac, okMax, wantMax)
	}
}

func TestRespawnNewlyRenderedViewerReceivesPlayerStats(t *testing.T) {
	s, viewer, inst, packets, packetsMu := newDotaCaptureConn(t)
	target := dotaPlayerConn(t, s, inst, 1001, dotaTeamHuman, 0, 0)
	target.selfPlayerID = 7007
	target.huntState.deaths = 2
	target.huntState.lastKiller = 9009

	inst.mu.Lock()
	now := float64(s.battleTime())
	s.renderAvatarForLocked(viewer, target, now)
	// Match the real corpse-hide path: the target and remote viewer no longer track
	// the avatar when the respawn tick runs.
	s.removeAvatarForLocked(viewer, target)
	target.huntState.tr.remove(target.objID)
	target.huntState.corpseHidden = true
	*packets = (*packets)[:0]
	s.respawnPlayerLocked(target, now)
	inst.mu.Unlock()

	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		packetsMu.Lock()
		found := false
		for _, p := range *packets {
			if p.Cmd != battleproto.CmdPlayerStats || p.Args == nil {
				continue
			}
			if id, _ := p.Args.GetInt("id"); id != target.selfPlayerID {
				continue
			}
			if deaths, _ := p.Args.GetInt("deaths"); deaths != target.huntState.deaths {
				continue
			}
			if killer, _ := p.Args.GetInt("killer"); killer != target.huntState.lastKiller {
				continue
			}
			found = true
			break
		}
		packetsMu.Unlock()
		if found {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("newly re-rendered viewer received no current PLAYER_STATS for player %d", target.selfPlayerID)
}

func syncFloatForIndex(data []byte, wantType uint64, wantIndex int) (float32, bool) {
	if wantIndex < 0 || len(data) < 14 {
		return 0, false
	}
	n := int(int16(binary.LittleEndian.Uint16(data[4:6])))
	off := 6 + 4*n
	if off+8 > len(data) {
		return 0, false
	}
	mask := binary.LittleEndian.Uint64(data[off : off+8])
	off += 8
	width := 1
	for typ := uint64(1); typ <= syncMaxMana; typ <<= 1 {
		if mask&typ == 0 {
			continue
		}
		if off+width > len(data) {
			return 0, false
		}
		bits := data[off]
		off += width
		for idx := 0; idx < 8; idx++ {
			if bits&(1<<idx) == 0 {
				continue
			}
			if off+4 > len(data) {
				return 0, false
			}
			value := math.Float32frombits(binary.LittleEndian.Uint32(data[off : off+4]))
			off += 4
			if typ == wantType && idx == wantIndex {
				return value, true
			}
		}
	}
	return 0, false
}
