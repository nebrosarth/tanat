package battleserver

import (
	"math/rand"
	"sort"
	"testing"

	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// manualDotaBots is a deterministic integration-test driver. It exercises the
// same authoritative member, bot, and DOTA ticks as a live instance, but moves
// the simulation clock explicitly instead of waiting for wall-clock timers.
type manualDotaBots struct {
	server *Server
	inst   *huntInstance
	clock  *manualBattleClock
}

func newManualDotaBots(t *testing.T, roomID int32, totalBots int) *manualDotaBots {
	t.Helper()
	clock := newManualBattleClock()
	server := New(session.NewStore())
	server.clock = clock
	server.rng = rand.New(rand.NewSource(int64(roomID)))
	inst := newDotaInstance(server, roomID, gamedata.DotaMaps()[0].ID)
	inst.mu.Lock()
	server.spawnDotaBotsLocked(inst, totalBots)
	inst.mu.Unlock()
	driver := &manualDotaBots{server: server, inst: inst, clock: clock}
	t.Cleanup(driver.close)
	return driver
}

func (d *manualDotaBots) step() bool {
	if d == nil || d.server == nil || d.inst == nil || d.clock == nil {
		return true
	}
	// Timers schedule callbacks that acquire inst.mu, so they must run before
	// the synchronous world pass acquires that same lock.
	d.clock.Advance(AssaultTick)
	now := d.clock.Now()
	d.inst.mu.Lock()
	defer d.inst.mu.Unlock()
	if d.inst.closed || d.inst.dota == nil || d.inst.dota.ended {
		return true
	}
	d.server.botRebalanceLanesLocked(d.inst, now)
	d.server.botPlanTeamsLocked(d.inst, now)
	members := d.inst.memberList()
	sort.Slice(members, func(i, j int) bool { return members[i].objID < members[j].objID })
	var rep *conn
	for _, member := range members {
		if member == nil || member.huntState == nil || member.huntState.closed {
			continue
		}
		d.server.memberTickLocked(member, now)
		rep = member
		if brain := d.inst.bots[member.objID]; brain != nil && botAIVersionForBrain(brain) != 40 {
			d.server.botTickLocked(brain, now)
		}
		if d.inst.dota.telemetry != nil {
			d.server.telemetrySnapshotLocked(d.inst.dota.telemetry, member, d.inst.dota.telemetryMatchTimeLocked(now), now)
		}
	}
	d.server.botAI40BatchTickLocked(d.inst, now)
	if rep != nil {
		d.server.dotaTickLocked(rep, now)
	}
	return d.inst.dota.ended
}

func (d *manualDotaBots) close() {
	if d == nil || d.inst == nil {
		return
	}
	d.inst.mu.Lock()
	d.inst.closed = true
	members := make([]*conn, 0, len(d.inst.members))
	for _, member := range d.inst.members {
		if member == nil {
			continue
		}
		if member.huntState != nil {
			member.huntState.closed = true
			member.huntState.attackSeq++
			member.stopArrivalLocked()
		}
		members = append(members, member)
	}
	if d.inst.dota != nil && d.inst.dota.telemetry != nil {
		d.inst.dota.telemetry.close()
		d.inst.dota.telemetry = nil
	}
	d.inst.mu.Unlock()
	for _, member := range members {
		if member.Conn != nil {
			_ = member.Conn.Close()
		}
	}
}
