package battleserver

import (
	"encoding/binary"
	"net"
	"testing"
	"time"

	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// syncHeader decodes a SYNC data blob's header: how many add/remove entries it
// carries, and the "present sync types" bitmask. Layout (see syncBlob.build):
// float32 time, int16 newIds count, int32 newIds[], uint64 type mask.
func syncHeader(t *testing.T, blob []byte) (newIDs int, mask uint64) {
	t.Helper()
	off := 4 // float32 time
	if len(blob) < off+2 {
		t.Fatalf("sync blob too short: %d bytes", len(blob))
	}
	n := int(int16(binary.LittleEndian.Uint16(blob[off:])))
	off += 2 + 4*n
	if len(blob) < off+8 {
		t.Fatalf("sync blob too short for type mask: %d bytes, need %d", len(blob), off+8)
	}
	return n, binary.LittleEndian.Uint64(blob[off:])
}

func syncTypeMask(t *testing.T, blob []byte) uint64 {
	t.Helper()
	_, mask := syncHeader(t, blob)
	return mask
}

// collectSyncs dials a conn whose pushes are decoded, returning a channel of SYNC
// data blobs.
func collectSyncs(t *testing.T) (*conn, <-chan []byte) {
	t.Helper()
	srv, cli := net.Pipe()
	t.Cleanup(func() { srv.Close(); cli.Close() })
	out := make(chan []byte, 256)
	go func() {
		defer close(out)
		r := battleproto.NewReader(cli)
		for {
			p, err := r.Read()
			if err != nil {
				return
			}
			if p.Cmd != battleproto.CmdSync || p.Args == nil {
				continue
			}
			if b, ok := p.Args.Assoc["data"].([]byte); ok {
				out <- b
			}
		}
	}()
	return &conn{Conn: srv}, out
}

// awaitViewRadiusSync waits for a SYNC blob that carries SyncType.VIEW_RADIUS.
func awaitViewRadiusSync(t *testing.T, syncs <-chan []byte) {
	t.Helper()
	deadline := time.After(3 * time.Second)
	for {
		select {
		case blob, ok := <-syncs:
			if !ok {
				t.Fatal("connection closed before any VIEW_RADIUS sync arrived")
			}
			if syncTypeMask(t, blob)&syncViewRadius != 0 {
				return
			}
		case <-deadline:
			t.Fatal("no SYNC carried SyncType.VIEW_RADIUS -- the WarFog plane would stay opaque and the map renders black")
		}
	}
}

// TestDotaStructureRevealCarriesViewRadius pins the «Штурм» fog-of-war fix. The
// map_1_0 scene bakes a WarFog plane that starts fully opaque; it is only cleared
// around each visible unit, by that unit's SyncType.VIEW_RADIUS. The server never sent
// VIEW_RADIUS, so every unit cleared a radius of 0 and the whole map rendered black.
// Every revealed base structure must now carry a non-zero vision radius.
func TestDotaStructureRevealCarriesViewRadius(t *testing.T) {
	s := New(session.NewStore())
	c, syncs := collectSyncs(t)

	dm := gamedata.DotaMaps()[0]
	inst := newDotaInstance(s, dm.ID, dm.ID)
	av := avatarByPrefab(t, "Avtr_Tank_Velial")
	c.objID = 1000
	hs := &huntState{
		av: av, kit: gamedata.SkillsFor(av),
		mobs: inst.mobs, summons: map[int32]*summonState{},
		hp: av.Health, mana: av.Mana,
	}
	hs.tr.add(c.objID)
	hs.inst = inst
	hs.worldReady = true
	c.huntState = hs
	c.inst = inst
	c.lk = &inst.mu
	inst.members[c.objID] = c

	var st *mobState
	for _, m := range inst.mobs {
		if m.structure {
			st = m
			break
		}
	}
	if st == nil {
		t.Fatal("precondition: DOTA instance seeded no structures")
	}

	go func() {
		c.lock()
		s.dotaRevealStructureLocked(c, st, float64(s.battleTime()))
		c.unlock()
	}()

	awaitViewRadiusSync(t, syncs)
}

// TestSelfAvatarWorldStateCarriesViewRadiusAndTeam drives the real CONNECT/READY
// handshake into a «Штурм» launch and pins BOTH fog gates on the player's own avatar
// (WarFogObject.Update only spawns its reveal zone when `mViewRadius > 0f &&
// friendliness != UNKNOWN`):
//
//	VIEW_RADIUS -> SyncedParams.mViewRadius; 0 (unset) fails the first half.
//	TEAM        -> SyncedParams.mTeamInited; without it TeamRecognizer.GetFriendliness
//	               returns UNKNOWN (`if (!mInited || !_params.IsTeamInited)`), failing
//	               the second half.
//
// The self avatar shipped with NEITHER (structures/mobs/other players' avatars all got
// TEAM, the self avatar never did), so the player never lit even their own feet.
func TestSelfAvatarWorldStateCarriesViewRadiusAndTeam(t *testing.T) {
	store := session.NewStore()
	s := New(store)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()
	go s.Serve(ln)

	const userID int32 = 1
	dm := gamedata.DotaMaps()[0]
	av := avatarByPrefab(t, "Avtr_Tank_Velial")
	store.SetPendingBattle(userID, session.PendingBattle{
		MapID: dm.ID, AvatarID: av.ID, Passwd: "pw", Scene: dm.Scene, Room: dm.ID,
	})

	cl, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer cl.Close()
	_ = cl.SetDeadline(time.Now().Add(5 * time.Second))
	r := battleproto.NewReader(cl)

	if err := battleproto.Write(cl, battleproto.Packet{
		Cmd: battleproto.CmdConnect, RequestID: 1, Status: true,
		Args: amf.NewArray().Set("clientId", userID).Set("pass", "pw"),
	}); err != nil {
		t.Fatalf("send CONNECT: %v", err)
	}
	if _, err := r.Read(); err != nil {
		t.Fatalf("read CONNECT reply: %v", err)
	}
	if err := battleproto.Write(cl, battleproto.Packet{
		Cmd: battleproto.CmdReady, RequestID: 2, Status: true, Args: amf.NewArray(),
	}); err != nil {
		t.Fatalf("send READY: %v", err)
	}

	// Scan the world-state burst for the AVATAR's stat sync. It must be identified
	// precisely: every «Штурм» structure reveal also carries VIEW_RADIUS|TEAM, so a
	// blind mask scan passes even with the avatar's own values removed (verified).
	// The avatar's stat blob is the one with NO add/remove entries -- it addresses the
	// already-registered avatar by tracking index (build(1), no addObject) -- whereas
	// every structure reveal carries exactly one add entry.
	const want = syncViewRadius | syncTeam
	for {
		p, err := r.Read()
		if err != nil {
			t.Fatalf("self avatar's stat sync never carried both VIEW_RADIUS and TEAM "+
				"(WarFogObject would never spawn its reveal zone -- map renders black): %v", err)
		}
		if p.Cmd != battleproto.CmdSync || p.Args == nil {
			continue
		}
		blob, ok := p.Args.Assoc["data"].([]byte)
		if !ok {
			continue
		}
		newIDs, mask := syncHeader(t, blob)
		if newIDs == 0 && mask&want == want {
			return // the avatar's own stat sync satisfies both fog gates
		}
	}
}

// TestViewRadiiAreNonZero guards the values themselves: a zero radius clears nothing,
// which is exactly the bug (an unset VIEW_RADIUS defaults to 0 on the client).
func TestViewRadiiAreNonZero(t *testing.T) {
	if avatarViewRadius <= 0 {
		t.Errorf("avatarViewRadius = %v; a zero vision radius clears no fog", avatarViewRadius)
	}
	if structViewRadius <= 0 {
		t.Errorf("structViewRadius = %v; a zero vision radius clears no fog", structViewRadius)
	}
}
