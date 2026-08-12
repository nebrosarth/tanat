package battleserver

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
	"net"
	"sync"
	"testing"
	"time"

	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
)

// botTeleportVisualConn captures battle writes without a socket or a reader
// goroutine. The captured bytes are decoded with the production battle reader,
// so these tests inspect the same packet stream a client would receive.
type botTeleportVisualConn struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (c *botTeleportVisualConn) Read([]byte) (int, error) { return 0, io.EOF }

func (c *botTeleportVisualConn) Write(p []byte) (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.buf.Write(p)
}

func (c *botTeleportVisualConn) Close() error { return nil }

func (c *botTeleportVisualConn) LocalAddr() net.Addr  { return botTeleportVisualAddr("local") }
func (c *botTeleportVisualConn) RemoteAddr() net.Addr { return botTeleportVisualAddr("visual-test") }

func (c *botTeleportVisualConn) SetDeadline(time.Time) error      { return nil }
func (c *botTeleportVisualConn) SetReadDeadline(time.Time) error  { return nil }
func (c *botTeleportVisualConn) SetWriteDeadline(time.Time) error { return nil }

func (c *botTeleportVisualConn) reset() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.buf.Reset()
}

func captureBotTeleportVisual(t *testing.T, bot *conn) *botTeleportVisualConn {
	t.Helper()
	wire := &botTeleportVisualConn{}
	bot.huntState.tr.add(bot.objID)
	bot.Conn = wire
	return wire
}

func (c *botTeleportVisualConn) packets(t *testing.T) []battleproto.Packet {
	t.Helper()
	c.mu.Lock()
	data := append([]byte(nil), c.buf.Bytes()...)
	c.mu.Unlock()

	r := battleproto.NewReader(bytes.NewReader(data))
	var out []battleproto.Packet
	for {
		p, err := r.Read()
		if err == io.EOF {
			return out
		}
		if err != nil {
			t.Fatalf("decode captured battle packets: %v", err)
		}
		out = append(out, p)
	}
}

type botTeleportVisualAddr string

func (a botTeleportVisualAddr) Network() string { return "bot-teleport-visual" }
func (a botTeleportVisualAddr) String() string  { return string(a) }

func addBotTeleportVisualViewer(t *testing.T, s *Server, inst *huntInstance, objID, team int32, x, y float32) (*conn, *botTeleportVisualConn) {
	t.Helper()
	av := avatarByPrefab(t, "Avtr_Tank_Velial")
	wire := &botTeleportVisualConn{}
	c := &conn{Conn: wire, objID: objID, selfPlayerID: objID, name: "visual-viewer"}
	c.x, c.y, c.snapT = x, y, s.battleTime()
	c.nav = inst.nav
	c.lk = &inst.mu
	c.inst = inst
	c.huntState = &huntState{
		av: av, kit: gamedata.SkillsFor(av), mobs: inst.mobs,
		summons: map[int32]*summonState{}, summonProtos: map[string]int32{},
		hp: av.Health, mana: av.Mana, team: team, worldReady: true,
	}
	c.huntState.inst = inst
	c.huntState.tr.add(objID)
	inst.members[objID] = c
	return c, wire
}

func botTeleportVisualSyncHeader(t *testing.T, data []byte) (int, uint64) {
	t.Helper()
	if len(data) < 14 {
		t.Fatalf("SYNC blob too short: %d", len(data))
	}
	n := int(int16(binary.LittleEndian.Uint16(data[4:6])))
	if n < 0 || len(data) < 14+4*n {
		t.Fatalf("SYNC blob has invalid visibility count %d (len %d)", n, len(data))
	}
	off := 6 + 4*n
	return n, binary.LittleEndian.Uint64(data[off : off+8])
}

func botTeleportVisualPosition(t *testing.T, data []byte, trackedCount, wantIndex int) (float32, float32) {
	t.Helper()
	n, mask := botTeleportVisualSyncHeader(t, data)
	if mask&syncPosition == 0 {
		t.Fatalf("SYNC mask %#x has no POSITION", mask)
	}
	width := (trackedCount + 7) / 8
	if width == 0 {
		width = 1
	}
	off := 14 + 4*n
	if len(data) < off+width {
		t.Fatalf("SYNC blob missing POSITION object mask")
	}
	objects := data[off : off+width]
	off += width
	for idx := 0; idx < trackedCount; idx++ {
		if objects[idx/8]&(1<<uint(idx%8)) == 0 {
			continue
		}
		if len(data) < off+20 {
			t.Fatalf("SYNC blob missing POSITION values for index %d", idx)
		}
		if idx == wantIndex {
			return math.Float32frombits(binary.LittleEndian.Uint32(data[off : off+4])),
				math.Float32frombits(binary.LittleEndian.Uint32(data[off+4 : off+8]))
		}
		off += 20
	}
	t.Fatalf("SYNC POSITION did not contain tracking index %d", wantIndex)
	return 0, 0
}

func botTeleportVisualSyncData(t *testing.T, p battleproto.Packet) []byte {
	t.Helper()
	if p.Cmd != battleproto.CmdSync || p.Args == nil {
		t.Fatalf("packet is %s, want SYNC", p.Cmd.Name())
	}
	data, ok := p.Args.Assoc["data"].([]byte)
	if !ok {
		t.Fatalf("SYNC data has type %T, want []byte", p.Args.Assoc["data"])
	}
	return data
}

func assertBotTeleportVisualReset(t *testing.T, packets []battleproto.Packet, ownerID int32, wantX, wantY float32, wantConfirmation bool, markerUID, targetUID, arrivalUID int32) {
	t.Helper()
	prepFXPrefix := 0
	if markerUID != 0 || targetUID != 0 {
		if markerUID == 0 || targetUID == 0 {
			t.Fatalf("incomplete preparation FX UIDs: marker=%d target=%d", markerUID, targetUID)
		}
		prepFXPrefix = 2
	}
	wantCommands := []battleproto.CmdID{
		battleproto.CmdSync, battleproto.CmdDeleteObject,
		battleproto.CmdPrototypeInfo, battleproto.CmdPlayerReg,
		battleproto.CmdCreateObject, battleproto.CmdSetAvatar,
		battleproto.CmdPrototypeInfo, battleproto.CmdAddEffector,
		battleproto.CmdSync,
	}
	if wantConfirmation {
		wantCommands = append(wantCommands, battleproto.CmdSync)
	}
	if prepFXPrefix != 0 {
		wantCommands = append([]battleproto.CmdID{battleproto.CmdEffectEnd, battleproto.CmdEffectEnd}, wantCommands...)
	}
	if arrivalUID != 0 {
		wantCommands = append(wantCommands, battleproto.CmdCreateObject, battleproto.CmdSync)
		wantCommands = append(wantCommands, battleproto.CmdEffectStart)
	}
	if len(packets) != len(wantCommands) {
		t.Fatalf("remote reset packet count = %d, want %d; commands=%v", len(packets), len(wantCommands), botTeleportVisualCommandNames(packets))
	}
	for i, want := range wantCommands {
		if packets[i].Cmd != want {
			t.Fatalf("remote reset packet %d = %s, want %s", i, packets[i].Cmd.Name(), want.Name())
		}
	}

	if prepFXPrefix != 0 {
		wantPrepEnds := []int32{markerUID, targetUID}
		for i, wantUID := range wantPrepEnds {
			if got, ok := packets[i].Args.GetInt("id"); !ok || got != wantUID {
				t.Fatalf("preparation EFFECT_END %d id = %d (present=%v), want %d", i, got, ok, wantUID)
			}
		}
	}

	arrivalPrefix := 0
	if arrivalUID != 0 {
		arrivalPrefix = 3 // CREATE_OBJECT + positioned SYNC for the Dummy anchor, then EFFECT_START.
	}
	resetPackets := packets[prepFXPrefix : len(packets)-arrivalPrefix]
	removed := botTeleportVisualSyncData(t, resetPackets[0])
	n, mask := botTeleportVisualSyncHeader(t, removed)
	if n != 1 || mask != 0 || binary.LittleEndian.Uint32(removed[6:10]) != syncRemMask|1 {
		t.Fatalf("tracker removal SYNC = n=%d mask=%#x entry=%#x, want one index-1 removal", n, mask, binary.LittleEndian.Uint32(removed[6:10]))
	}
	if id, _ := resetPackets[1].Args.GetInt("id"); id != ownerID {
		t.Fatalf("DELETE_OBJECT id = %d, want owner %d", id, ownerID)
	}
	if id, _ := resetPackets[4].Args.GetInt("id"); id != ownerID {
		t.Fatalf("CREATE_OBJECT id = %d, want owner %d", id, ownerID)
	}
	if id, _ := resetPackets[5].Args.GetInt("avatarID"); id != ownerID {
		t.Fatalf("SET_AVATAR avatarID = %d, want owner %d", id, ownerID)
	}
	if owner, ok := resetPackets[7].Args.GetInt("owner"); !ok || owner != ownerID {
		t.Fatalf("ADD_EFFECTOR owner = %d (present=%v), want owner %d", owner, ok, ownerID)
	}
	if parent, ok := resetPackets[7].Args.GetInt("parent"); !ok || parent != -1 {
		t.Fatalf("ADD_EFFECTOR parent = %d (present=%v), want unparented avatar", parent, ok)
	}
	if proto, ok := resetPackets[7].Args.GetInt("proto"); !ok || proto != attackProtoID(avatarByPrefab(t, "Avtr_Tank_Velial")) {
		t.Fatalf("ADD_EFFECTOR proto = %d (present=%v), want Velial attack prototype", proto, ok)
	}

	initialX, initialY := botTeleportVisualPosition(t, botTeleportVisualSyncData(t, resetPackets[8]), 2, 1)
	if math.Abs(float64(initialX-wantX)) > 0.001 || math.Abs(float64(initialY-wantY)) > 0.001 {
		t.Fatalf("recreated avatar POSITION = (%.3f, %.3f), want destination (%.3f, %.3f)", initialX, initialY, wantX, wantY)
	}
	if wantConfirmation {
		confirmX, confirmY := botTeleportVisualPosition(t, botTeleportVisualSyncData(t, resetPackets[9]), 2, 1)
		if math.Abs(float64(confirmX-wantX)) > 0.001 || math.Abs(float64(confirmY-wantY)) > 0.001 {
			t.Fatalf("destination confirmation POSITION = (%.3f, %.3f), want (%.3f, %.3f)", confirmX, confirmY, wantX, wantY)
		}
	}
	if arrivalUID != 0 {
		anchorCreate := packets[len(packets)-3]
		anchorID, ok := anchorCreate.Args.GetInt("id")
		proto, protoOK := anchorCreate.Args.GetInt("proto")
		if !ok || anchorCreate.Cmd != battleproto.CmdCreateObject || !protoOK || proto != trapAnchorProtoID {
			t.Fatalf("arrival anchor CREATE_OBJECT = %v, want Dummy proto", anchorCreate)
		}
		p := packets[len(packets)-1]
		uid, ok := p.Args.GetInt("effect")
		if !ok || uid != arrivalUID {
			t.Fatalf("arrival EFFECT_START uid = %d (present=%v), want %d", uid, ok, arrivalUID)
		}
		if fx, ok := p.Args.GetString("fx"); !ok || fx != botTeleportTargetFx {
			t.Fatalf("arrival EFFECT_START fx = %q (present=%v), want %q", fx, ok, botTeleportTargetFx)
		}
		if owner, ok := p.Args.GetInt("owner"); !ok || owner != anchorID {
			t.Fatalf("arrival EFFECT_START owner = %d (present=%v), want anchor %d", owner, ok, anchorID)
		}
		args, ok := p.Args.GetArray("args")
		if !ok {
			t.Fatal("arrival EFFECT_START has no args")
		}
		if target, ok := args.GetInt("target"); ok {
			t.Fatalf("arrival EFFECT_START target = %d, want no preparation target", target)
		}
		pos, ok := args.GetArray("targetPos")
		if !ok {
			t.Fatal("arrival EFFECT_START has no targetPos")
		}
		gotX, xOK := pos.GetFloat("x")
		gotY, yOK := pos.GetFloat("y")
		if !xOK || !yOK || math.Abs(gotX-float64(wantX)) > 0.001 || math.Abs(gotY-float64(wantY)) > 0.001 {
			t.Fatalf("arrival targetPos = (%.3f,%.3f), want (%.3f,%.3f)", gotX, gotY, wantX, wantY)
		}
	}
}

func boolInt(v bool) int {
	if v {
		return 1
	}
	return 0
}

func botTeleportVisualEffectPackets(packets []battleproto.Packet, cmd battleproto.CmdID) []battleproto.Packet {
	var effects []battleproto.Packet
	for _, p := range packets {
		if p.Cmd == cmd {
			effects = append(effects, p)
		}
	}
	return effects
}

func assertBotTeleportPreparationStarts(t *testing.T, packets []battleproto.Packet, ownerID, targetID int32, wantX, wantY float32) (int32, int32) {
	t.Helper()
	starts := botTeleportVisualEffectPackets(packets, battleproto.CmdEffectStart)
	if len(starts) != 2 {
		t.Fatalf("preparation EFFECT_START count = %d, want 2; commands=%v", len(starts), botTeleportVisualCommandNames(packets))
	}
	marker := starts[0]
	target := starts[1]
	uid, ok := marker.Args.GetInt("effect")
	if !ok || uid == 0 {
		t.Fatalf("marker EFFECT_START uid = %d (present=%v), want nonzero", uid, ok)
	}
	if fx, ok := marker.Args.GetString("fx"); !ok || fx != botTeleportMarkerFx {
		t.Fatalf("marker EFFECT_START fx = %q (present=%v), want %q", fx, ok, botTeleportMarkerFx)
	}
	if owner, ok := marker.Args.GetInt("owner"); !ok || owner != ownerID {
		t.Fatalf("marker EFFECT_START owner = %d (present=%v), want bot %d", owner, ok, ownerID)
	}
	markerArgs, ok := marker.Args.GetArray("args")
	if !ok {
		t.Fatal("marker EFFECT_START has no args")
	}
	if target, ok := markerArgs.GetInt("target"); ok {
		t.Fatalf("marker EFFECT_START target = %d, want no target", target)
	}
	if _, ok := markerArgs.GetArray("targetPos"); ok {
		t.Fatal("marker EFFECT_START has targetPos, want an unpositioned marker")
	}

	targetUID, ok := target.Args.GetInt("effect")
	if !ok || targetUID == 0 {
		t.Fatalf("target EFFECT_START uid = %d (present=%v), want nonzero", targetUID, ok)
	}
	if fx, ok := target.Args.GetString("fx"); !ok || fx != botTeleportTargetFx {
		t.Fatalf("target EFFECT_START fx = %q (present=%v), want %q", fx, ok, botTeleportTargetFx)
	}
	anchorID, ok := target.Args.GetInt("owner")
	if !ok || anchorID == 0 || anchorID == ownerID {
		t.Fatalf("target EFFECT_START owner = %d (present=%v), want stationary Dummy anchor", anchorID, ok)
	}
	var foundAnchor bool
	for i, p := range packets {
		if p.Cmd == battleproto.CmdCreateObject {
			id, _ := p.Args.GetInt("id")
			proto, _ := p.Args.GetInt("proto")
			if id == anchorID && proto == trapAnchorProtoID {
				foundAnchor = true
				if i+1 >= len(packets) || packets[i+1].Cmd != battleproto.CmdSync {
					t.Fatalf("target anchor %d was not positioned by an immediate SYNC", anchorID)
				}
			}
		}
	}
	if !foundAnchor {
		t.Fatalf("target EFFECT_START owner %d has no Dummy CREATE_OBJECT", anchorID)
	}
	args, ok := target.Args.GetArray("args")
	if !ok {
		t.Fatal("target EFFECT_START has no args")
	}
	if got, ok := args.GetInt("target"); !ok || got != targetID {
		t.Fatalf("target EFFECT_START target = %d (present=%v), want %d", got, ok, targetID)
	}
	pos, ok := args.GetArray("targetPos")
	if !ok {
		t.Fatal("target EFFECT_START has no fallback targetPos")
	}
	gotX, xOK := pos.GetFloat("x")
	gotY, yOK := pos.GetFloat("y")
	if !xOK || !yOK || math.Abs(gotX-float64(wantX)) > 0.001 || math.Abs(gotY-float64(wantY)) > 0.001 {
		t.Fatalf("target fallback targetPos = (%.3f,%.3f), want (%.3f,%.3f)", gotX, gotY, wantX, wantY)
	}
	return uid, targetUID
}

func assertBotTeleportPreparationEnds(t *testing.T, packets []battleproto.Packet, markerUID, targetUID int32) {
	t.Helper()
	ends := botTeleportVisualEffectPackets(packets, battleproto.CmdEffectEnd)
	if len(ends) != 2 {
		t.Fatalf("preparation EFFECT_END count = %d, want 2; commands=%v", len(ends), botTeleportVisualCommandNames(packets))
	}
	for i, wantUID := range []int32{markerUID, targetUID} {
		if got, ok := ends[i].Args.GetInt("id"); !ok || got != wantUID {
			t.Fatalf("preparation EFFECT_END %d id = %d (present=%v), want %d", i, got, ok, wantUID)
		}
	}
}

func assertBotTeleportTargetRefresh(t *testing.T, packets []battleproto.Packet, previousUID, previousAnchor, ownerID, targetID int32, wantX, wantY float32) int32 {
	t.Helper()
	if len(botTeleportVisualEffectPackets(packets, battleproto.CmdEffectEnd)) != 1 || len(botTeleportVisualEffectPackets(packets, battleproto.CmdEffectStart)) != 1 {
		t.Fatalf("target refresh packets = %v, want one EFFECT_END and one EFFECT_START", botTeleportVisualCommandNames(packets))
	}
	ends := botTeleportVisualEffectPackets(packets, battleproto.CmdEffectEnd)
	if got, ok := ends[0].Args.GetInt("id"); !ok || got != previousUID {
		t.Fatalf("refreshed target EFFECT_END id = %d (present=%v), want %d", got, ok, previousUID)
	}
	deleted := false
	for _, p := range packets {
		if p.Cmd == battleproto.CmdDeleteObject {
			if id, _ := p.Args.GetInt("id"); id == previousAnchor {
				deleted = true
			}
		}
	}
	if !deleted {
		t.Fatalf("refresh did not delete previous Dummy anchor %d: %v", previousAnchor, botTeleportVisualCommandNames(packets))
	}

	var p battleproto.Packet
	for _, candidate := range botTeleportVisualEffectPackets(packets, battleproto.CmdEffectStart) {
		p = candidate
	}
	uid, ok := p.Args.GetInt("effect")
	if !ok || uid == 0 {
		t.Fatalf("refreshed target EFFECT_START uid = %d (present=%v), want nonzero", uid, ok)
	}
	if fx, ok := p.Args.GetString("fx"); !ok || fx != botTeleportTargetFx {
		t.Fatalf("refreshed target EFFECT_START fx = %q (present=%v), want %q", fx, ok, botTeleportTargetFx)
	}
	if owner, ok := p.Args.GetInt("owner"); !ok || owner == ownerID || owner == previousAnchor {
		t.Fatalf("refreshed target EFFECT_START owner = %d (present=%v), want new Dummy anchor", owner, ok)
	}
	args, ok := p.Args.GetArray("args")
	if !ok {
		t.Fatal("refreshed target EFFECT_START has no args")
	}
	if got, ok := args.GetInt("target"); !ok || got != targetID {
		t.Fatalf("refreshed target EFFECT_START target = %d (present=%v), want cannon %d", got, ok, targetID)
	}
	pos, ok := args.GetArray("targetPos")
	if !ok {
		t.Fatal("refreshed target EFFECT_START has no fallback targetPos")
	}
	gotX, xOK := pos.GetFloat("x")
	gotY, yOK := pos.GetFloat("y")
	if !xOK || !yOK || math.Abs(gotX-float64(wantX)) > 0.001 || math.Abs(gotY-float64(wantY)) > 0.001 {
		t.Fatalf("refreshed target fallback targetPos = (%.3f,%.3f), want (%.3f,%.3f)", gotX, gotY, wantX, wantY)
	}
	return uid
}

func bakedBotTeleportCannon(t *testing.T, inst *huntInstance, bot *conn) *mobState {
	t.Helper()
	const structureID int32 = 8 // map_1_0's north-lane human DotaGun; stable baked ID.
	sc, ok := inst.dota.m.StructByID(structureID)
	if !ok || sc.Role != gamedata.DotaGun || sc.Prefab != "GA_Human_Gun_prop01" {
		t.Fatalf("setup: structure %d = %+v, want baked human DotaGun with GA_Human_Gun_prop01", structureID, sc)
	}
	target := inst.mobs[dotaStructIDBase+structureID]
	if target == nil || !target.structure || target.team != bot.playerTeam() || target.dotaRole != gamedata.DotaGun || target.dotaPrefab != sc.Prefab {
		t.Fatalf("setup: structure object = %+v, want live same-team baked DotaGun", target)
	}
	if got := botTeleportLane(bot, target); got != 0 {
		t.Fatalf("setup: baked cannon %d lane = %d, want bot lane 0", target.id, got)
	}
	return target
}

func startBakedBotTeleportCannon(t *testing.T) (*Server, *conn, *huntInstance, *botBrain, *mobState, *botTeleportVisualConn, float64, float32, float32, int32, int32, func()) {
	t.Helper()
	s, bot, inst, b, _, now, cleanup := newTeleportTestBot(t)
	wire := captureBotTeleportVisual(t, bot)
	target := bakedBotTeleportCannon(t, inst, bot)
	// A structure is a tactical reinforcement point only when its lane has
	// nearby activity. One enemy creep makes this baked cannon useful without
	// making the destination unsafe, while keeping the visual test focused on
	// the structure-target refresh lifecycle.
	pressure := teleportTestCreep(inst, 65991, dotaTeamElf, target.x, target.y)
	installTeleportTarget(inst, bot, target, pressure)

	bot.lock()
	wantX, wantY, ok := s.botTeleportDestinationLocked(bot, target)
	if !ok {
		bot.unlock()
		cleanup()
		t.Fatal("setup: baked cannon teleport destination unavailable")
	}
	if !s.botMaybeStartTeleportLocked(b, now) {
		bot.unlock()
		cleanup()
		t.Fatal("setup: teleport channel to baked cannon did not start")
	}
	markerUID, targetUID := assertBotTeleportPreparationStarts(t, wire.packets(t), bot.objID, target.id, wantX, wantY)
	bot.unlock()
	return s, bot, inst, b, target, wire, now, wantX, wantY, markerUID, targetUID, cleanup
}

func TestBotTeleportBakedCannonTargetRefreshEndsPreviousUID(t *testing.T) {
	for _, tc := range []struct {
		name   string
		finish func(*Server, *conn, *botBrain, *mobState, float64)
	}{
		{name: "cancel", finish: func(s *Server, bot *conn, b *botBrain, target *mobState, at float64) {
			target.dead = true
			s.botTickTeleportLocked(b, at)
		}},
		{name: "success", finish: func(s *Server, bot *conn, b *botBrain, _ *mobState, at float64) {
			s.botTickTeleportLocked(b, at)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, bot, _, b, target, wire, now, wantX, wantY, markerUID, targetUID, cleanup := startBakedBotTeleportCannon(t)
			defer cleanup()

			wire.reset()
			bot.lock()
			previousAnchor := b.pendingTeleport.targetAnchor
			targetUID2 := assertBotTeleportTargetRefresh(t, func() []battleproto.Packet {
				s.botTickTeleportLocked(b, now+2.0)
				return wire.packets(t)
			}(), targetUID, previousAnchor, bot.objID, target.id, wantX, wantY)
			bot.unlock()

			wire.reset()
			bot.lock()
			previousAnchor = b.pendingTeleport.targetAnchor
			targetUID3 := assertBotTeleportTargetRefresh(t, func() []battleproto.Packet {
				s.botTickTeleportLocked(b, now+4.0)
				return wire.packets(t)
			}(), targetUID2, previousAnchor, bot.objID, target.id, wantX, wantY)
			bot.unlock()
			if b.pendingTeleport == nil || b.pendingTeleport.targetFx != targetUID3 {
				t.Fatalf("latest pending target FX = %v, want %d", b.pendingTeleport, targetUID3)
			}

			wire.reset()
			bot.lock()
			complete := b.pendingTeleport.complete
			finishAt := now + 4.5
			if tc.name == "success" {
				finishAt = complete
			}
			tc.finish(s, bot, b, target, finishAt)
			bot.unlock()

			packets := wire.packets(t)
			if tc.name == "cancel" {
				if b.pendingTeleport != nil {
					t.Fatal("cancelled baked-cannon channel remained pending")
				}
				assertBotTeleportPreparationEnds(t, packets, markerUID, targetUID3)
				assertBotTeleportNoArrivalFX(t, packets)
				if len(bot.huntState.fxEnds) != 0 {
					t.Fatalf("cancelled baked-cannon channel left scheduled FX: %+v", bot.huntState.fxEnds)
				}
				return
			}

			if b.pendingTeleport != nil {
				t.Fatal("successful baked-cannon channel remained pending")
			}
			assertBotTeleportPreparationEnds(t, packets, markerUID, targetUID3)
			arrivalUID := assertBotTeleportArrivalStart(t, packets, bot.objID, bot.x, bot.y)
			wire.reset()
			bot.lock()
			s.memberTickLocked(bot, complete+2.0)
			bot.unlock()
			ends := botTeleportVisualEffectPackets(wire.packets(t), battleproto.CmdEffectEnd)
			if len(ends) != 1 {
				t.Fatalf("baked-cannon arrival cleanup EFFECT_END count = %d, want 1; commands=%v", len(ends), botTeleportVisualCommandNames(wire.packets(t)))
			}
			if id, ok := ends[0].Args.GetInt("id"); !ok || id != arrivalUID {
				t.Fatalf("baked-cannon arrival cleanup EFFECT_END id = %d (present=%v), want %d", id, ok, arrivalUID)
			}
			if len(bot.huntState.fxEnds) != 0 {
				t.Fatalf("baked-cannon arrival FX remained scheduled: %+v", bot.huntState.fxEnds)
			}
		})
	}
}

func assertBotTeleportNoArrivalFX(t *testing.T, packets []battleproto.Packet) {
	t.Helper()
	for _, p := range botTeleportVisualEffectPackets(packets, battleproto.CmdEffectStart) {
		if fx, _ := p.Args.GetString("fx"); fx == botTeleportTargetFx {
			args, ok := p.Args.GetArray("args")
			if !ok {
				t.Fatalf("cancelled teleport target EFFECT_START has no args: %v", botTeleportVisualCommandNames(packets))
			}
			if _, ok := args.GetInt("target"); !ok {
				t.Fatalf("cancelled teleport emitted final %s: %v", botTeleportTargetFx, botTeleportVisualCommandNames(packets))
			}
		}
	}
}

func assertBotTeleportArrivalStart(t *testing.T, packets []battleproto.Packet, ownerID int32, wantX, wantY float32) int32 {
	t.Helper()
	var starts []battleproto.Packet
	for _, p := range botTeleportVisualEffectPackets(packets, battleproto.CmdEffectStart) {
		if fx, _ := p.Args.GetString("fx"); fx == botTeleportTargetFx {
			args, ok := p.Args.GetArray("args")
			if !ok {
				t.Fatal("target EFFECT_START has no args")
			}
			if _, ok := args.GetInt("target"); !ok {
				starts = append(starts, p)
			}
		}
	}
	if len(starts) != 1 {
		t.Fatalf("arrival EFFECT_START count = %d, want 1; commands=%v", len(starts), botTeleportVisualCommandNames(packets))
	}
	p := starts[0]
	uid, ok := p.Args.GetInt("effect")
	if !ok || uid == 0 {
		t.Fatalf("arrival EFFECT_START uid = %d (present=%v), want nonzero", uid, ok)
	}
	anchorID, ok := p.Args.GetInt("owner")
	if !ok || anchorID == 0 || anchorID == ownerID {
		t.Fatalf("arrival EFFECT_START owner = %d (present=%v), want stationary Dummy anchor", anchorID, ok)
	}
	if len(packets) < 2 || packets[len(packets)-2].Cmd != battleproto.CmdSync {
		t.Fatalf("arrival anchor has no positioned SYNC: %v", botTeleportVisualCommandNames(packets))
	}
	args, ok := p.Args.GetArray("args")
	if !ok {
		t.Fatal("arrival EFFECT_START has no args")
	}
	pos, ok := args.GetArray("targetPos")
	if !ok {
		t.Fatal("arrival EFFECT_START has no targetPos")
	}
	gotX, xOK := pos.GetFloat("x")
	gotY, yOK := pos.GetFloat("y")
	if !xOK || !yOK || math.Abs(gotX-float64(wantX)) > 0.001 || math.Abs(gotY-float64(wantY)) > 0.001 {
		t.Fatalf("arrival targetPos = (%.3f,%.3f), want (%.3f,%.3f)", gotX, gotY, wantX, wantY)
	}
	return uid
}

func TestBotTeleportSuccessfulArrivalEndsMarkerAndCleansUpTargetFX(t *testing.T) {
	s, bot, inst, b, _, now, cleanup := newTeleportTestBot(t)
	defer cleanup()

	wire := captureBotTeleportVisual(t, bot)
	target := teleportTestCreep(inst, 64003, bot.playerTeam(), 0, 0)
	installTeleportTarget(inst, bot, target)

	bot.lock()
	wantX, wantY, ok := s.botTeleportDestinationLocked(bot, target)
	if !ok {
		bot.unlock()
		t.Fatal("setup: teleport destination unavailable")
	}
	if !s.botMaybeStartTeleportLocked(b, now) {
		bot.unlock()
		t.Fatal("setup: teleport channel did not start")
	}
	markerUID, targetUID := assertBotTeleportPreparationStarts(t, wire.packets(t), bot.objID, target.id, wantX, wantY)
	complete := b.pendingTeleport.complete
	s.botTickTeleportLocked(b, complete)
	bot.unlock()

	if b.pendingTeleport != nil {
		t.Fatal("successful teleport remained pending")
	}
	arrivalUID := assertBotTeleportArrivalStart(t, wire.packets(t), bot.objID, wantX, wantY)
	assertBotTeleportPreparationEnds(t, wire.packets(t), markerUID, targetUID)
	if len(bot.huntState.fxEnds) != 1 || bot.huntState.fxEnds[0].uid != arrivalUID || math.Abs(bot.huntState.fxEnds[0].at-(complete+2.0)) > 0.001 {
		t.Fatalf("scheduled arrival FX = %+v, want uid %d at %.3f", bot.huntState.fxEnds, arrivalUID, complete+2.0)
	}

	wire.reset()
	bot.lock()
	s.memberTickLocked(bot, complete+2.0)
	bot.unlock()
	ends := botTeleportVisualEffectPackets(wire.packets(t), battleproto.CmdEffectEnd)
	if len(ends) != 1 {
		t.Fatalf("arrival cleanup EFFECT_END count = %d, want 1; commands=%v", len(ends), botTeleportVisualCommandNames(wire.packets(t)))
	}
	if id, ok := ends[0].Args.GetInt("id"); !ok || id != arrivalUID {
		t.Fatalf("arrival cleanup EFFECT_END id = %d (present=%v), want %d", id, ok, arrivalUID)
	}
	if len(bot.huntState.fxEnds) != 0 {
		t.Fatalf("arrival FX remained scheduled after cleanup: %+v", bot.huntState.fxEnds)
	}
}

func TestBotTeleportAcceptedChannelEmitsMarkerFX(t *testing.T) {
	s, bot, inst, b, _, now, cleanup := newTeleportTestBot(t)
	defer cleanup()

	wire := captureBotTeleportVisual(t, bot)
	target := teleportTestCreep(inst, 64001, bot.playerTeam(), 0, 0)
	installTeleportTarget(inst, bot, target)

	bot.lock()
	wantX, wantY, ok := s.botTeleportDestinationLocked(bot, target)
	if !ok {
		bot.unlock()
		t.Fatal("setup: teleport destination unavailable")
	}
	started := s.botMaybeStartTeleportLocked(b, now)
	var markerUID, targetUID int32
	if started {
		markerUID, targetUID = assertBotTeleportPreparationStarts(t, wire.packets(t), bot.objID, target.id, wantX, wantY)
	}
	bot.unlock()
	if !started {
		t.Fatal("setup: teleport channel did not start")
	}
	if b.pendingTeleport == nil || b.pendingTeleport.markerFx != markerUID || b.pendingTeleport.targetFx != targetUID {
		t.Fatalf("pending preparation UIDs = %v, want marker=%d target=%d", b.pendingTeleport, markerUID, targetUID)
	}
}

func TestBotTeleportCancellationEndsMarkerWithoutArrivalFX(t *testing.T) {
	for _, tc := range []struct {
		name   string
		cancel func(*conn, *mobState, float64)
	}{
		{name: "invalid target", cancel: func(_ *conn, target *mobState, _ float64) { target.dead = true }},
		{name: "bot death", cancel: func(bot *conn, _ *mobState, now float64) { bot.huntState.deadUntil = now + 1 }},
		{name: "bot stun", cancel: func(bot *conn, _ *mobState, now float64) { bot.huntState.st.stunUntil = now + 1 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, bot, inst, b, _, now, cleanup := newTeleportTestBot(t)
			defer cleanup()
			wire := captureBotTeleportVisual(t, bot)
			target := teleportTestCreep(inst, 64002, bot.playerTeam(), 0, 0)
			installTeleportTarget(inst, bot, target)

			bot.lock()
			wantX, wantY, ok := s.botTeleportDestinationLocked(bot, target)
			if !ok {
				bot.unlock()
				t.Fatal("setup: teleport destination unavailable")
			}
			if !s.botMaybeStartTeleportLocked(b, now) {
				bot.unlock()
				t.Fatal("setup: teleport channel did not start")
			}
			markerUID, targetUID := assertBotTeleportPreparationStarts(t, wire.packets(t), bot.objID, target.id, wantX, wantY)
			tc.cancel(bot, target, now)
			s.botTickTeleportLocked(b, now+0.5)
			bot.unlock()

			if b.pendingTeleport != nil {
				t.Fatal("cancelled channel remained pending")
			}
			packets := wire.packets(t)
			assertBotTeleportPreparationEnds(t, packets, markerUID, targetUID)
			assertBotTeleportNoArrivalFX(t, packets)
		})
	}
}

func botTeleportVisualCommandNames(packets []battleproto.Packet) []string {
	commands := make([]string, len(packets))
	for i, p := range packets {
		commands[i] = p.Cmd.Name()
	}
	return commands
}

func TestBotTeleportHardResetRemoteViewer(t *testing.T) {
	s, bot, inst, _, _, _, cleanup := newTeleportTestBot(t)
	defer cleanup()

	tracked, trackedWire := addBotTeleportVisualViewer(t, s, inst, 71001, dotaTeamHuman, bot.x+4, bot.y)
	fog, fogWire := addBotTeleportVisualViewer(t, s, inst, 71002, dotaTeamElf, bot.x+avatarViewRadius+100, bot.y)
	now := float64(s.battleTime())
	wantX, wantY := bot.x+12, bot.y+7

	bot.lock()
	bot.x, bot.y, bot.vx, bot.vy, bot.snapT = wantX, wantY, 0, 0, float32(now)
	s.renderAvatarForLocked(tracked, bot, now)
	if tracked.huntState.tr.index(bot.objID) < 0 {
		bot.unlock()
		t.Fatal("setup: tracked remote viewer did not learn the bot avatar")
	}
	trackedWire.reset()
	fogWire.reset()
	s.hardResetAvatarForLocked(bot, now)
	bot.unlock()

	assertBotTeleportVisualReset(t, trackedWire.packets(t), bot.objID, wantX, wantY, false, 0, 0, 0)
	if got := tracked.huntState.tr.count(); got != 2 || tracked.huntState.tr.index(bot.objID) < 0 {
		t.Fatalf("tracked viewer tracker after reset = ids=%v count=%d, want one owner entry", tracked.huntState.tr.ids, got)
	}
	if len(fogWire.packets(t)) != 0 {
		t.Fatalf("untracked fog viewer received packets: %v", botTeleportVisualCommandNames(fogWire.packets(t)))
	}
	if fog.huntState.tr.index(bot.objID) >= 0 || fog.huntState.tr.count() != 1 {
		t.Fatalf("fog viewer tracker = ids=%v, want only self and no bot", fog.huntState.tr.ids)
	}
}

func TestBotTeleportVisualIntegrationHardResetBeforeDestinationConfirmation(t *testing.T) {
	s, bot, inst, b, target, _, now, cleanup := startTeleportTestChannel(t)
	defer cleanup()

	tracked, trackedWire := addBotTeleportVisualViewer(t, s, inst, 71003, dotaTeamHuman, bot.x+4, bot.y)
	fog, fogWire := addBotTeleportVisualViewer(t, s, inst, 71004, dotaTeamElf, bot.x+avatarViewRadius+100, bot.y)

	bot.lock()
	markerUID := b.pendingTeleport.markerFx
	targetUID := b.pendingTeleport.targetFx
	s.renderAvatarForLocked(tracked, bot, now)
	if tracked.huntState.tr.index(bot.objID) < 0 {
		bot.unlock()
		t.Fatal("setup: tracked remote viewer did not learn the bot avatar")
	}
	wantX, wantY, ok := s.botTeleportDestinationLocked(bot, target)
	if !ok {
		bot.unlock()
		t.Fatal("setup: teleport destination became unavailable")
	}
	complete := b.pendingTeleport.complete
	trackedWire.reset()
	fogWire.reset()
	s.botTickTeleportLocked(b, complete)
	bot.unlock()

	if b.pendingTeleport != nil {
		t.Fatal("successful teleport remained pending")
	}
	if math.Abs(float64(bot.x-wantX)) > 0.001 || math.Abs(float64(bot.y-wantY)) > 0.001 || bot.vx != 0 || bot.vy != 0 || bot.hasDest {
		t.Fatalf("authoritative bot state = pos(%.3f,%.3f) velocity(%.3f,%.3f) hasDest=%v, want destination and stopped", bot.x, bot.y, bot.vx, bot.vy, bot.hasDest)
	}
	if markerUID == 0 {
		t.Fatal("setup: channel has no marker FX uid")
	}
	if len(bot.huntState.fxEnds) != 1 {
		t.Fatalf("scheduled arrival FX count = %d, want 1", len(bot.huntState.fxEnds))
	}
	arrivalUID := bot.huntState.fxEnds[0].uid
	if arrivalUID == 0 || math.Abs(bot.huntState.fxEnds[0].at-(complete+2.0)) > 0.001 {
		t.Fatalf("scheduled arrival FX = %+v, want uid and expiry %.3f", bot.huntState.fxEnds[0], complete+2.0)
	}

	assertBotTeleportVisualReset(t, trackedWire.packets(t), bot.objID, wantX, wantY, true, markerUID, targetUID, arrivalUID)
	if got := tracked.huntState.tr.count(); got != 3 || tracked.huntState.tr.index(bot.objID) < 0 {
		t.Fatalf("tracked viewer tracker after teleport = ids=%v count=%d, want owner plus arrival anchor", tracked.huntState.tr.ids, got)
	}
	fogPackets := fogWire.packets(t)
	if len(fogPackets) != 5 || fogPackets[0].Cmd != battleproto.CmdEffectEnd || fogPackets[1].Cmd != battleproto.CmdEffectEnd ||
		fogPackets[2].Cmd != battleproto.CmdCreateObject || fogPackets[3].Cmd != battleproto.CmdSync || fogPackets[4].Cmd != battleproto.CmdEffectStart {
		t.Fatalf("fog viewer teleport FX packets = %v, want preparation ends plus visible arrival anchor", botTeleportVisualCommandNames(fogPackets))
	}
	for i, wantUID := range []int32{markerUID, targetUID} {
		if id, ok := fogPackets[i].Args.GetInt("id"); !ok || id != wantUID {
			t.Fatalf("fog preparation EFFECT_END %d id = %d (present=%v), want %d", i, id, ok, wantUID)
		}
	}
	if got, ok := fogPackets[4].Args.GetInt("effect"); !ok || got != arrivalUID {
		t.Fatalf("fog arrival EFFECT_START uid = %d (present=%v), want %d", got, ok, arrivalUID)
	}
	if fog.huntState.tr.index(bot.objID) >= 0 {
		t.Fatalf("fog viewer tracker changed on teleport: ids=%v", fog.huntState.tr.ids)
	}

	trackedWire.reset()
	fogWire.reset()
	bot.lock()
	s.memberTickLocked(bot, complete+2.0)
	bot.unlock()
	trackedEnds := botTeleportVisualEffectPackets(trackedWire.packets(t), battleproto.CmdEffectEnd)
	if len(trackedEnds) != 1 {
		t.Fatalf("arrival cleanup EFFECT_END count = %d, want 1; commands=%v", len(trackedEnds), botTeleportVisualCommandNames(trackedWire.packets(t)))
	}
	if id, ok := trackedEnds[0].Args.GetInt("id"); !ok || id != arrivalUID {
		t.Fatalf("arrival cleanup EFFECT_END id = %d (present=%v), want %d", id, ok, arrivalUID)
	}
	if len(bot.huntState.fxEnds) != 0 {
		t.Fatalf("arrival FX remained scheduled after cleanup: %+v", bot.huntState.fxEnds)
	}
	if fogEnds := botTeleportVisualEffectPackets(fogWire.packets(t), battleproto.CmdEffectEnd); len(fogEnds) != 1 {
		t.Fatalf("fog arrival cleanup EFFECT_END count = %d, want 1", len(fogEnds))
	}
}
