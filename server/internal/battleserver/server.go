// Package battleserver implements the raw-TCP "Battle" channel the client
// connects to after the Ctrl login/hero flow. It also powers the non-combat
// "central square" hub: the client won't show CentralSquareScreen until
// mCore.Battle exists, which only happens once this server completes the
// CONNECT handshake (BattleServerConnection.OnConnect -> Core.CreateBattle).
//
// Coverage so far: the connection handshake -- CONNECT, GET_TIME, ENTER, READY
// (see BattleServerConnection's state machine). Everything else is logged so we
// can point the real client at this server and observe exactly which packets it
// waits for next to actually render the square (the self-player registration
// chain: GAME_DATA + PROTOTYPE_INFO + CREATE_OBJECT + PLAYER_REG + SET_AVATAR,
// which drives SelfPlayer.Init -> BaseBattleScreen.ShowGui).
package battleserver

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"os"
	"strings"
	"sync"
	"time"

	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

type Server struct {
	Store *session.Store

	start        time.Time
	mu           sync.Mutex
	nextBattleID int32

	// insts holds the live shared hunt worlds keyed by room id. Several player
	// connections that were routed to the same room (open instance per map) share
	// one huntInstance: one authoritative mob simulation, one ticker, and every
	// member sees the others. Guarded by mu.
	insts map[int32]*huntInstance
}

func New(store *session.Store) *Server {
	return &Server{Store: store, start: time.Now(), nextBattleID: 1,
		insts: map[int32]*huntInstance{}}
}

// ListenAndServe accepts Battle connections on addr until the listener errors.
func (s *Server) ListenAndServe(addr string) error {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	log.Printf("battle: listening on %s", ln.Addr())
	return s.Serve(ln)
}

// Serve accepts Battle connections on an already-open listener (used by tests
// that bind an ephemeral port).
func (s *Server) Serve(ln net.Listener) error {
	for {
		conn, err := ln.Accept()
		if err != nil {
			return err
		}
		go s.handleConn(conn)
	}
}

// conn wraps a net.Conn with a write mutex so server-initiated pushes
// (PLAYER_REG etc.) can share the socket with handshake replies safely.
type conn struct {
	net.Conn
	r            *battleproto.Reader
	wm           sync.Mutex
	selfPlayerID int32
	objID        int32
	worldSent    bool

	// lk is the mutex guarding this connection's mutable state. For a lobby conn
	// (or a bare test conn) it is &mvMu -- private. For a hunt member it is
	// repointed at the shared huntInstance mutex so every member of one world
	// serializes on the same lock (movement + the shared mob simulation). All the
	// combat/movement code locks via c.lock()/c.unlock(); mvMu remains the backing
	// store for the non-hunt case.
	lk   *sync.Mutex
	inst *huntInstance

	// name is the display name used when this player's avatar is registered on
	// other members' clients (PLAYER_REG).
	name string

	// hunt is non-nil when this connection was launched through hunt|ready
	// (the Ctrl channel stored a PendingBattle and the client reconnected with
	// its one-time password). nil = the central-square lobby. huntState is the
	// live battle (mobs, skills, XP), created with the world state; both are
	// guarded by mvMu.
	hunt      *session.PendingBattle
	huntState *huntState
	nav       gamedata.Nav // walkability for the current scene (nil = free)

	// Movement state for the self avatar. The client dead-reckons from the last
	// POSITION sync, so we keep the last snapshot (x,y at snapT with velocity
	// vx,vy) and, while moving, an arrival timer that fires the stop sync.
	mvMu    sync.Mutex
	x, y    float32
	vx, vy  float32
	snapT   float32
	arrival *time.Timer

	// destX/destY is the whole move's final target; hasDest marks an in-flight
	// move so a live speed change (slow/haste) can re-issue it. moveGen is bumped
	// on every re-issue/stop so a stale fired arrival timer no-ops. path holds the
	// remaining waypoints of a routed move (last element == dest); the avatar
	// walks it leg by leg, one POSITION sync + arrival timer per leg.
	destX, destY float32
	hasDest      bool
	moveGen      int
	path         []gamedata.Vec2

	// chaseGoalX/Y is the goal the current combat chase (auto-attack pursuit or
	// approach-cast) last re-pathed to, and chaseRepathAt the battleTime of that
	// re-path. Together they gate the per-tick chase re-issue the way aimAlong's
	// staleness/drift throttle does for mobs (see chaseMoveLocked).
	chaseGoalX, chaseGoalY float32
	chaseRepathAt          float64
}

func (c *conn) send(p battleproto.Packet) error {
	c.wm.Lock()
	defer c.wm.Unlock()
	return battleproto.Write(c.Conn, p)
}

// lkOr returns the mutex guarding this conn's state: the shared instance lock
// once it joined a hunt world, else its private mvMu. Bare test conns that never
// set lk fall through to &mvMu, so a test that does c.lock() stays
// consistent with the code's c.lock().
func (c *conn) lkOr() *sync.Mutex {
	if c.lk != nil {
		return c.lk
	}
	return &c.mvMu
}

func (c *conn) lock()   { c.lkOr().Lock() }
func (c *conn) unlock() { c.lkOr().Unlock() }

// members returns the connections that share this conn's hunt world (so mob
// syncs, aggro and cross-player rendering fan out to all of them). Outside a
// shared instance -- the lobby, or a bare test conn -- it is just this conn, so
// every broadcast collapses to a single push and behaviour is unchanged.
func (c *conn) members() []*conn {
	if c.inst != nil {
		return c.inst.memberList()
	}
	return []*conn{c}
}

// testHookNewConn, when non-nil, receives each new server-side conn. Tests set it
// to reach into huntState (e.g. to pre-learn a level-gated skill); nil in prod.
var testHookNewConn func(*conn)

func (s *Server) handleConn(nc net.Conn) {
	defer nc.Close()
	remote := nc.RemoteAddr()
	log.Printf("battle: connection from %s", remote)
	c := &conn{Conn: nc, r: battleproto.NewReader(nc)}
	c.lk = &c.mvMu // private lock until this conn joins a shared hunt world
	if testHookNewConn != nil {
		testHookNewConn(c)
	}
	defer c.closeHunt() // invalidate combat/regen timers on disconnect
	for {
		p, err := c.r.Read()
		if err != nil {
			if errors.Is(err, io.EOF) {
				log.Printf("battle: %s disconnected", remote)
			} else {
				log.Printf("battle: %s read error: %v", remote, err)
			}
			return
		}
		s.handlePacket(c, p)
	}
}

func (s *Server) handlePacket(c *conn, p battleproto.Packet) {
	switch p.Cmd {
	case battleproto.CmdConnect:
		s.handleConnect(c, p)
	case battleproto.CmdGetTime:
		s.handleGetTime(c, p)
	case battleproto.CmdMovePlayer:
		s.handleMove(c, p)
	case battleproto.CmdStopPlayer:
		s.handleStop(c, p)
	case battleproto.CmdDoAction:
		s.handleDoAction(c, p)
	case battleproto.CmdUpgradeSkill:
		s.handleUpgradeSkill(c, p)
	case battleproto.CmdEnter:
		log.Printf("battle: %s ENTER", c.RemoteAddr())
		s.ack(c, p)
	case battleproto.CmdReady:
		log.Printf("battle: %s READY", c.RemoteAddr())
		s.ack(c, p)
		// The client has finished the CONNECT/ENTER/READY handshake and loaded
		// the local scene; now populate the world so the self player gets
		// initialised (which is what finally fires CentralSquareScreen.ShowGui
		// and replaces the loading screen). See sendWorldState.
		s.sendWorldState(c)
	default:
		log.Printf("battle: %s UNHANDLED %s req=%d args=%s",
			c.RemoteAddr(), p.Cmd.Name(), p.RequestID, dump(p.Args))
		// Reply with a success ack so validation-only client commands don't
		// stall; commands the client parses for data are still no-ops until
		// implemented, which the log above flags for follow-up.
		s.ack(c, p)
	}
}

// handleConnect answers CONNECT with a ConnectArg the client parses as
// {clientId -> mSelfPlayerId, battleId -> mBattleId}. We echo the client's own
// user id as its in-battle player id (deterministic and unambiguous).
func (s *Server) handleConnect(c *conn, p battleproto.Packet) {
	clientID := p.Args.IntOr("clientId", -1)
	pass := p.Args.StringOr("pass", "")
	battleID := s.newBattleID()
	c.selfPlayerID = clientID

	// A game-mode launch (hunt|ready) hands the client a one-time password; if
	// it matches the pending battle for this user, this connection is that
	// battle rather than the lobby. The pending entry is consumed either way a
	// match happens, so a later plain reconnect returns to the square.
	if pb, ok := s.Store.TakePendingBattle(clientID); ok {
		if pb.Passwd != "" && pb.Passwd == pass {
			c.hunt = &pb
			log.Printf("battle: %s CONNECT user=%d entering HUNT map=%d avatar=%d scene=%s",
				c.RemoteAddr(), clientID, pb.MapID, pb.AvatarID, pb.Scene)
		} else {
			log.Printf("battle: %s CONNECT user=%d wrong battle pass (got %q) -> lobby",
				c.RemoteAddr(), clientID, pass)
		}
	}

	log.Printf("battle: %s CONNECT clientId=%d pass=%q -> selfPlayerId=%d battleId=%d",
		c.RemoteAddr(), clientID, pass, clientID, battleID)

	args := amf.NewArray().
		Set("clientId", clientID).
		Set("battleId", battleID)
	if err := c.send(battleproto.Packet{
		Cmd:       battleproto.CmdConnect,
		Args:      args,
		RequestID: p.RequestID,
		Status:    true,
	}); err != nil {
		log.Printf("battle: %s CONNECT reply error: %v", c.RemoteAddr(), err)
	}
}

// handleGetTime answers GET_TIME with {time: <seconds since server start>},
// which the client feeds into BattleTimer.Sync (ServerTimeArgParser reads
// "time").
func (s *Server) handleGetTime(c *conn, p battleproto.Packet) {
	t := time.Since(s.start).Seconds()
	args := amf.NewArray().Set("time", t)
	if err := c.send(battleproto.Packet{
		Cmd:       battleproto.CmdGetTime,
		Args:      args,
		RequestID: p.RequestID,
		Status:    true,
	}); err != nil {
		log.Printf("battle: %s GET_TIME reply error: %v", c.RemoteAddr(), err)
	}
}

// ack replies with a bare success packet echoing the request id, for the
// client's validation-only commands (ENTER, READY, ...).
func (s *Server) ack(c *conn, p battleproto.Packet) {
	if err := c.send(battleproto.Packet{
		Cmd:       p.Cmd,
		Args:      amf.NewArray(),
		RequestID: p.RequestID,
		Status:    true,
	}); err != nil {
		log.Printf("battle: %s %s ack error: %v", c.RemoteAddr(), p.Cmd.Name(), err)
	}
}

const (
	heroPrototypeID int32 = 1
	// heroProtoDesc is the PROTOTYPE_INFO "desc" XML for the self player's
	// avatar. prefab "Hero" makes the client render the player's own customized
	// hero (appearance comes from its SelfHero data, not the battle channel);
	// PAvatar marks it as an avatar. Node names/shape are exactly what
	// PropertyHolder.RetrieveProperties -> BattlePrototype.P* loaders read.
	heroProtoDesc = `<Proto><PPrefab value="Hero"/><PAvatar value="true"/>` +
		`<PDesc><Name value="Hero"/><Short value=""/><Long value=""/><Icon value=""/></PDesc></Proto>`
)

// sendWorldState pushes the minimum battle state needed to initialise the self
// player and render the central square. The chain (order matters, see below)
// drives Battle.PerformAvatars -> SelfPlayer.Init -> BaseBattleScreen.ShowGui,
// which is the only thing that replaces the loading screen:
//
//	GAME_DATA      - teams/coefs (empty is fine for the hub)
//	PROTOTYPE_INFO - the "Hero" avatar prototype (must precede CREATE_OBJECT)
//	PLAYER_REG     - registers the self player (must precede SET_AVATAR)
//	CREATE_OBJECT  - builds the avatar GameObject (suspends client packet
//	                 processing until the async build finishes)
//	SET_AVATAR     - binds player<->avatar; with the object now present this
//	                 sets mSelfInited and triggers the bind -> Init -> ShowGui
func (s *Server) sendWorldState(c *conn) {
	if c.worldSent {
		return
	}
	c.worldSent = true
	self := c.selfPlayerID
	objID := 1000 + self
	c.objID = objID
	// Spawn at the lobby origin, unless this is a hunt: the combat maps put the
	// playable area far from (0,0), so use the map's measured spawn.
	sx, sy := spawnX, spawnY
	if c.hunt == nil {
		// Central-square lobby: enforce the scene's walkability and spawn at that
		// scene's own spawn point. The scene is chosen by the AREA the client asked
		// area_conf for (a player can visit the other race's city via the portal), so
		// use the recorded lobby area; fall back to the hero's home square by race for a
		// first entry that never hit area_conf (or a bare test conn). Elf square's
		// walkable plaza is far from (0,0), so the nav's Spawn() is authoritative.
		area := s.Store.LobbyArea(self)
		if area == 0 {
			area = 367 // human city default (Location.CS_HUMAN)
			if u, ok := s.Store.ByID(self); ok && u.Hero != nil && u.Hero.Race == 2 {
				area = gamedata.AreaCSElf
			}
		}
		if nav := gamedata.LobbyNav(area); nav != nil {
			c.nav = nav
			nx, ny := nav.Spawn()
			sx, sy = float32(nx), float32(ny)
		}
	}
	if c.hunt != nil {
		// Join (or create) the shared world for this launch's room. Ctrl assigns a
		// positive room id for the open per-map instance; a missing/<=0 room (tests,
		// or a solo launch) gets a private world keyed by the negative user id so it
		// never collides with a Ctrl room. joinInstance repoints c.lk at the
		// instance lock and starts the ticker for a fresh world.
		room := c.hunt.Room
		if room <= 0 {
			room = -self
		}
		inst := s.joinInstance(room, c.hunt.MapID, c)
		mx, my := inst.m.Spawn()
		sx, sy = float32(mx), float32(my)
		c.nav = inst.nav // enforce this scene's walkability on movement
	}
	c.lock()
	c.x, c.y, c.vx, c.vy, c.snapT = sx, sy, 0, 0, s.battleTime()
	c.unlock()
	name := "Hero"
	if u, ok := s.Store.ByID(self); ok && u.Username != "" {
		name = u.Username
	}
	c.name = name

	if c.hunt != nil {
		s.sendHuntWorldState(c, name)
		return
	}

	pkts := []battleproto.Packet{
		{Cmd: battleproto.CmdGameData, Args: amf.NewArray().
			Set("data", `<root><battle time_limit="0" frag_limit="0"/></root>`).
			Set("relics", amf.NewArray())},
		protoInfoPkt(heroPrototypeID, heroProtoDesc),
		{Cmd: battleproto.CmdPlayerReg, Args: amf.NewArray().
			Set("id", self).Set("name", name).Set("team", int32(1)).Set("avatar", int32(-1))},
		{Cmd: battleproto.CmdCreateObject, Args: amf.NewArray().
			Set("id", objID).Set("proto", heroPrototypeID)},
		{Cmd: battleproto.CmdSetAvatar, Args: amf.NewArray().
			Set("playerID", self).Set("avatarID", objID).Set("level", int32(1)).Set("points", int32(0))},
		// SET_AVATAR -> SelfPlayer.Init only *subscribes* to the avatar's next
		// POSITION sync (and sends SET_STATE); it is that first POSITION sync
		// that sets mInited and fires the ShowGui callback. So we must also push
		// a SYNC that makes the object visible and gives it a position.
		{Cmd: battleproto.CmdSync, Args: amf.NewArray().
			Set("data", positionSync(objID, true, sx, sy, 0, 0, s.battleTime()))},
		// A SPEED stat sync so the run animation actually plays: AnimationExt
		// sets the run clip's playback rate from syncedParams.mSpeed (the SPEED
		// stat). Without it the stat is 0 and the run clip is frozen -- the hero
		// slides without leg movement. Must follow the POSITION sync above (which
		// registers the object as tracking index 0).
		{Cmd: battleproto.CmdSync, Args: amf.NewArray().
			Set("data", speedSync(objID, lobbyMoveSpeed, s.battleTime()))},
	}
	log.Printf("battle: %s sending world state (self=%d obj=%d name=%q)", c.RemoteAddr(), self, objID, name)
	s.sendSeq(c, pkts)
}

// sendSeq pushes a list of server-initiated packets in order (Status true,
// RequestID -1), logging each; it stops on the first write error.
func (s *Server) sendSeq(c *conn, pkts []battleproto.Packet) {
	for _, p := range pkts {
		p.Status = true
		p.RequestID = -1
		if err := c.send(p); err != nil {
			log.Printf("battle: %s world %s send error: %v", c.RemoteAddr(), p.Cmd.Name(), err)
			return
		}
		log.Printf("battle: %s -> %s %s", c.RemoteAddr(), p.Cmd.Name(), dump(p.Args))
	}
}

// positionSync builds the SYNC packet "data" byte blob (parsed by
// SyncPacket.Parse) carrying one POSITION sample for objID. When add is true it
// also registers objID as a freshly-visible tracked object (tracking index 0),
// which is required exactly once, on the first sync. The layout is little-endian
// (the client uses BitConverter on Windows) and must be consumed exactly:
//
//	float32   time
//	int16     visibility-entry count (1 when add, else 0)
//	int32     objID | 0x80000000     (only when add: the "add" mask -> new id)
//	uint64    present sync-type bitmask (POSITION = bit 0)
//	byte      per-object bitmask for POSITION (ceil(idsCount/8)=1 byte, bit 0 set)
//	5×float32 POSITION values: x, y, velX, velY, snapshotTime
//
// The client extrapolates position by dead reckoning (pos = xy + vel*(t-snapT),
// see SyncedParams.GetPosition), so a moving avatar needs only two syncs: one at
// the start of a leg (with velocity) and one on arrival (velocity 0).
func positionSync(objID int32, add bool, x, y, velX, velY, t float32) []byte {
	const addMask = uint32(0x80000000) // TrackingIdManager: int.MinValue = "add"
	buf := new(bytes.Buffer)
	le := binary.LittleEndian
	_ = binary.Write(buf, le, t)
	if add {
		_ = binary.Write(buf, le, int16(1))
		_ = binary.Write(buf, le, uint32(objID)|addMask)
	} else {
		_ = binary.Write(buf, le, int16(0))
	}
	_ = binary.Write(buf, le, uint64(0x1)) // POSITION
	buf.WriteByte(0x01)                    // object at tracking index 0 has POSITION
	for _, v := range []float32{x, y, velX, velY, t} {
		_ = binary.Write(buf, le, v)
	}
	return buf.Bytes()
}

const (
	// spawnX/spawnY: where the hero appears in the central square (scene origin).
	spawnX float32 = 0
	spawnY float32 = 0
	// lobbyMoveSpeed is the hero's run speed in world units/sec. The client
	// renders whatever velocity we send; this only sets how fast it walks.
	lobbyMoveSpeed float32 = 4.0
)

func (s *Server) battleTime() float32 {
	return float32(time.Since(s.start).Seconds())
}

// speedSync builds a SYNC "data" blob carrying just the SPEED stat for objID
// (which must already be a registered tracking id, index 0). SyncType.SPEED =
// 0x400000; it is a single float. Layout (little-endian), consumed exactly:
//
//	float32  time
//	int16    visibility count = 0 (no add/remove)
//	uint64   type mask = SPEED
//	byte     per-object bitmask (bit 0 set)
//	float32  speed value
func speedSync(objID int32, speed, t float32) []byte {
	const speedType = uint64(0x400000) // SyncType.SPEED
	buf := new(bytes.Buffer)
	le := binary.LittleEndian
	_ = binary.Write(buf, le, t)
	_ = binary.Write(buf, le, int16(0))
	_ = binary.Write(buf, le, speedType)
	buf.WriteByte(0x01)
	_ = binary.Write(buf, le, speed)
	return buf.Bytes()
}

// handleMove processes MOVE_PLAYER: the client sends an absolute world target
// (both the minimap's rel=false and the ground-drag's rel=true carry world
// coords; see PlayerControl.Move). We ack (MOVE_PLAYER is validation-only on the
// client) and drive the avatar toward the target with POSITION syncs.
// calibrateNav, when HUNT_CALIBRATE=1, disables movement clipping (the avatar
// walks to any clicked point) and logs every click with the nav grid's verdict.
// Clicking points that are actually reachable then reveals where the generated
// grid is wrong (nav=BLOCKED on a reachable point => grid too tight there).
var calibrateNav = os.Getenv("HUNT_CALIBRATE") == "1"

// debugCombat logs each mob->player hit (attacker id, positions, alive-mob
// count) to help diagnose phantom/unseen-attacker damage. On by default during
// this debugging phase; set HUNT_DEBUG=0 to silence.
var debugCombat = os.Getenv("HUNT_DEBUG") != "0"

func (s *Server) handleMove(c *conn, p battleproto.Packet) {
	s.ack(c, p)
	tp, ok := p.Args.GetArray("targetPos")
	if !ok {
		return
	}
	tx, _ := tp.GetFloat("x")
	ty, _ := tp.GetFloat("y")
	// Log the click with the grid's verdict so a reachable-point click that the
	// grid rejects stands out as a calibration miss.
	verdict := "nofnav"
	if c.nav != nil {
		if c.nav.Walkable(float64(tx), float64(ty)) {
			verdict = "walk"
		} else {
			verdict = "BLOCKED"
		}
	}
	log.Printf("battle: %s CLICK target=(%.2f, %.2f) nav=%s", c.RemoteAddr(), tx, ty, verdict)

	c.lock()
	defer c.unlock()
	now := s.battleTime()
	// Reject movement while a cast animation roots the avatar; re-freeze in place
	// so any client-side move prediction snaps back to the cast spot.
	if hs := c.huntState; hs != nil && float64(now) < hs.castLockUntil {
		cx, cy := c.posAtLocked(now)
		c.stopArrivalLocked()
		c.hasDest = false
		c.x, c.y, c.vx, c.vy, c.snapT = cx, cy, 0, 0, now
		c.sendPosLocked(s, cx, cy, 0, 0, now)
		return
	}
	// A manual move order breaks the auto-attack session and any pending cast.
	if hs := c.huntState; hs != nil {
		if hs.attackTarget != 0 {
			s.stopAttackLocked(c, false)
		}
		s.cancelOrderLocked(c)
	}
	c.moveToLocked(s, float32(tx), float32(ty))
}

// handleStop processes STOP_PLAYER. CRITICAL: the client sends
// STOP_PLAYER{stop:false} on every right-button RELEASE ~100-200ms after the
// MOVE_PLAYER of a ground click (BattleInput.OnMouseClickRight MouseUp) — it
// only means "button released", NOT "halt". Halting on it made ground clicks
// walk a few tenths of a unit and freeze (while minimap clicks, which never
// send STOP_PLAYER, walked fine). Only stop:true (never sent by this client,
// but handled for completeness) freezes the avatar.
func (s *Server) handleStop(c *conn, p battleproto.Packet) {
	s.ack(c, p)
	if !p.Args.BoolOr("stop", false) {
		return
	}
	c.lock()
	defer c.unlock()
	now := s.battleTime()
	cx, cy := c.posAtLocked(now)
	c.stopArrivalLocked()
	c.hasDest = false // an explicit halt is not "moving": let channels keep running
	c.x, c.y, c.vx, c.vy, c.snapT = cx, cy, 0, 0, now
	c.sendPosLocked(s, cx, cy, 0, 0, now)
}

// moveToLocked starts a straight-line move to (tx,ty) for callers that already hold
// mvMu (combat chase): one POSITION sync with the leg's velocity now, plus a
// scheduled arrival sync (velocity 0) at the ETA. It routes AROUND walls: Nav.Path
// returns the waypoints, and the avatar walks them leg by leg (one POSITION sync +
// arrival timer each). A clear straight shot yields a single leg — identical to the
// old straight-line behaviour — so unobstructed moves and the lobby (nav==nil) are
// unchanged.
func (c *conn) moveToLocked(s *Server, tx, ty float32) {
	now := s.battleTime()
	cx, cy := c.posAtLocked(now)
	c.stopArrivalLocked()

	// Build a route around obstacles. Without a nav (lobby) or in calibration
	// mode, or if pathfinding can't resolve a route, fall back to a single
	// straight (clipped) leg — the historical behaviour.
	var route []gamedata.Vec2
	if c.nav != nil && !calibrateNav {
		route = c.nav.Path(float64(cx), float64(cy), float64(tx), float64(ty))
	}
	if len(route) == 0 {
		etx, ety := tx, ty
		if c.nav != nil && !calibrateNav {
			clx, cly := c.nav.Clip(float64(cx), float64(cy), float64(tx), float64(ty))
			etx, ety = float32(clx), float32(cly)
		}
		route = []gamedata.Vec2{{X: float64(etx), Y: float64(ety)}}
	}

	c.path = route
	c.destX, c.destY = float32(route[len(route)-1].X), float32(route[len(route)-1].Y)
	c.hasDest = true
	c.x, c.y, c.snapT = cx, cy, now
	c.startLegLocked(s, now)
}

// startLegLocked walks toward the head of c.path from the current snapshot: it
// sends one POSITION sync at the leg's velocity and schedules the arrival timer,
// which pops the waypoint and either starts the next leg or halts. Empty/degenerate
// waypoints are skipped. Caller holds mvMu.
func (c *conn) startLegLocked(s *Server, now float32) {
	for len(c.path) > 0 {
		wp := c.path[0]
		tx, ty := float32(wp.X), float32(wp.Y)
		dx, dy := tx-c.x, ty-c.y
		dist := float32(math.Hypot(float64(dx), float64(dy)))
		if dist < 0.05 {
			c.path = c.path[1:] // already on this waypoint; drop it
			c.x, c.y = tx, ty
			continue
		}
		speed := c.moveSpeedLocked(s)
		if speed <= 0 {
			speed = 0.0001
		}
		vx := dx / dist * speed
		vy := dy / dist * speed
		c.vx, c.vy, c.snapT = vx, vy, now
		c.sendPosLocked(s, c.x, c.y, vx, vy, now)

		// Capture this leg's generation so a fired-but-mvMu-blocked arrival from a
		// superseded move/stop becomes a no-op (Go can't cancel an already-fired
		// AfterFunc; stopArrivalLocked bumps moveGen on every re-issue/stop).
		gen := c.moveGen
		eta := time.Duration(float64(dist/speed) * float64(time.Second))
		c.arrival = time.AfterFunc(eta, func() {
			c.lock()
			defer c.unlock()
			if c.moveGen != gen {
				return // superseded by a newer move/stop
			}
			at := s.battleTime()
			c.arrival = nil
			c.x, c.y, c.snapT = tx, ty, at
			if len(c.path) > 0 {
				c.path = c.path[1:]
			}
			if len(c.path) == 0 {
				c.hasDest = false
				c.vx, c.vy = 0, 0
				c.sendPosLocked(s, tx, ty, 0, 0, at)
				return
			}
			c.startLegLocked(s, at) // turn onto the next leg
		})
		return
	}
	// No walkable leg to run (all waypoints coincided with the current spot).
	c.hasDest = false
	c.path = nil
	c.vx, c.vy = 0, 0
	c.sendPosLocked(s, c.x, c.y, 0, 0, now)
}

// resetChaseLocked forgets the chase throttle state so the next chaseMoveLocked
// re-paths unconditionally — a new chase session must not inherit the previous
// session's goal (which could sit within the drift tolerance by coincidence).
func (c *conn) resetChaseLocked() {
	c.chaseGoalX, c.chaseGoalY = float32(math.Inf(1)), float32(math.Inf(1))
	c.chaseRepathAt = 0
}

// chaseMoveLocked re-paths a combat chase (auto-attack pursuit / approach-cast)
// toward (tx,ty) only when it is worth it: immediately when the goal has
// drifted >1m since the last chase re-path, and at most once per second when
// the walker went idle short of the goal (a clipped/failed route, or the target
// nudged less than the tolerance). Without the gate every 200-250ms combat
// re-arm would re-run A*, cancel and recreate the arrival timer and push a
// redundant POSITION sync even for a target that has not moved — and a
// nil-route target would re-run a worst-case search every tick (the failure
// mode aimAlong's throttle guards against for mobs).
func (c *conn) chaseMoveLocked(s *Server, tx, ty float32) {
	now := float64(s.battleTime())
	drift := math.Hypot(float64(tx-c.chaseGoalX), float64(ty-c.chaseGoalY))
	if drift <= 1.0 && (c.hasDest || now-c.chaseRepathAt < 1.0) {
		return // same goal and the walker is busy (or retried recently): keep walking
	}
	c.chaseGoalX, c.chaseGoalY = tx, ty
	c.chaseRepathAt = now
	c.moveToLocked(s, tx, ty)
}

// moveStraightLocked walks a single straight leg to (tx,ty), clipped to walkable
// ground — no routing around walls. Used by burst-movement skills (dash) that
// should lunge in a straight line and stop at terrain rather than curve around
// it, matching the pre-pathfinding straight-line behaviour.
func (c *conn) moveStraightLocked(s *Server, tx, ty float32) {
	c.moveStraightExLocked(s, tx, ty, true)
}

// moveStraightExLocked is moveStraightLocked with an explicit clip flag: clip=false
// drives the leg straight to (tx,ty) THROUGH walls (a no-clip charge), used by
// obstacle-ignoring dashes; clip=true keeps the walkable-ground clamp.
func (c *conn) moveStraightExLocked(s *Server, tx, ty float32, clip bool) {
	now := s.battleTime()
	cx, cy := c.posAtLocked(now)
	c.stopArrivalLocked()
	if clip && c.nav != nil && !calibrateNav {
		clx, cly := c.nav.Clip(float64(cx), float64(cy), float64(tx), float64(ty))
		tx, ty = float32(clx), float32(cly)
	}
	c.path = []gamedata.Vec2{{X: float64(tx), Y: float64(ty)}}
	c.destX, c.destY, c.hasDest = tx, ty, true
	c.x, c.y, c.snapT = cx, cy, now
	c.startLegLocked(s, now)
}

// moveSpeedLocked is the avatar's current move speed: the status-modified hunt
// speed (slows/hastes) when in a hunt, else the flat lobby speed.
func (c *conn) moveSpeedLocked(s *Server) float32 {
	if c.huntState != nil {
		return float32(c.curSpeedLocked(float64(s.battleTime())))
	}
	return lobbyMoveSpeed
}

// posAtLocked returns the dead-reckoned position at time now. Caller holds mvMu.
func (c *conn) posAtLocked(now float32) (float32, float32) {
	dt := now - c.snapT
	return c.x + c.vx*dt, c.y + c.vy*dt
}

// stopArrivalLocked cancels the in-flight arrival timer, drops any remaining
// routed waypoints, and bumps the move generation so any already-fired
// (mvMu-blocked) arrival closure no-ops. Every stop/re-issue site funnels
// through here, so it is the single place the leftover path is cleared. (The
// per-leg arrival closure advances the path without calling this, so a move in
// progress is not disturbed.)
func (c *conn) stopArrivalLocked() {
	c.moveGen++
	c.path = nil
	if c.arrival != nil {
		c.arrival.Stop()
		c.arrival = nil
	}
}

// sendPosLocked pushes one POSITION sync for this avatar. In a hunt the avatar is
// a shared object: the move is broadcast to every member that renders it (each
// with its OWN tracking index and object count), so teammates see this player
// walk. The lobby (no huntState) uses the fixed index-0, 1-byte helper. Called
// under the world lock so syncs for a connection stay ordered.
func (c *conn) sendPosLocked(s *Server, x, y, vx, vy, t float32) {
	if c.huntState != nil {
		c.mobViewersLocked(c.objID, func(mem *conn, idx, count int) {
			s.push(mem, battleproto.CmdSync, amf.NewArray().Set("data",
				newSyncBlob(t).position(idx, x, y, vx, vy, t).build(count)))
		})
		return
	}
	data := positionSync(c.objID, false, x, y, vx, vy, t)
	if err := c.send(battleproto.Packet{
		Cmd:       battleproto.CmdSync,
		Args:      amf.NewArray().Set("data", data),
		RequestID: -1,
		Status:    true,
	}); err != nil {
		log.Printf("battle: %s pos sync error: %v", c.RemoteAddr(), err)
	}
}

func (s *Server) newBattleID() int32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	id := s.nextBattleID
	s.nextBattleID++
	return id
}

// dump renders a MixedArray compactly for logging (associative keys then dense
// values). It is best-effort and only used for observing unhandled packets.
func dump(m *amf.MixedArray) string {
	if m == nil {
		return "<nil>"
	}
	var b strings.Builder
	b.WriteByte('{')
	first := true
	for k, v := range m.Assoc {
		if !first {
			b.WriteString(", ")
		}
		first = false
		b.WriteString(k)
		b.WriteByte('=')
		writeVal(&b, v)
	}
	for i, v := range m.Dense {
		if !first {
			b.WriteString(", ")
		}
		first = false
		fmt.Fprintf(&b, "[%d]=", i)
		writeVal(&b, v)
	}
	b.WriteByte('}')
	return b.String()
}

func writeVal(b *strings.Builder, v interface{}) {
	if a, ok := v.(*amf.MixedArray); ok {
		b.WriteString(dump(a))
		return
	}
	fmt.Fprintf(b, "%v", v)
}
