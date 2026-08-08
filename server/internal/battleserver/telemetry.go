package battleserver

// «Штурм» bot telemetry: an opt-in, per-match recording of every hero's position and
// actions (bots AND real players, so a bot's play can be compared against the humans
// around it), written as JSON Lines to one file per match. Off by default -- nothing
// here runs, allocates, or touches disk unless TANAT_BOT_TELEMETRY is set, matching
// the existing HUNT_DEBUG/TANAT_DOTA_BOTS dev-switch convention (see server.go,
// bot.go). Purely observational: nothing in this file ever changes game state or an
// AI decision, only records ones already made elsewhere.
//
// Every event line carries `type` and `t` (seconds since this match's world was
// built, i.e. dotaState.startedAt) plus enough identity (id/name/is_bot/team) to be
// self-contained -- a line never needs a join against a separate roster record to be
// meaningful on its own.

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync/atomic"
	"time"
)

// botTelemetryDir returns the directory a match's telemetry file should be written
// into, or "" if telemetry is disabled. Unset/empty TANAT_BOT_TELEMETRY = off (the
// default); "1" = the literal directory "telemetry" (relative to the server's working
// directory); any other value = that directory path, so a full path can be given too.
func botTelemetryDir() string {
	switch v := os.Getenv("TANAT_BOT_TELEMETRY"); v {
	case "":
		return ""
	case "1":
		return "telemetry"
	default:
		return v
	}
}

// telemetryMatchSeq disambiguates two matches created in the same wall-clock second
// (a fast dev loop of short bot-only test matches hits this easily).
var telemetryMatchSeq int64

// telemetryBufferSize is how many pending events the writer goroutine can lag behind
// by before events start being dropped. Sized generously: 10 heroes snapshotting every
// 200ms is 50 events/sec at steady state, so this is ~80s of full buffering headroom.
const telemetryBufferSize = 4096

// telemetryRecorder buffers one match's events on a channel and serializes them to its
// own JSONL file on a dedicated goroutine, so a slow disk can never stall inst.mu --
// every call site in this package holds it. Owned by dotaState; created in
// newDotaInstance when telemetry is enabled, closed once in dotaEndLocked.
type telemetryRecorder struct {
	matchID string
	ch      chan any
	done    chan struct{}
	dropped int64
}

// newTelemetryRecorder creates the match's JSONL file and starts its writer goroutine.
// Returns nil (a valid, inert receiver -- every method on it nil-checks) if the
// directory or file can't be created, so a permissions/disk problem degrades to "no
// telemetry this match" rather than failing the match itself.
func newTelemetryRecorder(dir string, mapID int32) *telemetryRecorder {
	id := fmt.Sprintf("%d-%d-%d", mapID, time.Now().Unix(), atomic.AddInt64(&telemetryMatchSeq, 1))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		log.Printf("battle: telemetry: mkdir %s: %v (disabled for this match)", dir, err)
		return nil
	}
	path := filepath.Join(dir, id+".jsonl")
	f, err := os.Create(path)
	if err != nil {
		log.Printf("battle: telemetry: create %s: %v (disabled for this match)", path, err)
		return nil
	}
	r := &telemetryRecorder{matchID: id, ch: make(chan any, telemetryBufferSize), done: make(chan struct{})}
	go r.run(f)
	log.Printf("battle: telemetry: recording match %s to %s", id, path)
	return r
}

func (r *telemetryRecorder) run(f *os.File) {
	defer f.Close()
	enc := json.NewEncoder(f)
	for ev := range r.ch {
		if err := enc.Encode(ev); err != nil {
			log.Printf("battle: telemetry: write %s: %v", r.matchID, err)
		}
	}
	close(r.done)
}

// record enqueues an event without ever blocking the caller: a full buffer (the writer
// falling behind, or disk stalled) drops the event instead of stalling the match tick.
func (r *telemetryRecorder) record(ev any) {
	if r == nil {
		return
	}
	select {
	case r.ch <- ev:
	default:
		atomic.AddInt64(&r.dropped, 1)
	}
}

// close flushes and stops the writer goroutine, blocking until the file is closed (the
// match is already over at every call site, so there is nothing left for the caller to
// do concurrently anyway).
func (r *telemetryRecorder) close() {
	if r == nil {
		return
	}
	close(r.ch)
	<-r.done
	if d := atomic.LoadInt64(&r.dropped); d > 0 {
		log.Printf("battle: telemetry: match %s dropped %d events (buffer full)", r.matchID, d)
	}
}

// ---- event shapes ----

// telemetryEvent is embedded (and so flattened by encoding/json) into every event
// struct below: the two fields every line carries regardless of type.
type telemetryEvent struct {
	Type string  `json:"type"`
	T    float64 `json:"t"`
}

func newTelemetryEvent(typ string, matchTime float64) telemetryEvent {
	return telemetryEvent{Type: typ, T: matchTime}
}

type telemetryMatchStart struct {
	telemetryEvent
	MapID int32 `json:"map_id"`
}

// telemetrySnapshot is one hero's state on one tick (200ms cadence, see
// runInstanceTicker's member loop). Bot-only fields (Phase/Lane/EngageTarget/
// Retreating) are omitted for a real player.
type telemetrySnapshot struct {
	telemetryEvent
	ID           int32   `json:"id"`
	Name         string  `json:"name"`
	Avatar       string  `json:"avatar"`
	IsBot        bool    `json:"is_bot"`
	Team         int32   `json:"team"`
	X            float32 `json:"x"`
	Y            float32 `json:"y"`
	HPFrac       float64 `json:"hp_frac"`
	ManaFrac     float64 `json:"mana_frac"`
	Level        int32   `json:"level"`
	Dead         bool    `json:"dead"`
	AttackTarget int32   `json:"attack_target,omitempty"`
	PvpTarget    int32   `json:"pvp_target,omitempty"`
	Phase        string  `json:"phase,omitempty"`
	Lane         int     `json:"lane,omitempty"`
	EngageTarget int32   `json:"engage_target,omitempty"`
	Retreating   bool    `json:"retreating,omitempty"`
}

// telemetryCast is a genuinely-fired ability cast (mana/cooldown already passed --
// see execCastLocked, where this is recorded right after mana is spent).
type telemetryCast struct {
	telemetryEvent
	ActorID    int32   `json:"actor_id"`
	ActorIsBot bool    `json:"actor_is_bot"`
	Slot       int     `json:"slot"`
	TargetID   int32   `json:"target_id,omitempty"`
	X          float32 `json:"x,omitempty"`
	Y          float32 `json:"y,omitempty"`
}

// telemetryDamage is one landed hit on a HERO (hitPlayerFromLocked) -- creep/structure
// chip damage is not recorded here at all (see hitMobFlagsLocked's own hook, which
// only records a structure's actual destruction, not every swing against it -- the
// volume of ordinary creep/tower combat is not bot-behavior data).
type telemetryDamage struct {
	telemetryEvent
	VictimID       int32   `json:"victim_id"`
	VictimIsBot    bool    `json:"victim_is_bot"`
	AttackerID     int32   `json:"attacker_id"` // raw damager object id: a hero, or a mob/structure
	AttackerIsHero bool    `json:"attacker_is_hero"`
	AttackerIsBot  bool    `json:"attacker_is_bot,omitempty"`
	Damage         float64 `json:"damage"`
	Fatal          bool    `json:"fatal"`
}

type telemetryDeath struct {
	telemetryEvent
	VictimID     int32   `json:"victim_id"`
	VictimIsBot  bool    `json:"victim_is_bot"`
	KillerID     int32   `json:"killer_id"`
	KillerIsHero bool    `json:"killer_is_hero"`
	KillerIsBot  bool    `json:"killer_is_bot,omitempty"`
	X            float32 `json:"x"`
	Y            float32 `json:"y"`
}

type telemetryStructureDestroy struct {
	telemetryEvent
	StructureID  int32 `json:"structure_id"`
	Role         int   `json:"role"`
	Team         int32 `json:"team"`
	KillerID     int32 `json:"killer_id"`
	KillerIsHero bool  `json:"killer_is_hero"`
	KillerIsBot  bool  `json:"killer_is_bot,omitempty"`
}

type telemetryFinalStats struct {
	ID      int32  `json:"id"`
	Name    string `json:"name"`
	Avatar  string `json:"avatar"`
	IsBot   bool   `json:"is_bot"`
	Team    int32  `json:"team"`
	Frags   int32  `json:"frags"`
	Deaths  int32  `json:"deaths"`
	Assists int32  `json:"assists"`
	Level   int32  `json:"level"`
	Money   int32  `json:"money"`
}

type telemetryMatchEnd struct {
	telemetryEvent
	Winner   int32                 `json:"winner"`
	Duration float64               `json:"duration"`
	Final    []telemetryFinalStats `json:"final"`
}

// botPhaseName stringifies a bot's current phase for the snapshot stream.
func botPhaseName(p botPhase) string {
	switch p {
	case botPhaseRoam:
		return "roam"
	case botPhaseGroup:
		return "group"
	default:
		return "lane"
	}
}

// ---- recording entry points (called from the real game-logic call sites) ----

// telemetrySnapshotLocked records one member's current state. Called once per member
// per world tick (200ms) from runInstanceTicker; a no-op instantly if rec is nil
// (telemetry disabled) or the member isn't a live hero.
func (s *Server) telemetrySnapshotLocked(rec *telemetryRecorder, c *conn, matchTime, now float64) {
	if rec == nil {
		return
	}
	hs := c.huntState
	if hs == nil {
		return
	}
	px, py := c.posAtLocked(float32(now))
	var hpFrac, manaFrac float64
	if max := hs.maxHPLocked(now); max > 0 {
		hpFrac = hs.hp / max
	}
	if max := hs.maxManaLocked(now); max > 0 {
		manaFrac = hs.mana / max
	}
	snap := telemetrySnapshot{
		telemetryEvent: newTelemetryEvent("snapshot", matchTime),
		ID:             c.objID, Name: c.name, Avatar: hs.av.Prefab,
		IsBot: isBotConn(c), Team: c.playerTeam(),
		X: px, Y: py, HPFrac: hpFrac, ManaFrac: manaFrac, Level: hs.level,
		Dead:         hs.deadUntil > 0,
		AttackTarget: hs.attackTarget, PvpTarget: hs.pvpTarget,
	}
	if c.inst != nil {
		if brain := c.inst.bots[c.objID]; brain != nil {
			snap.Phase = botPhaseName(brain.phase)
			snap.Lane = brain.lane
			snap.EngageTarget = brain.engageTarget
			snap.Retreating = brain.retreating
		}
	}
	rec.record(snap)
}

// telemetryRecordCastLocked records a genuinely-fired ability cast (called from
// execCastLocked right after mana is spent -- a cast that failed the mana/cooldown
// gate never reaches this point, so this is real intent, not every button press).
func (s *Server) telemetryRecordCastLocked(rec *telemetryRecorder, c *conn, slot int, ms *mobState, px, py float32, hasPos bool, matchTime float64) {
	if rec == nil {
		return
	}
	ev := telemetryCast{
		telemetryEvent: newTelemetryEvent("cast", matchTime),
		ActorID:        c.objID, ActorIsBot: isBotConn(c), Slot: slot,
	}
	if ms != nil {
		ev.TargetID = ms.id
	} else if hasPos {
		ev.X, ev.Y = px, py
	}
	rec.record(ev)
}

// telemetryRecordDamageLocked records one landed hit on a hero, and (when fatal) the
// matching death -- both from hitPlayerFromLocked, the single choke point for all
// hero-received damage regardless of source (hero, creep, tower).
func (s *Server) telemetryRecordDamageLocked(rec *telemetryRecorder, c *conn, damagerID int32, pvpAttacker *conn, dmg float64, fatal bool, matchTime, now float64) {
	if rec == nil {
		return
	}
	rec.record(telemetryDamage{
		telemetryEvent: newTelemetryEvent("damage", matchTime),
		VictimID:       c.objID, VictimIsBot: isBotConn(c),
		AttackerID: damagerID, AttackerIsHero: pvpAttacker != nil,
		AttackerIsBot: pvpAttacker != nil && isBotConn(pvpAttacker),
		Damage:        dmg, Fatal: fatal,
	})
	if !fatal {
		return
	}
	px, py := c.posAtLocked(float32(now))
	rec.record(telemetryDeath{
		telemetryEvent: newTelemetryEvent("death", matchTime),
		VictimID:       c.objID, VictimIsBot: isBotConn(c),
		KillerID: damagerID, KillerIsHero: pvpAttacker != nil,
		KillerIsBot: pvpAttacker != nil && isBotConn(pvpAttacker),
		X:           px, Y: py,
	})
}

// telemetryRecordStructureDestroyLocked records a structure's destruction (called from
// hitMobFlagsLocked's death branch when ms.structure -- ordinary chip damage against a
// structure is not recorded, only the kill itself).
func (s *Server) telemetryRecordStructureDestroyLocked(rec *telemetryRecorder, ms *mobState, killer *conn, matchTime float64) {
	if rec == nil {
		return
	}
	ev := telemetryStructureDestroy{
		telemetryEvent: newTelemetryEvent("structure_destroy", matchTime),
		StructureID:    ms.id, Role: int(ms.dotaRole), Team: ms.team,
	}
	if killer != nil {
		ev.KillerID = killer.objID
		ev.KillerIsHero = true
		ev.KillerIsBot = isBotConn(killer)
	}
	rec.record(ev)
}

// telemetryRecordMatchEndLocked records the final scoreboard and stops the recorder.
// Called once from dotaEndLocked; safe to call with rec==nil (no-op).
func (s *Server) telemetryRecordMatchEndLocked(rec *telemetryRecorder, inst *huntInstance, winner int32, matchTime float64) {
	if rec == nil {
		return
	}
	var final []telemetryFinalStats
	for _, mem := range inst.members {
		hs := mem.huntState
		if hs == nil {
			continue
		}
		money, _, _ := s.Store.HeroMoney(mem.selfPlayerID)
		final = append(final, telemetryFinalStats{
			ID: mem.objID, Name: mem.name, Avatar: hs.av.Prefab,
			IsBot: isBotConn(mem), Team: mem.playerTeam(),
			Frags: hs.frags, Deaths: hs.deaths, Assists: hs.assists,
			Level: hs.level, Money: money,
		})
	}
	rec.record(telemetryMatchEnd{
		telemetryEvent: newTelemetryEvent("match_end", matchTime),
		Winner:         winner, Duration: matchTime, Final: final,
	})
	rec.close()
}
