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
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync/atomic"
	"time"

	"tanatserver/internal/gamedata"
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
	ID                 int32   `json:"id"`
	Name               string  `json:"name"`
	Avatar             string  `json:"avatar"`
	IsBot              bool    `json:"is_bot"`
	Team               int32   `json:"team"`
	AIVersion          int     `json:"ai_version,omitempty"`
	X                  float32 `json:"x"`
	Y                  float32 `json:"y"`
	HPFrac             float64 `json:"hp_frac"`
	ManaFrac           float64 `json:"mana_frac"`
	Level              int32   `json:"level"`
	Dead               bool    `json:"dead"`
	AttackTarget       int32   `json:"attack_target,omitempty"`
	AttackActionActive bool    `json:"attack_action_active,omitempty"`
	PvpTarget          int32   `json:"pvp_target,omitempty"`
	Phase              string  `json:"phase,omitempty"`
	Lane               int     `json:"lane,omitempty"`
	EngageTarget       int32   `json:"engage_target,omitempty"`
	Retreating         bool    `json:"retreating,omitempty"`
	RetreatMode        string  `json:"retreat_mode,omitempty"`
	PlanMode           string  `json:"plan_mode,omitempty"`
	PlanLane           int     `json:"plan_lane"`
	PlanObjective      int32   `json:"plan_objective"`
	Assignment         string  `json:"assignment,omitempty"`
	AssignmentReason   string  `json:"assignment_reason,omitempty"`
	AssignmentLane     int     `json:"assignment_lane"`
	Coverage           bool    `json:"coverage"`
	XP                 float64 `json:"xp,omitempty"`
	XPPerMinute        float64 `json:"xp_per_minute,omitempty"`
	FarmDecision       string  `json:"farm_decision,omitempty"`
	FarmTarget         int32   `json:"farm_target,omitempty"`
	FarmScore          float64 `json:"farm_score,omitempty"`
	FarmLane           int     `json:"farm_lane"`
	FarmCatchUp        bool    `json:"farm_catch_up,omitempty"`
	FarmLastHits       int     `json:"farm_last_hits,omitempty"`
	FarmWaveClears     int     `json:"farm_wave_clears,omitempty"`
	FarmXPEvents       int     `json:"farm_xp_events,omitempty"`
	FarmLastXPTAt      float64 `json:"farm_last_xp_at,omitempty"`
	FarmProgressAge    float64 `json:"farm_progress_age,omitempty"`
	FarmTargetDistance float64 `json:"farm_target_distance,omitempty"`
	FarmInXPRadius     bool    `json:"farm_in_xp_radius"`
	FarmDebt           int     `json:"farm_debt,omitempty"`
	FocusTarget        int32   `json:"focus_target,omitempty"`
}

// telemetryBotAttack records an attack that actually crossed the combat engine's
// commit point. A snapshot's attack_target is only an intention; these events are the
// authoritative distinction between a selected target and a visible swing/hit.
type telemetryBotAttack struct {
	telemetryEvent
	BotID    int32   `json:"bot_id"`
	TargetID int32   `json:"target_id"`
	Damage   float64 `json:"damage,omitempty"`
	Fatal    bool    `json:"fatal,omitempty"`
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

// telemetryXPGrant is one recipient's share of a Dota reward. raw_xp is the full
// source bounty before proximity splitting; received_xp is what this recipient actually
// received after splitting and any active XP multiplier. Keeping both makes a match log
// sufficient to distinguish farming efficiency from simple team stacking.
type telemetryXPGrant struct {
	telemetryEvent
	RecipientID    int32   `json:"recipient_id"`
	RecipientIsBot bool    `json:"recipient_is_bot"`
	Team           int32   `json:"team"`
	Source         string  `json:"source"`
	SourceID       int32   `json:"source_id,omitempty"`
	RawXP          float64 `json:"raw_xp"`
	ReceivedXP     float64 `json:"received_xp"`
	Recipients     int     `json:"recipients"`
	VictimLevel    int32   `json:"victim_level,omitempty"`
}

// telemetryCreepLifecycle makes the farm denominator observable. XP grants
// alone show what a bot received, but not how many lane creeps actually spawned
// or died outside every bot's XP radius.
type telemetryCreepLifecycle struct {
	telemetryEvent
	CreepID     int32   `json:"creep_id"`
	MobIndex    int     `json:"mob_index"`
	Team        int32   `json:"team"`
	Lane        int     `json:"lane"`
	LaneIndex   int     `json:"lane_index"`
	X           float32 `json:"x"`
	Y           float32 `json:"y"`
	KillerID    int32   `json:"killer_id,omitempty"`
	KillerIsBot bool    `json:"killer_is_bot,omitempty"`
}

type telemetryCreepXPMiss struct {
	telemetryEvent
	CreepID int32   `json:"creep_id"`
	Team    int32   `json:"team"`
	X       float32 `json:"x"`
	Y       float32 `json:"y"`
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

type telemetryAI40Action struct {
	telemetryEvent
	BotID     int32   `json:"bot_id"`
	Team      int32   `json:"team"`
	ModelID   string  `json:"model_id"`
	Kind      uint8   `json:"kind"`
	Target    uint16  `json:"target"`
	Direction uint8   `json:"direction"`
	Distance  uint8   `json:"distance"`
	LatencyMS float64 `json:"latency_ms"`
	Accepted  bool    `json:"accepted"`
}

type telemetryAI40Fallback struct {
	telemetryEvent
	Team    int32  `json:"team"`
	ModelID string `json:"model_id,omitempty"`
	Reason  string `json:"reason"`
}

// telemetryBotTeleport is the lifecycle record for a bot's battle-local
// teleport channel. FX ids are retained on every transition so a JSONL reader
// can correlate preparation/target visuals with the authoritative outcome.
type telemetryBotTeleport struct {
	telemetryEvent
	BotID        int32   `json:"bot_id"`
	TargetID     int32   `json:"target_id"`
	TargetKind   string  `json:"target_kind"`
	DestinationX float32 `json:"destination_x"`
	DestinationY float32 `json:"destination_y"`
	MarkerFX     int32   `json:"marker_fx"`
	TargetFX     int32   `json:"target_fx"`
	ArrivalFX    int32   `json:"arrival_fx"`
	CancelReason string  `json:"cancel_reason,omitempty"`
}

// telemetryBotPurchase makes bot economy observable in the same match log as
// movement, combat, and XP. A successful purchase is recorded after the
// atomic debit and includes both balances, so a log reader can distinguish a
// rejected affordability decision from a real item purchase.
type telemetryBotPurchase struct {
	telemetryEvent
	BotID       int32 `json:"bot_id"`
	ArticleID   int32 `json:"article_id"`
	TreeID      int32 `json:"tree_id"`
	Stage       int   `json:"stage"`
	Price       int32 `json:"price"`
	MoneyBefore int32 `json:"money_before"`
	MoneyAfter  int32 `json:"money_after"`
}

// telemetryBotStructureEscape records a bot's geometric response to a live
// enemy structure. It is deliberately independent of the teleport lifecycle
// so disabled telemetry and tests can call the same nil-safe helper.
type telemetryBotStructureEscape struct {
	telemetryEvent
	BotID        int32   `json:"bot_id"`
	ThreatID     int32   `json:"threat_id"`
	DestinationX float32 `json:"destination_x"`
	DestinationY float32 `json:"destination_y"`
	Reason       string  `json:"reason"`
	HPFraction   float64 `json:"hp_fraction"`
}

type telemetryBotRetreatDestination struct {
	X float32 `json:"x"`
	Y float32 `json:"y"`
}

type telemetryBotRetreatUtility struct {
	telemetryEvent
	BotID       int32                           `json:"bot_id"`
	Slot        int                             `json:"slot"`
	TargetID    int32                           `json:"target_id"`
	Reason      string                          `json:"reason"`
	Category    string                          `json:"category"`
	HPFraction  float64                         `json:"hp_fraction"`
	Destination *telemetryBotRetreatDestination `json:"destination,omitempty"`
}

type telemetryBotAssignment struct {
	BotID        int32  `json:"bot_id"`
	Role         string `json:"role"`
	Mode         string `json:"mode"`
	Reason       string `json:"reason"`
	Lane         int    `json:"lane"`
	FarmLane     int    `json:"farm_lane"`
	FarmLaneSet  bool   `json:"farm_lane_set"`
	BaselineLane int    `json:"baseline_lane"`
	Coverage     bool   `json:"coverage"`
	Aggressive   bool   `json:"aggressive"`
	Objective    int32  `json:"objective,omitempty"`
}

// telemetryBotTeamPlan is edge-triggered: a plan is recorded only when its
// live-state mode/lane/objective or an assignment changes, avoiding per-tick spam.
type telemetryBotTeamPlan struct {
	telemetryEvent
	AIVersion     int                      `json:"ai_version"`
	Team          int32                    `json:"team"`
	Mode          string                   `json:"mode"`
	Lane          int                      `json:"lane"`
	Objective     int32                    `json:"objective,omitempty"`
	ObjectiveKind string                   `json:"objective_kind,omitempty"`
	Reason        string                   `json:"reason"`
	FocusTarget   int32                    `json:"focus_target,omitempty"`
	Assignments   []telemetryBotAssignment `json:"assignments"`
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

func botRetreatModeName(mode botRetreatMode) string {
	if mode == botRetreatModeDisengage {
		return "disengage"
	}
	return "recovery"
}

func (s *Server) telemetryRecordBotTeleportLocked(c *conn, typ string, now float64, targetID int32, targetKind string, x, y float32, markerFX, targetFX, arrivalFX int32, reason string) {
	if c == nil || c.inst == nil || c.inst.dota == nil || c.inst.dota.telemetry == nil {
		return
	}
	c.inst.dota.telemetry.record(telemetryBotTeleport{
		telemetryEvent: newTelemetryEvent(typ, botMatchTime(c, now)),
		BotID:          c.objID, TargetID: targetID, TargetKind: targetKind,
		DestinationX: x, DestinationY: y,
		MarkerFX: markerFX, TargetFX: targetFX, ArrivalFX: arrivalFX,
		CancelReason: reason,
	})
}

func (s *Server) telemetryRecordBotPurchaseLocked(c *conn, it gamedata.AvatarItem, moneyBefore, moneyAfter int32, now float64) {
	if c == nil || !isBotConn(c) || c.inst == nil || c.inst.dota == nil || c.inst.dota.telemetry == nil {
		return
	}
	c.inst.dota.telemetry.record(telemetryBotPurchase{
		telemetryEvent: newTelemetryEvent("bot_item_purchase", botMatchTime(c, now)),
		BotID:          c.objID, ArticleID: it.ArticleID, TreeID: it.TreeID, Stage: it.Stage,
		Price: it.Price, MoneyBefore: moneyBefore, MoneyAfter: moneyAfter,
	})
}

func (s *Server) telemetryRecordBotStructureEscapeLocked(c *conn, threat *mobState, x, y float32, reason string, now float64) {
	if c == nil || c.inst == nil || c.inst.dota == nil || c.huntState == nil || c.inst.dota.telemetry == nil {
		return
	}
	threatID := int32(0)
	if threat != nil {
		threatID = threat.id
	}
	c.inst.dota.telemetry.record(telemetryBotStructureEscape{
		telemetryEvent: newTelemetryEvent("bot_structure_escape", botMatchTime(c, now)),
		BotID:          c.objID, ThreatID: threatID,
		DestinationX: x, DestinationY: y, Reason: reason,
		HPFraction: botHPFrac(c.huntState, now),
	})
}

func (s *Server) telemetryRecordBotRetreatUtilityLocked(c *conn, slot int, targetID int32, reason, category string, now float64, x, y float32, hasDestination bool) {
	if c == nil || c.inst == nil || c.inst.dota == nil || c.huntState == nil || c.inst.dota.telemetry == nil {
		return
	}
	var destination *telemetryBotRetreatDestination
	if hasDestination {
		destination = &telemetryBotRetreatDestination{X: x, Y: y}
	}
	c.inst.dota.telemetry.record(telemetryBotRetreatUtility{
		telemetryEvent: newTelemetryEvent("bot_retreat_utility", botMatchTime(c, now)),
		BotID:          c.objID, Slot: slot, TargetID: targetID, Reason: reason, Category: category,
		HPFraction: botHPFrac(c.huntState, now), Destination: destination,
	})
}

func (s *Server) telemetryRecordBotAttackStartLocked(c *conn, targetID int32, now float64) {
	if c == nil || !isBotConn(c) || c.inst == nil || c.inst.dota == nil || c.inst.dota.telemetry == nil {
		return
	}
	c.inst.dota.telemetry.record(telemetryBotAttack{
		telemetryEvent: newTelemetryEvent("bot_attack_start", botMatchTime(c, now)),
		BotID:          c.objID, TargetID: targetID,
	})
}

func (s *Server) telemetryRecordBotAttackHitLocked(c *conn, targetID int32, damage float64, fatal bool, now float64) {
	if c == nil || !isBotConn(c) || c.inst == nil || c.inst.dota == nil || c.inst.dota.telemetry == nil {
		return
	}
	c.inst.dota.telemetry.record(telemetryBotAttack{
		telemetryEvent: newTelemetryEvent("bot_attack_hit", botMatchTime(c, now)),
		BotID:          c.objID, TargetID: targetID, Damage: damage, Fatal: fatal,
	})
}

func (s *Server) telemetryRecordBotAttackCancelLocked(c *conn, targetID int32, reason string, now float64) {
	if c == nil || !isBotConn(c) || c.inst == nil || c.inst.dota == nil || c.inst.dota.telemetry == nil {
		return
	}
	c.inst.dota.telemetry.record(struct {
		telemetryEvent
		BotID    int32  `json:"bot_id"`
		TargetID int32  `json:"target_id"`
		Reason   string `json:"reason"`
	}{
		telemetryEvent: newTelemetryEvent("bot_attack_cancel", botMatchTime(c, now)),
		BotID:          c.objID, TargetID: targetID, Reason: reason,
	})
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
		AttackTarget: hs.attackTarget, AttackActionActive: hs.attackActionActive, PvpTarget: hs.pvpTarget,
	}
	if c.inst != nil {
		if brain := c.inst.bots[c.objID]; brain != nil {
			snap.AIVersion = botAIVersionForBrain(brain)
			if !botAIProfileForBrain(brain).UsesTeamOrchestrator() {
				snap.PlanMode = "legacy_local"
				snap.Assignment = "local_brain"
				snap.AssignmentReason = "ai0_no_orchestrator"
			}
			xpPerMinute := 0.0
			if c.inst.dota != nil {
				minutes := (now - c.inst.dota.startedAt) / 60
				if minutes > 0 {
					xpPerMinute = hs.xp / minutes
				}
			}
			snap.Phase = botPhaseName(brain.phase)
			snap.Lane = brain.lane
			snap.EngageTarget = brain.engageTarget
			snap.Retreating = brain.retreating
			if brain.retreating {
				snap.RetreatMode = botRetreatModeName(brain.retreatMode)
			}
			snap.PlanMode = brain.macroAssignment.Mode
			snap.PlanLane = brain.macroAssignment.Lane
			snap.PlanObjective = brain.macroAssignment.ObjectiveID
			snap.Assignment = brain.macroAssignment.Role
			snap.AssignmentReason = brain.macroAssignment.Reason
			snap.AssignmentLane = brain.macroAssignment.Lane
			snap.Coverage = brain.macroAssignment.Coverage
			snap.XP, snap.XPPerMinute = hs.xp, xpPerMinute
			snap.FarmDecision, snap.FarmTarget, snap.FarmScore = brain.farmDecision, brain.farmTarget, brain.farmTargetScore
			snap.FarmLane, snap.FarmCatchUp = brain.farmLane, brain.farmCatchUp
			snap.FarmLastHits, snap.FarmWaveClears = brain.farmLastHits, brain.farmWaveClears
			snap.FarmXPEvents, snap.FarmLastXPTAt = brain.farmXPEvents, brain.farmLastXPTAt
			if brain.farmLastXPTAt > 0 && now > brain.farmLastXPTAt {
				snap.FarmProgressAge = now - brain.farmLastXPTAt
			}
			if brain.farmTarget != 0 {
				if target := c.inst.mobs[brain.farmTarget]; target != nil && !target.dead {
					cx, cy := c.posAtLocked(float32(now))
					snap.FarmTargetDistance = math.Hypot(float64(target.x-cx), float64(target.y-cy))
					snap.FarmInXPRadius = snap.FarmTargetDistance <= dotaXPShareRadius
				}
			}
			snap.FarmDebt = botFarmDebtLocked(c.inst, brain)
			if c.inst.dota != nil {
				if plan, ok := c.inst.dota.teamPlans[c.playerTeam()]; ok {
					snap.FocusTarget = plan.FocusTarget
				}
			}
		}
	}
	rec.record(snap)
}

func (s *Server) telemetryRecordBotTeamPlanLocked(inst *huntInstance, plan botTeamPlan, now float64) {
	if inst == nil || inst.dota == nil || inst.dota.telemetry == nil {
		return
	}
	ids := make([]int32, 0, len(plan.Assignments))
	for id := range plan.Assignments {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	assignments := make([]telemetryBotAssignment, 0, len(ids))
	for _, id := range ids {
		a := plan.Assignments[id]
		assignments = append(assignments, telemetryBotAssignment{
			BotID: id, Role: a.Role, Mode: a.Mode, Reason: a.Reason,
			Lane: a.Lane, FarmLane: a.FarmLane, FarmLaneSet: a.FarmLaneSet,
			BaselineLane: a.BaselineLane, Coverage: a.Coverage, Aggressive: a.Aggressive,
			Objective: a.ObjectiveID,
		})
	}
	inst.dota.telemetry.record(telemetryBotTeamPlan{
		telemetryEvent: newTelemetryEvent("bot_team_plan", planTelemetryMatchTime(inst, now)),
		Team:           plan.Team, AIVersion: botPlanAIVersionLocked(inst, &plan), Mode: plan.Mode, Lane: plan.Lane, Objective: plan.ObjectiveID,
		ObjectiveKind: plan.Objective, Reason: plan.Reason, FocusTarget: plan.FocusTarget, Assignments: assignments,
	})
}

func planTelemetryMatchTime(inst *huntInstance, now float64) float64 {
	if inst == nil || inst.dota == nil {
		return 0
	}
	return inst.dota.telemetryMatchTimeLocked(now)
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
	s.telemetryRecordDamageWithCreditLocked(rec, c, damagerID, pvpAttacker, nil, dmg, fatal, matchTime, now)
}

// telemetryRecordDamageWithCreditLocked keeps the raw visual damager (a creep in
// the relevant case) while recording the hero that receives the authoritative kill
// credit. This makes telemetry agree with PLAYER_STATS/ON_KILL instead of hiding a
// corrected hero kill behind the creep's object id.
func (s *Server) telemetryRecordDamageWithCreditLocked(rec *telemetryRecorder, c *conn, damagerID int32, pvpAttacker, creditedKiller *conn, dmg float64, fatal bool, matchTime, now float64) {
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
	killer := pvpAttacker
	if creditedKiller != nil {
		killer = creditedKiller
	}
	px, py := c.posAtLocked(float32(now))
	rec.record(telemetryDeath{
		telemetryEvent: newTelemetryEvent("death", matchTime),
		VictimID:       c.objID, VictimIsBot: isBotConn(c),
		KillerID: func() int32 {
			if killer != nil {
				return killer.objID
			}
			return damagerID
		}(),
		KillerIsHero: killer != nil,
		KillerIsBot:  killer != nil && isBotConn(killer),
		X:            px, Y: py,
	})
}

// telemetryRecordXPGrantLocked records only calibrated Dota rewards. Hunt and Arena
// continue to use the same XP engine but intentionally do not enter this source-level
// stream, so their telemetry remains semantically separate.
func (s *Server) telemetryRecordXPGrantLocked(c *conn, team int32, source string, sourceID int32, rawXP, receivedXP float64, recipients int, victimLevel int32, now float64) {
	if source == "" || c == nil || c.inst == nil || c.inst.dota == nil || c.inst.dota.telemetry == nil {
		return
	}
	c.inst.dota.telemetry.record(telemetryXPGrant{
		telemetryEvent: newTelemetryEvent("xp_grant", c.inst.dota.telemetryMatchTimeLocked(now)),
		RecipientID:    c.objID, RecipientIsBot: isBotConn(c), Team: team,
		Source: source, SourceID: sourceID, RawXP: rawXP, ReceivedXP: receivedXP,
		Recipients: recipients, VictimLevel: victimLevel,
	})
}

func (s *Server) telemetryRecordCreepSpawnLocked(c *conn, m *mobState, now float64) {
	if c == nil || m == nil || m.structure || c.inst == nil || c.inst.dota == nil || c.inst.dota.telemetry == nil {
		return
	}
	lane := botLaneForCreep(c, m)
	c.inst.dota.telemetry.record(telemetryCreepLifecycle{
		telemetryEvent: newTelemetryEvent("creep_spawn", c.inst.dota.telemetryMatchTimeLocked(now)),
		CreepID:        m.id, MobIndex: m.mobIdx, Team: m.team, Lane: lane, LaneIndex: m.laneIdx,
		X: m.x, Y: m.y,
	})
}

func (s *Server) telemetryRecordCreepDeathLocked(c *conn, m *mobState, killer *conn, now float64) {
	killerID := int32(0)
	if killer != nil {
		killerID = killer.objID
	}
	s.telemetryRecordCreepDeathByIDLocked(c, m, killerID, killer, now)
}

func (s *Server) telemetryRecordCreepDeathByIDLocked(c *conn, m *mobState, killerID int32, killer *conn, now float64) {
	if c == nil || m == nil || m.structure || c.inst == nil || c.inst.dota == nil || c.inst.dota.telemetry == nil {
		return
	}
	lane := botLaneForCreep(c, m)
	ev := telemetryCreepLifecycle{
		telemetryEvent: newTelemetryEvent("creep_death", c.inst.dota.telemetryMatchTimeLocked(now)),
		CreepID:        m.id, MobIndex: m.mobIdx, Team: m.team, Lane: lane, LaneIndex: m.laneIdx,
		X: m.x, Y: m.y,
	}
	if killerID != 0 {
		ev.KillerID = killerID
		ev.KillerIsBot = killer != nil && isBotConn(killer)
	}
	c.inst.dota.telemetry.record(ev)
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
