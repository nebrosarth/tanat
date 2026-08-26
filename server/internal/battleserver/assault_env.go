package battleserver

import (
	"cmp"
	"crypto/sha256"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"strings"
	"time"

	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

const (
	AssaultHeroCount         = 10
	AssaultMaxEntities       = 96
	AssaultHeroFeatureSize   = 32
	AssaultEntityFeatures    = 16
	AssaultGlobalFeatures    = 32
	AssaultAbilityFeatures   = 40
	AssaultActionKinds       = 8
	AssaultNavigationOffsets = 81
	AssaultNavigationAnchors = 15
	AssaultTick              = 200 * time.Millisecond
)

// Teacher status is deliberately separate from the executed-action validity
// bit. The v13 teacher stream is a per-slot state machine: an authoritative
// representable order starts with ACTION, repeats as HOLD, retires as CANCEL,
// and is otherwise WAIT. A live order that cannot be represented by the actor's
// observation/masks is UNAVAILABLE rather than a fabricated action.
const (
	AssaultTeacherStatusNone        uint8 = 0
	AssaultTeacherStatusAction      uint8 = 1
	AssaultTeacherStatusWait        uint8 = 2
	AssaultTeacherStatusHold        uint8 = 3
	AssaultTeacherStatusCancel      uint8 = 4
	AssaultTeacherStatusUnavailable uint8 = 5
)

// AI-42 evaluation controls are transported separately from the factorized
// action. ISSUE applies the action payload, HOLD preserves the authoritative
// order, and IDLE guarantees that no order remains active. Keeping this outside
// HeroActionV1 preserves all established v4-v13 action layouts.
type AssaultControlV1 uint8

const (
	AssaultControlIssue AssaultControlV1 = iota
	AssaultControlHold
	AssaultControlIdle
	AssaultControlCount
)

// Rejection reasons are stable v13 wire values.  Unknown is the fail-closed
// value for an action that was not submitted to this controller slot.
const (
	AssaultRejectionReasonNone           uint8 = 0
	AssaultRejectionReasonMasked         uint8 = 1
	AssaultRejectionReasonInvalid        uint8 = 2
	AssaultRejectionReasonServerRejected uint8 = 3
	AssaultRejectionReasonSafety         uint8 = 4
	AssaultRejectionReasonTimeout        uint8 = 5
	AssaultRejectionReasonPolicyError    uint8 = 6
	AssaultRejectionReasonUnknown        uint8 = 255
)

// Short aliases keep the enum readable at protocol call sites without
// exposing a second set of numeric values.
const (
	TeacherStatusNone             = AssaultTeacherStatusNone
	TeacherStatusAction           = AssaultTeacherStatusAction
	TeacherStatusWait             = AssaultTeacherStatusWait
	TeacherStatusHold             = AssaultTeacherStatusHold
	TeacherStatusCancel           = AssaultTeacherStatusCancel
	TeacherStatusUnavailable      = AssaultTeacherStatusUnavailable
	RejectionReasonNone           = AssaultRejectionReasonNone
	RejectionReasonMasked         = AssaultRejectionReasonMasked
	RejectionReasonInvalid        = AssaultRejectionReasonInvalid
	RejectionReasonServerRejected = AssaultRejectionReasonServerRejected
	RejectionReasonSafety         = AssaultRejectionReasonSafety
	RejectionReasonTimeout        = AssaultRejectionReasonTimeout
	RejectionReasonPolicyError    = AssaultRejectionReasonPolicyError
	RejectionReasonUnknown        = AssaultRejectionReasonUnknown
)

const assaultSchemaV1 = "tanat.assault.v1|hero=10x32|entities=10x96x16|global=10x32|action=kind,target,direction,distance|mask=8+96+4x96"
const assaultSchemaV2 = "tanat.assault.v2|hero=10x32|abilities=10x4x40|entities=10x96x16|global=10x32|action=kind,target,direction,distance|conditioned=kind|mask=8+96+4x96"
const assaultSchemaV3 = "tanat.assault.v3|hero=10x32|abilities=10x4x40|entities=10x96x16|global=10x32|global_lane=active,onehot3,wrong|action=kind,target,direction,distance|conditioned=kind|mask=8+96+4x96"
const assaultSchemaV4 = "tanat.assault.v4|hero=10x32|abilities=10x4x40|entities=10x96x16|global=10x32|global_lane=active,onehot3,wrong|action=kind,target,offset81,anchor15|conditioned=kind|anchors=local,bases,lanes3x4|mask=8+96+4x96"
const assaultRewardSchemaV2 = "tanat.assault.reward.v2|xp=.002|money_gain=.04|money_spend=.004|hp=2|mana=.75|death=-1|hero_kill=-.6|creep_last_hit=-.16|structure=two_thirds_damage+one_third_destroy|win=5|zero_sum=1|team_spirit=.2"
const assaultRewardSchemaV3 = assaultRewardSchemaV2 + "|wrong_lane=-.15_per_second|lane_assignment=2-1-2|lane_until=360-600|lane_corridor=30"
const assaultRewardSchemaV4 = assaultRewardSchemaV3 + "|shaping_time_weight=.6^(elapsed/600s)|draw_timeout=-2_post_zero_sum"
const assaultRewardSchemaV5 = assaultRewardSchemaV4 + "|tanat_creep_last_hit_bonus=.24|standard_wave_last_hit_mean=.4"

var AssaultSchemaHashV1 = sha256.Sum256([]byte(assaultSchemaV1))
var AssaultSchemaHashV2 = sha256.Sum256([]byte(assaultSchemaV2))
var AssaultSchemaHashV3 = sha256.Sum256([]byte(assaultSchemaV3))
var AssaultSchemaHashV4 = sha256.Sum256([]byte(assaultSchemaV4))
var AssaultRewardHashV2 = sha256.Sum256([]byte(assaultRewardSchemaV2))
var AssaultRewardHashV3 = sha256.Sum256([]byte(assaultRewardSchemaV3))
var AssaultRewardHashV4 = sha256.Sum256([]byte(assaultRewardSchemaV4))
var AssaultRewardHashV5 = sha256.Sum256([]byte(assaultRewardSchemaV5))

const (
	assaultWrongLanePenaltyPerSecond = 0.15
	assaultLaneCorridor              = 30.0
	assaultLaneMinSeconds            = 360
	assaultLaneMaxSeconds            = 600
	assaultAttackRetargetInterval    = 1.0

	// Preserve the published OpenAI Five shaping weights as the base contract.
	// Tanat's bronze bounties are much smaller, so a separate additive calibration
	// restores the intended approximately +0.4 total reward per ordinary last hit:
	// 3 * (72*.002 + 4*.04 -.16 + .24) +
	//     (84*.002 + 5*.04 -.16 + .24) == 4*.4.
	assaultOpenAIFiveXPGainReward        = 0.002
	assaultTanatMoneyGainReward          = 0.04
	assaultTanatMoneySpendReward         = 0.004
	assaultOpenAIFiveHeroKillCorrection  = -0.6
	assaultOpenAIFiveCreepKillCorrection = -0.16
	assaultTanatCreepLastHitBonus        = 0.24
)

type AssaultControllerV1 uint8

const (
	AssaultControllerExternal AssaultControllerV1 = iota
	AssaultControllerAI20
	AssaultControllerAI30
	// AssaultControllerAI40 is driven by the external action batch just like
	// AssaultControllerExternal, but explicitly marks a hero as a member of the
	// shared AI-40 self-play policy.  Keeping the identity in the reset contract
	// prevents a training run from silently falling back to a scripted opponent.
	AssaultControllerAI40
)

type AssaultActionKindV1 uint8

const (
	AssaultActionWait AssaultActionKindV1 = iota
	AssaultActionMove
	AssaultActionAttack
	AssaultActionSkill1
	AssaultActionSkill2
	AssaultActionSkill3
	AssaultActionSkill4
	AssaultActionTeleport // reserved; masked in V1
)

// HeroActionV1 is deliberately factorized for policy heads. Target is an
// observation entity slot, never an internal server object id. Protocols v4-v7
// use Direction as a 16-way compass and Distance as 4/8/12u. Navigation
// protocols v8-v9 reuse the same compact wire fields for offset81/anchor15.
type HeroActionV1 struct {
	Kind      AssaultActionKindV1
	Target    uint16
	Direction uint8
	Distance  uint8
}

type ActionMaskV1 struct {
	Kinds       [AssaultActionKinds]uint8
	Targets     [AssaultMaxEntities]uint8
	SkillTarget [4][AssaultMaxEntities]uint8
}

type AssaultObservationV1 struct {
	Hero [AssaultHeroFeatureSize]float32
	// Abilities is ignored by protocol v4/AI-40 and serialized by protocol v5/AI-41.
	// Keeping it on the shared internal observation lets both policies use the same
	// authoritative simulation without changing the established V1 wire layout.
	Abilities  [4][AssaultAbilityFeatures]float32
	Entities   [AssaultMaxEntities][AssaultEntityFeatures]float32
	Global     [AssaultGlobalFeatures]float32
	EntityMask [AssaultMaxEntities]uint8
	ActionMask ActionMaskV1
}

type StepResultV1 struct {
	SchemaHash   [32]byte
	RewardHash   [32]byte
	Step         uint32
	Elapsed      float32
	Done         bool
	Winner       int32
	Invalid      [AssaultHeroCount]uint8
	Reward       [AssaultHeroCount]float32
	Observations [AssaultHeroCount]AssaultObservationV1
	// TeacherActions is populated only by the training-only AI-30 imitation
	// protocol.  TeacherValid deliberately excludes idle/wait frames: copying
	// them would teach a policy to be passive instead of reproducing decisions.
	TeacherActions [AssaultHeroCount]HeroActionV1
	TeacherValid   [AssaultHeroCount]uint8
	// AI-42 v13 keeps the v12 teacher bytes at the same offsets but gives them
	// authoritative names/status semantics, then appends execution telemetry.
	TeacherIntent   [AssaultHeroCount]HeroActionV1
	TeacherStatus   [AssaultHeroCount]uint8
	ExecutedActions [AssaultHeroCount]HeroActionV1
	ExecutedValid   [AssaultHeroCount]uint8
	RejectionReason [AssaultHeroCount]uint8
	// ActiveOrder is evaluation-only authoritative control state. It lets a
	// recurrent policy distinguish legal HOLD from a completed command.
	ActiveOrder [AssaultHeroCount]uint8
}

type AssaultResetV1 struct {
	Seed        int64
	Roster      [AssaultHeroCount]int32
	Controllers [AssaultHeroCount]AssaultControllerV1
	MaxSteps    uint32
}

type assaultRewardSnapshot struct {
	xp, hp, mana, objective float64
	money                   int32
	frags, creepKills       int32
	dead                    bool
}

// assaultTeacherState stores the last action exactly as the actor observed it.
// HOLD is valid only while that complete wire action remains equal; an internal
// object id or destination is not a safe lineage key because observation slots
// and relative navigation bins may change independently of it.
type assaultTeacherState struct {
	action HeroActionV1
	active bool
}

// assaultTeacherProjection separates authoritative teacher state from whether
// that state can be faithfully projected through the current actor observation.
// active=true with representable=false is the only path to UNAVAILABLE for an
// AI-30 slot.
type assaultTeacherProjection struct {
	action        HeroActionV1
	active        bool
	representable bool
}

// assaultTeacherFrame binds the observation seen immediately before one
// scripted AI-30 decision to the transition produced by that decision.  It is
// intentionally step-local: carrying it across ticks would recreate the
// one-tick label shift this frame prevents.
type assaultTeacherFrame struct {
	present      [AssaultHeroCount]bool
	observations [AssaultHeroCount]AssaultObservationV1
	projections  [AssaultHeroCount]assaultTeacherProjection
}

// AssaultEnv is a synchronous, network-client-free wrapper around the same
// authoritative DOTA simulation used by live matches. A Reset owns an isolated
// in-memory session store, so training can never mutate account economy.
type AssaultEnv struct {
	server           *Server
	inst             *huntInstance
	clock            *manualBattleClock
	heroes           [AssaultHeroCount]*conn
	brains           [AssaultHeroCount]*botBrain
	controllers      [AssaultHeroCount]AssaultControllerV1
	orderedMembers   []*conn
	entityIDs        [AssaultHeroCount][AssaultMaxEntities]int32
	actionMasks      [AssaultHeroCount]ActionMaskV1
	previous         [AssaultHeroCount]assaultRewardSnapshot
	entityScratch    [AssaultHeroCount][]assaultEntity
	entityTopScratch [AssaultHeroCount][]assaultEntity
	// Observations for all heroes are produced from one immutable world tick.
	// Cache each team's sources and visibility answers for that tick: fog is
	// observer-independent in a headless match, so recomputing it per hero only
	// duplicated work and allocated short-lived source slices.
	assaultVisionAt       float64
	assaultVisionOK       bool
	assaultVisionReady    [2]bool
	assaultVisionSources  [2][]dotaVisionSource
	assaultVisibleMembers [2]map[int32]uint8
	assaultVisibleMobs    [2]map[int32]uint8
	// AI-41-v2 training-only lane curriculum. The live server and the established
	// AI-40/AI-41-v1 protocols leave it disabled, preserving their reward contract.
	wrongLaneContract   bool
	wrongLaneCurriculum bool
	navigationContract  bool
	strategicReward     bool
	teacherActions      bool
	teacherState        [AssaultHeroCount]assaultTeacherState
	laneAssignment      [AssaultHeroCount]uint8
	laneAssignmentUntil float64
	laneRewardStep      uint32
	terminalRewarded    bool
	step                uint32
	maxSteps            uint32
	closed              bool
}

func NewAssaultEnv() *AssaultEnv { return &AssaultEnv{} }

// EnableWrongLaneCurriculum selects the AI-41-v2 reward/observation contract.
// Call it before Reset. It deliberately survives Reset's initial Close so one
// protocol-bound subprocess can reset many matches without losing its mode.
func (e *AssaultEnv) EnableWrongLaneCurriculum(enabled bool) {
	e.wrongLaneContract = enabled
	e.wrongLaneCurriculum = enabled
}

// ConfigureWrongLaneCurriculum retains the v3 schema/reward contract while
// optionally disabling randomized assignments for evaluation matches.
func (e *AssaultEnv) ConfigureWrongLaneCurriculum(contract, enabled bool) {
	e.wrongLaneContract = contract
	e.wrongLaneCurriculum = contract && enabled
}

// ConfigureNavigationActions switches Direction/Distance to the AI-41-v3
// contract: a 9x9 local offset and a semantic global navigation anchor.
// Protocols v4-v7 retain the legacy 16-way direction and 4/8/12 distance.
func (e *AssaultEnv) ConfigureNavigationActions(enabled bool) {
	e.navigationContract = enabled
}

// ConfigureStrategicReward enables time-weighted shaping and a post-zero-sum
// draw/timeout signal. It deliberately survives Reset like the wire contracts.
func (e *AssaultEnv) ConfigureStrategicReward(enabled bool) {
	e.strategicReward = enabled
}

// ConfigureTeacherActions exposes AI-30's resolved, externally representable
// movement and attack intents.  It is an offline imitation-learning aid and
// does not affect live matches or any established protocol.
func (e *AssaultEnv) ConfigureTeacherActions(enabled bool) {
	e.teacherActions = enabled
	if !enabled {
		e.teacherState = [AssaultHeroCount]assaultTeacherState{}
	}
}

func (e *AssaultEnv) Reset(cfg AssaultResetV1) (StepResultV1, error) {
	e.Close()
	if cfg.MaxSteps == 0 {
		cfg.MaxSteps = 30 * 60 * 5
	}
	for i, controller := range cfg.Controllers {
		if controller > AssaultControllerAI40 {
			return StepResultV1{}, fmt.Errorf("assault: controller slot %d has unsupported value %d", i, controller)
		}
	}
	dm, ok := gamedata.DotaMapByID(101)
	if !ok {
		maps := gamedata.DotaMaps()
		if len(maps) == 0 {
			return StepResultV1{}, errors.New("assault: no DOTA map configured")
		}
		dm = maps[0]
	}
	clock := newManualBattleClock()
	store := session.NewStore()
	s := New(store)
	s.clock = clock
	s.rng = rand.New(rand.NewSource(cfg.Seed))
	inst := newDotaInstance(s, 1, dm.ID)
	inst.dota.telemetry = nil
	inst.dota.botAIVersionByTeam[dotaTeamHuman] = 20
	inst.dota.botAIVersionByTeam[dotaTeamElf] = 20
	e.server, e.inst, e.clock, e.closed = s, inst, clock, false

	roster := botAvatarRoster()
	if len(roster) < AssaultHeroCount {
		return StepResultV1{}, fmt.Errorf("assault: bot roster has %d avatars, need %d", len(roster), AssaultHeroCount)
	}
	for i := 0; i < AssaultHeroCount; i++ {
		// The live server alternates picks between teams. Keep the headless
		// roster equivalent: handing ids 0..4 to one side and 5..9 to the other
		// accidentally produced a carry-heavy team versus a support-heavy team.
		rosterIndex := i
		if i < AssaultHeroCount/2 {
			rosterIndex = i * 2
		} else {
			rosterIndex = (i-AssaultHeroCount/2)*2 + 1
		}
		avatar := roster[rosterIndex]
		if cfg.Roster[i] != 0 {
			var found bool
			avatar, found = gamedata.AvatarByID(cfg.Roster[i])
			if !found {
				e.Close()
				return StepResultV1{}, fmt.Errorf("assault: roster slot %d has unknown avatar %d", i, cfg.Roster[i])
			}
		}
		side := gamedata.DotaSideHuman
		if i >= AssaultHeroCount/2 {
			side = gamedata.DotaSideElf
		}
		c := s.newHeadlessBotConnLocked(inst, i, side, avatar)
		e.heroes[i] = c
		e.brains[i] = inst.bots[c.objID]
		e.controllers[i] = cfg.Controllers[i]
		switch cfg.Controllers[i] {
		case AssaultControllerExternal:
			delete(inst.bots, c.objID)
		case AssaultControllerAI20:
			e.brains[i].aiVersion, e.brains[i].aiVersionSet = 20, true
		case AssaultControllerAI30:
			e.brains[i].aiVersion, e.brains[i].aiVersionSet = 30, true
		case AssaultControllerAI40:
			e.brains[i].aiVersion, e.brains[i].aiVersionSet = 40, true
			// Headless AI-40 actions arrive synchronously in Step.  Do not leave
			// the brain in inst.bots: that would also start the live ONNX sidecar
			// and give one hero two independent controllers.
			delete(inst.bots, c.objID)
		}
	}
	// The team orchestrator uses the strongest scripted controller present on
	// that side, while each hero retains its explicitly requested local profile.
	for i, controller := range cfg.Controllers {
		if controller == AssaultControllerAI30 {
			team := dotaTeamHuman
			if i >= AssaultHeroCount/2 {
				team = dotaTeamElf
			}
			inst.dota.botAIVersionByTeam[team] = 30
		} else if controller == AssaultControllerAI40 {
			team := dotaTeamHuman
			if i >= AssaultHeroCount/2 {
				team = dotaTeamElf
			}
			inst.dota.botAIVersionByTeam[team] = 40
		}
	}
	e.orderedMembers = assaultOrderedMembers(inst)
	e.step, e.maxSteps, e.closed, e.terminalRewarded = 0, cfg.MaxSteps, false, false
	e.teacherState = [AssaultHeroCount]assaultTeacherState{}
	e.assaultVisionOK = false
	e.laneAssignment = [AssaultHeroCount]uint8{}
	e.laneAssignmentUntil, e.laneRewardStep = 0, 0
	if e.wrongLaneCurriculum {
		// Every team gets a randomized 2-1-2 split. A dedicated RNG keeps the
		// assignment reproducible without perturbing combat randomness.
		laneRNG := rand.New(rand.NewSource(cfg.Seed ^ 0x41a41))
		for side := 0; side < 2; side++ {
			lanes := []uint8{0, 0, 1, 2, 2}
			laneRNG.Shuffle(len(lanes), func(i, j int) { lanes[i], lanes[j] = lanes[j], lanes[i] })
			copy(e.laneAssignment[side*5:(side+1)*5], lanes)
		}
		e.laneAssignmentUntil = assaultLaneMinSeconds +
			laneRNG.Float64()*(assaultLaneMaxSeconds-assaultLaneMinSeconds)
	}

	inst.mu.Lock()
	now := clock.Now()
	for i := range e.heroes {
		e.previous[i] = e.rewardSnapshotLocked(i, now)
	}
	result := e.resultLocked(nil, now)
	inst.mu.Unlock()
	return result, nil
}

func (e *AssaultEnv) Step(actions [AssaultHeroCount]HeroActionV1) (StepResultV1, error) {
	return e.executeStep(actions, nil)
}

// StepControlled executes the explicit AI-42 recurrent control contract.
// Legacy callers remain on Step and therefore retain byte-for-byte behavior.
func (e *AssaultEnv) StepControlled(
	actions [AssaultHeroCount]HeroActionV1,
	controls [AssaultHeroCount]AssaultControlV1,
) (StepResultV1, error) {
	return e.executeStep(actions, &controls)
}

func (e *AssaultEnv) executeStep(
	actions [AssaultHeroCount]HeroActionV1,
	controls *[AssaultHeroCount]AssaultControlV1,
) (StepResultV1, error) {
	if e.closed || e.server == nil || e.inst == nil || e.clock == nil {
		return StepResultV1{}, errors.New("assault: Step called before Reset or after Close")
	}
	if e.inst.dota.ended || e.step >= e.maxSteps {
		e.inst.mu.Lock()
		result := e.resultLocked(nil, e.clock.Now())
		e.inst.mu.Unlock()
		return result, nil
	}

	var invalid [AssaultHeroCount]uint8
	var executed [AssaultHeroCount]HeroActionV1
	var executedValid [AssaultHeroCount]uint8
	var rejectionReason [AssaultHeroCount]uint8
	e.inst.mu.Lock()
	for i := 0; i < AssaultHeroCount; i++ {
		if !assaultControllerUsesActions(e.controllers[i]) {
			rejectionReason[i] = AssaultRejectionReasonUnknown
			continue
		}
		if controls != nil && controls[i] != AssaultControlIssue {
			control := controls[i]
			switch control {
			case AssaultControlHold:
				// HOLD submits no new world action and preserves the order.
				rejectionReason[i] = AssaultRejectionReasonUnknown
			case AssaultControlIdle:
				e.cancelExternalActionLocked(i)
				rejectionReason[i] = AssaultRejectionReasonUnknown
			default:
				invalid[i] = 1
				rejectionReason[i] = AssaultRejectionReasonInvalid
			}
			continue
		}
		classification := e.classifyActionLocked(i, actions[i])
		accepted := e.applyActionLocked(i, actions[i])
		if !accepted {
			invalid[i] = 1
		}
		if accepted && classification == AssaultRejectionReasonNone {
			executed[i] = actions[i]
			executedValid[i] = 1
			rejectionReason[i] = AssaultRejectionReasonNone
		} else if classification != AssaultRejectionReasonNone {
			rejectionReason[i] = classification
		} else {
			rejectionReason[i] = AssaultRejectionReasonServerRejected
		}
	}
	e.inst.mu.Unlock()

	// Execute callbacks (movement arrival, swings, projectiles, corpses) outside
	// the instance lock; each callback acquires that lock through conn.lock.
	e.clock.Advance(AssaultTick)
	now := float64(e.server.battleTime())

	e.inst.mu.Lock()
	var teacherFrame assaultTeacherFrame
	haveTeacherFrame := false
	if !e.inst.dota.ended {
		e.server.botRebalanceLanesLocked(e.inst, now)
		e.server.botPlanTeamsLocked(e.inst, now)
		var rep *conn
		for _, c := range e.orderedMembers {
			if c.huntState == nil || c.huntState.closed {
				continue
			}
			e.server.memberTickLocked(c, now)
			rep = c
			if brain := e.inst.bots[c.objID]; brain != nil && botAIVersionForBrain(brain) != 40 {
				slot := e.assaultHeroSlot(c)
				if e.teacherActions && slot >= 0 && e.controllers[slot] == AssaultControllerAI30 {
					// memberTickLocked has applied authoritative lifecycle changes,
					// while botTickLocked has not yet chosen this hero's next order.
					// Capture exactly that decision state, then bind the resulting
					// teacher transition to it in the same wire row.
					// Earlier heroes in the deterministic member loop may have
					// changed deaths, summons, reveal or vision at the same clock
					// timestamp, so the old per-now visibility cache is no longer
					// an immutable-world snapshot for this teacher-only path.
					e.assaultVisionOK = false
					observation := e.observationLocked(slot, now)
					e.server.botTickLocked(brain, now)
					teacherFrame.present[slot] = true
					teacherFrame.observations[slot] = observation
					teacherFrame.projections[slot] = e.assaultTeacherProjectionLocked(slot, &observation, false, now)
					haveTeacherFrame = true
				} else {
					e.server.botTickLocked(brain, now)
				}
			}
		}
		e.server.botAI40BatchTickLocked(e.inst, now)
		// External policies still use the exact scripted purchase/skill rules.
		for i, brain := range e.brains {
			if assaultControllerUsesActions(e.controllers[i]) && brain != nil {
				e.server.botSpendSkillPointLocked(brain)
				e.server.botBuyItemsLocked(brain, now)
			}
		}
		if rep != nil {
			e.server.dotaTickLocked(rep, now)
		}
	}
	e.step++
	var teacherOverride *assaultTeacherFrame
	if haveTeacherFrame {
		teacherOverride = &teacherFrame
	}
	result := e.resultLockedWithTeacher(&invalid, now, teacherOverride)
	result.ExecutedActions = executed
	result.ExecutedValid = executedValid
	result.RejectionReason = rejectionReason
	e.inst.mu.Unlock()
	return result, nil
}

// cancelExternalActionLocked is the authoritative counterpart of AI-42 IDLE.
// It stops every command family an external policy can start and is harmless
// when the hero already has no active command.
// Caller holds the instance lock.
func (e *AssaultEnv) cancelExternalActionLocked(slot int) {
	if slot < 0 || slot >= AssaultHeroCount {
		return
	}
	c := e.heroes[slot]
	if c == nil || c.huntState == nil {
		return
	}
	hs := c.huntState
	e.server.breakInterruptibleChannelsLocked(c)
	if hs.attackTarget != 0 || (hs.attackActionActive && hs.pvpTarget == 0) {
		e.server.stopAttackLocked(c, true)
	}
	if hs.pvpTarget != 0 || (hs.attackActionActive && hs.attackTarget == 0) {
		e.server.stopPvpAttackLocked(c, true)
	}
	e.server.cancelOrderLocked(c)
	c.stopMovementLocked(e.server, float64(e.server.battleTime()))
}

func assaultControllerUsesActions(controller AssaultControllerV1) bool {
	return controller == AssaultControllerExternal || controller == AssaultControllerAI40
}

func (e *AssaultEnv) Close() {
	if e.closed {
		return
	}
	e.closed = true
	if e.inst != nil {
		e.inst.mu.Lock()
		e.inst.closed = true
		var ai40Runtime ai40PolicyRuntime
		var telemetry *telemetryRecorder
		if e.inst.dota != nil {
			ai40Runtime = e.inst.dota.ai40Runtime
			e.inst.dota.ai40Runtime = nil
			telemetry = e.inst.dota.telemetry
			e.inst.dota.telemetry = nil
		}
		for _, c := range e.heroes {
			if c != nil && c.Conn != nil {
				_ = c.Conn.Close()
			}
		}
		e.inst.mu.Unlock()
		if ai40Runtime != nil {
			_ = ai40Runtime.Close()
		}
		if telemetry != nil {
			telemetry.close()
		}
	}
	e.server, e.inst, e.clock = nil, nil, nil
	e.orderedMembers = nil
	e.teacherState = [AssaultHeroCount]assaultTeacherState{}
}

func assaultOrderedMembers(inst *huntInstance) []*conn {
	ids := make([]int32, 0, len(inst.members))
	for id := range inst.members {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	out := make([]*conn, 0, len(ids))
	for _, id := range ids {
		out = append(out, inst.members[id])
	}
	return out
}

func (e *AssaultEnv) applyActionLocked(slot int, action HeroActionV1) bool {
	c := e.heroes[slot]
	return e.applyConnActionWithMaskLocked(c, &e.entityIDs[slot], e.actionMasks[slot], action)
}

// classifyActionLocked is intentionally side-effect free. It supplies the
// v13 reason without changing the established applyActionLocked semantics.
// A successful safety no-op is reported as safety rather than as an executed
// action, while the legacy Invalid/reward bit remains driven only by apply's
// bool result.
func (e *AssaultEnv) classifyActionLocked(slot int, action HeroActionV1) uint8 {
	if slot < 0 || slot >= AssaultHeroCount || int(action.Kind) >= AssaultActionKinds {
		return AssaultRejectionReasonInvalid
	}
	c := e.heroes[slot]
	if c == nil || c.huntState == nil {
		return AssaultRejectionReasonServerRejected
	}
	if e.navigationContract && !assaultNavigationActionFieldsValid(action) {
		return AssaultRejectionReasonInvalid
	}
	mask := e.actionMasks[slot]
	if mask.Kinds[action.Kind] == 0 && action.Kind != AssaultActionWait {
		return AssaultRejectionReasonMasked
	}
	switch action.Kind {
	case AssaultActionWait:
		return AssaultRejectionReasonNone
	case AssaultActionMove:
		if e.navigationContract {
			if action.Distance != 0 {
				if _, _, ok := e.assaultNavigationAnchorLocked(c, int(action.Distance)); !ok {
					return AssaultRejectionReasonServerRejected
				}
			}
		}
		return AssaultRejectionReasonNone
	case AssaultActionAttack:
		id, ok := assaultActionTargetID(&e.entityIDs[slot], action.Target)
		if !ok {
			return AssaultRejectionReasonInvalid
		}
		if mask.Targets[action.Target] == 0 {
			return AssaultRejectionReasonMasked
		}
		if !assaultPolicyAttackTargetAllowed(c, id, float64(e.server.battleTime())) {
			return AssaultRejectionReasonSafety
		}
		return AssaultRejectionReasonNone
	case AssaultActionSkill1, AssaultActionSkill2, AssaultActionSkill3, AssaultActionSkill4:
		skill := int(action.Kind-AssaultActionSkill1) + 1
		def := c.huntState.skillDef(skill)
		if skillHasTargetFlag(def, "ENEMY") || skillHasTargetFlag(def, "FRIEND") {
			if int(action.Target) >= AssaultMaxEntities {
				return AssaultRejectionReasonInvalid
			}
			if mask.SkillTarget[skill-1][action.Target] == 0 {
				return AssaultRejectionReasonMasked
			}
			if _, ok := assaultActionTargetID(&e.entityIDs[slot], action.Target); !ok {
				return AssaultRejectionReasonInvalid
			}
		}
		return AssaultRejectionReasonNone
	default:
		return AssaultRejectionReasonInvalid
	}
}

// applyConnActionLocked is shared by headless external policies and the live
// AI-40 profile. Caller holds the instance lock.
func (e *AssaultEnv) applyConnActionLocked(c *conn, entityIDs *[AssaultMaxEntities]int32, action HeroActionV1) bool {
	if c == nil || c.huntState == nil || int(action.Kind) >= AssaultActionKinds {
		return false
	}
	now := float64(e.server.battleTime())
	mask := e.observationForConnLocked(c, entityIDs, now).ActionMask
	return e.applyConnActionWithMaskLocked(c, entityIDs, mask, action)
}

func (e *AssaultEnv) applyConnActionWithMaskLocked(c *conn, entityIDs *[AssaultMaxEntities]int32, mask ActionMaskV1, action HeroActionV1) bool {
	if c == nil || c.huntState == nil || int(action.Kind) >= AssaultActionKinds {
		return false
	}
	if e.navigationContract && !assaultNavigationActionFieldsValid(action) {
		return false
	}
	now := float64(e.server.battleTime())
	// The world cannot advance between the result returned by the previous Step
	// and this call: AssaultEnv owns a manual clock and runs all callbacks inside
	// Step. Reuse that result's mask instead of rebuilding the complete 96-entity
	// observation solely to validate one action.
	if mask.Kinds[action.Kind] == 0 {
		return action.Kind == AssaultActionWait
	}
	hs := c.huntState
	switch action.Kind {
	case AssaultActionWait:
		return true
	case AssaultActionMove:
		if e.navigationContract {
			x, y := c.posAtLocked(float32(now))
			if action.Distance == 0 {
				dx, dy := assaultLocalOffset(action.Direction)
				c.moveToLocked(e.server, x+dx, y+dy)
				return true
			}
			tx, ty, ok := e.assaultNavigationAnchorLocked(c, int(action.Distance))
			if !ok {
				return false
			}
			// A global route may span the complete map. Keep following it while
			// the policy repeats the same anchor instead of rebuilding A* at 5 Hz.
			if c.hasDest && math.Hypot(float64(c.destX-tx), float64(c.destY-ty)) <= 1 {
				return true
			}
			c.moveToLocked(e.server, tx, ty)
			return true
		}
		direction := int(action.Direction) % 16
		distance := []float64{4, 8, 12}[minInt(int(action.Distance), 2)]
		a := float64(direction) * 2 * math.Pi / 16
		x, y := c.posAtLocked(float32(now))
		c.moveToLocked(e.server, x+float32(math.Cos(a)*distance), y+float32(math.Sin(a)*distance))
		return true
	case AssaultActionAttack:
		id, ok := assaultActionTargetID(entityIDs, action.Target)
		if !ok || mask.Targets[action.Target] == 0 {
			return false
		}
		if !assaultPolicyAttackTargetAllowed(c, id, now) {
			return true
		}
		if mob := e.inst.mobs[id]; mob != nil && !mob.dead && mob.enemyOf(c.playerTeam()) {
			e.server.startAttackLocked(c, mob)
			return true
		}
		if victim := e.inst.members[id]; victim != nil && victim.huntState != nil && arenaEnemies(c, victim) {
			e.server.startPvpAttackLocked(c, victim)
			return true
		}
		return false
	case AssaultActionSkill1, AssaultActionSkill2, AssaultActionSkill3, AssaultActionSkill4:
		skill := int(action.Kind-AssaultActionSkill1) + 1
		def := hs.skillDef(skill)
		id := int32(0)
		if skillHasTargetFlag(def, "ENEMY") || skillHasTargetFlag(def, "FRIEND") {
			if int(action.Target) >= AssaultMaxEntities || mask.SkillTarget[skill-1][action.Target] == 0 {
				return false
			}
			var ok bool
			id, ok = assaultActionTargetID(entityIDs, action.Target)
			if !ok {
				return false
			}
		}
		x, y := c.posAtLocked(float32(now))
		if e.navigationContract {
			dx, dy := assaultLocalOffset(action.Direction)
			return e.server.startSkillOrderLocked(c, skill, id, x+dx, y+dy, true)
		}
		direction := int(action.Direction) % 16
		distance := []float64{4, 8, 12}[minInt(int(action.Distance), 2)]
		a := float64(direction) * 2 * math.Pi / 16
		return e.server.startSkillOrderLocked(c, skill, id,
			x+float32(math.Cos(a)*distance), y+float32(math.Sin(a)*distance), true)
	default:
		_ = hs
		return false
	}
}

func assaultNavigationActionFieldsValid(action HeroActionV1) bool {
	if action.Kind != AssaultActionMove &&
		(action.Kind < AssaultActionSkill1 || action.Kind > AssaultActionSkill4) {
		return true
	}
	if int(action.Direction) >= AssaultNavigationOffsets ||
		int(action.Distance) >= AssaultNavigationAnchors {
		return false
	}
	// v13 skill execution uses only the local offset. Global anchors are a
	// movement contract; accepting them for a skill would silently ignore the
	// anchor and train a false label.
	return action.Kind == AssaultActionMove || action.Distance == 0
}

// assaultLocalOffset maps the row-major 9x9 action grid to -12..+12 world
// units on each axis. Slot 40 is the centre; adjacent slots are three units
// apart. The diagonal reach is intentionally larger than the old 12u radius.
func assaultLocalOffset(raw uint8) (float32, float32) {
	index := int(raw) % 81
	return float32(index%9-4) * 3, float32(index/9-4) * 3
}

// assaultNavigationAnchorLocked resolves stable, team-relative semantics:
// 1 own base, 2 enemy base, then four progress points (20/40/60/80%) on each
// of north, centre and south lane. Lanes are authored Human->Elf, so Elf sees
// the progress order reversed. Anchor 0 is reserved for local-grid movement.
func (e *AssaultEnv) assaultNavigationAnchorLocked(c *conn, anchor int) (float32, float32, bool) {
	if c == nil || e.inst == nil || e.inst.dota == nil || anchor <= 0 ||
		anchor >= AssaultNavigationAnchors {
		return 0, 0, false
	}
	m := e.inst.dota.m
	team := c.playerTeam()
	if anchor <= 2 {
		own := m.SpawnHuman
		enemy := m.SpawnElf
		if team == dotaTeamElf {
			own, enemy = enemy, own
		}
		point := own
		if anchor == 2 {
			point = enemy
		}
		return float32(point.X), float32(point.Y), true
	}
	laneIndex := (anchor - 3) / 4
	progressIndex := (anchor - 3) % 4
	if laneIndex < 0 || laneIndex >= len(m.Lanes) {
		return 0, 0, false
	}
	fraction := float64(progressIndex+1) / 5
	if team == dotaTeamElf {
		fraction = 1 - fraction
	}
	point, ok := assaultLanePointAt(m.Lanes[laneIndex], fraction)
	return float32(point.X), float32(point.Y), ok
}

func assaultLanePointAt(lane []gamedata.Vec2, fraction float64) (gamedata.Vec2, bool) {
	if len(lane) < 2 {
		return gamedata.Vec2{}, false
	}
	fraction = max(0, min(1, fraction))
	total := 0.0
	for i := 0; i+1 < len(lane); i++ {
		total += math.Hypot(lane[i+1].X-lane[i].X, lane[i+1].Y-lane[i].Y)
	}
	if total <= 0 {
		return lane[0], true
	}
	want := total * fraction
	for i := 0; i+1 < len(lane); i++ {
		a, b := lane[i], lane[i+1]
		length := math.Hypot(b.X-a.X, b.Y-a.Y)
		if want <= length || i+2 == len(lane) {
			t := 0.0
			if length > 0 {
				t = min(1, want/length)
			}
			return gamedata.Vec2{X: a.X + (b.X-a.X)*t, Y: a.Y + (b.Y-a.Y)*t}, true
		}
		want -= length
	}
	return lane[len(lane)-1], true
}

// assaultPolicyAttackTargetAllowed prevents a sampled policy from cancelling
// its own approach/swing several times per second as target logits fluctuate.
// A dead/cancelled target clears attackTarget/pvpTarget and therefore never
// delays the next acquisition; only switching away from a still-live order is
// rate-limited for five 200ms simulation ticks.
func assaultPolicyAttackTargetAllowed(c *conn, target int32, now float64) bool {
	active := c.huntState.attackTarget
	if active == 0 {
		active = c.huntState.pvpTarget
	}
	if active != 0 && active != target &&
		now-c.policyAttackRetargetAt < assaultAttackRetargetInterval {
		return false
	}
	if active != target {
		c.policyAttackRetargetAt = now
	}
	return true
}

func (e *AssaultEnv) actionTargetID(slot int, target uint16) (int32, bool) {
	return assaultActionTargetID(&e.entityIDs[slot], target)
}

func assaultActionTargetID(entityIDs *[AssaultMaxEntities]int32, target uint16) (int32, bool) {
	if entityIDs == nil || int(target) >= AssaultMaxEntities {
		return 0, false
	}
	id := entityIDs[target]
	return id, id != 0
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

type assaultEntity struct {
	id       int32
	kind     int
	distance float64
	player   *conn
	mob      *mobState
}

func (e *AssaultEnv) assaultVisionSourcesLocked(team int32, now float64) []dotaVisionSource {
	if !e.assaultVisionOK || e.assaultVisionAt != now {
		e.assaultVisionAt = now
		e.assaultVisionOK = true
		for i := range e.assaultVisionReady {
			e.assaultVisionReady[i] = false
			clear(e.assaultVisibleMembers[i])
			clear(e.assaultVisibleMobs[i])
		}
	}
	index := assaultTeamIndex(team)
	if !e.assaultVisionReady[index] {
		e.assaultVisionSources[index] = dotaTeamVisionSourcesLocked(e.inst, team, now)
		e.assaultVisionReady[index] = true
	}
	return e.assaultVisionSources[index]
}

// Cached visibility is intentionally limited to headless Assault observations.
// Every caller uses already=false, so the answer is a pure function of team,
// entity and tick and does not inherit the live client's reveal hysteresis.
func (e *AssaultEnv) assaultVisibleEnemyMemberLocked(
	team int32, member *conn, now float64, sources []dotaVisionSource,
) bool {
	index := assaultTeamIndex(team)
	cache := e.assaultVisibleMembers[index]
	if cache == nil {
		cache = make(map[int32]uint8)
		e.assaultVisibleMembers[index] = cache
	}
	if value, ok := cache[member.objID]; ok {
		return value == 2
	}
	visible := dotaVisibleEnemyMemberLocked(e.inst, team, member, now, sources, false)
	if visible {
		cache[member.objID] = 2
	} else {
		cache[member.objID] = 1
	}
	return visible
}

func (e *AssaultEnv) assaultVisibleEnemyMobLocked(
	team int32, mob *mobState, sources []dotaVisionSource,
) bool {
	index := assaultTeamIndex(team)
	cache := e.assaultVisibleMobs[index]
	if cache == nil {
		cache = make(map[int32]uint8)
		e.assaultVisibleMobs[index] = cache
	}
	if value, ok := cache[mob.id]; ok {
		return value == 2
	}
	visible := botVisibleEnemyMobLocked(team, mob, sources)
	if visible {
		cache[mob.id] = 2
	} else {
		cache[mob.id] = 1
	}
	return visible
}

func (e *AssaultEnv) observationLocked(slot int, now float64) AssaultObservationV1 {
	c := e.heroes[slot]
	if c == nil {
		return AssaultObservationV1{}
	}
	sources := e.assaultVisionSourcesLocked(c.playerTeam(), now)
	return e.observationForConnWithScratchLocked(
		c, &e.entityIDs[slot], now, sources, &e.entityScratch[slot], &e.entityTopScratch[slot],
	)
}

func (e *AssaultEnv) observationForConnLocked(c *conn, entityIDs *[AssaultMaxEntities]int32, now float64) AssaultObservationV1 {
	sources := e.assaultVisionSourcesLocked(c.playerTeam(), now)
	var scratch []assaultEntity
	var topScratch []assaultEntity
	return e.observationForConnWithScratchLocked(c, entityIDs, now, sources, &scratch, &topScratch)
}

func (e *AssaultEnv) observationForConnWithScratchLocked(
	c *conn, entityIDs *[AssaultMaxEntities]int32, now float64,
	sources []dotaVisionSource, scratch, topScratch *[]assaultEntity,
) AssaultObservationV1 {
	var o AssaultObservationV1
	if c == nil || c.huntState == nil {
		return o
	}
	hs := c.huntState
	team := c.playerTeam()
	x, y := c.posAtLocked(float32(now))
	maxHP, maxMana := hs.maxHPLocked(now), hs.maxManaLocked(now)
	money, _, _ := e.server.Store.HeroMoney(c.selfPlayerID)
	o.Hero[0] = float32(hs.av.ID) / 100
	o.Hero[1] = float32(team)
	o.Hero[2], o.Hero[3] = x/160, y/160
	o.Hero[4] = safeFrac(hs.hp, maxHP)
	o.Hero[5] = safeFrac(hs.mana, maxMana)
	o.Hero[6] = float32(hs.level) / 20
	o.Hero[7] = float32(hs.xp) / 10000
	o.Hero[8] = float32(money) / 100000
	o.Hero[9] = boolFloat(hs.deadUntil > now)
	o.Hero[10] = boolFloat(hs.st.stunned(now))
	o.Hero[11] = boolFloat(hs.st.silenced(now))
	o.Hero[12] = boolFloat(hs.st.rooted(now))
	o.Hero[13] = boolFloat(now < hs.invisibleUntil)
	o.Hero[14] = boolFloat(hs.attackTarget != 0 || hs.pvpTarget != 0)
	for i := 0; i < 4; i++ {
		o.Hero[15+i] = float32(hs.skillLevel[i]) / 5
		remaining := math.Max(0, hs.cooldownUntil[i]-now)
		o.Hero[19+i] = float32(remaining / 60)
		o.Hero[23+i] = boolFloat(hs.toggleOn[i])
	}
	o.Hero[27] = float32(hs.points) / 10
	o.Hero[28] = float32(c.vx) / 10
	o.Hero[29] = float32(c.vy) / 10
	o.Hero[30] = float32(now) / 3600
	for skill := 0; skill < 4; skill++ {
		o.Abilities[skill] = assaultAbilityFeatures(hs, skill, now)
	}

	entities := (*scratch)[:0]
	if cap(entities) < len(e.inst.members)+len(e.inst.mobs) {
		entities = make([]assaultEntity, 0, len(e.inst.members)+len(e.inst.mobs))
	}
	// orderedMembers is fixed for the headless match.  Iterating it avoids ten
	// randomized map walks per result; the later distance/id ordering remains
	// exactly the same observation contract.
	for _, other := range e.orderedMembers {
		if other.huntState == nil {
			continue
		}
		visible := other.playerTeam() == team || e.assaultVisibleEnemyMemberLocked(team, other, now, sources)
		if !visible {
			continue
		}
		ox, oy := other.posAtLocked(float32(now))
		entities = append(entities, assaultEntity{id: other.objID, kind: 1,
			distance: math.Hypot(float64(ox-x), float64(oy-y)), player: other})
	}
	// dotaTickLocked already builds a deterministic complete mob list for this
	// exact world tick. Reuse it instead of rebuilding it from the map once for
	// every hero. Reset has no tick list yet, so retain the map fallback there.
	mobs := e.inst.dota.tickMobs
	if !e.inst.dota.tickMobsOK || e.inst.dota.tickMobsAt != now {
		mobs = nil
	}
	if mobs == nil {
		for _, mob := range e.inst.mobs {
			if mob.dead || (mob.teamVal() != team && !e.assaultVisibleEnemyMobLocked(team, mob, sources)) {
				continue
			}
			kind := 2
			if mob.structure {
				kind = 3
			}
			entities = append(entities, assaultEntity{id: mob.id, kind: kind,
				distance: math.Hypot(float64(mob.x-x), float64(mob.y-y)), mob: mob})
		}
	} else {
		for _, mob := range mobs {
			if mob.dead || (mob.teamVal() != team && !e.assaultVisibleEnemyMobLocked(team, mob, sources)) {
				continue
			}
			kind := 2
			if mob.structure {
				kind = 3
			}
			entities = append(entities, assaultEntity{id: mob.id, kind: kind,
				distance: math.Hypot(float64(mob.x-x), float64(mob.y-y)), mob: mob})
		}
	}
	*scratch = entities
	entities = assaultTopEntities(entities, topScratch)
	for i := range entityIDs {
		entityIDs[i] = 0
	}
	for i := 0; i < len(entities) && i < AssaultMaxEntities; i++ {
		ent := entities[i]
		entityIDs[i] = ent.id
		o.EntityMask[i] = 1
		f := &o.Entities[i]
		f[0] = float32(ent.kind) / 3
		f[1] = float32(ent.id) / 1000000
		f[6] = float32(ent.distance) / 200
		if ent.player != nil {
			other := ent.player
			oh := other.huntState
			ox, oy := other.posAtLocked(float32(now))
			f[2] = boolFloat(other.playerTeam() == team)
			f[3], f[4] = (ox-x)/200, (oy-y)/200
			f[5] = safeFrac(oh.hp, oh.maxHPLocked(now))
			f[7] = safeFrac(oh.mana, oh.maxManaLocked(now))
			f[8] = float32(oh.level) / 20
			f[9] = boolFloat(oh.deadUntil <= now)
			f[15] = float32(oh.av.ID) / 100
		} else {
			mob := ent.mob
			f[2] = boolFloat(mob.teamVal() == team)
			f[3], f[4] = (mob.x-x)/200, (mob.y-y)/200
			f[5] = safeFrac(mob.hp, mob.maxHealth())
			f[9] = 1
			f[10] = boolFloat(mob.structure)
			f[11] = boolFloat(mob.altar)
			f[12] = boolFloat(mob.structure && mob.dotaRole == gamedata.DotaGun)
			f[13] = boolFloat(mob.structure && mob.dotaRole == gamedata.DotaCreepTower)
			f[14] = boolFloat(mob.siege)
		}
	}

	o.ActionMask.Kinds[AssaultActionWait] = 1
	canAct := hs.deadUntil <= now && !hs.st.stunned(now)
	if canAct && !hs.st.rooted(now) {
		o.ActionMask.Kinds[AssaultActionMove] = 1
	}
	for i := 0; i < AssaultMaxEntities && o.EntityMask[i] != 0; i++ {
		id := entityIDs[i]
		enemy := false
		if mob := e.inst.mobs[id]; mob != nil {
			enemy = mob.enemyOf(team) && !(mob.altar && !e.inst.dota.altarVulnerableLocked(mob))
		} else if member := e.inst.members[id]; member != nil {
			enemy = member.playerTeam() != team && member.huntState.deadUntil <= now
		}
		if enemy {
			o.ActionMask.Targets[i] = 1
		}
	}
	if canAct {
		for i := 0; i < AssaultMaxEntities; i++ {
			if o.ActionMask.Targets[i] != 0 {
				o.ActionMask.Kinds[AssaultActionAttack] = 1
				break
			}
		}
		if !hs.st.silenced(now) {
			for skill := 0; skill < 4; skill++ {
				level := int(hs.skillLevel[skill])
				def := hs.skillDef(skill + 1)
				if level < 1 || def.Type == "PASSIVE" || now < hs.cooldownUntil[skill] ||
					hs.mana < skillManaCost(float64(def.ManaCost[level-1])) {
					continue
				}
				unitTarget := skillHasTargetFlag(def, "ENEMY") || skillHasTargetFlag(def, "FRIEND")
				if !unitTarget {
					// The target head still needs one legal category for self/point skills;
					// applyActionLocked deliberately converts it to object id 0.
					o.ActionMask.SkillTarget[skill][0] = 1
					o.ActionMask.Kinds[int(AssaultActionSkill1)+skill] = 1
					continue
				}
				for i := 0; i < AssaultMaxEntities && o.EntityMask[i] != 0; i++ {
					id := entityIDs[i]
					valid := false
					if mob := e.inst.mobs[id]; mob != nil {
						valid = (mob.enemyOf(team) && skillHasTargetFlag(def, "ENEMY")) ||
							(!mob.enemyOf(team) && skillHasTargetFlag(def, "FRIEND"))
						if valid && mob.structure && skillHasTargetFlag(def, "NOT_BUILDING") {
							valid = false
						}
					} else if member := e.inst.members[id]; member != nil && member.huntState.deadUntil <= now {
						valid = (member.playerTeam() != team && skillHasTargetFlag(def, "ENEMY")) ||
							(member.playerTeam() == team && skillHasTargetFlag(def, "FRIEND"))
					}
					if valid {
						o.ActionMask.SkillTarget[skill][i] = 1
						o.ActionMask.Kinds[int(AssaultActionSkill1)+skill] = 1
					}
				}
			}
		}
	}

	o.Global[0] = float32(now) / 3600
	o.Global[1] = float32(team)
	o.Global[2] = boolFloat(e.inst.dota.ended)
	o.Global[3] = float32(e.inst.dota.winner)
	for _, member := range e.orderedMembers {
		if member.huntState == nil || member.huntState.deadUntil > now {
			continue
		}
		if member.playerTeam() == team {
			o.Global[4]++
		} else if e.assaultVisibleEnemyMemberLocked(team, member, now, sources) {
			o.Global[5]++
		}
	}
	if mobs == nil {
		mobs = make([]*mobState, 0, len(e.inst.mobs))
		for _, mob := range e.inst.mobs {
			mobs = append(mobs, mob)
		}
	}
	// Accumulate normalized structure health in float64 and round only once.
	// The reset/teacher path can source mobs from a Go map; repeated float32
	// additions made this observation depend on that map's randomized iteration
	// order even though the authoritative match state was identical.
	var structureHealth [2]float64
	for _, mob := range mobs {
		if mob.dead || !mob.structure {
			continue
		}
		idx := 0
		if mob.teamVal() != team {
			idx = 1
		}
		structureHealth[idx] += float64(safeFrac(mob.hp, mob.maxHealth()))
	}
	o.Global[6] = float32(structureHealth[0])
	o.Global[7] = float32(structureHealth[1])
	if slot := e.assaultHeroSlot(c); slot >= 0 && e.wrongLaneCurriculum {
		lane := int(e.laneAssignment[slot])
		active := now < e.laneAssignmentUntil
		o.Global[8] = boolFloat(active)
		o.Global[9+lane] = 1
		o.Global[12] = boolFloat(e.assaultWrongLaneLocked(slot, now))
	}
	return o
}

func (e *AssaultEnv) assaultHeroSlot(c *conn) int {
	for slot, hero := range e.heroes {
		if hero == c {
			return slot
		}
	}
	return -1
}

func assaultDistanceToSegment(x, y float64, a, b gamedata.Vec2) float64 {
	dx, dy := b.X-a.X, b.Y-a.Y
	t := 0.0
	if denom := dx*dx + dy*dy; denom > 0 {
		t = ((x-a.X)*dx + (y-a.Y)*dy) / denom
		t = max(0, min(1, t))
	}
	return math.Hypot(x-(a.X+t*dx), y-(a.Y+t*dy))
}

func (e *AssaultEnv) assaultWrongLaneLocked(slot int, now float64) bool {
	if !e.wrongLaneCurriculum || slot < 0 || slot >= len(e.heroes) ||
		now >= e.laneAssignmentUntil || e.inst == nil || e.inst.dota == nil {
		return false
	}
	if controller := e.controllers[slot]; controller == AssaultControllerAI20 ||
		controller == AssaultControllerAI30 {
		return false // scripted teachers are observations/opponents, not policy samples
	}
	c := e.heroes[slot]
	if c == nil || c.huntState == nil || c.huntState.deadUntil > now {
		return false
	}
	laneIndex := int(e.laneAssignment[slot])
	if laneIndex < 0 || laneIndex >= len(e.inst.dota.m.Lanes) {
		return false
	}
	lane := e.inst.dota.m.Lanes[laneIndex]
	if len(lane) < 2 {
		return false
	}
	x, y := c.posAtLocked(float32(now))
	best := math.Inf(1)
	for i := 0; i+1 < len(lane); i++ {
		best = min(best, assaultDistanceToSegment(float64(x), float64(y), lane[i], lane[i+1]))
	}
	return best > assaultLaneCorridor
}

// assaultAbilityFeatures exposes authored, hero-visible skill semantics to AI-41.
// Indices are stable protocol fields; additions require a new observation hash/version.
//
//	0 slot, 1 hero id, 2 level, 3 max rank, 4 learned, 5 castable,
//	6 remaining cooldown, 7 base cooldown, 8 mana cost, 9 affordable,
//
// 10 active, 11 toggle, 12 passive, 13 enemy target, 14 friend target,
// 15 self/point target, 16 excludes buildings, 17 cast range, 18 AoE radius,
// 19 AoE width, 20 damage, 21 heal, 22 stun, 23 root, 24 slow,
// 25 silence, 26 shield/immunity, 27 mobility, 28 stealth,
// 29 summon/trap/ward, 30 channel, 31 buff, 32 debuff,
// 33 attack modifier/proc, 34 mana effect, 35 vision/reveal,
// 36 max absolute magnitude, 37 max duration, 38 max effect radius, 39 toggle on.
func assaultAbilityFeatures(hs *huntState, skill int, now float64) [AssaultAbilityFeatures]float32 {
	var f [AssaultAbilityFeatures]float32
	if hs == nil || skill < 0 || skill >= len(hs.skillLevel) {
		return f
	}
	def := hs.skillDef(skill + 1)
	level := int(hs.skillLevel[skill])
	rank := max(level, 1)
	if rank > len(hs.assaultAbilityStatic[skill]) {
		rank = len(hs.assaultAbilityStatic[skill])
	}
	if !hs.assaultAbilityStaticReady[skill][rank-1] {
		static := &hs.assaultAbilityStatic[skill][rank-1]
		static[0] = float32(skill+1) / 4
		static[1] = float32(hs.av.ID) / 100
		static[3] = float32(max(len(def.ManaCost), len(def.Cooldown))) / 5
		static[7] = float32(assaultRankInt(def.Cooldown, rank)) / 60
		static[8] = float32(assaultRankInt(def.ManaCost, rank)) / 500
		static[10] = boolFloat(def.Type == "ACTIVE")
		static[11] = boolFloat(def.Type == "TOGGLE")
		static[12] = boolFloat(def.Type == "PASSIVE")
		static[13] = boolFloat(skillHasTargetFlag(def, "ENEMY"))
		static[14] = boolFloat(skillHasTargetFlag(def, "FRIEND"))
		static[15] = boolFloat(!skillHasTargetFlag(def, "ENEMY") && !skillHasTargetFlag(def, "FRIEND"))
		static[16] = boolFloat(skillHasTargetFlag(def, "NOT_BUILDING"))
		static[17] = float32(def.Distance) / 20
		static[18] = float32(def.AoERadius) / 20
		static[19] = float32(def.AoEWidth) / 20
		assaultAccumulateAbilityOps(static, def.Ops, rank)
		hs.assaultAbilityStaticReady[skill][rank-1] = true
	}
	f = hs.assaultAbilityStatic[skill][rank-1]
	f[2] = float32(level) / 5
	f[4] = boolFloat(level > 0)
	remaining := math.Max(0, hs.cooldownUntil[skill]-now)
	manaCost := assaultRankInt(def.ManaCost, rank)
	castable := level > 0 && def.Type != "PASSIVE" && remaining <= 0 &&
		hs.mana >= skillManaCost(float64(manaCost)) && !hs.st.silenced(now) &&
		hs.deadUntil <= now && !hs.st.stunned(now)
	f[5] = boolFloat(castable)
	f[6] = float32(remaining / 60)
	f[9] = boolFloat(hs.mana >= skillManaCost(float64(manaCost)))
	f[39] = boolFloat(hs.toggleOn[skill])
	return f
}

func assaultRankInt(values []int, level int) int {
	if len(values) == 0 {
		return 0
	}
	if level < 1 {
		level = 1
	}
	if level > len(values) {
		level = len(values)
	}
	return values[level-1]
}

func assaultAccumulateAbilityOps(f *[AssaultAbilityFeatures]float32, ops []gamedata.Op, level int) {
	for _, op := range ops {
		switch op.Kind {
		case gamedata.OpDamage, gamedata.OpDot, gamedata.OpExecute,
			gamedata.OpShieldExplode, gamedata.OpCastMark,
			gamedata.OpManaScaledDamage, gamedata.OpAttackDamage:
			f[20] = 1
		case gamedata.OpAttackCleave:
			f[20], f[33] = 1, 1
		case gamedata.OpHeal, gamedata.OpHot, gamedata.OpHealOnKill,
			gamedata.OpChainHeal, gamedata.OpDamageShare, gamedata.OpLifestealHit:
			f[21] = 1
		case gamedata.OpStun:
			f[22] = 1
		case gamedata.OpRoot:
			f[23] = 1
		case gamedata.OpSlow, gamedata.OpAttackSlow, gamedata.OpChill:
			f[24] = 1
		case gamedata.OpSilence, gamedata.OpSilenceAll:
			f[25] = 1
		case gamedata.OpShield, gamedata.OpImmune, gamedata.OpBlockHit:
			f[26] = 1
		case gamedata.OpDash, gamedata.OpBlink, gamedata.OpPull, gamedata.OpKnockback:
			f[27] = 1
		case gamedata.OpStealth:
			f[28] = 1
		case gamedata.OpTreeForm:
			f[28], f[31] = 1, 1
		case gamedata.OpSummon, gamedata.OpTrap, gamedata.OpGrove:
			f[29] = 1
		case gamedata.OpVisionWard:
			f[29], f[35] = 1, 1
		case gamedata.OpChannel:
			f[30] = 1
		case gamedata.OpBuffStat, gamedata.OpAura, gamedata.OpDeathLink,
			gamedata.OpZoneArmor, gamedata.OpMeleeForm:
			f[31] = 1
		case gamedata.OpProc, gamedata.OpSelfRecoil, gamedata.OpHitStack,
			gamedata.OpMoveChargeAttack, gamedata.OpConsecutiveHit,
			gamedata.OpAttackSpeedStreak, gamedata.OpEmpowerNextHit:
			f[33] = 1
		case gamedata.OpAttackManaBonus:
			f[33], f[34] = 1, 1
		case gamedata.OpManaBurnHit, gamedata.OpManaBurnArea, gamedata.OpManaRestore,
			gamedata.OpManaOnKill, gamedata.OpManaScaledAttackSpeed:
			f[34] = 1
		case gamedata.OpRevealTarget:
			f[35] = 1
		}
		if strings.Contains(op.Stat, "slow") ||
			(op.Kind == gamedata.OpBuffStat && op.Value.At(level) < 0) {
			f[32] = 1
		}
		magnitude := math.Max(math.Abs(op.Value.At(level)), math.Abs(op.Value2.At(level)))
		f[36] = max(f[36], float32(magnitude/1000))
		f[37] = max(f[37], float32(math.Abs(op.Dur.At(level))/30))
		f[38] = max(f[38], float32(math.Abs(op.Radius)/20))
		assaultAccumulateAbilityOps(f, op.Ops, level)
	}
}

func (e *AssaultEnv) rewardSnapshotLocked(slot int, now float64) assaultRewardSnapshot {
	potentials := e.assaultObjectivePotentialsLocked()
	return e.rewardSnapshotWithObjectiveLocked(slot, now, potentials)
}

func (e *AssaultEnv) rewardSnapshotWithObjectiveLocked(
	slot int, now float64, potentials [2]float64,
) assaultRewardSnapshot {
	c := e.heroes[slot]
	if c == nil || c.huntState == nil {
		return assaultRewardSnapshot{}
	}
	money, _, _ := e.server.Store.HeroMoney(c.selfPlayerID)
	return assaultRewardSnapshot{
		xp:         c.huntState.xp,
		hp:         float64(safeFrac(c.huntState.hp, c.huntState.maxHPLocked(now))),
		mana:       float64(safeFrac(c.huntState.mana, c.huntState.maxManaLocked(now))),
		objective:  potentials[assaultTeamIndex(c.playerTeam())],
		money:      money,
		frags:      c.huntState.frags,
		creepKills: c.huntState.assaultCreepKills,
		dead:       c.huntState.deadUntil > now,
	}
}

func assaultShapingTimeWeight(elapsedSeconds float64) float64 {
	return math.Pow(0.6, elapsedSeconds/600)
}

func (e *AssaultEnv) resultLocked(invalid *[AssaultHeroCount]uint8, now float64) StepResultV1 {
	return e.resultLockedWithTeacher(invalid, now, nil)
}

func (e *AssaultEnv) resultLockedWithTeacher(
	invalid *[AssaultHeroCount]uint8, now float64, teacherFrame *assaultTeacherFrame,
) StepResultV1 {
	rewardHash := AssaultRewardHashV2
	if e.strategicReward {
		rewardHash = AssaultRewardHashV5
	} else if e.wrongLaneContract {
		rewardHash = AssaultRewardHashV3
	}
	r := StepResultV1{SchemaHash: AssaultSchemaHashV1, RewardHash: rewardHash, Step: e.step, Elapsed: float32(now)}
	r.Done = e.inst.dota.ended || e.step >= e.maxSteps
	r.Winner = e.inst.dota.winner
	if invalid != nil {
		r.Invalid = *invalid
	}
	var raw [AssaultHeroCount]float64
	// Structure state is team-wide, not hero-specific. Calculate both hostile
	// objective potentials once instead of scanning the map for every reward.
	potentials := e.assaultObjectivePotentialsLocked()
	for i := 0; i < AssaultHeroCount; i++ {
		cur := e.rewardSnapshotWithObjectiveLocked(i, now, potentials)
		prev := e.previous[i]
		shaping := (cur.xp - prev.xp) * assaultOpenAIFiveXPGainReward
		if cur.money > prev.money {
			shaping += float64(cur.money-prev.money) * assaultTanatMoneyGainReward
		} else if cur.money < prev.money {
			shaping += float64(prev.money-cur.money) * assaultTanatMoneySpendReward
		}
		shaping += (cur.hp - prev.hp) * 2
		shaping += (cur.mana - prev.mana) * 0.75
		shaping += cur.objective - prev.objective
		shaping += float64(cur.frags-prev.frags) * assaultOpenAIFiveHeroKillCorrection
		creepLastHits := float64(cur.creepKills - prev.creepKills)
		shaping += creepLastHits * assaultOpenAIFiveCreepKillCorrection
		if e.strategicReward {
			shaping += creepLastHits * assaultTanatCreepLastHitBonus
		}
		if cur.dead && !prev.dead {
			shaping -= 1
		}
		if invalid != nil && invalid[i] != 0 {
			shaping -= 0.01
		}
		if e.step > 0 && e.step != e.laneRewardStep && e.assaultWrongLaneLocked(i, now) {
			shaping -= assaultWrongLanePenaltyPerSecond * AssaultTick.Seconds()
		}
		if e.strategicReward {
			shaping *= assaultShapingTimeWeight(now)
		}
		reward := shaping
		if r.Done && r.Winner != 0 && e.heroes[i] != nil && !e.terminalRewarded {
			if e.heroes[i].playerTeam() == r.Winner {
				reward += 5
			}
		}
		raw[i] = reward
		team := dotaTeamHuman
		if e.heroes[i] != nil {
			team = e.heroes[i].playerTeam()
		}
		// Scripted AI-20/AI-30 consumes its own server-side perception in
		// botTickLocked. It never reads the expensive external-policy protocol
		// observation, and its submitted action is ignored. Keep its wire record
		// zeroed while avoiding five visibility/entity sorts per AI-30 match.
		if teacherFrame != nil && teacherFrame.present[i] {
			r.Observations[i] = teacherFrame.observations[i]
			e.actionMasks[i] = r.Observations[i].ActionMask
		} else if assaultControllerUsesActions(e.controllers[i]) ||
			(e.teacherActions && e.controllers[i] == AssaultControllerAI30) {
			sources := e.assaultVisionSourcesLocked(team, now)
			r.Observations[i] = e.observationForConnWithScratchLocked(
				e.heroes[i], &e.entityIDs[i], now, sources, &e.entityScratch[i], &e.entityTopScratch[i],
			)
			e.actionMasks[i] = r.Observations[i].ActionMask
		} else {
			e.actionMasks[i] = ActionMaskV1{}
		}
		e.previous[i] = cur
	}
	if e.teacherActions {
		for i := range r.Observations {
			action, status := HeroActionV1{}, uint8(AssaultTeacherStatusUnavailable)
			if teacherFrame != nil && teacherFrame.present[i] {
				projection := teacherFrame.projections[i]
				if r.Done {
					// Terminal state wins over an order chosen earlier in the
					// same simulation tick and retires its lineage exactly once.
					projection = e.assaultTeacherProjectionLocked(i, &r.Observations[i], true, now)
				}
				action, status = e.assaultTeacherTransitionFromProjectionLocked(i, projection)
			} else {
				action, status = e.assaultTeacherTransitionLocked(i, &r.Observations[i], r.Done, now)
			}
			r.TeacherStatus[i] = status
			if status == AssaultTeacherStatusAction {
				r.TeacherActions[i] = action
				r.TeacherValid[i] = 1
				r.TeacherIntent[i] = action
			}
		}
	}
	// OpenAI-Five-style zero-sum correction removes incentives that merely inflate
	// both teams' shaped score. Team spirit then shares 20% of the corrected team
	// average while preserving 80% individual credit assignment.
	var teamMean [2]float64
	for i := 0; i < AssaultHeroCount; i++ {
		teamMean[i/(AssaultHeroCount/2)] += raw[i] / (AssaultHeroCount / 2)
	}
	var corrected [AssaultHeroCount]float64
	var correctedMean [2]float64
	for i := 0; i < AssaultHeroCount; i++ {
		team := i / (AssaultHeroCount / 2)
		corrected[i] = raw[i] - teamMean[1-team]
		correctedMean[team] += corrected[i] / (AssaultHeroCount / 2)
	}
	for i := 0; i < AssaultHeroCount; i++ {
		team := i / (AssaultHeroCount / 2)
		r.Reward[i] = float32(0.8*corrected[i] + 0.2*correctedMean[team])
		if e.strategicReward && r.Done && r.Winner == 0 && !e.terminalRewarded {
			// Before zero-sum the same penalty on both teams would cancel to zero.
			r.Reward[i] -= 2
		}
	}
	if r.Done {
		e.terminalRewarded = true
	}
	for i, hero := range e.heroes {
		if assaultExternalOrderActiveLocked(hero, now) {
			r.ActiveOrder[i] = 1
		}
	}
	e.laneRewardStep = e.step
	return r
}

func assaultExternalOrderActiveLocked(c *conn, now float64) bool {
	if c == nil || c.huntState == nil {
		return false
	}
	hs := c.huntState
	if hs.closed || hs.deadUntil > now {
		return false
	}
	if c.hasDest || hs.order != nil || hs.attackTarget != 0 || hs.pvpTarget != 0 || hs.attackActionActive {
		return true
	}
	for _, channel := range hs.channels {
		if channel.interruptible && now <= channel.until {
			return true
		}
	}
	return false
}

// assaultTeacherTransitionLocked emits one deterministic v13 teacher status for
// a slot. The previous actor-visible action is updated only by ACTION or retired
// by CANCEL. UNAVAILABLE preserves a known lineage without leaking a payload;
// disappearance after such a gap still emits CANCEL.
func (e *AssaultEnv) assaultTeacherTransitionLocked(
	slot int, obs *AssaultObservationV1, terminal bool, now float64,
) (HeroActionV1, uint8) {
	if slot < 0 || slot >= AssaultHeroCount {
		return HeroActionV1{}, AssaultTeacherStatusUnavailable
	}
	if e.controllers[slot] != AssaultControllerAI30 {
		e.teacherState[slot] = assaultTeacherState{}
		return HeroActionV1{}, AssaultTeacherStatusUnavailable
	}
	projection := e.assaultTeacherProjectionLocked(slot, obs, terminal, now)
	return e.assaultTeacherTransitionFromProjectionLocked(slot, projection)
}

func (e *AssaultEnv) assaultTeacherTransitionFromProjectionLocked(
	slot int, projection assaultTeacherProjection,
) (HeroActionV1, uint8) {
	if slot < 0 || slot >= AssaultHeroCount {
		return HeroActionV1{}, AssaultTeacherStatusUnavailable
	}
	if e.controllers[slot] != AssaultControllerAI30 {
		e.teacherState[slot] = assaultTeacherState{}
		return HeroActionV1{}, AssaultTeacherStatusUnavailable
	}
	state := &e.teacherState[slot]
	if !projection.active {
		if state.active {
			*state = assaultTeacherState{}
			return HeroActionV1{}, AssaultTeacherStatusCancel
		}
		return HeroActionV1{}, AssaultTeacherStatusWait
	}
	if !projection.representable {
		return HeroActionV1{}, AssaultTeacherStatusUnavailable
	}
	if !state.active || state.action != projection.action {
		// Replacement is one ACTION transition. Retiring the old action and
		// installing the new actor-visible action are atomic, so no CANCEL is
		// emitted for an old lineage when a replacement is available this tick.
		state.action = projection.action
		state.active = true
		return projection.action, AssaultTeacherStatusAction
	}
	return HeroActionV1{}, AssaultTeacherStatusHold
}

// assaultTeacherProjectionLocked projects the current authoritative AI-30
// state into the actor-visible action contract. Internal object ids are used
// only to find the corresponding observation slot; they never enter the
// returned action. A present but masked/hidden order is active yet not
// representable, which is distinct from an idle AI-30 slot.
func (e *AssaultEnv) assaultTeacherProjectionLocked(
	slot int, obs *AssaultObservationV1, terminal bool, now float64,
) assaultTeacherProjection {
	if slot < 0 || slot >= AssaultHeroCount || e.controllers[slot] != AssaultControllerAI30 {
		return assaultTeacherProjection{active: true}
	}
	c := e.heroes[slot]
	if c == nil || c.huntState == nil {
		return assaultTeacherProjection{active: true}
	}
	hs := c.huntState
	if terminal || hs.deadUntil > now || hs.closed {
		if hs.deadUntil > now || terminal {
			c.assaultTeacherSkill = assaultTeacherSkillIntent{}
		}
		return assaultTeacherProjection{}
	}
	if obs == nil {
		return assaultTeacherProjection{active: true}
	}

	// A freshly accepted skill is the authoritative decision edge. It is
	// consumed once regardless of representability; a one-shot cast must not
	// become a fabricated hold on a later tick.
	if intent := c.assaultTeacherSkill; intent.sequence != 0 {
		c.assaultTeacherSkill = assaultTeacherSkillIntent{}
		return e.assaultTeacherSkillProjectionLocked(
			slot, c, obs, intent.slot, intent.target, intent.x, intent.y, intent.hasPos, now,
		)
	}
	// A skill approach remains an active resolved order until it reaches its
	// cast point or is cancelled. Derive it from the pending order so HOLD does
	// not degrade into a misleading movement label.
	if order := hs.order; order != nil {
		return e.assaultTeacherSkillProjectionLocked(
			slot, c, obs, order.slot, order.target, order.px, order.py, order.hasPos, now,
		)
	}

	if target := hs.attackTarget; target != 0 {
		return e.assaultTeacherAttackProjectionLocked(slot, obs, target)
	}
	if target := hs.pvpTarget; target != 0 {
		return e.assaultTeacherAttackProjectionLocked(slot, obs, target)
	}
	if !c.hasDest {
		return assaultTeacherProjection{}
	}
	if obs.ActionMask.Kinds[AssaultActionMove] == 0 {
		return assaultTeacherProjection{active: true}
	}
	action := e.assaultTeacherMoveActionLocked(c, obs, now)
	return assaultTeacherProjection{
		action:        action,
		active:        true,
		representable: true,
	}
}

func (e *AssaultEnv) assaultTeacherSkillProjectionLocked(
	slot int, c *conn, obs *AssaultObservationV1,
	skill int, target int32, px, py float32, hasPos bool, now float64,
) assaultTeacherProjection {
	skillIndex := skill - 1
	if skillIndex < 0 || skillIndex >= 4 {
		return assaultTeacherProjection{active: true}
	}
	kind := AssaultActionKindV1(int(AssaultActionSkill1) + skillIndex)
	if obs.ActionMask.Kinds[kind] == 0 {
		return assaultTeacherProjection{active: true}
	}
	action := HeroActionV1{Kind: kind}
	def := c.huntState.skillDef(skill)
	if skillHasTargetFlag(def, "ENEMY") || skillHasTargetFlag(def, "FRIEND") {
		if target == 0 {
			return assaultTeacherProjection{active: true}
		}
		enemyOnly := skillHasTargetFlag(def, "ENEMY") && !skillHasTargetFlag(def, "FRIEND")
		index, exists, alive, valid, visible := e.assaultTeacherTargetLocked(c, slot, obs, int32(target), now, enemyOnly)
		if !exists || !alive {
			return assaultTeacherProjection{}
		}
		if !valid || !visible || obs.ActionMask.SkillTarget[skillIndex][index] == 0 {
			return assaultTeacherProjection{active: true}
		}
		action.Target = uint16(index)
	}
	if e.navigationContract {
		x, y := c.posAtLocked(float32(now))
		dx, dy := 0, 0
		if hasPos {
			dx = int(math.Round(float64(px-x) / 3))
			dy = int(math.Round(float64(py-y) / 3))
		}
		dx = max(-4, min(4, dx))
		dy = max(-4, min(4, dy))
		action.Direction = uint8((dy+4)*9 + dx + 4)
	}
	return assaultTeacherProjection{
		action: action,
		active: true, representable: true,
	}
}

func (e *AssaultEnv) assaultTeacherAttackProjectionLocked(
	slot int, obs *AssaultObservationV1, target int32,
) assaultTeacherProjection {
	if obs.ActionMask.Kinds[AssaultActionAttack] == 0 {
		return assaultTeacherProjection{active: true}
	}
	c := e.heroes[slot]
	if c == nil {
		return assaultTeacherProjection{active: true}
	}
	index, exists, alive, valid, visible := e.assaultTeacherTargetLocked(c, slot, obs, target, e.clock.Now(), true)
	if !exists || !alive {
		return assaultTeacherProjection{}
	}
	if !valid || !visible || obs.ActionMask.Targets[index] == 0 {
		return assaultTeacherProjection{active: true}
	}
	return assaultTeacherProjection{
		action: HeroActionV1{Kind: AssaultActionAttack, Target: uint16(index)},
		active: true, representable: true,
	}
}

// assaultTeacherTargetLocked resolves a target without ever placing its hidden
// identity in the result. exists/alive distinguish a genuinely disappeared
// order from an enemy that is still authoritative but outside fog; valid is
// false for a stale/friendly target, and visible is derived solely from the
// current canonical observation slot list.
func (e *AssaultEnv) assaultTeacherTargetLocked(
	c *conn, slot int, obs *AssaultObservationV1, target int32, now float64, enemyOnly bool,
) (index int, exists, alive, valid, visible bool) {
	index = -1
	if e.inst == nil || c == nil || obs == nil || target == 0 {
		return index, false, false, false, false
	}
	if mob := e.inst.mobs[target]; mob != nil {
		exists, alive = true, !mob.dead
		valid = !enemyOnly || mob.enemyOf(c.playerTeam())
	} else if member := e.inst.members[target]; member != nil && member.huntState != nil {
		exists, alive = true, true
		if member.huntState.deadUntil > now {
			alive = false
		}
		valid = !enemyOnly || (member != c && arenaEnemies(c, member))
	}
	if !exists || !alive || !valid {
		return index, exists, alive, valid, false
	}
	for i, entityID := range e.entityIDs[slot] {
		if entityID == target && obs.EntityMask[i] != 0 {
			return i, true, true, true, true
		}
	}
	return index, true, true, true, false
}

func (e *AssaultEnv) assaultTeacherMoveActionLocked(c *conn, obs *AssaultObservationV1, now float64) HeroActionV1 {
	if e.navigationContract {
		for anchor := 1; anchor < AssaultNavigationAnchors; anchor++ {
			x, y, ok := e.assaultNavigationAnchorLocked(c, anchor)
			if ok && math.Hypot(float64(c.destX-x), float64(c.destY-y)) <= 6 {
				return HeroActionV1{Kind: AssaultActionMove, Distance: uint8(anchor)}
			}
		}
		x, y := c.posAtLocked(float32(now))
		dx := int(math.Round(float64(c.destX-x) / 3))
		dy := int(math.Round(float64(c.destY-y) / 3))
		dx = max(-4, min(4, dx))
		dy = max(-4, min(4, dy))
		return HeroActionV1{Kind: AssaultActionMove, Direction: uint8((dy+4)*9 + dx + 4)}
	}
	x, y := c.posAtLocked(float32(now))
	angle := math.Atan2(float64(c.destY-y), float64(c.destX-x))
	direction := int(math.Round(angle/(2*math.Pi)*16)) % 16
	if direction < 0 {
		direction += 16
	}
	distance := math.Hypot(float64(c.destX-x), float64(c.destY-y))
	distanceBin := uint8(0)
	if distance > 6 {
		distanceBin = 1
	}
	if distance > 10 {
		distanceBin = 2
	}
	return HeroActionV1{Kind: AssaultActionMove, Direction: uint8(direction), Distance: distanceBin}
}

// assaultTeacherActionLocked retains the historical projection helper for
// package-local callers. The v13 result path uses the transition method above
// so it can distinguish WAIT, HOLD, CANCEL and UNAVAILABLE.
func (e *AssaultEnv) assaultTeacherActionLocked(slot int, obs *AssaultObservationV1) (HeroActionV1, bool) {
	projection := e.assaultTeacherProjectionLocked(slot, obs, false, e.clock.Now())
	return projection.action, projection.active && projection.representable
}

func assaultEntityPriority(kind int) int {
	switch kind {
	case 1: // heroes always survive truncation
		return 0
	case 3: // strategic structures precede a large late-game creep set
		return 1
	default:
		return 2
	}
}

// assaultSortEntities preserves the published (kind, distance, id) order while
// avoiding cross-category comparisons. Heroes and structures occupy small,
// fixed groups, so sorting the three groups independently is materially cheaper
// than asking a generic sort to repeatedly compare their immutable priorities.
func assaultSortEntities(entities []assaultEntity) {
	heroes := 0
	for i := range entities {
		if assaultEntityPriority(entities[i].kind) != 0 {
			continue
		}
		entities[heroes], entities[i] = entities[i], entities[heroes]
		heroes++
	}
	structures := heroes
	for i := heroes; i < len(entities); i++ {
		if assaultEntityPriority(entities[i].kind) != 1 {
			continue
		}
		entities[structures], entities[i] = entities[i], entities[structures]
		structures++
	}
	byDistanceThenID := func(a, b assaultEntity) int {
		if a.distance != b.distance {
			return cmp.Compare(a.distance, b.distance)
		}
		return cmp.Compare(a.id, b.id)
	}
	slices.SortFunc(entities[:heroes], byDistanceThenID)
	slices.SortFunc(entities[heroes:structures], byDistanceThenID)
	slices.SortFunc(entities[structures:], byDistanceThenID)
}

func assaultEntityLess(a, b assaultEntity) bool {
	pa, pb := assaultEntityPriority(a.kind), assaultEntityPriority(b.kind)
	if pa != pb {
		return pa < pb
	}
	if a.distance != b.distance {
		return a.distance < b.distance
	}
	return a.id < b.id
}

// assaultTopEntities returns exactly the first AssaultMaxEntities elements of
// the published order.  Late matches can have hundreds of lane creeps but the
// wire contract exposes only 96 entities; a fixed max-heap avoids sorting
// entities that cannot possibly appear in an observation.  The final small sort
// restores the byte-for-byte canonical order.
func assaultTopEntities(entities []assaultEntity, scratch *[]assaultEntity) []assaultEntity {
	if len(entities) <= AssaultMaxEntities {
		assaultSortEntities(entities)
		return entities
	}
	top := (*scratch)[:0]
	if cap(top) < AssaultMaxEntities {
		top = make([]assaultEntity, 0, AssaultMaxEntities)
	}
	top = append(top, entities[:AssaultMaxEntities]...)
	// Max-heap: the least useful retained entity is at index zero.
	for parent := AssaultMaxEntities/2 - 1; parent >= 0; parent-- {
		assaultSiftWorstDown(top, parent)
	}
	for _, entity := range entities[AssaultMaxEntities:] {
		if !assaultEntityLess(entity, top[0]) {
			continue
		}
		top[0] = entity
		assaultSiftWorstDown(top, 0)
	}
	slices.SortFunc(top, func(a, b assaultEntity) int {
		if assaultEntityLess(a, b) {
			return -1
		}
		if assaultEntityLess(b, a) {
			return 1
		}
		return 0
	})
	*scratch = top
	return top
}

func assaultSiftWorstDown(heap []assaultEntity, root int) {
	for {
		left := root*2 + 1
		if left >= len(heap) {
			return
		}
		worst := left
		right := left + 1
		if right < len(heap) && assaultEntityLess(heap[left], heap[right]) {
			worst = right
		}
		if !assaultEntityLess(heap[root], heap[worst]) {
			return
		}
		heap[root], heap[worst] = heap[worst], heap[root]
		root = worst
	}
}

func (e *AssaultEnv) assaultObjectivePotentialLocked(team int32) float64 {
	return e.assaultObjectivePotentialsLocked()[assaultTeamIndex(team)]
}

func assaultTeamIndex(team int32) int {
	if team == dotaTeamElf {
		return 1
	}
	return 0
}

func (e *AssaultEnv) assaultObjectivePotentialsLocked() [2]float64 {
	var potential [2]float64
	for _, structure := range e.inst.dota.m.Structures {
		mob := e.inst.mobs[dotaStructIDBase+structure.ID]
		weight := 2.25
		switch structure.Role {
		case gamedata.DotaCreepTower:
			weight = 6
		case gamedata.DotaAltar:
			weight = 5
		}
		lost, dead := 1.0, true
		if mob != nil {
			lost = 1 - float64(safeFrac(mob.hp, mob.maxHealth()))
			dead = mob.dead
		}
		value := weight * (2.0 / 3.0) * lost
		if dead {
			value += weight / 3.0
		}
		structureTeam := dotaTeamHuman
		if structure.Side == gamedata.DotaSideElf {
			structureTeam = dotaTeamElf
		}
		if structureTeam == dotaTeamHuman {
			potential[assaultTeamIndex(dotaTeamElf)] += value
		} else {
			potential[assaultTeamIndex(dotaTeamHuman)] += value
		}
	}
	return potential
}

func safeFrac(v, max float64) float32 {
	if max <= 0 {
		return 0
	}
	return float32(math.Max(0, math.Min(1, v/max)))
}

func boolFloat(v bool) float32 {
	if v {
		return 1
	}
	return 0
}
