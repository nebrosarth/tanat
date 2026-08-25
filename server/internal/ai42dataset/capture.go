// Package ai42dataset contains the native AI-42 capture boundary.
//
// Capture is deliberately columnar: a caller supplies one authoritative
// battleserver result at a time, but the collector retains flat arrays rather
// than one Go object per hero and tick.  Finalize validates the complete
// capture and returns the same durable array columns consumed by the Python
// dataset writer, together with deterministic trajectory and incremental
// hashes.  The package does not import Python, NumPy, a learner, or an
// optimizer.
package ai42dataset

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"unicode/utf8"

	"tanatserver/internal/assaultproto"
	"tanatserver/internal/battleserver"
)

const (
	ProtocolVersion         uint16 = assaultproto.VersionAI42
	FrameRateHz             uint8  = 5
	DatasetSchemaVersion           = "AI42-dataset-v1"
	ShardSchemaVersion             = "AI42-go-shard-v1"
	TrajectorySchemaVersion        = 2

	HeroCount       = battleserver.AssaultHeroCount
	MaxEntities     = battleserver.AssaultMaxEntities
	HeroFeatures    = battleserver.AssaultHeroFeatureSize
	EntityFeatures  = battleserver.AssaultEntityFeatures
	GlobalFeatures  = battleserver.AssaultGlobalFeatures
	AbilityCount    = 4
	AbilityFeatures = battleserver.AssaultAbilityFeatures
	ActionKinds     = battleserver.AssaultActionKinds
	NavigationSlots = battleserver.AssaultNavigationOffsets
	AnchorSlots     = battleserver.AssaultNavigationAnchors
)

// Array names and order are the durable AI-42 array contract.  This matches
// dataset_ai42.py's _ARRAY_NAMES order, so a Python loader can map each
// compressed column without making per-row objects.
var arrayNames = [...]string{
	"hero", "abilities", "entities", "global", "entity_mask", "kind_mask",
	"target_mask", "skill_target_mask", "teacher_status", "teacher_action",
	"projected_action", "executed_action", "executed_valid", "rejection_reason",
	"rewards", "done", "winner", "step", "elapsed", "invalid",
}

// ArrayNames returns a copy of the stable durable column order.
func ArrayNames() []string { return append([]string(nil), arrayNames[:]...) }

// AI42SchemaHash is the exact v13 scalar/vector protocol schema hash emitted
// by assaultproto.  AI42RewardHash is the strategic V5 reward hash selected by
// protocol v13.
var (
	AI42SchemaHash = assaultproto.AI42SchemaHash
	AI42RewardHash = assaultproto.AI42RewardHash
)

const trajectorySchema = "tanat.ai42.teacher_trajectory.v2|" +
	"envelope(schema_version,schema_hash,trajectory_id,match_id,hero_id," +
	"expected_ticks,records)|" +
	"record(tick,sequence,hero_id,recurrent_parent_id,recurrent_boundary_id," +
	"observation_hash,action_hash,action_hash_source,mask_hash," +
	"original_ai30_intent,projected_neural_action,executed_action,valid," +
	"rejection_reason,outcome,integrity_hash)|" +
	"action(kind,target,point,anchor,timing,skill,lineage_id)|" +
	"outcome(reward,terminal,winner,hero_alive,team_reward,damage,kills," +
	"deaths,event)|canonical-json-utf8"

// AI42TrajectorySchemaHash is compatible with trajectory_ai42.py's v2
// schema hash.  It is stored in every shard header and is part of the match
// provenance, even though the native stream stores compact columns instead of
// materializing TeacherTrajectoryRecord values.
var AI42TrajectorySchemaHash = sha256.Sum256([]byte(trajectorySchema))

// Hash is a SHA-256 digest in the wire order used by the native stream.
type Hash [32]byte

// Action is the canonical v13 wire action.  It is intentionally separate from
// the battleserver type so the Python boundary only needs the five-byte wire
// representation and does not acquire gameplay package ownership.
type Action struct {
	Kind      uint8
	Target    uint16
	Direction uint8
	Distance  uint8
}

func actionFromBattle(value battleserver.HeroActionV1) Action {
	return Action{Kind: uint8(value.Kind), Target: value.Target, Direction: value.Direction, Distance: value.Distance}
}

func (value Action) isZero() bool { return value == (Action{}) }

func (value Action) wireBytes(dst []byte) {
	dst[0] = value.Kind
	binary.LittleEndian.PutUint16(dst[1:3], value.Target)
	dst[3] = value.Direction
	dst[4] = value.Distance
}

// Outcome retains the complete trajectory outcome envelope.  Winner and
// HeroAlive use explicit presence bits because the Python schema permits null.
// Reward and Terminal are authoritative and must agree with the result row.
type Outcome struct {
	Reward           float32
	Terminal         bool
	Winner           int32
	WinnerPresent    bool
	HeroAlive        bool
	HeroAlivePresent bool
	TeamReward       float32
	Damage           float32
	Kills            uint32
	Deaths           uint32
	Event            string
}

// Metadata is the exact match/runtime provenance needed by the Python loader.
// RuntimeManifest must be canonical UTF-8 JSON supplied by the control plane;
// it is hashed byte-for-byte and never normalized or discarded here.
type Metadata struct {
	ProtocolVersion      uint16
	TickHz               uint8
	MatchID              string
	HeroIDs              [HeroCount]string
	RuntimeManifest      []byte
	RuntimeManifestHash  Hash
	SchemaHash           Hash
	RewardHash           Hash
	TrajectorySchemaHash Hash
	Seed                 int64
	Scenario             string
	ControllerBySlot     [HeroCount]uint8
	RosterIDs            [HeroCount]int32
	SideBySlot           [HeroCount]uint8
}

// Prepared is a validated, immutable-by-convention view of one match.  Every
// slice is row-major and contiguous in the same leading dimensions listed in
// the shard header.  The collector stops accepting rows once Finalize returns.
type Prepared struct {
	Metadata  Metadata
	TickCount int

	Steps   []uint32
	Elapsed []float32
	Done    []uint8
	Winner  []int32

	Invalid   []uint8 // [tick, hero]
	Rewards   []float32
	Hero      []float32 // [tick, hero, 32]
	Abilities []float32 // [tick, hero, 4, 40]
	Entities  []float32 // [tick, hero, 96, 16]
	Global    []float32 // [tick, hero, 32]

	EntityMask      []uint8 // [tick, hero, 96]
	KindMask        []uint8 // [tick, hero, 8]
	TargetMask      []uint8 // [tick, hero, 96]
	SkillTargetMask []uint8 // [tick, hero, 4, 96]

	TeacherStatus   []uint8
	TeacherAction   []Action
	ProjectedAction []Action
	ExecutedAction  []Action
	ExecutedValid   []uint8
	RejectionReason []uint8

	PreviousRecurrentIDs []string // [tick, hero]
	RecurrentBoundaryIDs []string // [tick, hero]

	OutcomeReward           []float32
	OutcomeTerminal         []uint8
	OutcomeWinner           []int32
	OutcomeWinnerPresent    []uint8
	OutcomeHeroAlive        []uint8
	OutcomeHeroAlivePresent []uint8
	OutcomeTeamReward       []float32
	OutcomeDamage           []float32
	OutcomeKills            []uint32
	OutcomeDeaths           []uint32
	OutcomeEvent            []string

	TrajectoryIDs     [HeroCount]string
	TrajectoryHashes  [HeroCount]Hash
	IncrementalHashes [HeroCount][]Hash // [hero][tick]
	MatchHash         Hash
}

// ValidationError always identifies the durable field and, where applicable,
// the offending tick and hero slot.  This keeps errors actionable at the
// Python control-plane boundary and avoids opaque bulk-array failures.
type ValidationError struct {
	Field   string
	Tick    int
	Slot    int
	Message string
}

func (e *ValidationError) Error() string {
	if e.Tick >= 0 && e.Slot >= 0 {
		return fmt.Sprintf("ai42dataset: field=%s tick=%d slot=%d: %s", e.Field, e.Tick, e.Slot, e.Message)
	}
	if e.Tick >= 0 {
		return fmt.Sprintf("ai42dataset: field=%s tick=%d: %s", e.Field, e.Tick, e.Message)
	}
	return fmt.Sprintf("ai42dataset: field=%s: %s", e.Field, e.Message)
}

func fieldError(field string, tick, slot int, format string, args ...any) error {
	return &ValidationError{Field: field, Tick: tick, Slot: slot, Message: fmt.Sprintf(format, args...)}
}

func metadataError(field, format string, args ...any) error {
	return fieldError(field, -1, -1, format, args...)
}

// Capture incrementally accepts native simulation rows into flat columns.
type Capture struct {
	data     Prepared
	rolling  [HeroCount]Hash
	finished bool
}

// NewCapture validates and snapshots metadata.  The returned collector owns a
// copy of RuntimeManifest and can be fed directly from AssaultEnv.Step.
func NewCapture(metadata Metadata) (*Capture, error) {
	if err := validateMetadata(&metadata); err != nil {
		return nil, err
	}
	metadata.RuntimeManifest = append([]byte(nil), metadata.RuntimeManifest...)
	return &Capture{data: Prepared{Metadata: metadata}}, nil
}

// Reserve preallocates flat columns for expectedTicks.  It is optional and
// does not change the format or validation behavior.
func (c *Capture) Reserve(expectedTicks int) error {
	if c == nil {
		return metadataError("capture", "nil capture")
	}
	if c.finished {
		return metadataError("capture", "capture is already finalized")
	}
	if expectedTicks < 0 {
		return metadataError("expected_ticks", "must be non-negative")
	}
	var err error
	c.data.Steps, err = reserveTicks(c.data.Steps, expectedTicks, 1)
	if err != nil {
		return err
	}
	c.data.Elapsed, err = reserveTicks(c.data.Elapsed, expectedTicks, 1)
	if err != nil {
		return err
	}
	c.data.Done, err = reserveTicks(c.data.Done, expectedTicks, 1)
	if err != nil {
		return err
	}
	c.data.Winner, err = reserveTicks(c.data.Winner, expectedTicks, 1)
	if err != nil {
		return err
	}
	c.data.Invalid, err = reserveTicks(c.data.Invalid, expectedTicks, HeroCount)
	if err != nil {
		return err
	}
	c.data.Rewards, err = reserveTicks(c.data.Rewards, expectedTicks, HeroCount)
	if err != nil {
		return err
	}
	c.data.Hero, err = reserveTicks(c.data.Hero, expectedTicks, HeroCount*HeroFeatures)
	if err != nil {
		return err
	}
	c.data.Abilities, err = reserveTicks(c.data.Abilities, expectedTicks, HeroCount*AbilityCount*AbilityFeatures)
	if err != nil {
		return err
	}
	c.data.Entities, err = reserveTicks(c.data.Entities, expectedTicks, HeroCount*MaxEntities*EntityFeatures)
	if err != nil {
		return err
	}
	c.data.Global, err = reserveTicks(c.data.Global, expectedTicks, HeroCount*GlobalFeatures)
	if err != nil {
		return err
	}
	c.data.EntityMask, err = reserveTicks(c.data.EntityMask, expectedTicks, HeroCount*MaxEntities)
	if err != nil {
		return err
	}
	c.data.KindMask, err = reserveTicks(c.data.KindMask, expectedTicks, HeroCount*ActionKinds)
	if err != nil {
		return err
	}
	c.data.TargetMask, err = reserveTicks(c.data.TargetMask, expectedTicks, HeroCount*MaxEntities)
	if err != nil {
		return err
	}
	c.data.SkillTargetMask, err = reserveTicks(c.data.SkillTargetMask, expectedTicks, HeroCount*AbilityCount*MaxEntities)
	if err != nil {
		return err
	}
	c.data.TeacherStatus, err = reserveTicks(c.data.TeacherStatus, expectedTicks, HeroCount)
	if err != nil {
		return err
	}
	c.data.TeacherAction, err = reserveTicks(c.data.TeacherAction, expectedTicks, HeroCount)
	if err != nil {
		return err
	}
	c.data.ProjectedAction, err = reserveTicks(c.data.ProjectedAction, expectedTicks, HeroCount)
	if err != nil {
		return err
	}
	c.data.ExecutedAction, err = reserveTicks(c.data.ExecutedAction, expectedTicks, HeroCount)
	if err != nil {
		return err
	}
	c.data.ExecutedValid, err = reserveTicks(c.data.ExecutedValid, expectedTicks, HeroCount)
	if err != nil {
		return err
	}
	c.data.RejectionReason, err = reserveTicks(c.data.RejectionReason, expectedTicks, HeroCount)
	if err != nil {
		return err
	}
	c.data.PreviousRecurrentIDs, err = reserveTicks(c.data.PreviousRecurrentIDs, expectedTicks, HeroCount)
	if err != nil {
		return err
	}
	c.data.RecurrentBoundaryIDs, err = reserveTicks(c.data.RecurrentBoundaryIDs, expectedTicks, HeroCount)
	if err != nil {
		return err
	}
	c.data.OutcomeReward, err = reserveTicks(c.data.OutcomeReward, expectedTicks, HeroCount)
	if err != nil {
		return err
	}
	c.data.OutcomeTerminal, err = reserveTicks(c.data.OutcomeTerminal, expectedTicks, HeroCount)
	if err != nil {
		return err
	}
	c.data.OutcomeWinner, err = reserveTicks(c.data.OutcomeWinner, expectedTicks, HeroCount)
	if err != nil {
		return err
	}
	c.data.OutcomeWinnerPresent, err = reserveTicks(c.data.OutcomeWinnerPresent, expectedTicks, HeroCount)
	if err != nil {
		return err
	}
	c.data.OutcomeHeroAlive, err = reserveTicks(c.data.OutcomeHeroAlive, expectedTicks, HeroCount)
	if err != nil {
		return err
	}
	c.data.OutcomeHeroAlivePresent, err = reserveTicks(c.data.OutcomeHeroAlivePresent, expectedTicks, HeroCount)
	if err != nil {
		return err
	}
	c.data.OutcomeTeamReward, err = reserveTicks(c.data.OutcomeTeamReward, expectedTicks, HeroCount)
	if err != nil {
		return err
	}
	c.data.OutcomeDamage, err = reserveTicks(c.data.OutcomeDamage, expectedTicks, HeroCount)
	if err != nil {
		return err
	}
	c.data.OutcomeKills, err = reserveTicks(c.data.OutcomeKills, expectedTicks, HeroCount)
	if err != nil {
		return err
	}
	c.data.OutcomeDeaths, err = reserveTicks(c.data.OutcomeDeaths, expectedTicks, HeroCount)
	if err != nil {
		return err
	}
	c.data.OutcomeEvent, err = reserveTicks(c.data.OutcomeEvent, expectedTicks, HeroCount)
	if err != nil {
		return err
	}
	for hero := 0; hero < HeroCount; hero++ {
		c.data.IncrementalHashes[hero], err = reserveTicks(c.data.IncrementalHashes[hero], expectedTicks, 1)
		if err != nil {
			return err
		}
	}
	return nil
}

func reserveTicks[T any](values []T, expectedTicks, width int) ([]T, error) {
	maxInt := int(^uint(0) >> 1)
	if width <= 0 || expectedTicks > maxInt/width {
		return values, metadataError("expected_ticks", "capacity overflows native address space")
	}
	target := expectedTicks * width
	if cap(values) >= target {
		return values, nil
	}
	grown := make([]T, len(values), target)
	copy(grown, values)
	return grown, nil
}

// Append records one authoritative 5Hz result plus the caller's submitted
// actions, recurrent lineage, and complete per-slot outcomes.  No trajectory
// dataclass-equivalent value is created or retained per tick.
func (c *Capture) Append(
	result *battleserver.StepResultV1,
	submitted [HeroCount]Action,
	parents [HeroCount]string,
	boundaries [HeroCount]string,
	outcomes [HeroCount]Outcome,
) error {
	if c == nil {
		return metadataError("capture", "nil capture")
	}
	if c.finished {
		return metadataError("capture", "capture is already finalized")
	}
	if result == nil {
		return fieldError("result", len(c.data.Steps), -1, "nil result")
	}
	tick := len(c.data.Steps)
	if tick > 0 && c.data.Done[tick-1] != 0 {
		return fieldError("done", tick-1, -1, "terminal tick must be final")
	}
	if tick > 0 {
		previous := c.data.Steps[tick-1]
		if previous == ^uint32(0) || result.Step != previous+1 {
			return fieldError("result.step", tick, -1, "must be contiguous after %d, got %d", previous, result.Step)
		}
	}
	if result.SchemaHash != c.data.Metadata.SchemaHash {
		return fieldError("result.schema_hash", tick, -1, "does not match metadata")
	}
	if result.RewardHash != c.data.Metadata.RewardHash {
		return fieldError("result.reward_hash", tick, -1, "does not match metadata")
	}
	if !finite32(result.Elapsed) {
		return fieldError("result.elapsed", tick, -1, "must be finite")
	}
	if err := validateResultRows(result, tick); err != nil {
		return err
	}
	for slot := 0; slot < HeroCount; slot++ {
		if err := validateAction(submitted[slot], "submitted_action", tick, slot, true); err != nil {
			return err
		}
		if parents[slot] == "" || !utf8.ValidString(parents[slot]) {
			return fieldError("recurrent_parent_id", tick, slot, "must be a non-empty UTF-8 string")
		}
		if boundaries[slot] == "" || !utf8.ValidString(boundaries[slot]) {
			return fieldError("recurrent_boundary_id", tick, slot, "must be a non-empty UTF-8 string")
		}
		if parents[slot] == boundaries[slot] {
			return fieldError("recurrent_boundary_id", tick, slot, "must differ from recurrent parent")
		}
		if tick > 0 && parents[slot] != c.data.RecurrentBoundaryIDs[(tick-1)*HeroCount+slot] {
			return fieldError("recurrent_parent_id", tick, slot, "does not match prior recurrent boundary")
		}
		if err := validateOutcome(outcomes[slot], result, slot, tick); err != nil {
			return err
		}
	}

	appendResult(&c.data, result)
	for slot := 0; slot < HeroCount; slot++ {
		c.data.ProjectedAction = append(c.data.ProjectedAction, submitted[slot])
		c.data.PreviousRecurrentIDs = append(c.data.PreviousRecurrentIDs, parents[slot])
		c.data.RecurrentBoundaryIDs = append(c.data.RecurrentBoundaryIDs, boundaries[slot])
		appendOutcome(&c.data, outcomes[slot])
		c.data.IncrementalHashes[slot] = append(c.data.IncrementalHashes[slot], c.nextIncrementalHash(result, slot, submitted[slot], parents[slot], boundaries[slot], outcomes[slot]))
	}
	return nil
}

// Record is a readable alias for Append at integration call sites.
func (c *Capture) Record(
	result *battleserver.StepResultV1,
	submitted [HeroCount]Action,
	parents [HeroCount]string,
	boundaries [HeroCount]string,
	outcomes [HeroCount]Outcome,
) error {
	return c.Append(result, submitted, parents, boundaries, outcomes)
}

// Finalize validates all columns, computes trajectory hashes compatible with
// the Python fast collector, computes the match hash, and seals the collector.
func (c *Capture) Finalize() (*Prepared, error) {
	if c == nil {
		return nil, metadataError("capture", "nil capture")
	}
	if c.finished {
		return nil, metadataError("capture", "capture is already finalized")
	}
	if len(c.data.Steps) == 0 {
		return nil, fieldError("result", 0, -1, "at least one tick is required")
	}
	if c.data.Done[len(c.data.Done)-1] == 0 {
		return nil, fieldError("done", len(c.data.Done)-1, -1, "capture must end at its only terminal tick")
	}
	c.data.TickCount = len(c.data.Steps)
	if err := validatePrepared(&c.data); err != nil {
		return nil, err
	}
	for hero := 0; hero < HeroCount; hero++ {
		c.data.TrajectoryIDs[hero] = c.data.Metadata.MatchID + ":hero:" + c.data.Metadata.HeroIDs[hero]
		c.data.TrajectoryHashes[hero] = c.trajectoryHash(hero)
	}
	c.data.MatchHash = matchHash(&c.data)
	c.finished = true
	return &c.data, nil
}

// Validate checks a prepared value obtained from a native reader before it is
// handed to the Python loader.
func (p *Prepared) Validate() error {
	if p == nil {
		return metadataError("prepared", "nil prepared match")
	}
	return validatePrepared(p)
}

func validateMetadata(metadata *Metadata) error {
	if metadata.ProtocolVersion != ProtocolVersion {
		return metadataError("protocol_version", "got %d, want %d", metadata.ProtocolVersion, ProtocolVersion)
	}
	if metadata.TickHz != FrameRateHz {
		return metadataError("tick_hz", "got %d, want %d", metadata.TickHz, FrameRateHz)
	}
	if metadata.MatchID == "" || !utf8.ValidString(metadata.MatchID) {
		return metadataError("match_id", "must be a non-empty UTF-8 string")
	}
	if metadata.Scenario == "" || !utf8.ValidString(metadata.Scenario) {
		return metadataError("scenario", "must be a non-empty UTF-8 string")
	}
	seen := make(map[string]struct{}, HeroCount)
	rosterSeen := make(map[int32]struct{}, HeroCount)
	zeroSides, oneSides := 0, 0
	for slot, id := range metadata.HeroIDs {
		if id == "" || !utf8.ValidString(id) {
			return metadataError(fmt.Sprintf("hero_ids[%d]", slot), "must be a non-empty UTF-8 string")
		}
		if _, ok := seen[id]; ok {
			return metadataError(fmt.Sprintf("hero_ids[%d]", slot), "duplicates hero ID %q", id)
		}
		seen[id] = struct{}{}
		if metadata.ControllerBySlot[slot] > uint8(battleserver.AssaultControllerAI40) {
			return metadataError(fmt.Sprintf("controller_by_slot[%d]", slot), "unknown controller %d", metadata.ControllerBySlot[slot])
		}
		if metadata.SideBySlot[slot] > 1 {
			return metadataError(fmt.Sprintf("side_by_slot[%d]", slot), "must be 0 or 1")
		}
		if _, ok := rosterSeen[metadata.RosterIDs[slot]]; ok {
			return metadataError(fmt.Sprintf("roster_ids[%d]", slot), "duplicates roster ID %d", metadata.RosterIDs[slot])
		}
		rosterSeen[metadata.RosterIDs[slot]] = struct{}{}
		if metadata.SideBySlot[slot] == 0 {
			zeroSides++
		} else {
			oneSides++
		}
	}
	if zeroSides != HeroCount/2 || oneSides != HeroCount/2 {
		return metadataError("side_by_slot", "must contain five slots per side")
	}
	if len(metadata.RuntimeManifest) == 0 || !json.Valid(metadata.RuntimeManifest) {
		return metadataError("runtime_manifest", "must be non-empty valid JSON")
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(metadata.RuntimeManifest, &object); err != nil || len(object) == 0 {
		return metadataError("runtime_manifest", "must be a non-empty JSON object")
	}
	canonical, err := canonicalJSONValue(metadata.RuntimeManifest)
	if err != nil {
		return metadataError("runtime_manifest", "must be canonical JSON: %v", err)
	}
	if !bytes.Equal(canonical, metadata.RuntimeManifest) {
		return metadataError("runtime_manifest", "must use canonical UTF-8 JSON bytes")
	}
	if sha256.Sum256(metadata.RuntimeManifest) != metadata.RuntimeManifestHash {
		return metadataError("runtime_manifest_hash", "does not match runtime_manifest bytes")
	}
	if metadata.SchemaHash == (Hash{}) {
		return metadataError("schema_hash", "must not be zero")
	}
	if metadata.SchemaHash != AI42SchemaHash {
		return metadataError("schema_hash", "does not match AI42 protocol v13")
	}
	if metadata.RewardHash == (Hash{}) {
		return metadataError("reward_hash", "must not be zero")
	}
	if metadata.RewardHash != AI42RewardHash {
		return metadataError("reward_hash", "does not match AI42 protocol v13")
	}
	if metadata.TrajectorySchemaHash != AI42TrajectorySchemaHash {
		return metadataError("trajectory_schema_hash", "does not match trajectory schema v%d", TrajectorySchemaVersion)
	}
	return nil
}

func validateResultRows(result *battleserver.StepResultV1, tick int) error {
	for slot := 0; slot < HeroCount; slot++ {
		if result.Invalid[slot] > 1 {
			return fieldError("invalid", tick, slot, "must be zero or one")
		}
		if !finite32(result.Reward[slot]) {
			return fieldError("rewards", tick, slot, "must be finite")
		}
		observation := result.Observations[slot]
		for index, value := range observation.Hero {
			if !finite32(value) {
				return fieldError("hero", tick, slot, "contains non-finite value at index %d", index)
			}
		}
		for ability := range observation.Abilities {
			for index, value := range observation.Abilities[ability] {
				if !finite32(value) {
					return fieldError("abilities", tick, slot, "contains non-finite value at ability=%d index=%d", ability, index)
				}
			}
		}
		for entity := range observation.Entities {
			for index, value := range observation.Entities[entity] {
				if !finite32(value) {
					return fieldError("entities", tick, slot, "contains non-finite value at entity=%d index=%d", entity, index)
				}
			}
		}
		for index, value := range observation.Global {
			if !finite32(value) {
				return fieldError("global", tick, slot, "contains non-finite value at index %d", index)
			}
		}
		for index, value := range observation.EntityMask {
			if value > 1 {
				return fieldError("entity_mask", tick, slot, "must be zero or one at index %d", index)
			}
		}
		for index, value := range observation.ActionMask.Kinds {
			if value > 1 {
				return fieldError("kind_mask", tick, slot, "must be zero or one at index %d", index)
			}
		}
		for index, value := range observation.ActionMask.Targets {
			if value > 1 {
				return fieldError("target_mask", tick, slot, "must be zero or one at index %d", index)
			}
		}
		for ability := range observation.ActionMask.SkillTarget {
			for index, value := range observation.ActionMask.SkillTarget[ability] {
				if value > 1 {
					return fieldError("skill_target_mask", tick, slot, "must be zero or one at ability=%d index=%d", ability, index)
				}
			}
		}
		if result.ExecutedValid[slot] > 1 {
			return fieldError("executed_valid", tick, slot, "must be zero or one")
		}
		if result.TeacherStatus[slot] > battleserver.AssaultTeacherStatusUnavailable {
			return fieldError("teacher_status", tick, slot, "unknown v13 status %d", result.TeacherStatus[slot])
		}
		if err := validateAction(actionFromBattle(result.TeacherIntent[slot]), "teacher_intent", tick, slot, true); err != nil {
			return err
		}
		if err := validateAction(actionFromBattle(result.ExecutedActions[slot]), "executed_action", tick, slot, true); err != nil {
			return err
		}
		switch result.TeacherStatus[slot] {
		case battleserver.AssaultTeacherStatusAction:
			if actionFromBattle(result.TeacherIntent[slot]).Kind == 0 {
				return fieldError("teacher_intent", tick, slot, "action status cannot carry wait")
			}
		default:
			if !actionFromBattle(result.TeacherIntent[slot]).isZero() {
				return fieldError("teacher_intent", tick, slot, "control status must carry a zero action payload")
			}
		}
		reason := result.RejectionReason[slot]
		if !validRejectionReason(reason) {
			return fieldError("rejection_reason", tick, slot, "unknown v13 code %d", reason)
		}
		if result.ExecutedValid[slot] != 0 {
			if reason != battleserver.AssaultRejectionReasonNone {
				return fieldError("rejection_reason", tick, slot, "accepted action must have reason none")
			}
		} else {
			if reason == battleserver.AssaultRejectionReasonNone {
				return fieldError("rejection_reason", tick, slot, "rejected action must have a reason")
			}
			if !actionFromBattle(result.ExecutedActions[slot]).isZero() {
				return fieldError("executed_action", tick, slot, "rejected action must be zero")
			}
		}
	}
	return nil
}

func validateAction(action Action, field string, tick, slot int, canonical bool) error {
	if action.Kind >= ActionKinds {
		return fieldError(field, tick, slot, "kind %d is outside [0,%d)", action.Kind, ActionKinds)
	}
	if action.Direction >= NavigationSlots {
		return fieldError(field, tick, slot, "direction %d is outside [0,%d)", action.Direction, NavigationSlots)
	}
	if action.Distance >= AnchorSlots {
		return fieldError(field, tick, slot, "distance %d is outside [0,%d)", action.Distance, AnchorSlots)
	}
	if canonical {
		switch {
		case action.Kind == 0 && (action.Target != 0 || action.Direction != 0 || action.Distance != 0):
			return fieldError(field, tick, slot, "wait action must carry zero target/direction/distance")
		case action.Kind == uint8(battleserver.AssaultActionAttack) && (action.Direction != 0 || action.Distance != 0):
			return fieldError(field, tick, slot, "attack action cannot carry navigation fields")
		case action.Kind >= uint8(battleserver.AssaultActionSkill1) && action.Kind <= uint8(battleserver.AssaultActionSkill4) && action.Distance != 0:
			return fieldError(field, tick, slot, "skill action cannot carry an anchor")
		case action.Kind == uint8(battleserver.AssaultActionTeleport) && (action.Direction != 0 || action.Distance != 0):
			return fieldError(field, tick, slot, "teleport action cannot carry navigation fields")
		}
	}
	return nil
}

func validRejectionReason(value uint8) bool {
	switch value {
	case battleserver.AssaultRejectionReasonNone, battleserver.AssaultRejectionReasonMasked,
		battleserver.AssaultRejectionReasonInvalid, battleserver.AssaultRejectionReasonServerRejected,
		battleserver.AssaultRejectionReasonSafety, battleserver.AssaultRejectionReasonTimeout,
		battleserver.AssaultRejectionReasonPolicyError, battleserver.AssaultRejectionReasonUnknown:
		return true
	default:
		return false
	}
}

func validateOutcome(outcome Outcome, result *battleserver.StepResultV1, slot, tick int) error {
	if !finite32(outcome.Reward) || !finite32(outcome.TeamReward) || !finite32(outcome.Damage) {
		return fieldError("outcome", tick, slot, "reward/team_reward/damage must be finite")
	}
	if math.Float32bits(outcome.Reward) != math.Float32bits(result.Reward[slot]) {
		return fieldError("outcome.reward", tick, slot, "does not match result reward")
	}
	if outcome.Terminal != result.Done {
		return fieldError("outcome.terminal", tick, slot, "does not match result done")
	}
	if outcome.WinnerPresent && outcome.Winner != result.Winner {
		return fieldError("outcome.winner", tick, slot, "does not match result winner")
	}
	if outcome.Event != "" && !utf8.ValidString(outcome.Event) {
		return fieldError("outcome.event", tick, slot, "must be valid UTF-8")
	}
	return nil
}

func appendResult(data *Prepared, result *battleserver.StepResultV1) {
	data.Steps = append(data.Steps, result.Step)
	data.Elapsed = append(data.Elapsed, result.Elapsed)
	if result.Done {
		data.Done = append(data.Done, 1)
	} else {
		data.Done = append(data.Done, 0)
	}
	data.Winner = append(data.Winner, result.Winner)
	for slot := 0; slot < HeroCount; slot++ {
		data.Invalid = append(data.Invalid, result.Invalid[slot])
		data.Rewards = append(data.Rewards, result.Reward[slot])
		observation := result.Observations[slot]
		data.Hero = append(data.Hero, observation.Hero[:]...)
		for ability := range observation.Abilities {
			data.Abilities = append(data.Abilities, observation.Abilities[ability][:]...)
		}
		for entity := range observation.Entities {
			data.Entities = append(data.Entities, observation.Entities[entity][:]...)
		}
		data.Global = append(data.Global, observation.Global[:]...)
		data.EntityMask = append(data.EntityMask, observation.EntityMask[:]...)
		data.KindMask = append(data.KindMask, observation.ActionMask.Kinds[:]...)
		data.TargetMask = append(data.TargetMask, observation.ActionMask.Targets[:]...)
		for ability := range observation.ActionMask.SkillTarget {
			data.SkillTargetMask = append(data.SkillTargetMask, observation.ActionMask.SkillTarget[ability][:]...)
		}
		data.TeacherStatus = append(data.TeacherStatus, result.TeacherStatus[slot])
		data.TeacherAction = append(data.TeacherAction, actionFromBattle(result.TeacherIntent[slot]))
		data.ExecutedAction = append(data.ExecutedAction, actionFromBattle(result.ExecutedActions[slot]))
		data.ExecutedValid = append(data.ExecutedValid, result.ExecutedValid[slot])
		data.RejectionReason = append(data.RejectionReason, result.RejectionReason[slot])
	}
}

func appendOutcome(data *Prepared, outcome Outcome) {
	data.OutcomeReward = append(data.OutcomeReward, outcome.Reward)
	if outcome.Terminal {
		data.OutcomeTerminal = append(data.OutcomeTerminal, 1)
	} else {
		data.OutcomeTerminal = append(data.OutcomeTerminal, 0)
	}
	data.OutcomeWinner = append(data.OutcomeWinner, outcome.Winner)
	if outcome.WinnerPresent {
		data.OutcomeWinnerPresent = append(data.OutcomeWinnerPresent, 1)
	} else {
		data.OutcomeWinnerPresent = append(data.OutcomeWinnerPresent, 0)
	}
	if outcome.HeroAlive {
		data.OutcomeHeroAlive = append(data.OutcomeHeroAlive, 1)
	} else {
		data.OutcomeHeroAlive = append(data.OutcomeHeroAlive, 0)
	}
	if outcome.HeroAlivePresent {
		data.OutcomeHeroAlivePresent = append(data.OutcomeHeroAlivePresent, 1)
	} else {
		data.OutcomeHeroAlivePresent = append(data.OutcomeHeroAlivePresent, 0)
	}
	data.OutcomeTeamReward = append(data.OutcomeTeamReward, outcome.TeamReward)
	data.OutcomeDamage = append(data.OutcomeDamage, outcome.Damage)
	data.OutcomeKills = append(data.OutcomeKills, outcome.Kills)
	data.OutcomeDeaths = append(data.OutcomeDeaths, outcome.Deaths)
	data.OutcomeEvent = append(data.OutcomeEvent, outcome.Event)
}

func finite32(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}

func (c *Capture) nextIncrementalHash(result *battleserver.StepResultV1, slot int, submitted Action, parent, boundary string, outcome Outcome) Hash {
	row := sha256.New()
	row.Write([]byte("AI42-incremental-row-v1\x00"))
	writeU32(row, result.Step)
	writeString(row, parent)
	writeString(row, boundary)
	writeActionHash(row, submitted)
	writeActionHash(row, actionFromBattle(result.TeacherIntent[slot]))
	writeActionHash(row, actionFromBattle(result.ExecutedActions[slot]))
	writeByte(row, result.TeacherStatus[slot])
	writeByte(row, result.ExecutedValid[slot])
	writeByte(row, result.RejectionReason[slot])
	writeByte(row, result.Invalid[slot])
	writeF32(row, result.Reward[slot])
	writeOutcome(row, outcome)
	writeSlotObservation(row, result.Observations[slot])
	rowHash := row.Sum(nil)
	h := sha256.New()
	h.Write([]byte("AI42-incremental-v1\x00"))
	h.Write(c.rolling[slot][:])
	h.Write(rowHash)
	var resultHash Hash
	copy(resultHash[:], h.Sum(nil))
	c.rolling[slot] = resultHash
	return resultHash
}

func (c *Capture) trajectoryHash(hero int) Hash {
	digest := sha256.New()
	digest.Write([]byte("AI42-fast-trajectory-v1\x00"))
	evidence := map[string]any{
		"match_id":   c.data.Metadata.MatchID,
		"hero_id":    c.data.Metadata.HeroIDs[hero],
		"steps":      c.data.Steps,
		"parents":    columnStrings(c.data.PreviousRecurrentIDs, hero, HeroCount),
		"boundaries": columnStrings(c.data.RecurrentBoundaryIDs, hero, HeroCount),
	}
	digest.Write(canonicalJSON(evidence))
	for _, name := range arrayNames {
		column := c.arrayColumnHash(name, hero)
		digest.Write([]byte(name))
		digest.Write(column[:])
	}
	var result Hash
	copy(result[:], digest.Sum(nil))
	return result
}

func canonicalJSON(value any) []byte {
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	_ = encoder.Encode(value)
	result := buffer.Bytes()
	if len(result) > 0 && result[len(result)-1] == '\n' {
		result = result[:len(result)-1]
	}
	return result
}

// CanonicalizeJSON verifies that payload already is compact, sorted-key UTF-8
// JSON using the same number-preserving representation as the Python contract.
// It never silently rewrites schedule/provenance bytes.
func CanonicalizeJSON(payload []byte) ([]byte, error) {
	canonical, err := canonicalJSONValue(payload)
	if err != nil {
		return nil, err
	}
	if !bytes.Equal(canonical, payload) {
		return nil, fmt.Errorf("JSON bytes are not canonical")
	}
	return append([]byte(nil), payload...), nil
}

func canonicalJSONValue(payload []byte) ([]byte, error) {
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return nil, fmt.Errorf("multiple JSON values")
		}
		return nil, err
	}
	return canonicalJSON(value), nil
}

func columnStrings(values []string, hero, stride int) []string {
	result := make([]string, len(values)/stride)
	for tick := range result {
		result[tick] = values[tick*stride+hero]
	}
	return result
}

func (c *Capture) arrayColumnHash(name string, hero int) Hash {
	h := sha256.New()
	for tick := 0; tick < len(c.data.Steps); tick++ {
		index := tick*HeroCount + hero
		switch name {
		case "hero":
			for _, value := range c.data.Hero[index*HeroFeatures : (index+1)*HeroFeatures] {
				writeF32(h, value)
			}
		case "abilities":
			start, end := index*AbilityCount*AbilityFeatures, (index+1)*AbilityCount*AbilityFeatures
			for _, value := range c.data.Abilities[start:end] {
				writeF32(h, value)
			}
		case "entities":
			start, end := index*MaxEntities*EntityFeatures, (index+1)*MaxEntities*EntityFeatures
			for _, value := range c.data.Entities[start:end] {
				writeF32(h, value)
			}
		case "global":
			start, end := index*GlobalFeatures, (index+1)*GlobalFeatures
			for _, value := range c.data.Global[start:end] {
				writeF32(h, value)
			}
		case "entity_mask":
			h.Write(c.data.EntityMask[index*MaxEntities : (index+1)*MaxEntities])
		case "kind_mask":
			h.Write(c.data.KindMask[index*ActionKinds : (index+1)*ActionKinds])
		case "target_mask":
			h.Write(c.data.TargetMask[index*MaxEntities : (index+1)*MaxEntities])
		case "skill_target_mask":
			start, end := index*AbilityCount*MaxEntities, (index+1)*AbilityCount*MaxEntities
			h.Write(c.data.SkillTargetMask[start:end])
		case "teacher_status":
			h.Write([]byte{c.data.TeacherStatus[index]})
		case "teacher_action":
			writeActionHash(h, c.data.TeacherAction[index])
		case "projected_action":
			writeActionHash(h, c.data.ProjectedAction[index])
		case "executed_action":
			writeActionHash(h, c.data.ExecutedAction[index])
		case "executed_valid":
			h.Write([]byte{c.data.ExecutedValid[index]})
		case "rejection_reason":
			h.Write([]byte{c.data.RejectionReason[index]})
		case "rewards":
			writeF32(h, c.data.Rewards[index])
		case "done":
			h.Write([]byte{c.data.Done[tick]})
		case "winner":
			writeI32(h, c.data.Winner[tick])
		case "step":
			writeU32(h, c.data.Steps[tick])
		case "elapsed":
			writeF32(h, c.data.Elapsed[tick])
		case "invalid":
			h.Write([]byte{c.data.Invalid[index]})
		}
	}
	var result Hash
	copy(result[:], h.Sum(nil))
	return result
}

func matchHash(data *Prepared) Hash {
	h := sha256.New()
	h.Write([]byte("AI42-match-v1\x00"))
	h.Write(data.Metadata.RuntimeManifestHash[:])
	h.Write(data.Metadata.SchemaHash[:])
	h.Write(data.Metadata.RewardHash[:])
	h.Write(data.Metadata.TrajectorySchemaHash[:])
	for hero := 0; hero < HeroCount; hero++ {
		h.Write([]byte(data.TrajectoryIDs[hero]))
		h.Write(data.TrajectoryHashes[hero][:])
		if len(data.IncrementalHashes[hero]) > 0 {
			h.Write(data.IncrementalHashes[hero][len(data.IncrementalHashes[hero])-1][:])
		}
	}
	var result Hash
	copy(result[:], h.Sum(nil))
	return result
}

func validatePrepared(data *Prepared) error {
	if err := validateMetadata(&data.Metadata); err != nil {
		return err
	}
	ticks := data.TickCount
	if ticks < 1 {
		return fieldError("tick_count", -1, -1, "must be positive")
	}
	if len(data.Steps) != ticks || len(data.Elapsed) != ticks || len(data.Done) != ticks || len(data.Winner) != ticks {
		return fieldError("arrays", -1, -1, "scalar column lengths do not equal tick_count")
	}
	if data.Done[ticks-1] != 1 {
		return fieldError("done", ticks-1, -1, "must contain the only terminal tick")
	}
	for tick := 0; tick < ticks; tick++ {
		if tick > 0 && data.Steps[tick] != data.Steps[tick-1]+1 {
			return fieldError("step", tick, -1, "is not contiguous")
		}
		if data.Done[tick] > 1 {
			return fieldError("done", tick, -1, "must be zero or one")
		}
		if tick < ticks-1 && data.Done[tick] != 0 {
			return fieldError("done", tick, -1, "terminal tick must be final")
		}
		if !finite32(data.Elapsed[tick]) {
			return fieldError("elapsed", tick, -1, "must be finite")
		}
	}
	if err := validateColumnLengths(data, ticks); err != nil {
		return err
	}
	for tick := 0; tick < ticks; tick++ {
		for slot := 0; slot < HeroCount; slot++ {
			index := tick*HeroCount + slot
			if data.Invalid[index] > 1 {
				return fieldError("invalid", tick, slot, "must be zero or one")
			}
			if !finite32(data.Rewards[index]) {
				return fieldError("rewards", tick, slot, "must be finite")
			}
			heroStart := index * HeroFeatures
			for element, value := range data.Hero[heroStart : heroStart+HeroFeatures] {
				if !finite32(value) {
					return fieldError("hero", tick, slot, "contains non-finite value at index %d", element)
				}
			}
			abilityStart := index * AbilityCount * AbilityFeatures
			for element, value := range data.Abilities[abilityStart : abilityStart+AbilityCount*AbilityFeatures] {
				if !finite32(value) {
					return fieldError("abilities", tick, slot, "contains non-finite value at index %d", element)
				}
			}
			entityStart := index * MaxEntities * EntityFeatures
			for element, value := range data.Entities[entityStart : entityStart+MaxEntities*EntityFeatures] {
				if !finite32(value) {
					return fieldError("entities", tick, slot, "contains non-finite value at index %d", element)
				}
			}
			globalStart := index * GlobalFeatures
			for element, value := range data.Global[globalStart : globalStart+GlobalFeatures] {
				if !finite32(value) {
					return fieldError("global", tick, slot, "contains non-finite value at index %d", element)
				}
			}
			for element, value := range data.EntityMask[index*MaxEntities : (index+1)*MaxEntities] {
				if value > 1 {
					return fieldError("entity_mask", tick, slot, "must be zero or one at index %d", element)
				}
			}
			for element, value := range data.KindMask[index*ActionKinds : (index+1)*ActionKinds] {
				if value > 1 {
					return fieldError("kind_mask", tick, slot, "must be zero or one at index %d", element)
				}
			}
			for element, value := range data.TargetMask[index*MaxEntities : (index+1)*MaxEntities] {
				if value > 1 {
					return fieldError("target_mask", tick, slot, "must be zero or one at index %d", element)
				}
			}
			for element, value := range data.SkillTargetMask[index*AbilityCount*MaxEntities : (index+1)*AbilityCount*MaxEntities] {
				if value > 1 {
					return fieldError("skill_target_mask", tick, slot, "must be zero or one at index %d", element)
				}
			}
			if err := validateAction(data.TeacherAction[index], "teacher_action", tick, slot, true); err != nil {
				return err
			}
			if err := validateAction(data.ProjectedAction[index], "projected_action", tick, slot, true); err != nil {
				return err
			}
			if err := validateAction(data.ExecutedAction[index], "executed_action", tick, slot, true); err != nil {
				return err
			}
			if data.ExecutedValid[index] > 1 {
				return fieldError("executed_valid", tick, slot, "must be zero or one")
			}
			if !validRejectionReason(data.RejectionReason[index]) {
				return fieldError("rejection_reason", tick, slot, "unknown v13 code %d", data.RejectionReason[index])
			}
			status := data.TeacherStatus[index]
			if status > battleserver.AssaultTeacherStatusUnavailable {
				return fieldError("teacher_status", tick, slot, "unknown v13 status %d", status)
			}
			if status == battleserver.AssaultTeacherStatusAction && data.TeacherAction[index].Kind == 0 {
				return fieldError("teacher_action", tick, slot, "action status cannot carry wait")
			}
			if status != battleserver.AssaultTeacherStatusAction && !data.TeacherAction[index].isZero() {
				return fieldError("teacher_action", tick, slot, "control status must carry zero action")
			}
			if data.ExecutedValid[index] == 1 && data.RejectionReason[index] != battleserver.AssaultRejectionReasonNone {
				return fieldError("rejection_reason", tick, slot, "accepted action must have reason none")
			}
			if data.ExecutedValid[index] == 0 && data.RejectionReason[index] == battleserver.AssaultRejectionReasonNone {
				return fieldError("rejection_reason", tick, slot, "rejected action must have a reason")
			}
			if data.ExecutedValid[index] == 0 && !data.ExecutedAction[index].isZero() {
				return fieldError("executed_action", tick, slot, "rejected action must be zero")
			}
			if data.PreviousRecurrentIDs[index] == "" || data.RecurrentBoundaryIDs[index] == "" {
				return fieldError("recurrent_ids", tick, slot, "IDs must be non-empty")
			}
			if data.PreviousRecurrentIDs[index] == data.RecurrentBoundaryIDs[index] {
				return fieldError("recurrent_boundary_id", tick, slot, "must differ from parent")
			}
			if tick > 0 && data.PreviousRecurrentIDs[index] != data.RecurrentBoundaryIDs[(tick-1)*HeroCount+slot] {
				return fieldError("recurrent_parent_id", tick, slot, "does not match prior boundary")
			}
			if !finite32(data.OutcomeReward[index]) || !finite32(data.OutcomeTeamReward[index]) || !finite32(data.OutcomeDamage[index]) {
				return fieldError("outcome", tick, slot, "float fields must be finite")
			}
			if math.Float32bits(data.OutcomeReward[index]) != math.Float32bits(data.Rewards[index]) {
				return fieldError("outcome.reward", tick, slot, "does not match rewards")
			}
			if data.OutcomeTerminal[index] != data.Done[tick] {
				return fieldError("outcome.terminal", tick, slot, "does not match done")
			}
			if data.OutcomeWinnerPresent[index] > 1 || data.OutcomeHeroAlivePresent[index] > 1 || data.OutcomeHeroAlive[index] > 1 {
				return fieldError("outcome", tick, slot, "presence and boolean fields must be zero or one")
			}
		}
	}
	if err := validateLineage(data); err != nil {
		return err
	}
	return nil
}

func validateColumnLengths(data *Prepared, ticks int) error {
	expected := func(name string, got, want int) error {
		if got != want {
			return fieldError(name, -1, -1, "length=%d, want %d", got, want)
		}
		return nil
	}
	columns := []struct {
		name      string
		got, want int
	}{
		{"invalid", len(data.Invalid), ticks * HeroCount}, {"rewards", len(data.Rewards), ticks * HeroCount},
		{"hero", len(data.Hero), ticks * HeroCount * HeroFeatures}, {"abilities", len(data.Abilities), ticks * HeroCount * AbilityCount * AbilityFeatures},
		{"entities", len(data.Entities), ticks * HeroCount * MaxEntities * EntityFeatures}, {"global", len(data.Global), ticks * HeroCount * GlobalFeatures},
		{"entity_mask", len(data.EntityMask), ticks * HeroCount * MaxEntities}, {"kind_mask", len(data.KindMask), ticks * HeroCount * ActionKinds},
		{"target_mask", len(data.TargetMask), ticks * HeroCount * MaxEntities}, {"skill_target_mask", len(data.SkillTargetMask), ticks * HeroCount * AbilityCount * MaxEntities},
		{"teacher_status", len(data.TeacherStatus), ticks * HeroCount}, {"teacher_action", len(data.TeacherAction), ticks * HeroCount},
		{"projected_action", len(data.ProjectedAction), ticks * HeroCount}, {"executed_action", len(data.ExecutedAction), ticks * HeroCount},
		{"executed_valid", len(data.ExecutedValid), ticks * HeroCount}, {"rejection_reason", len(data.RejectionReason), ticks * HeroCount},
		{"previous_recurrent_ids", len(data.PreviousRecurrentIDs), ticks * HeroCount}, {"recurrent_boundary_ids", len(data.RecurrentBoundaryIDs), ticks * HeroCount},
		{"outcome_reward", len(data.OutcomeReward), ticks * HeroCount}, {"outcome_terminal", len(data.OutcomeTerminal), ticks * HeroCount},
		{"outcome_winner", len(data.OutcomeWinner), ticks * HeroCount}, {"outcome_winner_present", len(data.OutcomeWinnerPresent), ticks * HeroCount},
		{"outcome_hero_alive", len(data.OutcomeHeroAlive), ticks * HeroCount}, {"outcome_hero_alive_present", len(data.OutcomeHeroAlivePresent), ticks * HeroCount},
		{"outcome_team_reward", len(data.OutcomeTeamReward), ticks * HeroCount}, {"outcome_damage", len(data.OutcomeDamage), ticks * HeroCount},
		{"outcome_kills", len(data.OutcomeKills), ticks * HeroCount}, {"outcome_deaths", len(data.OutcomeDeaths), ticks * HeroCount}, {"outcome_event", len(data.OutcomeEvent), ticks * HeroCount},
	}
	for _, column := range columns {
		if err := expected(column.name, column.got, column.want); err != nil {
			return err
		}
	}
	for hero := 0; hero < HeroCount; hero++ {
		if len(data.IncrementalHashes[hero]) != ticks {
			return fieldError("incremental_hash", -1, hero, "length=%d, want %d", len(data.IncrementalHashes[hero]), ticks)
		}
	}
	return nil
}

func validateLineage(data *Prepared) error {
	for hero := 0; hero < HeroCount; hero++ {
		roots := make(map[string]string, data.TickCount)
		cancelled := make(map[string]struct{}, data.TickCount)
		for tick := 0; tick < data.TickCount; tick++ {
			index := tick*HeroCount + hero
			parent, boundary := data.PreviousRecurrentIDs[index], data.RecurrentBoundaryIDs[index]
			if _, duplicate := roots[boundary]; duplicate {
				return fieldError("recurrent_boundary_id", tick, hero, "duplicate boundary %q", boundary)
			}
			status := data.TeacherStatus[index]
			switch status {
			case battleserver.AssaultTeacherStatusHold:
				if tick == 0 || data.TeacherStatus[(tick-1)*HeroCount+hero] == battleserver.AssaultTeacherStatusWait || data.TeacherStatus[(tick-1)*HeroCount+hero] == battleserver.AssaultTeacherStatusCancel {
					return fieldError("teacher_status", tick, hero, "HOLD does not reference a holdable lineage")
				}
				root, ok := roots[parent]
				if !ok {
					return fieldError("recurrent_parent_id", tick, hero, "HOLD parent lineage is unknown")
				}
				roots[boundary] = root
			case battleserver.AssaultTeacherStatusCancel:
				if tick == 0 || data.TeacherStatus[(tick-1)*HeroCount+hero] == battleserver.AssaultTeacherStatusWait || data.TeacherStatus[(tick-1)*HeroCount+hero] == battleserver.AssaultTeacherStatusCancel {
					return fieldError("teacher_status", tick, hero, "CANCEL does not reference a cancellable lineage")
				}
				root, ok := roots[parent]
				if !ok {
					return fieldError("recurrent_parent_id", tick, hero, "CANCEL parent lineage is unknown")
				}
				if _, already := cancelled[root]; already {
					return fieldError("teacher_status", tick, hero, "CANCEL references an already cancelled lineage")
				}
				cancelled[root] = struct{}{}
				roots[boundary] = boundary
			default:
				roots[boundary] = boundary
			}
		}
	}
	return nil
}

// The following small writers intentionally target hash.Hash through the
// standard io.Writer interface.  Keeping all scalar writes little-endian makes
// trajectory hashes and shards identical on MSVC and Linux.
func writeByte(w interface{ Write([]byte) (int, error) }, value uint8) { _, _ = w.Write([]byte{value}) }
func writeU32(w interface{ Write([]byte) (int, error) }, value uint32) {
	var raw [4]byte
	binary.LittleEndian.PutUint32(raw[:], value)
	_, _ = w.Write(raw[:])
}
func writeI32(w interface{ Write([]byte) (int, error) }, value int32) { writeU32(w, uint32(value)) }
func writeF32(w interface{ Write([]byte) (int, error) }, value float32) {
	writeU32(w, math.Float32bits(value))
}
func writeString(w interface{ Write([]byte) (int, error) }, value string) {
	writeU32(w, uint32(len(value)))
	_, _ = w.Write([]byte(value))
}
func writeActionHash(w interface{ Write([]byte) (int, error) }, value Action) {
	var raw [5]byte
	value.wireBytes(raw[:])
	_, _ = w.Write(raw[:])
}
func writeOutcome(w interface{ Write([]byte) (int, error) }, value Outcome) {
	writeF32(w, value.Reward)
	writeByte(w, boolByte(value.Terminal))
	writeByte(w, boolByte(value.WinnerPresent))
	writeI32(w, value.Winner)
	writeByte(w, boolByte(value.HeroAlivePresent))
	writeByte(w, boolByte(value.HeroAlive))
	writeF32(w, value.TeamReward)
	writeF32(w, value.Damage)
	writeU32(w, value.Kills)
	writeU32(w, value.Deaths)
	writeString(w, value.Event)
}
func boolByte(value bool) uint8 {
	if value {
		return 1
	}
	return 0
}

func writeSlotObservation(w interface{ Write([]byte) (int, error) }, observation battleserver.AssaultObservationV1) {
	for _, value := range observation.Hero {
		writeF32(w, value)
	}
	for _, ability := range observation.Abilities {
		for _, value := range ability {
			writeF32(w, value)
		}
	}
	for _, entity := range observation.Entities {
		for _, value := range entity {
			writeF32(w, value)
		}
	}
	for _, value := range observation.Global {
		writeF32(w, value)
	}
	_, _ = w.Write(observation.EntityMask[:])
	_, _ = w.Write(observation.ActionMask.Kinds[:])
	_, _ = w.Write(observation.ActionMask.Targets[:])
	for _, target := range observation.ActionMask.SkillTarget {
		_, _ = w.Write(target[:])
	}
}
