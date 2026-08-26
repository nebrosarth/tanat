// Package assaultproto implements the versioned binary stdin/stdout boundary
// between Go rollout environments and Python AI-40 training workers.
package assaultproto

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
	"sync"

	"tanatserver/internal/battleserver"
)

const (
	// Version 4 adds vector commands so one Go process can own every rollout
	// environment and exchange one request/response batch per simulation tick.
	Version uint16 = 4
	// VersionAI41 appends four structured ability records to every hero observation.
	// Version 4 remains supported so active AI-40 campaigns can keep using the same binary.
	VersionAI41 uint16 = 5
	// VersionAI41WrongLane keeps the v5 wire layout and adds the randomized
	// lane-assignment observation/reward contract used by AI-41-v2.
	VersionAI41WrongLane uint16 = 6
	// VersionAI41Evaluation uses the v3 model contract with assignment features
	// zeroed and no lane penalty, matching training-only randomization semantics.
	VersionAI41Evaluation uint16 = 7
	// VersionAI41Navigation replaces the polar 16x3 position action with a 9x9
	// local grid plus 15 semantic navigation choices (local + 14 global anchors).
	VersionAI41Navigation uint16 = 8
	// VersionAI41NavigationEvaluation keeps the navigation action contract but
	// disables randomized lane assignments and their reward, as v7 does for v6.
	VersionAI41NavigationEvaluation uint16 = 9
	// Versions 10-11 retain the navigation wire layout and select the strategic
	// reward contract used by long-horizon historical self-play.
	VersionAI41Strategic           uint16 = 10
	VersionAI41StrategicEvaluation uint16 = 11
	// VersionAI41Teacher retains the strategic navigation contract and appends
	// resolved AI-30 teacher actions for offline behavior cloning only.
	VersionAI41Teacher uint16 = 12
	// VersionAI42 is the opt-in AI-42 append-only v12 observation contract.
	VersionAI42         uint16 = 13
	VersionAI42Reserved        = VersionAI42 // compatibility alias
	// VersionAI42Evaluation keeps v13 actor observations/rewards but removes
	// teacher-only response fields and adds one explicit control byte to every
	// submitted action. It is evaluation-only; v13 dataset lineage is unchanged.
	VersionAI42Evaluation uint16 = 14
	// VersionAI42DAgger adds a one-byte expert-intervention mask per actor and
	// returns both teacher/execution telemetry and authoritative active order.
	VersionAI42DAgger         uint16 = 15
	CommandReset              uint16 = 1
	CommandStep               uint16 = 2
	CommandClose              uint16 = 3
	CommandVectorReset        uint16 = 4
	CommandVectorStep         uint16 = 5
	CommandVectorResetIndices uint16 = 6
	ResponseResult            uint16 = 100
	ResponseVectorResult      uint16 = 101
	ResponseError             uint16 = 255
	maxFrameSize                     = 64 << 20
)

func versionHasAbilities(version uint16) bool {
	return version == VersionAI41 || version == VersionAI41WrongLane ||
		version == VersionAI41Evaluation || version == VersionAI41Navigation ||
		version == VersionAI41NavigationEvaluation || version == VersionAI41Strategic ||
		version == VersionAI41StrategicEvaluation || version == VersionAI41Teacher ||
		version == VersionAI42 || version == VersionAI42Evaluation || version == VersionAI42DAgger
}

func versionHasNavigation(version uint16) bool {
	return version == VersionAI41Navigation || version == VersionAI41NavigationEvaluation ||
		version == VersionAI41Strategic || version == VersionAI41StrategicEvaluation ||
		version == VersionAI41Teacher || version == VersionAI42 ||
		version == VersionAI42Evaluation || version == VersionAI42DAgger
}

func versionHasTeacher(version uint16) bool {
	return version == VersionAI41Teacher || version == VersionAI42
}

func versionHasAI42(version uint16) bool {
	return version == VersionAI42 || version == VersionAI42DAgger
}

func layoutHasAbilities(version uint16) bool {
	return versionHasAbilities(version)
}

func layoutHasTeacher(version uint16) bool {
	return versionHasTeacher(version)
}

func supportedVersion(version uint16) bool {
	return version == Version || versionHasAbilities(version)
}

// AI42SchemaHash is derived below from the v13 scalar and vector field tables;
// it is not a separately maintained/precomputed contract value. The reward
// hash intentionally remains the existing strategic V5 hash.
var AI42SchemaHash = sha256.Sum256([]byte(ai42SchemaText()))
var AI42EvaluationSchemaHash = sha256.Sum256([]byte(ai42EvaluationSchemaText()))
var AI42DAggerSchemaHash = sha256.Sum256([]byte(ai42DAggerSchemaText()))
var AI42RewardHash = battleserver.AssaultRewardHashV5

var magic = [4]byte{'T', 'A', 'N', 'T'}

type Request struct {
	Version             uint16
	Command             uint16
	Reset               battleserver.AssaultResetV1
	Actions             [battleserver.AssaultHeroCount]battleserver.HeroActionV1
	Controls            [battleserver.AssaultHeroCount]battleserver.AssaultControlV1
	VectorResets        []battleserver.AssaultResetV1
	VectorIndices       []uint32
	VectorActions       [][battleserver.AssaultHeroCount]battleserver.HeroActionV1
	VectorControls      [][battleserver.AssaultHeroCount]battleserver.AssaultControlV1
	Interventions       [battleserver.AssaultHeroCount]uint8
	VectorInterventions [][battleserver.AssaultHeroCount]uint8
}

const resetPayloadSize = 8 + 4 + battleserver.AssaultHeroCount*4 + battleserver.AssaultHeroCount
const actionPayloadSize = battleserver.AssaultHeroCount * 5
const controlledActionPayloadSize = battleserver.AssaultHeroCount * 6
const daggerActionPayloadSize = battleserver.AssaultHeroCount * 7

func decodeReset(p []byte) (battleserver.AssaultResetV1, error) {
	var reset battleserver.AssaultResetV1
	if len(p) != resetPayloadSize {
		if len(p) < resetPayloadSize {
			return reset, fmt.Errorf("assault protocol: field=RESET.payload offset=%d: truncated, got %d bytes, want %d",
				8+len(p), len(p), resetPayloadSize)
		}
		return reset, fmt.Errorf("assault protocol: field=RESET.trailing offset=%d: got %d extra bytes",
			8+resetPayloadSize, len(p)-resetPayloadSize)
	}
	reset.Seed = int64(binary.LittleEndian.Uint64(p[:8]))
	reset.MaxSteps = binary.LittleEndian.Uint32(p[8:12])
	off := 12
	for i := range reset.Roster {
		reset.Roster[i] = int32(binary.LittleEndian.Uint32(p[off : off+4]))
		off += 4
	}
	for i := range reset.Controllers {
		reset.Controllers[i] = battleserver.AssaultControllerV1(p[off])
		off++
	}
	return reset, nil
}

func decodeActions(p []byte) ([battleserver.AssaultHeroCount]battleserver.HeroActionV1, error) {
	var actions [battleserver.AssaultHeroCount]battleserver.HeroActionV1
	if len(p) != actionPayloadSize {
		if len(p) < actionPayloadSize {
			return actions, fmt.Errorf("assault protocol: field=STEP.actions offset=%d: truncated, got %d bytes, want %d",
				8+len(p), len(p), actionPayloadSize)
		}
		return actions, fmt.Errorf("assault protocol: field=STEP.trailing offset=%d: got %d extra bytes",
			8+actionPayloadSize, len(p)-actionPayloadSize)
	}
	for i := range actions {
		off := i * 5
		actions[i] = battleserver.HeroActionV1{
			Kind:      battleserver.AssaultActionKindV1(p[off]),
			Target:    binary.LittleEndian.Uint16(p[off+1 : off+3]),
			Direction: p[off+3],
			Distance:  p[off+4],
		}
	}
	return actions, nil
}

func decodeControlledActions(p []byte) (
	[battleserver.AssaultHeroCount]battleserver.HeroActionV1,
	[battleserver.AssaultHeroCount]battleserver.AssaultControlV1,
	error,
) {
	var actions [battleserver.AssaultHeroCount]battleserver.HeroActionV1
	var controls [battleserver.AssaultHeroCount]battleserver.AssaultControlV1
	if len(p) != controlledActionPayloadSize {
		return actions, controls, fmt.Errorf(
			"assault protocol: field=STEP.controlled_actions offset=%d: got %d bytes, want %d",
			8+len(p), len(p), controlledActionPayloadSize,
		)
	}
	for i := range actions {
		off := i * 6
		controls[i] = battleserver.AssaultControlV1(p[off])
		actions[i] = battleserver.HeroActionV1{
			Kind:      battleserver.AssaultActionKindV1(p[off+1]),
			Target:    binary.LittleEndian.Uint16(p[off+2 : off+4]),
			Direction: p[off+4], Distance: p[off+5],
		}
	}
	return actions, controls, nil
}

func decodeDaggerActions(p []byte) (
	[battleserver.AssaultHeroCount]battleserver.HeroActionV1,
	[battleserver.AssaultHeroCount]battleserver.AssaultControlV1,
	[battleserver.AssaultHeroCount]uint8,
	error,
) {
	var actions [battleserver.AssaultHeroCount]battleserver.HeroActionV1
	var controls [battleserver.AssaultHeroCount]battleserver.AssaultControlV1
	var interventions [battleserver.AssaultHeroCount]uint8
	if len(p) != daggerActionPayloadSize {
		return actions, controls, interventions, fmt.Errorf(
			"assault protocol: field=STEP.dagger_actions offset=%d: got %d bytes, want %d",
			8+len(p), len(p), daggerActionPayloadSize,
		)
	}
	for i := range actions {
		off := i * 7
		controls[i] = battleserver.AssaultControlV1(p[off])
		actions[i] = battleserver.HeroActionV1{
			Kind:      battleserver.AssaultActionKindV1(p[off+1]),
			Target:    binary.LittleEndian.Uint16(p[off+2 : off+4]),
			Direction: p[off+4], Distance: p[off+5],
		}
		interventions[i] = p[off+6]
		if interventions[i] > 1 {
			return actions, controls, interventions, fmt.Errorf(
				"assault protocol: field=STEP.intervention[%d] offset=%d: invalid value %d",
				i, 8+off+6, interventions[i],
			)
		}
	}
	return actions, controls, interventions, nil
}

func ReadRequest(r io.Reader) (Request, error) {
	body, err := readFrame(r)
	if err != nil {
		return Request{}, err
	}
	if len(body) < 8 || !bytes.Equal(body[:4], magic[:]) {
		if len(body) < 4 {
			return Request{}, fmt.Errorf("assault protocol: field=frame.magic offset=%d: truncated, got %d bytes",
				len(body), len(body))
		}
		return Request{}, errors.New("assault protocol: field=frame.magic offset=0: bad magic")
	}
	version := binary.LittleEndian.Uint16(body[4:6])
	if !supportedVersion(version) {
		return Request{}, fmt.Errorf("assault protocol: field=frame.version offset=4: unsupported version %d", version)
	}
	command := binary.LittleEndian.Uint16(body[6:8])
	p := body[8:]
	request := Request{Version: version, Command: command}
	switch command {
	case CommandReset:
		request.Reset, err = decodeReset(p)
		if err != nil {
			return Request{}, err
		}
	case CommandStep:
		if version == VersionAI42DAgger {
			request.Actions, request.Controls, request.Interventions, err = decodeDaggerActions(p)
		} else if version == VersionAI42Evaluation {
			request.Actions, request.Controls, err = decodeControlledActions(p)
		} else {
			request.Actions, err = decodeActions(p)
		}
		if err != nil {
			return Request{}, err
		}
	case CommandVectorReset:
		if len(p) < 4 {
			return Request{}, fmt.Errorf("assault protocol: field=VECTOR_RESET.count offset=8: truncated, got %d bytes",
				len(p))
		}
		count := int(binary.LittleEndian.Uint32(p[:4]))
		p = p[4:]
		want := count * resetPayloadSize
		if count < 1 {
			return Request{}, fmt.Errorf("assault protocol: field=VECTOR_RESET.count offset=8: invalid count %d", count)
		}
		if len(p) < want {
			return Request{}, fmt.Errorf("assault protocol: field=VECTOR_RESET[%d] offset=%d: truncated, got %d bytes, want %d",
				len(p)/resetPayloadSize, 12+len(p), len(p), want)
		}
		if len(p) > want {
			return Request{}, fmt.Errorf("assault protocol: field=VECTOR_RESET.trailing offset=%d: got %d extra bytes",
				12+want, len(p)-want)
		}
		request.VectorResets = make([]battleserver.AssaultResetV1, count)
		for i := range request.VectorResets {
			request.VectorResets[i], err = decodeReset(p[i*resetPayloadSize : (i+1)*resetPayloadSize])
			if err != nil {
				return Request{}, err
			}
		}
	case CommandVectorStep:
		if len(p) < 4 {
			return Request{}, fmt.Errorf("assault protocol: field=VECTOR_STEP.count offset=8: truncated, got %d bytes",
				len(p))
		}
		count := int(binary.LittleEndian.Uint32(p[:4]))
		p = p[4:]
		itemSize := actionPayloadSize
		if version == VersionAI42DAgger {
			itemSize = daggerActionPayloadSize
		} else if version == VersionAI42Evaluation {
			itemSize = controlledActionPayloadSize
		}
		want := count * itemSize
		if count < 1 {
			return Request{}, fmt.Errorf("assault protocol: field=VECTOR_STEP.count offset=8: invalid count %d", count)
		}
		if len(p) < want {
			return Request{}, fmt.Errorf("assault protocol: field=VECTOR_STEP[%d] offset=%d: truncated, got %d bytes, want %d",
				len(p)/itemSize, 12+len(p), len(p), want)
		}
		if len(p) > want {
			return Request{}, fmt.Errorf("assault protocol: field=VECTOR_STEP.trailing offset=%d: got %d extra bytes",
				12+want, len(p)-want)
		}
		request.VectorActions = make([][battleserver.AssaultHeroCount]battleserver.HeroActionV1, count)
		if version == VersionAI42Evaluation || version == VersionAI42DAgger {
			request.VectorControls = make([][battleserver.AssaultHeroCount]battleserver.AssaultControlV1, count)
		}
		if version == VersionAI42DAgger {
			request.VectorInterventions = make([][battleserver.AssaultHeroCount]uint8, count)
		}
		for i := range request.VectorActions {
			if version == VersionAI42DAgger {
				request.VectorActions[i], request.VectorControls[i], request.VectorInterventions[i], err = decodeDaggerActions(
					p[i*itemSize : (i+1)*itemSize],
				)
			} else if version == VersionAI42Evaluation {
				request.VectorActions[i], request.VectorControls[i], err = decodeControlledActions(
					p[i*itemSize : (i+1)*itemSize],
				)
			} else {
				request.VectorActions[i], err = decodeActions(p[i*itemSize : (i+1)*itemSize])
			}
			if err != nil {
				return Request{}, err
			}
		}
	case CommandVectorResetIndices:
		if len(p) < 4 {
			return Request{}, fmt.Errorf("assault protocol: field=VECTOR_RESET_INDICES.count offset=8: truncated, got %d bytes",
				len(p))
		}
		count := int(binary.LittleEndian.Uint32(p[:4]))
		p = p[4:]
		const indexedResetSize = 4 + resetPayloadSize
		want := count * indexedResetSize
		if count < 1 {
			return Request{}, fmt.Errorf("assault protocol: field=VECTOR_RESET_INDICES.count offset=8: invalid count %d", count)
		}
		if len(p) < want {
			return Request{}, fmt.Errorf("assault protocol: field=VECTOR_RESET_INDICES[%d] offset=%d: truncated, got %d bytes, want %d",
				len(p)/indexedResetSize, 12+len(p), len(p), want)
		}
		if len(p) > want {
			return Request{}, fmt.Errorf("assault protocol: field=VECTOR_RESET_INDICES.trailing offset=%d: got %d extra bytes",
				12+want, len(p)-want)
		}
		request.VectorIndices = make([]uint32, count)
		request.VectorResets = make([]battleserver.AssaultResetV1, count)
		for i := range request.VectorResets {
			off := i * indexedResetSize
			request.VectorIndices[i] = binary.LittleEndian.Uint32(p[off : off+4])
			request.VectorResets[i], err = decodeReset(p[off+4 : off+indexedResetSize])
			if err != nil {
				return Request{}, err
			}
		}
	case CommandClose:
		if len(p) != 0 {
			return Request{}, fmt.Errorf("assault protocol: field=CLOSE.trailing offset=8: got %d extra bytes", len(p))
		}
	default:
		return Request{}, fmt.Errorf("assault protocol: field=frame.command offset=6: unknown command %d", command)
	}
	return request, nil
}

func WriteResult(w io.Writer, result battleserver.StepResultV1) error {
	return NewResultEncoder().Write(w, &result)
}

// ReadResultVersion decodes one scalar result frame for a fixed protocol
// version. It is the inverse of ResultEncoder.WriteVersion and lets native
// capture sidecars consume the exact bytes already returned to a policy
// process instead of rebuilding large observation arrays through Python.
func ReadResultVersion(r io.Reader, expectedVersion uint16) (battleserver.StepResultV1, error) {
	var result battleserver.StepResultV1
	if !supportedVersion(expectedVersion) {
		return result, fmt.Errorf("assault protocol: unsupported expected result version %d", expectedVersion)
	}
	body, err := readFrame(r)
	if err != nil {
		return result, err
	}
	if len(body) < 8 || !bytes.Equal(body[:4], magic[:]) {
		return result, errors.New("assault protocol: field=frame.magic offset=0: bad magic")
	}
	version := binary.LittleEndian.Uint16(body[4:6])
	if version != expectedVersion {
		return result, fmt.Errorf(
			"assault protocol: field=frame.version offset=4: got %d, want %d",
			version, expectedVersion,
		)
	}
	response := binary.LittleEndian.Uint16(body[6:8])
	if response == ResponseError {
		return result, fmt.Errorf("assault protocol: remote error: %s", string(body[8:]))
	}
	if response != ResponseResult {
		return result, fmt.Errorf("assault protocol: field=frame.response offset=6: got %d, want %d", response, ResponseResult)
	}
	layout := newResultFrameLayout(version)
	if err := resultSizeError(layout, len(body)); err != nil {
		return result, err
	}
	copy(result.SchemaHash[:], body[8:40])
	copy(result.RewardHash[:], body[40:72])
	if _, _, err := resultHeaderHashes(version, &result); err != nil {
		return result, err
	}
	if version != Version {
		expectedSchema, expectedReward, _ := resultHeaderHashes(version, &result)
		if result.SchemaHash != expectedSchema {
			return result, fmt.Errorf("assault protocol: field=result.schema_hash offset=8: mismatch")
		}
		if result.RewardHash != expectedReward {
			return result, fmt.Errorf("assault protocol: field=result.reward_hash offset=40: mismatch")
		}
	}
	if err := decodeResultRecord(body[resultHeaderSize:], version, &result); err != nil {
		return battleserver.StepResultV1{}, err
	}
	return result, nil
}

func decodeResultRecord(body []byte, version uint16, result *battleserver.StepResultV1) error {
	layout := newResultFrameLayout(version)
	if len(body) != layout.recordSize {
		return resultSizeError(layout, len(body)+resultHeaderSize)
	}
	off := 0
	takeBytes := func(destination []byte) {
		copy(destination, body[off:off+len(destination)])
		off += len(destination)
	}
	takeU32 := func() uint32 {
		value := binary.LittleEndian.Uint32(body[off : off+4])
		off += 4
		return value
	}
	takeF32 := func() float32 { return math.Float32frombits(takeU32()) }
	takeF32s := func(destination []float32) {
		for index := range destination {
			destination[index] = takeF32()
		}
	}
	takeAction := func() battleserver.HeroActionV1 {
		value := battleserver.HeroActionV1{
			Kind:      battleserver.AssaultActionKindV1(body[off]),
			Target:    binary.LittleEndian.Uint16(body[off+1 : off+3]),
			Direction: body[off+3], Distance: body[off+4],
		}
		off += 5
		return value
	}

	result.Step = takeU32()
	result.Elapsed = takeF32()
	result.Done = body[off] != 0
	off += 4
	result.Winner = int32(takeU32())
	takeBytes(result.Invalid[:])
	takeF32s(result.Reward[:])
	for hero := range result.Observations {
		observation := &result.Observations[hero]
		takeF32s(observation.Hero[:])
		if layoutHasAbilities(version) {
			for ability := range observation.Abilities {
				takeF32s(observation.Abilities[ability][:])
			}
		}
		for entity := range observation.Entities {
			takeF32s(observation.Entities[entity][:])
		}
		takeF32s(observation.Global[:])
		takeBytes(observation.EntityMask[:])
		takeBytes(observation.ActionMask.Kinds[:])
		takeBytes(observation.ActionMask.Targets[:])
		for skill := range observation.ActionMask.SkillTarget {
			takeBytes(observation.ActionMask.SkillTarget[skill][:])
		}
	}
	if version == VersionAI41Teacher {
		for slot := range result.TeacherActions {
			result.TeacherActions[slot] = takeAction()
		}
		takeBytes(result.TeacherValid[:])
	} else if versionHasAI42(version) {
		for slot := range result.TeacherIntent {
			result.TeacherIntent[slot] = takeAction()
		}
		takeBytes(result.TeacherStatus[:])
		for slot := range result.ExecutedActions {
			result.ExecutedActions[slot] = takeAction()
		}
		takeBytes(result.ExecutedValid[:])
		takeBytes(result.RejectionReason[:])
		if version == VersionAI42DAgger {
			takeBytes(result.ActiveOrder[:])
		}
	} else if version == VersionAI42Evaluation {
		takeBytes(result.ActiveOrder[:])
	}
	if off != len(body) {
		return fmt.Errorf("assault protocol: field=result.trailing offset=%d: got %d bytes", resultHeaderSize+off, len(body)-off)
	}
	return nil
}

const resultBodySize = 138 + battleserver.AssaultHeroCount*(battleserver.AssaultHeroFeatureSize*4+
	battleserver.AssaultMaxEntities*battleserver.AssaultEntityFeatures*4+
	battleserver.AssaultGlobalFeatures*4+
	battleserver.AssaultMaxEntities+
	battleserver.AssaultActionKinds+
	battleserver.AssaultMaxEntities+
	4*battleserver.AssaultMaxEntities)
const resultBodySizeV2 = resultBodySize + battleserver.AssaultHeroCount*4*battleserver.AssaultAbilityFeatures*4
const resultBodySizeV3 = resultBodySizeV2 + battleserver.AssaultHeroCount*5 + battleserver.AssaultHeroCount
const resultHeaderSize = 72
const resultRecordSize = resultBodySize - resultHeaderSize
const resultRecordSizeV2 = resultBodySizeV2 - resultHeaderSize
const resultRecordSizeV3 = resultBodySizeV3 - resultHeaderSize
const resultBodySizeV13 = resultBodySizeV3 + battleserver.AssaultHeroCount*(5+1+1)
const resultRecordSizeV13 = resultBodySizeV13 - resultHeaderSize
const resultBodySizeV14 = resultBodySizeV2 + battleserver.AssaultHeroCount
const vectorResultHeaderSize = resultHeaderSize + 4

// The scalar result body is the authoritative frame layout. Every offset is
// relative to the body after the uint32 length prefix; all integer and float
// values are little-endian. The vector result has the same logical fields in
// structure-of-arrays order, after its count at offset 72.
//
//	0..4   magic
//	4..6   version
//	6..8   response
//	8..40  schema hash
//	40..72 reward hash
//	72..88 step, elapsed, done+padding, winner
//	88..98 invalid[10]
//	98..138 reward[10] (float32)
//	138..  observation[10] in hero, abilities (v5+), entities, global,
//	        entity mask, kind mask, target mask, skill-target mask order
//	...    teacher_action[10] (v12)
//	...    teacher_valid[10] (v12)
//	...    teacher_intent[10], teacher_status[10], executed_action[10],
//	       executed_valid[10], rejection_reason[10] (v13)
type resultLayoutField struct {
	name   string
	offset int
	size   int
}

type resultFrameLayout struct {
	bodySize, recordSize                                              int
	observationOffset, observationSize                                int
	teacherActionsOffset, teacherValidOffset                          int
	teacherIntentOffset, teacherStatusOffset                          int
	executedActionsOffset, executedValidOffset, rejectionReasonOffset int
	activeOrderOffset                                                 int
	fields                                                            []resultLayoutField
}

func newResultFrameLayout(version uint16) resultFrameLayout {
	off := 0
	fields := make([]resultLayoutField, 0, 64)
	add := func(name string, size int) int {
		start := off
		fields = append(fields, resultLayoutField{name: name, offset: start, size: size})
		off += size
		return start
	}
	add("frame.magic", 4)
	add("frame.version", 2)
	add("frame.response", 2)
	add("result.schema_hash", 32)
	add("result.reward_hash", 32)
	add("result.step", 4)
	add("result.elapsed", 4)
	add("result.done_padding", 4)
	add("result.winner", 4)
	add("result.invalid", battleserver.AssaultHeroCount)
	add("result.reward", battleserver.AssaultHeroCount*4)
	observationOffset := off
	observationSize := battleserver.AssaultHeroFeatureSize*4 +
		battleserver.AssaultMaxEntities*battleserver.AssaultEntityFeatures*4 +
		battleserver.AssaultGlobalFeatures*4 + battleserver.AssaultMaxEntities +
		battleserver.AssaultActionKinds + battleserver.AssaultMaxEntities +
		4*battleserver.AssaultMaxEntities
	if layoutHasAbilities(version) {
		observationSize += 4 * battleserver.AssaultAbilityFeatures * 4
	}
	for hero := 0; hero < battleserver.AssaultHeroCount; hero++ {
		prefix := fmt.Sprintf("result.observation[%d].", hero)
		add(prefix+"hero", battleserver.AssaultHeroFeatureSize*4)
		if layoutHasAbilities(version) {
			add(prefix+"abilities", 4*battleserver.AssaultAbilityFeatures*4)
		}
		add(prefix+"entities", battleserver.AssaultMaxEntities*battleserver.AssaultEntityFeatures*4)
		add(prefix+"global", battleserver.AssaultGlobalFeatures*4)
		add(prefix+"entity_mask", battleserver.AssaultMaxEntities)
		add(prefix+"kind_mask", battleserver.AssaultActionKinds)
		add(prefix+"target_mask", battleserver.AssaultMaxEntities)
		add(prefix+"skill_target_mask", 4*battleserver.AssaultMaxEntities)
	}
	teacherActionsOffset, teacherValidOffset := -1, -1
	teacherIntentOffset, teacherStatusOffset := -1, -1
	executedActionsOffset, executedValidOffset, rejectionReasonOffset := -1, -1, -1
	activeOrderOffset := -1
	if version == VersionAI41Teacher {
		teacherActionsOffset = add("result.teacher_actions", battleserver.AssaultHeroCount*5)
		teacherValidOffset = add("result.teacher_valid", battleserver.AssaultHeroCount)
	} else if versionHasAI42(version) {
		teacherIntentOffset = add("result.teacher_intent", battleserver.AssaultHeroCount*5)
		teacherStatusOffset = add("result.teacher_status", battleserver.AssaultHeroCount)
		executedActionsOffset = add("result.executed_actions", battleserver.AssaultHeroCount*5)
		executedValidOffset = add("result.executed_valid", battleserver.AssaultHeroCount)
		rejectionReasonOffset = add("result.rejection_reason", battleserver.AssaultHeroCount)
		if version == VersionAI42DAgger {
			activeOrderOffset = add("result.active_order", battleserver.AssaultHeroCount)
		}
	} else if version == VersionAI42Evaluation {
		activeOrderOffset = add("result.active_order", battleserver.AssaultHeroCount)
	}
	return resultFrameLayout{
		bodySize: off, recordSize: off - resultHeaderSize,
		observationOffset: observationOffset, observationSize: observationSize,
		teacherActionsOffset: teacherActionsOffset, teacherValidOffset: teacherValidOffset,
		teacherIntentOffset: teacherIntentOffset, teacherStatusOffset: teacherStatusOffset,
		executedActionsOffset: executedActionsOffset, executedValidOffset: executedValidOffset,
		rejectionReasonOffset: rejectionReasonOffset, activeOrderOffset: activeOrderOffset,
		fields: fields,
	}
}

func resultSizeError(layout resultFrameLayout, got int) error {
	if got == layout.bodySize {
		return nil
	}
	if got > layout.bodySize {
		return fmt.Errorf("assault protocol: field=result.trailing offset=%d: got %d bytes, want %d",
			layout.bodySize, got, layout.bodySize)
	}
	for _, field := range layout.fields {
		if got < field.offset+field.size {
			return fmt.Errorf("assault protocol: field=%s offset=%d: truncated, got %d bytes, need %d",
				field.name, got, got, field.offset+field.size)
		}
	}
	return fmt.Errorf("assault protocol: field=result.body offset=%d: got %d bytes, want %d",
		got, got, layout.bodySize)
}

func resultHeaderHashes(version uint16, result *battleserver.StepResultV1) ([32]byte, [32]byte, error) {
	if result == nil {
		return [32]byte{}, [32]byte{}, errors.New("assault protocol: field=result offset=0: nil result")
	}
	schema, reward := result.SchemaHash, result.RewardHash
	switch version {
	case Version:
		// Version 4 is the established AI-40 contract. Its hashes are carried by
		// the environment result because the original protocol predates the
		// version-specific AI-41 table.
	case VersionAI41:
		schema, reward = battleserver.AssaultSchemaHashV2, battleserver.AssaultRewardHashV2
	case VersionAI41WrongLane, VersionAI41Evaluation:
		schema, reward = battleserver.AssaultSchemaHashV3, battleserver.AssaultRewardHashV3
	case VersionAI41Navigation, VersionAI41NavigationEvaluation:
		schema, reward = battleserver.AssaultSchemaHashV4, battleserver.AssaultRewardHashV3
	case VersionAI41Strategic, VersionAI41StrategicEvaluation, VersionAI41Teacher:
		schema, reward = battleserver.AssaultSchemaHashV4, battleserver.AssaultRewardHashV5
	case VersionAI42:
		return AI42SchemaHash, AI42RewardHash, nil
	case VersionAI42Evaluation:
		return AI42EvaluationSchemaHash, AI42RewardHash, nil
	case VersionAI42DAgger:
		return AI42DAggerSchemaHash, AI42RewardHash, nil
	default:
		return [32]byte{}, [32]byte{}, fmt.Errorf("assault protocol: field=frame.version offset=4: unsupported version %d", version)
	}
	if version != Version && result.RewardHash != reward {
		return [32]byte{}, [32]byte{}, fmt.Errorf(
			"assault protocol: field=result.reward_hash offset=40: got %x, want %x",
			result.RewardHash[:4], reward[:4])
	}
	return schema, reward, nil
}

type vectorResultLayout struct {
	steps, elapsed, done, winners, invalid, rewards              int
	hero, abilities, entities, global, entityMask, kindMask      int
	targetMask, skillTargetMask, teacherActions, teacherValid    int
	teacherIntent, teacherStatus, executedActions, executedValid int
	rejectionReason, activeOrder, size                           int
	fields                                                       []resultLayoutField
}

func newVectorResultLayout(count int) vectorResultLayout {
	return newVectorResultLayoutVersion(count, Version)
}

func newVectorResultLayoutVersion(count int, version uint16) vectorResultLayout {
	off := vectorResultHeaderSize
	// Keep the complete frame header in the canonical field table just like
	// the scalar layout.  Python derives the v13 schema hash from the same full
	// table; omitting these five fields made byte-compatible frames advertise
	// different hashes across the Go/Python boundary.
	fields := []resultLayoutField{
		{name: "frame.magic", offset: 0, size: 4},
		{name: "frame.version", offset: 4, size: 2},
		{name: "frame.response", offset: 6, size: 2},
		{name: "result.schema_hash", offset: 8, size: 32},
		{name: "result.reward_hash", offset: 40, size: 32},
		{name: "result.count", offset: 72, size: 4},
	}
	take := func(name string, bytes int) int {
		start := off
		fields = append(fields, resultLayoutField{name: name, offset: start, size: bytes})
		off += bytes
		return start
	}
	actors := count * battleserver.AssaultHeroCount
	layout := vectorResultLayout{
		steps: take("result.steps", count*4), elapsed: take("result.elapsed", count*4),
		done: take("result.done", count), winners: take("result.winner", count*4),
		invalid: take("result.invalid", actors), rewards: take("result.reward", actors*4),
		hero: take("result.hero", actors*battleserver.AssaultHeroFeatureSize*4),
	}
	if layoutHasAbilities(version) {
		layout.abilities = take("result.abilities", actors*4*battleserver.AssaultAbilityFeatures*4)
	}
	layout.entities = take("result.entities", actors*battleserver.AssaultMaxEntities*battleserver.AssaultEntityFeatures*4)
	layout.global = take("result.global", actors*battleserver.AssaultGlobalFeatures*4)
	layout.entityMask = take("result.entity_mask", actors*battleserver.AssaultMaxEntities)
	layout.kindMask = take("result.kind_mask", actors*battleserver.AssaultActionKinds)
	layout.targetMask = take("result.target_mask", actors*battleserver.AssaultMaxEntities)
	layout.skillTargetMask = take("result.skill_target_mask", actors*4*battleserver.AssaultMaxEntities)
	if version == VersionAI41Teacher {
		layout.teacherActions = take("result.teacher_actions", actors*5)
		layout.teacherValid = take("result.teacher_valid", actors)
	} else if versionHasAI42(version) {
		layout.teacherIntent = take("result.teacher_intent", actors*5)
		layout.teacherStatus = take("result.teacher_status", actors)
		layout.executedActions = take("result.executed_actions", actors*5)
		layout.executedValid = take("result.executed_valid", actors)
		layout.rejectionReason = take("result.rejection_reason", actors)
		if version == VersionAI42DAgger {
			layout.activeOrder = take("result.active_order", actors)
		}
	} else if version == VersionAI42Evaluation {
		layout.activeOrder = take("result.active_order", actors)
	}
	layout.size = off
	layout.fields = fields
	return layout
}

func layoutFieldsText(fields []resultLayoutField) string {
	parts := make([]string, len(fields))
	for i, field := range fields {
		parts[i] = field.name + "@" + strconv.Itoa(field.offset) + ":" + strconv.Itoa(field.size)
	}
	return strings.Join(parts, ",")
}

// ai42SchemaText is generated from the same field tables used by both
// encoders. Keeping the offsets in the hash prevents a semantic description
// from silently surviving a serializer drift.
func ai42SchemaText() string {
	scalar := newResultFrameLayout(VersionAI42)
	vector := newVectorResultLayoutVersion(1, VersionAI42)
	return "tanat.assault.ai42.v13|frame=little-endian|action=kind,target,offset81,anchor15|" +
		"skill_navigation=offset81_only|" +
		"teacher_status=none,action,wait,hold,cancel,unavailable|" +
		"executed_reason=none,masked,invalid,server_rejected,safety,timeout,policy_error,unknown255|" +
		"scalar.body=" + strconv.Itoa(scalar.bodySize) + "|scalar.fields=" + layoutFieldsText(scalar.fields) +
		"|vector.body=" + strconv.Itoa(vector.size) + "|vector.fields=" + layoutFieldsText(vector.fields)
}

func ai42EvaluationSchemaText() string {
	scalar := newResultFrameLayout(VersionAI42Evaluation)
	vector := newVectorResultLayoutVersion(1, VersionAI42Evaluation)
	return "tanat.assault.ai42.evaluation.v14|frame=little-endian|" +
		"input=control,kind,target,offset81,anchor15|control=issue,hold,idle|" +
		"skill_navigation=offset81_only|" +
		"scalar.body=" + strconv.Itoa(scalar.bodySize) + "|scalar.fields=" + layoutFieldsText(scalar.fields) +
		"|vector.body=" + strconv.Itoa(vector.size) + "|vector.fields=" + layoutFieldsText(vector.fields)
}

func ai42DAggerSchemaText() string {
	scalar := newResultFrameLayout(VersionAI42DAgger)
	vector := newVectorResultLayoutVersion(1, VersionAI42DAgger)
	return "tanat.assault.ai42.dagger.v15|frame=little-endian|" +
		"input=control,kind,target,offset81,anchor15,intervention01|control=issue,hold,idle|" +
		"teacher_status=none,action,wait,hold,cancel,unavailable|" +
		"skill_navigation=offset81_only|" +
		"scalar.body=" + strconv.Itoa(scalar.bodySize) + "|scalar.fields=" + layoutFieldsText(scalar.fields) +
		"|vector.body=" + strconv.Itoa(vector.size) + "|vector.fields=" + layoutFieldsText(vector.fields)
}

func copyFloat32(body []byte, offset int, values []float32) {
	for index, value := range values {
		start := offset + index*4
		binary.LittleEndian.PutUint32(body[start:start+4], math.Float32bits(value))
	}
}

// ResultEncoder owns one reusable frame buffer. A rollout worker emits one
// fixed-size observation after every step, so rebuilding a bytes.Buffer and
// invoking reflective binary.Write once per float only creates GC pressure.
type ResultEncoder struct {
	body []byte
}

func NewResultEncoder() *ResultEncoder {
	return &ResultEncoder{body: make([]byte, resultBodySize)}
}

func encodeResultRecord(body []byte, result *battleserver.StepResultV1, version uint16) (int, error) {
	if result == nil {
		return 0, errors.New("assault protocol: field=result offset=0: nil result")
	}
	layout := newResultFrameLayout(version)
	if len(body) != layout.recordSize {
		if len(body) < layout.recordSize {
			return 0, fmt.Errorf("assault protocol: field=result.record offset=%d: truncated, got %d bytes, want %d",
				resultHeaderSize+len(body), len(body), layout.recordSize)
		}
		return 0, fmt.Errorf("assault protocol: field=result.record.trailing offset=%d: got %d extra bytes",
			resultHeaderSize+layout.recordSize, len(body)-layout.recordSize)
	}
	off := 0
	putBytes := func(values []byte) { copy(body[off:], values); off += len(values) }
	putU32 := func(value uint32) { binary.LittleEndian.PutUint32(body[off:off+4], value); off += 4 }
	putF32 := func(value float32) { putU32(math.Float32bits(value)) }
	putF32s := func(values []float32) {
		for _, value := range values {
			putF32(value)
		}
	}
	putU32(result.Step)
	putF32(result.Elapsed)
	if result.Done {
		body[off] = 1
	} else {
		body[off] = 0
	}
	off += 4
	putU32(uint32(result.Winner))
	putBytes(result.Invalid[:])
	for _, reward := range result.Reward {
		putF32(reward)
	}
	for i := range result.Observations {
		obs := &result.Observations[i]
		putF32s(obs.Hero[:])
		if layoutHasAbilities(version) {
			for _, ability := range obs.Abilities {
				putF32s(ability[:])
			}
		}
		for _, entity := range obs.Entities {
			putF32s(entity[:])
		}
		putF32s(obs.Global[:])
		putBytes(obs.EntityMask[:])
		putBytes(obs.ActionMask.Kinds[:])
		putBytes(obs.ActionMask.Targets[:])
		for j := range obs.ActionMask.SkillTarget {
			putBytes(obs.ActionMask.SkillTarget[j][:])
		}
	}
	if version == VersionAI41Teacher {
		for _, action := range result.TeacherActions {
			body[off] = byte(action.Kind)
			binary.LittleEndian.PutUint16(body[off+1:off+3], action.Target)
			body[off+3] = action.Direction
			body[off+4] = action.Distance
			off += 5
		}
		putBytes(result.TeacherValid[:])
	} else if versionHasAI42(version) {
		for _, action := range result.TeacherIntent {
			body[off] = byte(action.Kind)
			binary.LittleEndian.PutUint16(body[off+1:off+3], action.Target)
			body[off+3] = action.Direction
			body[off+4] = action.Distance
			off += 5
		}
		putBytes(result.TeacherStatus[:])
		for _, action := range result.ExecutedActions {
			body[off] = byte(action.Kind)
			binary.LittleEndian.PutUint16(body[off+1:off+3], action.Target)
			body[off+3] = action.Direction
			body[off+4] = action.Distance
			off += 5
		}
		putBytes(result.ExecutedValid[:])
		putBytes(result.RejectionReason[:])
		if version == VersionAI42DAgger {
			putBytes(result.ActiveOrder[:])
		}
	} else if version == VersionAI42Evaluation {
		putBytes(result.ActiveOrder[:])
	}
	want := layout.recordSize
	if off != want {
		return off, fmt.Errorf("assault protocol: field=result.record offset=%d: encoded %d bytes, want %d",
			resultHeaderSize+off, off, want)
	}
	return off, nil
}

func (e *ResultEncoder) Write(w io.Writer, result *battleserver.StepResultV1) error {
	return e.WriteVersion(w, result, Version)
}

func (e *ResultEncoder) WriteVersion(w io.Writer, result *battleserver.StepResultV1, version uint16) error {
	if result == nil {
		return errors.New("assault protocol: nil result")
	}
	if !supportedVersion(version) {
		return fmt.Errorf("assault protocol: field=frame.version offset=4: cannot encode version %d", version)
	}
	layout := newResultFrameLayout(version)
	schemaHash, rewardHash, err := resultHeaderHashes(version, result)
	if err != nil {
		return err
	}
	size := layout.bodySize
	if cap(e.body) < size {
		e.body = make([]byte, size)
	}
	body := e.body[:size]
	copy(body[:4], magic[:])
	binary.LittleEndian.PutUint16(body[4:6], version)
	binary.LittleEndian.PutUint16(body[6:8], ResponseResult)
	copy(body[8:40], schemaHash[:])
	copy(body[40:72], rewardHash[:])
	off, err := encodeResultRecord(body[resultHeaderSize:], result, version)
	if err != nil {
		return err
	}
	off += resultHeaderSize
	if off != size {
		return fmt.Errorf("assault protocol: field=result.body offset=%d: encoded %d bytes, want %d",
			off, off, size)
	}
	return writeFrame(w, body)
}

type VectorResultEncoder struct{ body []byte }

func NewVectorResultEncoder() *VectorResultEncoder { return &VectorResultEncoder{} }

func (e *VectorResultEncoder) Write(w io.Writer, results []*battleserver.StepResultV1) error {
	return e.WriteVersion(w, results, Version)
}

func (e *VectorResultEncoder) WriteVersion(w io.Writer, results []*battleserver.StepResultV1, version uint16) error {
	if len(results) < 1 {
		return errors.New("assault protocol: empty vector result")
	}
	if !supportedVersion(version) {
		return fmt.Errorf("assault protocol: field=frame.version offset=4: cannot encode vector version %d", version)
	}
	for index, result := range results {
		if result == nil {
			return fmt.Errorf("assault protocol: field=vector.result[%d] offset=%d: nil result",
				index, vectorResultHeaderSize)
		}
		if _, _, err := resultHeaderHashes(version, result); err != nil {
			return fmt.Errorf("assault protocol: vector result[%d]: %w", index, err)
		}
	}
	layout := newVectorResultLayoutVersion(len(results), version)
	size := layout.size
	if cap(e.body) < size {
		e.body = make([]byte, size)
	}
	body := e.body[:size]
	copy(body[:4], magic[:])
	binary.LittleEndian.PutUint16(body[4:6], version)
	binary.LittleEndian.PutUint16(body[6:8], ResponseVectorResult)
	schemaHash, rewardHash, err := resultHeaderHashes(version, results[0])
	if err != nil {
		return err
	}
	copy(body[8:40], schemaHash[:])
	copy(body[40:72], rewardHash[:])
	binary.LittleEndian.PutUint32(body[72:76], uint32(len(results)))
	for _, result := range results {
		resultSchema, resultReward, _ := resultHeaderHashes(version, result)
		if resultSchema != schemaHash || resultReward != rewardHash {
			return errors.New("assault protocol: field=vector.result_hashes offset=8: schema/reward mismatch")
		}
	}
	var wg sync.WaitGroup
	wg.Add(len(results))
	for i, result := range results {
		go func(index int, value *battleserver.StepResultV1) {
			defer wg.Done()
			binary.LittleEndian.PutUint32(body[layout.steps+index*4:], value.Step)
			binary.LittleEndian.PutUint32(body[layout.elapsed+index*4:], math.Float32bits(value.Elapsed))
			if value.Done {
				body[layout.done+index] = 1
			} else {
				body[layout.done+index] = 0
			}
			binary.LittleEndian.PutUint32(body[layout.winners+index*4:], uint32(value.Winner))
			actorBase := index * battleserver.AssaultHeroCount
			copy(body[layout.invalid+actorBase:], value.Invalid[:])
			copyFloat32(body, layout.rewards+actorBase*4, value.Reward[:])
			for heroIndex := range value.Observations {
				actor := actorBase + heroIndex
				obs := &value.Observations[heroIndex]
				copyFloat32(body, layout.hero+actor*battleserver.AssaultHeroFeatureSize*4, obs.Hero[:])
				if layoutHasAbilities(version) {
					for abilityIndex, ability := range obs.Abilities {
						copyFloat32(body, layout.abilities+
							(actor*4+abilityIndex)*battleserver.AssaultAbilityFeatures*4, ability[:])
					}
				}
				for entityIndex, entity := range obs.Entities {
					copyFloat32(body, layout.entities+
						(actor*battleserver.AssaultMaxEntities+entityIndex)*battleserver.AssaultEntityFeatures*4,
						entity[:])
				}
				copyFloat32(body, layout.global+actor*battleserver.AssaultGlobalFeatures*4, obs.Global[:])
				copy(body[layout.entityMask+actor*battleserver.AssaultMaxEntities:], obs.EntityMask[:])
				copy(body[layout.kindMask+actor*battleserver.AssaultActionKinds:], obs.ActionMask.Kinds[:])
				copy(body[layout.targetMask+actor*battleserver.AssaultMaxEntities:], obs.ActionMask.Targets[:])
				for skill := range obs.ActionMask.SkillTarget {
					start := layout.skillTargetMask + (actor*4+skill)*battleserver.AssaultMaxEntities
					copy(body[start:], obs.ActionMask.SkillTarget[skill][:])
				}
				if version == VersionAI41Teacher {
					action := value.TeacherActions[heroIndex]
					start := layout.teacherActions + actor*5
					body[start] = byte(action.Kind)
					binary.LittleEndian.PutUint16(body[start+1:start+3], action.Target)
					body[start+3] = action.Direction
					body[start+4] = action.Distance
					body[layout.teacherValid+actor] = value.TeacherValid[heroIndex]
				} else if versionHasAI42(version) {
					action := value.TeacherIntent[heroIndex]
					start := layout.teacherIntent + actor*5
					body[start] = byte(action.Kind)
					binary.LittleEndian.PutUint16(body[start+1:start+3], action.Target)
					body[start+3] = action.Direction
					body[start+4] = action.Distance
					body[layout.teacherStatus+actor] = value.TeacherStatus[heroIndex]
					action = value.ExecutedActions[heroIndex]
					start = layout.executedActions + actor*5
					body[start] = byte(action.Kind)
					binary.LittleEndian.PutUint16(body[start+1:start+3], action.Target)
					body[start+3] = action.Direction
					body[start+4] = action.Distance
					body[layout.executedValid+actor] = value.ExecutedValid[heroIndex]
					body[layout.rejectionReason+actor] = value.RejectionReason[heroIndex]
					if version == VersionAI42DAgger {
						body[layout.activeOrder+actor] = value.ActiveOrder[heroIndex]
					}
				} else if version == VersionAI42Evaluation {
					body[layout.activeOrder+actor] = value.ActiveOrder[heroIndex]
				}
			}
		}(i, result)
	}
	wg.Wait()
	if len(body) != layout.size {
		return fmt.Errorf("assault protocol: field=vector.body offset=%d: encoded %d bytes, want %d",
			len(body), len(body), layout.size)
	}
	return writeFrame(w, body)
}

func WriteError(w io.Writer, message string) error {
	return WriteErrorVersion(w, message, Version)
}

func WriteErrorVersion(w io.Writer, message string, version uint16) error {
	b := new(bytes.Buffer)
	b.Write(magic[:])
	writeU16(b, version)
	writeU16(b, ResponseError)
	b.WriteString(message)
	return writeFrame(w, b.Bytes())
}

func readFrame(r io.Reader) ([]byte, error) {
	var length uint32
	var prefix [4]byte
	if _, err := io.ReadFull(r, prefix[:]); err != nil {
		return nil, fmt.Errorf("assault protocol: field=frame.length offset=0: %w", err)
	}
	length = binary.LittleEndian.Uint32(prefix[:])
	if length < 8 || length > maxFrameSize {
		return nil, fmt.Errorf("assault protocol: field=frame.length offset=0: invalid frame length %d", length)
	}
	body := make([]byte, length)
	if n, err := io.ReadFull(r, body); err != nil {
		return nil, fmt.Errorf("assault protocol: field=frame.body offset=%d: truncated, got %d of %d bytes: %w",
			4+n, n, length, err)
	}
	return body, nil
}

func writeFrame(w io.Writer, body []byte) error {
	bw, ok := w.(*bufio.Writer)
	if !ok {
		bw = bufio.NewWriter(w)
	}
	var prefix [4]byte
	binary.LittleEndian.PutUint32(prefix[:], uint32(len(body)))
	if _, err := bw.Write(prefix[:]); err != nil {
		return err
	}
	if _, err := bw.Write(body); err != nil {
		return err
	}
	return bw.Flush()
}

func writeU16(w io.Writer, v uint16)  { _ = binary.Write(w, binary.LittleEndian, v) }
func writeU32(w io.Writer, v uint32)  { _ = binary.Write(w, binary.LittleEndian, v) }
func writeI32(w io.Writer, v int32)   { _ = binary.Write(w, binary.LittleEndian, v) }
func writeF32(w io.Writer, v float32) { writeU32(w, math.Float32bits(v)) }
func writeF32s(w io.Writer, values []float32) {
	for _, v := range values {
		writeF32(w, v)
	}
}
