package ai42dataset

import (
	"bytes"
	"crypto/sha256"
	"math"
	"os"
	"path/filepath"
	"testing"

	"tanatserver/internal/battleserver"
)

func testMetadata() Metadata {
	runtime := []byte(`{"fixture":"ai42-go"}`)
	metadata := Metadata{
		ProtocolVersion:      ProtocolVersion,
		TickHz:               FrameRateHz,
		MatchID:              "ai42-test-match",
		RuntimeManifest:      runtime,
		RuntimeManifestHash:  sha256Hash(runtime),
		SchemaHash:           AI42SchemaHash,
		RewardHash:           AI42RewardHash,
		TrajectorySchemaHash: AI42TrajectorySchemaHash,
		Scenario:             "ai30_mirror",
	}
	for slot := 0; slot < HeroCount; slot++ {
		metadata.HeroIDs[slot] = "ai42-test-match:hero:" + twoDigits(slot)
		metadata.ControllerBySlot[slot] = uint8(battleserver.AssaultControllerAI30)
		metadata.RosterIDs[slot] = int32(slot)
		if slot >= HeroCount/2 {
			metadata.SideBySlot[slot] = 1
		}
	}
	return metadata
}

func sha256Hash(value []byte) Hash {
	return sha256.Sum256(value)
}

func twoDigits(value int) string {
	if value < 10 {
		return "0" + string(rune('0'+value))
	}
	return string(rune('0'+value/10)) + string(rune('0'+value%10))
}

func testResult(step uint32, done bool) battleserver.StepResultV1 {
	result := battleserver.StepResultV1{
		SchemaHash: AI42SchemaHash,
		RewardHash: AI42RewardHash,
		Step:       step,
		Elapsed:    float32(step) * 0.2,
		Done:       done,
	}
	for slot := 0; slot < HeroCount; slot++ {
		result.TeacherStatus[slot] = battleserver.AssaultTeacherStatusUnavailable
		result.ExecutedValid[slot] = 1
		result.RejectionReason[slot] = battleserver.AssaultRejectionReasonNone
	}
	return result
}

func testRows(matchID string, tick int) ([HeroCount]Action, [HeroCount]string, [HeroCount]string, [HeroCount]Outcome) {
	var actions [HeroCount]Action
	var parents, boundaries [HeroCount]string
	var outcomes [HeroCount]Outcome
	for slot := 0; slot < HeroCount; slot++ {
		if tick == 0 {
			parents[slot] = matchID + ":root:" + twoDigits(slot)
		} else {
			parents[slot] = matchID + ":boundary:" + itoa(tick-1) + ":" + twoDigits(slot)
		}
		boundaries[slot] = matchID + ":boundary:" + itoa(tick) + ":" + twoDigits(slot)
	}
	return actions, parents, boundaries, outcomes
}

func itoa(value int) string {
	if value == 0 {
		return "0"
	}
	var digits [20]byte
	index := len(digits)
	for value > 0 {
		index--
		digits[index] = byte('0' + value%10)
		value /= 10
	}
	return string(digits[index:])
}

func TestCaptureValidatesTerminalCoverageAndHashes(t *testing.T) {
	capture, err := NewCapture(testMetadata())
	if err != nil {
		t.Fatal(err)
	}
	for tick := 0; tick < 2; tick++ {
		result := testResult(uint32(tick), tick == 1)
		actions, parents, boundaries, outcomes := testRows(testMetadata().MatchID, tick)
		for slot := 0; slot < HeroCount; slot++ {
			outcomes[slot] = Outcome{Reward: result.Reward[slot], Terminal: result.Done}
		}
		if err := capture.Append(&result, actions, parents, boundaries, outcomes); err != nil {
			t.Fatalf("append tick %d: %v", tick, err)
		}
	}
	prepared, err := capture.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	if prepared.TickCount != 2 || prepared.Done[1] != 1 {
		t.Fatalf("prepared shape/terminal = %d/%d", prepared.TickCount, prepared.Done[1])
	}
	for slot := 0; slot < HeroCount; slot++ {
		if prepared.TrajectoryHashes[slot] == (Hash{}) || len(prepared.IncrementalHashes[slot]) != 2 {
			t.Fatalf("missing deterministic hashes for slot %d", slot)
		}
	}
}

func TestCaptureReportsFieldTickAndSlot(t *testing.T) {
	capture, err := NewCapture(testMetadata())
	if err != nil {
		t.Fatal(err)
	}
	result := testResult(0, true)
	result.Observations[3].Hero[7] = float32(math.NaN())
	actions, parents, boundaries, outcomes := testRows(testMetadata().MatchID, 0)
	for slot := 0; slot < HeroCount; slot++ {
		outcomes[slot] = Outcome{Reward: result.Reward[slot], Terminal: true}
	}
	err = capture.Append(&result, actions, parents, boundaries, outcomes)
	if err == nil || !containsAll(err.Error(), "field=hero", "tick=0", "slot=3") {
		t.Fatalf("error=%v, want field/tick/slot diagnostics", err)
	}
}

func TestWriteGenerationIsDeterministic(t *testing.T) {
	build := func(root string) []byte {
		capture, err := NewCapture(testMetadata())
		if err != nil {
			t.Fatal(err)
		}
		result := testResult(0, true)
		actions, parents, boundaries, outcomes := testRows(testMetadata().MatchID, 0)
		for slot := 0; slot < HeroCount; slot++ {
			outcomes[slot] = Outcome{Reward: result.Reward[slot], Terminal: true}
		}
		if err := capture.Append(&result, actions, parents, boundaries, outcomes); err != nil {
			t.Fatal(err)
		}
		prepared, err := capture.Finalize()
		if err != nil {
			t.Fatal(err)
		}
		if err := WriteGeneration(root, prepared); err != nil {
			t.Fatal(err)
		}
		payload, err := os.ReadFile(filepath.Join(root, "shard-000000.a42"))
		if err != nil {
			t.Fatal(err)
		}
		return payload
	}
	first := build(filepath.Join(t.TempDir(), "generation"))
	second := build(filepath.Join(t.TempDir(), "generation"))
	if !bytes.Equal(first, second) {
		t.Fatal("identical captures produced different shard bytes")
	}
	if !bytes.HasPrefix(first, []byte(ShardMagic)) {
		t.Fatal("missing AI42GS1 magic")
	}
}

func containsAll(value string, parts ...string) bool {
	for _, part := range parts {
		if !bytes.Contains([]byte(value), []byte(part)) {
			return false
		}
	}
	return true
}
