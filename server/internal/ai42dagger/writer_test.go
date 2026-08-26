package ai42dagger

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"tanatserver/internal/ai42dataset"
	"tanatserver/internal/assaultproto"
	"tanatserver/internal/battleserver"
)

func writeTestSchedule(t *testing.T, root string) string {
	t.Helper()
	controllers := make([]int, ai42dataset.HeroCount)
	roster := make([]int32, ai42dataset.HeroCount)
	sides := make([]uint8, ai42dataset.HeroCount)
	for slot := 0; slot < ai42dataset.HeroCount; slot++ {
		controllers[slot] = int(battleserver.AssaultControllerAI40)
		roster[slot] = int32(slot + 1)
		if slot >= ai42dataset.HeroCount/2 {
			sides[slot] = 1
		}
	}
	value := map[string]any{
		"max_steps": 5, "split_seed": 7, "validation_fraction": 0.0,
		"match_schedule": []any{map[string]any{
			"index": 0, "match_id": "dagger-stream-test", "seed": 11,
			"scenario": "dagger", "controller_by_slot": controllers,
			"roster_ids": roster, "side_by_slot": sides,
		}},
	}
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := ai42dataset.CanonicalizeJSON(raw)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "schedule.json")
	if err := os.WriteFile(path, canonical, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func daggerStepFrame() []byte {
	body := make([]byte, 8+ai42dataset.HeroCount*7)
	copy(body[:4], []byte("TANT"))
	binary.LittleEndian.PutUint16(body[4:6], assaultproto.VersionAI42DAgger)
	binary.LittleEndian.PutUint16(body[6:8], assaultproto.CommandStep)
	frame := make([]byte, 4+len(body))
	binary.LittleEndian.PutUint32(frame[:4], uint32(len(body)))
	copy(frame[4:], body)
	return frame
}

func terminalResultFrame(t *testing.T) []byte {
	t.Helper()
	var result battleserver.StepResultV1
	result.RewardHash = battleserver.AssaultRewardHashV5
	result.Done = true
	for slot := 0; slot < ai42dataset.HeroCount; slot++ {
		result.TeacherStatus[slot] = battleserver.AssaultTeacherStatusUnavailable
		result.ExecutedValid[slot] = 1
	}
	var encoded bytes.Buffer
	if err := assaultproto.NewResultEncoder().WriteVersion(
		&encoded, &result, assaultproto.VersionAI42DAgger,
	); err != nil {
		t.Fatal(err)
	}
	return encoded.Bytes()
}

func TestWriteStreamPublishesStrictOneMatchGeneration(t *testing.T) {
	root := t.TempDir()
	schedule := writeTestSchedule(t, root)
	output := filepath.Join(root, "generation")
	stream := bytes.NewBuffer(daggerStepFrame())
	stream.Write(terminalResultFrame(t))
	result, err := WriteStream(stream, WriterOptions{
		SchedulePath: schedule, OutputPath: output, ReserveTicks: 5,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.MatchID != "dagger-stream-test" || result.Ticks != 1 {
		t.Fatalf("unexpected writer result: %+v", result)
	}
	generation, err := ai42dataset.OpenGeneration(output, "")
	if err != nil {
		t.Fatal(err)
	}
	if ids := generation.MatchIDs(); len(ids) != 1 || ids[0] != result.MatchID {
		t.Fatalf("unexpected generation matches: %v", ids)
	}
}

func TestWriteStreamFailsClosedOnIncompletePair(t *testing.T) {
	root := t.TempDir()
	output := filepath.Join(root, "generation")
	_, err := WriteStream(bytes.NewReader(daggerStepFrame()), WriterOptions{
		SchedulePath: writeTestSchedule(t, root), OutputPath: output, ReserveTicks: 1,
	})
	if err == nil {
		t.Fatal("writer accepted a request without its simulator result")
	}
	if _, statErr := os.Stat(output); !os.IsNotExist(statErr) {
		t.Fatalf("failed stream published partial output: %v", statErr)
	}
}
