// Package ai42schedule owns the canonical match schedule shared by native
// dataset producers. It deliberately contains no rollout or policy logic.
package ai42schedule

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"tanatserver/internal/ai42dataset"
)

type File struct {
	MaxSteps           uint32      `json:"max_steps"`
	SplitSeed          int64       `json:"split_seed"`
	ValidationFraction float64     `json:"validation_fraction"`
	MatchSchedule      []MatchSpec `json:"match_schedule"`
}

type MatchSpec struct {
	Index       int     `json:"index"`
	MatchID     string  `json:"match_id"`
	Seed        int64   `json:"seed"`
	Scenario    string  `json:"scenario"`
	Controllers []int   `json:"controller_by_slot"`
	Roster      []int32 `json:"roster_ids"`
	Sides       []uint8 `json:"side_by_slot"`
}

func Read(path string, stdin io.Reader) ([]byte, File, error) {
	var raw []byte
	var err error
	if path == "-" {
		raw, err = io.ReadAll(stdin)
	} else {
		raw, err = os.ReadFile(path)
	}
	if err != nil {
		return nil, File{}, fmt.Errorf("read schedule: %w", err)
	}
	if _, err := ai42dataset.CanonicalizeJSON(raw); err != nil {
		return nil, File{}, fmt.Errorf("schedule JSON must already be canonical: %w", err)
	}
	var schedule File
	if err := json.Unmarshal(raw, &schedule); err != nil {
		return nil, File{}, fmt.Errorf("decode schedule: %w", err)
	}
	if schedule.MaxSteps == 0 {
		return nil, File{}, fmt.Errorf("max_steps must be positive")
	}
	for index, spec := range schedule.MatchSchedule {
		if spec.Index != index {
			return nil, File{}, fmt.Errorf("match_schedule[%d].index=%d is not canonical", index, spec.Index)
		}
		if err := spec.Validate(); err != nil {
			return nil, File{}, fmt.Errorf("match_schedule[%d]: %w", index, err)
		}
	}
	return raw, schedule, nil
}

func (spec MatchSpec) Validate() error {
	if spec.MatchID == "" || spec.Scenario == "" {
		return fmt.Errorf("match_id and scenario are required")
	}
	if len(spec.Controllers) != ai42dataset.HeroCount || len(spec.Roster) != ai42dataset.HeroCount || len(spec.Sides) != ai42dataset.HeroCount {
		return fmt.Errorf("controller, roster, and side schedules must contain ten slots")
	}
	seen := make(map[int32]struct{}, ai42dataset.HeroCount)
	zero, one := 0, 0
	for slot := 0; slot < ai42dataset.HeroCount; slot++ {
		if spec.Controllers[slot] < 0 || spec.Controllers[slot] > 3 {
			return fmt.Errorf("controller_by_slot[%d]=%d is outside 0..3", slot, spec.Controllers[slot])
		}
		if _, ok := seen[spec.Roster[slot]]; ok {
			return fmt.Errorf("roster_ids[%d] duplicates %d", slot, spec.Roster[slot])
		}
		seen[spec.Roster[slot]] = struct{}{}
		switch spec.Sides[slot] {
		case 0:
			zero++
		case 1:
			one++
		default:
			return fmt.Errorf("side_by_slot[%d] must be 0 or 1", slot)
		}
	}
	if zero != 5 || one != 5 {
		return fmt.Errorf("side_by_slot must contain five slots per side")
	}
	return nil
}

func (spec MatchSpec) Metadata(runtimeManifest []byte) ai42dataset.Metadata {
	metadata := ai42dataset.Metadata{
		ProtocolVersion:      ai42dataset.ProtocolVersion,
		TickHz:               ai42dataset.FrameRateHz,
		MatchID:              spec.MatchID,
		RuntimeManifest:      runtimeManifest,
		RuntimeManifestHash:  sha256.Sum256(runtimeManifest),
		SchemaHash:           ai42dataset.AI42SchemaHash,
		RewardHash:           ai42dataset.AI42RewardHash,
		TrajectorySchemaHash: ai42dataset.AI42TrajectorySchemaHash,
		Seed:                 spec.Seed,
		Scenario:             spec.Scenario,
	}
	for slot := 0; slot < ai42dataset.HeroCount; slot++ {
		metadata.HeroIDs[slot] = fmt.Sprintf("%s:hero:%02d", spec.MatchID, slot)
		metadata.ControllerBySlot[slot] = uint8(spec.Controllers[slot])
		metadata.RosterIDs[slot] = spec.Roster[slot]
		metadata.SideBySlot[slot] = spec.Sides[slot]
	}
	return metadata
}
