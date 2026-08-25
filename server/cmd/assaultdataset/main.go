// Command assaultdataset runs one scheduled headless Assault match and emits
// the AI42GS1 generation consumed by tanat_ai40.go_shard_ai42.  The schedule
// is authored by Python; this command never recreates NumPy roster/seed
// permutations.
package main

import (
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"

	"tanatserver/internal/ai42dataset"
	"tanatserver/internal/assaultproto"
	"tanatserver/internal/battleserver"
)

type scheduleFile struct {
	MaxSteps           uint32      `json:"max_steps"`
	SplitSeed          int64       `json:"split_seed"`
	ValidationFraction float64     `json:"validation_fraction"`
	MatchSchedule      []matchSpec `json:"match_schedule"`
}

type matchSpec struct {
	Index       int     `json:"index"`
	MatchID     string  `json:"match_id"`
	Seed        int64   `json:"seed"`
	Scenario    string  `json:"scenario"`
	Controllers []int   `json:"controller_by_slot"`
	Roster      []int32 `json:"roster_ids"`
	Sides       []uint8 `json:"side_by_slot"`
}

func main() {
	if os.Getenv("TANAT_ASSAULTDATASET_DEBUG") == "" {
		log.SetOutput(io.Discard)
	}
	schedulePath := flag.String("schedule", "", "canonical Python runtime/schedule JSON")
	outputPath := flag.String("output", "", "generation directory to publish")
	matchIndex := flag.Int("match-index", 0, "schedule entry index to run")
	flag.Parse()
	if *schedulePath == "" || *outputPath == "" {
		fatal("-schedule and -output are required")
	}
	if *matchIndex < 0 {
		fatal("-match-index must be non-negative")
	}

	var runtimeManifest []byte
	var err error
	if *schedulePath == "-" {
		runtimeManifest, err = io.ReadAll(os.Stdin)
	} else {
		runtimeManifest, err = os.ReadFile(*schedulePath)
	}
	if err != nil {
		fatal("read schedule: %v", err)
	}
	_, err = ai42dataset.CanonicalizeJSON(runtimeManifest)
	if err != nil {
		fatal("schedule JSON must already be canonical: %v", err)
	}
	var schedule scheduleFile
	if err := json.Unmarshal(runtimeManifest, &schedule); err != nil {
		fatal("decode schedule: %v", err)
	}
	if *matchIndex >= len(schedule.MatchSchedule) {
		fatal("match index %d is outside schedule length %d", *matchIndex, len(schedule.MatchSchedule))
	}
	spec := schedule.MatchSchedule[*matchIndex]
	steps := schedule.MaxSteps
	if steps == 0 {
		fatal("max_steps must be positive")
	}
	if err := validateSpec(spec); err != nil {
		fatal("schedule entry: %v", err)
	}

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
	capture, err := ai42dataset.NewCapture(metadata)
	if err != nil {
		fatal("create native capture: %v", err)
	}
	if err := capture.Reserve(int(steps)); err != nil {
		fatal("reserve native capture: %v", err)
	}

	env := battleserver.NewAssaultEnv()
	defer env.Close()
	// These settings select the same v13 navigation/strategic/teacher contract
	// as the Python executor. The result hashes are normalized below because
	// AssaultEnv's direct Go API carries its simulator V1 hashes while the v13
	// protocol encoder advertises the AI42 hashes.
	env.ConfigureWrongLaneCurriculum(true, true)
	env.ConfigureNavigationActions(true)
	env.ConfigureStrategicReward(true)
	env.ConfigureTeacherActions(true)
	reset := battleserver.AssaultResetV1{Seed: spec.Seed, MaxSteps: steps}
	for slot := 0; slot < ai42dataset.HeroCount; slot++ {
		reset.Roster[slot] = spec.Roster[slot]
		reset.Controllers[slot] = battleserver.AssaultControllerV1(spec.Controllers[slot])
	}
	if _, err := env.Reset(reset); err != nil {
		fatal("reset Assault environment: %v", err)
	}

	var actions [ai42dataset.HeroCount]battleserver.HeroActionV1
	var submitted [ai42dataset.HeroCount]ai42dataset.Action
	for tick := 0; ; tick++ {
		result, err := env.Step(actions)
		if err != nil {
			fatal("step %d: %v", tick, err)
		}
		result.SchemaHash = assaultproto.AI42SchemaHash
		result.RewardHash = assaultproto.AI42RewardHash
		var parents, boundaries [ai42dataset.HeroCount]string
		var outcomes [ai42dataset.HeroCount]ai42dataset.Outcome
		for slot := 0; slot < ai42dataset.HeroCount; slot++ {
			if tick == 0 {
				parents[slot] = fmt.Sprintf("%s:root:%02d", spec.MatchID, slot)
			} else {
				parents[slot] = fmt.Sprintf("%s:boundary:%d:%02d", spec.MatchID, tick-1, slot)
			}
			boundaries[slot] = fmt.Sprintf("%s:boundary:%d:%02d", spec.MatchID, tick, slot)
			outcomes[slot] = ai42dataset.Outcome{
				Reward: result.Reward[slot], Terminal: result.Done,
				Winner: result.Winner, WinnerPresent: true,
			}
		}
		if err := capture.Append(&result, submitted, parents, boundaries, outcomes); err != nil {
			fatal("capture tick %d: %v", tick, err)
		}
		if result.Done {
			break
		}
		if uint32(tick+1) >= steps {
			fatal("match did not terminate at configured max_steps=%d", steps)
		}
	}
	prepared, err := capture.Finalize()
	if err != nil {
		fatal("finalize capture: %v", err)
	}
	if err := ai42dataset.WriteGenerationWithSplit(*outputPath, prepared, schedule.SplitSeed, schedule.ValidationFraction); err != nil {
		fatal("publish generation: %v", err)
	}
	fmt.Printf("match_id=%s ticks=%d output=%s\n", spec.MatchID, prepared.TickCount, *outputPath)
}

func validateSpec(spec matchSpec) error {
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

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "assaultdataset: "+format+"\n", args...)
	os.Exit(2)
}
