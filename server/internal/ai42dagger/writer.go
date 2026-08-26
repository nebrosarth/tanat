// Package ai42dagger owns the native streaming boundary between an external
// policy rollout and the durable AI-42 dataset format.
package ai42dagger

import (
	"errors"
	"fmt"
	"io"

	"tanatserver/internal/ai42dataset"
	"tanatserver/internal/ai42schedule"
	"tanatserver/internal/assaultproto"
)

type WriterOptions struct {
	SchedulePath string
	OutputPath   string
	MatchIndex   int
	ReserveTicks int
}

type WriterResult struct {
	MatchID string
	Ticks   int
}

func WriteStream(input io.Reader, options WriterOptions) (WriterResult, error) {
	if input == nil {
		return WriterResult{}, fmt.Errorf("input stream is required")
	}
	if options.SchedulePath == "" || options.SchedulePath == "-" || options.OutputPath == "" {
		return WriterResult{}, fmt.Errorf("schedule file and output directory are required")
	}
	if options.MatchIndex < 0 || options.ReserveTicks < 0 {
		return WriterResult{}, fmt.Errorf("match index and reserve ticks must be non-negative")
	}
	runtimeManifest, schedule, err := ai42schedule.Read(options.SchedulePath, nil)
	if err != nil {
		return WriterResult{}, err
	}
	if options.MatchIndex >= len(schedule.MatchSchedule) {
		return WriterResult{}, fmt.Errorf(
			"match index %d is outside schedule length %d",
			options.MatchIndex, len(schedule.MatchSchedule),
		)
	}
	spec := schedule.MatchSchedule[options.MatchIndex]
	capture, err := ai42dataset.NewCapture(spec.Metadata(runtimeManifest))
	if err != nil {
		return WriterResult{}, fmt.Errorf("create native capture: %w", err)
	}
	if err := capture.Reserve(min(options.ReserveTicks, int(schedule.MaxSteps))); err != nil {
		return WriterResult{}, fmt.Errorf("reserve native capture: %w", err)
	}

	for tick := 0; ; tick++ {
		request, err := assaultproto.ReadRequest(input)
		if err != nil {
			if errors.Is(err, io.EOF) {
				return WriterResult{}, fmt.Errorf("input ended before a terminal result")
			}
			return WriterResult{}, fmt.Errorf("read policy request at tick %d: %w", tick, err)
		}
		if request.Version != assaultproto.VersionAI42DAgger || request.Command != assaultproto.CommandStep {
			return WriterResult{}, fmt.Errorf("tick %d requires a scalar v15 STEP request", tick)
		}
		result, err := assaultproto.ReadResultVersion(input, assaultproto.VersionAI42DAgger)
		if err != nil {
			return WriterResult{}, fmt.Errorf("read simulator result at tick %d: %w", tick, err)
		}
		// v15 adds transport-only intervention and active-order fields. Durable
		// rows intentionally retain the established v13 observation/reward IDs.
		result.SchemaHash = ai42dataset.AI42SchemaHash
		result.RewardHash = ai42dataset.AI42RewardHash
		var submitted [ai42dataset.HeroCount]ai42dataset.Action
		var parents, boundaries [ai42dataset.HeroCount]string
		var outcomes [ai42dataset.HeroCount]ai42dataset.Outcome
		for slot := 0; slot < ai42dataset.HeroCount; slot++ {
			action := request.Actions[slot]
			submitted[slot] = ai42dataset.Action{
				Kind: uint8(action.Kind), Target: action.Target,
				Direction: action.Direction, Distance: action.Distance,
			}
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
			return WriterResult{}, fmt.Errorf("append tick %d: %w", tick, err)
		}
		if result.Done {
			break
		}
		if uint32(tick+1) >= schedule.MaxSteps {
			return WriterResult{}, fmt.Errorf("stream did not terminate at max_steps=%d", schedule.MaxSteps)
		}
	}
	prepared, err := capture.Finalize()
	if err != nil {
		return WriterResult{}, fmt.Errorf("finalize capture: %w", err)
	}
	if err := ai42dataset.WriteGenerationWithSplit(
		options.OutputPath, prepared, schedule.SplitSeed, schedule.ValidationFraction,
	); err != nil {
		return WriterResult{}, fmt.Errorf("publish generation: %w", err)
	}
	return WriterResult{MatchID: spec.MatchID, Ticks: prepared.TickCount}, nil
}
