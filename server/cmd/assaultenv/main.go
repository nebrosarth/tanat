// Command assaultenv exposes one synchronous AssaultEnv over stdin/stdout.
// Run multiple processes for parallel rollout workers.
package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"runtime"
	"runtime/pprof"
	"sync"

	"tanatserver/internal/assaultproto"
	"tanatserver/internal/battleserver"
)

type vectorJob struct {
	index    int
	reset    *battleserver.AssaultResetV1
	actions  *[battleserver.AssaultHeroCount]battleserver.HeroActionV1
	controls *[battleserver.AssaultHeroCount]battleserver.AssaultControlV1
}

type vectorWorker struct {
	env    *battleserver.AssaultEnv
	result battleserver.StepResultV1
	err    error
}

type vectorRunner struct {
	workers []*vectorWorker
	jobs    chan vectorJob
	pending sync.WaitGroup
	closed  sync.WaitGroup
}

func protocolModes(version uint16) (contract, wrongLane, navigation, strategic, teacher bool) {
	contract = version == assaultproto.VersionAI41WrongLane ||
		version == assaultproto.VersionAI41Evaluation ||
		version == assaultproto.VersionAI41Navigation ||
		version == assaultproto.VersionAI41NavigationEvaluation ||
		version == assaultproto.VersionAI41Strategic ||
		version == assaultproto.VersionAI41StrategicEvaluation ||
		version == assaultproto.VersionAI41Teacher ||
		version == assaultproto.VersionAI42 ||
		version == assaultproto.VersionAI42Evaluation
	wrongLane = version == assaultproto.VersionAI41WrongLane ||
		version == assaultproto.VersionAI41Navigation ||
		version == assaultproto.VersionAI41Strategic ||
		version == assaultproto.VersionAI41Teacher ||
		version == assaultproto.VersionAI42 ||
		version == assaultproto.VersionAI42Evaluation
	navigation = version == assaultproto.VersionAI41Navigation ||
		version == assaultproto.VersionAI41NavigationEvaluation ||
		version == assaultproto.VersionAI41Strategic ||
		version == assaultproto.VersionAI41StrategicEvaluation ||
		version == assaultproto.VersionAI41Teacher ||
		version == assaultproto.VersionAI42 ||
		version == assaultproto.VersionAI42Evaluation
	strategic = version == assaultproto.VersionAI41Strategic ||
		version == assaultproto.VersionAI41StrategicEvaluation ||
		version == assaultproto.VersionAI41Teacher ||
		version == assaultproto.VersionAI42 ||
		version == assaultproto.VersionAI42Evaluation
	teacher = version == assaultproto.VersionAI41Teacher || version == assaultproto.VersionAI42
	return
}

func newVectorRunner(count int) *vectorRunner {
	threads := min(runtime.GOMAXPROCS(0), count)
	r := &vectorRunner{
		workers: make([]*vectorWorker, count),
		jobs:    make(chan vectorJob, count),
	}
	for i := 0; i < count; i++ {
		r.workers[i] = &vectorWorker{env: battleserver.NewAssaultEnv()}
	}
	for i := 0; i < threads; i++ {
		r.closed.Add(1)
		go func() {
			defer r.closed.Done()
			for job := range r.jobs {
				w := r.workers[job.index]
				if job.reset != nil {
					w.result, w.err = w.env.Reset(*job.reset)
				} else if job.controls != nil {
					w.result, w.err = w.env.StepControlled(*job.actions, *job.controls)
				} else {
					w.result, w.err = w.env.Step(*job.actions)
				}
				r.pending.Done()
			}
		}()
	}
	return r
}

func (r *vectorRunner) close() {
	if r == nil {
		return
	}
	close(r.jobs)
	r.closed.Wait()
	for _, worker := range r.workers {
		worker.env.Close()
	}
}

func (r *vectorRunner) dispatch(jobs []vectorJob) error {
	r.pending.Add(len(jobs))
	for _, job := range jobs {
		r.jobs <- job
	}
	r.pending.Wait()
	for _, job := range jobs {
		if err := r.workers[job.index].err; err != nil {
			return err
		}
	}
	return nil
}

func (r *vectorRunner) reset(
	resets []battleserver.AssaultResetV1, contract, wrongLane, navigation, strategic, teacher bool,
) error {
	if len(resets) != len(r.workers) {
		return fmt.Errorf("vector reset count %d, want %d", len(resets), len(r.workers))
	}
	jobs := make([]vectorJob, len(resets))
	for i := range resets {
		r.workers[i].env.ConfigureWrongLaneCurriculum(contract, wrongLane)
		r.workers[i].env.ConfigureNavigationActions(navigation)
		r.workers[i].env.ConfigureStrategicReward(strategic)
		r.workers[i].env.ConfigureTeacherActions(teacher)
		jobs[i] = vectorJob{index: i, reset: &resets[i]}
	}
	return r.dispatch(jobs)
}

func (r *vectorRunner) resetIndices(
	indices []uint32, resets []battleserver.AssaultResetV1,
	contract, wrongLane, navigation, strategic, teacher bool,
) error {
	if len(indices) != len(resets) {
		return errors.New("vector reset index/value count mismatch")
	}
	seen := make(map[uint32]struct{}, len(indices))
	for _, raw := range indices {
		index := int(raw)
		if index < 0 || index >= len(r.workers) {
			return fmt.Errorf("vector reset index %d out of range", index)
		}
		if _, exists := seen[raw]; exists {
			return fmt.Errorf("duplicate vector reset index %d", index)
		}
		seen[raw] = struct{}{}
	}
	jobs := make([]vectorJob, len(resets))
	for i, raw := range indices {
		index := int(raw)
		r.workers[index].env.ConfigureWrongLaneCurriculum(contract, wrongLane)
		r.workers[index].env.ConfigureNavigationActions(navigation)
		r.workers[index].env.ConfigureStrategicReward(strategic)
		r.workers[index].env.ConfigureTeacherActions(teacher)
		jobs[i] = vectorJob{index: index, reset: &resets[i]}
	}
	return r.dispatch(jobs)
}

func (r *vectorRunner) step(actions [][battleserver.AssaultHeroCount]battleserver.HeroActionV1) error {
	return r.stepControlled(actions, nil)
}

func (r *vectorRunner) stepControlled(
	actions [][battleserver.AssaultHeroCount]battleserver.HeroActionV1,
	controls [][battleserver.AssaultHeroCount]battleserver.AssaultControlV1,
) error {
	if len(actions) != len(r.workers) {
		return fmt.Errorf("vector action count %d, want %d", len(actions), len(r.workers))
	}
	r.pending.Add(len(actions))
	for i := range actions {
		job := vectorJob{index: i, actions: &actions[i]}
		if controls != nil {
			job.controls = &controls[i]
		}
		r.jobs <- job
	}
	r.pending.Wait()
	for i := range actions {
		if err := r.workers[i].err; err != nil {
			return err
		}
	}
	return nil
}

func (r *vectorRunner) results(indices []uint32) []*battleserver.StepResultV1 {
	if indices == nil {
		results := make([]*battleserver.StepResultV1, len(r.workers))
		for i := range r.workers {
			results[i] = &r.workers[i].result
		}
		return results
	}
	results := make([]*battleserver.StepResultV1, len(indices))
	for i, index := range indices {
		results[i] = &r.workers[index].result
	}
	return results
}

func main() {
	// Battle-server logging is intentionally verbose enough to fill a subprocess
	// stderr pipe during a rollout. Keep workers silent unless explicitly debugging.
	if os.Getenv("TANAT_ASSAULTENV_DEBUG") == "" {
		log.SetOutput(io.Discard)
	}
	if profilePath := os.Getenv("TANAT_ASSAULTENV_CPUPROFILE"); profilePath != "" {
		profile, err := os.Create(profilePath)
		if err != nil {
			log.Fatalf("create CPU profile: %v", err)
		}
		if err := pprof.StartCPUProfile(profile); err != nil {
			_ = profile.Close()
			log.Fatalf("start CPU profile: %v", err)
		}
		defer func() {
			pprof.StopCPUProfile()
			_ = profile.Close()
		}()
	}
	if err := run(os.Stdin, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(input io.Reader, output io.Writer) error {
	env := battleserver.NewAssaultEnv()
	defer env.Close()
	var vector *vectorRunner
	defer func() { vector.close() }()
	resultEncoder := assaultproto.NewResultEncoder()
	vectorEncoder := assaultproto.NewVectorResultEncoder()
	in := bufio.NewReaderSize(input, 64<<10)
	out := bufio.NewWriterSize(output, 128<<10)
	for {
		request, err := assaultproto.ReadRequest(in)
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			_ = assaultproto.WriteError(out, err.Error())
			return err
		}
		writeError := func(message string) error {
			return assaultproto.WriteErrorVersion(out, message, request.Version)
		}
		switch request.Command {
		case assaultproto.CommandReset:
			contract, wrongLane, navigation, strategic, teacher := protocolModes(request.Version)
			env.ConfigureWrongLaneCurriculum(contract, wrongLane)
			env.ConfigureNavigationActions(navigation)
			env.ConfigureStrategicReward(strategic)
			env.ConfigureTeacherActions(teacher)
			result, err := env.Reset(request.Reset)
			if err != nil {
				if writeErr := writeError(err.Error()); writeErr != nil {
					return writeErr
				}
				continue
			}
			if err := resultEncoder.WriteVersion(out, &result, request.Version); err != nil {
				return err
			}
		case assaultproto.CommandStep:
			var result battleserver.StepResultV1
			if request.Version == assaultproto.VersionAI42Evaluation {
				result, err = env.StepControlled(request.Actions, request.Controls)
			} else {
				result, err = env.Step(request.Actions)
			}
			if err != nil {
				if writeErr := writeError(err.Error()); writeErr != nil {
					return writeErr
				}
				continue
			}
			if err := resultEncoder.WriteVersion(out, &result, request.Version); err != nil {
				return err
			}
		case assaultproto.CommandVectorReset:
			if vector == nil || len(vector.workers) != len(request.VectorResets) {
				vector.close()
				vector = newVectorRunner(len(request.VectorResets))
			}
			contract, wrongLane, navigation, strategic, teacher := protocolModes(request.Version)
			if err := vector.reset(request.VectorResets, contract, wrongLane,
				navigation, strategic, teacher); err != nil {
				if writeErr := writeError(err.Error()); writeErr != nil {
					return writeErr
				}
				continue
			}
			if err := vectorEncoder.WriteVersion(out, vector.results(nil), request.Version); err != nil {
				return err
			}
		case assaultproto.CommandVectorStep:
			if vector == nil {
				if err := writeError("vector environment is not initialized"); err != nil {
					return err
				}
				continue
			}
			var stepErr error
			if request.Version == assaultproto.VersionAI42Evaluation {
				stepErr = vector.stepControlled(request.VectorActions, request.VectorControls)
			} else {
				stepErr = vector.step(request.VectorActions)
			}
			if stepErr != nil {
				if writeErr := writeError(stepErr.Error()); writeErr != nil {
					return writeErr
				}
				continue
			}
			if err := vectorEncoder.WriteVersion(out, vector.results(nil), request.Version); err != nil {
				return err
			}
		case assaultproto.CommandVectorResetIndices:
			if vector == nil {
				if err := writeError("vector environment is not initialized"); err != nil {
					return err
				}
				continue
			}
			contract, wrongLane, navigation, strategic, teacher := protocolModes(request.Version)
			if err := vector.resetIndices(request.VectorIndices, request.VectorResets, contract,
				wrongLane, navigation, strategic, teacher); err != nil {
				if writeErr := writeError(err.Error()); writeErr != nil {
					return writeErr
				}
				continue
			}
			if err := vectorEncoder.WriteVersion(out, vector.results(request.VectorIndices), request.Version); err != nil {
				return err
			}
		case assaultproto.CommandClose:
			return nil
		}
	}
}
