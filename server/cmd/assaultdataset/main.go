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
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"time"

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

const defaultInitialReserveTicks = 2048

type performanceMetrics struct {
	MatchID              string  `json:"match_id"`
	Ticks                int     `json:"ticks"`
	SimulatedSeconds     float64 `json:"simulated_seconds"`
	TotalSeconds         float64 `json:"total_seconds"`
	ScheduleSeconds      float64 `json:"schedule_seconds"`
	SetupSeconds         float64 `json:"setup_seconds"`
	ResetSeconds         float64 `json:"reset_seconds"`
	SimulationSeconds    float64 `json:"simulation_seconds"`
	CaptureAppendSeconds float64 `json:"capture_append_seconds"`
	FinalizeSeconds      float64 `json:"finalize_seconds"`
	PublishSeconds       float64 `json:"publish_seconds"`
	InitialReserveTicks  int     `json:"initial_reserve_ticks"`
	HeapAllocBytes       uint64  `json:"heap_alloc_bytes"`
	HeapInuseBytes       uint64  `json:"heap_inuse_bytes"`
	TotalAllocBytes      uint64  `json:"total_alloc_bytes"`
	SystemBytes          uint64  `json:"system_bytes"`
}

func main() {
	totalStarted := time.Now()
	if os.Getenv("TANAT_ASSAULTDATASET_DEBUG") == "" {
		log.SetOutput(io.Discard)
	}
	schedulePath := flag.String("schedule", "", "canonical Python runtime/schedule JSON")
	outputPath := flag.String("output", "", "generation directory to publish")
	matchIndex := flag.Int("match-index", 0, "schedule entry index to run")
	cpuProfilePath := flag.String("cpuprofile", os.Getenv("TANAT_ASSAULTDATASET_CPUPROFILE"), "optional Go CPU profile path")
	memProfilePath := flag.String("memprofile", os.Getenv("TANAT_ASSAULTDATASET_MEMPROFILE"), "optional Go heap profile path")
	metricsPath := flag.String("metrics", os.Getenv("TANAT_ASSAULTDATASET_METRICS"), "optional phase metrics JSON path")
	flag.Parse()
	if *schedulePath == "" || *outputPath == "" {
		fatal("-schedule and -output are required")
	}
	if *matchIndex < 0 {
		fatal("-match-index must be non-negative")
	}

	var runtimeManifest []byte
	var err error
	scheduleStarted := time.Now()
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
	scheduleSeconds := time.Since(scheduleStarted).Seconds()
	stopCPUProfile := startCPUProfile(*cpuProfilePath)

	setupStarted := time.Now()
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
	initialReserveTicks := min(int(steps), defaultInitialReserveTicks)
	if err := capture.Reserve(initialReserveTicks); err != nil {
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
	setupSeconds := time.Since(setupStarted).Seconds()
	resetStarted := time.Now()
	reset := battleserver.AssaultResetV1{Seed: spec.Seed, MaxSteps: steps}
	for slot := 0; slot < ai42dataset.HeroCount; slot++ {
		reset.Roster[slot] = spec.Roster[slot]
		reset.Controllers[slot] = battleserver.AssaultControllerV1(spec.Controllers[slot])
	}
	if _, err := env.Reset(reset); err != nil {
		fatal("reset Assault environment: %v", err)
	}
	resetSeconds := time.Since(resetStarted).Seconds()

	var actions [ai42dataset.HeroCount]battleserver.HeroActionV1
	var submitted [ai42dataset.HeroCount]ai42dataset.Action
	var simulationSeconds, captureAppendSeconds float64
	for tick := 0; ; tick++ {
		simulationStarted := time.Now()
		result, err := env.Step(actions)
		simulationSeconds += time.Since(simulationStarted).Seconds()
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
		captureStarted := time.Now()
		err = capture.Append(&result, submitted, parents, boundaries, outcomes)
		captureAppendSeconds += time.Since(captureStarted).Seconds()
		if err != nil {
			fatal("capture tick %d: %v", tick, err)
		}
		if result.Done {
			break
		}
		if uint32(tick+1) >= steps {
			fatal("match did not terminate at configured max_steps=%d", steps)
		}
	}
	finalizeStarted := time.Now()
	prepared, err := capture.Finalize()
	finalizeSeconds := time.Since(finalizeStarted).Seconds()
	if err != nil {
		fatal("finalize capture: %v", err)
	}
	publishStarted := time.Now()
	if err := ai42dataset.WriteGenerationWithSplit(*outputPath, prepared, schedule.SplitSeed, schedule.ValidationFraction); err != nil {
		fatal("publish generation: %v", err)
	}
	publishSeconds := time.Since(publishStarted).Seconds()
	stopCPUProfile()
	writeHeapProfile(*memProfilePath)
	var memory runtime.MemStats
	runtime.ReadMemStats(&memory)
	metrics := performanceMetrics{
		MatchID: spec.MatchID, Ticks: prepared.TickCount,
		SimulatedSeconds: float64(prepared.TickCount) / float64(ai42dataset.FrameRateHz),
		TotalSeconds:     time.Since(totalStarted).Seconds(), ScheduleSeconds: scheduleSeconds,
		SetupSeconds: setupSeconds, ResetSeconds: resetSeconds, SimulationSeconds: simulationSeconds,
		CaptureAppendSeconds: captureAppendSeconds, FinalizeSeconds: finalizeSeconds, PublishSeconds: publishSeconds,
		InitialReserveTicks: initialReserveTicks, HeapAllocBytes: memory.HeapAlloc,
		HeapInuseBytes: memory.HeapInuse, TotalAllocBytes: memory.TotalAlloc, SystemBytes: memory.Sys,
	}
	if *metricsPath != "" {
		writeMetrics(*metricsPath, metrics)
	}
	fmt.Printf("match_id=%s ticks=%d output=%s\n", spec.MatchID, prepared.TickCount, *outputPath)
}

func startCPUProfile(path string) func() {
	if path == "" {
		return func() {}
	}
	profile, err := os.Create(path)
	if err != nil {
		fatal("create CPU profile: %v", err)
	}
	if err := pprof.StartCPUProfile(profile); err != nil {
		_ = profile.Close()
		fatal("start CPU profile: %v", err)
	}
	return func() {
		pprof.StopCPUProfile()
		if err := profile.Close(); err != nil {
			fatal("close CPU profile: %v", err)
		}
	}
}

func writeHeapProfile(path string) {
	if path == "" {
		return
	}
	runtime.GC()
	profile, err := os.Create(path)
	if err != nil {
		fatal("create heap profile: %v", err)
	}
	if err := pprof.WriteHeapProfile(profile); err != nil {
		_ = profile.Close()
		fatal("write heap profile: %v", err)
	}
	if err := profile.Close(); err != nil {
		fatal("close heap profile: %v", err)
	}
}

func writeMetrics(path string, metrics performanceMetrics) {
	payload, err := json.Marshal(metrics)
	if err != nil {
		fatal("encode performance metrics: %v", err)
	}
	temporary, err := os.CreateTemp(filepath.Dir(path), ".assaultdataset-metrics-*")
	if err != nil {
		fatal("create performance metrics: %v", err)
	}
	temporaryName := temporary.Name()
	defer os.Remove(temporaryName)
	if _, err := temporary.Write(payload); err != nil {
		_ = temporary.Close()
		fatal("write performance metrics: %v", err)
	}
	if err := temporary.Close(); err != nil {
		fatal("close performance metrics: %v", err)
	}
	if err := os.Rename(temporaryName, path); err != nil {
		fatal("publish performance metrics: %v", err)
	}
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
