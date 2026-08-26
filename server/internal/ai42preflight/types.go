// Package ai42preflight owns the native, fail-closed AI-42 preflight
// supervisor. It verifies the durable dataset in Go and delegates only the
// bounded tensor operation to the explicitly configured Torch worker.
package ai42preflight

import (
	"context"
	"time"

	"tanatserver/internal/ai42dataset"
)

const (
	TorchProtocol      = "AI42-torch-preflight-v1"
	BatchPlanVersion   = "AI42-bc-batch-plan-v2"
	ProfileFormat      = "AI42-bc-class-profile-v4"
	ProfileVersion     = ProfileFormat
	SupervisionVersion = "AI42-supervision-v2"
	ReplayScope        = "durable-v2"
	ProtocolVersion    = 13
	TorchWorkerModule  = "tanat_ai40.torch_probe_worker_ai42"
	MaxWorkerRequest   = 64 * 1024 * 1024
	MaxWorkerBundle    = 32 * 1024 * 1024
	MaxWorkerStdout    = 64 * 1024 * 1024
	MaxWorkerStderr    = 1 * 1024 * 1024
	MaxWorkerTimeout   = 10 * time.Minute
	MaxWarmStart       = 512 * 1024 * 1024
	MaxSequenceLength  = 64
	MaxBatchSize       = 64
	MaxBatchRows       = MaxSequenceLength * MaxBatchSize
)

// TorchOutput is the bounded result of one worker process. ExitCode is kept
// separate from Err so a malformed or failed worker response remains
// diagnosable without weakening the protocol checks.
type TorchOutput struct {
	Stdout   []byte
	Stderr   []byte
	ExitCode int
}

// TorchRunner is injectable for table and fault-injection tests. Production
// callers use CommandRunner, which starts the configured executable directly
// without a shell.
type TorchRunner interface {
	Run(context.Context, []byte) (TorchOutput, error)
}

// TargetedRowReader is the required bounded data-plane extension for bundle
// construction after the one full Verify pass. Ranges are sorted, disjoint,
// half-open tick intervals keyed by match ID. Implementations must operate on
// the same frozen generation, reject identity changes, call onRow exactly once
// for every requested physical row, and return that callback count without
// replaying unrelated shards or matches.
type TargetedRowReader interface {
	ReadTargetRows(context.Context, map[string][][2]int, func(ai42dataset.Row) error) (int, error)
}

// VerifiedSplitReader exposes split IDs independently recomputed by the
// durable reader during verification. Returning manifest labels without
// recomputation does not satisfy this contract.
type VerifiedSplitReader interface {
	VerifiedSplitMatchIDs() (map[string][]string, error)
}

// ModelConfig is the exact model portion accepted by the Torch worker.
type ModelConfig struct {
	HiddenSize   int
	ModelWidth   int
	EntityLayers int
	NumHeads     int
	FFMultiplier int
	TimingBins   int
}

// LearnerConfig is the worker's no-update learner configuration. ClassWeights
// are always populated from the verified train-only profile.
type LearnerConfig struct {
	LearningRate         float64
	WeightDecay          float64
	ClassBalancePower    float64
	MaxGradientNorm      float64
	HeadWeights          map[string]float64
	ClassWeights         map[string][]float64
	ClassWeightOverrides map[string][]float64
}

// Config is the strict native preflight configuration. Training controls in a
// Q3 config are parsed for compatibility but are never used to authorize an
// optimizer update.
type Config struct {
	ProtocolVersion int
	Model           ModelConfig
	SequenceLength  int
	BatchSize       int
	// ValidationProbeLimit follows the Q3 trainer: an explicit
	// validation_matches value wins, otherwise validation_batches is used.
	// Zero means retain every validated match (the validation-only config has
	// no trainer sampling limit).
	ValidationProbeLimit   int
	Learner                LearnerConfig
	Seed                   uint32
	SupervisionControllers []uint8
}

// Options controls one preflight execution.
type Options struct {
	ConfigPath                  string
	DatasetPath                 string
	DatasetHash                 string
	ProfilePath                 string
	ProfileHash                 string
	WarmStartPath               string
	OutputPath                  string
	ReportPath                  string
	Device                      string
	AllowWarmStartDatasetChange bool
	// SupervisionControllers, when non-empty, overrides the training config
	// for this exact preflight invocation.
	SupervisionControllers []uint8
	// TorchPythonPath is the only production worker selector. Run always
	// invokes TorchWorkerModule and never accepts caller-provided arguments.
	TorchPythonPath string
	// WorkerCommand is an internal fixed command assembled by the CLI. Run
	// accepts only [TorchPythonPath, "-m", TorchWorkerModule]; arbitrary
	// commands are rejected. Tests should inject Worker instead.
	WorkerCommand []string
	// Worker is intentionally injectable for tests and fault injection only.
	WorkerTimeout time.Duration
	Worker        TorchRunner
	// TargetedRows is an integration/test override. Production normally uses
	// the TargetedRowReader implemented by *ai42dataset.Generation.
	TargetedRows TargetedRowReader
	// SplitEvidence is the corresponding test/integration override for split
	// recomputation evidence.
	SplitEvidence VerifiedSplitReader
}

// Report is intentionally map-backed at the public boundary. The stable
// fields are documented by the JSON produced by Run; map-backed additive
// evidence lets the worker evolve without changing the Go API.
type Report map[string]any

// Run executes one complete native preflight and atomically publishes its
// profile, bundle, and report. It does not modify checkpoints, training runs,
// or accepted model pointers.
func Run(ctx context.Context, options Options) (Report, error) {
	return run(ctx, options)
}
