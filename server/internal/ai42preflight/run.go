package ai42preflight

import (
	"context"
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"tanatserver/internal/ai42dataset"
)

func run(parent context.Context, options Options) (Report, error) {
	if parent == nil {
		parent = context.Background()
	}
	started := time.Now()
	config, err := LoadConfig(options.ConfigPath)
	if err != nil {
		return nil, err
	}
	if len(options.SupervisionControllers) != 0 {
		config.SupervisionControllers = append([]uint8(nil), options.SupervisionControllers...)
		sort.Slice(config.SupervisionControllers, func(i, j int) bool {
			return config.SupervisionControllers[i] < config.SupervisionControllers[j]
		})
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	if options.DatasetPath == "" {
		return nil, fmt.Errorf("--dataset is required")
	}
	if options.WarmStartPath == "" {
		return nil, fmt.Errorf("--warm-start is required")
	}
	if options.OutputPath == "" {
		return nil, fmt.Errorf("--output is required")
	}
	if options.Device == "" {
		options.Device = "auto"
	}
	if options.DatasetHash != "" && !validHash(options.DatasetHash) {
		return nil, fmt.Errorf("dataset hash must be a lower-case SHA-256")
	}
	if options.WorkerTimeout <= 0 || options.WorkerTimeout > MaxWorkerTimeout {
		return nil, fmt.Errorf("worker timeout must be in (0,%s]", MaxWorkerTimeout)
	}
	if options.ProfileHash != "" && !validHash(options.ProfileHash) {
		return nil, fmt.Errorf("profile hash must be a lower-case SHA-256")
	}
	if options.Worker == nil && options.TorchPythonPath == "" && len(options.WorkerCommand) == 0 {
		return nil, fmt.Errorf("--torch-python is required; no Python fallback is available")
	}
	if options.ReportPath == "" {
		options.ReportPath = filepath.Join(options.OutputPath, "preflight_report.json")
	}
	if options.ProfilePath == "" {
		options.ProfilePath = filepath.Join(options.OutputPath, "class_profile_ai42.json")
	}
	outputDirectory, err := filepath.Abs(options.OutputPath)
	if err != nil {
		return nil, fmt.Errorf("resolve output: %w", err)
	}
	if err := ensureSafeDirectory(outputDirectory); err != nil {
		return nil, fmt.Errorf("create output: %w", err)
	}
	options.ReportPath, err = resolveOutputArtifactPath(outputDirectory, options.ReportPath, "preflight_report.json", "report")
	if err != nil {
		return nil, fmt.Errorf("resolve report: %w", err)
	}
	options.ProfilePath, err = resolveProfileArtifactPath(outputDirectory, options.ProfilePath, "class_profile_ai42.json")
	if err != nil {
		return nil, fmt.Errorf("resolve profile: %w", err)
	}
	if err := rejectExistingDestination(options.ReportPath, "report"); err != nil {
		return nil, err
	}
	options.WarmStartPath, err = filepath.Abs(options.WarmStartPath)
	if err != nil {
		return nil, fmt.Errorf("resolve warm-start: %w", err)
	}
	options.DatasetPath, err = filepath.Abs(options.DatasetPath)
	if err != nil {
		return nil, fmt.Errorf("resolve dataset: %w", err)
	}

	evidence := executionEvidence{}
	openStarted := time.Now()
	generation, datasetIdentityBefore, manifestContent, err := openDataset(options.DatasetPath, options.DatasetHash)
	if err != nil {
		return nil, err
	}
	evidence.OpenGenerationCalls = 1
	evidence.OpenElapsed = time.Since(openStarted)
	evidence.DatasetManifestFile = manifestContent
	targetedSource, err := resolveTargetedRows(options, generation)
	if err != nil {
		return nil, err
	}
	splitSource, err := resolveSplitEvidence(options, generation)
	if err != nil {
		return nil, err
	}
	accumulator := newAccumulator(config.SequenceLength)
	verifyStarted := time.Now()
	verification, verifyWorkers, err := verifyDataset(parent, generation, accumulator)
	if err != nil {
		return nil, fmt.Errorf("full durable-v2 dataset verification failed: %w", err)
	}
	evidence.FullVerifyPasses = 1
	if verifyWorkers > verification.Shards {
		verifyWorkers = verification.Shards
	}
	evidence.VerifyWorkers = verifyWorkers
	evidence.VerifyElapsed = time.Since(verifyStarted)
	if err := datasetIdentityBefore.Validate(options.DatasetPath); err != nil {
		return nil, fmt.Errorf("dataset changed during full verification")
	}
	evidence.IdentityChecks++
	profile, err := accumulator.profile(generation.ManifestHash(), config)
	if err != nil {
		return nil, err
	}
	if verification.ManifestHash != generation.ManifestHash() || verification.Matches != len(accumulator.Matches) {
		return nil, fmt.Errorf("dataset verification report is incomplete")
	}
	datasetContentExpected, err := buildDatasetContentExpectation(manifestContent, verification)
	if err != nil {
		return nil, err
	}

	planStarted := time.Now()
	plan, err := makeBatchPlan(accumulator.Matches, accumulator.Evidence, config)
	if err != nil {
		return nil, err
	}
	if err := validateVerifiedSplit(splitSource, accumulator.Matches, plan.SplitHash); err != nil {
		return nil, err
	}
	evidence.SplitRecomputed = true
	evidence.PlanElapsed = time.Since(planStarted)
	profileStarted := time.Now()
	profileBytes, err := profileBytes(profile)
	if err != nil {
		return nil, err
	}
	profile, profileWasExisting, profileFileHash, profileFileSize, err := publishOrValidateProfile(options.ProfilePath, outputDirectory, options.ProfileHash, profile, profileBytes)
	if err != nil {
		return nil, err
	}
	finalWeights := profile.Weights
	evidence.ProfileElapsed = time.Since(profileStarted)

	targetedShards, err := selectedShardNames(generation, plan)
	if err != nil {
		return nil, err
	}
	targetedStarted := time.Now()
	_, bundleBytes, supervised, targetedRows, err := captureBatch(parent, targetedSource, plan)
	if err != nil {
		return nil, err
	}
	// Generation.ReadTargetRows performs the authoritative VerifyMatchIDs pass
	// and forwards the requested rows from that same pass. Do not pre-verify
	// the shard separately: that would read and decompress it twice.
	evidence.TargetedShardRereadPasses = 1
	evidence.TargetedShardRereadShards = len(targetedShards)
	evidence.TargetedReadCalls = 1
	evidence.TargetedRows = targetedRows
	evidence.TargetedReadElapsed = time.Since(targetedStarted)
	evidence.TargetedShardRereadElapsed = evidence.TargetedReadElapsed
	if supervised < 1 {
		return nil, fmt.Errorf("selected batch contains no supported supervision")
	}
	bundleStarted := time.Now()
	bundleHash := sha256Hex(bundleBytes)
	bundlePath := filepath.Join(outputDirectory, "preflight_batch_bundle.json")
	if err := atomicCreate(bundlePath, bundleBytes); err != nil {
		return nil, err
	}
	if got, err := hashFile(bundlePath, MaxWorkerBundle); err != nil || got != bundleHash {
		return nil, fmt.Errorf("published batch bundle changed before worker")
	}
	evidence.BundleElapsed = time.Since(bundleStarted)

	warmStarted := time.Now()
	warmHash, err := hashFile(options.WarmStartPath, MaxWarmStart)
	if err != nil {
		return nil, fmt.Errorf("hash warm-start: %w", err)
	}
	evidence.WarmHashElapsed = time.Since(warmStarted)
	requestValue := map[string]any{
		"protocol": TorchProtocol, "seed": int(config.Seed), "device": options.Device,
		"model":      map[string]any{"hidden_size": config.Model.HiddenSize, "model_width": config.Model.ModelWidth, "entity_layers": config.Model.EntityLayers, "num_heads": config.Model.NumHeads, "ff_multiplier": config.Model.FFMultiplier},
		"learner":    map[string]any{"learning_rate": config.Learner.LearningRate, "weight_decay": config.Learner.WeightDecay, "class_balance_power": config.Learner.ClassBalancePower, "max_gradient_norm": config.Learner.MaxGradientNorm, "head_weights": scalarWeightsAsAny(config.Learner.HeadWeights), "class_weights": weightsAsAny(finalWeights)},
		"warm_start": map[string]any{"path": options.WarmStartPath, "sha256": warmHash, "dataset_hash": generation.ManifestHash(), "allow_dataset_change": options.AllowWarmStartDatasetChange},
		"batch":      map[string]any{"kind": "bundle", "path": bundlePath, "sha256": bundleHash},
	}
	unsignedBytes, err := canonicalJSON(requestValue)
	if err != nil {
		return nil, err
	}
	requestHash := sha256Hex(unsignedBytes)
	requestValue["request_sha256"] = requestHash
	requestBytes, err := canonicalJSON(requestValue)
	if err != nil {
		return nil, err
	}
	if len(requestBytes) > MaxWorkerRequest {
		return nil, fmt.Errorf("worker request exceeds %d-byte limit", MaxWorkerRequest)
	}

	worker, err := resolveWorker(options)
	if err != nil {
		return nil, err
	}
	workerStarted := time.Now()
	workerContext, cancel := context.WithTimeout(parent, options.WorkerTimeout)
	defer cancel()
	workerOutput, err := worker.Run(workerContext, requestBytes)
	if err != nil {
		return nil, fmt.Errorf("Torch worker failed: %w", err)
	}
	if err := workerContext.Err(); err != nil {
		return nil, fmt.Errorf("Torch worker exceeded supervision deadline: %w", err)
	}
	evidence.WorkerElapsed = time.Since(workerStarted)
	if got, err := hashFile(options.WarmStartPath, MaxWarmStart); err != nil || got != warmHash {
		return nil, fmt.Errorf("warm-start file changed during worker execution")
	}
	if got, err := hashFile(bundlePath, MaxWorkerBundle); err != nil || got != bundleHash {
		return nil, fmt.Errorf("batch bundle changed during worker execution")
	}
	postWorkerHashStarted := time.Now()
	postWorkerContent, err := hashDatasetContent(options.DatasetPath, datasetContentExpected)
	if err != nil {
		return nil, fmt.Errorf("dataset changed during post-worker content hash pass: %w", err)
	}
	evidence.PostWorkerContentHashPasses = 1
	evidence.PostWorkerContentHashFiles = len(postWorkerContent)
	evidence.PostWorkerContent = postWorkerContent
	evidence.PostWorkerContentHashElapsed = time.Since(postWorkerHashStarted)
	if err := datasetIdentityBefore.Validate(options.DatasetPath); err != nil {
		return nil, fmt.Errorf("dataset changed during worker execution")
	}
	evidence.IdentityChecks++
	profileAfterWorker, err := readProfileFile(options.ProfilePath)
	if err != nil {
		return nil, fmt.Errorf("class profile changed during worker execution: %w", err)
	}
	if profileAfterWorker.Hash != profileFileHash || profileAfterWorker.Size != profileFileSize {
		return nil, fmt.Errorf("class profile changed during worker execution")
	}
	workerResult, err := validateWorkerOutput(workerOutput, requestHash, bundleHash, warmHash, generation.ManifestHash(), options.AllowWarmStartDatasetChange, supervised, len(plan.Sequences), config)
	if err != nil {
		return nil, err
	}

	evidence.TotalElapsed = time.Since(started)
	report := makeReport(config, options, generation, verification, accumulator, profile, profileWasExisting, plan, finalWeights, profileFileHash, bundlePath, bundleHash, requestHash, warmHash, len(requestBytes), workerOutput, workerResult, datasetIdentityBefore.Hash, evidence)
	reportBytes, err := canonicalJSON(report)
	if err != nil {
		return nil, fmt.Errorf("encode report: %w", err)
	}
	if err := atomicCreate(options.ReportPath, reportBytes); err != nil {
		return nil, fmt.Errorf("publish report: %w", err)
	}
	return report, nil
}

func resolveWorker(options Options) (TorchRunner, error) {
	if options.Worker != nil {
		return options.Worker, nil
	}
	command := append([]string(nil), options.WorkerCommand...)
	if len(command) == 0 && options.TorchPythonPath != "" {
		command = []string{options.TorchPythonPath, "-m", TorchWorkerModule}
	}
	if len(command) != 3 || command[0] == "" || command[1] != "-m" || command[2] != TorchWorkerModule {
		return nil, fmt.Errorf("production worker command is fixed to [python, -m, %s]", TorchWorkerModule)
	}
	if options.TorchPythonPath != "" && command[0] != options.TorchPythonPath {
		return nil, fmt.Errorf("production worker command executable disagrees with torch-python")
	}
	return NewCommandRunner(command)
}

type datasetIdentityFile struct {
	Name    string
	Info    os.FileInfo
	Size    int64
	Mode    os.FileMode
	ModTime int64
}

type datasetIdentity struct {
	Files []datasetIdentityFile
	Hash  string
}

const (
	maxDatasetManifestBytes = 16 * 1024 * 1024
	maxDatasetShardBytes    = 512 * 1024 * 1024
	maxDatasetHashWorkers   = 8
)

type contentFileEvidence struct {
	Name   string
	Size   int64
	SHA256 string
}

type datasetContentExpectation struct {
	Files []contentFileEvidence
}

func buildDatasetContentExpectation(manifest contentFileEvidence, verification ai42dataset.VerificationReport) (datasetContentExpectation, error) {
	if manifest.Name == "" || manifest.Size < 0 || !validHash(manifest.SHA256) {
		return datasetContentExpectation{}, fmt.Errorf("dataset manifest content evidence is incomplete")
	}
	files := []contentFileEvidence{manifest}
	seen := map[string]struct{}{manifest.Name: {}}
	for _, file := range verification.Files {
		if file.Shard == "" || filepath.Base(file.Shard) != file.Shard || file.StoredBytes < 0 || !validHash(file.SHA256) {
			return datasetContentExpectation{}, fmt.Errorf("dataset verification contains invalid shard content evidence")
		}
		if _, exists := seen[file.Shard]; exists {
			return datasetContentExpectation{}, fmt.Errorf("dataset verification contains duplicate file %q", file.Shard)
		}
		seen[file.Shard] = struct{}{}
		files = append(files, contentFileEvidence{Name: file.Shard, Size: file.StoredBytes, SHA256: file.SHA256})
	}
	if len(files) != verification.Shards+1 {
		return datasetContentExpectation{}, fmt.Errorf("dataset verification content evidence is incomplete")
	}
	return datasetContentExpectation{Files: files}, nil
}

func hashDatasetContent(root string, expected datasetContentExpectation) (map[string]contentFileEvidence, error) {
	if len(expected.Files) == 0 {
		return nil, fmt.Errorf("dataset content evidence is empty")
	}
	workers := min(len(expected.Files), maxDatasetHashWorkers)
	type result struct {
		index int
		file  contentFileEvidence
		err   error
	}
	jobs := make(chan int)
	results := make(chan result, len(expected.Files))
	var wait sync.WaitGroup
	wait.Add(workers)
	for worker := 0; worker < workers; worker++ {
		go func() {
			defer wait.Done()
			for index := range jobs {
				file := expected.Files[index]
				if filepath.Base(file.Name) != file.Name || file.Name == "." || file.Name == ".." {
					results <- result{index: index, err: fmt.Errorf("dataset content file name %q is not a plain file name", file.Name)}
					continue
				}
				limit := maxDatasetShardBytes
				if file.Name == "manifest.json" {
					limit = maxDatasetManifestBytes
				}
				actual, err := hashFileEvidence(filepath.Join(root, file.Name), limit)
				if err == nil && (actual.Size != file.Size || actual.SHA256 != file.SHA256) {
					err = fmt.Errorf("%s content/size mismatch", file.Name)
				}
				results <- result{index: index, file: actual, err: err}
			}
		}()
	}
	go func() {
		defer close(jobs)
		for index := range expected.Files {
			jobs <- index
		}
	}()
	wait.Wait()
	close(results)
	actual := make(map[string]contentFileEvidence, len(expected.Files))
	var firstErr error
	for item := range results {
		if item.err != nil && firstErr == nil {
			firstErr = item.err
		}
		if item.err == nil {
			actual[expected.Files[item.index].Name] = item.file
		}
	}
	if firstErr != nil {
		return nil, firstErr
	}
	if len(actual) != len(expected.Files) {
		return nil, fmt.Errorf("dataset content hash pass did not cover every expected file")
	}
	return actual, nil
}

func openDataset(path, expected string) (*ai42dataset.Generation, *datasetIdentity, contentFileEvidence, error) {
	generation, err := ai42dataset.OpenGenerationWithOptions(path, ai42dataset.OpenOptions{ExpectedManifestHash: expected, DeferShardHashing: true})
	if err != nil {
		return nil, nil, contentFileEvidence{}, fmt.Errorf("open durable-v2 dataset: %w", err)
	}
	identity, err := captureDatasetIdentity(path)
	if err != nil {
		return nil, nil, contentFileEvidence{}, err
	}
	manifest, err := hashFileEvidence(filepath.Join(path, "manifest.json"), maxDatasetManifestBytes)
	if err != nil {
		return nil, nil, contentFileEvidence{}, fmt.Errorf("hash dataset manifest: %w", err)
	}
	return generation, identity, manifest, nil
}

func captureDatasetIdentity(path string) (*datasetIdentity, error) {
	root, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, err
	}
	files := make([]datasetIdentityFile, 0)
	for _, entry := range entries {
		if entry.Name() != "manifest.json" && filepath.Ext(entry.Name()) != ".a42" {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			return nil, err
		}
		if !info.Mode().IsRegular() {
			return nil, fmt.Errorf("dataset file %s is not regular", entry.Name())
		}
		files = append(files, datasetIdentityFile{Name: entry.Name(), Info: info, Size: info.Size(), Mode: info.Mode(), ModTime: info.ModTime().UnixNano()})
	}
	if len(files) < 2 {
		return nil, fmt.Errorf("dataset identity found %d manifest/shard files", len(files))
	}
	sort.Slice(files, func(i, j int) bool { return files[i].Name < files[j].Name })
	payload := make([]any, len(files))
	for index, value := range files {
		payload[index] = map[string]any{"name": value.Name, "bytes": value.Size, "mode": uint32(value.Mode), "mtime_ns": value.ModTime}
	}
	encoded, err := canonicalJSON(payload)
	if err != nil {
		return nil, err
	}
	return &datasetIdentity{Files: files, Hash: sha256Hex(encoded)}, nil
}

func (identity *datasetIdentity) Validate(path string) error {
	if identity == nil {
		return fmt.Errorf("missing dataset identity")
	}
	current, err := captureDatasetIdentity(path)
	if err != nil {
		return err
	}
	if len(current.Files) != len(identity.Files) {
		return fmt.Errorf("dataset file set changed")
	}
	for index, before := range identity.Files {
		after := current.Files[index]
		if before.Name != after.Name || before.Size != after.Size || before.Mode != after.Mode || before.ModTime != after.ModTime || !os.SameFile(before.Info, after.Info) {
			return fmt.Errorf("dataset file %q identity changed", before.Name)
		}
	}
	return nil
}

func resolveTargetedRows(options Options, generation *ai42dataset.Generation) (TargetedRowReader, error) {
	if options.TargetedRows != nil {
		return options.TargetedRows, nil
	}
	if source, ok := any(generation).(TargetedRowReader); ok {
		return source, nil
	}
	return nil, fmt.Errorf("durable-v2 reader lacks bounded target API: *ai42dataset.Generation must implement ReadTargetRows(context.Context, map[string][][2]int, func(ai42dataset.Row) error) (int, error)")
}

func resolveSplitEvidence(options Options, generation *ai42dataset.Generation) (VerifiedSplitReader, error) {
	if options.SplitEvidence != nil {
		return options.SplitEvidence, nil
	}
	if source, ok := any(generation).(VerifiedSplitReader); ok {
		return source, nil
	}
	return nil, fmt.Errorf("durable-v2 reader lacks split recomputation evidence API: *ai42dataset.Generation must implement VerifiedSplitMatchIDs() (map[string][]string, error)")
}

func validateVerifiedSplit(source VerifiedSplitReader, matches []matchInfo, expectedHash string) error {
	splits, err := source.VerifiedSplitMatchIDs()
	if err != nil {
		return fmt.Errorf("obtain reader split recomputation evidence: %w", err)
	}
	if len(splits) != 2 {
		return fmt.Errorf("reader split recomputation evidence must contain exactly train and validation")
	}
	train, trainOK := splits["train"]
	validation, validationOK := splits["validation"]
	if !trainOK || !validationOK {
		return fmt.Errorf("reader split recomputation evidence must contain exactly train and validation")
	}
	expectedTrain, expectedValidation := make([]string, 0), make([]string, 0)
	for _, match := range matches {
		switch match.Split {
		case "train":
			expectedTrain = append(expectedTrain, match.ID)
		case "validation":
			expectedValidation = append(expectedValidation, match.ID)
		default:
			return fmt.Errorf("verified match %q has unsupported split %q", match.ID, match.Split)
		}
	}
	if !equalStringSlices(train, expectedTrain) || !equalStringSlices(validation, expectedValidation) {
		return fmt.Errorf("reader split recomputation evidence disagrees with verified match order")
	}
	actualHash, err := canonicalHash(map[string]any{"train": stringsAny(train), "validation": stringsAny(validation)})
	if err != nil {
		return err
	}
	if actualHash != expectedHash {
		return fmt.Errorf("reader split recomputation hash mismatch")
	}
	return nil
}

func equalStringSlices(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func selectedShardNames(generation *ai42dataset.Generation, plan batchPlan) ([]string, error) {
	if generation == nil {
		return nil, fmt.Errorf("targeted shard reread requires a generation")
	}
	wantedMatches := make(map[string]struct{}, len(plan.Sequences))
	for _, sequence := range plan.Sequences {
		if sequence.MatchID == "" {
			return nil, fmt.Errorf("targeted batch contains an empty match ID")
		}
		wantedMatches[sequence.MatchID] = struct{}{}
	}
	if len(wantedMatches) == 0 {
		return nil, fmt.Errorf("targeted batch contains no matches")
	}
	seenShards := map[string]struct{}{}
	result := make([]string, 0, len(wantedMatches))
	for _, match := range generation.Matches() {
		if _, wanted := wantedMatches[match.MatchID]; !wanted {
			continue
		}
		if match.Shard == "" {
			return nil, fmt.Errorf("targeted match %q has no shard", match.MatchID)
		}
		if _, seen := seenShards[match.Shard]; seen {
			continue
		}
		seenShards[match.Shard] = struct{}{}
		result = append(result, match.Shard)
	}
	if len(result) == 0 {
		return nil, fmt.Errorf("targeted batch matches are absent from verified generation")
	}
	return result, nil
}

func hashFile(path string, limit int) (string, error) {
	evidence, err := hashFileEvidence(path, limit)
	if err != nil {
		return "", err
	}
	return evidence.SHA256, nil
}

func hashFileEvidence(path string, limit int) (contentFileEvidence, error) {
	if limit <= 0 {
		return contentFileEvidence{}, fmt.Errorf("hash limit must be positive")
	}
	lstat, err := os.Lstat(path)
	if err != nil {
		return contentFileEvidence{}, err
	}
	if lstat.Mode()&os.ModeSymlink != 0 {
		return contentFileEvidence{}, fmt.Errorf("%s must not be a symlink", path)
	}
	if !lstat.Mode().IsRegular() {
		return contentFileEvidence{}, fmt.Errorf("%s is not a regular file", path)
	}
	file, err := os.Open(path)
	if err != nil {
		return contentFileEvidence{}, err
	}
	defer file.Close()
	before, err := file.Stat()
	if err != nil {
		return contentFileEvidence{}, err
	}
	if !before.Mode().IsRegular() {
		return contentFileEvidence{}, fmt.Errorf("%s is not a regular file", path)
	}
	if !os.SameFile(lstat, before) {
		return contentFileEvidence{}, fmt.Errorf("%s changed while opening", path)
	}
	if before.Size() > int64(limit) {
		return contentFileEvidence{}, fmt.Errorf("%s exceeds %d-byte limit", path, limit)
	}
	digest := sha256.New()
	total, err := io.CopyBuffer(digest, &io.LimitedReader{R: file, N: int64(limit) + 1}, make([]byte, 128*1024))
	if err != nil {
		return contentFileEvidence{}, err
	}
	if total != before.Size() || total > int64(limit) {
		return contentFileEvidence{}, fmt.Errorf("%s changed while hashing or exceeds %d-byte limit", path, limit)
	}
	after, err := file.Stat()
	if err != nil {
		return contentFileEvidence{}, err
	}
	if after.Size() != before.Size() || after.ModTime() != before.ModTime() || after.Mode() != before.Mode() || !os.SameFile(before, after) {
		return contentFileEvidence{}, fmt.Errorf("%s changed while hashing", path)
	}
	return contentFileEvidence{Name: filepath.Base(path), Size: before.Size(), SHA256: fmt.Sprintf("%x", digest.Sum(nil))}, nil
}

func profileBytes(profile Profile) ([]byte, error) {
	unsigned := profileUnsigned(
		profile.DatasetManifestHash, profile.TrainMatchIDs, profile.TrainMatchIDsHash,
		profile.Counts, profile.Weights, profile.SupervisionControllers,
	)
	unsigned["profile_hash"] = profile.ProfileHash
	return canonicalJSON(unsigned)
}

type profileFileEvidence struct {
	Raw  []byte
	Hash string
	Size int64
}

func readProfileFile(path string) (profileFileEvidence, error) {
	lstat, err := os.Lstat(path)
	if err != nil {
		return profileFileEvidence{}, err
	}
	if lstat.Mode()&os.ModeSymlink != 0 {
		return profileFileEvidence{}, fmt.Errorf("profile must not be a symlink")
	}
	if !lstat.Mode().IsRegular() {
		return profileFileEvidence{}, fmt.Errorf("profile must be a regular file")
	}
	file, err := os.Open(path)
	if err != nil {
		return profileFileEvidence{}, err
	}
	defer file.Close()
	before, err := file.Stat()
	if err != nil {
		return profileFileEvidence{}, err
	}
	if !before.Mode().IsRegular() {
		return profileFileEvidence{}, fmt.Errorf("profile is not a regular file")
	}
	if !os.SameFile(lstat, before) {
		return profileFileEvidence{}, fmt.Errorf("profile changed while opening")
	}
	if before.Size() > 8*1024*1024 {
		return profileFileEvidence{}, fmt.Errorf("profile exceeds %d-byte limit", 8*1024*1024)
	}
	raw, err := io.ReadAll(io.LimitReader(file, 8*1024*1024+1))
	if err != nil {
		return profileFileEvidence{}, err
	}
	if int64(len(raw)) != before.Size() || len(raw) > 8*1024*1024 {
		return profileFileEvidence{}, fmt.Errorf("profile changed while reading or exceeds byte limit")
	}
	after, err := file.Stat()
	if err != nil {
		return profileFileEvidence{}, err
	}
	if after.Size() != before.Size() || after.ModTime() != before.ModTime() || after.Mode() != before.Mode() || !os.SameFile(before, after) {
		return profileFileEvidence{}, fmt.Errorf("profile changed while reading")
	}
	return profileFileEvidence{Raw: raw, Hash: sha256Hex(raw), Size: int64(len(raw))}, nil
}

func publishOrValidateProfile(path, outputDirectory, suppliedHash string, expected Profile, encoded []byte) (Profile, bool, string, int64, error) {
	info, statErr := os.Lstat(path)
	if statErr == nil {
		if info.Mode()&os.ModeSymlink != 0 {
			return Profile{}, false, "", 0, fmt.Errorf("profile must not be a symlink")
		}
		if !info.Mode().IsRegular() {
			return Profile{}, false, "", 0, fmt.Errorf("profile must be a regular file")
		}
		if suppliedHash == "" {
			return Profile{}, false, "", 0, fmt.Errorf("profile-hash is required when profile already exists")
		}
		file, err := readProfileFile(path)
		if err != nil {
			return Profile{}, false, "", 0, fmt.Errorf("read profile: %w", err)
		}
		if file.Hash != suppliedHash {
			return Profile{}, false, "", 0, fmt.Errorf("profile file hash does not match --profile-hash")
		}
		authoritative, err := loadExistingProfile(file.Raw, expected)
		return authoritative, true, file.Hash, file.Size, err
	}
	if !os.IsNotExist(statErr) {
		return Profile{}, false, "", 0, statErr
	}
	if !pathWithin(outputDirectory, path) {
		return Profile{}, false, "", 0, fmt.Errorf("profile may only be created inside output")
	}
	encodedHash := sha256Hex(encoded)
	if suppliedHash != "" && suppliedHash != encodedHash {
		return Profile{}, false, "", 0, fmt.Errorf("generated profile file hash does not match --profile-hash")
	}
	if err := atomicCreate(path, encoded); err != nil {
		if raceInfo, raceErr := os.Lstat(path); raceErr == nil && raceInfo.Mode().IsRegular() && raceInfo.Mode()&os.ModeSymlink == 0 {
			if suppliedHash == "" {
				return Profile{}, false, "", 0, fmt.Errorf("profile appeared during atomic create; --profile-hash is required")
			}
			file, readErr := readProfileFile(path)
			if readErr != nil {
				return Profile{}, false, "", 0, readErr
			}
			if file.Hash != suppliedHash {
				return Profile{}, false, "", 0, fmt.Errorf("profile file hash does not match --profile-hash")
			}
			authoritative, validateErr := loadExistingProfile(file.Raw, expected)
			return authoritative, true, file.Hash, file.Size, validateErr
		}
		return Profile{}, false, "", 0, err
	}
	file, err := readProfileFile(path)
	if err != nil {
		return Profile{}, false, "", 0, err
	}
	if file.Hash != encodedHash {
		return Profile{}, false, "", 0, fmt.Errorf("published profile bytes changed during atomic create")
	}
	return expected, false, file.Hash, file.Size, nil
}

func atomicCreate(path string, payload []byte) error {
	if err := ensureSafeDirectory(filepath.Dir(path)); err != nil {
		return err
	}
	if err := rejectExistingDestination(path, "artifact"); err != nil {
		return err
	}
	temporary, err := os.CreateTemp(filepath.Dir(path), ".ai42-profile-*")
	if err != nil {
		return err
	}
	name := temporary.Name()
	defer os.Remove(name)
	if _, err := temporary.Write(payload); err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Sync(); err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	if err := os.Link(name, path); err != nil {
		return err
	}
	return nil
}

func ensureSafeDirectory(path string) error {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	absolute = filepath.Clean(absolute)
	ancestors := pathAncestors(absolute)
	for index := len(ancestors) - 1; index >= 0; index-- {
		candidate := ancestors[index]
		info, err := os.Lstat(candidate)
		if os.IsNotExist(err) {
			if err := os.Mkdir(candidate, 0o755); err != nil && !os.IsExist(err) {
				return fmt.Errorf("create directory %s: %w", candidate, err)
			}
			info, err = os.Lstat(candidate)
		}
		if err != nil {
			return err
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("path component %s must not be a symlink", candidate)
		}
		if !info.IsDir() {
			return fmt.Errorf("path component %s is not a directory", candidate)
		}
	}
	return nil
}

func pathAncestors(path string) []string {
	result := []string{}
	for current := filepath.Clean(path); ; current = filepath.Dir(current) {
		result = append(result, current)
		parent := filepath.Dir(current)
		if parent == current {
			break
		}
	}
	return result
}

func validateNoSymlinkComponents(path string, allowMissingLeaf bool) error {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	ancestors := pathAncestors(filepath.Clean(absolute))
	for index := len(ancestors) - 1; index >= 0; index-- {
		candidate := ancestors[index]
		info, err := os.Lstat(candidate)
		if os.IsNotExist(err) {
			if allowMissingLeaf && index == 0 {
				return nil
			}
			return fmt.Errorf("path component %s does not exist", candidate)
		}
		if err != nil {
			return err
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("path component %s must not be a symlink", candidate)
		}
	}
	return nil
}

func rejectExistingDestination(path, name string) error {
	info, err := os.Lstat(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("inspect %s: %w", name, err)
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("%s must not be an existing symlink", name)
	}
	return fmt.Errorf("%s already exists; refusing to overwrite", name)
}

func resolveOutputArtifactPath(outputDirectory, requested, defaultName, name string) (string, error) {
	if requested == "" {
		requested = filepath.Join(outputDirectory, defaultName)
	}
	abs, err := filepath.Abs(requested)
	if err != nil {
		return "", err
	}
	abs = filepath.Clean(abs)
	if !pathWithin(outputDirectory, abs) {
		return "", fmt.Errorf("%s path must resolve inside output", name)
	}
	if abs == filepath.Clean(outputDirectory) {
		return "", fmt.Errorf("%s path must name a file", name)
	}
	if err := ensureSafeDirectory(filepath.Dir(abs)); err != nil {
		return "", err
	}
	if err := validateNoSymlinkComponents(abs, true); err != nil {
		return "", err
	}
	return abs, nil
}

func resolveProfileArtifactPath(outputDirectory, requested, defaultName string) (string, error) {
	if requested == "" {
		requested = filepath.Join(outputDirectory, defaultName)
	}
	abs, err := filepath.Abs(requested)
	if err != nil {
		return "", err
	}
	abs = filepath.Clean(abs)
	if pathWithin(outputDirectory, abs) {
		if abs == filepath.Clean(outputDirectory) {
			return "", fmt.Errorf("profile path must name a file")
		}
		if err := ensureSafeDirectory(filepath.Dir(abs)); err != nil {
			return "", err
		}
		if err := validateNoSymlinkComponents(abs, true); err != nil {
			return "", err
		}
		return abs, nil
	}
	if err := validateNoSymlinkComponents(abs, false); err != nil {
		return "", fmt.Errorf("external profile path is not a safe existing file: %w", err)
	}
	return abs, nil
}

func pathWithin(root, candidate string) bool {
	root = filepath.Clean(root)
	candidate = filepath.Clean(candidate)
	rel, err := filepath.Rel(root, candidate)
	if err != nil || filepath.IsAbs(rel) || rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		return false
	}
	return rel != "."
}

func weightsAsAny(value map[string][]float64) map[string]any {
	result := map[string]any{}
	for _, head := range profileHeads {
		values := make([]any, len(value[head]))
		for index, item := range value[head] {
			values[index] = item
		}
		result[head] = values
	}
	return result
}
