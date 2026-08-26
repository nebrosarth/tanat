package ai42preflight

import (
	"bytes"
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"tanatserver/internal/ai42dataset"
	"tanatserver/internal/battleserver"
)

func TestCanonicalJSONMatchesTorchNumberConvention(t *testing.T) {
	encoded, err := canonicalJSON(map[string]any{"integer": 2, "one": 1.0, "small": 0.0003, "unicode": "é"})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(encoded), `{"integer":2,"one":1.0,"small":0.0003,"unicode":"é"}`; got != want {
		t.Fatalf("canonical=%s, want %s", got, want)
	}
	value, err := decodeCanonical(encoded, "fixture", 1024)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := value.(map[string]any); !ok {
		t.Fatalf("decoded type=%T", value)
	}
}

func TestPythonParityHashesForOrderedIDsAndBatchPlan(t *testing.T) {
	idHash, err := canonicalHash([]string{"match-a", "match-b"})
	if err != nil {
		t.Fatal(err)
	}
	if idHash != "a1000c5c1503809fea54c95bea25926b5358573ce42794970cad25dc1e26f18e" {
		t.Fatalf("ordered ID hash=%s", idHash)
	}
	plan, err := makeBatchPlan([]matchInfo{
		{ID: "match-a", Split: "train", Scenario: "test", TickCount: 1},
		{ID: "match-b", Split: "train", Scenario: "test", TickCount: 1},
		{ID: "match-v", Split: "validation", Scenario: "test", TickCount: 1},
	}, map[string]*matchEvidence{
		"match-a": {SupervisedByWindow: []uint16{1}},
		"match-b": {SupervisedByWindow: []uint16{1}},
		"match-v": {SupervisedByWindow: []uint16{0}},
	}, Config{Seed: 4242, SequenceLength: 64, BatchSize: 8, ValidationProbeLimit: 1})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := strings.Join(plan.TrainMatchIDs, ","), "match-a,match-b"; got != want {
		t.Fatalf("ranked train IDs=%s, want %s", got, want)
	}
	for name, want := range map[string]string{
		"plan":       "b1ec2f03ecc8027d8c3cecfcd0a65efd961fbcd42d29eaa0a28dfc07826fd8ef",
		"validation": "5747bab193b724249cbc6f51315797d18f10b0f0759e2294fb41e8f4c1cf60a8",
		"split":      "8d587f040eaaa11031e6764410d5562060b13b73aa76862e74ad00db7627b6a4",
	} {
		got := map[string]string{"plan": plan.Hash, "validation": plan.ValidationHash, "split": plan.SplitHash}[name]
		if got != want {
			t.Fatalf("%s hash=%s, want %s", name, got, want)
		}
	}
}

func TestMakeBatchPlanSkipsFirstEmptyPhysicalBatch(t *testing.T) {
	plan, err := makeBatchPlan(
		[]matchInfo{{ID: "match-a", Split: "train", Scenario: "test", TickCount: 1}},
		map[string]*matchEvidence{"match-a": {SupervisedByWindow: []uint16{1 << 2}}},
		Config{Seed: 7, SequenceLength: 1, BatchSize: 2},
	)
	if err != nil {
		t.Fatal(err)
	}
	if plan.PhysicalBatches != 2 || plan.EmptyBatches != 1 || plan.SequencesScanned != 4 {
		t.Fatalf("batch scan counters=%d/%d/%d", plan.PhysicalBatches, plan.EmptyBatches, plan.SequencesScanned)
	}
	if len(plan.Sequences) != 2 || plan.Sequences[0].Hero != 2 || plan.Sequences[1].Hero != 3 {
		t.Fatalf("selected sequences=%+v", plan.Sequences)
	}
}

func TestMakeBatchPlanSupervisesOnlySelectedController(t *testing.T) {
	info := matchInfo{ID: "match-a", Split: "train", Scenario: "test", TickCount: 1}
	info.ControllerBySlot[7] = 3
	plan, err := makeBatchPlan(
		[]matchInfo{info},
		map[string]*matchEvidence{"match-a": {SupervisedByWindow: []uint16{1 << 7}}},
		Config{Seed: 7, SequenceLength: 1, BatchSize: 1, SupervisionControllers: []uint8{3}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Sequences) != 1 || plan.Sequences[0].Hero != 7 {
		t.Fatalf("selected sequences=%+v, want only controller-3 hero 7", plan.Sequences)
	}
}

func TestTargetProfileWeightsAreUniformAcrossExchangeableSlots(t *testing.T) {
	counts := zeroCounts()
	counts["target"][0] = 100
	counts["target"][17] = 1
	profile, err := buildProfile(strings.Repeat("a", 64), []string{"train-a"}, counts, []uint8{3})
	if err != nil {
		t.Fatal(err)
	}
	for index, weight := range profile.Weights["target"] {
		if weight != 1 {
			t.Fatalf("target slot %d weight=%v, want 1", index, weight)
		}
	}
	if _, err := mergeClassWeights(profile, map[string][]float64{"target": profile.Weights["target"]}); err == nil || !strings.Contains(err.Error(), "permutation-equivariant") {
		t.Fatalf("target override error=%v", err)
	}
}

func TestEligibilityEvidenceDoesNotOvercountControlMasks(t *testing.T) {
	accumulator := newAccumulator(1)
	row := testTargetRow("match-a", 0)
	for index := range row.TeacherStatus {
		row.TeacherStatus[index] = 0
		row.TeacherAction[index] = ai42dataset.Action{}
	}
	row.TeacherStatus[2] = 1
	row.TeacherAction[2] = ai42dataset.Action{Kind: 1}
	if err := accumulator.onRow(row); err != nil {
		t.Fatal(err)
	}
	evidence := accumulator.Evidence["match-a"]
	if evidence.SupervisedByWindow[0] != 1<<2 {
		t.Fatalf("supervision bits=%010b", evidence.SupervisedByWindow[0])
	}
	total := 0
	for _, count := range evidence.Counts["control"] {
		total += count
	}
	if total != 1 {
		t.Fatalf("control count=%d, want 1", total)
	}
}

func TestLoadConfigRejectsUnknownAndAcceptsQ3Overrides(t *testing.T) {
	q3 := filepath.Join("..", "..", "ai40", "config", "ai42_bc_training_q3.json")
	config, err := LoadConfig(q3)
	if err != nil {
		t.Fatal(err)
	}
	if config.Learner.LearningRate != 1e-4 || len(config.Learner.ClassWeightOverrides["control"]) != 4 {
		t.Fatalf("unexpected Q3 config: %+v", config.Learner)
	}
	q4 := filepath.Join("..", "..", "ai40", "config", "ai42_bc_training_q4.json")
	q4Config, err := LoadConfig(q4)
	if err != nil {
		t.Fatal(err)
	}
	wantHeadWeights := map[string]float64{"control": 1, "kind": 1.5, "target": 1, "offset": 2, "anchor": 1}
	if !reflect.DeepEqual(q4Config.Learner.HeadWeights, wantHeadWeights) {
		t.Fatalf("unexpected Q4 head weights: %#v", q4Config.Learner.HeadWeights)
	}
	path := filepath.Join(t.TempDir(), "unknown.json")
	if err := os.WriteFile(path, []byte(`{"protocol_version":13,"model":{"hidden_size":1,"model_width":1,"entity_layers":1,"num_heads":1,"ff_multiplier":1,"timing_bins":1},"recurrent_batch":{"sequence_length":1,"batch_size":1},"learner":{"class_balance_power":0.5,"max_gradient_norm":1,"timing_loss_enabled":false,"optimizer_step_allowed_in_preflight":false,"unknown":true}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadConfig(path); err == nil || !strings.Contains(err.Error(), "unknown") {
		t.Fatalf("unknown config error=%v", err)
	}
}

func TestRunPublishesProfileBundleAndReportWithInjectedRunner(t *testing.T) {
	dataset := writeTestGeneration(t)
	warmPath := filepath.Join(t.TempDir(), "warm.pt")
	if err := os.WriteFile(warmPath, []byte("checkpoint"), 0o600); err != nil {
		t.Fatal(err)
	}
	outputPath := filepath.Join(t.TempDir(), "run")
	runner := &fakeRunner{datasetHash: dataset.ManifestHash}
	report, err := Run(context.Background(), runOptions(dataset, warmPath, outputPath, runner))
	if err != nil {
		t.Fatal(err)
	}
	if report["replay_scope"] != ReplayScope || report["python_fallback"] != false {
		t.Fatalf("unexpected report evidence: %#v", report)
	}
	for _, name := range []string{"class_profile_ai42.json", "preflight_batch_bundle.json", "preflight_report.json"} {
		if _, err := os.Stat(filepath.Join(outputPath, name)); err != nil {
			t.Fatalf("missing %s: %v", name, err)
		}
	}
	profileRaw, err := os.ReadFile(filepath.Join(outputPath, "class_profile_ai42.json"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := decodeCanonical(profileRaw, "profile", 8*1024*1024); err != nil {
		t.Fatal(err)
	}
	if runner.calls != 1 {
		t.Fatalf("worker calls=%d", runner.calls)
	}
	requestValue, err := decodeCanonical(runner.lastRequest, "request", MaxWorkerRequest)
	if err != nil {
		t.Fatal(err)
	}
	requestRoot, err := object(requestValue, "request")
	if err != nil {
		t.Fatal(err)
	}
	requestLearner, err := object(requestRoot["learner"], "request.learner")
	if err != nil {
		t.Fatal(err)
	}
	requestHeadWeights, err := canonicalJSON(requestLearner["head_weights"])
	if err != nil {
		t.Fatal(err)
	}
	reportHeadWeights, err := canonicalJSON(report["head_weights"])
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(requestHeadWeights, reportHeadWeights) {
		t.Fatalf("request head weights=%s, report=%s", requestHeadWeights, reportHeadWeights)
	}
	if report["head_weights_hash"] != sha256Hex(reportHeadWeights) {
		t.Fatalf("head weights hash=%v", report["head_weights_hash"])
	}
	native := report["native_evidence"].(map[string]any)
	counters := native["io_counters"].(map[string]any)
	if counters["open_generation_calls"] != 1 || counters["full_verify_passes"] != 1 || counters["targeted_read_calls"] != 1 || counters["targeted_shard_reread_passes"] != 1 || counters["post_worker_content_hash_passes"] != 1 {
		t.Fatalf("unexpected I/O counters: %#v", counters)
	}
	invariants := report["invariants"].(map[string]any)
	if invariants["no_redundant_full_generation_pass"] != true {
		t.Fatalf("full generation pass evidence=%#v", invariants)
	}
	if invariants["deterministic_split_checked"] != true {
		t.Fatalf("split recomputation was not reported")
	}
}

func TestRunUsesAuthoritativeExistingProfileWithFloatRoundoff(t *testing.T) {
	dataset := writeTestGeneration(t)
	warmPath := filepath.Join(t.TempDir(), "warm.pt")
	if err := os.WriteFile(warmPath, []byte("checkpoint"), 0o600); err != nil {
		t.Fatal(err)
	}
	outputPath := filepath.Join(t.TempDir(), "run")
	profilePath := filepath.Join(outputPath, "class_profile_ai42.json")
	if _, err := Run(context.Background(), runOptions(dataset, warmPath, outputPath, &fakeRunner{datasetHash: dataset.ManifestHash})); err != nil {
		t.Fatal(err)
	}

	profileRaw, err := os.ReadFile(profilePath)
	if err != nil {
		t.Fatal(err)
	}
	profileValue, err := decodeCanonical(profileRaw, "profile", 8*1024*1024)
	if err != nil {
		t.Fatal(err)
	}
	profileRoot, err := object(profileValue, "profile")
	if err != nil {
		t.Fatal(err)
	}
	weights, err := object(profileRoot["weights"], "profile.weights")
	if err != nil {
		t.Fatal(err)
	}
	adjusted := false
	for _, head := range profileHeads {
		values, ok := weights[head].([]any)
		if !ok {
			t.Fatalf("profile.weights[%s] has type %T", head, weights[head])
		}
		for index, item := range values {
			value, err := asNumber(item, fmt.Sprintf("profile.weights[%s][%d]", head, index))
			if err != nil {
				t.Fatal(err)
			}
			if value > 0 {
				values[index] = value + 5e-7
				adjusted = true
				break
			}
		}
		if adjusted {
			break
		}
	}
	if !adjusted {
		t.Fatal("profile has no supported class weight to adjust")
	}
	delete(profileRoot, "profile_hash")
	unsigned, err := canonicalJSON(profileRoot)
	if err != nil {
		t.Fatal(err)
	}
	profileRoot["profile_hash"] = sha256Hex(unsigned)
	authoritativeRaw, err := canonicalJSON(profileRoot)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(profilePath, authoritativeRaw, 0o600); err != nil {
		t.Fatal(err)
	}
	exactProfileHash := sha256Hex(authoritativeRaw)
	wantProfileFile := append([]byte(nil), authoritativeRaw...)
	wantHash, err := asString(profileRoot["profile_hash"], "profile.profile_hash")
	if err != nil {
		t.Fatal(err)
	}
	wantWeights, err := canonicalJSON(weights)
	if err != nil {
		t.Fatal(err)
	}

	runner := &fakeRunner{datasetHash: dataset.ManifestHash}
	secondOptions := runOptions(dataset, warmPath, filepath.Join(t.TempDir(), "second-run"), runner)
	secondOptions.ProfilePath = profilePath
	secondOptions.ProfileHash = exactProfileHash
	secondOptions.ReportPath = filepath.Join(secondOptions.OutputPath, "authoritative-report.json")
	report, err := Run(context.Background(), secondOptions)
	if err != nil {
		t.Fatal(err)
	}
	gotProfile, ok := report["profile"].(map[string]any)
	if !ok {
		t.Fatalf("report profile type=%T", report["profile"])
	}
	if gotProfile["hash"] != wantHash {
		t.Fatalf("report profile hash=%v, want %s", gotProfile["hash"], wantHash)
	}
	gotWeights, err := canonicalJSON(gotProfile["weights"])
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(gotWeights, wantWeights) {
		t.Fatalf("report weights=%s, want %s", gotWeights, wantWeights)
	}
	gotClassWeights, err := canonicalJSON(report["class_weights"])
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(gotClassWeights, wantWeights) {
		t.Fatalf("report class_weights=%s, want %s", gotClassWeights, wantWeights)
	}
	requestValue, err := decodeCanonical(runner.lastRequest, "request", MaxWorkerRequest)
	if err != nil {
		t.Fatal(err)
	}
	requestRoot, err := object(requestValue, "request")
	if err != nil {
		t.Fatal(err)
	}
	learner, err := object(requestRoot["learner"], "request.learner")
	if err != nil {
		t.Fatal(err)
	}
	requestWeights, err := canonicalJSON(learner["class_weights"])
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(requestWeights, wantWeights) {
		t.Fatalf("request class_weights=%s, want %s", requestWeights, wantWeights)
	}
	gotProfileFile, err := os.ReadFile(profilePath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(gotProfileFile, wantProfileFile) {
		t.Fatal("authoritative profile file changed")
	}
}

func TestRunRejectsDatasetMutationFromWorker(t *testing.T) {
	dataset := writeTestGeneration(t)
	warmPath := filepath.Join(t.TempDir(), "warm.pt")
	if err := os.WriteFile(warmPath, []byte("checkpoint"), 0o600); err != nil {
		t.Fatal(err)
	}
	outputPath := filepath.Join(t.TempDir(), "run")
	runner := &fakeRunner{datasetHash: dataset.ManifestHash, mutateDataset: dataset.Path}
	if _, err := Run(context.Background(), runOptions(dataset, warmPath, outputPath, runner)); err == nil || !strings.Contains(err.Error(), "dataset changed") {
		t.Fatalf("mutation error=%v", err)
	}
	if _, err := os.Stat(filepath.Join(outputPath, "preflight_report.json")); !os.IsNotExist(err) {
		t.Fatalf("report should not publish after mutation: %v", err)
	}
}

func TestRunRejectsSameSizeDatasetMutationAfterWorker(t *testing.T) {
	dataset := writeTestGeneration(t)
	warmPath := filepath.Join(t.TempDir(), "warm.pt")
	if err := os.WriteFile(warmPath, []byte("checkpoint"), 0o600); err != nil {
		t.Fatal(err)
	}
	outputPath := filepath.Join(t.TempDir(), "run")
	runner := &fakeRunner{datasetHash: dataset.ManifestHash, mutateDataset: dataset.Path, mutateDatasetSameSize: true}
	_, err := Run(context.Background(), runOptions(dataset, warmPath, outputPath, runner))
	if err == nil || !strings.Contains(err.Error(), "post-worker content hash") {
		t.Fatalf("same-size mutation error=%v", err)
	}
}

func TestRunRejectsProfileMutationAfterWorker(t *testing.T) {
	dataset := writeTestGeneration(t)
	warmPath := filepath.Join(t.TempDir(), "warm.pt")
	if err := os.WriteFile(warmPath, []byte("checkpoint"), 0o600); err != nil {
		t.Fatal(err)
	}
	outputPath := filepath.Join(t.TempDir(), "run")
	runner := &fakeRunner{datasetHash: dataset.ManifestHash, mutateProfile: filepath.Join(outputPath, "class_profile_ai42.json")}
	_, err := Run(context.Background(), runOptions(dataset, warmPath, outputPath, runner))
	if err == nil || !strings.Contains(err.Error(), "class profile changed") {
		t.Fatalf("profile mutation error=%v", err)
	}
}

func TestRunRejectsExistingProfileWithoutExactFileHash(t *testing.T) {
	dataset := writeTestGeneration(t)
	warmPath := filepath.Join(t.TempDir(), "warm.pt")
	if err := os.WriteFile(warmPath, []byte("checkpoint"), 0o600); err != nil {
		t.Fatal(err)
	}
	firstOutput := filepath.Join(t.TempDir(), "first")
	if _, err := Run(context.Background(), runOptions(dataset, warmPath, firstOutput, &fakeRunner{datasetHash: dataset.ManifestHash})); err != nil {
		t.Fatal(err)
	}
	profileRaw, err := os.ReadFile(filepath.Join(firstOutput, "class_profile_ai42.json"))
	if err != nil {
		t.Fatal(err)
	}
	options := runOptions(dataset, warmPath, filepath.Join(t.TempDir(), "second"), &fakeRunner{datasetHash: dataset.ManifestHash})
	options.ProfilePath = filepath.Join(firstOutput, "class_profile_ai42.json")
	if _, err := Run(context.Background(), options); err == nil || !strings.Contains(err.Error(), "profile-hash is required") {
		t.Fatalf("missing profile hash error=%v", err)
	}
	wrongHash := options
	wrongHash.OutputPath = filepath.Join(t.TempDir(), "wrong-hash")
	wrongHash.ReportPath = ""
	wrongHash.ProfileHash = strings.Repeat("f", 64)
	if _, err := Run(context.Background(), wrongHash); err == nil || !strings.Contains(err.Error(), "does not match --profile-hash") {
		t.Fatalf("wrong profile hash error=%v", err)
	}
	generatedHash := sha256Hex(profileRaw)
	generated := runOptions(dataset, warmPath, filepath.Join(t.TempDir(), "generated-with-pin"), &fakeRunner{datasetHash: dataset.ManifestHash})
	generated.ProfileHash = generatedHash
	if _, err := Run(context.Background(), generated); err != nil {
		t.Fatalf("generated profile hash error=%v", err)
	}
}

func TestRunRejectsReportOutsideOutputAndExistingReport(t *testing.T) {
	dataset := writeTestGeneration(t)
	warmPath := filepath.Join(t.TempDir(), "warm.pt")
	if err := os.WriteFile(warmPath, []byte("checkpoint"), 0o600); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(t.TempDir(), "outside-report.json")
	options := runOptions(dataset, warmPath, filepath.Join(t.TempDir(), "run"), &fakeRunner{datasetHash: dataset.ManifestHash})
	options.ReportPath = outside
	if _, err := Run(context.Background(), options); err == nil || !strings.Contains(err.Error(), "inside output") {
		t.Fatalf("outside report error=%v", err)
	}
	outputPath := filepath.Join(t.TempDir(), "published")
	if _, err := Run(context.Background(), runOptions(dataset, warmPath, outputPath, &fakeRunner{datasetHash: dataset.ManifestHash})); err != nil {
		t.Fatal(err)
	}
	if _, err := Run(context.Background(), runOptions(dataset, warmPath, outputPath, &fakeRunner{datasetHash: dataset.ManifestHash})); err == nil || !strings.Contains(err.Error(), "report already exists") {
		t.Fatalf("existing report error=%v", err)
	}
}

func TestRunRejectsSymlinkedOutput(t *testing.T) {
	realOutput := filepath.Join(t.TempDir(), "real-output")
	if err := os.Mkdir(realOutput, 0o755); err != nil {
		t.Fatal(err)
	}
	linkedOutput := filepath.Join(t.TempDir(), "linked-output")
	if err := os.Symlink(realOutput, linkedOutput); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	dataset := writeTestGeneration(t)
	warmPath := filepath.Join(t.TempDir(), "warm.pt")
	if err := os.WriteFile(warmPath, []byte("checkpoint"), 0o600); err != nil {
		t.Fatal(err)
	}
	options := runOptions(dataset, warmPath, linkedOutput, &fakeRunner{datasetHash: dataset.ManifestHash})
	if _, err := Run(context.Background(), options); err == nil || !strings.Contains(err.Error(), "symlink") {
		t.Fatalf("symlink output error=%v", err)
	}
}

func TestRunRejectsWorkerTimeoutAboveBound(t *testing.T) {
	options := Options{DatasetPath: "dataset", WarmStartPath: "warm", OutputPath: "output", WorkerTimeout: MaxWorkerTimeout + time.Nanosecond, Worker: &fakeRunner{}}
	if _, err := Run(context.Background(), options); err == nil || !strings.Contains(err.Error(), "worker timeout") {
		t.Fatalf("timeout bound error=%v", err)
	}
}

func TestResolveWorkerUsesFixedModule(t *testing.T) {
	runner, err := resolveWorker(Options{TorchPythonPath: os.Args[0]})
	if err != nil {
		t.Fatal(err)
	}
	command, ok := runner.(*commandRunner)
	if !ok {
		t.Fatalf("runner type=%T", runner)
	}
	want := []string{os.Args[0], "-m", TorchWorkerModule}
	if !reflect.DeepEqual(command.command, want) {
		t.Fatalf("worker command=%q, want %q", command.command, want)
	}
	if _, err := resolveWorker(Options{WorkerCommand: []string{os.Args[0], "--unsafe"}}); err == nil || !strings.Contains(err.Error(), "fixed") {
		t.Fatalf("arbitrary worker command error=%v", err)
	}
}

func TestRunRejectsWarmStartDatasetLineageMismatch(t *testing.T) {
	dataset := writeTestGeneration(t)
	warmPath := filepath.Join(t.TempDir(), "warm.pt")
	if err := os.WriteFile(warmPath, []byte("checkpoint"), 0o600); err != nil {
		t.Fatal(err)
	}
	runner := &fakeRunner{datasetHash: strings.Repeat("f", 64)}
	_, err := Run(context.Background(), runOptions(dataset, warmPath, filepath.Join(t.TempDir(), "run"), runner))
	if err == nil || !strings.Contains(err.Error(), "dataset hash does not match verified generation") {
		t.Fatalf("lineage error=%v", err)
	}
}

func TestRunAllowsExplicitModelOnlyWarmStartDatasetChange(t *testing.T) {
	dataset := writeTestGeneration(t)
	warmPath := filepath.Join(t.TempDir(), "warm.pt")
	if err := os.WriteFile(warmPath, []byte("checkpoint"), 0o600); err != nil {
		t.Fatal(err)
	}
	runner := &fakeRunner{datasetHash: strings.Repeat("f", 64)}
	options := runOptions(dataset, warmPath, filepath.Join(t.TempDir(), "run"), runner)
	options.AllowWarmStartDatasetChange = true
	report, err := Run(context.Background(), options)
	if err != nil {
		t.Fatal(err)
	}
	warm := report["warm_start"].(map[string]any)
	if warm["dataset_changed"] != true || warm["dataset_change_allowed"] != true {
		t.Fatalf("warm-start evidence=%v", warm)
	}
}

func TestWarmStartHashRejectsFileLargerThan512MiB(t *testing.T) {
	path := filepath.Join(t.TempDir(), "oversized.pt")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Truncate(MaxWarmStart + 1); err != nil {
		_ = file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := hashFile(path, MaxWarmStart); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("oversized warm-start error=%v", err)
	}
}

func TestRunRejectsOversizedWarmStartBeforeWorker(t *testing.T) {
	dataset := writeTestGeneration(t)
	warmPath := filepath.Join(t.TempDir(), "oversized.pt")
	file, err := os.Create(warmPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Truncate(MaxWarmStart + 1); err != nil {
		_ = file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	runner := &fakeRunner{datasetHash: dataset.ManifestHash}
	_, err = Run(context.Background(), runOptions(dataset, warmPath, filepath.Join(t.TempDir(), "run"), runner))
	if err == nil || !strings.Contains(err.Error(), "exceeds 536870912-byte limit") {
		t.Fatalf("oversized warm-start error=%v", err)
	}
	if runner.calls != 0 {
		t.Fatalf("worker calls=%d, want zero", runner.calls)
	}
}

func TestValidateWorkerOutputRejectsProtocolFaults(t *testing.T) {
	cases := []struct {
		name   string
		output TorchOutput
		want   string
	}{
		{name: "nonzero exit", output: TorchOutput{ExitCode: 7}, want: "exited"},
		{name: "noncanonical JSON", output: TorchOutput{Stdout: []byte("{ }")}, want: "canonical"},
		{name: "unknown root shape", output: TorchOutput{Stdout: []byte("{}")}, want: "missing"},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			_, err := validateWorkerOutput(testCase.output, strings.Repeat("a", 64), strings.Repeat("b", 64), strings.Repeat("c", 64), strings.Repeat("d", 64), false, 1, 1, defaultConfig)
			if err == nil || !strings.Contains(err.Error(), testCase.want) {
				t.Fatalf("error=%v, want substring %q", err, testCase.want)
			}
		})
	}
}

func BenchmarkCanonicalWorkerRequest(b *testing.B) {
	value := map[string]any{
		"protocol": TorchProtocol, "request_sha256": strings.Repeat("a", 64), "seed": 4242, "device": "cpu",
		"model":      map[string]any{"hidden_size": 384, "model_width": 384, "entity_layers": 4, "num_heads": 8, "ff_multiplier": 4, "timing_bins": 4},
		"learner":    map[string]any{"learning_rate": 3e-4, "weight_decay": 1e-4, "class_balance_power": 0.5, "max_gradient_norm": 1.0},
		"warm_start": map[string]any{"path": "warm.pt", "sha256": strings.Repeat("b", 64), "dataset_hash": strings.Repeat("c", 64), "allow_dataset_change": false},
		"batch":      map[string]any{"kind": "bundle", "path": "batch.json", "sha256": strings.Repeat("c", 64)},
	}
	b.ReportAllocs()
	for index := 0; index < b.N; index++ {
		if _, err := canonicalJSON(value); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkProfileWeights(b *testing.B) {
	counts := make([]int, 96)
	for index := range counts {
		counts[index] = index + 1
	}
	b.ReportAllocs()
	for index := 0; index < b.N; index++ {
		_ = classBalanceWeights(counts)
	}
}

func BenchmarkBundleSerialization(b *testing.B) {
	row := ai42dataset.Row{
		Hero: make([]float32, ai42dataset.HeroCount*ai42dataset.HeroFeatures), Abilities: make([]float32, ai42dataset.HeroCount*ai42dataset.AbilityCount*ai42dataset.AbilityFeatures),
		Entities: make([]float32, ai42dataset.HeroCount*ai42dataset.MaxEntities*ai42dataset.EntityFeatures), Global: make([]float32, ai42dataset.HeroCount*ai42dataset.GlobalFeatures),
		EntityMask: make([]uint8, ai42dataset.HeroCount*ai42dataset.MaxEntities), KindMask: make([]uint8, ai42dataset.HeroCount*ai42dataset.ActionKinds), TargetMask: make([]uint8, ai42dataset.HeroCount*ai42dataset.MaxEntities),
		SkillTargetMask: make([]uint8, ai42dataset.HeroCount*ai42dataset.AbilityCount*ai42dataset.MaxEntities), TeacherStatus: make([]uint8, ai42dataset.HeroCount), TeacherAction: make([]ai42dataset.Action, ai42dataset.HeroCount),
	}
	row.EntityMask[0], row.KindMask[1], row.TeacherStatus[0], row.TeacherAction[0].Kind = 1, 1, 1, 1
	plan := batchPlan{SequenceLength: 1, BatchSize: 1, Sequences: []batchSequence{{MatchID: "match", Hero: 0, Start: 0, Stop: 1, Step: 0}}}
	rows := map[string]map[int]ai42dataset.Row{"match": {0: row}}
	b.ReportAllocs()
	for index := 0; index < b.N; index++ {
		bundle, _, err := buildBundle(plan, rows)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := canonicalJSON(bundle); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEligibleBatchPlan(b *testing.B) {
	matches := []matchInfo{{ID: "match-a", Split: "train", Scenario: "test", TickCount: 640}}
	evidence := &matchEvidence{SupervisedByWindow: make([]uint16, 10)}
	evidence.SupervisedByWindow[9] = 1 << 9
	config := Config{Seed: 4242, SequenceLength: 64, BatchSize: 8}
	b.ReportAllocs()
	for index := 0; index < b.N; index++ {
		if _, err := makeBatchPlan(matches, map[string]*matchEvidence{"match-a": evidence}, config); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTargetRanges(b *testing.B) {
	sequences := make([]batchSequence, 0, MaxBatchSize)
	for hero := 0; hero < MaxBatchSize; hero++ {
		sequences = append(sequences, batchSequence{MatchID: "match-a", Hero: hero % ai42dataset.HeroCount, Start: hero * MaxSequenceLength, Stop: (hero + 1) * MaxSequenceLength})
	}
	plan := batchPlan{Sequences: sequences}
	b.ReportAllocs()
	for index := 0; index < b.N; index++ {
		if _, _, err := targetRanges(plan); err != nil {
			b.Fatal(err)
		}
	}
}

type fakeRunner struct {
	calls                 int
	mutateDataset         string
	mutateDatasetSameSize bool
	mutateProfile         string
	datasetHash           string
	lastRequest           []byte
}

func (runner *fakeRunner) Run(_ context.Context, request []byte) (TorchOutput, error) {
	runner.calls++
	runner.lastRequest = append(runner.lastRequest[:0], request...)
	value, err := decodeCanonical(request, "request", MaxWorkerRequest)
	if err != nil {
		return TorchOutput{}, err
	}
	root := value.(map[string]any)
	requestHash := root["request_sha256"].(string)
	batch := root["batch"].(map[string]any)
	if batch["kind"] != "bundle" {
		return TorchOutput{}, fmt.Errorf("unexpected batch kind %v", batch["kind"])
	}
	warm := root["warm_start"].(map[string]any)
	bundleHash := batch["sha256"].(string)
	warmHash := warm["sha256"].(string)
	hashFields := map[string]any{}
	for key := range workerHashFields {
		hashFields[key] = strings.Repeat("a", 64)
	}
	hashFields["request_sha256"], hashFields["warm_start_sha256"] = requestHash, warmHash
	invariants := map[string]any{}
	for key := range workerInvariantFields {
		invariants[key] = true
	}
	invariants["optimizer_step_called"], invariants["optimizer_authorized"], invariants["final_report_published"] = false, false, false
	result := map[string]any{
		"protocol": TorchProtocol, "ai42_protocol_version": ProtocolVersion, "device": "cpu", "parameter_count": 1, "estimated_parameter_count": 1,
		"batch":      map[string]any{"kind": "bundle", "sha256": bundleHash, "batch_size": 8, "sequence_length": 64, "supervised_count": 8},
		"warm_start": map[string]any{"file_sha256": warmHash, "manifest_digest": strings.Repeat("a", 64), "payload_digest": strings.Repeat("a", 64), "model_hash": strings.Repeat("a", 64), "model_artifact_hash": strings.Repeat("a", 64), "optimizer_artifact_hash": strings.Repeat("a", 64), "rng_artifact_hash": strings.Repeat("a", 64), "dataset_hash": runner.datasetHash, "dataset_changed": runner.datasetHash != warm["dataset_hash"], "dataset_change_allowed": warm["allow_dataset_change"], "source_step": 0, "source_epoch": 0, "source_cursor": nil},
		"loss":       map[string]any{"value": 0.0, "gradient_norm": 0.0, "gradient_norm_after_repeat": 0.0, "gradient_digest_before_clip": strings.Repeat("a", 64), "gradient_digest": strings.Repeat("a", 64), "repeat_gradient_digest": strings.Repeat("a", 64), "summary": map[string]any{"loss": 0.0, "head_losses": map[string]any{}, "metrics": map[string]any{}, "class_counts": map[string]any{}, "control_counts": map[string]any{}, "skill_metrics": map[string]any{}, "head_weighted_numerators": map[string]any{}, "head_weighted_denominators": map[string]any{}}}, "hashes": hashFields, "timings_ms": map[string]any{"forward": 0.0, "backward_and_clip": 0.0, "checkpoint_roundtrip": 0.0, "total": 0.0}, "invariants": invariants,
	}
	response, err := canonicalJSON(map[string]any{"protocol": TorchProtocol, "request_sha256": requestHash, "ok": true, "error": nil, "result": result})
	if err != nil {
		return TorchOutput{}, err
	}
	if runner.mutateDataset != "" {
		path := filepath.Join(runner.mutateDataset, "shard-000000.a42")
		if runner.mutateDatasetSameSize {
			info, err := os.Stat(path)
			if err != nil {
				return TorchOutput{}, err
			}
			payload, err := os.ReadFile(path)
			if err != nil {
				return TorchOutput{}, err
			}
			if len(payload) == 0 {
				return TorchOutput{}, fmt.Errorf("test shard is empty")
			}
			payload[0] ^= 0xff
			if err := os.WriteFile(path, payload, info.Mode().Perm()); err != nil {
				return TorchOutput{}, err
			}
			if err := os.Chtimes(path, info.ModTime(), info.ModTime()); err != nil {
				return TorchOutput{}, err
			}
		} else {
			file, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
			if err != nil {
				return TorchOutput{}, err
			}
			_, _ = file.Write([]byte("mutation"))
			_ = file.Close()
		}
	}
	if runner.mutateProfile != "" {
		file, err := os.OpenFile(runner.mutateProfile, os.O_APPEND|os.O_WRONLY, 0)
		if err != nil {
			return TorchOutput{}, err
		}
		_, _ = file.Write([]byte("mutation"))
		_ = file.Close()
	}
	return TorchOutput{Stdout: response}, nil
}

type testGeneration struct {
	Path         string
	ManifestHash string
	Reader       *fakeDatasetEvidence
}

type fakeDatasetEvidence struct {
	splits map[string][]string
	rows   map[string]map[int]ai42dataset.Row
}

func (reader *fakeDatasetEvidence) ReadTargetRows(ctx context.Context, ranges map[string][][2]int, onRow func(ai42dataset.Row) error) (int, error) {
	count := 0
	matchIDs := make([]string, 0, len(ranges))
	for matchID := range ranges {
		matchIDs = append(matchIDs, matchID)
	}
	sort.Strings(matchIDs)
	for _, matchID := range matchIDs {
		for _, tickRange := range ranges[matchID] {
			for tick := tickRange[0]; tick < tickRange[1]; tick++ {
				if err := ctx.Err(); err != nil {
					return count, err
				}
				row, ok := reader.rows[matchID][tick]
				if !ok {
					return count, fmt.Errorf("missing fake row %s/%d", matchID, tick)
				}
				if err := onRow(row); err != nil {
					return count, err
				}
				count++
			}
		}
	}
	return count, nil
}

func (reader *fakeDatasetEvidence) VerifiedSplitMatchIDs() (map[string][]string, error) {
	return map[string][]string{"train": append([]string(nil), reader.splits["train"]...), "validation": append([]string(nil), reader.splits["validation"]...)}, nil
}

func runOptions(dataset testGeneration, warmPath, outputPath string, runner TorchRunner) Options {
	return Options{DatasetPath: dataset.Path, WarmStartPath: warmPath, OutputPath: outputPath, Device: "cpu", WorkerTimeout: time.Second, Worker: runner, TargetedRows: dataset.Reader, SplitEvidence: dataset.Reader}
}

func writeTestGeneration(t *testing.T) testGeneration {
	t.Helper()
	root := filepath.Join(t.TempDir(), "generation")
	manifest := []byte(`{"runtime":"test"}`)
	runtimeHash := sha256.Sum256(manifest)
	metadata := ai42dataset.Metadata{ProtocolVersion: ai42dataset.ProtocolVersion, TickHz: ai42dataset.FrameRateHz, MatchID: "match-a", RuntimeManifest: manifest, RuntimeManifestHash: runtimeHash, SchemaHash: ai42dataset.AI42SchemaHash, RewardHash: ai42dataset.AI42RewardHash, TrajectorySchemaHash: ai42dataset.AI42TrajectorySchemaHash, Seed: 7, Scenario: "test"}
	for hero := 0; hero < ai42dataset.HeroCount; hero++ {
		metadata.HeroIDs[hero] = "hero-" + string(rune('a'+hero))
		metadata.ControllerBySlot[hero] = 2
		metadata.RosterIDs[hero] = int32(hero + 1)
		if hero >= 5 {
			metadata.SideBySlot[hero] = 1
		}
	}
	capture, err := ai42dataset.NewCapture(metadata)
	if err != nil {
		t.Fatal(err)
	}
	var result battleserver.StepResultV1
	result.SchemaHash, result.RewardHash, result.Step, result.Done = metadata.SchemaHash, metadata.RewardHash, 0, true
	var submitted [ai42dataset.HeroCount]ai42dataset.Action
	var parents, boundaries [ai42dataset.HeroCount]string
	var outcomes [ai42dataset.HeroCount]ai42dataset.Outcome
	for hero := 0; hero < ai42dataset.HeroCount; hero++ {
		result.Observations[hero].Hero[0] = float32(hero) / 100
		result.Observations[hero].EntityMask[0] = 1
		result.Observations[hero].ActionMask.Kinds[1] = 1
		result.Observations[hero].ActionMask.Targets[0] = 1
		result.TeacherStatus[hero] = battleserver.AssaultTeacherStatusAction
		result.TeacherIntent[hero] = battleserver.HeroActionV1{Kind: battleserver.AssaultActionMove}
		result.ExecutedActions[hero] = result.TeacherIntent[hero]
		result.ExecutedValid[hero] = 1
		submitted[hero] = ai42dataset.Action{Kind: 1}
		parents[hero] = "match-a:root:" + twoDigits(hero)
		boundaries[hero] = "match-a:boundary:0:" + twoDigits(hero)
		outcomes[hero] = ai42dataset.Outcome{Terminal: true, Winner: 0, WinnerPresent: true, HeroAlive: true, HeroAlivePresent: true}
	}
	if err := capture.Append(&result, submitted, parents, boundaries, outcomes); err != nil {
		t.Fatal(err)
	}
	prepared, err := capture.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	if err := ai42dataset.WriteGeneration(root, prepared); err != nil {
		t.Fatal(err)
	}
	generation, err := ai42dataset.OpenGenerationWithOptions(root, ai42dataset.OpenOptions{DeferShardHashing: true})
	if err != nil {
		t.Fatal(err)
	}
	splits := map[string][]string{"train": {}, "validation": {}}
	for _, match := range generation.Matches() {
		splits[match.Split] = append(splits[match.Split], match.MatchID)
	}
	reader := &fakeDatasetEvidence{splits: splits, rows: map[string]map[int]ai42dataset.Row{"match-a": {0: testTargetRow("match-a", 0)}}}
	return testGeneration{Path: root, ManifestHash: generation.ManifestHash(), Reader: reader}
}

func testTargetRow(matchID string, tick int) ai42dataset.Row {
	row := ai42dataset.Row{MatchID: matchID, Tick: tick, Step: uint32(tick),
		Hero: make([]float32, ai42dataset.HeroCount*ai42dataset.HeroFeatures), Abilities: make([]float32, ai42dataset.HeroCount*ai42dataset.AbilityCount*ai42dataset.AbilityFeatures),
		Entities: make([]float32, ai42dataset.HeroCount*ai42dataset.MaxEntities*ai42dataset.EntityFeatures), Global: make([]float32, ai42dataset.HeroCount*ai42dataset.GlobalFeatures),
		EntityMask: make([]uint8, ai42dataset.HeroCount*ai42dataset.MaxEntities), KindMask: make([]uint8, ai42dataset.HeroCount*ai42dataset.ActionKinds), TargetMask: make([]uint8, ai42dataset.HeroCount*ai42dataset.MaxEntities),
		SkillTargetMask: make([]uint8, ai42dataset.HeroCount*ai42dataset.AbilityCount*ai42dataset.MaxEntities), TeacherStatus: make([]uint8, ai42dataset.HeroCount), TeacherAction: make([]ai42dataset.Action, ai42dataset.HeroCount)}
	for hero := 0; hero < ai42dataset.HeroCount; hero++ {
		row.Hero[hero*ai42dataset.HeroFeatures] = float32(hero) / 100
		row.EntityMask[hero*ai42dataset.MaxEntities] = 1
		row.KindMask[hero*ai42dataset.ActionKinds+1] = 1
		row.TeacherStatus[hero] = 1
		row.TeacherAction[hero] = ai42dataset.Action{Kind: 1}
	}
	return row
}

func twoDigits(value int) string {
	if value < 10 {
		return "0" + string(rune('0'+value))
	}
	return string(rune('0'+value/10)) + string(rune('0'+value%10))
}
