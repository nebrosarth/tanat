package ai42preflight

import (
	"time"

	"tanatserver/internal/ai42dataset"
)

type executionEvidence struct {
	OpenElapsed                  time.Duration
	VerifyElapsed                time.Duration
	PlanElapsed                  time.Duration
	ProfileElapsed               time.Duration
	TargetedReadElapsed          time.Duration
	TargetedShardRereadElapsed   time.Duration
	BundleElapsed                time.Duration
	WarmHashElapsed              time.Duration
	WorkerElapsed                time.Duration
	PostWorkerContentHashElapsed time.Duration
	TotalElapsed                 time.Duration
	OpenGenerationCalls          int
	FullVerifyPasses             int
	VerifyWorkers                int
	TargetedReadCalls            int
	TargetedShardRereadPasses    int
	TargetedShardRereadShards    int
	TargetedRows                 int
	PostWorkerContentHashPasses  int
	PostWorkerContentHashFiles   int
	IdentityChecks               int
	SplitRecomputed              bool
	DatasetManifestFile          contentFileEvidence
	PostWorkerContent            map[string]contentFileEvidence
}

func makeReport(config Config, options Options, generation *ai42dataset.Generation, verification ai42dataset.VerificationReport, accumulator *profileAccumulator, profile Profile, profileExisting bool, plan batchPlan, weights map[string][]float64, profileFileHash, bundlePath, bundleHash, requestHash, warmHash string, requestBytes int, workerOutput TorchOutput, workerResult map[string]any, datasetIdentityHash string, evidence executionEvidence) Report {
	device := options.Device
	if value, ok := workerResult["device"].(string); ok {
		device = value
	}
	parameterCount := int64(0)
	if value, err := asInt(workerResult["parameter_count"], "worker.result.parameter_count", 0, 1<<62); err == nil {
		parameterCount = value
	}
	trainIDs := append([]string(nil), profile.TrainMatchIDs...)
	validationIDs := append([]string(nil), plan.ValidationMatchIDs...)
	datasetContentFingerprint := datasetContentFingerprint(verification, evidence.DatasetManifestFile)
	headWeights := scalarWeightsAsAny(config.Learner.HeadWeights)
	headWeightsBytes, _ := canonicalJSON(headWeights)
	headWeightsHash := sha256Hex(headWeightsBytes)
	profilePayload := map[string]any{"path": options.ProfilePath, "hash": profile.ProfileHash, "file_sha256": profileFileHash, "expected_file_sha256": options.ProfileHash, "dataset_manifest_hash": profile.DatasetManifestHash, "train_match_ids_hash": profile.TrainMatchIDsHash, "train_match_ids": stringsAny(trainIDs), "supervision_controllers": uint8sAny(profile.SupervisionControllers), "counts": countsAsAny(profile.Counts), "weights": weightsAsAny(profile.Weights), "existing_validated": profileExisting}
	manifest := map[string]any{"protocol_version": ProtocolVersion, "dataset_schema_version": "AI42-dataset-v1", "shard_schema_version": "AI42-go-shard-v2", "dataset_manifest_hash": generation.ManifestHash(), "profile_hash": profile.ProfileHash, "train_match_ids_hash": profile.TrainMatchIDsHash, "supervision_controllers": uint8sAny(profile.SupervisionControllers), "head_weights": headWeights, "head_weights_hash": headWeightsHash}
	fileEvidence := make([]any, len(verification.Files))
	for index, file := range verification.Files {
		fileEvidence[index] = map[string]any{"shard": file.Shard, "stored_bytes": file.StoredBytes, "sha256": file.SHA256, "compressed_bytes": file.CompressedBytes, "payload_sha256": file.PayloadSHA256, "raw_bytes": file.RawBytes, "raw_sha256": file.RawSHA256}
	}
	dataset := map[string]any{"provided": true, "path": options.DatasetPath, "manifest_hash": generation.ManifestHash(), "dataset_schema_version": "AI42-dataset-v1", "shard_schema_version": "AI42-go-shard-v2", "matches": verification.Matches, "train_matches": len(trainIDs), "validation_matches": len(accumulator.Matches) - len(trainIDs), "shards": verification.Shards, "rows": verification.Rows, "fingerprint": datasetContentFingerprint, "identity_mode": "content-hash-plus-stat", "identity_hash": datasetIdentityHash, "post_worker_content_hash_verified": evidence.PostWorkerContentHashPasses == 1}
	batchPlan := map[string]any{"version": plan.Version, "hash": plan.Hash, "split_hash": plan.SplitHash, "validation_probe_hash": plan.ValidationHash, "batch_cursor": 0, "train_match_ids": stringsAny(plan.TrainMatchIDs), "validation_match_ids": stringsAny(validationIDs), "supervision_controllers": uint8sAny(profile.SupervisionControllers), "sequence_length": plan.SequenceLength, "batch_size": plan.BatchSize, "selected_batch_size": len(plan.Sequences), "physical_batches_scanned": plan.PhysicalBatches, "empty_batches_skipped": plan.EmptyBatches, "sequences_scanned": plan.SequencesScanned}
	hashes := map[string]any{"dataset_manifest": generation.ManifestHash(), "dataset_manifest_file_sha256": evidence.DatasetManifestFile.SHA256, "dataset_fingerprint": datasetContentFingerprint, "dataset_identity": datasetIdentityHash, "split": plan.SplitHash, "profile": profile.ProfileHash, "profile_file_sha256": profileFileHash, "train_match_ids": profile.TrainMatchIDsHash, "head_weights": headWeightsHash, "batch_bundle": bundleHash, "request_sha256": requestHash, "warm_start_sha256": warmHash}
	invariants := map[string]any{"durable_v2_full_verify": evidence.FullVerifyPasses == 1, "single_full_verify_pass": evidence.FullVerifyPasses == 1, "no_redundant_full_generation_pass": evidence.FullVerifyPasses == 1, "train_only_profile": true, "ordered_train_ids_hash_checked": true, "deterministic_split_checked": evidence.SplitRecomputed, "deterministic_batch_plan_checked": true, "profile_immutable_or_atomically_published": true, "bounded_bundle": true, "targeted_bundle_read": evidence.TargetedReadCalls == 1, "targeted_verified_shard_reread": evidence.TargetedShardRereadPasses == 1, "request_hash_checked": true, "worker_response_canonical": true, "worker_exit_zero": workerOutput.ExitCode == 0, "worker_ok": true, "dataset_unchanged": evidence.PostWorkerContentHashPasses == 1, "warm_start_unchanged": true, "bundle_unchanged": true, "no_python_fallback": true, "optimizer_step_authorized": false, "report_atomically_published": true}
	postWorkerFiles := postWorkerContentEvidence(verification, evidence.DatasetManifestFile, evidence.PostWorkerContent)
	ioCounters := map[string]any{"open_generation_calls": evidence.OpenGenerationCalls, "full_verify_passes": evidence.FullVerifyPasses, "full_verify_workers": evidence.VerifyWorkers, "targeted_shard_reread_passes": evidence.TargetedShardRereadPasses, "targeted_shard_reread_shards": evidence.TargetedShardRereadShards, "targeted_read_calls": evidence.TargetedReadCalls, "targeted_rows": evidence.TargetedRows, "identity_checks": evidence.IdentityChecks, "post_worker_content_hash_passes": evidence.PostWorkerContentHashPasses, "post_worker_content_hash_files": evidence.PostWorkerContentHashFiles}
	nativeEvidence := map[string]any{"replay_scope": ReplayScope, "dataset_verification": map[string]any{"manifest_hash": verification.ManifestHash, "shards": verification.Shards, "matches": verification.Matches, "rows": verification.Rows, "files": fileEvidence}, "dataset_identity": map[string]any{"mode": "content-hash-plus-stat", "content_hash": datasetContentFingerprint, "stat_hash": datasetIdentityHash, "manifest_commits_shard_digests": true}, "post_worker_content_hash": map[string]any{"completed": evidence.PostWorkerContentHashPasses == 1, "passes": evidence.PostWorkerContentHashPasses, "files": postWorkerFiles, "expected_source": "manifest file plus VerificationReport.Files", "compares_exact_size_and_sha256": true}, "io_counters": ioCounters, "profile": profilePayload, "split": map[string]any{"hash": plan.SplitHash, "train_match_ids": stringsAny(plan.TrainMatchIDs), "validation_match_ids": stringsAny(plan.ValidationMatchIDs), "reader_recomputed": evidence.SplitRecomputed}, "eligibility": map[string]any{"physical_batches_scanned": plan.PhysicalBatches, "empty_batches_skipped": plan.EmptyBatches, "sequences_scanned": plan.SequencesScanned, "selected_batch_size": len(plan.Sequences)}, "batch_bundle": map[string]any{"path": bundlePath, "sha256": bundleHash, "targeted_rows": evidence.TargetedRows}, "request": map[string]any{"sha256": requestHash, "bytes": requestBytes}, "warm_start": map[string]any{"path": options.WarmStartPath, "sha256": warmHash, "maximum_bytes": MaxWarmStart, "dataset_change_allowed": options.AllowWarmStartDatasetChange}, "worker": map[string]any{"protocol": TorchProtocol, "exit_code": workerOutput.ExitCode, "stdout_bytes": len(workerOutput.Stdout), "stderr_bytes": len(workerOutput.Stderr), "stderr_sha256": sha256Hex(workerOutput.Stderr)}}
	timings := map[string]any{"open_generation_ms": durationMS(evidence.OpenElapsed), "full_verify_ms": durationMS(evidence.VerifyElapsed), "dataset_verify_ms": durationMS(evidence.OpenElapsed + evidence.VerifyElapsed), "batch_plan_ms": durationMS(evidence.PlanElapsed), "profile_ms": durationMS(evidence.ProfileElapsed), "targeted_shard_reread_ms": durationMS(evidence.TargetedShardRereadElapsed), "targeted_read_ms": durationMS(evidence.TargetedReadElapsed), "bundle_publish_ms": durationMS(evidence.BundleElapsed), "bundle_ms": durationMS(evidence.TargetedReadElapsed + evidence.BundleElapsed), "warm_start_hash_ms": durationMS(evidence.WarmHashElapsed), "worker_ms": durationMS(evidence.WorkerElapsed), "post_worker_content_hash_ms": durationMS(evidence.PostWorkerContentHashElapsed), "total_ms": durationMS(evidence.TotalElapsed)}
	return Report{
		"format": "AI42-bc-run-report-v1", "mode": "preflight", "status": "ok", "ok": true, "accepted": false, "execute_required_for_training": true, "training_implemented": true, "python_fallback": false,
		"device": device, "seed": int(config.Seed), "deterministic_order": true, "parameter_count": parameterCount, "parameters_unchanged": workerInvariant(workerResult, "parameters_unchanged"),
		"manifest": manifest, "dataset": dataset, "loss": workerResult["loss"], "checkpoint": workerResult["warm_start"], "warm_start": workerResult["warm_start"], "profile": profilePayload,
		"class_weight_overrides": overridesAsAny(config.Learner.ClassWeightOverrides), "class_weights": weightsAsAny(weights), "head_weights": headWeights, "head_weights_hash": headWeightsHash, "trainable_scope": config.Learner.TrainableScope, "batch_plan": batchPlan, "hashes": hashes,
		"torch": workerResult, "native_evidence": nativeEvidence, "replay_scope": ReplayScope, "timings": timings, "invariants": invariants,
	}
}

func workerInvariant(result map[string]any, name string) bool {
	values, ok := result["invariants"].(map[string]any)
	if !ok {
		return false
	}
	value, ok := values[name].(bool)
	return ok && value
}
func countsAsAny(value map[string][]int) map[string]any {
	result := map[string]any{}
	for _, head := range profileHeads {
		items := make([]any, len(value[head]))
		for index, item := range value[head] {
			items[index] = item
		}
		result[head] = items
	}
	return result
}
func overridesAsAny(value map[string][]float64) map[string]any {
	result := map[string]any{}
	for head, values := range value {
		items := make([]any, len(values))
		for index, item := range values {
			items[index] = item
		}
		result[head] = items
	}
	return result
}

func scalarWeightsAsAny(value map[string]float64) map[string]any {
	result := make(map[string]any, len(value))
	for head, weight := range value {
		result[head] = weight
	}
	return result
}

func postWorkerContentEvidence(verification ai42dataset.VerificationReport, manifest contentFileEvidence, actual map[string]contentFileEvidence) []any {
	files := make([]any, 0, len(verification.Files)+1)
	appendFile := func(expected contentFileEvidence) {
		observed, ok := actual[expected.Name]
		files = append(files, map[string]any{"name": expected.Name, "expected_bytes": expected.Size, "expected_sha256": expected.SHA256, "observed_bytes": observed.Size, "observed_sha256": observed.SHA256, "matched": ok && observed.Size == expected.Size && observed.SHA256 == expected.SHA256})
	}
	appendFile(manifest)
	for _, file := range verification.Files {
		appendFile(contentFileEvidence{Name: file.Shard, Size: file.StoredBytes, SHA256: file.SHA256})
	}
	return files
}

func datasetContentFingerprint(verification ai42dataset.VerificationReport, manifest contentFileEvidence) string {
	files := make([]any, 0, len(verification.Files)+1)
	files = append(files, map[string]any{"name": manifest.Name, "bytes": manifest.Size, "sha256": manifest.SHA256})
	for _, file := range verification.Files {
		files = append(files, map[string]any{"name": file.Shard, "bytes": file.StoredBytes, "sha256": file.SHA256})
	}
	encoded, err := canonicalJSON(files)
	if err != nil {
		return ""
	}
	return sha256Hex(encoded)
}

func durationMS(value time.Duration) float64 { return float64(value) / float64(time.Millisecond) }
