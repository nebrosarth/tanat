package ai42preflight

import (
	"fmt"
	"strings"
)

const maxWorkerFailureDiagnostic = 4096

var workerResultFields = map[string]struct{}{"protocol": {}, "ai42_protocol_version": {}, "device": {}, "parameter_count": {}, "estimated_parameter_count": {}, "batch": {}, "warm_start": {}, "loss": {}, "hashes": {}, "timings_ms": {}, "invariants": {}}
var workerLossFields = map[string]struct{}{"value": {}, "gradient_norm": {}, "gradient_norm_after_repeat": {}, "gradient_digest_before_clip": {}, "gradient_digest": {}, "repeat_gradient_digest": {}, "summary": {}}
var workerLossSummaryFields = map[string]struct{}{"loss": {}, "head_losses": {}, "metrics": {}, "class_counts": {}, "control_counts": {}, "skill_metrics": {}, "head_weighted_numerators": {}, "head_weighted_denominators": {}}
var workerBatchFields = map[string]struct{}{"kind": {}, "sha256": {}, "batch_size": {}, "sequence_length": {}, "supervised_count": {}}
var workerWarmFields = map[string]struct{}{"file_sha256": {}, "manifest_digest": {}, "payload_digest": {}, "model_hash": {}, "model_artifact_hash": {}, "optimizer_artifact_hash": {}, "rng_artifact_hash": {}, "dataset_hash": {}, "dataset_changed": {}, "dataset_change_allowed": {}, "source_step": {}, "source_epoch": {}, "source_cursor": {}}
var workerHashFields = map[string]struct{}{"request_sha256": {}, "warm_start_sha256": {}, "model_before_warm_start": {}, "model_after_warm_start": {}, "model_after": {}, "optimizer_before": {}, "optimizer_after": {}, "rng_before_warm_start": {}, "rng_after_warm_start": {}, "rng_after": {}, "forward_first": {}, "forward_second": {}, "roundtrip_payload": {}}
var workerTimingFields = map[string]struct{}{"forward": {}, "backward_and_clip": {}, "checkpoint_roundtrip": {}, "total": {}}
var workerInvariantFields = map[string]struct{}{"finite_outputs": {}, "finite_loss": {}, "finite_gradients": {}, "gradient_clip_checked": {}, "deterministic_recurrent_forward": {}, "deterministic_masked_backward": {}, "parameters_unchanged": {}, "optimizer_unchanged": {}, "rng_unchanged": {}, "optimizer_not_restored": {}, "rng_not_restored": {}, "cursor_not_restored": {}, "optimizer_step_called": {}, "optimizer_authorized": {}, "final_report_published": {}, "checkpoint_roundtrip": {}, "exact_checkpoint_bytes_loaded": {}, "checkpoint_source_unchanged": {}}

func validateWorkerOutput(output TorchOutput, requestHash, bundleHash, warmHash, datasetHash string, allowDatasetChange bool, supervised, expectedBatchSize int, config Config) (map[string]any, error) {
	if len(output.Stdout) > MaxWorkerStdout || len(output.Stderr) > MaxWorkerStderr {
		return nil, fmt.Errorf("Torch worker output exceeded bounded stream limits")
	}
	if output.ExitCode != 0 {
		diagnostic := strings.TrimSpace(string(output.Stderr))
		if diagnostic == "" {
			diagnostic = strings.TrimSpace(string(output.Stdout))
		}
		if len(diagnostic) > maxWorkerFailureDiagnostic {
			diagnostic = diagnostic[:maxWorkerFailureDiagnostic] + "..."
		}
		if diagnostic != "" {
			return nil, fmt.Errorf("Torch worker exited with code %d: %s", output.ExitCode, diagnostic)
		}
		return nil, fmt.Errorf("Torch worker exited with code %d", output.ExitCode)
	}
	value, err := decodeCanonical(output.Stdout, "Torch worker response", MaxWorkerStdout)
	if err != nil {
		return nil, err
	}
	root, err := object(value, "Torch worker response")
	if err != nil {
		return nil, err
	}
	if err := exactFields(root, map[string]struct{}{"protocol": {}, "request_sha256": {}, "ok": {}, "error": {}, "result": {}}, nil, "Torch worker response"); err != nil {
		return nil, err
	}
	protocol, err := asString(root["protocol"], "worker.protocol")
	if err != nil || protocol != TorchProtocol {
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("worker protocol is incompatible")
	}
	responseHash, err := asString(root["request_sha256"], "worker.request_sha256")
	if err != nil || responseHash != requestHash {
		return nil, fmt.Errorf("worker response request hash does not match request")
	}
	ok, err := asBool(root["ok"], "worker.ok")
	if err != nil {
		return nil, err
	}
	if !ok || root["error"] != nil {
		return nil, fmt.Errorf("Torch worker returned a failure response")
	}
	result, err := object(root["result"], "worker.result")
	if err != nil {
		return nil, err
	}
	if err := exactFields(result, workerResultFields, nil, "worker.result"); err != nil {
		return nil, err
	}
	if protocolValue, err := asString(result["protocol"], "worker.result.protocol"); err != nil || protocolValue != TorchProtocol {
		return nil, fmt.Errorf("worker result protocol is incompatible")
	}
	version, err := asInt(result["ai42_protocol_version"], "worker.result.ai42_protocol_version", ProtocolVersion, ProtocolVersion)
	if err != nil {
		return nil, err
	}
	_ = version
	if _, err := asString(result["device"], "worker.result.device"); err != nil {
		return nil, err
	}
	if _, err := asInt(result["parameter_count"], "worker.result.parameter_count", 1, 1<<62); err != nil {
		return nil, err
	}
	if _, err := asInt(result["estimated_parameter_count"], "worker.result.estimated_parameter_count", 1, 1<<62); err != nil {
		return nil, err
	}
	batch, err := object(result["batch"], "worker.result.batch")
	if err != nil {
		return nil, err
	}
	if err := exactFields(batch, workerBatchFields, nil, "worker.result.batch"); err != nil {
		return nil, err
	}
	if kind, err := asString(batch["kind"], "worker.result.batch.kind"); err != nil || kind != "bundle" {
		return nil, fmt.Errorf("worker result batch kind is not bundle")
	}
	batchSHA, err := asString(batch["sha256"], "worker.result.batch.sha256")
	if err != nil || batchSHA != bundleHash {
		return nil, fmt.Errorf("worker result batch hash does not match bundle")
	}
	batchSize, err := asInt(batch["batch_size"], "worker.result.batch.batch_size", 1, MaxBatchSize)
	if err != nil || int(batchSize) != expectedBatchSize {
		return nil, fmt.Errorf("worker result batch size does not match request")
	}
	sequenceLength, err := asInt(batch["sequence_length"], "worker.result.batch.sequence_length", 1, MaxSequenceLength)
	if err != nil || int(sequenceLength) != config.SequenceLength {
		return nil, fmt.Errorf("worker result sequence length does not match request")
	}
	supervisedCount, err := asInt(batch["supervised_count"], "worker.result.batch.supervised_count", 1, 1<<62)
	if err != nil {
		return nil, err
	}
	if int(supervisedCount) != supervised {
		return nil, fmt.Errorf("worker result supervised count does not match bundle")
	}
	warmStart, err := object(result["warm_start"], "worker.result.warm_start")
	if err != nil {
		return nil, err
	}
	if err := exactFields(warmStart, workerWarmFields, nil, "worker.result.warm_start"); err != nil {
		return nil, err
	}
	fileHash, err := asString(warmStart["file_sha256"], "worker.result.warm_start.file_sha256")
	if err != nil || fileHash != warmHash {
		return nil, fmt.Errorf("worker warm-start hash does not match request")
	}
	for _, key := range []string{"manifest_digest", "payload_digest", "model_hash", "model_artifact_hash", "optimizer_artifact_hash", "rng_artifact_hash", "dataset_hash"} {
		value, err := asString(warmStart[key], "worker.result.warm_start."+key)
		if err != nil || !validHash(value) {
			return nil, fmt.Errorf("worker warm-start %s is not a valid hash", key)
		}
	}
	sourceDatasetHash := warmStart["dataset_hash"].(string)
	datasetChanged := sourceDatasetHash != datasetHash
	workerChanged, err := asBool(warmStart["dataset_changed"], "worker.result.warm_start.dataset_changed")
	if err != nil || workerChanged != datasetChanged {
		return nil, fmt.Errorf("worker warm-start dataset-change evidence does not match lineage")
	}
	workerAllowed, err := asBool(warmStart["dataset_change_allowed"], "worker.result.warm_start.dataset_change_allowed")
	if err != nil || workerAllowed != allowDatasetChange {
		return nil, fmt.Errorf("worker warm-start dataset-change authorization does not match request")
	}
	if datasetChanged && !allowDatasetChange {
		return nil, fmt.Errorf("worker warm-start dataset hash does not match verified generation")
	}
	for _, key := range []string{"source_step", "source_epoch"} {
		if _, err := asInt(warmStart[key], "worker.result.warm_start."+key, 0, 1<<62); err != nil {
			return nil, err
		}
	}
	if cursor := warmStart["source_cursor"]; cursor != nil {
		if _, err := asInt(cursor, "worker.result.warm_start.source_cursor", 0, 1<<62); err != nil {
			return nil, err
		}
	}
	loss, err := object(result["loss"], "worker.result.loss")
	if err != nil {
		return nil, err
	}
	if err := exactFields(loss, workerLossFields, nil, "worker.result.loss"); err != nil {
		return nil, err
	}
	for _, key := range []string{"value", "gradient_norm", "gradient_norm_after_repeat"} {
		value, err := asNumber(loss[key], "worker.result.loss."+key)
		if err != nil || value < 0 {
			return nil, fmt.Errorf("worker loss %s is invalid", key)
		}
	}
	for _, key := range []string{"gradient_digest_before_clip", "gradient_digest", "repeat_gradient_digest"} {
		value, err := asString(loss[key], "worker.result.loss."+key)
		if err != nil || !validHash(value) {
			return nil, fmt.Errorf("worker loss %s is invalid", key)
		}
	}
	summary, err := object(loss["summary"], "worker.result.loss.summary")
	if err != nil {
		return nil, err
	}
	if err := exactFields(summary, workerLossSummaryFields, nil, "worker.result.loss.summary"); err != nil {
		return nil, err
	}
	hashes, err := object(result["hashes"], "worker.result.hashes")
	if err != nil {
		return nil, err
	}
	if err := exactFields(hashes, workerHashFields, nil, "worker.result.hashes"); err != nil {
		return nil, err
	}
	for key := range workerHashFields {
		value, err := asString(hashes[key], "worker.result.hashes."+key)
		if err != nil || !validHash(value) {
			return nil, fmt.Errorf("worker result hash %s is invalid", key)
		}
	}
	if hashes["request_sha256"] != requestHash || hashes["warm_start_sha256"] != warmHash {
		return nil, fmt.Errorf("worker result hashes do not match request inputs")
	}
	timings, err := object(result["timings_ms"], "worker.result.timings_ms")
	if err != nil {
		return nil, err
	}
	if err := exactFields(timings, workerTimingFields, nil, "worker.result.timings_ms"); err != nil {
		return nil, err
	}
	for key := range workerTimingFields {
		value, err := asNumber(timings[key], "worker.result.timings_ms."+key)
		if err != nil || value < 0 {
			return nil, fmt.Errorf("worker timing %s is invalid", key)
		}
	}
	invariants, err := object(result["invariants"], "worker.result.invariants")
	if err != nil {
		return nil, err
	}
	if err := exactFields(invariants, workerInvariantFields, nil, "worker.result.invariants"); err != nil {
		return nil, err
	}
	for key := range workerInvariantFields {
		value, err := asBool(invariants[key], "worker.result.invariants."+key)
		if err != nil {
			return nil, err
		}
		if key == "optimizer_step_called" || key == "optimizer_authorized" || key == "final_report_published" {
			if value {
				return nil, fmt.Errorf("worker invariant %s was violated", key)
			}
		} else if !value {
			return nil, fmt.Errorf("worker invariant %s was not proven", key)
		}
	}
	return result, nil
}
