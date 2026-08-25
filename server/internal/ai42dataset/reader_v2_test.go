package ai42dataset

import (
	"bytes"
	"compress/flate"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestOpenGenerationAndVerifyV2(t *testing.T) {
	root := filepath.Join(t.TempDir(), "generation")
	if err := WriteGeneration(root, acceptancePrepared(t)); err != nil {
		t.Fatal(err)
	}
	manifestBytes, err := os.ReadFile(filepath.Join(root, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var manifest map[string]any
	if err := json.Unmarshal(manifestBytes, &manifest); err != nil {
		t.Fatal(err)
	}
	expectedHash, ok := manifest["manifest_hash"].(string)
	if !ok {
		t.Fatal("manifest hash is not a string")
	}
	generation, err := OpenGeneration(root, expectedHash)
	if err != nil {
		t.Fatal(err)
	}
	rows, matches := 0, 0
	report, err := generation.Verify(context.Background(), VerifyOptions{
		OnRow: func(row Row) error {
			rows++
			if row.MatchID == "" || row.Tick != 0 || row.Step != 0 || row.Done != 1 || len(row.Hero) != HeroCount*HeroFeatures {
				t.Fatalf("unexpected row: match=%q tick=%d step=%d done=%d hero=%d", row.MatchID, row.Tick, row.Step, row.Done, len(row.Hero))
			}
			return nil
		},
		OnMatch: func(match MatchMetadata) error {
			matches++
			if match.MatchID != "ai42-test-match" || len(match.TrajectoryHashes) != HeroCount {
				t.Fatalf("unexpected match metadata: %+v", match)
			}
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if report.ManifestHash != expectedHash || report.Shards != 1 || report.Matches != 1 || report.Rows != 1 || rows != 1 || matches != 1 {
		t.Fatalf("unexpected verification report: %+v rows=%d matches=%d", report, rows, matches)
	}
	if len(report.Files) != 1 || report.Files[0].Shard != "shard-000000.a42" || report.Files[0].SHA256 == "" || report.Files[0].PayloadSHA256 == "" || report.Files[0].RawSHA256 == "" {
		t.Fatalf("missing authoritative file evidence: %+v", report.Files)
	}
}

func TestOpenGenerationDeferredHashingAndAuthoritativeVerify(t *testing.T) {
	root := filepath.Join(t.TempDir(), "generation")
	if err := WriteGeneration(root, acceptancePrepared(t)); err != nil {
		t.Fatal(err)
	}
	shardPath := filepath.Join(root, "shard-000000.a42")
	payload, err := os.ReadFile(shardPath)
	if err != nil {
		t.Fatal(err)
	}
	payload[len(payload)-1] ^= 0x80
	if err := os.WriteFile(shardPath, payload, 0o600); err != nil {
		t.Fatal(err)
	}

	generation, err := OpenGenerationWithOptions(root, OpenOptions{DeferShardHashing: true})
	if err != nil {
		t.Fatalf("deferred open rejected a header-valid shard before authoritative hashing: %v", err)
	}
	if _, err := OpenGeneration(root, ""); err == nil || !strings.Contains(err.Error(), "file hash/size mismatch") {
		t.Fatalf("strict open error=%v", err)
	}
	report, err := generation.Verify(context.Background(), VerifyOptions{})
	if err == nil || (!strings.Contains(err.Error(), "payload hash mismatch") && !strings.Contains(err.Error(), "file hash mismatch") && !strings.Contains(err.Error(), "compressed payload is corrupt")) {
		t.Fatalf("authoritative verification error=%v", err)
	}
	if len(report.Files) != 0 {
		t.Fatalf("failed verification exposed file evidence: %+v", report.Files)
	}
}

func TestOpenGenerationDeferredHashingStillValidatesShardHeader(t *testing.T) {
	root := filepath.Join(t.TempDir(), "generation")
	if err := WriteGeneration(root, acceptancePrepared(t)); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "shard-000000.a42")
	payload, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	payload[0] ^= 0xff
	if err := os.WriteFile(path, payload, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenGenerationWithOptions(root, OpenOptions{DeferShardHashing: true}); err == nil || !strings.Contains(err.Error(), "magic mismatch") {
		t.Fatalf("deferred header validation error=%v", err)
	}
}

func TestOpenGenerationRejectsTamperedDeterministicSplit(t *testing.T) {
	root := filepath.Join(t.TempDir(), "generation")
	if err := WriteGeneration(root, acceptancePrepared(t)); err != nil {
		t.Fatal(err)
	}
	rewriteGenerationManifest(t, root, func(manifest map[string]any) {
		manifest["matches"].([]any)[0].(map[string]any)["split"] = "validation"
	})
	if _, err := OpenGenerationWithOptions(root, OpenOptions{DeferShardHashing: true}); err == nil || !strings.Contains(err.Error(), "does not match deterministic split") {
		t.Fatalf("split tamper error=%v", err)
	}
}

func TestValidateDeterministicSplitsMatchesPythonGolden(t *testing.T) {
	// Python _deterministic_split(seed=17, fraction=.4) ranks these IDs as
	// c, e, b, d, a and therefore selects c/e for validation.
	expected := map[string]string{
		"match-a": "train", "match-b": "train", "match-c": "validation",
		"match-d": "train", "match-e": "validation",
	}
	matches := make([]*matchRecord, 0, len(expected))
	for _, id := range []string{"match-a", "match-b", "match-c", "match-d", "match-e"} {
		matches = append(matches, &matchRecord{metadata: MatchMetadata{MatchID: id, Split: expected[id]}})
	}
	if err := validateDeterministicSplits(matches, 17, 0.4); err != nil {
		t.Fatal(err)
	}
	matches[0].metadata.Split = "validation"
	if err := validateDeterministicSplits(matches, 17, 0.4); err == nil || !strings.Contains(err.Error(), "deterministic split") {
		t.Fatalf("golden split tamper error=%v", err)
	}
}

func TestValidateDeterministicStratifiedSplitsMatchesProductionMerge(t *testing.T) {
	// The global Python ranking selects match-01 and match-00 for validation,
	// while the production scenario quotas select one match from each group:
	// match-01 and match-02.
	splits := map[string]string{
		"match-00": "train", "match-01": "validation",
		"match-02": "validation", "match-03": "train",
	}
	matches := make([]*matchRecord, 0, len(splits))
	for _, id := range []string{"match-00", "match-01", "match-02", "match-03"} {
		scenario := "scenario-a"
		if id >= "match-02" {
			scenario = "scenario-b"
		}
		matches = append(matches, &matchRecord{metadata: MatchMetadata{
			MatchID: id, Split: splits[id], Scenario: scenario,
		}})
	}
	runtime := stratifiedRuntimeManifest([]string{
		"match-00", "match-01", "match-02", "match-03",
	}, []string{"scenario-a", "scenario-a", "scenario-b", "scenario-b"})
	if err := validateDeterministicSplits(matches, -1, 0.5, runtime); err != nil {
		t.Fatalf("production-like stratified split rejected: %v", err)
	}
	if err := validateDeterministicSplits(matches, -1, 0.5); err == nil || !strings.Contains(err.Error(), "deterministic split") {
		t.Fatalf("global split unexpectedly accepted stratified assignments: %v", err)
	}

	counts := map[string]map[string]int{}
	for _, match := range matches {
		if _, exists := counts[match.metadata.Scenario]; !exists {
			counts[match.metadata.Scenario] = map[string]int{"train": 0, "validation": 0}
		}
		counts[match.metadata.Scenario][match.metadata.Split]++
	}
	for scenario, count := range counts {
		if count["train"] != 1 || count["validation"] != 1 {
			t.Fatalf("scenario %s has non-exact split quota: %v", scenario, count)
		}
	}
}

func TestValidateDeterministicStratifiedSplitsUsesManifestScenariosWithoutSchedule(t *testing.T) {
	matches := []*matchRecord{
		{metadata: MatchMetadata{MatchID: "match-00", Split: "train", Scenario: "scenario-a"}},
		{metadata: MatchMetadata{MatchID: "match-01", Split: "validation", Scenario: "scenario-a"}},
		{metadata: MatchMetadata{MatchID: "match-02", Split: "validation", Scenario: "scenario-b"}},
		{metadata: MatchMetadata{MatchID: "match-03", Split: "train", Scenario: "scenario-b"}},
	}
	runtime := map[string]any{
		"scenario_mix": map[string]any{
			"scenario-a": map[string]any{"train": json.Number("1"), "validation": json.Number("1")},
			"scenario-b": map[string]any{"train": json.Number("1"), "validation": json.Number("1")},
		},
	}
	if err := validateDeterministicSplits(matches, -1, 0.5, runtime); err != nil {
		t.Fatalf("stratified split without schedule rejected: %v", err)
	}
}

func TestValidateDeterministicStratifiedSplitsRejectsTamperingAndMalformedInputs(t *testing.T) {
	tests := []struct {
		name               string
		mutate             func(map[string]any)
		tamperMatch        bool
		validationFraction float64
		wantString         string
	}{
		{
			name:        "tampered split",
			tamperMatch: true,
			wantString:  "deterministic split",
		},
		{
			name: "malformed quota fields",
			mutate: func(runtime map[string]any) {
				runtime["scenario_mix"].(map[string]any)["scenario-a"] = map[string]any{
					"train": json.Number("1"), "validation": json.Number("1"), "extra": json.Number("0"),
				}
			},
			wantString: "scenario_mix",
		},
		{
			name: "quota does not match scenario",
			mutate: func(runtime map[string]any) {
				runtime["scenario_mix"].(map[string]any)["scenario-a"].(map[string]any)["train"] = json.Number("0")
			},
			wantString: "quota",
		},
		{
			name:               "validation fraction mismatch",
			validationFraction: 0.25,
			wantString:         "validation_fraction",
		},
		{
			name: "malformed schedule type",
			mutate: func(runtime map[string]any) {
				runtime["match_schedule"] = map[string]any{}
			},
			wantString: "match_schedule",
		},
		{
			name: "malformed schedule entry",
			mutate: func(runtime map[string]any) {
				runtime["match_schedule"].([]any)[0].(map[string]any)["scenario"] = nil
			},
			wantString: "match_schedule",
		},
		{
			name: "schedule does not match generation",
			mutate: func(runtime map[string]any) {
				runtime["match_schedule"].([]any)[0].(map[string]any)["match_id"] = "unknown-match"
			},
			wantString: "frozen schedule",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			matches := stratifiedTestMatches()
			runtime := stratifiedRuntimeManifest(
				[]string{"match-00", "match-01", "match-02", "match-03"},
				[]string{"scenario-a", "scenario-a", "scenario-b", "scenario-b"},
			)
			if test.tamperMatch {
				matches[2].metadata.Split = "train"
			}
			if test.mutate != nil {
				test.mutate(runtime)
			}
			fraction := test.validationFraction
			if fraction == 0 {
				fraction = 0.5
			}
			err := validateDeterministicSplits(matches, -1, fraction, runtime)
			if err == nil || !strings.Contains(err.Error(), test.wantString) {
				t.Fatalf("error=%v, want substring %q", err, test.wantString)
			}
		})
	}
}

func stratifiedTestMatches() []*matchRecord {
	splits := map[string]string{
		"match-00": "train", "match-01": "validation",
		"match-02": "validation", "match-03": "train",
	}
	matches := make([]*matchRecord, 0, len(splits))
	for _, id := range []string{"match-00", "match-01", "match-02", "match-03"} {
		scenario := "scenario-a"
		if id >= "match-02" {
			scenario = "scenario-b"
		}
		matches = append(matches, &matchRecord{metadata: MatchMetadata{
			MatchID: id, Split: splits[id], Scenario: scenario,
		}})
	}
	return matches
}

func stratifiedRuntimeManifest(matchIDs, scenarios []string) map[string]any {
	schedule := make([]any, len(matchIDs))
	for index := range matchIDs {
		schedule[index] = map[string]any{
			"match_id": matchIDs[index], "scenario": scenarios[index],
		}
	}
	return map[string]any{
		"scenario_mix": map[string]any{
			"scenario-a": map[string]any{"train": json.Number("1"), "validation": json.Number("1")},
			"scenario-b": map[string]any{"train": json.Number("1"), "validation": json.Number("1")},
		},
		"match_schedule": schedule,
	}
}

func TestGenerationMatchMetadataIsImmutableByCopy(t *testing.T) {
	root := filepath.Join(t.TempDir(), "generation")
	if err := WriteGeneration(root, acceptancePrepared(t)); err != nil {
		t.Fatal(err)
	}
	generation, err := OpenGenerationWithOptions(root, OpenOptions{DeferShardHashing: true})
	if err != nil {
		t.Fatal(err)
	}
	matches := generation.Matches()
	matches[0].HeroIDs[0] = "mutated"
	metadata, ok := generation.MatchMetadata("ai42-test-match")
	if !ok || metadata.HeroIDs[0] == "mutated" {
		t.Fatalf("generation metadata was mutable through returned slices: %+v", metadata)
	}
}

func TestReadTargetRowsFiltersExactRangesAfterAuthoritativeVerification(t *testing.T) {
	root := filepath.Join(t.TempDir(), "generation")
	prepared := preparedForMatchTicks(t, "target-match", 5)
	if err := WriteGeneration(root, prepared); err != nil {
		t.Fatal(err)
	}
	generation, err := OpenGenerationWithOptions(root, OpenOptions{DeferShardHashing: true})
	if err != nil {
		t.Fatal(err)
	}
	var ticks []int
	count, err := generation.ReadTargetRows(context.Background(), map[string][][2]int{
		"target-match": {{0, 1}, {2, 4}},
	}, func(row Row) error {
		if row.MatchID != "target-match" {
			t.Fatalf("unexpected callback identity: %s", row.MatchID)
		}
		ticks = append(ticks, row.Tick)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 || !equalInts(ticks, []int{0, 2, 3}) {
		t.Fatalf("target rows count=%d ticks=%v", count, ticks)
	}

	shardPath := filepath.Join(root, "shard-000000.a42")
	payload, err := os.ReadFile(shardPath)
	if err != nil {
		t.Fatal(err)
	}
	payload[len(payload)-1] ^= 0x20
	if err := os.WriteFile(shardPath, payload, 0o600); err != nil {
		t.Fatal(err)
	}
	callbacks := 0
	if _, err := generation.ReadTargetRows(context.Background(), map[string][][2]int{
		"target-match": {{0, 1}},
	}, func(Row) error {
		callbacks++
		return nil
	}); err == nil {
		t.Fatal("targeted read accepted a corrupt containing shard")
	}
	if callbacks != 0 {
		t.Fatalf("corrupt shard delivered %d callbacks before authoritative verification", callbacks)
	}
}

func TestReadTargetRowsRejectsInvalidRangesAndUnavailableState(t *testing.T) {
	root := filepath.Join(t.TempDir(), "generation")
	if err := WriteGeneration(root, preparedForMatchTicks(t, "target-match", 5)); err != nil {
		t.Fatal(err)
	}
	generation, err := OpenGenerationWithOptions(root, OpenOptions{DeferShardHashing: true})
	if err != nil {
		t.Fatal(err)
	}
	callback := func(Row) error { return nil }
	tests := []struct {
		name    string
		targets map[string][][2]int
		want    string
	}{
		{name: "empty selection", targets: nil, want: "non-empty"},
		{name: "unknown match", targets: map[string][][2]int{"missing": {{0, 1}}}, want: "unknown target match"},
		{name: "empty ranges", targets: map[string][][2]int{"target-match": {}}, want: "has no ranges"},
		{name: "too many ranges", targets: map[string][][2]int{"target-match": {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {4, 5}}}, want: "range count"},
		{name: "negative", targets: map[string][][2]int{"target-match": {{-1, 1}}}, want: "outside"},
		{name: "empty interval", targets: map[string][][2]int{"target-match": {{2, 2}}}, want: "outside"},
		{name: "inverted", targets: map[string][][2]int{"target-match": {{3, 2}}}, want: "outside"},
		{name: "past end", targets: map[string][][2]int{"target-match": {{4, 6}}}, want: "outside"},
		{name: "unsorted", targets: map[string][][2]int{"target-match": {{3, 4}, {0, 1}}}, want: "sorted and disjoint"},
		{name: "overlap", targets: map[string][][2]int{"target-match": {{0, 3}, {2, 4}}}, want: "sorted and disjoint"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := generation.ReadTargetRows(context.Background(), test.targets, callback); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error=%v, want %q", err, test.want)
			}
		})
	}
	if _, err := generation.ReadTargetRows(context.Background(), map[string][][2]int{"target-match": {{0, 1}}}, nil); err == nil || !strings.Contains(err.Error(), "non-nil") {
		t.Fatalf("nil callback error=%v", err)
	}
	var unavailable *Generation
	if _, err := unavailable.ReadTargetRows(context.Background(), map[string][][2]int{"target-match": {{0, 1}}}, callback); err == nil || !strings.Contains(err.Error(), "unavailable") {
		t.Fatalf("unavailable state error=%v", err)
	}
}

func TestTargetRowStateRejectsMissingDuplicateAndIdentityCallbacks(t *testing.T) {
	newState := func() *targetRowState {
		return &targetRowState{matchID: "target-match", tickCount: 4, ranges: [][2]int{{1, 3}}, nextTick: 1, lastTick: -1}
	}
	t.Run("duplicate", func(t *testing.T) {
		state := newState()
		if forwarded, err := state.consume(1); err != nil || !forwarded {
			t.Fatalf("first callback forwarded=%t err=%v", forwarded, err)
		}
		if _, err := state.consume(1); err == nil || !strings.Contains(err.Error(), "duplicate") {
			t.Fatalf("duplicate callback error=%v", err)
		}
	})
	t.Run("missing", func(t *testing.T) {
		state := newState()
		if _, err := state.consume(2); err == nil || !strings.Contains(err.Error(), "missing") {
			t.Fatalf("missing callback error=%v", err)
		}
	})
	t.Run("finish missing", func(t *testing.T) {
		state := newState()
		if err := state.finish(); err == nil || !strings.Contains(err.Error(), "missing") {
			t.Fatalf("finish error=%v", err)
		}
	})
	t.Run("identity outside match", func(t *testing.T) {
		state := newState()
		if _, err := state.consume(4); err == nil || !strings.Contains(err.Error(), "outside") {
			t.Fatalf("identity error=%v", err)
		}
	})
}

func TestReadTargetRowsRejectsManifestIdentityMutation(t *testing.T) {
	root := filepath.Join(t.TempDir(), "generation")
	if err := WriteGeneration(root, preparedForMatchTicks(t, "target-match", 2)); err != nil {
		t.Fatal(err)
	}
	generation, err := OpenGenerationWithOptions(root, OpenOptions{DeferShardHashing: true})
	if err != nil {
		t.Fatal(err)
	}
	manifestPath := filepath.Join(root, "manifest.json")
	payload, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(manifestPath, append(payload, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	callbacks := 0
	_, err = generation.ReadTargetRows(context.Background(), map[string][][2]int{
		"target-match": {{0, 1}},
	}, func(Row) error {
		callbacks++
		return nil
	})
	if err == nil || !strings.Contains(err.Error(), "manifest changed after open") {
		t.Fatalf("manifest identity error=%v", err)
	}
	if callbacks != 0 {
		t.Fatalf("mutated manifest delivered %d callbacks", callbacks)
	}
}

func TestReadTargetRowsRejectsManifestMutationDuringCallback(t *testing.T) {
	root := filepath.Join(t.TempDir(), "generation")
	if err := WriteGeneration(root, preparedForMatchTicks(t, "target-match", 2)); err != nil {
		t.Fatal(err)
	}
	generation, err := OpenGenerationWithOptions(root, OpenOptions{DeferShardHashing: true})
	if err != nil {
		t.Fatal(err)
	}
	manifestPath := filepath.Join(root, "manifest.json")
	count, err := generation.ReadTargetRows(context.Background(), map[string][][2]int{
		"target-match": {{0, 1}},
	}, func(Row) error {
		payload, readErr := os.ReadFile(manifestPath)
		if readErr != nil {
			return readErr
		}
		return os.WriteFile(manifestPath, append(payload, '\n'), 0o600)
	})
	if err == nil || !strings.Contains(err.Error(), "manifest changed after open") {
		t.Fatalf("manifest mutation error=%v", err)
	}
	if count != 1 {
		t.Fatalf("completed callback count=%d, want 1 before post-verification identity rejection", count)
	}
}

func TestVerifiedSplitMatchIDsReturnsDefensiveManifestOrder(t *testing.T) {
	root := writeMultiShardGeneration(t, 3)
	generation, err := OpenGenerationWithOptions(root, OpenOptions{DeferShardHashing: true})
	if err != nil {
		t.Fatal(err)
	}
	splits, err := generation.VerifiedSplitMatchIDs()
	if err != nil {
		t.Fatal(err)
	}
	wantTrain := []string{multiMatchID(0), multiMatchID(1), multiMatchID(2)}
	if !equalStrings(splits["train"], wantTrain) || len(splits["validation"]) != 0 {
		t.Fatalf("split IDs=%v", splits)
	}
	splits["train"][0] = "mutated"
	delete(splits, "validation")
	again, err := generation.VerifiedSplitMatchIDs()
	if err != nil {
		t.Fatal(err)
	}
	if !equalStrings(again["train"], wantTrain) || again["validation"] == nil {
		t.Fatalf("split result was not defensive: %v", again)
	}

	validationRoot := filepath.Join(t.TempDir(), "validation-generation")
	if err := WriteGenerationWithSplit(validationRoot, acceptancePrepared(t), 17, 1); err != nil {
		t.Fatal(err)
	}
	validationGeneration, err := OpenGenerationWithOptions(validationRoot, OpenOptions{DeferShardHashing: true})
	if err != nil {
		t.Fatal(err)
	}
	validation, err := validationGeneration.VerifiedSplitMatchIDs()
	if err != nil {
		t.Fatal(err)
	}
	if len(validation["train"]) != 0 || !equalStrings(validation["validation"], []string{"ai42-test-match"}) {
		t.Fatalf("validation split IDs=%v", validation)
	}
}

func TestVerifiedSplitMatchIDsRejectsUnavailableAndMutatedState(t *testing.T) {
	var unavailable *Generation
	if _, err := unavailable.VerifiedSplitMatchIDs(); err == nil || !strings.Contains(err.Error(), "unavailable") {
		t.Fatalf("nil generation error=%v", err)
	}
	if _, err := (&Generation{}).VerifiedSplitMatchIDs(); err == nil || !strings.Contains(err.Error(), "unavailable") {
		t.Fatalf("empty generation error=%v", err)
	}
	root := filepath.Join(t.TempDir(), "generation")
	if err := WriteGeneration(root, acceptancePrepared(t)); err != nil {
		t.Fatal(err)
	}
	generation, err := OpenGenerationWithOptions(root, OpenOptions{DeferShardHashing: true})
	if err != nil {
		t.Fatal(err)
	}
	manifestPath := filepath.Join(root, "manifest.json")
	payload, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(manifestPath, append(payload, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := generation.VerifiedSplitMatchIDs(); err == nil || !strings.Contains(err.Error(), "manifest changed after open") {
		t.Fatalf("mutated manifest error=%v", err)
	}
}

func TestVerifyGenerationChecksDerivedV2LineageAcrossRows(t *testing.T) {
	metadata := testMetadata()
	capture, err := NewCapture(metadata)
	if err != nil {
		t.Fatal(err)
	}
	for tick := 0; tick < 2; tick++ {
		result := testResult(uint32(tick), tick == 1)
		actions, parents, boundaries, outcomes := testRows(metadata.MatchID, tick)
		for slot := 0; slot < HeroCount; slot++ {
			outcomes[slot].Reward = result.Reward[slot]
			outcomes[slot].Terminal = result.Done
		}
		if err := capture.Append(&result, actions, parents, boundaries, outcomes); err != nil {
			t.Fatal(err)
		}
	}
	prepared, err := capture.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Join(t.TempDir(), "generation")
	if err := WriteGeneration(root, prepared); err != nil {
		t.Fatal(err)
	}
	generation, err := OpenGeneration(root, "")
	if err != nil {
		t.Fatal(err)
	}
	report, err := generation.Verify(context.Background(), VerifyOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if report.Rows != 2 {
		t.Fatalf("rows=%d, want 2", report.Rows)
	}
}

func TestVerifyMatchIDsTargetsOnlyReferencedShards(t *testing.T) {
	root := writeMultiShardGeneration(t, 3)
	generation, err := OpenGenerationWithOptions(root, OpenOptions{DeferShardHashing: true})
	if err != nil {
		t.Fatal(err)
	}
	corruptPath := filepath.Join(root, "shard-000002.a42")
	payload, err := os.ReadFile(corruptPath)
	if err != nil {
		t.Fatal(err)
	}
	payload[len(payload)-1] ^= 0x40
	if err := os.WriteFile(corruptPath, payload, 0o600); err != nil {
		t.Fatal(err)
	}

	var rows, matches []string
	target := multiMatchID(1)
	report, err := generation.VerifyMatchIDs(context.Background(), []string{target}, VerifyOptions{
		OnRow: func(row Row) error {
			rows = append(rows, row.MatchID)
			return nil
		},
		OnMatch: func(match MatchMetadata) error {
			matches = append(matches, match.MatchID)
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if report.Shards != 1 || report.Matches != 1 || report.Rows != 1 || len(report.Files) != 1 || report.Files[0].Shard != "shard-000001.a42" {
		t.Fatalf("unexpected targeted report: %+v", report)
	}
	if len(rows) != 1 || rows[0] != target || len(matches) != 1 || matches[0] != target {
		t.Fatalf("targeted callbacks rows=%v matches=%v", rows, matches)
	}
	if _, err := generation.Verify(context.Background(), VerifyOptions{}); err == nil {
		t.Fatal("full verification did not inspect the corrupt unselected shard")
	}
}

func TestVerifySelectionAndWorkerBounds(t *testing.T) {
	root := writeMultiShardGeneration(t, 2)
	generation, err := OpenGenerationWithOptions(root, OpenOptions{DeferShardHashing: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name string
		run  func() error
		want string
	}{
		{name: "negative workers", run: func() error {
			_, err := generation.Verify(context.Background(), VerifyOptions{Workers: -1})
			return err
		}, want: "workers"},
		{name: "too many workers", run: func() error {
			_, err := generation.Verify(context.Background(), VerifyOptions{Workers: maxVerificationWorkers + 1})
			return err
		}, want: "workers"},
		{name: "duplicate match", run: func() error {
			_, err := generation.VerifyMatchIDs(context.Background(), []string{multiMatchID(0), multiMatchID(0)}, VerifyOptions{})
			return err
		}, want: "duplicate selected match"},
		{name: "unknown match", run: func() error {
			_, err := generation.VerifyMatchIDs(context.Background(), []string{"missing"}, VerifyOptions{})
			return err
		}, want: "unknown selected match"},
		{name: "duplicate shard", run: func() error {
			_, err := generation.VerifyShards(context.Background(), []string{"shard-000000.a42", "shard-000000.a42"}, VerifyOptions{})
			return err
		}, want: "duplicate selected shard"},
		{name: "unknown shard", run: func() error {
			_, err := generation.VerifyShards(context.Background(), []string{"missing.a42"}, VerifyOptions{})
			return err
		}, want: "unknown selected shard"},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := test.run(); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error=%v, want %q", err, test.want)
			}
		})
	}
}

func TestVerifyWorkersCallbacksAreConcurrentAndReportIsDeterministic(t *testing.T) {
	root := writeMultiShardGeneration(t, 4)
	generation, err := OpenGenerationWithOptions(root, OpenOptions{DeferShardHashing: true})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	gate := make(chan struct{})
	var closeGate sync.Once
	var entered atomic.Int32
	seen := make(map[string]int)
	var seenMu sync.Mutex
	report, err := generation.Verify(ctx, VerifyOptions{
		Workers: 4,
		OnRow: func(row Row) error {
			if entered.Add(1) == 4 {
				closeGate.Do(func() { close(gate) })
			}
			select {
			case <-gate:
			case <-ctx.Done():
				return ctx.Err()
			}
			seenMu.Lock()
			seen[row.MatchID]++
			seenMu.Unlock()
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if entered.Load() != 4 || len(seen) != 4 || report.Shards != 4 || report.Matches != 4 || report.Rows != 4 || len(report.Files) != 4 {
		t.Fatalf("parallel result entered=%d seen=%v report=%+v", entered.Load(), seen, report)
	}
	for index, evidence := range report.Files {
		want := fmt.Sprintf("shard-%06d.a42", index)
		if evidence.Shard != want {
			t.Fatalf("evidence order[%d]=%q, want %q", index, evidence.Shard, want)
		}
	}
}

func TestVerifyWorkersReturnCanonicalConcreteError(t *testing.T) {
	root := writeMultiShardGeneration(t, 3)
	generation, err := OpenGenerationWithOptions(root, OpenOptions{DeferShardHashing: true})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	gate := make(chan struct{})
	var closeGate sync.Once
	var entered atomic.Int32
	report, err := generation.Verify(ctx, VerifyOptions{
		Workers: 3,
		OnRow: func(row Row) error {
			if entered.Add(1) == 3 {
				closeGate.Do(func() { close(gate) })
			}
			select {
			case <-gate:
				return fmt.Errorf("callback failure for %s", row.MatchID)
			case <-ctx.Done():
				return ctx.Err()
			}
		},
	})
	if err == nil || !strings.Contains(err.Error(), multiMatchID(0)) {
		t.Fatalf("error=%v, want canonical first shard callback error", err)
	}
	if report.Shards != 0 || len(report.Files) != 0 {
		t.Fatalf("failed shards exposed successful evidence: %+v", report)
	}
}

func TestVerifyHonorsExternalCancellation(t *testing.T) {
	root := writeMultiShardGeneration(t, 2)
	generation, err := OpenGenerationWithOptions(root, OpenOptions{DeferShardHashing: true})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := generation.Verify(ctx, VerifyOptions{Workers: 2}); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation error=%v", err)
	}
}

func TestVerifyCleansPerShardTemporaryFilesAfterCallbackError(t *testing.T) {
	root := writeMultiShardGeneration(t, 3)
	generation, err := OpenGenerationWithOptions(root, OpenOptions{DeferShardHashing: true})
	if err != nil {
		t.Fatal(err)
	}
	spoolRoot := filepath.Join(t.TempDir(), "spool")
	if err := os.Mkdir(spoolRoot, 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("TMP", spoolRoot)
	t.Setenv("TEMP", spoolRoot)
	t.Setenv("TMPDIR", spoolRoot)
	_, err = generation.Verify(context.Background(), VerifyOptions{
		Workers: 3,
		OnRow: func(Row) error {
			return errors.New("stop after verified row")
		},
	})
	if err == nil || !strings.Contains(err.Error(), "stop after verified row") {
		t.Fatalf("callback error=%v", err)
	}
	entries, err := os.ReadDir(spoolRoot)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("verification left temporary files behind: %v", entries)
	}
}

func TestOpenGenerationRejectsPinnedHashAndContainedPathViolations(t *testing.T) {
	root := filepath.Join(t.TempDir(), "generation")
	if err := WriteGeneration(root, acceptancePrepared(t)); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenGeneration(root, strings.Repeat("0", 64)); err == nil || !strings.Contains(err.Error(), "expected manifest hash") {
		t.Fatalf("hash pin error=%v", err)
	}
	manifestPath := filepath.Join(root, "manifest.json")
	payload, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	var manifest map[string]any
	if err := json.Unmarshal(payload, &manifest); err != nil {
		t.Fatal(err)
	}
	shards := manifest["shards"].([]any)
	shards[0].(map[string]any)["name"] = "../outside.a42"
	unsigned := make(map[string]any, len(manifest)-1)
	for key, value := range manifest {
		if key != "manifest_hash" {
			unsigned[key] = value
		}
	}
	manifest["manifest_hash"] = sha256Hex(canonicalJSON(unsigned))
	if err := os.WriteFile(manifestPath, canonicalJSON(manifest), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenGeneration(root, ""); err == nil || !strings.Contains(err.Error(), "contained") {
		t.Fatalf("path containment error=%v", err)
	}
}

func TestDecodeCanonicalJSONRejectsDuplicateAndUnknownFields(t *testing.T) {
	if _, err := decodeCanonicalJSON([]byte(`{"a":1,"a":1}`), "fixture"); err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("duplicate key error=%v", err)
	}
	if err := requireFields(map[string]any{"known": true, "extra": true}, map[string]struct{}{"known": {}}, "fixture"); err == nil || !strings.Contains(err.Error(), "unknown") {
		t.Fatalf("unknown field error=%v", err)
	}
}

func TestOpenGenerationRejectsProductionLimitExploits(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(map[string]any)
		want   string
	}{
		{
			name: "oversized raw declaration",
			mutate: func(manifest map[string]any) {
				manifest["shards"].([]any)[0].(map[string]any)["raw_bytes"] = maxShardRawBytes + 1
			},
			want: "raw_bytes",
		},
		{
			name: "oversized stored declaration",
			mutate: func(manifest map[string]any) {
				manifest["shards"].([]any)[0].(map[string]any)["stored_bytes"] = maxShardStoredBytes + 1
			},
			want: "stored_bytes",
		},
		{
			name: "oversized match row dimension",
			mutate: func(manifest map[string]any) {
				manifest["matches"].([]any)[0].(map[string]any)["tick_count"] = maxMatchRows + 1
			},
			want: "tick_count",
		},
		{
			name: "oversized shard row dimension",
			mutate: func(manifest map[string]any) {
				manifest["shards"].([]any)[0].(map[string]any)["row_count"] = maxShardRows + 1
			},
			want: "row_count",
		},
		{
			name: "oversized match count",
			mutate: func(manifest map[string]any) {
				entry := manifest["matches"].([]any)[0]
				matches := make([]any, maxGenerationMatches+1)
				for index := range matches {
					matches[index] = entry
				}
				manifest["matches"] = matches
			},
			want: "matches count",
		},
		{
			name: "oversized shard count",
			mutate: func(manifest map[string]any) {
				entry := manifest["shards"].([]any)[0]
				shards := make([]any, maxGenerationShards+1)
				for index := range shards {
					shards[index] = entry
				}
				manifest["shards"] = shards
			},
			want: "shards count",
		},
		{
			name: "oversized shard match count",
			mutate: func(manifest map[string]any) {
				ids := make([]string, maxMatchesPerShard+1)
				for index := range ids {
					ids[index] = "match-" + itoa(index)
				}
				manifest["shards"].([]any)[0].(map[string]any)["match_ids"] = ids
			},
			want: "match_ids count",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			root := filepath.Join(t.TempDir(), "generation")
			if err := WriteGeneration(root, acceptancePrepared(t)); err != nil {
				t.Fatal(err)
			}
			rewriteGenerationManifest(t, root, test.mutate)
			if _, err := OpenGeneration(root, ""); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("OpenGeneration error=%v, want %q", err, test.want)
			}
		})
	}
}

func TestOpenGenerationRejectsOversizedManifestBeforeRead(t *testing.T) {
	root := filepath.Join(t.TempDir(), "generation")
	if err := WriteGeneration(root, acceptancePrepared(t)); err != nil {
		t.Fatal(err)
	}
	if err := os.Truncate(filepath.Join(root, "manifest.json"), maxManifestBytes+1); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenGeneration(root, ""); err == nil || !strings.Contains(err.Error(), "outside") {
		t.Fatalf("oversized manifest error=%v", err)
	}
}

func TestVerifyRejectsZipBombBeyondDeclaredRawBytes(t *testing.T) {
	root := filepath.Join(t.TempDir(), "generation")
	if err := WriteGeneration(root, acceptancePrepared(t)); err != nil {
		t.Fatal(err)
	}
	shardPath := filepath.Join(root, "shard-000000.a42")
	shardBytes, err := os.ReadFile(shardPath)
	if err != nil {
		t.Fatal(err)
	}
	headerLength := int(binary.LittleEndian.Uint32(shardBytes[len(ShardMagic):]))
	headerStart := len(ShardMagic) + 4
	value, err := decodeCanonicalJSON(shardBytes[headerStart:headerStart+headerLength], "test.header")
	if err != nil {
		t.Fatal(err)
	}
	header := value.(map[string]any)
	declaredRaw, err := int64Field(header["raw_bytes"], "raw_bytes", 1)
	if err != nil {
		t.Fatal(err)
	}
	var compressed bytes.Buffer
	compressor, err := flate.NewWriter(&compressed, 6)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := compressor.Write(make([]byte, declaredRaw+64*1024)); err != nil {
		t.Fatal(err)
	}
	if err := compressor.Close(); err != nil {
		t.Fatal(err)
	}
	header["stored_bytes"] = compressed.Len()
	header["payload_sha256"] = sha256Hex(compressed.Bytes())
	headerBytes := canonicalJSON(header)
	var rebuilt bytes.Buffer
	rebuilt.WriteString(ShardMagic)
	var length [4]byte
	binary.LittleEndian.PutUint32(length[:], uint32(len(headerBytes)))
	rebuilt.Write(length[:])
	rebuilt.Write(headerBytes)
	rebuilt.Write(compressed.Bytes())
	if err := os.WriteFile(shardPath, rebuilt.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	rewriteGenerationManifest(t, root, func(manifest map[string]any) {
		shard := manifest["shards"].([]any)[0].(map[string]any)
		shard["sha256"] = sha256Hex(rebuilt.Bytes())
		shard["stored_bytes"] = rebuilt.Len()
	})
	generation, err := OpenGeneration(root, "")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := generation.Verify(context.Background(), VerifyOptions{}); err == nil || !strings.Contains(err.Error(), "exceeds declared raw_bytes") {
		t.Fatalf("zip-bomb verification error=%v", err)
	}
}

func TestVerifyRejectsPostOpenManifestMutation(t *testing.T) {
	root := filepath.Join(t.TempDir(), "generation")
	if err := WriteGeneration(root, acceptancePrepared(t)); err != nil {
		t.Fatal(err)
	}
	generation, err := OpenGeneration(root, "")
	if err != nil {
		t.Fatal(err)
	}
	manifestPath := filepath.Join(root, "manifest.json")
	payload, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(manifestPath, append(payload, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := generation.Verify(context.Background(), VerifyOptions{}); err == nil || !strings.Contains(err.Error(), "manifest changed after open") {
		t.Fatalf("manifest mutation verification error=%v", err)
	}
}

func TestVerifyRejectsPostOpenManifestReplacementWithSameBytes(t *testing.T) {
	root := filepath.Join(t.TempDir(), "generation")
	if err := WriteGeneration(root, acceptancePrepared(t)); err != nil {
		t.Fatal(err)
	}
	generation, err := OpenGeneration(root, "")
	if err != nil {
		t.Fatal(err)
	}
	manifestPath := filepath.Join(root, "manifest.json")
	payload, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	replacement := filepath.Join(root, "manifest.replacement")
	if err := os.WriteFile(replacement, payload, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(manifestPath); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(replacement, manifestPath); err != nil {
		t.Fatal(err)
	}
	if _, err := generation.Verify(context.Background(), VerifyOptions{}); err == nil || !strings.Contains(err.Error(), "file identity mismatch") {
		t.Fatalf("manifest replacement verification error=%v", err)
	}
}

func TestVerifyRejectsPostOpenManifestSymlinkEscape(t *testing.T) {
	root := filepath.Join(t.TempDir(), "generation")
	if err := WriteGeneration(root, acceptancePrepared(t)); err != nil {
		t.Fatal(err)
	}
	generation, err := OpenGeneration(root, "")
	if err != nil {
		t.Fatal(err)
	}
	manifestPath := filepath.Join(root, "manifest.json")
	payload, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(t.TempDir(), "outside-manifest.json")
	if err := os.WriteFile(outside, payload, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(manifestPath); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, manifestPath); err != nil {
		t.Skipf("symlink creation unavailable: %v", err)
	}
	if _, err := generation.Verify(context.Background(), VerifyOptions{}); err == nil || !strings.Contains(err.Error(), "manifest changed after open") {
		t.Fatalf("manifest symlink verification error=%v", err)
	}
}

func TestAddCappedTotalRejectsOverflowAndLimit(t *testing.T) {
	for _, test := range []struct {
		name         string
		total, value int64
	}{
		{name: "limit", total: 8, value: 3},
		{name: "overflow", total: math.MaxInt64 - 1, value: 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := addCappedTotal(test.total, test.value, 10, "fixture"); err == nil {
				t.Fatal("addCappedTotal accepted unsafe cumulative value")
			}
		})
	}
}

func preparedForMatch(tb testing.TB, matchID string) *Prepared {
	tb.Helper()
	return preparedForMatchTicks(tb, matchID, 1)
}

func preparedForMatchTicks(tb testing.TB, matchID string, ticks int) *Prepared {
	tb.Helper()
	metadata := testMetadata()
	metadata.MatchID = matchID
	for slot := 0; slot < HeroCount; slot++ {
		metadata.HeroIDs[slot] = matchID + ":hero:" + twoDigits(slot)
	}
	capture, err := NewCapture(metadata)
	if err != nil {
		tb.Fatal(err)
	}
	for tick := 0; tick < ticks; tick++ {
		result := testResult(uint32(tick), tick == ticks-1)
		actions, parents, boundaries, outcomes := testRows(matchID, tick)
		for slot := 0; slot < HeroCount; slot++ {
			outcomes[slot] = Outcome{Reward: result.Reward[slot], Terminal: result.Done}
		}
		if err := capture.Append(&result, actions, parents, boundaries, outcomes); err != nil {
			tb.Fatal(err)
		}
	}
	prepared, err := capture.Finalize()
	if err != nil {
		tb.Fatal(err)
	}
	return prepared
}

func multiMatchID(index int) string {
	return fmt.Sprintf("ai42-test-match-%06d", index)
}

func writeMultiShardGeneration(tb testing.TB, count int) string {
	tb.Helper()
	root := filepath.Join(tb.TempDir(), "generation")
	if err := os.Mkdir(root, 0o700); err != nil {
		tb.Fatal(err)
	}
	matches := make([]any, 0, count)
	shards := make([]any, 0, count)
	var first *Prepared
	for index := 0; index < count; index++ {
		prepared := preparedForMatch(tb, multiMatchID(index))
		if first == nil {
			first = prepared
		}
		name := fmt.Sprintf("shard-%06d.a42", index)
		shardBytes, err := encodeShard(prepared, "train")
		if err != nil {
			tb.Fatal(err)
		}
		shardBytes = rebindEncodedShard(tb, shardBytes, name)
		if err := os.WriteFile(filepath.Join(root, name), shardBytes, 0o600); err != nil {
			tb.Fatal(err)
		}
		matches = append(matches, matchEntry(prepared, name, "train"))
		shards = append(shards, map[string]any{
			"name": name, "sha256": sha256Hex(shardBytes), "match_ids": []string{prepared.Metadata.MatchID},
			"row_count": prepared.TickCount, "raw_bytes": rawBytesFromShard(shardBytes),
			"stored_bytes": len(shardBytes), "compression": ShardCodec,
		})
	}
	if first == nil {
		tb.Fatal("multi-shard fixture requires at least one match")
	}
	runtimeValue, err := decodeCanonicalJSON(first.Metadata.RuntimeManifest, "fixture.runtime_manifest")
	if err != nil {
		tb.Fatal(err)
	}
	unsigned := map[string]any{
		"dataset_schema_version": DatasetSchemaVersion, "shard_schema_version": ShardSchemaVersionV2,
		"protocol_version": int(ProtocolVersion), "schema_hash": hashHex(first.Metadata.SchemaHash),
		"reward_hash": hashHex(first.Metadata.RewardHash), "trajectory_schema_hash": hashHex(first.Metadata.TrajectorySchemaHash),
		"runtime_manifest_hash": hashHex(first.Metadata.RuntimeManifestHash), "runtime_manifest": runtimeValue,
		"split_seed": int64(0), "validation_fraction": 0.0, "matches": matches, "shards": shards,
	}
	manifest := make(map[string]any, len(unsigned)+1)
	for key, value := range unsigned {
		manifest[key] = value
	}
	manifest["manifest_hash"] = sha256Hex(canonicalJSON(unsigned))
	if err := os.WriteFile(filepath.Join(root, "manifest.json"), canonicalJSON(manifest), 0o600); err != nil {
		tb.Fatal(err)
	}
	return root
}

func rebindEncodedShard(tb testing.TB, payload []byte, shardName string) []byte {
	tb.Helper()
	headerLength := int(binary.LittleEndian.Uint32(payload[len(ShardMagic):]))
	headerStart := len(ShardMagic) + 4
	value, err := decodeCanonicalJSON(payload[headerStart:headerStart+headerLength], "fixture.header")
	if err != nil {
		tb.Fatal(err)
	}
	header := value.(map[string]any)
	header["matches"].([]any)[0].(map[string]any)["shard"] = shardName
	headerBytes := canonicalJSON(header)
	result := make([]byte, 0, len(payload)-headerLength+len(headerBytes))
	result = append(result, []byte(ShardMagic)...)
	var length [4]byte
	binary.LittleEndian.PutUint32(length[:], uint32(len(headerBytes)))
	result = append(result, length[:]...)
	result = append(result, headerBytes...)
	result = append(result, payload[headerStart+headerLength:]...)
	return result
}

func rewriteGenerationManifest(t *testing.T, root string, mutate func(map[string]any)) {
	t.Helper()
	path := filepath.Join(root, "manifest.json")
	payload, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var manifest map[string]any
	if err := json.Unmarshal(payload, &manifest); err != nil {
		t.Fatal(err)
	}
	mutate(manifest)
	unsigned := make(map[string]any, len(manifest)-1)
	for key, value := range manifest {
		if key != "manifest_hash" {
			unsigned[key] = value
		}
	}
	manifest["manifest_hash"] = sha256Hex(canonicalJSON(unsigned))
	if err := os.WriteFile(path, canonicalJSON(manifest), 0o600); err != nil {
		t.Fatal(err)
	}
}

func FuzzDecodeCanonicalJSONNeverPanics(f *testing.F) {
	f.Add([]byte(`{"a":1}`))
	f.Add([]byte(`{"a":1,"a":2}`))
	f.Add([]byte(`not-json`))
	f.Fuzz(func(t *testing.T, payload []byte) {
		_, _ = decodeCanonicalJSON(payload, "fuzz")
	})
}

func BenchmarkVerifyGenerationWorkers(b *testing.B) {
	root := writeMultiShardGeneration(b, 8)
	generation, err := OpenGenerationWithOptions(root, OpenOptions{DeferShardHashing: true})
	if err != nil {
		b.Fatal(err)
	}
	for _, workers := range []int{1, 4} {
		b.Run(fmt.Sprintf("workers-%d", workers), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := generation.Verify(context.Background(), VerifyOptions{Workers: workers}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkOpenGenerationDeferredHashing(b *testing.B) {
	root := writeMultiShardGeneration(b, 8)
	for _, deferred := range []bool{false, true} {
		b.Run(fmt.Sprintf("deferred-%t", deferred), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := OpenGenerationWithOptions(root, OpenOptions{DeferShardHashing: deferred}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
