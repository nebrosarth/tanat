package ai42dataset

import (
	"crypto/sha256"
	"math"
	"os"
	"path/filepath"
	"testing"

	"tanatserver/internal/battleserver"
)

func acceptancePrepared(t *testing.T) *Prepared {
	t.Helper()
	metadata := testMetadata()
	capture, err := NewCapture(metadata)
	if err != nil {
		t.Fatal(err)
	}
	result := testResult(0, true)
	actions, parents, boundaries, outcomes := testRows(metadata.MatchID, 0)
	for slot := 0; slot < HeroCount; slot++ {
		outcomes[slot] = Outcome{Reward: result.Reward[slot], Terminal: true}
	}
	if err := capture.Append(&result, actions, parents, boundaries, outcomes); err != nil {
		t.Fatal(err)
	}
	prepared, err := capture.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	return prepared
}

func TestNativeAcceptanceDestinationIsImmutable(t *testing.T) {
	root := t.TempDir()
	sentinel := filepath.Join(root, "sentinel")
	if err := os.WriteFile(sentinel, []byte("keep"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := WriteGeneration(root, acceptancePrepared(t)); err == nil {
		t.Errorf("WriteGeneration accepted an existing destination")
	}
	if got, err := os.ReadFile(sentinel); err != nil || string(got) != "keep" {
		t.Errorf("existing destination was modified: err=%v contents=%q", err, got)
	}
}

func TestNativeAcceptancePublicationIsWholeGenerationAtomic(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "manifest.json"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := WriteGeneration(root, acceptancePrepared(t)); err == nil {
		t.Fatalf("WriteGeneration unexpectedly succeeded with an unpublishable manifest path")
	}
	if _, err := os.Stat(filepath.Join(root, "shard-000000.a42")); err == nil {
		t.Errorf("failed publication left a visible shard without a manifest")
	}
}

func TestNativeAcceptanceReservePreservesCapturedRows(t *testing.T) {
	metadata := testMetadata()
	capture, err := NewCapture(metadata)
	if err != nil {
		t.Fatal(err)
	}
	first := testResult(0, false)
	actions, parents, boundaries, outcomes := testRows(metadata.MatchID, 0)
	if err := capture.Append(&first, actions, parents, boundaries, outcomes); err != nil {
		t.Fatal(err)
	}
	if err := capture.Reserve(100); err != nil {
		t.Fatal(err)
	}
	second := testResult(1, true)
	actions, parents, boundaries, outcomes = testRows(metadata.MatchID, 1)
	for slot := 0; slot < HeroCount; slot++ {
		outcomes[slot] = Outcome{Reward: second.Reward[slot], Terminal: true}
	}
	if err := capture.Append(&second, actions, parents, boundaries, outcomes); err != nil {
		t.Fatal(err)
	}
	prepared, err := capture.Finalize()
	if err != nil {
		t.Fatalf("Reserve lost or misaligned the first row: %v", err)
	}
	if prepared.TickCount != 2 {
		t.Fatalf("Reserve changed tick count to %d", prepared.TickCount)
	}
}

func TestNativeAcceptanceMetadataMatchesPythonProvenanceRules(t *testing.T) {
	nonCanonical := testMetadata()
	nonCanonical.RuntimeManifest = []byte(`{"b":1,"a":2}`)
	nonCanonical.RuntimeManifestHash = sha256.Sum256(nonCanonical.RuntimeManifest)
	if _, err := NewCapture(nonCanonical); err == nil {
		t.Errorf("NewCapture accepted non-canonical runtime_manifest JSON")
	}

	unbalanced := testMetadata()
	for slot := range unbalanced.SideBySlot {
		unbalanced.SideBySlot[slot] = 0
	}
	if _, err := NewCapture(unbalanced); err == nil {
		t.Errorf("NewCapture accepted side_by_slot without five slots per side")
	}

	duplicateRoster := testMetadata()
	duplicateRoster.RosterIDs[1] = duplicateRoster.RosterIDs[0]
	if _, err := NewCapture(duplicateRoster); err == nil {
		t.Errorf("NewCapture accepted duplicate roster IDs")
	}
}

func TestNativeAcceptanceRejectsMalformedCanonicalJSONAndSplitNumbers(t *testing.T) {
	if got, err := CanonicalizeJSON([]byte(`{"b":2, "a":1}`)); err == nil && string(got) != `{"b":2, "a":1}` {
		t.Errorf("CanonicalizeJSON accepted non-canonical JSON and rewrote it to %s", got)
	}
	root := filepath.Join(t.TempDir(), "generation")
	if err := WriteGenerationWithSplit(root, acceptancePrepared(t), 0, math.NaN()); err == nil {
		t.Errorf("WriteGenerationWithSplit accepted NaN validation_fraction")
	}
}

func TestNativeAcceptanceRejectsTerminalTeacherAndRecurrentNegatives(t *testing.T) {
	t.Run("missing terminal", func(t *testing.T) {
		metadata := testMetadata()
		capture, err := NewCapture(metadata)
		if err != nil {
			t.Fatal(err)
		}
		result := testResult(0, false)
		actions, parents, boundaries, outcomes := testRows(metadata.MatchID, 0)
		if err := capture.Append(&result, actions, parents, boundaries, outcomes); err != nil {
			t.Fatal(err)
		}
		if _, err := capture.Finalize(); err == nil {
			t.Errorf("Finalize accepted a capture without a terminal row")
		}
	})

	t.Run("teacher action status cannot wait", func(t *testing.T) {
		metadata := testMetadata()
		capture, err := NewCapture(metadata)
		if err != nil {
			t.Fatal(err)
		}
		result := testResult(0, true)
		result.TeacherStatus[0] = battleserver.AssaultTeacherStatusAction
		actions, parents, boundaries, outcomes := testRows(metadata.MatchID, 0)
		if err := capture.Append(&result, actions, parents, boundaries, outcomes); err == nil {
			t.Errorf("Append accepted ACTION teacher status with zero/wait payload")
		}
	})

	t.Run("recurrent parent must continue prior boundary", func(t *testing.T) {
		metadata := testMetadata()
		capture, err := NewCapture(metadata)
		if err != nil {
			t.Fatal(err)
		}
		first := testResult(0, false)
		actions, parents, boundaries, outcomes := testRows(metadata.MatchID, 0)
		if err := capture.Append(&first, actions, parents, boundaries, outcomes); err != nil {
			t.Fatal(err)
		}
		second := testResult(1, true)
		actions, parents, boundaries, outcomes = testRows(metadata.MatchID, 1)
		parents[0] = "wrong-parent"
		if err := capture.Append(&second, actions, parents, boundaries, outcomes); err == nil {
			t.Errorf("Append accepted a recurrent parent disconnected from the prior boundary")
		}
	})
}
