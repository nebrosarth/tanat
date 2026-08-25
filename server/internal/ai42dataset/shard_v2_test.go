package ai42dataset

import (
	"bytes"
	"compress/flate"
	"encoding/binary"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func TestShardV2IsDeterministicCanonicalAndCompact(t *testing.T) {
	prepared := acceptancePrepared(t)
	var first, second bytes.Buffer
	if err := WriteShard(&first, prepared); err != nil {
		t.Fatal(err)
	}
	if err := WriteShard(&second, prepared); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first.Bytes(), second.Bytes()) {
		t.Fatal("identical captures produced different AI42GS2 shard bytes")
	}
	if !bytes.HasPrefix(first.Bytes(), []byte(ShardMagic)) {
		t.Fatalf("magic=%q, want %q", first.Bytes()[:8], ShardMagic)
	}

	headerBytes, header := decodeShardV2Header(t, first.Bytes())
	if got := header["shard_schema_version"]; got != ShardSchemaVersionV2 {
		t.Fatalf("shard schema=%v, want %s", got, ShardSchemaVersionV2)
	}
	if !bytes.Equal(headerBytes, canonicalJSON(header)) {
		t.Fatal("shard header is not canonical JSON")
	}
	matches, ok := header["matches"].([]any)
	if !ok || len(matches) != 1 {
		t.Fatalf("matches=%T/%v, want one match", header["matches"], header["matches"])
	}
	match, ok := matches[0].(map[string]any)
	if !ok {
		t.Fatalf("match metadata type=%T, want object", matches[0])
	}
	for _, field := range []string{"ticks", "recurrent_parent_ids", "recurrent_boundary_ids"} {
		if _, exists := match[field]; exists {
			t.Errorf("v2 match metadata retained removed field %q", field)
		}
	}
	if got := match["first_step"]; got != float64(prepared.Steps[0]) {
		t.Errorf("first_step=%v, want %d", got, prepared.Steps[0])
	}
	if got := match["recurrent_lineage_schema"]; got != RecurrentLineageSchemaV2 {
		t.Errorf("lineage schema=%v, want %s", got, RecurrentLineageSchemaV2)
	}
	if len(match) != 15 {
		t.Errorf("v2 match metadata fields=%d, want 15", len(match))
	}
	trajectoryHashes, ok := match["trajectory_hashes"].([]any)
	if !ok || len(trajectoryHashes) != HeroCount {
		t.Fatalf("trajectory hashes=%T/%v, want ten hashes", match["trajectory_hashes"], match["trajectory_hashes"])
	}
	for hero, value := range trajectoryHashes {
		if value != hashHex(prepared.TrajectoryHashes[hero]) {
			t.Errorf("trajectory hash %d=%v, want %s", hero, value, hashHex(prepared.TrajectoryHashes[hero]))
		}
	}

	legacy := make(map[string]any, len(match)+1)
	for key, value := range match {
		legacy[key] = value
	}
	delete(legacy, "first_step")
	delete(legacy, "recurrent_lineage_schema")
	ticks := append([]uint32(nil), prepared.Steps...)
	parents := make([]any, prepared.TickCount)
	boundaries := make([]any, prepared.TickCount)
	for tick := 0; tick < prepared.TickCount; tick++ {
		parents[tick] = append([]string(nil), prepared.PreviousRecurrentIDs[tick*HeroCount:(tick+1)*HeroCount]...)
		boundaries[tick] = append([]string(nil), prepared.RecurrentBoundaryIDs[tick*HeroCount:(tick+1)*HeroCount]...)
	}
	legacy["ticks"], legacy["recurrent_parent_ids"], legacy["recurrent_boundary_ids"] = ticks, parents, boundaries
	if len(canonicalJSON(match)) >= len(canonicalJSON(legacy)) {
		t.Fatalf("v2 metadata was not reduced: v2=%d legacy=%d", len(canonicalJSON(match)), len(canonicalJSON(legacy)))
	}

	compressed := shardCompressedPayload(first.Bytes(), len(headerBytes))
	if got := sha256Hex(compressed); got != header["payload_sha256"] {
		t.Fatalf("payload hash=%s, header=%v", got, header["payload_sha256"])
	}
	decompressor := flate.NewReader(bytes.NewReader(compressed))
	raw, err := io.ReadAll(decompressor)
	if closeErr := decompressor.Close(); err != nil {
		t.Fatal(err)
	} else if closeErr != nil {
		t.Fatal(closeErr)
	}
	expectedRaw, _, err := encodeArrays(prepared)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw, expectedRaw) {
		t.Fatal("v2 shard payload arrays differ from prepared arrays")
	}
}

func TestShardV2GenerationManifestUsesCanonicalMetadata(t *testing.T) {
	root := filepath.Join(t.TempDir(), "generation")
	if err := WriteGeneration(root, acceptancePrepared(t)); err != nil {
		t.Fatal(err)
	}
	payload, err := os.ReadFile(filepath.Join(root, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var manifest map[string]any
	if err := json.Unmarshal(payload, &manifest); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(payload, canonicalJSON(manifest)) {
		t.Fatal("v2 generation manifest is not canonical JSON")
	}
	if got := manifest["shard_schema_version"]; got != ShardSchemaVersionV2 {
		t.Fatalf("manifest shard schema=%v, want %s", got, ShardSchemaVersionV2)
	}
	match := manifest["matches"].([]any)[0].(map[string]any)
	if _, exists := match["recurrent_parent_ids"]; exists {
		t.Fatal("v2 generation manifest retained recurrent parents")
	}
}

func decodeShardV2Header(t *testing.T, payload []byte) ([]byte, map[string]any) {
	t.Helper()
	if len(payload) < len(ShardMagic)+4 || !bytes.HasPrefix(payload, []byte(ShardMagic)) {
		t.Fatalf("invalid AI42GS2 shard prefix")
	}
	headerLength := int(binary.LittleEndian.Uint32(payload[len(ShardMagic) : len(ShardMagic)+4]))
	headerStart := len(ShardMagic) + 4
	headerEnd := headerStart + headerLength
	if headerLength < 1 || headerEnd > len(payload) {
		t.Fatalf("invalid header length %d", headerLength)
	}
	headerBytes := payload[headerStart:headerEnd]
	var header map[string]any
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		t.Fatalf("decode header: %v", err)
	}
	return headerBytes, header
}

func shardCompressedPayload(payload []byte, headerLength int) []byte {
	return payload[len(ShardMagic)+4+headerLength:]
}
