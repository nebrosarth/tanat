package ai42dataset

import (
	"bytes"
	"compress/flate"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
)

const (
	ShardMagic               = "AI42GS2\x00"
	ShardCodec               = "deflate-raw-6"
	ShardSchemaVersionV2     = "AI42-go-shard-v2"
	RecurrentLineageSchemaV2 = "implicit-match-boundary-v1"
)

// WriteShard writes one exact AI42GS2 shard. The compressed payload is raw
// DEFLATE level 6; the header and every array descriptor are canonical JSON.
func WriteShard(w io.Writer, prepared *Prepared) error {
	if prepared == nil {
		return fmt.Errorf("ai42dataset: nil prepared match")
	}
	if err := prepared.Validate(); err != nil {
		return err
	}
	if err := validatePublishIdentity(prepared); err != nil {
		return err
	}
	shard, err := encodeShard(prepared, "train")
	if err != nil {
		return err
	}
	_, err = w.Write(shard)
	return err
}

// WriteGeneration atomically publishes manifest.json and shard-000000.a42.
// It is intentionally one-match for the first native integration slice;
// sharding multiple matches can reuse WriteShard without changing this wire
// contract.
func WriteGeneration(root string, prepared *Prepared) error {
	return WriteGenerationWithSplit(root, prepared, 0, 0)
}

// WriteGenerationWithSplit uses the frozen Python split metadata. A zero
// validation fraction publishes the supplied match as train, which is the
// deterministic one-match default.
func WriteGenerationWithSplit(root string, prepared *Prepared, splitSeed int64, validationFraction float64) error {
	if prepared == nil {
		return fmt.Errorf("ai42dataset: nil prepared match")
	}
	if math.IsNaN(validationFraction) || math.IsInf(validationFraction, 0) || validationFraction < 0 || validationFraction > 1 {
		return fmt.Errorf("ai42dataset: validation_fraction must be between zero and one")
	}
	if root == "" {
		return fmt.Errorf("ai42dataset: generation destination must be non-empty")
	}
	if _, err := os.Lstat(root); err == nil {
		return fmt.Errorf("ai42dataset: generation destination already exists: %s", root)
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("ai42dataset: inspect generation destination: %w", err)
	}
	if err := prepared.Validate(); err != nil {
		return err
	}
	if err := validatePublishIdentity(prepared); err != nil {
		return err
	}
	parent := filepath.Dir(root)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return fmt.Errorf("ai42dataset: create generation parent: %w", err)
	}
	staging, err := os.MkdirTemp(parent, "."+filepath.Base(root)+".staging-*")
	if err != nil {
		return fmt.Errorf("ai42dataset: create staging generation: %w", err)
	}
	defer os.RemoveAll(staging)
	shardName := "shard-000000.a42"
	shardPath := filepath.Join(staging, shardName)
	split := deterministicSplit(prepared.Metadata.MatchID, splitSeed, validationFraction)
	shard, err := encodeShard(prepared, split)
	if err != nil {
		return err
	}
	if err := atomicWrite(shardPath, shard); err != nil {
		return err
	}
	match := matchEntry(prepared, shardName, split)
	manifestUnsigned := map[string]any{
		"dataset_schema_version": DatasetSchemaVersion,
		"shard_schema_version":   ShardSchemaVersionV2,
		"protocol_version":       int(ProtocolVersion),
		"schema_hash":            hashHex(prepared.Metadata.SchemaHash),
		"reward_hash":            hashHex(prepared.Metadata.RewardHash),
		"trajectory_schema_hash": hashHex(prepared.Metadata.TrajectorySchemaHash),
		"runtime_manifest_hash":  hashHex(prepared.Metadata.RuntimeManifestHash),
		"runtime_manifest":       json.RawMessage(prepared.Metadata.RuntimeManifest),
		"split_seed":             splitSeed,
		"validation_fraction":    validationFraction,
		"matches":                []any{match},
		"shards": []any{map[string]any{
			"name":         shardName,
			"sha256":       sha256Hex(shard),
			"match_ids":    []string{prepared.Metadata.MatchID},
			"row_count":    prepared.TickCount,
			"raw_bytes":    rawBytesFromShard(shard),
			"stored_bytes": len(shard),
			"compression":  ShardCodec,
		}},
	}
	manifest := make(map[string]any, len(manifestUnsigned)+1)
	for key, value := range manifestUnsigned {
		manifest[key] = value
	}
	manifest["manifest_hash"] = sha256Hex(canonicalJSON(manifestUnsigned))
	manifestPath := filepath.Join(staging, "manifest.json")
	if err := atomicWrite(manifestPath, canonicalJSON(manifest)); err != nil {
		return err
	}
	if err := os.Rename(staging, root); err != nil {
		return fmt.Errorf("ai42dataset: publish generation: %w", err)
	}
	return nil
}

func encodeShard(prepared *Prepared, split string) ([]byte, error) {
	raw, descriptors, err := encodeArrays(prepared)
	if err != nil {
		return nil, err
	}
	var compressed bytes.Buffer
	compressor, err := flate.NewWriter(&compressed, 6)
	if err != nil {
		return nil, fmt.Errorf("ai42dataset: create raw deflate writer: %w", err)
	}
	if _, err := compressor.Write(raw); err != nil {
		return nil, fmt.Errorf("ai42dataset: compress shard payload: %w", err)
	}
	if err := compressor.Close(); err != nil {
		return nil, fmt.Errorf("ai42dataset: close shard compressor: %w", err)
	}
	match := matchEntry(prepared, "shard-000000.a42", split)
	header := map[string]any{
		"shard_schema_version":   ShardSchemaVersionV2,
		"protocol_version":       int(ProtocolVersion),
		"schema_hash":            hashHex(prepared.Metadata.SchemaHash),
		"reward_hash":            hashHex(prepared.Metadata.RewardHash),
		"trajectory_schema_hash": hashHex(prepared.Metadata.TrajectorySchemaHash),
		"runtime_manifest_hash":  hashHex(prepared.Metadata.RuntimeManifestHash),
		"codec":                  ShardCodec,
		"raw_bytes":              len(raw),
		"stored_bytes":           compressed.Len(),
		"raw_sha256":             sha256Hex(raw),
		"payload_sha256":         sha256Hex(compressed.Bytes()),
		"arrays":                 descriptors,
		"matches":                []any{match},
	}
	headerBytes := canonicalJSON(header)
	if len(headerBytes) > 16*1024*1024 {
		return nil, fmt.Errorf("ai42dataset: shard header is too large")
	}
	var output bytes.Buffer
	output.WriteString(ShardMagic)
	var length [4]byte
	binary.LittleEndian.PutUint32(length[:], uint32(len(headerBytes)))
	output.Write(length[:])
	output.Write(headerBytes)
	output.Write(compressed.Bytes())
	return output.Bytes(), nil
}

func encodeArrays(prepared *Prepared) ([]byte, []map[string]any, error) {
	var raw bytes.Buffer
	descriptors := make([]map[string]any, 0, len(arrayNames))
	for _, name := range arrayNames {
		offset := raw.Len()
		switch name {
		case "hero":
			writeF32Slice(&raw, prepared.Hero)
		case "abilities":
			writeF32Slice(&raw, prepared.Abilities)
		case "entities":
			writeF32Slice(&raw, prepared.Entities)
		case "global":
			writeF32Slice(&raw, prepared.Global)
		case "entity_mask":
			raw.Write(prepared.EntityMask)
		case "kind_mask":
			raw.Write(prepared.KindMask)
		case "target_mask":
			raw.Write(prepared.TargetMask)
		case "skill_target_mask":
			raw.Write(prepared.SkillTargetMask)
		case "teacher_status":
			raw.Write(prepared.TeacherStatus)
		case "teacher_action":
			writeActionSlice(&raw, prepared.TeacherAction)
		case "projected_action":
			writeActionSlice(&raw, prepared.ProjectedAction)
		case "executed_action":
			writeActionSlice(&raw, prepared.ExecutedAction)
		case "executed_valid":
			raw.Write(prepared.ExecutedValid)
		case "rejection_reason":
			raw.Write(prepared.RejectionReason)
		case "rewards":
			writeF32Slice(&raw, prepared.Rewards)
		case "done":
			raw.Write(prepared.Done)
		case "winner":
			writeI32Slice(&raw, prepared.Winner)
		case "step":
			writeU32Slice(&raw, prepared.Steps)
		case "elapsed":
			writeF32Slice(&raw, prepared.Elapsed)
		case "invalid":
			raw.Write(prepared.Invalid)
		default:
			return nil, nil, fmt.Errorf("ai42dataset: unsupported array %q", name)
		}
		shape := arrayShape(name, prepared.TickCount)
		descriptors = append(descriptors, map[string]any{
			"name": name, "dtype": arrayDType(name), "shape": shape,
			"offset": offset, "nbytes": raw.Len() - offset,
		})
	}
	return raw.Bytes(), descriptors, nil
}

func arrayDType(name string) string {
	switch name {
	case "hero", "abilities", "entities", "global", "rewards", "elapsed":
		return "<f4"
	case "winner":
		return "<i4"
	case "step":
		return "<u4"
	case "teacher_action", "projected_action", "executed_action":
		return "action"
	default:
		return "u1"
	}
}

func arrayShape(name string, rows int) []int {
	switch name {
	case "hero":
		return []int{rows, HeroCount, HeroFeatures}
	case "abilities":
		return []int{rows, HeroCount, AbilityCount, AbilityFeatures}
	case "entities":
		return []int{rows, HeroCount, MaxEntities, EntityFeatures}
	case "global":
		return []int{rows, HeroCount, GlobalFeatures}
	case "entity_mask", "target_mask":
		return []int{rows, HeroCount, MaxEntities}
	case "kind_mask":
		return []int{rows, HeroCount, ActionKinds}
	case "skill_target_mask":
		return []int{rows, HeroCount, AbilityCount, MaxEntities}
	case "teacher_status", "teacher_action", "projected_action", "executed_action", "executed_valid", "rejection_reason", "rewards", "invalid":
		return []int{rows, HeroCount}
	default:
		return []int{rows}
	}
}

func writeF32Slice(w io.Writer, values []float32) {
	for _, value := range values {
		writeF32(w, value)
	}
}
func writeI32Slice(w io.Writer, values []int32) {
	for _, value := range values {
		writeI32(w, value)
	}
}
func writeU32Slice(w io.Writer, values []uint32) {
	for _, value := range values {
		writeU32(w, value)
	}
}
func writeActionSlice(w io.Writer, values []Action) {
	for _, value := range values {
		writeActionHash(w, value)
	}
}

func matchEntry(prepared *Prepared, shardName, split string) map[string]any {
	heroes := make([]string, HeroCount)
	controllers := make([]int, HeroCount)
	roster := make([]int32, HeroCount)
	sides := make([]int, HeroCount)
	copy(heroes, prepared.Metadata.HeroIDs[:])
	for slot := 0; slot < HeroCount; slot++ {
		controllers[slot] = int(prepared.Metadata.ControllerBySlot[slot])
		sides[slot] = int(prepared.Metadata.SideBySlot[slot])
	}
	copy(roster, prepared.Metadata.RosterIDs[:])
	trajectoryIDs := make([]string, HeroCount)
	trajectoryHashes := make([]string, HeroCount)
	for hero := 0; hero < HeroCount; hero++ {
		trajectoryIDs[hero] = prepared.TrajectoryIDs[hero]
		trajectoryHashes[hero] = hashHex(prepared.TrajectoryHashes[hero])
	}
	return map[string]any{
		"match_id": prepared.Metadata.MatchID, "split": split, "shard": shardName,
		"row_offset": 0, "tick_count": prepared.TickCount, "first_step": prepared.Steps[0],
		"recurrent_lineage_schema": RecurrentLineageSchemaV2,
		"hero_ids":                 heroes, "trajectory_ids": trajectoryIDs, "trajectory_hashes": trajectoryHashes,
		"seed": prepared.Metadata.Seed, "scenario": prepared.Metadata.Scenario,
		"controller_by_slot": controllers, "roster_ids": roster, "side_by_slot": sides,
	}
}

func deterministicSplit(matchID string, seed int64, fraction float64) string {
	_ = matchID
	_ = seed
	if fraction >= 1 {
		return "validation"
	}
	// With exactly one match the frozen Python rule computes zero validation
	// rows, regardless of a non-zero fraction.
	return "train"
}

func validatePublishIdentity(prepared *Prepared) error {
	view := Capture{data: *prepared}
	for hero := 0; hero < HeroCount; hero++ {
		expectedID := prepared.Metadata.MatchID + ":hero:" + prepared.Metadata.HeroIDs[hero]
		if prepared.TrajectoryIDs[hero] != expectedID {
			return fieldError("trajectory_ids", -1, hero, "is not bound to match metadata")
		}
		expectedHash := view.trajectoryHash(hero)
		if prepared.TrajectoryHashes[hero] != expectedHash {
			return fieldError("trajectory_hashes", -1, hero, "does not match prepared trajectory data and metadata")
		}
	}
	if prepared.MatchHash != matchHash(prepared) {
		return metadataError("match_hash", "does not match prepared trajectory provenance")
	}
	return nil
}

func hashHex(value Hash) string { return hex.EncodeToString(value[:]) }
func sha256Hex(value []byte) string {
	digest := sha256.Sum256(value)
	return hex.EncodeToString(digest[:])
}

func rawBytesFromShard(shard []byte) int {
	if len(shard) < 12 {
		return 0
	}
	headerLength := int(binary.LittleEndian.Uint32(shard[8:12]))
	if headerLength < 0 || 12+headerLength > len(shard) {
		return 0
	}
	compressed := shard[12+headerLength:]
	decompressor := flate.NewReader(bytes.NewReader(compressed))
	defer decompressor.Close()
	raw, err := io.ReadAll(decompressor)
	if err != nil {
		return 0
	}
	return len(raw)
}

func atomicWrite(path string, payload []byte) error {
	directory := filepath.Dir(path)
	temporary, err := os.CreateTemp(directory, ".ai42-atomic-*")
	if err != nil {
		return fmt.Errorf("ai42dataset: create temporary file: %w", err)
	}
	temporaryName := temporary.Name()
	defer os.Remove(temporaryName)
	if _, err := temporary.Write(payload); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("ai42dataset: write %s: %w", path, err)
	}
	if err := temporary.Sync(); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("ai42dataset: sync %s: %w", path, err)
	}
	if err := temporary.Close(); err != nil {
		return fmt.Errorf("ai42dataset: close %s: %w", path, err)
	}
	if err := os.Rename(temporaryName, path); err != nil {
		return fmt.Errorf("ai42dataset: publish %s: %w", path, err)
	}
	return nil
}
