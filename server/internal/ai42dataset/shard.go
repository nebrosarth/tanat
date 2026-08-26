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
	ShardCodec               = "deflate-raw-3"
	ShardSchemaVersionV2     = "AI42-go-shard-v2"
	RecurrentLineageSchemaV2 = "implicit-match-boundary-v1"
	shardCompressionLevel    = 3
	legacyShardCodec         = "deflate-raw-6"
)

func supportedShardCodec(codec string) bool {
	return codec == ShardCodec || codec == legacyShardCodec
}

// WriteShard writes one exact AI42GS2 shard. The compressed payload is raw
// DEFLATE level 3; the header and every array descriptor are canonical JSON.
func WriteShard(w io.Writer, prepared *Prepared) error {
	if prepared == nil {
		return fmt.Errorf("ai42dataset: nil prepared match")
	}
	// Finalize already performs the exhaustive value/shape validation.  The
	// identity pass below re-hashes every published array and therefore also
	// detects mutation after Finalize without repeating the full scalar scan.
	if err := validatePublishIdentity(prepared); err != nil {
		return err
	}
	encoded, err := encodeShardWithStats(prepared, "train")
	if err != nil {
		return err
	}
	_, err = w.Write(encoded.payload)
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
	// See WriteShard: publication verifies the sealed content identity.  The
	// exhaustive semantic validation belongs to Capture.Finalize and strict
	// readers, not a second hot-path pass over the same tens of megabytes.
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
	encoded, err := writeGenerationShard(shardPath, prepared, split)
	if err != nil {
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
			"sha256":       encoded.sha256,
			"match_ids":    []string{prepared.Metadata.MatchID},
			"row_count":    prepared.TickCount,
			"raw_bytes":    encoded.rawBytes,
			"stored_bytes": encoded.storedBytes,
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
	encoded, err := encodeShardWithStats(prepared, split)
	if err != nil {
		return nil, err
	}
	return encoded.payload, nil
}

type encodedShard struct {
	payload  []byte
	rawBytes int
}

type storedShard struct {
	rawBytes    int
	storedBytes int
	sha256      string
}

func encodeShardWithStats(prepared *Prepared, split string) (encodedShard, error) {
	raw, descriptors, err := encodeArrays(prepared)
	if err != nil {
		return encodedShard{}, err
	}
	var compressed bytes.Buffer
	compressor, err := flate.NewWriter(&compressed, shardCompressionLevel)
	if err != nil {
		return encodedShard{}, fmt.Errorf("ai42dataset: create raw deflate writer: %w", err)
	}
	if _, err := compressor.Write(raw); err != nil {
		return encodedShard{}, fmt.Errorf("ai42dataset: compress shard payload: %w", err)
	}
	if err := compressor.Close(); err != nil {
		return encodedShard{}, fmt.Errorf("ai42dataset: close shard compressor: %w", err)
	}
	header := shardHeader(
		prepared, split, descriptors, len(raw), compressed.Len(),
		sha256Hex(raw), sha256Hex(compressed.Bytes()),
	)
	headerBytes := canonicalJSON(header)
	if len(headerBytes) > 16*1024*1024 {
		return encodedShard{}, fmt.Errorf("ai42dataset: shard header is too large")
	}
	var output bytes.Buffer
	output.Grow(len(ShardMagic) + 4 + len(headerBytes) + compressed.Len())
	output.WriteString(ShardMagic)
	var length [4]byte
	binary.LittleEndian.PutUint32(length[:], uint32(len(headerBytes)))
	output.Write(length[:])
	output.Write(headerBytes)
	output.Write(compressed.Bytes())
	return encodedShard{payload: output.Bytes(), rawBytes: len(raw)}, nil
}

func shardHeader(
	prepared *Prepared,
	split string,
	descriptors []map[string]any,
	rawBytes, compressedBytes int,
	rawHash, payloadHash string,
) map[string]any {
	match := matchEntry(prepared, "shard-000000.a42", split)
	return map[string]any{
		"shard_schema_version":   ShardSchemaVersionV2,
		"protocol_version":       int(ProtocolVersion),
		"schema_hash":            hashHex(prepared.Metadata.SchemaHash),
		"reward_hash":            hashHex(prepared.Metadata.RewardHash),
		"trajectory_schema_hash": hashHex(prepared.Metadata.TrajectorySchemaHash),
		"runtime_manifest_hash":  hashHex(prepared.Metadata.RuntimeManifestHash),
		"codec":                  ShardCodec,
		"raw_bytes":              rawBytes,
		"stored_bytes":           compressedBytes,
		"raw_sha256":             rawHash,
		"payload_sha256":         payloadHash,
		"arrays":                 descriptors,
		"matches":                []any{match},
	}
}

func writeGenerationShard(path string, prepared *Prepared, split string) (storedShard, error) {
	payload, err := os.CreateTemp(filepath.Dir(path), ".ai42-payload-*")
	if err != nil {
		return storedShard{}, fmt.Errorf("ai42dataset: create compressed payload: %w", err)
	}
	payloadName := payload.Name()
	defer os.Remove(payloadName)

	payloadHash := sha256.New()
	payloadCounter := &countingWriter{writer: io.MultiWriter(payload, payloadHash)}
	compressor, err := flate.NewWriter(payloadCounter, shardCompressionLevel)
	if err != nil {
		_ = payload.Close()
		return storedShard{}, fmt.Errorf("ai42dataset: create raw deflate writer: %w", err)
	}
	rawHash := sha256.New()
	rawCounter := &countingWriter{writer: io.MultiWriter(compressor, rawHash)}
	descriptors, err := writeEncodedArrays(rawCounter, prepared)
	if err != nil {
		_ = compressor.Close()
		_ = payload.Close()
		return storedShard{}, err
	}
	if err := compressor.Close(); err != nil {
		_ = payload.Close()
		return storedShard{}, fmt.Errorf("ai42dataset: close shard compressor: %w", err)
	}
	if err := payload.Sync(); err != nil {
		_ = payload.Close()
		return storedShard{}, fmt.Errorf("ai42dataset: sync compressed payload: %w", err)
	}
	if err := payload.Close(); err != nil {
		return storedShard{}, fmt.Errorf("ai42dataset: close compressed payload: %w", err)
	}

	header := shardHeader(
		prepared, split, descriptors, rawCounter.count, payloadCounter.count,
		hex.EncodeToString(rawHash.Sum(nil)), hex.EncodeToString(payloadHash.Sum(nil)),
	)
	headerBytes := canonicalJSON(header)
	if len(headerBytes) > 16*1024*1024 {
		return storedShard{}, fmt.Errorf("ai42dataset: shard header is too large")
	}

	payload, err = os.Open(payloadName)
	if err != nil {
		return storedShard{}, fmt.Errorf("ai42dataset: reopen compressed payload: %w", err)
	}
	defer payload.Close()
	shard, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return storedShard{}, fmt.Errorf("ai42dataset: create shard: %w", err)
	}
	shardHash := sha256.New()
	shardCounter := &countingWriter{writer: io.MultiWriter(shard, shardHash)}
	var length [4]byte
	binary.LittleEndian.PutUint32(length[:], uint32(len(headerBytes)))
	for _, part := range [][]byte{[]byte(ShardMagic), length[:], headerBytes} {
		if _, err := shardCounter.Write(part); err != nil {
			_ = shard.Close()
			return storedShard{}, fmt.Errorf("ai42dataset: write shard header: %w", err)
		}
	}
	if _, err := io.Copy(shardCounter, payload); err != nil {
		_ = shard.Close()
		return storedShard{}, fmt.Errorf("ai42dataset: write shard payload: %w", err)
	}
	if err := shard.Sync(); err != nil {
		_ = shard.Close()
		return storedShard{}, fmt.Errorf("ai42dataset: sync shard: %w", err)
	}
	if err := shard.Close(); err != nil {
		return storedShard{}, fmt.Errorf("ai42dataset: close shard: %w", err)
	}
	return storedShard{
		rawBytes: rawCounter.count, storedBytes: shardCounter.count,
		sha256: hex.EncodeToString(shardHash.Sum(nil)),
	}, nil
}

type countingWriter struct {
	writer io.Writer
	count  int
}

func (w *countingWriter) Write(payload []byte) (int, error) {
	written, err := w.writer.Write(payload)
	w.count += written
	return written, err
}

func writeEncodedArrays(w io.Writer, prepared *Prepared) ([]map[string]any, error) {
	counter, ok := w.(*countingWriter)
	if !ok {
		counter = &countingWriter{writer: w}
	}
	scratch := make([]byte, 64*1024)
	descriptors := make([]map[string]any, 0, len(arrayNames))
	for _, name := range arrayNames {
		start := counter.count
		var err error
		switch name {
		case "hero":
			err = writeF32Blocks(counter, scratch, prepared.Hero)
		case "abilities":
			err = writeF32Blocks(counter, scratch, prepared.Abilities)
		case "entities":
			err = writeF32Blocks(counter, scratch, prepared.Entities)
		case "global":
			err = writeF32Blocks(counter, scratch, prepared.Global)
		case "entity_mask":
			err = writeAll(counter, prepared.EntityMask)
		case "kind_mask":
			err = writeAll(counter, prepared.KindMask)
		case "target_mask":
			err = writeAll(counter, prepared.TargetMask)
		case "skill_target_mask":
			err = writeAll(counter, prepared.SkillTargetMask)
		case "teacher_status":
			err = writeAll(counter, prepared.TeacherStatus)
		case "teacher_action":
			err = writeActionBlocks(counter, scratch, prepared.TeacherAction)
		case "projected_action":
			err = writeActionBlocks(counter, scratch, prepared.ProjectedAction)
		case "executed_action":
			err = writeActionBlocks(counter, scratch, prepared.ExecutedAction)
		case "executed_valid":
			err = writeAll(counter, prepared.ExecutedValid)
		case "rejection_reason":
			err = writeAll(counter, prepared.RejectionReason)
		case "rewards":
			err = writeF32Blocks(counter, scratch, prepared.Rewards)
		case "done":
			err = writeAll(counter, prepared.Done)
		case "winner":
			err = writeI32Blocks(counter, scratch, prepared.Winner)
		case "step":
			err = writeU32Blocks(counter, scratch, prepared.Steps)
		case "elapsed":
			err = writeF32Blocks(counter, scratch, prepared.Elapsed)
		case "invalid":
			err = writeAll(counter, prepared.Invalid)
		default:
			return nil, fmt.Errorf("ai42dataset: unsupported array %q", name)
		}
		if err != nil {
			return nil, fmt.Errorf("ai42dataset: encode array %s: %w", name, err)
		}
		descriptors = append(descriptors, map[string]any{
			"name": name, "dtype": arrayDType(name), "shape": arrayShape(name, prepared.TickCount),
			"offset": start, "nbytes": counter.count - start,
		})
	}
	return descriptors, nil
}

func encodeArrays(prepared *Prepared) ([]byte, []map[string]any, error) {
	total, err := encodedArraysSize(prepared)
	if err != nil {
		return nil, nil, err
	}
	raw := make([]byte, total)
	offset := 0
	descriptors := make([]map[string]any, 0, len(arrayNames))
	for _, name := range arrayNames {
		start := offset
		switch name {
		case "hero":
			offset = putF32Slice(raw, offset, prepared.Hero)
		case "abilities":
			offset = putF32Slice(raw, offset, prepared.Abilities)
		case "entities":
			offset = putF32Slice(raw, offset, prepared.Entities)
		case "global":
			offset = putF32Slice(raw, offset, prepared.Global)
		case "entity_mask":
			offset += copy(raw[offset:], prepared.EntityMask)
		case "kind_mask":
			offset += copy(raw[offset:], prepared.KindMask)
		case "target_mask":
			offset += copy(raw[offset:], prepared.TargetMask)
		case "skill_target_mask":
			offset += copy(raw[offset:], prepared.SkillTargetMask)
		case "teacher_status":
			offset += copy(raw[offset:], prepared.TeacherStatus)
		case "teacher_action":
			offset = putActionSlice(raw, offset, prepared.TeacherAction)
		case "projected_action":
			offset = putActionSlice(raw, offset, prepared.ProjectedAction)
		case "executed_action":
			offset = putActionSlice(raw, offset, prepared.ExecutedAction)
		case "executed_valid":
			offset += copy(raw[offset:], prepared.ExecutedValid)
		case "rejection_reason":
			offset += copy(raw[offset:], prepared.RejectionReason)
		case "rewards":
			offset = putF32Slice(raw, offset, prepared.Rewards)
		case "done":
			offset += copy(raw[offset:], prepared.Done)
		case "winner":
			offset = putI32Slice(raw, offset, prepared.Winner)
		case "step":
			offset = putU32Slice(raw, offset, prepared.Steps)
		case "elapsed":
			offset = putF32Slice(raw, offset, prepared.Elapsed)
		case "invalid":
			offset += copy(raw[offset:], prepared.Invalid)
		default:
			return nil, nil, fmt.Errorf("ai42dataset: unsupported array %q", name)
		}
		shape := arrayShape(name, prepared.TickCount)
		descriptors = append(descriptors, map[string]any{
			"name": name, "dtype": arrayDType(name), "shape": shape,
			"offset": start, "nbytes": offset - start,
		})
	}
	if offset != len(raw) {
		return nil, nil, fmt.Errorf("ai42dataset: encoded array size=%d, expected=%d", offset, len(raw))
	}
	return raw, descriptors, nil
}

func encodedArraysSize(prepared *Prepared) (int, error) {
	maxInt := int(^uint(0) >> 1)
	total := 0
	add := func(count, width int) error {
		if count < 0 || width <= 0 || count > (maxInt-total)/width {
			return fmt.Errorf("ai42dataset: encoded array size overflows native address space")
		}
		total += count * width
		return nil
	}
	for _, item := range []struct{ count, width int }{
		{len(prepared.Hero), 4}, {len(prepared.Abilities), 4}, {len(prepared.Entities), 4},
		{len(prepared.Global), 4}, {len(prepared.EntityMask), 1}, {len(prepared.KindMask), 1},
		{len(prepared.TargetMask), 1}, {len(prepared.SkillTargetMask), 1}, {len(prepared.TeacherStatus), 1},
		{len(prepared.TeacherAction), 5}, {len(prepared.ProjectedAction), 5}, {len(prepared.ExecutedAction), 5},
		{len(prepared.ExecutedValid), 1}, {len(prepared.RejectionReason), 1}, {len(prepared.Rewards), 4},
		{len(prepared.Done), 1}, {len(prepared.Winner), 4}, {len(prepared.Steps), 4},
		{len(prepared.Elapsed), 4}, {len(prepared.Invalid), 1},
	} {
		if err := add(item.count, item.width); err != nil {
			return 0, err
		}
	}
	return total, nil
}

func putF32Slice(dst []byte, offset int, values []float32) int {
	for _, value := range values {
		binary.LittleEndian.PutUint32(dst[offset:offset+4], math.Float32bits(value))
		offset += 4
	}
	return offset
}

func putI32Slice(dst []byte, offset int, values []int32) int {
	for _, value := range values {
		binary.LittleEndian.PutUint32(dst[offset:offset+4], uint32(value))
		offset += 4
	}
	return offset
}

func putU32Slice(dst []byte, offset int, values []uint32) int {
	for _, value := range values {
		binary.LittleEndian.PutUint32(dst[offset:offset+4], value)
		offset += 4
	}
	return offset
}

func putActionSlice(dst []byte, offset int, values []Action) int {
	for _, value := range values {
		value.wireBytes(dst[offset : offset+5])
		offset += 5
	}
	return offset
}

func writeAll(w io.Writer, payload []byte) error {
	for len(payload) > 0 {
		written, err := w.Write(payload)
		if written > 0 {
			payload = payload[written:]
		}
		if err != nil {
			return err
		}
		if written == 0 {
			return io.ErrShortWrite
		}
	}
	return nil
}

func writeF32Blocks(w io.Writer, scratch []byte, values []float32) error {
	const width = 4
	for len(values) > 0 {
		count := min(len(values), len(scratch)/width)
		payload := scratch[:count*width]
		putF32Slice(payload, 0, values[:count])
		if err := writeAll(w, payload); err != nil {
			return err
		}
		values = values[count:]
	}
	return nil
}

func writeI32Blocks(w io.Writer, scratch []byte, values []int32) error {
	const width = 4
	for len(values) > 0 {
		count := min(len(values), len(scratch)/width)
		payload := scratch[:count*width]
		putI32Slice(payload, 0, values[:count])
		if err := writeAll(w, payload); err != nil {
			return err
		}
		values = values[count:]
	}
	return nil
}

func writeU32Blocks(w io.Writer, scratch []byte, values []uint32) error {
	const width = 4
	for len(values) > 0 {
		count := min(len(values), len(scratch)/width)
		payload := scratch[:count*width]
		putU32Slice(payload, 0, values[:count])
		if err := writeAll(w, payload); err != nil {
			return err
		}
		values = values[count:]
	}
	return nil
}

func writeActionBlocks(w io.Writer, scratch []byte, values []Action) error {
	const width = 5
	for len(values) > 0 {
		count := min(len(values), len(scratch)/width)
		payload := scratch[:count*width]
		putActionSlice(payload, 0, values[:count])
		if err := writeAll(w, payload); err != nil {
			return err
		}
		values = values[count:]
	}
	return nil
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
	if err := validatePreparedShape(prepared); err != nil {
		return err
	}
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
