package ai42dataset

// This file contains the bounded v2 generation reader.  It intentionally
// keeps the writer's compact wire format: opening a generation verifies its
// control plane and, by default, file hashes, while Verify streams bounded
// shards through temporary raw-payload files and exposes one row at a time to
// the caller. No complete raw payload or array set is retained in memory.

import (
	"bufio"
	"bytes"
	"compress/flate"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"unicode/utf8"

	"tanatserver/internal/battleserver"
)

const (
	// Reader limits are deliberately above the production generation (320
	// matches, one <=4500-row shard per match) while bounding every attacker-
	// controlled allocation, file walk, and inflation target.
	maxManifestBytes         int64 = 16 * 1024 * 1024
	maxShardHeaderBytes            = 16 * 1024 * 1024
	maxGenerationMatches           = 4096
	maxGenerationShards            = 4096
	maxMatchesPerShard             = 256
	maxMatchRows                   = 4500
	maxShardRows                   = maxMatchRows
	maxGenerationRows        int64 = 16 * 1024 * 1024
	maxShardRawBytes         int64 = 512 * 1024 * 1024
	maxShardStoredBytes      int64 = 512 * 1024 * 1024
	maxGenerationRawBytes    int64 = 1 << 40 // 1 TiB.
	maxGenerationStoredBytes int64 = 1 << 40 // 1 TiB.
	maxArrayDimensions             = 4
	maxVerificationWorkers         = 16
)

var (
	manifestFields = map[string]struct{}{
		"dataset_schema_version": {}, "shard_schema_version": {}, "protocol_version": {},
		"schema_hash": {}, "reward_hash": {}, "trajectory_schema_hash": {},
		"runtime_manifest_hash": {}, "runtime_manifest": {}, "split_seed": {},
		"validation_fraction": {}, "matches": {}, "shards": {}, "manifest_hash": {},
	}
	shardManifestFields = map[string]struct{}{
		"name": {}, "sha256": {}, "match_ids": {}, "row_count": {}, "raw_bytes": {},
		"stored_bytes": {}, "compression": {},
	}
	matchV2Fields = map[string]struct{}{
		"match_id": {}, "split": {}, "shard": {}, "row_offset": {}, "tick_count": {},
		"first_step": {}, "recurrent_lineage_schema": {}, "hero_ids": {},
		"trajectory_ids": {}, "trajectory_hashes": {}, "seed": {}, "scenario": {},
		"controller_by_slot": {}, "roster_ids": {}, "side_by_slot": {},
	}
	headerFields = map[string]struct{}{
		"shard_schema_version": {}, "protocol_version": {}, "schema_hash": {},
		"reward_hash": {}, "trajectory_schema_hash": {}, "runtime_manifest_hash": {},
		"codec": {}, "raw_bytes": {}, "stored_bytes": {}, "raw_sha256": {},
		"payload_sha256": {}, "arrays": {}, "matches": {},
	}
	arrayDescriptorFields = map[string]struct{}{
		"name": {}, "dtype": {}, "shape": {}, "offset": {}, "nbytes": {},
	}
)

// OpenOptions controls generation opening.  An empty expected hash disables
// the caller-supplied pin, while the manifest's own self-hash is always
// checked.
type OpenOptions struct {
	ExpectedManifestHash string
	// DeferShardHashing skips whole-file shard SHA-256 at open. Shard headers,
	// paths, sizes, schemas, dimensions, and hard caps are still validated.
	// Any authoritative Verify method always performs the whole-file hash.
	DeferShardHashing bool
}

// Generation is an immutable, verified v2 generation handle.  It retains
// only validated manifest metadata; shard payloads are opened during Verify.
type Generation struct {
	root              string
	manifest          map[string]any
	manifestHash      string
	manifestPath      string
	manifestInfo      os.FileInfo
	manifestBytes     []byte
	manifestBytesHash string
	matches           []*matchRecord
	shards            []*shardRecord
}

// MatchMetadata is the compact v2 match metadata passed to OnMatch and exposed
// by Generation metadata accessors. Every returned value owns copies of its
// slices, so callers cannot mutate the opened generation.
type MatchMetadata struct {
	MatchID                string
	Split                  string
	Shard                  string
	RowOffset              int
	TickCount              int
	FirstStep              uint32
	RecurrentLineageSchema string
	HeroIDs                []string
	TrajectoryIDs          []string
	TrajectoryHashes       []Hash
	Seed                   int64
	Scenario               string
	ControllerBySlot       []uint8
	RosterIDs              []int32
	SideBySlot             []uint8
}

// Row is one decoded row.  Its slices are reused for the next callback and
// are valid only during the synchronous OnRow call.  This makes callback use
// bounded even for very large shards; callers that need to retain data must
// copy it.
type Row struct {
	MatchID string
	Tick    int
	Step    uint32
	Elapsed float32
	Done    uint8
	Winner  int32

	Invalid         []uint8
	Rewards         []float32
	Hero            []float32
	Abilities       []float32
	Entities        []float32
	Global          []float32
	EntityMask      []uint8
	KindMask        []uint8
	TargetMask      []uint8
	SkillTargetMask []uint8

	TeacherStatus   []uint8
	TeacherAction   []Action
	ProjectedAction []Action
	ExecutedAction  []Action
	ExecutedValid   []uint8
	RejectionReason []uint8
}

// VerifyOptions supplies bounded callbacks.  OnMatch is called after a match
// has passed all row and trajectory checks.  OnRow is called in shard/row
// order after the row's durable semantic checks have passed.
type VerifyOptions struct {
	OnMatch func(MatchMetadata) error
	OnRow   func(Row) error
	// Workers controls independent shard verification. Zero selects one.
	// Values outside [1,16] after defaulting are rejected. Callbacks may run
	// concurrently when Workers is greater than one and must be concurrency-safe.
	Workers int
}

// VerificationFileEvidence records the immutable file and payload evidence
// established by one successful authoritative shard verification.
type VerificationFileEvidence struct {
	Shard           string
	StoredBytes     int64
	SHA256          string
	CompressedBytes int64
	PayloadSHA256   string
	RawBytes        int64
	RawSHA256       string
}

// VerificationReport summarizes a complete Verify pass.
type VerificationReport struct {
	ManifestHash string
	Shards       int
	Matches      int
	Rows         int
	Files        []VerificationFileEvidence
}

type matchRecord struct {
	raw               map[string]any
	metadata          MatchMetadata
	trajectoryHashHex []string
	rowEnd            int
	shard             *shardRecord
}

type matchVerification struct {
	match            *matchRecord
	lastStatus       [HeroCount]uint8
	lineageRoots     [HeroCount]map[string]string
	lineageCancelled [HeroCount]map[string]struct{}
	columnHashes     [HeroCount][]hash.Hash
}

type shardRecord struct {
	raw         map[string]any
	name        string
	sha256      string
	rowCount    int
	rawBytes    int64
	storedBytes int64
	matchIDs    []string
	matches     []*matchRecord
}

type arrayDescriptor struct {
	name     string
	dtype    string
	shape    []int
	offset   int64
	nbytes   int64
	rowBytes int64
}

type shardPreamble struct {
	prefix          []byte
	headerBytes     []byte
	header          map[string]any
	descriptors     []arrayDescriptor
	compressedBytes int64
}

type rowData struct {
	MatchID string
	Tick    int
	Step    uint32
	Elapsed float32
	Done    uint8
	Winner  int32

	Invalid         []uint8
	Rewards         []float32
	Hero            []float32
	Abilities       []float32
	Entities        []float32
	Global          []float32
	EntityMask      []uint8
	KindMask        []uint8
	TargetMask      []uint8
	SkillTargetMask []uint8
	TeacherStatus   []uint8
	TeacherAction   []Action
	ProjectedAction []Action
	ExecutedAction  []Action
	ExecutedValid   []uint8
	RejectionReason []uint8
}

// OpenGeneration verifies manifest canonicality, the expected manifest hash,
// all manifest references, path containment, and every shard file hash.  It
// does not decompress shard payloads; call Verify for bounded replay checks.
func OpenGeneration(root, expectedManifestHash string) (*Generation, error) {
	return OpenGenerationWithOptions(root, OpenOptions{ExpectedManifestHash: expectedManifestHash})
}

// OpenGenerationWithOptions is the options form of OpenGeneration.
func OpenGenerationWithOptions(root string, options OpenOptions) (*Generation, error) {
	resolvedRoot, err := resolveGenerationRoot(root)
	if err != nil {
		return nil, err
	}
	manifestPath, err := containedRegularFile(resolvedRoot, "manifest.json")
	if err != nil {
		return nil, err
	}
	payload, manifestInfo, err := readBoundedRegularFile(manifestPath, maxManifestBytes)
	if err != nil {
		return nil, fmt.Errorf("ai42dataset: read manifest: %w", err)
	}
	value, err := decodeCanonicalJSON(payload, manifestPath)
	if err != nil {
		return nil, err
	}
	manifest, ok := value.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("ai42dataset: manifest root must be an object")
	}
	if err := requireFields(manifest, manifestFields, "manifest"); err != nil {
		return nil, err
	}
	manifestHash, err := hashField(manifest["manifest_hash"], "manifest.manifest_hash")
	if err != nil {
		return nil, err
	}
	unsigned := make(map[string]any, len(manifest)-1)
	for key, item := range manifest {
		if key != "manifest_hash" {
			unsigned[key] = item
		}
	}
	if got := sha256Hex(canonicalJSON(unsigned)); got != manifestHash {
		return nil, fmt.Errorf("ai42dataset: manifest hash mismatch: got %s, want %s", got, manifestHash)
	}
	if options.ExpectedManifestHash != "" {
		expected, err := normalizeHash(options.ExpectedManifestHash, "expected_manifest_hash")
		if err != nil {
			return nil, err
		}
		if expected != manifestHash {
			return nil, fmt.Errorf("ai42dataset: expected manifest hash %s, got %s", expected, manifestHash)
		}
	}

	generation, err := validateManifest(resolvedRoot, manifest, manifestHash)
	if err != nil {
		return nil, err
	}
	generation.manifestPath = manifestPath
	generation.manifestInfo = manifestInfo
	generation.manifestBytes = append([]byte(nil), payload...)
	generation.manifestBytesHash = sha256Hex(payload)
	for _, shard := range generation.shards {
		if err := generation.inspectShardHeader(shard); err != nil {
			return nil, err
		}
		if !options.DeferShardHashing {
			path, err := containedRegularFile(resolvedRoot, shard.name)
			if err != nil {
				return nil, err
			}
			size, digest, err := digestFile(path, maxShardStoredBytes)
			if err != nil {
				return nil, fmt.Errorf("ai42dataset: shard %s: %w", shard.name, err)
			}
			if size != shard.storedBytes || digest != shard.sha256 {
				return nil, fmt.Errorf("ai42dataset: shard %s file hash/size mismatch", shard.name)
			}
		}
	}
	return generation, nil
}

// OpenGenerationWithExpectedManifestHash is an explicit-name convenience for
// callers that prefer not to use the two-argument OpenGeneration form.
func OpenGenerationWithExpectedManifestHash(root, expectedManifestHash string) (*Generation, error) {
	return OpenGeneration(root, expectedManifestHash)
}

// ManifestHash returns the verified manifest SHA-256 in lowercase hex.
func (g *Generation) ManifestHash() string {
	if g == nil {
		return ""
	}
	return g.manifestHash
}

// MatchIDs returns validated match IDs in canonical manifest order.
func (g *Generation) MatchIDs() []string {
	if g == nil {
		return nil
	}
	result := make([]string, len(g.matches))
	for index, match := range g.matches {
		result[index] = match.metadata.MatchID
	}
	return result
}

// Matches returns immutable-by-copy validated metadata in canonical manifest
// order. Mutating a returned value cannot change the opened generation.
func (g *Generation) Matches() []MatchMetadata {
	if g == nil {
		return nil
	}
	result := make([]MatchMetadata, len(g.matches))
	for index, match := range g.matches {
		result[index] = copyMatchMetadata(match.metadata)
	}
	return result
}

// MatchMetadata returns one immutable-by-copy validated metadata snapshot.
func (g *Generation) MatchMetadata(matchID string) (MatchMetadata, bool) {
	if g == nil {
		return MatchMetadata{}, false
	}
	index := sort.Search(len(g.matches), func(index int) bool {
		return g.matches[index].metadata.MatchID >= matchID
	})
	if index == len(g.matches) || g.matches[index].metadata.MatchID != matchID {
		return MatchMetadata{}, false
	}
	return copyMatchMetadata(g.matches[index].metadata), true
}

// VerifiedSplitMatchIDs returns defensive train/validation ID slices in
// canonical manifest order. Opening established the deterministic split; the
// manifest identity is checked again before the verified state is exposed.
func (g *Generation) VerifiedSplitMatchIDs() (map[string][]string, error) {
	if g == nil || g.manifest == nil || g.manifestInfo == nil || len(g.matches) == 0 {
		return nil, fmt.Errorf("ai42dataset: verified generation state is unavailable")
	}
	if err := g.revalidateManifest(); err != nil {
		return nil, err
	}
	result := map[string][]string{
		"train":      make([]string, 0, len(g.matches)),
		"validation": make([]string, 0, len(g.matches)),
	}
	for _, match := range g.matches {
		if match == nil || match.metadata.MatchID == "" {
			return nil, fmt.Errorf("ai42dataset: verified generation state is unavailable")
		}
		ids, ok := result[match.metadata.Split]
		if !ok {
			return nil, fmt.Errorf("ai42dataset: verified generation has invalid split %q", match.metadata.Split)
		}
		result[match.metadata.Split] = append(ids, match.metadata.MatchID)
	}
	return result, nil
}

type targetRowState struct {
	matchID        string
	tickCount      int
	ranges         [][2]int
	rangeIndex     int
	nextTick       int
	lastTick       int
	sawRow         bool
	matchCallbacks int
}

// ReadTargetRows authoritatively verifies every shard containing a requested
// match, but forwards only rows in the requested sorted, disjoint half-open
// tick ranges. Each requested row is delivered exactly once.
func (g *Generation) ReadTargetRows(ctx context.Context, targets map[string][][2]int, callback func(Row) error) (int, error) {
	if g == nil || g.manifest == nil || len(g.matches) == 0 {
		return 0, fmt.Errorf("ai42dataset: verified generation state is unavailable")
	}
	if callback == nil {
		return 0, fmt.Errorf("ai42dataset: target row callback must be non-nil")
	}
	if len(targets) == 0 {
		return 0, fmt.Errorf("ai42dataset: target row ranges must be non-empty")
	}
	if len(targets) > maxGenerationMatches {
		return 0, fmt.Errorf("ai42dataset: target match count %d exceeds %d", len(targets), maxGenerationMatches)
	}

	requestedIDs := make([]string, 0, len(targets))
	for matchID := range targets {
		requestedIDs = append(requestedIDs, matchID)
	}
	sort.Strings(requestedIDs)
	states := make(map[string]*targetRowState, len(requestedIDs))
	expectedRows := 0
	for _, matchID := range requestedIDs {
		metadata, ok := g.MatchMetadata(matchID)
		if !ok {
			return 0, fmt.Errorf("ai42dataset: unknown target match %q", matchID)
		}
		ranges := targets[matchID]
		if len(ranges) == 0 {
			return 0, fmt.Errorf("ai42dataset: target match %s has no ranges", matchID)
		}
		if len(ranges) > metadata.TickCount {
			return 0, fmt.Errorf("ai42dataset: target match %s range count %d exceeds tick count %d", matchID, len(ranges), metadata.TickCount)
		}
		previousEnd := -1
		for index, interval := range ranges {
			start, end := interval[0], interval[1]
			if start < 0 || end < 0 || start >= end || end > metadata.TickCount {
				return 0, fmt.Errorf("ai42dataset: target match %s range[%d] [%d,%d) is outside [0,%d)", matchID, index, start, end, metadata.TickCount)
			}
			if index > 0 && start < previousEnd {
				return 0, fmt.Errorf("ai42dataset: target match %s ranges must be sorted and disjoint", matchID)
			}
			width := end - start
			if expectedRows > math.MaxInt-width {
				return 0, fmt.Errorf("ai42dataset: target row count overflows int")
			}
			expectedRows += width
			previousEnd = end
		}
		copied := append([][2]int(nil), ranges...)
		states[matchID] = &targetRowState{
			matchID: matchID, tickCount: metadata.TickCount, ranges: copied,
			nextTick: copied[0][0], lastTick: -1,
		}
	}

	canonicalIDs := make([]string, 0, len(states))
	for _, match := range g.matches {
		if _, selected := states[match.metadata.MatchID]; selected {
			canonicalIDs = append(canonicalIDs, match.metadata.MatchID)
		}
	}
	forwarded := 0
	_, err := g.VerifyMatchIDs(ctx, canonicalIDs, VerifyOptions{
		OnRow: func(row Row) error {
			state, selected := states[row.MatchID]
			if !selected {
				return nil
			}
			shouldForward, err := state.consume(row.Tick)
			if err != nil {
				return err
			}
			if !shouldForward {
				return nil
			}
			if err := callback(row); err != nil {
				return err
			}
			forwarded++
			return nil
		},
		OnMatch: func(metadata MatchMetadata) error {
			state, selected := states[metadata.MatchID]
			if !selected {
				return nil
			}
			if state.matchCallbacks != 0 {
				return fmt.Errorf("ai42dataset: duplicate verified match callback for %s", metadata.MatchID)
			}
			if err := state.finish(); err != nil {
				return err
			}
			state.matchCallbacks++
			return nil
		},
	})
	if err != nil {
		return forwarded, err
	}
	if err := g.revalidateManifest(); err != nil {
		return forwarded, err
	}
	for _, matchID := range canonicalIDs {
		state := states[matchID]
		if state.matchCallbacks != 1 {
			return forwarded, fmt.Errorf("ai42dataset: missing verified match callback for %s", matchID)
		}
		if err := state.finish(); err != nil {
			return forwarded, err
		}
	}
	if forwarded != expectedRows {
		return forwarded, fmt.Errorf("ai42dataset: delivered %d target rows, want %d", forwarded, expectedRows)
	}
	return forwarded, nil
}

func (s *targetRowState) consume(tick int) (bool, error) {
	if tick < 0 || tick >= s.tickCount {
		return false, fmt.Errorf("ai42dataset: target match %s callback tick %d is outside [0,%d)", s.matchID, tick, s.tickCount)
	}
	if s.sawRow && tick <= s.lastTick {
		return false, fmt.Errorf("ai42dataset: duplicate or out-of-order row callback for %s tick %d", s.matchID, tick)
	}
	s.sawRow = true
	s.lastTick = tick
	for s.rangeIndex < len(s.ranges) && tick >= s.ranges[s.rangeIndex][1] {
		if s.nextTick != s.ranges[s.rangeIndex][1] {
			return false, fmt.Errorf("ai42dataset: missing row callback for %s tick %d", s.matchID, s.nextTick)
		}
		s.rangeIndex++
		if s.rangeIndex < len(s.ranges) {
			s.nextTick = s.ranges[s.rangeIndex][0]
		}
	}
	if s.rangeIndex == len(s.ranges) || tick < s.ranges[s.rangeIndex][0] {
		return false, nil
	}
	if tick != s.nextTick {
		return false, fmt.Errorf("ai42dataset: missing or duplicate row callback for %s: got tick %d, want %d", s.matchID, tick, s.nextTick)
	}
	s.nextTick++
	return true, nil
}

func (s *targetRowState) finish() error {
	for s.rangeIndex < len(s.ranges) {
		if s.nextTick != s.ranges[s.rangeIndex][1] {
			return fmt.Errorf("ai42dataset: missing row callback for %s tick %d", s.matchID, s.nextTick)
		}
		s.rangeIndex++
		if s.rangeIndex < len(s.ranges) {
			s.nextTick = s.ranges[s.rangeIndex][0]
		}
	}
	return nil
}

// Verify performs a complete bounded replay/verification pass.
func (g *Generation) Verify(ctx context.Context, options VerifyOptions) (VerificationReport, error) {
	if g == nil {
		return VerificationReport{}, fmt.Errorf("ai42dataset: nil generation")
	}
	return g.verifyShards(ctx, g.shards, options)
}

// VerifyShards authoritatively verifies only the named shards. Names must be
// unique and are processed in canonical manifest order.
func (g *Generation) VerifyShards(ctx context.Context, shardNames []string, options VerifyOptions) (VerificationReport, error) {
	if g == nil {
		return VerificationReport{}, fmt.Errorf("ai42dataset: nil generation")
	}
	if len(shardNames) == 0 {
		return VerificationReport{ManifestHash: g.manifestHash}, fmt.Errorf("ai42dataset: no shards selected")
	}
	wanted := make(map[string]struct{}, len(shardNames))
	for _, name := range shardNames {
		if name == "" {
			return VerificationReport{ManifestHash: g.manifestHash}, fmt.Errorf("ai42dataset: selected shard name must be non-empty")
		}
		if _, duplicate := wanted[name]; duplicate {
			return VerificationReport{ManifestHash: g.manifestHash}, fmt.Errorf("ai42dataset: duplicate selected shard %q", name)
		}
		wanted[name] = struct{}{}
	}
	selected := make([]*shardRecord, 0, len(wanted))
	for _, shard := range g.shards {
		if _, ok := wanted[shard.name]; ok {
			selected = append(selected, shard)
			delete(wanted, shard.name)
		}
	}
	if len(wanted) != 0 {
		for _, name := range shardNames {
			if _, unknown := wanted[name]; unknown {
				return VerificationReport{ManifestHash: g.manifestHash}, fmt.Errorf("ai42dataset: unknown selected shard %q", name)
			}
		}
	}
	return g.verifyShards(ctx, selected, options)
}

// VerifyMatchIDs authoritatively verifies every complete shard referenced by
// the selected matches. Match IDs must be unique. Callbacks may therefore see
// other matches co-located in a selected shard.
func (g *Generation) VerifyMatchIDs(ctx context.Context, matchIDs []string, options VerifyOptions) (VerificationReport, error) {
	if g == nil {
		return VerificationReport{}, fmt.Errorf("ai42dataset: nil generation")
	}
	if len(matchIDs) == 0 {
		return VerificationReport{ManifestHash: g.manifestHash}, fmt.Errorf("ai42dataset: no matches selected")
	}
	wanted := make(map[string]struct{}, len(matchIDs))
	for _, id := range matchIDs {
		if id == "" {
			return VerificationReport{ManifestHash: g.manifestHash}, fmt.Errorf("ai42dataset: selected match ID must be non-empty")
		}
		if _, duplicate := wanted[id]; duplicate {
			return VerificationReport{ManifestHash: g.manifestHash}, fmt.Errorf("ai42dataset: duplicate selected match %q", id)
		}
		wanted[id] = struct{}{}
	}
	selectedNames := make(map[string]struct{}, len(wanted))
	for _, match := range g.matches {
		if _, ok := wanted[match.metadata.MatchID]; ok {
			selectedNames[match.shard.name] = struct{}{}
			delete(wanted, match.metadata.MatchID)
		}
	}
	if len(wanted) != 0 {
		for _, id := range matchIDs {
			if _, unknown := wanted[id]; unknown {
				return VerificationReport{ManifestHash: g.manifestHash}, fmt.Errorf("ai42dataset: unknown selected match %q", id)
			}
		}
	}
	selected := make([]*shardRecord, 0, len(selectedNames))
	for _, shard := range g.shards {
		if _, ok := selectedNames[shard.name]; ok {
			selected = append(selected, shard)
		}
	}
	return g.verifyShards(ctx, selected, options)
}

type shardVerificationResult struct {
	report VerificationReport
	err    error
}

func (g *Generation) verifyShards(ctx context.Context, shards []*shardRecord, options VerifyOptions) (VerificationReport, error) {
	report := VerificationReport{ManifestHash: g.manifestHash}
	workers := options.Workers
	if workers == 0 {
		workers = 1
	}
	if workers < 1 || workers > maxVerificationWorkers {
		return report, fmt.Errorf("ai42dataset: verification workers %d is outside [1,%d]", options.Workers, maxVerificationWorkers)
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := g.revalidateManifest(); err != nil {
		return report, err
	}
	if err := ctx.Err(); err != nil {
		return report, err
	}
	if len(shards) == 0 {
		return report, nil
	}
	if workers > len(shards) {
		workers = len(shards)
	}
	verifyCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	jobs := make(chan int)
	results := make([]shardVerificationResult, len(shards))
	var wait sync.WaitGroup
	wait.Add(workers)
	for worker := 0; worker < workers; worker++ {
		go func() {
			defer wait.Done()
			for index := range jobs {
				local := VerificationReport{ManifestHash: g.manifestHash}
				err := g.verifyShard(verifyCtx, shards[index], options, &local)
				if err == nil {
					local.Shards = 1
				}
				results[index] = shardVerificationResult{report: local, err: err}
				if err != nil {
					cancel()
				}
			}
		}()
	}
	func() {
		defer close(jobs)
		for index := range shards {
			select {
			case jobs <- index:
			case <-verifyCtx.Done():
				return
			}
		}
	}()
	wait.Wait()

	var firstConcrete error
	for _, result := range results {
		if result.err != nil && !errors.Is(result.err, context.Canceled) && !errors.Is(result.err, context.DeadlineExceeded) && firstConcrete == nil {
			firstConcrete = result.err
		}
		if result.err == nil {
			report.Shards += result.report.Shards
			report.Matches += result.report.Matches
			report.Rows += result.report.Rows
			report.Files = append(report.Files, result.report.Files...)
		}
	}
	if firstConcrete != nil {
		return report, firstConcrete
	}
	if err := ctx.Err(); err != nil {
		return report, err
	}
	for _, result := range results {
		if result.err != nil {
			return report, result.err
		}
	}
	return report, nil
}

func (g *Generation) revalidateManifest() error {
	path, err := containedRegularFile(g.root, "manifest.json")
	if err != nil {
		return fmt.Errorf("ai42dataset: manifest changed after open: %w", err)
	}
	payload, info, err := readBoundedRegularFile(path, maxManifestBytes)
	if err != nil {
		return fmt.Errorf("ai42dataset: manifest changed after open: %w", err)
	}
	if g.manifestInfo == nil || !os.SameFile(g.manifestInfo, info) {
		return fmt.Errorf("ai42dataset: manifest changed after open: file identity mismatch")
	}
	if path != g.manifestPath || int64(len(payload)) != g.manifestInfo.Size() ||
		sha256Hex(payload) != g.manifestBytesHash || !bytes.Equal(payload, g.manifestBytes) {
		return fmt.Errorf("ai42dataset: manifest changed after open: byte/hash mismatch")
	}
	return nil
}

// Replay is a semantic alias for Verify at integration call sites.
func (g *Generation) Replay(ctx context.Context, options VerifyOptions) (VerificationReport, error) {
	return g.Verify(ctx, options)
}

func resolveGenerationRoot(root string) (string, error) {
	if root == "" {
		return "", fmt.Errorf("ai42dataset: generation root must be non-empty")
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return "", fmt.Errorf("ai42dataset: resolve generation root: %w", err)
	}
	resolved, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return "", fmt.Errorf("ai42dataset: resolve generation root: %w", err)
	}
	info, err := os.Stat(resolved)
	if err != nil {
		return "", fmt.Errorf("ai42dataset: stat generation root: %w", err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("ai42dataset: generation root is not a directory")
	}
	return filepath.Clean(resolved), nil
}

func containedRegularFile(root, name string) (string, error) {
	if name == "" || filepath.IsAbs(name) || filepath.VolumeName(name) != "" ||
		strings.ContainsAny(name, `/\\`) || name == "." || name == ".." {
		return "", fmt.Errorf("ai42dataset: path %q is not a contained file name", name)
	}
	joined := filepath.Join(root, name)
	rel, err := filepath.Rel(root, joined)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
		return "", fmt.Errorf("ai42dataset: path %q escapes generation root", name)
	}
	resolved, err := filepath.EvalSymlinks(joined)
	if err != nil {
		return "", fmt.Errorf("ai42dataset: resolve %s: %w", name, err)
	}
	resolved, err = filepath.Abs(resolved)
	if err != nil {
		return "", fmt.Errorf("ai42dataset: resolve %s: %w", name, err)
	}
	rootAbs, err := filepath.Abs(root)
	if err != nil {
		return "", fmt.Errorf("ai42dataset: resolve generation root: %w", err)
	}
	rel, err = filepath.Rel(rootAbs, resolved)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
		return "", fmt.Errorf("ai42dataset: path %q escapes generation root through a symlink", name)
	}
	info, err := os.Stat(resolved)
	if err != nil {
		return "", fmt.Errorf("ai42dataset: stat %s: %w", name, err)
	}
	if !info.Mode().IsRegular() {
		return "", fmt.Errorf("ai42dataset: %s is not a regular file", name)
	}
	return resolved, nil
}

func readBoundedRegularFile(path string, limit int64) ([]byte, os.FileInfo, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nil, nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, nil, fmt.Errorf("not a regular file")
	}
	if info.Size() < 1 || info.Size() > limit {
		return nil, nil, fmt.Errorf("size %d is outside [1,%d]", info.Size(), limit)
	}
	reader := &io.LimitedReader{R: file, N: limit + 1}
	payload, err := io.ReadAll(reader)
	if err != nil {
		return nil, nil, err
	}
	if int64(len(payload)) != info.Size() || int64(len(payload)) > limit {
		return nil, nil, fmt.Errorf("file changed while reading or exceeds %d bytes", limit)
	}
	return payload, info, nil
}

func validateManifest(root string, manifest map[string]any, manifestHash string) (*Generation, error) {
	if got, ok := manifest["dataset_schema_version"].(string); !ok || got != DatasetSchemaVersion {
		return nil, fmt.Errorf("ai42dataset: manifest dataset schema mismatch")
	}
	if got, ok := manifest["shard_schema_version"].(string); !ok || got != ShardSchemaVersionV2 {
		return nil, fmt.Errorf("ai42dataset: manifest shard schema mismatch")
	}
	if got, err := intField(manifest["protocol_version"], "manifest.protocol_version", 0); err != nil || uint16(got) != ProtocolVersion {
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("ai42dataset: manifest protocol mismatch")
	}
	for name, expected := range map[string]string{
		"schema_hash": hashHex(AI42SchemaHash), "reward_hash": hashHex(AI42RewardHash),
		"trajectory_schema_hash": hashHex(AI42TrajectorySchemaHash),
	} {
		if got, err := hashField(manifest[name], "manifest."+name); err != nil {
			return nil, err
		} else if got != expected {
			return nil, fmt.Errorf("ai42dataset: manifest %s mismatch", name)
		}
	}
	runtime, ok := manifest["runtime_manifest"].(map[string]any)
	if !ok || len(runtime) == 0 {
		return nil, fmt.Errorf("ai42dataset: manifest.runtime_manifest must be a non-empty object")
	}
	runtimeHash, err := hashField(manifest["runtime_manifest_hash"], "manifest.runtime_manifest_hash")
	if err != nil {
		return nil, err
	}
	if sha256Hex(canonicalJSON(runtime)) != runtimeHash {
		return nil, fmt.Errorf("ai42dataset: manifest runtime manifest hash mismatch")
	}
	splitSeed, err := int64Field(manifest["split_seed"], "manifest.split_seed", math.MinInt64)
	if err != nil {
		return nil, err
	}
	validationFraction, err := floatField(manifest["validation_fraction"], "manifest.validation_fraction")
	if err != nil || validationFraction < 0 || validationFraction > 1 {
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("ai42dataset: manifest.validation_fraction must be between zero and one")
	}

	matchValues, ok := manifest["matches"].([]any)
	if !ok || len(matchValues) == 0 {
		return nil, fmt.Errorf("ai42dataset: manifest.matches must be non-empty")
	}
	if len(matchValues) > maxGenerationMatches {
		return nil, fmt.Errorf("ai42dataset: manifest.matches count %d exceeds %d", len(matchValues), maxGenerationMatches)
	}
	shardValues, ok := manifest["shards"].([]any)
	if !ok || len(shardValues) == 0 {
		return nil, fmt.Errorf("ai42dataset: manifest.shards must be non-empty")
	}
	if len(shardValues) > maxGenerationShards {
		return nil, fmt.Errorf("ai42dataset: manifest.shards count %d exceeds %d", len(shardValues), maxGenerationShards)
	}
	generation := &Generation{root: root, manifest: manifest, manifestHash: manifestHash}
	byID := make(map[string]*matchRecord, len(matchValues))
	lastMatchID := ""
	var totalMatchRows int64
	for index, value := range matchValues {
		object, ok := value.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("ai42dataset: manifest.matches[%d] must be an object", index)
		}
		match, err := parseMatch(object, fmt.Sprintf("manifest.matches[%d]", index))
		if err != nil {
			return nil, err
		}
		totalMatchRows, err = addCappedTotal(totalMatchRows, int64(match.metadata.TickCount), maxGenerationRows, "manifest match rows")
		if err != nil {
			return nil, err
		}
		if index > 0 && match.metadata.MatchID <= lastMatchID {
			return nil, fmt.Errorf("ai42dataset: manifest matches must be unique and ordered")
		}
		lastMatchID = match.metadata.MatchID
		if _, exists := byID[match.metadata.MatchID]; exists {
			return nil, fmt.Errorf("ai42dataset: duplicate match ID %q", match.metadata.MatchID)
		}
		byID[match.metadata.MatchID] = match
		generation.matches = append(generation.matches, match)
	}
	if err := validateDeterministicSplits(generation.matches, splitSeed, validationFraction, runtime); err != nil {
		return nil, err
	}

	byShard := make(map[string]*shardRecord, len(shardValues))
	lastShardName := ""
	var totalShardRows, totalRawBytes, totalStoredBytes int64
	for index, value := range shardValues {
		object, ok := value.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("ai42dataset: manifest.shards[%d] must be an object", index)
		}
		if err := requireFields(object, shardManifestFields, fmt.Sprintf("manifest.shards[%d]", index)); err != nil {
			return nil, err
		}
		name, ok := object["name"].(string)
		if !ok || name == "" {
			return nil, fmt.Errorf("ai42dataset: manifest.shards[%d].name is invalid", index)
		}
		if index > 0 && name <= lastShardName {
			return nil, fmt.Errorf("ai42dataset: manifest shards must be unique and ordered")
		}
		lastShardName = name
		if _, exists := byShard[name]; exists {
			return nil, fmt.Errorf("ai42dataset: duplicate shard name %q", name)
		}
		if _, err := containedRegularFile(root, name); err != nil {
			return nil, err
		}
		sha, err := hashField(object["sha256"], fmt.Sprintf("manifest.shards[%d].sha256", index))
		if err != nil {
			return nil, err
		}
		compression, ok := object["compression"].(string)
		if !ok || compression != ShardCodec {
			return nil, fmt.Errorf("ai42dataset: manifest shard %s compression mismatch", name)
		}
		rowCount, err := intFromField(object["row_count"], fmt.Sprintf("manifest.shards[%d].row_count", index), 1)
		if err != nil {
			return nil, err
		}
		if rowCount > maxShardRows {
			return nil, fmt.Errorf("ai42dataset: manifest shard %s row_count %d exceeds %d", name, rowCount, maxShardRows)
		}
		rawBytes, err := int64Field(object["raw_bytes"], fmt.Sprintf("manifest.shards[%d].raw_bytes", index), 1)
		if err != nil {
			return nil, err
		}
		if rawBytes > maxShardRawBytes {
			return nil, fmt.Errorf("ai42dataset: manifest shard %s raw_bytes %d exceeds %d", name, rawBytes, maxShardRawBytes)
		}
		expectedRawBytes, err := expectedRawBytesForRows(rowCount)
		if err != nil {
			return nil, fmt.Errorf("ai42dataset: manifest shard %s: %w", name, err)
		}
		if rawBytes != expectedRawBytes {
			return nil, fmt.Errorf("ai42dataset: manifest shard %s raw_bytes %d does not match %d rows (%d)", name, rawBytes, rowCount, expectedRawBytes)
		}
		storedBytes, err := int64Field(object["stored_bytes"], fmt.Sprintf("manifest.shards[%d].stored_bytes", index), 1)
		if err != nil {
			return nil, err
		}
		if storedBytes > maxShardStoredBytes {
			return nil, fmt.Errorf("ai42dataset: manifest shard %s stored_bytes %d exceeds %d", name, storedBytes, maxShardStoredBytes)
		}
		totalShardRows, err = addCappedTotal(totalShardRows, int64(rowCount), maxGenerationRows, "manifest shard rows")
		if err != nil {
			return nil, err
		}
		totalRawBytes, err = addCappedTotal(totalRawBytes, rawBytes, maxGenerationRawBytes, "manifest raw bytes")
		if err != nil {
			return nil, err
		}
		totalStoredBytes, err = addCappedTotal(totalStoredBytes, storedBytes, maxGenerationStoredBytes, "manifest stored bytes")
		if err != nil {
			return nil, err
		}
		idValues, ok := object["match_ids"].([]any)
		if !ok || len(idValues) == 0 {
			return nil, fmt.Errorf("ai42dataset: manifest shard %s match_ids must be non-empty", name)
		}
		if len(idValues) > maxMatchesPerShard {
			return nil, fmt.Errorf("ai42dataset: manifest shard %s match_ids count %d exceeds %d", name, len(idValues), maxMatchesPerShard)
		}
		ids := make([]string, len(idValues))
		for idIndex, value := range idValues {
			id, ok := value.(string)
			if !ok || id == "" || (idIndex > 0 && id <= ids[idIndex-1]) {
				return nil, fmt.Errorf("ai42dataset: manifest shard %s match_ids must be ordered non-empty strings", name)
			}
			ids[idIndex] = id
		}
		shard := &shardRecord{raw: object, name: name, sha256: sha, rowCount: rowCount, rawBytes: rawBytes, storedBytes: storedBytes, matchIDs: ids}
		if _, exists := byShard[name]; exists {
			return nil, fmt.Errorf("ai42dataset: duplicate shard %q", name)
		}
		byShard[name] = shard
		generation.shards = append(generation.shards, shard)
	}
	if totalMatchRows != totalShardRows {
		return nil, fmt.Errorf("ai42dataset: manifest match/shard cumulative row counts differ")
	}

	assigned := make(map[string]string, len(byID))
	for _, shard := range generation.shards {
		rowOffset := 0
		expectedIDs := make([]string, 0, len(shard.matchIDs))
		for _, match := range generation.matches {
			if match.metadata.Shard == shard.name {
				expectedIDs = append(expectedIDs, match.metadata.MatchID)
			}
		}
		if !equalStrings(shard.matchIDs, expectedIDs) {
			return nil, fmt.Errorf("ai42dataset: shard %s match ordering mismatch", shard.name)
		}
		for _, id := range shard.matchIDs {
			match, ok := byID[id]
			if !ok {
				return nil, fmt.Errorf("ai42dataset: shard %s references unknown match %q", shard.name, id)
			}
			if previous, exists := assigned[id]; exists {
				return nil, fmt.Errorf("ai42dataset: match %s assigned to shards %s and %s", id, previous, shard.name)
			}
			if match.metadata.Shard != shard.name || match.metadata.RowOffset != rowOffset {
				return nil, fmt.Errorf("ai42dataset: shard %s match offsets are not contiguous", shard.name)
			}
			assigned[id] = shard.name
			match.shard = shard
			if match.metadata.TickCount > int(^uint(0)>>1)-rowOffset {
				return nil, fmt.Errorf("ai42dataset: shard %s row offsets overflow", shard.name)
			}
			match.rowEnd = rowOffset + match.metadata.TickCount
			shard.matches = append(shard.matches, match)
			rowOffset = match.rowEnd
		}
		if rowOffset != shard.rowCount {
			return nil, fmt.Errorf("ai42dataset: shard %s row_count mismatch", shard.name)
		}
	}
	if len(assigned) != len(byID) {
		return nil, fmt.Errorf("ai42dataset: manifest does not assign every match exactly once")
	}
	for _, match := range generation.matches {
		if match.shard == nil {
			return nil, fmt.Errorf("ai42dataset: match %s is not assigned to a shard", match.metadata.MatchID)
		}
	}
	return generation, nil
}

func validateDeterministicSplits(matches []*matchRecord, seed int64, fraction float64, runtimeValues ...map[string]any) error {
	if math.IsNaN(fraction) || math.IsInf(fraction, 0) || fraction < 0 || fraction > 1 {
		return fmt.Errorf("ai42dataset: validation_fraction must be between zero and one")
	}
	if len(runtimeValues) > 1 {
		return fmt.Errorf("ai42dataset: deterministic split received multiple runtime manifests")
	}
	if len(runtimeValues) == 1 && runtimeValues[0] != nil {
		if scenarioMix, present := runtimeValues[0]["scenario_mix"]; present && scenarioMix != nil {
			return validateDeterministicStratifiedSplits(matches, seed, fraction, runtimeValues[0], scenarioMix)
		}
	}
	return validateDeterministicSimpleSplits(matches, seed, fraction)
}

type rankedSplitMatch struct {
	match  *matchRecord
	digest Hash
}

func validateDeterministicSimpleSplits(matches []*matchRecord, seed int64, fraction float64) error {
	seen := make(map[string]struct{}, len(matches))
	ranked := make([]rankedSplitMatch, len(matches))
	for index, match := range matches {
		if match == nil || match.metadata.MatchID == "" {
			return fmt.Errorf("ai42dataset: deterministic split contains an invalid match")
		}
		if _, exists := seen[match.metadata.MatchID]; exists {
			return fmt.Errorf("ai42dataset: match IDs must be unique")
		}
		seen[match.metadata.MatchID] = struct{}{}
		ranked[index] = rankedSplitMatch{
			match:  match,
			digest: sha256.Sum256([]byte(strconv.FormatInt(seed, 10) + "\x00" + match.metadata.MatchID)),
		}
	}
	sort.Slice(ranked, func(left, right int) bool {
		comparison := bytes.Compare(ranked[left].digest[:], ranked[right].digest[:])
		if comparison != 0 {
			return comparison < 0
		}
		return ranked[left].match.metadata.MatchID < ranked[right].match.metadata.MatchID
	})
	validationCount := 0
	switch {
	case fraction <= 0:
		validationCount = 0
	case fraction >= 1:
		validationCount = len(matches)
	case len(matches) <= 1:
		validationCount = 0
	default:
		validationCount = int(math.Ceil(float64(len(matches)) * fraction))
		if validationCount < 1 {
			validationCount = 1
		}
		if validationCount > len(matches)-1 {
			validationCount = len(matches) - 1
		}
	}
	validation := make(map[string]struct{}, validationCount)
	for index := 0; index < validationCount; index++ {
		validation[ranked[index].match.metadata.MatchID] = struct{}{}
	}
	for _, match := range matches {
		expected := "train"
		if _, ok := validation[match.metadata.MatchID]; ok || fraction >= 1 {
			expected = "validation"
		}
		if match.metadata.Split != expected {
			return fmt.Errorf("ai42dataset: match %s split %q does not match deterministic split %q", match.metadata.MatchID, match.metadata.Split, expected)
		}
	}
	return nil
}

func validateDeterministicStratifiedSplits(matches []*matchRecord, seed int64, fraction float64, runtime map[string]any, scenarioMixValue any) error {
	type scenarioQuota struct {
		train      int
		validation int
	}

	if len(matches) == 0 {
		return fmt.Errorf("ai42dataset: match IDs must be non-empty")
	}
	matchByID := make(map[string]*matchRecord, len(matches))
	for _, match := range matches {
		if match == nil || match.metadata.MatchID == "" {
			return fmt.Errorf("ai42dataset: match IDs must be non-empty")
		}
		if _, exists := matchByID[match.metadata.MatchID]; exists {
			return fmt.Errorf("ai42dataset: match IDs must be unique")
		}
		matchByID[match.metadata.MatchID] = match
	}

	pairs, err := splitScenarioPairs(matches, runtime)
	if err != nil {
		return err
	}
	if len(pairs) != len(matches) {
		return fmt.Errorf("ai42dataset: match set does not match the frozen schedule")
	}

	scenarioMix, ok := scenarioMixValue.(map[string]any)
	if !ok || len(scenarioMix) == 0 {
		return fmt.Errorf("ai42dataset: scenario_mix must be a non-empty mapping")
	}
	if len(scenarioMix) > maxGenerationMatches {
		return fmt.Errorf("ai42dataset: scenario_mix contains too many scenarios")
	}
	normalized := make(map[string]scenarioQuota, len(scenarioMix))
	for scenario, value := range scenarioMix {
		if scenario == "" || !utf8.ValidString(scenario) {
			return fmt.Errorf("ai42dataset: scenario_mix keys must be non-empty UTF-8 strings")
		}
		quota, ok := value.(map[string]any)
		if !ok || len(quota) != 2 {
			return fmt.Errorf("ai42dataset: scenario_mix[%q] must contain exactly train and validation quotas", scenario)
		}
		train, err := intFromField(quota["train"], fmt.Sprintf("runtime_manifest.scenario_mix[%q].train", scenario), 0)
		if err != nil {
			return err
		}
		validation, err := intFromField(quota["validation"], fmt.Sprintf("runtime_manifest.scenario_mix[%q].validation", scenario), 0)
		if err != nil {
			return err
		}
		if train > maxGenerationMatches || validation > maxGenerationMatches {
			return fmt.Errorf("ai42dataset: scenario_mix[%q] quota exceeds %d matches", scenario, maxGenerationMatches)
		}
		normalized[scenario] = scenarioQuota{train: train, validation: validation}
	}

	observed := make(map[string]struct{}, len(pairs))
	grouped := make(map[string][]string, len(normalized))
	for _, pair := range pairs {
		if _, exists := matchByID[pair.matchID]; !exists {
			return fmt.Errorf("ai42dataset: frozen schedule references unknown match %q", pair.matchID)
		}
		if _, exists := observed[pair.scenario]; !exists {
			observed[pair.scenario] = struct{}{}
		}
		grouped[pair.scenario] = append(grouped[pair.scenario], pair.matchID)
	}
	if len(observed) != len(normalized) {
		return fmt.Errorf("ai42dataset: scenario_mix scenarios must exactly match the schedule")
	}
	for scenario := range normalized {
		if _, exists := observed[scenario]; !exists {
			return fmt.Errorf("ai42dataset: scenario_mix scenarios must exactly match the schedule")
		}
	}

	validationTotal := 0
	for scenario, matchIDs := range grouped {
		quota := normalized[scenario]
		if quota.train > len(matchIDs) || quota.validation > len(matchIDs) || quota.train > len(matchIDs)-quota.validation || quota.train+quota.validation != len(matchIDs) {
			return fmt.Errorf("ai42dataset: scenario_mix quota does not match %q match count", scenario)
		}
		validationTotal += quota.validation
	}
	if float64(validationTotal) != float64(len(pairs))*fraction {
		return fmt.Errorf("ai42dataset: scenario_mix validation quotas do not match validation_fraction")
	}

	for scenario, matchIDs := range grouped {
		quota := normalized[scenario]
		ranked := make([]rankedSplitMatch, len(matchIDs))
		for index, matchID := range matchIDs {
			ranked[index] = rankedSplitMatch{
				match:  matchByID[matchID],
				digest: sha256.Sum256([]byte(strconv.FormatInt(seed, 10) + "\x00" + matchID)),
			}
		}
		sort.Slice(ranked, func(left, right int) bool {
			comparison := bytes.Compare(ranked[left].digest[:], ranked[right].digest[:])
			if comparison != 0 {
				return comparison < 0
			}
			return ranked[left].match.metadata.MatchID < ranked[right].match.metadata.MatchID
		})
		validation := make(map[string]struct{}, quota.validation)
		for index := 0; index < quota.validation; index++ {
			validation[ranked[index].match.metadata.MatchID] = struct{}{}
		}
		for _, matchID := range matchIDs {
			expected := "train"
			if _, ok := validation[matchID]; ok {
				expected = "validation"
			}
			if matchByID[matchID].metadata.Split != expected {
				return fmt.Errorf("ai42dataset: match %s split %q does not match deterministic split %q", matchID, matchByID[matchID].metadata.Split, expected)
			}
		}
	}
	return nil
}

type splitScenarioPair struct {
	matchID  string
	scenario string
}

func splitScenarioPairs(matches []*matchRecord, runtime map[string]any) ([]splitScenarioPair, error) {
	if scheduleValue, present := runtime["match_schedule"]; present && scheduleValue != nil {
		schedule, ok := scheduleValue.([]any)
		if !ok {
			return nil, fmt.Errorf("ai42dataset: runtime_manifest.match_schedule must be an array")
		}
		if len(schedule) > 0 {
			if len(schedule) > maxGenerationMatches {
				return nil, fmt.Errorf("ai42dataset: runtime_manifest.match_schedule count %d exceeds %d", len(schedule), maxGenerationMatches)
			}
			pairs := make([]splitScenarioPair, 0, len(schedule))
			seen := make(map[string]struct{}, len(schedule))
			for index, value := range schedule {
				object, ok := value.(map[string]any)
				if !ok {
					return nil, fmt.Errorf("ai42dataset: runtime_manifest.match_schedule[%d] must be an object", index)
				}
				matchID, err := nonEmptyString(object["match_id"], fmt.Sprintf("runtime_manifest.match_schedule[%d].match_id", index))
				if err != nil {
					return nil, err
				}
				scenario, err := nonEmptyString(object["scenario"], fmt.Sprintf("runtime_manifest.match_schedule[%d].scenario", index))
				if err != nil {
					return nil, err
				}
				if _, exists := seen[matchID]; exists {
					return nil, fmt.Errorf("ai42dataset: runtime_manifest.match_schedule contains duplicate match ID %q", matchID)
				}
				seen[matchID] = struct{}{}
				pairs = append(pairs, splitScenarioPair{matchID: matchID, scenario: scenario})
			}
			return pairs, nil
		}
	}

	pairs := make([]splitScenarioPair, len(matches))
	for index, match := range matches {
		if match == nil {
			return nil, fmt.Errorf("ai42dataset: match IDs must be non-empty")
		}
		pairs[index] = splitScenarioPair{matchID: match.metadata.MatchID, scenario: match.metadata.Scenario}
	}
	return pairs, nil
}

func parseMatch(object map[string]any, path string) (*matchRecord, error) {
	if err := requireFields(object, matchV2Fields, path); err != nil {
		return nil, err
	}
	matchID, err := nonEmptyString(object["match_id"], path+".match_id")
	if err != nil {
		return nil, err
	}
	split, err := nonEmptyString(object["split"], path+".split")
	if err != nil || (split != "train" && split != "validation") {
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("ai42dataset: %s.split is invalid", path)
	}
	shardName, err := nonEmptyString(object["shard"], path+".shard")
	if err != nil {
		return nil, err
	}
	rowOffset, err := intFromField(object["row_offset"], path+".row_offset", 0)
	if err != nil {
		return nil, err
	}
	if rowOffset > maxShardRows {
		return nil, fmt.Errorf("ai42dataset: %s.row_offset %d exceeds %d", path, rowOffset, maxShardRows)
	}
	tickCount, err := intFromField(object["tick_count"], path+".tick_count", 1)
	if err != nil {
		return nil, err
	}
	if tickCount > maxMatchRows {
		return nil, fmt.Errorf("ai42dataset: %s.tick_count %d exceeds %d", path, tickCount, maxMatchRows)
	}
	firstStep, err := uint32Field(object["first_step"], path+".first_step")
	if err != nil {
		return nil, err
	}
	if uint64(firstStep)+uint64(tickCount-1) > math.MaxUint32 {
		return nil, fmt.Errorf("ai42dataset: %s step range overflows uint32", path)
	}
	lineage, err := nonEmptyString(object["recurrent_lineage_schema"], path+".recurrent_lineage_schema")
	if err != nil || lineage != RecurrentLineageSchemaV2 {
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("ai42dataset: %s.recurrent_lineage_schema mismatch", path)
	}
	heroIDs, err := stringArray(object["hero_ids"], HeroCount, path+".hero_ids", true)
	if err != nil {
		return nil, err
	}
	trajectoryIDs, err := stringArray(object["trajectory_ids"], HeroCount, path+".trajectory_ids", true)
	if err != nil {
		return nil, err
	}
	trajectoryHashes, err := hashArray(object["trajectory_hashes"], path+".trajectory_hashes")
	if err != nil {
		return nil, err
	}
	for hero := 0; hero < HeroCount; hero++ {
		expected := matchID + ":hero:" + heroIDs[hero]
		if trajectoryIDs[hero] != expected {
			return nil, fmt.Errorf("ai42dataset: %s.trajectory_ids[%d] is not bound to match metadata", path, hero)
		}
	}
	seed, err := int64Field(object["seed"], path+".seed", math.MinInt64)
	if err != nil {
		return nil, err
	}
	scenario, err := nonEmptyString(object["scenario"], path+".scenario")
	if err != nil {
		return nil, err
	}
	controllers, err := uint8Array(object["controller_by_slot"], HeroCount, path+".controller_by_slot")
	if err != nil {
		return nil, err
	}
	for index, value := range controllers {
		if value > uint8(battleserver.AssaultControllerAI40) {
			return nil, fmt.Errorf("ai42dataset: %s.controller_by_slot[%d] is invalid", path, index)
		}
	}
	roster, err := int32Array(object["roster_ids"], HeroCount, path+".roster_ids")
	if err != nil {
		return nil, err
	}
	sides, err := uint8Array(object["side_by_slot"], HeroCount, path+".side_by_slot")
	if err != nil {
		return nil, err
	}
	zero, one := 0, 0
	for _, value := range sides {
		if value > 1 {
			return nil, fmt.Errorf("ai42dataset: %s.side_by_slot contains invalid side", path)
		}
		if value == 0 {
			zero++
		} else {
			one++
		}
	}
	if zero != HeroCount/2 || one != HeroCount/2 {
		return nil, fmt.Errorf("ai42dataset: %s.side_by_slot must contain five slots per side", path)
	}
	return &matchRecord{
		raw: object,
		metadata: MatchMetadata{MatchID: matchID, Split: split, Shard: shardName, RowOffset: rowOffset, TickCount: tickCount,
			FirstStep: firstStep, RecurrentLineageSchema: lineage, HeroIDs: heroIDs, TrajectoryIDs: trajectoryIDs,
			TrajectoryHashes: trajectoryHashes, Seed: seed, Scenario: scenario, ControllerBySlot: controllers,
			RosterIDs: roster, SideBySlot: sides},
		trajectoryHashHex: make([]string, HeroCount),
	}, nil
}

func (g *Generation) inspectShardHeader(shard *shardRecord) error {
	path, err := containedRegularFile(g.root, shard.name)
	if err != nil {
		return err
	}
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("ai42dataset: open shard %s: %w", shard.name, err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return fmt.Errorf("ai42dataset: stat shard %s: %w", shard.name, err)
	}
	_, err = g.readShardPreamble(file, info, shard)
	return err
}

func (g *Generation) verifyShard(ctx context.Context, shard *shardRecord, options VerifyOptions, report *VerificationReport) error {
	path, err := containedRegularFile(g.root, shard.name)
	if err != nil {
		return err
	}
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("ai42dataset: open shard %s: %w", shard.name, err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return fmt.Errorf("ai42dataset: stat shard %s: %w", shard.name, err)
	}
	preamble, err := g.readShardPreamble(file, info, shard)
	if err != nil {
		return err
	}
	wholeHash := sha256.New()
	_, _ = wholeHash.Write(preamble.prefix)
	_, _ = wholeHash.Write(preamble.headerBytes)
	return g.streamAndVerifyRaw(ctx, file, shard, preamble, wholeHash, options, report)
}

func (g *Generation) readShardPreamble(file *os.File, info os.FileInfo, shard *shardRecord) (shardPreamble, error) {
	var result shardPreamble
	if !info.Mode().IsRegular() {
		return result, fmt.Errorf("ai42dataset: shard %s is not a regular file", shard.name)
	}
	if info.Size() != shard.storedBytes {
		return result, fmt.Errorf("ai42dataset: shard %s stored byte count mismatch", shard.name)
	}
	prefix := make([]byte, len(ShardMagic)+4)
	if _, err := io.ReadFull(file, prefix); err != nil {
		return result, fmt.Errorf("ai42dataset: shard %s prefix is truncated: %w", shard.name, err)
	}
	if !bytes.Equal(prefix[:len(ShardMagic)], []byte(ShardMagic)) {
		return result, fmt.Errorf("ai42dataset: shard %s magic mismatch", shard.name)
	}
	headerLength := int(binary.LittleEndian.Uint32(prefix[len(ShardMagic):]))
	if headerLength < 2 || headerLength > maxShardHeaderBytes {
		return result, fmt.Errorf("ai42dataset: shard %s header length is invalid", shard.name)
	}
	headerBytes := make([]byte, headerLength)
	if _, err := io.ReadFull(file, headerBytes); err != nil {
		return result, fmt.Errorf("ai42dataset: shard %s header is truncated: %w", shard.name, err)
	}
	value, err := decodeCanonicalJSON(headerBytes, shard.name+".header")
	if err != nil {
		return result, err
	}
	header, ok := value.(map[string]any)
	if !ok {
		return result, fmt.Errorf("ai42dataset: shard %s header must be an object", shard.name)
	}
	if err := requireFields(header, headerFields, shard.name+".header"); err != nil {
		return result, err
	}
	if err := validateHeader(header, shard, g.manifest, shard.name); err != nil {
		return result, err
	}
	descriptors, err := parseDescriptors(header["arrays"], shard.rowCount, shard.name+".arrays")
	if err != nil {
		return result, err
	}
	var descriptorBytes int64
	for _, descriptor := range descriptors {
		if descriptorBytes > math.MaxInt64-descriptor.nbytes {
			return result, fmt.Errorf("ai42dataset: shard %s array byte count overflows", shard.name)
		}
		descriptorBytes += descriptor.nbytes
	}
	headerRawBytes, err := int64Field(header["raw_bytes"], shard.name+".header.raw_bytes", 1)
	if err != nil {
		return result, err
	}
	if descriptorBytes != headerRawBytes {
		return result, fmt.Errorf("ai42dataset: shard %s descriptor byte count mismatch", shard.name)
	}
	compressedBytes := info.Size() - int64(len(prefix)) - int64(headerLength)
	if compressedBytes < 1 {
		return result, fmt.Errorf("ai42dataset: shard %s has no compressed payload", shard.name)
	}
	headerStored, err := int64Field(header["stored_bytes"], shard.name+".header.stored_bytes", 1)
	if err != nil {
		return result, err
	}
	if headerStored != compressedBytes {
		return result, fmt.Errorf("ai42dataset: shard %s compressed byte count mismatch", shard.name)
	}
	result.prefix = prefix
	result.headerBytes = headerBytes
	result.header = header
	result.descriptors = descriptors
	result.compressedBytes = compressedBytes
	return result, nil
}

func validateHeader(header map[string]any, shard *shardRecord, manifest map[string]any, name string) error {
	if got, ok := header["shard_schema_version"].(string); !ok || got != ShardSchemaVersionV2 {
		return fmt.Errorf("ai42dataset: shard %s schema mismatch", name)
	}
	protocol, err := intField(header["protocol_version"], name+".protocol_version", 0)
	if err != nil {
		return err
	}
	if uint16(protocol) != ProtocolVersion {
		return fmt.Errorf("ai42dataset: shard %s protocol mismatch", name)
	}
	for _, field := range []string{"schema_hash", "reward_hash", "trajectory_schema_hash", "runtime_manifest_hash"} {
		headerValue, err := hashField(header[field], name+"."+field)
		if err != nil {
			return err
		}
		manifestValue, err := hashField(manifest[field], "manifest."+field)
		if err != nil {
			return err
		}
		if headerValue != manifestValue {
			return fmt.Errorf("ai42dataset: shard %s %s mismatch", name, field)
		}
	}
	if codec, ok := header["codec"].(string); !ok || codec != ShardCodec {
		return fmt.Errorf("ai42dataset: shard %s codec mismatch", name)
	}
	if rawBytes, err := int64Field(header["raw_bytes"], name+".raw_bytes", 1); err != nil {
		return err
	} else if rawBytes != shard.rawBytes {
		return fmt.Errorf("ai42dataset: shard %s raw byte metadata mismatch", name)
	}
	payloadHash, err := hashField(header["payload_sha256"], name+".payload_sha256")
	if err != nil {
		return err
	}
	if payloadHash == "" {
		return fmt.Errorf("ai42dataset: shard %s payload hash is empty", name)
	}
	if _, err := hashField(header["raw_sha256"], name+".raw_sha256"); err != nil {
		return err
	}
	matchValues, ok := header["matches"].([]any)
	if !ok || len(matchValues) != len(shard.matches) {
		return fmt.Errorf("ai42dataset: shard %s match metadata count mismatch", name)
	}
	expected := make([]any, len(shard.matches))
	manifestMatches, _ := manifest["matches"].([]any)
	for index, match := range shard.matches {
		for _, value := range manifestMatches {
			object, ok := value.(map[string]any)
			if ok && object["match_id"] == match.metadata.MatchID {
				expected[index] = object
				break
			}
		}
		if expected[index] == nil || !bytes.Equal(canonicalJSON(matchValues[index]), canonicalJSON(expected[index])) {
			return fmt.Errorf("ai42dataset: shard %s match metadata mismatch", name)
		}
	}
	return nil
}

func parseDescriptors(value any, rows int, path string) ([]arrayDescriptor, error) {
	if rows < 1 || rows > maxShardRows {
		return nil, fmt.Errorf("ai42dataset: %s row dimension %d is outside [1,%d]", path, rows, maxShardRows)
	}
	values, ok := value.([]any)
	if !ok || len(values) != len(arrayNames) {
		return nil, fmt.Errorf("ai42dataset: %s must contain the ordered AI42 array set", path)
	}
	descriptors := make([]arrayDescriptor, len(values))
	var expectedOffset int64
	for index, value := range values {
		object, ok := value.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("ai42dataset: %s[%d] must be an object", path, index)
		}
		itemPath := fmt.Sprintf("%s[%d]", path, index)
		if err := requireFields(object, arrayDescriptorFields, itemPath); err != nil {
			return nil, err
		}
		name, ok := object["name"].(string)
		if !ok || index >= len(arrayNames) || name != arrayNames[index] {
			return nil, fmt.Errorf("ai42dataset: %s.name is not the expected ordered array", itemPath)
		}
		dtype, ok := object["dtype"].(string)
		if !ok || dtype != arrayDType(name) {
			return nil, fmt.Errorf("ai42dataset: %s.dtype mismatch", itemPath)
		}
		shape, err := shapeArray(object["shape"], itemPath+".shape")
		if err != nil {
			return nil, err
		}
		expectedShape := arrayShape(name, rows)
		if !equalInts(shape, expectedShape) {
			return nil, fmt.Errorf("ai42dataset: %s.shape mismatch", itemPath)
		}
		offset, err := int64Field(object["offset"], itemPath+".offset", 0)
		if err != nil {
			return nil, err
		}
		nbytes, err := int64Field(object["nbytes"], itemPath+".nbytes", 1)
		if err != nil {
			return nil, err
		}
		if offset != expectedOffset {
			return nil, fmt.Errorf("ai42dataset: %s.offset is not contiguous", itemPath)
		}
		itemsize, ok := dtypeSize(dtype)
		if !ok {
			return nil, fmt.Errorf("ai42dataset: %s.dtype is unknown", itemPath)
		}
		product, ok := shapeProduct(shape)
		if !ok || product > math.MaxInt64/int64(itemsize) || product*int64(itemsize) != nbytes {
			return nil, fmt.Errorf("ai42dataset: %s.nbytes mismatch", itemPath)
		}
		if nbytes%int64(rows) != 0 {
			return nil, fmt.Errorf("ai42dataset: %s.nbytes is not row divisible", itemPath)
		}
		if expectedOffset > maxShardRawBytes-nbytes {
			return nil, fmt.Errorf("ai42dataset: %s exceeds shard raw-byte cap %d", itemPath, maxShardRawBytes)
		}
		descriptors[index] = arrayDescriptor{name: name, dtype: dtype, shape: shape, offset: offset, nbytes: nbytes, rowBytes: nbytes / int64(rows)}
		expectedOffset += nbytes
	}
	return descriptors, nil
}

func dtypeSize(dtype string) (int, bool) {
	switch dtype {
	case "u1":
		return 1, true
	case "<f4", "<i4", "<u4":
		return 4, true
	case "action":
		return 5, true
	default:
		return 0, false
	}
}

func expectedRawBytesForRows(rows int) (int64, error) {
	if rows < 1 || rows > maxShardRows {
		return 0, fmt.Errorf("row count %d is outside [1,%d]", rows, maxShardRows)
	}
	var total int64
	for _, name := range arrayNames {
		itemSize, ok := dtypeSize(arrayDType(name))
		if !ok {
			return 0, fmt.Errorf("unknown dtype for array %q", name)
		}
		items, ok := shapeProduct(arrayShape(name, rows))
		if !ok || items > math.MaxInt64/int64(itemSize) {
			return 0, fmt.Errorf("array %q dimensions overflow", name)
		}
		bytes := items * int64(itemSize)
		if total > maxShardRawBytes-bytes {
			return 0, fmt.Errorf("raw dimensions exceed %d bytes", maxShardRawBytes)
		}
		total += bytes
	}
	return total, nil
}

func (g *Generation) streamAndVerifyRaw(ctx context.Context, file *os.File, shard *shardRecord, preamble shardPreamble, wholeHash hash.Hash, options VerifyOptions, report *VerificationReport) error {
	header := preamble.header
	// The caller has consumed the prefix and header.  The remaining file bytes
	// are bounded by the manifest's stored size and are hashed as flate reads.
	remaining := preamble.compressedBytes
	if remaining < 1 {
		return fmt.Errorf("ai42dataset: shard %s compressed payload is empty", shard.name)
	}
	if remaining > maxShardStoredBytes {
		return fmt.Errorf("ai42dataset: shard %s compressed payload exceeds %d bytes", shard.name, maxShardStoredBytes)
	}
	expectedRaw, err := int64Field(header["raw_bytes"], shard.name+".header.raw_bytes", 1)
	if err != nil {
		return err
	}
	if expectedRaw > maxShardRawBytes {
		return fmt.Errorf("ai42dataset: shard %s raw payload declaration exceeds %d bytes", shard.name, maxShardRawBytes)
	}
	limited := &countingReader{reader: io.LimitReader(file, remaining)}
	compressedHash := sha256.New()
	hashedCompressed := io.TeeReader(limited, io.MultiWriter(compressedHash, wholeHash))
	buffered := bufio.NewReader(hashedCompressed)
	decompressor := flate.NewReader(buffered)
	defer decompressor.Close()

	temporary, err := os.CreateTemp("", "ai42dataset-raw-*")
	if err != nil {
		return fmt.Errorf("ai42dataset: create bounded raw spool: %w", err)
	}
	temporaryName := temporary.Name()
	defer func() {
		_ = temporary.Close()
		_ = os.Remove(temporaryName)
	}()
	rawHash := sha256.New()
	copyBuffer := make([]byte, 128*1024)
	boundedRaw := &io.LimitedReader{R: &contextReader{ctx: ctx, reader: decompressor}, N: expectedRaw + 1}
	written, err := io.CopyBuffer(io.MultiWriter(temporary, rawHash), boundedRaw, copyBuffer)
	if err != nil {
		return fmt.Errorf("ai42dataset: shard %s compressed payload is corrupt: %w", shard.name, err)
	}
	if written > expectedRaw || boundedRaw.N == 0 {
		return fmt.Errorf("ai42dataset: shard %s inflated payload exceeds declared raw_bytes %d", shard.name, expectedRaw)
	}
	if written != expectedRaw {
		return fmt.Errorf("ai42dataset: shard %s inflated payload size %d does not equal raw_bytes %d", shard.name, written, expectedRaw)
	}
	// flate may stop at the end of the first stream.  Draining the bounded
	// source detects both trailing compressed bytes and a reader that did not
	// consume the declared payload.
	trailing, err := io.Copy(io.Discard, buffered)
	if err != nil {
		return fmt.Errorf("ai42dataset: read compressed trailing data: %w", err)
	}
	if trailing != 0 || limited.read != remaining {
		return fmt.Errorf("ai42dataset: shard %s compressed payload has trailing or incomplete data", shard.name)
	}
	if info, err := file.Stat(); err != nil {
		return fmt.Errorf("ai42dataset: restat shard %s: %w", shard.name, err)
	} else if !info.Mode().IsRegular() || info.Size() != shard.storedBytes {
		return fmt.Errorf("ai42dataset: shard %s changed while verifying", shard.name)
	}
	payloadHash := hex.EncodeToString(compressedHash.Sum(nil))
	if payloadHash != mustHashField(header["payload_sha256"]) {
		return fmt.Errorf("ai42dataset: shard %s payload hash mismatch", shard.name)
	}
	fileHash := hex.EncodeToString(wholeHash.Sum(nil))
	if fileHash != shard.sha256 {
		return fmt.Errorf("ai42dataset: shard %s file hash mismatch", shard.name)
	}
	if stat, err := temporary.Stat(); err != nil {
		return fmt.Errorf("ai42dataset: stat raw spool: %w", err)
	} else if stat.Size() != expectedRaw {
		return fmt.Errorf("ai42dataset: shard %s raw byte count mismatch", shard.name)
	}
	rawHashHex := hex.EncodeToString(rawHash.Sum(nil))
	if rawHashHex != mustHashField(header["raw_sha256"]) {
		return fmt.Errorf("ai42dataset: shard %s raw payload hash mismatch", shard.name)
	}
	if err := temporary.Close(); err != nil {
		return fmt.Errorf("ai42dataset: close raw spool: %w", err)
	}
	if err := verifyRows(ctx, temporaryName, shard, preamble.descriptors, options, report); err != nil {
		return err
	}
	report.Files = append(report.Files, VerificationFileEvidence{
		Shard: shard.name, StoredBytes: shard.storedBytes, SHA256: fileHash,
		CompressedBytes: remaining, PayloadSHA256: payloadHash,
		RawBytes: expectedRaw, RawSHA256: rawHashHex,
	})
	return nil
}

func verifyRows(ctx context.Context, rawPath string, shard *shardRecord, descriptors []arrayDescriptor, options VerifyOptions, report *VerificationReport) error {
	rawFile, err := os.Open(rawPath)
	if err != nil {
		return fmt.Errorf("ai42dataset: open raw spool: %w", err)
	}
	defer rawFile.Close()
	states := make([]*matchVerification, len(shard.matches))
	for index, match := range shard.matches {
		states[index] = newMatchVerification(match)
	}
	row := newRowData()
	rowBuffers := make([][]byte, len(descriptors))
	for index, descriptor := range descriptors {
		rowBuffers[index] = make([]byte, descriptor.rowBytes)
	}
	for globalRow := 0; globalRow < shard.rowCount; globalRow++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		state := verificationForRow(states, globalRow)
		match := state.match
		resetRow(row, match.metadata.MatchID, globalRow-match.metadata.RowOffset, match.metadata.FirstStep+uint32(globalRow-match.metadata.RowOffset))
		for descriptorIndex, descriptor := range descriptors {
			rowBytes := rowBuffers[descriptorIndex]
			readOffset := descriptor.offset + int64(globalRow)*descriptor.rowBytes
			if _, err := rawFile.ReadAt(rowBytes, readOffset); err != nil {
				return fmt.Errorf("ai42dataset: shard %s array %s row %d: %w", shard.name, descriptor.name, globalRow, err)
			}
			updateColumnHashes(state, descriptor.name, rowBytes)
			if err := decodeRowColumn(row, descriptor.name, rowBytes); err != nil {
				return fmt.Errorf("ai42dataset: shard %s row %d: %w", shard.name, globalRow, err)
			}
		}
		if err := validateDecodedRow(state, row); err != nil {
			return fmt.Errorf("ai42dataset: shard %s row %d: %w", shard.name, globalRow, err)
		}
		if options.OnRow != nil {
			if err := options.OnRow(rowView(row)); err != nil {
				return fmt.Errorf("ai42dataset: shard %s row %d callback: %w", shard.name, globalRow, err)
			}
		}
		report.Rows++
	}
	for _, state := range states {
		match := state.match
		if err := verifyMatchHashes(state); err != nil {
			return err
		}
		if options.OnMatch != nil {
			if err := options.OnMatch(copyMatchMetadata(match.metadata)); err != nil {
				return fmt.Errorf("ai42dataset: match %s callback: %w", match.metadata.MatchID, err)
			}
		}
		report.Matches++
	}
	return nil
}

func copyMatchMetadata(metadata MatchMetadata) MatchMetadata {
	metadata.HeroIDs = append([]string(nil), metadata.HeroIDs...)
	metadata.TrajectoryIDs = append([]string(nil), metadata.TrajectoryIDs...)
	metadata.TrajectoryHashes = append([]Hash(nil), metadata.TrajectoryHashes...)
	metadata.ControllerBySlot = append([]uint8(nil), metadata.ControllerBySlot...)
	metadata.RosterIDs = append([]int32(nil), metadata.RosterIDs...)
	metadata.SideBySlot = append([]uint8(nil), metadata.SideBySlot...)
	return metadata
}

func newMatchVerification(match *matchRecord) *matchVerification {
	return &matchVerification{match: match}
}

func verificationForRow(states []*matchVerification, row int) *matchVerification {
	index := sort.Search(len(states), func(index int) bool { return states[index].match.rowEnd > row })
	return states[index]
}

func newRowData() *rowData {
	return &rowData{
		Invalid: make([]uint8, HeroCount), Rewards: make([]float32, HeroCount), Hero: make([]float32, HeroCount*HeroFeatures),
		Abilities: make([]float32, HeroCount*AbilityCount*AbilityFeatures), Entities: make([]float32, HeroCount*MaxEntities*EntityFeatures),
		Global: make([]float32, HeroCount*GlobalFeatures), EntityMask: make([]uint8, HeroCount*MaxEntities),
		KindMask: make([]uint8, HeroCount*ActionKinds), TargetMask: make([]uint8, HeroCount*MaxEntities),
		SkillTargetMask: make([]uint8, HeroCount*AbilityCount*MaxEntities), TeacherStatus: make([]uint8, HeroCount),
		TeacherAction: make([]Action, HeroCount), ProjectedAction: make([]Action, HeroCount), ExecutedAction: make([]Action, HeroCount),
		ExecutedValid: make([]uint8, HeroCount), RejectionReason: make([]uint8, HeroCount),
	}
}

func resetRow(row *rowData, matchID string, tick int, step uint32) {
	row.MatchID, row.Tick, row.Step = matchID, tick, step
	row.Elapsed, row.Done, row.Winner = 0, 0, 0
	clear(row.Invalid)
	clear(row.Rewards)
	clear(row.Hero)
	clear(row.Abilities)
	clear(row.Entities)
	clear(row.Global)
	clear(row.EntityMask)
	clear(row.KindMask)
	clear(row.TargetMask)
	clear(row.SkillTargetMask)
	clear(row.TeacherStatus)
	clear(row.TeacherAction)
	clear(row.ProjectedAction)
	clear(row.ExecutedAction)
	clear(row.ExecutedValid)
	clear(row.RejectionReason)
}

func rowView(row *rowData) Row {
	return Row{MatchID: row.MatchID, Tick: row.Tick, Step: row.Step, Elapsed: row.Elapsed, Done: row.Done, Winner: row.Winner,
		Invalid: row.Invalid, Rewards: row.Rewards, Hero: row.Hero, Abilities: row.Abilities, Entities: row.Entities, Global: row.Global,
		EntityMask: row.EntityMask, KindMask: row.KindMask, TargetMask: row.TargetMask, SkillTargetMask: row.SkillTargetMask,
		TeacherStatus: row.TeacherStatus, TeacherAction: row.TeacherAction, ProjectedAction: row.ProjectedAction,
		ExecutedAction: row.ExecutedAction, ExecutedValid: row.ExecutedValid, RejectionReason: row.RejectionReason}
}

func decodeRowColumn(row *rowData, name string, data []byte) error {
	readF32 := func(values []float32) error {
		if len(data) != len(values)*4 {
			return fmt.Errorf("array %s row byte count mismatch", name)
		}
		for index := range values {
			values[index] = math.Float32frombits(binary.LittleEndian.Uint32(data[index*4:]))
		}
		return nil
	}
	readU8 := func(values []uint8) error {
		if len(data) != len(values) {
			return fmt.Errorf("array %s row byte count mismatch", name)
		}
		copy(values, data)
		return nil
	}
	readActions := func(values []Action) error {
		if len(data) != len(values)*5 {
			return fmt.Errorf("array %s row byte count mismatch", name)
		}
		for index := range values {
			base := index * 5
			values[index] = Action{Kind: data[base], Target: binary.LittleEndian.Uint16(data[base+1 : base+3]), Direction: data[base+3], Distance: data[base+4]}
		}
		return nil
	}
	readI32 := func() error {
		if len(data) != 4 {
			return fmt.Errorf("array %s row byte count mismatch", name)
		}
		row.Winner = int32(binary.LittleEndian.Uint32(data))
		return nil
	}
	switch name {
	case "hero":
		return readF32(row.Hero)
	case "abilities":
		return readF32(row.Abilities)
	case "entities":
		return readF32(row.Entities)
	case "global":
		return readF32(row.Global)
	case "entity_mask":
		return readU8(row.EntityMask)
	case "kind_mask":
		return readU8(row.KindMask)
	case "target_mask":
		return readU8(row.TargetMask)
	case "skill_target_mask":
		return readU8(row.SkillTargetMask)
	case "teacher_status":
		return readU8(row.TeacherStatus)
	case "teacher_action":
		return readActions(row.TeacherAction)
	case "projected_action":
		return readActions(row.ProjectedAction)
	case "executed_action":
		return readActions(row.ExecutedAction)
	case "executed_valid":
		return readU8(row.ExecutedValid)
	case "rejection_reason":
		return readU8(row.RejectionReason)
	case "rewards":
		return readF32(row.Rewards)
	case "done":
		if len(data) != 1 {
			return fmt.Errorf("array %s row byte count mismatch", name)
		}
		row.Done = data[0]
		return nil
	case "winner":
		return readI32()
	case "step":
		if len(data) != 4 {
			return fmt.Errorf("array %s row byte count mismatch", name)
		}
		row.Step = binary.LittleEndian.Uint32(data)
		return nil
	case "elapsed":
		if len(data) != 4 {
			return fmt.Errorf("array %s row byte count mismatch", name)
		}
		row.Elapsed = math.Float32frombits(binary.LittleEndian.Uint32(data))
		return nil
	case "invalid":
		return readU8(row.Invalid)
	default:
		return fmt.Errorf("unknown array %q", name)
	}
}

func validateDecodedRow(state *matchVerification, row *rowData) error {
	match := state.match
	if row.Step != match.metadata.FirstStep+uint32(row.Tick) {
		return fmt.Errorf("field=step tick=%d: is not contiguous", row.Tick)
	}
	if row.Done > 1 || (row.Done != 0 && row.Tick != match.metadata.TickCount-1) {
		return fmt.Errorf("field=done tick=%d: terminal tick must be final", row.Tick)
	}
	if row.Tick == match.metadata.TickCount-1 && row.Done != 1 {
		return fmt.Errorf("field=done tick=%d: must contain the terminal tick", row.Tick)
	}
	if !finite32(row.Elapsed) {
		return fmt.Errorf("field=elapsed tick=%d: must be finite", row.Tick)
	}
	for slot := 0; slot < HeroCount; slot++ {
		index := slot
		if row.Invalid[index] > 1 || row.ExecutedValid[index] > 1 {
			return fmt.Errorf("field=invalid tick=%d slot=%d: must be zero or one", row.Tick, slot)
		}
		if !finite32(row.Rewards[index]) {
			return fmt.Errorf("field=rewards tick=%d slot=%d: must be finite", row.Tick, slot)
		}
		for element, value := range row.Hero[slot*HeroFeatures : (slot+1)*HeroFeatures] {
			if !finite32(value) {
				return fmt.Errorf("field=hero tick=%d slot=%d: non-finite index %d", row.Tick, slot, element)
			}
		}
		for element, value := range row.Abilities[slot*AbilityCount*AbilityFeatures : (slot+1)*AbilityCount*AbilityFeatures] {
			if !finite32(value) {
				return fmt.Errorf("field=abilities tick=%d slot=%d: non-finite index %d", row.Tick, slot, element)
			}
		}
		for element, value := range row.Entities[slot*MaxEntities*EntityFeatures : (slot+1)*MaxEntities*EntityFeatures] {
			if !finite32(value) {
				return fmt.Errorf("field=entities tick=%d slot=%d: non-finite index %d", row.Tick, slot, element)
			}
		}
		for element, value := range row.Global[slot*GlobalFeatures : (slot+1)*GlobalFeatures] {
			if !finite32(value) {
				return fmt.Errorf("field=global tick=%d slot=%d: non-finite index %d", row.Tick, slot, element)
			}
		}
		for field, values := range map[string][]uint8{"entity_mask": row.EntityMask[slot*MaxEntities : (slot+1)*MaxEntities], "kind_mask": row.KindMask[slot*ActionKinds : (slot+1)*ActionKinds], "target_mask": row.TargetMask[slot*MaxEntities : (slot+1)*MaxEntities], "skill_target_mask": row.SkillTargetMask[slot*AbilityCount*MaxEntities : (slot+1)*AbilityCount*MaxEntities]} {
			for element, value := range values {
				if value > 1 {
					return fmt.Errorf("field=%s tick=%d slot=%d: value at index %d must be zero or one", field, row.Tick, slot, element)
				}
			}
		}
		for field, action := range map[string]Action{"teacher_action": row.TeacherAction[slot], "projected_action": row.ProjectedAction[slot], "executed_action": row.ExecutedAction[slot]} {
			if err := validateAction(action, field, row.Tick, slot, true); err != nil {
				return err
			}
		}
		status := row.TeacherStatus[slot]
		if status > battleserver.AssaultTeacherStatusUnavailable {
			return fmt.Errorf("field=teacher_status tick=%d slot=%d: unknown v13 status %d", row.Tick, slot, status)
		}
		if status == battleserver.AssaultTeacherStatusAction && row.TeacherAction[slot].Kind == 0 {
			return fmt.Errorf("field=teacher_action tick=%d slot=%d: action status cannot carry wait", row.Tick, slot)
		}
		if status != battleserver.AssaultTeacherStatusAction && !row.TeacherAction[slot].isZero() {
			return fmt.Errorf("field=teacher_action tick=%d slot=%d: control status must carry zero action", row.Tick, slot)
		}
		if !validRejectionReason(row.RejectionReason[slot]) {
			return fmt.Errorf("field=rejection_reason tick=%d slot=%d: unknown v13 code %d", row.Tick, slot, row.RejectionReason[slot])
		}
		if row.ExecutedValid[slot] == 1 && row.RejectionReason[slot] != battleserver.AssaultRejectionReasonNone {
			return fmt.Errorf("field=rejection_reason tick=%d slot=%d: accepted action must have reason none", row.Tick, slot)
		}
		if row.ExecutedValid[slot] == 0 && row.RejectionReason[slot] == battleserver.AssaultRejectionReasonNone {
			return fmt.Errorf("field=rejection_reason tick=%d slot=%d: rejected action must have a reason", row.Tick, slot)
		}
		if row.ExecutedValid[slot] == 0 && !row.ExecutedAction[slot].isZero() {
			return fmt.Errorf("field=executed_action tick=%d slot=%d: rejected action must be zero", row.Tick, slot)
		}
		if err := validateDerivedLineage(state, row.Tick, slot, status); err != nil {
			return err
		}
	}
	return nil
}

func validateDerivedLineage(state *matchVerification, tick, slot int, status uint8) error {
	match := state.match
	if state.lineageRoots[slot] == nil {
		state.lineageRoots[slot] = make(map[string]string, match.metadata.TickCount)
	}
	if state.lineageCancelled[slot] == nil {
		state.lineageCancelled[slot] = make(map[string]struct{}, match.metadata.TickCount)
	}
	parent := derivedParentID(match.metadata.MatchID, tick, slot)
	boundary := derivedBoundaryID(match.metadata.MatchID, tick, slot)
	if parent == boundary {
		return fmt.Errorf("field=recurrent_boundary_id tick=%d slot=%d: must differ from parent", tick, slot)
	}
	if _, duplicate := state.lineageRoots[slot][boundary]; duplicate {
		return fmt.Errorf("field=recurrent_boundary_id tick=%d slot=%d: duplicate boundary", tick, slot)
	}
	if tick > 0 && state.lastStatus[slot] == battleserver.AssaultTeacherStatusWait && status == battleserver.AssaultTeacherStatusHold {
		return fmt.Errorf("field=teacher_status tick=%d slot=%d: HOLD does not reference a holdable lineage", tick, slot)
	}
	if tick > 0 && state.lastStatus[slot] == battleserver.AssaultTeacherStatusCancel && status == battleserver.AssaultTeacherStatusHold {
		return fmt.Errorf("field=teacher_status tick=%d slot=%d: HOLD does not reference a holdable lineage", tick, slot)
	}
	switch status {
	case battleserver.AssaultTeacherStatusHold:
		if tick == 0 {
			return fmt.Errorf("field=teacher_status tick=%d slot=%d: HOLD does not reference a holdable lineage", tick, slot)
		}
		root, ok := state.lineageRoots[slot][parent]
		if !ok {
			return fmt.Errorf("field=recurrent_parent_id tick=%d slot=%d: HOLD parent lineage is unknown", tick, slot)
		}
		state.lineageRoots[slot][boundary] = root
	case battleserver.AssaultTeacherStatusCancel:
		if tick == 0 || state.lastStatus[slot] == battleserver.AssaultTeacherStatusWait || state.lastStatus[slot] == battleserver.AssaultTeacherStatusCancel {
			return fmt.Errorf("field=teacher_status tick=%d slot=%d: CANCEL does not reference a cancellable lineage", tick, slot)
		}
		root, ok := state.lineageRoots[slot][parent]
		if !ok {
			return fmt.Errorf("field=recurrent_parent_id tick=%d slot=%d: CANCEL parent lineage is unknown", tick, slot)
		}
		if _, already := state.lineageCancelled[slot][root]; already {
			return fmt.Errorf("field=teacher_status tick=%d slot=%d: CANCEL references an already cancelled lineage", tick, slot)
		}
		state.lineageCancelled[slot][root] = struct{}{}
		state.lineageRoots[slot][boundary] = boundary
	default:
		state.lineageRoots[slot][boundary] = boundary
	}
	state.lastStatus[slot] = status
	return nil
}

func derivedParentID(matchID string, tick, slot int) string {
	if tick == 0 {
		return fmt.Sprintf("%s:root:%02d", matchID, slot)
	}
	return fmt.Sprintf("%s:boundary:%d:%02d", matchID, tick-1, slot)
}
func derivedBoundaryID(matchID string, tick, slot int) string {
	return fmt.Sprintf("%s:boundary:%d:%02d", matchID, tick, slot)
}

func updateColumnHashes(state *matchVerification, name string, row []byte) {
	if len(state.columnHashes[0]) == 0 {
		for hero := 0; hero < HeroCount; hero++ {
			state.columnHashes[hero] = make([]hash.Hash, len(arrayNames))
			for index := range arrayNames {
				state.columnHashes[hero][index] = sha256.New()
			}
		}
	}
	column := 0
	for index, value := range arrayNames {
		if value == name {
			column = index
			break
		}
	}
	width := len(row) / HeroCount
	if name == "done" || name == "winner" || name == "step" || name == "elapsed" {
		for hero := 0; hero < HeroCount; hero++ {
			_, _ = state.columnHashes[hero][column].Write(row)
		}
		return
	}
	for hero := 0; hero < HeroCount; hero++ {
		start := hero * width
		_, _ = state.columnHashes[hero][column].Write(row[start : start+width])
	}
}

func verifyMatchHashes(state *matchVerification) error {
	match := state.match
	for hero := 0; hero < HeroCount; hero++ {
		evidence := map[string]any{"match_id": match.metadata.MatchID, "hero_id": match.metadata.HeroIDs[hero], "steps": make([]uint32, match.metadata.TickCount), "parents": make([]string, match.metadata.TickCount), "boundaries": make([]string, match.metadata.TickCount)}
		steps := evidence["steps"].([]uint32)
		parents := evidence["parents"].([]string)
		boundaries := evidence["boundaries"].([]string)
		for tick := 0; tick < match.metadata.TickCount; tick++ {
			steps[tick] = match.metadata.FirstStep + uint32(tick)
			parents[tick] = derivedParentID(match.metadata.MatchID, tick, hero)
			boundaries[tick] = derivedBoundaryID(match.metadata.MatchID, tick, hero)
		}
		digest := sha256.New()
		_, _ = digest.Write([]byte("AI42-fast-trajectory-v1\x00"))
		_, _ = digest.Write(canonicalJSON(evidence))
		for index, name := range arrayNames {
			_, _ = digest.Write([]byte(name))
			sum := state.columnHashes[hero][index].Sum(nil)
			_, _ = digest.Write(sum)
		}
		got := hex.EncodeToString(digest.Sum(nil))
		if got != hashHex(match.metadata.TrajectoryHashes[hero]) {
			return fmt.Errorf("ai42dataset: match %s trajectory hash mismatch for hero %d", match.metadata.MatchID, hero)
		}
	}
	return nil
}

func decodeCanonicalJSON(payload []byte, path string) (any, error) {
	if err := rejectDuplicateJSONKeys(payload); err != nil {
		return nil, fmt.Errorf("ai42dataset: %s: %w", path, err)
	}
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, fmt.Errorf("ai42dataset: %s: invalid JSON: %w", path, err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return nil, fmt.Errorf("ai42dataset: %s: trailing JSON value", path)
		}
		return nil, fmt.Errorf("ai42dataset: %s: trailing data: %w", path, err)
	}
	if !bytes.Equal(canonicalJSON(value), payload) {
		return nil, fmt.Errorf("ai42dataset: %s: JSON is not canonical", path)
	}
	return value, nil
}

func rejectDuplicateJSONKeys(payload []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	var walk func() error
	walk = func() error {
		token, err := decoder.Token()
		if err != nil {
			return err
		}
		delim, ok := token.(json.Delim)
		if !ok {
			return nil
		}
		switch delim {
		case '{':
			seen := map[string]struct{}{}
			for decoder.More() {
				key, err := decoder.Token()
				if err != nil {
					return err
				}
				keyString, ok := key.(string)
				if !ok {
					return errors.New("object key is not a string")
				}
				if _, exists := seen[keyString]; exists {
					return fmt.Errorf("duplicate object key %q", keyString)
				}
				seen[keyString] = struct{}{}
				if err := walk(); err != nil {
					return err
				}
			}
		case '[':
			for decoder.More() {
				if err := walk(); err != nil {
					return err
				}
			}
		default:
			return nil
		}
		_, err = decoder.Token()
		return err
	}
	if err := walk(); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return errors.New("multiple JSON values")
		}
		return err
	}
	return nil
}

func requireFields(object map[string]any, expected map[string]struct{}, path string) error {
	for key := range object {
		if _, ok := expected[key]; !ok {
			return fmt.Errorf("ai42dataset: %s unknown field %q", path, key)
		}
	}
	if len(object) != len(expected) {
		return fmt.Errorf("ai42dataset: %s field set mismatch", path)
	}
	return nil
}

func nonEmptyString(value any, path string) (string, error) {
	text, ok := value.(string)
	if !ok || text == "" || !utf8.ValidString(text) {
		return "", fmt.Errorf("ai42dataset: %s must be a non-empty UTF-8 string", path)
	}
	return text, nil
}
func normalizeHash(value, path string) (string, error) {
	text, err := nonEmptyString(value, path)
	if err != nil {
		return "", err
	}
	if len(text) != sha256.Size*2 || strings.ToLower(text) != text {
		return "", fmt.Errorf("ai42dataset: %s must be lowercase SHA-256 hex", path)
	}
	raw, err := hex.DecodeString(text)
	if err != nil || len(raw) != sha256.Size {
		return "", fmt.Errorf("ai42dataset: %s must be lowercase SHA-256 hex", path)
	}
	return text, nil
}
func hashField(value any, path string) (string, error) {
	text, err := nonEmptyString(value, path)
	if err != nil {
		return "", err
	}
	return normalizeHash(text, path)
}
func hashArray(value any, path string) ([]Hash, error) {
	values, ok := value.([]any)
	if !ok || len(values) != HeroCount {
		return nil, fmt.Errorf("ai42dataset: %s must contain ten hashes", path)
	}
	result := make([]Hash, HeroCount)
	seen := map[string]struct{}{}
	for index, item := range values {
		textValue, err := nonEmptyString(item, fmt.Sprintf("%s[%d]", path, index))
		if err != nil {
			return nil, err
		}
		text, err := normalizeHash(textValue, fmt.Sprintf("%s[%d]", path, index))
		if err != nil {
			return nil, err
		}
		if _, exists := seen[text]; exists {
			return nil, fmt.Errorf("ai42dataset: %s contains duplicate hash", path)
		}
		seen[text] = struct{}{}
		raw, _ := hex.DecodeString(text)
		copy(result[index][:], raw)
	}
	return result, nil
}
func stringArray(value any, length int, path string, unique bool) ([]string, error) {
	values, ok := value.([]any)
	if !ok || len(values) != length {
		return nil, fmt.Errorf("ai42dataset: %s must contain %d values", path, length)
	}
	result := make([]string, length)
	seen := map[string]struct{}{}
	for index, item := range values {
		text, err := nonEmptyString(item, fmt.Sprintf("%s[%d]", path, index))
		if err != nil {
			return nil, err
		}
		if unique {
			if _, exists := seen[text]; exists {
				return nil, fmt.Errorf("ai42dataset: %s contains duplicate value", path)
			}
			seen[text] = struct{}{}
		}
		result[index] = text
	}
	return result, nil
}
func uint8Array(value any, length int, path string) ([]uint8, error) {
	values, ok := value.([]any)
	if !ok || len(values) != length {
		return nil, fmt.Errorf("ai42dataset: %s must contain %d values", path, length)
	}
	result := make([]uint8, length)
	for index, item := range values {
		number, err := intField(item, fmt.Sprintf("%s[%d]", path, index), 0)
		if err != nil || number > math.MaxUint8 {
			if err != nil {
				return nil, err
			}
			return nil, fmt.Errorf("ai42dataset: %s[%d] is outside uint8", path, index)
		}
		result[index] = uint8(number)
	}
	return result, nil
}
func int32Array(value any, length int, path string) ([]int32, error) {
	values, ok := value.([]any)
	if !ok || len(values) != length {
		return nil, fmt.Errorf("ai42dataset: %s must contain %d values", path, length)
	}
	result := make([]int32, length)
	seen := map[int32]struct{}{}
	for index, item := range values {
		number, err := int64Field(item, fmt.Sprintf("%s[%d]", path, index), math.MinInt32)
		if err != nil || number > math.MaxInt32 {
			if err != nil {
				return nil, err
			}
			return nil, fmt.Errorf("ai42dataset: %s[%d] is outside int32", path, index)
		}
		result[index] = int32(number)
		if _, exists := seen[result[index]]; exists {
			return nil, fmt.Errorf("ai42dataset: %s contains duplicate value", path)
		}
		seen[result[index]] = struct{}{}
	}
	return result, nil
}
func shapeArray(value any, path string) ([]int, error) {
	values, ok := value.([]any)
	if !ok {
		return nil, fmt.Errorf("ai42dataset: %s must be an integer array", path)
	}
	if len(values) < 1 || len(values) > maxArrayDimensions {
		return nil, fmt.Errorf("ai42dataset: %s dimension count %d is outside [1,%d]", path, len(values), maxArrayDimensions)
	}
	result := make([]int, len(values))
	for index, item := range values {
		number, err := intField(item, fmt.Sprintf("%s[%d]", path, index), 0)
		if err != nil || number > int64(math.MaxInt) {
			if err != nil {
				return nil, err
			}
			return nil, fmt.Errorf("ai42dataset: %s[%d] is too large", path, index)
		}
		result[index] = int(number)
	}
	return result, nil
}
func equalInts(left, right []int) bool {
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
func equalStrings(left, right []string) bool {
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
func shapeProduct(shape []int) (int64, bool) {
	product := int64(1)
	for _, value := range shape {
		if value < 0 || int64(value) > math.MaxInt64/product {
			return 0, false
		}
		product *= int64(value)
	}
	return product, true
}
func addCappedTotal(total, value, limit int64, path string) (int64, error) {
	if total < 0 || value < 0 || total > limit || value > limit-total {
		return 0, fmt.Errorf("ai42dataset: %s exceed cumulative limit %d", path, limit)
	}
	return total + value, nil
}
func intField(value any, path string, minimum int64) (int64, error) {
	number, ok := value.(json.Number)
	if !ok {
		return 0, fmt.Errorf("ai42dataset: %s must be an integer", path)
	}
	parsed, err := strconv.ParseInt(string(number), 10, 64)
	if err != nil || parsed < minimum {
		return 0, fmt.Errorf("ai42dataset: %s must be an integer >= %d", path, minimum)
	}
	return parsed, nil
}
func intFromField(value any, path string, minimum int64) (int, error) {
	number, err := intField(value, path, minimum)
	if err != nil {
		return 0, err
	}
	if number > int64(^uint(0)>>1) {
		return 0, fmt.Errorf("ai42dataset: %s is too large for int", path)
	}
	return int(number), nil
}
func int64Field(value any, path string, minimum int64) (int64, error) {
	return intField(value, path, minimum)
}
func uint32Field(value any, path string) (uint32, error) {
	number, err := intField(value, path, 0)
	if err != nil || number > math.MaxUint32 {
		if err != nil {
			return 0, err
		}
		return 0, fmt.Errorf("ai42dataset: %s is outside uint32", path)
	}
	return uint32(number), nil
}
func floatField(value any, path string) (float64, error) {
	number, ok := value.(json.Number)
	if !ok {
		return 0, fmt.Errorf("ai42dataset: %s must be a number", path)
	}
	parsed, err := strconv.ParseFloat(string(number), 64)
	if err != nil || math.IsNaN(parsed) || math.IsInf(parsed, 0) {
		return 0, fmt.Errorf("ai42dataset: %s must be finite", path)
	}
	return parsed, nil
}
func mustHashField(value any) string { text, _ := hashField(value, "hash"); return text }
func mustInt64Field(value any) int64 { number, _ := int64Field(value, "integer", 0); return number }
func digestFile(path string, limit int64) (int64, string, error) {
	file, err := os.Open(path)
	if err != nil {
		return 0, "", err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return 0, "", err
	}
	if !info.Mode().IsRegular() || info.Size() < 1 || info.Size() > limit {
		return 0, "", fmt.Errorf("file size %d is outside [1,%d]", info.Size(), limit)
	}
	digest := sha256.New()
	size, err := io.CopyBuffer(digest, &io.LimitedReader{R: file, N: limit + 1}, make([]byte, 128*1024))
	if err != nil {
		return 0, "", err
	}
	if size != info.Size() || size > limit {
		return 0, "", fmt.Errorf("file changed while hashing or exceeds %d bytes", limit)
	}
	return size, hex.EncodeToString(digest.Sum(nil)), nil
}

type countingReader struct {
	reader io.Reader
	read   int64
}

func (r *countingReader) Read(data []byte) (int, error) {
	count, err := r.reader.Read(data)
	r.read += int64(count)
	return count, err
}

type contextReader struct {
	ctx    context.Context
	reader io.Reader
}

func (r *contextReader) Read(data []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	count, err := r.reader.Read(data)
	if err == nil {
		if contextErr := r.ctx.Err(); contextErr != nil {
			return count, contextErr
		}
	}
	return count, err
}
