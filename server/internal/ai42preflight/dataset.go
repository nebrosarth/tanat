package ai42preflight

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"sort"
	"sync"

	"tanatserver/internal/ai42dataset"
)

type matchInfo struct {
	ID        string
	Split     string
	Scenario  string
	TickCount int
	FirstStep uint32
}

type matchEvidence struct {
	Counts             map[string][]int
	SupervisedByWindow []uint16
}

type profileAccumulator struct {
	mu             sync.Mutex
	Matches        []matchInfo
	ByID           map[string]matchInfo
	Evidence       map[string]*matchEvidence
	SequenceLength int
}

func newAccumulator(sequenceLength int) *profileAccumulator {
	return &profileAccumulator{ByID: map[string]matchInfo{}, Evidence: map[string]*matchEvidence{}, SequenceLength: sequenceLength}
}

func (a *profileAccumulator) onMatch(value ai42dataset.MatchMetadata) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if value.MatchID == "" || (value.Split != "train" && value.Split != "validation") {
		return fmt.Errorf("dataset match %q has an invalid split", value.MatchID)
	}
	info := matchInfo{ID: value.MatchID, Split: value.Split, Scenario: value.Scenario, TickCount: value.TickCount, FirstStep: value.FirstStep}
	if _, exists := a.ByID[info.ID]; exists {
		return fmt.Errorf("dataset match %q was reported twice", info.ID)
	}
	a.ByID[info.ID] = info
	a.Matches = append(a.Matches, info)
	evidence := a.Evidence[info.ID]
	expectedWindows := (info.TickCount + a.SequenceLength - 1) / a.SequenceLength
	if evidence == nil || len(evidence.SupervisedByWindow) != expectedWindows {
		return fmt.Errorf("dataset match %q produced incomplete window evidence", info.ID)
	}
	return nil
}

func (a *profileAccumulator) onRow(row ai42dataset.Row) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if row.MatchID == "" {
		return fmt.Errorf("dataset row has no match ID")
	}
	evidence := a.Evidence[row.MatchID]
	if evidence == nil {
		evidence = &matchEvidence{Counts: zeroCounts()}
		a.Evidence[row.MatchID] = evidence
	}
	if len(row.TeacherStatus) != ai42dataset.HeroCount || len(row.TeacherAction) != ai42dataset.HeroCount {
		return fmt.Errorf("dataset row %q has invalid action columns", row.MatchID)
	}
	window := row.Tick / a.SequenceLength
	for len(evidence.SupervisedByWindow) <= window {
		evidence.SupervisedByWindow = append(evidence.SupervisedByWindow, 0)
	}
	for hero := 0; hero < ai42dataset.HeroCount; hero++ {
		status := row.TeacherStatus[hero]
		if status >= 1 && status <= 4 {
			evidence.SupervisedByWindow[window] |= uint16(1) << hero
		}
		if err := addRowCounts(evidence.Counts, row, hero); err != nil {
			return fmt.Errorf("dataset row %q hero %d: %w", row.MatchID, hero, err)
		}
	}
	return nil
}

func (a *profileAccumulator) profile(datasetHash string) (Profile, error) {
	trainIDs := make([]string, 0)
	counts := zeroCounts()
	for _, info := range a.Matches {
		if info.Split != "train" {
			continue
		}
		trainIDs = append(trainIDs, info.ID)
		evidence := a.Evidence[info.ID]
		if evidence == nil {
			return Profile{}, fmt.Errorf("train match %q produced no rows", info.ID)
		}
		for _, head := range profileHeads {
			for index, value := range evidence.Counts[head] {
				counts[head][index] += value
			}
		}
	}
	return buildProfile(datasetHash, trainIDs, counts)
}

func verifyDataset(ctx context.Context, generation *ai42dataset.Generation, accumulator *profileAccumulator) (ai42dataset.VerificationReport, int, error) {
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	if workers > 16 {
		workers = 16
	}
	report, err := generation.Verify(ctx, ai42dataset.VerifyOptions{OnMatch: accumulator.onMatch, OnRow: accumulator.onRow, Workers: workers})
	if err != nil {
		return report, workers, err
	}
	if err := accumulator.restoreManifestOrder(generation.Matches()); err != nil {
		return report, workers, err
	}
	return report, workers, nil
}

func (a *profileAccumulator) restoreManifestOrder(metadata []ai42dataset.MatchMetadata) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(metadata) != len(a.ByID) {
		return fmt.Errorf("verified match callback coverage is incomplete")
	}
	ordered := make([]matchInfo, len(metadata))
	for index, match := range metadata {
		info, ok := a.ByID[match.MatchID]
		if !ok {
			return fmt.Errorf("verified match callback omitted %q", match.MatchID)
		}
		ordered[index] = info
	}
	a.Matches = ordered
	return nil
}

func zeroCounts() map[string][]int {
	return map[string][]int{"control": make([]int, 4), "kind": make([]int, 8), "target": make([]int, 96), "offset": make([]int, 81), "anchor": make([]int, 15)}
}

func addRowCounts(counts map[string][]int, row ai42dataset.Row, hero int) error {
	status := row.TeacherStatus[hero]
	control, controlOK := map[uint8]int{1: 0, 2: 1, 3: 2, 4: 3}[status]
	if controlOK {
		counts["control"][control]++
	}
	if status != 1 {
		return nil
	}
	action := row.TeacherAction[hero]
	kind := int(action.Kind)
	if kind < 0 || kind >= len(counts["kind"]) {
		return fmt.Errorf("kind label %d is outside the model vocabulary", kind)
	}
	counts["kind"][kind]++
	entityMask := row.EntityMask[hero*ai42dataset.MaxEntities : (hero+1)*ai42dataset.MaxEntities]
	anyTarget := false
	if kind == 2 {
		for index, value := range row.TargetMask[hero*ai42dataset.MaxEntities : (hero+1)*ai42dataset.MaxEntities] {
			if value != 0 && entityMask[index] != 0 {
				anyTarget = true
				break
			}
		}
	} else if kind >= 3 && kind <= 6 {
		base := (hero*4 + kind - 3) * ai42dataset.MaxEntities
		for index, value := range row.SkillTargetMask[base : base+ai42dataset.MaxEntities] {
			if value != 0 && entityMask[index] != 0 {
				anyTarget = true
				break
			}
		}
	}
	if kind == 2 || (kind >= 3 && kind <= 6 && anyTarget) {
		target := int(action.Target)
		if target < 0 || target >= len(counts["target"]) {
			return fmt.Errorf("target label %d is outside the model vocabulary", target)
		}
		counts["target"][target]++
	}
	if kind >= 3 && kind <= 6 || (kind == 1 && action.Distance == 0) {
		offset := int(action.Direction)
		if offset >= len(counts["offset"]) {
			return fmt.Errorf("offset label %d is outside the model vocabulary", offset)
		}
		counts["offset"][offset]++
	}
	if kind == 1 && action.Distance > 0 {
		anchor := int(action.Distance)
		if anchor >= len(counts["anchor"]) {
			return fmt.Errorf("anchor label %d is outside the model vocabulary", anchor)
		}
		counts["anchor"][anchor]++
	}
	return nil
}

type batchSequence struct {
	MatchID string
	Hero    int
	Start   int
	Stop    int
	Step    uint32
}

type batchPlan struct {
	Version            string
	Seed               uint32
	SequenceLength     int
	BatchSize          int
	TrainMatchIDs      []string
	ValidationMatchIDs []string
	TrainProbeMatchIDs []string
	Hash               string
	ValidationHash     string
	SplitHash          string
	Sequences          []batchSequence
	PhysicalBatches    int
	EmptyBatches       int
	SequencesScanned   int
}

func makeBatchPlan(matches []matchInfo, evidence map[string]*matchEvidence, config Config) (batchPlan, error) {
	train, validation := make([]matchInfo, 0), make([]matchInfo, 0)
	for _, info := range matches {
		if info.Split == "train" {
			train = append(train, info)
		} else if info.Split == "validation" {
			validation = append(validation, info)
		}
	}
	if len(train) == 0 {
		return batchPlan{}, fmt.Errorf("validated dataset has no train matches")
	}
	trainIDs := rankedIDs(train, "train", config.Seed, 0)
	validationIDs := rankedIDs(validation, "validation", config.Seed, config.ValidationProbeLimit)
	payload := map[string]any{"version": BatchPlanVersion, "seed": int(config.Seed), "sequence_length": config.SequenceLength, "batch_size": config.BatchSize, "train_match_ids": stringsAny(trainIDs), "validation_match_ids": stringsAny(validationIDs)}
	hash, err := canonicalHash(payload)
	if err != nil {
		return batchPlan{}, err
	}
	splitHash, err := canonicalHash(map[string]any{"train": stringsAny(extractIDs(train)), "validation": stringsAny(extractIDs(validation))})
	if err != nil {
		return batchPlan{}, err
	}
	validationHash, err := canonicalHash(map[string]any{"version": BatchPlanVersion, "seed": int(config.Seed), "match_ids": stringsAny(validationIDs)})
	if err != nil {
		return batchPlan{}, err
	}
	sequences := make([]batchSequence, 0, config.BatchSize)
	selected := make([]batchSequence, 0, config.BatchSize)
	physicalBatches, emptyBatches, sequencesScanned := 0, 0, 0
	batchEligible := false
	finishBatch := func() bool {
		if len(sequences) == 0 {
			return false
		}
		physicalBatches++
		if batchEligible {
			selected = append(selected[:0], sequences...)
			return true
		}
		emptyBatches++
		sequences = sequences[:0]
		batchEligible = false
		return false
	}
	found := false
	for _, id := range trainIDs {
		info := matchByID(matches, id)
		matchEvidence := evidence[id]
		if matchEvidence == nil {
			return batchPlan{}, fmt.Errorf("train match %q has no compact eligibility evidence", id)
		}
		windows := (info.TickCount + config.SequenceLength - 1) / config.SequenceLength
		for hero := 0; hero < ai42dataset.HeroCount; hero++ {
			for window := 0; window < windows; window++ {
				if window >= len(matchEvidence.SupervisedByWindow) {
					return batchPlan{}, fmt.Errorf("train match %q has incomplete compact eligibility evidence", id)
				}
				start := window * config.SequenceLength
				stop := start + config.SequenceLength
				if stop > info.TickCount {
					stop = info.TickCount
				}
				sequences = append(sequences, batchSequence{MatchID: id, Hero: hero, Start: start, Stop: stop, Step: info.FirstStep + uint32(start)})
				sequencesScanned++
				batchEligible = batchEligible || matchEvidence.SupervisedByWindow[window]&(uint16(1)<<hero) != 0
				if len(sequences) == config.BatchSize && finishBatch() {
					found = true
					break
				}
			}
			if found {
				break
			}
		}
		if found {
			break
		}
	}
	if !found && finishBatch() {
		found = true
	}
	if !found {
		return batchPlan{}, fmt.Errorf("validated dataset contains no eligible recurrent train batch")
	}
	return batchPlan{Version: BatchPlanVersion, Seed: config.Seed, SequenceLength: config.SequenceLength, BatchSize: config.BatchSize, TrainMatchIDs: trainIDs, ValidationMatchIDs: validationIDs, TrainProbeMatchIDs: trainIDs[:1], Hash: hash, ValidationHash: validationHash, SplitHash: splitHash, Sequences: selected, PhysicalBatches: physicalBatches, EmptyBatches: emptyBatches, SequencesScanned: sequencesScanned}, nil
}

func rankedIDs(matches []matchInfo, split string, seed uint32, limit int) []string {
	type ranked struct{ id, scenario, hash string }
	groups := map[string][]ranked{}
	for _, info := range matches {
		if info.Split != split {
			continue
		}
		scenario := info.Scenario
		if scenario == "" {
			scenario = "default"
		}
		payload := fmt.Sprintf("%s\x00%d\x00%s\x00%s\x00%s", BatchPlanVersion, seed, split, scenario, info.ID)
		digest := sha256.Sum256([]byte(payload))
		groups[scenario] = append(groups[scenario], ranked{id: info.ID, scenario: scenario, hash: fmt.Sprintf("%x", digest[:])})
	}
	scenarios := make([]string, 0, len(groups))
	for scenario := range groups {
		scenarios = append(scenarios, scenario)
	}
	sort.Strings(scenarios)
	for _, scenario := range scenarios {
		sort.Slice(groups[scenario], func(i, j int) bool {
			if groups[scenario][i].hash == groups[scenario][j].hash {
				return groups[scenario][i].id < groups[scenario][j].id
			}
			return groups[scenario][i].hash < groups[scenario][j].hash
		})
	}
	result := make([]string, 0, len(matches))
	for len(groups) > 0 && (limit == 0 || len(result) < limit) {
		for _, scenario := range scenarios {
			values := groups[scenario]
			if len(values) > 0 {
				result = append(result, values[0].id)
				groups[scenario] = values[1:]
				if limit > 0 && len(result) >= limit {
					break
				}
			}
			if len(groups[scenario]) == 0 {
				delete(groups, scenario)
			}
		}
		if len(groups) == 0 {
			break
		}
	}
	return result
}

func matchByID(matches []matchInfo, id string) matchInfo {
	for _, info := range matches {
		if info.ID == id {
			return info
		}
	}
	return matchInfo{}
}

func extractIDs(matches []matchInfo) []string {
	result := make([]string, len(matches))
	for index, value := range matches {
		result[index] = value.ID
	}
	return result
}
func stringsAny(values []string) []any {
	result := make([]any, len(values))
	for index, value := range values {
		result[index] = value
	}
	return result
}

type capturedRows struct {
	ByMatch map[string]map[int]ai42dataset.Row
	Ranges  map[string][][2]int
	Rows    int
}

func captureBatch(ctx context.Context, source TargetedRowReader, plan batchPlan) (map[string]any, []byte, int, int, error) {
	ranges, expectedRows, err := targetRanges(plan)
	if err != nil {
		return nil, nil, 0, 0, err
	}
	captured := &capturedRows{ByMatch: map[string]map[int]ai42dataset.Row{}, Ranges: ranges}
	reportedRows, err := source.ReadTargetRows(ctx, ranges, func(row ai42dataset.Row) error {
		if !tickInRanges(row.Tick, ranges[row.MatchID]) {
			return fmt.Errorf("targeted reader returned unrequested row %q tick %d", row.MatchID, row.Tick)
		}
		rows := captured.ByMatch[row.MatchID]
		if rows == nil {
			rows = map[int]ai42dataset.Row{}
			captured.ByMatch[row.MatchID] = rows
		}
		if _, exists := rows[row.Tick]; exists {
			return fmt.Errorf("targeted reader returned duplicate row %q tick %d", row.MatchID, row.Tick)
		}
		rows[row.Tick] = copyRow(row)
		captured.Rows++
		return nil
	})
	if err != nil {
		return nil, nil, 0, captured.Rows, fmt.Errorf("targeted batch read failed: %w", err)
	}
	if reportedRows != captured.Rows || captured.Rows != expectedRows {
		return nil, nil, 0, captured.Rows, fmt.Errorf("targeted reader returned %d/%d rows, expected %d", reportedRows, captured.Rows, expectedRows)
	}
	bundle, supervised, err := buildBundle(plan, captured.ByMatch)
	if err != nil {
		return nil, nil, 0, captured.Rows, err
	}
	encoded, err := canonicalJSON(bundle)
	if err != nil {
		return nil, nil, 0, captured.Rows, err
	}
	if len(encoded) > MaxWorkerBundle {
		return nil, nil, 0, captured.Rows, fmt.Errorf("batch bundle exceeds %d-byte limit", MaxWorkerBundle)
	}
	return bundle, encoded, supervised, captured.Rows, nil
}

func targetRanges(plan batchPlan) (map[string][][2]int, int, error) {
	raw := map[string][][2]int{}
	for _, sequence := range plan.Sequences {
		start := sequence.Start
		if start > 0 {
			start--
		}
		if start < 0 || sequence.Stop <= start {
			return nil, 0, fmt.Errorf("invalid target range for %q", sequence.MatchID)
		}
		raw[sequence.MatchID] = append(raw[sequence.MatchID], [2]int{start, sequence.Stop})
	}
	result := map[string][][2]int{}
	total := 0
	for matchID, values := range raw {
		sort.Slice(values, func(i, j int) bool {
			if values[i][0] == values[j][0] {
				return values[i][1] < values[j][1]
			}
			return values[i][0] < values[j][0]
		})
		merged := make([][2]int, 0, len(values))
		for _, value := range values {
			if len(merged) == 0 || value[0] > merged[len(merged)-1][1] {
				merged = append(merged, value)
				continue
			}
			if value[1] > merged[len(merged)-1][1] {
				merged[len(merged)-1][1] = value[1]
			}
		}
		result[matchID] = merged
		for _, value := range merged {
			total += value[1] - value[0]
		}
	}
	if total < 1 || total > MaxBatchRows+MaxBatchSize {
		return nil, 0, fmt.Errorf("targeted row count %d exceeds bounded batch allowance", total)
	}
	return result, total, nil
}

func tickInRanges(tick int, ranges [][2]int) bool {
	index := sort.Search(len(ranges), func(index int) bool { return ranges[index][1] > tick })
	return index < len(ranges) && tick >= ranges[index][0]
}

func copyRow(row ai42dataset.Row) ai42dataset.Row {
	row.Invalid = append([]uint8(nil), row.Invalid...)
	row.Rewards = append([]float32(nil), row.Rewards...)
	row.Hero = append([]float32(nil), row.Hero...)
	row.Abilities = append([]float32(nil), row.Abilities...)
	row.Entities = append([]float32(nil), row.Entities...)
	row.Global = append([]float32(nil), row.Global...)
	row.EntityMask = append([]uint8(nil), row.EntityMask...)
	row.KindMask = append([]uint8(nil), row.KindMask...)
	row.TargetMask = append([]uint8(nil), row.TargetMask...)
	row.SkillTargetMask = append([]uint8(nil), row.SkillTargetMask...)
	row.TeacherStatus = append([]uint8(nil), row.TeacherStatus...)
	row.TeacherAction = append([]ai42dataset.Action(nil), row.TeacherAction...)
	return row
}

func buildBundle(plan batchPlan, rows map[string]map[int]ai42dataset.Row) (map[string]any, int, error) {
	observations := map[string]any{"hero": []any{}, "abilities": []any{}, "entities": []any{}, "global_state": []any{}, "hero_ids": []any{}}
	masks := map[string]any{"entity_mask": []any{}, "kind_mask": []any{}, "target_mask": []any{}, "skill_target_mask": []any{}}
	labels := map[string]any{"teacher_actions": []any{}, "teacher_status": []any{}}
	root := map[string]any{"observations": observations, "masks": masks, "labels": labels, "padding_mask": []any{}, "loss_mask": []any{}, "reset_mask": []any{}, "death_mask": []any{}, "sequence_ids": []any{}, "time_indices": []any{}}
	supervised := 0
	for _, sequence := range plan.Sequences {
		matchRows := rows[sequence.MatchID]
		heroRows, abilitiesRows, entitiesRows, globalRows := []any{}, []any{}, []any{}, []any{}
		entityMasks, kindMasks, targetMasks, skillMasks := []any{}, []any{}, []any{}, []any{}
		actions, statuses := []any{}, []any{}
		padding, loss, reset, death, times, heroIDs := []any{}, []any{}, []any{}, []any{}, []any{}, []any{}
		for local := 0; local < plan.SequenceLength; local++ {
			tick := sequence.Start + local
			actual := tick < sequence.Stop
			var row ai42dataset.Row
			if actual {
				var ok bool
				row, ok = matchRows[tick]
				if !ok {
					return nil, 0, fmt.Errorf("batch capture is missing row %s tick %d", sequence.MatchID, tick)
				}
			}
			padding = append(padding, !actual)
			loss = append(loss, actual)
			reset = append(reset, local == 0)
			times = append(times, int64(sequence.Step)+int64(local))
			if !actual {
				death = append(death, false)
				heroIDs = append(heroIDs, int64(0))
				heroRows = append(heroRows, floatZeros(ai42dataset.HeroFeatures))
				abilitiesRows = append(abilitiesRows, floatMatrixZeros(ai42dataset.AbilityCount, ai42dataset.AbilityFeatures))
				entitiesRows = append(entitiesRows, floatMatrixZeros(ai42dataset.MaxEntities, ai42dataset.EntityFeatures))
				globalRows = append(globalRows, floatZeros(ai42dataset.GlobalFeatures))
				entityMasks = append(entityMasks, boolZeros(ai42dataset.MaxEntities))
				kindMasks = append(kindMasks, boolZeros(ai42dataset.ActionKinds))
				targetMasks = append(targetMasks, boolZeros(ai42dataset.MaxEntities))
				skillMasks = append(skillMasks, boolMatrixZeros(ai42dataset.AbilityCount, ai42dataset.MaxEntities))
				actions = append(actions, []any{0, 0, 0, 0})
				statuses = append(statuses, 0)
				continue
			}
			baseHero := sequence.Hero * ai42dataset.HeroFeatures
			hero := row.Hero[baseHero : baseHero+ai42dataset.HeroFeatures]
			heroRows = append(heroRows, floatValues(hero))
			heroIDs = append(heroIDs, int64(roundEven(float64(hero[0])*100)))
			baseAbilities := sequence.Hero * ai42dataset.AbilityCount * ai42dataset.AbilityFeatures
			abilitiesRows = append(abilitiesRows, floatMatrixValues(row.Abilities[baseAbilities:baseAbilities+ai42dataset.AbilityCount*ai42dataset.AbilityFeatures], ai42dataset.AbilityCount, ai42dataset.AbilityFeatures))
			baseEntities := sequence.Hero * ai42dataset.MaxEntities * ai42dataset.EntityFeatures
			entitiesRows = append(entitiesRows, floatMatrixValues(row.Entities[baseEntities:baseEntities+ai42dataset.MaxEntities*ai42dataset.EntityFeatures], ai42dataset.MaxEntities, ai42dataset.EntityFeatures))
			baseGlobal := sequence.Hero * ai42dataset.GlobalFeatures
			globalRows = append(globalRows, floatValues(row.Global[baseGlobal:baseGlobal+ai42dataset.GlobalFeatures]))
			baseEntity := sequence.Hero * ai42dataset.MaxEntities
			entityMasks = append(entityMasks, boolValues(row.EntityMask[baseEntity:baseEntity+ai42dataset.MaxEntities]))
			kindMasks = append(kindMasks, boolValues(row.KindMask[sequence.Hero*ai42dataset.ActionKinds:(sequence.Hero+1)*ai42dataset.ActionKinds]))
			targetMasks = append(targetMasks, boolValues(row.TargetMask[baseEntity:baseEntity+ai42dataset.MaxEntities]))
			skillBase := sequence.Hero * ai42dataset.AbilityCount * ai42dataset.MaxEntities
			skillMasks = append(skillMasks, boolMatrixValues(row.SkillTargetMask[skillBase:skillBase+ai42dataset.AbilityCount*ai42dataset.MaxEntities], ai42dataset.AbilityCount, ai42dataset.MaxEntities))
			action := row.TeacherAction[sequence.Hero]
			actions = append(actions, []any{int(action.Kind), int(action.Target), int(action.Direction), int(action.Distance)})
			status := row.TeacherStatus[sequence.Hero]
			statuses = append(statuses, int(status))
			if status >= 1 && status <= 4 {
				supervised++
			}
			dead := hero[9] >= 0.5
			if tick > 0 {
				previous, ok := matchRows[tick-1]
				if !ok {
					return nil, 0, fmt.Errorf("batch capture is missing previous row %s tick %d", sequence.MatchID, tick-1)
				}
				previousBase := sequence.Hero * ai42dataset.HeroFeatures
				dead = dead || previous.Hero[previousBase+9] >= 0.5
			}
			death = append(death, dead)
		}
		observations["hero"] = append(observations["hero"].([]any), heroRows)
		observations["abilities"] = append(observations["abilities"].([]any), abilitiesRows)
		observations["entities"] = append(observations["entities"].([]any), entitiesRows)
		observations["global_state"] = append(observations["global_state"].([]any), globalRows)
		observations["hero_ids"] = append(observations["hero_ids"].([]any), heroIDs)
		masks["entity_mask"] = append(masks["entity_mask"].([]any), entityMasks)
		masks["kind_mask"] = append(masks["kind_mask"].([]any), kindMasks)
		masks["target_mask"] = append(masks["target_mask"].([]any), targetMasks)
		masks["skill_target_mask"] = append(masks["skill_target_mask"].([]any), skillMasks)
		labels["teacher_actions"] = append(labels["teacher_actions"].([]any), actions)
		labels["teacher_status"] = append(labels["teacher_status"].([]any), statuses)
		root["padding_mask"] = append(root["padding_mask"].([]any), padding)
		root["loss_mask"] = append(root["loss_mask"].([]any), loss)
		root["reset_mask"] = append(root["reset_mask"].([]any), reset)
		root["death_mask"] = append(root["death_mask"].([]any), death)
		root["time_indices"] = append(root["time_indices"].([]any), times)
		root["sequence_ids"] = append(root["sequence_ids"].([]any), sequenceID(sequence.MatchID, sequence.Hero, sequence.Step))
	}
	if supervised == 0 {
		return nil, 0, fmt.Errorf("selected recurrent batch contains no supported supervision")
	}
	return root, supervised, nil
}

func sequenceID(matchID string, hero int, step uint32) int64 {
	digest := sha256.Sum256([]byte(fmt.Sprintf("%s\x00%d\x00%d", matchID, hero, step)))
	return int64(binary.LittleEndian.Uint64(digest[:8]) & uint64(math.MaxInt64))
}
func roundEven(value float64) int64 {
	floor := math.Floor(value)
	fraction := value - floor
	if fraction < 0.5 {
		return int64(floor)
	}
	if fraction > 0.5 {
		return int64(floor + 1)
	}
	if int64(floor)%2 == 0 {
		return int64(floor)
	}
	return int64(floor + 1)
}
func floatValues(values []float32) []any {
	result := make([]any, len(values))
	for index, value := range values {
		result[index] = float64(value)
	}
	return result
}
func floatZeros(length int) []any {
	result := make([]any, length)
	for index := range result {
		result[index] = float64(0)
	}
	return result
}
func floatMatrixValues(values []float32, rows, columns int) []any {
	result := make([]any, rows)
	for row := 0; row < rows; row++ {
		result[row] = floatValues(values[row*columns : (row+1)*columns])
	}
	return result
}
func floatMatrixZeros(rows, columns int) []any {
	result := make([]any, rows)
	for row := range result {
		result[row] = floatZeros(columns)
	}
	return result
}
func boolValues(values []uint8) []any {
	result := make([]any, len(values))
	for index, value := range values {
		result[index] = value != 0
	}
	return result
}
func boolZeros(length int) []any {
	result := make([]any, length)
	for index := range result {
		result[index] = false
	}
	return result
}
func boolMatrixValues(values []uint8, rows, columns int) []any {
	result := make([]any, rows)
	for row := 0; row < rows; row++ {
		result[row] = boolValues(values[row*columns : (row+1)*columns])
	}
	return result
}
func boolMatrixZeros(rows, columns int) []any {
	result := make([]any, rows)
	for row := range result {
		result[row] = boolZeros(columns)
	}
	return result
}
