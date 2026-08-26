package ai42preflight

import (
	"fmt"
	"math"
)

var profileHeads = []string{"control", "kind", "target", "offset", "anchor"}

type Profile struct {
	DatasetManifestHash    string
	TrainMatchIDs          []string
	TrainMatchIDsHash      string
	Counts                 map[string][]int
	Weights                map[string][]float64
	ProfileHash            string
	SupervisionControllers []uint8
}

func buildProfile(datasetHash string, trainIDs []string, counts map[string][]int, controllers []uint8) (Profile, error) {
	if !validHash(datasetHash) {
		return Profile{}, fmt.Errorf("dataset manifest hash is not a lower-case SHA-256")
	}
	if len(trainIDs) == 0 {
		return Profile{}, fmt.Errorf("validated dataset has no train matches")
	}
	weights := make(map[string][]float64, len(profileHeads))
	for _, head := range profileHeads {
		values, ok := counts[head]
		if !ok {
			return Profile{}, fmt.Errorf("class counts are missing %s", head)
		}
		weights[head] = profileClassWeights(head, values)
	}
	idHash, err := canonicalHash(trainIDs)
	if err != nil {
		return Profile{}, err
	}
	unsigned := profileUnsigned(datasetHash, trainIDs, idHash, counts, weights, controllers)
	encoded, err := canonicalJSON(unsigned)
	if err != nil {
		return Profile{}, err
	}
	return Profile{DatasetManifestHash: datasetHash, TrainMatchIDs: append([]string(nil), trainIDs...), TrainMatchIDsHash: idHash, Counts: cloneCounts(counts), Weights: cloneWeights(weights), ProfileHash: sha256Hex(encoded), SupervisionControllers: append([]uint8(nil), controllers...)}, nil
}

func profileUnsigned(datasetHash string, ids []string, idHash string, counts map[string][]int, weights map[string][]float64, controllers []uint8) map[string]any {
	countPayload, weightPayload := map[string]any{}, map[string]any{}
	for _, head := range profileHeads {
		countValues := make([]any, len(counts[head]))
		for index, value := range counts[head] {
			countValues[index] = value
		}
		weightValues := make([]any, len(weights[head]))
		for index, value := range weights[head] {
			weightValues[index] = value
		}
		countPayload[head], weightPayload[head] = countValues, weightValues
	}
	return map[string]any{
		"format": ProfileFormat, "profile_version": ProfileVersion, "supervision_version": SupervisionVersion, "protocol_version": ProtocolVersion,
		"dataset_schema_version": "AI42-dataset-v1", "shard_schema_version": "AI42-go-shard-v2", "dataset_manifest_hash": datasetHash,
		"train_match_ids": append([]string(nil), ids...), "train_match_ids_hash": idHash, "class_balance_power": 0.5,
		"supervision_controllers": uint8sAny(controllers),
		"counts":                  countPayload, "weights": weightPayload,
	}
}

func classBalanceWeights(counts []int) []float64 {
	result := make([]float64, len(counts))
	total := float32(0)
	present := 0
	for _, count := range counts {
		if count > 0 {
			total += float32(count)
			present++
		}
	}
	if present == 0 {
		return result
	}
	mean := float32(0)
	for index, count := range counts {
		if count > 0 {
			// PyTorch's class_balance_weights operates in float32. Keep each
			// intermediate rounded to float32 before converting for JSON.
			value := float32(math.Sqrt(float64(total / float32(count))))
			result[index] = float64(value)
			mean += value
		}
	}
	mean /= float32(present)
	for index, count := range counts {
		if count > 0 {
			result[index] = float64(float32(float32(result[index]) / mean))
		}
	}
	return result
}

func profileClassWeights(head string, counts []int) []float64 {
	if head == "target" {
		result := make([]float64, len(counts))
		for index := range result {
			result[index] = 1
		}
		return result
	}
	return classBalanceWeights(counts)
}

func mergeClassWeights(profile Profile, overrides map[string][]float64) (map[string][]float64, error) {
	result := cloneWeights(profile.Weights)
	for head, values := range overrides {
		if head == "target" {
			return nil, fmt.Errorf("target class-weight overrides are forbidden because entity slots are permutation-equivariant")
		}
		counts, ok := profile.Counts[head]
		if !ok {
			return nil, fmt.Errorf("class-weight override contains unknown head %q", head)
		}
		if len(values) != len(counts) {
			return nil, fmt.Errorf("class-weight override %q has wrong length", head)
		}
		supported := 0
		mean := 0.0
		for index, value := range values {
			if !math.IsNaN(value) && !math.IsInf(value, 0) && value >= 0 {
			} else {
				return nil, fmt.Errorf("class-weight override %s[%d] is non-finite or negative", head, index)
			}
			if counts[index] == 0 && value != 0 {
				return nil, fmt.Errorf("class-weight override for absent %s class %d must be zero", head, index)
			}
			if counts[index] > 0 {
				if value <= 0 {
					return nil, fmt.Errorf("class-weight override for supported %s class %d must be positive", head, index)
				}
				supported++
				mean += value
			}
		}
		if supported == 0 {
			for _, value := range values {
				if value != 0 {
					return nil, fmt.Errorf("class-weight override %s must be all zero when absent", head)
				}
			}
		} else if math.Abs(mean/float64(supported)-1) > 2e-6 {
			return nil, fmt.Errorf("class-weight override %s must be mean-one over supported classes", head)
		}
		result[head] = append([]float64(nil), values...)
	}
	return result, nil
}

func loadExistingProfile(raw []byte, expected Profile) (Profile, error) {
	value, err := decodeCanonical(raw, "profile", 8*1024*1024)
	if err != nil {
		return Profile{}, err
	}
	root, err := object(value, "profile")
	if err != nil {
		return Profile{}, err
	}
	required := map[string]struct{}{"format": {}, "profile_version": {}, "supervision_version": {}, "protocol_version": {}, "dataset_schema_version": {}, "shard_schema_version": {}, "dataset_manifest_hash": {}, "train_match_ids": {}, "train_match_ids_hash": {}, "class_balance_power": {}, "supervision_controllers": {}, "counts": {}, "weights": {}, "profile_hash": {}}
	if err := exactFields(root, required, nil, "profile"); err != nil {
		return Profile{}, err
	}
	for _, field := range []string{"format", "profile_version", "supervision_version", "dataset_schema_version", "shard_schema_version", "dataset_manifest_hash", "train_match_ids_hash", "profile_hash"} {
		if _, err := asString(root[field], "profile."+field); err != nil {
			return Profile{}, err
		}
	}
	if root["format"] != ProfileFormat || root["profile_version"] != ProfileVersion || root["supervision_version"] != SupervisionVersion {
		return Profile{}, fmt.Errorf("profile format/version is incompatible")
	}
	protocol, err := asInt(root["protocol_version"], "profile.protocol_version", ProtocolVersion, ProtocolVersion)
	if err != nil {
		return Profile{}, err
	}
	_ = protocol
	datasetHash, _ := asString(root["dataset_manifest_hash"], "profile.dataset_manifest_hash")
	if !validHash(datasetHash) {
		return Profile{}, fmt.Errorf("profile.dataset_manifest_hash is not a lower-case SHA-256")
	}
	ids, err := stringList(root["train_match_ids"], "profile.train_match_ids")
	if err != nil || len(ids) == 0 {
		if err != nil {
			return Profile{}, err
		}
		return Profile{}, fmt.Errorf("profile.train_match_ids must be non-empty")
	}
	idHash, _ := asString(root["train_match_ids_hash"], "profile.train_match_ids_hash")
	computedIDHash, _ := canonicalHash(ids)
	if idHash != computedIDHash {
		return Profile{}, fmt.Errorf("profile train-match ID hash does not match its ordered IDs")
	}
	power, err := asNumber(root["class_balance_power"], "profile.class_balance_power")
	if err != nil || power != 0.5 {
		if err != nil {
			return Profile{}, err
		}
		return Profile{}, fmt.Errorf("AI-42 BC-v2 requires class_balance_power=0.5")
	}
	controllers, err := controllerList(root["supervision_controllers"], "profile.supervision_controllers")
	if err != nil {
		return Profile{}, err
	}
	counts, err := parseProfileCounts(root["counts"])
	if err != nil {
		return Profile{}, err
	}
	weights, err := parseProfileWeights(root["weights"], counts)
	if err != nil {
		return Profile{}, err
	}
	unsigned := profileUnsigned(datasetHash, ids, idHash, counts, weights, controllers)
	encoded, err := canonicalJSON(unsigned)
	if err != nil {
		return Profile{}, err
	}
	profileHash, _ := asString(root["profile_hash"], "profile.profile_hash")
	if profileHash != sha256Hex(encoded) {
		return Profile{}, fmt.Errorf("profile_hash does not match profile contents")
	}
	if datasetHash != expected.DatasetManifestHash || !sameStrings(ids, expected.TrainMatchIDs) || !sameCounts(counts, expected.Counts) || !sameUint8s(controllers, expected.SupervisionControllers) {
		return Profile{}, fmt.Errorf("profile lineage or counts are incompatible with verified dataset")
	}
	for _, head := range profileHeads {
		for index, value := range weights[head] {
			if math.Abs(value-expected.Weights[head][index]) > 1e-6*math.Max(1, math.Abs(expected.Weights[head][index]))+1e-7 {
				return Profile{}, fmt.Errorf("profile weights[%s][%d] do not match frozen class counts", head, index)
			}
		}
	}
	return Profile{DatasetManifestHash: datasetHash, TrainMatchIDs: ids, TrainMatchIDsHash: idHash, Counts: counts, Weights: weights, ProfileHash: profileHash, SupervisionControllers: controllers}, nil
}

func controllerList(value any, name string) ([]uint8, error) {
	raw, ok := value.([]any)
	if !ok || len(raw) == 0 {
		return nil, fmt.Errorf("%s must be a non-empty controller list", name)
	}
	result := make([]uint8, len(raw))
	previous := int64(-1)
	for index, item := range raw {
		controller, err := asInt(item, fmt.Sprintf("%s[%d]", name, index), 0, 3)
		if err != nil {
			return nil, err
		}
		if controller <= previous {
			return nil, fmt.Errorf("%s must contain sorted unique controllers", name)
		}
		previous = controller
		result[index] = uint8(controller)
	}
	return result, nil
}

func parseProfileCounts(value any) (map[string][]int, error) {
	root, err := object(value, "profile.counts")
	if err != nil {
		return nil, err
	}
	if len(root) != len(profileHeads) {
		return nil, fmt.Errorf("profile counts must contain exactly the five AI-42 heads")
	}
	result := map[string][]int{}
	for _, head := range profileHeads {
		raw, ok := root[head]
		if !ok {
			return nil, fmt.Errorf("profile.counts missing %s", head)
		}
		values, ok := raw.([]any)
		if !ok || len(values) != headSizes[head] {
			return nil, fmt.Errorf("profile.counts[%s] has wrong shape", head)
		}
		result[head] = make([]int, len(values))
		for index, item := range values {
			number, err := asInt(item, fmt.Sprintf("profile.counts[%s][%d]", head, index), 0, int64(^uint(0)>>1))
			if err != nil {
				return nil, err
			}
			result[head][index] = int(number)
		}
	}
	return result, nil
}

func parseProfileWeights(value any, counts map[string][]int) (map[string][]float64, error) {
	root, err := object(value, "profile.weights")
	if err != nil {
		return nil, err
	}
	if len(root) != len(profileHeads) {
		return nil, fmt.Errorf("profile weights must contain exactly the five AI-42 heads")
	}
	result := map[string][]float64{}
	for _, head := range profileHeads {
		raw, ok := root[head]
		if !ok {
			return nil, fmt.Errorf("profile.weights missing %s", head)
		}
		values, ok := raw.([]any)
		if !ok || len(values) != len(counts[head]) {
			return nil, fmt.Errorf("profile.weights[%s] has wrong shape", head)
		}
		result[head] = make([]float64, len(values))
		supported, mean := 0, 0.0
		for index, item := range values {
			number, err := asNumber(item, fmt.Sprintf("profile.weights[%s][%d]", head, index))
			if err != nil || number < 0 {
				if err != nil {
					return nil, err
				}
				return nil, fmt.Errorf("profile.weights[%s][%d] is invalid", head, index)
			}
			if head == "target" {
				if number <= 0 {
					return nil, fmt.Errorf("profile target-slot weights must all be positive")
				}
				supported++
				mean += number
			} else if counts[head][index] == 0 && number != 0 {
				return nil, fmt.Errorf("profile weight for absent %s class %d must be zero", head, index)
			} else if counts[head][index] > 0 {
				if number <= 0 {
					return nil, fmt.Errorf("profile weight for supported %s class %d must be positive", head, index)
				}
				supported++
				mean += number
			}
			result[head][index] = number
		}
		if supported > 0 && math.Abs(mean/float64(supported)-1) > 2e-6 {
			return nil, fmt.Errorf("profile weights[%s] are not mean-one", head)
		}
	}
	return result, nil
}

func stringList(value any, name string) ([]string, error) {
	raw, ok := value.([]any)
	if !ok {
		return nil, fmt.Errorf("%s must be a string list", name)
	}
	result := make([]string, len(raw))
	seen := map[string]struct{}{}
	for index, item := range raw {
		text, err := asString(item, fmt.Sprintf("%s[%d]", name, index))
		if err != nil {
			return nil, err
		}
		if _, exists := seen[text]; exists {
			return nil, fmt.Errorf("%s contains duplicate value", name)
		}
		seen[text] = struct{}{}
		result[index] = text
	}
	return result, nil
}
func canonicalHash(value any) (string, error) {
	payload, err := canonicalJSON(value)
	if err != nil {
		return "", err
	}
	return sha256Hex(payload), nil
}
func cloneCounts(value map[string][]int) map[string][]int {
	result := map[string][]int{}
	for key, values := range value {
		result[key] = append([]int(nil), values...)
	}
	return result
}
func cloneWeights(value map[string][]float64) map[string][]float64 {
	result := map[string][]float64{}
	for key, values := range value {
		result[key] = append([]float64(nil), values...)
	}
	return result
}
func sameStrings(left, right []string) bool {
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
func sameCounts(left, right map[string][]int) bool {
	for _, head := range profileHeads {
		if len(left[head]) != len(right[head]) {
			return false
		}
		for index := range left[head] {
			if left[head][index] != right[head][index] {
				return false
			}
		}
	}
	return true
}

func sameUint8s(left, right []uint8) bool {
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
