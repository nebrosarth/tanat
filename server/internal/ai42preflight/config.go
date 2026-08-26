package ai42preflight

import (
	"fmt"
	"os"
	"path/filepath"
)

var defaultConfig = Config{
	ProtocolVersion:     ProtocolVersion,
	Model:               ModelConfig{HiddenSize: 192, ModelWidth: 192, EntityLayers: 2, NumHeads: 6, FFMultiplier: 4},
	SequenceLength:      64,
	BatchSize:           8,
	ValidationBatchSize: 8,
	Learner: LearnerConfig{
		LearningRate: 3e-4, WeightDecay: 1e-4, ClassBalancePower: 1.0, OffsetDistanceLossWeight: 1.0, MaxGradientNorm: 1.0,
		HeadWeights: map[string]float64{"control": 1, "kind": 1, "target": 1, "offset": 1, "anchor": 1},
	},
	Seed: 4242, ValidationProbeLimit: 0,
	SupervisionControllers: []uint8{0, 1, 2, 3},
}

var modelFields = map[string]struct{}{
	"hidden_size": {}, "model_width": {}, "entity_layers": {}, "num_heads": {}, "ff_multiplier": {},
}
var recurrentFields = map[string]struct{}{"sequence_length": {}, "batch_size": {}}
var trainingFields = map[string]struct{}{
	"seed": {}, "max_optimizer_seconds": {}, "max_steps": {}, "epochs": {}, "validation_batches": {}, "validation_epsilon": {},
	"checkpoint_interval": {}, "validation_matches": {}, "supervision_controllers": {}, "validation_batch_size": {},
}

// LoadConfig reads the single production training config. Training settings
// describe the artifact but never authorize optimizer work in this package.
func LoadConfig(path string) (Config, error) {
	if path == "" {
		return defaultConfig, nil
	}
	absolute, err := filepath.Abs(path)
	if err != nil {
		return Config{}, fmt.Errorf("resolve config: %w", err)
	}
	raw, err := os.ReadFile(absolute)
	if err != nil {
		return Config{}, fmt.Errorf("read config: %w", err)
	}
	value, err := decodeStrictJSON(raw, "config")
	if err != nil {
		return Config{}, err
	}
	root, err := object(value, "config")
	if err != nil {
		return Config{}, err
	}
	config := defaultConfig
	if err := parseTrainingConfig(root, &config); err != nil {
		return Config{}, err
	}
	if err := config.Validate(); err != nil {
		return Config{}, fmt.Errorf("config: %w", err)
	}
	return config, nil
}

func decodeStrictJSON(raw []byte, name string) (any, error) {
	var value any
	if err := decodeWithDuplicateCheck(raw, &value); err != nil {
		return nil, fmt.Errorf("%s is invalid JSON: %w", name, err)
	}
	return value, nil
}

func parseTrainingConfig(root map[string]any, config *Config) error {
	required := map[string]struct{}{"protocol_version": {}, "model": {}, "recurrent_batch": {}, "learner": {}, "training": {}}
	if err := exactFields(root, required, nil, "training config"); err != nil {
		return err
	}
	if err := parseProtocol(root, config); err != nil {
		return err
	}
	if err := parseModel(root["model"], &config.Model); err != nil {
		return err
	}
	if err := parseRecurrent(root["recurrent_batch"], config); err != nil {
		return err
	}
	learner, err := object(root["learner"], "training config.learner")
	if err != nil {
		return err
	}
	requiredLearner := map[string]struct{}{"class_balance_power": {}, "offset_distance_loss_weight": {}, "max_gradient_norm": {}, "learning_rate": {}, "weight_decay": {}}
	if err := exactFields(learner, requiredLearner, map[string]struct{}{"head_weights": {}}, "training config.learner"); err != nil {
		return err
	}
	config.Learner.ClassBalancePower, err = asNumber(learner["class_balance_power"], "training config.learner.class_balance_power")
	if err != nil {
		return err
	}
	config.Learner.OffsetDistanceLossWeight, err = asNumber(learner["offset_distance_loss_weight"], "training config.learner.offset_distance_loss_weight")
	if err != nil {
		return err
	}
	config.Learner.MaxGradientNorm, err = asNumber(learner["max_gradient_norm"], "training config.learner.max_gradient_norm")
	if err != nil {
		return err
	}
	config.Learner.LearningRate, err = asNumber(learner["learning_rate"], "training config.learner.learning_rate")
	if err != nil {
		return err
	}
	config.Learner.WeightDecay, err = asNumber(learner["weight_decay"], "training config.learner.weight_decay")
	if err != nil {
		return err
	}
	if raw, ok := learner["head_weights"]; ok {
		if err := parseHeadWeights(raw, &config.Learner.HeadWeights); err != nil {
			return err
		}
	}
	training, err := object(root["training"], "training config.training")
	if err != nil {
		return err
	}
	if err := exactFields(training, map[string]struct{}{"seed": {}, "max_optimizer_seconds": {}, "max_steps": {}, "epochs": {}, "validation_batches": {}, "validation_epsilon": {}}, map[string]struct{}{"checkpoint_interval": {}, "validation_matches": {}, "supervision_controllers": {}, "validation_batch_size": {}}, "training config.training"); err != nil {
		return err
	}
	seed, err := asInt(training["seed"], "training config.training.seed", 0, 1<<32-1)
	if err != nil {
		return err
	}
	config.Seed = uint32(seed)
	budget, err := asNumber(training["max_optimizer_seconds"], "training config.training.max_optimizer_seconds")
	if err != nil || budget <= 0 || budget > 300 {
		if err != nil {
			return err
		}
		return fmt.Errorf("training config.training.max_optimizer_seconds must be in (0,300]")
	}
	if _, err := asInt(training["max_steps"], "training config.training.max_steps", 0, 1<<31-1); err != nil {
		return err
	}
	if _, err := asInt(training["epochs"], "training config.training.epochs", 1, 1<<31-1); err != nil {
		return err
	}
	validationBatches, err := asInt(training["validation_batches"], "training config.training.validation_batches", 1, 1<<31-1)
	if err != nil {
		return err
	}
	config.ValidationProbeLimit = int(validationBatches)
	config.ValidationBatchSize = config.BatchSize
	if value, ok := training["validation_batch_size"]; ok {
		validationBatchSize, err := asInt(value, "training config.training.validation_batch_size", 1, MaxValidationBatchSize)
		if err != nil {
			return err
		}
		config.ValidationBatchSize = int(validationBatchSize)
	}
	if value, ok := training["validation_matches"]; ok {
		validationMatches, err := asInt(value, "training config.training.validation_matches", 1, 1<<31-1)
		if err != nil {
			return err
		}
		config.ValidationProbeLimit = int(validationMatches)
	}
	validationEpsilon, err := asNumber(training["validation_epsilon"], "training config.training.validation_epsilon")
	if err != nil || validationEpsilon < 0 {
		if err != nil {
			return err
		}
		return fmt.Errorf("training config.training.validation_epsilon must be non-negative")
	}
	if value, ok := training["checkpoint_interval"]; ok {
		if _, err := asInt(value, "training config.training.checkpoint_interval", 1, 1<<31-1); err != nil {
			return err
		}
	}
	if value, ok := training["supervision_controllers"]; ok {
		values, ok := value.([]any)
		if !ok || len(values) == 0 {
			return fmt.Errorf("training config.training.supervision_controllers must be a non-empty array")
		}
		seen := [4]bool{}
		for index, value := range values {
			controller, err := asInt(value, fmt.Sprintf("training config.training.supervision_controllers[%d]", index), 0, 3)
			if err != nil {
				return err
			}
			if seen[controller] {
				return fmt.Errorf("training config.training.supervision_controllers contains duplicate %d", controller)
			}
			seen[controller] = true
		}
		controllers := make([]uint8, 0, len(values))
		for controller, present := range seen {
			if present {
				controllers = append(controllers, uint8(controller))
			}
		}
		config.SupervisionControllers = controllers
	}
	return nil
}

func parseProtocol(root map[string]any, config *Config) error {
	protocol, err := asInt(root["protocol_version"], "config.protocol_version", ProtocolVersion, ProtocolVersion)
	if err != nil {
		return err
	}
	config.ProtocolVersion = int(protocol)
	return nil
}

func parseModel(value any, model *ModelConfig) error {
	objectValue, err := object(value, "config.model")
	if err != nil {
		return err
	}
	if err := exactFields(objectValue, modelFields, nil, "config.model"); err != nil {
		return err
	}
	values := []*int{&model.HiddenSize, &model.ModelWidth, &model.EntityLayers, &model.NumHeads, &model.FFMultiplier}
	names := []string{"hidden_size", "model_width", "entity_layers", "num_heads", "ff_multiplier"}
	for index, name := range names {
		number, err := asInt(objectValue[name], "config.model."+name, 1, 4096)
		if err != nil {
			return err
		}
		*values[index] = int(number)
	}
	return nil
}

func parseRecurrent(value any, config *Config) error {
	objectValue, err := object(value, "config.recurrent_batch")
	if err != nil {
		return err
	}
	if err := exactFields(objectValue, recurrentFields, nil, "config.recurrent_batch"); err != nil {
		return err
	}
	sequence, err := asInt(objectValue["sequence_length"], "config.recurrent_batch.sequence_length", 1, MaxSequenceLength)
	if err != nil {
		return err
	}
	batch, err := asInt(objectValue["batch_size"], "config.recurrent_batch.batch_size", 1, MaxBatchSize)
	if err != nil {
		return err
	}
	config.SequenceLength, config.BatchSize = int(sequence), int(batch)
	return nil
}

var headSizes = map[string]int{"control": 4, "kind": 8, "target": 96, "offset": 81, "anchor": 15}

func parseHeadWeights(value any, destination *map[string]float64) error {
	objectValue, err := object(value, "config.learner.head_weights")
	if err != nil {
		return err
	}
	expected := map[string]struct{}{"control": {}, "kind": {}, "target": {}, "offset": {}, "anchor": {}}
	if err := exactFields(objectValue, expected, nil, "config.learner.head_weights"); err != nil {
		return err
	}
	result := make(map[string]float64, len(expected))
	for head := range expected {
		number, err := asNumber(objectValue[head], "config.learner.head_weights."+head)
		if err != nil {
			return err
		}
		if number < 0 {
			return fmt.Errorf("head_weights[%q] must be non-negative", head)
		}
		result[head] = number
	}
	*destination = result
	return nil
}

func (config Config) Validate() error {
	if config.ProtocolVersion != ProtocolVersion {
		return fmt.Errorf("protocol_version must be %d", ProtocolVersion)
	}
	if config.Model.ModelWidth%config.Model.NumHeads != 0 {
		return fmt.Errorf("model_width must be divisible by num_heads")
	}
	if config.SequenceLength < 1 || config.SequenceLength > MaxSequenceLength || config.BatchSize < 1 || config.BatchSize > MaxBatchSize {
		return fmt.Errorf("recurrent batch exceeds bounded dimensions")
	}
	if config.BatchSize*config.SequenceLength > MaxBatchRows {
		return fmt.Errorf("recurrent batch exceeds bounded row limit")
	}
	if config.ValidationBatchSize < 1 || config.ValidationBatchSize > MaxValidationBatchSize {
		return fmt.Errorf("validation_batch_size must be in [1,%d]", MaxValidationBatchSize)
	}
	if config.Learner.ClassBalancePower != 1.0 {
		return fmt.Errorf("AI-42 BC-v2 freezes class_balance_power at 1")
	}
	for name, value := range map[string]float64{"learning_rate": config.Learner.LearningRate, "weight_decay": config.Learner.WeightDecay, "class_balance_power": config.Learner.ClassBalancePower, "offset_distance_loss_weight": config.Learner.OffsetDistanceLossWeight, "max_gradient_norm": config.Learner.MaxGradientNorm} {
		if value != value || value > 1e308 || value < -1e308 {
			return fmt.Errorf("%s must be finite", name)
		}
	}
	if config.Learner.LearningRate <= 0 || config.Learner.WeightDecay < 0 || config.Learner.OffsetDistanceLossWeight < 0 || config.Learner.MaxGradientNorm <= 0 {
		return fmt.Errorf("learner rates and gradient norm are outside their valid ranges")
	}
	if len(config.Learner.HeadWeights) != 5 {
		return fmt.Errorf("head_weights must contain exactly the five AI-42 loss heads")
	}
	positive := false
	for _, head := range []string{"control", "kind", "target", "offset", "anchor"} {
		value, ok := config.Learner.HeadWeights[head]
		if !ok || value != value || value > 1e308 || value < 0 {
			return fmt.Errorf("head_weights[%q] is outside its valid range", head)
		}
		positive = positive || value > 0
	}
	if !positive {
		return fmt.Errorf("head_weights must enable at least one loss head")
	}
	if len(config.SupervisionControllers) == 0 {
		return fmt.Errorf("supervision_controllers must not be empty")
	}
	seenControllers := [4]bool{}
	for _, controller := range config.SupervisionControllers {
		if controller > 3 || seenControllers[controller] {
			return fmt.Errorf("supervision_controllers must contain unique IDs in [0,3]")
		}
		seenControllers[controller] = true
	}
	return nil
}
