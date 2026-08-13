package battleserver

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	ai40WireVersion   = uint16(1)
	ai40RequestInfer  = uint16(1)
	ai40RequestReset  = uint16(2)
	ai40ResponseOK    = uint16(100)
	ai40ResponseError = uint16(255)
	ai40MaxFrame      = 2 << 20
	ai40LatencyLimit  = 150 * time.Millisecond
)

var ai40Magic = [4]byte{'A', 'I', '4', '0'}

type ai40PolicyRuntime interface {
	Infer(ids []int32, observations []AssaultObservationV1) ([]HeroActionV1, time.Duration, error)
	Reset(ids []int32) error
	Close() error
}

type ai40Manifest struct {
	Version       string   `json:"version"`
	ModelID       string   `json:"model_id"`
	SchemaHash    string   `json:"schema_hash"`
	RewardHash    string   `json:"reward_hash"`
	Model         string   `json:"model"`
	Inputs        []string `json:"inputs"`
	Outputs       []string `json:"outputs"`
	RecurrentSize int      `json:"recurrent_size"`
}

type ai40Sidecar struct {
	cmd    *exec.Cmd
	in     io.WriteCloser
	out    *bufio.Reader
	stderr bytes.Buffer
	model  string
	mu     sync.Mutex
	close  sync.Once
}

func (p *ai40Sidecar) ModelID() string { return p.model }

func ai40ManifestPath() string { return strings.TrimSpace(os.Getenv("TANAT_AI40_MANIFEST")) }

func ai40PythonCommand() string {
	raw := strings.TrimSpace(os.Getenv("TANAT_AI40_PYTHON"))
	if raw == "" {
		return "python"
	}
	// This setting is an executable path, not a shell command. Keeping it as one
	// argument is important on Windows installations whose path contains spaces.
	return strings.Trim(raw, "\"")
}

func loadAI40Manifest(path string) (ai40Manifest, string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return ai40Manifest{}, "", err
	}
	var manifest ai40Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return ai40Manifest{}, "", err
	}
	if manifest.SchemaHash != fmt.Sprintf("%x", AssaultSchemaHashV1) {
		return ai40Manifest{}, "", errors.New("AI-40 manifest observation schema mismatch")
	}
	if manifest.RewardHash != fmt.Sprintf("%x", AssaultRewardHashV2) {
		return ai40Manifest{}, "", errors.New("AI-40 manifest reward schema mismatch")
	}
	if manifest.Version == "" || manifest.RecurrentSize <= 0 || manifest.Model == "" {
		return ai40Manifest{}, "", errors.New("AI-40 manifest is incomplete")
	}
	wantInputs := []string{"hero", "entities", "global", "entity_mask", "h", "c"}
	wantOutputs := []string{"kind", "target", "direction", "distance", "value", "next_h", "next_c"}
	if !sameAI40Names(manifest.Inputs, wantInputs) || !sameAI40Names(manifest.Outputs, wantOutputs) {
		return ai40Manifest{}, "", errors.New("AI-40 manifest tensor names mismatch")
	}
	model := manifest.Model
	if !filepath.IsAbs(model) {
		model = filepath.Join(filepath.Dir(path), model)
	}
	model, err = filepath.Abs(model)
	if err != nil {
		return ai40Manifest{}, "", err
	}
	if _, err := os.Stat(model); err != nil {
		return ai40Manifest{}, "", fmt.Errorf("AI-40 model: %w", err)
	}
	return manifest, model, nil
}

func sameAI40Names(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range want {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func startAI40Sidecar() (ai40PolicyRuntime, error) {
	manifestPath := ai40ManifestPath()
	if manifestPath == "" {
		return nil, errors.New("TANAT_AI40_MANIFEST is not set")
	}
	manifest, model, err := loadAI40Manifest(manifestPath)
	if err != nil {
		return nil, err
	}
	python := ai40PythonCommand()
	args := []string{"-m", "tanat_ai40.inference_server", "--model", model,
		"--recurrent-size", strconv.Itoa(manifest.RecurrentSize)}
	cmd := exec.Command(python, args...)
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")
	in, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		_ = in.Close()
		return nil, err
	}
	modelID := manifest.ModelID
	if modelID == "" {
		modelID = manifest.Version
	}
	runtime := &ai40Sidecar{cmd: cmd, in: in, out: bufio.NewReader(out), model: modelID}
	cmd.Stderr = &runtime.stderr
	if err := cmd.Start(); err != nil {
		_ = in.Close()
		return nil, err
	}
	ready := make(chan error, 1)
	go func() {
		body, response, readErr := readAI40Frame(runtime.out)
		if readErr != nil {
			ready <- readErr
			return
		}
		if response != ai40ResponseOK || string(body) != "ready" {
			ready <- fmt.Errorf("AI-40 sidecar readiness failed: %s", body)
			return
		}
		ready <- nil
	}()
	select {
	case err := <-ready:
		if err != nil {
			_ = runtime.Close()
			if stderr := strings.TrimSpace(runtime.stderr.String()); stderr != "" {
				return nil, fmt.Errorf("%w: %s", err, stderr)
			}
			return nil, err
		}
	case <-time.After(30 * time.Second):
		_ = runtime.Close()
		if stderr := strings.TrimSpace(runtime.stderr.String()); stderr != "" {
			return nil, fmt.Errorf("AI-40 sidecar startup timeout: %s", stderr)
		}
		return nil, errors.New("AI-40 sidecar startup timeout")
	}
	return runtime, nil
}

func (p *ai40Sidecar) Infer(ids []int32, observations []AssaultObservationV1) ([]HeroActionV1, time.Duration, error) {
	if len(ids) == 0 || len(ids) != len(observations) || len(ids) > AssaultHeroCount {
		return nil, 0, errors.New("AI-40 invalid inference batch")
	}
	payload := new(bytes.Buffer)
	payload.WriteByte(byte(len(ids)))
	for i, id := range ids {
		_ = binary.Write(payload, binary.LittleEndian, id)
		writeAI40Observation(payload, &observations[i])
	}
	type inferenceResult struct {
		body     []byte
		response uint16
		err      error
	}
	result := make(chan inferenceResult, 1)
	started := time.Now()
	go func() {
		p.mu.Lock()
		defer p.mu.Unlock()
		if err := writeAI40Frame(p.in, ai40RequestInfer, payload.Bytes()); err != nil {
			result <- inferenceResult{err: err}
			return
		}
		body, response, err := readAI40Frame(p.out)
		result <- inferenceResult{body: body, response: response, err: err}
	}()
	var received inferenceResult
	select {
	case received = <-result:
	case <-time.After(ai40LatencyLimit):
		if p.cmd != nil && p.cmd.Process != nil {
			_ = p.cmd.Process.Kill()
		}
		return nil, time.Since(started), errors.New("AI-40 inference timeout")
	}
	body, response, err := received.body, received.response, received.err
	latency := time.Since(started)
	if err != nil {
		return nil, latency, err
	}
	if response == ai40ResponseError {
		return nil, latency, fmt.Errorf("AI-40 sidecar: %s", body)
	}
	if response != ai40ResponseOK || len(body) != len(ids)*5 {
		return nil, latency, errors.New("AI-40 malformed inference response")
	}
	actions := make([]HeroActionV1, len(ids))
	for i := range actions {
		off := i * 5
		actions[i] = HeroActionV1{Kind: AssaultActionKindV1(body[off]),
			Target: binary.LittleEndian.Uint16(body[off+1 : off+3]), Direction: body[off+3], Distance: body[off+4]}
		if int(actions[i].Kind) >= AssaultActionKinds || actions[i].Target >= AssaultMaxEntities ||
			actions[i].Direction >= 16 || actions[i].Distance >= 3 {
			return nil, latency, errors.New("AI-40 action outside discrete bounds")
		}
	}
	return actions, latency, nil
}

func (p *ai40Sidecar) Reset(ids []int32) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	payload := new(bytes.Buffer)
	payload.WriteByte(byte(len(ids)))
	for _, id := range ids {
		_ = binary.Write(payload, binary.LittleEndian, id)
	}
	if err := writeAI40Frame(p.in, ai40RequestReset, payload.Bytes()); err != nil {
		return err
	}
	body, response, err := readAI40Frame(p.out)
	if err != nil {
		return err
	}
	if response != ai40ResponseOK {
		return fmt.Errorf("AI-40 reset: %s", body)
	}
	return nil
}

func (p *ai40Sidecar) Close() error {
	p.close.Do(func() {
		// Kill first so a timed-out reader holding p.mu is released immediately.
		// Wait is bounded because shutdown is also called while ending a match.
		if p.cmd != nil && p.cmd.Process != nil {
			_ = p.cmd.Process.Kill()
		}
		if p.in != nil {
			_ = p.in.Close()
		}
		if p.cmd != nil && p.cmd.Process != nil {
			done := make(chan error, 1)
			go func() { done <- p.cmd.Wait() }()
			select {
			case <-time.After(time.Second):
			case <-done:
			}
		}
	})
	return nil
}

func (s *Server) botAI40BatchTickLocked(inst *huntInstance, now float64) {
	if inst == nil || inst.dota == nil {
		return
	}
	var brains []*botBrain
	for _, brain := range inst.bots {
		if brain != nil && botAIVersionForBrain(brain) == 40 && now >= brain.ai40NextAt {
			brains = append(brains, brain)
		}
	}
	if len(brains) == 0 {
		return
	}
	sort.Slice(brains, func(i, j int) bool { return brains[i].c.objID < brains[j].c.objID })
	if inst.dota.ai40Runtime == nil {
		if inst.dota.ai40StartAttempted {
			return
		}
		inst.dota.ai40StartAttempted = true
		runtime, err := startAI40Sidecar()
		if err != nil {
			s.botAI40FallbackAllLocked(inst, "startup: "+err.Error())
			return
		}
		inst.dota.ai40Runtime = runtime
	}
	envView := &AssaultEnv{server: s, inst: inst, clock: nil}
	ids := make([]int32, len(brains))
	observations := make([]AssaultObservationV1, len(brains))
	for i, brain := range brains {
		// Item purchases and automatic skill levelling deliberately stay scripted;
		// the trained policy owns only the same factorized combat/movement action
		// exposed by AssaultEnv.
		s.botSpendSkillPointLocked(brain)
		s.botBuyItemsLocked(brain, now)
		ids[i] = brain.c.objID
		observations[i] = envView.observationForConnLocked(brain.c, &brain.ai40EntityIDs, now)
		brain.ai40NextAt = now + AssaultTick.Seconds()
	}
	actions, latency, err := inst.dota.ai40Runtime.Infer(ids, observations)
	if err == nil && len(actions) != len(brains) {
		err = errors.New("AI-40 inference returned wrong action count")
	}
	if err != nil || latency > ai40LatencyLimit {
		reason := fmt.Sprintf("inference latency=%s", latency)
		if err != nil {
			reason = fmt.Sprintf("%s (latency=%s)", err, latency)
		}
		s.botAI40FallbackAllLocked(inst, reason)
		return
	}
	for i, brain := range brains {
		if botAIVersionForBrain(brain) != 40 {
			continue
		}
		accepted := envView.applyConnActionLocked(brain.c, &brain.ai40EntityIDs, actions[i])
		s.telemetryRecordAI40ActionLocked(inst, brain, actions[i], latency, accepted)
		if !accepted {
			s.botAI40FallbackTeamLocked(inst, brain.c.playerTeam(), "invalid action")
		}
	}
	if inst.dota.botAIVersionByTeam[dotaTeamHuman] != 40 &&
		inst.dota.botAIVersionByTeam[dotaTeamElf] != 40 && inst.dota.ai40Runtime != nil {
		runtime := inst.dota.ai40Runtime
		inst.dota.ai40Runtime = nil
		go runtime.Close()
	}
}

func (s *Server) botAI40TickLocked(b *botBrain, now float64) { _ = b; _ = now }

func (s *Server) botAI40FallbackAllLocked(inst *huntInstance, reason string) {
	s.botAI40FallbackTeamLocked(inst, dotaTeamHuman, reason)
	s.botAI40FallbackTeamLocked(inst, dotaTeamElf, reason)
	if inst.dota.ai40Runtime != nil {
		runtime := inst.dota.ai40Runtime
		inst.dota.ai40Runtime = nil
		go runtime.Close()
	}
}

func (s *Server) botAI40FallbackTeamLocked(inst *huntInstance, team int32, reason string) {
	if inst.dota.botAIVersionByTeam[team] != 40 {
		return
	}
	modelID := ""
	if metadata, ok := inst.dota.ai40Runtime.(interface{ ModelID() string }); ok {
		modelID = metadata.ModelID()
	}
	inst.dota.botAIVersionByTeam[team] = 20
	inst.dota.ai40FallbackByTeam[team] = reason
	for _, brain := range inst.bots {
		if brain != nil && brain.c.playerTeam() == team {
			brain.aiVersion, brain.aiVersionSet = 20, true
			brain.nextThinkAt = 0
		}
	}
	if inst.dota.telemetry != nil {
		inst.dota.telemetry.record(telemetryAI40Fallback{
			telemetryEvent: newTelemetryEvent("ai40_fallback", inst.dota.telemetryMatchTimeLocked(float64(s.battleTime()))),
			Team:           team, ModelID: modelID, Reason: reason,
		})
	}
	log.Printf("battle: AI-40 room=%d team=%d fallback to AI-20: %s", inst.id, team, reason)
}

func (s *Server) telemetryRecordAI40ActionLocked(inst *huntInstance, brain *botBrain, action HeroActionV1, latency time.Duration, accepted bool) {
	if inst == nil || inst.dota == nil || inst.dota.telemetry == nil || brain == nil || brain.c == nil {
		return
	}
	modelID := ""
	if metadata, ok := inst.dota.ai40Runtime.(interface{ ModelID() string }); ok {
		modelID = metadata.ModelID()
	}
	inst.dota.telemetry.record(telemetryAI40Action{
		telemetryEvent: newTelemetryEvent("ai40_action", inst.dota.telemetryMatchTimeLocked(float64(s.battleTime()))),
		BotID:          brain.c.objID, Team: brain.c.playerTeam(), ModelID: modelID,
		Kind: uint8(action.Kind), Target: action.Target, Direction: action.Direction, Distance: action.Distance,
		LatencyMS: float64(latency) / float64(time.Millisecond), Accepted: accepted,
	})
}

func writeAI40Observation(w io.Writer, obs *AssaultObservationV1) {
	writeAI40F32s(w, obs.Hero[:])
	for i := range obs.Entities {
		writeAI40F32s(w, obs.Entities[i][:])
	}
	writeAI40F32s(w, obs.Global[:])
	_, _ = w.Write(obs.EntityMask[:])
	_, _ = w.Write(obs.ActionMask.Kinds[:])
	_, _ = w.Write(obs.ActionMask.Targets[:])
	for i := range obs.ActionMask.SkillTarget {
		_, _ = w.Write(obs.ActionMask.SkillTarget[i][:])
	}
}

func writeAI40F32s(w io.Writer, values []float32) {
	for _, value := range values {
		_ = binary.Write(w, binary.LittleEndian, math.Float32bits(value))
	}
}

func writeAI40Frame(w io.Writer, command uint16, payload []byte) error {
	body := new(bytes.Buffer)
	body.Write(ai40Magic[:])
	_ = binary.Write(body, binary.LittleEndian, ai40WireVersion)
	_ = binary.Write(body, binary.LittleEndian, command)
	body.Write(payload)
	if err := binary.Write(w, binary.LittleEndian, uint32(body.Len())); err != nil {
		return err
	}
	_, err := w.Write(body.Bytes())
	return err
}

func readAI40Frame(r io.Reader) ([]byte, uint16, error) {
	var size uint32
	if err := binary.Read(r, binary.LittleEndian, &size); err != nil {
		return nil, 0, err
	}
	if size < 8 || size > ai40MaxFrame {
		return nil, 0, errors.New("AI-40 invalid frame size")
	}
	body := make([]byte, size)
	if _, err := io.ReadFull(r, body); err != nil {
		return nil, 0, err
	}
	if !bytes.Equal(body[:4], ai40Magic[:]) || binary.LittleEndian.Uint16(body[4:6]) != ai40WireVersion {
		return nil, 0, errors.New("AI-40 invalid frame header")
	}
	return body[8:], binary.LittleEndian.Uint16(body[6:8]), nil
}
