package battleserver

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

type fakeAI40Runtime struct {
	actions []HeroActionV1
	latency time.Duration
	err     error
	closed  atomic.Bool
}

func (f *fakeAI40Runtime) Infer(ids []int32, observations []AssaultObservationV1) ([]HeroActionV1, time.Duration, error) {
	return append([]HeroActionV1(nil), f.actions...), f.latency, f.err
}
func (f *fakeAI40Runtime) Reset(ids []int32) error { return nil }
func (f *fakeAI40Runtime) Close() error            { f.closed.Store(true); return nil }

func waitAI40Closed(t *testing.T, runtime *fakeAI40Runtime) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for !runtime.closed.Load() && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
}

func TestAI40WireObservationSizeAndHeader(t *testing.T) {
	var obs AssaultObservationV1
	var payload bytes.Buffer
	writeAI40Observation(&payload, &obs)
	want := (AssaultHeroFeatureSize+AssaultMaxEntities*AssaultEntityFeatures+AssaultGlobalFeatures)*4 +
		AssaultMaxEntities + AssaultActionKinds + AssaultMaxEntities + 4*AssaultMaxEntities
	if payload.Len() != want {
		t.Fatalf("observation bytes=%d, want %d", payload.Len(), want)
	}
	var frame bytes.Buffer
	if err := writeAI40Frame(&frame, ai40RequestInfer, []byte{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	rawFrame := append([]byte(nil), frame.Bytes()...)
	body, response, err := readAI40Frame(&frame)
	if err != nil || response != ai40RequestInfer || !bytes.Equal(body, []byte{1, 2, 3}) {
		t.Fatalf("round trip body=%v response=%d err=%v", body, response, err)
	}
	if binary.LittleEndian.Uint32(rawFrame[:4]) != 11 {
		t.Fatal("bad frame length")
	}
}

func TestAI40PythonExecutablePreservesSpaces(t *testing.T) {
	want := `E:\Tanat Online\ai40\.venv\Scripts\python.exe`
	t.Setenv("TANAT_AI40_PYTHON", `"`+want+`"`)
	if got := ai40PythonCommand(); got != want {
		t.Fatalf("python executable=%q, want %q", got, want)
	}
}

func TestAI40InvalidActionFallsBackTeamToAI20(t *testing.T) {
	env := NewAssaultEnv()
	defer env.Close()
	cfg := externalAssaultReset(71, 20)
	for i := range cfg.Controllers {
		cfg.Controllers[i] = AssaultControllerAI20
	}
	if _, err := env.Reset(cfg); err != nil {
		t.Fatal(err)
	}
	env.inst.mu.Lock()
	for i := 0; i < AssaultHeroCount/2; i++ {
		brain := env.brains[i]
		brain.aiVersion, brain.aiVersionSet = 40, true
	}
	env.inst.dota.botAIVersionByTeam[dotaTeamHuman] = 40
	runtime := &fakeAI40Runtime{actions: make([]HeroActionV1, AssaultHeroCount/2)}
	runtime.actions[0] = HeroActionV1{Kind: AssaultActionTeleport}
	env.inst.dota.ai40Runtime = runtime
	env.inst.dota.ai40StartAttempted = true
	env.server.botAI40BatchTickLocked(env.inst, 0)
	teamVersion := env.inst.dota.botAIVersionByTeam[dotaTeamHuman]
	brainVersions := make([]int, AssaultHeroCount/2)
	for i := 0; i < AssaultHeroCount/2; i++ {
		brainVersions[i] = botAIVersionForBrain(env.brains[i])
	}
	env.inst.mu.Unlock()
	waitAI40Closed(t, runtime)
	if teamVersion != 20 {
		t.Fatal("invalid neural action did not fall back team")
	}
	for i, version := range brainVersions {
		if version != 20 {
			t.Fatalf("brain %d did not latch AI-20 fallback", i)
		}
	}
}

func TestAI40LatencyFallsBackEveryNeuralTeamAndClosesRuntime(t *testing.T) {
	env := NewAssaultEnv()
	defer env.Close()
	cfg := externalAssaultReset(73, 20)
	for i := range cfg.Controllers {
		cfg.Controllers[i] = AssaultControllerAI20
	}
	if _, err := env.Reset(cfg); err != nil {
		t.Fatal(err)
	}
	env.inst.mu.Lock()
	for _, brain := range env.brains {
		brain.aiVersion, brain.aiVersionSet = 40, true
	}
	env.inst.dota.botAIVersionByTeam[dotaTeamHuman] = 40
	env.inst.dota.botAIVersionByTeam[dotaTeamElf] = 40
	runtime := &fakeAI40Runtime{actions: make([]HeroActionV1, AssaultHeroCount), latency: ai40LatencyLimit + time.Millisecond}
	env.inst.dota.ai40Runtime = runtime
	env.inst.dota.ai40StartAttempted = true
	env.server.botAI40BatchTickLocked(env.inst, 0)
	humanVersion := env.inst.dota.botAIVersionByTeam[dotaTeamHuman]
	elfVersion := env.inst.dota.botAIVersionByTeam[dotaTeamElf]
	env.inst.mu.Unlock()
	waitAI40Closed(t, runtime)
	if !runtime.closed.Load() || humanVersion != 20 ||
		elfVersion != 20 {
		t.Fatal("latency fallback did not latch both teams and close runtime")
	}
}

func TestAI40WrongActionCountFallsBackWithoutPanic(t *testing.T) {
	env := NewAssaultEnv()
	defer env.Close()
	cfg := externalAssaultReset(77, 20)
	for i := range cfg.Controllers {
		cfg.Controllers[i] = AssaultControllerAI20
	}
	if _, err := env.Reset(cfg); err != nil {
		t.Fatal(err)
	}
	env.inst.mu.Lock()
	for i := 0; i < AssaultHeroCount/2; i++ {
		env.brains[i].aiVersion, env.brains[i].aiVersionSet = 40, true
	}
	env.inst.dota.botAIVersionByTeam[dotaTeamHuman] = 40
	runtime := &fakeAI40Runtime{}
	env.inst.dota.ai40Runtime = runtime
	env.inst.dota.ai40StartAttempted = true
	env.server.botAI40BatchTickLocked(env.inst, 0)
	version := env.inst.dota.botAIVersionByTeam[dotaTeamHuman]
	env.inst.mu.Unlock()
	waitAI40Closed(t, runtime)
	if version != 20 || !runtime.closed.Load() {
		t.Fatal("wrong action count did not trigger fail-closed fallback")
	}
}

func TestAI40SidecarIntegration(t *testing.T) {
	if os.Getenv("TANAT_AI40_INTEGRATION") != "1" {
		t.Skip("set TANAT_AI40_INTEGRATION=1 with a real exported model")
	}
	runtime, err := startAI40Sidecar()
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.Close()
	var obs AssaultObservationV1
	obs.ActionMask.Kinds[AssaultActionWait] = 1
	actions, latency, err := runtime.Infer([]int32{900001}, []AssaultObservationV1{obs})
	if err != nil {
		t.Fatal(err)
	}
	if len(actions) != 1 || actions[0].Kind != AssaultActionWait {
		t.Fatalf("sidecar action=%+v, want one masked wait", actions)
	}
	if latency > ai40LatencyLimit {
		t.Fatalf("sidecar latency=%s, limit=%s", latency, ai40LatencyLimit)
	}
}

func TestAI40LiveBatchAppliesValidMove(t *testing.T) {
	env := NewAssaultEnv()
	defer env.Close()
	if _, err := env.Reset(externalAssaultReset(79, 20)); err != nil {
		t.Fatal(err)
	}
	env.inst.mu.Lock()
	brain := env.brains[0]
	brain.aiVersion, brain.aiVersionSet = 40, true
	env.inst.bots[brain.c.objID] = brain
	env.inst.dota.botAIVersionByTeam[dotaTeamHuman] = 40
	runtime := &fakeAI40Runtime{actions: []HeroActionV1{{Kind: AssaultActionMove, Direction: 0, Distance: 0}}}
	env.inst.dota.ai40Runtime = runtime
	env.inst.dota.ai40StartAttempted = true
	startX, startY := brain.c.x, brain.c.y
	env.server.botAI40BatchTickLocked(env.inst, 0)
	validMove := brain.c.hasDest && brain.c.destX > startX && brain.c.destY == startY
	stillAI40 := botAIVersionForBrain(brain) == 40
	env.inst.mu.Unlock()
	if !validMove {
		t.Fatalf("valid AI-40 move was not applied: start=(%g,%g) destination=(%g,%g)",
			startX, startY, brain.c.destX, brain.c.destY)
	}
	if !stillAI40 {
		t.Fatal("valid AI-40 action caused fallback")
	}
}

func TestLoadAI40ManifestValidatesHashesAndModel(t *testing.T) {
	dir := t.TempDir()
	model := dir + string(filepath.Separator) + "ai40.onnx"
	if err := os.WriteFile(model, []byte("model"), 0600); err != nil {
		t.Fatal(err)
	}
	manifestPath := dir + string(filepath.Separator) + "ai40.manifest.json"
	manifest := fmt.Sprintf(`{"version":"AI-40-v2","schema_hash":"%x","reward_hash":"%x","model":"ai40.onnx","inputs":["hero","entities","global","entity_mask","h","c"],"outputs":["kind","target","direction","distance","value","next_h","next_c"],"recurrent_size":256}`,
		AssaultSchemaHashV1, AssaultRewardHashV2)
	if err := os.WriteFile(manifestPath, []byte(manifest), 0600); err != nil {
		t.Fatal(err)
	}
	got, gotModel, err := loadAI40Manifest(manifestPath)
	if err != nil || got.RecurrentSize != 256 || gotModel != model {
		t.Fatalf("manifest=%+v model=%q err=%v", got, gotModel, err)
	}
	bad := strings.Replace(manifest, fmt.Sprintf("%x", AssaultSchemaHashV1), strings.Repeat("0", 64), 1)
	if err := os.WriteFile(manifestPath, []byte(bad), 0600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := loadAI40Manifest(manifestPath); err == nil {
		t.Fatal("schema mismatch was accepted")
	}
}
