# AI-40 training

This directory contains the Python side of the versioned `AssaultEnvV1`
training boundary. The Go process owns the authoritative match and the Python
process owns rollout collection and learning.

## Build and verify the environment

From `server` in PowerShell:

```powershell
go build -o assaultenv.exe ./cmd/assaultenv
python ai40/smoke_env.py assaultenv.exe
$env:PYTHONPATH = "$PWD\ai40\src"
$env:TANAT_ASSAULTENV = "$PWD\assaultenv.exe"
python -m unittest discover -s ai40/tests -v
```

The transport test needs NumPy only. One `assaultenv.exe` process represents
one rollout worker; launch several processes rather than sharing one process
between threads.

Headless controller values are `0` for a generic external policy, `1` for
AI-20, `2` for the scripted AI-30 teacher/opponent and `3` for AI-40. Mixed
controller teams remain supported, but `tanat-ai40-train` and
`tanat-ai40-eval` deliberately reset all ten slots to controller `3`: five
AI-40 heroes play against five AI-40 heroes using the same current policy
weights. Recurrent state is still independent per hero. The ten-avatar roster
is shuffled between the two sides on every reset so an avatar does not become
specialized to one faction. The live server selects profiles independently with
`TANAT_DOTA_BOT_AI_TEAM1` and `TANAT_DOTA_BOT_AI_TEAM2`; AI-30 is the scripted
default when those variables are unset.

## Simulation speed

Headless training has no real-time speed multiplier. Every `Step` advances the
authoritative world by exactly 200 ms and returns as soon as the CPU completes
the work, so `TANAT_DOTA_SIM_SPEED` and a live-client setting such as x20 do not
apply. Run rollout workers as fast as possible without changing the 200 ms
step. On the measured Ryzen 9 9950X3D machine, 64 workers split into asynchronous
groups of 32 give the best full-training throughput; larger worker counts no
longer improve it.

## Training environment

Use Python 3.12. Create an isolated environment and install the current PyTorch
CUDA wheel selected for the training machine from the official PyTorch wheel
index, then install this package:

```powershell
py -3.12 -m venv ai40\.venv
ai40\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install the CUDA PyTorch build appropriate for the RTX 5090 first.
python -m pip install -e .\ai40
# ONNX export/runtime verification support:
python -m pip install -e ".\ai40[onnx]"
```

Run a small PPO/GAE verification update:

```powershell
tanat-ai40-smoke --env .\assaultenv.exe --steps 32 --workers 2 --updates 1 --device cuda
```

Run resumable multi-process training (one independent Go process per worker):

```powershell
tanat-ai40-train --env .\assaultenv.exe --workers 12 --steps 256 `
  --updates 1000 --device cuda --output ai40\checkpoints\run-001

tanat-ai40-train --env .\assaultenv.exe --workers 12 --steps 256 `
  --updates 1000 --device cuda --output ai40\checkpoints\run-001 `
  --resume ai40\checkpoints\run-001\latest.pt
```

These training commands always use mirror self-play: the checkpoint being
updated controls both teams. There is no fixed scripted opponent in this mode.

For a fixed 100-match mixed curriculum (50 mirror matches and 50 against the
scripted AI-30), run:

```powershell
tanat-ai40-train-matches --env .\assaultenv.exe --ai40-matches 50 `
  --ai30-matches 50 --workers 32 --steps 256 --max-steps 4500 --device cuda `
  --minibatch-size 2048 --env-gomaxprocs 1 `
  --output ai40\checkpoints\mixed-100
```

The preferred runner collects fixed-policy rollout groups independently, so a
fast group can advance without waiting at a global per-tick barrier for every
simulation:

```powershell
tanat-ai40-train-async --env .\assaultenv.exe --ai40-matches 50 `
  --ai30-matches 50 --workers 64 --group-size 32 --steps 256 `
  --max-steps 4500 --device cuda --minibatch-size 2048 `
  --env-gomaxprocs 1 --output ai40\checkpoints\async-100
```

All groups use the same frozen actor during one 256-step horizon. By default,
the next rollout overlaps the current PPO update using a separate actor copy.
This introduces at most one PPO update of policy lag; the behavior log
probabilities are retained for PPO importance ratios and the observed lag is
written to every metrics row. Use `--no-pipeline` for strictly sequential
collection and learning.

## Long training campaign and AI-30 win rate

Use the campaign wrapper for unattended training. It resumes after Ctrl+C,
freezes a checkpoint after every stage, evaluates it on unseen deterministic
matches against AI-30, and maintains `best.pt`:

```powershell
.\run-ai40-training.ps1
```

With no arguments, the wrapper uses `mixed-100/latest.pt` as its initial model,
writes the campaign to `ai40/checkpoints/long-001`, and selects the measured
64-worker/32-group settings, 256-step rollouts, 4,500-tick (15-minute) matches,
2,048-sample PPO minibatches and adaptive evaluations. Evaluation starts with
50 matches, expands to 200 at 40% win rate, and uses 500 confirmation matches
at 55%. Repeating the same bare command resumes `long-001`. Use
`-FromScratch -RunName <new-name>` only when a
new randomly initialized policy is intentional. Add `-DryRun -SkipBuild` to
print the resolved defaults without starting training.

The default campaign contains ten stages. Each stage adds 100 mirror matches
and 100 matches against AI-30, then runs the adaptive evaluation with AI-40
alternating between factions. With 4,500 simulation ticks per match this is at
most approximately 67.5 million new trainable hero-steps before matches that
finish naturally are accounted for. Stop it safely with Ctrl+C and run the
same command to continue. To start from random weights, pass `-FromScratch` and
use a new run name.

Useful overrides can be passed directly or after the wrapper parameters:

```powershell
.\run-ai40-training.ps1 -RunName long-001 -Stages 20 `
  -Resume .\ai40\checkpoints\mixed-100\latest.pt `
  --stop-win-rate 0.65 --stop-ci-low 0.55
```

The run directory contains:

- `latest.pt`: most recent resumable model and optimizer state;
- `best.pt`: checkpoint with the highest lower 95% Wilson win-rate bound;
- `checkpoints/stage-NNN.pt`: immutable checkpoints after each stage;
- `evaluations/stage-NNN.json`: full evaluation results;
- `metrics.csv` and `metrics.jsonl`: stage-level win rate, score rate, confidence
  interval, evaluation level, faction results, invalid-action rate, reward and
  match duration;
- `training_metrics.jsonl`: policy/value loss, entropy, approximate KL, clipping,
  gradient norm, explained variance, CUDA peak memory, rollout/PPO time,
  throughput and cumulative training outcomes for every update;
- `campaign.json`: atomic resume state and current best stage.

The campaign stops early only when both the observed win rate reaches 60% and
the lower 95% confidence bound reaches 50%. Training match outcomes are not
used as the headline win rate because weights change during those matches;
only the frozen post-stage evaluation determines `best.pt`.

Only AI-40-controlled hero samples enter PPO in teacher matches. AI-30 changes
faction between matches. The learner keeps rollout tensors in host memory and
moves one minibatch at a time to CUDA; `--minibatch-size` is the primary
RAM/VRAM control. To continue an interrupted curriculum without repeating
completed matches, add `--resume ai40\checkpoints\mixed-100\latest.pt`.

On a Ryzen 9 9950X3D with an RTX 5090, the measured mixed-training default is
64 workers in groups of 32, `--minibatch-size 2048` and `--env-gomaxprocs 1`.
Actions are packed
as one NumPy batch, observations are decoded as structured zero-copy NumPy
views, target masks stay vectorized on CUDA, and the Go runner reuses one
fixed-size result encoder. The same 32-worker short benchmark improved from
about 845 to 1,995 environment steps/second after the transport changes. The
asynchronous 64-worker short rollout reaches about 3,107 environment
steps/second. A mixed 128-match, 3,000-tick benchmark completed 2.88 million
trainable hero-steps in 227.3 seconds (12,670 hero-steps/second), about 41% more
throughput than the comparable synchronous 32-worker run. Tests at 80 and 96
workers did not improve full-training throughput because PPO time and CPU
contention offset the larger rollout batch.

Headless participants suppress Battle-client packet encoding and use a
zero-buffer discard connection. Their authoritative combat state and object
trackers are retained, but rendering-only AMF/SYNC packets are not serialized
for clients that do not exist. A 9,000-tick worst-case mirror run measured
31.1 MiB working set and 65.9 MiB private commit for one `assaultenv` process;
16 concurrent late-match workers measured 24.4 MiB average and 28.1 MiB maximum
working set each. Keep 100 MiB per worker as the regression ceiling.

Evaluate a checkpoint with deterministic argmax actions:

```powershell
tanat-ai40-eval ai40\checkpoints\run-001\latest.pt --env .\assaultenv.exe `
  --matches 16 --workers 4 --device cuda
```

Export a checkpoint to ONNX and write its manifest:

```powershell
$env:PYTHONUTF8 = "1" # required by PyTorch exporter on legacy Windows consoles
python -m tanat_ai40.export_onnx ai40\checkpoints\run-001\latest.pt `
  --output ai40\checkpoints\ai40.onnx
```

## Run AI-40 in a live Assault match

The live server runs one isolated CPU ONNX Runtime sidecar per match. Install
the package and ONNX Runtime into the selected Python environment, then set the
model and team profiles before starting the battle server:

```powershell
python -m pip install -e ".\ai40[onnx]"
$env:PYTHONPATH = "$PWD\ai40\src"
$env:TANAT_AI40_PYTHON = "$PWD\ai40\.venv\Scripts\python.exe"
$env:TANAT_AI40_MANIFEST = "$PWD\ai40\checkpoints\ai40.manifest.json"
$env:TANAT_DOTA_BOT_AI_TEAM1 = "AI-40"
$env:TANAT_DOTA_BOT_AI_TEAM2 = "AI-40"
```

`TANAT_AI40_PYTHON` is an executable path and may contain spaces. The server
validates the observation hash, reward hash, tensor names, recurrent size and
model file before activation. Inference runs at 5 Hz in one batch for all AI-40
heroes. Missing/incompatible models, protocol errors, NaN/Inf, invalid actions,
or inference over 150 ms latch the affected neural team (all neural teams for a
runtime failure) to AI-20 for the rest of the match. Item purchases and skill
levelling remain scripted.

With `TANAT_BOT_TELEMETRY` enabled, JSONL includes `ai40_action` events with
model ID, selected factors, acceptance and latency, plus `ai40_fallback` events
with the exact reason.

Run the opt-in real sidecar integration test against an exported model:

```powershell
$env:TANAT_AI40_INTEGRATION = "1"
go test ./internal/battleserver -run TestAI40SidecarIntegration -v
```

## V1 schema

- 10 hero slots, shared policy weights and separate recurrent states.
- Hero tensor: `[10, 32]`.
- Fog-filtered entity tensor: `[10, 96, 16]`.
- Centralized team/value tensor: `[10, 32]`.
- Action heads: kind, entity target, 16-way direction, 3 distance bins.
- Server masks: action kinds, attack targets, and skill targets.
- One environment step is one authoritative 200 ms tick (5 Hz policy).
- Each `Reset` creates an in-memory account/economy store. Training never
  writes persistent accounts, currency, ratings, fight logs, or telemetry.

`AssaultSchemaHashV1` and `AssaultRewardHashV2` are embedded in every response,
checkpoint and ONNX manifest. Change their source strings whenever tensor or
reward meaning changes; old clients/checkpoints then fail closed instead of
silently training against a different contract.

## Current training scope

V3 adds explicit AI-40 mirror self-play, randomized side assignment for the
complete hero roster and records `training_mode=ai40_mirror_self_play` plus the
ten AI-40 controllers in every checkpoint. V2 added parallel rollout processes, resumable optimizer checkpoints,
deterministic evaluation and the expanded reward contract: XP, bronze,
spending, HP/mana potential, death, hero kills, creep last hits, structure
damage/destruction, terminal win, zero-sum correction and 20% team spirit.
Scripted skill upgrades and item purchases are retained.

Historical policy-pool opponent sampling, DAgger dataset storage, mana/HP time decay and
Aegis/boss hooks remain follow-up layers on the same boundary. AI-30 teacher
behavior, live ONNX inference and latency fallback are implemented.
