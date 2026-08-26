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

## AI-42 pre-training gates

### Production AI-42 dataset

The production collector is Go-backed, training-free, resumable, and uses the
frozen stratified schedule in `config/ai42_dataset01.json`: 320 matches, 40
validation matches, 4,500 ticks (15 simulated minutes), and eight bounded
workers/in-flight matches. From `server`, run:

```powershell
$env:PYTHONPATH = "$PWD\ai40\src"
python -m tanat_ai40.build_ai42_dataset_go `
  .\ai40\data\ai42_dataset01 `
  --config .\ai40\config\ai42_dataset01.json
```

The destination is immutable; an interrupted run resumes only from the exact
staging contract. Use `--python-fallback` only for an explicit reference-path
diagnostic, never for the production dataset.

To compact an existing immutable Go v1 generation without decompressing its
payloads, publish to a new destination:

```powershell
$env:PYTHONPATH = "$PWD\ai40\src"
python -m tanat_ai40.compact_ai42_dataset `
  .\ai40\data\ai42_dataset-v1 `
  .\ai40\data\ai42_dataset-v2
```

AI42 Go v2 keeps `trajectory_hashes` and compressed payload bytes unchanged but
stores only `first_step` and the versioned implicit lineage schema per match.
Call `dataset.match_metadata(match_id, derive=True)` when legacy tick or
recurrent-lineage matrices are explicitly needed.

AI-42 behavior cloning is intentionally validation-only until training is
explicitly authorized. Build the environment from the same source revision,
capture and strictly reload a short ten-hero AI-30 episode, then run the
native Go preflight against that dataset. The wrapper requires both the Go
toolchain and a Python interpreter that can import Torch; it has no Python
preflight fallback:

```powershell
$env:PYTHONPATH = "$PWD\ai40\src"
python -m tanat_ai40.smoke_ai42_dataset .\assaultenv.exe <new-dataset-dir> `
  --seed 4242 --max-steps 8
$warmStart = '<accepted-checkpoint.pt>'
$run = '<preflight-run-dir>'
$profile = Join-Path $run 'class_profile_ai42.json'
$report = Join-Path $run 'preflight_report.json'
.\run-ai42-bc-preflight.ps1 `
  --dataset <new-dataset-dir> `
  --dataset-hash <lower-case-dataset-manifest-sha256> `
  --profile $profile `
  --profile-hash <lower-case-profile-sha256> `
  --warm-start $warmStart `
  --output $run `
  --report $report `
  --supervision-controller 3 `
  --allow-warm-start-dataset-change `
  --device cuda
```

On Linux, use the same native flags with `./run-ai42-bc-preflight.sh`:

```bash
export PYTHONPATH="$PWD/ai40/src"
./run-ai42-bc-preflight.sh \
  --dataset <new-dataset-dir> \
  --dataset-hash <lower-case-dataset-manifest-sha256> \
  --profile <preflight-run-dir>/class_profile_ai42.json \
  --profile-hash <lower-case-profile-sha256> \
  --warm-start <accepted-checkpoint.pt> \
  --output <preflight-run-dir> \
  --report <preflight-run-dir>/preflight_report.json \
  --supervision-controller 3 \
  --allow-warm-start-dataset-change \
  --device cuda
```

The wrappers add the default preflight config, pass the exact selected
interpreter as `--torch-python`, and own a fixed positive `--worker-timeout`
of `5m`. They reject caller-supplied `--torch-python`, legacy
`--worker-command`/`--worker-arg`, `--worker-timeout`, wrapper-owned
`--config`, and duplicate control flags before invoking Go. The Go command
owns full durable-data verification and report publication. Python is only
the fixed bounded `-m tanat_ai40.torch_probe_worker_ai42` probe launched by
Go; it never owns the report or a production hot path. An external
`--profile` always requires its explicit `--profile-hash`, and `--report` must
be located under `--output`. Paths and flags are passed as separate process
arguments, including paths containing spaces. The native command validates
the durable dataset, recurrent batching, control/action losses, finite
gradients, clipping and checkpoint roundtrip without calling
`optimizer.step`; it atomically publishes the profile, batch bundle and
report. `train_ai42_bc` no longer runs a Python preflight: invoking it without
`--execute` fails closed with the native-wrapper instruction. ONNX is not
required for PyTorch/CUDA training and remains a later production-inference
gate.

`--supervision-controller` is repeatable. DAgger generations use controller
`3` for the AI-42 candidate and controller `2` for the AI-30 opponent, so
preflight and training select only controller `3`. This preserves complete
recurrent sequences for the candidate while preventing ordinary opponent
labels from overwhelming the intervention/correction signal. Omitting the
flag retains the backward-compatible all-controller profile.
The dataset-change flag is required only when an accepted model is carried
into a new immutable DAgger generation. The worker still validates the whole
source checkpoint and proves that optimizer state, RNG state and cursor were
not restored.

Class-profile v4 keeps inverse-frequency balancing for semantic heads, but
forces every `target` class weight to exactly `1`. Target indices are
exchangeable entity slots rather than semantic classes; weighting them by
slot frequency would break the actor's target permutation equivariance.
There is no manual class-weight override path.

### Current AI-42 workflow

AI-42 uses one production configuration:
[`config/ai42_bc_training.json`](config/ai42_bc_training.json). Historical Q-series
configs were experiments against a repeatedly inspected validation set and are
not part of the product surface.

The actor is deliberately compact: width 192, two entity-attention layers, and
an LSTM state per hero. Non-skill actions are classified from recurrent state.
Skill1-Skill4 are scored from their corresponding ability token plus recurrent
context, preserving slot-specific cooldown/readiness information. The actor
exports only the heads consumed by protocol v13: control, kind, target, offset,
anchor, and recurrent state. Timing heads, a privileged centralized critic,
and a separate macro network remain outside the runtime contract. PPO adds a
small training-only value head over recurrent belief state; its value gradient
is detached from the actor, so critic regression cannot erase BC behavior.

Class balancing is derived once from the immutable training split profile.
Manual class-weight overrides, repeated-kind curricula, and partial-head
training modes are intentionally unsupported. Per-class recall remains visible
as diagnostics; promotion uses validation loss, end-to-end action accuracy,
aggregate head floors, and navigation distance. An untouched match schedule is
the final deployment gate; training does not tune against those matches.

Run native preflight from `server` before a checkpoint-backed training run:

```powershell
.\run-ai42-bc-preflight.ps1 `
  -Config .\ai40\config\ai42_bc_training.json `
  -Dataset <dataset-generation> `
  -WarmStart <checkpoint> `
  -Output <preflight-directory>
```

The compact v2 architecture is checkpoint-incompatible with the removed
experimental actor. Its first baseline therefore starts from the deterministic
seed without `--warm-start`; later iterations use the native preflight above.
Run the explicitly authorized five-minute BC iteration:

```powershell
$env:PYTHONPATH = "$PWD\ai40\src"
python -m tanat_ai40.train_ai42_bc `
  --config .\ai40\config\ai42_bc_training.json `
  --dataset <dataset-generation> --output <run-directory> `
  --device cuda --max-optimizer-seconds 300 --execute
```

The deterministic batch plan persists its hash and exact cursor. Resume with
`--resume <run-directory>\latest.pt`. Accepted candidates are immutable
generations selected through the atomically replaced `accepted_pointer.json`.

Validation uses the same frozen sequences and class profile as training, but
packs independent sequences with `training.validation_batch_size` (256 in the
production config). This does not change recurrent boundaries or checkpoint
lineage. On the RTX 5090 reference host it reduced one 16-match validation
probe from roughly 154 seconds at batch 8 to 17.2 seconds at batch 256 while
keeping loss within `1.4e-8` and action/navigation metrics unchanged.

Retained checkpoints can be ranked without reloading the dataset for every
candidate:

```powershell
python -m tanat_ai40.evaluate_ai42_bc_checkpoints `
  --config .\ai40\config\ai42_bc_training.json `
  --dataset <dataset-generation> --profile <class-profile.json> `
  --checkpoint <step-000100.pt> --checkpoint <step-000200.pt> `
  --output <curve-report.json> --selection-matches 1 `
  --evaluation-batch-size 256 --patience 3 --device cuda
```

`--selection-matches` is a cheap, deterministic ranking probe. A selected
candidate must still pass the full frozen validation probe and the existing
promotion gate; the proxy never promotes a model by itself.

DAgger collection remains the next data stage. It should collect policy-reached
states with AI-30 intervention labels, alternate sides, and freeze policy,
environment, writer, and ONNX hashes in generation provenance. Do not tune the
architecture or promotion gate against the same validation matches.

Use margin interventions only when low confidence is a reliable proxy for a
bad action. A policy that is confidently wrong should use the deterministic
periodic mixture instead. With five controlled heroes and a five-tick period,
one different hero yields its decision to AI-30 each tick, so every hero gets
one correction per simulated second without replacing the whole team at once:

```powershell
python -m tanat_ai40.collect_dagger_generation_ai42 <checkpoint.pt> `
  --config .\ai40\config\ai42_bc_training.json `
  --env .\assaultenv.exe --writer .\ai42daggerwriter.exe `
  --output <dagger-dataset> --onnx <frozen-actor.onnx> `
  --seed 52000 --matches 32 --workers 8 --max-steps 4500 `
  --intervention-strategy periodic --intervention-gap-ticks 5 `
  --split-seed 4242 --validation-fraction 0.25 --device cpu
```

The intervention strategy and its parameters are part of the immutable
collector-v2 schedule. The legacy default remains `margin` with threshold
`0.08` and a five-tick per-hero cooldown.

Behavior cloning and DAgger are bootstrap stages, not the final optimization
loop. Continue an accepted BC or PPO generation with fresh on-policy recurrent
rollouts and clipped PPO:

```powershell
python -m tanat_ai40.train_ai42_ppo <champion-or-resume.pt> `
  --config .\ai40\config\ai42_bc_training.json `
  --env .\assaultenv.exe --output <ppo-run> --device cuda `
  --workers 8 --rollout-steps 64 --train-seconds 900 `
  --past-opponent-fraction 0.2
```

Eighty percent of the default worker batch is current-policy mirror self-play.
The remaining twenty percent uses the immutable input checkpoint as a frozen
past opponent, with candidate sides alternated. The learner replays complete
per-hero sequences, computes terminal-aware GAE, and stores actor, critic,
optimizer, lineage, RNG, and metrics in an atomic PPO checkpoint. Reward-free
opening windows with an untrained zero critic perform no optimizer step.

Never promote from training loss. Compare the candidate directly with its
champion on alternating sides and identical seed policy:

```powershell
python -m tanat_ai40.evaluate_ai42_pair `
  <ppo-run>\final.pt <champion.pt> `
  --config .\ai40\config\ai42_bc_training.json `
  --env .\assaultenv.exe --matches 40 --workers 8 --device cuda `
  --output <ppo-run>\paired-evaluation.json
```

The paired report records wins/losses/draws, side splits, rewards, invalid
actions, issued actions, match duration, and immutable checkpoint lineage.
The existing `tanat-ai42-eval` remains the independent AI-30 anchor.

The complete default 15-minute train/evaluate iteration is one command:

```powershell
python -m tanat_ai40.cycle_ai42_ppo <champion.pt> `
  --config .\ai40\config\ai42_bc_training.json `
  --env .\assaultenv.exe --output <cycle-directory> --device cuda
```

It evaluates candidate and champion on the same AI-30 seeds and writes a
promotion recommendation. It never replaces a champion automatically.

ONNX is the deployment/runtime format. Export validates the exact actor-only
interface and PyTorch/ONNX parity; CUDA execution must fail closed if ONNX
Runtime silently falls back to CPU.

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
  --ai30-matches 50 --workers 64 --group-size 64 --steps 256 `
  --max-steps 4500 --device cuda --minibatch-size 8192 `
  --env-gomaxprocs 16 --output ai40\checkpoints\async-100
```

Protocol v4 keeps all environments in one batched Go process per rollout
group. The default 64-worker group therefore uses one `assaultenv.exe`, one
STEP request and one structure-of-arrays result frame per tick. Splitting the
workers with `--group-size 32` uses two vector processes when lower rollout
latency matters more than the one-process constraint.

All groups use the same frozen actor during one 256-step horizon. By default,
the next rollout overlaps the current PPO update using a separate actor copy.
This introduces at most one PPO update of policy lag; the behavior log
probabilities are retained for PPO importance ratios and the observed lag is
written to every metrics row. Use `--no-pipeline` for strictly sequential
collection and learning. CUDA training uses BF16 and 8,192-sample minibatches
by default. PPO forward graphs use `torch.compile`; pass `--no-bf16` or
`--no-compile` for diagnostics. Windows compile support is supplied by the
`triton-windows` project dependency.

## Long training campaign and AI-30 win rate

Use the campaign wrapper for unattended training. It resumes after Ctrl+C,
freezes a checkpoint after every stage, evaluates it on unseen deterministic
matches against AI-30, and maintains `best.pt`:

```powershell
.\run-ai40-training.ps1
```

With no arguments, the wrapper uses `mixed-100/latest.pt` as its initial model,
writes the campaign to `ai40/checkpoints/long-001`, and selects the measured
64 workers in one vector process, 256-step rollouts, 4,500-tick (15-minute)
matches, 8,192-sample BF16 PPO minibatches and adaptive evaluations. Evaluation starts with
50 matches, expands to 200 at 40% win rate, and uses 500 confirmation matches
at 55%. Repeating the same bare command resumes `long-001`. Use
`-FromScratch -RunName <new-name>` only when a
new randomly initialized policy is intentional. Add `-DryRun -SkipBuild` to
print the resolved defaults without starting training.

The default campaign contains 100 stages. Each stage adds 100 mirror matches
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

## AI-41 strategic historical self-play

AI-41-v5 uses protocol v10 for training and protocol v11 for evaluation alongside
the unchanged AI-40 protocol v4 and legacy AI-41 protocols v5-v7. Each hero
receives a `[4, 40]` ability tensor containing rank/readiness, cooldown and
mana, targeting/range/AoE, authored effect categories (damage, heal, control,
mobility, summon, channel, buffs/debuffs and vision), magnitude/duration and
toggle state. The policy also has a hero-ID embedding and separate
target/offset/anchor contexts for every action kind; skill actions receive
the encoded record for their own slot.

Position actions use an OpenAI-Five-style row-major 9x9 grid. The 81 cells are
spaced three world units apart and cover `[-12,+12]` on both axes; cell 40 is
the caster's current position. For `Move`, the former distance factor is now a
15-way navigation selector:

- `0`: use the local 9x9 offset;
- `1`, `2`: own base and enemy base;
- `3..6`: north lane at 20/40/60/80% progress from the hero's own side;
- `7..10`: the same four centre-lane anchors;
- `11..14`: the same four south-lane anchors.

The server resolves anchors relative to the hero's team and runs A* to the
resulting map point. Repeating an anchor continues the existing route instead
of rebuilding it every tick. Point-target abilities use the same local grid;
unit-target abilities still use the 96-slot entity head. Training metrics add
`global_anchor_move_rate`, `global_anchor_steps`, and `move_steps`.

Training resets independently randomize a `2-1-2` north/centre/south assignment
for each team. `global[8]` marks the curriculum active, `global[9:12]` is the
hero's one-hot lane, `global[12]` marks the hero outside its 30-unit lane
corridor. The random 6-to-10-minute cutoff is intentionally hidden from the
policy; it only sees `global[8]` switch to zero when the assignment expires. An
alive policy hero outside its lane loses `0.15` reward per simulated second
until that cutoff. Scripted AI-20/AI-30 opponents are exempt. Evaluation
protocol v9 zeros these fields and disables the penalty, so win rate measures
play without a forced lane assignment. Per-update metrics record
`wrong_lane_rate`, its numerator and the active sample count.

On Windows, prepare and start the default campaign with:

```powershell
.\run-ai41-training.ps1
```

On Ubuntu, prepare and start the default campaign with:

```bash
./run-ai41-training.sh
```

The bare launcher starts `ai40/checkpoints/ai41-tanat-reward-stable-002` from
the promoted stage-002 checkpoint of `ai41-tanat-reward-stable-001`. Its Adam
state and cumulative counters are reset, so the smaller candidate updates are
measured as a separate campaign. The original stage-005 file remains immutable
and is the first historical opponent.

Each default stage runs 50 current-policy mirror matches, 25 matches against
AI-30 and 50 matches against the historical pool. PPO samples are retained only
for the current-policy side; a frozen policy owns the other side and has an
independent recurrent state in the same vectorized environment process. Accepted
former `latest` checkpoints enter the bounded eight-model pool.

The default conservative proposal profile uses Adam `5e-5`, one PPO epoch per rollout and an
approximate-KL guard of `0.01`. It keeps each candidate close to the promoted
checkpoint, avoiding the observed policy-collapse regime
(KL near `0.2` and clipping near `60%`). `training_metrics.jsonl` records
`ppo_epochs_completed` and `ppo_early_stopped`; pass `-LearningRate`,
`-PpoEpochs` or `-TargetKl` to deliberately use another profile. A campaign
locks these values after its first update, so use a new run name for a different
schedule.

Credit assignment uses a 1,200-second discount horizon and a 180-second GAE
horizon. The default rollout is 1,024 ticks (204.8 simulated seconds), long
enough to observe that trace rather than truncating it after the former 51.2
seconds. Non-terminal shaping is multiplied by `0.6^(elapsed/600 seconds)`.
A timeout/draw applies `-2` after zero-sum and team-spirit correction, so the
same penalty on both teams cannot cancel out.

Every stage reports deterministic win rate and score against AI-30, the frozen
stage-005 anchor, and the remaining historical pool. The candidate is compared
with the currently promoted checkpoint on identical seeds. `latest.pt` is
promoted only when the composite score does not regress and no category loses
more than five percentage points; otherwise `latest.pt` is restored from
`promoted.pt`. Full candidate/reference suites and the decision are stored in
the stage evaluation JSON and summarized in `metrics.csv`.

The Ubuntu AI-41 runner defaults to two independent rollout groups of 32
matches with 16 Go execution threads each. Chase routes use a validated
weighted A* path, reuse their final segment for moving targets, and hold a live
attack target for at least five simulation ticks.

To select a different navigation source explicitly:

```bash
ai40/.venv/bin/python -m tanat_ai40.migrate_ai41_navigation \
  ai40/checkpoints/ai41-lanes-001/latest.pt \
  --output ai40/checkpoints/ai41-nav-custom-bootstrap/latest.pt
./run-ai41-training.sh --run-name ai41-nav-custom \
  --resume ai40/checkpoints/ai41-nav-custom-bootstrap/latest.pt
```

AI-41 strategic checkpoints use `AI-41-v5-tanat-reward-selfplay`, the V4 observation
hash and RewardV5. RewardV5 preserves OpenAI Five's `-0.16` last-hit correction
and adds a separate Tanat calibration of `+0.24`. A standard 3-melee + 1-ranged
wave therefore averages `+0.4` raw reward per hero last hit. These checkpoints
fail closed if passed to an AI-40 evaluator. Live ONNX activation remains on
the AI-40 contract until an AI-41 sidecar/export contract is installed.

On a Ryzen 9 9950X3D with an RTX 5090, two 32-worker vector groups complete a
warm 64-worker, 256-tick AI-41-v3 navigation rollout in about 1.65 seconds, or
roughly 9,900 environment steps/second. Actor inference and the packed pinned
transfer take about 0.56 seconds; batched Go simulation and result transfer
take about 1.05 seconds at the initial 0.47% anchor exploration rate. Strategic
historical inference and the default 1,024-tick horizon add work beyond that
legacy benchmark. GAE is computed per rollout group so the
large entity tensor is not copied into an intermediate time-major batch. Actor
forward and action sampling use fixed-shape CUDA graphs; sampling receives
freshly generated exponential noise on every tick so graph capture cannot
freeze its random choices. The first rollout additionally pays about 1.2
seconds of graph compilation. BF16 with 8,192-sample minibatches keeps a short
eager PPO update near 1.9 seconds for the larger navigation heads. Per-update
metrics include
`actor_inference_seconds` and `environment_step_seconds` for checking this
balance on other machines. Campaigns compile the fixed-shape actor and sampler
by default but keep PPO eager: compiling the learner has a large cold-start cost
on every campaign stage. `--compile-learner` remains available for long stages;
`--no-compile` disables actor compilation as well.

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
