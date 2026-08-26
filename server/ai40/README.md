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

### Executable five-minute AI-42 BC workflow

The explicit training path is implemented by
[`src/tanat_ai40/train_ai42_bc.py`](src/tanat_ai40/train_ai42_bc.py) and uses
[`config/ai42_bc_training.json`](config/ai42_bc_training.json). It requires
the same source revision on the training machine and the checkout: commit
`e0056a2` atop `80c3600`. It requires `--execute`, uses CUDA, and caps
optimizer time at 300 seconds. The first run used the immutable dataset at
`E:\code\Tanat Online\tanat\server\ai42_datasets\clone-v13-dataset01-v2`.
From `server` in PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\ai40\src"
$dataset = 'E:\code\Tanat Online\tanat\server\ai42_datasets\clone-v13-dataset01-v2'
$run = 'E:\code\Tanat Online\ai42_runs\bc5m-e0056a2-01'
python -m tanat_ai40.train_ai42_bc `
  --config .\ai40\config\ai42_bc_training.json `
  --dataset $dataset --output $run --device cuda `
  --batch-size 8 --max-optimizer-seconds 300 --execute
```

The batch plan is deterministic: match IDs are SHA-256 hash-ranked and
scenario-stratified. Only batches with effective supervised rows are eligible
for optimizer updates. Checkpoints persist the plan hash and exact
`batch_cursor`; resume with `--resume <run>\latest.pt` to continue the same
stream. Periodic and final candidates are written to `latest.pt`. An accepted
candidate is written as an immutable checkpoint generation, fully digest
validated, and promoted through atomic `accepted_pointer.json`. `accepted.pt`
and `best.pt` are compatibility aliases, not the authority.
The config sets `max_steps=1000`; the actual run stopped at step 131 because
the 300-second deadline was reached.

First accepted run: `E:\code\Tanat Online\ai42_runs\bc5m-e0056a2-01`, CUDA,
131 steps in 300 seconds. Accepted generation SHA-256:
`fe3c111789c9594a76a5ab7f125566ebc2ceae5642b94243c44b44c9c9482f3c`.
Validation loss changed from `13.2954028457` to `12.5962044984` (`-5.26%`);
train probe changed from `7.021938324` to `5.387267113` (`-23.28%`). Control
accuracy nevertheless fell from `.5592` to `.3164` while control loss
improved; offset was `1.346%`. This is an accepted BC candidate, not live or
gameplay proof.

Next gates are frozen global class weights, confusion/per-class metrics,
macro-F1, balanced accuracy, end-to-end action correctness, offset top-k and
distance, followed by repeated five-minute/resume runs and headless
evaluation. No live deployment, ONNX, or PPO was performed.

### Q4 result and headless rejection

The Q4 five-minute CUDA proposal uses
`config/ai42_bc_training_q4.json` and immutable dataset hash
`98709a6c0606c4a4b64b59370236cab28f2d22ad0fbf7570567e61c32178ccae`.
It completed 362 optimizer steps in exactly 300 optimizer seconds. Validation
loss improved from `19.4502037027` to `14.1587547074`; control balanced
accuracy improved from `.26828` to `.41805`, and end-to-end action accuracy
improved from `1.500%` to `2.138%`. The immutable generation SHA-256 is
`d7302ec8d2527b4a08511e93152fdc03d507b057c2057571acf25050fe981923`.

That offline acceptance is not deployment acceptance. Protocol v14 is the
evaluation-only AI-42 boundary. It keeps the v13 actor observations, lane
contract, and reward, omits teacher-only response fields, and transports three
runtime controls:

- `ISSUE`: submit a new factorized action;
- `HOLD`: preserve an authoritative active order;
- `IDLE`: guarantee that no order remains active.

The four-head checkpoint remains compatible: inference combines the trained
`WAIT` and `CANCEL` probabilities into `IDLE`. The server returns one
authoritative `active_order` bit per hero, so `HOLD` is masked after actual
movement arrival, attack completion, cast completion, death, respawn, or
reset. The v14 Python/Go schema hash is
`a54f64514781db87ed2624720916c454d21a41ee2aabca6f094b0924e58e8bef`.

Q4 was evaluated against AI-30 for 40 deterministic matches, alternating
sides, with 32 workers and a 4,500-tick/15-minute cap. It lost all 40 matches
on both sides, with zero invalid actions, and matches ended after 5.704 minutes
on average. The policy emitted 322,200 moves, 374 waits, and zero attacks or
skills. Held-out kind metrics explain the failure: move recall was `99.74%`,
while attack, skill-1, and skill-2 recall were all zero despite non-zero
support. Q4 is therefore an offline BC artifact only and must not be deployed.

The promotion gate now fails closed when any supported action kind has zero
recall and preserves a per-kind recall regression floor. Future five-minute
proposals must first recover attack/skill coverage, then pass the same
side-balanced AI-42-versus-AI-30 headless suite. This prevents aggregate loss
or move-dominated micro-accuracy from promoting a non-combat policy.

### Q5 combat-recovery result

Q5 uses `config/ai42_bc_training_q5.json` and model-only warm-starts from the
rejected Q4 generation. It gives the kind head 40%, 30%, 15%, 6%, 4.5% and
4.5% of effective supervised mass for move, attack and skills 1-4,
respectively. The kind head weight is `6.0`; offset and anchor are temporarily
reduced to `0.5`. Model shape, dataset, split, seed and the 300-second optimizer
budget are unchanged.

The CUDA run completed 361 steps in exactly 300 optimizer seconds. Validation
loss improved from `11.7898046763` to `10.1870572756`, kind balanced accuracy
from `.39147` to `.52076`, and end-to-end action accuracy from `2.138%` to
`2.412%`. Skill-1 recall recovered from zero to `62.5%` and skill-3 reached
`100%`, but attack and skill-2 recall remained zero. Move recall fell from
`99.736%` to `97.879%`, exceeding its one-percentage-point regression floor;
offset distance also regressed. The fail-closed gate therefore rejected Q5 and
created no accepted generation. The diagnostic `latest.pt` SHA-256 is
`0a93dc5791a358a9e3b341c734abd666db110ada630a9d280d7512e71cd89085`.

A protocol-faithful diagnostic evaluation of that rejected checkpoint emitted
3,570 combat actions (3,067 attacks, 360 skill-1 and 143 skill-3), versus zero
for Q4, with no invalid actions. It still lost all 40 side-balanced matches to
AI-30; skill-2 and skill-4 were never emitted. Mean match duration increased
from 5.704 to 7.454 minutes. Q5 is useful evidence that targeted kind weighting
recovers combat, but it is not deployable. The run-report SHA-256 is
`a85d514f2587dc9c3da45e80f0acc4b333cd5b0472c174635a40d6107a8066f4`;
the diagnostic evaluation SHA-256 is
`73f33ac99f17d7c98a1d179beb0dc3fb35f603f11e1658ebed7a6fd2715e11a8`.

### Q6-Q9 dataset02 and bounded candidate selection

Dataset02 contains 320 deterministic AI-30 mirror matches under manifest hash
`f551d9c152f3ed21d1ece0fcf9fd90bfcc82764d0471b02c5751dfe75e29fcc6`.
Its 647,848 ticks occupy 7.75 GB after native raw-deflate compression. Q6
proved that repeating one mixed combat mask collapses the policy into ordinary
attack; Q7's sequential per-kind masks collapsed it back into movement.

Q8/Q9 instead accumulate eight deterministic focused batches into one averaged
gradient before clipping and stepping. The persisted cursor advances by all
eight batches only after a successful optimizer step, preserving exact resume
across deadline interruption. Q9 also retains immutable candidates every five
steps so the useful transition between movement-only and attack-heavy policies
is not overwritten by the final checkpoint. The 300-second CUDA run completed
47 optimizer steps and retained steps 5 through 45.

All nine periodic candidates were screened in 10 side-balanced matches against
AI-30. Steps 25 and 45 then received the 40-match gate. Step 25 scored `.4375`
(2 wins, 7 losses, 31 draws). Step 45 scored `.6125` (14 wins, 5 losses,
21 draws), emitted no invalid actions, and used every skill. Its action mix was
11.39% move, 81.72% attack, 4.36% skill-2, and 1.22% across skills 1/3/4.
Checkpoint SHA-256:
`b2b9fd3c181502da118af0fd2d932c53490ea0284d008259020d17c44d4bb56b`.
This is the current BC/DAgger seed, not a deployment candidate: its ordinary
attack rate remains too high and it won only from side 1 in this fixed-side
evaluation schedule. The next data stage must aggregate policy-reached combat
states and teacher corrections instead of further tuning global class counts.

### Intervention DAgger protocol

Protocol v15 is the append-only intervention boundary for the next dataset.
It extends each v14 controlled action with one `intervention` byte. A marked
external/AI-40 slot suppresses its submitted command for that decision tick,
clears the active external order, and lets the retained AI-30 brain execute one
authoritative replacement decision. The response combines v13 teacher intent,
status and execution telemetry with v14 `active_order`. Invalid intervention
bytes and scripted-controller intervention attempts fail closed. Scalar and
vector schema SHA-256:
`06369e1df3d48649c080938403ff0f5d7310a74b65b33903d1454872a45d1a28`.
The temporary AI-30 local/team profile is restored before the result leaves the
server, so expert takeover cannot silently convert a policy slot or team into a
scripted controller for later ticks.

`tanat-ai42-collect-dagger` runs one Q9 policy side through ONNX Runtime and
streams each exact v15 scalar STEP request/result pair to `ai42daggerwriter`.
The Go writer decodes those existing frames directly into the native columnar
capture, performs strict replay once, and atomically publishes an AI42GS2 shard;
it never rebuilds observations as per-tick Python records. The schedule freezes
checkpoint lineage, side, roster, and intervention parameters into dataset
provenance. The intervention threshold is mandatory rather than hidden in the
collector. On Q9 step 45, the measured masked top-two margin distribution was
p25=0.0806 and p50=0.1776; a 50-tick smoke with threshold 0.08 and a five-tick
per-hero gap produced 17 interventions, 16 policy-side teacher labels, and zero
invalid actions, then passed the strict dataset audit.

```powershell
go build -o <ai42daggerwriter.exe> .\cmd\ai42daggerwriter
tanat-ai42-collect-dagger <generation.pt> `
  --config .\ai40\config\ai42_bc_training_q9.json `
  --env <assaultenv.exe> --writer <ai42daggerwriter.exe> `
  --output <dataset-dir> --onnx <actor.onnx> `
  --seed 54001 --candidate-side 1 --max-steps 4500 `
  --intervention-margin 0.08 --intervention-gap-ticks 5 --device cuda
```

`tanat-ai42-collect-dagger-generation` reuses one ONNX session across a bounded
batch of scalar simulation processes, alternates the candidate between sides,
and publishes one immutable generation with one schedule and policy lineage.
Every match is decoded, strictly replayed, and compressed by its own Go writer;
the final merge copies compressed payloads without decompression. The schedule
also records SHA-256 and byte size for the exact environment, writer, and ONNX
artifacts, so a server-semantics change creates distinguishable provenance even
when tensor and reward schemas stay compatible. A four-match,
four-worker, 400-tick integration run completed in 5.47 seconds (73.1 aggregate
ticks/second), produced no invalid actions, and passed strict train/validation
audits with one candidate-side-1 and one candidate-side-2 match in each split.

```powershell
tanat-ai42-collect-dagger-generation <generation.pt> `
  --config .\ai40\config\ai42_bc_training_q9.json `
  --env <assaultenv.exe> --writer <ai42daggerwriter.exe> `
  --output <dataset-dir> --onnx <actor.onnx> `
  --seed 54200 --matches 8 --workers 4 --max-steps 4500 `
  --intervention-margin 0.08 --intervention-gap-ticks 5 `
  --validation-fraction 0.25 --split-seed 42 --device cuda
```

The match count must be even. Train/validation assignment is stratified by the
candidate side, and the validation fraction must therefore select an integer
number of matches from each side.

### AI-42 native-inference benchmark

`tanat-ai42-benchmark-inference` exports an immutable generation to the complete
actor-only ONNX interface and compares synchronous PyTorch and ONNX Runtime
calls at the rollout boundary. The interface includes `control`, every action
head, and recurrent `next_h`/`next_c`; export fails if parity does not hold.
CUDA benchmarks also fail closed if ONNX Runtime silently falls back to CPU.

On the RTX 3070 Laptop GPU, Q9 step 45 at batch 10 measured 14.69 ms per batch
with eager PyTorch BF16 and 7.51 ms with ONNX Runtime CUDA, a 1.96x speedup.
The full two-match evaluator attributed 88.9% of its 600-tick wall time to model
inference and only 6.0% to the Go simulation. Peak PyTorch reserved VRAM was
78 MiB, so this actor can coexist with a larger GPU workload by limiting its
batch without treating VRAM as the primary bottleneck.

The same two-match, two-worker, 600-tick rollout completed in 7.76 seconds with
the ONNX Runtime backend versus 18.46 seconds with PyTorch BF16 (2.38x end-to-end).
Policy throughput increased from 366 to 1,026 rows/second, with zero invalid
actions. The evaluator exports the exact loaded checkpoint before opening the
session and records the selected backend in `runtime_profile`.

For CUDA 12 PyTorch wheels use the `onnx-gpu` optional dependency. It pins ORT
below 1.27 because ORT 1.27 and newer PyPI GPU wheels require CUDA 13; the
benchmark preloads PyTorch's matching CUDA/cuDNN DLLs before creating the CUDA
execution provider.

```powershell
tanat-ai42-benchmark-inference <generation.pt> `
  --config .\ai40\config\ai42_bc_training_q9.json `
  --batch 10 --iterations 100 --warmup 20 --device cuda `
  --onnx <actor.onnx> --output <benchmark.json>
```

Run the evaluator from `server` with:

```powershell
$env:PYTHONPATH = "$PWD\ai40\src"
python -m tanat_ai40.evaluate_ai42 <generation.pt> `
  --config .\ai40\config\ai42_bc_training_q4.json `
  --env .\assaultenv.exe --matches 40 --workers 32 `
  --max-steps 4500 --device cuda --backend onnxruntime `
  --onnx <actor.onnx> --output <evaluation.json>
```

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
