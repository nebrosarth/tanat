#!/usr/bin/env bash
# Ubuntu runner for the AI-40 campaign.
# The PowerShell wrapper is kept for Windows; this file is the native Linux entrypoint.

set -Eeuo pipefail

SERVER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
AI40_DIR="$SERVER_DIR/ai40"
VENV_DIR="${TANAT_AI40_VENV:-$AI40_DIR/.venv}"
ENV_BINARY="${TANAT_ASSAULTENV:-$SERVER_DIR/assaultenv}"
GO_VERSION="${TANAT_GO_VERSION:-1.26.6}"
TOOLS_DIR="${TANAT_AI40_TOOLS:-$SERVER_DIR/.tools}"
TRAINING_VARIANT="${TANAT_TRAINING_VARIANT:-ai40}"
CAMPAIGN_MODULE="tanat_ai40.train_campaign"
MODEL_LABEL="AI-40"

RUN_NAME="long-001"
STAGES=100
MIRROR_PER_STAGE=100
AI30_PER_STAGE=100
HISTORICAL_PER_STAGE=0
HISTORICAL_ANCHOR=""
HISTORICAL_POOL_SIZE=8
EVAL_MATCHES=50
EVAL_MEDIUM_MATCHES=200
EVAL_FINAL_MATCHES=500
EVAL_MEDIUM_WIN_RATE=0.40
EVAL_FINAL_WIN_RATE=0.55
EVAL_WORKERS=64
WORKERS=64
GROUP_SIZE=64
STEPS=256
MAX_STEPS=4500
MINIBATCH_SIZE=8192
LEARNING_RATE=0.0003
PPO_EPOCHS=3
TARGET_KL=0
ENV_GOMAXPROCS=16
DEVICE="cuda"
STOP_WIN_RATE=0.60
STOP_CI_LOW=0.50
PROMOTION_EVAL_MATCHES=50
PROMOTION_TOLERANCE=0
PROMOTION_MAX_CATEGORY_REGRESSION=0.05
DISCOUNT_HORIZON_SECONDS=19.8998324946844
GAE_HORIZON_SECONDS=3.26032220809386
RESUME="$AI40_DIR/checkpoints/mixed-100/latest.pt"
FROM_SCRATCH=0
NO_PIPELINE=0
NO_BF16=0
NO_COMPILE=0
COMPILE_LEARNER=0
SKIP_BUILD=0
DRY_RUN=0
SKIP_SETUP=0
SETUP_ONLY=0
NO_PAUSE=0
EXTRA_ARGS=()

pause_before_close() {
    local exit_code=$?
    if (( ! NO_PAUSE )) && [[ -t 0 ]]; then
        printf '\nTraining finished with exit code %s. Press Enter to close this window.\n' "$exit_code"
        read -r _ || true
    fi
}

trap pause_before_close EXIT

if [[ "$TRAINING_VARIANT" == "ai41" ]]; then
    CAMPAIGN_MODULE="tanat_ai40.ai41"
    MODEL_LABEL="AI-41"
    # Start a fresh conservative campaign from the best checkpoint of the
    # stalled first reward-contract run.  Later bare invocations resume it.
    RUN_NAME="ai41-tanat-reward-stable-002"
    RESUME="$AI40_DIR/checkpoints/ai41-tanat-reward-stable-001/promoted.pt"
    # Candidate stages restart from the promoted checkpoint.  Keep their PPO
    # displacement small so the fixed-opponent promotion suite sees stable,
    # incremental proposals instead of a frequent one-stage regression.
    MIRROR_PER_STAGE=50
    AI30_PER_STAGE=25
    HISTORICAL_PER_STAGE=50
    HISTORICAL_ANCHOR="$AI40_DIR/checkpoints/ai41-nav-001/checkpoints/stage-005.pt"
    STEPS=1024
    DISCOUNT_HORIZON_SECONDS=1200
    GAE_HORIZON_SECONDS=180
    LEARNING_RATE=0.00005
    PPO_EPOCHS=1
    TARGET_KL=0.01
    # Two independent 32-match rollout groups avoid the 64-environment tick
    # barrier and measured ~12% more env steps/s on the training workstation.
    GROUP_SIZE=32
    ENV_GOMAXPROCS=16
elif [[ "$TRAINING_VARIANT" != "ai40" ]]; then
    echo "error: unsupported TANAT_TRAINING_VARIANT=$TRAINING_VARIANT" >&2
    exit 1
fi

cd -- "$SERVER_DIR"

die() {
    echo "error: $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: ./run-ai40-training.sh [options] [-- campaign-options]

The first run creates ai40/.venv, installs tanat-ai40 and downloads a local
Go toolchain if Ubuntu does not provide one. The remaining options mirror
run-ai40-training.ps1; unknown options are passed to train_campaign.py.

  --run-name NAME              Campaign directory name (default: long-001)
  --stages N                   Number of stages (default: 100)
  --workers N                  Rollout workers (default: 64)
  --group-size N               Workers per Go process (default: 64)
  --steps N                    Rollout steps (default: 256)
  --max-steps N                Match tick limit (default: 4500)
  --minibatch-size N           PPO minibatch size (default: 8192)
  --learning-rate N            PPO Adam learning rate
  --ppo-epochs N               PPO passes over each rollout
  --target-kl N                Early-stop PPO update when approximate KL exceeds N (0 disables)
  --device NAME                cuda or cpu (default: cuda)
  --resume FILE                Initial checkpoint
  --from-scratch               Start with random weights
  --no-pipeline                Disable actor/learner overlap
  --no-bf16                    Disable BF16 training
  --no-compile                 Disable torch.compile
  --compile-learner            Also compile PPO (slow startup; experimental)
  --skip-build                 Reuse the existing Linux assaultenv binary
  --skip-setup                 Do not create/update the Python environment
  --setup-only                 Prepare dependencies and build assaultenv, then exit
  --dry-run                    Print the resolved command without starting training
  --no-pause                   Do not wait for Enter when the script finishes
  -h, --help                   Show this help
EOF
}

need_value() {
    (($# >= 2)) || die "option $1 requires a value"
}

while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --run-name) need_value "$@"; RUN_NAME=$2; shift 2 ;;
        --stages) need_value "$@"; STAGES=$2; shift 2 ;;
        --mirror-per-stage) need_value "$@"; MIRROR_PER_STAGE=$2; shift 2 ;;
        --ai30-per-stage) need_value "$@"; AI30_PER_STAGE=$2; shift 2 ;;
        --historical-per-stage) need_value "$@"; HISTORICAL_PER_STAGE=$2; shift 2 ;;
        --historical-anchor) need_value "$@"; HISTORICAL_ANCHOR=$2; shift 2 ;;
        --historical-pool-size) need_value "$@"; HISTORICAL_POOL_SIZE=$2; shift 2 ;;
        --eval-matches) need_value "$@"; EVAL_MATCHES=$2; shift 2 ;;
        --eval-medium-matches) need_value "$@"; EVAL_MEDIUM_MATCHES=$2; shift 2 ;;
        --eval-final-matches) need_value "$@"; EVAL_FINAL_MATCHES=$2; shift 2 ;;
        --eval-medium-win-rate) need_value "$@"; EVAL_MEDIUM_WIN_RATE=$2; shift 2 ;;
        --eval-final-win-rate) need_value "$@"; EVAL_FINAL_WIN_RATE=$2; shift 2 ;;
        --eval-workers) need_value "$@"; EVAL_WORKERS=$2; shift 2 ;;
        --workers) need_value "$@"; WORKERS=$2; shift 2 ;;
        --group-size) need_value "$@"; GROUP_SIZE=$2; shift 2 ;;
        --steps) need_value "$@"; STEPS=$2; shift 2 ;;
        --max-steps) need_value "$@"; MAX_STEPS=$2; shift 2 ;;
        --minibatch-size) need_value "$@"; MINIBATCH_SIZE=$2; shift 2 ;;
        --learning-rate) need_value "$@"; LEARNING_RATE=$2; shift 2 ;;
        --ppo-epochs) need_value "$@"; PPO_EPOCHS=$2; shift 2 ;;
        --target-kl) need_value "$@"; TARGET_KL=$2; shift 2 ;;
        --env-gomaxprocs) need_value "$@"; ENV_GOMAXPROCS=$2; shift 2 ;;
        --device) need_value "$@"; DEVICE=$2; shift 2 ;;
        --stop-win-rate) need_value "$@"; STOP_WIN_RATE=$2; shift 2 ;;
        --stop-ci-low) need_value "$@"; STOP_CI_LOW=$2; shift 2 ;;
        --promotion-eval-matches) need_value "$@"; PROMOTION_EVAL_MATCHES=$2; shift 2 ;;
        --promotion-tolerance) need_value "$@"; PROMOTION_TOLERANCE=$2; shift 2 ;;
        --promotion-max-category-regression) need_value "$@"; PROMOTION_MAX_CATEGORY_REGRESSION=$2; shift 2 ;;
        --discount-horizon-seconds) need_value "$@"; DISCOUNT_HORIZON_SECONDS=$2; shift 2 ;;
        --gae-horizon-seconds) need_value "$@"; GAE_HORIZON_SECONDS=$2; shift 2 ;;
        --resume) need_value "$@"; RESUME=$2; shift 2 ;;
        --from-scratch) FROM_SCRATCH=1; shift ;;
        --no-pipeline) NO_PIPELINE=1; shift ;;
        --no-bf16) NO_BF16=1; shift ;;
        --no-compile) NO_COMPILE=1; shift ;;
        --compile-learner) COMPILE_LEARNER=1; shift ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --skip-setup) SKIP_SETUP=1; shift ;;
        --setup-only) SETUP_ONLY=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --no-pause) NO_PAUSE=1; shift ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

command_exists() { command -v "$1" >/dev/null 2>&1; }

setup_python() {
    command_exists python3 || die "python3 is required (Ubuntu package: python3, python3-venv)"
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        if [[ -e "$VENV_DIR" ]]; then
            local backup="${VENV_DIR}.windows.$(date +%Y%m%d-%H%M%S)"
            mv -- "$VENV_DIR" "$backup"
            echo "Saved incompatible Python environment as: $backup"
        fi
        echo "Creating Python environment: $VENV_DIR"
        python3 -m venv "$VENV_DIR" || die "cannot create venv; install python3-venv"
    fi

    local python="$VENV_DIR/bin/python"
    if ! "$python" -c 'import numpy, torch, tanat_ai40' >/dev/null 2>&1; then
        echo "Installing AI-40 Python dependencies (this may download PyTorch/CUDA wheels)..."
        "$python" -m pip install --upgrade pip setuptools wheel
        "$python" -m pip install --editable "$AI40_DIR"
    fi
    "$python" -c 'import sys, numpy, torch; print(f"Python {sys.version.split()[0]}, NumPy {numpy.__version__}, PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}")'
}

setup_go() {
    if command_exists go; then
        GO_BIN=$(command -v go)
        return
    fi

    local go_root="$TOOLS_DIR/go-$GO_VERSION"
    GO_BIN="$go_root/bin/go"
    if [[ ! -x "$GO_BIN" ]]; then
        command_exists curl || die "curl is required to download Go"
        mkdir -p "$TOOLS_DIR"
        local archive="$TOOLS_DIR/go${GO_VERSION}.linux-amd64.tar.gz"
        echo "Downloading Go $GO_VERSION to $go_root"
        curl --fail --location --retry 3 --output "$archive" \
            "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz"
        rm -rf -- "$go_root"
        tar -xzf "$archive" -C "$TOOLS_DIR"
        mv -- "$TOOLS_DIR/go" "$go_root"
        rm -f -- "$archive"
    fi
}

check_compile_prerequisites() {
    ((NO_COMPILE)) && return
    command_exists gcc || {
        echo "Warning: gcc is unavailable; disabling torch.compile. Use --no-compile to silence this message." >&2
        NO_COMPILE=1
        return
    }

    local python_include
    python_include="$("$VENV_DIR/bin/python" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
    if [[ ! -f "$python_include/Python.h" ]]; then
        echo "Warning: $python_include/Python.h is missing; disabling torch.compile." >&2
        echo "Install python3.12-dev and rerun to enable compilation: sudo apt install python3.12-dev" >&2
        NO_COMPILE=1
    fi
}

build_environment() {
    if ((SKIP_BUILD)); then
        [[ -x "$ENV_BINARY" ]] || die "Linux environment not found: $ENV_BINARY"
        return
    fi
    echo "Building Linux assaultenv: $ENV_BINARY"
    (cd "$SERVER_DIR" && "$GO_BIN" build -o "$ENV_BINARY" ./cmd/assaultenv)
    chmod +x "$ENV_BINARY"
}

if (( ! SKIP_SETUP )); then
    setup_python
fi
check_compile_prerequisites
setup_go
build_environment

PYTHON="$VENV_DIR/bin/python"
[[ -x "$PYTHON" ]] || die "Python environment not found: $PYTHON"
[[ -x "$ENV_BINARY" ]] || die "Training environment not found: $ENV_BINARY"

if ((SETUP_ONLY)); then
    echo "$MODEL_LABEL Ubuntu environment is ready."
    echo "Python: $PYTHON"
    echo "Environment: $ENV_BINARY"
    exit 0
fi

if ((FROM_SCRATCH)); then
    RESUME=""
fi
if [[ "$TRAINING_VARIANT" == "ai41" && "$RESUME" == "$AI40_DIR/checkpoints/ai41-tanat-reward-stable-bootstrap/latest.pt" && ! -f "$RESUME" ]]; then
    [[ -f "$HISTORICAL_ANCHOR" ]] || die "frozen stage-005 checkpoint not found: $HISTORICAL_ANCHOR"
    echo "Migrating stage-005 weights to calibrated Tanat reward contract: $RESUME"
    "$PYTHON" -m tanat_ai40.migrate_ai41_strategic "$HISTORICAL_ANCHOR" --output "$RESUME"
fi
if [[ -n "$RESUME" ]]; then
    [[ -f "$RESUME" ]] || die "resume checkpoint not found: $RESUME"
    RESUME="$(cd -- "$(dirname -- "$RESUME")" && pwd -P)/$(basename -- "$RESUME")"
fi

OUTPUT="$AI40_DIR/checkpoints/$RUN_NAME"
ARGS=(
    -m "$CAMPAIGN_MODULE"
    --env "$ENV_BINARY"
    --output "$OUTPUT"
    --stages "$STAGES"
    --mirror-per-stage "$MIRROR_PER_STAGE"
    --ai30-per-stage "$AI30_PER_STAGE"
    --historical-per-stage "$HISTORICAL_PER_STAGE"
    --historical-pool-size "$HISTORICAL_POOL_SIZE"
    --eval-matches "$EVAL_MATCHES"
    --eval-medium-matches "$EVAL_MEDIUM_MATCHES"
    --eval-final-matches "$EVAL_FINAL_MATCHES"
    --eval-medium-win-rate "$EVAL_MEDIUM_WIN_RATE"
    --eval-final-win-rate "$EVAL_FINAL_WIN_RATE"
    --eval-workers "$EVAL_WORKERS"
    --workers "$WORKERS"
    --group-size "$GROUP_SIZE"
    --steps "$STEPS"
    --max-steps "$MAX_STEPS"
    --minibatch-size "$MINIBATCH_SIZE"
    --learning-rate "$LEARNING_RATE"
    --ppo-epochs "$PPO_EPOCHS"
    --target-kl "$TARGET_KL"
    --env-gomaxprocs "$ENV_GOMAXPROCS"
    --device "$DEVICE"
    --stop-win-rate "$STOP_WIN_RATE"
    --stop-ci-low "$STOP_CI_LOW"
    --promotion-eval-matches "$PROMOTION_EVAL_MATCHES"
    --promotion-tolerance "$PROMOTION_TOLERANCE"
    --promotion-max-category-regression "$PROMOTION_MAX_CATEGORY_REGRESSION"
    --discount-horizon-seconds "$DISCOUNT_HORIZON_SECONDS"
    --gae-horizon-seconds "$GAE_HORIZON_SECONDS"
)
if [[ -n "$HISTORICAL_ANCHOR" ]]; then
    [[ -f "$HISTORICAL_ANCHOR" ]] || die "historical anchor not found: $HISTORICAL_ANCHOR"
    ARGS+=(--historical-anchor "$HISTORICAL_ANCHOR")
fi
if [[ -n "$RESUME" ]]; then ARGS+=(--resume "$RESUME"); fi
if ((NO_PIPELINE)); then ARGS+=(--no-pipeline); fi
if ((NO_BF16)); then ARGS+=(--no-bf16); fi
if ((NO_COMPILE)); then ARGS+=(--no-compile); fi
if ((COMPILE_LEARNER)); then ARGS+=(--compile-learner); fi
ARGS+=("${EXTRA_ARGS[@]}")

echo "$MODEL_LABEL campaign: $OUTPUT"
if [[ -n "$RESUME" ]]; then echo "Initial checkpoint: $RESUME"; else echo "Initial checkpoint: random weights"; fi
echo "Training: workers=$WORKERS groups=$GROUP_SIZE rollout=$STEPS max_steps=$MAX_STEPS minibatch=$MINIBATCH_SIZE device=$DEVICE"
echo "PPO: learning_rate=$LEARNING_RATE epochs=$PPO_EPOCHS target_kl=$TARGET_KL (0 disables early stop)"
echo "Historical self-play: per_stage=$HISTORICAL_PER_STAGE pool=$HISTORICAL_POOL_SIZE anchor=$HISTORICAL_ANCHOR"
echo "Credit assignment: discount=${DISCOUNT_HORIZON_SECONDS}s GAE=${GAE_HORIZON_SECONDS}s"
echo "Actor-learner pipeline: $(( ! NO_PIPELINE )); BF16: $(( ! NO_BF16 )); torch.compile actor=$(( ! NO_COMPILE )) learner=$(( COMPILE_LEARNER && ! NO_COMPILE ))"
echo "Evaluation: $EVAL_MATCHES -> $EVAL_MEDIUM_MATCHES at $EVAL_MEDIUM_WIN_RATE -> $EVAL_FINAL_MATCHES at $EVAL_FINAL_WIN_RATE; workers=$EVAL_WORKERS"
echo "Stop target: winrate=$STOP_WIN_RATE CI-low=$STOP_CI_LOW"
echo "Press Ctrl+C to stop. Run the same command again to resume."

if ((DRY_RUN)); then
    printf 'Dry run: %q' "$PYTHON"
    printf ' %q' "${ARGS[@]}"
    printf '\n'
    exit 0
fi

set +e
"$PYTHON" -u "${ARGS[@]}"
exit_code=$?
set -e
exit "$exit_code"
