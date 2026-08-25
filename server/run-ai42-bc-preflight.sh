#!/usr/bin/env bash
set -Eeuo pipefail

SERVER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PACKAGE_ROOT="$SERVER_ROOT/ai40"

declare -A seen_control_flags=()
reject_control_flag_injection() {
    local argument flag canonical
    for argument in "$@"; do
        flag="${argument%%=*}"
        canonical="$flag"
        if [[ "$canonical" == -* ]]; then
            canonical="${canonical#-}"
            canonical="${canonical#-}"
            canonical="--$canonical"
        fi
        case "$canonical" in
            --config|--torch-python|--worker-timeout|--worker-command|--worker-arg)
                echo "The AI-42 preflight wrapper owns '$canonical'; do not pass it explicitly." >&2
                return 2
                ;;
            --dataset|--dataset-hash|--profile|--profile-hash|--warm-start|--output|--report|--device)
                if [[ -n "${seen_control_flags[$canonical]+present}" ]]; then
                    echo "Duplicate AI-42 preflight control flag '$canonical'." >&2
                    return 2
                fi
                seen_control_flags["$canonical"]=1
                ;;
        esac
    done
}

# Reject wrapper-owned, legacy worker, and duplicate control flags before any
# external tool is started. Go owns the production preflight and launches only
# the fixed torch_probe_worker_ai42 module.
reject_control_flag_injection "$@"

GO_BIN="$(command -v go || true)"
if [[ -z "$GO_BIN" || ! -x "$GO_BIN" ]]; then
    echo "Go toolchain not found. Install Go and make 'go' available on PATH for native AI-42 BC preflight." >&2
    exit 2
fi
cd -- "$SERVER_ROOT"

PYTHON_BIN=""
shopt -s nullglob
python_candidates=(
    "$PACKAGE_ROOT/.venv/bin/python"
    "$PACKAGE_ROOT"/.venv.linux.*/bin/python
)
shopt -u nullglob
for candidate in "${python_candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
        PYTHON_BIN="$candidate"
        break
    fi
done
if [[ -z "$PYTHON_BIN" ]]; then
    for python_name in python3 python; do
        candidate="$(command -v "$python_name" 2>/dev/null || true)"
        if [[ -n "$candidate" ]]; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi
if [[ -z "$PYTHON_BIN" ]]; then
    echo "No Python interpreter found for the AI-42 Torch worker. Install Python/Torch or create ai40/.venv." >&2
    exit 2
fi
if ! "$PYTHON_BIN" -c 'import torch' >/dev/null 2>&1; then
    echo "Python interpreter '$PYTHON_BIN' cannot import Torch for the AI-42 worker." >&2
    exit 2
fi

export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
WORKER_TIMEOUT="5m"

# Arrays preserve each user value as one argument; exec/go never invokes a
# shell for the worker or for user-supplied paths and flags. The selected
# interpreter is passed as a fixed path; worker module selection remains in Go.
native_args=(run ./cmd/ai42preflight --config "$PACKAGE_ROOT/config/ai42_bc_preflight.json")
native_args+=("$@")
native_args+=(
    --torch-python "$PYTHON_BIN"
    --worker-timeout "$WORKER_TIMEOUT"
)
exec "$GO_BIN" "${native_args[@]}"
