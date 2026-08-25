#!/usr/bin/env bash
set -Eeuo pipefail

SERVER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PACKAGE_ROOT="$SERVER_ROOT/ai40"
PYTHON_BIN="$PACKAGE_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi

export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m tanat_ai40.smoke_ai42_actor "$@"
