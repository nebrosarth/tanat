#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export TANAT_TRAINING_VARIANT=ai41
exec "$SCRIPT_DIR/run-ai40-training.sh" "$@"
