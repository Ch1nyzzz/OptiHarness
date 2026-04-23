#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m optiharness.cli locomo prepare
python -m optiharness.cli evolve \
  --split train \
  --model "${OPTIHARNESS_MODEL:-${MEMOMEMO_MODEL:-/data/home/yuhan/model_zoo/Qwen3-8B}}" \
  --base-url "${OPTIHARNESS_BASE_URL:-${MEMOMEMO_BASE_URL:-http://127.0.0.1:8000/v1}}" \
  --out "${OPTIHARNESS_OUT:-${MEMOMEMO_OUT:-runs/locomo_memory_seed_run}}"
