#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m memomemo.cli locomo prepare
python -m memomemo.cli evolve \
  --split train \
  --model "${MEMOMEMO_MODEL:-/data/home/yuhan/model_zoo/Qwen3-8B}" \
  --base-url "${MEMOMEMO_BASE_URL:-http://127.0.0.1:8000/v1}" \
  --out "${MEMOMEMO_OUT:-runs/locomo_memory_seed_run}"
