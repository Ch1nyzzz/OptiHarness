#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m memomemo.cli locomo prepare
python -m memomemo.cli optimize \
  --run-id "${MEMOMEMO_RUN_ID:-locomo_memory_opt}" \
  --iterations "${MEMOMEMO_ITERATIONS:-10}" \
  --split "${MEMOMEMO_SPLIT:-train}" \
  --limit "${MEMOMEMO_LIMIT:-40}" \
  --model "${MEMOMEMO_MODEL:-/data/home/yuhan/model_zoo/Qwen3-8B}" \
  --base-url "${MEMOMEMO_BASE_URL:-http://127.0.0.1:8000/v1}" \
  --claude-model "${MEMOMEMO_CLAUDE_MODEL:-claude-sonnet-4-6}"
