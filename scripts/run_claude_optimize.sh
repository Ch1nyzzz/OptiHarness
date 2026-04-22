#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m memomemo.cli locomo prepare

args=(
  --run-id "${MEMOMEMO_RUN_ID:-locomo_memory_opt}"
  --iterations "${MEMOMEMO_ITERATIONS:-20}"
  --split "${MEMOMEMO_SPLIT:-train}"
  --scaffold-extra-json "${MEMOMEMO_SCAFFOLD_EXTRA_JSON:-@configs/source_memory.example.json}"
  --model "${MEMOMEMO_MODEL:-/data/home/yuhan/model_zoo/Qwen3-8B}"
  --base-url "${MEMOMEMO_BASE_URL:-http://127.0.0.1:8000/v1}"
  --claude-model "${MEMOMEMO_CLAUDE_MODEL:-claude-sonnet-4-6}"
)

if [[ -n "${MEMOMEMO_BASELINE_DIR:-}" ]]; then
  args+=(--baseline-dir "${MEMOMEMO_BASELINE_DIR}")
elif [[ -d runs/baselines ]]; then
  args+=(--baseline-dir runs/baselines)
fi

python -m memomemo.cli optimize "${args[@]}"
