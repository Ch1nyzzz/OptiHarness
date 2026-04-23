#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m optiharness.cli locomo prepare

args=(
  --run-id "${OPTIHARNESS_RUN_ID:-${MEMOMEMO_RUN_ID:-locomo_memory_opt}}"
  --iterations "${OPTIHARNESS_ITERATIONS:-${MEMOMEMO_ITERATIONS:-20}}"
  --split "${OPTIHARNESS_SPLIT:-${MEMOMEMO_SPLIT:-train}}"
  --scaffold-extra-json "${OPTIHARNESS_SCAFFOLD_EXTRA_JSON:-${MEMOMEMO_SCAFFOLD_EXTRA_JSON:-@configs/source_memory.example.json}}"
  --model "${OPTIHARNESS_MODEL:-${MEMOMEMO_MODEL:-/data/home/yuhan/model_zoo/Qwen3-8B}}"
  --base-url "${OPTIHARNESS_BASE_URL:-${MEMOMEMO_BASE_URL:-http://127.0.0.1:8000/v1}}"
  --claude-model "${OPTIHARNESS_CLAUDE_MODEL:-${MEMOMEMO_CLAUDE_MODEL:-claude-sonnet-4-6}}"
)

if [[ -n "${OPTIHARNESS_BASELINE_DIR:-${MEMOMEMO_BASELINE_DIR:-}}" ]]; then
  args+=(--baseline-dir "${OPTIHARNESS_BASELINE_DIR:-${MEMOMEMO_BASELINE_DIR}}")
elif [[ -d runs/baselines ]]; then
  args+=(--baseline-dir runs/baselines)
fi

python -m optiharness.cli optimize "${args[@]}"
