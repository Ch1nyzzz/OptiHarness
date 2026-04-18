# MemoMemo

MemoMemo is a clean memory-specialized evolution harness. It is separate from
`skillevolve` and is scoped to LOCOMO conversational-memory QA.

The first objective axis is `passrate` (maximize). The second objective axis is
`token_consuming` (minimize). Every run writes a Pareto frontier JSON with those
two fields.

## What Is Included

- LOCOMO importer and deterministic split writer.
- OpenAI-compatible local model client. Defaults match the local model setup:
  `/data/home/yuhan/model_zoo/Qwen3-8B` at `http://127.0.0.1:8000/v1`.
- Three initial memory scaffolds:
  - `bm25`: lexical retrieval scaffold compatible with `rank_bm25`; falls back to a
    built-in BM25 implementation when the package is not installed.
  - `amem`: A-Mem-style atomic memory scaffold with entity and temporal scoring.
  - `mem0`: Mem0-style fact memory scaffold with speaker/entity-aware retrieval.
- Pareto frontier writer over `passrate` and `token_consuming`.
- Reference manifest and fetch script for:
  - https://github.com/dorianbrown/rank_bm25.git
  - https://github.com/WujiangXu/A-mem
  - https://github.com/mem0ai/mem0

## Install

```bash
cd /data/home/yuhan/MemoMemo
python -m pip install -e '.[dev]'
```

Optional external libraries:

```bash
python -m pip install -e '.[dev,external]'
```

## Prepare LOCOMO

The setup can reuse the local SkillEvolve cache if present:

```bash
memomemo locomo prepare
```

To allow downloading when no local cache exists:

```bash
memomemo locomo prepare --allow-download
```

## Run Initial Memory Frontier

Dry-run retrieval scoring, useful for a quick plumbing check:

```bash
memomemo evolve --split train --limit 20 --dry-run --out runs/locomo_memory_scaffold_smoke
```

Full local-model scoring:

```bash
memomemo evolve \
  --split train \
  --limit 40 \
  --model /data/home/yuhan/model_zoo/Qwen3-8B \
  --base-url http://127.0.0.1:8000/v1 \
  --out runs/locomo_memory_scaffold_run
```

Key outputs:

- `runs/<run>/candidate_results/*.json`
- `runs/<run>/pareto_frontier.json`
- `runs/<run>/run_summary.json`

## Run Reusable Baselines

Run the built-in `bm25`, `amem`, and `mem0` baselines once across train/test,
with three repeated trials per split:

```bash
memomemo baseline \
  --splits train,test \
  --repeats 3 \
  --model /data/home/yuhan/model_zoo/Qwen3-8B \
  --base-url http://127.0.0.1:8000/v1 \
  --out runs/baselines
```

The command reuses existing repeat directories by default. Add `--force` when
you intentionally want to rerun the cached baseline trials.

Key outputs:

- `runs/baselines/baseline_summary.json`
- `runs/baselines/train/repeat_01/run_summary.json`
- `runs/baselines/test/repeat_01/run_summary.json`
- `runs/baselines/<split>/repeat_<NN>/candidate_results/*.json`

## Run Claude Proposer Optimization

This is the real optimization loop. It follows the `skillevolve` /
`meta-harness` pattern:

1. evaluate built-in scaffold candidates as iteration 0,
2. call `claude -p` to propose new candidate memory-scaffold code,
3. require Claude to write `pending_eval.json`,
4. import and evaluate those candidates,
5. update `pareto_frontier.json` over `passrate` and `token_consuming`.

The proposer prompt is aligned with the stricter `meta-harness` discipline,
adapted to MemoMemo's two-objective Pareto frontier: each iteration must
produce exactly two candidates, write missing post-eval reports, prototype
mechanisms before final implementation, avoid parameter-only tuning, and
self-critique candidates before writing `pending_eval.json`.

Small dry-run:

```bash
memomemo optimize --run-id smoke_opt --iterations 1 --limit 3 --dry-run
```

Full local-model run:

```bash
memomemo optimize \
  --run-id locomo_memory_opt \
  --iterations 10 \
  --split train \
  --limit 40 \
  --baseline-dir runs/baselines \
  --model /data/home/yuhan/model_zoo/Qwen3-8B \
  --base-url http://127.0.0.1:8000/v1 \
  --claude-model claude-sonnet-4-6
```

When `--baseline-dir` is set, iteration 0 loads the precomputed baseline
candidates for the selected split instead of rerunning `bm25`, `amem`, and
`mem0`.

Claude writes generated candidates under:

- `src/memomemo/generated/`

The harness writes proposer/eval artifacts under:

- `runs/<run-id>/claude_sessions/`
- `runs/<run-id>/pending_eval.json`
- `runs/<run-id>/reports/`
- `runs/<run-id>/candidate_results/`
- `runs/<run-id>/evolution_summary.jsonl`
- `runs/<run-id>/pareto_frontier.json`

## Fetch Reference Repos

The adapters are intentionally clean and local. To inspect the upstream
reference repositories side-by-side:

```bash
scripts/fetch_reference_repos.sh
```
