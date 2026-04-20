# MemoMemo

MemoMemo is a clean memory-specialized evolution harness. It is separate from
`skillevolve` and is scoped to LOCOMO conversational-memory QA.

The first objective axis is `passrate` (maximize). The second objective axis is
`token_consuming` (minimize). `average_score` is reported as a diagnostic but is
not optimized because it is strongly coupled to `passrate`. Every run writes a
Pareto frontier JSON with those fields.

## What Is Included

- LOCOMO importer and deterministic split writer.
- OpenAI-compatible local model client. Defaults match the local model setup:
  `/data/home/yuhan/model_zoo/Qwen3-8B` at `http://127.0.0.1:8000/v1`.
- Built-in memory scaffolds:
  - `bm25`: lexical retrieval scaffold compatible with `rank_bm25`; falls back to a
    built-in BM25 implementation when the package is not installed.
  - `mem0_source`: calls the checked-out mem0 `Memory.from_config/add/search`
    source.
  - `memgpt_source`: source-informed MemGPT/Letta seed that reproduces the core
    memory hierarchy: core memory blocks, recall search, archival search, and
    context-compaction summaries.
  - `membank_source`: source-informed MemoryBank/SiliconFriend seed with daily
    memory docs, deterministic summaries and personality memory, an Ebbinghaus
    retention score, retrieval reinforcement, and date-grouped memory search.
- Pareto frontier writer over `passrate` and `token_consuming`.
- Reference manifest and fetch script for:
  - https://github.com/dorianbrown/rank_bm25.git
  - https://github.com/mem0ai/mem0
  - https://github.com/cpacker/MemGPT
  - https://github.com/zhongwanjun/MemoryBank-SiliconFriend

## Install

```bash
cd /data/home/yuhan/MemoMemo
python -m pip install -e '.[dev]'
```

Upstream-source integrations:

```bash
scripts/fetch_reference_repos.sh
python -m pip install -e '.[dev,source]'
```

The mem0 scaffold is source-backed and requires upstream dependencies plus
LLM/embedding configuration:

```bash
memomemo evolve \
  --split train \
  --limit 20 \
  --scaffolds mem0_source \
  --scaffold-extra-json @configs/source_memory.example.json \
  --out runs/locomo_memory_source_scaffold_run
```

For optimization, source scaffolds can be used as seed mechanisms:

```bash
memomemo optimize \
  --run-id locomo_memory_source_opt \
  --iterations 20 \
  --split train \
  --limit 40 \
  --scaffolds mem0_source,memgpt_source,membank_source \
  --scaffold-extra-json @configs/source_memory.example.json \
  --model /data/home/yuhan/model_zoo/Qwen3-8B \
  --base-url http://127.0.0.1:8000/v1
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
memomemo evolve \
  --split train \
  --limit 20 \
  --dry-run \
  --scaffold-extra-json @configs/source_memory.example.json \
  --out runs/locomo_memory_scaffold_smoke
```

Full local-model scoring:

```bash
memomemo evolve \
  --split train \
  --limit 40 \
  --scaffold-extra-json @configs/source_memory.example.json \
  --model /data/home/yuhan/model_zoo/Qwen3-8B \
  --base-url http://127.0.0.1:8000/v1 \
  --out runs/locomo_memory_scaffold_run
```

Key outputs:

- `runs/<run>/candidate_results/*.json`
- `runs/<run>/pareto_frontier.json`
- `runs/<run>/run_summary.json`

## Run Reusable Baselines

Run the built-in `bm25`, `mem0_source`, `memgpt_source`, and
`membank_source` baselines once across train/test, with three repeated trials
per split:

```bash
memomemo baseline \
  --splits train,test \
  --repeats 3 \
  --scaffold-extra-json @configs/source_memory.example.json \
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
produce exactly one candidate in the default flow, write missing post-eval
reports, prototype the mechanism before final implementation, avoid
parameter-only tuning, and self-critique the candidate before writing
`pending_eval.json`.

Optimization seeds exactly one default top-k candidate per source scaffold:
`mem0_source=top30`, `memgpt_source=top12`, and `membank_source=top10`.
Top-k sweeps remain available for `evolve` and `baseline`, but not for the
optimizer seed set. The source evolution seed families are `mem0`, `memgpt`,
and `membank`; `bm25` remains a lexical baseline family.

Small dry-run:

```bash
memomemo optimize \
  --run-id smoke_opt \
  --iterations 1 \
  --limit 3 \
  --dry-run \
  --scaffold-extra-json @configs/source_memory.example.json
```

Full local-model run:

```bash
memomemo optimize \
  --run-id locomo_memory_opt \
  --iterations 20 \
  --split train \
  --limit 40 \
  --baseline-dir runs/baselines \
  --scaffold-extra-json @configs/source_memory.example.json \
  --model /data/home/yuhan/model_zoo/Qwen3-8B \
  --base-url http://127.0.0.1:8000/v1 \
  --claude-model claude-sonnet-4-6
```

When `--baseline-dir` is set, iteration 0 loads the precomputed baseline
candidates for the selected split instead of rerunning the selected built-in
scaffolds. It loads the default top-k candidate for each selected scaffold.

Claude writes generated candidates and source snapshots under the run output:

- `runs/<run-id>/generated/`
- `runs/<run-id>/generated/source_snapshots/`

The harness writes proposer/eval artifacts under:

- `runs/<run-id>/claude_sessions/`
- `runs/<run-id>/proposer_calls/` when `--selection-policy ucb` is used
- `runs/<run-id>/pending_eval.json`
- `runs/<run-id>/reports/`
- `runs/<run-id>/candidate_results/`
- `runs/<run-id>/trace_slices/`
- `runs/<run-id>/evolution_summary.jsonl`
- `runs/<run-id>/pareto_frontier.json`

Each Claude proposer session also writes `meta.json`, `tool_access.json`, and
`metrics.json`. These include input/output/cache tokens, estimated USD cost,
duration, tool counts, per-file Read counts and line counts, and Write/Edit line
counts. The optimizer appends the same proposer metrics to
`evolution_summary.jsonl` as `proposer_result` events and aggregates them in
`optimizer_summary.json`.

In UCB mode, low-budget contexts copy at most 3 full trace cases per candidate
per iteration, medium copies at most 10, and high may inspect all trace cases.

## Fetch Reference Repos

The adapters are intentionally clean and local. To inspect the upstream
reference repositories side-by-side:

```bash
scripts/fetch_reference_repos.sh
```
