# OptiHarness

OptiHarness is a clean memory-specialized evolution harness. It is separate from
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
cd /data/home/yuhan/OptiHarness
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
optiharness evolve \
  --split train \
  --limit 20 \
  --scaffolds mem0_source \
  --scaffold-extra-json @configs/source_memory.example.json \
  --out runs/locomo_memory_source_scaffold_run
```

For optimization, source scaffolds can be used as seed mechanisms:

```bash
optiharness optimize \
  --run-id locomo_memory_source_opt \
  --iterations 20 \
  --split train \
  --scaffolds mem0_source,memgpt_source,membank_source \
  --scaffold-extra-json @configs/source_memory.example.json \
  --model /data/home/yuhan/model_zoo/Qwen3-8B \
  --base-url http://127.0.0.1:8000/v1
```

## Prepare LOCOMO

The setup can reuse the local SkillEvolve cache if present:

```bash
optiharness locomo prepare
```

To allow downloading when no local cache exists:

```bash
optiharness locomo prepare --allow-download
```

## Run LongMemEval

LongMemEval uses the same memory-scaffold base as LOCOMO and defaults to
`memgpt_source`. Scoring follows the official LongMemEval LLM-as-judge
yes/no check, with Together AI `openai/gpt-oss-120b` as the default judge:

```bash
export TOGETHER_API_KEY=...
```

Prepare the cleaned LongMemEval-S file from Hugging Face:

```bash
optiharness longmemeval prepare --variant s --allow-download
```

Run a dry-run smoke benchmark:

```bash
optiharness longmemeval benchmark \
  --variant s \
  --limit 3 \
  --out runs/longmemeval_memory_smoke
```

Optimize it through the shared MemGPT proposer path:

```bash
optiharness optimize \
  --task longmemeval \
  --longmemeval-variant s \
  --iterations 20
```

## Run Initial Memory Frontier

Dry-run retrieval scoring, useful for a quick plumbing check:

```bash
optiharness evolve \
  --split train \
  --limit 20 \
  --dry-run \
  --scaffold-extra-json @configs/source_memory.example.json \
  --out runs/locomo_memory_scaffold_smoke
```

Full local-model scoring:

```bash
optiharness evolve \
  --split train \
  --scaffold-extra-json @configs/source_memory.example.json \
  --model /data/home/yuhan/model_zoo/Qwen3-8B \
  --base-url http://127.0.0.1:8000/v1 \
  --out runs/locomo_memory_scaffold_run
```

Key outputs:

- `runs/<run>/candidate_results/*.json`
- `runs/<run>/best_candidates.json`
- `runs/<run>/run_summary.json`

## Run Reusable Baselines

Run the built-in `bm25`, `mem0_source`, `memgpt_source`, and
`membank_source` baselines once across train/test, with three repeated trials
per split:

```bash
optiharness baseline \
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

## Run Text Classification Benchmark

OptiHarness also includes the Meta-Harness text-classification benchmark adapter.
The default datasets now follow the paper setup: USPTO-50k single-step
retrosynthesis, Symptom2Disease diagnosis prediction, and LawBench crime
prediction from the MCE artifact. The older patent IPC-section task remains
available as `USPTO_IPC_SECTION`. Result aggregation and Pareto output use
OptiHarness's own runner and schemas.

Install the optional dataset dependency:

```bash
python -m pip install -e '.[dev,benchmark]'
```

Fetch the paper-compatible USPTO-50k JSONL splits:

```bash
scripts/fetch_reference_repos.sh
```

The loader expects MCE artifact data under `references/vendor/mce-artifact/env/`.
To use another checkout, set `MEMOMEMO_USPTO50K_DATA_DIR`,
`MEMOMEMO_SYMPTOM2DISEASE_DATA_DIR`, or `MEMOMEMO_LAWBENCH_DATA_DIR`.

Quick plumbing check without model calls:

```bash
optiharness text-classification benchmark \
  --dry-run \
  --num-train 2 \
  --num-val 1 \
  --num-test 2 \
  --out runs/text_classification_smoke
```

Full local-model baseline:

```bash
optiharness text-classification benchmark \
  --memory-systems no_memory,fewshot_all \
  --mode offline \
  --model /data/home/yuhan/model_zoo/Qwen3-8B \
  --base-url http://127.0.0.1:8000/v1 \
  --out runs/text_classification_baselines
```

Key outputs:

- `runs/text_classification_baselines/run_summary.json`
- `runs/text_classification_baselines/candidate_results/*.json`
- `runs/text_classification_baselines/pareto_frontier.json`

Optimize the few-shot memory construction with the same proposer/sandbox
pattern as the LOCOMO optimizer, selected through `--task text_classification`.
Run online and offline as separate tasks:

```bash
optiharness optimize \
  --task text_classification \
  --run-id textcls_fewshot_offline_opt \
  --text-classification-mode offline \
  --iterations 20 \
  --text-classification-num-train 50 \
  --text-classification-num-val 30 \
  --text-classification-num-test 0

optiharness optimize \
  --task text_classification \
  --run-id textcls_fewshot_online_opt \
  --text-classification-mode online \
  --iterations 20 \
  --text-classification-num-train 50 \
  --text-classification-num-val 30 \
  --text-classification-num-test 0
```

For text-classification optimization, the proposer workspace copies only the
minimal text-classification source surface. It does not include LOCOMO source,
LOCOMO data, memory scaffolds, or upstream vendor repositories.

## Run Claude Proposer Optimization

This is the real optimization loop. It follows the `skillevolve` /
`meta-harness` pattern:

1. evaluate built-in scaffold candidates as iteration 0,
2. call `claude -p` to propose new candidate memory-scaffold code,
3. require Claude to write `pending_eval.json`,
4. import and evaluate those candidates,
5. update `best_candidates.json` by highest `passrate`.

The proposer prompt is aligned with the stricter `meta-harness` discipline:
each iteration must produce exactly one candidate, start from a clean scoped
source snapshot, use historical iterations as diagnostic references only, avoid
parameter-only tuning, and write `pending_eval.json`.

Optimization seeds exactly one default top-k candidate per source scaffold:
`mem0_source=top30`, `memgpt_source=top12`, and `membank_source=top10`.
Top-k sweeps remain available for `evolve` and `baseline`, but not for the
optimizer seed set. The source evolution seed families are `mem0`, `memgpt`,
and `membank`; `bm25` remains a lexical baseline family.

Small dry-run:

```bash
optiharness optimize \
  --run-id smoke_opt \
  --iterations 1 \
  --limit 3 \
  --dry-run \
  --scaffold-extra-json @configs/source_memory.example.json
```

Full local-model run:

```bash
optiharness optimize \
  --run-id locomo_memory_opt \
  --iterations 20 \
  --split train \
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

The harness writes proposer/eval artifacts under:

- `runs/<run-id>/proposer_calls/iter_<NNN>/`
- `runs/<run-id>/proposer_calls/iter_<NNN>/workspace/`
- `runs/<run-id>/proposer_calls/iter_<NNN>/source_snapshot/`
- `runs/<run-id>/proposer_calls/iter_<NNN>/eval/`
- `runs/<run-id>/pending_eval.json`
- `runs/<run-id>/reports/`
- `runs/<run-id>/candidate_results/`
- `runs/<run-id>/trace_slices/`
- `runs/<run-id>/evolution_summary.jsonl`
- `runs/<run-id>/best_candidates.json`
- `runs/<run-id>/candidate_score_table.json`
- `runs/<run-id>/retrieval_diagnostics_summary.json`
- `runs/<run-id>/iteration_index.json`
- `runs/<run-id>/diff_summary.jsonl`
- `runs/<run-id>/progressive_state.json` when adaptive progressive loading is used

Each Claude proposer session also writes `meta.json`, `tool_access.json`, and
`metrics.json`. These include input/output/cache tokens, estimated USD cost,
duration, tool counts, per-file Read counts and line counts, and Write/Edit line
counts. The optimizer appends the same proposer metrics to
`evolution_summary.jsonl` as `proposer_result` events and aggregates them in
`optimizer_summary.json`.

Proposer runs use the Docker filesystem sandbox by default. Provide
`--proposer-docker-image` for the image that contains the selected code-agent
CLI and any auth/config mounts needed by that agent. Use
`--proposer-sandbox none` only when intentionally running the proposer directly
on the host.

Default optimization uses the same scoped workspace builder with a fixed high
context budget. `--selection-policy progressive` starts with low context for
the first five proposer iterations, then escalates low -> medium -> high on
passrate stagnation and resets to low after a passrate improvement. Every
budget sees full cumulative summaries in `workspace/summaries/`; raw
per-iteration artifacts are copied into `workspace/reference_iterations/`
according to the current budget. Low and medium trace slices prioritize
failures that no previous iteration has answered correctly.

## Fetch Reference Repos

The adapters are intentionally clean and local. To inspect the upstream
reference repositories side-by-side:

```bash
scripts/fetch_reference_repos.sh
```
