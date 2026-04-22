# MemoMemo Progressive Iteration Workspace Plan

## Goal

Make MemoMemo optimization runs reproducible, inspectable, and comparable by
turning every iteration into a self-contained artifact bundle and replacing the
current UCB parent-selection structure with progressive context loading.

The core idea:

- Every iteration starts from the same clean source snapshot copied from the
  initial `src` implementation, such as the memgpt scaffold source.
- Historical iterations are used as diagnostic references, not as source-code
  parents.
- Summary-level history is always fully visible.
- Raw per-iteration details are loaded progressively according to a simple
  low/medium/high budget schedule.
- Default/full and progressive modes use the same Docker workspace boundary, so
  proposers cannot see unrelated previous experiments.

This should make each candidate answer a clearer question:

> Given the original source and the historical evidence available under this
> budget, what mechanism should we implement next?

It avoids mixing two variables: source inheritance and information budget.

## Current Problems

The current layout and proposer flow have several issues:

- Iteration artifacts are split across `proposer_calls/`, `generated/`,
  `candidate_results/`, `trace_slices/`, and run-level summary files.
- `candidate_results/` contains highly useful `tasks[].retrieved` evidence, but
  it is stored globally rather than under each iteration.
- Default proposer runs from project root and can inspect unrelated source,
  run outputs, and previous experiments.
- UCB workspaces are scoped, but UCB currently combines context selection,
  parent selection, and source snapshot construction.
- Parent source lineage is hard to maintain for source-edit candidates because
  edited code can live in old `generated/source_snapshots/iter_xxx` paths.
- `fs_access=all` in UCB means all copied scoped context, not the same visibility
  as current default.

## New Design

Replace UCB parent selection with progressive context loading.

Every iteration:

1. Creates a clean source snapshot from the initial `src` implementation.
2. Builds a Docker-visible workspace containing full cumulative summaries.
3. Adds raw reference artifacts from selected prior iterations according to the
   current context budget.
4. Runs the proposer in Docker with only that workspace mounted.
5. Evaluates the proposed candidate.
6. Stores all artifacts produced by the iteration under the iteration directory.
7. Updates run-level cumulative summaries and the progressive budget state.

There is no parent source inheritance. A candidate may be informed by historical
iterations, but its editable code starts from the clean source snapshot.

## Canonical Iteration Directory

Each iteration should be inspectable from one directory:

```text
runs/<run_id>/proposer_calls/iter_016/
  assignment.json
  workspace_manifest.json
  access_policy.json

  workspace/
    source_snapshot/
      candidate/
        SNAPSHOT.md
        project_source/
          src/memomemo/
        original_project_source/
          src/memomemo/
        upstream_source/
          MemGPT/
        candidate_files/

    summaries/
      evolution_summary.jsonl
      best_candidates.json
      candidate_score_table.json
      retrieval_diagnostics_summary.json
      iteration_index.json
      diff_summary.jsonl

    reference_iterations/
      iter_007/
        assignment.json
        pending_eval.json
        eval/
          candidate_result.json
          candidate_result.compact.json
          retrieval_diagnostics.json
          eval_summary.json
        trace_slices/
          low/<candidate_id>.json
          medium/<candidate_id>.json
          high/<candidate_id>.json
        source_snapshot/
        diff.patch
        diff_digest.md

      iter_014/
        ...

    pending_eval.json

  pending_eval.raw.json
  pending_eval.json

  eval/
    candidate_result.json
    candidate_result.compact.json
    retrieval_diagnostics.json
    eval_summary.json

  trace_slices/
    low/<candidate_id>.json
    medium/<candidate_id>.json
    high/<candidate_id>.json

  source_snapshot/
    candidate/
      project_source/
      original_project_source/
      upstream_source/
      candidate_files/

  generated/
    candidate modules created by this iteration

  diff.patch
  diff_digest.md

  agent/
    prompt.md
    stdout.md
    stderr.txt
    metrics.json
    tool_access.json
    stream.jsonl
```

The `workspace/` directory is what Docker sees. The rest of the iteration
directory is the host-side canonical archive after the attempt finishes.

Do not use symlinks inside Docker-visible directories. Copy files so a workspace
is self-contained.

## Global Run Indexes

Keep global run-level files as indexes and caches:

```text
runs/<run_id>/candidate_results/
runs/<run_id>/generated/
runs/<run_id>/trace_slices/
runs/<run_id>/best_candidates.json
runs/<run_id>/evolution_summary.jsonl
runs/<run_id>/candidate_score_table.json
runs/<run_id>/retrieval_diagnostics_summary.json
runs/<run_id>/iteration_index.json
runs/<run_id>/progressive_state.json
```

These files are written by the harness, but proposer Docker containers should
not mount the whole run root. The workspace builder copies the needed run-level
indexes into `workspace/summaries/`.

## Full Summary Visibility

Summary-level information is always available in every budget.

For iteration `N`, `workspace/summaries/` must contain cumulative information
through iteration `N - 1`, not just the selected reference iterations.

Required summary files:

- `evolution_summary.jsonl`
  Full run summary through the previous iteration. This includes all candidate
  events, proposal metadata, metrics, and best/frontier records.

- `best_candidates.json`
  Current best/frontier candidates through the previous iteration.

- `candidate_score_table.json`
  A compact table of all evaluated candidates:
  `iteration`, `candidate_id`, `scaffold_name`, `passrate`, `average_score`,
  `token_consuming`, `source_family`, `build_tag`, `result_path`,
  `iteration_dir`, and whether it is currently best/worst.

- `retrieval_diagnostics_summary.json`
  A cumulative compact summary of retrieval and failure patterns across all
  evaluated iterations.

- `iteration_index.json`
  One row per iteration with paths to that iteration's candidate result,
  compact result, retrieval diagnostics, traces, diff, source snapshot, and
  generated files.

- `diff_summary.jsonl`
  One compact diff digest per iteration. This lets proposers understand what
  changed without always reading full source.

This is important: if the best iteration is `iter_007` and the current
iteration is `iter_016`, a low-budget workspace still needs to know what
happened in `iter_008` through `iter_015` from the cumulative summaries.

## Raw Reference Loading

Only raw per-iteration artifacts are budget-controlled.

Raw artifacts include:

- full `eval/candidate_result.json`
- compact `eval/candidate_result.compact.json`
- `eval/retrieval_diagnostics.json`
- trace slices
- full source snapshot
- generated candidate files
- `pending_eval.json`
- `diff.patch`
- `diff_digest.md`
- agent metrics and access logs when useful

Budget rules:

### Low

Load raw artifacts for:

- the current best iteration
- the current worst iteration

Always include all cumulative summaries.

Low gives the proposer one positive example and one negative example. The
positive example shows what works; the negative example helps avoid repeating a
bad mechanism. The full summary files provide awareness of every other
iteration without exposing all raw details.

### Medium

Load raw artifacts for:

- the current best three iterations
- the current worst iteration

Always include all cumulative summaries.

Medium gives broader successful mechanism coverage while still keeping raw
context bounded.

### High

Load raw artifacts for:

- all prior iterations in the current run

Always include all cumulative summaries.

High is the exhaustive analysis mode. It is equivalent to default/full context,
but still scoped to the current run and Docker workspace.

## Selecting Best and Worst Iterations

Best and worst are selected from evaluated candidates in the current run.

Default ranking:

- best: highest `passrate`, then highest `average_score`, then lower
  `token_consuming`
- worst: lowest `passrate`, then lowest `average_score`, then higher
  `token_consuming`

Rejected, failed-import, failed-eval, or proposer-failed iterations should be
represented in summaries, but raw worst selection should prefer evaluated
candidates unless there are no evaluated candidates. Failed iterations can have
their own diagnostic list:

```text
summaries/failed_iteration_table.json
```

Deduplicate selected raw references. If the best and worst are the same
iteration, load it once.

## Progressive Budget Schedule

Remove UCB.

Use a deterministic progressive schedule:

```text
state.budget in {"low", "medium", "high"}
state.stagnation_count: int
```

Initial rule:

- The first five proposer iterations use `low`.

After the first five iterations:

- If the current iteration improves the best passrate, set budget to `low` and
  reset stagnation count to 0.
- If there is no improvement and current budget is `low`, move next iteration
  to `medium`.
- If there is no improvement and current budget is `medium`, move next
  iteration to `high`.
- If there is no improvement and current budget is `high`, keep `high`.
- If `high` improves the best passrate, move next iteration back to `low`.

More explicitly:

```text
if iteration <= 5:
    next_budget = "low"
elif improved:
    next_budget = "low"
elif current_budget == "low":
    next_budget = "medium"
elif current_budget == "medium":
    next_budget = "high"
else:
    next_budget = "high"
```

Improvement should initially mean:

```text
new_best_passrate > previous_best_passrate
```

Optionally, later we can support a tie-break improvement:

```text
same passrate but higher average_score or lower token_consuming
```

Store state in:

```text
runs/<run_id>/progressive_state.json
```

Example:

```json
{
  "current_budget": "low",
  "next_budget": "medium",
  "stagnation_count": 1,
  "best_passrate": 0.4,
  "best_candidate_id": "iter012_example_top12",
  "last_improved_iteration": 12
}
```

## Source Snapshot Model

Every iteration starts from clean source.

For memgpt-only experiments, copy from:

```text
src/memomemo/
```

into:

```text
proposer_calls/iter_016/workspace/source_snapshot/candidate/project_source/src/memomemo/
```

Also copy this clean copy into:

```text
proposer_calls/iter_016/workspace/source_snapshot/candidate/original_project_source/src/memomemo/
```

The proposer may modify only:

```text
workspace/source_snapshot/candidate/**
workspace/generated/**
workspace/pending_eval.json
```

After the proposer finishes, archive the final source snapshot to:

```text
proposer_calls/iter_016/source_snapshot/
```

Do not fork from previous source snapshots. Previous source snapshots are
reference material only.

## Diff Maintenance

Every iteration should maintain source diffs.

Generate:

```text
proposer_calls/iter_016/diff.patch
proposer_calls/iter_016/diff_digest.md
```

The diff should compare:

```text
source_snapshot/candidate/original_project_source/
vs
source_snapshot/candidate/project_source/
```

For generated wrapper candidates, include diffs or full files from:

```text
generated/
```

Also append a compact record to:

```text
diff_summary.jsonl
```

Suggested fields:

```json
{
  "iteration": 16,
  "candidate_id": "iter016_example_top12",
  "files_changed": [
    "src/memomemo/scaffolds/memgpt_scaffold.py"
  ],
  "insertions": 120,
  "deletions": 40,
  "mechanism_summary": "Added query-specific temporal evidence packets.",
  "risk_notes": "Changes answer prompt and retrieval ordering."
}
```

Diffs are first-class optimization context. In many cases, a proposer can
understand prior mechanisms from `diff_digest.md` and
`summaries/diff_summary.jsonl` without reading all source code.

## Candidate Result Artifacts

Keep writing the global candidate result:

```text
runs/<run_id>/candidate_results/<candidate_id>.json
```

Also copy it into the iteration:

```text
proposer_calls/iter_016/eval/candidate_result.json
```

The full result contains:

- `candidate`: aggregate metrics and config
- `tasks`: per-task question, gold answer, prediction, score, pass/fail,
  token counts, and retrieved hits
- `build_cache`: build cache status and sample reuse

`tasks[].retrieved` is critical optimization signal because it shows whether a
failure was caused by retrieval miss, context ordering, answer prompt behavior,
or another mechanism.

## Compact Candidate Result

Create:

```text
proposer_calls/iter_016/eval/candidate_result.compact.json
```

Suggested content:

```json
{
  "candidate": {
    "candidate_id": "...",
    "passrate": 0.4,
    "average_score": 0.52,
    "token_consuming": 149907
  },
  "tasks": [
    {
      "task_id": "...",
      "question": "...",
      "prediction": "...",
      "gold_answer": "...",
      "score": 0.0,
      "passed": false,
      "retrieved": [
        {
          "text": "truncated hit text",
          "score": 0.12,
          "source": "memgpt_source",
          "memory_tier": "archival",
          "tool": "archival_memory_search",
          "rank": 0,
          "passage_id": "...",
          "turn_indices": [1, 2, 3],
          "search_mode": "bm25+semantic"
        }
      ]
    }
  ]
}
```

The compact result should cap:

- number of tasks, or include all tasks with small per-task limits
- number of retrieved hits per task
- hit text length

Open decision: keep `gold_answer` in compact diagnostics or provide two compact
views:

- `candidate_result.compact.json`
- `candidate_result.compact_no_gold.json`

For training optimization, full results already include gold answers, so this
is mainly about making the proposer-facing default safer and cleaner.

## Retrieval Diagnostics

Create:

```text
proposer_calls/iter_016/eval/retrieval_diagnostics.json
```

Suggested fields:

- candidate id, iteration, budget, source family
- aggregate metrics
- failed tasks
- low-score tasks
- retrieved-but-failed tasks
- likely retrieval-miss tasks
- memory-tier distribution
- top-hit tier distribution
- average retrieved count
- token distribution
- examples grouped by failure pattern

Also update cumulative:

```text
runs/<run_id>/retrieval_diagnostics_summary.json
```

and copy it into future workspaces:

```text
workspace/summaries/retrieval_diagnostics_summary.json
```

## Proposal Metadata

Remove parent-centric requirements from the prompt and schema.

New proposal fields:

```json
{
  "name": "memgpt_example",
  "scaffold_name": "memgpt_source",
  "top_k": 12,
  "window": 1,
  "source_family": "memgpt",
  "budget": "low",
  "reference_iterations": [7, 14],
  "build_tag": "stable_build_identifier",
  "source_snapshot_path": "proposer_calls/iter_016/source_snapshot",
  "extra": {
    "source_project_path": "proposer_calls/iter_016/source_snapshot/candidate/project_source"
  },
  "hypothesis": "why this should improve passrate",
  "changes": "brief implementation summary"
}
```

`reference_iterations` records which raw iteration bundles were made available.
It is not a parent list.

The harness should fill in missing metadata after path normalization:

- `budget`
- `reference_iterations`
- `source_snapshot_path`
- host path for `extra.source_project_path`

## Docker Policy

All proposer modes should run in Docker by default for optimization runs.

Mount only the current workspace:

```text
docker run --rm -i \
  -v <workspace_host_path>:/workspace:rw \
  -w /workspace \
  <image> ...
```

Do not mount:

- repo root
- global `runs/`
- `references/vendor/`
- previous experiment directories

If agent auth/config is needed, mount only explicit files or directories as
read-only through `--proposer-docker-mount`.

Access checks should allow reads and writes only under `/workspace` after host
path mapping.

Save:

```text
workspace_manifest.json
access_policy.json
agent/tool_access.json
```

## Default Mode

Default should be redefined as progressive full context without budget
adaptation:

- clean source snapshot
- Docker workspace
- full cumulative summaries
- all prior raw iteration bundles
- no UCB
- no parent source inheritance

Default is equivalent to `budget=high` for every iteration.

This makes default comparable to progressive loading:

- progressive: low -> medium -> high based on stagnation
- default: high every iteration

Both should be scoped to the current run and unable to see unrelated previous
experiments.

## Prompt Changes

Prompts should say:

- Start from the clean source snapshot in `source_snapshot/candidate/`.
- Historical iterations are references for diagnosis and inspiration.
- Do not treat any reference iteration as a source parent.
- Do not copy a prior candidate mechanically; implement one intentional
  mechanism from the clean source.
- Full cumulative summaries are in `summaries/`.
- Raw reference details are in `reference_iterations/` and reflect the current
  budget.
- Write exactly one candidate to `pending_eval.json`.
- Do not read raw LOCOMO data, global `candidate_results`, global run
  directories, or scoring helpers.
- Do not reduce recall solely to save tokens. Compression, filtering, reranking,
  and context budgeting are valid when they are expected to improve answer
  quality by removing noise or surfacing stronger evidence.
- Parameter changes are allowed only as supporting details of a mechanism
  change. A candidate whose substantive change is only `top_k`, window size,
  thresholds, weights, prompt length, or context budget will be rejected.
- Use gold answers only to classify failure modes; do not encode task-specific
  answers, names, dates, or scorer quirks into runtime behavior.
- All copied project source under `project_source/src/memomemo/**` is editable
  for this candidate, including scaffolds, base classes, model/prompt helpers,
  dynamic-loading helpers, and utils.

Remove prompt requirements like:

- "candidate must derive from parent"
- "parent_candidate_id is required"
- UCB/bandit terminology

Use `reference_iterations` instead.

## Implementation Phases

### Phase 1: Iteration Bundle Schema

Add helpers:

- `_iteration_dir(iteration) -> Path`
- `_workspace_dir(iteration) -> Path`
- `_write_workspace_manifest(...)`
- `_write_access_policy(...)`
- `_archive_workspace_outputs(...)`
- `_copy_candidate_result_to_iteration(...)`
- `_write_compact_candidate_result(...)`
- `_write_retrieval_diagnostics(...)`
- `_copy_trace_slices_to_iteration(...)`
- `_write_diff_artifacts(...)`

Keep global outputs for compatibility.

### Phase 2: Cumulative Summaries

Create and maintain:

- `candidate_score_table.json`
- `retrieval_diagnostics_summary.json`
- `iteration_index.json`
- `diff_summary.jsonl`
- `progressive_state.json`

Every new workspace copies the current versions into `workspace/summaries/`.

### Phase 3: Clean Source Snapshot Builder

Implement a builder that always copies clean source from the current repo source
or a frozen initial seed snapshot.

For memgpt-only:

- copy `src/memomemo/**`
- copy `references/vendor/MemGPT/**`
- copy `original_project_source` for policy diffing

Do not copy previous iteration source as the editable base.

### Phase 4: Reference Iteration Selector

Implement:

- `_best_iterations(k)`
- `_worst_iteration()`
- `_reference_iterations_for_budget(budget)`

Rules:

- low: best 1 + worst 1
- medium: best 3 + worst 1
- high: all prior evaluated iterations

Always dedupe.

### Phase 5: Progressive Budget Scheduler

Remove or bypass UCB state and arm selection.

Implement progressive state:

- first five iterations low
- improvement -> next low
- low stagnation -> medium
- medium stagnation -> high
- high stagnation -> high
- high improvement -> low

Persist state in `progressive_state.json`.

### Phase 6: Docker-Scoped Default

Make default run through the same workspace builder.

Default budget:

- always high

Default should no longer run from project root.

### Phase 7: Prompt Refactor

Update `proposer_prompt.py`:

- one progressive prompt builder
- optional flag for default/high vs adaptive budget
- no parent requirements
- no UCB terminology
- workspace-local paths only
- mention summaries and reference iterations

### Phase 8: Compatibility

Old runs should remain readable.

Fallbacks:

- if iteration bundle eval result is missing, read global
  `candidate_results/<candidate_id>.json`
- if iteration source snapshot is missing, read old
  `generated/source_snapshots/iter_xxx`
- if compact diagnostics are missing, generate them lazily from full candidate
  result

New runs should use the new bundle layout.

## Tests

### Progressive Schedule

- first five iterations use low
- low no-improvement moves to medium
- medium no-improvement moves to high
- high no-improvement stays high
- any improvement moves next budget to low
- high improvement moves next budget to low

### Reference Selection

- low selects best one and worst one
- medium selects best three and worst one
- high selects all prior evaluated iterations
- best/worst overlap is deduped
- failed iterations are indexed but not selected as raw worst when evaluated
  candidates exist

### Workspace Summaries

- every budget copies full cumulative `evolution_summary.jsonl`
- low still sees summary records for iterations not selected as raw references
- `candidate_score_table.json` includes all evaluated candidates
- `iteration_index.json` maps iteration to artifact paths
- `diff_summary.jsonl` includes all prior diffs

### Candidate Results

- global `candidate_results/<id>.json` is written
- per-iteration `eval/candidate_result.json` is copied
- compact candidate result preserves retrieved metadata
- retrieval diagnostics are written and included in cumulative summary

### Source Snapshots

- every iteration starts from clean source
- previous best source snapshot is available only under `reference_iterations/`
- editable `source_snapshot/` is not copied from prior iteration source
- `original_project_source` enables policy diff checks

### Diff Artifacts

- `diff.patch` compares original clean source to modified source
- `diff_digest.md` summarizes changed mechanism
- `diff_summary.jsonl` is copied into future workspaces

### Docker and Access

- proposer cwd is workspace
- Docker mount is workspace only
- reads outside workspace are rejected or retried
- writes outside workspace are rejected or retried
- default cannot read repo root or unrelated `runs/`
- progressive mode cannot read global candidate results directly

### Prompt

- prompt references `summaries/`
- prompt references `reference_iterations/`
- prompt states clean source is the editable base
- prompt does not require parent candidate
- prompt does not mention UCB or bandit
- prompt does not expose global `runs/<run_id>/candidate_results/`

## Open Decisions

1. Should compact diagnostics include `gold_answer` by default?

2. Should high include all prior raw agent telemetry, or only eval/source/diff
   artifacts?

3. Should the clean source snapshot come from live repo `src` or from an
   explicit frozen `iter_000` seed snapshot?

4. Should default be a separate mode, or simply `--progressive-budget high`
   fixed for all iterations?

5. Should improvement use only passrate, or passrate plus tie-breakers?

Initial recommendation:

- compact diagnostics include gold for now because full candidate results
  already include it
- high includes eval/source/diff/traces, but not full raw agent streams unless
  debugging
- use an explicit frozen `iter_000` seed snapshot for reproducibility
- default is fixed high with the same workspace builder
- improvement starts as strict passrate improvement

## Success Criteria

- Every iteration can be understood from its own `proposer_calls/iter_xxx/`
  directory.
- Every proposer Docker run sees only the current workspace.
- Every budget sees full cumulative summaries through the previous iteration.
- Low sees raw artifacts for best one and worst one.
- Medium sees raw artifacts for best three and worst one.
- High sees raw artifacts for all prior iterations.
- Every iteration edits a clean source snapshot, not a parent snapshot.
- Diffs are maintained so source reading is optional in many cases.
- Default/high and progressive modes are comparable because both are scoped.
- Existing global summary/frontier/loading behavior remains compatible.
- `pytest -q` passes.
