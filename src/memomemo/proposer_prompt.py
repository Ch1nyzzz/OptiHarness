"""Prompt builder for proposer iterations."""

from __future__ import annotations

from pathlib import Path


def build_progressive_proposer_prompt(
    *,
    run_id: str,
    iteration: int,
    run_dir: Path,
    pending_eval_path: Path,
    summaries_dir: Path,
    reference_iterations_dir: Path,
    generated_dir: Path,
    source_snapshot_dir: Path,
    budget: str,
    reference_iterations: tuple[int, ...],
    target_system: str,
    optimization_directions: tuple[str, ...],
    split: str,
    limit: int,
    selection_policy: str = "progressive",
    benchmark_name: str = "LOCOMO conversational-memory QA",
    raw_data_policy: str = "raw LOCOMO data",
) -> str:
    """Build the proposer prompt for scoped progressive-context runs."""

    direction_lines = "\n".join(f"- {line}" for line in optimization_directions)
    focus_section = ""
    if direction_lines:
        focus_section = f"""
## Optimization Focus

You may choose one of these mechanism directions, combine them, or make an
overall system-level redesign:

{direction_lines}
"""

    workspace_dir = run_dir

    def show(path: Path) -> str:
        try:
            return str(path.relative_to(workspace_dir))
        except ValueError:
            return str(path)

    refs = ", ".join(f"iter_{item:03d}" for item in reference_iterations) or "none"
    refs_json = ", ".join(str(item) for item in reference_iterations)
    pending_eval_display = show(pending_eval_path)
    summaries_display = show(summaries_dir)
    reference_display = show(reference_iterations_dir)
    source_snapshot_display = show(source_snapshot_dir)
    generated_display = show(generated_dir)

    _ = (budget, selection_policy)

    return f"""# OptiHarness Proposer — iteration {iteration}

You are optimizing the memory layer for {benchmark_name}.

Run exactly one iteration. The outer OptiHarness harness will import and evaluate
the candidate after this session exits. Do not run the full harness evaluation.

## Assignment

- Run id: `{run_id}`
- Target system: `{target_system}`
- Eval split: `{split}`
- Eval limit: `{limit}` (`0` means full split)
- Cumulative summaries: `{summaries_display}/`
- Raw reference iterations: `{reference_display}/` ({refs})
- Writable clean source snapshot: `{source_snapshot_display}/candidate/`
- Generated wrapper directory: `{generated_display}/`
- Required output: `{pending_eval_display}`

Every iteration starts from the clean source snapshot in
`{source_snapshot_display}/candidate/`. Historical iterations are diagnostic
references only. Do not treat any reference iteration as a source parent and do
not mechanically copy a prior candidate; implement one intentional mechanism
from the clean source.

## Objective

Primary objective: maximize `passrate`.

Only optimize `passrate`. `average_score` and `token_consuming` are reported
diagnostics, but they are not optimization objectives. Do not reduce recall
solely to save tokens. Compression, filtering, reranking, and context budgeting
are valid when they are expected to improve answer quality by removing noise or
surfacing stronger evidence.

Optimize for expected generalization, not the reported training split alone.
Use raw task traces to identify failure modes, recurring evidence gaps, and
bad evidence-ordering behavior. Do not use traces to create answer-surface
patches, scorer-specific strings, annotation typo fixes, or deterministic
shortcuts for known saved tasks. Use gold answers only to classify failure
modes; do not encode task-specific answers, names, dates, or scorer quirks into
runtime behavior.

{focus_section}

## Available Files

- `{summaries_display}/evolution_summary.jsonl` — full cumulative event history
  through the previous iteration.
- `{summaries_display}/best_candidates.json` — current best passrate candidates.
- `{summaries_display}/candidate_score_table.json` — compact metrics for all
  evaluated candidates.
- `{summaries_display}/retrieval_diagnostics_summary.json` — cumulative failure
  and retrieval-pattern summary.
- `{summaries_display}/iteration_index.json` — paths for prior iteration
  artifacts.
- `{summaries_display}/diff_summary.jsonl` — compact source-change records.
- `{reference_display}/` — raw iteration bundles copied into this workspace for
  detailed diagnosis. Cumulative summaries may mention iterations whose raw
  bundles are not present here.
- `{source_snapshot_display}/candidate/project_source/src/memomemo/` — editable
  project source for this candidate.
- `{source_snapshot_display}/candidate/original_project_source/src/memomemo/` —
  clean project source used for diffing and policy checks.
- `{source_snapshot_display}/candidate/upstream_source/` — copied upstream
  source when available.
- `{generated_display}/` — optional importable wrapper modules for this
  iteration.

Do not read global run directories, global `candidate_results`, repo-root
`src/`, `references/vendor`, {raw_data_policy}, or OptiHarness scoring helpers.
Candidate runtime code must not access benchmark raw data, `candidate_results/**`,
or `memomemo.metrics.score_prediction`.
The copied `memomemo` package is intentionally benchmark-scoped and incomplete.
Do not add runtime imports from repo-root harness modules such as
`memomemo.evaluation`, `memomemo.pareto`, `memomemo.metrics`, optimizer modules,
or any module not listed in Available Files. Keep `memomemo/__init__.py`
minimal; do not make it import top-level repository APIs.

## Edit Scope

You may edit only:

- `{source_snapshot_display}/candidate/**`
- `{generated_display}/**`
- `{pending_eval_display}`

All copied project source under
`{source_snapshot_display}/candidate/project_source/src/memomemo/**` is editable
for this candidate, including scaffolds, base classes, model/prompt helpers,
dynamic-loading helpers, and utils.

Source-backed baseline memories and source bases are read-only and expensive
to rebuild. If your source edit changes build/database-construction logic or
other persisted memory construction semantics, use a fresh `source_base_dir`
and a new stable `build_tag`. For upstream source edits, route explicit paths
such as `mem0_source_path` through the copied source snapshot.

## Quality Gate

Before writing `pending_eval.json`, verify that the candidate is a real
mechanism change, is not just a `top_k`/`window`/threshold/weight variant, does
not use gold answers at inference time, does not hardcode benchmark-specific
answers, and uses the isolated source snapshot for source edits.
Parameter changes are allowed only as supporting details of a mechanism change.
A candidate whose substantive change is only `top_k`, window size, thresholds,
weights, prompt length, or context budget will be rejected.
Run a lightweight syntax/import smoke check against the edited snapshot before
writing `pending_eval.json`; do not run the full harness evaluation.

## Required Output

Write exactly this JSON file:
`{pending_eval_display}`

Schema:

```json
{{
  "candidates": [
    {{
      "name": "short_unique_name",
      "scaffold_name": "{target_system}_source",
      "top_k": 8,
      "window": 1,
      "source_family": "{target_system}",
      "reference_iterations": [{refs_json}],
      "build_tag": "stable_build_identifier",
      "source_snapshot_path": "{source_snapshot_display}",
      "extra": {{
        "source_project_path": "{source_snapshot_display}/candidate/project_source"
      }},
      "hypothesis": "why this should improve passrate",
      "changes": "brief implementation summary"
    }}
  ]
}}
```

Notes:

- The `candidates` array must contain exactly one candidate.
- `top_k` must be a single integer.
- Use a source-backed scaffold such as `memgpt_source` when editing the copied
  scaffold source.
- `extra.source_project_path` must point to the edited snapshot project source
  when files under `project_source/src/memomemo` are modified.
- If you create a wrapper module in `{generated_display}`, keep it small and
  route source-backed mechanisms through the clean edited snapshot.
- `reference_iterations` records the raw bundles available for diagnosis; it is
  not a parent list.
"""
