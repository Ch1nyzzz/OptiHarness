"""Prompt builder for text-classification proposer iterations."""

from __future__ import annotations

from pathlib import Path


def build_text_classification_proposer_prompt(
    *,
    run_id: str,
    iteration: int,
    run_dir: Path,
    pending_eval_path: Path,
    summaries_dir: Path,
    generated_dir: Path,
    source_snapshot_dir: Path,
    mode: str,
    dataset: str,
    num_train: int | None,
    num_val: int | None,
    num_test: int | None,
    source_files: tuple[str, ...],
    primary_source_file: str,
    selection_policy: str = "default",
    context_budget: str = "high",
) -> str:
    """Build the proposer prompt for text-classification memory optimization."""

    workspace_dir = run_dir

    def show(path: Path) -> str:
        try:
            return str(path.relative_to(workspace_dir))
        except ValueError:
            return str(path)

    pending_eval_display = show(pending_eval_path)
    summaries_display = show(summaries_dir)
    source_snapshot_display = show(source_snapshot_dir)
    generated_display = show(generated_dir)
    objective_split = "test" if (num_test is None or num_test > 0) else "val"
    source_file_lines = "\n".join(
        "- "
        f"`{source_snapshot_display}/candidate/project_source/src/memomemo/{rel}`"
        for rel in source_files
    )

    return f"""# OptiHarness Text-Classification Proposer — iteration {iteration}

You are optimizing the few-shot memory layer for a text-classification
benchmark. The workspace is benchmark-scoped: it contains only the source files
declared for this benchmark plus run summaries needed for this iteration.

Run exactly one iteration. The outer OptiHarness harness will import and evaluate
the candidate after this session exits. Do not run the full harness evaluation.

## Assignment

- Run id: `{run_id}`
- Task: `text_classification_{mode}`
- Dataset: `{dataset}`
- Train examples: `{num_train if num_train is not None else "paper-default per dataset"}`
- Val examples: `{num_val if num_val is not None else "paper-default per dataset"}`
- Test examples: `{num_test if num_test is not None else "paper-default per dataset"}`
- Optimization metric: `{objective_split}_accuracy`
- Cumulative summaries: `{summaries_display}/`
- Writable clean source snapshot: `{source_snapshot_display}/candidate/`
- Generated wrapper directory: `{generated_display}/`
- Required output: `{pending_eval_display}`
- Selection policy: `{selection_policy}`
- Context budget: `{context_budget}`

Every iteration starts from the clean source snapshot in
`{source_snapshot_display}/candidate/`. Historical summaries are diagnostic
references only.

For `default` selection policy, use the available summaries as fixed high-context
run history. For `progressive` selection policy, follow the current context
budget: low means inspect the score table and frontier first, medium adds recent
summary rows, and high allows deeper summary review before making the mechanism
change.

## Objective

Optimize the `fewshot_all` / few-shot memory construction for the
Meta-Harness text-classification suite. The default suite contains USPTO-50k
single-step retrosynthesis, Symptom2Disease diagnosis prediction, and LawBench
crime prediction. `no_memory` is a reference baseline only; do not optimize it
and do not propose a no-memory candidate.

Primary objective: maximize `{objective_split}_accuracy`.

Keep the mechanism general. Do not hardcode validation/test examples, answers
for specific products/symptoms/cases, prompt hashes, row order, or scorer
quirks. Runtime code may use training labels through `learn_from_batch()`, but
it must not access validation/test labels except through the harness evaluator.

## Mode Semantics

This run uses `{mode}` mode.

- `offline`: train first receives gold-labeled examples via
  `learn_from_batch()`, then the fixed memory is evaluated.
- `online`: each train example is predicted before feedback; then
  `learn_from_batch()` receives the prediction and gold label.

Design the memory mechanism for this mode. Good mechanisms include, but are
not limited to, task-aware exemplar selection, compact per-task rules, retrieval
over train examples, confusion-aware examples, transformation or diagnosis
pattern summaries, and contrastive examples from similar inputs. A candidate
that is only a parameter change or prompt-length tweak will be rejected.

## Available Files

- `{summaries_display}/evolution_summary.jsonl` — cumulative event history.
- `{summaries_display}/best_candidates.json` — current best passrate candidates.
- `{summaries_display}/candidate_score_table.json` — compact metrics.
{source_file_lines}
- `{generated_display}/` — optional importable wrapper modules for this
  iteration.

Do not try to read benchmark code, repository source, global run artifacts, or
previous candidate results from outside this workspace.
The copied `memomemo` package is intentionally benchmark-scoped and incomplete.
Do not add runtime imports from repo-root harness modules such as
`memomemo.evaluation`, `memomemo.optimizer`, benchmark optimizers, or modules
not listed above. Runtime imports must resolve from the source files in this
workspace.

## Edit Scope

You may edit only:

- `{source_snapshot_display}/candidate/**`
- `{generated_display}/**`
- `{pending_eval_display}`

Prefer editing
`{source_snapshot_display}/candidate/project_source/src/memomemo/{primary_source_file}`.
If you create a wrapper in `{generated_display}`, keep it small.

Before writing `pending_eval.json`, run a lightweight syntax/import smoke check
for the edited snapshot. If you introduce helpers such as tokenizers, counters,
or retrieval utilities, import or define them explicitly and guard empty
retrieval/class lists. Do not run the full harness evaluation.

## Required Output

Write exactly this JSON file:
`{pending_eval_display}`

Schema:

```json
{{
  "candidates": [
    {{
      "name": "short_unique_name",
      "memory_system": "fewshot_all",
      "source_family": "text_classification_fewshot",
      "source_snapshot_path": "{source_snapshot_display}",
      "extra": {{
        "source_project_path": "{source_snapshot_display}/candidate/project_source",
        "mode": "{mode}"
      }},
      "hypothesis": "why this should improve {objective_split}_accuracy",
      "changes": "brief implementation summary"
    }}
  ]
}}
```

The `candidates` array must contain exactly one candidate.
"""
