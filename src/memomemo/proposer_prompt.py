"""Prompt builder for Claude proposer iterations."""

from __future__ import annotations

from pathlib import Path


def build_proposer_prompt(
    *,
    run_id: str,
    iteration: int,
    run_dir: Path,
    generated_dir: Path,
    source_snapshot_dir: Path | None = None,
    pending_eval_path: Path,
    frontier_path: Path,
    summary_path: Path,
    split: str,
    limit: int,
) -> str:
    """Build the Claude proposer prompt."""

    reports_dir = run_dir / "reports"
    snapshot_dir = source_snapshot_dir or generated_dir / "source_snapshots" / f"iter_{iteration:03d}"

    return f"""# MemoMemo Proposer — iteration {iteration}

You are optimizing the memory layer for LOCOMO conversational-memory QA.

Run ONE iteration of memory-scaffold evolution. Do all work in this main Claude
session. Do not delegate to subagents.

You do NOT run the full harness evaluation. The outer MemoMemo harness imports
and evaluates the candidate after this session exits.

## Assignment

- Pareto frontier: `{frontier_path}`
- Evolution summary: `{summary_path}`
- Generated workspace: `{generated_dir}`
- Writable source snapshot: `{snapshot_dir}`
- Eval split: `{split}`
- Eval limit: `{limit}` (`0` means full split)

## Objective

Primary objective: maximize `passrate`.

Secondary objective: reduce `token_consuming` only among candidates with
comparable `passrate`. `average_score` is reported as a diagnostic because it
is strongly coupled to `passrate`, but it is not an optimization objective. Do
not propose a lower-recall compression mechanism just because it is cheaper; a
candidate that saves tokens but is expected to lower answer quality is a failed
candidate.

## Critical Constraints

- You MUST implement exactly 1 new candidate every iteration.
- Do not stop early, claim the frontier is optimal, or write an empty
  `pending_eval.json`.
- The candidate must derive from one explicitly chosen parent candidate or
  source scaffold.
- Avoid candidates that are only parameter tuning. Changing only `top_k`,
  `window`, thresholds, weights, or context budgets is not enough.
- Good candidates change an agent-system mechanism: memory construction,
  retrieval algorithm, evidence selection, context formatting, answering
  prompt, diversity/coverage policy, temporal reasoning, entity handling, or
  compression strategy.
- No dataset-specific hints or hardcoded LOCOMO answers. General conversation
  memory heuristics are fine.
- Source-backed baseline memories under `runs/source_base_memory/**` are
  read-only artifacts. If you change mem0, MemGPT/Letta, or
  MemoryBank/SiliconFriend source code, do it only in the candidate-specific
  source snapshot and route the candidate through that snapshot with explicit
  source paths; never mutate `references/vendor/**` or an existing source-base
  directory. Rebuilding source bases is expensive, so make source construction
  changes intentional and scoped.

## Available Files

- `{summary_path}` — evolution summary.
- `{frontier_path}` — current Pareto frontier.
- `{reports_dir}/` — report workspace and existing concise post-eval reports.
- `runs/{run_id}/candidate_results/` — evaluated candidate summaries and traces.
- `src/memomemo/` — project source.
- `src/memomemo/scaffolds/` — built-in scaffold source.
- `references/vendor/` — upstream source checkouts.
- `{generated_dir}/` — writable experiment candidate workspace.
- `{snapshot_dir}/candidate/` — pre-created writable source snapshot for this
  iteration. Use `extra.source_project_path` to route built-in source
  scaffolds through edited `project_source`, and use source-specific paths such
  as `extra.mem0_source_path` for edited upstream code.

The harness has already created a candidate-specific writable source snapshot:
`{snapshot_dir}/candidate/`. Modify that snapshot, not `references/vendor/**`.
It contains full project source under `project_source/src/memomemo` and
upstream source under `upstream_source` for mem0, MemGPT/Letta, and
MemoryBank/SiliconFriend. You may modify any copied source file, including
memory construction, extraction, evolution, embedding, persistence, retrieval,
context formatting, and answer logic. Source-code variants that alter persisted
memories must use a variant-specific `source_base_dir` so the harness rebuilds
or loads the matching base memories.

The candidate should have a genuinely different mechanism; it must not be just
a constant-only variant of an existing scaffold.

## Scaffold References

Implemented memory scaffolds are available for reference:

- `src/memomemo/scaffolds/bm25_scaffold.py`
  Reference: https://github.com/dorianbrown/rank_bm25.git
- `src/memomemo/scaffolds/mem0_scaffold.py`
  Source-backed scaffold: `mem0_source`
  Reference: https://github.com/mem0ai/mem0
- `src/memomemo/scaffolds/memgpt_scaffold.py`
  Source-informed scaffold: `memgpt_source`
  Reference: https://github.com/cpacker/MemGPT
- `src/memomemo/scaffolds/membank_scaffold.py`
  Source-informed scaffold: `membank_source`
  Reference: https://github.com/zhongwanjun/MemoryBank-SiliconFriend
- `references/sources.yaml`
- `references/vendor/mem0/`
- `references/vendor/MemGPT/`
- `references/vendor/MemoryBank-SiliconFriend/`

Candidate mechanisms may compose these patterns, simplify them, add scoring
features, change memory formatting, change retrieval mechanisms, change the
answering prompt, or create a new scaffold class.

## Allowed Edit Scope

You may edit:

- `{generated_dir}/**`
- `runs/{run_id}/reports/**`

When modifying source-backed logic, edit
`{snapshot_dir}/candidate/`, then either write a small importable wrapper under
`{generated_dir}/` or route a built-in source scaffold through the snapshot
with `extra.source_project_path`.
The copied source is editable for experiment candidates, including upstream
mem0 build/database-construction logic and copied memgpt/membank scaffold
logic. Keep existing persistent
source bases read-only; use a new candidate-specific `source_base_dir` when the
candidate changes persisted memory semantics.
Do not edit package source under `src/` for experiment candidates unless the
outer task explicitly asks for harness changes.
Do not edit LOCOMO raw data or past result JSON.

## Quality Gate

Before writing `pending_eval.json`, verify that the candidate is a real
mechanism change, is not just a `top_k`/`window`/threshold/weight variant, does
not use gold answers at inference time, does not hardcode LOCOMO-specific
answers, and uses an isolated source snapshot plus fresh `source_base_dir` for
any source edit that changes persisted memory construction.

## Candidate Interface

Each proposed candidate should be importable and instantiate a `MemoryScaffold`.
Recommended pattern:

```python
from memomemo.scaffolds.base import MemoryScaffold, ScaffoldConfig, ScaffoldRun

class MyCandidateScaffold(MemoryScaffold):
    name = "my_candidate"
    def build(self, example, config): ...
    def answer(
        self,
        state,
        example,
        client,
        config: ScaffoldConfig,
        *,
        max_context_chars: int,
        dry_run: bool,
    ) -> ScaffoldRun: ...
```

## Required Output

Write exactly this JSON file:
`{pending_eval_path}`

Schema:

```json
{{
  "candidates": [
    {{
      "name": "short_unique_name",
      "module": "my_candidate",
      "class": "MyCandidateScaffold",
      "top_k": 8,
      "window": 1,
      "parent_candidate_id": "chosen_parent_candidate_id",
      "source_family": "mem0|memgpt|membank|fusion",
      "build_tag": "stable_build_identifier",
      "extra": {{}},
      "hypothesis": "why this should improve passrate",
      "changes": "brief implementation summary"
    }}
  ]
}}
```

Notes:

- `module` + `class` are required for generated candidates.
- `module` is resolved from the run-local generated workspace above. Use a bare
  module name such as `"my_candidate"` for `{generated_dir}/my_candidate.py`.
- `top_k` must be a single integer. Do not propose multiple `top_k` values;
  this iteration should test one mechanism, not spawn parameter variants.
- You can also propose a built-in scaffold by using `"scaffold_name": "bm25"` only as
  part of a generated composition or explicit control candidate; generated
  candidates are preferred.
- The `candidates` array MUST contain exactly 1 candidate.
- `parent_candidate_id` and `source_family` must reflect the parent/source you
  selected during analysis.
- `build_tag` identifies memory-construction logic. Reuse the parent's build
  tag only when persisted memory construction is unchanged. If the candidate
  changes mem0 construction/extraction/embedding/persistence
  code, choose a new stable `build_tag`, set `extra.source_base_dir` to a fresh
  absolute path under `runs/{run_id}/source_base_variants/`, and set
  `extra.mem0_source_path` to the edited snapshot upstream directory. For mem0 it is
  `{snapshot_dir}/candidate/upstream_source/mem0`. If you edit
  `project_source/src/memomemo/scaffolds/mem0_scaffold.py`,
  `memgpt_scaffold.py`, or `membank_scaffold.py`, set
  `extra.source_project_path` to `{snapshot_dir}/candidate/project_source`.
- `changes` must summarize the implemented mechanism, not just parameter
  values.
"""


def build_ucb_proposer_prompt(
    *,
    run_id: str,
    iteration: int,
    run_dir: Path,
    pending_eval_path: Path,
    frontier_path: Path,
    summary_path: Path,
    context_dir: Path,
    generated_dir: Path,
    source_snapshot_dir: Path,
    intend_path: Path,
    parent_candidate_id: str,
    source_family: str,
    cost_level: str,
    split: str,
    limit: int,
) -> str:
    """Build the UCB-guided Claude proposer prompt."""

    return f"""# MemoMemo UCB Proposer — iteration {iteration}

You are optimizing the memory layer for LOCOMO conversational-memory QA.
This is a UCB-guided iteration: the outer optimizer already selected one
lineage parent and one context budget.

Do NOT run the full harness evaluation. The outer MemoMemo harness imports and
evaluates your candidate after this session exits.

## UCB Assignment

- Parent candidate: `{parent_candidate_id}`
- Source lineage: `{source_family}`
- Context budget: `{cost_level}`
- Context snapshot: `{context_dir}`
- Generated workspace: `{generated_dir}`
- Writable source snapshot: `{source_snapshot_dir}`
- Pareto frontier: `{frontier_path}`
- Full evolution summary: `{summary_path}`. For `low`/`medium`, the context
  snapshot also contains a recent summary copy.
- Eval split: `{split}`
- Eval limit: `{limit}` (`0` means full split)

Budget meaning:

- `low`: selected parent and the most recent 2 prior iteration bundles copied
  into the context snapshot. Trace slices contain at most 3 full task traces
  per candidate per iteration.
- `medium`: selected parent and the most recent 5 prior iteration bundles
  copied into the context snapshot. Trace slices contain at most 10 full task
  traces per candidate per iteration.
- `high`: source, parent, trace, and candidate-result manifests, including
  full trace slices.

## Objective

Primary objective: maximize `passrate`.

Secondary objective: reduce `token_consuming` only among candidates with
comparable `passrate`. `average_score` is reported as a diagnostic because it
is strongly coupled to `passrate`, but it is not an optimization objective. The
frontier uses a quality threshold: candidates that are much lower quality can
be excluded even when they are cheaper. Do not optimize token use by
sacrificing answer quality; if broader evidence coverage is likely to recover
correct answers, prefer quality over compression.

## Constraints

- Implement exactly 1 new candidate.
- The candidate must be derived from the assigned parent lineage.
- Avoid candidates that are only parameter tuning. Changing only `top_k`,
  `window`, thresholds, weights, or context budgets is not enough.
- No dataset-specific hints or hardcoded LOCOMO answers.
- Do not edit LOCOMO raw data or past result JSON.
- Source-backed baseline memories under `runs/source_base_memory/**` are
  read-only artifacts. You may modify copied mem0, MemGPT/Letta, and
  MemoryBank/SiliconFriend source code in the candidate-specific source
  snapshot, including construction, extraction, evolution, embedding, schema,
  persistence, retrieval, context formatting, and answer logic. If
  persisted-memory semantics change, route the candidate through the edited
  snapshot with an explicit source path and fresh `source_base_dir`. Rebuilding
  source bases is expensive, so make source construction changes intentional
  and scoped.

## Available Files

- `{context_dir}` — context snapshot assembled for this UCB iteration.
- `{context_dir}/CONTEXT.md` — manifest-style guide to snapshot contents.
- `{source_snapshot_dir}/candidate/` — writable source snapshot for this
  candidate. It is a candidate-specific copy of the selected parent/source, not
  the ignored vendor checkout. It includes full project source under
  `project_source/src/memomemo` and relevant upstream source under
  `upstream_source`. You may edit any copied source file. Set
  `extra.source_project_path` to route built-in source scaffolds through edited
  project source. Source edits that alter persisted memories must use a new
  candidate-specific `source_base_dir` so the harness does not reuse
  incompatible base memories.
- `{generated_dir}` — writable generated module workspace.
- `{intend_path}` — optional design-intent note location.
- `{frontier_path}` — current Pareto frontier.
- `{summary_path}` — full evolution summary.
- `runs/{run_id}/candidate_results/` — candidate summaries and task traces.
- `src/memomemo/` — project source.
- `references/vendor/` — upstream source checkouts.

## Allowed Edit Scope

You may edit:

- `{generated_dir}/**`
- `{intend_path}`

The candidate-specific snapshot is under `{source_snapshot_dir}/candidate`.
Generated importable wrappers belong under `{generated_dir}`. You may also
propose a built-in source scaffold with `extra.source_project_path` pointing to
`{source_snapshot_dir}/candidate/project_source`.
The copied source is editable for experiment candidates, including upstream
mem0 build/database-construction logic and the copied memgpt/membank scaffold
logic. Keep existing persistent
source bases read-only; use a new candidate-specific `source_base_dir` when the
candidate changes persisted memory semantics.

## Quality Gate

Before writing `pending_eval.json`, verify that the candidate is a real
mechanism change, is not just a `top_k`/`window`/threshold/weight variant, does
not use gold answers at inference time, does not hardcode LOCOMO-specific
answers, and uses an isolated source snapshot plus fresh `source_base_dir` for
any source edit that changes persisted memory construction.

## Required Output

Write exactly this JSON file:

`{pending_eval_path}`

Schema:

```json
{{
  "candidates": [
    {{
      "name": "short_unique_name",
      "module": "my_candidate",
      "class": "MyCandidateScaffold",
      "top_k": 8,
      "window": 1,
      "parent_candidate_id": "{parent_candidate_id}",
      "source_family": "{source_family}",
      "cost_level": "{cost_level}",
      "build_tag": "stable_build_identifier",
      "extra": {{}},
      "hypothesis": "falsifiable mechanism claim",
      "changes": "brief mechanism summary"
    }}
  ]
}}
```

Notes:

- `module` + `class` are required for generated candidates.
- `module` is resolved from the run-local generated workspace above. Use bare
  module names such as `"my_candidate"` for files directly under
  `{generated_dir}/`.
- `top_k` must be a single integer. Do not propose multiple `top_k` values;
  this iteration should test one mechanism, not spawn parameter variants.
- `build_tag` identifies memory-construction logic. Reuse the parent's tag only
  when persisted memory construction is unchanged. If the candidate changes
  mem0 construction/extraction/embedding/persistence code,
  choose a new stable `build_tag`, set `extra.source_base_dir` to a fresh
  absolute path under `runs/{run_id}/source_base_variants/`, and set
  `extra.mem0_source_path` to the edited upstream snapshot directory. For this
  iteration, mem0 source lives at
  `{source_snapshot_dir}/candidate/upstream_source/mem0`. If you edit
  `project_source/src/memomemo/scaffolds/mem0_scaffold.py`,
  `memgpt_scaffold.py`, or `membank_scaffold.py`, set
  `extra.source_project_path` to `{source_snapshot_dir}/candidate/project_source`.
- `hypothesis` and `changes` must describe mechanisms, not just constants.
- The candidates array MUST contain exactly 1 candidate.
"""
