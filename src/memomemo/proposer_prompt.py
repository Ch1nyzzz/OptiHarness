"""Prompt builder for Claude proposer iterations."""

from __future__ import annotations

from pathlib import Path


def build_proposer_prompt(
    *,
    run_id: str,
    iteration: int,
    run_dir: Path,
    pending_eval_path: Path,
    frontier_path: Path,
    summary_path: Path,
    split: str,
    limit: int,
) -> str:
    """Build the Claude proposer prompt."""

    reports_dir = run_dir / "reports"

    return f"""# MemoMemo Proposer — iteration {iteration}

You are optimizing the memory layer for LOCOMO conversational-memory QA.
This project is intentionally separate from `skillevolve`, but follows the
same proposer pattern: inspect summarized history and selected traces, edit code, write
`pending_eval.json`, and let the harness evaluate your candidates.

Run ONE iteration of memory-scaffold evolution. Do all work in this main Claude
session. Do not delegate to subagents.

You do NOT run the full harness evaluation. You analyze prior results,
prototype scaffold ideas locally, implement new candidate scaffolds, validate
imports/lightweight behavior, and write `pending_eval.json`. The outer
MemoMemo harness imports and evaluates the candidates after this session exits.

## Objective

Maximize `passrate` and minimize `token_consuming`.

The Pareto frontier is stored at:
`{frontier_path}`

The run summary is stored at:
`{summary_path}`

Current eval split: `{split}`
Current eval limit: `{limit}` (`0` means full split)

## Critical Constraints

- You MUST implement exactly 2 new candidates every iteration.
- Do not stop early, claim the frontier is optimal, or write an empty
  `pending_eval.json`.
- Mix exploitation and exploration: at least one candidate should improve a
  current frontier mechanism, and at least one should try a distinct mechanism.
- Avoid candidates that are only parameter tuning. Changing only `top_k`,
  `window`, thresholds, weights, or context budgets is not enough.
- Good candidates change an agent-system mechanism: memory construction,
  retrieval algorithm, evidence selection, context formatting, answering
  prompt, diversity/coverage policy, temporal reasoning, entity handling, or
  compression strategy.
- No dataset-specific hints or hardcoded LOCOMO answers. General conversation
  memory heuristics are fine.

## Workflow

### Step 0: Post-eval reports

Check `{reports_dir}`. For any completed previous iteration that appears in
`{summary_path}` but has no report, write a concise report under
`{reports_dir}/iter_<NNN>.md`.

Each report should be <=30 lines and cover:

- what changed,
- which candidates improved or regressed on `passrate` / `token_consuming`,
- likely reasons based on task-level retrieved evidence,
- one takeaway for future iterations.

### Step 1: Analyze

Read the compact state first:

- `{summary_path}` — what has been tried,
- `{frontier_path}` — current Pareto frontier,
- `{reports_dir}/` — concise post-eval reports,
- relevant scaffold implementations in `src/memomemo/scaffolds/` and
  `src/memomemo/generated/`.

Then deep-read trajectory traces only as needed. Use recent or selected
`runs/{run_id}/candidate_results/<candidate_id>.json` files to inspect
task-level predictions, retrieved hits, scores, and token usage for the
mechanisms you are considering. Do not scan every candidate result by default.

Formulate exactly 2 falsifiable hypotheses. Each should target a different
mechanism, not only a different parameter value.

### Step 2: Prototype — mandatory

Before writing final candidate files, prototype each mechanism in `/tmp/`.

For each candidate:

1. Write a small `/tmp/` script that exercises the core retrieval/scoring logic
   in isolation.
2. Pull real examples or retrieved traces from recent or selected
   `candidate_results/<candidate_id>.json` files.
3. Try 2-3 variants of the mechanism and compare them qualitatively or with a
   small deterministic proxy.
4. Delete the `/tmp/` script when done.

Do not skip prototyping. If the prototype shows the mechanism is weak, change
the mechanism before implementing the final candidate.

### Step 3: Implement

Implement exactly 2 candidate scaffolds, preferably under `src/memomemo/generated/`.
Each candidate should have a genuinely different mechanism. It is fine to
compose BM25, A-Mem, and Mem0-style patterns, but the result must not be just a
constant-only variant of an existing scaffold.

After implementing each candidate, self-critique it:

- Is this candidate really a new mechanism?
- Did it change retrieval/memory behavior beyond `top_k`, `window`, or numeric
  weights?
- Can it work without seeing gold answers at inference time?
- Is it general conversation-memory logic rather than LOCOMO-specific leakage?

If the answer is weak, rewrite the candidate before continuing.

Validate each candidate with a lightweight import or smoke command before
writing `pending_eval.json`.

## Scaffold References

Initial memory scaffolds are already implemented:

- `src/memomemo/scaffolds/bm25_scaffold.py`
  Reference: https://github.com/dorianbrown/rank_bm25.git
- `src/memomemo/scaffolds/amem_scaffold.py`
  Reference: https://github.com/WujiangXu/A-mem
- `src/memomemo/scaffolds/mem0_scaffold.py`
  Reference: https://github.com/mem0ai/mem0

Reference manifest:
`references/sources.yaml`

You may compose these patterns, simplify them, add scoring features, change
memory formatting, change retrieval mechanisms, change the answering prompt,
or create a new scaffold class.

## Allowed Edit Scope

You may edit:

- `src/memomemo/scaffolds/**`
- `src/memomemo/generated/**`
- `src/memomemo/utils/**`
- `src/memomemo/metrics.py`

Prefer writing new candidates under `src/memomemo/generated/`.
Do not edit LOCOMO raw data or past result JSON.

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

### Step 4: Write `pending_eval.json`

## Required Output

Write exactly this JSON file:
`{pending_eval_path}`

Schema:

```json
{{
  "candidates": [
    {{
      "name": "short_unique_name_a",
      "module": "memomemo.generated.my_candidate_a",
      "class": "MyCandidateAScaffold",
      "top_k": [4, 8],
      "window": 1,
      "hypothesis": "why this should improve passrate/token_consuming",
      "changes": "brief implementation summary"
    }},
    {{
      "name": "short_unique_name_b",
      "module": "memomemo.generated.my_candidate_b",
      "class": "MyCandidateBScaffold",
      "top_k": [4, 8],
      "window": 1,
      "hypothesis": "why this should improve passrate/token_consuming",
      "changes": "brief implementation summary"
    }}
  ]
}}
```

Notes:

- `module` + `class` are required for generated candidates.
- `top_k` may be an integer or a list of integers.
- You can also propose a built-in scaffold by using `"scaffold_name": "bm25"` only as
  part of a generated composition or explicit control candidate; generated
  candidates are preferred.
- The `candidates` array MUST contain exactly 2 candidates.
- `hypothesis` must be falsifiable and name the mechanism being tested.
- `changes` must summarize the implemented mechanism, not just parameter
  values.

## Useful Files

- `src/memomemo/evolution.py` — evaluation loop
- `src/memomemo/scaffolds/base.py` — scaffold interface and helper methods
- `src/memomemo/schemas.py` — data structures
- `src/memomemo/locomo.py` — dataset loader
- `runs/{run_id}/candidate_results/<candidate_id>.json` — selected task-level
  retrieval traces and scores
- `runs/{run_id}/pareto_frontier.json` — current frontier
- `{reports_dir}` — concise post-eval reports for future proposer iterations

Start by reading the frontier, evolution summary, and reports. Then inspect a
few recent or selected task failures from candidate results only when needed.
Diagnose first, write missing reports, prototype two mechanisms, implement
exactly two candidates, self-critique and validate them, then write
`pending_eval.json`.
"""
