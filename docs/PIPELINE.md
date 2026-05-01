# MemoMemo Optimization Pipeline

This document consolidates, in one place, the MemoMemo proposer/evaluator
optimization loop, the three context-selection policies
(**default / progressive / bandit v3**), and the experimental results
collected so far. After reading it you should be able to:

1. Read `src/memomemo/optimizer.py` and follow the outer loop end-to-end;
2. Make an informed choice between the three policies;
3. Reproduce or extend the existing results on LoCoMo / LongMemEval / SWE-bench mini.

Detailed per-run numbers, cost tables, and the full set of run paths still
live in [`EXPERIMENT_RESULTS.md`](../EXPERIMENT_RESULTS.md) at the repo root;
this document only excerpts the headline numbers.

---

## 0. Shared Skeleton (used by all three policies)

Implemented in `LocomoOptimizer.run()` (`src/memomemo/optimizer.py:152-284`).
`SwebenchOptimizer` inherits the same outer loop and only overrides example
loading, the seed frontier, and candidate evaluation.

The three policies **diverge at exactly two points**: ① how each iteration
picks its budget / reference iterations / file hints, and ② what state it
writes back after evaluation. Everything else (workspace assembly, proposer
invocation, frontier persistence) is identical.

![Shared Skeleton](memomemo_shared_skeleton.svg)

Every iteration writes all of its artifacts to
`runs/<run_id>/proposer_calls/iter_NNN/`. The proposer itself runs inside a
Docker sandbox mounted at `/workspace/`; it cannot see the repository root,
the raw benchmark data, or the scoring helpers — those paths are blocked by
`access_policy.json`.

---

## 1. Default Policy (fixed-high baseline)

**Entry point**: `OptimizerConfig.selection_policy = "default"`
(`src/memomemo/optimizer.py:220-221`).

**Decision rule**: every iteration is hard-coded to `budget = "high"`; no
state is ever read or written.

![Default Policy: fixed high](default_policy_fixed_high.svg)

**Properties**:

- Largest context and highest cost per iteration (cache miss + long prompt +
  full copies of every reference iteration).
- No feedback loop: the proposer always sees the same global view.
- Serves as a sanity baseline for progressive / bandit.

---

## 2. Progressive Policy

**Entry point**: `OptimizerConfig.selection_policy = "progressive"`.
**State file**: `runs/<run_id>/progressive_state.json`.

**Core idea**: explicitly modulate context size with three budget tiers
(`low / medium / high`), and **promote / demote between tiers based on
whether the previous iteration produced a Pareto-frontier improvement**. As
soon as an improvement lands, drop back to `low` instead of paying the
`high` token cost iteration after iteration.

![Progressive Policy state machine](progressive_policy_state_machine.svg)

| budget | trace_scope | refs                  | prompt length |
|--------|-------------|-----------------------|---------------|
| low    | last1       | best1 + worst         | shortest      |
| medium | last3       | best3 + worst         | medium        |
| high   | all         | full history          | longest       |

**Per-iteration flow**:

1. `_progressive_budget_for_iteration(k)` runs the state machine to pick a budget;
2. `_reference_iterations_for_budget` selects refs for that budget;
3. The workspace is assembled, trimming each ref bundle's `trace_slices`
   according to `trace_scope`;
4. `build_progressive_proposer_prompt` injects
   `selection_policy="progressive"` plus best/worst role hints and the
   Optimization Focus block (memgpt: 4 cells / mini-swe-agent: 5 cells);
5. The proposer runs inside Docker and emits `pending_eval.json`;
6. `_evaluate_proposed` produces a `CandidateResult`;
7. `_update_progressive_state`: did this iteration enter a new frontier?
   If yes → `improved=True`, reset stagnation counter; if no → bump the
   budget tier upward.

**Why this works**: cheap, narrow contexts cast a wide net during the early
iterations; the budget only escalates when the run is genuinely stuck; any
new improvement immediately drops back to `low`. The average per-iteration
cost ends up far below default.

---

## 3. Bandit v3 Policy

**Entry point**: `OptimizerConfig.selection_policy = "bandit"`.
**State file**: `runs/<run_id>/bandit_state.json`.

**Core idea**: treat every readable file in the workspace as the arm of a
multi-armed bandit. Each iteration scores files by "did reading this file
correlate with a positive z-scored reward in the past?", and the top-N
files are pinned into the prompt as `hot` / `warm` so the proposer
immediately attends to the files that have actually paid off.

![Bandit v3 decision flow](bandit_v3_decision_flow.svg)

### 3.1 Top-level decision

When iteration k starts, the optimizer reads `bandit_state.json`:

```
state["files"] = {
    path → {read_iters, success_iters, reward_sum, read_lines,
            write_iters, changed_iters, utility, policy_score, …}
}
```

Files are sorted by `policy_score` (with `required core` files always pinned
into `hot` and excluded from ranking):

- `hot` = core_files + top-8, read budget = 800 lines
- `warm` = next 12, read budget = 300 lines

Budget / `trace_scope` selection:

```
iter == 1 or no file statistics yet      → low / last1 / refs=()
stagnation ≥ bandit_stagnation_threshold (=4)
                                          → high / all / full history refs
otherwise                                 → medium / last3 / (iters where
                                            hot files appeared
                                            ∪ best3 ∪ last_improved, cap 5)
```

### 3.2 Reward (the key change in v3)

```
best_eval_passrate = max(c.passrate for c in evaluated)   # v3: passrate only
history.append(best_eval_passrate)
recent = history[-bandit_reward_window:]                   # default 8;
                                                           # the published v3
                                                           # runs use 16

if not evaluated:
    reward = -clip * 0.25
elif len(recent) < 2:                                      # warm-up
    reward = clip((best_eval_passrate − previous_best) * 10, ±clip)
else:
    μ, σ = mean/std(recent),  σ ≥ 0.02
    reward = clip((best_eval_passrate − μ) / σ, ±clip)     # rolling z-score
success = reward > 0
```

Two changes from v1/v2:

1. **Passrate-only reward**: `average_score` is no longer mixed in. Mixed
   rewards on claudekimi runs were strictly worse on every metric — the
   bandit kept chasing "half-correct" candidates that gained on train but
   did not transfer to test.
2. **Rolling-window z-score**: rewards are normalized against a recent
   window rather than the global best, so a single meaningful rebound
   during a stagnation stretch can still earn a positive score.

### 3.3 Per-file utility update

Once the iteration's reward is determined:

```
For every path the proposer actually read this iteration:
    files[path].read_iters   += 1
    files[path].read_calls   += reads in this iteration
    files[path].read_lines   += lines read in this iteration
    files[path].reward_sum   += this iteration's reward
    if success: files[path].success_iters += 1

For every path that was written or appears in the diff:
    files[path].write_iters   += 1   (written)
    files[path].changed_iters += 1   (in the diff)
    (these do NOT contribute to the reward denominator)
```

`_recompute_bandit_scores` then re-scores globally (only over the "free
read" pool — `required core` is excluded):

```
p_global         = scored_success_iters / scored_read_iters
mean_reward_glob = scored_reward_sum    / scored_read_iters

p_file       = (success_iters + α·p_global) / (read_iters + α)        # Beta-smoothed
mean_reward  = (reward_sum    + α·prior_w·mean_reward_glob)
               / (read_iters + α)
avg_lines    = read_lines / read_iters
cost         = cost_λ · log1p(avg_lines / line_scale)                  # length penalty
bonus        = c · sqrt(log(total_iters + 1) / (read_iters + 1))       # UCB exploration
binary_util  = p_file       - p_global
reward_util  = mean_reward  - mean_reward_glob

policy_score = 0.7·binary_util + 0.3·reward_util − cost + bonus
```

| Hyperparameter                  | Default |
|---------------------------------|--------:|
| `bandit_prior_alpha`            |   2.0   |
| `bandit_prior_weight`           |   0.4   |
| `bandit_exploration_c`          |   0.15  |
| `bandit_cost_lambda`            |   0.05  |
| `bandit_line_scale`             |   500   |
| `bandit_reward_window`          |   8     |
| `bandit_reward_sigma_floor`     |   0.02  |
| `bandit_reward_clip`            |   2.0   |
| `bandit_stagnation_threshold`   |   4     |
| `bandit_failed_iter_penalty`    |   0.5   |

### 3.4 Required core files (always hot, never scored)

`_bandit_core_files` (`src/memomemo/optimizer.py:1951`) keeps a fixed set of
"foundation files" pinned into the hot list regardless of statistics:

- The scaffold source file for the current `source_family`
  (e.g. `memgpt_source.py`);
- `scaffolds/base.py`, `model.py`, `schemas.py`;
- The six summary files under `summaries/`
  (`evolution_summary.jsonl`, `best_candidates.json`,
  `candidate_score_table.json`, `retrieval_diagnostics_summary.json`,
  `diff_summary.jsonl`, `iteration_index.json`);
- `pending_eval.json`.

These files are excluded from bandit scoring so that small-sample noise
cannot demote essential files into warm/cold.

---

## 4. Side-by-side comparison

| Dimension              | default              | progressive                                  | bandit v3                                            |
|------------------------|----------------------|----------------------------------------------|------------------------------------------------------|
| Budget selection       | always `high`        | state machine (`low`→…→`high`)               | heuristic + stagnation threshold (`low/medium/high`) |
| Reference iterations   | full history         | by budget: best k + worst                    | iters where hot files appeared, fallback to best3 / last_improved |
| Trace-scope trim       | `all`                | `last1 / last3 / all`                        | same three tiers, derived from budget                |
| Prompt extras          | Optimization Focus   | + Progressive role hints                     | + Bandit Context Policy (hot/warm lists)             |
| Feedback signal        | none                 | did this iter enter a new frontier?          | rolling-window z-score (passrate only)               |
| State file             | none                 | `progressive_state.json`                     | `bandit_state.json` (per-file stats)                 |
| Explore vs exploit     | none (always max)    | implicit (only escalates on stagnation)      | explicit (UCB bonus + Beta smoothing)                |
| Cost profile           | highest (every iter is high) | medium (most iters are low)          | medium-high (more reads + policy meta)               |
| Strengths              | reproduces baseline; sanity check | best on LoCoMo/LongMemEval for claudekimi/opus | only policy where codex54 beats progressive on LoCoMo test |
| Weaknesses             | never converges      | hard to drop back from `high` once escalated | needs a warm-up phase; mixed reward overfits train  |

---

## 5. Experimental results summary

Detailed numbers, per-iteration costs, and all run paths are in
[`EXPERIMENT_RESULTS.md`](../EXPERIMENT_RESULTS.md). This section lists only
the conclusions.

### 5.1 LoCoMo (train=80, test=1449)

**Test passrate** (bold = best within that proposer family):

| proposer family | default | progressive (docker) | bandit (docker) | bandit v2 (docker) | bandit v3 (docker) | best          |
|-----------------|--------:|---------------------:|----------------:|-------------------:|-------------------:|---------------|
| claudekimi      | 0.3409  | **0.3734**           | 0.3616          | 0.3395             | 0.3589             | progressive   |
| claude opus     | 0.3306  | **0.3982**           | 0.3230          | N/A                | N/A                | progressive ★ |
| codex54         | 0.3471  | 0.3589               | 0.3140          | 0.3575             | **0.3865**         | bandit v3     |

Highlights:

- **Global best on LoCoMo: claude opus progressive docker, 0.3982 test.**
- progressive wins outright on claudekimi / opus; codex54 is the only
  family where the bandit overtakes progressive. bandit v3 (0.3865) is
  also the only bandit result on any proposer that beats progressive.
- bandit v3 nearly halves codex54's cost (input/iter 2.39M → 1.13M) with
  no test regression.

### 5.2 LongMemEval (train=100, test=400)

| proposer family | default     | progressive          | bandit  | best        |
|-----------------|------------:|---------------------:|--------:|-------------|
| claude opus46   | N/A         | failed: Together 500  | N/A     | —           |
| claudekimi      | 0.4700      | **0.5000**            | 0.4400  | progressive |
| codex54         | **0.4875**  | 0.4725                | 0.4575  | default     |

Takeaway: bandit v2 no longer leads on LongMemEval test —
progressive/default come out ahead. opus46 has the strongest train number
(0.6300) but its test-frontier run aborted on a Together 500 and is not
counted.

### 5.3 SWE-bench mini

The source-code backend (`mini_swe_agent_source`) is wired into the
optimize CLI (`--swebench`). The current run with a meaningful signal is
mimo v2.5 trainfirst30:

| policy                 | source baseline | best optimizer candidate                            | best passrate | iters     |
|------------------------|----------------:|-----------------------------------------------------|--------------:|----------:|
| claudekimi default     | 0.4667          | iter002 `stack_trace_context` and 4 ties            | **0.5000**    | 20/30     |
| claudekimi progressive | 0.4000          | `iter016_final_fallback_traceback_retrieval_v1`     | **0.5333**    | 20/30     |

The more important signal is the full DeepSeek v4 Flash evaluation on the
500-problem verified set:

| candidate                                                               | resolved/500 | passrate    |
|-------------------------------------------------------------------------|-------------:|------------:|
| source baseline                                                         | 220 / 500    | 0.4400      |
| default optimized (`iter002_stack_trace_context`)                       | 229 / 500    | 0.4580      |
| progressive optimized (`iter016_final_fallback_traceback_retrieval_v1`) | 310 / 500    | **0.6200**  |

The progressive-optimized candidate beats the source baseline by +18.0
percentage points on full verified — currently the strongest SWE-bench
result. verified_test10 (10 problems) is too small and saturates too
quickly for the optimizer to make stable improvements.

### 5.4 Overall takeaways

- LoCoMo global best: claude opus progressive docker @ 0.3982 test.
- LongMemEval no longer supports the "bandit always wins" claim:
  progressive / default beat bandit v2 on test.
- On LoCoMo, progressive is the safe winner for claudekimi / opus;
  codex54 is the family that benefits from bandit v3.
- train80 LoCoMo on its own is too noisy and should always be paired with
  test 1449; claudekimi bandit is the clearest example of train/test
  inconsistency.
- The first real optimization signal on SWE-bench mini comes from
  progressive: trainfirst30 0.5333 (vs baseline 0.4667), and DeepSeek
  Flash full500 0.6200 (vs baseline 0.4400).
- All progressive / bandit results above use the docker sandbox;
  non-docker results are not counted.
