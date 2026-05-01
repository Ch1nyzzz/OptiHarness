# MemoMemo Optimization Pipeline

This document explains the three context-selection policies used by the
MemoMemo proposer/evaluator loop — **default**, **progressive**, and
**bandit v3** — and how they plug into the shared optimizer skeleton.
The goal is to give a collaborator enough mental model to read
`src/memomemo/optimizer.py` and reproduce/extend any of the runs reported
in `EXPERIMENT_RESULTS.md`.

All three policies share the same OptiHarness-style outer loop. They only
differ in how each iteration picks its context (budget, reference
iterations, file hints) and what state it writes back after evaluation.

---

## 0. Shared Skeleton (identical across policies)

Implemented in `LocomoOptimizer.run()` (`src/memomemo/optimizer.py:152-284`).
`SwebenchOptimizer` inherits the same skeleton and only overrides the
example loader, seed-frontier function, and candidate evaluation.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     LocomoOptimizer.run()                            │
│             (SwebenchOptimizer inherits the same skeleton)           │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
        ┌─────────────────── Setup phase ───────────────────┐
        │ 1) _load_examples()    load LOCOMO / SWE-bench    │
        │ 2) _run_seed_frontier() OR baseline_dir loader    │
        │    – evaluate seed scaffolds as iteration-0 cands │
        │ 3) _save_best_candidates / _refresh_run_indexes   │
        │    – write best_candidates.json + indexes         │
        └───────────────────────────────────────────────────┘
                               │
        ┌──────── for iteration in 1 ... N ────────┐
        ▼                                            │
┌────────────────────────────────────────────┐       │
│ ① Pick context strategy (3-policy fork)    │       │
│   default     → budget = "high"            │       │
│   progressive → _progressive_budget_…()    │       │
│   bandit      → _bandit_policy_for_…()     │       │
└────────────────────────────────────────────┘       │
        │                                            │
        ▼                                            │
┌────────────────────────────────────────────────┐    │
│ ② _run_progressive_proposer_iteration(...)     │    │
│   a. _build_progressive_workspace              │    │
│      - copy clean source_snapshot/             │    │
│      - copy summaries/ (cumulative state)      │    │
│      - copy reference_iterations/ bundles      │    │
│      - write access_policy.json (sandbox)      │    │
│   b. build_progressive_proposer_prompt(...)    │    │
│      - inject budget / refs / focus / policy   │    │
│   c. _run_proposer_agent(...)                  │    │
│      - run claude / kimi / codex inside docker │    │
│      - expect workspace/pending_eval.json out  │    │
│   d. _normalize / _archive_workspace_outputs   │    │
│      - copy diff, tool_access, metrics to      │    │
│        proposer_calls/iter_NNN/                │    │
│   e. _evaluate_proposed                        │    │
│      - load_candidate_scaffold + EvaluationRunner   │
│      - run on LOCOMO / SWE-bench split         │    │
│      - emit CandidateResult (passrate, score) │    │
└────────────────────────────────────────────────┘    │
        │                                            │
        ▼                                            │
┌────────────────────────────────────────────────┐    │
│ ③ State write-back (policy-specific)           │    │
│   default      → no-op                         │    │
│   progressive  → _update_progressive_state(...)│    │
│   bandit       → _update_bandit_state(...)     │    │
└────────────────────────────────────────────────┘    │
        │                                            │
        ▼                                            │
┌────────────────────────────────────────────────┐    │
│ ④ Persist frontier and rolling summaries       │    │
│   - best_candidates.json (Pareto frontier)     │    │
│   - candidate_score_table.json                 │    │
│   - evolution_summary.jsonl                    │    │
│   - diff_summary.jsonl, iteration_index.json   │    │
└────────────────────────────────────────────────┘    │
        └────────────────────────────────────────────┘
                               │
                               ▼
              optimizer_summary.json + optional test_frontier
```

Each iteration writes everything under
`runs/<run_id>/proposer_calls/iter_NNN/`. The proposer runs inside a
Docker sandbox mounted at `/workspace/`; it cannot see the repo root,
the raw benchmark data, or scoring helpers — those are blocked by
`access_policy.json`.

---

## 1. Default Policy (fixed-high baseline)

**Entry**: `OptimizerConfig.selection_policy = "default"`
(`src/memomemo/optimizer.py:220-221`).

**Decision rule**: every iteration hard-codes `budget = "high"`. No
state file is read or written.

```
┌──────────────────────────────────┐
│   Per-iteration k                │
│                                  │
│   budget        = "high"         │  ← hard-coded
│   trace_scope   = "all"          │  ← _trace_scope_for_budget("high")
│   refs          = all past iters │  ← _reference_iterations_for_budget
│   bandit_policy = None           │
│   adaptive      = False          │
└──────────────────────────────────┘
              │
              ▼
   Workspace contents:
   - reference_iterations/iter_001..k-1 (full set)
   - trace_slices keep low + medium + high
   - prompt has NO “Optimization Focus”
   - prompt has NO “Bandit Context Policy”
              │
              ▼
   Evaluate → CandidateResult; only frontier files updated
```

**Properties**

- Largest, most expensive context every iteration (cache miss + huge
  prompt + many ref-iter copies).
- No feedback loop: the proposer sees the same global view forever.
- Acts as the sanity baseline for both progressive and bandit.

---

## 2. Progressive Policy

**Entry**: `OptimizerConfig.selection_policy = "progressive"`.
**State**: `runs/<run_id>/progressive_state.json`.

**Idea**: explicitly modulate context size with three budgets
(low / medium / high), and **promote/demote based on whether the
previous iteration produced a Pareto-frontier improvement**.

```
┌─────────────── budget state machine ───────────────┐
│                                                     │
│   iter ≤ progressive_initial_low_iterations         │
│   (default 5)  ────────────────► force "low"        │
│                                                     │
│   otherwise read progressive_state.json:            │
│     improved == True   →  next "low"   (reset)      │
│     improved == False  →  low    → medium           │
│                            medium → high            │
│                            high   → high  (sat)     │
└─────────────────────────────────────────────────────┘

           ┌─────────────┬──────────────┬─────────────┐
budget     │  low        │  medium      │  high       │
trace_scope│  last1      │  last3       │  all        │
refs       │  best1+worst│  best3+worst │  all iters  │
prompt     │  shortest   │  medium      │  longest    │
─────────────────────────────────────────────────────
```

**Per-iteration flow**

```
  ┌── enter iteration k ──┐
  │
  │ ① budget = _progressive_budget_for_iteration(k)
  │     (state machine above)
  │
  │ ② refs = _reference_iterations_for_budget(budget, …)
  │     - low:    best_iterations(k=1) + worst_iteration
  │     - medium: best_iterations(k=3) + worst_iteration
  │     - high:   all available iterations
  │
  │ ③ Workspace assembly
  │     - reference_iterations/iter_xxx (per refs)
  │     - prune each bundle’s trace_slices by trace_scope:
  │         last1 → keep trace_slices/low only
  │         last3 → keep trace_slices/medium only
  │         all   → keep low + medium + high
  │
  │ ④ build_progressive_proposer_prompt
  │     - selection_policy = "progressive"
  │     - inject "Progressive reference roles: best=… worst=…"
  │     - inject "Optimization Focus" cells
  │       (memgpt: 4 cells / mini-swe-agent: 5 cells)
  │
  │ ⑤ Proposer in docker → pending_eval.json
  │ ⑥ _evaluate_proposed → CandidateResult
  │ ⑦ _update_progressive_state(...):
  │     improved = any evaluated candidate enters
  │                (frontier_ids \ previous_frontier_ids)
  │     stagnation_count = 0 if improved else +1
  │     next_budget = (state machine above)
  └───────────────────────┘
```

**Why it works**: cheap narrow context for the first few iterations
samples the search space efficiently; only when stagnation hits does
the budget escalate. As soon as a real improvement lands, the budget
collapses back to `low`, so we don’t pay `high` token cost forever.

---

## 3. Bandit v3 Policy

**Entry**: `OptimizerConfig.selection_policy = "bandit"`.
**State**: `runs/<run_id>/bandit_state.json`.

**Idea**: treat each readable file in the workspace as an arm of a
multi-armed bandit. After every iteration, score each file by how
correlated its reads are with positive z-score reward. Surface the top
files in the prompt (`hot` / `warm`) so the proposer starts where past
iterations have actually paid off.

### 3.1 Top-level decision diagram

```
┌──────────────────────────────────────────────────────────┐
│  iter k entry: _bandit_policy_for_workspace              │
│                                                          │
│  state ← bandit_state.json                               │
│    state["files"] = {                                    │
│       path → {read_iters, success_iters, reward_sum,     │
│               read_lines, write_iters, changed_iters,    │
│               utility, policy_score, …}                  │
│    }                                                     │
└──────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│   Sort files by policy_score (excluding required core)   │
│   hot   = core_files + top-8                             │
│   warm  = next 12                                        │
│   read_budget_lines: hot = 800, warm = 300               │
└──────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│   Budget / trace_scope decision                          │
│                                                          │
│   if iter == 1 OR no file stats yet:                     │
│       budget = "low",    trace_scope = "last1", refs=()  │
│   elif stagnation ≥ bandit_stagnation_threshold (=4):    │
│       budget = "high",   trace_scope = "all",            │
│       refs = _bandit_reference_iterations(budget="high") │
│              → all known historical iterations           │
│   else:                                                  │
│       budget = "medium", trace_scope = "last3",          │
│       refs = iters that hot files appeared in            │
│              ∪ best3 ∪ last_improved   (cap = 5)         │
└──────────────────────────────────────────────────────────┘
                       │
                       ▼
   prompt injects "Bandit Context Policy":
       trace_scope, hot_files, warm_files (advisory)
       best_iterations, worst_iteration
   workspace's access_policy.json carries hot/warm/read_budget
                       │
                       ▼
                proposer runs → pending_eval.json
                       │
                       ▼
                _evaluate_proposed → CandidateResult
                       │
                       ▼
            ┌──────────────────────────────────────┐
            │   ③ _update_bandit_state(...)         │
            └──────────────────────────────────────┘
```

### 3.2 Reward computation (key v3 change)

```
best_eval_passrate = max(c.passrate for c in evaluated)   # v3: passrate ONLY
history.append(best_eval_passrate)
recent = history[-bandit_reward_window:]                   # default 8
                                                            (the v3 runs in
                                                             EXPERIMENT_RESULTS.md
                                                             also test 16)

if not evaluated:
    reward = -clip * 0.25                                   # failure penalty
elif len(recent) < 2:                                       # warm-up
    raw = best_eval_passrate - previous_best_passrate
    reward = clip(raw * 10, -clip, +clip)
else:
    μ, σ = mean/std(recent),  σ ≥ sigma_floor (= 0.02)
    reward = clip((best_eval_passrate − μ) / σ, ±clip)      # z-score
success = reward > 0
```

What v3 changed vs v1/v2 (verified against
`src/memomemo/optimizer.py` and the experiment notes):

1. **passrate-only reward**. `average_score` is no longer mixed into the
   reward. The mixed-reward variant chased partial-quality wins that did
   not transfer from train to test (claudekimi mixed-reward run was worse
   on every metric).
2. **Sliding-window z-score**. Reward is the standardized deviation
   against the recent window rather than absolute improvement vs the
   global best, so a meaningful jump during a stagnation phase still
   earns positive credit.

### 3.3 File-level utility update

After each iteration’s reward is fixed:

```
For each path the proposer actually read this iter:
    files[path].read_iters   += 1
    files[path].read_calls   += this iter's read count
    files[path].read_lines   += this iter's read line total
    files[path].reward_sum   += reward (the iter-level reward)
    if success:
        files[path].success_iters += 1

For each path written or appearing in diff this iter:
    files[path].write_iters   += 1   (if written)
    files[path].changed_iters += 1   (if in diff)
    (these do NOT count toward reward denominator)
```

Then `_recompute_bandit_scores` re-scores everything globally:

```
Discretionary-read pool only (read_iters > 0 AND not in required core):
    p_global         = scored_success_iters / scored_read_iters
    mean_reward_glob = scored_reward_sum    / scored_read_iters

For each scored file path:
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

Hyperparameters (all live in `OptimizerConfig`):

| field | default |
|---|---:|
| `bandit_prior_alpha` | 2.0 |
| `bandit_prior_weight` | 0.4 |
| `bandit_exploration_c` | 0.15 |
| `bandit_cost_lambda` | 0.05 |
| `bandit_line_scale` | 500 |
| `bandit_reward_window` | 8 |
| `bandit_reward_sigma_floor` | 0.02 |
| `bandit_reward_clip` | 2.0 |
| `bandit_stagnation_threshold` | 4 |
| `bandit_failed_iter_penalty` | 0.5 |

### 3.4 Required core files (always hot, never scored)

`_bandit_core_files` (`src/memomemo/optimizer.py:1951`) keeps a fixed
set of files in the hot list every iteration regardless of statistics:

- The scaffold source for the active `source_family`
  (e.g. `memgpt_source.py`).
- `scaffolds/base.py`, `model.py`, `schemas.py`.
- The six summary files under `summaries/`:
  `evolution_summary.jsonl`, `best_candidates.json`,
  `candidate_score_table.json`, `retrieval_diagnostics_summary.json`,
  `diff_summary.jsonl`, `iteration_index.json`.
- `pending_eval.json`.

These "foundation" files are excluded from bandit scoring so the
algorithm never under-prioritizes essentials based on small-sample
noise.

---

## 4. Side-by-side Comparison

| Dimension | default | progressive | bandit v3 |
|---|---|---|---|
| Budget selection | always `high` | state machine (`low`→…→`medium`→`high`) | heuristic + stagnation threshold (`low/medium/high`) |
| Reference iters | all history | by budget: best k + worst | iters where hot files appeared, fallback best3 / last_improved |
| Trace-scope pruning | `all` | `last1 / last3 / all` | same three tiers, derived from budget |
| Prompt extras | Optimization Focus | + Progressive role hints | + Bandit Context Policy (hot/warm lists) |
| Feedback signal | none | did this iter enter a new quality frontier? | sliding-window z-score reward (passrate only) |
| State file | none | `progressive_state.json` | `bandit_state.json` (per-file stats) |
| Explore vs exploit | none (always max) | implicit (escalate on stagnation) | explicit (UCB bonus + Beta smoothing) |
| Cost profile | highest (`high` every iter) | medium (mostly `low`) | medium-high (more reads + policy meta) |
| Strength | reproduce baseline; sanity check | most stable on LOCOMO / LongMemEval for claudekimi & opus | only bandit family that beats progressive on LoCoMo test (codex54 → 0.3865) |
| Weakness | no convergence | once `high`, hard to drop back fast | needs cold-start; mixed-reward variants overfit train |

Empirical wins (from `EXPERIMENT_RESULTS.md`, train = 80, test = 1449):

- **LOCOMO test**: claude opus progressive 0.3982 (overall best);
  claudekimi progressive 0.3734; codex54 bandit v3 0.3865.
- **LongMemEval test400**: claudekimi progressive 0.5000 (best);
  codex54 default 0.4875 close behind.
- **SWE-bench mini (mimo v2.5, train30)**: progressive 0.5333
  vs source baseline 0.4000–0.4667.

---

## 5. File Map for Reviewers

| Concept | Location |
|---|---|
| Outer optimizer (3-policy fork) | `src/memomemo/optimizer.py:207-258` |
| Progressive state machine | `src/memomemo/optimizer.py:1568-1638` |
| Bandit policy assembly | `src/memomemo/optimizer.py:1661-1744` |
| Bandit reward + per-file stats | `src/memomemo/optimizer.py:1801-1873` |
| Bandit scoring formula | `src/memomemo/optimizer.py:1875-1935` |
| Required core files | `src/memomemo/optimizer.py:1951-1981` |
| Pareto frontier definition | `src/memomemo/pareto.py` |
| Proposer prompt template | `src/memomemo/proposer_prompt.py` |
| Optimization cells (memgpt + mini-swe-agent) | `src/memomemo/optimization_cells.py` |
| SWE-bench subclass | `src/memomemo/swebench_optimizer.py` |
| CLI `--selection-policy` flag | `src/memomemo/cli.py:255-263` |
| Experiment results | `EXPERIMENT_RESULTS.md` |

---

## 6. Cheat Sheet — Running Each Policy

The Docker sandbox rules from `AGENTS.md` apply (the kimi proposer must
use `docker-claude-kimi:latest`; claude proposer needs the host
`.claude` and `.claude.json` mounted). API keys live in `.env` and must
be sourced before launch.

```bash
# Source credentials first
set -a && source /data/home/yuhan/MemoMemo/.env && set +a

# 1) DEFAULT (fixed-high baseline) — no state, no docker requirement
python -m memomemo.cli optimize \
  --run-id default_demo \
  --iterations 30 --selection-policy default \
  --proposer-agent codex --proposer-sandbox none

# 2) PROGRESSIVE (claudekimi, docker)
python -m memomemo.cli optimize \
  --run-id progressive_demo --iterations 30 \
  --selection-policy progressive \
  --proposer-agent kimi --proposer-sandbox docker \
  --proposer-docker-image docker-claude-kimi:latest \
  --proposer-docker-user 1023:1023 --proposer-docker-home /tmp \
  --proposer-docker-env KIMI_API_KEY

# 3) BANDIT v3 (codex54, docker)
python -m memomemo.cli optimize \
  --run-id bandit_v3_demo --iterations 30 \
  --selection-policy bandit \
  --proposer-agent codex --proposer-sandbox docker \
  --proposer-docker-image docker-codex:latest \
  --bandit-reward-window 16   # v3 runs in the paper used 16
```

After a training run, evaluate the frontier on test using
`scripts/evaluate_candidate_json.py` with a spec built from
`runs/<train_run>/best_candidates.json` (recipe in `AGENTS.md`).
