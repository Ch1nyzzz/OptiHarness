# Experiment Insights

This note captures cross-run observations that are not visible from the
headline result tables alone. The scope is the retained experiments in
`docs/PIPELINE.md`, with the following exclusions for budget-stage analysis:

- default and default+direction are excluded because every iteration is fixed
  `high`;
- claude opus is excluded because there is no retained v3 bandit comparison;
- text classification is excluded from budget-stage comparisons because no
  bandit result is retained.

Unless otherwise stated, a breakthrough means a training iteration whose
candidate score improved over all earlier candidates in that same run. Score is
ordered by `passrate`, then `average_score`; equal-score ties are not counted
as new breakthroughs.

## Final Frontier vs Process Breakthroughs

Do not infer where optimization happens from `best_candidates.json` alone.
That file records the candidates left on the final frontier, so it can collapse
many earlier improvements into one or two retained candidates.

For example, LoCoMo claudekimi progressive has only one final retained best
candidate, but it had five score-improvement events during training:

| iteration | budget | passrate |
|---:|---|---:|
| 1 | low | 0.3750 |
| 2 | low | 0.3875 |
| 5 | low | 0.4125 |
| 13 | high | 0.4250 |
| 20 | high | 0.4375 |

LoCoMo claudekimi bandit likewise has one final retained best candidate, but
six process breakthroughs:

| iteration | budget | passrate |
|---:|---|---:|
| 1 | low | 0.3000 |
| 2 | medium | 0.3000 |
| 3 | medium | 0.3750 |
| 9 | high | 0.3875 |
| 12 | medium | 0.4125 |
| 22 | high | 0.4375 |

The `iter002` row is counted because `average_score` improved even though the
displayed passrate remained 0.3000.

## Breakthroughs By Budget

Across all retained adaptive runs in `PIPELINE.md`, excluding default and opus,
the process-breakthrough distribution is:

| benchmark | run | low | medium | high | total |
|---|---|---:|---:|---:|---:|
| LoCoMo | claudekimi progressive | 3 | 0 | 2 | 5 |
| LoCoMo | claudekimi bandit | 1 | 3 | 2 | 6 |
| LoCoMo | codex54 progressive | 3 | 0 | 1 | 4 |
| LoCoMo | codex54 bandit | 1 | 2 | 1 | 4 |
| LongMemEval | claudekimi progressive | 2 | 2 | 2 | 6 |
| LongMemEval | claudekimi bandit | 1 | 5 | 1 | 7 |
| LongMemEval | codex54 progressive | 3 | 1 | 2 | 6 |
| LongMemEval | codex54 bandit | 1 | 3 | 1 | 5 |
| SWE-bench mini | mimo progressive | 1 | 0 | 1 | 2 |
| SWE-bench mini | DeepSeek bandit | 1 | 2 | 0 | 3 |
| **total** |  | **17** | **18** | **13** | **48** |

On the full training horizon, low and medium account for most score
improvements. This is the optimistic view of the adaptive policies: narrow
contexts are often enough to find early gains, and medium budget is especially
productive for bandit.

## After Iteration 5

The full-horizon count is biased by the first few iterations. Early candidates
start from a low baseline, so they have more headroom and often improve without
needing much context. A stricter view is to count only breakthroughs after
iteration 5.

Raw breakthrough counts after iteration 5:

| benchmark | run | low | medium | high | total |
|---|---|---:|---:|---:|---:|
| LoCoMo | claudekimi progressive | 0 | 0 | 2 | 2 |
| LoCoMo | claudekimi bandit | 0 | 1 | 2 | 3 |
| LoCoMo | codex54 progressive | 1 | 0 | 1 | 2 |
| LoCoMo | codex54 bandit | 0 | 0 | 1 | 1 |
| LongMemEval | claudekimi progressive | 0 | 2 | 2 | 4 |
| LongMemEval | claudekimi bandit | 0 | 3 | 1 | 4 |
| LongMemEval | codex54 progressive | 1 | 1 | 2 | 4 |
| LongMemEval | codex54 bandit | 0 | 0 | 1 | 1 |
| SWE-bench mini | mimo progressive | 0 | 0 | 1 | 1 |
| SWE-bench mini | DeepSeek bandit | 0 | 0 | 0 | 0 |
| **total** |  | **2** | **7** | **13** | **22** |

However, raw counts are still not enough, because after warm-up the policies
spend many more iterations at `high`. The denominator matters:

| budget | post-5 iterations | post-5 breakthroughs | breakthrough rate |
|---|---:|---:|---:|
| low | 18 | 2 | 11.1% |
| medium | 66 | 7 | 10.6% |
| high | 142 | 13 | 9.2% |
| **total** | **226** | **22** | **9.7%** |

So the post-warm-up distribution changes materially in two different ways:
`high` produces the most absolute breakthroughs, but it also gets by far the
most opportunities. Normalized by how often each budget is used, `high` is not
more efficient per iteration than low or medium in this sample.

Per-run post-5 exposure and breakthrough rates:

| benchmark | run | low | medium | high |
|---|---|---:|---:|---:|
| LoCoMo | claudekimi progressive | 0/3 | 0/3 | 2/18 |
| LoCoMo | claudekimi bandit | 0/0 | 1/13 | 2/12 |
| LoCoMo | codex54 progressive | 1/2 | 0/2 | 1/21 |
| LoCoMo | codex54 bandit | 0/0 | 0/8 | 1/14 |
| LongMemEval | claudekimi progressive | 0/4 | 2/5 | 2/16 |
| LongMemEval | claudekimi bandit | 0/0 | 3/13 | 1/12 |
| LongMemEval | codex54 progressive | 1/4 | 1/4 | 2/17 |
| LongMemEval | codex54 bandit | 0/0 | 0/8 | 1/17 |
| SWE-bench mini | mimo progressive | 0/5 | 0/6 | 1/4 |
| SWE-bench mini | DeepSeek bandit | 0/0 | 0/4 | 0/11 |

The better interpretation is:

- low and medium are useful for early exploration and cheap candidate
  discovery;
- medium remains useful after warm-up, especially for claudekimi bandit on
  LongMemEval;
- high is where most later-stage gains happen in absolute count, mostly because
  the state machines spend far more post-warm-up iterations at high;
- per opportunity, post-warm-up high is not clearly more productive than low or
  medium in the retained runs;
- SWE-bench DeepSeek bandit is an important exception: its best train30
  improvement happens by iteration 5, so the post-5 table shows no additional
  breakthrough even though the medium-budget phase produced the key 0.5333
  train result.

## Token Trend Context

The budget-stage breakthrough counts should be read alongside proposer-token
curves:

- bandit usually ramps up: `low` is cheap, `medium` is moderate, and `high`
  approaches default-like context cost;
- progressive does not always save tokens. It saves clearly for claudekimi
  memory runs, but can be flat or more expensive for codex54 and SWE-bench
  mini because the proposer may compensate for smaller granted context by
  reading fewer but larger or more information-dense files;
- default should not be mixed into budget-stage breakthrough counts, because
  default has no budget transition. Its breakthroughs are all `default-high`
  by construction.
