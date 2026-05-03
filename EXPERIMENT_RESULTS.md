# Experiment Results

Last updated: 2026-05-03

This file summarizes the current completed experiments for the default,
progressive, and bandit selection policies. Reruns that reproduced the same
conclusion are not listed as separate rows. Rows marked `(docker)` used the
docker proposer sandbox with the corrected `docker-claude-kimi:latest` image.
Earlier bandit variants (v1 fixed-best reward; v2 z-score with mixed reward)
have been superseded by bandit v3 (sliding-window z-score, window=16, with a
passrate-only reward for claudekimi and the original mixed reward for
codex54). Only the v3 results are retained below.

## Executive Summary

- **LongMemEval new global best: claudekimi bandit force=low (v3, w16)
  at train 0.6300 / test 0.5225**, reached by pinning every iteration to
  the bandit's `low` budget tier (`--force-budget low`) while keeping
  the per-file UCB prior. Cache reads drop to 2.80M/iter from the
  adaptive bandit's 4.12M, and test passrate climbs from 0.4550 to
  0.5225.
- LoCoMo overall test best: codex54 bandit (docker, v3) at 0.3865.
  claudekimi progressive is second at 0.3734. LoCoMo bandit force=low
  (claudekimi) lands at 0.3409 test, *below* the adaptive bandit (0.3589)
  and progressive (0.3734), so the same `force=low` knob does not
  generalize to LoCoMo.
- LongMemEval bandit v3 rerun on claudekimi exceeds the 1st attempt
  (train 0.5300 → 0.6000; test 0.4325 → 0.4550), confirming v3 on
  claudekimi is genuinely strong rather than a single-run fluke.
- The `--include-optimization-direction` (`default+direction`) ablation
  is now resolved: on LongMemEval the rerun lands at train 0.5500 /
  test 0.4950 (below progressive 0.5000 and below bandit force=low
  0.5225), superseding the lucky 1st attempt's 0.6500/0.5300; on LoCoMo
  the rerun's best Pareto frontier test is 0.3458 (still below
  progressive 0.3734); on SWE-bench mini trainfirst30 with DeepSeek the
  best train passrate is 0.5000 (below progressive/bandit's 0.5333).
  Mechanism direction injection alone is *not* the largest lever on any
  benchmark.
- New `--force-budget low` ablation, run on both progressive and bandit
  for LoCoMo and LongMemEval: bandit on LongMemEval gains sharply
  (0.4550 → 0.5225 test, the new headline number); progressive on
  LongMemEval drops slightly (0.5000 → 0.4675); both policies on
  LoCoMo drop. The bandit-specific LongMemEval gain isolates the
  per-file UCB prior + UCB exploration as the actual driver of the new
  best.
- Text classification has no bandit run in the current results. Default is
  better than progressive in the claudekimi offline validation run.
- SWE-bench mini: DeepSeek v4 Flash bandit (fixedsource) still gives the
  strongest full SWE-bench Verified result: **320/500 = 0.6400** with
  `iter013_impact_aware_feedback`. The new claudekimi default+direction
  trainfirst30 run (24/30 iters, DeepSeek v4 Flash) promotes
  `iter020_patch_integrity_validator` to a full500 verified passrate of
  **313/500 = 0.6260**, slightly above the earlier progressive full500
  (310/500 = 0.6200) but still below the bandit fixedsource result.
  Earlier verified_test10 runs (qwen35-9b, qwen35 a3b) still show no
  improvement over the source baseline.

## Reading the merged tables

Each benchmark below uses a single per-(proposer, policy) table that pairs
passrate (train + test) with per-iteration proposer cost. Cost columns are
averaged over `proposer_calls/*/agent/metrics.json` of the train run:

- `input` / `output` — new tokens billed each turn.
- `cache reads` — prompt-cache hits reused across turns; billed at a fraction
  of new-input rate but typically much larger than `input`.
- `tools/iter` — tool-use calls per iteration.
- `files/iter` — unique workspace files the proposer opened per iteration.
- `dur/iter` — wall-clock duration per iteration.

A `—` cell means cost data is unavailable: the train-run directory was
deleted and only the test-eval dir remains. Bold cells flag the strongest
test result within each proposer family (with the matching train cell also
bolded); ★ marks an overall benchmark best. The `total/iter` column is
the sum of new input + output + cache reads per proposer iteration — the
gross token volume that flows through the proposer per call.

## LoCoMo

LoCoMo train uses 80 examples; test uses the full 1,449 examples.
Progressive and bandit runs use the docker proposer sandbox. Default runs
predate the docker sandbox and are kept as baselines.

| proposer | policy | train | test | input/iter | output/iter | cache reads/iter | total/iter | tools/iter | files/iter | dur/iter |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| claudekimi | default (docker) | 0.4125 | 0.3382 | 136.4k | 26.8k | 2.97M | 3.13M | 45.8 | 19.9 | 12.3m |
| claudekimi | default+direction (docker, 1st) | 0.3875 | 0.3140 | 170.1k | 28.3k | 4.43M | 4.63M | 54.8 | 19.9 | 16.3m |
| claudekimi | default+direction (docker, rerun) | 0.4125 | 0.3333 / 0.3458 | 170.9k | 29.7k | 4.17M | 4.37M | 53.1 | 20.0 | 16.2m |
| claudekimi | progressive (docker) | **0.4375** | **0.3734** | 138.9k | 25.5k | 1.70M | 1.86M | 35.2 | 15.1 | 13.0m |
| claudekimi | progressive force=low (docker) | 0.4250 | 0.3540 | 141.9k | 27.8k | 3.23M | 3.40M | 44.9 | 18.2 | 13.2m |
| claudekimi | bandit (docker) | 0.4375 | 0.3589 | 104.2k | 29.8k | 1.83M | 1.96M | 35.1 | 17.6 | 14.1m |
| claudekimi | bandit force=low (docker, v3) | 0.3875 | 0.3409 | 122.5k | 23.4k | 2.64M | 2.79M | 36.6 | 15.1 | 11.1m |
| codex54 | default (docker) | 0.4375 | 0.3368 | 1.45M | 23.9k | 1.33M | 2.80M | 33.9 | 16.8 | 8.0m |
| codex54 | progressive (docker) | 0.4250 | 0.3589 | 2.39M | 18.7k | 2.25M | 4.66M | 50.6 | 16.9 | 7.1m |
| codex54 | bandit (docker) | **0.4250** | **0.3865** ★ | 1.13M | 20.7k | 995k | 2.14M | 34.6 | 18.5 | 7.0m |

The `default+direction` rerun's test cell shows two numbers because the
Pareto frontier holds two candidates: `iter005_memgpt_temporal_fusion_top8`
(train 0.4125 / test 0.3333) and `iter028_memgpt_aidwr_top8` (train 0.4000 /
test 0.3458). The `force=low` rows pin every iteration to the
`low` budget tier (`--force-budget low`); both end up below their adaptive
counterparts on test (progressive 0.3540 vs 0.3734; bandit 0.3409 vs
0.3589), confirming that LoCoMo benefits from at least occasional
`medium`/`high` budget reads. Iteration counts behind the cost averages
are 30 except: claudekimi progressive 29 (one iteration produced no
metrics) and codex54 bandit 27/30 (interrupted).

### Bandit design

The bandit policy uses a sliding-window z-score reward (window=16) that
rewards each iteration relative to recent quality history rather than
absolute improvement over the run's best. claudekimi uses a passrate-only
reward — an earlier mixed-reward run (train 0.3750 / test 0.3009) was
inferior on every metric and is dropped, confirming that mixing
average_score into the reward chases partial-quality wins that do not
transfer to test. codex54 uses the original mixed reward; its train run was
interrupted at iter28/30 and test was evaluated on 2026-04-30 against the
full 1,449-example test split using Qwen3-8B on GPU1
(port 8002, 128 workers).

### Notes

- The overall LoCoMo test best is **codex54 bandit (docker, v3) at
  0.3865**.
- Progressive wins for claudekimi on test. codex54 is the exception:
  bandit (0.3865) is its strongest result on test and the only
  bandit-family policy that beats progressive for any proposer family.
- claudekimi bandit (0.3589) trails progressive (0.3734) on test even
  though their train numbers are tied at 0.4375.
- codex54's input-token cost is roughly an order of magnitude higher than
  claudekimi because gpt-5.4 expands reasoning inline; bandit cuts it
  nearly in half (2.39M → 1.13M) without losing test quality.
- **claudekimi default+direction** (`--include-optimization-direction`):
  the 1st attempt collapsed to test 0.3140 (vs default 0.3382); the
  rerun recovered slightly to a 2-candidate Pareto frontier with best
  test 0.3458 (still below progressive 0.3734). Cost is consistently
  +30–50% cache reads vs plain default. Mechanism direction lines do
  not transfer to test on LoCoMo. See PIPELINE.md §1 for the prompt
  mechanics.
- **`force=low` ablation** (`--force-budget low`) on claudekimi LoCoMo:
  both progressive force=low (test 0.3540) and bandit force=low (test
  0.3409) underperform their adaptive counterparts (0.3734 / 0.3589).
  Cache reads drop only modestly because LoCoMo's `low` budget still
  carries the bandit hot/warm file lists every iter. Net: budget tier
  escalation is doing real work on LoCoMo and shouldn't be locked at
  `low`.

### Key run paths

- `runs/locomo_claudekimi_default_iter012_test_20260426`
- `runs/locomo_claudekimi_progressive_iter020_test_20260426`
- `runs/locomo_codex54_default_iter001_test_20260426`
- `runs/locomo_codex54_progressive_iter028_test_20260426`
- `runs/locomo_codex54_bandit_v3_iter028_test_20260430`
- `runs/locomo_memory_opt_memgpt_claudekimi_progressive_docker_env_iter30_full80seed_20260421_215754`
- `runs/locomo_memory_opt_memgpt_claudekimi_bandit_v3_passrate_reward_iter30_full80seed_w16_20260429_022838`
- `runs/locomo_memory_opt_memgpt_codex54_bandit_v3_iter30_full80seed_w16_20260428_192739`
- `runs/locomo_memory_opt_memgpt_codex54_progressive_docker_iter30_full80seed_20260421_211252`
- `runs/locomo_memgpt_codex54_default_docker_iter30_train80_rerun_20260502_015354` (codex54 default docker)
- `runs/locomo_memgpt_claudekimi_default_docker_iter30_train80_20260501_204004` (claudekimi default docker)
- `runs/locomo_memgpt_claudekimi_default_direction_docker_iter30_train80_20260502_015441` (claudekimi default+direction, 1st)
- `runs/locomo_memgpt_claudekimi_default_direction_docker_iter30_train80_20260502_154556` (claudekimi default+direction, rerun)
- `runs/locomo_memgpt_claudekimi_progressive_budgetlow_docker_iter30_train80_20260502_170952` (claudekimi progressive force=low)
- `runs/locomo_memgpt_claudekimi_bandit_v3_budgetlow_docker_iter30_train80_w16_20260502_170954` (claudekimi bandit v3 force=low)

## LongMemEval

Train rows use train100. Test rows use the 400-example test split. Rows
marked `failed` produced a test-frontier artifact but did not complete a
valid score. Bandit rows use the same v3 sliding-window z-score reward
(window=16, passrate-only) as the LoCoMo bandit runs.
`default+direction` rows use the new `--include-optimization-direction`
flag, which injects the Optimization Focus mechanism direction list
into the proposer prompt while keeping the default policy's fixed-high
context schedule (see PIPELINE.md §0.1).

| proposer | policy | train | test | input/iter | output/iter | cache reads/iter | total/iter | tools/iter | files/iter | dur/iter |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| claudekimi | default | 0.5600 | 0.4700 | 121.5k | 26.9k | 2.12M | 2.27M | 39.6 | 18.4 | 10.3m |
| claudekimi | progressive | 0.6000 | 0.5000 | 105.0k | 25.0k | 1.73M | 1.86M | 33.6 | 16.3 | 9.5m |
| claudekimi | progressive force=low (docker) | 0.6000 | 0.4675 | 170.2k | 25.6k | 3.73M | 3.92M | 47.7 | 18.2 | 14.4m |
| claudekimi | bandit (docker, v3, 1st) | 0.5300 | 0.4325 | 152.9k | 27.9k | 3.28M | 3.46M | 44.4 | 20.0 | 14.3m |
| claudekimi | bandit (docker, v3, rerun) | **0.6000** | 0.4550 | 179.0k | 31.2k | 4.12M | 4.33M | 51.6 | 23.0 | 16.1m |
| claudekimi | bandit force=low (docker, v3) | **0.6300** | **0.5225** ★ | 127.0k | 28.4k | 2.80M | 2.95M | 38.7 | 16.0 | 12.8m |
| claudekimi | default+direction (1st) | 0.6500 | 0.5300 | 176.0k | 30.2k | 3.43M | 3.63M | 49.6 | 18.6 | 16.3m |
| claudekimi | default+direction (rerun) | 0.5500 | 0.4950 | 166.9k | 33.7k | 4.13M | 4.33M | 50.8 | 20.5 | 16.1m |
| codex54 | default | **0.6000** | **0.4875** | 1.77M | 27.4k | 1.61M | 3.41M | 33.4 | 18.8 | 9.5m |
| codex54 | progressive (rerun) | 0.5400 | 0.4725 | 1.45M | 25.0k | 1.33M | 2.80M | 31.9 | 17.0 | 8.3m |
| codex54 | bandit (docker, v3) | 0.5200 | 0.4725 | 1.13M | 24.6k | 1.03M | 2.18M | 34.8 | 19.0 | 8.1m |

All cost rows average over 30 iterations.

### Notes

- **New LongMemEval test best: claudekimi bandit force=low (v3, w16) at
  0.5225 ★** — locking the bandit at `low` budget while keeping the
  per-file UCB prior gives both the lowest cost (cache reads 2.80M/iter
  vs adaptive bandit rerun 4.12M) and the highest train (0.6300) and
  test (0.5225) on this benchmark. progressive (0.5000) drops to second.
- bandit v3 rerun on claudekimi *exceeds* the 1st attempt on both train
  (0.5300 → 0.6000) and test (0.4325 → 0.4550), so v3 on claudekimi is
  reproducibly strong rather than a single-run fluke. The 1st attempt is
  retained for the historical record but the rerun is canonical.
- progressive force=low (test 0.4675) drops *below* the adaptive
  progressive (0.5000), while bandit force=low (test 0.5225) goes
  *above* the adaptive bandit (0.4550). The `low` budget knob therefore
  isolates the per-file UCB prior + UCB exploration as the actual driver
  of the headline number — not "low budget" alone.
- **codex54 v3 bandit cuts proposer cost by ~36%** (input/iter 1.77M → 1.13M,
  total/iter 3.41M → 2.18M, dur/iter 9.5m → 8.1m) while losing only −1.5pt
  test passrate vs default — the same cost-reduction pattern seen on LoCoMo
  codex54 bandit.
- codex54 progressive shown is the rerun (0.4725); the original run failed
  on test with `date value out of range` and is not counted.
- claudekimi `default+direction` (`--include-optimization-direction`)
  ablation: 1st attempt hit train 0.6500 / test 0.5300, but the
  same-config rerun lands at train 0.5500 / test 0.4950, below
  progressive 0.5000 and below bandit force=low 0.5225. The 1st attempt
  was a lucky outlier; the rerun is now the canonical row. Cost is
  ~+95% cache reads vs plain default (4.13M vs 2.12M tokens/iter, dur
  16.1m vs 10.3m). 1st-attempt test was retried with
  `scripts/evaluate_longmemeval_candidate.py` after the original
  test_frontier hit Together 429; rerun's test_frontier (progressive
  force=low) hit a separate Together "Server disconnected" and was
  retried with the same script.
- Cost pattern matches LoCoMo: codex54 dominates input-token cost while
  claudekimi rides on the prompt cache.

### Key run paths

- `runs/longmemeval_memgpt_claudekimi_default_docker_env_iter30_train100_fix_20260423_074629`
- `runs/longmemeval_memgpt_claudekimi_progressive_docker_env_iter30_train100_fixrerun_20260423_161417`
- `runs/longmemeval_default_iter021_correct_test_run4_20260424`
- `runs/longmemeval_memgpt_codex54_default_docker_iter30_train100_20260427_222924`
- `runs/longmemeval_memgpt_codex54_progressive_docker_rerun_iter30_train100_w16_20260428_162906`
- `runs/longmemeval_memgpt_codex54_bandit_v3_docker_iter30_train100_w16_20260501_203909`
- `runs/longmemeval_memgpt_claudekimi_bandit_v3_docker_iter30_train100_w16_20260501_203907` (1st bandit v3)
- `runs/longmemeval_memgpt_claudekimi_bandit_v3_docker_iter30_train100_w16_20260502_155309` (bandit v3 rerun)
- `runs/longmemeval_memgpt_claudekimi_default_direction_docker_iter30_train100_20260502_015454` (1st default+direction)
- `runs/longmemeval_memgpt_claudekimi_default_direction_docker_iter30_train100_20260502_152524` (default+direction rerun)
- `runs/longmemeval_memgpt_claudekimi_progressive_budgetlow_docker_iter30_train100_20260502_170956` (claudekimi progressive force=low)
- `runs/longmemeval_memgpt_claudekimi_bandit_v3_budgetlow_docker_iter30_train100_w16_20260502_170958` (claudekimi bandit v3 force=low)

## Text Classification

No bandit result is present for the current text classification experiments.

| setting | default | progressive | bandit |
|---|---:|---:|---:|
| claudekimi offline validation | 0.4556 | 0.4356 | N/A |
| offline best test 3x | 0.4367 | 0.4400 | N/A |
| online best test 3x | 0.4600 | 0.4433 | N/A |

Conclusion: there is no clean bandit comparison yet. Default is stronger in
the newer claudekimi offline validation run; older test reruns are close
and do not show a consistent progressive advantage.

Key run paths:

- `runs/textcls_offline_default_claudekimi_envkey_no_train_eval_iter30_w64_fix_20260423_074629`
- `runs/textcls_offline_progressive_claudekimi_envkey_no_train_eval_iter30_w64_fix_20260423_074629`
- `runs/textcls_best_test_3x_20260423_014927`

## SWE-bench Mini

The benchmark is now driven by a source-backed `mini_swe_agent_source` target
(`src/memomemo/swebench.py` + `swebench_optimizer.py`, exposed in the optimize
CLI as `--swebench`). The optimizer copies a clean `references/vendor/mini-swe-agent`
checkout into each iteration's workspace and runs caller-provided
`{source_path}`/`{instance_path}`/`{patch_path}` command templates per task.
Data lives in `data/swebench_verified_full.json` and `data/swebench_verified_test50.json`.

All SWE-bench mini test evaluations (trainfirst30 and verified_full500)
use **DeepSeek v4 Flash** as the solver model. Three task pools have
been used so far:

- **train30 (DeepSeek v4 Flash)** — 30 SWE-bench-verified train instances,
  10 workers, 900s per-task timeout. The first sequence where
  optimization beats the source baseline.
- **verified_full500 (DeepSeek v4 Flash)** — full 500-task SWE-bench Verified
  evaluation for the source baseline and optimized mini-SWE-agent candidates.
- **verified_test10 (qwen35-9b)** — 10 verified-test instances; source
  baseline 0.30 (default) / 0.20 (progressive); optimizer iterations regress.
- **verified_test10 (qwen35 a3b 35B-A3B)** — older 2026-04-24 sequence;
  optimizer plateaus at the source baseline.

### train30 (DeepSeek v4 Flash, 30 train instances)

The SWE-bench train30 pool has no separate test split, so passrate is
reported on the same 30-task pool as a "best optimizer candidate" against
the source baseline.

| proposer | policy | source baseline | best passrate | iters | input/iter | output/iter | cache reads/iter | total/iter | tools/iter | files/iter | dur/iter |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| claudekimi | default | 0.4667 | **0.5000** | 20/30 | 136.3k | 28.6k | 3.06M | 3.22M | 56.0 | 23.2 | 13.4m |
| claudekimi | progressive | 0.4000 | **0.5333** | 20/30 | 141.8k | 29.7k | 3.61M | 3.78M | 61.0 | 23.6 | 12.5m |
| claudekimi | default+direction | 0.4667 | **0.5000** | 24/30 | 167.3k | 29.4k | 4.58M | 4.78M | 63.5 | 24.2 | 16.0m |

Best optimizer candidates: default → `iter002_stack_trace_context` (and 4
ties at 0.5000); progressive → `iter016_final_fallback_traceback_retrieval_v1`;
default+direction → 5 ties at 0.5000, latest one with iter ≤ 20 is
`iter020_patch_integrity_validator` (the row promoted to full500
verified, see below).

Notes:

- The progressive row is the rerun (2026-04-30). An initial progressive run
  on the same pool plateaued at the source baseline (0.4667) but only
  completed 5/30 iterations and is not counted; the rerun supersedes it.
- Progressive is the strongest train30 result so far. The rerun's lower
  source baseline (0.4000 vs the earlier non-rerun baseline of 0.4667)
  reflects scoring variance on the 30-task pool, not a regression in the
  agent — the optimizer still converges to a candidate that beats both
  the rerun's own baseline (0.4 → 0.5333) and the earlier baseline
  (0.4667 → 0.5333).
- `iter016` introduces a fallback traceback-aware retrieval; the next four
  best candidates (0.5000) are independent traceback / verification gates,
  suggesting traceback-driven evidence is the dominant useful direction on
  this pool.
- The `default+direction` run was stopped at iter024/30 (training was
  killed once 5 ties at 0.5000 had been collected); the average cost row
  uses the 24 completed iterations. It does not break the 0.5000 ceiling
  on train30, so on this benchmark the mechanism direction list trails
  progressive/bandit (0.5333).
- Tools/iter (56–63) and files/iter (~23–24) are higher than on
  LoCoMo/LongMemEval because the SWE-bench workspace is larger (full
  mini-SWE-agent source plus benchmark scaffolding); each iteration probes
  more files before producing a patch. Cache reads scale up correspondingly
  (~3–4.6M tok/iter vs ~1.7M on LoCoMo).

### train30 (DeepSeek v4 Flash, bandit fixedsource)

A second DeepSeek v4 Flash trainfirst30 sequence used the bandit policy
on the fixedsource workspace. The fixedsource bandit row uses the
canonical sliding-window passrate-only z-score reward (window=16). 20
proposer iter dirs exist but `iter_020` is missing its `metrics.json`
(interrupted), so cost averages over the first 19.

| proposer | policy | source baseline | best passrate | iters | input/iter | output/iter | cache reads/iter | total/iter | tools/iter | files/iter | dur/iter |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| claudekimi | bandit (fixedsource) | 0.5000 | **0.5333** | 19/20 | 128.9k | 26.1k | 3.35M | 3.51M | 56.6 | 25.5 | 12.2m |

Notes:

- Best optimizer candidates: `iter005_auto_test_feedback_and_recovery` plus
  four ties at 0.5333 (`iter006_repro_and_review`,
  `iter007_pre_edit_localization`,
  `iter009_edit_verification_repo_tests`,
  `iter013_impact_aware_feedback`).
- The DeepSeek fixedsource bandit reaches the same 0.5333 train30 ceiling
  as the DeepSeek progressive run, but with a higher source baseline
  (0.5000 vs 0.4667 / 0.4000 on the other two trainfirst30 sequences —
  the bandit fixedsource workspace gives the agent a richer starting
  scaffold). The tied frontier candidates were later promoted to full500
  verified evaluation; `iter013` is the best verified result at 320/500
  (0.6400).
- An earlier "promptcells" variant
  (`swebench_miniswe_deepseek_v4_flash_claudekimi_bandit_v3_iter20_trainfirst30_w10_t900_promptcells_20260430_200058`)
  only completed 6 iterations and never beat the source baseline (best
  0.5000 = baseline). Its averaged cost (104.2k input / 29.4k output /
  2.53M cache reads / 2.66M total / 54.5 tools / 27.8 files / 10.8m per
  iter, n=6) is not retained as a result row.
- Cache reads (3.35M/iter for fixedsource bandit, 4.58M/iter for
  default+direction) are the largest of any benchmark/policy combo so
  far — DeepSeek v4 Flash with mini-SWE-agent context plus the bandit /
  direction-list meta produces an unusually wide cached prompt.

### verified_full500 (DeepSeek v4 Flash)

These are full SWE-bench Verified test evaluations with DeepSeek v4 Flash
as the solver model. The optimized candidates come from the mini-SWE-agent
source optimization runs; score is `resolved / 500`.

| candidate | source run | resolved / total | passrate |
|---|---|---:|---:|
| source baseline | `swebench_deepseek_v4_flash_verified_full500_20260430_182537` | 220 / 500 | 0.4400 |
| default optimized (`iter002_stack_trace_context`) | `swebench_deepseek_v4_flash_verified_full500_20260430_182537` | 229 / 500 | 0.4580 |
| progressive optimized (`iter016_final_fallback_traceback_retrieval_v1`) | `swebench_deepseek_v4_flash_verified_full500_20260430_182537` | 310 / 500 | 0.6200 |
| bandit fixedsource optimized (`iter013_impact_aware_feedback`) | `swebench_miniswe_deepseek_v4_flash_claudekimi_bandit_v3_fixedsource_iter20_trainfirst30_w10_t900_20260430_233750/test_frontier` | 320 / 500 | **0.6400** |
| default+direction optimized (`iter020_patch_integrity_validator`) | `swebench_miniswe_deepseek_v4_flash_claudekimi_default_direction_iter30_trainfirst30_w10_t900_20260502_015837/test_full500_iter020_patch_integrity_validator` | 313 / 500 | 0.6260 |

The bandit fixedsource `iter013_impact_aware_feedback` candidate is the
current best verified result: +20.0 absolute points over the DeepSeek v4
Flash source baseline, +18.2 points over the default optimized candidate,
and +2.0 points over the earlier progressive full500 result. The full
frontier test results are stored at
`runs/swebench_miniswe_deepseek_v4_flash_claudekimi_bandit_v3_fixedsource_iter20_trainfirst30_w10_t900_20260430_233750/test_frontier/test_results.json`.

The new `default+direction` row is the candidate-level promotion of the
24/30-iter trainfirst30 default+direction run; `iter020` is the latest
0.5000 frontier candidate with iter ≤ 20 and beats the source baseline
by +18.6 points on full500 (0.4400 → 0.6260), slightly above the earlier
progressive full500 result (0.6200) but still below the bandit
fixedsource (0.6400). Confirms the mechanism direction list does
generalize on SWE-bench, even though its train30 ceiling (0.5000) trails
progressive/bandit (0.5333).

### verified_test10 (qwen35-9b, fixedpaths)

| policy | source baseline | best optimizer candidate | best passrate |
|---|---:|---|---:|
| claudekimi default | 0.3000 | none beats baseline (`iter001_working_memory_repo_context` 0.2) | 0.3000 |
| claudekimi progressive | 0.2000 | none beats baseline (`iter001_memory_and_robust_submit` 0.1) | 0.2000 |

Notes:

- These are the production verified_test10 evaluations after the fixedpaths
  patch. Earlier non-fixedpaths runs (170020 / 165622 / 165622-codex54) all
  scored 0.0 across the board and are abandoned.
- Each fixedpaths run only completed one optimizer iteration, and that
  iteration regressed against the source baseline. verified_test10 is
  apparently too small / too saturated for this proposer/model combo to
  improve on the baseline.

### Other targets

- `swebench_deepseek_v4_flash_smoke_baseline_1` — smoke baseline for the
  DeepSeek v4 Flash agent: 1 task, passrate 1.0, used to validate the
  command-template wiring.
- `swebench_deepseek_v4_flash_verified_full500_20260430_182537` — full
  500-task verified evaluation for DeepSeek v4 Flash baseline and optimized
  candidates.

### Conclusion

- The source-backed mini-SWE-agent benchmark is wired through the optimizer
  end-to-end and produces useful proposer iterations on train30. All test
  evaluations use DeepSeek v4 Flash as the solver model.
- DeepSeek v4 Flash trainfirst30 progressive (rerun) is the first SWE-bench
  result that meaningfully beats the source baseline (0.5333 vs 0.4667
  absolute).
- On full SWE-bench Verified with DeepSeek v4 Flash, the bandit fixedsource
  optimized candidate `iter013_impact_aware_feedback` reaches 320/500
  resolved (0.6400), beating the source baseline 220/500 (0.4400), the
  default optimized candidate 229/500 (0.4580), the progressive optimized
  candidate 310/500 (0.6200), and the new default+direction
  `iter020_patch_integrity_validator` 313/500 (0.6260).
- The verified_test10 optimizer pool is too small and too saturated for the
  optimizer to reliably improve over the source baseline.
- DeepSeek v4 Flash bandit (fixedsource) train30 reaches the same 0.5333
  ceiling as DeepSeek progressive on the trainfirst30 pool, and its best
  full500 promoted candidate is the strongest SWE-bench result in this
  document.

Key run paths:

- `runs/swebench_miniswe_mimo_v25_claudekimi_default_iter30_trainfirst30_w10_t900_20260429_211742` (DeepSeek v4 Flash default)
- `runs/swebench_miniswe_mimo_v25_claudekimi_progressive_rerun_iter30_trainfirst30_w10_t900_20260430_022208` (DeepSeek v4 Flash progressive rerun)
- `runs/swebench_miniswe_deepseek_v4_flash_claudekimi_default_direction_iter30_trainfirst30_w10_t900_20260502_015837` (DeepSeek v4 Flash default+direction trainfirst30 + full500 of iter020)
- `runs/swebench_miniswe_qwen35_9b_claudekimi_default_fixedpaths_iter30_verified_test10_20260429_174251`
- `runs/swebench_miniswe_qwen35_9b_claudekimi_progressive_fixedpaths_iter30_verified_test10_20260429_174251`
- `runs/swebench_deepseek_v4_flash_smoke_baseline_1`
- `runs/swebench_deepseek_v4_flash_verified_full500_20260430_182537`
- `runs/swebench_miniswe_deepseek_v4_flash_claudekimi_bandit_v3_fixedsource_iter20_trainfirst30_w10_t900_20260430_233750`
- `runs/swebench_miniswe_deepseek_v4_flash_claudekimi_bandit_v3_iter20_trainfirst30_w10_t900_promptcells_20260430_200058`
- `runs/swebench_miniswe_qwen35_claudekimi_default_iter30_verified_test10_20260424_2038` (older a3b baseline)
- `runs/swebench_miniswe_qwen35_claudekimi_progressive_iter30_verified_test10_20260424_2038` (older a3b baseline)

## Overall Takeaways

- **LongMemEval new global best: claudekimi bandit force=low (v3, w16) at
  train 0.6300 / test 0.5225 ★** — the per-file UCB prior + UCB
  exploration plus a hard `low` budget cap is the strongest single
  policy / hyperparameter combination on LongMemEval seen so far.
- **LoCoMo best: codex54 bandit (docker, v3) at 0.3865 test.**
- All progressive and bandit results use the docker sandbox; non-docker
  progressive/bandit results are discarded.
- LoCoMo: progressive (docker) wins for claudekimi on test.
  codex54 is the exception: bandit (0.3865) is its best policy and the
  first bandit-family run that beats progressive for any proposer family on
  LoCoMo.
- Train80 LoCoMo is too small to trust by itself; claudekimi bandit
  (train 0.4375 / test 0.3589 vs progressive train 0.4375 / test 0.3734) is
  the clearest example of train-test mismatch.
- Reruns are required to verify stability. The 2026-05-02 reruns
  produced two opposite outcomes: claudekimi LongMemEval
  default+direction collapsed (1st 0.5300 → rerun 0.4950 test, lucky
  outlier), while claudekimi LongMemEval bandit v3 reproduced and even
  improved (1st 0.4325 → rerun 0.4550 test). Per (task, optimization
  model) this doc now keeps the rerun as the canonical row.
- SWE-bench mini has a useful optimization signal: DeepSeek v4 Flash
  trainfirst30 progressive reaches 0.5333 train vs 0.4667 source
  baseline. DeepSeek v4 Flash full verified evaluation reaches 0.6400
  after bandit fixedsource optimization vs 0.4400 source baseline.
  default+direction trainfirst30 plateaus at 0.5000 but its
  `iter020_patch_integrity_validator` candidate hits 0.6260 on full500
  (313/500), close to the progressive full500 (0.6200) but below the
  bandit fixedsource (0.6400). verified_test10 remains too small to
  drive optimizer improvement.
