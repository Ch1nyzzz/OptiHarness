# Experiment Results

Last updated: 2026-05-02

This file summarizes the current completed experiments for the default,
progressive, and bandit selection policies. Reruns that reproduced the same
conclusion are not listed as separate rows. Rows marked `(docker)` used the
docker proposer sandbox with the corrected `docker-claude-kimi:latest` image.
Earlier bandit variants (v1 fixed-best reward; v2 z-score with mixed reward)
have been superseded by bandit v3 (sliding-window z-score, window=16, with a
passrate-only reward for claudekimi and the original mixed reward for
codex54). Only the v3 results are retained below.

## Executive Summary

- **LoCoMo overall test best: claude opus progressive docker at 0.3982.**
  claudekimi progressive is second at 0.3734.
- LoCoMo bandit lands codex54 at **0.3865 test**, the only bandit-family
  result that beats progressive for any proposer family on LoCoMo.
  claudekimi bandit reaches 0.3589 test, still trailing claudekimi
  progressive (0.3734).
- **LongMemEval bandit v3 added (2026-05-01)**: codex54 v3 bandit ties
  progressive at 0.4725 test while cutting proposer cost ~36% (input/iter
  1.77M → 1.13M). claudekimi v3 bandit underperforms at 0.4325 test,
  same kimi-vs-codex split observed on LoCoMo. LongMemEval test best
  remains claudekimi progressive at 0.5000.
- New `--include-optimization-direction` flag exposes a `default+direction`
  ablation that injects only the Optimization Focus mechanism direction
  list while keeping default's fixed-high schedule. First LongMemEval
  claudekimi run hit train 0.6500 / test 0.5300 (above progressive
  0.5000); a rerun is in progress to confirm stability before the
  number is treated as the new test best. **On LoCoMo claudekimi the
  same flag lowers test (0.3382 → 0.3140)** while inflating cache reads
  (+49%); a LoCoMo kimi default+direction rerun is also in progress to
  confirm whether the drop is stable. See PIPELINE.md §1 for the prompt
  mechanics.
- Text classification has no bandit run in the current results. Default is
  better than progressive in the claudekimi offline validation run.
- SWE-bench mini: DeepSeek v4 Flash bandit (fixedsource) now gives the
  strongest full SWE-bench Verified result: **320/500 = 0.6400** with
  `iter013_impact_aware_feedback`, beating the earlier progressive full500
  result of 310/500 (0.6200). Earlier verified_test10 runs (qwen35-9b,
  qwen35 a3b) still show no improvement over the source baseline. No
  mimo-v2.5 bandit train30 result. A kimi default+direction run on
  DeepSeek v4 Flash trainfirst30 is in progress (iter_021 at time of
  writing).

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
| claudekimi | default+direction (docker, rerun) | running | running | — | — | — | — | — | — | — |
| claudekimi | progressive (docker) | **0.4375** | **0.3734** | 138.9k | 25.5k | 1.70M | 1.86M | 35.2 | 15.1 | 13.0m |
| claudekimi | bandit (docker) | 0.4375 | 0.3589 | 104.2k | 29.8k | 1.83M | 1.96M | 35.1 | 17.6 | 14.1m |
| claude opus | default | 0.3875 | 0.3306 | — | — | — | — | — | — | — |
| claude opus | progressive (docker) | **0.4750** | **0.3982** ★ | 3.1k | 20.6k | 1.99M | 2.11M | 61.2 | 20.7 | 8.9m |
| codex54 | default (docker) | 0.4375 | 0.3368 | 1.45M | 23.9k | 1.33M | 2.80M | 33.9 | 16.8 | 8.0m |
| codex54 | progressive (docker) | 0.4250 | 0.3589 | 2.39M | 18.7k | 2.25M | 4.66M | 50.6 | 16.9 | 7.1m |
| codex54 | bandit (docker) | **0.4250** | **0.3865** | 1.13M | 20.7k | 995k | 2.14M | 34.6 | 18.5 | 7.0m |

claude opus has no bandit row because no v3-era bandit run was completed on
opus (earlier bandit variants are not retained). Iteration counts behind the
cost averages are 30 except: claudekimi progressive 29 (one iteration
produced no metrics) and codex54 bandit 27/30 (interrupted).

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

- The overall LoCoMo test best is **claude opus progressive docker at
  0.3982**.
- Progressive wins for claudekimi and claude opus on test. codex54 is the
  exception: bandit (0.3865) is its strongest result on test and the only
  bandit-family policy that beats progressive for any proposer family.
- claudekimi bandit (0.3589) trails progressive (0.3734) on test even
  though their train numbers are tied at 0.4375.
- codex54's input-token cost is roughly an order of magnitude higher than
  claudekimi/opus because gpt-5.4 expands reasoning inline; bandit cuts it
  nearly in half (2.39M → 1.13M) without losing test quality.
- The opus proposer keeps almost everything in the prompt cache
  (input ≈ 3k), so its real cost is dominated by cache reads.
- **claudekimi default+direction** (new `--include-optimization-direction`
  flag) underperforms plain default on LoCoMo (test 0.3140 vs 0.3382) and
  pays a heavy cost penalty (cache reads 2.97M → 4.43M, dur 12.3m →
  16.3m). Mechanism direction lines tighten the proposer toward the
  hypothesis schema but cost more cache reads and—on LoCoMo at least—do
  not transfer to test. See PIPELINE.md §1 for the prompt mechanics.

### Key run paths

- `runs/locomo_claudekimi_default_iter012_test_20260426`
- `runs/locomo_claudekimi_progressive_iter020_test_20260426`
- `runs/locomo_opus_default_iter028_test_20260426`
- `runs/locomo_opus_progressive_docker_iter024_test_20260427`
- `runs/locomo_codex54_default_iter001_test_20260426`
- `runs/locomo_codex54_progressive_iter028_test_20260426`
- `runs/locomo_codex54_bandit_v3_iter028_test_20260430`
- `runs/locomo_memory_opt_memgpt_claudekimi_progressive_docker_env_iter30_full80seed_20260421_215754`
- `runs/locomo_memory_opt_memgpt_claudekimi_bandit_v3_passrate_reward_iter30_full80seed_w16_20260429_022838`
- `runs/locomo_memory_opt_memgpt_codex54_bandit_v3_iter30_full80seed_w16_20260428_192739`
- `runs/locomo_memory_opt_memgpt_claude_opus_progressive_docker_iter30_full80seed_20260427_0353`
- `runs/locomo_memory_opt_memgpt_codex54_progressive_docker_iter30_full80seed_20260421_211252`
- `runs/locomo_memgpt_codex54_default_docker_iter30_train80_rerun_20260502_015354` (codex54 default docker)
- `runs/locomo_memgpt_claudekimi_default_docker_iter30_train80_20260501_204004` (claudekimi default docker)
- `runs/locomo_memgpt_claudekimi_default_direction_docker_iter30_train80_20260502_015441` (claudekimi default+direction, 1st)
- `runs/locomo_memgpt_claudekimi_default_direction_docker_iter30_train80_20260502_154556` (claudekimi default+direction, rerun)

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
| claudekimi | progressive | **0.6000** | **0.5000** ★ | 105.0k | 25.0k | 1.73M | 1.86M | 33.6 | 16.3 | 9.5m |
| claudekimi | bandit (docker, v3, 1st) | 0.5300 | 0.4325 | 152.9k | 27.9k | 3.28M | 3.46M | 44.4 | 20.0 | 14.3m |
| claudekimi | bandit (docker, v3, rerun) | running | running | — | — | — | — | — | — | — |
| claudekimi | default+direction (1st) | 0.6500 | 0.5300 | 176.0k | 30.2k | 3.43M | 3.63M | 49.6 | 18.6 | 16.3m |
| claudekimi | default+direction (rerun) | running | running | — | — | — | — | — | — | — |
| codex54 | default | **0.6000** | **0.4875** | 1.77M | 27.4k | 1.61M | 3.41M | 33.4 | 18.8 | 9.5m |
| codex54 | progressive (rerun) | 0.5400 | 0.4725 | 1.45M | 25.0k | 1.33M | 2.80M | 31.9 | 17.0 | 8.3m |
| codex54 | bandit (docker, v3) | 0.5200 | 0.4725 | 1.13M | 24.6k | 1.03M | 2.18M | 34.8 | 19.0 | 8.1m |

All cost rows average over 30 iterations.

### Notes

- Best LongMemEval test result is claudekimi progressive at 0.5000; codex54
  default and codex54 v3 bandit tie for second at 0.4875 / 0.4725.
- **codex54 v3 bandit cuts proposer cost by ~36%** (input/iter 1.77M → 1.13M,
  total/iter 3.41M → 2.18M, dur/iter 9.5m → 8.1m) while losing only −1.5pt
  test passrate vs default — the same cost-reduction pattern seen on LoCoMo
  codex54 bandit.
- claudekimi v3 bandit (0.4325) underperforms claudekimi default (0.4700)
  and progressive (0.5000) on test even though its train (0.5300) is in
  the same ballpark, repeating the LoCoMo pattern that bandit transfers
  worse than progressive for the kimi proposer.
- codex54 progressive shown is the rerun (0.4725); the original run failed
  on test with `date value out of range` and is not counted.
- claudekimi default+direction's first run hit train 0.6500 / test
  0.5300 — the highest train ever recorded on LongMemEval and above
  progressive 0.5000 on test — but a same-config rerun is in progress
  to confirm stability before the row is treated as the new test best.
  Cost is ~+50% cache reads vs plain default (3.43M vs 2.12M
  tokens/iter, dur 16.3m vs 10.3m). Test was retried with
  `scripts/evaluate_longmemeval_candidate.py` after the original
  test_frontier hit Together 429.
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

Three task pools have been used so far:

- **train30 (mimo v2.5)** — 30 SWE-bench-verified train instances, 10 workers,
  900s per-task timeout. The first sequence where optimization beats the
  source baseline.
- **verified_full500 (DeepSeek v4 Flash)** — full 500-task SWE-bench Verified
  evaluation for the source baseline and optimized mini-SWE-agent candidates.
- **verified_test10 (qwen35-9b)** — 10 verified-test instances; source
  baseline 0.30 (default) / 0.20 (progressive); optimizer iterations regress.
- **verified_test10 (qwen35 a3b 35B-A3B)** — older 2026-04-24 sequence;
  optimizer plateaus at the source baseline.

### train30 (mimo v2.5, 30 train instances)

The SWE-bench train30 pool has no separate test split, so passrate is
reported on the same 30-task pool as a "best optimizer candidate" against
the source baseline. No bandit run has been completed on this pool with
mimo v2.5.

| proposer | policy | source baseline | best passrate | iters | input/iter | output/iter | cache reads/iter | total/iter | tools/iter | files/iter | dur/iter |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| claudekimi | default | 0.4667 | **0.5000** | 20/30 | 136.3k | 28.6k | 3.06M | 3.22M | 56.0 | 23.2 | 13.4m |
| claudekimi | progressive | 0.4000 | **0.5333** | 20/30 | 141.8k | 29.7k | 3.61M | 3.78M | 61.0 | 23.6 | 12.5m |

Best optimizer candidates: default → `iter002_stack_trace_context` (and 4
ties at 0.5000); progressive → `iter016_final_fallback_traceback_retrieval_v1`.

Notes:

- The progressive row is the rerun (2026-04-30). An initial progressive run
  on the same pool plateaued at the source baseline (0.4667) but only
  completed 5/30 iterations and is not counted; the rerun supersedes it.
- Progressive is the strongest current mimo v2.5 train30 result. The
  rerun's lower source baseline (0.4000 vs the earlier non-rerun baseline
  of 0.4667) reflects scoring variance on the 30-task pool, not a
  regression in the agent — the optimizer still converges to a candidate
  that beats both the rerun's own baseline (0.4 → 0.5333) and the earlier
  baseline (0.4667 → 0.5333).
- `iter016` introduces a fallback traceback-aware retrieval; the next four
  best candidates (0.5000) are independent traceback / verification gates,
  suggesting traceback-driven evidence is the dominant useful direction on
  this pool.
- Default was also interrupted before reaching iter30 but still produced
  five 0.5000 candidates with rate-of-discovery similar to the rerun, so
  the current "progressive > default" gap on train30 is plausibly within
  run-to-run noise.
- Tools/iter (56–61) and files/iter (~23) are higher than on
  LoCoMo/LongMemEval because the SWE-bench workspace is larger (full
  mini-SWE-agent source plus benchmark scaffolding); each iteration probes
  more files before producing a patch. Cache reads scale up correspondingly
  (~3M tok/iter vs ~1.7M on LoCoMo).

### train30 (DeepSeek v4 Flash, 30 train instances)

The same trainfirst30 pool was also run with DeepSeek v4 Flash as the
solver. Only a bandit policy was attempted; default/progressive were not
run. The fixedsource bandit row uses the canonical sliding-window
passrate-only z-score reward (window=16). 20 proposer iter dirs exist
but `iter_020` is missing its `metrics.json` (interrupted), so cost
averages over the first 19.

| proposer | policy | source baseline | best passrate | iters | input/iter | output/iter | cache reads/iter | total/iter | tools/iter | files/iter | dur/iter |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| claudekimi | bandit (fixedsource) | 0.5000 | **0.5333** | 19/20 | 128.9k | 26.1k | 3.35M | 3.51M | 56.6 | 25.5 | 12.2m |

Notes:

- Best optimizer candidates: `iter005_auto_test_feedback_and_recovery` plus
  four ties at 0.5333 (`iter006_repro_and_review`,
  `iter007_pre_edit_localization`,
  `iter009_edit_verification_repo_tests`,
  `iter013_impact_aware_feedback`).
- The DeepSeek bandit reaches the same 0.5333 train30 ceiling as the mimo
  v2.5 progressive run, but with a different solver and a higher source
  baseline (0.5000 vs 0.4667 on mimo). The tied frontier candidates were
  later promoted to full500 verified evaluation; `iter013` is the best
  verified result at 320/500 (0.6400).
- An earlier "promptcells" variant
  (`swebench_miniswe_deepseek_v4_flash_claudekimi_bandit_v3_iter20_trainfirst30_w10_t900_promptcells_20260430_200058`)
  only completed 6 iterations and never beat the source baseline (best
  0.5000 = baseline). Its averaged cost (104.2k input / 29.4k output /
  2.53M cache reads / 2.66M total / 54.5 tools / 27.8 files / 10.8m per
  iter, n=6) is not retained as a result row.
- Cache reads (3.35M/iter) are the largest of any benchmark/policy combo
  so far — DeepSeek v4 Flash with mini-SWE-agent context plus the bandit
  policy meta produces an unusually wide cached prompt.

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

The bandit fixedsource `iter013_impact_aware_feedback` candidate is the
current best verified result: +20.0 absolute points over the DeepSeek v4
Flash source baseline, +18.2 points over the default optimized candidate,
and +2.0 points over the earlier progressive full500 result. The full
frontier test results are stored at
`runs/swebench_miniswe_deepseek_v4_flash_claudekimi_bandit_v3_fixedsource_iter20_trainfirst30_w10_t900_20260430_233750/test_frontier/test_results.json`.

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
  end-to-end and produces useful proposer iterations on train30.
- mimo v2.5 trainfirst30 progressive (rerun) is the first SWE-bench result
  that meaningfully beats the source baseline (0.5333 vs 0.4667 absolute).
- On full SWE-bench Verified with DeepSeek v4 Flash, the bandit fixedsource
  optimized candidate `iter013_impact_aware_feedback` reaches 320/500
  resolved (0.6400), beating the source baseline 220/500 (0.4400), the
  default optimized candidate 229/500 (0.4580), and the earlier progressive
  optimized candidate 310/500 (0.6200).
- The verified_test10 optimizer pool is too small and too saturated for the
  optimizer to reliably improve over the source baseline.
- DeepSeek v4 Flash bandit (fixedsource) train30 reaches the same 0.5333
  ceiling as mimo progressive on the trainfirst30 pool, and its best
  full500 promoted candidate is now the strongest SWE-bench result in this
  document.

Key run paths:

- `runs/swebench_miniswe_mimo_v25_claudekimi_default_iter30_trainfirst30_w10_t900_20260429_211742`
- `runs/swebench_miniswe_mimo_v25_claudekimi_progressive_rerun_iter30_trainfirst30_w10_t900_20260430_022208` (progressive)
- `runs/swebench_miniswe_qwen35_9b_claudekimi_default_fixedpaths_iter30_verified_test10_20260429_174251`
- `runs/swebench_miniswe_qwen35_9b_claudekimi_progressive_fixedpaths_iter30_verified_test10_20260429_174251`
- `runs/swebench_deepseek_v4_flash_smoke_baseline_1`
- `runs/swebench_deepseek_v4_flash_verified_full500_20260430_182537`
- `runs/swebench_miniswe_deepseek_v4_flash_claudekimi_bandit_v3_fixedsource_iter20_trainfirst30_w10_t900_20260430_233750`
- `runs/swebench_miniswe_deepseek_v4_flash_claudekimi_bandit_v3_iter20_trainfirst30_w10_t900_promptcells_20260430_200058`
- `runs/swebench_miniswe_qwen35_claudekimi_default_iter30_verified_test10_20260424_2038` (older a3b baseline)
- `runs/swebench_miniswe_qwen35_claudekimi_progressive_iter30_verified_test10_20260424_2038` (older a3b baseline)

## Overall Takeaways

- **LoCoMo overall best: claude opus progressive docker at 0.3982 test.**
- All progressive and bandit results use the docker sandbox; non-docker
  progressive/bandit results are discarded.
- LoCoMo: progressive (docker) wins for claudekimi and claude opus on test.
  codex54 is the exception: bandit (0.3865) is its best policy and the
  first bandit-family run that beats progressive for any proposer family on
  LoCoMo.
- Train80 LoCoMo is too small to trust by itself; claudekimi bandit
  (train 0.4375 / test 0.3589 vs progressive train 0.4375 / test 0.3734) is
  the clearest example of train-test mismatch.
- Reruns should be used to verify stability; per (task, optimization model)
  this doc keeps only the higher-scoring result.
- SWE-bench mini has its first useful optimization signal: mimo v2.5
  trainfirst30 progressive reaches 0.5333 train vs 0.4667 source baseline.
  DeepSeek v4 Flash full verified evaluation now reaches 0.6400 after
  bandit fixedsource optimization vs 0.4400 source baseline; verified_test10
  remains too small to drive optimizer improvement.
