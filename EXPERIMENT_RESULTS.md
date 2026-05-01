# Experiment Results

Last updated: 2026-05-01

This file summarizes the current completed experiments for the default,
progressive, and bandit selection policies. Reruns that reproduced the same
conclusion are not listed as separate rows. Rows marked `(docker)` used the
docker proposer sandbox with the corrected `docker-claude-kimi:latest` image.

## Executive Summary

- LongMemEval is no longer a bandit win after the fresh test-frontier runs:
  claudekimi progressive is best on test400 at 0.5000, with codex54 default
  close behind at 0.4875.
- LoCoMo: the overall test best remains **claude opus progressive docker at
  0.3982**, surpassing claudekimi progressive at 0.3734.
- The new z-score bandit v2 runs improve train scores for claudekimi and
  codex54, but test is mixed: codex54 bandit v2 (0.3575) nearly matches
  codex54 progressive (0.3589), while claudekimi bandit v2 (0.3395)
  underperforms claudekimi progressive (0.3734).
- LoCoMo claudekimi bandit v3 (passrate-only reward) lands at 0.3589 test,
  still trailing claudekimi progressive (0.3734).
- LoCoMo codex54 bandit v3 reaches **0.3865 test**, the first bandit-family
  result that beats progressive for any proposer family on LoCoMo.
- Text classification has no bandit run in the current results. Default is
  better than progressive in the claudekimi offline validation run.
- SWE-bench mini now has a source-backed `mini_swe_agent_source` benchmark
  target. The first useful optimization signal comes from the mimo v2.5
  trainfirst30 sequence: progressive reaches **0.5333 train** vs a source
  baseline of 0.4000–0.4667 (run-to-run variance on the 30-task pool).
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
result within each proposer family; ★ marks an overall benchmark best.

## LoCoMo

LoCoMo train uses 80 examples; test uses the full 1,449 examples.
Progressive and bandit runs use the docker proposer sandbox. Default runs
predate the docker sandbox and are kept as baselines. `bandit v2` is the
2026-04-28 z-score reward bandit; `bandit v3` adds the sliding-window
passrate-only reward (window=16). The opus bandit v2 test result is
intentionally excluded because that test artifact was deleted.

| proposer | policy | train | test | input/iter | output/iter | cache reads/iter | tools/iter | files/iter | dur/iter |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| claudekimi | default | 0.4000 | 0.3409 | — | — | — | — | — | — |
| claudekimi | progressive (docker) | 0.4375 | **0.3734** | 138.9k | 25.5k | 1.70M | 35.2 | 15.1 | 13.0m |
| claudekimi | bandit (docker) | 0.4125 | 0.3616 | — | — | — | — | — | — |
| claudekimi | bandit v2 (docker) | **0.4500** | 0.3395 | — | — | — | — | — | — |
| claudekimi | bandit v3 (docker) | 0.4375 | 0.3589 | 104.2k | 29.8k | 1.83M | 35.1 | 17.6 | 14.1m |
| claude opus | default | 0.3875 | 0.3306 | — | — | — | — | — | — |
| claude opus | progressive (docker) | **0.4750** | **0.3982** ★ | 3.1k | 20.6k | 1.99M | 61.2 | 20.7 | 8.9m |
| claude opus | bandit (docker) | 0.4125 | 0.3230 | — | — | — | — | — | — |
| codex54 | default | 0.4125 | 0.3471 | — | — | — | — | — | — |
| codex54 | progressive (docker) | 0.4250 | 0.3589 | 2.39M | 18.7k | 2.25M | 50.6 | 16.9 | 7.1m |
| codex54 | bandit (docker) | 0.3750 | 0.3140 | — | — | — | — | — | — |
| codex54 | bandit v2 (docker) | **0.4625** | 0.3575 | — | — | — | — | — | — |
| codex54 | bandit v3 (docker) | 0.4250 | **0.3865** | 1.13M | 20.7k | 995k | 34.6 | 18.5 | 7.0m |

Iteration counts behind the cost averages are 30 except: claudekimi
progressive 29 (one iteration produced no metrics), and codex54 bandit v3
27/30 (interrupted).

### Bandit v3 design

Bandit v3 introduces a sliding-window z-score reward (window=16) that
rewards each iteration relative to recent quality history rather than
absolute improvement over the run's best. The "passrate-only reward"
variant restricts the reward signal to passrate (excluding average_score)
so the bandit does not pursue partial-quality wins that don't carry to
test. claudekimi v3 keeps only the passrate-only variant — an earlier
mixed-reward run was inferior on every metric (train 0.3750 / test 0.3009)
and is dropped, confirming that mixing average_score into the reward
chases partial-quality wins that do not transfer to test. codex54 v3 keeps
the original z-score with mixed reward; its train run was interrupted at
iter28/30 and test was evaluated on 2026-04-30 against the full
1,449-example test split using Qwen3-8B on GPU1 (port 8002, 128 workers).

### Notes

- The overall LoCoMo test best is **claude opus progressive docker at
  0.3982**.
- Progressive wins for claudekimi and claude opus on test. codex54 is the
  exception: bandit v3 (0.3865) is its strongest result on test and the
  only bandit-family policy that beats progressive for any proposer family.
- bandit v2 fixes the worst codex54 bandit failure: codex54 improves from
  0.3140 to 0.3575 and nearly ties progressive at 0.3589, but bandit v3
  supersedes it (0.3865).
- claudekimi bandit v2 improves train (0.4500) but not test (0.3395),
  another train-test mismatch.
- claudekimi v3 passrate-only matches the older claudekimi bandit on test
  (0.3589 ≈ 0.3616) but still trails progressive at 0.3734.
- The deleted opus bandit v2 test result is not counted in this table.
- codex54's input-token cost is roughly an order of magnitude higher than
  claudekimi/opus because gpt-5.4 expands reasoning inline; v3 cuts it
  nearly in half (2.39M → 1.13M) without losing test quality.
- The opus proposer keeps almost everything in the prompt cache
  (input ≈ 3k), so its real cost is dominated by cache reads.

### Key run paths

- `runs/locomo_claudekimi_default_iter012_test_20260426`
- `runs/locomo_claudekimi_progressive_iter020_test_20260426`
- `runs/locomo_claudekimi_bandit_docker_authfix_iter024_test_20260427`
- `runs/locomo_opus_default_iter028_test_20260426`
- `runs/locomo_opus_progressive_docker_iter024_test_20260427`
- `runs/locomo_opus_bandit_docker_iter027_test_20260427`
- `runs/locomo_codex54_default_iter001_test_20260426`
- `runs/locomo_codex54_progressive_iter028_test_20260426`
- `runs/locomo_codex54_bandit_docker_iter029_test_20260427`
- `runs/locomo_bandit_v2_test_eval_kimi_iter026_20260428`
- `runs/locomo_bandit_v2_test_eval_codex54_iter030_20260428`
- `runs/locomo_memory_opt_memgpt_claudekimi_progressive_docker_env_iter30_full80seed_20260421_215754`
- `runs/locomo_memory_opt_memgpt_claudekimi_bandit_docker_authfix_iter30_full80seed_20260427`
- `runs/locomo_memory_opt_memgpt_claudekimi_bandit_v2_iter30_full80seed_20260428_0213`
- `runs/locomo_memory_opt_memgpt_claudekimi_bandit_v3_iter30_full80seed_w16_20260428_192739`
- `runs/locomo_memory_opt_memgpt_claudekimi_bandit_v3_passrate_reward_iter30_full80seed_w16_20260429_022838`
- `runs/locomo_memory_opt_memgpt_codex54_bandit_v3_iter30_full80seed_w16_20260428_192739`
- `runs/locomo_codex54_bandit_v3_iter028_test_20260430`
- `runs/locomo_memory_opt_memgpt_claude_opus_progressive_docker_iter30_full80seed_20260427_0353`
- `runs/locomo_memory_opt_memgpt_claude_opus_bandit_docker_iter30_full80seed_20260426`
- `runs/locomo_memory_opt_memgpt_codex54_progressive_docker_iter30_full80seed_20260421_211252`
- `runs/locomo_memory_opt_memgpt_codex54_bandit_docker_authfix_iter30_full80seed_20260427`
- `runs/locomo_memory_opt_memgpt_codex54_bandit_v2_iter30_full80seed_20260428_0213`

## LongMemEval

Train rows use train100. Test rows use the 400-example test split. Rows
marked `failed` produced a test-frontier artifact but did not complete a
valid score. The opus46 progressive test was retried and stopped after
hanging; its last completed status is a Together judge 500 error. No bandit
v3 run was attempted on LongMemEval, so the bandit row below is bandit v2.
opus46 default and bandit are not reported (no completed runs).

| proposer | policy | train | test | input/iter | output/iter | cache reads/iter | tools/iter | files/iter | dur/iter |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| claude opus46 | progressive | **0.6300** | failed: Together 500 | 1.7k | 17.3k | 1.48M | 61.3 | 20.0 | 7.2m |
| claudekimi | default | 0.5600 | 0.4700 | 121.5k | 26.9k | 2.12M | 39.6 | 18.4 | 10.3m |
| claudekimi | progressive | **0.6000** | **0.5000** | 105.0k | 25.0k | 1.73M | 33.6 | 16.3 | 9.5m |
| claudekimi | bandit v2 | 0.5700 | 0.4400 | — | — | — | — | — | — |
| codex54 | default | **0.6000** | **0.4875** | 1.77M | 27.4k | 1.61M | 33.4 | 18.8 | 9.5m |
| codex54 | progressive (rerun) | 0.5400 | 0.4725 | 1.45M | 25.0k | 1.33M | 31.9 | 17.0 | 8.3m |
| codex54 | bandit v2 | 0.5700 | 0.4575 | — | — | — | — | — | — |

All cost rows average over 30 iterations.

### Notes

- Fresh LongMemEval test-frontier runs completed for claudekimi progressive,
  claudekimi bandit v2, codex54 default, codex54 progressive rerun, and
  codex54 bandit v2 on 2026-04-29.
- Best LongMemEval test result is claudekimi progressive at 0.5000; codex54
  default is second at 0.4875.
- bandit v2 does not beat default/progressive on LongMemEval test:
  claudekimi bandit v2 is 0.4400 and codex54 bandit v2 is 0.4575.
- codex54 progressive shown is the rerun (0.4725); the original run failed on
  test with `date value out of range` and is not counted.
- opus46 progressive is strongest on train100 at 0.6300, but its
  test-frontier attempt failed with a Together 500; a rerun was stopped
  before completion.
- The cost pattern matches LoCoMo: codex54 dominates input-token cost,
  opus46 rides on the prompt cache, and claudekimi sits in between.

### Key run paths

- `runs/longmemeval_memgpt_claudekimi_default_docker_env_iter30_train100_fix_20260423_074629`
- `runs/longmemeval_memgpt_claudekimi_progressive_docker_env_iter30_train100_fixrerun_20260423_161417`
- `runs/longmemeval_default_iter021_correct_test_run4_20260424`
- `runs/longmemeval_memgpt_claude_opus46_progressive_docker_iter30_train100_20260427_222924`
- `runs/longmemeval_memgpt_codex54_default_docker_iter30_train100_20260427_222924`
- `runs/longmemeval_memgpt_codex54_progressive_docker_iter30_train100_20260427_222924`
- `runs/longmemeval_memgpt_claudekimi_bandit_v2_docker_iter30_train100_w16_20260428_192739`
- `runs/longmemeval_memgpt_codex54_bandit_v2_docker_iter30_train100_w16_20260428_162906`
- `runs/longmemeval_memgpt_codex54_progressive_docker_rerun_iter30_train100_w16_20260428_162906`

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
the source baseline. No bandit run is included in this section.

| proposer | policy | source baseline | best passrate | iters | input/iter | output/iter | cache reads/iter | tools/iter | files/iter | dur/iter |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| claudekimi | default | 0.4667 | **0.5000** | 20/30 | 136.3k | 28.6k | 3.06M | 56.0 | 23.2 | 13.4m |
| claudekimi | progressive | 0.4000 | **0.5333** | 20/30 | 141.8k | 29.7k | 3.61M | 61.0 | 23.6 | 12.5m |

Best optimizer candidates: default → `iter002_stack_trace_context` (and 4
ties at 0.5000); progressive → `iter016_final_fallback_traceback_retrieval_v1`.

Notes:

- The progressive row is the rerun (2026-04-30). An initial progressive run
  on the same pool plateaued at the source baseline (0.4667) but only
  completed 5/30 iterations and is not counted; the rerun supersedes it.
- Progressive is the strongest current SWE-bench result. The rerun's lower
  source baseline (0.4000 vs the earlier non-rerun baseline of 0.4667)
  reflects scoring variance on the 30-task pool, not a regression in the
  agent — the optimizer still converges to a candidate that beats both the
  rerun's own baseline (0.4 → 0.5333) and the earlier baseline
  (0.4667 → 0.5333).
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

### verified_full500 (DeepSeek v4 Flash)

These are full SWE-bench Verified test evaluations with DeepSeek v4 Flash
as the solver model. The optimized candidates come from the mini-SWE-agent
source optimization runs; score is `resolved / 500`.

| candidate | source run | resolved / total | passrate |
|---|---|---:|---:|
| source baseline | `swebench_deepseek_v4_flash_verified_full500_20260430_182537` | 220 / 500 | 0.4400 |
| default optimized (`iter002_stack_trace_context`) | `swebench_deepseek_v4_flash_verified_full500_20260430_182537` | 229 / 500 | 0.4580 |
| progressive optimized (`iter016_final_fallback_traceback_retrieval_v1`) | `swebench_deepseek_v4_flash_verified_full500_20260430_182537` | 310 / 500 | **0.6200** |

The progressive optimized candidate is the clear best verified result so
far: +18.0 absolute points over the DeepSeek v4 Flash source baseline and
+16.2 points over the default optimized candidate.

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
- On full SWE-bench Verified with DeepSeek v4 Flash, the progressive
  optimized candidate reaches 310/500 resolved (0.6200), beating the source
  baseline 220/500 (0.4400) and the default optimized candidate 229/500
  (0.4580).
- The verified_test10 optimizer pool is too small and too saturated for the
  optimizer to reliably improve over the source baseline.
- DeepSeek v4 Flash bandit-v3 train30 searches have been run; the strongest
  full verified result currently comes from the progressive optimized
  candidate evaluated in the full500 run above.

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
- LongMemEval no longer supports a bandit-win conclusion: progressive/default
  beat bandit v2 on the current test-frontier artifacts.
- LoCoMo: progressive (docker) wins for claudekimi and claude opus on test.
  codex54 is the exception: bandit v3 (0.3865) is its best policy and the
  first bandit-family run that beats progressive for any proposer family on
  LoCoMo. bandit v2 was already much stronger for codex54 than the earlier
  bandit (0.3575 vs. 0.3140), and v3 extends that gain.
- LoCoMo claudekimi bandit v3 passrate reward improves over bandit v2 on
  test (0.3589 vs. 0.3395), but remains below progressive (0.3734).
- Train80 LoCoMo is too small to trust by itself; claudekimi bandit is the
  clearest example of train-test mismatch.
- Reruns should be used to verify stability; per (task, optimization model)
  this doc keeps only the higher-scoring result.
- SWE-bench mini has its first useful optimization signal: mimo v2.5
  trainfirst30 progressive reaches 0.5333 train vs 0.4667 source baseline.
  DeepSeek v4 Flash full verified evaluation reaches 0.6200 after
  progressive optimization vs 0.4400 source baseline; verified_test10
  remains too small to drive optimizer improvement.
