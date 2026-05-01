# MemoMemo Optimization Pipeline

本文档把 MemoMemo proposer/evaluator 优化循环、三种上下文选择策略（**default / progressive / bandit v3**），以及当前阶段的实验结果合并到一处。读完后你应当能够：

1. 看懂 `src/memomemo/optimizer.py` 的整体外环；
2. 在三种策略之间做出有依据的选择；
3. 复现或扩展 LoCoMo / LongMemEval / SWE-bench mini 上的现有结果。

详细的 per-run 数据、cost 表、run path 全集仍保留在仓库根目录的
[`EXPERIMENT_RESULTS.md`](../EXPERIMENT_RESULTS.md) 中，本文只摘录关键数字。

---

## 0. Shared Skeleton（三策略共用的外环）

实现位置：`LocomoOptimizer.run()`（`src/memomemo/optimizer.py:152-284`）。
`SwebenchOptimizer` 继承同一外环，只覆盖样例加载、seed frontier 与候选评估。

三种策略**只在两个点上分叉**：①每轮如何选 budget / 参考迭代 / 文件提示；
③评估完之后写回什么 state。其它（workspace 装配、proposer 调用、frontier
持久化）完全一致。

![Shared Skeleton](memomemo_shared_skeleton.svg)

每轮迭代会把全部产物写到 `runs/<run_id>/proposer_calls/iter_NNN/`；proposer
本身跑在挂载到 `/workspace/` 的 Docker sandbox 里，看不到仓库根、原始 benchmark
数据或评分助手——这些被 `access_policy.json` 阻断。

---

## 1. Default 策略（fixed-high baseline）

**入口**：`OptimizerConfig.selection_policy = "default"`
（`src/memomemo/optimizer.py:220-221`）。

**决策规则**：每一轮硬编码 `budget = "high"`，不读不写任何 state。

![Default Policy: fixed high](default_policy_fixed_high.svg)

**特性**：

- 每轮上下文最大、成本最高（cache miss + 长 prompt + 全量 ref-iter 副本）。
- 没有反馈环：proposer 永远看到同一个全局视图。
- 作为 progressive / bandit 的 sanity baseline。

---

## 2. Progressive 策略

**入口**：`OptimizerConfig.selection_policy = "progressive"`。
**State 文件**：`runs/<run_id>/progressive_state.json`。

**核心思路**：用 `low / medium / high` 三档 budget 显式地调节上下文规模，
**根据上一轮是否产生 Pareto frontier 改进来升降档**。一旦命中改进就回到 `low`，
不持续付 `high` 的 token 代价。

![Progressive Policy state machine](progressive_policy_state_machine.svg)

| budget | trace_scope | refs                  | prompt 长度 |
|--------|-------------|-----------------------|-------------|
| low    | last1       | best1 + worst         | 最短        |
| medium | last3       | best3 + worst         | 中等        |
| high   | all         | 所有历史迭代          | 最长        |

**单轮流程**：

1. `_progressive_budget_for_iteration(k)` 跑 state machine 取 budget；
2. `_reference_iterations_for_budget` 按 budget 取 refs；
3. workspace 装配，按 trace_scope 修剪每个 ref bundle 的 trace_slices；
4. `build_progressive_proposer_prompt` 注入 `selection_policy="progressive"`，
   写入 best/worst role hint 与 Optimization Focus（memgpt: 4 cells / mini-swe-agent: 5 cells）；
5. proposer 在 docker 内产出 `pending_eval.json`；
6. `_evaluate_proposed` 算 CandidateResult；
7. `_update_progressive_state`：本轮是否进入新 frontier？是→`improved=True`、stagnation 清零；否→升档。

**为什么有效**：前几轮用低成本窄上下文广撒网；只有真正卡住才升档；新增改进
立即回到 `low`，整体平均成本远低于 default。

---

## 3. Bandit v3 策略

**入口**：`OptimizerConfig.selection_policy = "bandit"`。
**State 文件**：`runs/<run_id>/bandit_state.json`。

**核心思路**：把 workspace 中每个可读文件视为多臂赌博机的一条 arm。每轮根据"读了
该文件并取得 z-score 正奖励"的相关性给文件打分，把 top-N 作为 `hot` / `warm`
显式塞进 prompt，让 proposer 一上手就去过去真正有回报的文件。

![Bandit v3 decision flow](bandit_v3_decision_flow.svg)

### 3.1 顶层决策

进入第 k 轮时读取 `bandit_state.json`：

```
state["files"] = {
    path → {read_iters, success_iters, reward_sum, read_lines,
            write_iters, changed_iters, utility, policy_score, …}
}
```

按 `policy_score` 排序（required core 永远在 hot 里、不参与排序）：
- `hot` = core_files + top-8，read budget = 800 行
- `warm` = next 12，read budget = 300 行

budget / trace_scope 的决定：

```
iter == 1 或没有任何文件统计 → low / last1 / refs=()
stagnation ≥ bandit_stagnation_threshold(=4) → high / all / 全部历史 refs
否则 → medium / last3 / (hot 出现过的迭代 ∪ best3 ∪ last_improved，cap 5)
```

### 3.2 Reward（v3 的关键改动）

```
best_eval_passrate = max(c.passrate for c in evaluated)   # v3：仅 passrate
history.append(best_eval_passrate)
recent = history[-bandit_reward_window:]                   # 默认 8，论文里的 v3 跑用 16

if not evaluated:
    reward = -clip * 0.25
elif len(recent) < 2:                                      # warm-up
    reward = clip((best_eval_passrate − previous_best) * 10, ±clip)
else:
    μ, σ = mean/std(recent),  σ ≥ 0.02
    reward = clip((best_eval_passrate − μ) / σ, ±clip)     # 滑动窗 z-score
success = reward > 0
```

v3 vs v1/v2 的两点变化：
1. **passrate-only reward**：`average_score` 不再混入。混合奖励的 claudekimi
   跑在每个指标上都更差——它会去追那些 train 上提分、test 上不迁移的"半正确"
   候选。
2. **滑动窗 z-score**：奖励标准化到近期窗口而非全局最佳，停滞期里出现一次有
   意义的反弹也能拿到正分。

### 3.3 文件级 utility 更新

每轮 reward 确定后：

```
本轮 proposer 实际读过的每个 path:
    files[path].read_iters   += 1
    files[path].read_calls   += 当轮 read 次数
    files[path].read_lines   += 当轮 read 行数
    files[path].reward_sum   += 当轮 reward
    if success: files[path].success_iters += 1

本轮被写过 / 出现在 diff 的 path:
    files[path].write_iters   += 1（写过）
    files[path].changed_iters += 1（在 diff 里）
    （这些不计入 reward 分母）
```

之后 `_recompute_bandit_scores` 全局重新打分（仅在"自由读"池里——`required core` 不算）：

```
p_global         = scored_success_iters / scored_read_iters
mean_reward_glob = scored_reward_sum    / scored_read_iters

p_file       = (success_iters + α·p_global) / (read_iters + α)        # Beta-smoothed
mean_reward  = (reward_sum    + α·prior_w·mean_reward_glob)
               / (read_iters + α)
avg_lines    = read_lines / read_iters
cost         = cost_λ · log1p(avg_lines / line_scale)                  # 长度惩罚
bonus        = c · sqrt(log(total_iters + 1) / (read_iters + 1))       # UCB exploration
binary_util  = p_file       - p_global
reward_util  = mean_reward  - mean_reward_glob

policy_score = 0.7·binary_util + 0.3·reward_util − cost + bonus
```

| 超参                            | 默认  |
|---------------------------------|------:|
| `bandit_prior_alpha`            |  2.0  |
| `bandit_prior_weight`           |  0.4  |
| `bandit_exploration_c`          |  0.15 |
| `bandit_cost_lambda`            |  0.05 |
| `bandit_line_scale`             |  500  |
| `bandit_reward_window`          |  8    |
| `bandit_reward_sigma_floor`     |  0.02 |
| `bandit_reward_clip`            |  2.0  |
| `bandit_stagnation_threshold`   |  4    |
| `bandit_failed_iter_penalty`    |  0.5  |

### 3.4 Required core files（永远 hot、永不打分）

`_bandit_core_files`（`src/memomemo/optimizer.py:1951`）保持一组固定的"地基文件"
始终在 hot 列表内，与统计无关：

- 当前 `source_family` 的 scaffold 源文件（如 `memgpt_source.py`）；
- `scaffolds/base.py`、`model.py`、`schemas.py`；
- `summaries/` 下的六个汇总文件（`evolution_summary.jsonl`、
  `best_candidates.json`、`candidate_score_table.json`、
  `retrieval_diagnostics_summary.json`、`diff_summary.jsonl`、
  `iteration_index.json`）；
- `pending_eval.json`。

这些文件不参与 bandit 打分，避免小样本噪声把 essentials 排到 warm/cold。

---

## 4. 三策略对照表

| 维度                | default              | progressive                          | bandit v3                                        |
|---------------------|----------------------|--------------------------------------|--------------------------------------------------|
| Budget 选择         | 永远 `high`          | state machine（`low`→…→`high`）       | 启发式 + 停滞阈值（`low/medium/high`）           |
| 参考迭代            | 全历史               | 按 budget：best k + worst             | hot 文件出现过的迭代，回退 best3 / last_improved |
| Trace-scope 修剪    | `all`                | `last1 / last3 / all`                 | 同三档，由 budget 推导                            |
| Prompt 附加段       | Optimization Focus   | + Progressive role hints             | + Bandit Context Policy（hot/warm 列表）         |
| 反馈信号            | 无                   | 本轮是否进入新 frontier              | 滑动窗 z-score（passrate-only）                  |
| State 文件          | 无                   | `progressive_state.json`              | `bandit_state.json`（per-file 统计）             |
| Explore vs exploit  | 无（永远最大）       | 隐式（停滞才升档）                    | 显式（UCB bonus + Beta smoothing）               |
| 成本画像            | 最高（每轮 high）    | 中（多数轮 low）                      | 中高（更多读 + policy meta）                     |
| 长处                | 复现 baseline；体检   | claudekimi/opus 在 LoCoMo/LongMemEval 上最稳 | codex54 在 LoCoMo test 上唯一压过 progressive    |
| 短处                | 不会收敛             | 一旦 high 难以快速回落                | 冷启动期需要积累；混合奖励会过拟训练            |

---

## 5. 实验结果摘要

详细数据、per-iter cost、所有 run path 见
[`EXPERIMENT_RESULTS.md`](../EXPERIMENT_RESULTS.md)。本节只列结论性数字。

### 5.1 LoCoMo（train=80，test=1449）

**Test 集 passrate**（粗体 = 该 proposer 家族最优）：

| proposer family | default | progressive (docker) | bandit (docker) | bandit v2 (docker) | bandit v3 (docker) | best        |
|-----------------|--------:|---------------------:|----------------:|-------------------:|-------------------:|-------------|
| claudekimi      | 0.3409  | **0.3734**           | 0.3616          | 0.3395             | 0.3589             | progressive |
| claude opus     | 0.3306  | **0.3982**           | 0.3230          | N/A                | N/A                | progressive ★ |
| codex54         | 0.3471  | 0.3589               | 0.3140          | 0.3575             | **0.3865**         | bandit v3   |

要点：

- **LoCoMo 全局最佳：claude opus progressive docker，0.3982 test。**
- claudekimi / opus 上 progressive 全面胜出；codex54 是唯一 bandit-family 压过
  progressive 的家族——bandit v3（0.3865）也是当前唯一在任何 proposer 上都打败
  progressive 的 bandit 结果。
- bandit v3 把 codex54 的成本接近砍半（input/iter 2.39M→1.13M）且 test 不退步。

### 5.2 LongMemEval（train=100，test=400）

| proposer family | default     | progressive          | bandit  | best        |
|-----------------|------------:|---------------------:|--------:|-------------|
| claude opus46   | N/A         | failed: Together 500  | N/A     | —           |
| claudekimi      | 0.4700      | **0.5000**            | 0.4400  | progressive |
| codex54         | **0.4875**  | 0.4725                | 0.4575  | default     |

要点：bandit v2 在 LongMemEval test 上不再领先，progressive/default 占优；
opus46 train 0.6300 最强，但 test-frontier 跑因 Together 500 失败未计。

### 5.3 SWE-bench mini

源代码后端 (`mini_swe_agent_source`) 已经 wired 到 optimize CLI（`--swebench`），
当前出有意义信号的是 mimo v2.5 trainfirst30：

| policy         | source baseline | best optimizer candidate                   | best passrate | iters     |
|----------------|----------------:|--------------------------------------------|--------------:|----------:|
| claudekimi default     | 0.4667 | iter002 `stack_trace_context` 等 5 个并列    | **0.5000**    | 20/30     |
| claudekimi progressive | 0.4000 | `iter016_final_fallback_traceback_retrieval_v1` | **0.5333**    | 20/30     |

更关键的是 DeepSeek v4 Flash 全 500 题 verified 评估：

| candidate                                                    | resolved/500 | passrate    |
|--------------------------------------------------------------|-------------:|------------:|
| source baseline                                              | 220 / 500    | 0.4400      |
| default optimized (`iter002_stack_trace_context`)            | 229 / 500    | 0.4580      |
| progressive optimized (`iter016_final_fallback_traceback_retrieval_v1`) | 310 / 500    | **0.6200**  |

progressive 优化候选在全量 verified 上比源 baseline 高 +18.0 个百分点，是当前最强
SWE-bench 结果。verified_test10（10 题）pool 太小且过饱和，optimizer 无法稳定改进。

### 5.4 总体 takeaway

- LoCoMo 全局最优：claude opus progressive docker @ 0.3982 test。
- LongMemEval 不再支持"bandit 必胜"的结论：progressive / default 在 test 上压
  bandit v2。
- LoCoMo 上 progressive 是 claudekimi / opus 的稳赢策略；codex54 受益于 bandit v3。
- train80 LoCoMo 单看不够稳，应配 test 1449；claudekimi bandit 是 train-test
  不一致最明显的反例。
- SWE-bench mini 第一个真正的优化信号来自 progressive：trainfirst30 0.5333（vs
  baseline 0.4667），DeepSeek Flash full500 0.6200（vs baseline 0.4400）。
- 所有 progressive / bandit 结果都用 docker sandbox；非 docker 结果不计入。

---

## 6. 评审者文件索引

| 概念                                | 位置                                       |
|-------------------------------------|--------------------------------------------|
| 外环（三策略 fork）                  | `src/memomemo/optimizer.py:207-258`        |
| Progressive state machine           | `src/memomemo/optimizer.py:1568-1638`      |
| Bandit policy 装配                   | `src/memomemo/optimizer.py:1661-1744`      |
| Bandit reward + per-file 统计         | `src/memomemo/optimizer.py:1801-1873`      |
| Bandit 打分公式                      | `src/memomemo/optimizer.py:1875-1935`      |
| Required core files                 | `src/memomemo/optimizer.py:1951-1981`      |
| Pareto frontier 定义                | `src/memomemo/pareto.py`                   |
| Proposer prompt 模板                | `src/memomemo/proposer_prompt.py`          |
| Optimization cells                  | `src/memomemo/optimization_cells.py`       |
| SWE-bench 子类                      | `src/memomemo/swebench_optimizer.py`       |
| CLI `--selection-policy` flag       | `src/memomemo/cli.py:255-263`              |
| 实验结果详细                         | [`EXPERIMENT_RESULTS.md`](../EXPERIMENT_RESULTS.md) |

---

## 7. Cheat Sheet — 跑每种策略

`AGENTS.md` 的 docker 沙箱规则适用：kimi proposer 必须用
`docker-claude-kimi:latest`；claude proposer 需要把宿主机 `.claude` 与
`.claude.json` 挂载进去。API key 放在 `.env`，启动前必须 source。

```bash
# 先 source 凭证
set -a && source /data/home/yuhan/MemoMemo/.env && set +a

# 1) DEFAULT（fixed-high baseline）— 无 state、无 docker 强依赖
python -m memomemo.cli optimize \
  --run-id default_demo \
  --iterations 30 --selection-policy default \
  --proposer-agent codex --proposer-sandbox none

# 2) PROGRESSIVE（claudekimi，docker）
python -m memomemo.cli optimize \
  --run-id progressive_demo --iterations 30 \
  --selection-policy progressive \
  --proposer-agent kimi --proposer-sandbox docker \
  --proposer-docker-image docker-claude-kimi:latest \
  --proposer-docker-user 1023:1023 --proposer-docker-home /tmp \
  --proposer-docker-env KIMI_API_KEY

# 3) BANDIT v3（codex54，docker）
python -m memomemo.cli optimize \
  --run-id bandit_v3_demo --iterations 30 \
  --selection-policy bandit \
  --proposer-agent codex --proposer-sandbox docker \
  --proposer-docker-image docker-codex:latest \
  --bandit-reward-window 16   # 论文里的 v3 跑用 16
```

训练跑完后用 `scripts/evaluate_candidate_json.py` 把 `runs/<train_run>/best_candidates.json`
喂进去做 test 评估（具体 recipe 见 `AGENTS.md`）。
