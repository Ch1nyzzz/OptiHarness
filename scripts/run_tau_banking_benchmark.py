#!/usr/bin/env python
"""Run tau3 banking_knowledge through the OptiHarness adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from memomemo.tau_banking import TauBankingRunConfig, run_tau_banking_benchmark


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/tau_banking_baseline"))
    parser.add_argument("--tau2-root", type=Path, default=Path("references/vendor/tau2-bench"))
    parser.add_argument("--python", dest="python_executable", default="python")
    parser.add_argument("--agent-module", type=Path, default=None)
    parser.add_argument("--agent-name", default="optiharness_banking_base")
    parser.add_argument("--factory-name", default="create_banking_knowledge_base_agent")
    parser.add_argument("--retrieval-configs", default="bm25")
    parser.add_argument("--retrieval-config-kwargs", default="{}")
    parser.add_argument("--task-split-name", default="base")
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--agent-llm", default="openai/gpt-4.1-mini")
    parser.add_argument("--user-llm", default="openai/gpt-4.1-mini")
    parser.add_argument("--agent-llm-args", default='{"temperature": 0.0}')
    parser.add_argument("--user-llm-args", default='{"temperature": 0.0}')
    parser.add_argument("--process-timeout-s", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    num_tasks = None if args.num_tasks < 0 else args.num_tasks
    config = TauBankingRunConfig(
        tau2_root=args.tau2_root,
        python_executable=args.python_executable,
        agent_name=args.agent_name,
        agent_module=args.agent_module,
        factory_name=args.factory_name,
        retrieval_configs=tuple(_csv(args.retrieval_configs)),
        retrieval_config_kwargs=json.loads(args.retrieval_config_kwargs),
        task_split_name=args.task_split_name,
        task_ids=tuple(_csv(args.task_ids)),
        num_tasks=num_tasks,
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        agent_llm=args.agent_llm,
        user_llm=args.user_llm,
        agent_llm_args=json.loads(args.agent_llm_args),
        user_llm_args=json.loads(args.user_llm_args),
        process_timeout_s=args.process_timeout_s or None,
    )
    summary = run_tau_banking_benchmark(
        out_dir=args.out,
        config=config,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
