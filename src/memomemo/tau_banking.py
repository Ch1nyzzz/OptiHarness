"""Adapter for tau3 banking_knowledge benchmark runs.

The tau3 banking_knowledge implementation lives in the tau2-bench repository
and currently has a different Python runtime floor from OptiHarness. This module
therefore treats tau2-bench as an external runner: it writes a small agent
module, executes a self-contained tau2 runner script, and converts tau2 rewards
into OptiHarness candidate-result artifacts.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from memomemo.pareto import ParetoPoint, save_frontier
from memomemo.schemas import CandidateResult


TAU_BANKING_DOMAIN = "banking_knowledge"
TAU2_BENCH_REPO_URL = "https://github.com/sierra-research/tau2-bench"
DEFAULT_TAU2_ROOT = Path("references/vendor/tau2-bench")
DEFAULT_TAU_BANKING_RETRIEVAL_CONFIGS = ("bm25",)
DEFAULT_TAU_BANKING_AGENT_NAME = "optiharness_banking_base"
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_AGENT_SOURCE = PACKAGE_ROOT / "tau_agents" / "banking_knowledge_base_agent.py"
DEFAULT_RUNTIME_SOURCE = PACKAGE_ROOT / "tau_agent_runtime"


@dataclass(frozen=True)
class TauBankingRunConfig:
    """Runtime config for an external tau3 banking_knowledge evaluation."""

    tau2_root: Path = DEFAULT_TAU2_ROOT
    python_executable: str = "python"
    agent_name: str = DEFAULT_TAU_BANKING_AGENT_NAME
    agent_module: Path | None = None
    agent_snapshot_root: Path | None = None
    factory_name: str = "create_banking_knowledge_base_agent"
    retrieval_configs: tuple[str, ...] = DEFAULT_TAU_BANKING_RETRIEVAL_CONFIGS
    retrieval_config_kwargs: dict[str, Any] = field(default_factory=dict)
    task_split_name: str = "base"
    task_ids: tuple[str, ...] = ()
    num_tasks: int | None = 5
    num_trials: int = 1
    max_steps: int = 200
    max_errors: int = 10
    max_concurrency: int = 1
    seed: int = 300
    agent_llm: str = "openai/gpt-4.1-mini"
    user_llm: str = "openai/gpt-4.1-mini"
    agent_llm_args: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.0})
    user_llm_args: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.0})
    process_timeout_s: int | None = None
    extra_env: dict[str, str] = field(default_factory=dict)


def write_base_agent(path: Path) -> Path:
    """Write the default banking_knowledge base agent module."""

    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DEFAULT_AGENT_SOURCE, path)
    return path


def write_base_agent_snapshot(snapshot_dir: Path) -> Path:
    """Write a candidate source snapshot containing agent and local runtime."""

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = snapshot_dir / "tau_agent_runtime"
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    shutil.copytree(DEFAULT_RUNTIME_SOURCE, runtime_dir)
    return write_base_agent(snapshot_dir / "banking_knowledge_base_agent.py")


def run_tau_banking_benchmark(
    *,
    out_dir: Path,
    config: TauBankingRunConfig | None = None,
    force: bool = False,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    pareto_quality_threshold: float = 0.0,
) -> dict[str, Any]:
    """Run tau3 banking_knowledge and write OptiHarness-compatible artifacts."""

    cfg = config or TauBankingRunConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    row_dir = out_dir / "rows"
    row_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = out_dir / "candidate_results"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    agent_module = cfg.agent_module or write_base_agent_snapshot(out_dir / "agent_snapshot")
    agent_snapshot_root = cfg.agent_snapshot_root or agent_module.parent
    runner_script = _write_runner_script(out_dir / "tau2_banking_runner.py")

    started = time.time()
    rows: list[dict[str, Any]] = []
    for retrieval_config in cfg.retrieval_configs:
        row_path = row_dir / f"{_safe_name(cfg.agent_name)}__{_safe_name(retrieval_config)}.json"
        if row_path.exists() and not force:
            rows.append(json.loads(row_path.read_text(encoding="utf-8")))
            continue

        tau_output_path = row_dir / f"{_safe_name(retrieval_config)}.tau2_summary.json"
        cmd = build_tau_banking_command(
            runner_script=runner_script,
            output_path=tau_output_path,
            config=cfg,
            agent_module=agent_module,
            agent_snapshot_root=agent_snapshot_root,
            retrieval_config=retrieval_config,
        )
        env = os.environ.copy()
        env.update(cfg.extra_env)
        completed = command_runner(
            cmd,
            cwd=str(cfg.tau2_root),
            env=env,
            text=True,
            capture_output=True,
            timeout=cfg.process_timeout_s,
        )
        if completed.returncode != 0:
            failure = {
                "benchmark": "tau_banking_knowledge",
                "agent": cfg.agent_name,
                "retrieval_config": retrieval_config,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "command": cmd,
            }
            row_path.write_text(json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8")
            raise RuntimeError(
                f"tau2 banking_knowledge run failed for retrieval_config={retrieval_config!r}; "
                f"details written to {row_path}"
            )

        tau_payload = json.loads(tau_output_path.read_text(encoding="utf-8"))
        row = summarize_tau_banking_payload(
            tau_payload,
            agent_name=cfg.agent_name,
            retrieval_config=retrieval_config,
            model=cfg.agent_llm,
        )
        row["command"] = cmd
        row["stdout_tail"] = (completed.stdout or "")[-4000:]
        row["stderr_tail"] = (completed.stderr or "")[-4000:]
        row_path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append(row)

    candidates = aggregate_tau_banking_candidates(rows, candidate_dir=candidate_dir)
    frontier_path = out_dir / "pareto_frontier.json"
    save_frontier(
        frontier_path,
        [
            ParetoPoint(
                candidate_id=item.candidate_id,
                scaffold_name=item.scaffold_name,
                passrate=item.passrate,
                token_consuming=item.token_consuming,
                avg_token_consuming=item.avg_token_consuming,
                average_score=item.average_score,
                result_path=item.result_path,
                config=item.config,
            )
            for item in candidates
        ],
        quality_gap_threshold=pareto_quality_threshold,
    )

    summary = {
        "benchmark": "tau_banking_knowledge",
        "out_dir": str(out_dir),
        "tau2_root": str(cfg.tau2_root),
        "agent_name": cfg.agent_name,
        "agent_module": str(agent_module),
        "agent_snapshot_root": str(agent_snapshot_root),
        "retrieval_configs": list(cfg.retrieval_configs),
        "task_split_name": cfg.task_split_name,
        "task_ids": list(cfg.task_ids),
        "num_tasks": cfg.num_tasks,
        "num_trials": cfg.num_trials,
        "model": cfg.agent_llm,
        "user_model": cfg.user_llm,
        "duration_s": time.time() - started,
        "row_count": len(rows),
        "candidate_count": len(candidates),
        "rows": rows,
        "candidates": [candidate.to_dict() for candidate in candidates],
        "pareto_frontier_path": str(frontier_path),
        "reference_urls": [
            TAU2_BENCH_REPO_URL,
            "https://taubench.com/blog/tau-knowledge.html",
            "https://arxiv.org/abs/2603.04370",
        ],
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def build_tau_banking_command(
    *,
    runner_script: Path,
    output_path: Path,
    config: TauBankingRunConfig,
    agent_module: Path,
    agent_snapshot_root: Path,
    retrieval_config: str,
) -> list[str]:
    """Build the external Python command that executes tau2-bench."""

    cmd = [
        *shlex.split(config.python_executable),
        str(runner_script),
        "--tau2-root",
        str(config.tau2_root),
        "--agent-module",
        str(agent_module),
        "--agent-snapshot-root",
        str(agent_snapshot_root),
        "--factory-name",
        config.factory_name,
        "--agent-name",
        config.agent_name,
        "--agent-llm",
        config.agent_llm,
        "--user-llm",
        config.user_llm,
        "--agent-llm-args",
        json.dumps(config.agent_llm_args),
        "--user-llm-args",
        json.dumps(config.user_llm_args),
        "--retrieval-config",
        retrieval_config,
        "--retrieval-config-kwargs",
        json.dumps(config.retrieval_config_kwargs),
        "--task-split-name",
        config.task_split_name,
        "--num-trials",
        str(config.num_trials),
        "--max-steps",
        str(config.max_steps),
        "--max-errors",
        str(config.max_errors),
        "--max-concurrency",
        str(config.max_concurrency),
        "--seed",
        str(config.seed),
        "--output",
        str(output_path),
    ]
    if config.num_tasks is not None:
        cmd.extend(["--num-tasks", str(config.num_tasks)])
    if config.task_ids:
        cmd.extend(["--task-ids", json.dumps(list(config.task_ids))])
    return cmd


def summarize_tau_banking_payload(
    payload: Mapping[str, Any],
    *,
    agent_name: str,
    retrieval_config: str,
    model: str,
) -> dict[str, Any]:
    """Summarize the small JSON emitted by the external tau2 runner."""

    simulations = list(payload.get("simulations") or [])
    rewards = [float(item.get("reward") or 0.0) for item in simulations]
    agent_costs = [float(item.get("agent_cost") or 0.0) for item in simulations]
    user_costs = [float(item.get("user_cost") or 0.0) for item in simulations]
    message_counts = [int(item.get("message_count") or 0) for item in simulations]
    total_agent_cost = sum(agent_costs)
    token_units = (
        int(round(total_agent_cost * 1_000_000))
        if total_agent_cost > 0
        else sum(message_counts)
    )

    total = len(simulations)
    passed = sum(1 for reward in rewards if reward >= 1.0)
    return {
        "benchmark": "tau_banking_knowledge",
        "agent": agent_name,
        "model": model,
        "retrieval_config": retrieval_config,
        "timestamp": datetime.now().isoformat(),
        "task_count": total,
        "passed": passed,
        "passrate": passed / total if total else 0.0,
        "average_reward": sum(rewards) / total if total else 0.0,
        "agent_cost": total_agent_cost,
        "user_cost": sum(user_costs),
        "cost_unit": "agent_cost_micro_usd" if total_agent_cost > 0 else "message_count",
        "token_consuming": token_units,
        "avg_token_consuming": token_units / total if total else 0.0,
        "simulations": simulations,
        "tau2_info": payload.get("info", {}),
    }


def aggregate_tau_banking_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_dir: Path,
) -> list[CandidateResult]:
    """Aggregate tau banking rows into OptiHarness CandidateResult files."""

    by_candidate: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_candidate[(str(row["agent"]), str(row["retrieval_config"]))].append(row)

    candidates: list[CandidateResult] = []
    for (agent_name, retrieval_config), grouped_rows in sorted(by_candidate.items()):
        counts = [int(row.get("task_count", 0) or 0) for row in grouped_rows]
        total_count = sum(counts)
        total_tokens = sum(int(row.get("token_consuming", 0) or 0) for row in grouped_rows)
        rewards = [float(row.get("average_reward", 0.0) or 0.0) for row in grouped_rows]
        passrates = [float(row.get("passrate", 0.0) or 0.0) for row in grouped_rows]
        candidate_id = f"tau_banking_{_safe_name(agent_name)}_{_safe_name(retrieval_config)}"
        result_path = candidate_dir / f"{candidate_id}.json"
        passrate = sum(passrates) / len(passrates) if passrates else 0.0
        average_score = sum(rewards) / len(rewards) if rewards else 0.0
        candidate = CandidateResult(
            candidate_id=candidate_id,
            scaffold_name=agent_name,
            passrate=passrate,
            average_score=average_score,
            token_consuming=total_tokens,
            avg_token_consuming=total_tokens / total_count if total_count else 0.0,
            avg_prompt_tokens=0.0,
            avg_completion_tokens=0.0,
            count=total_count,
            config={
                "benchmark": "tau_banking_knowledge",
                "agent": agent_name,
                "retrieval_config": retrieval_config,
                "cost_unit": grouped_rows[0].get("cost_unit", "unknown"),
                "rows": len(grouped_rows),
            },
            result_path=str(result_path),
        )
        result_path.write_text(
            json.dumps({"candidate": candidate.to_dict(), "rows": list(grouped_rows)}, indent=2),
            encoding="utf-8",
        )
        candidates.append(candidate)
    return candidates


def _write_runner_script(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_TAU2_RUNNER_SOURCE, encoding="utf-8")
    return path


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)
    return safe.strip("_") or "item"


_TAU2_RUNNER_SOURCE = r'''
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau2-root", required=True)
    parser.add_argument("--agent-module", required=True)
    parser.add_argument("--agent-snapshot-root", required=True)
    parser.add_argument("--factory-name", required=True)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--agent-llm", required=True)
    parser.add_argument("--user-llm", required=True)
    parser.add_argument("--agent-llm-args", default="{}")
    parser.add_argument("--user-llm-args", default="{}")
    parser.add_argument("--retrieval-config", required=True)
    parser.add_argument("--retrieval-config-kwargs", default="{}")
    parser.add_argument("--task-split-name", default="base")
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tau2_root = Path(args.tau2_root).resolve()
    sys.path.insert(0, str(tau2_root / "src"))
    sys.path.insert(0, str(Path(args.agent_snapshot_root).resolve()))

    module_path = Path(args.agent_module).resolve()
    spec = importlib.util.spec_from_file_location("memomemo_tau_agent", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, args.factory_name)

    from tau2.data_model.simulation import TextRunConfig
    from tau2.registry import registry
    from tau2.run import run_domain

    registry.register_agent_factory(factory, args.agent_name)
    task_ids = json.loads(args.task_ids) if args.task_ids else None
    config = TextRunConfig(
        domain="banking_knowledge",
        agent=args.agent_name,
        llm_agent=args.agent_llm,
        llm_args_agent=json.loads(args.agent_llm_args),
        llm_user=args.user_llm,
        llm_args_user=json.loads(args.user_llm_args),
        retrieval_config=args.retrieval_config,
        retrieval_config_kwargs=json.loads(args.retrieval_config_kwargs),
        task_split_name=args.task_split_name,
        task_ids=task_ids,
        num_tasks=args.num_tasks,
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        save_to=f"optiharness_{args.agent_name}_{args.retrieval_config}",
        log_level="ERROR",
    )
    results = run_domain(config)
    simulations = []
    for sim in results.simulations:
        reward_info = getattr(sim, "reward_info", None)
        messages = sim.get_messages() if hasattr(sim, "get_messages") else (sim.messages or [])
        simulations.append(
            {
                "id": getattr(sim, "id", None),
                "task_id": getattr(sim, "task_id", None),
                "trial": getattr(sim, "trial", None),
                "reward": getattr(reward_info, "reward", 0.0) if reward_info else 0.0,
                "agent_cost": getattr(sim, "agent_cost", 0.0) or 0.0,
                "user_cost": getattr(sim, "user_cost", 0.0) or 0.0,
                "duration": getattr(sim, "duration", None),
                "termination_reason": getattr(sim, "termination_reason", None),
                "message_count": len(messages),
            }
        )
    payload = {
        "info": results.info.model_dump(mode="json") if hasattr(results.info, "model_dump") else {},
        "simulations": simulations,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
