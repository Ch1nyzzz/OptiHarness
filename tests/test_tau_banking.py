from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from memomemo.tau_banking import (
    TauBankingRunConfig,
    build_tau_banking_command,
    run_tau_banking_benchmark,
    summarize_tau_banking_payload,
    write_base_agent,
    write_base_agent_snapshot,
)


def test_write_base_agent_exposes_tau2_factory(tmp_path: Path) -> None:
    agent_path = write_base_agent(tmp_path / "agent.py")

    text = agent_path.read_text(encoding="utf-8")
    assert "class BankingKnowledgeBaseAgent" in text
    assert "def create_banking_knowledge_base_agent" in text
    assert "from tau_agent_runtime.base_agent import HalfDuplexAgent" in text
    assert "Search the knowledge base before applying any product rule" in text


def test_write_base_agent_snapshot_includes_local_runtime(tmp_path: Path) -> None:
    agent_path = write_base_agent_snapshot(tmp_path / "snapshot")

    assert agent_path == tmp_path / "snapshot" / "banking_knowledge_base_agent.py"
    runtime_path = tmp_path / "snapshot" / "tau_agent_runtime" / "base_agent.py"
    assert runtime_path.exists()
    assert "class HalfDuplexAgent" in runtime_path.read_text(encoding="utf-8")


def test_summarize_tau_banking_payload_uses_reward_and_agent_cost() -> None:
    row = summarize_tau_banking_payload(
        {
            "simulations": [
                {"task_id": "1", "reward": 1.0, "agent_cost": 0.001, "message_count": 10},
                {"task_id": "2", "reward": 0.5, "agent_cost": 0.002, "message_count": 20},
                {"task_id": "3", "reward": 0.0, "agent_cost": 0.0, "message_count": 30},
            ]
        },
        agent_name="base",
        retrieval_config="bm25",
        model="model",
    )

    assert row["task_count"] == 3
    assert row["passed"] == 1
    assert row["passrate"] == 1 / 3
    assert row["average_reward"] == 0.5
    assert row["token_consuming"] == 3000
    assert row["cost_unit"] == "agent_cost_micro_usd"


def test_build_tau_banking_command_accepts_multiword_python_command(tmp_path: Path) -> None:
    command = build_tau_banking_command(
        runner_script=tmp_path / "runner.py",
        output_path=tmp_path / "out.json",
        config=TauBankingRunConfig(
            tau2_root=tmp_path / "tau2",
            python_executable="uv run python",
        ),
        agent_module=tmp_path / "agent.py",
        agent_snapshot_root=tmp_path,
        retrieval_config="bm25",
    )

    assert command[:3] == ["uv", "run", "python"]


def test_tau_banking_benchmark_writes_candidate_results_with_fake_runner(tmp_path: Path) -> None:
    tau2_root = tmp_path / "tau2"
    tau2_root.mkdir()
    calls: list[list[str]] = []

    def fake_runner(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        calls.append(cmd)
        output = Path(cmd[cmd.index("--output") + 1])
        output.write_text(
            json.dumps(
                {
                    "info": {"domain": "banking_knowledge"},
                    "simulations": [
                        {
                            "id": "a",
                            "task_id": "1",
                            "trial": 0,
                            "reward": 1.0,
                            "agent_cost": 0.001,
                            "user_cost": 0.003,
                            "message_count": 12,
                        },
                        {
                            "id": "b",
                            "task_id": "2",
                            "trial": 0,
                            "reward": 0.0,
                            "agent_cost": 0.002,
                            "user_cost": 0.004,
                            "message_count": 18,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    summary = run_tau_banking_benchmark(
        out_dir=tmp_path / "run",
        config=TauBankingRunConfig(
            tau2_root=tau2_root,
            python_executable="python3.12",
            retrieval_configs=("bm25",),
            num_tasks=2,
            agent_llm="openai/gpt-4.1-mini",
            user_llm="openai/gpt-4.1-mini",
        ),
        command_runner=fake_runner,
    )

    assert len(calls) == 1
    assert "--retrieval-config" in calls[0]
    assert summary["row_count"] == 1
    assert summary["candidate_count"] == 1
    candidate = summary["candidates"][0]
    assert candidate["passrate"] == 0.5
    assert candidate["average_score"] == 0.5
    assert candidate["token_consuming"] == 3000
    assert (tmp_path / "run" / "pareto_frontier.json").exists()
    candidate_path = (
        tmp_path / "run" / "candidate_results" / candidate["candidate_id"]
    ).with_suffix(".json")
    assert candidate_path.exists()
