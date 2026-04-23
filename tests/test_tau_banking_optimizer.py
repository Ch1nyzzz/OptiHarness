from __future__ import annotations

import json
from types import SimpleNamespace

from memomemo.benchmark_workspaces import TAU3_BANKING_WORKSPACE_SPEC
from memomemo import tau_banking_optimizer as optimizer_module
from memomemo.tau_banking_optimizer import (
    TauBankingOptimizer,
    TauBankingOptimizerConfig,
)


def test_tau3_workspace_uses_declared_source_scope(tmp_path) -> None:
    optimizer = TauBankingOptimizer(
        TauBankingOptimizerConfig(
            run_id="tau",
            out_dir=tmp_path,
            proposer_sandbox="none",
        )
    )

    workspace = optimizer._build_workspace(
        iteration=1,
        call_dir=tmp_path / "proposer_calls" / "iter_001",
    )
    source_pkg = (
        workspace
        / "source_snapshot"
        / "candidate"
        / "project_source"
        / "src"
        / "memomemo"
    )
    manifest = json.loads(
        (workspace / "source_snapshot" / "manifest.json").read_text(encoding="utf-8")
    )

    copied = sorted(str(path.relative_to(source_pkg)) for path in source_pkg.rglob("*.py"))
    expected = sorted(TAU3_BANKING_WORKSPACE_SPEC.source_files)
    assert copied == expected
    assert sorted(manifest["source_files"]) == expected
    assert manifest["benchmark"] == TAU3_BANKING_WORKSPACE_SPEC.benchmark
    assert manifest["primary_source_file"] == TAU3_BANKING_WORKSPACE_SPEC.primary_source_file
    assert manifest["default_agent_module"].endswith(
        "tau_agents/banking_knowledge_base_agent.py"
    )


def test_tau3_optimizer_dispatches_one_candidate(tmp_path, monkeypatch) -> None:
    optimizer = TauBankingOptimizer(
        TauBankingOptimizerConfig(
            run_id="tau",
            out_dir=tmp_path,
            iterations=1,
            skip_baseline_eval=True,
            proposer_sandbox="none",
        )
    )
    captured = {}

    def fake_run_code_agent_prompt(prompt, **kwargs):
        captured["prompt"] = prompt
        workspace = kwargs["cwd"]
        (workspace / "pending_eval.json").write_text(
            json.dumps(
                {
                    "candidates": [
                        {
                            "name": "retrieval_disciplined",
                            "source_project_path": (
                                "source_snapshot/candidate/project_source"
                            ),
                            "agent_module": (
                                "source_snapshot/candidate/project_source/src/"
                                "memomemo/tau_agents/banking_knowledge_base_agent.py"
                            ),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            returncode=0,
            timed_out=False,
            stderr="",
            usage=None,
            metrics={},
            tool_access={},
        )

    def fake_evaluate(iteration, proposed):
        captured["iteration"] = iteration
        captured["proposed"] = proposed
        return []

    monkeypatch.setattr(
        optimizer_module,
        "run_code_agent_prompt",
        fake_run_code_agent_prompt,
    )
    monkeypatch.setattr(optimizer, "_evaluate_proposed", fake_evaluate)

    payload = optimizer.run()

    assert payload["task"] == "tau3_banking_knowledge"
    assert captured["iteration"] == 1
    assert captured["proposed"][0]["name"] == "retrieval_disciplined"
    assert captured["proposed"][0]["source_family"] == "tau3_banking_agent"
    assert "tau3 banking_knowledge base agent" in captured["prompt"]


def test_tau3_optimizer_accepts_top_level_candidate_list(tmp_path, monkeypatch) -> None:
    optimizer = TauBankingOptimizer(
        TauBankingOptimizerConfig(
            run_id="tau",
            out_dir=tmp_path,
            iterations=1,
            skip_baseline_eval=True,
            proposer_sandbox="none",
        )
    )
    captured = {}

    def fake_run_code_agent_prompt(prompt, **kwargs):
        (kwargs["cwd"] / "pending_eval.json").write_text(
            json.dumps(
                [
                    {
                        "name": "list_candidate",
                        "source_project_path": "source_snapshot/candidate/project_source",
                        "agent_module": (
                            "source_snapshot/candidate/project_source/src/"
                            "memomemo/tau_agents/banking_knowledge_base_agent.py"
                        ),
                    }
                ]
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            returncode=0,
            timed_out=False,
            stderr="",
            usage=None,
            metrics={},
            tool_access={},
        )

    def fake_evaluate(iteration, proposed):
        captured["proposed"] = proposed
        return []

    monkeypatch.setattr(
        optimizer_module,
        "run_code_agent_prompt",
        fake_run_code_agent_prompt,
    )
    monkeypatch.setattr(optimizer, "_evaluate_proposed", fake_evaluate)

    optimizer.run()

    assert captured["proposed"][0]["name"] == "list_candidate"
    pending = json.loads(optimizer.pending_eval_path.read_text(encoding="utf-8"))
    assert pending["candidates"][0]["name"] == "list_candidate"


def test_tau3_candidate_policy_rejects_benchmark_output_read(tmp_path) -> None:
    source_project = tmp_path / "snap" / "candidate" / "project_source"
    package = source_project / "src" / "memomemo" / "tau_agents"
    package.mkdir(parents=True)
    (package / "banking_knowledge_base_agent.py").write_text(
        "open('candidate_results/previous.json').read()\n",
        encoding="utf-8",
    )
    optimizer = TauBankingOptimizer(
        TauBankingOptimizerConfig(run_id="tau", out_dir=tmp_path)
    )

    violations = optimizer._candidate_policy_violations(
        {"source_project_path": str(source_project)}
    )

    assert any(item["marker"] == "candidate_results" for item in violations)
