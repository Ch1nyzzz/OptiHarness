from __future__ import annotations

import json
from types import SimpleNamespace

from memomemo.benchmark_workspaces import TEXT_CLASSIFICATION_WORKSPACE_SPEC
from memomemo import text_classification_optimizer as optimizer_module
from memomemo.text_classification_optimizer import (
    TextClassificationOptimizer,
    TextClassificationOptimizerConfig,
)


def test_text_classification_workspace_uses_declared_source_scope(tmp_path) -> None:
    optimizer = TextClassificationOptimizer(
        TextClassificationOptimizerConfig(
            run_id="text",
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
    manifest_path = workspace / "source_snapshot" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    copied = sorted(
        str(path.relative_to(source_pkg))
        for path in source_pkg.rglob("*.py")
        if path.relative_to(source_pkg).parts != ("__init__.py",)
    )
    expected = sorted(TEXT_CLASSIFICATION_WORKSPACE_SPEC.source_files)
    assert copied == expected
    assert sorted(manifest["source_files"]) == expected
    assert manifest["benchmark"] == TEXT_CLASSIFICATION_WORKSPACE_SPEC.benchmark
    assert (
        manifest["primary_source_file"]
        == TEXT_CLASSIFICATION_WORKSPACE_SPEC.primary_source_file
    )
    assert (source_pkg / "__init__.py").read_text(encoding="utf-8") == ""


def test_text_classification_optimizer_dispatches_one_candidate(tmp_path, monkeypatch) -> None:
    optimizer = TextClassificationOptimizer(
        TextClassificationOptimizerConfig(
            run_id="text",
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
        pending = workspace / "pending_eval.json"
        pending.write_text(
            json.dumps(
                {
                    "candidates": [
                        {
                            "name": "label_balanced",
                            "memory_system": "fewshot_all",
                            "extra": {
                                "source_project_path": (
                                    "source_snapshot/candidate/project_source"
                                )
                            },
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

    assert payload["task"] == "text_classification_offline"
    assert captured["iteration"] == 1
    assert captured["proposed"][0]["name"] == "label_balanced"
    assert captured["proposed"][0]["memory_system"] == "fewshot_all"
    assert captured["proposed"][0]["source_family"] == "text_classification_fewshot"
    assert "benchmark-scoped" in captured["prompt"]


def test_text_classification_proposer_retries_when_pending_eval_missing(
    tmp_path,
    monkeypatch,
) -> None:
    optimizer = TextClassificationOptimizer(
        TextClassificationOptimizerConfig(
            run_id="text",
            out_dir=tmp_path,
            iterations=1,
            skip_baseline_eval=True,
            proposer_sandbox="none",
        )
    )
    prompts = []
    captured = {}

    def fake_run_code_agent_prompt(prompt, **kwargs):
        prompts.append(prompt)
        if len(prompts) == 2:
            (kwargs["cwd"] / "pending_eval.json").write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "name": "retry_candidate",
                                "memory_system": "fewshot_all",
                                "extra": {
                                    "source_project_path": (
                                        "source_snapshot/candidate/project_source"
                                    )
                                },
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
        captured["proposed"] = proposed
        return []

    monkeypatch.setattr(
        optimizer_module,
        "run_code_agent_prompt",
        fake_run_code_agent_prompt,
    )
    monkeypatch.setattr(optimizer, "_evaluate_proposed", fake_evaluate)

    optimizer.run()

    assert len(prompts) == 2
    assert "Required Repair" in prompts[1]
    assert captured["proposed"][0]["name"] == "retry_candidate"
    assert "proposer_missing_pending_retry" in optimizer.summary_path.read_text(
        encoding="utf-8"
    )


def test_text_classification_optimizer_accepts_top_level_candidate_list(
    tmp_path,
    monkeypatch,
) -> None:
    optimizer = TextClassificationOptimizer(
        TextClassificationOptimizerConfig(
            run_id="text",
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
                        "memory_system": "fewshot_all",
                        "extra": {
                            "source_project_path": (
                                "source_snapshot/candidate/project_source"
                            )
                        },
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


def test_text_classification_progressive_policy_records_budget(
    tmp_path,
    monkeypatch,
) -> None:
    optimizer = TextClassificationOptimizer(
        TextClassificationOptimizerConfig(
            run_id="text",
            out_dir=tmp_path,
            iterations=1,
            skip_baseline_eval=True,
            selection_policy="progressive",
            proposer_sandbox="none",
        )
    )
    captured = {}

    def fake_run_code_agent_prompt(prompt, **kwargs):
        captured["prompt"] = prompt
        workspace = kwargs["cwd"]
        (workspace / "pending_eval.json").write_text(
            json.dumps({"candidates": []}),
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

    monkeypatch.setattr(
        optimizer_module,
        "run_code_agent_prompt",
        fake_run_code_agent_prompt,
    )

    payload = optimizer.run()
    state = json.loads((tmp_path / "progressive_state.json").read_text(encoding="utf-8"))

    assert payload["selection_policy"] == "progressive"
    assert "Selection policy: `progressive`" in captured["prompt"]
    assert "Context budget: `low`" in captured["prompt"]
    assert state["last_budget"] == "low"


def test_text_classification_candidate_policy_rejects_out_of_scope_import(tmp_path) -> None:
    source_project = tmp_path / "snap" / "candidate" / "project_source"
    package = source_project / "src" / "memomemo"
    package.mkdir(parents=True)
    (package / "text_classification.py").write_text(
        "from memomemo.optimizer import MemoOptimizer\n",
        encoding="utf-8",
    )
    optimizer = TextClassificationOptimizer(
        TextClassificationOptimizerConfig(run_id="text", out_dir=tmp_path)
    )

    violations = optimizer._candidate_policy_violations(
        {"extra": {"source_project_path": str(source_project)}}
    )

    assert any(item["marker"] == "memomemo.optimizer" for item in violations)


def test_text_classification_policy_allows_baseline_candidate_results_helper(
    tmp_path,
) -> None:
    optimizer = TextClassificationOptimizer(
        TextClassificationOptimizerConfig(run_id="text", out_dir=tmp_path)
    )
    workspace = optimizer._build_workspace(
        iteration=1,
        call_dir=tmp_path / "proposer_calls" / "iter_001",
    )
    source_project = (
        workspace
        / "source_snapshot"
        / "candidate"
        / "project_source"
    )

    violations = optimizer._candidate_policy_violations(
        {"extra": {"source_project_path": str(source_project)}}
    )

    assert not any(item["marker"] == "candidate_results" for item in violations)


def test_text_classification_policy_rejects_new_candidate_results_access(
    tmp_path,
) -> None:
    optimizer = TextClassificationOptimizer(
        TextClassificationOptimizerConfig(run_id="text", out_dir=tmp_path)
    )
    workspace = optimizer._build_workspace(
        iteration=1,
        call_dir=tmp_path / "proposer_calls" / "iter_001",
    )
    source_project = (
        workspace
        / "source_snapshot"
        / "candidate"
        / "project_source"
    )
    text_cls = source_project / "src" / "memomemo" / "text_classification.py"
    with text_cls.open("a", encoding="utf-8") as handle:
        handle.write("\nLEAK_PATH = 'candidate_results'\n")

    violations = optimizer._candidate_policy_violations(
        {"extra": {"source_project_path": str(source_project)}}
    )

    assert any(item["marker"] == "candidate_results" for item in violations)
