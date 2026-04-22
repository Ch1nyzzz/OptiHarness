import json
from types import SimpleNamespace

from memomemo import optimizer as optimizer_module
from memomemo.optimizer import MemoOptimizer, OptimizerConfig, _single_top_k
from memomemo.schemas import CandidateResult
import pytest


def test_default_proposer_evaluates_only_one_candidate(tmp_path, monkeypatch):
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))

    def fake_run_code_agent_prompt(*args, **kwargs):
        captured["prompt"] = args[0]
        optimizer.pending_eval_path.write_text(
            json.dumps(
                {
                    "candidates": [
                        {"name": "first", "scaffold_name": "bm25"},
                        {"name": "second", "scaffold_name": "bm25"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, timed_out=False, stderr="")

    captured = {}

    def fake_evaluate(iteration, proposed, examples):
        captured["iteration"] = iteration
        captured["proposed"] = proposed
        captured["examples"] = examples
        return []

    monkeypatch.setattr(optimizer_module, "run_code_agent_prompt", fake_run_code_agent_prompt)
    monkeypatch.setattr(optimizer, "_evaluate_proposed", fake_evaluate)

    result = optimizer._run_default_proposer_iteration(1, examples=[])

    assert result == []
    assert captured["iteration"] == 1
    assert captured["proposed"][0]["name"] == "first"
    assert captured["proposed"][0]["scaffold_name"] == "bm25"
    assert captured["proposed"][0]["budget"] == "high"
    assert captured["proposed"][0]["reference_iterations"] == []
    assert captured["examples"] == []
    assert "MemoMemo Proposer" in captured["prompt"]
    assert "Context budget" not in captured["prompt"]
    assert "Context scope" not in captured["prompt"]
    assert '"budget":' not in captured["prompt"]
    assert "Optimization Focus" not in captured["prompt"]
    assert "mechanism directions" not in captured["prompt"]
    snapshot = tmp_path / "proposer_calls" / "iter_001" / "source_snapshot" / "candidate"
    assert (snapshot / "project_source" / "src" / "memomemo" / "optimizer.py").exists()
    assert (snapshot / "upstream_source" / "MemGPT" / "letta" / "schemas" / "memory.py").exists()
    assert "candidate_count_adjusted" in optimizer.summary_path.read_text(
        encoding="utf-8"
    )


def test_optimizer_can_run_codex_proposer_agent(tmp_path, monkeypatch):
    optimizer = MemoOptimizer(
        OptimizerConfig(
            run_id="r",
            out_dir=tmp_path,
            proposer_agent="codex",
            codex_model="gpt-test-codex",
        )
    )
    captured = {}

    def fake_run_code_agent_prompt(prompt, **kwargs):
        captured.update(kwargs)
        optimizer.pending_eval_path.write_text(
            json.dumps({"candidates": [{"name": "first", "scaffold_name": "bm25"}]}),
            encoding="utf-8",
        )
        return SimpleNamespace(
            returncode=0,
            timed_out=False,
            stderr="",
            metrics={},
            usage=None,
            tool_access={},
        )

    monkeypatch.setattr(optimizer_module, "run_code_agent_prompt", fake_run_code_agent_prompt)
    monkeypatch.setattr(optimizer, "_evaluate_proposed", lambda iteration, proposed, examples: [])

    optimizer._run_default_proposer_iteration(1, examples=[])

    assert captured["agent"] == "codex"
    assert captured["model"] == "gpt-test-codex"
    assert captured["sandbox"].kind == "docker"


def test_optimizer_can_disable_default_docker_sandbox(tmp_path):
    optimizer = MemoOptimizer(
        OptimizerConfig(
            run_id="r",
            out_dir=tmp_path,
            proposer_sandbox="none",
        )
    )

    assert optimizer._proposer_sandbox_config() is None


def test_optimizer_can_run_kimi_proposer_agent(tmp_path, monkeypatch):
    optimizer = MemoOptimizer(
        OptimizerConfig(
            run_id="r",
            out_dir=tmp_path,
            proposer_agent="kimi",
            kimi_model="kimi-test",
        )
    )
    captured = {}

    def fake_run_code_agent_prompt(prompt, **kwargs):
        captured.update(kwargs)
        optimizer.pending_eval_path.write_text(
            json.dumps({"candidates": [{"name": "first", "scaffold_name": "bm25"}]}),
            encoding="utf-8",
        )
        return SimpleNamespace(
            returncode=0,
            timed_out=False,
            stderr="",
            metrics={},
            usage=None,
            tool_access={},
        )

    monkeypatch.setattr(optimizer_module, "run_code_agent_prompt", fake_run_code_agent_prompt)
    monkeypatch.setattr(optimizer, "_evaluate_proposed", lambda iteration, proposed, examples: [])

    optimizer._run_default_proposer_iteration(1, examples=[])

    assert captured["agent"] == "kimi"
    assert captured["model"] == "kimi-test"


def test_optimizer_rejects_baseline_with_mismatched_count(tmp_path, monkeypatch):
    optimizer = MemoOptimizer(
        OptimizerConfig(
            run_id="r",
            out_dir=tmp_path,
            iterations=0,
            limit=0,
            baseline_dir=tmp_path / "baseline",
            scaffolds=("memgpt_source",),
        )
    )
    baseline = CandidateResult(
        candidate_id="memgpt_source_top12",
        scaffold_name="memgpt_source",
        passrate=0.4,
        average_score=0.5,
        token_consuming=100,
        avg_token_consuming=10,
        avg_prompt_tokens=8,
        avg_completion_tokens=2,
        count=1,
        config={"top_k": 12, "extra": {"source_family": "memgpt"}},
        result_path="baseline.json",
    )

    monkeypatch.setattr(optimizer, "_load_examples", lambda: [object(), object()])
    monkeypatch.setattr(
        optimizer_module,
        "load_baseline_candidates",
        lambda *args, **kwargs: [baseline.to_dict()],
    )

    with pytest.raises(ValueError, match="Baseline candidate count does not match"):
        optimizer.run()


def test_run_writes_initial_trace_slices(tmp_path, monkeypatch):
    optimizer = MemoOptimizer(
        OptimizerConfig(
            run_id="r",
            out_dir=tmp_path,
            iterations=0,
            selection_policy="progressive",
        )
    )
    result_path = tmp_path / "candidate_results" / "seed.json"
    candidate = CandidateResult(
        candidate_id="seed",
        scaffold_name="memgpt_source",
        passrate=0.0,
        average_score=0.0,
        token_consuming=10,
        avg_token_consuming=5,
        avg_prompt_tokens=4,
        avg_completion_tokens=1,
        count=1,
        config={"top_k": 8, "extra": {"source_family": "memgpt"}},
        result_path=str(result_path),
    )

    def fake_run_initial_frontier(*args, **kwargs):
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "candidate": candidate.to_dict(),
                    "tasks": [
                        {
                            "task_id": "hard-case",
                            "question": "question",
                            "gold_answer": "gold",
                            "prediction": "pred",
                            "score": 0.0,
                            "passed": False,
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "retrieved": [],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return {"candidates": [candidate.to_dict()]}

    monkeypatch.setattr(optimizer, "_load_examples", lambda: [object()])
    monkeypatch.setattr(optimizer_module, "run_initial_frontier", fake_run_initial_frontier)

    optimizer.run()

    low = json.loads((tmp_path / "trace_slices" / "low" / "seed.json").read_text())
    assert low["case_limit"] == 10
    assert low["cases"][0]["task_id"] == "hard-case"


def test_candidate_code_policy_rejects_runtime_trace_and_scorer_access(tmp_path):
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "bad_candidate.py").write_text(
        "\n".join(
            [
                "from memomemo.metrics import score_prediction",
                "",
                "def leak():",
                "    return open('runs/r/candidate_results/iter001.json').read()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))

    violations = optimizer._candidate_code_policy_violations(
        {
            "module": "bad_candidate",
            "class": "BadCandidate",
            "candidate_root": str(generated),
        }
    )

    markers = {item["marker"] for item in violations}
    assert "candidate_results" in markers
    assert "score_prediction" in markers


def test_candidate_code_policy_rejects_raw_locomo_access_in_source_snapshot(tmp_path):
    project_source = tmp_path / "snap" / "candidate" / "project_source"
    scaffold_dir = project_source / "src" / "memomemo" / "scaffolds"
    scaffold_dir.mkdir(parents=True)
    (scaffold_dir / "memgpt_scaffold.py").write_text(
        "from pathlib import Path\nPath('data/locomo/locomo10.json').read_text()\n",
        encoding="utf-8",
    )
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))

    violations = optimizer._candidate_code_policy_violations(
        {
            "scaffold_name": "memgpt_source",
            "extra": {"source_project_path": str(project_source)},
        }
    )

    assert any(item["marker"] == "data/locomo" for item in violations)


def test_source_candidate_policy_allows_preexisting_scorer_import(tmp_path):
    candidate_source = tmp_path / "snap" / "candidate" / "project_source"
    original_source = tmp_path / "snap" / "candidate" / "original_project_source"
    for root in (candidate_source, original_source):
        scaffold_dir = root / "src" / "memomemo" / "scaffolds"
        scaffold_dir.mkdir(parents=True)
        (scaffold_dir / "base.py").write_text(
            "from memomemo.metrics import retrieval_oracle_prediction\n",
            encoding="utf-8",
        )
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))

    violations = optimizer._candidate_code_policy_violations(
        {
            "scaffold_name": "memgpt_source",
            "extra": {"source_project_path": str(candidate_source)},
        }
    )

    assert violations == []


def test_source_candidate_policy_rejects_new_scorer_import(tmp_path):
    candidate_source = tmp_path / "snap" / "candidate" / "project_source"
    original_source = tmp_path / "snap" / "candidate" / "original_project_source"
    for root in (candidate_source, original_source):
        scaffold_dir = root / "src" / "memomemo" / "scaffolds"
        scaffold_dir.mkdir(parents=True)
        (scaffold_dir / "base.py").write_text(
            "class BaseMemoryScaffold:\n    pass\n",
            encoding="utf-8",
        )
    (candidate_source / "src" / "memomemo" / "scaffolds" / "memgpt_scaffold.py").write_text(
        "from memomemo.metrics import retrieval_oracle_prediction\n",
        encoding="utf-8",
    )
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))

    violations = optimizer._candidate_code_policy_violations(
        {
            "scaffold_name": "memgpt_source",
            "extra": {"source_project_path": str(candidate_source)},
        }
    )

    assert any(item["marker"] == "memomemo.metrics" for item in violations)


def test_single_top_k_uses_first_value_from_list():
    assert _single_top_k([4, 8]) == (4, True)
    assert _single_top_k([8]) == (8, False)
    assert _single_top_k(12) == (12, False)


def test_optimizer_copies_full_source_context(tmp_path):
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))
    dest = tmp_path / "context"

    optimizer._copy_project_source_context(dest)
    optimizer._copy_upstream_source_context("fusion", dest)

    assert (dest / "project_source" / "src" / "memomemo" / "optimizer.py").exists()
    assert (dest / "project_source" / "src" / "memomemo" / "scaffolds" / "mem0_scaffold.py").exists()
    assert (dest / "upstream_source" / "mem0" / "mem0" / "memory" / "main.py").exists()
    assert (dest / "upstream_source" / "MemGPT" / "letta" / "schemas" / "memory.py").exists()
    assert (dest / "upstream_source" / "MemoryBank-SiliconFriend" / "memory_bank" / "summarize_memory.py").exists()


def test_source_snapshot_uses_single_candidate_dir(tmp_path):
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))
    call_dir = tmp_path / "call"
    call_dir.mkdir()

    snapshot_root = optimizer._build_source_snapshot_workspace(
        iteration=3,
        source_family="mem0",
        call_dir=call_dir,
    )

    manifest = json.loads((snapshot_root / "manifest.json").read_text(encoding="utf-8"))
    assert (snapshot_root / "candidate" / "SNAPSHOT.md").exists()
    assert not (snapshot_root / "candidate_a").exists()
    assert not (snapshot_root / "candidate_b").exists()
    assert manifest["candidate_dir"] == str(snapshot_root / "candidate")
    assert "trace_scope" not in manifest
    assert "optimization_cell" not in manifest
    assert "cost_level" not in manifest
    assert "slots" not in manifest


def test_reference_bundle_prunes_trace_slices_to_budget_access(tmp_path):
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))
    src = tmp_path / "proposer_calls" / "iter_001"
    for level in ("low", "medium", "high"):
        trace_dir = src / "trace_slices" / level
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / "candidate.json").write_text("{}", encoding="utf-8")

    low_dest = tmp_path / "low_bundle"
    optimizer._copy_iteration_bundle(src, low_dest, trace_scope="last1")
    assert (low_dest / "trace_slices" / "low" / "candidate.json").exists()
    assert not (low_dest / "trace_slices" / "medium").exists()
    assert not (low_dest / "trace_slices" / "high").exists()

    medium_dest = tmp_path / "medium_bundle"
    optimizer._copy_iteration_bundle(src, medium_dest, trace_scope="last3")
    assert not (medium_dest / "trace_slices" / "low").exists()
    assert (medium_dest / "trace_slices" / "medium" / "candidate.json").exists()
    assert not (medium_dest / "trace_slices" / "high").exists()

    high_dest = tmp_path / "high_bundle"
    optimizer._copy_iteration_bundle(src, high_dest, trace_scope="all")
    assert (high_dest / "trace_slices" / "low" / "candidate.json").exists()
    assert (high_dest / "trace_slices" / "medium" / "candidate.json").exists()
    assert (high_dest / "trace_slices" / "high" / "candidate.json").exists()


def test_append_summary_writes_only_global_summary(tmp_path):
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))
    candidate = _candidate("memgpt_source_top8")

    optimizer._append_summary(iteration=0, candidate=candidate)

    summary = tmp_path / "evolution_summary.jsonl"
    assert summary.exists()
    assert "memgpt_source_top8" in summary.read_text(encoding="utf-8")
    assert not (tmp_path / "cells").exists()


def test_progressive_workspace_outputs_sync_and_normalize_candidate_paths(
    tmp_path, monkeypatch
):
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))
    call_dir = tmp_path / "proposer_calls" / "iter_004"
    workspace = call_dir / "workspace"
    captured = {}

    monkeypatch.setattr(
        optimizer,
        "_build_progressive_workspace",
        lambda **kwargs: (workspace, (1,)),
    )

    def fake_run_code_agent_prompt(prompt, **kwargs):
        assert kwargs["cwd"] == workspace
        (workspace / "generated").mkdir(parents=True)
        (workspace / "generated" / "progressive_candidate.py").write_text(
            "class ProgressiveCandidate: pass\n",
            encoding="utf-8",
        )
        (workspace / "pending_eval.json").write_text(
            json.dumps(
                {
                    "candidates": [
                        {
                            "name": "progressive_candidate",
                            "module": "progressive_candidate",
                            "class": "ProgressiveCandidate",
                            "extra": {
                                "source_project_path": "source_snapshot/candidate/project_source",
                                "source_base_dir": "source_base_variants/progressive_candidate",
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
            metrics={},
            usage=None,
            tool_access={
                "files_read": {},
                "files_written": {
                    "generated/progressive_candidate.py": {"writes": 1},
                    "pending_eval.json": {"writes": 1},
                },
            },
        )

    def fake_evaluate(iteration, proposed, examples):
        captured["proposed"] = proposed
        return []

    monkeypatch.setattr(optimizer_module, "run_code_agent_prompt", fake_run_code_agent_prompt)
    monkeypatch.setattr(optimizer, "_evaluate_proposed", fake_evaluate)

    optimizer._run_progressive_proposer_iteration(
        4,
        [_candidate("seed")],
        examples=[],
        budget="low",
        adaptive=True,
    )

    proposed = captured["proposed"]
    assert (tmp_path / "generated" / "progressive_candidate.py").exists()
    assert proposed[0]["candidate_root"] == str(tmp_path / "generated")
    assert "optimization_target" not in proposed[0]
    assert "bandit_arm" not in proposed[0]
    assert proposed[0]["extra"]["source_project_path"] == str(
        (call_dir / "source_snapshot" / "candidate" / "project_source").resolve()
    )
    assert proposed[0]["extra"]["source_base_dir"] == str(
        (workspace / "source_base_variants" / "progressive_candidate").resolve()
    )
    pending = json.loads((tmp_path / "pending_eval.json").read_text(encoding="utf-8"))
    assert pending["candidates"][0]["candidate_root"] == str(tmp_path / "generated")
    assert "bandit_arm" not in pending["candidates"][0]


def test_progressive_docker_sandbox_maps_container_workspace_paths(tmp_path, monkeypatch):
    optimizer = MemoOptimizer(
        OptimizerConfig(
            run_id="r",
            out_dir=tmp_path,
            selection_policy="progressive",
            proposer_sandbox="docker",
            proposer_docker_image="memo-proposer:test",
            proposer_docker_workspace="/workspace",
        )
    )
    call_dir = tmp_path / "proposer_calls" / "iter_004"
    workspace = call_dir / "workspace"
    captured = {}

    monkeypatch.setattr(
        optimizer,
        "_build_progressive_workspace",
        lambda **kwargs: (workspace, (1,)),
    )

    def fake_run_code_agent_prompt(prompt, **kwargs):
        captured["sandbox"] = kwargs["sandbox"]
        assert kwargs["cwd"] == workspace
        (workspace / "generated").mkdir(parents=True)
        (workspace / "source_snapshot" / "candidate" / "project_source" / "src").mkdir(
            parents=True
        )
        (workspace / "pending_eval.json").write_text(
            json.dumps(
                {
                    "candidates": [
                        {
                            "name": "progressive_source_candidate",
                            "scaffold_name": "memgpt_source",
                            "extra": {
                                "source_project_path": (
                                    "/workspace/source_snapshot/candidate/project_source"
                                ),
                                "source_base_dir": "/workspace/source_base_variants/build",
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
            metrics={},
            usage=None,
            tool_access={
                "files_read": {"/workspace/summaries/best_candidates.json": {"reads": 1}},
                "files_written": {"/workspace/pending_eval.json": {"writes": 1}},
            },
        )

    def fake_evaluate(iteration, proposed, examples):
        captured["proposed"] = proposed
        return []

    monkeypatch.setattr(optimizer_module, "run_code_agent_prompt", fake_run_code_agent_prompt)
    monkeypatch.setattr(optimizer, "_evaluate_proposed", fake_evaluate)

    optimizer._run_progressive_proposer_iteration(
        4,
        [_candidate("seed")],
        examples=[],
        budget="low",
        adaptive=True,
    )

    assert captured["sandbox"].kind == "docker"
    assert captured["sandbox"].docker_image == "memo-proposer:test"
    assert captured["proposed"][0]["extra"]["source_project_path"] == str(
        (call_dir / "source_snapshot" / "candidate" / "project_source").resolve()
    )
    assert captured["proposed"][0]["extra"]["source_base_dir"] == str(
        (workspace / "source_base_variants" / "build").resolve()
    )
    summary = (tmp_path / "evolution_summary.jsonl").read_text(encoding="utf-8")
    assert "proposer_access_rejected" not in summary


def test_progressive_access_violation_retries_with_boundary_feedback(tmp_path, monkeypatch):
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))
    workspace = tmp_path / "proposer_calls" / "iter_004" / "workspace"
    prompts = []
    build_calls = []
    captured = {}

    def fake_build_workspace(**kwargs):
        build_calls.append(kwargs)
        if workspace.exists():
            import shutil

            shutil.rmtree(workspace)
        (workspace / "generated").mkdir(parents=True)
        return workspace, (1,)

    monkeypatch.setattr(optimizer, "_build_progressive_workspace", fake_build_workspace)

    def fake_run_code_agent_prompt(prompt, **kwargs):
        prompts.append(prompt)
        assert kwargs["cwd"] == workspace
        if len(prompts) == 1:
            return SimpleNamespace(
                returncode=0,
                timed_out=False,
                stderr="",
                metrics={},
                usage=None,
                tool_access={"files_read": {str(tmp_path / "generated" / "leak.py"): {"reads": 1}}},
            )
        (workspace / "generated" / "progressive_retry_candidate.py").write_text(
            "class ProgressiveRetryCandidate: pass\n",
            encoding="utf-8",
        )
        (workspace / "pending_eval.json").write_text(
            json.dumps(
                {
                    "candidates": [
                        {
                            "name": "progressive_retry_candidate",
                            "module": "progressive_retry_candidate",
                            "class": "ProgressiveRetryCandidate",
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
            metrics={},
            usage=None,
            tool_access={
                "files_read": {"summaries/best_candidates.json": {"reads": 1}},
                "files_written": {
                    "generated/progressive_retry_candidate.py": {"writes": 1},
                    "pending_eval.json": {"writes": 1},
                },
            },
        )

    def fake_evaluate(iteration, proposed, examples):
        captured["proposed"] = proposed
        return []

    monkeypatch.setattr(optimizer_module, "run_code_agent_prompt", fake_run_code_agent_prompt)
    monkeypatch.setattr(optimizer, "_evaluate_proposed", fake_evaluate)

    optimizer._run_progressive_proposer_iteration(
        4,
        [_candidate("seed")],
        examples=[],
        budget="low",
        adaptive=True,
    )

    assert len(build_calls) == 2
    assert len(prompts) == 2
    assert "Retry Required: Filesystem Boundary Violation" in prompts[1]
    assert str(tmp_path / "generated" / "leak.py") in prompts[1]
    assert captured["proposed"][0]["name"] == "progressive_retry_candidate"
    summary = (tmp_path / "evolution_summary.jsonl").read_text(encoding="utf-8")
    assert "proposer_access_retry" in summary
    assert "proposer_access_rejected" not in summary


def test_progressive_budget_schedule_transitions(tmp_path):
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))

    assert [optimizer._progressive_budget_for_iteration(item) for item in range(1, 6)] == [
        "low",
        "low",
        "low",
        "low",
        "low",
    ]

    seed = _scored_candidate("seed", passrate=0.2)
    same = _scored_candidate("iter005_same_top8", passrate=0.2)
    optimizer._update_progressive_state(
        iteration=5,
        budget="low",
        previous_best_passrate=0.2,
        candidates=[seed, same],
        evaluated=[same],
    )
    assert optimizer._progressive_budget_for_iteration(6) == "medium"

    medium_same = _scored_candidate("iter006_same_top8", passrate=0.2)
    optimizer._update_progressive_state(
        iteration=6,
        budget="medium",
        previous_best_passrate=0.2,
        candidates=[seed, same, medium_same],
        evaluated=[medium_same],
    )
    assert optimizer._progressive_budget_for_iteration(7) == "high"

    improved = _scored_candidate("iter007_better_top8", passrate=0.3)
    optimizer._update_progressive_state(
        iteration=7,
        budget="high",
        previous_best_passrate=0.2,
        candidates=[seed, same, medium_same, improved],
        evaluated=[improved],
    )
    assert optimizer._progressive_budget_for_iteration(8) == "low"


def test_progressive_reference_selection_uses_best_and_worst_iterations(tmp_path):
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))
    for iteration in (1, 2, 3, 4):
        call_dir = tmp_path / "proposer_calls" / f"iter_{iteration:03d}"
        call_dir.mkdir(parents=True)
        (call_dir / "assignment.json").write_text("{}", encoding="utf-8")

    candidates = [
        _scored_candidate("iter001_mid_top8", passrate=0.3, average_score=0.4, tokens=200),
        _scored_candidate("iter002_best_top8", passrate=0.7, average_score=0.5, tokens=300),
        _scored_candidate("iter003_worst_top8", passrate=0.1, average_score=0.2, tokens=100),
        _scored_candidate("iter004_second_top8", passrate=0.6, average_score=0.6, tokens=150),
    ]

    assert optimizer._reference_iterations_for_budget(
        "low",
        iteration=5,
        candidates=candidates,
    ) == (2, 3)
    assert optimizer._reference_iterations_for_budget(
        "medium",
        iteration=5,
        candidates=candidates,
    ) == (2, 4, 1, 3)
    assert optimizer._reference_iterations_for_budget(
        "high",
        iteration=5,
        candidates=candidates,
    ) == (1, 2, 3, 4)


def test_progressive_workspace_copies_full_summaries_and_selected_raw_refs(tmp_path):
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))
    optimizer.summary_path.write_text(
        "\n".join(
            json.dumps({"iteration": item, "candidate": {"candidate_id": f"iter{item:03d}_x"}})
            for item in range(1, 5)
        )
        + "\n",
        encoding="utf-8",
    )
    optimizer.frontier_path.write_text("[]\n", encoding="utf-8")
    optimizer.candidate_score_table_path.write_text("[]\n", encoding="utf-8")
    optimizer.retrieval_diagnostics_summary_path.write_text("[]\n", encoding="utf-8")
    optimizer.iteration_index_path.write_text("[]\n", encoding="utf-8")
    optimizer.diff_summary_path.write_text(
        json.dumps({"iteration": 4, "files_changed": ["x.py"]}) + "\n",
        encoding="utf-8",
    )
    for iteration in (1, 2):
        call_dir = tmp_path / "proposer_calls" / f"iter_{iteration:03d}"
        (call_dir / "eval").mkdir(parents=True)
        (call_dir / "eval" / "candidate_result.compact.json").write_text(
            "{}",
            encoding="utf-8",
        )
        (call_dir / "trace_slices" / "low").mkdir(parents=True)
        (call_dir / "trace_slices" / "low" / "candidate.json").write_text(
            "{}",
            encoding="utf-8",
        )

    candidates = [
        _scored_candidate("iter001_best_top8", passrate=0.8),
        _scored_candidate("iter002_worst_top8", passrate=0.1),
    ]
    workspace, refs = optimizer._build_progressive_workspace(
        iteration=3,
        budget="low",
        existing_candidates=candidates,
        call_dir=tmp_path / "proposer_calls" / "iter_003",
    )

    assert refs == (1, 2)
    summary_lines = (workspace / "summaries" / "evolution_summary.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(summary_lines) == 4
    assert (workspace / "summaries" / "diff_summary.jsonl").exists()
    assert (
        workspace
        / "reference_iterations"
        / "iter_001"
        / "eval"
        / "candidate_result.compact.json"
    ).exists()
    assert (
        workspace
        / "source_snapshot"
        / "candidate"
        / "original_project_source"
        / "src"
        / "memomemo"
        / "optimizer.py"
    ).exists()
    assert (workspace / "workspace_manifest.json").exists()
    assert (workspace / "access_policy.json").exists()


def test_optimizer_records_and_aggregates_proposer_metrics(tmp_path):
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))
    result = SimpleNamespace(
        returncode=0,
        timed_out=False,
        usage={"total_cost_usd": 0.25},
        metrics={
            "input_tokens": 100,
            "output_tokens": 25,
            "total_tokens": 125,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 15,
            "total_reported_tokens": 150,
            "estimated_cost_usd": 0.25,
            "duration_s": 3.5,
            "tool_calls": 4,
            "tool_counts": {"Read": 2, "Write": 1, "Bash": 1},
            "read_file_calls": 2,
            "unique_files_read": 1,
            "read_lines": 42,
            "write_file_calls": 1,
            "written_lines": 7,
        },
        tool_access={
            "files_read": {"src/memomemo/optimizer.py": {"reads": 2, "lines": 42}},
            "files_written": {"runs/r/generated/candidate.py": {"writes": 1, "lines_written": 7}},
            "grep_requests": [],
            "tool_counts": {"Read": 2, "Write": 1, "Bash": 1},
        },
    )

    optimizer._append_proposer_result_event(
        iteration=1,
        result=result,
        selection_policy="progressive",
        extra={"budget": "low"},
    )

    row = json.loads(optimizer.summary_path.read_text(encoding="utf-8").strip())
    aggregate = optimizer._aggregate_proposer_metrics()

    assert row["event"] == "proposer_result"
    assert row["files_read"] == {"src/memomemo/optimizer.py": {"reads": 2, "lines": 42}}
    assert aggregate["calls"] == 1
    assert aggregate["estimated_cost_usd"] == 0.25
    assert aggregate["input_tokens"] == 100
    assert aggregate["tool_counts"] == {"Bash": 1, "Read": 2, "Write": 1}
    assert aggregate["read_file_calls"] == 2
    assert aggregate["unique_files_read"] == 1
    assert aggregate["written_lines"] == 7


def _candidate(candidate_id: str) -> CandidateResult:
    return CandidateResult(
        candidate_id=candidate_id,
        scaffold_name=candidate_id,
        passrate=0.1,
        average_score=0.2,
        token_consuming=100,
        avg_token_consuming=10,
        avg_prompt_tokens=8,
        avg_completion_tokens=2,
        count=10,
        config={"extra": {"source_family": "memgpt"}},
        result_path=f"{candidate_id}.json",
    )


def _scored_candidate(
    candidate_id: str,
    *,
    passrate: float,
    average_score: float = 0.2,
    tokens: int = 100,
) -> CandidateResult:
    return CandidateResult(
        candidate_id=candidate_id,
        scaffold_name="memgpt_source",
        passrate=passrate,
        average_score=average_score,
        token_consuming=tokens,
        avg_token_consuming=float(tokens),
        avg_prompt_tokens=8,
        avg_completion_tokens=2,
        count=10,
        config={"extra": {"source_family": "memgpt"}},
        result_path=f"{candidate_id}.json",
    )
