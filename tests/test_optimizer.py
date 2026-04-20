import json
from types import SimpleNamespace

from memomemo import optimizer as optimizer_module
from memomemo.optimizer import MemoOptimizer, OptimizerConfig, _single_top_k
from memomemo.schemas import CandidateResult


def test_default_proposer_evaluates_only_one_candidate(tmp_path, monkeypatch):
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))

    def fake_run_claude_prompt(*args, **kwargs):
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

    monkeypatch.setattr(optimizer_module, "run_claude_prompt", fake_run_claude_prompt)
    monkeypatch.setattr(optimizer, "_evaluate_proposed", fake_evaluate)

    result = optimizer._run_default_proposer_iteration(1, examples=[])

    assert result == []
    assert captured["iteration"] == 1
    assert captured["proposed"] == [{"name": "first", "scaffold_name": "bm25"}]
    assert captured["examples"] == []
    snapshot = tmp_path / "generated" / "source_snapshots" / "iter_001" / "candidate"
    assert (snapshot / "project_source" / "src" / "memomemo" / "optimizer.py").exists()
    assert (snapshot / "upstream_source" / "mem0" / "mem0" / "memory" / "main.py").exists()
    assert (snapshot / "upstream_source" / "MemGPT" / "letta" / "schemas" / "memory.py").exists()
    assert (
        snapshot / "upstream_source" / "MemoryBank-SiliconFriend" / "memory_bank" / "summarize_memory.py"
    ).exists()
    assert "default_candidate_count_adjusted" in optimizer.summary_path.read_text(
        encoding="utf-8"
    )


def test_single_top_k_uses_first_value_from_list():
    assert _single_top_k([4, 8]) == (4, True)
    assert _single_top_k([8]) == (8, False)
    assert _single_top_k(12) == (12, False)


def test_ucb_default_arms_exclude_bm25_seed_family(tmp_path):
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))

    arms = optimizer._available_ucb_arms([])

    assert {arm.source_family for arm in arms} == {"mem0", "memgpt", "membank", "fusion"}


def test_ucb_first_three_iterations_restrict_to_low_budget(tmp_path, monkeypatch):
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))
    selected = []

    def fake_select_ucb_arm(state, *, available_arms, exploration_c):
        selected.append(list(available_arms))
        return available_arms[0]

    monkeypatch.setattr(optimizer_module, "select_ucb_arm", fake_select_ucb_arm)
    monkeypatch.setattr(optimizer, "_select_parent_for_arm", lambda candidates, arm: _candidate("p"))
    monkeypatch.setattr(optimizer, "_build_context_snapshot", lambda **kwargs: tmp_path / "context")
    monkeypatch.setattr(optimizer, "_build_source_snapshot_workspace", lambda **kwargs: tmp_path / "source")
    monkeypatch.setattr(
        optimizer_module,
        "run_claude_prompt",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, timed_out=False, stderr="", metrics={}),
    )
    monkeypatch.setattr(optimizer, "_capture_diff", lambda call_dir: None)
    monkeypatch.setattr(optimizer_module, "write_diff_digest", lambda call_dir: None)

    optimizer._run_ucb_proposer_iteration(3, [_candidate("seed")], examples=[])

    assert selected
    assert {arm.cost_level for arm in selected[0]} == {"low"}


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


def test_ucb_source_snapshot_uses_single_candidate_dir(tmp_path):
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))
    call_dir = tmp_path / "call"
    call_dir.mkdir()

    snapshot_root = optimizer._build_source_snapshot_workspace(
        iteration=3,
        source_family="mem0",
        parent=_candidate("mem0_parent"),
        call_dir=call_dir,
        cost_level="medium",
    )

    manifest = json.loads((snapshot_root / "manifest.json").read_text(encoding="utf-8"))
    assert (snapshot_root / "candidate" / "SNAPSHOT.md").exists()
    assert not (snapshot_root / "candidate_a").exists()
    assert not (snapshot_root / "candidate_b").exists()
    assert manifest["candidate_dir"] == str(snapshot_root / "candidate")
    assert "slots" not in manifest


def test_ucb_context_prunes_trace_slices_to_budget(tmp_path):
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))
    src = tmp_path / "proposer_calls" / "iter_001"
    for level in ("low", "medium", "high"):
        trace_dir = src / "trace_slices" / level
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / "candidate.json").write_text("{}", encoding="utf-8")

    low_dest = tmp_path / "low_bundle"
    optimizer._copy_iteration_bundle(src, low_dest, cost_level="low")
    assert (low_dest / "trace_slices" / "low" / "candidate.json").exists()
    assert not (low_dest / "trace_slices" / "medium").exists()
    assert not (low_dest / "trace_slices" / "high").exists()

    medium_dest = tmp_path / "medium_bundle"
    optimizer._copy_iteration_bundle(src, medium_dest, cost_level="medium")
    assert not (medium_dest / "trace_slices" / "low").exists()
    assert (medium_dest / "trace_slices" / "medium" / "candidate.json").exists()
    assert not (medium_dest / "trace_slices" / "high").exists()

    high_dest = tmp_path / "high_bundle"
    optimizer._copy_iteration_bundle(src, high_dest, cost_level="high")
    assert (high_dest / "trace_slices" / "low" / "candidate.json").exists()
    assert (high_dest / "trace_slices" / "medium" / "candidate.json").exists()
    assert (high_dest / "trace_slices" / "high" / "candidate.json").exists()


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
        selection_policy="ucb",
        extra={"arm": "mem0|low"},
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
        config={"extra": {"source_family": "mem0"}},
        result_path=f"{candidate_id}.json",
    )
