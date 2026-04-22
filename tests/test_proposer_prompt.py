from pathlib import Path

from memomemo.proposer_prompt import build_progressive_proposer_prompt


def test_progressive_prompt_uses_workspace_summaries_and_reference_iterations():
    prompt = build_progressive_proposer_prompt(
        run_id="r",
        iteration=6,
        run_dir=Path("runs/r/proposer_calls/iter_006/workspace"),
        pending_eval_path=Path("runs/r/proposer_calls/iter_006/workspace/pending_eval.json"),
        summaries_dir=Path("runs/r/proposer_calls/iter_006/workspace/summaries"),
        reference_iterations_dir=Path(
            "runs/r/proposer_calls/iter_006/workspace/reference_iterations"
        ),
        generated_dir=Path("runs/r/proposer_calls/iter_006/workspace/generated"),
        source_snapshot_dir=Path("runs/r/proposer_calls/iter_006/workspace/source_snapshot"),
        budget="low",
        reference_iterations=(2, 3),
        target_system="memgpt",
        optimization_directions=("retrieval_policy: Improve evidence ranking.",),
        split="train",
        limit=0,
    )

    assert "summaries/evolution_summary.jsonl" in prompt
    assert "summaries/best_candidates.json" in prompt
    assert "summaries/candidate_score_table.json" in prompt
    assert "summaries/retrieval_diagnostics_summary.json" in prompt
    assert "summaries/diff_summary.jsonl" in prompt
    assert "MemoMemo Proposer" in prompt
    assert "Context budget" not in prompt
    assert "Context scope" not in prompt
    assert '"budget":' not in prompt
    assert "Optimization Focus" in prompt
    assert "mechanism directions" in prompt
    assert "retrieval_policy: Improve evidence ranking." in prompt
    assert "reference_iterations/" in prompt
    assert "iter_002, iter_003" in prompt
    assert "clean source snapshot" in prompt
    assert "diagnostic\nreferences only" in prompt
    assert "source parent" in prompt
    assert "UCB" not in prompt
    assert "bandit" not in prompt.lower()
    assert "parent_candidate_id" not in prompt
    assert '"reference_iterations": [2, 3]' in prompt
    assert "`candidate_results/**`" in prompt
    assert "build/database-construction logic" in prompt
    assert "amem_source_path" not in prompt
    assert "mem0_source_path" in prompt
    assert "fresh `source_base_dir`" in prompt
    assert "source bases" in prompt
    assert "expensive" in prompt


def test_progressive_prompt_requires_mechanism_changes_not_parameter_only():
    prompt = build_progressive_proposer_prompt(
        run_id="r",
        iteration=7,
        run_dir=Path("runs/r/proposer_calls/iter_007/workspace"),
        pending_eval_path=Path("runs/r/proposer_calls/iter_007/workspace/pending_eval.json"),
        summaries_dir=Path("runs/r/proposer_calls/iter_007/workspace/summaries"),
        reference_iterations_dir=Path(
            "runs/r/proposer_calls/iter_007/workspace/reference_iterations"
        ),
        generated_dir=Path("runs/r/proposer_calls/iter_007/workspace/generated"),
        source_snapshot_dir=Path("runs/r/proposer_calls/iter_007/workspace/source_snapshot"),
        budget="medium",
        reference_iterations=(1, 4, 5),
        target_system="memgpt",
        optimization_directions=(),
        split="train",
        limit=0,
    )

    assert "Parameter changes are allowed only as supporting details" in prompt
    assert "substantive change is only `top_k`, window size, thresholds" in prompt
    assert "Do not reduce recall\nsolely to save tokens" in prompt
    assert "Use gold answers only to classify failure\nmodes" in prompt
    assert "All copied project source under" in prompt
    assert "scaffolds, base classes, model/prompt helpers" in prompt
    assert "exactly one candidate" in prompt
    assert "top_k" in prompt
    assert '"top_k": [4, 8]' not in prompt


def test_default_prompt_uses_neutral_context_description():
    prompt = build_progressive_proposer_prompt(
        run_id="r",
        iteration=3,
        run_dir=Path("runs/r/proposer_calls/iter_003/workspace"),
        pending_eval_path=Path("runs/r/proposer_calls/iter_003/workspace/pending_eval.json"),
        summaries_dir=Path("runs/r/proposer_calls/iter_003/workspace/summaries"),
        reference_iterations_dir=Path(
            "runs/r/proposer_calls/iter_003/workspace/reference_iterations"
        ),
        generated_dir=Path("runs/r/proposer_calls/iter_003/workspace/generated"),
        source_snapshot_dir=Path("runs/r/proposer_calls/iter_003/workspace/source_snapshot"),
        budget="high",
        reference_iterations=(1, 2),
        target_system="memgpt",
        optimization_directions=(),
        split="train",
        limit=0,
        selection_policy="default",
    )

    assert "MemoMemo Proposer" in prompt
    assert "Context budget" not in prompt
    assert "Context scope" not in prompt
    assert '"budget":' not in prompt
    assert "Optimization Focus" not in prompt
    assert "mechanism directions" not in prompt
    assert "Cumulative summaries may mention iterations whose raw\n  bundles are not present here" in prompt
