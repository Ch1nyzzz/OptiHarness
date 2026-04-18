from pathlib import Path

from memomemo.proposer_prompt import build_proposer_prompt


def test_proposer_prompt_names_pending_eval_and_objectives():
    prompt = build_proposer_prompt(
        run_id="r",
        iteration=1,
        run_dir=Path("runs/r"),
        pending_eval_path=Path("runs/r/pending_eval.json"),
        frontier_path=Path("runs/r/pareto_frontier.json"),
        summary_path=Path("runs/r/evolution_summary.jsonl"),
        split="train",
        limit=40,
    )
    assert "pending_eval.json" in prompt
    assert "passrate" in prompt
    assert "token_consuming" in prompt


def test_proposer_prompt_aligns_with_meta_harness_discipline():
    prompt = build_proposer_prompt(
        run_id="r",
        iteration=2,
        run_dir=Path("runs/r"),
        pending_eval_path=Path("runs/r/pending_eval.json"),
        frontier_path=Path("runs/r/pareto_frontier.json"),
        summary_path=Path("runs/r/evolution_summary.jsonl"),
        split="train",
        limit=40,
    )

    assert "exactly 2 new candidates" in prompt
    assert "exactly 3 new candidates" not in prompt
    assert "Prototype — mandatory" in prompt
    assert "Post-eval reports" in prompt
    assert "Avoid candidates that are only parameter tuning" in prompt
    assert "self-critique" in prompt
    assert "MemoryScaffold" in prompt
    assert "Do not scan every candidate result by default" in prompt
    assert "candidate_results/*.json" not in prompt
    assert "runs/r/reports" in prompt
