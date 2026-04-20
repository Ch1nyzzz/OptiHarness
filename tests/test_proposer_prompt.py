from pathlib import Path

from memomemo.proposer_prompt import build_proposer_prompt, build_ucb_proposer_prompt


def test_proposer_prompt_names_pending_eval_and_objectives():
    prompt = build_proposer_prompt(
        run_id="r",
        iteration=1,
        run_dir=Path("runs/r"),
        generated_dir=Path("runs/r/generated"),
        pending_eval_path=Path("runs/r/pending_eval.json"),
        frontier_path=Path("runs/r/pareto_frontier.json"),
        summary_path=Path("runs/r/evolution_summary.jsonl"),
        split="train",
        limit=40,
    )
    assert "pending_eval.json" in prompt
    assert "passrate" in prompt
    assert "average_score" in prompt
    assert "not an optimization objective" in prompt
    assert "maximize `passrate` and `average_score`" not in prompt
    assert "token_consuming" in prompt


def test_proposer_prompt_aligns_with_meta_harness_discipline():
    prompt = build_proposer_prompt(
        run_id="r",
        iteration=2,
        run_dir=Path("runs/r"),
        generated_dir=Path("runs/r/generated"),
        pending_eval_path=Path("runs/r/pending_eval.json"),
        frontier_path=Path("runs/r/pareto_frontier.json"),
        summary_path=Path("runs/r/evolution_summary.jsonl"),
        split="train",
        limit=40,
    )

    assert "exactly 1 new candidate" in prompt
    assert "exactly 2 new candidates" not in prompt
    assert "exactly 3 new candidates" not in prompt
    assert "Avoid candidates that are only parameter tuning" in prompt
    assert "MemoryScaffold" in prompt
    assert "Available Files" in prompt
    assert "Prototype — mandatory" not in prompt
    assert "Post-eval reports" not in prompt
    assert "Quality Gate" in prompt
    assert "gold answers at inference time" in prompt
    assert "self-critique" not in prompt
    assert "Start by reading" not in prompt
    assert "Do not scan every candidate result by default" not in prompt
    assert "candidate_results/*.json" not in prompt
    assert "runs/r/reports" in prompt
    assert "runs/r/generated" in prompt
    assert '"module": "my_candidate"' in prompt
    assert '"extra": {}' in prompt
    assert '"top_k": 8' in prompt
    assert '"top_k": [4, 8]' not in prompt
    assert "choose the parent candidate yourself" not in prompt
    assert "read-only artifacts" in prompt
    assert "build/database-construction logic" in prompt
    assert "amem_source_path" not in prompt
    assert "mem0_source_path" in prompt
    assert "fresh `source_base_dir`" in prompt
    assert "expensive" in prompt


def test_ucb_proposer_prompt_assigns_parent_and_intent_file():
    prompt = build_ucb_proposer_prompt(
        run_id="r",
        iteration=3,
        run_dir=Path("runs/r"),
        pending_eval_path=Path("runs/r/pending_eval.json"),
        frontier_path=Path("runs/r/pareto_frontier.json"),
        summary_path=Path("runs/r/evolution_summary.jsonl"),
        context_dir=Path("runs/r/proposer_calls/iter_003/context"),
        generated_dir=Path("runs/r/generated"),
        source_snapshot_dir=Path("runs/r/generated/source_snapshots/iter_003"),
        intend_path=Path("runs/r/proposer_calls/iter_003/intend.md"),
        parent_candidate_id="mem0_source_top30",
        source_family="mem0",
        cost_level="medium",
        split="train",
        limit=40,
    )

    assert "UCB Assignment" in prompt
    assert "mem0_source_top30" in prompt
    assert "source_family" in prompt
    assert "intend.md" in prompt
    assert "average_score" in prompt
    assert "not an optimization objective" in prompt
    assert "maximize `passrate` and `average_score`" not in prompt
    assert "10 full task" in prompt
    assert "Available Files" in prompt
    assert "Writable source snapshot" in prompt
    assert "candidate/" in prompt
    assert "candidate_a" not in prompt
    assert "candidate_b" not in prompt
    assert "Start by reading" not in prompt
    assert "Prototype" not in prompt
    assert "Self-critique" not in prompt
    assert "Quality Gate" in prompt
    assert "gold answers at inference time" in prompt
    assert "you may use" not in prompt
    assert '"module": "my_candidate"' in prompt
    assert '"parent_candidate_id": "mem0_source_top30"' in prompt
    assert '"extra": {}' in prompt
    assert '"top_k": 8' in prompt
    assert '"top_k": [4, 8]' not in prompt
    assert "read-only artifacts" in prompt
    assert "build/database-construction logic" in prompt
    assert "amem_source_path" not in prompt
    assert "mem0_source_path" in prompt
    assert "fresh `source_base_dir`" in prompt
    assert "source bases" in prompt
    assert "expensive" in prompt
