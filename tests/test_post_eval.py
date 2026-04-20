import json

from memomemo.post_eval import write_post_eval_artifacts
from memomemo.schemas import CandidateResult


def test_post_eval_trace_slices_use_budgeted_full_cases(tmp_path):
    result_path = tmp_path / "candidate_results" / "c.json"
    result_path.parent.mkdir(parents=True)
    long_text = "x" * 500
    tasks = [
        {
            "task_id": f"task-{idx}",
            "question": f"question {idx}",
            "gold_answer": "gold",
            "prediction": "pred",
            "score": 0.0 if idx < 6 else 1.0,
            "passed": idx >= 6,
            "prompt_tokens": idx,
            "completion_tokens": 1,
            "retrieved": [
                {
                    "text": long_text,
                    "score": 1.0,
                    "source": "bm25",
                    "metadata": {"rank": 0, "task": idx},
                }
            ],
        }
        for idx in range(12)
    ]
    candidate = CandidateResult(
        candidate_id="c",
        scaffold_name="bm25",
        passrate=0.5,
        average_score=0.5,
        token_consuming=100,
        avg_token_consuming=10,
        avg_prompt_tokens=8,
        avg_completion_tokens=2,
        count=12,
        config={"top_k": 8},
        result_path=str(result_path),
    )
    result_path.write_text(
        json.dumps({"candidate": candidate.to_dict(), "tasks": tasks}),
        encoding="utf-8",
    )

    call_dir = tmp_path / "proposer_calls" / "iter_001"
    write_post_eval_artifacts(
        run_dir=tmp_path,
        call_dir=call_dir,
        iteration=1,
        candidates=[candidate],
        frontier_ids={"c"},
    )

    low = json.loads((tmp_path / "trace_slices" / "low" / "c.json").read_text())
    medium = json.loads((tmp_path / "trace_slices" / "medium" / "c.json").read_text())
    high = json.loads((tmp_path / "trace_slices" / "high" / "c.json").read_text())

    assert low["case_limit"] == 3
    assert medium["case_limit"] == 10
    assert high["case_limit"] is None
    assert len(low["cases"]) == 3
    assert len(medium["cases"]) == 10
    assert len(high["cases"]) == 12
    assert low["cases"][0]["retrieved_preview"][0]["text"] == long_text
    assert low["cases"][0]["retrieved_preview"][0]["metadata"]["rank"] == 0
    assert (call_dir / "trace_slices" / "high" / "c.json").exists()
