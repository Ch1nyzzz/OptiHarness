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
            "score": 0.0 if idx < 25 else 1.0,
            "passed": idx >= 25,
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
        for idx in range(40)
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
        count=40,
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

    assert low["case_limit"] == 10
    assert medium["case_limit"] == 30
    assert high["case_limit"] is None
    assert len(low["cases"]) == 10
    assert len(medium["cases"]) == 30
    assert len(high["cases"]) == 40
    assert low["selection_strategy"] == "progressive_unresolved_failures_first"
    assert low["cases"][0]["retrieved_preview"][0]["text"] == long_text
    assert low["cases"][0]["retrieved_preview"][0]["metadata"]["rank"] == 0
    assert (call_dir / "trace_slices" / "high" / "c.json").exists()
    compact = json.loads((call_dir / "eval" / "candidate_result.compact.json").read_text())
    diagnostics = json.loads((call_dir / "eval" / "retrieval_diagnostics.json").read_text())
    table = json.loads((tmp_path / "candidate_score_table.json").read_text())
    diagnostics_summary = json.loads(
        (tmp_path / "retrieval_diagnostics_summary.json").read_text()
    )
    assert compact["candidate"]["candidate_id"] == "c"
    assert compact["tasks"][0]["retrieved"][0]["metadata"]["rank"] == 0
    assert diagnostics["retrieved_but_failed_count"] == 25
    assert table[0]["candidate_id"] == "c"
    assert diagnostics_summary[0]["candidate_id"] == "c"


def test_post_eval_trace_slices_prioritize_unresolved_failures(tmp_path):
    previous_path = tmp_path / "candidate_results" / "iter001_previous.json"
    current_path = tmp_path / "candidate_results" / "iter002_current.json"
    previous_path.parent.mkdir(parents=True)
    previous_candidate = CandidateResult(
        candidate_id="iter001_previous",
        scaffold_name="bm25",
        passrate=0.5,
        average_score=0.5,
        token_consuming=10,
        avg_token_consuming=5,
        avg_prompt_tokens=4,
        avg_completion_tokens=1,
        count=2,
        config={"top_k": 8},
        result_path=str(previous_path),
    )
    current_candidate = CandidateResult(
        candidate_id="iter002_current",
        scaffold_name="bm25",
        passrate=0.0,
        average_score=0.0,
        token_consuming=10,
        avg_token_consuming=5,
        avg_prompt_tokens=4,
        avg_completion_tokens=1,
        count=3,
        config={"top_k": 8},
        result_path=str(current_path),
    )
    previous_path.write_text(
        json.dumps(
            {
                "candidate": previous_candidate.to_dict(),
                "tasks": [
                    _task("previously-passed", passed=True),
                    _task("still-unresolved", passed=False),
                ],
            }
        ),
        encoding="utf-8",
    )
    current_path.write_text(
        json.dumps(
            {
                "candidate": current_candidate.to_dict(),
                "tasks": [
                    _task("previously-passed", passed=False),
                    _task("still-unresolved", passed=False),
                    _task("new-unresolved", passed=False),
                ],
            }
        ),
        encoding="utf-8",
    )

    write_post_eval_artifacts(
        run_dir=tmp_path,
        call_dir=None,
        iteration=2,
        candidates=[current_candidate],
        frontier_ids=set(),
    )

    low = json.loads((tmp_path / "trace_slices" / "low" / "iter002_current.json").read_text())

    assert low["previously_passed_task_count"] == 1
    assert low["progressive_failure_count"] == 2
    assert [case["task_id"] for case in low["cases"][:3]] == [
        "still-unresolved",
        "new-unresolved",
        "previously-passed",
    ]


def _task(task_id: str, *, passed: bool) -> dict[str, object]:
    return {
        "task_id": task_id,
        "question": f"question {task_id}",
        "gold_answer": "gold",
        "prediction": "pred",
        "score": 1.0 if passed else 0.0,
        "passed": passed,
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "retrieved": [],
    }
