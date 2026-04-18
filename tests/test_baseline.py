import json

from memomemo.baseline import load_baseline_candidates


def test_load_baseline_candidates_filters_split(tmp_path):
    train_dir = tmp_path / "train" / "repeat_01"
    test_dir = tmp_path / "test" / "repeat_01"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    train_summary = train_dir / "run_summary.json"
    test_summary = test_dir / "run_summary.json"
    train_summary.write_text(
        json.dumps(
            {
                "split": "train",
                "candidates": [
                    {
                        "candidate_id": "train_r01_bm25_top4",
                        "scaffold_name": "bm25",
                        "passrate": 1.0,
                        "average_score": 1.0,
                        "token_consuming": 10,
                        "avg_token_consuming": 10.0,
                        "avg_prompt_tokens": 8.0,
                        "avg_completion_tokens": 2.0,
                        "count": 1,
                        "config": {"top_k": 4},
                        "result_path": "train/result.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    test_summary.write_text(
        json.dumps({"split": "test", "candidates": [{"candidate_id": "test_r01_bm25_top4"}]}),
        encoding="utf-8",
    )
    (tmp_path / "baseline_summary.json").write_text(
        json.dumps(
            {
                "runs": [
                    {"split": "train", "summary_path": str(train_summary)},
                    {"split": "test", "summary_path": str(test_summary)},
                ]
            }
        ),
        encoding="utf-8",
    )

    candidates = load_baseline_candidates(tmp_path, split="train")

    assert [item["candidate_id"] for item in candidates] == ["train_r01_bm25_top4"]
    assert candidates[0]["scaffold_name"] == "bm25"


def test_load_baseline_candidates_can_filter_scaffolds(tmp_path):
    run_dir = tmp_path / "train" / "repeat_01"
    run_dir.mkdir(parents=True)
    summary = run_dir / "run_summary.json"
    base_candidate = {
        "passrate": 1.0,
        "average_score": 1.0,
        "token_consuming": 10,
        "avg_token_consuming": 10.0,
        "avg_prompt_tokens": 8.0,
        "avg_completion_tokens": 2.0,
        "count": 1,
        "config": {"top_k": 4},
        "result_path": "result.json",
    }
    summary.write_text(
        json.dumps(
            {
                "split": "train",
                "candidates": [
                    {
                        **base_candidate,
                        "candidate_id": "train_r01_bm25_top4",
                        "scaffold_name": "bm25",
                    },
                    {
                        **base_candidate,
                        "candidate_id": "train_r01_no_memory_top0",
                        "scaffold_name": "no_memory",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    candidates = load_baseline_candidates(tmp_path, split="train", scaffolds=("bm25",))

    assert [item["candidate_id"] for item in candidates] == ["train_r01_bm25_top4"]


def test_load_baseline_candidates_can_filter_top_k_by_scaffold(tmp_path):
    run_dir = tmp_path / "train" / "repeat_01"
    run_dir.mkdir(parents=True)
    summary = run_dir / "run_summary.json"
    base_candidate = {
        "passrate": 1.0,
        "average_score": 1.0,
        "token_consuming": 10,
        "avg_token_consuming": 10.0,
        "avg_prompt_tokens": 8.0,
        "avg_completion_tokens": 2.0,
        "count": 1,
        "result_path": "result.json",
    }
    summary.write_text(
        json.dumps(
            {
                "split": "train",
                "candidates": [
                    {
                        **base_candidate,
                        "candidate_id": "train_r01_mem0_top8",
                        "scaffold_name": "mem0",
                        "config": {"top_k": 8},
                    },
                    {
                        **base_candidate,
                        "candidate_id": "train_r01_mem0_top12",
                        "scaffold_name": "mem0",
                        "config": {"top_k": 12},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    candidates = load_baseline_candidates(
        tmp_path,
        split="train",
        scaffolds=("mem0",),
        top_k_by_scaffold={"mem0": 8},
    )

    assert [item["candidate_id"] for item in candidates] == ["train_r01_mem0_top8"]
