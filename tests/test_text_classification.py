from __future__ import annotations

import json

from memomemo.text_classification import (
    ClassificationExample,
    TextClassificationSplits,
    extract_json_field,
    load_text_classification_splits,
    run_text_classification_benchmark,
)


def test_extract_json_field_handles_fenced_json() -> None:
    text = 'prefix ```json\n{"final_answer": "B"}\n``` suffix'

    assert extract_json_field(text, "final_answer") == "B"


def test_uspto50k_loader_uses_mce_artifact_jsonl(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "uspto"
    data_dir.mkdir()
    rows = {
        "train.jsonl": [
            {"question": "Context: The reaction type is Protections.\nInput: P\nAnswer: ", "target": "A.B"},
            {"question": "Context: The reaction type is Oxidations.\nInput: Q\nAnswer: ", "target": "C"},
        ],
        "val.jsonl": [
            {"question": "Context: The reaction type is Reductions.\nInput: R\nAnswer: ", "target": "D.E"},
        ],
        "test.jsonl": [
            {"question": "Context: The reaction type is FGI.\nInput: S\nAnswer: ", "target": "F"},
        ],
    }
    for name, items in rows.items():
        (data_dir / name).write_text(
            "\n".join(json.dumps(item) for item in items) + "\n",
            encoding="utf-8",
        )
    monkeypatch.setenv("MEMOMEMO_USPTO50K_DATA_DIR", str(data_dir))

    splits = load_text_classification_splits(
        "USPTO",
        num_train=2,
        num_val=1,
        num_test=1,
    )

    assert splits.train[0].task_id == "USPTO50K::train::0"
    assert splits.train[0].target == "A.B"
    assert splits.val[0].raw_question.startswith("Context: The reaction type")
    assert splits.evaluator("b.a", "A.B")
    assert splits.evaluator('{"final_answer": "E.D"}', "D.E")
    assert not splits.evaluator("A.C", "A.B")


def test_text_classification_benchmark_writes_memomemo_results(tmp_path) -> None:
    def loader(
        dataset: str,
        *,
        num_train: int,
        num_val: int,
        num_test: int,
        shuffle_seed: int,
    ) -> TextClassificationSplits:
        del dataset, num_train, num_val, num_test, shuffle_seed

        def example(split: str, index: int) -> ClassificationExample:
            return ClassificationExample(
                task_id=f"{split}-{index}",
                input=(
                    "You are classifying patents by IPC section.\n\n"
                    "Choose exactly one section letter from these options:\n"
                    "A: Human Necessities\nB: Performing Operations; Transporting\n\n"
                    "Patent text:\nexample\n\n"
                    "Answer with only the IPC section letter."
                ),
                target="A",
                raw_question="Classify this patent.",
            )

        return TextClassificationSplits(
            train=[example("train", 0), example("train", 1)],
            val=[example("val", 0)],
            test=[example("test", 0), example("test", 1)],
            evaluator=lambda prediction, target, **_: prediction.strip() == target,
        )

    summary = run_text_classification_benchmark(
        out_dir=tmp_path,
        datasets=("USPTO",),
        memory_systems=("no_memory", "fewshot_all"),
        dry_run=True,
        split_loader=loader,
        num_train=2,
        num_val=1,
        num_test=2,
    )

    assert summary["row_count"] == 2
    assert summary["candidate_count"] == 2
    assert {item["scaffold_name"] for item in summary["candidates"]} == {
        "no_memory",
        "fewshot_all",
    }
    assert all(item["passrate"] == 1.0 for item in summary["candidates"])
    assert (tmp_path / "run_summary.json").exists()
    assert (tmp_path / "pareto_frontier.json").exists()

    payload = json.loads((tmp_path / "candidate_results" / "no_memory.json").read_text())
    assert payload["candidate"]["config"]["benchmark"] == "text_classification"
    no_memory_row = next(row for row in summary["rows"] if row["memory"] == "no_memory")
    fewshot_row = next(row for row in summary["rows"] if row["memory"] == "fewshot_all")
    assert no_memory_row["train_accuracy"] is None
    assert no_memory_row["train_correct"] is None
    assert no_memory_row["train_total"] == 2
    assert fewshot_row["train_accuracy"] is None
    assert fewshot_row["train_correct"] is None
    assert fewshot_row["train_total"] == 2
    assert no_memory_row["llm_calls"] == 3
    assert fewshot_row["llm_calls"] == 3


def test_mce_artifact_loaders_support_paper_datasets(tmp_path, monkeypatch) -> None:
    root = tmp_path / "mce"
    specs = {
        "uspto": (
            "MEMOMEMO_USPTO50K_DATA_DIR",
            {"question": "Context: The reaction type is Protections.\nInput: P\nAnswer: ", "target": "A.B"},
        ),
        "s2d": (
            "MEMOMEMO_SYMPTOM2DISEASE_DATA_DIR",
            {"question": "I feel thirsty and tired.", "answer": "diabetes"},
        ),
        "law": (
            "MEMOMEMO_LAWBENCH_DATA_DIR",
            {
                "instruction": "请你模拟法官依据下面事实给出罪名。",
                "question": "事实: 偷走手机。",
                "answer": "罪名:盗窃",
            },
        ),
    }
    for dirname, (env_name, row) in specs.items():
        data_dir = root / dirname
        data_dir.mkdir(parents=True)
        for filename in ("train.jsonl", "val.jsonl", "test.jsonl"):
            (data_dir / filename).write_text(json.dumps(row) + "\n", encoding="utf-8")
        monkeypatch.setenv(env_name, str(data_dir))

    uspto = load_text_classification_splits("USPTO", num_train=1, num_val=1, num_test=1)
    s2d = load_text_classification_splits("Symptom2Disease", num_train=1, num_val=1, num_test=1)
    law = load_text_classification_splits("LawBench", num_train=1, num_val=1, num_test=1)

    assert uspto.evaluator("B.A", "A.B")
    assert s2d.evaluator("[DIAGNOSIS]Diabetes[/DIAGNOSIS]", "diabetes")
    assert law.evaluator("[罪名]盗窃<eoa>", "罪名:盗窃")
    assert s2d.train[0].task_id.startswith("Symptom2Disease::train")
    assert law.train[0].input.startswith("请你模拟法官")
