from collections import Counter
from typing import Any

from memomemo.evaluation import EvaluationRunner
from memomemo.schemas import ConversationTurn, LocomoExample, RetrievalHit
from memomemo.model import LocalModelClient
from memomemo.scaffolds.base import RetrievalMemoryScaffold, ScaffoldConfig, ScaffoldRun


def test_source_family_build_cache_reuses_one_state_per_sample(tmp_path) -> None:
    scaffold = CountingRetrievalScaffold()
    examples = _examples({"sample-a": 3, "sample-b": 2})
    runner = EvaluationRunner(examples=examples, out_dir=tmp_path, dry_run=True, max_eval_workers=4)

    result = runner.evaluate_scaffold(
        scaffold=scaffold,
        scaffold_name="candidate_mem0",
        config=ScaffoldConfig(top_k=1, extra={"source_family": "mem0"}),
        candidate_id="candidate_mem0_top1",
    )

    assert result.count == 5
    assert Counter(scaffold.build_sample_ids) == {"sample-a": 1, "sample-b": 1}
    payload = (tmp_path / "candidate_results" / "candidate_mem0_top1.json").read_text(
        encoding="utf-8"
    )
    assert '"enabled": true' in payload
    assert '"sample_count": 2' in payload


def test_build_cache_is_not_used_for_plain_scaffolds(tmp_path) -> None:
    scaffold = CountingRetrievalScaffold()
    examples = _examples({"sample-a": 3})
    runner = EvaluationRunner(examples=examples, out_dir=tmp_path, dry_run=True)

    runner.evaluate_scaffold(
        scaffold=scaffold,
        scaffold_name="plain_candidate",
        config=ScaffoldConfig(top_k=1),
        candidate_id="plain_candidate_top1",
    )

    assert scaffold.build_sample_ids == ["sample-a", "sample-a", "sample-a"]


def test_build_cache_reuses_across_candidates_with_same_build_tag(tmp_path) -> None:
    first = CountingRetrievalScaffold()
    second = CountingRetrievalScaffold()
    examples = _examples({"sample-a": 3, "sample-b": 2})
    runner = EvaluationRunner(examples=examples, out_dir=tmp_path, dry_run=True)
    config = ScaffoldConfig(
        top_k=1,
        extra={"source_family": "mem0", "build_tag": "same-build"},
    )

    runner.evaluate_scaffold(
        scaffold=first,
        scaffold_name="first_candidate",
        config=config,
        candidate_id="first_candidate_top1",
    )
    runner.evaluate_scaffold(
        scaffold=second,
        scaffold_name="second_candidate",
        config=config,
        candidate_id="second_candidate_top1",
    )

    assert Counter(first.build_sample_ids) == {"sample-a": 1, "sample-b": 1}
    assert second.build_sample_ids == []
    payload = (tmp_path / "candidate_results" / "second_candidate_top1.json").read_text(
        encoding="utf-8"
    )
    assert '"built_samples": []' in payload
    assert '"reused_samples": [' in payload


def test_build_cache_rebuilds_when_build_tag_changes(tmp_path) -> None:
    first = CountingRetrievalScaffold()
    second = CountingRetrievalScaffold()
    examples = _examples({"sample-a": 3})
    runner = EvaluationRunner(examples=examples, out_dir=tmp_path, dry_run=True)

    runner.evaluate_scaffold(
        scaffold=first,
        scaffold_name="first_candidate",
        config=ScaffoldConfig(top_k=1, extra={"source_family": "mem0", "build_tag": "old"}),
        candidate_id="first_candidate_top1",
    )
    runner.evaluate_scaffold(
        scaffold=second,
        scaffold_name="second_candidate",
        config=ScaffoldConfig(top_k=1, extra={"source_family": "mem0", "build_tag": "new"}),
        candidate_id="second_candidate_top1",
    )

    assert first.build_sample_ids == ["sample-a"]
    assert second.build_sample_ids == ["sample-a"]


def test_build_cache_preserves_custom_answer_method(tmp_path) -> None:
    scaffold = CustomAnswerScaffold()
    runner = EvaluationRunner(examples=_examples({"sample-a": 2}), out_dir=tmp_path, dry_run=True)

    result = runner.evaluate_scaffold(
        scaffold=scaffold,
        scaffold_name="candidate_mem0",
        config=ScaffoldConfig(top_k=1, extra={"source_family": "mem0"}),
        candidate_id="candidate_mem0_top1",
    )

    assert result.passrate == 1.0
    assert scaffold.answer_calls == 2


class CountingRetrievalScaffold(RetrievalMemoryScaffold):
    name = "counting"

    def __init__(self) -> None:
        self.build_sample_ids: list[str] = []

    def build(self, example: LocomoExample, config: ScaffoldConfig) -> dict[str, Any]:
        self.build_sample_ids.append(example.sample_id)
        return {"sample_id": example.sample_id}

    def retrieve(
        self,
        state: dict[str, Any],
        question: str,
        config: ScaffoldConfig,
    ) -> list[RetrievalHit]:
        return [
            RetrievalHit(
                text=f"{state['sample_id']} answer",
                score=1.0,
                source=self.name,
            )
        ]


class CustomAnswerScaffold(CountingRetrievalScaffold):
    def __init__(self) -> None:
        super().__init__()
        self.answer_calls = 0

    def answer(
        self,
        state: dict[str, Any],
        example: LocomoExample,
        client: LocalModelClient,
        config: ScaffoldConfig,
        *,
        max_context_chars: int,
        dry_run: bool,
    ) -> ScaffoldRun:
        self.answer_calls += 1
        return ScaffoldRun(
            prediction="FINAL ANSWER: answer",
            prompt_tokens=1,
            completion_tokens=1,
            retrieved=[],
        )


def _examples(counts: dict[str, int]) -> list[LocomoExample]:
    turn = ConversationTurn(
        session="session_1",
        session_date="2024-01-01",
        dia_id="dia_1",
        speaker="Alice",
        text="answer",
        global_index=0,
    )
    examples: list[LocomoExample] = []
    for sample_id, count in counts.items():
        for idx in range(count):
            examples.append(
                LocomoExample(
                    task_id=f"LOCOMO::{sample_id}::qa::{idx}",
                    sample_id=sample_id,
                    question="question",
                    answer="answer",
                    category=1,
                    evidence=(),
                    conversation=(turn,),
                )
            )
    return examples
