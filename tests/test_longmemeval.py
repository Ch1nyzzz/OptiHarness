from __future__ import annotations

import json
from pathlib import Path

from memomemo.longmemeval import (
    build_splits,
    build_judge_prompt,
    flatten_haystack_sessions,
    load_longmemeval_examples,
    select_split,
)
from memomemo.schemas import ConversationTurn, LocomoExample


def test_load_longmemeval_examples_converts_haystack_to_memory_qa(tmp_path: Path) -> None:
    data_path = tmp_path / "longmemeval_s_cleaned.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q001",
                    "question_type": "temporal-reasoning",
                    "question": "When did Sam buy the blue notebook?",
                    "answer": "January 3",
                    "question_date": "2024-02-01",
                    "haystack_session_ids": ["sess_a"],
                    "haystack_dates": ["2024-01-03"],
                    "haystack_sessions": [
                        [
                            {
                                "role": "user",
                                "content": "I bought a blue notebook today.",
                                "has_answer": True,
                            },
                            {"role": "assistant", "content": "Nice choice."},
                        ]
                    ],
                    "answer_session_ids": ["sess_a"],
                }
            ]
        ),
        encoding="utf-8",
    )

    examples = load_longmemeval_examples(data_path=data_path, variant="s")

    assert len(examples) == 1
    example = examples[0]
    assert example.task_id == "LONGMEMEVAL::s::q001"
    assert example.sample_id == "q001"
    assert example.category == 2
    assert example.evidence == ("sess_a",)
    assert example.metadata["question_id"] == "q001"
    assert example.metadata["question_type"] == "temporal-reasoning"
    assert example.conversation[0].session == "sess_a"
    assert example.conversation[0].session_date == "2024-01-03"
    assert example.conversation[0].speaker == "user"
    assert "has_answer" not in example.conversation[0].text


def test_flatten_haystack_sessions_uses_fallback_ids_and_dates() -> None:
    turns = flatten_haystack_sessions(
        {
            "question_date": "2024-04-10",
            "haystack_sessions": [[{"role": "user", "content": "remember pesto"}]],
        }
    )

    assert turns[0].session == "session_1"
    assert turns[0].session_date == "2024-04-10"
    assert turns[0].dia_id == "session_1:turn_0"


def test_longmemeval_build_splits_are_deterministic() -> None:
    turn = ConversationTurn(
        session="session_1",
        session_date="2024-01-01",
        dia_id="session_1:turn_0",
        speaker="user",
        text="fact",
        global_index=0,
    )
    examples = [
        LocomoExample(
            task_id=f"LONGMEMEVAL::s::q{idx}",
            sample_id=f"q{idx}",
            question=f"q{idx}",
            answer=f"a{idx}",
            category=1,
            evidence=(),
            conversation=(turn,),
        )
        for idx in range(10)
    ]

    payload = build_splits(examples, variant="s", warmup_size=2, train_size=3, seed=7)
    repeated = build_splits(examples, variant="s", warmup_size=2, train_size=3, seed=7)

    assert payload == repeated
    assert payload["benchmark"] == "longmemeval"
    assert payload["variant"] == "s"
    assert len(payload["splits"]["warmup"]) == 2
    assert len(payload["splits"]["train"]) == 3
    assert len(payload["splits"]["test"]) == 5


def test_select_split_materializes_missing_custom_split_path(tmp_path: Path) -> None:
    turn = ConversationTurn(
        session="session_1",
        session_date="2024-01-01",
        dia_id="session_1:turn_0",
        speaker="user",
        text="fact",
        global_index=0,
    )
    examples = [
        LocomoExample(
            task_id=f"LONGMEMEVAL::s::q{idx}",
            sample_id=f"q{idx}",
            question=f"q{idx}",
            answer=f"a{idx}",
            category=1,
            evidence=(),
            conversation=(turn,),
        )
        for idx in range(120)
    ]
    split_path = tmp_path / "custom_splits.json"

    selected = select_split(examples, split="train", variant="s", split_path=split_path)

    assert split_path.exists()
    assert len(selected) == 100


def test_longmemeval_judge_prompt_uses_official_question_type_rules() -> None:
    prompt = build_judge_prompt(
        question_type="knowledge-update",
        question="Where does Alex live now?",
        answer="Seattle",
        response="Alex moved from Boston to Seattle.",
    )

    assert "updated answer" in prompt
    assert "Correct Answer: Seattle" in prompt
    assert prompt.endswith("Answer yes or no only.")


def test_longmemeval_judge_prompt_handles_abstention_questions() -> None:
    prompt = build_judge_prompt(
        question_type="multi-session",
        question="What is the user's passport number?",
        answer="The history never states a passport number.",
        response="I do not know.",
        abstention=True,
    )

    assert "unanswerable question" in prompt
    assert "Explanation:" in prompt
    assert "Does the model correctly identify" in prompt
