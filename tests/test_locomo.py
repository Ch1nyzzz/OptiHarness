from memomemo.locomo import build_splits
from memomemo.schemas import ConversationTurn, LocomoExample


def test_build_splits_draws_train_from_one_auto_sample() -> None:
    examples = _examples({"small": 20, "large": 100})

    payload = build_splits(examples, warmup_size=0, train_size=80, train_sample_id="auto", seed=7)

    train_ids = payload["splits"]["train"]
    assert payload["splits"]["warmup"] == []
    assert payload["train_sample_id"] == "large"
    assert len(train_ids) == 80
    assert all("::large::" in task_id for task_id in train_ids)
    assert train_ids != [f"LOCOMO::large::qa::{idx}" for idx in range(80)]


def test_build_splits_can_pin_train_sample() -> None:
    examples = _examples({"left": 90, "right": 90})

    payload = build_splits(examples, warmup_size=0, train_size=80, train_sample_id="left", seed=13)

    assert payload["train_sample_id"] == "left"
    assert len(payload["splits"]["train"]) == 80
    assert all("::left::" in task_id for task_id in payload["splits"]["train"])


def _examples(counts: dict[str, int]) -> list[LocomoExample]:
    turn = ConversationTurn(
        session="session_1",
        session_date="2024-01-01",
        dia_id="dia_1",
        speaker="Alice",
        text="hello",
        global_index=0,
    )
    examples: list[LocomoExample] = []
    for sample_id, count in counts.items():
        for idx in range(count):
            examples.append(
                LocomoExample(
                    task_id=f"LOCOMO::{sample_id}::qa::{idx}",
                    sample_id=sample_id,
                    question=f"q{idx}",
                    answer=f"a{idx}",
                    category=1,
                    evidence=(),
                    conversation=(turn,),
                )
            )
    return examples
