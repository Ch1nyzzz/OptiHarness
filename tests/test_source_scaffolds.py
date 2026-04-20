import json

from memomemo.scaffolds.base import ScaffoldConfig
from memomemo.scaffolds.mem0_scaffold import Mem0SourceScaffold, _mem0_build_fingerprint
from memomemo.schemas import ConversationTurn, LocomoExample


def example():
    turns = (
        ConversationTurn("session_1", "2024-01-01", "D1", "Alice", "I adopted a dog named Max.", 0),
        ConversationTurn("session_2", "2024-01-02", "D2", "Alice", "Max likes blue frisbees.", 1),
    )
    return LocomoExample(
        task_id="T1",
        sample_id="S1",
        question="What is Alice's dog named?",
        answer="Max",
        category=1,
        evidence=("D1",),
        conversation=turns,
    )


def two_speaker_example():
    turns = (
        ConversationTurn("session_1", "2024-01-01", "D1", "Alice", "I adopted a dog named Max.", 0),
        ConversationTurn("session_1", "2024-01-01", "D2", "Bob", "Max sounds sweet.", 1),
    )
    return LocomoExample(
        task_id="T1",
        sample_id="S1",
        question="What is Alice's dog named?",
        answer="Max",
        category=1,
        evidence=("D1",),
        conversation=turns,
    )


def test_mem0_source_scaffold_uses_upstream_api(monkeypatch):
    class FakeMemory:
        def __init__(self):
            self.added = []
            self.searched = []

        @classmethod
        def from_config(cls, config):
            instance = cls()
            instance.config = config
            return instance

        def add(self, messages, *, user_id, metadata, infer):
            self.added.append((messages, user_id, metadata, infer))
            return {"results": [{"event": "ADD"}]}

        def search(self, query, *, top_k, filters, threshold, rerank):
            self.searched.append((query, top_k, filters, threshold, rerank))
            speaker = "Alice" if filters["user_id"].startswith("Alice") else "Bob"
            return {
                "results": [
                    {
                        "id": "m1",
                        "memory": f"{speaker} remembers Alice adopted a dog named Max.",
                        "score": 0.9,
                        "metadata": {"speaker": speaker, "timestamp": "2024-01-01"},
                    }
                ]
            }

    monkeypatch.setattr("memomemo.scaffolds.mem0_scaffold.load_mem0_memory_class", lambda: FakeMemory)

    scaffold = Mem0SourceScaffold()
    state = scaffold.build(two_speaker_example(), ScaffoldConfig(top_k=1, extra={"infer": False}))
    hits = scaffold.retrieve(state, "What is Alice's dog named?", ScaffoldConfig(top_k=1))

    assert len(state.memory.added) == 2
    messages, user_id, metadata, infer = state.memory.added[0]
    assert len(messages) == 2
    assert user_id == "Alice_S1"
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert metadata["source"] == "locomo"
    assert metadata["sample_id"] == "S1"
    assert metadata["timestamp"] == "2024-01-01"
    assert metadata["target_speaker"] == "Alice"
    assert infer is False
    reverse_messages, reverse_user_id, reverse_metadata, _ = state.memory.added[1]
    assert reverse_user_id == "Bob_S1"
    assert reverse_messages[0]["role"] == "assistant"
    assert reverse_messages[1]["role"] == "user"
    assert reverse_metadata["target_speaker"] == "Bob"
    assert state.user_ids == ("Alice_S1", "Bob_S1")
    assert [item[2]["user_id"] for item in state.memory.searched] == ["Alice_S1", "Bob_S1"]
    assert hits
    assert hits[0].source == "mem0_source"
    assert "Max" in hits[0].text
    assert "2024-01-01" in hits[0].text
    assert hits[0].metadata["source_impl"] == "mem0"


def test_mem0_source_scaffold_can_load_source_snapshot(monkeypatch, tmp_path):
    captured = {}

    class FakeMemory:
        def __init__(self):
            self.added = []

        @classmethod
        def from_config(cls, config):
            return cls()

        def add(self, messages, *, user_id, metadata, infer):
            self.added.append((messages, user_id, metadata, infer))

        def search(self, query, *, top_k, filters, threshold, rerank):
            return {"results": []}

    def fake_load_mem0_memory_class(**kwargs):
        captured.update(kwargs)
        return FakeMemory

    monkeypatch.setattr("memomemo.scaffolds.mem0_scaffold.load_mem0_memory_class", fake_load_mem0_memory_class)
    snapshot = tmp_path / "upstream_source" / "mem0"

    scaffold = Mem0SourceScaffold()
    state = scaffold.build(
        two_speaker_example(),
        ScaffoldConfig(top_k=1, extra={"infer": False, "mem0_source_path": str(snapshot)}),
    )

    assert captured == {"source_path": snapshot}
    assert len(state.memory.added) == 2


def test_mem0_source_scaffold_reuses_existing_base_with_recorded_fingerprint(tmp_path, monkeypatch):
    class FakeMemory:
        @classmethod
        def from_config(cls, config):
            instance = cls()
            instance.config = config
            return instance

    base_dir = tmp_path / "mem0_source" / "S1"
    (base_dir / "qdrant").mkdir(parents=True)
    (base_dir / "history.db").write_text("", encoding="utf-8")
    (base_dir / ".done").write_text("done\n", encoding="utf-8")
    config = ScaffoldConfig(
        top_k=1,
        extra={"batch_add": True, "infer": True, "source_base_dir": str(tmp_path)},
    )
    (base_dir / "manifest.json").write_text(
        json.dumps(
            {
                "scaffold_name": "mem0_source",
                "sample_id": "S1",
                "turn_count": 2,
                "build_fingerprint": _mem0_build_fingerprint(config),
                "config": {"extra": {"batch_add": True, "infer": True}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("memomemo.scaffolds.mem0_scaffold.load_mem0_memory_class", lambda: FakeMemory)

    scaffold = Mem0SourceScaffold()
    state = scaffold.build(
        example(),
        config,
    )

    assert state.base_dir == base_dir
    assert state.user_id == "Alice_S1"
    assert state.temp_dir is None
