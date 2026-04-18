from memomemo.evolution import make_initial_candidate_grid
from memomemo.scaffolds import (
    DEFAULT_MEMORY_SCAFFOLDS,
    DEFAULT_SCAFFOLD_TOP_KS,
    available_scaffolds,
    build_scaffold,
)
from memomemo.scaffolds.base import RetrievalMemoryScaffold, ScaffoldConfig
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


def test_all_retrieval_scaffolds_retrieve_relevant_memory():
    ex = example()
    for scaffold_name in available_scaffolds():
        scaffold = build_scaffold(scaffold_name)
        if not isinstance(scaffold, RetrievalMemoryScaffold) or scaffold_name == "no_memory":
            continue
        state = scaffold.build(ex, ScaffoldConfig(top_k=2))
        hits = scaffold.retrieve(state, ex.question, ScaffoldConfig(top_k=2))
        assert hits, scaffold_name
        assert any("Max" in hit.text for hit in hits)


def test_default_experiment_scaffolds_exclude_controls():
    assert DEFAULT_MEMORY_SCAFFOLDS == ("bm25", "amem", "mem0")
    assert DEFAULT_SCAFFOLD_TOP_KS["bm25"] == 8
    assert DEFAULT_SCAFFOLD_TOP_KS["amem"] == 12
    assert DEFAULT_SCAFFOLD_TOP_KS["mem0"] == 8
    assert "no_memory" in available_scaffolds()
    assert "no_memory" not in DEFAULT_MEMORY_SCAFFOLDS
    assert [item[2] for item in make_initial_candidate_grid()] == [
        "bm25_top8",
        "amem_top12",
        "mem0_top8",
    ]
