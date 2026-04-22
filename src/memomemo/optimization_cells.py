"""Prompt-guided optimization-cell registry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizationCell:
    """One prompt-guided optimization direction for a target system."""

    name: str
    target_system: str
    description: str
    focus_functions: tuple[str, ...]
    prompt_guidance: str


MEMGPT_OPTIMIZATION_CELLS = {
    "core_summary": OptimizationCell(
        name="core_summary",
        target_system="memgpt",
        description="Optimize core memory construction and summary/compaction behavior.",
        focus_functions=(
            "_build_core_memory",
            "_build_summary_message",
            "_compile_core_memory",
            "_compile_memory_metadata",
        ),
        prompt_guidance="Focus on how stable memory and compressed history are represented.",
    ),
    "memory_representation": OptimizationCell(
        name="memory_representation",
        target_system="memgpt",
        description="Optimize how conversation turns become recall messages and archival passages.",
        focus_functions=(
            "MemGPTSourceScaffold.build",
            "_build_recall_messages",
            "_build_archival_passages",
        ),
        prompt_guidance=(
            "Focus on message-to-memory transformation, archival chunking, "
            "and memory representation."
        ),
    ),
    "retrieval_policy": OptimizationCell(
        name="retrieval_policy",
        target_system="memgpt",
        description=(
            "Optimize memory-tier mixing, ranking, expansion, deduplication, "
            "and retrieval result formatting."
        ),
        focus_functions=(
            "MemGPTSourceScaffold.retrieve",
            "_hybrid_rank",
            "_expand_recall_indices",
            "_dedupe_hits",
            "_core_hit",
            "_format_archival_result",
            "_format_recall_result",
        ),
        prompt_guidance=(
            "Focus on retrieval policy and evidence assembly, not only scalar "
            "parameter tuning."
        ),
    ),
    "all": OptimizationCell(
        name="all",
        target_system="memgpt",
        description="Global redesign / fusion across all memgpt cells.",
        focus_functions=(),
        prompt_guidance="You may fuse ideas across multiple cells and redesign boundaries if justified.",
    ),
}


def get_target_cells(target_system: str) -> list[OptimizationCell]:
    """Return optimization cells for the requested target system."""

    if target_system.lower() != "memgpt":
        return []
    return list(MEMGPT_OPTIMIZATION_CELLS.values())


def get_cell(name: str, target_system: str = "memgpt") -> OptimizationCell:
    """Return one optimization cell by name."""

    if target_system.lower() != "memgpt":
        raise KeyError(f"unknown target system: {target_system}")
    try:
        return MEMGPT_OPTIMIZATION_CELLS[name]
    except KeyError as exc:
        raise KeyError(f"unknown {target_system} optimization cell: {name}") from exc
