"""Memory scaffold registry."""

from __future__ import annotations

from memomemo.scaffolds.amem_scaffold import AtomicMemoryScaffold
from memomemo.scaffolds.base import MemoryScaffold, RetrievalMemoryScaffold, ScaffoldConfig, ScaffoldRun
from memomemo.scaffolds.bm25_scaffold import RankBM25Scaffold
from memomemo.scaffolds.mem0_scaffold import Mem0StyleScaffold
from memomemo.scaffolds.no_memory_scaffold import NoMemoryScaffold


SCAFFOLD_REGISTRY: dict[str, type[MemoryScaffold]] = {
    RankBM25Scaffold.name: RankBM25Scaffold,
    AtomicMemoryScaffold.name: AtomicMemoryScaffold,
    Mem0StyleScaffold.name: Mem0StyleScaffold,
    NoMemoryScaffold.name: NoMemoryScaffold,
}

DEFAULT_MEMORY_SCAFFOLDS = (
    RankBM25Scaffold.name,
    AtomicMemoryScaffold.name,
    Mem0StyleScaffold.name,
)

DEFAULT_SCAFFOLD_TOP_KS = {
    RankBM25Scaffold.name: 8,
    AtomicMemoryScaffold.name: 12,
    Mem0StyleScaffold.name: 8,
    NoMemoryScaffold.name: 0,
}


def available_scaffolds() -> tuple[str, ...]:
    return tuple(sorted(SCAFFOLD_REGISTRY))


def build_scaffold(name: str) -> MemoryScaffold:
    try:
        return SCAFFOLD_REGISTRY[name]()
    except KeyError as exc:
        available = ", ".join(available_scaffolds())
        raise ValueError(f"unknown scaffold {name!r}; available: {available}") from exc


__all__ = [
    "MemoryScaffold",
    "RetrievalMemoryScaffold",
    "ScaffoldConfig",
    "ScaffoldRun",
    "DEFAULT_MEMORY_SCAFFOLDS",
    "DEFAULT_SCAFFOLD_TOP_KS",
    "available_scaffolds",
    "build_scaffold",
]
