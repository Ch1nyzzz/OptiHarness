"""No-memory baseline scaffold."""

from __future__ import annotations

from memomemo.schemas import LocomoExample, RetrievalHit
from memomemo.scaffolds.base import RetrievalMemoryScaffold, ScaffoldConfig


class NoMemoryScaffold(RetrievalMemoryScaffold):
    """Pure reader baseline with no retrieved memory context."""

    name = "no_memory"

    def build(self, example: LocomoExample, config: ScaffoldConfig) -> None:
        return None

    def retrieve(self, state: None, question: str, config: ScaffoldConfig) -> list[RetrievalHit]:
        return []
