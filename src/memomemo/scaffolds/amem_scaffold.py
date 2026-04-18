"""A-Mem-style atomic memory scaffold."""

from __future__ import annotations

import re
from dataclasses import dataclass

from memomemo.schemas import LocomoExample, RetrievalHit
from memomemo.scaffolds.base import RetrievalMemoryScaffold, ScaffoldConfig
from memomemo.scaffolds.bm25_scaffold import SimpleBM25Okapi
from memomemo.utils.text import tokenize


@dataclass(frozen=True)
class AtomicNote:
    text: str
    turn_index: int
    entities: frozenset[str]
    recency: float


@dataclass
class AtomicState:
    notes: list[AtomicNote]
    bm25: SimpleBM25Okapi


class AtomicMemoryScaffold(RetrievalMemoryScaffold):
    """A-Mem-inspired scaffold using atomic notes and lightweight links."""

    name = "amem"
    reference_urls = ("https://github.com/WujiangXu/A-mem.git",)

    def build(self, example: LocomoExample, config: ScaffoldConfig) -> AtomicState:
        notes: list[AtomicNote] = []
        total = max(1, len(example.conversation) - 1)
        for turn in example.conversation:
            for sentence in _split_sentences(turn.text):
                text = f"{turn.speaker} ({turn.session}, {turn.session_date}, {turn.dia_id}): {sentence}"
                notes.append(
                    AtomicNote(
                        text=text,
                        turn_index=turn.global_index,
                        entities=frozenset(_entities(turn.speaker + " " + sentence)),
                        recency=turn.global_index / total,
                    )
                )
        bm25 = SimpleBM25Okapi([tokenize(note.text) for note in notes])
        return AtomicState(notes=notes, bm25=bm25)

    def retrieve(self, state: AtomicState, question: str, config: ScaffoldConfig) -> list[RetrievalHit]:
        if not state.notes:
            return []
        query_tokens = tokenize(question)
        query_entities = set(_entities(question))
        lexical_scores = state.bm25.get_scores(query_tokens)
        scored: list[tuple[float, int]] = []
        for idx, note in enumerate(state.notes):
            entity_overlap = len(query_entities & set(note.entities))
            temporal_weight = float(config.extra.get("temporal_weight", 0.12))
            entity_weight = float(config.extra.get("entity_weight", 0.65))
            score = lexical_scores[idx] + entity_weight * entity_overlap + temporal_weight * note.recency
            scored.append((score, idx))
        ranked = sorted(scored, key=lambda pair: pair[0], reverse=True)
        hits: list[RetrievalHit] = []
        for score, idx in ranked[: max(1, config.top_k)]:
            if score <= 0:
                continue
            note = state.notes[idx]
            hits.append(
                RetrievalHit(
                    text=note.text,
                    score=float(score),
                    source=self.name,
                    metadata={
                        "turn_index": note.turn_index,
                        "entities": sorted(note.entities),
                    },
                )
            )
        return hits


def _split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text or "") if part.strip()]
    return parts or [text.strip()] if text.strip() else []


def _entities(text: str) -> list[str]:
    return re.findall(r"\b[A-Z][A-Za-z0-9_'-]{1,}\b", text or "")
