"""Mem0-style fact memory scaffold."""

from __future__ import annotations

import re
from dataclasses import dataclass

from memomemo.schemas import LocomoExample, RetrievalHit
from memomemo.scaffolds.base import RetrievalMemoryScaffold, ScaffoldConfig
from memomemo.scaffolds.bm25_scaffold import SimpleBM25Okapi
from memomemo.utils.text import tokenize


@dataclass(frozen=True)
class FactMemory:
    text: str
    speaker: str
    turn_index: int
    entities: frozenset[str]


@dataclass
class Mem0State:
    memories: list[FactMemory]
    bm25: SimpleBM25Okapi


class Mem0StyleScaffold(RetrievalMemoryScaffold):
    """Mem0-inspired scaffold with compact speaker-aware fact memories."""

    name = "mem0"
    reference_urls = ("https://github.com/mem0ai/mem0.git",)

    def build(self, example: LocomoExample, config: ScaffoldConfig) -> Mem0State:
        memories: list[FactMemory] = []
        for turn in example.conversation:
            fact = _compact_fact(turn.speaker, turn.text)
            memories.append(
                FactMemory(
                    text=f"{turn.speaker}: {fact} [{turn.session} {turn.session_date} {turn.dia_id}]",
                    speaker=turn.speaker,
                    turn_index=turn.global_index,
                    entities=frozenset(_entities(turn.speaker + " " + turn.text)),
                )
            )
        bm25 = SimpleBM25Okapi([tokenize(memory.text) for memory in memories])
        return Mem0State(memories=memories, bm25=bm25)

    def retrieve(self, state: Mem0State, question: str, config: ScaffoldConfig) -> list[RetrievalHit]:
        query_tokens = tokenize(question)
        query_entities = set(_entities(question))
        lexical_scores = state.bm25.get_scores(query_tokens)
        scored: list[tuple[float, int]] = []
        for idx, memory in enumerate(state.memories):
            speaker_bonus = 0.4 if memory.speaker.lower() in question.lower() else 0.0
            entity_bonus = 0.55 * len(query_entities & set(memory.entities))
            score = lexical_scores[idx] + speaker_bonus + entity_bonus
            scored.append((score, idx))
        ranked = sorted(scored, key=lambda pair: pair[0], reverse=True)
        hits: list[RetrievalHit] = []
        for score, idx in ranked[: max(1, config.top_k)]:
            if score <= 0:
                continue
            memory = state.memories[idx]
            hits.append(
                RetrievalHit(
                    text=memory.text,
                    score=float(score),
                    source=self.name,
                    metadata={
                        "speaker": memory.speaker,
                        "turn_index": memory.turn_index,
                        "entities": sorted(memory.entities),
                    },
                )
            )
        return hits


def _compact_fact(speaker: str, text: str) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""
    # Preserve first-person facts as speaker-scoped memories.
    text = re.sub(r"\bI\b", speaker, text)
    text = re.sub(r"\bmy\b", f"{speaker}'s", text, flags=re.IGNORECASE)
    return text


def _entities(text: str) -> list[str]:
    return re.findall(r"\b[A-Z][A-Za-z0-9_'-]{1,}\b", text or "")
