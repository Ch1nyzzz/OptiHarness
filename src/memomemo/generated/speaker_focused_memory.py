"""Speaker-focused hierarchical memory scaffold.

Mechanism (distinct from baselines):
- Build per-speaker sub-indices of *compacted* facts (pronouns rewritten to
  speaker names) and a global compact-fact index.
- At retrieval, extract speakers referenced in the question. For each
  referenced speaker, pull their within-speaker BM25-top-N facts as a
  profile block. Merge with a small number of global top-N compact hits.
- Also prepend a short "conversation timeline" header (session date +
  participants) so temporal questions have anchors.

This differs from:
- `bm25`: no per-speaker index, no fact compaction, no timeline header.
- `mem0`: adds a fixed speaker_bonus but still ranks globally; no
  within-speaker retrieval buckets and no timeline header.
- `amem`: note index is sentence-level and global; no speaker sub-index.

The structural change is the per-speaker retrieval bucket plus the timeline
header, not a parameter tweak.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from memomemo.schemas import ConversationTurn, LocomoExample, RetrievalHit
from memomemo.scaffolds.base import RetrievalMemoryScaffold, ScaffoldConfig
from memomemo.scaffolds.bm25_scaffold import SimpleBM25Okapi
from memomemo.utils.text import tokenize


_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9'-]{1,}\b")


@dataclass
class _SpeakerIndex:
    indices: list[int]
    bm25: SimpleBM25Okapi


@dataclass
class SpeakerFocusedState:
    turns: tuple[ConversationTurn, ...]
    compact_docs: list[str]
    tokenized_docs: list[list[str]]
    global_bm25: SimpleBM25Okapi
    by_speaker: dict[str, _SpeakerIndex]
    timeline: str
    speakers: frozenset[str]


class SpeakerFocusedMemoryScaffold(RetrievalMemoryScaffold):
    """Hierarchical memory: per-speaker buckets + global fallback + timeline."""

    name = "speaker_focused"
    reference_urls = (
        "https://github.com/mem0ai/mem0.git",
        "https://github.com/WujiangXu/A-mem.git",
    )

    def build(self, example: LocomoExample, config: ScaffoldConfig) -> SpeakerFocusedState:
        turns = example.conversation
        compact_docs: list[str] = []
        for turn in turns:
            compact = _compact_fact(turn.speaker, turn.text)
            compact_docs.append(
                f"[{turn.session_date} | {turn.dia_id}] {turn.speaker}: {compact[:260]}"
            )
        tokenized_docs = [tokenize(doc) for doc in compact_docs]
        global_bm25 = SimpleBM25Okapi(tokenized_docs)

        speaker_indices: dict[str, list[int]] = {}
        for idx, turn in enumerate(turns):
            if not turn.speaker:
                continue
            speaker_indices.setdefault(turn.speaker, []).append(idx)

        by_speaker: dict[str, _SpeakerIndex] = {}
        for spk, idxs in speaker_indices.items():
            sub_tokens = [tokenized_docs[i] for i in idxs]
            if not sub_tokens:
                continue
            by_speaker[spk] = _SpeakerIndex(indices=idxs, bm25=SimpleBM25Okapi(sub_tokens))

        timeline = _render_timeline(turns)
        return SpeakerFocusedState(
            turns=turns,
            compact_docs=compact_docs,
            tokenized_docs=tokenized_docs,
            global_bm25=global_bm25,
            by_speaker=by_speaker,
            timeline=timeline,
            speakers=frozenset(by_speaker.keys()),
        )

    def retrieve(
        self,
        state: SpeakerFocusedState,
        question: str,
        config: ScaffoldConfig,
    ) -> list[RetrievalHit]:
        if not state.turns:
            return []
        query = tokenize(question)
        if not query:
            return []

        profile_cap = int(config.extra.get("profile_cap", 6))
        timeline_included = bool(config.extra.get("include_timeline", True))

        q_entities = set(_ENTITY_RE.findall(question))
        mentioned_speakers = sorted(state.speakers & q_entities)

        profile_picks: list[tuple[int, float, str]] = []
        per_speaker_budget = max(1, profile_cap)
        for speaker in mentioned_speakers:
            bucket = state.by_speaker.get(speaker)
            if bucket is None:
                continue
            bucket_scores = list(bucket.bm25.get_scores(query))
            ordered_local = sorted(
                range(len(bucket.indices)),
                key=lambda local_i: bucket_scores[local_i],
                reverse=True,
            )
            added = 0
            for local_i in ordered_local:
                if added >= per_speaker_budget:
                    break
                score = bucket_scores[local_i]
                if score <= 0:
                    continue
                global_idx = bucket.indices[local_i]
                profile_picks.append((global_idx, float(score), speaker))
                added += 1

        global_scores = list(state.global_bm25.get_scores(query))
        # Entity bonus: turns whose text references any non-speaker question entity.
        non_speaker_entities = {ent for ent in q_entities if ent not in state.speakers}
        if non_speaker_entities:
            for idx, turn in enumerate(state.turns):
                text_entities = set(_ENTITY_RE.findall(turn.text))
                overlap = len(text_entities & non_speaker_entities)
                if overlap:
                    global_scores[idx] += 0.6 * overlap

        already = {pick[0] for pick in profile_picks}
        global_rank = sorted(
            range(len(state.turns)),
            key=lambda idx: global_scores[idx],
            reverse=True,
        )
        remaining_budget = max(1, config.top_k)
        global_picks: list[tuple[int, float]] = []
        for idx in global_rank:
            if idx in already:
                continue
            if global_scores[idx] <= 0:
                continue
            global_picks.append((idx, float(global_scores[idx])))
            if len(global_picks) >= remaining_budget:
                break

        merged: list[tuple[int, float, str]] = list(profile_picks)
        merged.extend((idx, score, "global") for idx, score in global_picks)
        merged.sort(key=lambda pair: pair[0])  # chronological order

        hits: list[RetrievalHit] = []
        if timeline_included and state.timeline and mentioned_speakers:
            hits.append(
                RetrievalHit(
                    text=f"[TIMELINE] {state.timeline}",
                    score=0.0,
                    source=self.name,
                    metadata={"layer": "timeline"},
                )
            )

        for idx, score, layer in merged:
            hits.append(
                RetrievalHit(
                    text=state.compact_docs[idx],
                    score=score,
                    source=self.name,
                    metadata={
                        "turn_index": idx,
                        "speaker": state.turns[idx].speaker,
                        "layer": layer,
                    },
                )
            )
        return hits


def _compact_fact(speaker: str, text: str) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""
    text = re.sub(r"\bI\b", speaker, text)
    text = re.sub(r"\bmy\b", f"{speaker}'s", text, flags=re.IGNORECASE)
    text = re.sub(r"\bme\b", speaker, text, flags=re.IGNORECASE)
    return text


def _render_timeline(turns: tuple[ConversationTurn, ...]) -> str:
    by_session: dict[str, list[ConversationTurn]] = {}
    for turn in turns:
        by_session.setdefault(turn.session, []).append(turn)
    parts: list[str] = []
    for session, session_turns in sorted(by_session.items(), key=lambda kv: kv[0]):
        date = session_turns[0].session_date or "?"
        speakers = sorted({t.speaker for t in session_turns if t.speaker})
        parts.append(f"{session}@{date}[{'/'.join(speakers)}]")
    return " | ".join(parts)
