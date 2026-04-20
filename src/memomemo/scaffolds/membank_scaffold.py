"""Source-informed MemoryBank/SiliconFriend scaffold."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable

from memomemo.schemas import ConversationTurn, LocomoExample, RetrievalHit
from memomemo.scaffolds.base import RetrievalMemoryScaffold, ScaffoldConfig
from memomemo.scaffolds.bm25_scaffold import SimpleBM25Okapi
from memomemo.utils.text import estimate_tokens, tokenize


MEMBANK_SOURCE_IMPL = "MemoryBank-SiliconFriend"
MEMBANK_BUILD_FORMAT = "daily_memory_bank_v1"


@dataclass(frozen=True)
class MemoryBankDocument:
    memory_id: str
    date: str
    date_key: int
    session: str
    doc_type: str
    text: str
    memory_strength: float
    last_recall_key: int
    turn_indices: tuple[int, ...]
    tokens: tuple[str, ...]


@dataclass
class MemoryBankState:
    documents: tuple[MemoryBankDocument, ...]
    docs_tokens: tuple[list[str], ...]
    bm25: SimpleBM25Okapi
    latest_date_key: int
    recall_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _ScoredDocument:
    index: int
    score: float
    lexical_score: float
    semantic_score: float
    retention: float


class MemoryBankSourceScaffold(RetrievalMemoryScaffold):
    """Daily memory-bank seed inspired by MemoryBank/SiliconFriend."""

    name = "membank_source"
    reference_urls = (
        "https://arxiv.org/abs/2305.10250",
        "https://github.com/zhongwanjun/MemoryBank-SiliconFriend.git",
    )

    def build(self, example: LocomoExample, config: ScaffoldConfig) -> MemoryBankState:
        documents = tuple(_build_memory_bank_documents(example, config.extra))
        docs_tokens = tuple(list(doc.tokens) for doc in documents)
        latest_date_key = max((doc.date_key for doc in documents), default=0)
        return MemoryBankState(
            documents=documents,
            docs_tokens=docs_tokens,
            bm25=SimpleBM25Okapi([list(tokens) for tokens in docs_tokens]),
            latest_date_key=latest_date_key,
        )

    def retrieve(
        self,
        state: MemoryBankState,
        question: str,
        config: ScaffoldConfig,
    ) -> list[RetrievalHit]:
        query_tokens = tokenize(question)
        if not query_tokens or not state.documents:
            return []

        ranked = _rank_documents(state, query_tokens, config.extra)
        if not ranked:
            ranked = _recent_fallback_rank(state, limit=max(1, int(config.top_k)))

        selected = _expand_selected_documents(
            state,
            ranked[: max(1, int(config.top_k))],
            window=max(0, int(config.window)),
            max_docs=max(1, int(config.extra.get("max_group_source_docs", max(4, config.top_k * 2)))),
        )
        for scored in selected:
            doc = state.documents[scored.index]
            state.recall_counts[doc.memory_id] = state.recall_counts.get(doc.memory_id, 0) + 1
        return _date_grouped_hits(state, selected, limit=max(1, int(config.top_k)))


def _build_memory_bank_documents(example: LocomoExample, extra: dict[str, Any]) -> list[MemoryBankDocument]:
    documents: list[MemoryBankDocument] = []
    sessions = _session_groups(example.conversation)
    date_keys = _date_keys_by_session(sessions)
    exchange_size = max(1, int(extra.get("exchange_size", 2)))
    exchange_stride = max(1, int(extra.get("exchange_stride", exchange_size)))

    for session_index, (session, turns) in enumerate(sessions):
        if not turns:
            continue
        session_date = turns[0].session_date or session
        date_key = date_keys.get(session, session_index)
        for start in range(0, len(turns), exchange_stride):
            chunk = turns[start : start + exchange_size]
            if not chunk:
                continue
            text = _format_exchange(session_date, chunk)
            documents.append(
                _document(
                    example=example,
                    session=session,
                    date_text=session_date,
                    date_key=date_key,
                    doc_type="dialogue",
                    local_id=f"dialogue:{start}",
                    text=text,
                    base_strength=1.0,
                    turns=chunk,
                )
            )
            if start + exchange_size >= len(turns):
                break

        summary_text = _session_summary(session, session_date, turns)
        documents.append(
            _document(
                example=example,
                session=session,
                date_text=session_date,
                date_key=date_key,
                doc_type="summary",
                local_id="summary",
                text=summary_text,
                base_strength=1.7,
                turns=turns,
            )
        )

        personality_text = _session_personality(session, session_date, turns)
        if personality_text:
            documents.append(
                _document(
                    example=example,
                    session=session,
                    date_text=session_date,
                    date_key=date_key,
                    doc_type="personality",
                    local_id="personality",
                    text=personality_text,
                    base_strength=1.45,
                    turns=turns,
                )
            )

    if sessions:
        documents.extend(_overall_documents(example, sessions, date_keys))
    return documents


def _document(
    *,
    example: LocomoExample,
    session: str,
    date_text: str,
    date_key: int,
    doc_type: str,
    local_id: str,
    text: str,
    base_strength: float,
    turns: list[ConversationTurn],
) -> MemoryBankDocument:
    tokens = tuple(tokenize(text))
    memory_strength = base_strength + min(2.0, len(set(tokens)) / 45.0)
    if _contains_preference_or_emotion(text):
        memory_strength += 0.35
    return MemoryBankDocument(
        memory_id=f"{example.sample_id}:{session}:{local_id}",
        date=date_text,
        date_key=date_key,
        session=session,
        doc_type=doc_type,
        text=text,
        memory_strength=memory_strength,
        last_recall_key=date_key,
        turn_indices=tuple(turn.global_index for turn in turns),
        tokens=tokens,
    )


def _rank_documents(
    state: MemoryBankState,
    query_tokens: list[str],
    extra: dict[str, Any],
) -> list[_ScoredDocument]:
    lexical_scores = list(state.bm25.get_scores(query_tokens))
    semantic_scores = [_cosine_score(query_tokens, list(doc.tokens)) for doc in state.documents]
    max_lexical = max(lexical_scores, default=0.0) or 1.0
    query_set = set(query_tokens)
    scored: list[_ScoredDocument] = []
    for idx, doc in enumerate(state.documents):
        lexical = lexical_scores[idx] / max_lexical
        semantic = semantic_scores[idx]
        if lexical <= 0 and semantic <= 0:
            continue
        retention = _retention_probability(
            state=state,
            doc=doc,
            recall_count=state.recall_counts.get(doc.memory_id, 0),
            extra=extra,
        )
        doc_type_boost = {
            "dialogue": 0.03,
            "summary": 0.08,
            "personality": 0.06,
            "overall_history": 0.04,
            "overall_personality": 0.04,
        }.get(doc.doc_type, 0.0)
        date_overlap = len(query_set & set(tokenize(doc.date)))
        recall_boost = min(0.2, 0.04 * state.recall_counts.get(doc.memory_id, 0))
        score = (
            0.62 * lexical
            + 0.55 * semantic
            + float(extra.get("retention_weight", 0.12)) * retention
            + 0.035 * math.log1p(doc.memory_strength)
            + 0.04 * date_overlap
            + doc_type_boost
            + recall_boost
        )
        scored.append(
            _ScoredDocument(
                index=idx,
                score=score,
                lexical_score=lexical_scores[idx],
                semantic_score=semantic,
                retention=retention,
            )
        )
    return sorted(scored, key=lambda item: item.score, reverse=True)


def _retention_probability(
    *,
    state: MemoryBankState,
    doc: MemoryBankDocument,
    recall_count: int,
    extra: dict[str, Any],
) -> float:
    elapsed = max(0, state.latest_date_key - doc.last_recall_key)
    strength = max(0.1, doc.memory_strength + recall_count)
    if str(extra.get("forgetting_formula") or "memorybank").lower() == "source":
        return math.exp(-elapsed / float(extra.get("source_decay_days", 5.0)) * strength)
    half_life = max(1.0, float(extra.get("retention_half_life_days", 14.0)))
    return math.exp(-elapsed / (half_life * strength))


def _recent_fallback_rank(state: MemoryBankState, *, limit: int) -> list[_ScoredDocument]:
    ranked = sorted(
        range(len(state.documents)),
        key=lambda idx: (state.documents[idx].date_key, state.documents[idx].memory_strength),
        reverse=True,
    )
    return [
        _ScoredDocument(index=idx, score=0.01, lexical_score=0.0, semantic_score=0.0, retention=1.0)
        for idx in ranked[:limit]
    ]


def _expand_selected_documents(
    state: MemoryBankState,
    anchors: list[_ScoredDocument],
    *,
    window: int,
    max_docs: int,
) -> list[_ScoredDocument]:
    scored_by_index = {item.index: item for item in anchors}
    if window > 0:
        anchor_turns = {
            turn_idx
            for item in anchors
            for turn_idx in state.documents[item.index].turn_indices
        }
        for idx, doc in enumerate(state.documents):
            if idx in scored_by_index or not doc.turn_indices:
                continue
            if any(abs(turn_idx - anchor) <= window for turn_idx in doc.turn_indices for anchor in anchor_turns):
                scored_by_index[idx] = _ScoredDocument(
                    index=idx,
                    score=0.02,
                    lexical_score=0.0,
                    semantic_score=0.0,
                    retention=1.0,
                )
    return sorted(scored_by_index.values(), key=lambda item: item.score, reverse=True)[:max_docs]


def _date_grouped_hits(
    state: MemoryBankState,
    selected: list[_ScoredDocument],
    *,
    limit: int,
) -> list[RetrievalHit]:
    groups: dict[tuple[str, str], list[_ScoredDocument]] = defaultdict(list)
    order: list[tuple[str, str]] = []
    for item in selected:
        doc = state.documents[item.index]
        key = (doc.date, doc.session)
        if key not in groups:
            order.append(key)
        groups[key].append(item)

    hits: list[RetrievalHit] = []
    for key in order[:limit]:
        items = sorted(groups[key], key=lambda item: item.score, reverse=True)
        docs = [state.documents[item.index] for item in items]
        body = "\n".join(f"- ({doc.doc_type}, strength={doc.memory_strength:.2f}) {doc.text}" for doc in docs)
        score = max(item.score for item in items)
        metadata = {
            "memory_tier": "memory_bank",
            "source_impl": MEMBANK_SOURCE_IMPL,
            "build_format": MEMBANK_BUILD_FORMAT,
            "tool": "memory_bank_search",
            "date": key[0],
            "session": key[1],
            "memory_ids": [doc.memory_id for doc in docs],
            "doc_types": [doc.doc_type for doc in docs],
            "turn_indices": sorted({turn_idx for doc in docs for turn_idx in doc.turn_indices}),
            "retention": max(item.retention for item in items),
            "context_tokens": estimate_tokens(body),
        }
        hits.append(
            RetrievalHit(
                text=(
                    "MemoryBank search result\n"
                    f"date: {key[0]}\n"
                    f"session: {key[1]}\n"
                    f"content:\n{body}"
                ),
                score=score,
                source=MemoryBankSourceScaffold.name,
                metadata=metadata,
            )
        )
    return hits


def _format_exchange(date_text: str, turns: list[ConversationTurn]) -> str:
    prefix = f"Conversation content on {date_text}:"
    body = "; ".join(f"[|{turn.speaker}|]: {turn.text.strip()}" for turn in turns)
    return f"{prefix} {body}"


def _session_summary(session: str, date_text: str, turns: list[ConversationTurn]) -> str:
    speakers = _unique(turn.speaker for turn in turns)
    terms = _top_terms(turn.text for turn in turns)
    preview = "; ".join(_shorten(turn.text, 90) for turn in turns[:3])
    return (
        f"The summary of the conversation on {date_text} ({session}) is: "
        f"{', '.join(speakers)} discussed {', '.join(terms) if terms else 'their recent experiences'}. "
        f"Representative details: {preview}."
    )


def _session_personality(session: str, date_text: str, turns: list[ConversationTurn]) -> str:
    evidence = [
        turn
        for turn in turns
        if _contains_preference_or_emotion(turn.text)
    ][:6]
    if not evidence:
        return ""
    lines = "; ".join(f"{turn.speaker}: {_shorten(turn.text, 100)}" for turn in evidence)
    return (
        f"At {date_text} ({session}), the user's exhibited personality traits, emotions, "
        f"preferences, and response strategy cues include: {lines}."
    )


def _overall_documents(
    example: LocomoExample,
    sessions: list[tuple[str, list[ConversationTurn]]],
    date_keys: dict[str, int],
) -> list[MemoryBankDocument]:
    all_turns = [turn for _, turns in sessions for turn in turns]
    speakers = _unique(turn.speaker for turn in all_turns)
    first_date = all_turns[0].session_date if all_turns else "unknown"
    last_date = all_turns[-1].session_date if all_turns else "unknown"
    history_text = (
        f"Overall MemoryBank history for sample {example.sample_id}: "
        f"{len(all_turns)} messages across {len(sessions)} sessions from {first_date} to {last_date}. "
        f"Participants: {', '.join(speakers)}. Main recurring topics: "
        f"{', '.join(_top_terms((turn.text for turn in all_turns), limit=16))}."
    )
    personality_turns = [turn for turn in all_turns if _contains_preference_or_emotion(turn.text)][:10]
    personality_text = (
        "Overall MemoryBank personality and preference profile: "
        + "; ".join(f"{turn.speaker}: {_shorten(turn.text, 80)}" for turn in personality_turns)
    )
    latest_key = max(date_keys.values(), default=0)
    latest_session = sessions[-1][0]
    latest_date = sessions[-1][1][0].session_date if sessions and sessions[-1][1] else "unknown"
    return [
        _document(
            example=example,
            session=latest_session,
            date_text=latest_date,
            date_key=latest_key,
            doc_type="overall_history",
            local_id="overall_history",
            text=history_text,
            base_strength=2.1,
            turns=all_turns,
        ),
        _document(
            example=example,
            session=latest_session,
            date_text=latest_date,
            date_key=latest_key,
            doc_type="overall_personality",
            local_id="overall_personality",
            text=personality_text,
            base_strength=2.0,
            turns=personality_turns or all_turns,
        ),
    ]


def _session_groups(turns: Iterable[ConversationTurn]) -> list[tuple[str, list[ConversationTurn]]]:
    groups: list[tuple[str, list[ConversationTurn]]] = []
    by_session: dict[str, list[ConversationTurn]] = {}
    for turn in turns:
        if turn.session not in by_session:
            by_session[turn.session] = []
            groups.append((turn.session, by_session[turn.session]))
        by_session[turn.session].append(turn)
    return groups


def _date_keys_by_session(sessions: list[tuple[str, list[ConversationTurn]]]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    fallback_base = 1_000_000
    for idx, (session, turns) in enumerate(sessions):
        raw = turns[0].session_date if turns else ""
        parsed[session] = _parse_date_key(raw) or (fallback_base + idx)
    return parsed


def _parse_date_key(text: str) -> int | None:
    if not text:
        return None
    match = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", text)
    if not match:
        match = re.search(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", text)
        if match:
            month_name, day_text, year_text = match.groups()
        else:
            return None
    else:
        day_text, month_name, year_text = match.groups()
    month = _MONTHS.get(month_name.lower()[:3])
    if month is None:
        return None
    try:
        return date(int(year_text), month, int(day_text)).toordinal()
    except ValueError:
        return None


def _top_terms(texts: Iterable[str], *, limit: int = 10) -> list[str]:
    stop = {
        "about",
        "after",
        "again",
        "been",
        "could",
        "from",
        "have",
        "just",
        "like",
        "that",
        "their",
        "there",
        "they",
        "this",
        "what",
        "when",
        "with",
        "would",
        "your",
    }
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(token for token in tokenize(text) if token not in stop and not token.isdigit())
    return [token for token, _ in counts.most_common(limit)]


def _contains_preference_or_emotion(text: str) -> bool:
    tokens = set(tokenize(text))
    markers = {
        "adore",
        "anxious",
        "care",
        "excited",
        "feel",
        "felt",
        "happy",
        "hope",
        "interested",
        "love",
        "need",
        "prefer",
        "proud",
        "sad",
        "scared",
        "thankful",
        "want",
        "worried",
    }
    return bool(tokens & markers)


def _cosine_score(query_tokens: list[str], doc_tokens: list[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    query = Counter(query_tokens)
    doc = Counter(doc_tokens)
    dot = sum(query[token] * doc.get(token, 0) for token in query)
    if dot <= 0:
        return 0.0
    query_norm = math.sqrt(sum(value * value for value in query.values()))
    doc_norm = math.sqrt(sum(value * value for value in doc.values()))
    if query_norm == 0 or doc_norm == 0:
        return 0.0
    return dot / (query_norm * doc_norm)


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
