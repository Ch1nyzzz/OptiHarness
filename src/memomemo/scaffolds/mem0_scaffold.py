"""Source-backed mem0 scaffold."""

from __future__ import annotations

import os
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memomemo.schemas import LocomoExample, RetrievalHit
from memomemo.scaffolds.base import RetrievalMemoryScaffold, ScaffoldConfig
from memomemo.source_base import PROJECT_ROOT, build_fingerprint, source_base_sample_dir, validate_source_base
from memomemo.upstream import load_mem0_memory_class


MEM0_SOURCE_PATH_KEYS = ("mem0_source_path", "source_path", "upstream_source_path")


@dataclass
class Mem0SourceState:
    memory: Any
    user_id: str
    user_ids: tuple[str, ...] = ()
    speakers_by_user_id: dict[str, str] = field(default_factory=dict)
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    base_dir: Path | None = None


DEFAULT_LOCOMO_CUSTOM_INSTRUCTIONS = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name; do not use "user" while creating memories
   - Personal details, hobbies, life circumstances, and future plans
   - Emotional states and reactions
   - Specific dates or timeframes when they are available

2. Include meaningful personal narratives focusing on:
   - Identity, family, relationships, hobbies, health, education, career, and major life events
   - Concrete details rather than generic statements

3. Extract memories only from user messages, not from assistant responses.

4. Format each memory as a concise paragraph that captures the person's experience and relevant context.
""".strip()


class Mem0SourceScaffold(RetrievalMemoryScaffold):
    """Adapter around upstream mem0 Memory.add/search."""

    name = "mem0_source"
    reference_urls = ("https://github.com/mem0ai/mem0.git",)

    def build(self, example: LocomoExample, config: ScaffoldConfig) -> Mem0SourceState:
        cached = _load_base_mem0_state(example, config)
        if cached is not None:
            return cached

        _disable_mem0_telemetry(config.extra)
        memory_cls = _load_mem0_memory_class(config)
        mem0_config, temp_dir = _mem0_config(config, example)
        memory = memory_cls.from_config(mem0_config)
        infer = bool(config.extra.get("infer", True))
        speaker_user_ids, add_calls = _locomo_mem0_add_calls(example, config.extra)
        for user_id, messages, metadata in add_calls:
            memory.add(messages, user_id=user_id, metadata=metadata, infer=infer)

        user_ids = tuple(speaker_user_ids.values())
        primary_user_id = user_ids[0] if user_ids else str(config.extra.get("user_id") or f"locomo-{example.sample_id}")
        return Mem0SourceState(
            memory=memory,
            user_id=primary_user_id,
            user_ids=user_ids or (primary_user_id,),
            speakers_by_user_id={user_id: speaker for speaker, user_id in speaker_user_ids.items()},
            temp_dir=temp_dir,
        )

    def retrieve(self, state: Mem0SourceState, question: str, config: ScaffoldConfig) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        user_ids = state.user_ids or (state.user_id,)
        for user_idx, user_id in enumerate(user_ids):
            payload = state.memory.search(
                question,
                top_k=max(1, config.top_k),
                filters={"user_id": user_id},
                threshold=float(config.extra.get("threshold", 0.1)),
                rerank=bool(config.extra.get("rerank", False)),
            )
            results = payload.get("results", payload) if isinstance(payload, dict) else payload
            speaker = state.speakers_by_user_id.get(user_id, "")
            for idx, item in enumerate(results or []):
                if not isinstance(item, dict):
                    continue
                text = str(item.get("memory") or item.get("text") or item.get("data") or "")
                if not text:
                    continue
                item_metadata = dict(item.get("metadata") or {})
                timestamp = str(item_metadata.get("timestamp") or item_metadata.get("session_date") or "")
                formatted_text = _format_retrieved_memory(text, speaker=speaker, timestamp=timestamp)
                metadata = dict(item_metadata)
                metadata.update(
                    {
                        "memory_id": item.get("id"),
                        "source_impl": "mem0",
                        "rank": idx,
                        "speaker_rank": user_idx,
                        "user_id": user_id,
                    }
                )
                if speaker:
                    metadata.setdefault("speaker", speaker)
                hits.append(
                    RetrievalHit(
                        text=formatted_text,
                        score=float(item.get("score") or max(0.0, 1.0 - idx * 0.01)),
                        source=self.name,
                        metadata=metadata,
                    )
                )
        return hits


def _mem0_config(
    config: ScaffoldConfig,
    example: LocomoExample,
) -> tuple[dict[str, Any], tempfile.TemporaryDirectory[str] | None]:
    raw = config.extra.get("mem0_config")
    if isinstance(raw, dict):
        return raw, None

    persist_dir = config.extra.get("persist_dir")
    temp_dir = None if persist_dir else tempfile.TemporaryDirectory(prefix="memomemo_mem0_")
    base_dir = str(persist_dir or (temp_dir.name if temp_dir is not None else ""))
    collection = str(
        config.extra.get("collection_name")
        or re.sub(r"[^A-Za-z0-9_]", "_", f"memomemo_{example.sample_id}_{example.task_id}_{uuid.uuid4().hex[:8]}")
    )
    provider = str(config.extra.get("vector_store_provider") or "qdrant")
    vector_store_config = dict(config.extra.get("vector_store_config") or {})
    vector_store_config.setdefault("path", base_dir)
    vector_store_config.setdefault("collection_name", collection)
    mem0_config: dict[str, Any] = {
        "vector_store": {
            "provider": provider,
            "config": vector_store_config,
        },
        "llm": {
            "provider": str(config.extra.get("llm_provider") or "openai"),
            "config": dict(config.extra.get("llm_config") or {}),
        },
        "embedder": {
            "provider": str(config.extra.get("embedder_provider") or "openai"),
            "config": dict(config.extra.get("embedder_config") or {}),
        },
        "history_db_path": str(config.extra.get("history_db_path") or f"{base_dir}/history.db"),
    }
    if config.extra.get("custom_instructions"):
        mem0_config["custom_instructions"] = str(config.extra["custom_instructions"])
    elif bool(config.extra.get("default_locomo_custom_instructions", True)):
        mem0_config["custom_instructions"] = DEFAULT_LOCOMO_CUSTOM_INSTRUCTIONS
    if config.extra.get("reranker"):
        mem0_config["reranker"] = config.extra["reranker"]
    return mem0_config, temp_dir


def _chunk_messages(messages: list[dict[str, Any]], extra: dict[str, Any]) -> list[list[dict[str, Any]]]:
    max_messages = int(extra.get("add_chunk_max_messages", 40) or 0)
    max_chars = int(extra.get("add_chunk_max_chars", 20000) or 0)
    if max_messages <= 0 and max_chars <= 0:
        return [messages]

    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for message in messages:
        message_chars = len(str(message.get("content") or ""))
        over_message_budget = bool(max_messages > 0 and len(current) >= max_messages)
        over_char_budget = bool(max_chars > 0 and current and current_chars + message_chars > max_chars)
        if over_message_budget or over_char_budget:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(message)
        current_chars += message_chars
    if current:
        chunks.append(current)
    return chunks


def _locomo_mem0_add_calls(
    example: LocomoExample,
    extra: dict[str, Any],
    *,
    base_memory: bool = False,
) -> tuple[dict[str, str], list[tuple[str, list[dict[str, Any]], dict[str, Any]]]]:
    """Build official-style LOCOMO Mem0 add calls.

    LOCOMO conversations contain two speakers. Mem0's benchmark adapter stores
    each conversation twice, once from each speaker's perspective: the target
    speaker is the ``user`` and the other speaker is the ``assistant``. This
    preserves Mem0's user-memory semantics while still exposing both sides at
    retrieval time.
    """

    speakers = _locomo_speakers(example)
    speaker_user_ids = _speaker_user_ids(example, speakers, extra)
    batch_size = _mem0_add_batch_size(extra)
    calls: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = []

    for session, session_date, turns in _session_groups(example):
        if not turns:
            continue
        target_speakers = speakers or [turns[0].speaker]
        for target_speaker in target_speakers:
            user_id = speaker_user_ids[target_speaker]
            messages = [
                {
                    "role": "user" if turn.speaker == target_speaker else "assistant",
                    "content": _mem0_turn_content(turn, extra),
                    "name": _safe_name(turn.speaker),
                }
                for turn in turns
            ]
            for chunk_index, chunk in enumerate(_chunks_by_count(messages, batch_size)):
                metadata: dict[str, Any] = {
                    "sample_id": example.sample_id,
                    "source": "locomo",
                    "session": session,
                    "session_date": session_date,
                    "timestamp": session_date,
                    "target_speaker": target_speaker,
                    "speaker": target_speaker,
                    "speaker_user_id": user_id,
                    "batch_index": chunk_index,
                }
                if base_memory:
                    metadata["base_memory"] = True
                if example.task_id and not base_memory:
                    metadata["task_id"] = example.task_id
                calls.append((user_id, chunk, metadata))
    return speaker_user_ids, calls


def _locomo_speakers(example: LocomoExample) -> list[str]:
    speakers: list[str] = []
    seen: set[str] = set()
    for turn in example.conversation:
        speaker = str(turn.speaker or "").strip()
        if speaker and speaker not in seen:
            speakers.append(speaker)
            seen.add(speaker)
    return speakers[:2] if len(speakers) > 2 else speakers


def _speaker_user_ids(example: LocomoExample, speakers: list[str], extra: dict[str, Any]) -> dict[str, str]:
    explicit = extra.get("speaker_user_ids")
    if isinstance(explicit, dict):
        return {
            speaker: str(explicit.get(speaker) or _default_speaker_user_id(example.sample_id, speaker))
            for speaker in speakers
        }
    if len(speakers) == 1 and extra.get("user_id"):
        return {speakers[0]: str(extra["user_id"])}
    return {
        speaker: _default_speaker_user_id(example.sample_id, speaker)
        for speaker in speakers
    }


def _default_speaker_user_id(sample_id: str, speaker: str) -> str:
    safe_speaker = _safe_name(speaker) or "speaker"
    safe_sample = re.sub(r"[^A-Za-z0-9_-]", "_", str(sample_id))[:96] or "sample"
    return f"{safe_speaker}_{safe_sample}"


def _session_groups(example: LocomoExample) -> list[tuple[str, str, list[Any]]]:
    groups: list[tuple[str, str, list[Any]]] = []
    index_by_session: dict[str, int] = {}
    for turn in example.conversation:
        if turn.session not in index_by_session:
            index_by_session[turn.session] = len(groups)
            groups.append((turn.session, turn.session_date, []))
        groups[index_by_session[turn.session]][2].append(turn)
    return groups


def _mem0_add_batch_size(extra: dict[str, Any]) -> int:
    if extra.get("batch_add") is False:
        return 1
    return max(1, int(extra.get("mem0_batch_size") or extra.get("add_batch_size") or 2))


def _chunks_by_count(messages: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [messages[idx : idx + batch_size] for idx in range(0, len(messages), max(1, batch_size))]


def _mem0_turn_content(turn: Any, extra: dict[str, Any]) -> str:
    max_chars = int(extra.get("turn_max_chars", 2000))
    text = str(turn.text or "")
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return f"{turn.speaker}: {text}"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(value or ""))[:64]


def _format_retrieved_memory(text: str, *, speaker: str, timestamp: str) -> str:
    prefix_parts = []
    if speaker:
        prefix_parts.append(str(speaker))
    if timestamp:
        prefix_parts.append(str(timestamp))
    if not prefix_parts:
        return text
    return f"[{' | '.join(prefix_parts)}] {text}"


def _load_base_mem0_state(example: LocomoExample, config: ScaffoldConfig) -> Mem0SourceState | None:
    base_dir = source_base_sample_dir(Mem0SourceScaffold.name, example.sample_id, config.extra)
    fingerprints = [_mem0_build_fingerprint(config)]
    if _mem0_source_path(config) is None:
        fingerprints.append(_mem0_legacy_build_fingerprint(config))
    source_path = _mem0_source_path(config)
    if source_path is None:
        recorded = _recorded_source_base_fingerprint(base_dir)
        if recorded:
            fingerprints.append(recorded)
    if not any(
        validate_source_base(
            scaffold_name=Mem0SourceScaffold.name,
            sample_id=example.sample_id,
            turn_count=len(example.conversation),
            extra=config.extra,
            base_dir=base_dir,
            build_fingerprint=fingerprint,
        )
        for fingerprint in fingerprints
    ):
        return None
    qdrant_dir = base_dir / "qdrant"
    history_db = base_dir / "history.db"
    if not qdrant_dir.exists() or not history_db.exists():
        return None

    _disable_mem0_telemetry(config.extra)
    memory_cls = _load_mem0_memory_class(config)
    memory = memory_cls.from_config(_base_mem0_config(config, example, base_dir))
    speaker_user_ids = _speaker_user_ids(example, _locomo_speakers(example), config.extra)
    user_ids = tuple(speaker_user_ids.values())
    primary_user_id = user_ids[0] if user_ids else str(config.extra.get("user_id") or f"locomo-{example.sample_id}")
    return Mem0SourceState(
        memory=memory,
        user_id=primary_user_id,
        user_ids=user_ids or (primary_user_id,),
        speakers_by_user_id={user_id: speaker for speaker, user_id in speaker_user_ids.items()},
        temp_dir=None,
        base_dir=base_dir,
    )


def _recorded_source_base_fingerprint(base_dir: Path) -> str | None:
    manifest_path = base_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    recorded = str(manifest.get("build_fingerprint") or "").strip()
    return recorded or None


def _base_mem0_config(config: ScaffoldConfig, example: LocomoExample, base_dir: Path) -> dict[str, Any]:
    vector_store_config = dict(config.extra.get("vector_store_config") or {})
    vector_store_config.update(
        {
            "path": str(base_dir / "qdrant"),
            "collection_name": str(
                config.extra.get("collection_name")
                or re.sub(r"[^A-Za-z0-9_]", "_", f"memomemo_{example.sample_id}")
            ),
        }
    )
    mem0_config: dict[str, Any] = {
        "vector_store": {
            "provider": str(config.extra.get("vector_store_provider") or "qdrant"),
            "config": vector_store_config,
        },
        "llm": {
            "provider": str(config.extra.get("llm_provider") or "openai"),
            "config": dict(config.extra.get("llm_config") or {}),
        },
        "embedder": {
            "provider": str(config.extra.get("embedder_provider") or "openai"),
            "config": dict(config.extra.get("embedder_config") or {}),
        },
        "history_db_path": str(config.extra.get("history_db_path") or base_dir / "history.db"),
    }
    if config.extra.get("custom_instructions"):
        mem0_config["custom_instructions"] = str(config.extra["custom_instructions"])
    elif bool(config.extra.get("default_locomo_custom_instructions", True)):
        mem0_config["custom_instructions"] = DEFAULT_LOCOMO_CUSTOM_INSTRUCTIONS
    if config.extra.get("reranker"):
        mem0_config["reranker"] = config.extra["reranker"]
    return mem0_config


def _disable_mem0_telemetry(extra: dict[str, Any]) -> None:
    if "mem0_telemetry" in extra:
        os.environ["MEM0_TELEMETRY"] = str(bool(extra["mem0_telemetry"]))
    else:
        os.environ["MEM0_TELEMETRY"] = "False"


def _load_mem0_memory_class(config: ScaffoldConfig) -> Any:
    source_path = _mem0_source_path(config)
    if source_path is None:
        return load_mem0_memory_class()
    return load_mem0_memory_class(source_path=source_path)


def _mem0_source_path(config: ScaffoldConfig) -> Path | None:
    for key in MEM0_SOURCE_PATH_KEYS:
        value = config.extra.get(key)
        if value:
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            return path
    return None


def _mem0_build_fingerprint(config: ScaffoldConfig) -> str:
    source_root = _mem0_source_path(config) or PROJECT_ROOT / "references" / "vendor" / "mem0"
    mem0_logic_paths = sorted((source_root / "mem0").glob("**/*.py")) if (source_root / "mem0").exists() else []
    return build_fingerprint(
        scaffold_name=Mem0SourceScaffold.name,
        extra=config.extra,
        logic_paths=[
            PROJECT_ROOT / "src" / "memomemo" / "scaffolds" / "mem0_scaffold.py",
            PROJECT_ROOT / "src" / "memomemo" / "upstream.py",
            *mem0_logic_paths,
        ],
    )


def _mem0_legacy_build_fingerprint(config: ScaffoldConfig) -> str:
    source_root = PROJECT_ROOT / "references" / "vendor" / "mem0"
    return build_fingerprint(
        scaffold_name=Mem0SourceScaffold.name,
        extra=config.extra,
        logic_paths=[
            PROJECT_ROOT / "src" / "memomemo" / "scaffolds" / "mem0_scaffold.py",
            PROJECT_ROOT / "src" / "memomemo" / "upstream.py",
            source_root / "mem0" / "memory" / "main.py",
            source_root / "mem0" / "memory" / "base.py",
            source_root / "mem0" / "llms" / "openai.py",
            source_root / "mem0" / "configs" / "llms" / "openai.py",
        ],
    )
