"""Persistent source-backed base memory helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_BASE_DIR = PROJECT_ROOT / "runs" / "source_base_memory"
BUILD_FINGERPRINT_VERSION = "source-base-v1"


def source_base_enabled(extra: Mapping[str, Any]) -> bool:
    return bool(extra.get("use_source_base_memory", True))


def source_base_sample_dir(scaffold_name: str, sample_id: str, extra: Mapping[str, Any]) -> Path:
    root = Path(str(extra.get("source_base_dir") or extra.get("base_memory_dir") or DEFAULT_SOURCE_BASE_DIR))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root / scaffold_name / sample_id


def validate_source_base(
    *,
    scaffold_name: str,
    sample_id: str,
    turn_count: int,
    extra: Mapping[str, Any],
    base_dir: Path,
    build_fingerprint: str,
) -> bool:
    """Return True when a base memory can be safely reused read-only."""

    if not source_base_enabled(extra):
        return False
    if not (base_dir / ".done").exists():
        return False
    manifest_path = base_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("sample_id") != sample_id:
        return False
    if manifest.get("scaffold_name") != scaffold_name:
        return False
    if int(manifest.get("turn_count", -1)) != int(turn_count):
        return False

    manifest_extra = ((manifest.get("config") or {}).get("extra") or {})
    if _build_relevant_config(manifest_extra) != _build_relevant_config(extra):
        return False

    recorded = str(manifest.get("build_fingerprint") or "")
    if recorded:
        return recorded == build_fingerprint
    return bool(extra.get("allow_legacy_source_base", False))


def build_fingerprint(
    *,
    scaffold_name: str,
    extra: Mapping[str, Any],
    logic_paths: list[Path],
) -> str:
    payload = {
        "version": BUILD_FINGERPRINT_VERSION,
        "scaffold_name": scaffold_name,
        "build_config": _build_relevant_config(extra),
        "logic": [_file_digest(path) for path in logic_paths],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _build_relevant_config(extra: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {
        "allow_legacy_source_base",
        "base_memory_dir",
        "build_cache_tag",
        "build_tag",
        "answer_max_tokens",
        "answer_temperature",
        "persist_dir",
        "query_keywords",
        "query_temperature",
        "raw_context",
        "rerank",
        "source_base_dir",
        "source_family",
        "threshold",
        "use_source_base_memory",
    }
    return {
        str(key): value
        for key, value in extra.items()
        if str(key) not in ignored
    }


def _file_digest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {"path": str(path), "sha256": "missing"}
    return {
        "path": str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
