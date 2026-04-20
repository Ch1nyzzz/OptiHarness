#!/usr/bin/env python3
"""Build persistent source-backed base memories for LOCOMO samples."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from memomemo.locomo import load_locomo_examples, prepare_locomo, select_split
from memomemo.scaffolds.base import ScaffoldConfig
from memomemo.scaffolds.mem0_scaffold import (
    DEFAULT_LOCOMO_CUSTOM_INSTRUCTIONS,
    _locomo_mem0_add_calls,
    _mem0_build_fingerprint,
    _mem0_source_path,
)
from memomemo.upstream import load_mem0_memory_class


def main() -> int:
    args = _parse_args()
    extra = _load_scaffold_extra(args.scaffold_extra_json)
    samples = _load_samples(args.split, args.sample_id, args.max_samples)
    args.out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "out": str(args.out),
        "split": args.split,
        "sample_count": len(samples),
        "scaffolds": args.scaffolds,
        "started_at": _now(),
        "samples": [sample.sample_id for sample in samples],
    }
    _write_json(args.out / "build_summary.started.json", summary)

    for scaffold_name in args.scaffolds:
        if scaffold_name == "mem0_source":
            _build_samples(
                scaffold_name=scaffold_name,
                samples=samples,
                out_root=args.out,
                config=ScaffoldConfig(top_k=args.top_k, extra=dict(extra.get(scaffold_name, {}))),
                force=args.force,
                progress_every=args.progress_every,
                sample_workers=args.sample_workers,
            )
        else:
            raise ValueError(f"unsupported source scaffold: {scaffold_name}")

    summary["finished_at"] = _now()
    _write_json(args.out / "build_summary.json", summary)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scaffolds",
        default="mem0_source",
        help="Comma-separated source scaffolds to build: mem0_source.",
    )
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/source_base_memory"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--scaffold-extra-json", default="@configs/source_memory.example.json")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--sample-workers",
        type=int,
        default=1,
        help="Number of samples to build concurrently. Turns inside each sample remain serial.",
    )
    args = parser.parse_args()
    args.scaffolds = [item.strip() for item in args.scaffolds.split(",") if item.strip()]
    args.sample_workers = max(1, int(args.sample_workers))
    return args


def _build_samples(
    *,
    scaffold_name: str,
    samples: list[Any],
    out_root: Path,
    config: ScaffoldConfig,
    force: bool,
    progress_every: int,
    sample_workers: int,
) -> None:
    jobs = [
        (scaffold_name, sample, out_root / scaffold_name / sample.sample_id, config, force, progress_every)
        for sample in samples
    ]
    if sample_workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            _build_one_sample(job)
        return

    workers = min(sample_workers, len(jobs))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(_build_one_sample, jobs):
            pass


def _build_one_sample(job: tuple[str, Any, Path, ScaffoldConfig, bool, int]) -> None:
    scaffold_name, sample, out_dir, config, force, progress_every = job
    if scaffold_name == "mem0_source":
        _build_mem0_sample(sample, out_dir, config, force=force, progress_every=progress_every)
    else:
        raise ValueError(f"unsupported source scaffold: {scaffold_name}")


def _load_scaffold_extra(value: str | None) -> dict[str, dict[str, Any]]:
    if not value:
        return {}
    path = Path(value[1:]) if value.startswith("@") else None
    text = path.read_text(encoding="utf-8") if path is not None else value
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("--scaffold-extra-json must be a JSON object")
    out: dict[str, dict[str, Any]] = {}
    for key, val in payload.items():
        if not isinstance(val, dict):
            raise ValueError(f"extra config for {key!r} must be an object")
        out[str(key)] = dict(val)
    return out


def _load_samples(split: str, sample_ids: list[str] | None, max_samples: int) -> list[Any]:
    try:
        examples = load_locomo_examples()
    except FileNotFoundError:
        prepare_locomo()
        examples = load_locomo_examples()
    selected = examples if split == "all" else select_split(examples, split=split)
    if sample_ids:
        allowed = set(sample_ids)
        selected = [example for example in selected if example.sample_id in allowed]
    by_sample: dict[str, Any] = {}
    for example in selected:
        by_sample.setdefault(example.sample_id, example)
    samples = [by_sample[key] for key in sorted(by_sample)]
    return samples[:max_samples] if max_samples else samples


def _build_mem0_sample(
    sample: Any,
    out_dir: Path,
    config: ScaffoldConfig,
    *,
    force: bool,
    progress_every: int,
) -> None:
    done_path = out_dir / ".done"
    if done_path.exists() and not force:
        _stamp_existing_manifest(out_dir, "mem0_source", config)
        print(f"[mem0_source] skip existing {sample.sample_id}", flush=True)
        return
    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_path = out_dir / "progress.json"
    _write_json(progress_path, _progress_payload(sample, "mem0_source", "starting", 0))
    started = time.time()
    try:
        os.environ.setdefault("MEM0_TELEMETRY", "False")
        source_path = _mem0_source_path(config)
        memory_cls = (
            load_mem0_memory_class()
            if source_path is None
            else load_mem0_memory_class(source_path=source_path)
        )
        memory = memory_cls.from_config(_persistent_mem0_config(sample, out_dir, config.extra))
        infer = bool(config.extra.get("infer", True))
        speaker_user_ids, add_calls = _locomo_mem0_add_calls(sample, config.extra, base_memory=True)
        for idx, (user_id, messages, metadata) in enumerate(add_calls, start=1):
            memory.add(messages, user_id=user_id, metadata=metadata, infer=infer)
            if idx % max(1, progress_every) == 0 or idx == len(add_calls):
                _write_json(
                    progress_path,
                    {
                        **_progress_payload(sample, "mem0_source", "running", idx),
                        "total_chunks": len(add_calls),
                        "turns": len(sample.conversation),
                    },
                )

        _close_mem0_memory(memory)
        _write_manifest(
            out_dir,
            sample=sample,
            scaffold_name="mem0_source",
            config=config,
            started=started,
            extra={
                "user_ids": list(speaker_user_ids.values()),
                "speaker_user_ids": speaker_user_ids,
                "chunk_count": len(add_calls),
                "qdrant_path": str(out_dir / "qdrant"),
                "history_db_path": str(out_dir / "history.db"),
            },
        )
        done_path.write_text(_now() + "\n", encoding="utf-8")
        _write_json(
            progress_path,
            {
                **_progress_payload(sample, "mem0_source", "done", len(add_calls)),
                "total_chunks": len(add_calls),
                "turns": len(sample.conversation),
            },
        )
        print(f"[mem0_source] built {sample.sample_id}: {len(sample.conversation)} turns", flush=True)
    except Exception as exc:
        if "memory" in locals():
            _close_mem0_memory(memory)
        _write_failure(out_dir, sample, "mem0_source", exc)
        raise


def _persistent_mem0_config(sample: Any, out_dir: Path, extra: dict[str, Any]) -> dict[str, Any]:
    collection = str(extra.get("collection_name") or _safe_collection(f"memomemo_{sample.sample_id}"))
    vector_store_config = dict(extra.get("vector_store_config") or {})
    vector_store_config.update(
        {
            "path": str(out_dir / "qdrant"),
            "collection_name": collection,
        }
    )
    mem0_config: dict[str, Any] = {
        "vector_store": {
            "provider": str(extra.get("vector_store_provider") or "qdrant"),
            "config": vector_store_config,
        },
        "llm": {
            "provider": str(extra.get("llm_provider") or "openai"),
            "config": dict(extra.get("llm_config") or {}),
        },
        "embedder": {
            "provider": str(extra.get("embedder_provider") or "openai"),
            "config": dict(extra.get("embedder_config") or {}),
        },
        "history_db_path": str(extra.get("history_db_path") or out_dir / "history.db"),
    }
    if extra.get("custom_instructions"):
        mem0_config["custom_instructions"] = str(extra["custom_instructions"])
    elif bool(extra.get("default_locomo_custom_instructions", True)):
        mem0_config["custom_instructions"] = DEFAULT_LOCOMO_CUSTOM_INSTRUCTIONS
    if extra.get("reranker"):
        mem0_config["reranker"] = extra["reranker"]
    return mem0_config


def _close_mem0_memory(memory: Any) -> None:
    if hasattr(memory, "close"):
        memory.close()
    for attr in ("_entity_store", "_telemetry_vector_store", "vector_store"):
        store = getattr(memory, attr, None)
        client = getattr(store, "client", None)
        if client is not None and hasattr(client, "close"):
            try:
                client.close()
            except Exception:
                pass


def _safe_collection(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _progress_payload(sample: Any, scaffold_name: str, status: str, completed: int) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "scaffold_name": scaffold_name,
        "status": status,
        "completed": completed,
        "total": len(sample.conversation),
        "updated_at": _now(),
    }


def _write_manifest(
    out_dir: Path,
    *,
    sample: Any,
    scaffold_name: str,
    config: ScaffoldConfig,
    started: float,
    extra: dict[str, Any],
) -> None:
    _write_json(
        out_dir / "manifest.json",
        {
            "sample_id": sample.sample_id,
            "scaffold_name": scaffold_name,
            "turn_count": len(sample.conversation),
            "config": config.to_dict(),
            "build_fingerprint": _source_build_fingerprint(scaffold_name, config),
            "started_at_epoch": started,
            "finished_at_epoch": time.time(),
            "elapsed_s": round(time.time() - started, 3),
            **extra,
        },
    )


def _stamp_existing_manifest(out_dir: Path, scaffold_name: str, config: ScaffoldConfig) -> None:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    current = _source_build_fingerprint(scaffold_name, config)
    recorded = payload.get("build_fingerprint")
    if recorded == current:
        return
    if recorded:
        return
    payload["build_fingerprint"] = current
    _write_json(manifest_path, payload)


def _source_build_fingerprint(scaffold_name: str, config: ScaffoldConfig) -> str:
    if scaffold_name == "mem0_source":
        return _mem0_build_fingerprint(config)
    raise ValueError(f"unsupported scaffold for fingerprint: {scaffold_name}")


def _write_failure(out_dir: Path, sample: Any, scaffold_name: str, exc: Exception) -> None:
    _write_json(
        out_dir / "failed.json",
        {
            "sample_id": sample.sample_id,
            "scaffold_name": scaffold_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_at": _now(),
        },
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


if __name__ == "__main__":
    raise SystemExit(main())
