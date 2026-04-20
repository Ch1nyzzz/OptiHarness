"""Post-eval compact artifacts for future proposer context."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memomemo.schemas import CandidateResult


def write_post_eval_artifacts(
    *,
    run_dir: Path,
    call_dir: Path | None,
    iteration: int,
    candidates: list[CandidateResult],
    frontier_ids: set[str],
) -> None:
    """Write compact eval summaries and trace slices for evaluated candidates."""

    if not candidates:
        return

    trace_root = run_dir / "trace_slices"
    low_dir = trace_root / "low"
    medium_dir = trace_root / "medium"
    high_dir = trace_root / "high"
    low_dir.mkdir(parents=True, exist_ok=True)
    medium_dir.mkdir(parents=True, exist_ok=True)
    high_dir.mkdir(parents=True, exist_ok=True)
    call_low_dir: Path | None = None
    call_medium_dir: Path | None = None
    call_high_dir: Path | None = None
    if call_dir is not None:
        call_low_dir = call_dir / "trace_slices" / "low"
        call_medium_dir = call_dir / "trace_slices" / "medium"
        call_high_dir = call_dir / "trace_slices" / "high"
        call_low_dir.mkdir(parents=True, exist_ok=True)
        call_medium_dir.mkdir(parents=True, exist_ok=True)
        call_high_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        payload = _read_result_payload(candidate)
        tasks = payload.get("tasks") if isinstance(payload, dict) else []
        if not isinstance(tasks, list):
            tasks = []

        summary = {
            "iteration": iteration,
            "candidate_id": candidate.candidate_id,
            "scaffold_name": candidate.scaffold_name,
            "passrate": candidate.passrate,
            "average_score": candidate.average_score,
            "token_consuming": candidate.token_consuming,
            "avg_token_consuming": candidate.avg_token_consuming,
            "entered_frontier": candidate.candidate_id in frontier_ids,
            "result_path": candidate.result_path,
            "config": candidate.config,
        }
        summaries.append(summary)

        low_payload = _trace_slice(candidate, tasks, limit=3, slice_level="low")
        medium_payload = _trace_slice(candidate, tasks, limit=10, slice_level="medium")
        high_payload = _trace_slice(candidate, tasks, limit=None, slice_level="high")
        _write_json(
            low_dir / f"{candidate.candidate_id}.json",
            low_payload,
        )
        _write_json(
            medium_dir / f"{candidate.candidate_id}.json",
            medium_payload,
        )
        _write_json(
            high_dir / f"{candidate.candidate_id}.json",
            high_payload,
        )
        if (
            call_low_dir is not None
            and call_medium_dir is not None
            and call_high_dir is not None
        ):
            _write_json(call_low_dir / f"{candidate.candidate_id}.json", low_payload)
            _write_json(call_medium_dir / f"{candidate.candidate_id}.json", medium_payload)
            _write_json(call_high_dir / f"{candidate.candidate_id}.json", high_payload)

    if call_dir is not None:
        call_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            call_dir / "eval_summary.json",
            {
                "iteration": iteration,
                "candidates": summaries,
            },
        )


def write_diff_digest(*, call_dir: Path) -> None:
    """Write a compact placeholder digest from the saved diff patch."""

    diff_path = call_dir / "diff.patch"
    digest_path = call_dir / "diff_digest.md"
    if not diff_path.exists():
        digest_path.write_text("No diff patch was captured.\n", encoding="utf-8")
        return

    text = diff_path.read_text(encoding="utf-8", errors="replace")
    changed_files = []
    for line in text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                changed_files.append(parts[3].removeprefix("b/"))
    payload = ["# Diff Digest", ""]
    if changed_files:
        payload.append("Changed files:")
        payload.extend(f"- {path}" for path in sorted(set(changed_files)))
    else:
        payload.append("No file-level diff entries were captured.")
    payload.append("")
    payload.append(f"Patch size: {len(text)} characters")
    digest_path.write_text("\n".join(payload) + "\n", encoding="utf-8")


def _read_result_payload(candidate: CandidateResult) -> dict[str, Any]:
    try:
        return json.loads(Path(candidate.result_path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _trace_slice(
    candidate: CandidateResult,
    tasks: list[object],
    *,
    limit: int | None,
    slice_level: str,
) -> dict[str, Any]:
    normalized = [item for item in tasks if isinstance(item, dict)]
    failures = [item for item in normalized if not item.get("passed")]
    successes = [item for item in normalized if item.get("passed")]
    failures.sort(key=lambda item: (float(item.get("score", 0.0)), -_task_tokens(item)))
    successes.sort(key=lambda item: (-float(item.get("score", 0.0)), -_task_tokens(item)))

    if limit is None:
        selected = failures + successes
    else:
        selected = failures[:limit]
        if len(selected) < limit:
            selected.extend(successes[: max(0, limit - len(selected))])

    return {
        "candidate_id": candidate.candidate_id,
        "passrate": candidate.passrate,
        "avg_token_consuming": candidate.avg_token_consuming,
        "slice_level": slice_level,
        "case_limit": limit,
        "cases": [_case_preview(item) for item in selected],
    }


def _case_preview(task: dict[str, Any]) -> dict[str, Any]:
    retrieved = task.get("retrieved") or []
    if not isinstance(retrieved, list):
        retrieved = []
    out: dict[str, Any] = {
        "task_id": task.get("task_id"),
        "question": task.get("question"),
        "gold_answer": task.get("gold_answer"),
        "prediction": task.get("prediction"),
        "score": task.get("score"),
        "passed": task.get("passed"),
        "prompt_tokens": task.get("prompt_tokens"),
        "completion_tokens": task.get("completion_tokens"),
        "retrieved_preview": [
            _hit_preview(hit)
            for hit in retrieved
            if isinstance(hit, dict)
        ],
    }
    return out


def _hit_preview(hit: dict[str, Any]) -> dict[str, Any]:
    text = str(hit.get("text") or "")
    return {
        "text": text,
        "score": hit.get("score"),
        "source": hit.get("source"),
        "metadata": hit.get("metadata") or {},
    }


def _task_tokens(task: dict[str, Any]) -> int:
    return int(task.get("prompt_tokens") or 0) + int(task.get("completion_tokens") or 0)
