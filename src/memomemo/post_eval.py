"""Post-eval compact artifacts for future proposer context."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memomemo.schemas import CandidateResult

TRACE_SLICE_LIMITS = {
    "low": 10,
    "medium": 30,
    "high": None,
}


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

    current_candidate_ids = {candidate.candidate_id for candidate in candidates}
    previously_passed_task_ids = _load_previously_passed_task_ids(
        run_dir,
        iteration=iteration,
        exclude_candidate_ids=current_candidate_ids,
    )

    summaries: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
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
            "is_best_passrate": candidate.candidate_id in frontier_ids,
            "entered_frontier": candidate.candidate_id in frontier_ids,
            "result_path": candidate.result_path,
            "config": candidate.config,
        }
        summaries.append(summary)
        diagnostic = _retrieval_diagnostics(
            candidate,
            tasks,
            iteration=iteration,
            is_best=candidate.candidate_id in frontier_ids,
        )
        diagnostics.append(diagnostic)

        low_payload = _trace_slice(
            candidate,
            tasks,
            limit=TRACE_SLICE_LIMITS["low"],
            slice_level="low",
            previously_passed_task_ids=previously_passed_task_ids,
        )
        medium_payload = _trace_slice(
            candidate,
            tasks,
            limit=TRACE_SLICE_LIMITS["medium"],
            slice_level="medium",
            previously_passed_task_ids=previously_passed_task_ids,
        )
        high_payload = _trace_slice(
            candidate,
            tasks,
            limit=TRACE_SLICE_LIMITS["high"],
            slice_level="high",
            previously_passed_task_ids=previously_passed_task_ids,
        )
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
            eval_dir = call_dir / "eval"
            _copy_result_payload(
                candidate,
                eval_dir / "candidate_result.json",
            )
            _write_json(
                eval_dir / "candidate_result.compact.json",
                _compact_result_payload(candidate, tasks),
            )
            _write_json(eval_dir / "retrieval_diagnostics.json", diagnostic)

    if call_dir is not None:
        call_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            call_dir / "eval" / "eval_summary.json",
            {
                "iteration": iteration,
                "candidates": summaries,
            },
        )
        _write_json(
            call_dir / "eval_summary.json",
            {
                "iteration": iteration,
                "candidates": summaries,
            },
        )

    _append_candidate_score_table(run_dir, summaries)
    _append_retrieval_diagnostics_summary(run_dir, diagnostics)


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


def _load_previously_passed_task_ids(
    run_dir: Path,
    *,
    iteration: int,
    exclude_candidate_ids: set[str],
) -> set[str]:
    out: set[str] = set()
    result_dir = run_dir / "candidate_results"
    if not result_dir.exists():
        return out

    for path in sorted(result_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue

        candidate_payload = payload.get("candidate")
        candidate_id = ""
        if isinstance(candidate_payload, dict):
            candidate_id = str(candidate_payload.get("candidate_id") or "")
        if candidate_id in exclude_candidate_ids:
            continue
        candidate_iteration = _candidate_iteration(candidate_id)
        if candidate_iteration is not None and candidate_iteration >= iteration:
            continue

        tasks = payload.get("tasks")
        if not isinstance(tasks, list):
            continue
        for task in tasks:
            if not isinstance(task, dict) or not task.get("passed"):
                continue
            task_id = _task_id(task)
            if task_id is not None:
                out.add(task_id)
    return out


def _candidate_iteration(candidate_id: str) -> int | None:
    if not candidate_id.startswith("iter"):
        return None
    digits = []
    for char in candidate_id[4:]:
        if not char.isdigit():
            break
        digits.append(char)
    if not digits:
        return None
    return int("".join(digits))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _copy_result_payload(candidate: CandidateResult, dest: Path) -> None:
    payload = _read_result_payload(candidate)
    if not payload:
        payload = {"candidate": candidate.to_dict(), "tasks": []}
    _write_json(dest, payload)


def _compact_result_payload(
    candidate: CandidateResult,
    tasks: list[object],
    *,
    task_limit: int = 80,
    retrieved_limit: int = 5,
    hit_text_limit: int = 500,
) -> dict[str, Any]:
    normalized = [item for item in tasks if isinstance(item, dict)]
    compact_tasks = []
    for task in normalized[:task_limit]:
        retrieved = task.get("retrieved") or []
        if not isinstance(retrieved, list):
            retrieved = []
        compact_tasks.append(
            {
                "task_id": task.get("task_id"),
                "question": task.get("question"),
                "prediction": task.get("prediction"),
                "gold_answer": task.get("gold_answer"),
                "score": task.get("score"),
                "passed": task.get("passed"),
                "retrieved": [
                    _compact_hit(hit, hit_text_limit=hit_text_limit)
                    for hit in retrieved[:retrieved_limit]
                    if isinstance(hit, dict)
                ],
            }
        )
    return {
        "candidate": {
            "candidate_id": candidate.candidate_id,
            "scaffold_name": candidate.scaffold_name,
            "passrate": candidate.passrate,
            "average_score": candidate.average_score,
            "token_consuming": candidate.token_consuming,
            "avg_token_consuming": candidate.avg_token_consuming,
            "count": candidate.count,
            "config": candidate.config,
            "result_path": candidate.result_path,
        },
        "task_count": len(normalized),
        "tasks": compact_tasks,
    }


def _compact_hit(hit: dict[str, Any], *, hit_text_limit: int) -> dict[str, Any]:
    text = str(hit.get("text") or "")
    if len(text) > hit_text_limit:
        text = text[:hit_text_limit] + "..."
    metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    return {
        "text": text,
        "score": hit.get("score"),
        "source": hit.get("source"),
        "memory_tier": metadata.get("memory_tier") or metadata.get("tier"),
        "tool": metadata.get("tool"),
        "rank": metadata.get("rank"),
        "passage_id": metadata.get("passage_id"),
        "turn_indices": metadata.get("turn_indices"),
        "search_mode": metadata.get("search_mode"),
        "metadata": metadata,
    }


def _retrieval_diagnostics(
    candidate: CandidateResult,
    tasks: list[object],
    *,
    iteration: int,
    is_best: bool,
) -> dict[str, Any]:
    normalized = [item for item in tasks if isinstance(item, dict)]
    failures = [item for item in normalized if not item.get("passed")]
    low_score = [item for item in normalized if float(item.get("score") or 0.0) < 0.5]
    retrieved_but_failed = [
        item for item in failures if isinstance(item.get("retrieved"), list) and item["retrieved"]
    ]
    retrieval_misses = [
        item for item in failures if not (isinstance(item.get("retrieved"), list) and item["retrieved"])
    ]
    tier_counts: dict[str, int] = {}
    top_hit_tiers: dict[str, int] = {}
    retrieved_counts: list[int] = []
    token_counts: list[int] = []

    for task in normalized:
        retrieved = task.get("retrieved") or []
        if not isinstance(retrieved, list):
            retrieved = []
        retrieved_counts.append(len(retrieved))
        token_counts.append(_task_tokens(task))
        for index, hit in enumerate(retrieved):
            if not isinstance(hit, dict):
                continue
            metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
            tier = str(metadata.get("memory_tier") or metadata.get("tier") or hit.get("source") or "unknown")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            if index == 0:
                top_hit_tiers[tier] = top_hit_tiers.get(tier, 0) + 1

    return {
        "iteration": iteration,
        "candidate_id": candidate.candidate_id,
        "scaffold_name": candidate.scaffold_name,
        "passrate": candidate.passrate,
        "average_score": candidate.average_score,
        "token_consuming": candidate.token_consuming,
        "is_best_passrate": is_best,
        "failed_task_count": len(failures),
        "low_score_task_count": len(low_score),
        "retrieved_but_failed_count": len(retrieved_but_failed),
        "likely_retrieval_miss_count": len(retrieval_misses),
        "memory_tier_distribution": dict(sorted(tier_counts.items())),
        "top_hit_tier_distribution": dict(sorted(top_hit_tiers.items())),
        "average_retrieved_count": (
            sum(retrieved_counts) / len(retrieved_counts) if retrieved_counts else 0.0
        ),
        "token_distribution": {
            "min": min(token_counts) if token_counts else 0,
            "max": max(token_counts) if token_counts else 0,
            "average": sum(token_counts) / len(token_counts) if token_counts else 0.0,
        },
        "failed_tasks": [_case_preview(item) for item in failures[:20]],
        "retrieved_but_failed_tasks": [
            _case_preview(item) for item in retrieved_but_failed[:20]
        ],
        "likely_retrieval_miss_tasks": [
            _case_preview(item) for item in retrieval_misses[:20]
        ],
    }


def _append_candidate_score_table(
    run_dir: Path,
    summaries: list[dict[str, Any]],
) -> None:
    path = run_dir / "candidate_score_table.json"
    rows = _read_json_list(path)
    by_id = {
        str(row.get("candidate_id")): row
        for row in rows
        if isinstance(row, dict) and row.get("candidate_id")
    }
    for summary in summaries:
        config = summary.get("config") if isinstance(summary.get("config"), dict) else {}
        extra = config.get("extra") if isinstance(config.get("extra"), dict) else {}
        by_id[str(summary["candidate_id"])] = {
            "iteration": summary["iteration"],
            "candidate_id": summary["candidate_id"],
            "scaffold_name": summary["scaffold_name"],
            "passrate": summary["passrate"],
            "average_score": summary["average_score"],
            "token_consuming": summary["token_consuming"],
            "source_family": extra.get("source_family"),
            "build_tag": extra.get("build_tag") or config.get("build_tag"),
            "result_path": summary["result_path"],
            "iteration_dir": str(run_dir / "proposer_calls" / f"iter_{summary['iteration']:03d}"),
            "is_best_passrate": summary["is_best_passrate"],
        }
    ordered = sorted(
        by_id.values(),
        key=lambda row: (int(row.get("iteration") or 0), str(row.get("candidate_id") or "")),
    )
    _write_json(path, ordered)


def _append_retrieval_diagnostics_summary(
    run_dir: Path,
    diagnostics: list[dict[str, Any]],
) -> None:
    path = run_dir / "retrieval_diagnostics_summary.json"
    rows = _read_json_list(path)
    by_id = {
        str(row.get("candidate_id")): row
        for row in rows
        if isinstance(row, dict) and row.get("candidate_id")
    }
    for item in diagnostics:
        by_id[str(item["candidate_id"])] = {
            key: item[key]
            for key in (
                "iteration",
                "candidate_id",
                "scaffold_name",
                "passrate",
                "average_score",
                "token_consuming",
                "is_best_passrate",
                "failed_task_count",
                "low_score_task_count",
                "retrieved_but_failed_count",
                "likely_retrieval_miss_count",
                "memory_tier_distribution",
                "top_hit_tier_distribution",
                "average_retrieved_count",
                "token_distribution",
            )
            if key in item
        }
    ordered = sorted(
        by_id.values(),
        key=lambda row: (int(row.get("iteration") or 0), str(row.get("candidate_id") or "")),
    )
    _write_json(path, ordered)


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _trace_slice(
    candidate: CandidateResult,
    tasks: list[object],
    *,
    limit: int | None,
    slice_level: str,
    previously_passed_task_ids: set[str] | None = None,
) -> dict[str, Any]:
    normalized = [item for item in tasks if isinstance(item, dict)]
    previous_passes = previously_passed_task_ids or set()
    failures = [item for item in normalized if not item.get("passed")]
    successes = [item for item in normalized if item.get("passed")]
    failures.sort(key=lambda item: (float(item.get("score", 0.0)), -_task_tokens(item)))
    successes.sort(key=lambda item: (-float(item.get("score", 0.0)), -_task_tokens(item)))
    progressive_failures = [
        item for item in failures if _task_id(item) not in previous_passes
    ]
    regressed_failures = [
        item for item in failures if _task_id(item) in previous_passes
    ]

    if limit is None:
        selected = failures + successes
    else:
        selected = (progressive_failures + regressed_failures)[:limit]
        if len(selected) < limit:
            selected.extend(successes[: max(0, limit - len(selected))])

    return {
        "candidate_id": candidate.candidate_id,
        "passrate": candidate.passrate,
        "avg_token_consuming": candidate.avg_token_consuming,
        "slice_level": slice_level,
        "case_limit": limit,
        "selection_strategy": (
            "progressive_unresolved_failures_first" if limit is not None else "all_cases"
        ),
        "previously_passed_task_count": len(previous_passes),
        "progressive_failure_count": len(progressive_failures),
        "cases": [_case_preview(item) for item in selected],
    }


def _task_id(task: dict[str, Any]) -> str | None:
    value = task.get("task_id")
    if value is None:
        return None
    return str(value)


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
