"""Reusable baseline evaluation suite."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Mapping

from memomemo.evolution import make_initial_candidate_grid, run_initial_frontier
from memomemo.model import DEFAULT_BASE_URL, DEFAULT_MODEL
from memomemo.scaffolds import DEFAULT_MEMORY_SCAFFOLDS
from memomemo.schemas import CandidateResult


DEFAULT_BASELINE_SPLITS = ("train", "test")
DEFAULT_BASELINE_REPEATS = 3


def run_baseline_suite(
    *,
    out_dir: Path,
    splits: Iterable[str] = DEFAULT_BASELINE_SPLITS,
    repeats: int = DEFAULT_BASELINE_REPEATS,
    limit: int = 0,
    scaffolds: Iterable[str] | None = None,
    top_k_variants: Iterable[int] | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = "EMPTY",
    timeout_s: int = 300,
    dry_run: bool = False,
    max_context_chars: int = 6000,
    max_eval_workers: int = 1,
    force: bool = False,
) -> dict[str, object]:
    """Evaluate built-in baselines across splits and repeated trials.

    Existing repeat directories are reused by default so the suite can be run
    before optimization and treated as a cache.
    """

    selected_splits = [str(split) for split in splits]
    selected_scaffolds = list(scaffolds or DEFAULT_MEMORY_SCAFFOLDS)
    selected_top_k = None if top_k_variants is None else [int(item) for item in top_k_variants]
    expected_pairs = {
        (scaffold_name, config.top_k)
        for scaffold_name, config, _ in make_initial_candidate_grid(
            scaffolds=selected_scaffolds,
            top_k_variants=selected_top_k,
        )
    }
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    runs: list[dict[str, object]] = []
    for split in selected_splits:
        for repeat in range(1, repeats + 1):
            repeat_dir = baseline_repeat_dir(out_dir, split=split, repeat=repeat)
            summary_path = repeat_dir / "run_summary.json"
            reused = False
            if summary_path.exists() and not force:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                reused = _summary_matches_request(
                    summary,
                    split=split,
                    limit=limit,
                    expected_pairs=expected_pairs,
                    model=model,
                    base_url=base_url,
                    dry_run=dry_run,
                    max_context_chars=max_context_chars,
                )
                if not reused:
                    summary = run_initial_frontier(
                        split=split,
                        limit=limit,
                        out_dir=repeat_dir,
                        scaffolds=selected_scaffolds,
                        top_k_variants=selected_top_k,
                        model=model,
                        base_url=base_url,
                        api_key=api_key,
                        timeout_s=timeout_s,
                        dry_run=dry_run,
                        max_context_chars=max_context_chars,
                        candidate_id_prefix=f"{split}_r{repeat:02d}_",
                        max_eval_workers=max_eval_workers,
                        force=True,
                    )
            else:
                summary = run_initial_frontier(
                    split=split,
                    limit=limit,
                    out_dir=repeat_dir,
                    scaffolds=selected_scaffolds,
                    top_k_variants=selected_top_k,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    timeout_s=timeout_s,
                    dry_run=dry_run,
                    max_context_chars=max_context_chars,
                    candidate_id_prefix=f"{split}_r{repeat:02d}_",
                    max_eval_workers=max_eval_workers,
                    force=force,
                )
            runs.append(
                {
                    "split": split,
                    "repeat": repeat,
                    "reused": reused,
                    "summary_path": str(summary_path),
                    "candidate_count": summary.get("candidate_count", 0),
                    "count": summary.get("count", 0),
                    "duration_s": summary.get("duration_s", 0.0),
                }
            )

    aggregate = {
        "out_dir": str(out_dir),
        "splits": selected_splits,
        "repeats": repeats,
        "limit": limit,
        "dry_run": dry_run,
        "model": model,
        "base_url": base_url,
        "max_context_chars": max_context_chars,
        "max_eval_workers": max_eval_workers,
        "scaffolds": selected_scaffolds,
        "top_k_variants": selected_top_k,
        "scaffold_top_k": {
            scaffold_name: top_k
            for scaffold_name, top_k in sorted(expected_pairs)
        },
        "duration_s": time.time() - started,
        "runs": runs,
    }
    (out_dir / "baseline_summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return aggregate


def baseline_repeat_dir(root: Path, *, split: str, repeat: int) -> Path:
    """Return the standard directory for one baseline split/repeat."""

    return root / split / f"repeat_{repeat:02d}"


def _summary_matches_request(
    summary: dict[str, object],
    *,
    split: str,
    limit: int,
    expected_pairs: set[tuple[str, int]],
    model: str,
    base_url: str,
    dry_run: bool,
    max_context_chars: int,
) -> bool:
    candidates = summary.get("candidates", [])
    if not isinstance(candidates, list):
        return False
    try:
        candidate_pairs = set()
        for item in candidates:
            if not isinstance(item, dict) or not isinstance(item.get("config"), dict):
                continue
            result_path = item.get("result_path")
            if not result_path or not Path(str(result_path)).exists():
                continue
            scaffold_name = item.get("scaffold_name") or item.get("seed_name")
            candidate_pairs.add((str(scaffold_name), int(item.get("config", {}).get("top_k"))))
    except (TypeError, ValueError):
        return False
    try:
        summary_limit = int(summary.get("limit", -1))
    except (TypeError, ValueError):
        return False
    return (
        summary.get("split") == split
        and summary_limit == limit
        and bool(summary.get("dry_run")) == dry_run
        and summary.get("model") == model
        and summary.get("base_url") == base_url
        and int(summary.get("max_context_chars", -1)) == max_context_chars
        and candidate_pairs == expected_pairs
    )


def load_baseline_candidates(
    root: Path,
    *,
    split: str | None = None,
    scaffolds: Iterable[str] | None = None,
    top_k_by_scaffold: Mapping[str, int] | None = None,
) -> list[dict[str, object]]:
    """Load candidate summaries from a baseline suite or a single run directory."""

    selected_scaffolds = {str(item) for item in scaffolds} if scaffolds is not None else None
    selected_top_k_by_scaffold = (
        {str(scaffold): int(top_k) for scaffold, top_k in top_k_by_scaffold.items()}
        if top_k_by_scaffold is not None
        else None
    )
    summary_paths: list[Path]
    suite_summary = root / "baseline_summary.json"
    if suite_summary.exists():
        suite = json.loads(suite_summary.read_text(encoding="utf-8"))
        summary_paths = [
            Path(item["summary_path"])
            for item in suite.get("runs", [])
            if split is None or item.get("split") == split
        ]
    elif (root / "run_summary.json").exists():
        summary_paths = [root / "run_summary.json"]
    else:
        summary_paths = sorted(root.glob("**/run_summary.json"))

    candidates: list[dict[str, object]] = []
    for path in summary_paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if split is not None and payload.get("split") != split:
            continue
        for candidate in payload.get("candidates", []):
            try:
                item = CandidateResult.from_dict(candidate)
            except (KeyError, TypeError):
                continue
            if selected_scaffolds is not None and item.scaffold_name not in selected_scaffolds:
                continue
            if selected_top_k_by_scaffold is not None:
                expected_top_k = selected_top_k_by_scaffold.get(item.scaffold_name)
                if expected_top_k is None or int(item.config.get("top_k", -1)) != expected_top_k:
                    continue
            candidates.append(item.to_dict())
    return candidates
