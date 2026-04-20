#!/usr/bin/env python3
"""Evaluate selected generated memory scaffolds on a LOCOMO split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from memomemo.dynamic import load_candidate_scaffold
from memomemo.evaluation import EvaluationRunner
from memomemo.locomo import load_locomo_examples, prepare_locomo, select_split
from memomemo.model import DEFAULT_BASE_URL, DEFAULT_MODEL
from memomemo.scaffolds.base import ScaffoldConfig


DEFAULT_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "test_iter008_prf_stemmed_temporal_bm25_top8",
        "name": "prf_stemmed_temporal_bm25",
        "module": "prf_stemmed_temporal_bm25",
        "class": "PRFStemmedTemporalBM25Scaffold",
        "top_k": 8,
        "window": 1,
    },
    {
        "candidate_id": "test_iter009_prefix_prf_bm25_top8",
        "name": "prefix_prf_bm25",
        "module": "prefix_prf_bm25",
        "class": "PrefixPRFBM25Scaffold",
        "top_k": 8,
        "window": 1,
    },
    {
        "candidate_id": "test_iter007_extended_temporal_bm25_top8",
        "name": "extended_temporal_bm25",
        "module": "extended_temporal_bm25",
        "class": "ExtendedTemporalBM25Scaffold",
        "top_k": 8,
        "window": 1,
    },
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("warmup", "train", "test"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/test_top3_passrate_generated"))
    parser.add_argument("--candidate-root", type=Path, default=Path("runs/locomo_memory_opt/generated"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--eval-workers", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_path = Path("data/locomo/locomo10.json")
    if not data_path.exists():
        prepare_locomo()
    examples = select_split(load_locomo_examples(), split=args.split)
    if args.limit:
        examples = examples[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    runner = EvaluationRunner(
        examples=examples,
        out_dir=args.out,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_s=args.timeout_s,
        max_context_chars=args.max_context_chars,
        max_eval_workers=args.eval_workers,
        force=args.force,
    )

    project_root = Path(__file__).resolve().parents[1]
    candidates = []
    for raw in DEFAULT_CANDIDATES:
        raw = dict(raw)
        raw.setdefault("candidate_root", str(args.candidate_root))
        scaffold = load_candidate_scaffold(raw, project_root=project_root)
        config = ScaffoldConfig(top_k=int(raw["top_k"]), window=int(raw["window"]))
        result = runner.evaluate_scaffold(
            scaffold=scaffold,
            scaffold_name=str(raw["name"]),
            config=config,
            candidate_id=str(raw["candidate_id"]),
        )
        candidates.append(result.to_dict())
        print(json.dumps(result.to_dict(), ensure_ascii=False), flush=True)

    summary = {
        "split": args.split,
        "limit": args.limit,
        "count": len(examples),
        "model": args.model,
        "base_url": args.base_url,
        "max_context_chars": args.max_context_chars,
        "max_eval_workers": args.eval_workers,
        "candidates": candidates,
    }
    (args.out / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
