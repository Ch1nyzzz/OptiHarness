#!/usr/bin/env python3
"""Evaluate one generated candidate from a JSON spec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from memomemo.dynamic import load_candidate_scaffold
from memomemo.evaluation import EvaluationRunner
from memomemo.locomo import load_locomo_examples, prepare_locomo, select_split
from memomemo.model import DEFAULT_BASE_URL, DEFAULT_MODEL
from memomemo.scaffolds.base import ScaffoldConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("warmup", "train", "test"))
    parser.add_argument("--limit", type=int, default=0)
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
    candidate = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    project_root = Path.cwd()
    scaffold = load_candidate_scaffold(candidate, project_root=project_root)
    config = ScaffoldConfig(
        top_k=int(candidate.get("top_k", 12)),
        window=int(candidate.get("window", 1)),
        extra=dict(candidate.get("extra") or {}),
    )
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
    result = runner.evaluate_scaffold(
        scaffold=scaffold,
        scaffold_name=str(candidate["name"]),
        config=config,
        candidate_id=str(candidate["candidate_id"]),
    )
    summary = {
        "split": args.split,
        "limit": args.limit,
        "count": len(examples),
        "model": args.model,
        "base_url": args.base_url,
        "max_context_chars": args.max_context_chars,
        "max_eval_workers": args.eval_workers,
        "candidate": result.to_dict(),
        "candidate_spec": candidate,
    }
    (args.out / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
