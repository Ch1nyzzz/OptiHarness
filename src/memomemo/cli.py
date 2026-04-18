"""Command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from memomemo.baseline import (
    DEFAULT_BASELINE_REPEATS,
    DEFAULT_BASELINE_SPLITS,
    run_baseline_suite,
)
from memomemo.evolution import run_initial_frontier
from memomemo.locomo import prepare_locomo
from memomemo.model import DEFAULT_BASE_URL, DEFAULT_MODEL
from memomemo.optimizer import MemoOptimizer, OptimizerConfig
from memomemo.scaffolds import DEFAULT_MEMORY_SCAFFOLDS, DEFAULT_SCAFFOLD_TOP_KS, available_scaffolds


def main() -> int:
    parser = argparse.ArgumentParser(prog="memomemo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    locomo = subparsers.add_parser("locomo")
    locomo_sub = locomo.add_subparsers(dest="locomo_command", required=True)
    prepare = locomo_sub.add_parser("prepare")
    prepare.add_argument("--source", type=Path, default=None)
    prepare.add_argument("--dest", type=Path, default=None)
    prepare.add_argument("--allow-download", action="store_true")
    prepare.add_argument("--warmup-size", type=int, default=40)
    prepare.add_argument("--train-size", type=int, default=40)
    prepare.add_argument("--seed", type=int, default=13)

    evolve = subparsers.add_parser("evolve")
    evolve.add_argument("--split", choices=("warmup", "train", "test"), default="train")
    evolve.add_argument("--limit", type=int, default=0)
    evolve.add_argument("--out", type=Path, default=Path("runs/locomo_memory_scaffold_run"))
    evolve.add_argument("--model", default=DEFAULT_MODEL)
    evolve.add_argument("--base-url", default=DEFAULT_BASE_URL)
    evolve.add_argument("--api-key", default="EMPTY")
    evolve.add_argument("--timeout-s", type=int, default=300)
    evolve.add_argument("--dry-run", action="store_true")
    evolve.add_argument("--max-context-chars", type=int, default=6000)
    evolve.add_argument("--eval-workers", type=int, default=1)
    evolve.add_argument(
        "--force",
        action="store_true",
        help="Rerun existing candidate result files instead of reusing them.",
    )
    evolve.add_argument(
        "--scaffolds",
        default=",".join(DEFAULT_MEMORY_SCAFFOLDS),
        help=f"Comma-separated scaffolds. Available: {', '.join(available_scaffolds())}",
    )
    evolve.add_argument(
        "--top-k",
        default=None,
        help=(
            "Comma-separated top-k variants. Omit to use fixed scaffold defaults: "
            f"{_format_scaffold_top_k_defaults()}."
        ),
    )

    baseline = subparsers.add_parser("baseline")
    baseline.add_argument(
        "--splits",
        default=",".join(DEFAULT_BASELINE_SPLITS),
        help="Comma-separated splits to evaluate.",
    )
    baseline.add_argument("--repeats", type=int, default=DEFAULT_BASELINE_REPEATS)
    baseline.add_argument("--limit", type=int, default=0)
    baseline.add_argument("--out", type=Path, default=Path("runs/baselines"))
    baseline.add_argument("--model", default=DEFAULT_MODEL)
    baseline.add_argument("--base-url", default=DEFAULT_BASE_URL)
    baseline.add_argument("--api-key", default="EMPTY")
    baseline.add_argument("--timeout-s", type=int, default=300)
    baseline.add_argument("--dry-run", action="store_true")
    baseline.add_argument("--max-context-chars", type=int, default=6000)
    baseline.add_argument("--eval-workers", type=int, default=1)
    baseline.add_argument(
        "--scaffolds",
        default=",".join(DEFAULT_MEMORY_SCAFFOLDS),
        help=f"Comma-separated scaffolds. Available: {', '.join(available_scaffolds())}",
    )
    baseline.add_argument(
        "--top-k",
        default=None,
        help=(
            "Comma-separated top-k variants. Omit to use fixed scaffold defaults: "
            f"{_format_scaffold_top_k_defaults()}."
        ),
    )
    baseline.add_argument(
        "--force",
        action="store_true",
        help="Rerun existing baseline repeat directories instead of reusing them.",
    )

    optimize = subparsers.add_parser("optimize")
    optimize.add_argument("--run-id", default="locomo_memory_opt")
    optimize.add_argument("--iterations", type=int, default=3)
    optimize.add_argument("--split", choices=("warmup", "train", "test"), default="train")
    optimize.add_argument("--limit", type=int, default=40)
    optimize.add_argument("--out", type=Path, default=None)
    optimize.add_argument("--model", default=DEFAULT_MODEL)
    optimize.add_argument("--base-url", default=DEFAULT_BASE_URL)
    optimize.add_argument("--api-key", default="EMPTY")
    optimize.add_argument("--eval-timeout-s", type=int, default=300)
    optimize.add_argument("--claude-model", default="claude-sonnet-4-6")
    optimize.add_argument("--propose-timeout-s", type=int, default=2400)
    optimize.add_argument("--dry-run", action="store_true")
    optimize.add_argument("--max-context-chars", type=int, default=6000)
    optimize.add_argument("--eval-workers", type=int, default=1)
    optimize.add_argument(
        "--skip-scaffold-eval",
        action="store_true",
        help="Resume from existing candidate_results instead of rerunning built-in scaffolds.",
    )
    optimize.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="Load precomputed baseline candidates from this directory.",
    )

    args = parser.parse_args()
    if args.command == "locomo" and args.locomo_command == "prepare":
        payload = prepare_locomo(
            dest=args.dest,
            source=args.source,
            allow_download=args.allow_download,
            warmup_size=args.warmup_size,
            train_size=args.train_size,
            seed=args.seed,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.command == "evolve":
        selected_scaffolds = _csv(args.scaffolds)
        top_k = None if args.top_k is None else [int(item) for item in _csv(args.top_k)]
        payload = run_initial_frontier(
            split=args.split,
            limit=args.limit,
            out_dir=args.out,
            scaffolds=selected_scaffolds,
            top_k_variants=top_k,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_s=args.timeout_s,
            dry_run=args.dry_run,
            max_context_chars=args.max_context_chars,
            max_eval_workers=args.eval_workers,
            force=args.force,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.command == "baseline":
        selected_splits = _csv(args.splits)
        selected_scaffolds = _csv(args.scaffolds)
        top_k = None if args.top_k is None else [int(item) for item in _csv(args.top_k)]
        payload = run_baseline_suite(
            out_dir=args.out,
            splits=selected_splits,
            repeats=args.repeats,
            limit=args.limit,
            scaffolds=selected_scaffolds,
            top_k_variants=top_k,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_s=args.timeout_s,
            dry_run=args.dry_run,
            max_context_chars=args.max_context_chars,
            max_eval_workers=args.eval_workers,
            force=args.force,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.command == "optimize":
        out_dir = args.out or Path("runs") / args.run_id
        optimizer = MemoOptimizer(
            OptimizerConfig(
                run_id=args.run_id,
                out_dir=out_dir,
                iterations=args.iterations,
                split=args.split,
                limit=args.limit,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                eval_timeout_s=args.eval_timeout_s,
                claude_model=args.claude_model,
                propose_timeout_s=args.propose_timeout_s,
                dry_run=args.dry_run,
                max_context_chars=args.max_context_chars,
                max_eval_workers=args.eval_workers,
                skip_scaffold_eval=args.skip_scaffold_eval,
                baseline_dir=args.baseline_dir,
            )
        )
        payload = optimizer.run()
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    return 1


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _format_scaffold_top_k_defaults() -> str:
    return ", ".join(
        f"{scaffold}=top{DEFAULT_SCAFFOLD_TOP_KS[scaffold]}"
        for scaffold in DEFAULT_MEMORY_SCAFFOLDS
    )


if __name__ == "__main__":
    raise SystemExit(main())
