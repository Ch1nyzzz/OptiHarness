"""Command line interface."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from memomemo.benchmark_tasks import TASK_CHOICES, normalize_task_name, task_spec
from memomemo.baseline import (
    DEFAULT_BASELINE_REPEATS,
    DEFAULT_BASELINE_SPLITS,
    run_baseline_suite,
)
from memomemo.evaluation import run_initial_frontier
from memomemo.locomo import prepare_locomo
from memomemo.locomo_optimizer import LocomoOptimizer, LocomoOptimizerConfig
from memomemo.longmemeval import (
    DEFAULT_LONGMEMEVAL_SCAFFOLDS,
    DEFAULT_LONGMEMEVAL_JUDGE_BASE_URL,
    DEFAULT_LONGMEMEVAL_JUDGE_MODEL,
    prepare_longmemeval,
    run_longmemeval_frontier,
)
from memomemo.longmemeval_optimizer import (
    LongMemEvalOptimizer,
    LongMemEvalOptimizerConfig,
)
from memomemo.model import DEFAULT_BASE_URL, DEFAULT_MODEL
from memomemo.claude_runner import DEFAULT_CODEX_MODEL
from memomemo.scaffolds import (
    DEFAULT_BASELINE_SCAFFOLDS,
    DEFAULT_EVOLUTION_SEED_SCAFFOLDS,
    DEFAULT_SCAFFOLD_TOP_KS,
    available_scaffolds,
)
from memomemo.text_classification import (
    ALL_TEXT_CLASSIFICATION_TASKS,
    DEFAULT_TEXT_CLASSIFICATION_BASELINES,
    DEFAULT_TEXT_CLASSIFICATION_DATASETS,
    DEFAULT_TEXT_CLASSIFICATION_SEEDS,
    DEFAULT_TEXT_CLASSIFICATION_SPLITS,
    available_text_classification_memories,
    run_text_classification_benchmark,
)
from memomemo.text_classification_optimizer import (
    TextClassificationOptimizer,
    TextClassificationOptimizerConfig,
)
from memomemo.tau_banking import DEFAULT_TAU2_ROOT, DEFAULT_TAU_BANKING_RETRIEVAL_CONFIGS
from memomemo.tau_banking_optimizer import TauBankingOptimizer, TauBankingOptimizerConfig


def main() -> int:
    parser = argparse.ArgumentParser(prog="optiharness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    locomo = subparsers.add_parser("locomo")
    locomo_sub = locomo.add_subparsers(dest="locomo_command", required=True)
    prepare = locomo_sub.add_parser("prepare")
    prepare.add_argument("--source", type=Path, default=None)
    prepare.add_argument("--dest", type=Path, default=None)
    prepare.add_argument("--allow-download", action="store_true")
    prepare.add_argument("--warmup-size", type=int, default=0)
    prepare.add_argument("--train-size", type=int, default=80)
    prepare.add_argument(
        "--train-sample-id",
        default="auto",
        help="Sample id to draw the train split from, or 'auto' for the largest available sample.",
    )
    prepare.add_argument("--seed", type=int, default=13)

    longmemeval = subparsers.add_parser("longmemeval")
    longmemeval_sub = longmemeval.add_subparsers(
        dest="longmemeval_command",
        required=True,
    )
    lme_prepare = longmemeval_sub.add_parser("prepare")
    lme_prepare.add_argument("--variant", choices=("s", "m", "oracle"), default="s")
    lme_prepare.add_argument("--source", type=Path, default=None)
    lme_prepare.add_argument("--dest", type=Path, default=None)
    lme_prepare.add_argument("--allow-download", action="store_true")
    lme_prepare.add_argument("--warmup-size", type=int, default=0)
    lme_prepare.add_argument("--train-size", type=int, default=100)
    lme_prepare.add_argument("--seed", type=int, default=13)

    lme_benchmark = longmemeval_sub.add_parser("benchmark")
    lme_benchmark.add_argument("--variant", choices=("s", "m", "oracle"), default="s")
    lme_benchmark.add_argument("--data-path", type=Path, default=None)
    lme_benchmark.add_argument("--split-path", type=Path, default=None)
    lme_benchmark.add_argument("--split", choices=("warmup", "train", "test"), default="train")
    lme_benchmark.add_argument("--limit", type=int, default=0)
    lme_benchmark.add_argument(
        "--out",
        type=Path,
        default=Path("runs/longmemeval_memory_scaffold_run"),
    )
    lme_benchmark.add_argument("--model", default=DEFAULT_MODEL)
    lme_benchmark.add_argument("--base-url", default=DEFAULT_BASE_URL)
    lme_benchmark.add_argument("--api-key", default="EMPTY")
    lme_benchmark.add_argument("--timeout-s", type=int, default=300)
    lme_benchmark.add_argument("--dry-run", action="store_true")
    lme_benchmark.add_argument("--max-context-chars", type=int, default=6000)
    lme_benchmark.add_argument("--eval-workers", type=int, default=1)
    lme_benchmark.add_argument("--judge-model", default=DEFAULT_LONGMEMEVAL_JUDGE_MODEL)
    lme_benchmark.add_argument("--judge-base-url", default=DEFAULT_LONGMEMEVAL_JUDGE_BASE_URL)
    lme_benchmark.add_argument("--judge-api-key", default=None)
    lme_benchmark.add_argument("--judge-timeout-s", type=int, default=300)
    lme_benchmark.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Use local token/F1 scoring instead of LongMemEval's LLM-as-judge scorer.",
    )
    lme_benchmark.add_argument("--pareto-quality-threshold", type=float, default=0.125)
    lme_benchmark.add_argument("--force", action="store_true")
    lme_benchmark.add_argument(
        "--scaffolds",
        default=",".join(DEFAULT_LONGMEMEVAL_SCAFFOLDS),
        help=f"Comma-separated scaffolds. Default: {', '.join(DEFAULT_LONGMEMEVAL_SCAFFOLDS)}.",
    )
    lme_benchmark.add_argument("--top-k", default=None)
    lme_benchmark.add_argument("--question-types", default="")
    lme_benchmark.add_argument("--scaffold-extra-json", default=None)

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
        "--pareto-quality-threshold",
        type=float,
        default=0.125,
        help="Passrate gap above which lower-quality cheap candidates are excluded from Pareto.",
    )
    evolve.add_argument(
        "--force",
        action="store_true",
        help="Rerun existing candidate result files instead of reusing them.",
    )
    evolve.add_argument(
        "--scaffolds",
        default=",".join(DEFAULT_EVOLUTION_SEED_SCAFFOLDS),
        help=f"Comma-separated scaffolds. Available: {', '.join(available_scaffolds())}",
    )
    evolve.add_argument(
        "--top-k",
        default=None,
        help=(
            "Comma-separated top-k variants. Omit to use fixed scaffold defaults: "
            f"{_format_scaffold_top_k_defaults(DEFAULT_EVOLUTION_SEED_SCAFFOLDS)}."
        ),
    )
    evolve.add_argument(
        "--scaffold-extra-json",
        default=None,
        help="JSON object of per-scaffold extra config, or @path to a JSON file.",
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
        default=",".join(DEFAULT_BASELINE_SCAFFOLDS),
        help=f"Comma-separated scaffolds. Available: {', '.join(available_scaffolds())}",
    )
    baseline.add_argument(
        "--top-k",
        default=None,
        help=(
            "Comma-separated top-k variants. Omit to use fixed scaffold defaults: "
            f"{_format_scaffold_top_k_defaults(DEFAULT_BASELINE_SCAFFOLDS)}."
        ),
    )
    baseline.add_argument(
        "--scaffold-extra-json",
        default=None,
        help="JSON object of per-scaffold extra config, or @path to a JSON file.",
    )
    baseline.add_argument(
        "--force",
        action="store_true",
        help="Rerun existing baseline repeat directories instead of reusing them.",
    )

    text_cls = subparsers.add_parser("text-classification")
    text_cls_sub = text_cls.add_subparsers(dest="text_classification_command", required=True)
    text_benchmark = text_cls_sub.add_parser("benchmark")
    text_benchmark.add_argument(
        "--datasets",
        default=",".join(DEFAULT_TEXT_CLASSIFICATION_DATASETS),
        help=f"Comma-separated datasets. Available: {', '.join(ALL_TEXT_CLASSIFICATION_TASKS)}.",
    )
    text_benchmark.add_argument(
        "--memory-systems",
        default=",".join(DEFAULT_TEXT_CLASSIFICATION_BASELINES),
        help=(
            "Comma-separated memory systems. Available: "
            f"{', '.join(available_text_classification_memories())}."
        ),
    )
    text_benchmark.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_TEXT_CLASSIFICATION_SEEDS),
        help="Comma-separated shuffle seeds.",
    )
    text_benchmark.add_argument(
        "--num-train",
        type=int,
        default=None,
        help="Override train split size for every selected dataset. Default: paper-specific sizes.",
    )
    text_benchmark.add_argument(
        "--num-val",
        type=int,
        default=None,
        help="Override val split size for every selected dataset. Default: paper-specific sizes.",
    )
    text_benchmark.add_argument(
        "--num-test",
        type=int,
        default=None,
        help="Override test split size for every selected dataset. Default: paper-specific sizes.",
    )
    text_benchmark.add_argument(
        "--mode",
        choices=("online", "offline"),
        default="offline",
        help="Training mode: online predict-then-learn, or offline learn labels then eval.",
    )
    text_benchmark.add_argument("--num-epochs", type=int, default=1)
    text_benchmark.add_argument(
        "--out",
        type=Path,
        default=Path("runs/text_classification_baselines"),
    )
    text_benchmark.add_argument("--model", default=DEFAULT_MODEL)
    text_benchmark.add_argument("--base-url", default=DEFAULT_BASE_URL)
    text_benchmark.add_argument("--api-key", default="EMPTY")
    text_benchmark.add_argument("--timeout-s", type=int, default=300)
    text_benchmark.add_argument("--dry-run", action="store_true")
    text_benchmark.add_argument("--temperature", type=float, default=0.0)
    text_benchmark.add_argument("--eval-workers", type=int, default=1)
    text_benchmark.add_argument(
        "--pareto-quality-threshold",
        type=float,
        default=0.0,
        help="Quality gap threshold passed to OptiHarness's Pareto writer.",
    )
    text_benchmark.add_argument(
        "--force",
        action="store_true",
        help="Rerun existing row result files instead of reusing them.",
    )

    optimize = subparsers.add_parser("optimize")
    optimize.add_argument("--run-id", default=None)
    optimize.add_argument(
        "--task",
        choices=TASK_CHOICES,
        default=None,
        help="Benchmark task to optimize. Backward-compatible aliases are accepted.",
    )
    task_flags = optimize.add_mutually_exclusive_group()
    task_flags.add_argument(
        "--locomo",
        dest="task_locomo",
        action="store_true",
        help="Optimize the LOCOMO memory benchmark.",
    )
    task_flags.add_argument(
        "--tau3",
        dest="task_tau3",
        action="store_true",
        help="Optimize the tau3 banking_knowledge benchmark.",
    )
    task_flags.add_argument(
        "--text-classification",
        dest="task_text_classification",
        action="store_true",
        help="Optimize the text-classification benchmark.",
    )
    task_flags.add_argument(
        "--longmemeval",
        dest="task_longmemeval",
        action="store_true",
        help="Optimize the LongMemEval memory benchmark.",
    )
    optimize.add_argument("--iterations", type=int, default=20)
    optimize.add_argument("--split", choices=("warmup", "train", "test"), default="train")
    optimize.add_argument("--limit", type=int, default=0)
    optimize.add_argument("--out", type=Path, default=None)
    optimize.add_argument("--model", default=DEFAULT_MODEL)
    optimize.add_argument("--base-url", default=DEFAULT_BASE_URL)
    optimize.add_argument("--api-key", default="EMPTY")
    optimize.add_argument("--eval-timeout-s", type=int, default=300)
    optimize.add_argument(
        "--proposer-agent",
        choices=("claude", "codex", "kimi"),
        default="claude",
        help="Code agent used to generate candidates.",
    )
    optimize.add_argument("--claude-model", default="claude-sonnet-4-6")
    optimize.add_argument("--codex-model", default=DEFAULT_CODEX_MODEL)
    optimize.add_argument(
        "--kimi-model",
        default="",
        help="Kimi model override. Omit to use the default model from Kimi config.",
    )
    optimize.add_argument("--propose-timeout-s", type=int, default=2400)
    optimize.add_argument("--dry-run", action="store_true")
    optimize.add_argument("--max-context-chars", type=int, default=6000)
    optimize.add_argument("--eval-workers", type=int, default=1)
    optimize.add_argument(
        "--scaffolds",
        default=None,
        help=f"Comma-separated seed scaffolds. Available: {', '.join(available_scaffolds())}",
    )
    optimize.add_argument(
        "--scaffold-extra-json",
        default=None,
        help="JSON object of per-scaffold seed extra config, or @path to a JSON file.",
    )
    optimize.add_argument(
        "--selection-policy",
        choices=("default", "progressive"),
        default="default",
        help=(
            "Use fixed-high scoped context (default) or adaptive progressive "
            "context loading."
        ),
    )
    optimize.add_argument(
        "--proposer-sandbox",
        choices=("none", "docker"),
        default="docker",
        help="Run proposer code agents directly or inside a Docker filesystem sandbox.",
    )
    optimize.add_argument(
        "--proposer-docker-image",
        default="",
        help="Docker image used when --proposer-sandbox=docker.",
    )
    optimize.add_argument(
        "--proposer-docker-workspace",
        default="/workspace",
        help="Container path for the mounted proposer workspace.",
    )
    optimize.add_argument(
        "--proposer-docker-env",
        action="append",
        default=[],
        help="Extra environment variable name to pass into the proposer container.",
    )
    optimize.add_argument(
        "--proposer-docker-mount",
        action="append",
        default=[],
        help="Extra Docker volume mount spec, for example ~/.codex:/root/.codex:ro.",
    )
    optimize.add_argument(
        "--proposer-docker-kimi-cli-kind",
        choices=("claude", "legacy"),
        default="claude",
        help="Kimi CLI style available inside the proposer Docker image.",
    )
    optimize.add_argument(
        "--proposer-docker-user",
        default="",
        help="Optional Docker user spec for proposer containers, for example 1000:1000.",
    )
    optimize.add_argument(
        "--proposer-docker-home",
        default="",
        help="Optional HOME value inside the proposer container.",
    )
    optimize.add_argument(
        "--pareto-quality-threshold",
        type=float,
        default=0.125,
        help="Passrate gap above which lower-quality cheap candidates are excluded from Pareto.",
    )
    optimize.add_argument(
        "--skip-scaffold-eval",
        action="store_true",
        help="Resume from existing candidate_results instead of rerunning built-in scaffolds.",
    )
    optimize.add_argument(
        "--force",
        action="store_true",
        help="Rerun task-specific baseline/seed evaluations instead of reusing cached rows.",
    )
    optimize.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="Load precomputed baseline candidates from this directory.",
    )
    optimize.add_argument(
        "--text-classification-mode",
        choices=("online", "offline"),
        default="offline",
        help="Text-classification optimization task variant.",
    )
    optimize.add_argument(
        "--text-classification-datasets",
        default=",".join(DEFAULT_TEXT_CLASSIFICATION_DATASETS),
        help="Comma-separated text-classification datasets.",
    )
    optimize.add_argument(
        "--text-classification-seeds",
        default=",".join(str(seed) for seed in DEFAULT_TEXT_CLASSIFICATION_SEEDS),
        help="Comma-separated text-classification shuffle seeds.",
    )
    optimize.add_argument(
        "--text-classification-num-train",
        type=int,
        default=None,
    )
    optimize.add_argument(
        "--text-classification-num-val",
        type=int,
        default=None,
    )
    optimize.add_argument(
        "--text-classification-num-test",
        type=int,
        default=None,
    )
    optimize.add_argument("--text-classification-temperature", type=float, default=0.0)
    optimize.add_argument("--longmemeval-variant", choices=("s", "m", "oracle"), default="s")
    optimize.add_argument("--longmemeval-data-path", type=Path, default=None)
    optimize.add_argument("--longmemeval-split-path", type=Path, default=None)
    optimize.add_argument("--longmemeval-question-types", default="")
    optimize.add_argument("--longmemeval-judge-model", default=DEFAULT_LONGMEMEVAL_JUDGE_MODEL)
    optimize.add_argument("--longmemeval-judge-base-url", default=DEFAULT_LONGMEMEVAL_JUDGE_BASE_URL)
    optimize.add_argument("--longmemeval-judge-api-key", default=None)
    optimize.add_argument("--longmemeval-judge-timeout-s", type=int, default=300)
    optimize.add_argument("--longmemeval-no-llm-judge", action="store_true")
    optimize.add_argument(
        "--tau3-root",
        type=Path,
        default=None,
        help="tau2-bench checkout root. Defaults to TAU3_BENCH_ROOT/TAU2_BENCH_ROOT or references/vendor/tau2-bench.",
    )
    optimize.add_argument(
        "--tau3-python",
        default=None,
        help="Python command for tau2-bench. Defaults to TAU_PYTHON or python.",
    )
    optimize.add_argument(
        "--tau3-retrieval-configs",
        default=",".join(DEFAULT_TAU_BANKING_RETRIEVAL_CONFIGS),
        help="Comma-separated tau3 retrieval configs.",
    )
    optimize.add_argument("--tau3-retrieval-config-kwargs-json", default="{}")
    optimize.add_argument("--tau3-task-split-name", default="base")
    optimize.add_argument("--tau3-task-ids", default="")
    optimize.add_argument("--tau3-num-tasks", type=int, default=5)
    optimize.add_argument("--tau3-num-trials", type=int, default=1)
    optimize.add_argument("--tau3-max-steps", type=int, default=200)
    optimize.add_argument("--tau3-max-errors", type=int, default=10)
    optimize.add_argument("--tau3-max-concurrency", type=int, default=1)
    optimize.add_argument("--tau3-seed", type=int, default=300)
    optimize.add_argument("--tau3-agent-llm", default="openai/gpt-4.1-mini")
    optimize.add_argument("--tau3-user-llm", default="openai/gpt-4.1-mini")
    optimize.add_argument("--tau3-agent-llm-args-json", default='{"temperature": 0.0}')
    optimize.add_argument("--tau3-user-llm-args-json", default='{"temperature": 0.0}')
    optimize.add_argument("--tau3-process-timeout-s", type=int, default=0)

    args = parser.parse_args()
    if args.command == "locomo" and args.locomo_command == "prepare":
        payload = prepare_locomo(
            dest=args.dest,
            source=args.source,
            allow_download=args.allow_download,
            warmup_size=args.warmup_size,
            train_size=args.train_size,
            train_sample_id=args.train_sample_id,
            seed=args.seed,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.command == "longmemeval" and args.longmemeval_command == "prepare":
        payload = prepare_longmemeval(
            variant=args.variant,
            dest=args.dest,
            source=args.source,
            allow_download=args.allow_download,
            warmup_size=args.warmup_size,
            train_size=args.train_size,
            seed=args.seed,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.command == "longmemeval" and args.longmemeval_command == "benchmark":
        selected_scaffolds = _csv(args.scaffolds)
        top_k = None if args.top_k is None else [int(item) for item in _csv(args.top_k)]
        scaffold_extra = _scaffold_extra(args.scaffold_extra_json)
        payload = run_longmemeval_frontier(
            split=args.split,
            limit=args.limit,
            out_dir=args.out,
            variant=args.variant,
            data_path=args.data_path,
            split_path=args.split_path,
            question_types=tuple(_csv(args.question_types)),
            scaffolds=selected_scaffolds,
            top_k_variants=top_k,
            scaffold_extra=scaffold_extra,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_s=args.timeout_s,
            dry_run=args.dry_run,
            max_context_chars=args.max_context_chars,
            max_eval_workers=args.eval_workers,
            force=args.force,
            pareto_quality_threshold=args.pareto_quality_threshold,
            judge_model=args.judge_model,
            judge_base_url=args.judge_base_url,
            judge_api_key=args.judge_api_key,
            judge_timeout_s=args.judge_timeout_s,
            use_llm_judge=not args.no_llm_judge,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.command == "evolve":
        selected_scaffolds = _csv(args.scaffolds)
        top_k = None if args.top_k is None else [int(item) for item in _csv(args.top_k)]
        scaffold_extra = _scaffold_extra(args.scaffold_extra_json)
        payload = run_initial_frontier(
            split=args.split,
            limit=args.limit,
            out_dir=args.out,
            scaffolds=selected_scaffolds,
            top_k_variants=top_k,
            scaffold_extra=scaffold_extra,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_s=args.timeout_s,
            dry_run=args.dry_run,
            max_context_chars=args.max_context_chars,
            max_eval_workers=args.eval_workers,
            force=args.force,
            pareto_quality_threshold=args.pareto_quality_threshold,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.command == "baseline":
        selected_splits = _csv(args.splits)
        selected_scaffolds = _csv(args.scaffolds)
        top_k = None if args.top_k is None else [int(item) for item in _csv(args.top_k)]
        scaffold_extra = _scaffold_extra(args.scaffold_extra_json)
        payload = run_baseline_suite(
            out_dir=args.out,
            splits=selected_splits,
            repeats=args.repeats,
            limit=args.limit,
            scaffolds=selected_scaffolds,
            top_k_variants=top_k,
            scaffold_extra=scaffold_extra,
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

    if args.command == "text-classification" and args.text_classification_command == "benchmark":
        payload = run_text_classification_benchmark(
            out_dir=args.out,
            datasets=_csv(args.datasets),
            memory_systems=_csv(args.memory_systems),
            seeds=[int(item) for item in _csv(args.seeds)],
            num_train=args.num_train,
            num_val=args.num_val,
            num_test=args.num_test,
            mode=args.mode,
            num_epochs=args.num_epochs,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_s=args.timeout_s,
            dry_run=args.dry_run,
            temperature=args.temperature,
            max_eval_workers=args.eval_workers,
            force=args.force,
            pareto_quality_threshold=args.pareto_quality_threshold,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.command == "optimize":
        try:
            selected_task = _optimize_task(args)
        except ValueError as exc:
            parser.error(str(exc))
        selected_spec = task_spec(selected_task)
        run_id = args.run_id or selected_spec.default_run_id
        out_dir = args.out or Path("runs") / run_id
        if selected_task == "text_classification":
            optimizer = TextClassificationOptimizer(
                TextClassificationOptimizerConfig(
                    run_id=run_id,
                    out_dir=out_dir,
                    iterations=args.iterations,
                    mode=args.text_classification_mode,
                    datasets=tuple(_csv(args.text_classification_datasets)),
                    seeds=tuple(int(item) for item in _csv(args.text_classification_seeds)),
                    num_train=args.text_classification_num_train,
                    num_val=args.text_classification_num_val,
                    num_test=args.text_classification_num_test,
                    model=args.model,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    eval_timeout_s=args.eval_timeout_s,
                    proposer_agent=args.proposer_agent,
                    claude_model=args.claude_model,
                    codex_model=args.codex_model,
                    kimi_model=args.kimi_model,
                    propose_timeout_s=args.propose_timeout_s,
                    dry_run=args.dry_run,
                    temperature=args.text_classification_temperature,
                    max_eval_workers=args.eval_workers,
                    skip_baseline_eval=args.skip_scaffold_eval,
                    force=args.force,
                    selection_policy=args.selection_policy,
                    proposer_sandbox=args.proposer_sandbox,
                    proposer_docker_image=args.proposer_docker_image,
                    proposer_docker_workspace=args.proposer_docker_workspace,
                    proposer_docker_env=tuple(_csv_many(args.proposer_docker_env)),
                    proposer_docker_mount=tuple(args.proposer_docker_mount or ()),
                    proposer_docker_kimi_cli_kind=args.proposer_docker_kimi_cli_kind,
                    proposer_docker_user=args.proposer_docker_user,
                    proposer_docker_home=args.proposer_docker_home,
                )
            )
            payload = optimizer.run()
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

        if selected_task == "longmemeval":
            selected_scaffolds = (
                _csv(args.scaffolds)
                if args.scaffolds
                else list(DEFAULT_LONGMEMEVAL_SCAFFOLDS)
            )
            scaffold_extra = _scaffold_extra(args.scaffold_extra_json)
            optimizer = LongMemEvalOptimizer(
                LongMemEvalOptimizerConfig(
                    run_id=run_id,
                    out_dir=out_dir,
                    iterations=args.iterations,
                    split=args.split,
                    limit=args.limit,
                    dataset_variant=args.longmemeval_variant,
                    data_path=args.longmemeval_data_path,
                    split_path=args.longmemeval_split_path,
                    question_types=tuple(_csv(args.longmemeval_question_types)),
                    judge_model=args.longmemeval_judge_model,
                    judge_base_url=args.longmemeval_judge_base_url,
                    judge_api_key=args.longmemeval_judge_api_key,
                    judge_timeout_s=args.longmemeval_judge_timeout_s,
                    use_llm_judge=not args.longmemeval_no_llm_judge,
                    model=args.model,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    eval_timeout_s=args.eval_timeout_s,
                    proposer_agent=args.proposer_agent,
                    claude_model=args.claude_model,
                    codex_model=args.codex_model,
                    kimi_model=args.kimi_model,
                    propose_timeout_s=args.propose_timeout_s,
                    dry_run=args.dry_run,
                    max_context_chars=args.max_context_chars,
                    max_eval_workers=args.eval_workers,
                    skip_scaffold_eval=args.skip_scaffold_eval,
                    baseline_dir=args.baseline_dir,
                    scaffolds=tuple(selected_scaffolds),
                    scaffold_extra=scaffold_extra,
                    selection_policy=args.selection_policy,
                    pareto_quality_threshold=args.pareto_quality_threshold,
                    proposer_sandbox=args.proposer_sandbox,
                    proposer_docker_image=args.proposer_docker_image,
                    proposer_docker_workspace=args.proposer_docker_workspace,
                    proposer_docker_env=tuple(_csv_many(args.proposer_docker_env)),
                    proposer_docker_mount=tuple(args.proposer_docker_mount or ()),
                    proposer_docker_kimi_cli_kind=args.proposer_docker_kimi_cli_kind,
                    proposer_docker_user=args.proposer_docker_user,
                    proposer_docker_home=args.proposer_docker_home,
                )
            )
            payload = optimizer.run()
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

        if selected_task == "tau3":
            optimizer = TauBankingOptimizer(
                TauBankingOptimizerConfig(
                    run_id=run_id,
                    out_dir=out_dir,
                    iterations=args.iterations,
                    tau2_root=_tau3_root(args.tau3_root),
                    python_executable=_tau3_python(args.tau3_python),
                    retrieval_configs=tuple(_csv(args.tau3_retrieval_configs)),
                    retrieval_config_kwargs=_json_object(
                        args.tau3_retrieval_config_kwargs_json,
                        "--tau3-retrieval-config-kwargs-json",
                    ),
                    task_split_name=args.tau3_task_split_name,
                    task_ids=tuple(_csv(args.tau3_task_ids)),
                    num_tasks=None if args.tau3_num_tasks < 0 else args.tau3_num_tasks,
                    num_trials=args.tau3_num_trials,
                    max_steps=args.tau3_max_steps,
                    max_errors=args.tau3_max_errors,
                    max_concurrency=args.tau3_max_concurrency,
                    seed=args.tau3_seed,
                    agent_llm=args.tau3_agent_llm,
                    user_llm=args.tau3_user_llm,
                    agent_llm_args=_json_object(
                        args.tau3_agent_llm_args_json,
                        "--tau3-agent-llm-args-json",
                    ),
                    user_llm_args=_json_object(
                        args.tau3_user_llm_args_json,
                        "--tau3-user-llm-args-json",
                    ),
                    process_timeout_s=(
                        None if args.tau3_process_timeout_s <= 0 else args.tau3_process_timeout_s
                    ),
                    eval_timeout_s=args.eval_timeout_s,
                    proposer_agent=args.proposer_agent,
                    claude_model=args.claude_model,
                    codex_model=args.codex_model,
                    kimi_model=args.kimi_model,
                    propose_timeout_s=args.propose_timeout_s,
                    skip_baseline_eval=args.skip_scaffold_eval,
                    force=args.force,
                    proposer_sandbox=args.proposer_sandbox,
                    proposer_docker_image=args.proposer_docker_image,
                    proposer_docker_workspace=args.proposer_docker_workspace,
                    proposer_docker_env=tuple(_csv_many(args.proposer_docker_env)),
                    proposer_docker_mount=tuple(args.proposer_docker_mount or ()),
                    proposer_docker_kimi_cli_kind=args.proposer_docker_kimi_cli_kind,
                    proposer_docker_user=args.proposer_docker_user,
                    proposer_docker_home=args.proposer_docker_home,
                )
            )
            payload = optimizer.run()
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

        selected_scaffolds = (
            _csv(args.scaffolds)
            if args.scaffolds
            else list(DEFAULT_EVOLUTION_SEED_SCAFFOLDS)
        )
        scaffold_extra = _scaffold_extra(args.scaffold_extra_json)
        optimizer = LocomoOptimizer(
            LocomoOptimizerConfig(
                run_id=run_id,
                out_dir=out_dir,
                iterations=args.iterations,
                split=args.split,
                limit=args.limit,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                eval_timeout_s=args.eval_timeout_s,
                proposer_agent=args.proposer_agent,
                claude_model=args.claude_model,
                codex_model=args.codex_model,
                kimi_model=args.kimi_model,
                propose_timeout_s=args.propose_timeout_s,
                dry_run=args.dry_run,
                max_context_chars=args.max_context_chars,
                max_eval_workers=args.eval_workers,
                skip_scaffold_eval=args.skip_scaffold_eval,
                baseline_dir=args.baseline_dir,
                scaffolds=tuple(selected_scaffolds),
                scaffold_extra=scaffold_extra,
                selection_policy=args.selection_policy,
                pareto_quality_threshold=args.pareto_quality_threshold,
                proposer_sandbox=args.proposer_sandbox,
                proposer_docker_image=args.proposer_docker_image,
                proposer_docker_workspace=args.proposer_docker_workspace,
                proposer_docker_env=tuple(_csv_many(args.proposer_docker_env)),
                proposer_docker_mount=tuple(args.proposer_docker_mount or ()),
                proposer_docker_kimi_cli_kind=args.proposer_docker_kimi_cli_kind,
                proposer_docker_user=args.proposer_docker_user,
                proposer_docker_home=args.proposer_docker_home,
            )
        )
        payload = optimizer.run()
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    return 1


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_many(values: list[str] | tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for value in values:
        out.extend(_csv(value))
    return out


def _scaffold_extra(value: str | None) -> dict[str, dict[str, object]]:
    if not value:
        return {}
    text = Path(value[1:]).read_text(encoding="utf-8") if value.startswith("@") else value
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("--scaffold-extra-json must be a JSON object")
    out: dict[str, dict[str, object]] = {}
    for name, extra in payload.items():
        if not isinstance(extra, dict):
            raise ValueError(f"extra config for {name!r} must be a JSON object")
        out[str(name)] = dict(extra)
    return out


def _json_object(value: str, flag: str) -> dict[str, object]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError(f"{flag} must be a JSON object")
    return dict(payload)


def _optimize_task(args: argparse.Namespace) -> str:
    flag_tasks = [
        task
        for task, enabled in (
            ("locomo", args.task_locomo),
            ("longmemeval", args.task_longmemeval),
            ("tau3", args.task_tau3),
            ("text_classification", args.task_text_classification),
        )
        if enabled
    ]
    if args.task and flag_tasks:
        raise ValueError("Use either --task or one benchmark flag, not both.")
    if flag_tasks:
        return flag_tasks[0]
    return normalize_task_name(args.task)


def _tau3_root(value: Path | None) -> Path:
    if value is not None:
        return value
    env_value = os.environ.get("TAU3_BENCH_ROOT") or os.environ.get("TAU2_BENCH_ROOT")
    return Path(env_value) if env_value else DEFAULT_TAU2_ROOT


def _tau3_python(value: str | None) -> str:
    return value or os.environ.get("TAU_PYTHON") or "python"


def _format_scaffold_top_k_defaults(scaffolds: tuple[str, ...]) -> str:
    return ", ".join(
        f"{scaffold}=top{DEFAULT_SCAFFOLD_TOP_KS[scaffold]}"
        for scaffold in scaffolds
    )


if __name__ == "__main__":
    raise SystemExit(main())
