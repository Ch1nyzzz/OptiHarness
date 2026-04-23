"""LongMemEval optimization entry point using the LOCOMO memory scaffold system."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memomemo.evaluation import EvaluationRunner
from memomemo.benchmark_workspaces import LONGMEMEVAL_WORKSPACE_SPEC, BenchmarkWorkspaceSpec
from memomemo.longmemeval import (
    DEFAULT_LONGMEMEVAL_JUDGE_API_KEY_ENV,
    DEFAULT_LONGMEMEVAL_SCAFFOLDS,
    DEFAULT_LONGMEMEVAL_JUDGE_BASE_URL,
    DEFAULT_LONGMEMEVAL_JUDGE_MODEL,
    LongMemEvalJudge,
    _fallback_score_run,
    load_longmemeval_examples,
    prepare_longmemeval,
    run_longmemeval_frontier,
    select_split,
)
from memomemo.optimizer import LocomoOptimizer, OptimizerConfig
from memomemo.schemas import LocomoExample


@dataclass(frozen=True)
class LongMemEvalOptimizerConfig(OptimizerConfig):
    """Configuration for LongMemEval runs over the MemGPT memory scaffold."""

    dataset_variant: str = "s"
    data_path: Path | None = None
    split_path: Path | None = None
    question_types: tuple[str, ...] = ()
    judge_model: str = DEFAULT_LONGMEMEVAL_JUDGE_MODEL
    judge_base_url: str = DEFAULT_LONGMEMEVAL_JUDGE_BASE_URL
    judge_api_key: str | None = None
    judge_timeout_s: int = 300
    use_llm_judge: bool = True
    scaffolds: tuple[str, ...] = DEFAULT_LONGMEMEVAL_SCAFFOLDS
    progressive_target_system: str = "memgpt"


class LongMemEvalOptimizer(LocomoOptimizer):
    """Proposer loop for LongMemEval using the LOCOMO/MemGPT scaffold base."""

    workspace_spec: BenchmarkWorkspaceSpec = LONGMEMEVAL_WORKSPACE_SPEC
    config: LongMemEvalOptimizerConfig

    def __init__(self, config: LongMemEvalOptimizerConfig) -> None:
        super().__init__(config)

    def _load_examples(self) -> list[LocomoExample]:
        data_path = self.config.data_path
        if data_path is None or not data_path.exists():
            prepare_longmemeval(
                variant=self.config.dataset_variant,
                dest=data_path,
            )
        examples = load_longmemeval_examples(
            data_path=data_path,
            variant=self.config.dataset_variant,
            question_types=self.config.question_types,
        )
        selected = select_split(
            examples,
            split=self.config.split,
            variant=self.config.dataset_variant,
            split_path=self.config.split_path,
        )
        if self.config.limit:
            selected = selected[: self.config.limit]
        return selected

    def _run_seed_frontier(self) -> dict[str, Any]:
        return run_longmemeval_frontier(
            split=self.config.split,
            limit=self.config.limit,
            out_dir=self.run_dir,
            variant=self.config.dataset_variant,
            data_path=self.config.data_path,
            split_path=self.config.split_path,
            question_types=self.config.question_types,
            judge_model=self.config.judge_model,
            judge_base_url=self.config.judge_base_url,
            judge_api_key=self.config.judge_api_key,
            judge_timeout_s=self.config.judge_timeout_s,
            use_llm_judge=self.config.use_llm_judge,
            model=self.config.model,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout_s=self.config.eval_timeout_s,
            dry_run=self.config.dry_run,
            max_context_chars=self.config.max_context_chars,
            max_eval_workers=self.config.max_eval_workers,
            pareto_quality_threshold=self.config.pareto_quality_threshold,
            scaffolds=self.config.scaffolds,
            scaffold_extra=self.config.scaffold_extra,
        )

    def _benchmark_prompt_name(self) -> str:
        return "LongMemEval long-term memory QA"

    def _raw_data_policy_name(self) -> str:
        return "raw LongMemEval data"

    def _make_evaluation_runner(self, examples: list[LocomoExample]) -> EvaluationRunner:
        score_run = _fallback_score_run
        if self.config.use_llm_judge and not self.config.dry_run:
            judge_api_key = self.config.judge_api_key or os.environ.get(
                DEFAULT_LONGMEMEVAL_JUDGE_API_KEY_ENV,
                "",
            )
            if not judge_api_key:
                raise ValueError(
                    "LongMemEval LLM-as-judge requires a Together API key. "
                    f"Set {DEFAULT_LONGMEMEVAL_JUDGE_API_KEY_ENV} or pass "
                    "--longmemeval-judge-api-key."
                )
            score_run = LongMemEvalJudge(
                model=self.config.judge_model,
                base_url=self.config.judge_base_url,
                api_key=judge_api_key,
                timeout_s=self.config.judge_timeout_s,
            ).score_run
        return EvaluationRunner(
            examples=examples,
            out_dir=self.run_dir,
            model=self.config.model,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout_s=self.config.eval_timeout_s,
            dry_run=self.config.dry_run,
            max_context_chars=self.config.max_context_chars,
            max_eval_workers=self.config.max_eval_workers,
            score_run=score_run,
        )

    def _candidate_extra_defaults(self) -> dict[str, object]:
        scoring_method = (
            "longmemeval_llm_judge"
            if self.config.use_llm_judge and not self.config.dry_run
            else "token_f1"
        )
        out: dict[str, object] = {
            "benchmark": "longmemeval",
            "scoring_method": scoring_method,
        }
        if scoring_method == "longmemeval_llm_judge":
            out["judge_model"] = self.config.judge_model
        return out
