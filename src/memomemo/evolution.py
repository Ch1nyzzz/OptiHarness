"""Initial memory-scaffold evolution runner."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from memomemo.locomo import default_data_path, load_locomo_examples, prepare_locomo, select_split
from memomemo.metrics import passed, score_prediction
from memomemo.model import DEFAULT_BASE_URL, DEFAULT_MODEL, LocalModelClient
from memomemo.pareto import ParetoPoint, save_frontier
from memomemo.schemas import CandidateResult, LocomoExample, TaskResult
from memomemo.scaffolds import DEFAULT_MEMORY_SCAFFOLDS, DEFAULT_SCAFFOLD_TOP_KS, build_scaffold
from memomemo.scaffolds.base import MemoryScaffold, ScaffoldConfig


DEFAULT_TOP_K_VARIANTS = (4, 8, 12)


class EvolutionRunner:
    """Evaluate memory scaffold candidates and write a Pareto frontier."""

    def __init__(
        self,
        *,
        examples: list[LocomoExample],
        out_dir: Path,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "EMPTY",
        timeout_s: int = 300,
        dry_run: bool = False,
        max_context_chars: int = 6000,
        max_eval_workers: int = 1,
        force: bool = False,
    ) -> None:
        self.examples = examples
        self.out_dir = out_dir
        self.dry_run = dry_run
        self.max_context_chars = max_context_chars
        self.max_eval_workers = max(1, int(max_eval_workers))
        self.force = force
        self.client = LocalModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_s=timeout_s,
        )

    def evaluate_candidate(
        self,
        *,
        scaffold_name: str,
        config: ScaffoldConfig,
        candidate_id: str,
    ) -> CandidateResult:
        scaffold = build_scaffold(scaffold_name)
        return self.evaluate_scaffold(
            scaffold=scaffold,
            scaffold_name=scaffold_name,
            config=config,
            candidate_id=candidate_id,
        )

    def evaluate_scaffold(
        self,
        *,
        scaffold: MemoryScaffold,
        scaffold_name: str,
        config: ScaffoldConfig,
        candidate_id: str,
    ) -> CandidateResult:
        """Evaluate a built-in or dynamically proposed memory scaffold."""

        candidate_dir = self.out_dir / "candidate_results"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        result_path = candidate_dir / f"{candidate_id}.json"
        if not self.force:
            existing = _load_candidate_result(
                result_path,
                candidate_id=candidate_id,
                scaffold_name=scaffold_name,
                config=config,
            )
            if existing is not None:
                return existing

        if self.max_eval_workers == 1 or len(self.examples) <= 1:
            task_results = [self._evaluate_example(scaffold, config, example) for example in self.examples]
        else:
            with ThreadPoolExecutor(max_workers=self.max_eval_workers) as pool:
                task_results = list(
                    pool.map(
                        lambda example: self._evaluate_example(scaffold, config, example),
                        self.examples,
                    )
                )

        count = len(task_results)
        passrate = sum(1 for item in task_results if item.passed) / count if count else 0.0
        average_score = sum(item.score for item in task_results) / count if count else 0.0
        prompt_tokens = sum(item.prompt_tokens for item in task_results)
        completion_tokens = sum(item.completion_tokens for item in task_results)
        token_consuming = prompt_tokens + completion_tokens
        candidate = CandidateResult(
            candidate_id=candidate_id,
            scaffold_name=scaffold_name,
            passrate=passrate,
            average_score=average_score,
            token_consuming=token_consuming,
            avg_token_consuming=(token_consuming / count if count else 0.0),
            avg_prompt_tokens=(prompt_tokens / count if count else 0.0),
            avg_completion_tokens=(completion_tokens / count if count else 0.0),
            count=count,
            config=config.to_dict(),
            result_path=str(result_path),
        )
        payload = {
            "candidate": candidate.to_dict(),
            "tasks": [item.to_dict() for item in task_results],
        }
        result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return candidate

    def _evaluate_example(
        self,
        scaffold: MemoryScaffold,
        config: ScaffoldConfig,
        example: LocomoExample,
    ) -> TaskResult:
        run = scaffold.run(
            example,
            self.client,
            config,
            max_context_chars=self.max_context_chars,
            dry_run=self.dry_run,
        )
        score = score_prediction(run.prediction, example.answer)
        return TaskResult(
            task_id=example.task_id,
            question=example.question,
            gold_answer=example.answer,
            prediction=run.prediction,
            score=score,
            passed=passed(score),
            prompt_tokens=run.prompt_tokens,
            completion_tokens=run.completion_tokens,
            retrieved=[asdict(hit) for hit in run.retrieved],
        )


def make_initial_candidate_grid(
    *,
    scaffolds: Iterable[str] | None = None,
    top_k_variants: Iterable[int] | None = None,
) -> list[tuple[str, ScaffoldConfig, str]]:
    """Build the initial scaffold/config grid for evolution."""

    selected = list(scaffolds or DEFAULT_MEMORY_SCAFFOLDS)
    out: list[tuple[str, ScaffoldConfig, str]] = []
    for scaffold_name in selected:
        top_k_values = (
            [DEFAULT_SCAFFOLD_TOP_KS.get(scaffold_name, 8)]
            if top_k_variants is None
            else [int(item) for item in top_k_variants]
        )
        for top_k in top_k_values:
            config = ScaffoldConfig(top_k=int(top_k), window=1)
            out.append((scaffold_name, config, f"{scaffold_name}_top{top_k}"))
    return out


def run_initial_frontier(
    *,
    split: str = "train",
    limit: int = 0,
    out_dir: Path,
    scaffolds: Iterable[str] | None = None,
    top_k_variants: Iterable[int] | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = "EMPTY",
    timeout_s: int = 300,
    dry_run: bool = False,
    max_context_chars: int = 6000,
    candidate_id_prefix: str = "",
    max_eval_workers: int = 1,
    force: bool = False,
) -> dict[str, object]:
    """Evaluate initial scaffolds and write summary + Pareto frontier."""

    if not default_data_path().exists():
        prepare_locomo()
    all_examples = load_locomo_examples()
    examples = select_split(all_examples, split=split)
    if limit:
        examples = examples[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    runner = EvolutionRunner(
        examples=examples,
        out_dir=out_dir,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout_s=timeout_s,
        dry_run=dry_run,
        max_context_chars=max_context_chars,
        max_eval_workers=max_eval_workers,
        force=force,
    )

    selected_scaffolds = list(scaffolds or DEFAULT_MEMORY_SCAFFOLDS)
    selected_top_k = None if top_k_variants is None else [int(item) for item in top_k_variants]
    grid = make_initial_candidate_grid(
        scaffolds=selected_scaffolds,
        top_k_variants=selected_top_k,
    )
    started = time.time()
    summary_candidates: list[CandidateResult] = []
    for scaffold_name, config, candidate_id in grid:
        candidate_id = f"{candidate_id_prefix}{candidate_id}"
        summary_candidates.append(
            runner.evaluate_candidate(
                scaffold_name=scaffold_name,
                config=config,
                candidate_id=candidate_id,
            )
        )

    summary_candidates = sorted(
        summary_candidates,
        key=lambda item: (item.scaffold_name, int(item.config.get("top_k", 0)), item.candidate_id),
    )
    frontier_path = out_dir / "pareto_frontier.json"
    save_frontier(
        frontier_path,
        [
            ParetoPoint(
                candidate_id=item.candidate_id,
                scaffold_name=item.scaffold_name,
                passrate=item.passrate,
                token_consuming=item.token_consuming,
                avg_token_consuming=item.avg_token_consuming,
                average_score=item.average_score,
                result_path=item.result_path,
                config=item.config,
            )
            for item in summary_candidates
        ],
    )
    summary = {
        "split": split,
        "limit": limit,
        "count": len(examples),
        "dry_run": dry_run,
        "model": model,
        "base_url": base_url,
        "max_context_chars": max_context_chars,
        "max_eval_workers": max_eval_workers,
        "force": force,
        "scaffolds": selected_scaffolds,
        "top_k_variants": selected_top_k,
        "scaffold_top_k": {
            item.scaffold_name: int(item.config.get("top_k", 0))
            for item in summary_candidates
        },
        "duration_s": time.time() - started,
        "candidate_count": len(summary_candidates),
        "candidates": [candidate.to_dict() for candidate in summary_candidates],
        "pareto_frontier_path": str(frontier_path),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _load_all_candidate_results(candidate_dir: Path) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    if not candidate_dir.exists():
        return candidates
    for path in sorted(candidate_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            candidates.append(CandidateResult.from_dict(payload["candidate"]))
        except Exception:
            continue
    return candidates


def _load_candidate_result(
    result_path: Path,
    *,
    candidate_id: str,
    scaffold_name: str,
    config: ScaffoldConfig,
) -> CandidateResult | None:
    if not result_path.exists():
        return None
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        candidate = CandidateResult.from_dict(payload["candidate"])
    except Exception:
        return None
    if (
        candidate.candidate_id != candidate_id
        or candidate.scaffold_name != scaffold_name
        or candidate.config != config.to_dict()
    ):
        return None
    return candidate
