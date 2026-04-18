"""Claude-proposer optimization loop for MemoMemo."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from memomemo.baseline import load_baseline_candidates
from memomemo.claude_runner import run_claude_prompt
from memomemo.dynamic import load_candidate_scaffold
from memomemo.evolution import EvolutionRunner, run_initial_frontier
from memomemo.locomo import default_data_path, load_locomo_examples, prepare_locomo, select_split
from memomemo.model import DEFAULT_BASE_URL, DEFAULT_MODEL
from memomemo.pareto import ParetoPoint, pareto_frontier, save_frontier
from memomemo.proposer_prompt import build_proposer_prompt
from memomemo.scaffolds import DEFAULT_MEMORY_SCAFFOLDS, DEFAULT_SCAFFOLD_TOP_KS
from memomemo.scaffolds.base import ScaffoldConfig
from memomemo.schemas import CandidateResult, LocomoExample


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for the Claude proposer loop."""

    run_id: str
    out_dir: Path
    iterations: int = 3
    split: str = "train"
    limit: int = 40
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str = "EMPTY"
    eval_timeout_s: int = 300
    claude_model: str = "claude-sonnet-4-6"
    propose_timeout_s: int = 2400
    dry_run: bool = False
    max_context_chars: int = 6000
    max_eval_workers: int = 1
    skip_scaffold_eval: bool = False
    baseline_dir: Path | None = None


class MemoOptimizer:
    """Meta-harness-style proposer loop for memory scaffolds."""

    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config
        self.project_root = Path(__file__).resolve().parents[2]
        self.run_dir = config.out_dir
        self.pending_eval_path = self.run_dir / "pending_eval.json"
        self.frontier_path = self.run_dir / "pareto_frontier.json"
        self.summary_path = self.run_dir / "evolution_summary.jsonl"

    def run(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        examples = self._load_examples()

        candidates: list[CandidateResult] = []
        if self.config.baseline_dir is not None:
            baseline_candidates = load_baseline_candidates(
                self.config.baseline_dir,
                split=self.config.split,
                scaffolds=DEFAULT_MEMORY_SCAFFOLDS,
                top_k_by_scaffold=DEFAULT_SCAFFOLD_TOP_KS,
            )
            if not baseline_candidates:
                raise ValueError(
                    f"No baseline candidates found for split '{self.config.split}' "
                    f"under {self.config.baseline_dir}"
                )
            for item in baseline_candidates:
                candidate = CandidateResult.from_dict(item)
                candidates.append(candidate)
                self._append_summary(iteration=0, candidate=candidate)
        elif not self.config.skip_scaffold_eval:
            scaffold_summary = run_initial_frontier(
                split=self.config.split,
                limit=self.config.limit,
                out_dir=self.run_dir,
                model=self.config.model,
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout_s=self.config.eval_timeout_s,
                dry_run=self.config.dry_run,
                max_context_chars=self.config.max_context_chars,
                max_eval_workers=self.config.max_eval_workers,
            )
            for item in scaffold_summary.get("candidates", []):
                candidate = CandidateResult.from_dict(item)
                candidates.append(candidate)
                self._append_summary(iteration=0, candidate=candidate)
        else:
            candidates.extend(self._load_existing_candidates())

        self._save_frontier_from_candidates(candidates)

        for iteration in range(1, self.config.iterations + 1):
            if self.pending_eval_path.exists():
                self.pending_eval_path.unlink()
            (self.run_dir / "reports").mkdir(parents=True, exist_ok=True)
            prompt = build_proposer_prompt(
                run_id=self.config.run_id,
                iteration=iteration,
                run_dir=self.run_dir,
                pending_eval_path=self.pending_eval_path,
                frontier_path=self.frontier_path,
                summary_path=self.summary_path,
                split=self.config.split,
                limit=self.config.limit,
            )
            result = run_claude_prompt(
                prompt,
                cwd=self.project_root,
                log_dir=self.run_dir / "claude_sessions",
                name=f"iter_{iteration:03d}",
                model=self.config.claude_model,
                timeout_s=self.config.propose_timeout_s,
            )
            if result.returncode != 0 or result.timed_out or not self.pending_eval_path.exists():
                self._append_event(
                    {
                        "iteration": iteration,
                        "event": "proposer_failed",
                        "returncode": result.returncode,
                        "timed_out": result.timed_out,
                        "stderr": result.stderr[:1000],
                    }
                )
                continue

            pending = json.loads(self.pending_eval_path.read_text(encoding="utf-8"))
            proposed = pending.get("candidates") or []
            evaluated = self._evaluate_proposed(iteration, proposed, examples)
            candidates.extend(evaluated)
            self._save_frontier_from_candidates(candidates)

        final_summary = {
            "run_id": self.config.run_id,
            "out_dir": str(self.run_dir),
            "iterations": self.config.iterations,
            "candidate_count": len(candidates),
            "pareto_frontier_path": str(self.frontier_path),
        }
        (self.run_dir / "optimizer_summary.json").write_text(
            json.dumps(final_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return final_summary

    def _load_examples(self) -> list[LocomoExample]:
        if not default_data_path().exists():
            prepare_locomo()
        examples = load_locomo_examples()
        selected = select_split(examples, split=self.config.split)
        if self.config.limit:
            selected = selected[: self.config.limit]
        return selected

    def _evaluate_proposed(
        self,
        iteration: int,
        proposed: list[dict[str, Any]],
        examples: list[LocomoExample],
    ) -> list[CandidateResult]:
        runner = EvolutionRunner(
            examples=examples,
            out_dir=self.run_dir,
            model=self.config.model,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout_s=self.config.eval_timeout_s,
            dry_run=self.config.dry_run,
            max_context_chars=self.config.max_context_chars,
            max_eval_workers=self.config.max_eval_workers,
        )
        results: list[CandidateResult] = []
        for raw in proposed:
            try:
                scaffold = load_candidate_scaffold(raw, project_root=self.project_root)
            except Exception as exc:  # noqa: BLE001 - log and continue
                self._append_event(
                    {
                        "iteration": iteration,
                        "event": "candidate_import_failed",
                        "candidate": raw,
                        "error": str(exc),
                    }
                )
                continue

            top_k_values = raw.get("top_k", [8])
            if isinstance(top_k_values, int):
                top_k_values = [top_k_values]
            for top_k in top_k_values:
                config = ScaffoldConfig(
                    top_k=int(top_k),
                    window=int(raw.get("window", 1)),
                    extra=dict(raw.get("extra") or {}),
                )
                candidate_name = str(raw.get("name") or scaffold.name)
                candidate_id = f"iter{iteration:03d}_{candidate_name}_top{top_k}"
                try:
                    result = runner.evaluate_scaffold(
                        scaffold=scaffold,
                        scaffold_name=candidate_name,
                        config=config,
                        candidate_id=candidate_id,
                    )
                except Exception as exc:  # noqa: BLE001 - log and continue
                    self._append_event(
                        {
                            "iteration": iteration,
                            "event": "candidate_eval_failed",
                            "candidate": raw,
                            "candidate_id": candidate_id,
                            "error": str(exc),
                        }
                    )
                    continue
                results.append(result)
                self._append_summary(iteration=iteration, candidate=result, proposal=raw)
        return results

    def _load_existing_candidates(self) -> list[CandidateResult]:
        out: list[CandidateResult] = []
        for path in sorted((self.run_dir / "candidate_results").glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                out.append(CandidateResult.from_dict(payload["candidate"]))
            except Exception:
                continue
        return out

    def _save_frontier_from_candidates(self, candidates: list[CandidateResult]) -> None:
        points = [
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
            for item in candidates
        ]
        save_frontier(self.frontier_path, points)

    def _append_summary(
        self,
        *,
        iteration: int,
        candidate: CandidateResult,
        proposal: dict[str, Any] | None = None,
    ) -> None:
        frontier = pareto_frontier(
            [
                ParetoPoint(
                    candidate_id=candidate.candidate_id,
                    scaffold_name=candidate.scaffold_name,
                    passrate=candidate.passrate,
                    token_consuming=candidate.token_consuming,
                    avg_token_consuming=candidate.avg_token_consuming,
                    average_score=candidate.average_score,
                    result_path=candidate.result_path,
                    config=candidate.config,
                )
            ]
        )
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "iteration": iteration,
            "candidate": candidate.to_dict(),
            "proposal": proposal or {},
            "self_frontier": [asdict(item) for item in frontier],
        }
        self._append_event(row)

    def _append_event(self, row: dict[str, Any]) -> None:
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with self.summary_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
