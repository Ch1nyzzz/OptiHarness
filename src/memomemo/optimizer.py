"""Claude-proposer optimization loop for MemoMemo."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from memomemo.bandit import (
    BanditArm,
    SOURCE_FAMILIES,
    cost_penalty,
    load_bandit_state,
    save_bandit_state,
    select_ucb_arm,
    update_bandit_state,
)
from memomemo.baseline import load_baseline_candidates
from memomemo.claude_runner import run_claude_prompt
from memomemo.dynamic import load_candidate_scaffold
from memomemo.evaluation import EvaluationRunner, run_initial_frontier
from memomemo.locomo import default_data_path, load_locomo_examples, prepare_locomo, select_split
from memomemo.model import DEFAULT_BASE_URL, DEFAULT_MODEL
from memomemo.pareto import ParetoPoint, pareto_frontier, save_frontier
from memomemo.post_eval import write_diff_digest, write_post_eval_artifacts
from memomemo.proposer_prompt import build_proposer_prompt, build_ucb_proposer_prompt
from memomemo.scaffolds import DEFAULT_EVOLUTION_SEED_SCAFFOLDS, DEFAULT_SCAFFOLD_TOP_KS
from memomemo.scaffolds.base import ScaffoldConfig
from memomemo.schemas import CandidateResult, LocomoExample


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for the Claude proposer loop."""

    run_id: str
    out_dir: Path
    iterations: int = 20
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
    scaffolds: tuple[str, ...] = DEFAULT_EVOLUTION_SEED_SCAFFOLDS
    scaffold_extra: dict[str, dict[str, object]] | None = None
    selection_policy: str = "default"
    ucb_exploration_c: float = 0.6
    ucb_alpha: float = 0.25
    ucb_gamma: float = 0.95
    pareto_quality_threshold: float = 0.125


class MemoOptimizer:
    """Meta-harness-style proposer loop for memory scaffolds."""

    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config
        self.project_root = Path(__file__).resolve().parents[2]
        self.run_dir = config.out_dir
        self.pending_eval_path = self.run_dir / "pending_eval.json"
        self.frontier_path = self.run_dir / "pareto_frontier.json"
        self.summary_path = self.run_dir / "evolution_summary.jsonl"
        self.bandit_path = self.run_dir / "bandit_state.json"
        self.generated_dir = self.run_dir / "generated"

    def run(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_package_dirs(self.generated_dir)
        examples = self._load_examples()

        candidates: list[CandidateResult] = []
        if self.config.baseline_dir is not None:
            baseline_candidates = load_baseline_candidates(
                self.config.baseline_dir,
                split=self.config.split,
                scaffolds=self.config.scaffolds,
                top_k_by_scaffold={
                    scaffold: DEFAULT_SCAFFOLD_TOP_KS.get(scaffold, 8)
                    for scaffold in self.config.scaffolds
                },
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
                pareto_quality_threshold=self.config.pareto_quality_threshold,
                scaffolds=self.config.scaffolds,
                scaffold_extra=self.config.scaffold_extra,
            )
            for item in scaffold_summary.get("candidates", []):
                candidate = CandidateResult.from_dict(item)
                candidates.append(candidate)
                self._append_summary(iteration=0, candidate=candidate)
        else:
            candidates.extend(self._load_existing_candidates())

        self._save_frontier_from_candidates(candidates)

        for iteration in range(1, self.config.iterations + 1):
            if self.config.selection_policy == "ucb":
                evaluated = self._run_ucb_proposer_iteration(iteration, candidates, examples)
            else:
                evaluated = self._run_default_proposer_iteration(
                    iteration,
                    examples,
                    existing_candidates=candidates,
                )
            candidates.extend(evaluated)
            self._save_frontier_from_candidates(candidates)
            frontier_ids = self._frontier_ids(candidates)
            write_post_eval_artifacts(
                run_dir=self.run_dir,
                call_dir=None,
                iteration=iteration,
                candidates=evaluated,
                frontier_ids=frontier_ids,
            )

        final_summary = {
            "run_id": self.config.run_id,
            "out_dir": str(self.run_dir),
            "iterations": self.config.iterations,
            "candidate_count": len(candidates),
            "pareto_frontier_path": str(self.frontier_path),
            "selection_policy": self.config.selection_policy,
            "pareto_quality_threshold": self.config.pareto_quality_threshold,
            "proposer_metrics": self._aggregate_proposer_metrics(),
        }
        (self.run_dir / "optimizer_summary.json").write_text(
            json.dumps(final_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return final_summary

    def _run_default_proposer_iteration(
        self,
        iteration: int,
        examples: list[LocomoExample],
        existing_candidates: list[CandidateResult] | None = None,
    ) -> list[CandidateResult]:
        if self.pending_eval_path.exists():
            self.pending_eval_path.unlink()
        (self.run_dir / "reports").mkdir(parents=True, exist_ok=True)
        call_dir = self.run_dir / "proposer_calls" / f"iter_{iteration:03d}"
        call_dir.mkdir(parents=True, exist_ok=True)
        source_snapshot_dir = self._build_source_snapshot_workspace(
            iteration=iteration,
            source_family="fusion",
            parent=self._select_default_parent(existing_candidates or []),
            call_dir=call_dir,
            cost_level=None,
        )
        prompt = build_proposer_prompt(
            run_id=self.config.run_id,
            iteration=iteration,
            run_dir=self.run_dir,
            generated_dir=self.generated_dir,
            source_snapshot_dir=source_snapshot_dir,
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
        self._append_proposer_result_event(
            iteration=iteration,
            result=result,
            selection_policy="default",
        )
        if result.returncode != 0 or result.timed_out or not self.pending_eval_path.exists():
            self._append_event(
                {
                    "iteration": iteration,
                    "event": "proposer_failed",
                    "returncode": result.returncode,
                    "timed_out": result.timed_out,
                    "stderr": result.stderr[:1000],
                    "proposer_metrics": getattr(result, "metrics", {}),
                }
            )
            return []

        pending = json.loads(self.pending_eval_path.read_text(encoding="utf-8"))
        proposed = pending.get("candidates") or []
        if not isinstance(proposed, list):
            proposed = []
        if len(proposed) != 1:
            self._append_event(
                {
                    "iteration": iteration,
                    "event": "default_candidate_count_adjusted",
                    "requested_count": len(proposed),
                    "evaluated_count": min(len(proposed), 1),
                }
            )
            proposed = proposed[:1]
        return self._evaluate_proposed(iteration, proposed, examples)

    def _run_ucb_proposer_iteration(
        self,
        iteration: int,
        existing_candidates: list[CandidateResult],
        examples: list[LocomoExample],
    ) -> list[CandidateResult]:
        if self.pending_eval_path.exists():
            self.pending_eval_path.unlink()

        bandit_state = load_bandit_state(self.bandit_path)
        available_arms = self._available_ucb_arms(existing_candidates)
        if iteration <= 3:
            available_arms = [
                arm for arm in available_arms if arm.cost_level == "low"
            ]
        arm = select_ucb_arm(
            bandit_state,
            available_arms=available_arms,
            exploration_c=self.config.ucb_exploration_c,
        )
        parent = self._select_parent_for_arm(existing_candidates, arm)
        call_dir = self.run_dir / "proposer_calls" / f"iter_{iteration:03d}"
        context_dir = self._build_context_snapshot(
            iteration=iteration,
            arm=arm,
            parent=parent,
            call_dir=call_dir,
        )
        source_snapshot_dir = self._build_source_snapshot_workspace(
            iteration=iteration,
            source_family=arm.source_family,
            parent=parent,
            call_dir=call_dir,
            cost_level=arm.cost_level,
        )
        intend_path = call_dir / "intend.md"
        prompt = build_ucb_proposer_prompt(
            run_id=self.config.run_id,
            iteration=iteration,
            run_dir=self.run_dir,
            pending_eval_path=self.pending_eval_path,
            frontier_path=self.frontier_path,
            summary_path=self.summary_path,
            context_dir=context_dir,
            generated_dir=self.generated_dir,
            source_snapshot_dir=source_snapshot_dir,
            intend_path=intend_path,
            parent_candidate_id=parent.candidate_id,
            source_family=arm.source_family,
            cost_level=arm.cost_level,
            split=self.config.split,
            limit=self.config.limit,
        )
        result = run_claude_prompt(
            prompt,
            cwd=self.project_root,
            log_dir=call_dir / "claude_session",
            name="proposer",
            model=self.config.claude_model,
            timeout_s=self.config.propose_timeout_s,
        )
        self._append_proposer_result_event(
            iteration=iteration,
            result=result,
            selection_policy="ucb",
            extra={
                "arm": arm.key,
                "parent_candidate_id": parent.candidate_id,
                "call_dir": str(call_dir),
            },
        )
        self._capture_diff(call_dir)
        write_diff_digest(call_dir=call_dir)

        if result.returncode != 0 or result.timed_out or not self.pending_eval_path.exists():
            self._append_event(
                {
                    "iteration": iteration,
                    "event": "proposer_failed",
                    "selection_policy": "ucb",
                    "arm": arm.key,
                    "parent_candidate_id": parent.candidate_id,
                    "returncode": result.returncode,
                    "timed_out": result.timed_out,
                    "stderr": result.stderr[:1000],
                    "proposer_metrics": getattr(result, "metrics", {}),
                }
            )
            reward = -cost_penalty(arm.cost_level)
            updated = update_bandit_state(
                bandit_state,
                arm,
                reward=reward,
                entered_frontier=False,
                iteration=iteration,
                alpha=self.config.ucb_alpha,
                gamma=self.config.ucb_gamma,
            )
            save_bandit_state(self.bandit_path, updated)
            return []

        pending = json.loads(self.pending_eval_path.read_text(encoding="utf-8"))
        proposed = pending.get("candidates") or []
        if not isinstance(proposed, list):
            proposed = []
        if len(proposed) != 1:
            self._append_event(
                {
                    "iteration": iteration,
                    "event": "ucb_candidate_count_adjusted",
                    "requested_count": len(proposed),
                    "evaluated_count": min(len(proposed), 1),
                }
            )
            proposed = proposed[:1]
        for raw in proposed:
            if isinstance(raw, dict):
                raw.setdefault("parent_candidate_id", parent.candidate_id)
                raw.setdefault("source_family", arm.source_family)
                raw.setdefault("cost_level", arm.cost_level)
                raw.setdefault(
                    "bandit_arm",
                    {"source_family": arm.source_family, "cost_level": arm.cost_level},
                )
        (call_dir / "pending_eval.json").write_text(
            json.dumps({"candidates": proposed}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        evaluated = self._evaluate_proposed(iteration, proposed, examples)
        candidate_pool = existing_candidates + evaluated
        frontier_ids = self._frontier_ids(candidate_pool)
        write_post_eval_artifacts(
            run_dir=self.run_dir,
            call_dir=call_dir,
            iteration=iteration,
            candidates=evaluated,
            frontier_ids=frontier_ids,
        )
        reward = self._ucb_round_reward(
            arm=arm,
            parent=parent,
            evaluated=evaluated,
            frontier_ids=frontier_ids,
        )
        updated = update_bandit_state(
            bandit_state,
            arm,
            reward=reward,
            entered_frontier=any(item.candidate_id in frontier_ids for item in evaluated),
            iteration=iteration,
            alpha=self.config.ucb_alpha,
            gamma=self.config.ucb_gamma,
        )
        save_bandit_state(self.bandit_path, updated)
        self._append_event(
            {
                "iteration": iteration,
                "event": "ucb_update",
                "arm": arm.key,
                "parent_candidate_id": parent.candidate_id,
                "reward": reward,
                "evaluated": [item.candidate_id for item in evaluated],
            }
        )
        return evaluated

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
        runner = EvaluationRunner(
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
            if isinstance(raw, dict):
                raw = dict(raw)
                raw.setdefault("candidate_root", str(self.generated_dir))
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

            top_k, top_k_adjusted = _single_top_k(raw.get("top_k", 8))
            if top_k_adjusted:
                self._append_event(
                    {
                        "iteration": iteration,
                        "event": "candidate_top_k_adjusted",
                        "candidate": raw,
                        "evaluated_top_k": top_k,
                    }
                )
            extra = dict(raw.get("extra") or {})
            for key in (
                "bandit_arm",
                "build_tag",
                "class",
                "cost_level",
                "factory",
                "module",
                "module_path",
                "parent_candidate_id",
                "project_source_path",
                "source_base_dir",
                "source_family",
                "source_path",
                "source_project_path",
                "upstream_source_path",
                "mem0_source_path",
                "memgpt_source_path",
                "membank_source_path",
            ):
                if key in raw and key not in extra:
                    extra[key] = raw[key]
            config = ScaffoldConfig(
                top_k=top_k,
                window=int(raw.get("window", 1)),
                extra=extra,
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

    def _available_ucb_arms(self, candidates: list[CandidateResult]) -> list[BanditArm]:
        sources = {self._infer_source_family(item) for item in candidates}
        allowed_sources = set(SOURCE_FAMILIES)
        sources.update(allowed_sources)
        return [
            BanditArm(source_family=source, cost_level=cost)
            for source in sorted(sources)
            if source in allowed_sources
            for cost in ("low", "medium", "high")
        ]

    def _select_parent_for_arm(
        self,
        candidates: list[CandidateResult],
        arm: BanditArm,
    ) -> CandidateResult:
        if not candidates:
            raise ValueError("cannot select a UCB parent without evaluated candidates")

        frontier_ids = self._frontier_ids(candidates)
        source_pool = [
            item for item in candidates if self._infer_source_family(item) == arm.source_family
        ]
        if arm.source_family == "fusion" and not source_pool:
            source_pool = list(candidates)
        if not source_pool:
            source_pool = list(candidates)

        frontier_pool = [item for item in source_pool if item.candidate_id in frontier_ids]
        pool = frontier_pool or source_pool

        def score(item: CandidateResult) -> tuple[float, int, str]:
            return (item.passrate, -item.token_consuming, item.candidate_id)

        return max(pool, key=score)

    def _select_default_parent(self, candidates: list[CandidateResult]) -> CandidateResult | None:
        if not candidates:
            return None

        frontier_ids = self._frontier_ids(candidates)
        pool = [item for item in candidates if item.candidate_id in frontier_ids] or candidates

        def score(item: CandidateResult) -> tuple[float, int, str]:
            return (item.passrate, -item.token_consuming, item.candidate_id)

        return max(pool, key=score)

    def _infer_source_family(self, candidate: CandidateResult) -> str:
        extra = candidate.config.get("extra") if isinstance(candidate.config, dict) else None
        if isinstance(extra, dict):
            source_family = str(extra.get("source_family") or "").lower()
            if source_family in {"bm25", "mem0", "memgpt", "membank", "fusion"}:
                return source_family
        text = f"{candidate.candidate_id} {candidate.scaffold_name}".lower()
        if "memgpt" in text or "letta" in text:
            return "memgpt"
        if "membank" in text or "memorybank" in text:
            return "membank"
        if "mem0" in text:
            return "mem0"
        if "bm25" in text:
            return "bm25"
        if candidate.scaffold_name == "bm25":
            return "bm25"
        if candidate.scaffold_name == "mem0_source":
            return "mem0"
        if candidate.scaffold_name == "memgpt_source":
            return "memgpt"
        if candidate.scaffold_name == "membank_source":
            return "membank"
        return "fusion"

    def _build_context_snapshot(
        self,
        *,
        iteration: int,
        arm: BanditArm,
        parent: CandidateResult,
        call_dir: Path,
    ) -> Path:
        call_dir.mkdir(parents=True, exist_ok=True)
        context_dir = call_dir / "context"
        if context_dir.exists():
            shutil.rmtree(context_dir)
        context_dir.mkdir(parents=True, exist_ok=True)

        assignment = {
            "iteration": iteration,
            "arm": {"source_family": arm.source_family, "cost_level": arm.cost_level},
            "parent_candidate_id": parent.candidate_id,
            "parent_result_path": parent.result_path,
            "generated_dir": str(self.generated_dir),
            "source_snapshot_dir": str(
                self.generated_dir / "source_snapshots" / f"iter_{iteration:03d}"
            ),
        }
        (call_dir / "assignment.json").write_text(
            json.dumps(assignment, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (context_dir / "assignment.json").write_text(
            json.dumps(assignment, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        recent_limit = 2 if arm.cost_level == "low" else 5
        if arm.cost_level == "high":
            recent_limit = 999999

        current_dir = context_dir / "current"
        current_dir.mkdir(parents=True, exist_ok=True)
        for path in (self.frontier_path, self.bandit_path):
            if path.exists():
                shutil.copy2(path, current_dir / path.name)
        self._write_context_summary_tail(current_dir, limit=recent_limit)

        parent_dir = context_dir / "selected_parent"
        parent_dir.mkdir(parents=True, exist_ok=True)
        (parent_dir / "candidate_summary.json").write_text(
            json.dumps(parent.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._copy_if_exists(Path(parent.result_path), parent_dir / Path(parent.result_path).name)
        self._copy_parent_traces_if_exists(
            parent.candidate_id,
            parent_dir,
            cost_level=arm.cost_level,
        )
        parent_source = self._candidate_source_path(parent)
        if parent_source is not None:
            self._copy_if_exists(parent_source, parent_dir / parent_source.name)

        scaffold_source = self._source_scaffold_path(arm.source_family)
        if scaffold_source is not None:
            self._copy_if_exists(scaffold_source, parent_dir / scaffold_source.name)
        self._copy_project_source_context(parent_dir)
        self._copy_upstream_source_context(arm.source_family, parent_dir)
        self._copy_if_exists(
            self.project_root / "src" / "memomemo" / "scaffolds" / "base.py",
            parent_dir / "base.py",
        )
        self._copy_if_exists(
            self.project_root / "src" / "memomemo" / "schemas.py",
            parent_dir / "schemas.py",
        )

        self._copy_recent_iteration_bundles(
            context_dir,
            iteration=iteration,
            limit=recent_limit,
            cost_level=arm.cost_level,
        )
        if arm.cost_level == "high":
            self._write_high_budget_manifest(context_dir)
        self._write_context_readme(
            context_dir,
            cost_level=arm.cost_level,
            recent_limit=recent_limit,
            parent=parent,
        )

        return context_dir

    def _write_context_summary_tail(self, current_dir: Path, *, limit: int) -> None:
        if not self.summary_path.exists():
            return
        lines = [line for line in self.summary_path.read_text(encoding="utf-8").splitlines() if line]
        selected = lines if limit >= 999999 else lines[-limit:]
        (current_dir / "evolution_summary_recent.jsonl").write_text(
            "\n".join(selected) + ("\n" if selected else ""),
            encoding="utf-8",
        )
        if limit >= 999999:
            self._copy_if_exists(self.summary_path, current_dir / self.summary_path.name)

    def _build_source_snapshot_workspace(
        self,
        *,
        iteration: int,
        source_family: str,
        parent: CandidateResult | None,
        call_dir: Path,
        cost_level: str | None = None,
    ) -> Path:
        snapshot_root = (
            self.generated_dir
            / "source_snapshots"
            / f"iter_{iteration:03d}"
        )
        if snapshot_root.exists():
            shutil.rmtree(snapshot_root)
        snapshot_root.mkdir(parents=True, exist_ok=True)
        self._ensure_package_dirs(snapshot_root)

        parent_source = self._candidate_source_path(parent) if parent is not None else None
        scaffold_source = self._source_scaffold_path(source_family)
        source_files = [
            path for path in (parent_source, scaffold_source) if path is not None and path.exists()
        ]

        candidate_dir = snapshot_root / "candidate"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_package_dirs(candidate_dir)
        for path in source_files:
            self._copy_if_exists(path, candidate_dir / path.name)
        self._copy_project_source_context(candidate_dir)
        self._copy_upstream_source_context(source_family, candidate_dir)
        self._copy_if_exists(
            self.project_root / "src" / "memomemo" / "scaffolds" / "base.py",
            candidate_dir / "base.py",
        )
        self._copy_if_exists(
            self.project_root / "src" / "memomemo" / "schemas.py",
            candidate_dir / "schemas.py",
        )
        (candidate_dir / "SNAPSHOT.md").write_text(
            "\n".join(
                [
                    "# Source Snapshot Candidate",
                    "",
                    f"Iteration: {iteration}",
                    f"Source family: {source_family}",
                    f"Parent candidate: {parent.candidate_id if parent is not None else 'none'}",
                    "",
                    "This directory is a writable candidate-specific copy of the parent/source.",
                    "It also contains full project source under `project_source/src/memomemo`",
                    "and relevant upstream source under `upstream_source` for inspection.",
                    "Existing source-backed base memories are read-only. You may edit",
                    "copied build/database-construction paths such as add/build/schema/",
                    "extraction/evolution/embedding or persistence layout, but source",
                    "edits that alter persisted memories must use a fresh source_base_dir",
                    "and build_tag in pending_eval.json.",
                    "Modify files here for the mechanism under test, then expose a scaffold",
                    f"through a wrapper module under `{self.generated_dir}` and reference",
                    "that wrapper in `pending_eval.json`.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        manifest = {
            "iteration": iteration,
            "source_family": source_family,
            "parent_candidate_id": parent.candidate_id if parent is not None else None,
            "cost_level": cost_level,
            "candidate_dir": str(candidate_dir),
            "source_files": [str(path) for path in source_files],
        }
        (snapshot_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (call_dir / "source_snapshot_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return snapshot_root

    def _ensure_package_dirs(self, path: Path) -> None:
        generated_root = self.generated_dir
        current = generated_root
        while True:
            current.mkdir(parents=True, exist_ok=True)
            init = current / "__init__.py"
            if not init.exists():
                init.write_text('"""Generated source snapshot package."""\n', encoding="utf-8")
            if current == path:
                break
            try:
                rel = path.relative_to(current)
            except ValueError:
                break
            parts = rel.parts
            if not parts:
                break
            current = current / parts[0]

    def _copy_recent_iteration_bundles(
        self,
        context_dir: Path,
        *,
        iteration: int,
        limit: int,
        cost_level: str,
    ) -> None:
        calls_root = self.run_dir / "proposer_calls"
        if not calls_root.exists():
            return
        recent_dir = context_dir / "recent_iteration_bundles"
        recent_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for call_dir in sorted(calls_root.glob("iter_*"), reverse=True):
            if call_dir.name == f"iter_{iteration:03d}":
                continue
            if copied >= limit:
                break
            dest = recent_dir / call_dir.name
            self._copy_iteration_bundle(call_dir, dest, cost_level=cost_level)
            copied += 1

    def _copy_iteration_bundle(self, src: Path, dest: Path, *, cost_level: str) -> None:
        if not src.exists():
            return
        if dest.exists():
            shutil.rmtree(dest)
        ignore = shutil.ignore_patterns("context", "claude_session", "__pycache__")
        shutil.copytree(src, dest, ignore=ignore)
        self._prune_trace_slices_for_budget(dest, cost_level=cost_level)

    def _prune_trace_slices_for_budget(self, bundle_dir: Path, *, cost_level: str) -> None:
        trace_dir = bundle_dir / "trace_slices"
        if not trace_dir.exists():
            return
        if cost_level == "low":
            allowed = {"low"}
        elif cost_level == "medium":
            allowed = {"medium"}
        else:
            allowed = {"low", "medium", "high"}
        for child in trace_dir.iterdir():
            if child.is_dir() and child.name not in allowed:
                shutil.rmtree(child)

    def _write_high_budget_manifest(self, context_dir: Path) -> None:
        manifest = {
            "source_files": [
                str(path)
                for path in sorted((self.project_root / "src" / "memomemo").glob("**/*.py"))
            ],
            "candidate_results": [
                str(path)
                for path in sorted((self.run_dir / "candidate_results").glob("*.json"))
            ],
            "trace_slices": [
                str(path)
                for path in sorted((self.run_dir / "trace_slices").glob("**/*.json"))
            ],
            "proposer_calls": [
                str(path)
                for path in sorted((self.run_dir / "proposer_calls").glob("iter_*"))
            ],
            "generated_files": [
                str(path)
                for path in sorted(self.generated_dir.glob("**/*"))
                if path.is_file()
            ],
        }
        (context_dir / "high_budget_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _copy_parent_traces_if_exists(
        self,
        candidate_id: str,
        dest_dir: Path,
        *,
        cost_level: str,
    ) -> None:
        level = "low" if cost_level == "low" else "medium" if cost_level == "medium" else "high"
        path = self.run_dir / "trace_slices" / level / f"{candidate_id}.json"
        self._copy_if_exists(path, dest_dir / f"trace_{level}.json")

    def _write_context_readme(
        self,
        context_dir: Path,
        *,
        cost_level: str,
        recent_limit: int,
        parent: CandidateResult,
    ) -> None:
        recent_text = "all prior" if recent_limit >= 999999 else f"last {recent_limit}"
        context_dir.joinpath("CONTEXT.md").write_text(
            "\n".join(
                [
                    "# UCB Context Snapshot",
                    "",
                    f"Budget: `{cost_level}`",
                    f"Selected parent: `{parent.candidate_id}`",
                    "",
                    "This snapshot is assembled by the optimizer. In UCB mode the",
                    "optimizer chooses the parent and context budget; the proposer uses",
                    "this snapshot instead of choosing its own global context.",
                    "",
                    "Directory guide:",
                    "",
                    "- `current/`: current frontier, bandit state, and a recent summary tail.",
                    "- `selected_parent/`: selected parent summary, source/result files, and parent traces when available.",
                    "- `selected_parent/project_source/src/memomemo/`: full project source snapshot for inspection.",
                    "- `selected_parent/upstream_source/`: full relevant upstream source snapshot for source-backed lineages.",
                    "- `source_snapshot_dir` in `assignment.json`: writable candidate-specific parent/source copies under the run-local `generated/source_snapshots/` directory.",
                    f"- `recent_iteration_bundles/`: {recent_text} proposer iteration bundles copied from `proposer_calls/`.",
                    "",
                    "Source-backed baseline memories under `runs/source_base_memory/**`",
                    "are read-only. You may modify copied source for retrieval, scoring,",
                    "filtering, context formatting, answering prompts, wrappers, and",
                    "build/database-construction paths such as add/build/schema/",
                    "extraction/evolution/embedding or persistence layout. Source edits",
                    "that alter persisted memories must use a fresh source_base_dir and",
                    "build_tag in pending_eval.json.",
                    "",
                    "Trace access is budgeted per iteration bundle: low includes up to",
                    "3 full trace cases, medium includes up to 10 full trace cases, and",
                    "high may inspect all trace cases. High also includes a manifest",
                    "pointing to all source, parent, trace, and candidate-result files",
                    "that may be read selectively.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def _copy_if_exists(self, src: Path, dest: Path) -> None:
        if src.exists() and src.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    def _copy_tree_if_exists(self, src: Path, dest: Path) -> None:
        if not src.exists() or not src.is_dir():
            return
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(
            src,
            dest,
            ignore=shutil.ignore_patterns(
                ".git",
                ".mypy_cache",
                ".pytest_cache",
                ".ruff_cache",
                "__pycache__",
                "*.pyc",
            ),
        )

    def _copy_project_source_context(self, dest_dir: Path) -> None:
        self._copy_tree_if_exists(
            self.project_root / "src" / "memomemo",
            dest_dir / "project_source" / "src" / "memomemo",
        )

    def _candidate_source_path(self, candidate: CandidateResult) -> Path | None:
        if candidate.scaffold_name in {"bm25", "mem0_source", "memgpt_source", "membank_source", "no_memory"}:
            return None
        stem = candidate.scaffold_name.replace("-", "_")
        exact = self.generated_dir / f"{stem}.py"
        if exact.exists():
            return exact
        return None

    def _source_scaffold_path(self, source_family: str) -> Path | None:
        mapping = {
            "bm25": "bm25_scaffold.py",
            "mem0": "mem0_scaffold.py",
            "memgpt": "memgpt_scaffold.py",
            "membank": "membank_scaffold.py",
        }
        name = mapping.get(source_family)
        if not name:
            return None
        return self.project_root / "src" / "memomemo" / "scaffolds" / name

    def _copy_upstream_source_context(self, source_family: str, dest_dir: Path) -> None:
        upstream_dir = dest_dir / "upstream_source"
        if source_family in {"mem0", "fusion"}:
            self._copy_tree_if_exists(
                self.project_root / "references" / "vendor" / "mem0",
                upstream_dir / "mem0",
            )
        if source_family in {"memgpt", "fusion"}:
            self._copy_tree_if_exists(
                self.project_root / "references" / "vendor" / "MemGPT",
                upstream_dir / "MemGPT",
            )
        if source_family in {"membank", "fusion"}:
            self._copy_tree_if_exists(
                self.project_root / "references" / "vendor" / "MemoryBank-SiliconFriend",
                upstream_dir / "MemoryBank-SiliconFriend",
            )

    def _capture_diff(self, call_dir: Path) -> None:
        call_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "git",
            "diff",
            "--",
            "src/memomemo/scaffolds",
            "src/memomemo/utils",
            "src/memomemo/metrics.py",
        ]
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                text=True,
                capture_output=True,
                timeout=30,
            )
            text = completed.stdout or completed.stderr or ""
        except Exception as exc:  # noqa: BLE001 - best-effort artifact
            text = f"Failed to capture diff: {exc}\n"
        (call_dir / "diff.patch").write_text(text, encoding="utf-8")

    def _ucb_round_reward(
        self,
        *,
        arm: BanditArm,
        parent: CandidateResult,
        evaluated: list[CandidateResult],
        frontier_ids: set[str],
    ) -> float:
        if not evaluated:
            return -cost_penalty(arm.cost_level)

        def candidate_reward(item: CandidateResult) -> float:
            entered = 1.0 if item.candidate_id in frontier_ids else 0.0
            pass_gain = max(0.0, item.passrate - parent.passrate)
            return entered + pass_gain

        best = max(candidate_reward(item) for item in evaluated)
        return best - cost_penalty(arm.cost_level)

    def _frontier_ids(self, candidates: list[CandidateResult]) -> set[str]:
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
        return {
            item.candidate_id
            for item in pareto_frontier(
                points,
                quality_gap_threshold=self.config.pareto_quality_threshold,
            )
        }

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
        save_frontier(
            self.frontier_path,
            points,
            quality_gap_threshold=self.config.pareto_quality_threshold,
        )

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
            ],
            quality_gap_threshold=self.config.pareto_quality_threshold,
        )
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "iteration": iteration,
            "candidate": candidate.to_dict(),
            "proposal": proposal or {},
            "self_frontier": [asdict(item) for item in frontier],
        }
        self._append_event(row)

    def _append_proposer_result_event(
        self,
        *,
        iteration: int,
        result: Any,
        selection_policy: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        tool_access = getattr(result, "tool_access", {}) or {}
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "iteration": iteration,
            "event": "proposer_result",
            "selection_policy": selection_policy,
            "returncode": getattr(result, "returncode", None),
            "timed_out": bool(getattr(result, "timed_out", False)),
            "proposer_metrics": getattr(result, "metrics", {}) or {},
            "usage": getattr(result, "usage", None),
            "files_read": tool_access.get("files_read", {})
            if isinstance(tool_access, dict)
            else {},
            "files_written": tool_access.get("files_written", {})
            if isinstance(tool_access, dict)
            else {},
            "grep_requests": tool_access.get("grep_requests", [])
            if isinstance(tool_access, dict)
            else [],
            "tool_counts": tool_access.get("tool_counts", {})
            if isinstance(tool_access, dict)
            else {},
        }
        if extra:
            row.update(extra)
        self._append_event(row)

    def _aggregate_proposer_metrics(self) -> dict[str, Any]:
        if not self.summary_path.exists():
            return {}

        totals = {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "total_reported_tokens": 0,
            "estimated_cost_usd": 0.0,
            "duration_s": 0.0,
            "tool_calls": 0,
            "read_file_calls": 0,
            "read_lines": 0,
            "write_file_calls": 0,
            "written_lines": 0,
        }
        tool_counts: dict[str, int] = {}
        unique_files_read: set[str] = set()

        for line in self.summary_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "proposer_result":
                continue

            metrics = row.get("proposer_metrics") or {}
            if not isinstance(metrics, dict):
                metrics = {}
            totals["calls"] += 1
            for key in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "total_reported_tokens",
                "tool_calls",
                "read_file_calls",
                "read_lines",
                "write_file_calls",
                "written_lines",
            ):
                totals[key] += _int_metric(metrics.get(key))
            for key in ("estimated_cost_usd", "duration_s"):
                totals[key] += _float_metric(metrics.get(key))

            row_tool_counts = row.get("tool_counts") or metrics.get("tool_counts") or {}
            if isinstance(row_tool_counts, dict):
                for name, count in row_tool_counts.items():
                    tool_counts[str(name)] = tool_counts.get(str(name), 0) + _int_metric(
                        count
                    )

            files_read = row.get("files_read") or {}
            if isinstance(files_read, dict):
                unique_files_read.update(str(path) for path in files_read)

        totals["estimated_cost_usd"] = round(totals["estimated_cost_usd"], 6)
        totals["duration_s"] = round(totals["duration_s"], 3)
        totals["unique_files_read"] = len(unique_files_read)
        totals["tool_counts"] = dict(sorted(tool_counts.items()))
        return totals

    def _append_event(self, row: dict[str, Any]) -> None:
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with self.summary_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _single_top_k(raw: Any) -> tuple[int, bool]:
    if isinstance(raw, int):
        return raw, False
    if isinstance(raw, list) and raw:
        return int(raw[0]), len(raw) != 1
    return int(raw or 8), raw != 8


def _int_metric(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float_metric(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
