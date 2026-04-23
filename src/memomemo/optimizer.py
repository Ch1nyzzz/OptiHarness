"""Claude-proposer optimization loop for OptiHarness."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memomemo.baseline import load_baseline_candidates
from memomemo.benchmark_workspaces import (
    LOCOMO_WORKSPACE_SPEC,
    BenchmarkWorkspaceSpec,
    copy_benchmark_project_source,
)
from memomemo.claude_runner import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_DOCKER_ENV_VARS,
    DEFAULT_KIMI_MODEL,
    ProposerSandboxConfig,
    _float_metric,
    _int_metric,
    run_code_agent_prompt,
)
from memomemo.dynamic import load_candidate_scaffold
from memomemo.evaluation import EvaluationRunner, run_initial_frontier
from memomemo.locomo import default_data_path, load_locomo_examples, prepare_locomo, select_split
from memomemo.model import DEFAULT_BASE_URL, DEFAULT_MODEL
from memomemo.optimization_cells import get_target_cells
from memomemo.post_eval import write_diff_digest, write_post_eval_artifacts
from memomemo.proposer_prompt import build_progressive_proposer_prompt
from memomemo.scaffolds import DEFAULT_EVOLUTION_SEED_SCAFFOLDS, DEFAULT_SCAFFOLD_TOP_KS
from memomemo.scaffolds.base import ScaffoldConfig
from memomemo.schemas import CandidateResult, LocomoExample


def _pending_candidates(payload: Any) -> list[Any]:
    """Accept either {"candidates": [...]} or a top-level candidate list."""

    if isinstance(payload, dict):
        candidates = payload.get("candidates") or []
    elif isinstance(payload, list):
        candidates = payload
    else:
        candidates = []
    return candidates if isinstance(candidates, list) else []


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for the Claude proposer loop."""

    run_id: str
    out_dir: Path
    iterations: int = 20
    split: str = "train"
    limit: int = 0
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str = "EMPTY"
    eval_timeout_s: int = 300
    proposer_agent: str = "claude"
    claude_model: str = "claude-sonnet-4-6"
    codex_model: str = DEFAULT_CODEX_MODEL
    kimi_model: str = DEFAULT_KIMI_MODEL
    propose_timeout_s: int = 2400
    dry_run: bool = False
    max_context_chars: int = 6000
    max_eval_workers: int = 1
    skip_scaffold_eval: bool = False
    baseline_dir: Path | None = None
    scaffolds: tuple[str, ...] = DEFAULT_EVOLUTION_SEED_SCAFFOLDS
    scaffold_extra: dict[str, dict[str, object]] | None = None
    selection_policy: str = "default"
    progressive_target_system: str = "memgpt"
    progressive_initial_low_iterations: int = 5
    pareto_quality_threshold: float = 0.125
    proposer_sandbox: str = "docker"
    proposer_docker_image: str = ""
    proposer_docker_workspace: str = "/workspace"
    proposer_docker_env: tuple[str, ...] = ()
    proposer_docker_mount: tuple[str, ...] = ()
    proposer_docker_kimi_cli_kind: str = "claude"
    proposer_docker_user: str = ""
    proposer_docker_home: str = ""


class LocomoOptimizer:
    """Meta-harness-style proposer loop for LOCOMO memory scaffolds."""

    workspace_spec: BenchmarkWorkspaceSpec = LOCOMO_WORKSPACE_SPEC

    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config
        self.project_root = Path(__file__).resolve().parents[2]
        self.run_dir = config.out_dir
        self.pending_eval_path = self.run_dir / "pending_eval.json"
        self.frontier_path = self.run_dir / "best_candidates.json"
        self.summary_path = self.run_dir / "evolution_summary.jsonl"
        self.generated_dir = self.run_dir / "generated"
        self.progressive_state_path = self.run_dir / "progressive_state.json"
        self.candidate_score_table_path = self.run_dir / "candidate_score_table.json"
        self.retrieval_diagnostics_summary_path = (
            self.run_dir / "retrieval_diagnostics_summary.json"
        )
        self.iteration_index_path = self.run_dir / "iteration_index.json"
        self.diff_summary_path = self.run_dir / "diff_summary.jsonl"

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
                if int(candidate.count) != len(examples):
                    raise ValueError(
                        "Baseline candidate count does not match current evaluation set: "
                        f"{candidate.candidate_id} has count={candidate.count}, "
                        f"but split={self.config.split!r} limit={self.config.limit} "
                        f"selects {len(examples)} examples. Recompute the baseline with "
                        "the same split/limit before using it as --baseline-dir."
                    )
                candidates.append(candidate)
                self._append_summary(iteration=0, candidate=candidate)
        elif not self.config.skip_scaffold_eval:
            scaffold_summary = self._run_seed_frontier()
            for item in scaffold_summary.get("candidates", []):
                candidate = CandidateResult.from_dict(item)
                candidates.append(candidate)
                self._append_summary(iteration=0, candidate=candidate)
        else:
            candidates.extend(self._load_existing_candidates())

        if candidates and not self.config.skip_scaffold_eval:
            best_ids = self._best_passrate_ids(candidates)
            write_post_eval_artifacts(
                run_dir=self.run_dir,
                call_dir=None,
                iteration=0,
                candidates=candidates,
                frontier_ids=best_ids,
            )

        self._save_best_candidates(candidates)
        self._refresh_run_indexes(candidates)

        for iteration in range(1, self.config.iterations + 1):
            previous_best_passrate = self._best_passrate(candidates)
            if self.config.selection_policy == "progressive":
                budget = self._progressive_budget_for_iteration(iteration)
            else:
                budget = "high"
            evaluated = self._run_progressive_proposer_iteration(
                iteration,
                candidates,
                examples,
                budget=budget,
                adaptive=self.config.selection_policy == "progressive",
            )
            candidates.extend(evaluated)
            self._save_best_candidates(candidates)
            self._refresh_run_indexes(candidates)
            best_ids = self._best_passrate_ids(candidates)
            write_post_eval_artifacts(
                run_dir=self.run_dir,
                call_dir=None,
                iteration=iteration,
                candidates=evaluated,
                frontier_ids=best_ids,
            )
            if self.config.selection_policy == "progressive":
                self._update_progressive_state(
                    iteration=iteration,
                    budget=budget,
                    previous_best_passrate=previous_best_passrate,
                    candidates=candidates,
                    evaluated=evaluated,
                )

        final_summary = {
            "run_id": self.config.run_id,
            "out_dir": str(self.run_dir),
            "iterations": self.config.iterations,
            "candidate_count": len(candidates),
            "best_candidates_path": str(self.frontier_path),
            "selection_policy": self.config.selection_policy,
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
        return self._run_progressive_proposer_iteration(
            iteration,
            existing_candidates or [],
            examples,
            budget="high",
            adaptive=False,
        )

    def _run_progressive_proposer_iteration(
        self,
        iteration: int,
        existing_candidates: list[CandidateResult],
        examples: list[LocomoExample],
        *,
        budget: str,
        adaptive: bool,
    ) -> list[CandidateResult]:
        if self.pending_eval_path.exists():
            self.pending_eval_path.unlink()
        call_dir = self.run_dir / "proposer_calls" / f"iter_{iteration:03d}"
        retry_note = ""
        max_attempts = 2
        result: Any | None = None
        workspace_dir = call_dir / "workspace"
        workspace_generated_dir = workspace_dir / "generated"
        reference_iterations: tuple[int, ...] = ()
        for attempt in range(1, max_attempts + 1):
            workspace_dir, reference_iterations = self._build_progressive_workspace(
                iteration=iteration,
                budget=budget,
                existing_candidates=existing_candidates,
                call_dir=call_dir,
            )
            workspace_generated_dir = workspace_dir / "generated"
            workspace_source_snapshot_dir = workspace_dir / "source_snapshot"
            workspace_pending_eval_path = workspace_dir / "pending_eval.json"
            prompt = build_progressive_proposer_prompt(
                run_id=self.config.run_id,
                iteration=iteration,
                run_dir=workspace_dir,
                pending_eval_path=workspace_pending_eval_path,
                summaries_dir=workspace_dir / "summaries",
                reference_iterations_dir=workspace_dir / "reference_iterations",
                generated_dir=workspace_generated_dir,
                source_snapshot_dir=workspace_source_snapshot_dir,
                budget=budget,
                reference_iterations=reference_iterations,
                target_system=self.config.progressive_target_system,
                optimization_directions=(
                    self._optimization_direction_lines(self.config.progressive_target_system)
                    if adaptive
                    else ()
                ),
                split=self.config.split,
                limit=self.config.limit,
                selection_policy="progressive" if adaptive else "default",
                benchmark_name=self._benchmark_prompt_name(),
                raw_data_policy=self._raw_data_policy_name(),
            )
            if retry_note:
                prompt = f"{prompt}\n\n{retry_note}"
            result = self._run_proposer_agent(
                prompt,
                log_dir=call_dir / "agent" / f"attempt_{attempt:02d}",
                name="proposer",
                cwd=workspace_dir,
            )
            self._append_proposer_result_event(
                iteration=iteration,
                result=result,
                selection_policy="progressive" if adaptive else "default",
                extra={
                    "budget": budget,
                    "reference_iterations": list(reference_iterations),
                    "target_system": self.config.progressive_target_system,
                    "call_dir": str(call_dir),
                    "workspace_dir": str(workspace_dir),
                    "attempt": attempt,
                },
            )
            access_violations = self._proposer_access_violations(
                result,
                workspace_dir=workspace_dir,
            )
            if not access_violations:
                break
            if attempt < max_attempts:
                self._append_event(
                    {
                        "iteration": iteration,
                        "event": "proposer_access_retry",
                        "selection_policy": "progressive" if adaptive else "default",
                        "budget": budget,
                        "attempt": attempt,
                        "violations": access_violations,
                    }
                )
                retry_note = self._access_retry_note(
                    violations=access_violations,
                    workspace_dir=workspace_dir,
                )
                continue
            self._append_event(
                {
                    "iteration": iteration,
                    "event": "proposer_access_rejected",
                    "selection_policy": "progressive" if adaptive else "default",
                    "budget": budget,
                    "attempt": attempt,
                    "violations": access_violations,
                }
            )
            return []

        assert result is not None
        self._archive_workspace_outputs(
            workspace_dir=workspace_dir,
            call_dir=call_dir,
            result=result,
        )
        if (
            result.returncode == 0
            and not result.timed_out
            and not self.pending_eval_path.exists()
        ):
            self._append_event(
                {
                    "iteration": iteration,
                    "event": "proposer_missing_pending_retry",
                    "selection_policy": "progressive" if adaptive else "default",
                    "budget": budget,
                    "attempt": max_attempts,
                }
            )
            repair_prompt = (
                f"{prompt}\n\n"
                "## Required Repair\n\n"
                "The previous proposer attempt exited without writing "
                f"`{workspace_pending_eval_path}`. Continue in the same "
                "workspace, make a concrete candidate source change if needed, "
                "and write exactly one valid `pending_eval.json`. Do not run "
                "the full harness evaluation."
            )
            result = self._run_proposer_agent(
                repair_prompt,
                log_dir=call_dir / "agent" / "missing_pending_retry",
                name="proposer",
                cwd=workspace_dir,
            )
            self._append_proposer_result_event(
                iteration=iteration,
                result=result,
                selection_policy="progressive" if adaptive else "default",
                extra={
                    "budget": budget,
                    "reference_iterations": list(reference_iterations),
                    "target_system": self.config.progressive_target_system,
                    "call_dir": str(call_dir),
                    "workspace_dir": str(workspace_dir),
                    "attempt": "missing_pending_retry",
                },
            )
            access_violations = self._proposer_access_violations(
                result,
                workspace_dir=workspace_dir,
            )
            if access_violations:
                self._append_event(
                    {
                        "iteration": iteration,
                        "event": "proposer_access_rejected",
                        "selection_policy": "progressive" if adaptive else "default",
                        "budget": budget,
                        "attempt": "missing_pending_retry",
                        "violations": access_violations,
                    }
                )
                return []
            self._archive_workspace_outputs(
                workspace_dir=workspace_dir,
                call_dir=call_dir,
                result=result,
            )
        if result.returncode != 0 or result.timed_out or not self.pending_eval_path.exists():
            self._append_event(
                {
                    "iteration": iteration,
                    "event": "proposer_failed",
                    "selection_policy": "progressive" if adaptive else "default",
                    "budget": budget,
                    "returncode": result.returncode,
                    "timed_out": result.timed_out,
                    "stderr": result.stderr[:1000],
                    "proposer_metrics": getattr(result, "metrics", {}),
                }
            )
            return []

        pending = json.loads(self.pending_eval_path.read_text(encoding="utf-8"))
        proposed = _pending_candidates(pending)
        if len(proposed) != 1:
            self._append_event(
                {
                    "iteration": iteration,
                    "event": "candidate_count_adjusted",
                    "selection_policy": "progressive" if adaptive else "default",
                    "budget": budget,
                    "requested_count": len(proposed),
                    "evaluated_count": min(len(proposed), 1),
                }
            )
            proposed = proposed[:1]
        for raw in proposed:
            if isinstance(raw, dict):
                self._normalize_workspace_candidate_paths(
                    raw,
                    workspace_dir=workspace_dir,
                    workspace_generated_dir=workspace_generated_dir,
                )
                self._rewrite_workspace_source_paths_to_archive(
                    raw,
                    workspace_dir=workspace_dir,
                    archived_source_snapshot=call_dir / "source_snapshot",
                )
                raw.setdefault("source_family", self.config.progressive_target_system)
                raw.setdefault("budget", budget)
                raw.setdefault("reference_iterations", list(reference_iterations))
                raw.setdefault("source_snapshot_path", str(call_dir / "source_snapshot"))
        normalized_pending = json.dumps(
            {"candidates": proposed},
            indent=2,
            ensure_ascii=False,
        )
        self.pending_eval_path.write_text(normalized_pending, encoding="utf-8")
        (call_dir / "pending_eval.json").write_text(normalized_pending, encoding="utf-8")

        evaluated = self._evaluate_proposed(iteration, proposed, examples)
        best_ids = self._best_passrate_ids(existing_candidates + evaluated)
        write_post_eval_artifacts(
            run_dir=self.run_dir,
            call_dir=call_dir,
            iteration=iteration,
            candidates=evaluated,
            frontier_ids=best_ids,
        )
        self._refresh_run_indexes(existing_candidates + evaluated)
        return evaluated

    def _build_progressive_workspace(
        self,
        *,
        iteration: int,
        budget: str,
        existing_candidates: list[CandidateResult],
        call_dir: Path,
    ) -> tuple[Path, tuple[int, ...]]:
        call_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir = call_dir / "workspace"
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)

        workspace_generated_dir = workspace_dir / "generated"
        workspace_generated_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_package_dirs(workspace_generated_dir, root=workspace_generated_dir)

        reference_iterations = self._reference_iterations_for_budget(
            budget,
            iteration=iteration,
            candidates=existing_candidates,
        )
        assignment = {
            "iteration": iteration,
            "target_system": self.config.progressive_target_system,
            "budget": budget,
            "reference_iterations": list(reference_iterations),
            "generated_dir": str(workspace_generated_dir),
            "source_snapshot_dir": str(workspace_dir / "source_snapshot"),
            "pending_eval_path": str(workspace_dir / "pending_eval.json"),
        }
        for dest in (call_dir / "assignment.json", workspace_dir / "assignment.json"):
            dest.write_text(json.dumps(assignment, indent=2, ensure_ascii=False), encoding="utf-8")

        self._copy_workspace_summaries(workspace_dir / "summaries")
        self._copy_reference_iterations(
            workspace_dir / "reference_iterations",
            reference_iterations=reference_iterations,
            budget=budget,
        )
        self._build_source_snapshot_workspace(
            iteration=iteration,
            source_family=self.config.progressive_target_system,
            call_dir=call_dir,
            target_system=self.config.progressive_target_system,
            snapshot_root=workspace_dir / "source_snapshot",
            generated_dir=workspace_generated_dir,
        )
        self._write_workspace_manifest(
            workspace_dir,
            call_dir=call_dir,
            assignment=assignment,
        )
        self._write_access_policy(
            workspace_dir,
            source_snapshot_dir=workspace_dir / "source_snapshot",
            generated_dir=workspace_generated_dir,
            pending_eval_path=workspace_dir / "pending_eval.json",
        )
        return workspace_dir, reference_iterations

    def _copy_workspace_summaries(self, summaries_dir: Path) -> None:
        summaries_dir.mkdir(parents=True, exist_ok=True)
        summary_files = (
            (self.summary_path, "evolution_summary.jsonl", ""),
            (self.frontier_path, "best_candidates.json", "[]\n"),
            (self.candidate_score_table_path, "candidate_score_table.json", "[]\n"),
            (
                self.retrieval_diagnostics_summary_path,
                "retrieval_diagnostics_summary.json",
                "[]\n",
            ),
            (self.iteration_index_path, "iteration_index.json", "[]\n"),
            (self.diff_summary_path, "diff_summary.jsonl", ""),
        )
        for src, name, default_text in summary_files:
            dest = summaries_dir / name
            if src.exists():
                shutil.copy2(src, dest)
            else:
                dest.write_text(default_text, encoding="utf-8")

    def _copy_reference_iterations(
        self,
        reference_dir: Path,
        *,
        reference_iterations: tuple[int, ...],
        budget: str,
    ) -> None:
        reference_dir.mkdir(parents=True, exist_ok=True)
        trace_scope = self._trace_scope_for_budget(budget)
        for item in reference_iterations:
            src = self._iteration_dir(item)
            if not src.exists():
                continue
            self._copy_iteration_bundle(
                src,
                reference_dir / f"iter_{item:03d}",
                trace_scope=trace_scope,
            )

    def _write_workspace_manifest(
        self,
        workspace_dir: Path,
        *,
        call_dir: Path,
        assignment: dict[str, Any],
    ) -> None:
        manifest = {
            "workspace_dir": str(workspace_dir),
            "call_dir": str(call_dir),
            "assignment": assignment,
            "summaries_dir": str(workspace_dir / "summaries"),
            "reference_iterations_dir": str(workspace_dir / "reference_iterations"),
            "source_snapshot_dir": str(workspace_dir / "source_snapshot"),
            "generated_dir": str(workspace_dir / "generated"),
            "pending_eval_path": str(workspace_dir / "pending_eval.json"),
        }
        for dest in (
            workspace_dir / "workspace_manifest.json",
            call_dir / "workspace_manifest.json",
        ):
            dest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_access_policy(
        self,
        workspace_dir: Path,
        *,
        source_snapshot_dir: Path,
        generated_dir: Path,
        pending_eval_path: Path,
    ) -> None:
        policy = {
            "read_roots": [str(workspace_dir)],
            "write_roots": [
                str(source_snapshot_dir / "candidate"),
                str(generated_dir),
            ],
            "write_files": [str(pending_eval_path)],
            "forbidden_roots": [
                str(self.project_root),
                str(self.run_dir),
                str(self.project_root / "references" / "vendor"),
                str(self.run_dir / "candidate_results"),
            ],
            "notes": [
                "The proposer workspace is self-contained.",
                (
                    "Do not read global runs, repo-root source, references/vendor, "
                    f"{self._raw_data_policy_name()}, or scoring helpers."
                ),
            ],
        }
        for dest in (
            workspace_dir / "access_policy.json",
            workspace_dir.parent / "access_policy.json",
        ):
            dest.write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")

    def _archive_workspace_outputs(
        self,
        *,
        workspace_dir: Path,
        call_dir: Path,
        result: Any,
    ) -> None:
        self._sync_workspace_outputs(workspace_dir=workspace_dir, call_dir=call_dir)

        source_snapshot = workspace_dir / "source_snapshot"
        archived_snapshot = call_dir / "source_snapshot"
        if source_snapshot.exists():
            if archived_snapshot.exists():
                shutil.rmtree(archived_snapshot)
            shutil.copytree(
                source_snapshot,
                archived_snapshot,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )

        for name in ("workspace_manifest.json", "access_policy.json"):
            self._copy_if_exists(workspace_dir / name, call_dir / name)

        agent_dir = call_dir / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        tool_access = getattr(result, "tool_access", None)
        if isinstance(tool_access, dict):
            (agent_dir / "tool_access.json").write_text(
                json.dumps(tool_access, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        metrics = getattr(result, "metrics", None)
        if isinstance(metrics, dict):
            (agent_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        self._write_source_snapshot_diff(call_dir)
        write_diff_digest(call_dir=call_dir)
        self._append_diff_summary(call_dir)

    def _write_source_snapshot_diff(self, call_dir: Path) -> None:
        original = call_dir / "source_snapshot" / "candidate" / "original_project_source"
        updated = call_dir / "source_snapshot" / "candidate" / "project_source"
        if not original.exists() or not updated.exists():
            (call_dir / "diff.patch").write_text(
                "Source snapshot diff unavailable.\n",
                encoding="utf-8",
            )
            return
        cmd = ["git", "diff", "--no-index", "--", str(original), str(updated)]
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
            text = f"Failed to capture source snapshot diff: {exc}\n"
        (call_dir / "diff.patch").write_text(text, encoding="utf-8")

    def _append_diff_summary(self, call_dir: Path) -> None:
        iteration = _iteration_from_dir_name(call_dir.name) or 0
        diff_path = call_dir / "diff.patch"
        text = diff_path.read_text(encoding="utf-8", errors="replace") if diff_path.exists() else ""
        files_changed: list[str] = []
        insertions = 0
        deletions = 0
        for line in text.splitlines():
            if line.startswith("diff --git "):
                parts = line.split()
                if len(parts) >= 4:
                    files_changed.append(parts[3].removeprefix("b/"))
            elif line.startswith("+") and not line.startswith("+++"):
                insertions += 1
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1
        row = {
            "iteration": iteration,
            "iteration_dir": str(call_dir),
            "diff_path": str(diff_path),
            "diff_digest_path": str(call_dir / "diff_digest.md"),
            "files_changed": sorted(set(files_changed)),
            "insertions": insertions,
            "deletions": deletions,
        }
        rows: list[dict[str, Any]] = []
        if self.diff_summary_path.exists():
            for raw in self.diff_summary_path.read_text(encoding="utf-8").splitlines():
                if not raw:
                    continue
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict) and int(item.get("iteration") or -1) != iteration:
                    rows.append(item)
        rows.append(row)
        self.diff_summary_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows),
            encoding="utf-8",
        )

    def _load_examples(self) -> list[LocomoExample]:
        if not default_data_path().exists():
            prepare_locomo()
        examples = load_locomo_examples()
        selected = select_split(examples, split=self.config.split)
        if self.config.limit:
            selected = selected[: self.config.limit]
        return selected

    def _run_seed_frontier(self) -> dict[str, Any]:
        return run_initial_frontier(
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

    def _benchmark_prompt_name(self) -> str:
        return "LOCOMO conversational-memory QA"

    def _raw_data_policy_name(self) -> str:
        return "raw LOCOMO data"

    def _run_proposer_agent(
        self,
        prompt: str,
        *,
        log_dir: Path,
        name: str,
        cwd: Path | None = None,
    ) -> Any:
        agent = self.config.proposer_agent.strip().lower()
        model_by_agent = {
            "claude": self.config.claude_model,
            "codex": self.config.codex_model,
            "kimi": self.config.kimi_model,
        }
        model = model_by_agent.get(agent, self.config.claude_model)
        return run_code_agent_prompt(
            prompt,
            agent=agent,
            cwd=cwd or self.project_root,
            log_dir=log_dir,
            name=name,
            model=model,
            timeout_s=self.config.propose_timeout_s,
            sandbox=self._proposer_sandbox_config(),
        )

    def _proposer_sandbox_config(self) -> ProposerSandboxConfig | None:
        kind = self.config.proposer_sandbox.strip().lower()
        if kind == "none":
            return None
        if kind != "docker":
            raise ValueError(f"unsupported proposer sandbox: {self.config.proposer_sandbox!r}")
        docker_env = _dedupe_tuple(DEFAULT_DOCKER_ENV_VARS + self.config.proposer_docker_env)
        return ProposerSandboxConfig(
            kind="docker",
            docker_image=self.config.proposer_docker_image,
            docker_workspace=self.config.proposer_docker_workspace or "/workspace",
            docker_env_vars=docker_env,
            docker_mounts=self.config.proposer_docker_mount,
            docker_kimi_cli_kind=self.config.proposer_docker_kimi_cli_kind,
            docker_user=self.config.proposer_docker_user,
            docker_home=self.config.proposer_docker_home,
        )

    def _evaluate_proposed(
        self,
        iteration: int,
        proposed: list[dict[str, Any]],
        examples: list[LocomoExample],
    ) -> list[CandidateResult]:
        runner = self._make_evaluation_runner(examples)
        results: list[CandidateResult] = []
        for raw in proposed:
            if isinstance(raw, dict):
                raw = dict(raw)
                raw.setdefault("candidate_root", str(self.generated_dir))
            violations = self._candidate_code_policy_violations(raw)
            if violations:
                self._append_event(
                    {
                        "iteration": iteration,
                        "event": "candidate_policy_rejected",
                        "candidate": raw,
                        "violations": violations,
                    }
                )
                continue
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
                "build_tag",
                "class",
                "cost_level",
                "factory",
                "module",
                "module_path",
                "project_source_path",
                "source_base_dir",
                "source_family",
                "source_path",
                "source_project_path",
                "upstream_source_path",
                "mem0_source_path",
                "memgpt_source_path",
                "membank_source_path",
                "optimization_target",
            ):
                if key in raw and key not in extra:
                    extra[key] = raw[key]
            for key, value in self._candidate_extra_defaults().items():
                extra.setdefault(key, value)
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

    def _make_evaluation_runner(self, examples: list[LocomoExample]) -> EvaluationRunner:
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
        )

    def _candidate_extra_defaults(self) -> dict[str, object]:
        return {}

    def _candidate_code_policy_violations(self, candidate: Any) -> list[dict[str, str]]:
        if not isinstance(candidate, dict):
            return []
        paths = self._candidate_policy_scan_paths(candidate)
        violations: list[dict[str, str]] = []
        forbidden = {
            "candidate_results": "runtime code must not read previous candidate results",
            "data/locomo": "runtime code must not read raw LOCOMO data",
            "data\\locomo": "runtime code must not read raw LOCOMO data",
            "locomo10.json": "runtime code must not read raw LOCOMO data",
            "data/longmemeval": "runtime code must not read raw LongMemEval data",
            "data\\longmemeval": "runtime code must not read raw LongMemEval data",
            "longmemeval_s_cleaned.json": "runtime code must not read raw LongMemEval data",
            "longmemeval_m_cleaned.json": "runtime code must not read raw LongMemEval data",
            "longmemeval_oracle.json": "runtime code must not read raw LongMemEval data",
            "score_prediction": "runtime code must not call OptiHarness scoring helpers",
            "memomemo.metrics": "runtime code must not import OptiHarness scoring helpers",
            "load_locomo_examples": "runtime code must not load the full LOCOMO dataset",
            "load_longmemeval_examples": "runtime code must not load the full LongMemEval dataset",
        }
        source_project = self._candidate_source_project_root(candidate)
        original_source_project = (
            self._candidate_original_source_project_root(source_project)
            if source_project is not None
            else None
        )
        for path in paths:
            text = self._candidate_policy_scan_text(
                path,
                source_project=source_project,
                original_source_project=original_source_project,
            )
            if text is None:
                continue
            lower = text.lower()
            for marker, reason in forbidden.items():
                if marker.lower() in lower:
                    violations.append(
                        {
                            "path": str(path),
                            "marker": marker,
                            "reason": reason,
                        }
                    )
        return violations

    def _candidate_policy_scan_text(
        self,
        path: Path,
        *,
        source_project: Path | None,
        original_source_project: Path | None,
    ) -> str | None:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None
        if source_project is None or original_source_project is None:
            return text
        try:
            rel = path.resolve(strict=False).relative_to(source_project.resolve(strict=False))
        except ValueError:
            return text
        original_path = original_source_project / rel
        try:
            original_text = original_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return text
        if text == original_text:
            return ""
        return _added_policy_lines(original_text, text)

    def _proposer_access_violations(
        self,
        result: Any,
        *,
        workspace_dir: Path,
    ) -> list[dict[str, str]]:
        tool_access = getattr(result, "tool_access", None)
        if not isinstance(tool_access, dict):
            return []

        allowed_roots = [workspace_dir]
        violations: list[dict[str, str]] = []

        for raw_path in sorted((tool_access.get("files_read") or {}).keys()):
            path = self._normalize_agent_access_path(raw_path, base_dir=workspace_dir)
            if not self._path_is_under_any(path, allowed_roots):
                violations.append(
                    {
                        "operation": "read",
                        "path": str(path),
                        "reason": "proposer reads must stay inside the scoped workspace",
                    }
                )

        for raw_path in sorted((tool_access.get("files_written") or {}).keys()):
            path = self._normalize_agent_access_path(raw_path, base_dir=workspace_dir)
            if not self._path_is_under_any(path, allowed_roots):
                violations.append(
                    {
                        "operation": "write",
                        "path": str(path),
                        "reason": "proposer writes must stay inside the scoped workspace",
                    }
                )
        return violations

    def _access_retry_note(
        self,
        *,
        violations: list[dict[str, str]],
        workspace_dir: Path,
    ) -> str:
        lines = [
            "## Retry Required: Filesystem Boundary Violation",
            "",
            "Your previous attempt read or wrote files outside the proposer workspace.",
            f"Allowed workspace root: `{workspace_dir.resolve(strict=False)}`",
            "",
            "Do not use absolute paths or `..` paths that leave this directory.",
            "Use only the files copied into the current working directory for this proposer call.",
            "Recreate exactly one candidate from scratch and write a fresh `pending_eval.json`.",
            "",
            "Violations from the previous attempt:",
        ]
        for item in violations:
            operation = item.get("operation", "access")
            path = item.get("path", "")
            reason = item.get("reason", "")
            lines.append(f"- {operation}: `{path}` ({reason})")
        return "\n".join(lines)

    def _normalize_agent_access_path(
        self,
        raw_path: str,
        *,
        base_dir: Path | None = None,
    ) -> Path:
        path = Path(str(raw_path)).expanduser()
        if path.is_absolute() and base_dir is not None:
            path = self._map_container_workspace_path(path, workspace_dir=base_dir)
        if not path.is_absolute():
            path = (base_dir or self.project_root) / path
        return path.resolve(strict=False)

    def _path_is_under_any(self, path: Path, roots: list[Path]) -> bool:
        normalized = path.resolve(strict=False)
        for root in roots:
            root_path = root.resolve(strict=False)
            if normalized == root_path or root_path in normalized.parents:
                return True
        return False

    def _path_matches_any(self, path: Path, files: list[Path]) -> bool:
        normalized = path.resolve(strict=False)
        return any(normalized == item.resolve(strict=False) for item in files)

    def _candidate_policy_scan_paths(self, candidate: dict[str, Any]) -> list[Path]:
        out: list[Path] = []
        module_path = str(candidate.get("module_path") or "").strip()
        if module_path:
            path = Path(module_path).expanduser()
            if not path.is_absolute():
                path = self.project_root / path
            out.append(path)

        candidate_root = candidate.get("candidate_root") or candidate.get("generated_dir")
        root: Path | None = None
        if candidate_root:
            root = Path(str(candidate_root)).expanduser()
            if not root.is_absolute():
                root = self.project_root / root
        module_name = str(candidate.get("module") or "").strip()
        if root is not None and root.exists():
            if module_name:
                rel = Path(*module_name.split(".")).with_suffix(".py")
                module_file = root / rel
                if module_file.exists():
                    out.append(module_file)
            # Historical candidates may import earlier run-local parent modules.
            # Scan that workspace so contaminated parent modules remain visible
            # to the policy check.
            out.extend(sorted(root.glob("*.py")))

        source_project = self._candidate_source_project_root(candidate)
        if source_project is not None:
            package_dir = source_project / "src" / "memomemo"
            if package_dir.exists():
                out.extend(sorted(package_dir.rglob("*.py")))
        return sorted(set(out))

    def _candidate_source_project_root(self, candidate: dict[str, Any]) -> Path | None:
        extra = candidate.get("extra") if isinstance(candidate.get("extra"), dict) else {}
        for key in ("source_project_path", "project_source_path", "memomemo_source_path"):
            value = candidate.get(key) or extra.get(key)
            if not value:
                continue
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = self.project_root / path
            if (path / "src").exists():
                return path
            if path.name == "src":
                return path.parent
            return path
        return None

    def _candidate_original_source_project_root(self, source_project: Path) -> Path | None:
        candidate_dir = source_project.parent if source_project.name == "project_source" else None
        if candidate_dir is not None:
            original = candidate_dir / "original_project_source"
            if (original / "src").exists():
                return original

        for parent in source_project.parents:
            manifest_path = parent / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            original_value = manifest.get("original_project_source")
            if not original_value:
                continue
            original = Path(str(original_value)).expanduser()
            if not original.is_absolute():
                original = manifest_path.parent / original
            if (original / "src").exists():
                return original
        return None

    def _load_existing_candidates(self) -> list[CandidateResult]:
        out: list[CandidateResult] = []
        for path in sorted((self.run_dir / "candidate_results").glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                out.append(CandidateResult.from_dict(payload["candidate"]))
            except Exception:
                continue
        return out

    def _iteration_dir(self, iteration: int) -> Path:
        return self.run_dir / "proposer_calls" / f"iter_{iteration:03d}"

    def _workspace_dir(self, iteration: int) -> Path:
        return self._iteration_dir(iteration) / "workspace"

    def _progressive_budget_for_iteration(self, iteration: int) -> str:
        if iteration <= self.config.progressive_initial_low_iterations:
            return "low"
        state = self._load_progressive_state()
        budget = str(state.get("next_budget") or state.get("current_budget") or "low")
        if budget not in {"low", "medium", "high"}:
            return "low"
        return budget

    def _load_progressive_state(self) -> dict[str, Any]:
        if not self.progressive_state_path.exists():
            return {
                "current_budget": "low",
                "next_budget": "low",
                "stagnation_count": 0,
                "best_passrate": 0.0,
                "best_candidate_id": None,
                "last_improved_iteration": 0,
            }
        try:
            payload = json.loads(self.progressive_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _update_progressive_state(
        self,
        *,
        iteration: int,
        budget: str,
        previous_best_passrate: float,
        candidates: list[CandidateResult],
        evaluated: list[CandidateResult],
    ) -> None:
        best = max(candidates, key=_candidate_score) if candidates else None
        best_passrate = best.passrate if best is not None else 0.0
        improved = bool(evaluated and best_passrate > previous_best_passrate)
        prior = self._load_progressive_state()
        stagnation = 0 if improved else int(prior.get("stagnation_count") or 0) + 1
        if iteration < self.config.progressive_initial_low_iterations:
            next_budget = "low"
        elif improved:
            next_budget = "low"
        elif budget == "low":
            next_budget = "medium"
        elif budget == "medium":
            next_budget = "high"
        else:
            next_budget = "high"
        state = {
            "current_budget": budget,
            "next_budget": next_budget,
            "stagnation_count": stagnation,
            "best_passrate": best_passrate,
            "best_candidate_id": best.candidate_id if best is not None else None,
            "last_improved_iteration": (
                iteration if improved else int(prior.get("last_improved_iteration") or 0)
            ),
        }
        self.progressive_state_path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _reference_iterations_for_budget(
        self,
        budget: str,
        *,
        iteration: int,
        candidates: list[CandidateResult],
    ) -> tuple[int, ...]:
        available = {
            item
            for item in self._candidate_iterations(candidates)
            if 0 < item < iteration and self._iteration_dir(item).exists()
        }
        if budget == "high":
            return tuple(sorted(available))

        selected: list[int] = []
        selected.extend(self._best_iterations(candidates, k=3 if budget == "medium" else 1))
        worst = self._worst_iteration(candidates)
        if worst is not None:
            selected.append(worst)
        out: list[int] = []
        seen: set[int] = set()
        for item in selected:
            if item not in available or item in seen:
                continue
            seen.add(item)
            out.append(item)
        return tuple(out)

    def _best_iterations(self, candidates: list[CandidateResult], *, k: int) -> list[int]:
        out: list[int] = []
        seen: set[int] = set()
        for candidate in sorted(candidates, key=_candidate_best_rank):
            iteration = _candidate_iteration(candidate.candidate_id)
            if iteration is None or iteration <= 0 or iteration in seen:
                continue
            seen.add(iteration)
            out.append(iteration)
            if len(out) >= k:
                break
        return out

    def _worst_iteration(self, candidates: list[CandidateResult]) -> int | None:
        evaluated = [
            item for item in candidates if (_candidate_iteration(item.candidate_id) or 0) > 0
        ]
        if not evaluated:
            return None
        return _candidate_iteration(min(evaluated, key=_candidate_worst_rank).candidate_id)

    def _candidate_iterations(self, candidates: list[CandidateResult]) -> set[int]:
        out: set[int] = set()
        for candidate in candidates:
            iteration = _candidate_iteration(candidate.candidate_id)
            if iteration is not None:
                out.add(iteration)
        return out

    def _trace_scope_for_budget(self, budget: str) -> str:
        if budget == "low":
            return "last1"
        if budget == "medium":
            return "last3"
        return "all"

    def _best_passrate(self, candidates: list[CandidateResult]) -> float:
        return max((item.passrate for item in candidates), default=0.0)

    def _refresh_run_indexes(self, candidates: list[CandidateResult]) -> None:
        self._write_candidate_score_table_from_candidates(candidates)
        if not self.retrieval_diagnostics_summary_path.exists():
            self.retrieval_diagnostics_summary_path.write_text("[]\n", encoding="utf-8")
        self._write_iteration_index(candidates)
        if not self.diff_summary_path.exists():
            self.diff_summary_path.write_text("", encoding="utf-8")

    def _write_candidate_score_table_from_candidates(
        self,
        candidates: list[CandidateResult],
    ) -> None:
        rows = []
        best_ids = self._best_passrate_ids(candidates)
        for candidate in sorted(
            candidates,
            key=lambda item: ((_candidate_iteration(item.candidate_id) or 0), item.candidate_id),
        ):
            extra = self._candidate_extra(candidate)
            iteration = _candidate_iteration(candidate.candidate_id) or 0
            rows.append(
                {
                    "iteration": iteration,
                    "candidate_id": candidate.candidate_id,
                    "scaffold_name": candidate.scaffold_name,
                    "passrate": candidate.passrate,
                    "average_score": candidate.average_score,
                    "token_consuming": candidate.token_consuming,
                    "source_family": extra.get("source_family"),
                    "build_tag": extra.get("build_tag"),
                    "result_path": candidate.result_path,
                    "iteration_dir": str(self._iteration_dir(iteration)),
                    "is_best_passrate": candidate.candidate_id in best_ids,
                }
            )
        self.candidate_score_table_path.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _write_iteration_index(self, candidates: list[CandidateResult]) -> None:
        by_iteration: dict[int, dict[str, Any]] = {}
        for candidate in candidates:
            iteration = _candidate_iteration(candidate.candidate_id) or 0
            call_dir = self._iteration_dir(iteration)
            row = by_iteration.setdefault(
                iteration,
                {
                    "iteration": iteration,
                    "iteration_dir": str(call_dir),
                    "candidate_ids": [],
                    "candidate_result_paths": [],
                    "compact_result_path": str(call_dir / "eval" / "candidate_result.compact.json"),
                    "retrieval_diagnostics_path": str(
                        call_dir / "eval" / "retrieval_diagnostics.json"
                    ),
                    "trace_slices_dir": str(call_dir / "trace_slices"),
                    "diff_path": str(call_dir / "diff.patch"),
                    "diff_digest_path": str(call_dir / "diff_digest.md"),
                    "source_snapshot_dir": str(call_dir / "source_snapshot"),
                    "generated_dir": str(call_dir / "generated"),
                },
            )
            row["candidate_ids"].append(candidate.candidate_id)
            row["candidate_result_paths"].append(candidate.result_path)

        for call_dir in sorted((self.run_dir / "proposer_calls").glob("iter_*")):
            iteration = _iteration_from_dir_name(call_dir.name)
            if iteration is None:
                continue
            by_iteration.setdefault(
                iteration,
                {
                    "iteration": iteration,
                    "iteration_dir": str(call_dir),
                    "candidate_ids": [],
                    "candidate_result_paths": [],
                    "compact_result_path": str(call_dir / "eval" / "candidate_result.compact.json"),
                    "retrieval_diagnostics_path": str(
                        call_dir / "eval" / "retrieval_diagnostics.json"
                    ),
                    "trace_slices_dir": str(call_dir / "trace_slices"),
                    "diff_path": str(call_dir / "diff.patch"),
                    "diff_digest_path": str(call_dir / "diff_digest.md"),
                    "source_snapshot_dir": str(call_dir / "source_snapshot"),
                    "generated_dir": str(call_dir / "generated"),
                },
            )

        rows = [by_iteration[key] for key in sorted(by_iteration)]
        self.iteration_index_path.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _optimization_direction_lines(self, target_system: str | None = None) -> tuple[str, ...]:
        """Return prompt-only optimization directions for proposers."""

        lines: list[str] = []
        target = target_system or self.config.progressive_target_system
        for cell in get_target_cells(target):
            focus = ", ".join(cell.focus_functions) if cell.focus_functions else "all functions"
            lines.append(f"{cell.name}: {cell.description} Focus areas: {focus}.")
        if not lines:
            lines.append(
                "global: improve memory construction, retrieval, evidence selection, "
                "and answering."
            )
        return tuple(lines)

    def _candidate_extra(self, candidate: CandidateResult) -> dict[str, Any]:
        extra = candidate.config.get("extra") if isinstance(candidate.config, dict) else None
        if isinstance(extra, dict):
            return dict(extra)
        return {}

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
        return "fusion"

    def _build_source_snapshot_workspace(
        self,
        *,
        iteration: int,
        source_family: str,
        call_dir: Path,
        target_system: str | None = None,
        snapshot_root: Path | None = None,
        generated_dir: Path | None = None,
    ) -> Path:
        generated_dir = generated_dir or self.generated_dir
        snapshot_root = snapshot_root or (
            generated_dir / "source_snapshots" / f"iter_{iteration:03d}"
        )
        if snapshot_root.exists():
            shutil.rmtree(snapshot_root)
        snapshot_root.mkdir(parents=True, exist_ok=True)
        self._ensure_package_dirs(snapshot_root, root=snapshot_root)

        scaffold_source = self._source_scaffold_path(source_family)
        source_files = [path for path in (scaffold_source,) if path is not None and path.exists()]

        candidate_dir = snapshot_root / "candidate"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_package_dirs(candidate_dir, root=snapshot_root)
        for path in source_files:
            self._copy_if_exists(path, candidate_dir / path.name)
        self._copy_project_source_context(candidate_dir)
        original_project_source = candidate_dir / "original_project_source"
        project_source = candidate_dir / "project_source"
        if project_source.exists():
            if original_project_source.exists():
                shutil.rmtree(original_project_source)
            shutil.copytree(
                project_source,
                original_project_source,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
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
                    f"Target system: {target_system or source_family}",
                    "",
                    "This directory is a writable candidate-specific clean source snapshot.",
                    "It also contains benchmark-scoped project source under",
                    "`project_source/src/memomemo` and relevant upstream source",
                    "under `upstream_source` for inspection.",
                    "Historical iterations are diagnostic references only; do not treat",
                    "their source snapshots as editable parents.",
                    "Existing source-backed base memories are read-only. You may edit",
                    "copied build/database-construction paths such as add/build/schema/",
                    "extraction/evolution/embedding or persistence layout, but source",
                    "edits that alter persisted memories must use a fresh source_base_dir",
                    "and build_tag in pending_eval.json.",
                    "Modify files here for the mechanism under test, then expose the",
                    "edited built-in source scaffold in `pending_eval.json` with",
                    "`scaffold_name` and `extra.source_project_path`.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        manifest = {
            "iteration": iteration,
            "source_family": source_family,
            "target_system": target_system or source_family,
            "benchmark": self.workspace_spec.benchmark,
            "candidate_dir": str(candidate_dir),
            "project_source": str(project_source),
            "original_project_source": str(original_project_source),
            "primary_source_file": self.workspace_spec.primary_source_file,
            "project_source_files": list(self.workspace_spec.source_files),
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

    def _ensure_package_dirs(self, path: Path, *, root: Path | None = None) -> None:
        generated_root = root or self.generated_dir
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

    def _copy_iteration_bundle(self, src: Path, dest: Path, *, trace_scope: str) -> None:
        if not src.exists():
            return
        if dest.exists():
            shutil.rmtree(dest)
        ignore = shutil.ignore_patterns(
            "workspace",
            "context",
            "claude_session",
            "__pycache__",
            "*.pyc",
        )
        shutil.copytree(src, dest, ignore=ignore)
        self._prune_trace_slices_for_scope(dest, trace_scope=trace_scope)

    def _prune_trace_slices_for_scope(self, bundle_dir: Path, *, trace_scope: str) -> None:
        trace_dir = bundle_dir / "trace_slices"
        if not trace_dir.exists():
            return
        if trace_scope == "last1":
            allowed = {"low"}
        elif trace_scope == "last3":
            allowed = {"medium"}
        else:
            allowed = {"low", "medium", "high"}
        for child in trace_dir.iterdir():
            if child.is_dir() and child.name not in allowed:
                shutil.rmtree(child)

    def _copy_if_exists(self, src: Path, dest: Path) -> None:
        if src.exists() and src.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    def _sync_workspace_outputs(
        self,
        *,
        workspace_dir: Path,
        call_dir: Path,
    ) -> None:
        workspace_generated_dir = workspace_dir / "generated"
        if workspace_generated_dir.exists():
            self.generated_dir.mkdir(parents=True, exist_ok=True)
            self._ensure_package_dirs(self.generated_dir)
            call_generated_dir = call_dir / "generated"
            if call_generated_dir.exists():
                shutil.rmtree(call_generated_dir)
            call_generated_dir.mkdir(parents=True, exist_ok=True)
            for src in sorted(workspace_generated_dir.rglob("*")):
                if not src.is_file():
                    continue
                if "__pycache__" in src.parts or src.suffix == ".pyc":
                    continue
                rel = src.relative_to(workspace_generated_dir)
                dest = self.generated_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                call_dest = call_generated_dir / rel
                call_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, call_dest)

        workspace_pending = workspace_dir / "pending_eval.json"
        if workspace_pending.exists():
            self._copy_if_exists(workspace_pending, self.pending_eval_path)
            self._copy_if_exists(workspace_pending, call_dir / "pending_eval.raw.json")

    def _normalize_workspace_candidate_paths(
        self,
        candidate: dict[str, Any],
        *,
        workspace_dir: Path,
        workspace_generated_dir: Path,
    ) -> None:
        candidate["candidate_root"] = str(self.generated_dir)
        if candidate.get("generated_dir"):
            candidate["generated_dir"] = str(self.generated_dir)
        self._normalize_workspace_path_fields(candidate, workspace_dir, workspace_generated_dir)
        extra = candidate.get("extra")
        if isinstance(extra, dict):
            self._normalize_workspace_path_fields(extra, workspace_dir, workspace_generated_dir)

    def _normalize_workspace_path_fields(
        self,
        payload: dict[str, Any],
        workspace_dir: Path,
        workspace_generated_dir: Path,
    ) -> None:
        for key in (
            "module_path",
            "source_path",
            "source_project_path",
            "project_source_path",
            "memomemo_source_path",
            "upstream_source_path",
            "mem0_source_path",
            "memgpt_source_path",
            "membank_source_path",
            "source_base_dir",
            "base_memory_dir",
        ):
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            payload[key] = str(
                self._resolve_workspace_path(
                    value,
                    workspace_dir=workspace_dir,
                    workspace_generated_dir=workspace_generated_dir,
                )
            )

    def _rewrite_workspace_source_paths_to_archive(
        self,
        candidate: dict[str, Any],
        *,
        workspace_dir: Path,
        archived_source_snapshot: Path,
    ) -> None:
        self._rewrite_workspace_source_path_fields(
            candidate,
            workspace_dir=workspace_dir,
            archived_source_snapshot=archived_source_snapshot,
        )
        extra = candidate.get("extra")
        if isinstance(extra, dict):
            self._rewrite_workspace_source_path_fields(
                extra,
                workspace_dir=workspace_dir,
                archived_source_snapshot=archived_source_snapshot,
            )

    def _rewrite_workspace_source_path_fields(
        self,
        payload: dict[str, Any],
        *,
        workspace_dir: Path,
        archived_source_snapshot: Path,
    ) -> None:
        workspace_source = (workspace_dir / "source_snapshot").resolve(strict=False)
        archived_source = archived_source_snapshot.resolve(strict=False)
        for key in (
            "source_snapshot_path",
            "source_project_path",
            "project_source_path",
            "memomemo_source_path",
            "upstream_source_path",
            "mem0_source_path",
            "memgpt_source_path",
            "membank_source_path",
        ):
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = workspace_dir / path
            path = self._map_container_workspace_path(path, workspace_dir=workspace_dir)
            resolved = path.resolve(strict=False)
            if resolved == workspace_source or workspace_source in resolved.parents:
                rel = resolved.relative_to(workspace_source)
                payload[key] = str((archived_source / rel).resolve(strict=False))

    def _resolve_workspace_path(
        self,
        value: str,
        *,
        workspace_dir: Path,
        workspace_generated_dir: Path,
    ) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            path = self._map_container_workspace_path(path, workspace_dir=workspace_dir)
        if not path.is_absolute():
            path = workspace_dir / path
        resolved = path.resolve(strict=False)
        workspace_generated = workspace_generated_dir.resolve(strict=False)
        if resolved == workspace_generated or workspace_generated in resolved.parents:
            rel = resolved.relative_to(workspace_generated)
            return (self.generated_dir / rel).resolve(strict=False)
        return resolved

    def _map_container_workspace_path(self, path: Path, *, workspace_dir: Path) -> Path:
        if self.config.proposer_sandbox.strip().lower() != "docker":
            return path
        container_root = Path(self.config.proposer_docker_workspace or "/workspace")
        try:
            rel = path.relative_to(container_root)
        except ValueError:
            return path
        return workspace_dir / rel

    def _copy_tree_if_exists(
        self,
        src: Path,
        dest: Path,
        *,
        ignore_names: tuple[str, ...] = (),
    ) -> None:
        if not src.exists() or not src.is_dir():
            return
        if dest.exists():
            shutil.rmtree(dest)
        ignore_patterns = (
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "__pycache__",
            "*.pyc",
            *ignore_names,
        )
        shutil.copytree(
            src,
            dest,
            ignore=shutil.ignore_patterns(*ignore_patterns),
        )

    def _copy_project_source_context(self, dest_dir: Path) -> None:
        source_pkg = dest_dir / "project_source" / "src" / "memomemo"
        copied = copy_benchmark_project_source(
            project_root=self.project_root,
            dest_pkg=source_pkg,
            spec=self.workspace_spec,
        )
        (dest_dir / "project_source_manifest.json").write_text(
            json.dumps(
                {
                    "benchmark": self.workspace_spec.benchmark,
                    "primary_source_file": self.workspace_spec.primary_source_file,
                    "source_files": list(copied),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

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

    def _best_passrate_ids(self, candidates: list[CandidateResult]) -> set[str]:
        if not candidates:
            return set()
        best = max(item.passrate for item in candidates)
        return {item.candidate_id for item in candidates if item.passrate == best}

    def _save_best_candidates(self, candidates: list[CandidateResult]) -> None:
        self.frontier_path.parent.mkdir(parents=True, exist_ok=True)
        best_ids = self._best_passrate_ids(candidates)
        payload = [
            item.to_dict()
            for item in sorted(
                candidates,
                key=lambda item: (-item.passrate, item.token_consuming, item.candidate_id),
            )
            if item.candidate_id in best_ids
        ]
        self.frontier_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _append_summary(
        self,
        *,
        iteration: int,
        candidate: CandidateResult,
        proposal: dict[str, Any] | None = None,
    ) -> None:
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "iteration": iteration,
            "candidate": candidate.to_dict(),
            "proposal": proposal or {},
            "self_best": [candidate.to_dict()],
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
        tool_access_raw = getattr(result, "tool_access", {}) or {}
        tool_access = tool_access_raw if isinstance(tool_access_raw, dict) else {}
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "iteration": iteration,
            "event": "proposer_result",
            "selection_policy": selection_policy,
            "proposer_agent": self.config.proposer_agent,
            "returncode": getattr(result, "returncode", None),
            "timed_out": bool(getattr(result, "timed_out", False)),
            "proposer_metrics": getattr(result, "metrics", {}) or {},
            "usage": getattr(result, "usage", None),
            "files_read": tool_access.get("files_read", {}),
            "files_written": tool_access.get("files_written", {}),
            "grep_requests": tool_access.get("grep_requests", []),
            "tool_counts": tool_access.get("tool_counts", {}),
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


LocomoOptimizerConfig = OptimizerConfig
MemoOptimizer = LocomoOptimizer


def _candidate_score(item: CandidateResult) -> tuple[float, int, str]:
    return (item.passrate, -item.token_consuming, item.candidate_id)


def _candidate_best_rank(item: CandidateResult) -> tuple[float, float, int, str]:
    return (-item.passrate, -item.average_score, item.token_consuming, item.candidate_id)


def _candidate_worst_rank(item: CandidateResult) -> tuple[float, float, int, str]:
    return (item.passrate, item.average_score, -item.token_consuming, item.candidate_id)


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


def _iteration_from_dir_name(name: str) -> int | None:
    if not name.startswith("iter_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def _added_policy_lines(original: str, updated: str) -> str:
    original_lines = original.splitlines()
    updated_lines = updated.splitlines()
    prefix = 0
    limit = min(len(original_lines), len(updated_lines))
    while prefix < limit and original_lines[prefix] == updated_lines[prefix]:
        prefix += 1

    suffix = 0
    original_remaining = len(original_lines) - prefix
    updated_remaining = len(updated_lines) - prefix
    while (
        suffix < original_remaining
        and suffix < updated_remaining
        and original_lines[len(original_lines) - 1 - suffix]
        == updated_lines[len(updated_lines) - 1 - suffix]
    ):
        suffix += 1

    end = len(updated_lines) - suffix if suffix else len(updated_lines)
    return "\n".join(updated_lines[prefix:end])


def _single_top_k(raw: Any) -> tuple[int, bool]:
    if isinstance(raw, int):
        return raw, False
    if isinstance(raw, list) and raw:
        return int(raw[0]), len(raw) != 1
    return int(raw or 8), raw != 8


def _dedupe_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)
