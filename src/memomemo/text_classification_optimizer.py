"""Proposer optimizer for text-classification memory systems."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from memomemo.benchmark_workspaces import (
    TEXT_CLASSIFICATION_WORKSPACE_SPEC,
    BenchmarkWorkspaceSpec,
    copy_benchmark_project_source,
)
from memomemo.claude_runner import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_DOCKER_ENV_VARS,
    DEFAULT_KIMI_MODEL,
    ProposerSandboxConfig,
    run_code_agent_prompt,
)
from memomemo.model import DEFAULT_BASE_URL, DEFAULT_MODEL, LocalModelClient
from memomemo.pareto import ParetoPoint, save_frontier
from memomemo.schemas import CandidateResult
from memomemo.text_classification import (
    DEFAULT_TEXT_CLASSIFICATION_BASELINES,
    DEFAULT_TEXT_CLASSIFICATION_DATASETS,
    DEFAULT_TEXT_CLASSIFICATION_SEEDS,
    DEFAULT_TEXT_CLASSIFICATION_SPLITS,
    PromptLLM,
    evaluate_text_classification_memory,
    load_text_classification_splits,
    run_text_classification_benchmark,
)
from memomemo.text_classification_dynamic import load_candidate_text_memory
from memomemo.text_classification_proposer_prompt import (
    build_text_classification_proposer_prompt,
)


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
class TextClassificationOptimizerConfig:
    """Configuration for text-classification few-shot optimization."""

    run_id: str
    out_dir: Path
    iterations: int = 20
    mode: str = "offline"
    datasets: tuple[str, ...] = DEFAULT_TEXT_CLASSIFICATION_DATASETS
    seeds: tuple[int, ...] = DEFAULT_TEXT_CLASSIFICATION_SEEDS
    num_train: int | None = None
    num_val: int | None = None
    num_test: int | None = None
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
    temperature: float = 0.0
    max_eval_workers: int = 1
    skip_baseline_eval: bool = False
    force: bool = False
    selection_policy: str = "default"
    progressive_initial_low_iterations: int = 5
    proposer_sandbox: str = "docker"
    proposer_docker_image: str = ""
    proposer_docker_workspace: str = "/workspace"
    proposer_docker_env: tuple[str, ...] = ()
    proposer_docker_mount: tuple[str, ...] = ()
    proposer_docker_kimi_cli_kind: str = "claude"
    proposer_docker_user: str = ""
    proposer_docker_home: str = ""


class TextClassificationOptimizer:
    """Meta-harness-style optimizer for text-classification few-shot memory."""

    workspace_spec: BenchmarkWorkspaceSpec = TEXT_CLASSIFICATION_WORKSPACE_SPEC

    def __init__(self, config: TextClassificationOptimizerConfig) -> None:
        self.config = config
        self.project_root = Path(__file__).resolve().parents[2]
        self.run_dir = config.out_dir
        self.pending_eval_path = self.run_dir / "pending_eval.json"
        self.frontier_path = self.run_dir / "best_candidates.json"
        self.summary_path = self.run_dir / "evolution_summary.jsonl"
        self.generated_dir = self.run_dir / "generated"
        self.progressive_state_path = self.run_dir / "progressive_state.json"
        self.candidate_score_table_path = self.run_dir / "candidate_score_table.json"
        self.iteration_index_path = self.run_dir / "iteration_index.json"

    def run(self) -> dict[str, Any]:
        if self.config.mode not in {"online", "offline"}:
            raise ValueError("--text-classification-mode must be online or offline")
        if self.config.selection_policy not in {"default", "progressive"}:
            raise ValueError("--selection-policy must be default or progressive")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_package_dirs(self.generated_dir)

        candidates: list[CandidateResult] = []
        if not self.config.skip_baseline_eval:
            baseline_summary = run_text_classification_benchmark(
                out_dir=self.run_dir,
                datasets=self.config.datasets,
                memory_systems=DEFAULT_TEXT_CLASSIFICATION_BASELINES,
                seeds=self.config.seeds,
                num_train=self.config.num_train,
                num_val=self.config.num_val,
                num_test=self.config.num_test,
                mode=self.config.mode,
                model=self.config.model,
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout_s=self.config.eval_timeout_s,
                dry_run=self.config.dry_run,
                temperature=self.config.temperature,
                max_eval_workers=self.config.max_eval_workers,
                force=self.config.force,
            )
            for item in baseline_summary.get("candidates", []):
                candidate = CandidateResult.from_dict(item)
                candidates.append(candidate)
                self._append_summary(iteration=0, candidate=candidate)
        else:
            candidates.extend(self._load_existing_candidates())

        self._save_best_candidates(candidates)
        self._refresh_run_indexes(candidates)

        for iteration in range(1, self.config.iterations + 1):
            previous_best_passrate = self._best_passrate(candidates)
            budget = (
                self._progressive_budget_for_iteration(iteration)
                if self.config.selection_policy == "progressive"
                else "high"
            )
            evaluated = self._run_proposer_iteration(
                iteration,
                candidates,
                budget=budget,
            )
            candidates.extend(evaluated)
            self._save_best_candidates(candidates)
            self._refresh_run_indexes(candidates)
            self._write_pareto(candidates)
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
            "task": f"text_classification_{self.config.mode}",
            "out_dir": str(self.run_dir),
            "iterations": self.config.iterations,
            "candidate_count": len(candidates),
            "best_candidates_path": str(self.frontier_path),
            "selection_policy": self.config.selection_policy,
        }
        (self.run_dir / "optimizer_summary.json").write_text(
            json.dumps(final_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return final_summary

    def _run_proposer_iteration(
        self,
        iteration: int,
        existing_candidates: list[CandidateResult],
        *,
        budget: str,
    ) -> list[CandidateResult]:
        if self.pending_eval_path.exists():
            self.pending_eval_path.unlink()

        call_dir = self.run_dir / "proposer_calls" / f"iter_{iteration:03d}"
        workspace_dir = self._build_workspace(
            iteration=iteration,
            call_dir=call_dir,
        )
        base_prompt = build_text_classification_proposer_prompt(
            run_id=self.config.run_id,
            iteration=iteration,
            run_dir=workspace_dir,
            pending_eval_path=workspace_dir / "pending_eval.json",
            summaries_dir=workspace_dir / "summaries",
            generated_dir=workspace_dir / "generated",
            source_snapshot_dir=workspace_dir / "source_snapshot",
            mode=self.config.mode,
            dataset=",".join(self.config.datasets),
            num_train=self.config.num_train,
            num_val=self.config.num_val,
            num_test=self.config.num_test,
            source_files=self.workspace_spec.source_files,
            primary_source_file=self.workspace_spec.primary_source_file,
            selection_policy=self.config.selection_policy,
            context_budget=budget,
        )
        result: Any | None = None
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            prompt = base_prompt
            if attempt > 1:
                prompt = (
                    f"{base_prompt}\n\n"
                    "## Required Repair\n\n"
                    "The previous proposer attempt exited without writing "
                    f"`{workspace_dir / 'pending_eval.json'}`. Continue in the "
                    "same workspace, make a concrete candidate source change if "
                    "needed, and write exactly one valid `pending_eval.json`. "
                    "Do not run the full harness evaluation."
                )
            result = self._run_proposer_agent(
                prompt,
                log_dir=call_dir / "agent" / f"attempt_{attempt:02d}",
                name="text-classification-proposer",
                cwd=workspace_dir,
            )
            self._append_event(
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "iteration": iteration,
                    "event": "proposer_result",
                    "task": f"text_classification_{self.config.mode}",
                    "selection_policy": self.config.selection_policy,
                    "context_budget": budget,
                    "attempt": attempt,
                    "returncode": getattr(result, "returncode", None),
                    "timed_out": bool(getattr(result, "timed_out", False)),
                    "proposer_metrics": getattr(result, "metrics", {}) or {},
                    "usage": getattr(result, "usage", None),
                }
            )
            self._archive_workspace_outputs(workspace_dir=workspace_dir, call_dir=call_dir)
            if (
                result.returncode != 0
                or result.timed_out
                or self.pending_eval_path.exists()
            ):
                break
            if attempt < max_attempts:
                self._append_event(
                    {
                        "iteration": iteration,
                        "event": "proposer_missing_pending_retry",
                        "selection_policy": self.config.selection_policy,
                        "context_budget": budget,
                        "attempt": attempt,
                    }
                )

        assert result is not None

        if (
            result.returncode != 0
            or result.timed_out
            or not self.pending_eval_path.exists()
        ):
            self._append_event(
                {
                    "iteration": iteration,
                    "event": "proposer_failed",
                    "returncode": result.returncode,
                    "timed_out": result.timed_out,
                    "stderr": result.stderr[:1000],
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
                    "requested_count": len(proposed),
                }
            )
            proposed = proposed[:1]
        for raw in proposed:
            if isinstance(raw, dict):
                self._normalize_candidate_paths(
                    raw,
                    workspace_dir=workspace_dir,
                    call_dir=call_dir,
                )
                raw.setdefault("source_family", "text_classification_fewshot")
                raw.setdefault("memory_system", "fewshot_all")
                raw.setdefault("task", f"text_classification_{self.config.mode}")
        self.pending_eval_path.write_text(
            json.dumps({"candidates": proposed}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (call_dir / "pending_eval.json").write_text(
            json.dumps({"candidates": proposed}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        evaluated = self._evaluate_proposed(iteration, proposed)
        self._refresh_run_indexes(existing_candidates + evaluated)
        return evaluated

    def _best_passrate(self, candidates: list[CandidateResult]) -> float:
        return max((item.passrate for item in candidates), default=0.0)

    def _progressive_budget_for_iteration(self, iteration: int) -> str:
        if iteration <= self.config.progressive_initial_low_iterations:
            return "low"
        try:
            state = json.loads(self.progressive_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "medium"
        next_budget = str(state.get("next_budget") or "medium").lower()
        return next_budget if next_budget in {"low", "medium", "high"} else "medium"

    def _update_progressive_state(
        self,
        *,
        iteration: int,
        budget: str,
        previous_best_passrate: float,
        candidates: list[CandidateResult],
        evaluated: list[CandidateResult],
    ) -> None:
        current_best = self._best_passrate(candidates)
        improved = current_best > previous_best_passrate
        if improved:
            next_budget = "low"
        elif budget == "low":
            next_budget = "medium"
        elif budget == "medium":
            next_budget = "high"
        else:
            next_budget = "high"
        payload = {
            "selection_policy": "progressive",
            "last_iteration": iteration,
            "last_budget": budget,
            "next_budget": next_budget,
            "previous_best_passrate": previous_best_passrate,
            "current_best_passrate": current_best,
            "improved": improved,
            "evaluated_candidate_ids": [item.candidate_id for item in evaluated],
        }
        self.progressive_state_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _evaluate_proposed(
        self,
        iteration: int,
        proposed: list[dict[str, Any]],
    ) -> list[CandidateResult]:
        results: list[CandidateResult] = []
        for raw in proposed:
            if not isinstance(raw, dict):
                continue
            raw = dict(raw)
            raw.setdefault("candidate_root", str(self.generated_dir))
            violations = self._candidate_policy_violations(raw)
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

            candidate_name = str(raw.get("name") or "text_memory")
            candidate_id = f"iter{iteration:03d}_{candidate_name}"
            try:
                candidate = self._evaluate_candidate(candidate_id, candidate_name, raw)
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
            results.append(candidate)
            self._append_summary(iteration=iteration, candidate=candidate, proposal=raw)
        return results

    def _evaluate_candidate(
        self,
        candidate_id: str,
        candidate_name: str,
        raw: dict[str, Any],
    ) -> CandidateResult:
        rows: list[dict[str, Any]] = []
        for dataset in self.config.datasets:
            for seed in self.config.seeds:
                splits = load_text_classification_splits(
                    dataset,
                    num_train=self.config.num_train,
                    num_val=self.config.num_val,
                    num_test=self.config.num_test,
                    shuffle_seed=seed,
                )
                client = LocalModelClient(
                    model=self.config.model,
                    base_url=self.config.base_url,
                    api_key=self.config.api_key,
                    timeout_s=self.config.eval_timeout_s,
                )
                llm = PromptLLM(
                    client=client,
                    dry_run=self.config.dry_run,
                    temperature=self.config.temperature,
                )
                memory = load_candidate_text_memory(
                    raw,
                    project_root=self.project_root,
                    llm=llm,
                )
                rows.append(
                    evaluate_text_classification_memory(
                        memory=memory,
                        llm=llm,
                        splits=splits,
                        dataset=dataset,
                        seed=seed,
                        memory_name=candidate_name,
                        mode=self.config.mode,
                        num_epochs=1,
                        model=self.config.model,
                        base_url=self.config.base_url,
                        dry_run=self.config.dry_run,
                        max_eval_workers=self.config.max_eval_workers,
                    )
                )

        use_test_metric = self.config.num_test is None or self.config.num_test > 0
        metric = "test_accuracy" if use_test_metric else "val_accuracy"
        total_key = "test_total" if use_test_metric else "val_total"
        accuracies = [float(row.get(metric, 0.0) or 0.0) for row in rows]
        count = sum(int(row.get(total_key, 0) or 0) for row in rows)
        prompt_tokens = sum(int(row.get("llm_input_tokens", 0) or 0) for row in rows)
        completion_tokens = sum(
            int(row.get("llm_output_tokens", 0) or 0) for row in rows
        )
        total_tokens = prompt_tokens + completion_tokens
        passrate = sum(accuracies) / len(accuracies) if accuracies else 0.0
        result_path = self.run_dir / "candidate_results" / f"{candidate_id}.json"
        candidate = CandidateResult(
            candidate_id=candidate_id,
            scaffold_name=candidate_name,
            passrate=passrate,
            average_score=passrate,
            token_consuming=total_tokens,
            avg_token_consuming=total_tokens / count if count else 0.0,
            avg_prompt_tokens=prompt_tokens / count if count else 0.0,
            avg_completion_tokens=completion_tokens / count if count else 0.0,
            count=count,
            config={
                "benchmark": "text_classification",
                "mode": self.config.mode,
                "memory_system": raw.get("memory_system", "fewshot_all"),
                "metric": metric,
                "extra": dict(raw.get("extra") or {}),
            },
            result_path=str(result_path),
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "candidate": candidate.to_dict(),
                    "proposal": raw,
                    "rows": rows,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return candidate

    def _build_workspace(self, *, iteration: int, call_dir: Path) -> Path:
        call_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir = call_dir / "workspace"
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)

        generated_dir = workspace_dir / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_package_dirs(generated_dir, root=generated_dir)

        summaries_dir = workspace_dir / "summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)
        for src, name, default_text in (
            (self.summary_path, "evolution_summary.jsonl", ""),
            (self.frontier_path, "best_candidates.json", "[]\n"),
            (self.candidate_score_table_path, "candidate_score_table.json", "[]\n"),
        ):
            dest = summaries_dir / name
            if src.exists():
                shutil.copy2(src, dest)
            else:
                dest.write_text(default_text, encoding="utf-8")

        self._build_source_snapshot(iteration, call_dir, workspace_dir / "source_snapshot")
        self._write_access_policy(workspace_dir)
        return workspace_dir

    def _build_source_snapshot(
        self,
        iteration: int,
        call_dir: Path,
        snapshot_root: Path,
    ) -> Path:
        if snapshot_root.exists():
            shutil.rmtree(snapshot_root)
        candidate_dir = snapshot_root / "candidate"
        project_source = candidate_dir / "project_source"
        source_pkg = project_source / "src" / "memomemo"
        source_pkg.mkdir(parents=True, exist_ok=True)
        self._ensure_package_dirs(source_pkg, root=project_source / "src")

        copied_files = copy_benchmark_project_source(
            project_root=self.project_root,
            dest_pkg=source_pkg,
            spec=self.workspace_spec,
        )
        original = candidate_dir / "original_project_source"
        shutil.copytree(
            project_source,
            original,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        (candidate_dir / "SNAPSHOT.md").write_text(
            "\n".join(
                [
                    "# Text Classification Source Snapshot",
                    "",
                    f"Iteration: {iteration}",
                    f"Task: text_classification_{self.config.mode}",
                    f"Benchmark: {self.workspace_spec.benchmark}",
                    "",
                    "This snapshot contains the benchmark-scoped source files declared",
                    "by the workspace spec. It is the complete editable source surface",
                    "for this proposer iteration.",
                    "",
                    "Copied source files:",
                    *[f"- project_source/src/memomemo/{rel}" for rel in copied_files],
                    "",
                ]
            ),
            encoding="utf-8",
        )
        manifest = {
            "iteration": iteration,
            "task": f"text_classification_{self.config.mode}",
            "benchmark": self.workspace_spec.benchmark,
            "candidate_dir": str(candidate_dir),
            "project_source": str(project_source),
            "original_project_source": str(original),
            "primary_source_file": self.workspace_spec.primary_source_file,
            "source_files": list(copied_files),
            "allowed_memomemo_modules": list(
                self.workspace_spec.allowed_memomemo_modules
            ),
        }
        snapshot_root.mkdir(parents=True, exist_ok=True)
        (snapshot_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (call_dir / "source_snapshot_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return snapshot_root

    def _write_access_policy(self, workspace_dir: Path) -> None:
        policy = {
            "read_roots": [str(workspace_dir)],
            "write_roots": [
                str(workspace_dir / "source_snapshot" / "candidate"),
                str(workspace_dir / "generated"),
            ],
            "write_files": [str(workspace_dir / "pending_eval.json")],
            "forbidden_roots": [
                str(self.project_root),
                str(self.run_dir),
            ],
            "notes": [
                "This proposer workspace is benchmark-scoped and self-contained.",
                "Read and write only the files copied into this workspace.",
                "Do not read repo-root source, global run artifacts, or previous candidate results.",
            ],
        }
        (workspace_dir / "access_policy.json").write_text(
            json.dumps(policy, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _archive_workspace_outputs(self, *, workspace_dir: Path, call_dir: Path) -> None:
        workspace_generated = workspace_dir / "generated"
        if workspace_generated.exists():
            self.generated_dir.mkdir(parents=True, exist_ok=True)
            self._ensure_package_dirs(self.generated_dir)
            call_generated = call_dir / "generated"
            if call_generated.exists():
                shutil.rmtree(call_generated)
            call_generated.mkdir(parents=True, exist_ok=True)
            for src in sorted(workspace_generated.rglob("*")):
                if src.is_dir() or "__pycache__" in src.parts or src.suffix == ".pyc":
                    continue
                rel = src.relative_to(workspace_generated)
                dest = self.generated_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                call_dest = call_generated / rel
                call_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, call_dest)

        for src, dest in (
            (workspace_dir / "pending_eval.json", self.pending_eval_path),
            (workspace_dir / "pending_eval.json", call_dir / "pending_eval.raw.json"),
        ):
            if src.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)

        source_snapshot = workspace_dir / "source_snapshot"
        archived = call_dir / "source_snapshot"
        if source_snapshot.exists():
            if archived.exists():
                shutil.rmtree(archived)
            shutil.copytree(
                source_snapshot,
                archived,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        self._write_source_snapshot_diff(call_dir)

    def _normalize_candidate_paths(
        self,
        candidate: dict[str, Any],
        *,
        workspace_dir: Path,
        call_dir: Path,
    ) -> None:
        candidate["candidate_root"] = str(self.generated_dir)
        if candidate.get("generated_dir"):
            candidate["generated_dir"] = str(self.generated_dir)
        self._normalize_path_fields(candidate, workspace_dir=workspace_dir, call_dir=call_dir)
        extra = candidate.get("extra")
        if isinstance(extra, dict):
            self._normalize_path_fields(extra, workspace_dir=workspace_dir, call_dir=call_dir)

    def _normalize_path_fields(
        self,
        item: dict[str, Any],
        *,
        workspace_dir: Path,
        call_dir: Path,
    ) -> None:
        for key in (
            "source_project_path",
            "project_source_path",
            "memomemo_source_path",
            "source_snapshot_path",
            "module_path",
        ):
            value = item.get(key)
            if not value:
                continue
            resolved = self._resolve_workspace_path(str(value), workspace_dir=workspace_dir)
            try:
                rel = resolved.relative_to(workspace_dir / "source_snapshot")
                item[key] = str((call_dir / "source_snapshot" / rel).resolve(False))
            except ValueError:
                try:
                    rel = resolved.relative_to(workspace_dir / "generated")
                    item[key] = str((self.generated_dir / rel).resolve(False))
                except ValueError:
                    item[key] = str(resolved)

    def _resolve_workspace_path(self, value: str, *, workspace_dir: Path) -> Path:
        path = Path(value).expanduser()
        docker_workspace = Path(self.config.proposer_docker_workspace or "/workspace")
        if path.is_absolute():
            try:
                rel = path.relative_to(docker_workspace)
                return (workspace_dir / rel).resolve(False)
            except ValueError:
                return path.resolve(False)
        return (workspace_dir / path).resolve(False)

    def _candidate_policy_violations(self, candidate: dict[str, Any]) -> list[dict[str, str]]:
        paths = self._candidate_policy_scan_paths(candidate)
        violations: list[dict[str, str]] = []
        forbidden = {
            "candidate_results": "runtime code must not read previous candidate results",
            "score_prediction": "runtime code must not call OptiHarness scoring helpers",
            "memomemo.metrics": "runtime code must not import OptiHarness scoring helpers",
        }
        for path in paths:
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            lowered = text.lower()
            for marker, reason in forbidden.items():
                if marker in lowered and self._policy_marker_is_candidate_added(
                    path,
                    marker,
                    text,
                ):
                    violations.append(
                        {"path": str(path), "marker": marker, "reason": reason}
                    )
            violations.extend(self._candidate_import_scope_violations(path, text))
        return violations

    def _policy_marker_is_candidate_added(
        self,
        path: Path,
        marker: str,
        text: str,
    ) -> bool:
        """Return true when a forbidden marker is new candidate code, not baseline code."""

        lowered_marker = marker.lower()
        try:
            package_path = path.resolve(False)
            source_pkg = self.project_root / "src" / "memomemo"
            rel = package_path.relative_to(source_pkg.resolve(False))
        except ValueError:
            rel = None

        if rel is None:
            try:
                parts = path.parts
                index = len(parts) - 1 - parts[::-1].index("memomemo")
                rel = Path(*parts[index + 1 :])
            except ValueError:
                return True

        original = self.project_root / "src" / "memomemo" / rel
        try:
            original_text = original.read_text(encoding="utf-8")
        except OSError:
            return True

        candidate_count = text.lower().count(lowered_marker)
        original_count = original_text.lower().count(lowered_marker)
        return candidate_count > original_count

    def _candidate_import_scope_violations(
        self,
        path: Path,
        text: str,
    ) -> list[dict[str, str]]:
        allowed = set(self.workspace_spec.allowed_memomemo_modules)
        violations: list[dict[str, str]] = []
        for match in re.finditer(
            r"^\s*(?:from\s+memomemo(?:\.([A-Za-z_]\w*))?|import\s+memomemo\.([A-Za-z_]\w*))",
            text,
            flags=re.MULTILINE,
        ):
            module = match.group(1) or match.group(2)
            if not module:
                continue
            if module not in allowed:
                marker = f"memomemo.{module}"
                violations.append(
                    {
                        "path": str(path),
                        "marker": marker,
                        "reason": (
                            "runtime code may import only memomemo modules declared "
                            "by the benchmark workspace spec"
                        ),
                    }
                )
        return violations

    def _candidate_policy_scan_paths(self, candidate: dict[str, Any]) -> list[Path]:
        out: list[Path] = []
        candidate_root = candidate.get("candidate_root") or candidate.get("generated_dir")
        if candidate_root:
            root = Path(str(candidate_root)).expanduser()
            if not root.is_absolute():
                root = self.project_root / root
            if root.exists():
                out.extend(sorted(root.glob("*.py")))

        extra = candidate.get("extra") if isinstance(candidate.get("extra"), dict) else {}
        source_project_value = (
            candidate.get("source_project_path")
            or candidate.get("project_source_path")
            or extra.get("source_project_path")
            or extra.get("project_source_path")
        )
        if source_project_value:
            source_project = Path(str(source_project_value)).expanduser()
            if not source_project.is_absolute():
                source_project = self.project_root / source_project
            package_dir = source_project / "src" / "memomemo"
            if package_dir.exists():
                out.extend(sorted(package_dir.rglob("*.py")))
        return sorted(set(out))

    def _run_proposer_agent(
        self,
        prompt: str,
        *,
        log_dir: Path,
        name: str,
        cwd: Path,
    ) -> Any:
        agent = self.config.proposer_agent.strip().lower()
        model_by_agent = {
            "claude": self.config.claude_model,
            "codex": self.config.codex_model,
            "kimi": self.config.kimi_model,
        }
        return run_code_agent_prompt(
            prompt,
            agent=agent,
            cwd=cwd,
            log_dir=log_dir,
            name=name,
            model=model_by_agent.get(agent, self.config.claude_model),
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

    def _load_existing_candidates(self) -> list[CandidateResult]:
        candidates = []
        for path in sorted((self.run_dir / "candidate_results").glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                candidates.append(CandidateResult.from_dict(payload["candidate"]))
            except Exception:
                continue
        return candidates

    def _write_source_snapshot_diff(self, call_dir: Path) -> None:
        original = call_dir / "source_snapshot" / "candidate" / "original_project_source"
        updated = call_dir / "source_snapshot" / "candidate" / "project_source"
        if not original.exists() or not updated.exists():
            (call_dir / "diff.patch").write_text("Source snapshot diff unavailable.\n")
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

    def _refresh_run_indexes(self, candidates: list[CandidateResult]) -> None:
        best_ids = self._best_passrate_ids(candidates)
        score_rows = [
            {
                "candidate_id": item.candidate_id,
                "scaffold_name": item.scaffold_name,
                "passrate": item.passrate,
                "average_score": item.average_score,
                "token_consuming": item.token_consuming,
                "result_path": item.result_path,
                "is_best_passrate": item.candidate_id in best_ids,
            }
            for item in sorted(candidates, key=lambda item: item.candidate_id)
        ]
        self.candidate_score_table_path.write_text(
            json.dumps(score_rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        iterations = sorted(
            {
                int(item.candidate_id[4:7])
                for item in candidates
                if item.candidate_id.startswith("iter") and item.candidate_id[4:7].isdigit()
            }
        )
        self.iteration_index_path.write_text(
            json.dumps(
                [
                    {
                        "iteration": item,
                        "iteration_dir": str(
                            self.run_dir / "proposer_calls" / f"iter_{item:03d}"
                        ),
                    }
                    for item in iterations
                ],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _write_pareto(self, candidates: list[CandidateResult]) -> None:
        save_frontier(
            self.run_dir / "pareto_frontier.json",
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
                for item in candidates
            ],
            quality_gap_threshold=0.0,
        )

    def _best_passrate_ids(self, candidates: list[CandidateResult]) -> set[str]:
        if not candidates:
            return set()
        best = max(item.passrate for item in candidates)
        return {item.candidate_id for item in candidates if item.passrate == best}

    def _save_best_candidates(self, candidates: list[CandidateResult]) -> None:
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
        self._append_event(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "iteration": iteration,
                "task": f"text_classification_{self.config.mode}",
                "candidate": candidate.to_dict(),
                "proposal": proposal or {},
            }
        )

    def _append_event(self, row: dict[str, Any]) -> None:
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with self.summary_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _ensure_package_dirs(self, path: Path, *, root: Path | None = None) -> None:
        path.mkdir(parents=True, exist_ok=True)
        root = root or path
        current = path
        while current == root or root in current.parents:
            init = current / "__init__.py"
            if not init.exists():
                init.write_text("", encoding="utf-8")
            if current == root:
                break
            current = current.parent


def _dedupe_tuple(values: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)
