"""Proposer optimizer for tau3 banking_knowledge agents."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from memomemo.benchmark_workspaces import (
    TAU3_BANKING_WORKSPACE_SPEC,
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
from memomemo.pareto import ParetoPoint, save_frontier
from memomemo.schemas import CandidateResult
from memomemo.tau_banking import (
    DEFAULT_TAU2_ROOT,
    DEFAULT_TAU_BANKING_AGENT_NAME,
    DEFAULT_TAU_BANKING_RETRIEVAL_CONFIGS,
    TauBankingRunConfig,
    run_tau_banking_benchmark,
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
class TauBankingOptimizerConfig:
    """Configuration for tau3 banking_knowledge agent optimization."""

    run_id: str
    out_dir: Path
    iterations: int = 20
    tau2_root: Path = DEFAULT_TAU2_ROOT
    python_executable: str = "python"
    retrieval_configs: tuple[str, ...] = DEFAULT_TAU_BANKING_RETRIEVAL_CONFIGS
    retrieval_config_kwargs: dict[str, Any] = field(default_factory=dict)
    task_split_name: str = "base"
    task_ids: tuple[str, ...] = ()
    num_tasks: int | None = 5
    num_trials: int = 1
    max_steps: int = 200
    max_errors: int = 10
    max_concurrency: int = 1
    seed: int = 300
    agent_llm: str = "openai/gpt-4.1-mini"
    user_llm: str = "openai/gpt-4.1-mini"
    agent_llm_args: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.0})
    user_llm_args: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.0})
    process_timeout_s: int | None = None
    eval_timeout_s: int = 300
    proposer_agent: str = "claude"
    claude_model: str = "claude-sonnet-4-6"
    codex_model: str = DEFAULT_CODEX_MODEL
    kimi_model: str = DEFAULT_KIMI_MODEL
    propose_timeout_s: int = 2400
    skip_baseline_eval: bool = False
    force: bool = False
    proposer_sandbox: str = "docker"
    proposer_docker_image: str = ""
    proposer_docker_workspace: str = "/workspace"
    proposer_docker_env: tuple[str, ...] = ()
    proposer_docker_mount: tuple[str, ...] = ()
    proposer_docker_kimi_cli_kind: str = "claude"
    proposer_docker_user: str = ""
    proposer_docker_home: str = ""


class TauBankingOptimizer:
    """Meta-harness-style optimizer for the tau3 banking base agent."""

    workspace_spec: BenchmarkWorkspaceSpec = TAU3_BANKING_WORKSPACE_SPEC

    def __init__(self, config: TauBankingOptimizerConfig) -> None:
        self.config = config
        self.project_root = Path(__file__).resolve().parents[2]
        self.run_dir = config.out_dir
        self.pending_eval_path = self.run_dir / "pending_eval.json"
        self.frontier_path = self.run_dir / "best_candidates.json"
        self.summary_path = self.run_dir / "evolution_summary.jsonl"
        self.generated_dir = self.run_dir / "generated"
        self.candidate_score_table_path = self.run_dir / "candidate_score_table.json"
        self.iteration_index_path = self.run_dir / "iteration_index.json"

    def run(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_package_dirs(self.generated_dir)

        candidates: list[CandidateResult] = []
        if not self.config.skip_baseline_eval:
            baseline = run_tau_banking_benchmark(
                out_dir=self.run_dir / "baseline",
                config=self._tau_config(agent_name=DEFAULT_TAU_BANKING_AGENT_NAME),
                force=self.config.force,
            )
            candidates.extend(self._candidate_results_from_summary(baseline))
            for candidate in candidates:
                self._append_summary(iteration=0, candidate=candidate)
        else:
            candidates.extend(self._load_existing_candidates())

        self._save_best_candidates(candidates)
        self._refresh_run_indexes(candidates)
        self._write_pareto(candidates)

        for iteration in range(1, self.config.iterations + 1):
            evaluated = self._run_proposer_iteration(iteration, candidates)
            candidates.extend(evaluated)
            self._save_best_candidates(candidates)
            self._refresh_run_indexes(candidates)
            self._write_pareto(candidates)

        final_summary = {
            "run_id": self.config.run_id,
            "task": "tau3_banking_knowledge",
            "out_dir": str(self.run_dir),
            "iterations": self.config.iterations,
            "candidate_count": len(candidates),
            "best_candidates_path": str(self.frontier_path),
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
    ) -> list[CandidateResult]:
        if self.pending_eval_path.exists():
            self.pending_eval_path.unlink()
        call_dir = self.run_dir / "proposer_calls" / f"iter_{iteration:03d}"
        workspace_dir = self._build_workspace(iteration=iteration, call_dir=call_dir)
        prompt = self._build_proposer_prompt(
            iteration=iteration,
            workspace_dir=workspace_dir,
            existing_candidates=existing_candidates,
        )
        result = self._run_proposer_agent(
            prompt,
            log_dir=call_dir / "agent",
            name="tau3-banking-proposer",
            cwd=workspace_dir,
        )
        self._append_event(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "iteration": iteration,
                "event": "proposer_result",
                "task": "tau3_banking_knowledge",
                "returncode": getattr(result, "returncode", None),
                "timed_out": bool(getattr(result, "timed_out", False)),
                "proposer_metrics": getattr(result, "metrics", {}) or {},
                "usage": getattr(result, "usage", None),
            }
        )
        self._archive_workspace_outputs(workspace_dir=workspace_dir, call_dir=call_dir)
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
                self._normalize_candidate_paths(raw, workspace_dir=workspace_dir, call_dir=call_dir)
                raw.setdefault("source_family", "tau3_banking_agent")
                raw.setdefault("task", "tau3_banking_knowledge")
                raw.setdefault("factory_name", "create_banking_knowledge_base_agent")
        self.pending_eval_path.write_text(
            json.dumps({"candidates": proposed}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (call_dir / "pending_eval.json").write_text(
            json.dumps({"candidates": proposed}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return self._evaluate_proposed(iteration, proposed)

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

            candidate_name = _safe_name(str(raw.get("name") or "banking_agent"))
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
        source_project = self._candidate_source_project(raw)
        agent_module = self._candidate_agent_module(raw, source_project=source_project)
        agent_snapshot_root = agent_module.parent
        if agent_module.parent.name == "tau_agents":
            agent_snapshot_root = agent_module.parent.parent

        candidate_run_dir = self.run_dir / "candidate_runs" / candidate_id
        summary = run_tau_banking_benchmark(
            out_dir=candidate_run_dir,
            config=self._tau_config(
                agent_name=candidate_name,
                agent_module=agent_module,
                agent_snapshot_root=agent_snapshot_root,
                factory_name=str(raw.get("factory_name") or "create_banking_knowledge_base_agent"),
            ),
            force=True,
        )
        return self._aggregate_candidate_summary(
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            raw=raw,
            summary=summary,
        )

    def _aggregate_candidate_summary(
        self,
        *,
        candidate_id: str,
        candidate_name: str,
        raw: dict[str, Any],
        summary: dict[str, Any],
    ) -> CandidateResult:
        rows = list(summary.get("rows") or [])
        total_count = sum(int(row.get("task_count", 0) or 0) for row in rows)
        total_tokens = sum(int(row.get("token_consuming", 0) or 0) for row in rows)
        passrates = [float(row.get("passrate", 0.0) or 0.0) for row in rows]
        scores = [float(row.get("average_reward", 0.0) or 0.0) for row in rows]
        passrate = sum(passrates) / len(passrates) if passrates else 0.0
        average_score = sum(scores) / len(scores) if scores else 0.0
        result_path = self.run_dir / "candidate_results" / f"{candidate_id}.json"
        candidate = CandidateResult(
            candidate_id=candidate_id,
            scaffold_name=candidate_name,
            passrate=passrate,
            average_score=average_score,
            token_consuming=total_tokens,
            avg_token_consuming=total_tokens / total_count if total_count else 0.0,
            avg_prompt_tokens=0.0,
            avg_completion_tokens=0.0,
            count=total_count,
            config={
                "benchmark": "tau3_banking_knowledge",
                "agent": candidate_name,
                "retrieval_configs": list(self.config.retrieval_configs),
                "source_family": raw.get("source_family", "tau3_banking_agent"),
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
                    "tau_summary": summary,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return candidate

    def _candidate_results_from_summary(self, summary: dict[str, Any]) -> list[CandidateResult]:
        out: list[CandidateResult] = []
        for item in summary.get("candidates", []):
            try:
                candidate = CandidateResult.from_dict(item)
            except Exception:
                continue
            candidate_id = f"baseline_{candidate.candidate_id}"
            result_path = self.run_dir / "candidate_results" / f"{candidate_id}.json"
            copied = CandidateResult(
                candidate_id=candidate_id,
                scaffold_name=candidate.scaffold_name,
                passrate=candidate.passrate,
                average_score=candidate.average_score,
                token_consuming=candidate.token_consuming,
                avg_token_consuming=candidate.avg_token_consuming,
                avg_prompt_tokens=candidate.avg_prompt_tokens,
                avg_completion_tokens=candidate.avg_completion_tokens,
                count=candidate.count,
                config=candidate.config,
                result_path=str(result_path),
            )
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps({"candidate": copied.to_dict(), "tau_summary": summary}, indent=2),
                encoding="utf-8",
            )
            out.append(copied)
        return out

    def _tau_config(
        self,
        *,
        agent_name: str,
        agent_module: Path | None = None,
        agent_snapshot_root: Path | None = None,
        factory_name: str = "create_banking_knowledge_base_agent",
    ) -> TauBankingRunConfig:
        return TauBankingRunConfig(
            tau2_root=self.config.tau2_root,
            python_executable=self.config.python_executable,
            agent_name=agent_name,
            agent_module=agent_module,
            agent_snapshot_root=agent_snapshot_root,
            factory_name=factory_name,
            retrieval_configs=self.config.retrieval_configs,
            retrieval_config_kwargs=self.config.retrieval_config_kwargs,
            task_split_name=self.config.task_split_name,
            task_ids=self.config.task_ids,
            num_tasks=self.config.num_tasks,
            num_trials=self.config.num_trials,
            max_steps=self.config.max_steps,
            max_errors=self.config.max_errors,
            max_concurrency=self.config.max_concurrency,
            seed=self.config.seed,
            agent_llm=self.config.agent_llm,
            user_llm=self.config.user_llm,
            agent_llm_args=self.config.agent_llm_args,
            user_llm_args=self.config.user_llm_args,
            process_timeout_s=self.config.process_timeout_s or self.config.eval_timeout_s,
        )

    def _build_workspace(self, *, iteration: int, call_dir: Path) -> Path:
        call_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir = call_dir / "workspace"
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)

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

        generated_dir = workspace_dir / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_package_dirs(generated_dir, root=generated_dir)
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
                    "# tau3 Banking Source Snapshot",
                    "",
                    f"Iteration: {iteration}",
                    "Task: tau3_banking_knowledge",
                    "",
                    "This snapshot contains only the tau3 banking base agent,",
                    "the small local runtime it imports, and OptiHarness schemas needed",
                    "for candidate evaluation. Edit the copied agent/runtime only.",
                    "",
                    "Default candidate path:",
                    "- project_source/src/memomemo/tau_agents/banking_knowledge_base_agent.py",
                    "",
                    "Write exactly one candidate to `pending_eval.json`.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        manifest = {
            "iteration": iteration,
            "task": "tau3_banking_knowledge",
            "benchmark": self.workspace_spec.benchmark,
            "candidate_dir": str(candidate_dir),
            "project_source": str(project_source),
            "original_project_source": str(original),
            "primary_source_file": self.workspace_spec.primary_source_file,
            "source_files": list(copied_files),
            "default_agent_module": str(
                project_source
                / "src"
                / "memomemo"
                / "tau_agents"
                / "banking_knowledge_base_agent.py"
            ),
            "allowed_memomemo_modules": list(self.workspace_spec.allowed_memomemo_modules),
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

    def _build_proposer_prompt(
        self,
        *,
        iteration: int,
        workspace_dir: Path,
        existing_candidates: list[CandidateResult],
    ) -> str:
        best = sorted(
            existing_candidates,
            key=lambda item: (-item.passrate, item.token_consuming, item.candidate_id),
        )[:5]
        best_lines = [
            f"- {item.candidate_id}: passrate={item.passrate:.4f}, "
            f"average_reward={item.average_score:.4f}, cost={item.token_consuming}"
            for item in best
        ] or ["- No evaluated candidates yet."]
        return "\n".join(
            [
                "# tau3 Banking Agent Optimization",
                "",
                f"Run id: {self.config.run_id}",
                f"Iteration: {iteration}",
                f"Workspace: {workspace_dir}",
                "",
                "Optimize the tau3 banking_knowledge base agent. The benchmark is",
                "external tau2-bench, so the only editable surface is the scoped",
                "source snapshot under `source_snapshot/candidate/project_source`.",
                "",
                "Current best candidates:",
                *best_lines,
                "",
                "Editable files:",
                "- source_snapshot/candidate/project_source/src/memomemo/tau_agents/banking_knowledge_base_agent.py",
                "- source_snapshot/candidate/project_source/src/memomemo/tau_agent_runtime/base_agent.py",
                "",
                "Do not read repository root source, global run artifacts, tau2-bench",
                "vendor source, candidate_results, or benchmark gold/reward internals.",
                "Use the summaries copied into `summaries/` only.",
                "",
                "Write `pending_eval.json` with exactly one candidate:",
                "",
                "```json",
                "{",
                '  "candidates": [',
                "    {",
                '      "name": "short_candidate_name",',
                '      "source_project_path": "source_snapshot/candidate/project_source",',
                '      "agent_module": "source_snapshot/candidate/project_source/src/memomemo/tau_agents/banking_knowledge_base_agent.py",',
                '      "factory_name": "create_banking_knowledge_base_agent",',
                '      "notes": "what changed"',
                "    }",
                "  ]",
                "}",
                "```",
                "",
                "Focus on general tool-use policy, knowledge retrieval discipline,",
                "state tracking, and concise user-facing actions. Do not encode",
                "task-specific answers or reward hacks.",
            ]
        )

    def _write_access_policy(self, workspace_dir: Path) -> None:
        policy = {
            "read_roots": [str(workspace_dir)],
            "write_roots": [str(workspace_dir / "source_snapshot" / "candidate")],
            "write_files": [str(workspace_dir / "pending_eval.json")],
            "forbidden_roots": [
                str(self.project_root),
                str(self.run_dir),
                str(self.config.tau2_root),
            ],
            "notes": [
                "This proposer workspace is benchmark-scoped and self-contained.",
                "Do not read repo-root source, global run artifacts, or tau2-bench source.",
            ],
        }
        (workspace_dir / "access_policy.json").write_text(
            json.dumps(policy, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _archive_workspace_outputs(self, *, workspace_dir: Path, call_dir: Path) -> None:
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
            "agent_module",
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
        forbidden = {
            "candidate_results": "runtime code must not read previous candidate results",
            "run_summary.json": "runtime code must not read benchmark outputs",
            "tau2_banking_runner": "runtime code must not call the benchmark runner",
            "reward_info": "runtime code must not inspect reward internals",
        }
        violations: list[dict[str, str]] = []
        for path in paths:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            lowered = text.lower()
            for marker, reason in forbidden.items():
                if marker.lower() in lowered:
                    violations.append({"path": str(path), "marker": marker, "reason": reason})
            violations.extend(self._candidate_import_scope_violations(path, text))
        return violations

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
            if module and module not in allowed:
                violations.append(
                    {
                        "path": str(path),
                        "marker": f"memomemo.{module}",
                        "reason": (
                            "runtime code may import only memomemo modules declared "
                            "by the tau3 workspace spec"
                        ),
                    }
                )
        return violations

    def _candidate_policy_scan_paths(self, candidate: dict[str, Any]) -> list[Path]:
        out: list[Path] = []
        source_project = self._candidate_source_project(candidate)
        if source_project is not None:
            package_dir = source_project / "src" / "memomemo"
            if package_dir.exists():
                out.extend(sorted(package_dir.rglob("*.py")))
        try:
            agent_module = self._candidate_agent_module(candidate, source_project=source_project)
        except ValueError:
            return sorted(set(out))
        else:
            if agent_module.exists():
                out.append(agent_module)
        return sorted(set(out))

    def _candidate_source_project(self, candidate: dict[str, Any]) -> Path | None:
        extra = candidate.get("extra") if isinstance(candidate.get("extra"), dict) else {}
        for key in ("source_project_path", "project_source_path", "memomemo_source_path"):
            value = candidate.get(key) or extra.get(key)
            if not value:
                continue
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = self.project_root / path
            return path
        return None

    def _candidate_agent_module(
        self,
        candidate: dict[str, Any],
        *,
        source_project: Path | None,
    ) -> Path:
        value = candidate.get("agent_module") or candidate.get("module_path")
        if value:
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = self.project_root / path
            return path
        if source_project is not None:
            return (
                source_project
                / "src"
                / "memomemo"
                / "tau_agents"
                / "banking_knowledge_base_agent.py"
            )
        raise ValueError("tau3 candidate must specify source_project_path or agent_module")

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
                "task": "tau3_banking_knowledge",
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


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)
    return safe.strip("_") or "candidate"


def _dedupe_tuple(values: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)
