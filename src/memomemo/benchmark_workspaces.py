"""Benchmark-scoped source workspaces for proposer optimization."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


MINIMAL_BENCHMARK_PACKAGE_INIT = (
    '"""Benchmark-scoped candidate package."""\n\n'
    "__all__: list[str] = []\n"
)


@dataclass(frozen=True)
class BenchmarkWorkspaceSpec:
    """Source files that define one benchmark's editable proposer workspace."""

    benchmark: str
    source_files: tuple[str, ...]
    primary_source_file: str

    @property
    def allowed_memomemo_modules(self) -> tuple[str, ...]:
        """Top-level memomemo modules candidates may import from this snapshot."""

        modules: set[str] = set()
        for rel in self.source_files:
            parts = Path(rel).parts
            if not parts:
                continue
            top = parts[0]
            if top == "__init__.py":
                continue
            if top.endswith(".py"):
                modules.add(top.removesuffix(".py"))
            else:
                modules.add(top)
        return tuple(sorted(modules))


LOCOMO_WORKSPACE_SPEC = BenchmarkWorkspaceSpec(
    benchmark="locomo",
    primary_source_file="scaffolds/base.py",
    source_files=(
        "__init__.py",
        "dynamic.py",
        "metrics.py",
        "model.py",
        "schemas.py",
        "source_base.py",
        "upstream.py",
        "scaffolds/__init__.py",
        "scaffolds/base.py",
        "scaffolds/bm25_scaffold.py",
        "scaffolds/mem0_scaffold.py",
        "scaffolds/membank_scaffold.py",
        "scaffolds/memgpt_scaffold.py",
        "scaffolds/no_memory_scaffold.py",
        "utils/__init__.py",
        "utils/text.py",
    ),
)


LONGMEMEVAL_WORKSPACE_SPEC = BenchmarkWorkspaceSpec(
    benchmark="longmemeval",
    primary_source_file="scaffolds/base.py",
    source_files=(
        *LOCOMO_WORKSPACE_SPEC.source_files,
        "longmemeval.py",
    ),
)


TEXT_CLASSIFICATION_WORKSPACE_SPEC = BenchmarkWorkspaceSpec(
    benchmark="text_classification",
    primary_source_file="text_classification.py",
    source_files=(
        "model.py",
        "pareto.py",
        "schemas.py",
        "text_classification.py",
        "utils/__init__.py",
        "utils/text.py",
    ),
)


TAU3_BANKING_WORKSPACE_SPEC = BenchmarkWorkspaceSpec(
    benchmark="tau3_banking_knowledge",
    primary_source_file="tau_agents/banking_knowledge_base_agent.py",
    source_files=(
        "__init__.py",
        "pareto.py",
        "schemas.py",
        "tau_banking.py",
        "tau_agents/__init__.py",
        "tau_agents/banking_knowledge_base_agent.py",
        "tau_agent_runtime/__init__.py",
        "tau_agent_runtime/base_agent.py",
    ),
)


def copy_benchmark_project_source(
    *,
    project_root: Path,
    dest_pkg: Path,
    spec: BenchmarkWorkspaceSpec,
) -> tuple[str, ...]:
    """Copy exactly the source files declared by a benchmark workspace spec."""

    source_pkg = project_root / "src" / "memomemo"
    copied: list[str] = []
    for rel in spec.source_files:
        src = source_pkg / rel
        if not src.exists():
            raise FileNotFoundError(f"benchmark source file does not exist: {src}")
        dest = dest_pkg / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if Path(rel).parts == ("__init__.py",):
            dest.write_text(MINIMAL_BENCHMARK_PACKAGE_INIT, encoding="utf-8")
        else:
            shutil.copy2(src, dest)
        copied.append(rel)
    return tuple(copied)
