"""Central benchmark task registry for optimization entry points."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkTaskSpec:
    """Stable metadata used to route a benchmark to its optimizer/base agent."""

    slug: str
    aliases: tuple[str, ...]
    benchmark: str
    base_agent_system: str
    optimizer_kind: str
    default_run_id: str
    description: str

    @property
    def cli_names(self) -> tuple[str, ...]:
        return (self.slug, *self.aliases)


LOCOMO_TASK = BenchmarkTaskSpec(
    slug="locomo",
    aliases=("locomo_subset", "memory_qa"),
    benchmark="locomo",
    base_agent_system="locomo_memory_scaffolds",
    optimizer_kind="locomo_memory",
    default_run_id="locomo_memory_opt",
    description="LOCOMO conversational-memory QA over memory scaffold base agents.",
)

LONGMEMEVAL_TASK = BenchmarkTaskSpec(
    slug="longmemeval",
    aliases=("long-mem-eval", "long_mem_eval", "lme"),
    benchmark="longmemeval",
    base_agent_system=LOCOMO_TASK.base_agent_system,
    optimizer_kind="longmemeval_memory",
    default_run_id="longmemeval_memory_opt",
    description="LongMemEval long-term memory QA over the LOCOMO/MemGPT memory scaffold base agent.",
)

TAU3_TASK = BenchmarkTaskSpec(
    slug="tau3",
    aliases=("tau", "tau_banking", "tau3_banking", "banking_knowledge"),
    benchmark="tau3_banking_knowledge",
    base_agent_system="tau3_banking_knowledge_base_agent",
    optimizer_kind="tau3_banking_agent",
    default_run_id="tau3_banking_opt",
    description="tau3/tau2-bench banking_knowledge tool-agent-user benchmark.",
)

TEXT_CLASSIFICATION_TASK = BenchmarkTaskSpec(
    slug="text_classification",
    aliases=("text-classification", "textcls", "classification"),
    benchmark="text_classification",
    base_agent_system="text_classification_fewshot_memory",
    optimizer_kind="text_classification_memory",
    default_run_id="text_classification_opt",
    description="Meta-Harness text-classification few-shot memory benchmark.",
)


BENCHMARK_TASKS: tuple[BenchmarkTaskSpec, ...] = (
    LOCOMO_TASK,
    LONGMEMEVAL_TASK,
    TAU3_TASK,
    TEXT_CLASSIFICATION_TASK,
)

_TASK_BY_NAME = {
    name: task
    for task in BENCHMARK_TASKS
    for name in task.cli_names
}

TASK_CHOICES = tuple(sorted(_TASK_BY_NAME))


def normalize_task_name(value: str | None) -> str:
    """Return the canonical task slug for a CLI task name or alias."""

    raw = (value or LOCOMO_TASK.slug).strip()
    if raw not in _TASK_BY_NAME:
        choices = ", ".join(TASK_CHOICES)
        raise ValueError(f"unknown benchmark task {raw!r}; expected one of: {choices}")
    return _TASK_BY_NAME[raw].slug


def task_spec(value: str | None) -> BenchmarkTaskSpec:
    """Return the task spec for a CLI task name or alias."""

    return _TASK_BY_NAME[normalize_task_name(value)]
