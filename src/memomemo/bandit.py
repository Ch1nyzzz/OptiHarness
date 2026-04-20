"""UCB policy for choosing lineage and context-budget arms."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SOURCE_FAMILIES = ("mem0", "memgpt", "membank", "fusion")
COST_LEVELS = ("low", "medium", "high")
DEFAULT_COST_PENALTY = {
    "low": 0.0,
    "medium": 0.005,
    "high": 0.01,
}


@dataclass(frozen=True)
class BanditArm:
    """One UCB arm: a lineage source plus a context budget."""

    source_family: str
    cost_level: str

    @property
    def key(self) -> str:
        return f"{self.source_family}|{self.cost_level}"

    @classmethod
    def from_key(cls, key: str) -> "BanditArm":
        source_family, cost_level = key.split("|", 1)
        return cls(source_family=source_family, cost_level=cost_level)


@dataclass
class ArmStats:
    """Running reward statistics for one arm."""

    pulls: float = 0.0
    mean_reward: float = 0.0
    last_reward: float = 0.0
    frontier_hits: int = 0
    last_iteration: int = 0


@dataclass
class BanditState:
    """Serializable UCB state."""

    total_pulls: float
    arms: dict[str, ArmStats]


def all_arms() -> list[BanditArm]:
    """Return the default arm grid."""

    return [
        BanditArm(source_family=source_family, cost_level=cost_level)
        for source_family in SOURCE_FAMILIES
        for cost_level in COST_LEVELS
    ]


def load_bandit_state(path: Path) -> BanditState:
    """Load UCB state, creating empty stats for missing arms."""

    if not path.exists():
        return BanditState(total_pulls=0.0, arms={})
    payload = json.loads(path.read_text(encoding="utf-8"))
    arms = {
        key: ArmStats(**value)
        for key, value in (payload.get("arms") or {}).items()
        if isinstance(value, dict)
    }
    return BanditState(total_pulls=float(payload.get("total_pulls", 0.0)), arms=arms)


def save_bandit_state(path: Path, state: BanditState) -> None:
    """Write UCB state JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "total_pulls": state.total_pulls,
        "arms": {key: asdict(stats) for key, stats in sorted(state.arms.items())},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def select_ucb_arm(
    state: BanditState,
    *,
    available_arms: Iterable[BanditArm] | None = None,
    exploration_c: float = 0.6,
) -> BanditArm:
    """Select the highest-UCB arm, prioritizing cold-start coverage."""

    candidates = list(available_arms or all_arms())
    if not candidates:
        raise ValueError("no available UCB arms")

    for arm in sorted(candidates, key=_arm_sort_key):
        stats = state.arms.get(arm.key)
        if stats is None or stats.pulls <= 0:
            return arm

    total = max(1.0, state.total_pulls)

    def score(arm: BanditArm) -> tuple[float, str]:
        stats = state.arms.get(arm.key) or ArmStats()
        bonus = exploration_c * math.sqrt(math.log(total + 1.0) / max(1e-9, stats.pulls))
        return (stats.mean_reward + bonus, arm.key)

    return max(candidates, key=score)


def update_bandit_state(
    state: BanditState,
    arm: BanditArm,
    *,
    reward: float,
    entered_frontier: bool,
    iteration: int,
    alpha: float = 0.25,
    gamma: float = 0.95,
) -> BanditState:
    """Apply a non-stationary moving-average reward update."""

    stats = state.arms.get(arm.key) or ArmStats()
    if stats.pulls <= 0:
        stats.mean_reward = float(reward)
        stats.pulls = 1.0
    else:
        stats.mean_reward = (1.0 - alpha) * stats.mean_reward + alpha * float(reward)
        stats.pulls = gamma * stats.pulls + 1.0
    stats.last_reward = float(reward)
    stats.frontier_hits += int(bool(entered_frontier))
    stats.last_iteration = int(iteration)
    state.arms[arm.key] = stats
    state.total_pulls = gamma * max(0.0, state.total_pulls) + 1.0
    return state


def cost_penalty(cost_level: str, penalties: dict[str, float] | None = None) -> float:
    """Return a fixed budget penalty for reward shaping."""

    table = penalties or DEFAULT_COST_PENALTY
    return float(table.get(cost_level, table["medium"]))


def _arm_sort_key(arm: BanditArm) -> tuple[int, int, str]:
    source_order = {name: idx for idx, name in enumerate(SOURCE_FAMILIES)}
    cost_order = {name: idx for idx, name in enumerate(COST_LEVELS)}
    return (
        source_order.get(arm.source_family, len(source_order)),
        cost_order.get(arm.cost_level, len(cost_order)),
        arm.key,
    )
