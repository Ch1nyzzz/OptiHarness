"""Pareto frontier over passrate and token consumption."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ParetoPoint:
    """One point in the memory-evolution Pareto plane."""

    candidate_id: str
    scaffold_name: str
    passrate: float
    token_consuming: int
    avg_token_consuming: float
    average_score: float
    result_path: str
    config: dict


def dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
    """Return True if `a` dominates `b`.

    Higher passrate is better. Lower token_consuming is better.
    """

    passrate_ge = a.passrate >= b.passrate
    tokens_le = a.token_consuming <= b.token_consuming
    strict = a.passrate > b.passrate or a.token_consuming < b.token_consuming
    return passrate_ge and tokens_le and strict


def pareto_frontier(points: Iterable[ParetoPoint]) -> list[ParetoPoint]:
    """Filter a collection of points down to the non-dominated frontier."""

    pool = list(points)
    frontier: list[ParetoPoint] = []
    for point in pool:
        if any(other is not point and dominates(other, point) for other in pool):
            continue
        frontier.append(point)
    frontier.sort(key=lambda item: (-item.passrate, item.token_consuming, item.candidate_id))
    return frontier


def save_frontier(path: Path, points: Iterable[ParetoPoint]) -> None:
    """Write the frontier JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(point) for point in pareto_frontier(points)]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_frontier(path: Path) -> list[ParetoPoint]:
    """Load a frontier JSON."""

    data = json.loads(path.read_text(encoding="utf-8"))
    return [ParetoPoint(**_normalize_point(item)) for item in data]


def _normalize_point(item: dict) -> dict:
    data = dict(item)
    if "scaffold_name" not in data and "seed_name" in data:
        data["scaffold_name"] = data.pop("seed_name")
    return data
