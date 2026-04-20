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


def dominates(a: ParetoPoint, b: ParetoPoint, *, quality_gap_threshold: float = 0.0) -> bool:
    """Return True if `a` dominates `b`.

    Higher passrate is better. Lower token_consuming is better.
    If `quality_gap_threshold` is positive, a point with a passrate advantage
    larger than that threshold dominates lower-quality points even when it
    spends more tokens.
    """

    if quality_gap_threshold > 0 and (a.passrate - b.passrate) > quality_gap_threshold:
        return True

    passrate_ge = a.passrate >= b.passrate
    tokens_le = a.token_consuming <= b.token_consuming
    strict = a.passrate > b.passrate or a.token_consuming < b.token_consuming
    return passrate_ge and tokens_le and strict


def pareto_frontier(
    points: Iterable[ParetoPoint],
    *,
    quality_gap_threshold: float = 0.0,
) -> list[ParetoPoint]:
    """Filter a collection of points down to the non-dominated frontier."""

    pool = list(points)
    frontier: list[ParetoPoint] = []
    for point in pool:
        if any(
            other is not point
            and dominates(other, point, quality_gap_threshold=quality_gap_threshold)
            for other in pool
        ):
            continue
        frontier.append(point)
    frontier.sort(key=lambda item: (-item.passrate, item.token_consuming, item.candidate_id))
    return frontier


def save_frontier(
    path: Path,
    points: Iterable[ParetoPoint],
    *,
    quality_gap_threshold: float = 0.0,
) -> None:
    """Write the frontier JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        asdict(point)
        for point in pareto_frontier(points, quality_gap_threshold=quality_gap_threshold)
    ]
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
