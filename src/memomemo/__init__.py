"""OptiHarness benchmark optimization harness.

The historical Python package name remains ``memomemo`` for backward
compatibility. New user-facing entry points should use ``optiharness``.
"""

from memomemo.evaluation import EvaluationRunner, run_initial_frontier
from memomemo.pareto import ParetoPoint, pareto_frontier

__all__ = [
    "EvaluationRunner",
    "ParetoPoint",
    "pareto_frontier",
    "run_initial_frontier",
]
