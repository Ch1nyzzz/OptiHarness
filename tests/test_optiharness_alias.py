from __future__ import annotations

import optiharness
from optiharness.cli import main as optiharness_main
from memomemo.cli import main as memomemo_main


def test_optiharness_public_alias_imports_cli() -> None:
    assert optiharness_main is memomemo_main
    assert hasattr(optiharness, "EvaluationRunner")
