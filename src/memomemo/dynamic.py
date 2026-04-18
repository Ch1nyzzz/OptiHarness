"""Dynamic candidate loading for Claude-proposed memory scaffolds."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from memomemo.scaffolds import build_scaffold
from memomemo.scaffolds.base import MemoryScaffold


def load_candidate_scaffold(candidate: dict[str, Any], *, project_root: Path) -> MemoryScaffold:
    """Instantiate a memory scaffold from pending_eval candidate metadata."""

    src_path = str(project_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    scaffold_name = candidate.get("scaffold_name") or candidate.get("seed_name")
    if scaffold_name:
        return build_scaffold(str(scaffold_name))

    module_name = str(candidate.get("module") or "").strip()
    class_name = str(candidate.get("class") or "").strip()
    factory_name = str(candidate.get("factory") or "").strip()
    if not module_name:
        raise ValueError("candidate must provide `module` or `scaffold_name`")

    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    module = importlib.reload(module)

    if class_name:
        cls = getattr(module, class_name)
        scaffold = cls()
    elif factory_name:
        scaffold = getattr(module, factory_name)()
    elif hasattr(module, "build_scaffold"):
        scaffold = module.build_scaffold()
    elif hasattr(module, "SCAFFOLD_CLASS"):
        scaffold = module.SCAFFOLD_CLASS()
    else:
        raise ValueError(
            f"{module_name} must expose class/factory/build_scaffold/SCAFFOLD_CLASS"
        )

    if not isinstance(scaffold, MemoryScaffold):
        required = ("build", "answer", "name")
        if not all(hasattr(scaffold, attr) for attr in required):
            raise TypeError(f"{module_name} did not produce a MemoryScaffold-compatible object")
    return scaffold
