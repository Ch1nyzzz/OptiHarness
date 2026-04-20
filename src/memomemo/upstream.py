"""Helpers for loading vendored upstream memory systems."""

from __future__ import annotations

import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = PROJECT_ROOT / "references" / "vendor"


def vendor_path(name: str) -> Path:
    """Return the checked-out reference repository path."""

    return VENDOR_ROOT / name


def _resolve_source_path(path: str | Path | None, *, default_name: str) -> Path:
    root = Path(path).expanduser() if path else vendor_path(default_name)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root


@contextmanager
def prepend_sys_path(path: Path) -> Iterator[None]:
    """Temporarily prepend a path for upstream imports."""

    text = str(path)
    inserted = False
    if text not in sys.path:
        sys.path.insert(0, text)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(text)
            except ValueError:
                pass


def load_mem0_memory_class(*, source_path: str | Path | None = None):
    """Load mem0 Memory, preferring an installed package then vendored source.

    The vendored mem0 package imports its distribution metadata in ``mem0/__init__.py``.
    A source checkout is not installed as a package, so for source loading we create a
    minimal package shim and import submodules directly from the checkout.
    """

    if not source_path:
        try:
            from mem0 import Memory

            return Memory
        except Exception:
            pass

    repo = _resolve_source_path(source_path, default_name="mem0")
    package = repo / "mem0"
    if not package.exists():
        raise FileNotFoundError(
            f"mem0 source checkout not found at {repo}. Run scripts/fetch_reference_repos.sh."
        )

    if source_path:
        for name in list(sys.modules):
            if name == "mem0" or name.startswith("mem0."):
                sys.modules.pop(name, None)
    else:
        existing = sys.modules.get("mem0")
        existing_path = getattr(existing, "__path__", None)
        if existing is not None and str(package) in [str(item) for item in existing_path or []]:
            with prepend_sys_path(repo):
                importlib.invalidate_caches()
                module = importlib.import_module("mem0.memory.main")
            return module.Memory

    shim = types.ModuleType("mem0")
    shim.__path__ = [str(package)]  # type: ignore[attr-defined]
    shim.__version__ = "source"
    sys.modules["mem0"] = shim

    with prepend_sys_path(repo):
        importlib.invalidate_caches()
        module = importlib.import_module("mem0.memory.main")
    return module.Memory
