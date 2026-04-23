"""Dynamic loading for text-classification memory candidates."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

from memomemo.text_classification import (
    ClassificationMemorySystem,
    PromptLLM,
    build_text_classification_memory,
)


SOURCE_PROJECT_PATH_KEYS = (
    "source_project_path",
    "project_source_path",
    "memomemo_source_path",
)


def load_candidate_text_memory(
    candidate: dict[str, Any],
    *,
    project_root: Path,
    llm: PromptLLM,
) -> ClassificationMemorySystem:
    """Instantiate a text-classification memory candidate."""

    src_path = str(project_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    candidate_root = _candidate_root(candidate, project_root=project_root)
    if candidate_root is not None:
        root_path = str(candidate_root)
        if root_path not in sys.path:
            sys.path.insert(0, root_path)

    source_project_path = _source_project_path(candidate, project_root=project_root)
    context = (
        _isolated_memomemo_project(source_project_path)
        if source_project_path is not None
        else nullcontext()
    )

    memory_name = str(
        candidate.get("memory_system")
        or candidate.get("baseline")
        or candidate.get("seed_name")
        or "fewshot_all"
    )
    module_path = str(candidate.get("module_path") or "").strip()
    module_name = str(candidate.get("module") or "").strip()
    class_name = str(candidate.get("class") or "").strip()
    factory_name = str(candidate.get("factory") or "").strip()

    with context:
        if module_path or module_name:
            module = (
                _load_module_path(module_path, project_root=project_root)
                if module_path
                else _import_candidate_module(
                    module_name,
                    candidate_root=candidate_root,
                )
            )
            if class_name:
                memory = getattr(module, class_name)(llm=llm)
            elif factory_name:
                memory = getattr(module, factory_name)(llm=llm)
            elif hasattr(module, "build_memory"):
                memory = module.build_memory(llm=llm)
            elif hasattr(module, "MEMORY_CLASS"):
                memory = module.MEMORY_CLASS(llm=llm)
            else:
                raise ValueError(
                    f"{module_name or module_path} must expose class/factory/build_memory/MEMORY_CLASS"
                )
        else:
            if source_project_path is not None:
                text_cls = importlib.import_module("memomemo.text_classification")
                memory = text_cls.build_text_classification_memory(memory_name, llm)
            else:
                memory = build_text_classification_memory(memory_name, llm)

    if not _is_text_memory(memory):
        raise TypeError("candidate did not produce a text-classification memory object")
    return memory


def _is_text_memory(memory: Any) -> bool:
    required = ("predict", "learn_from_batch", "get_state", "set_state")
    return isinstance(memory, ClassificationMemorySystem) or all(
        hasattr(memory, attr) for attr in required
    )


def _candidate_root(candidate: dict[str, Any], *, project_root: Path) -> Path | None:
    value = candidate.get("candidate_root") or candidate.get("generated_dir")
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path


def _source_project_path(candidate: dict[str, Any], *, project_root: Path) -> Path | None:
    extra = candidate.get("extra") if isinstance(candidate.get("extra"), dict) else {}
    for key in SOURCE_PROJECT_PATH_KEYS:
        value = candidate.get(key) or extra.get(key)
        if value:
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = project_root / path
            return _source_project_src_root(path)
    return None


def _source_project_src_root(path: Path) -> Path:
    candidates = [
        path,
        path / "src",
        path / "project_source",
        path / "project_source" / "src",
    ]
    for item in candidates:
        if (item / "memomemo").is_dir():
            return item
    raise FileNotFoundError(
        f"source project path must contain memomemo package: {path}"
    )


def _import_candidate_module(module_name: str, *, candidate_root: Path | None) -> object:
    importlib.invalidate_caches()
    if candidate_root is not None and module_name.startswith("memomemo.generated."):
        module_name = module_name.removeprefix("memomemo.generated.")
    if candidate_root is not None and module_name in sys.modules:
        del sys.modules[module_name]
    module = importlib.import_module(module_name)
    return importlib.reload(module)


@contextmanager
def _isolated_memomemo_project(src_root: Path) -> Iterator[None]:
    """Temporarily import memomemo modules from a copied source tree."""

    saved_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "memomemo" or name.startswith("memomemo.")
    }
    for name in saved_modules:
        sys.modules.pop(name, None)

    source_text = str(src_root)
    inserted = False
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
        inserted = True
    try:
        yield
    finally:
        for name in [
            item
            for item in list(sys.modules)
            if item == "memomemo" or item.startswith("memomemo.")
        ]:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        if inserted:
            try:
                sys.path.remove(source_text)
            except ValueError:
                pass


def _load_module_path(module_path: str, *, project_root: Path) -> object:
    path = Path(module_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        raise FileNotFoundError(f"candidate module_path does not exist: {path}")

    module_name = f"_memomemo_text_candidate_{abs(hash(path.resolve()))}"
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load candidate module_path: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
