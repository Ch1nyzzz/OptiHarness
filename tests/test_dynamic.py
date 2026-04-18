from pathlib import Path

from memomemo.dynamic import load_candidate_scaffold
from memomemo.scaffolds.base import MemoryScaffold


def test_load_builtin_candidate_scaffold():
    scaffold = load_candidate_scaffold({"scaffold_name": "bm25"}, project_root=Path.cwd())
    assert isinstance(scaffold, MemoryScaffold)
    assert scaffold.name == "bm25"
