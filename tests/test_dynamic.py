from pathlib import Path

from memomemo.dynamic import load_candidate_scaffold
from memomemo.scaffolds.base import MemoryScaffold


def test_load_builtin_candidate_scaffold():
    scaffold = load_candidate_scaffold({"scaffold_name": "bm25"}, project_root=Path.cwd())
    assert isinstance(scaffold, MemoryScaffold)
    assert scaffold.name == "bm25"


def test_load_run_local_generated_candidate(tmp_path):
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    generated_dir.joinpath("run_candidate.py").write_text(
        "\n".join(
            [
                "from memomemo.scaffolds.bm25_scaffold import RankBM25Scaffold",
                "",
                "class RunCandidateScaffold(RankBM25Scaffold):",
                '    name = "run_candidate"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    scaffold = load_candidate_scaffold(
        {
            "module": "run_candidate",
            "class": "RunCandidateScaffold",
            "candidate_root": str(generated_dir),
        },
        project_root=Path.cwd(),
    )

    assert isinstance(scaffold, MemoryScaffold)
    assert scaffold.name == "run_candidate"


def test_load_builtin_scaffold_from_source_project_snapshot(tmp_path):
    source_root = tmp_path / "snapshot" / "candidate" / "project_source" / "src"
    scaffold_dir = source_root / "memomemo" / "scaffolds"
    scaffold_dir.mkdir(parents=True)
    (source_root / "memomemo" / "__init__.py").write_text("", encoding="utf-8")
    (scaffold_dir / "__init__.py").write_text("", encoding="utf-8")
    (scaffold_dir / "base.py").write_text(
        "\n".join(
            [
                "class MemoryScaffold:",
                "    pass",
            ]
        ),
        encoding="utf-8",
    )
    (scaffold_dir / "memgpt_scaffold.py").write_text(
        "\n".join(
            [
                "from memomemo.scaffolds.base import MemoryScaffold",
                "",
                "class MemGPTSourceScaffold(MemoryScaffold):",
                '    name = "snapshot_memgpt"',
                "    def build(self, example, config):",
                "        return None",
                "    def answer(self, state, example, client, config, *, max_context_chars, dry_run):",
                "        return None",
            ]
        ),
        encoding="utf-8",
    )

    scaffold = load_candidate_scaffold(
        {
            "scaffold_name": "memgpt_source",
            "extra": {
                "source_project_path": str(tmp_path / "snapshot" / "candidate" / "project_source"),
            },
        },
        project_root=Path.cwd(),
    )

    assert not isinstance(scaffold, MemoryScaffold)
    assert scaffold.name == "snapshot_memgpt"
