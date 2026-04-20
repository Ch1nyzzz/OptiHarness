from memomemo.upstream import load_mem0_memory_class


def test_load_mem0_memory_class_from_source_snapshot(tmp_path):
    package = tmp_path / "mem0" / "mem0" / "memory"
    package.mkdir(parents=True)
    for path in (tmp_path / "mem0" / "mem0", tmp_path / "mem0" / "mem0" / "memory"):
        (path / "__init__.py").write_text("", encoding="utf-8")
    (package / "main.py").write_text(
        "class Memory:\n"
        "    marker = 'snapshot'\n",
        encoding="utf-8",
    )

    memory_cls = load_mem0_memory_class(source_path=tmp_path / "mem0")

    assert memory_cls.marker == "snapshot"
