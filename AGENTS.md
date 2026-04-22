# Repository Guidelines

## Project Structure & Module Organization

MemoMemo is a Python 3.11 `src`-layout package. Core package code lives in
`src/memomemo/`; memory scaffold implementations are under
`src/memomemo/scaffolds/`, and small helpers live in `src/memomemo/utils/`.
Tests are in `tests/` and mirror major modules such as `test_optimizer.py`,
`test_pareto.py`, and `test_scaffolds.py`. CLI and experiment helpers are in
`scripts/`. Configuration examples are in `configs/`. Runtime outputs belong in
`runs/` and `logs/`; do not treat generated run artifacts as source changes.
External reference checkouts are expected under `references/vendor/`.

## Build, Test, and Development Commands

Install the editable package with development dependencies:

```bash
python -m pip install -e '.[dev]'
```

Install source-backed scaffold dependencies when working on mem0, MemGPT, or
MemoryBank integrations:

```bash
python -m pip install -e '.[dev,source]'
scripts/fetch_reference_repos.sh
```

Run the test suite:

```bash
pytest -q
```

Run a quick dry-run optimization smoke test:

```bash
memomemo optimize --run-id smoke_opt --iterations 1 --limit 3 --dry-run \
  --scaffold-extra-json @configs/source_memory.example.json
```

## Coding Style & Naming Conventions

Use 4-space indentation, type hints, and small focused functions. Follow the
existing module style: dataclasses for configuration records, snake_case for
functions and variables, PascalCase for classes, and descriptive test names.
Prefer `pathlib.Path` for filesystem work and structured JSON/YAML parsing over
ad hoc string parsing. Keep generated candidate code under run-local
`generated/` directories, not `src/`.

## Testing Guidelines

Pytest is the only configured test framework. Add or update tests beside the
behavior you change, using filenames `tests/test_<feature>.py` and test
functions named `test_<expected_behavior>`. For optimizer, prompt, dynamic
loading, and scaffold changes, include regression tests that exercise file
paths and serialized JSON payloads. Run `pytest -q` before handing off changes.

## Commit & Pull Request Guidelines

The current history uses concise imperative commit subjects, for example
`Add source-backed memory scaffolds`. Keep commits focused and avoid mixing
runtime artifacts with source edits. Pull requests should include a short
summary, the commands run, relevant run IDs or output paths, and linked issues
when applicable. Include screenshots only for UI-facing changes; most changes
should instead include CLI output or JSON artifact paths.

## Security & Configuration Tips

Do not commit secrets, model API keys, local cache paths, or large generated
outputs. Source-backed scaffolds may depend on local model endpoints and vendor
repositories; document non-default paths in the PR description rather than
hardcoding them.
