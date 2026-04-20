"""Small Claude Code CLI wrapper for proposer calls."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_ALLOWED_TOOLS = ("Read", "Write", "Edit", "Bash", "Grep", "Glob")


@dataclass(frozen=True)
class ClaudeResult:
    """One proposer invocation result."""

    returncode: int | None
    timed_out: bool
    stdout: str
    stderr: str
    raw_stdout: str
    command: tuple[str, ...]
    usage: dict[str, Any] | None
    tool_access: dict[str, Any]
    duration_s: float
    metrics: dict[str, Any] = field(default_factory=dict)


def has_claude_cli() -> bool:
    return shutil.which("claude") is not None


def run_claude_prompt(
    prompt: str,
    *,
    cwd: Path,
    log_dir: Path,
    name: str,
    model: str = "claude-sonnet-4-6",
    timeout_s: int = 2400,
    allowed_tools: tuple[str, ...] = DEFAULT_ALLOWED_TOOLS,
    strip_anthropic_api_key: bool = True,
) -> ClaudeResult:
    """Run `claude -p` non-interactively and persist logs."""

    command = (
        "claude",
        "-p",
        "--model",
        model,
        "--allowedTools",
        ",".join(allowed_tools),
        "--output-format",
        "stream-json",
        "--verbose",
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if not has_claude_cli():
        result = ClaudeResult(
            returncode=None,
            timed_out=False,
            stdout="",
            stderr="claude CLI not found on PATH",
            raw_stdout="",
            command=command,
            usage=None,
            tool_access={},
            duration_s=0.0,
            metrics={},
        )
        _write_logs(result, log_dir=log_dir, name=name, prompt=prompt)
        return result

    env = os.environ.copy()
    saved_key = env.pop("ANTHROPIC_API_KEY", None) if strip_anthropic_api_key else None
    try:
        completed = subprocess.run(
            command,
            input=prompt,
            cwd=str(cwd),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        raw_stdout = completed.stdout or ""
        stdout, usage = _extract_stream_result(raw_stdout)
        tool_access = _extract_tool_access(raw_stdout, cwd=cwd)
        duration_s = time.time() - started
        metrics = _extract_session_metrics(
            usage=usage,
            tool_access=tool_access,
            duration_s=duration_s,
        )
        result = ClaudeResult(
            returncode=completed.returncode,
            timed_out=False,
            stdout=stdout,
            stderr=completed.stderr or "",
            raw_stdout=raw_stdout,
            command=command,
            usage=usage,
            tool_access=tool_access,
            duration_s=duration_s,
            metrics=metrics,
        )
    except subprocess.TimeoutExpired as exc:
        raw_stdout = _coerce(exc.stdout)
        tool_access = _extract_tool_access(raw_stdout, cwd=cwd)
        duration_s = time.time() - started
        result = ClaudeResult(
            returncode=None,
            timed_out=True,
            stdout=raw_stdout,
            stderr=_coerce(exc.stderr),
            raw_stdout=raw_stdout,
            command=command,
            usage=None,
            tool_access=tool_access,
            duration_s=duration_s,
            metrics=_extract_session_metrics(
                usage=None,
                tool_access=tool_access,
                duration_s=duration_s,
            ),
        )
    finally:
        if saved_key:
            os.environ["ANTHROPIC_API_KEY"] = saved_key

    _write_logs(result, log_dir=log_dir, name=name, prompt=prompt)
    return result


def _extract_stream_result(raw_stdout: str) -> tuple[str, dict[str, Any] | None]:
    events: list[dict[str, Any]] = []
    for line in raw_stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    assistant_usage = _aggregate_assistant_usage(events)
    result_event = next(
        (event for event in reversed(events) if event.get("type") == "result"),
        None,
    )
    if not isinstance(result_event, dict):
        text_chunks: list[str] = []
        for event in events:
            if event.get("type") != "assistant":
                continue
            message = event.get("message") or {}
            for item in message.get("content") or []:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_chunks.append(str(item.get("text") or ""))
        fallback_usage = {"usage": assistant_usage} if assistant_usage else None
        return "".join(text_chunks) or raw_stdout, fallback_usage

    usage: dict[str, Any] = {}
    for key in (
        "usage",
        "total_cost_usd",
        "duration_ms",
        "duration_api_ms",
        "num_turns",
        "session_id",
    ):
        if key in result_event:
            usage[key] = result_event[key]
    if "usage" not in usage and assistant_usage:
        usage["usage"] = assistant_usage
    return str(result_event.get("result") or ""), usage or None


def _aggregate_assistant_usage(events: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for event in events:
        if event.get("type") != "assistant":
            continue
        message = event.get("message") or {}
        if not isinstance(message, dict):
            continue
        usage = message.get("usage") or {}
        if not isinstance(usage, dict):
            continue
        for key in (
            "input_tokens",
            "output_tokens",
            "prompt_tokens",
            "completion_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            if key in usage:
                totals[key] = totals.get(key, 0) + _int_metric(usage.get(key))
    return totals


def _extract_tool_access(raw_stdout: str, *, cwd: Path | str | None = None) -> dict[str, Any]:
    tool_uses: list[dict[str, Any]] = []
    tool_by_id: dict[str, dict[str, Any]] = {}
    read_files: list[str] = []
    grep_requests: list[dict[str, Any]] = []

    for line in raw_stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        message = event.get("message") or {}
        if not isinstance(message, dict):
            continue

        if event.get("type") == "assistant":
            for item in message.get("content") or []:
                if not isinstance(item, dict) or item.get("type") != "tool_use":
                    continue
                name = str(item.get("name") or "")
                tool_input = item.get("input") or {}
                if not isinstance(tool_input, dict):
                    tool_input = {}

                record = {
                    "id": item.get("id"),
                    "name": name,
                    "input": tool_input,
                }
                tool_uses.append(record)
                if record["id"]:
                    tool_by_id[str(record["id"])] = record

                if name == "Read":
                    file_path = tool_input.get("file_path")
                    if isinstance(file_path, str) and file_path:
                        read_files.append(_make_relative(file_path, cwd))
                elif name == "Grep":
                    grep_requests.append(
                        {
                            "pattern": tool_input.get("pattern"),
                            "path": tool_input.get("path"),
                            "glob": tool_input.get("glob"),
                        }
                    )
        elif event.get("type") == "user":
            for item in message.get("content") or []:
                if not isinstance(item, dict) or item.get("type") != "tool_result":
                    continue
                tool_id = str(item.get("tool_use_id") or "")
                record = tool_by_id.get(tool_id)
                if record is None:
                    continue
                record["_output"] = _stringify_tool_result_content(item.get("content"))
                record["is_error"] = bool(item.get("is_error", False))

    files_read: dict[str, dict[str, int]] = {}
    files_written: dict[str, dict[str, int]] = {}
    for record in tool_uses:
        name = record.get("name")
        tool_input = record.get("input") if isinstance(record.get("input"), dict) else {}
        if name == "Read":
            file_path = tool_input.get("file_path")
            if not isinstance(file_path, str) or not file_path:
                continue
            path = _make_relative(file_path, cwd)
            lines = _count_read_lines(str(record.get("_output") or ""))
            current = files_read.setdefault(path, {"reads": 0, "lines": 0})
            current["reads"] += 1
            current["lines"] += lines
            if lines:
                record["output_lines"] = lines
        elif name == "Write":
            file_path = tool_input.get("file_path")
            if not isinstance(file_path, str) or not file_path:
                continue
            _add_written_lines(
                files_written,
                _make_relative(file_path, cwd),
                _count_text_lines(tool_input.get("content")),
            )
        elif name == "Edit":
            file_path = tool_input.get("file_path")
            if not isinstance(file_path, str) or not file_path:
                continue
            _add_written_lines(
                files_written,
                _make_relative(file_path, cwd),
                _count_text_lines(tool_input.get("new_string")),
            )
        elif name == "MultiEdit":
            file_path = tool_input.get("file_path")
            if not isinstance(file_path, str) or not file_path:
                continue
            edits = tool_input.get("edits") or []
            lines = 0
            if isinstance(edits, list):
                for edit in edits:
                    if isinstance(edit, dict):
                        lines += _count_text_lines(edit.get("new_string"))
            _add_written_lines(files_written, _make_relative(file_path, cwd), lines)

    for record in tool_uses:
        record.pop("_output", None)

    return {
        "read_files": sorted(set(read_files)),
        "grep_requests": _dedupe_dicts(grep_requests),
        "tool_uses": tool_uses,
        "tool_counts": dict(
            sorted(Counter(str(item.get("name") or "") for item in tool_uses).items())
        ),
        "files_read": dict(sorted(files_read.items())),
        "files_written": dict(sorted(files_written.items())),
    }


def _extract_session_metrics(
    *,
    usage: dict[str, Any] | None,
    tool_access: dict[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    usage = usage or {}
    token_usage = usage.get("usage") if isinstance(usage.get("usage"), dict) else {}
    input_tokens = _int_metric(
        token_usage.get("input_tokens", token_usage.get("prompt_tokens", 0))
    )
    output_tokens = _int_metric(
        token_usage.get("output_tokens", token_usage.get("completion_tokens", 0))
    )
    cache_creation_tokens = _int_metric(token_usage.get("cache_creation_input_tokens", 0))
    cache_read_tokens = _int_metric(token_usage.get("cache_read_input_tokens", 0))
    files_read = tool_access.get("files_read") if isinstance(tool_access, dict) else {}
    files_written = (
        tool_access.get("files_written") if isinstance(tool_access, dict) else {}
    )
    if not isinstance(files_read, dict):
        files_read = {}
    if not isinstance(files_written, dict):
        files_written = {}

    read_count = sum(_int_metric(item.get("reads", 0)) for item in files_read.values())
    read_lines = sum(_int_metric(item.get("lines", 0)) for item in files_read.values())
    write_count = sum(
        _int_metric(item.get("writes", 0)) for item in files_written.values()
    )
    written_lines = sum(
        _int_metric(item.get("lines_written", 0)) for item in files_written.values()
    )
    tool_uses = tool_access.get("tool_uses") if isinstance(tool_access, dict) else []
    if not isinstance(tool_uses, list):
        tool_uses = []

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cache_creation_input_tokens": cache_creation_tokens,
        "cache_read_input_tokens": cache_read_tokens,
        "total_reported_tokens": (
            input_tokens + output_tokens + cache_creation_tokens + cache_read_tokens
        ),
        "estimated_cost_usd": _float_metric(usage.get("total_cost_usd", 0.0)),
        "duration_s": round(float(duration_s), 3),
        "tool_calls": len(tool_uses),
        "tool_counts": (
            tool_access.get("tool_counts", {}) if isinstance(tool_access, dict) else {}
        ),
        "read_file_calls": read_count,
        "unique_files_read": len(files_read),
        "read_lines": read_lines,
        "write_file_calls": write_count,
        "written_lines": written_lines,
    }


def _dedupe_dicts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _make_relative(filepath: str, cwd: Path | str | None) -> str:
    if not cwd:
        return filepath
    try:
        if not os.path.isabs(filepath):
            return filepath
        rel = os.path.relpath(filepath, str(cwd))
    except ValueError:
        return filepath
    if rel == ".." or rel.startswith(f"..{os.sep}"):
        return filepath
    return rel


def _count_read_lines(output: str) -> int:
    return sum(1 for line in output.splitlines() if _is_numbered_read_line(line))


def _is_numbered_read_line(line: str) -> bool:
    stripped = line.lstrip()
    idx = 0
    while idx < len(stripped) and stripped[idx].isdigit():
        idx += 1
    return idx > 0 and stripped[idx:].startswith("\u2192")


def _count_text_lines(value: object) -> int:
    if not isinstance(value, str) or not value:
        return 0
    return value.count("\n") + 1


def _add_written_lines(
    files_written: dict[str, dict[str, int]],
    path: str,
    lines: int,
) -> None:
    current = files_written.setdefault(path, {"writes": 0, "lines_written": 0})
    current["writes"] += 1
    current["lines_written"] += lines


def _stringify_tool_result_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text is not None:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _int_metric(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float_metric(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _write_logs(result: ClaudeResult, *, log_dir: Path, name: str, prompt: str) -> None:
    prefix = log_dir / name
    prefix.mkdir(parents=True, exist_ok=True)
    (prefix / "prompt.md").write_text(prompt, encoding="utf-8")
    (prefix / "stdout.md").write_text(result.stdout or "", encoding="utf-8")
    (prefix / "stderr.txt").write_text(result.stderr or "", encoding="utf-8")
    (prefix / "stream.jsonl").write_text(result.raw_stdout or "", encoding="utf-8")
    meta = {
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "command": list(result.command),
        "usage": result.usage,
        "tool_access": result.tool_access,
        "metrics": result.metrics,
        "duration_s": result.duration_s,
    }
    (prefix / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (prefix / "tool_access.json").write_text(
        json.dumps(result.tool_access, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (prefix / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _coerce(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
