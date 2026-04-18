"""Small Claude Code CLI wrapper for proposer calls."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
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
        tool_access = _extract_tool_access(raw_stdout)
        result = ClaudeResult(
            returncode=completed.returncode,
            timed_out=False,
            stdout=stdout,
            stderr=completed.stderr or "",
            raw_stdout=raw_stdout,
            command=command,
            usage=usage,
            tool_access=tool_access,
            duration_s=time.time() - started,
        )
    except subprocess.TimeoutExpired as exc:
        result = ClaudeResult(
            returncode=None,
            timed_out=True,
            stdout=_coerce(exc.stdout),
            stderr=_coerce(exc.stderr),
            raw_stdout=_coerce(exc.stdout),
            command=command,
            usage=None,
            tool_access=_extract_tool_access(_coerce(exc.stdout)),
            duration_s=time.time() - started,
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
        return "".join(text_chunks) or raw_stdout, None

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
    return str(result_event.get("result") or ""), usage or None


def _extract_tool_access(raw_stdout: str) -> dict[str, Any]:
    tool_uses: list[dict[str, Any]] = []
    read_files: list[str] = []
    grep_requests: list[dict[str, Any]] = []

    for line in raw_stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "assistant":
            continue

        message = event.get("message") or {}
        if not isinstance(message, dict):
            continue
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

            if name == "Read":
                file_path = tool_input.get("file_path")
                if isinstance(file_path, str) and file_path:
                    read_files.append(file_path)
            elif name == "Grep":
                grep_requests.append(
                    {
                        "pattern": tool_input.get("pattern"),
                        "path": tool_input.get("path"),
                        "glob": tool_input.get("glob"),
                    }
                )

    return {
        "read_files": sorted(set(read_files)),
        "grep_requests": _dedupe_dicts(grep_requests),
        "tool_uses": tool_uses,
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


def _coerce(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
