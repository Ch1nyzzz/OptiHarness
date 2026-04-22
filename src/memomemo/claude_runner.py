"""Small code-agent CLI wrappers for proposer calls."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_ALLOWED_TOOLS = ("Read", "Write", "Edit", "Bash", "Grep", "Glob")
DEFAULT_CODEX_MODEL = "gpt-5.4"
DEFAULT_KIMI_MODEL = ""
KIMI_CLAUDE_EXECUTABLE = "claude-kimi"
LEGACY_KIMI_EXECUTABLE = "kimi"
DEFAULT_DOCKER_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "KIMI_API_KEY",
    "MOONSHOT_API_KEY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)


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


@dataclass(frozen=True)
class ProposerSandboxConfig:
    """Filesystem isolation settings for proposer code-agent invocations."""

    kind: str = "none"
    docker_image: str = ""
    docker_workspace: str = "/workspace"
    docker_env_vars: tuple[str, ...] = DEFAULT_DOCKER_ENV_VARS
    docker_mounts: tuple[str, ...] = ()
    docker_kimi_cli_kind: str = "claude"
    docker_user: str = ""
    docker_home: str = ""


@dataclass(frozen=True)
class _PreparedAgentCommand:
    command: tuple[str, ...]
    run_cwd: Path
    extract_cwd: Path
    error: str = ""


def has_claude_cli() -> bool:
    return shutil.which("claude") is not None


def has_codex_cli() -> bool:
    return shutil.which("codex") is not None


def has_kimi_cli() -> bool:
    return _kimi_cli_kind() is not None


def _kimi_cli_kind() -> str | None:
    if shutil.which(KIMI_CLAUDE_EXECUTABLE) is not None:
        return "claude"
    if shutil.which(LEGACY_KIMI_EXECUTABLE) is not None:
        return "legacy"
    return None


def _uses_docker_sandbox(sandbox: ProposerSandboxConfig | None) -> bool:
    return sandbox is not None and sandbox.kind.strip().lower() == "docker"


def _agent_visible_cwd(cwd: Path, *, sandbox: ProposerSandboxConfig | None) -> Path:
    if not _uses_docker_sandbox(sandbox):
        return cwd
    return Path(str(sandbox.docker_workspace or "/workspace"))


def _docker_kimi_cli_kind(sandbox: ProposerSandboxConfig | None) -> str:
    kind = (sandbox.docker_kimi_cli_kind if sandbox is not None else "claude").strip().lower()
    return "legacy" if kind == "legacy" else "claude"


def _prepare_agent_command(
    command: tuple[str, ...],
    *,
    cwd: Path,
    sandbox: ProposerSandboxConfig | None,
) -> _PreparedAgentCommand:
    if sandbox is None or sandbox.kind.strip().lower() == "none":
        return _PreparedAgentCommand(command=command, run_cwd=cwd, extract_cwd=cwd)

    if not _uses_docker_sandbox(sandbox):
        return _PreparedAgentCommand(
            command=command,
            run_cwd=cwd,
            extract_cwd=cwd,
            error=f"unsupported proposer sandbox: {sandbox.kind!r}",
        )

    image = sandbox.docker_image.strip()
    if not image:
        return _PreparedAgentCommand(
            command=command,
            run_cwd=cwd,
            extract_cwd=cwd,
            error="--proposer-docker-image is required when --proposer-sandbox=docker",
        )
    if shutil.which("docker") is None:
        docker_command = ("docker", "run", "--rm", "-i", image, *command)
        return _PreparedAgentCommand(
            command=docker_command,
            run_cwd=cwd,
            extract_cwd=_agent_visible_cwd(cwd, sandbox=sandbox),
            error="docker CLI not found on PATH",
        )

    workspace = str(sandbox.docker_workspace or "/workspace")
    docker_parts: list[str] = [
        "docker",
        "run",
        "--rm",
        "-i",
        "-v",
        f"{cwd.resolve(strict=False)}:{workspace}:rw",
        "-w",
        workspace,
        "--entrypoint",
        "",
    ]
    if sandbox.docker_user.strip():
        docker_parts.extend(["--user", sandbox.docker_user.strip()])
    if sandbox.docker_home.strip():
        docker_parts.extend(["-e", f"HOME={sandbox.docker_home.strip()}"])
    for env_name in _dedupe_strings(sandbox.docker_env_vars):
        if env_name in os.environ:
            docker_parts.extend(["-e", env_name])
    for mount in sandbox.docker_mounts:
        mount_arg = _normalize_docker_mount(mount)
        if mount_arg:
            docker_parts.extend(["-v", mount_arg])
    docker_parts.append(image)
    docker_parts.extend(command)
    return _PreparedAgentCommand(
        command=tuple(docker_parts),
        run_cwd=cwd,
        extract_cwd=Path(workspace),
    )


def _dedupe_strings(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)


def _normalize_docker_mount(spec: str) -> str:
    text = str(spec).strip()
    if not text:
        return ""
    parts = text.split(":", 1)
    if len(parts) != 2:
        return text
    host, rest = parts
    if host.startswith(("~", ".", "/")):
        host = str(Path(host).expanduser().resolve(strict=False))
    return f"{host}:{rest}"


def run_code_agent_prompt(
    prompt: str,
    *,
    agent: str,
    cwd: Path,
    log_dir: Path,
    name: str,
    model: str,
    timeout_s: int = 2400,
    sandbox: ProposerSandboxConfig | None = None,
) -> ClaudeResult:
    """Run a configured proposer code agent."""

    normalized = agent.strip().lower()
    if normalized == "claude":
        return run_claude_prompt(
            prompt,
            cwd=cwd,
            log_dir=log_dir,
            name=name,
            model=model,
            timeout_s=timeout_s,
            sandbox=sandbox,
        )
    if normalized == "codex":
        return run_codex_prompt(
            prompt,
            cwd=cwd,
            log_dir=log_dir,
            name=name,
            model=model,
            timeout_s=timeout_s,
            sandbox=sandbox,
        )
    if normalized == "kimi":
        return run_kimi_prompt(
            prompt,
            cwd=cwd,
            log_dir=log_dir,
            name=name,
            model=model,
            timeout_s=timeout_s,
            sandbox=sandbox,
        )
    raise ValueError(f"unsupported proposer agent: {agent!r}")


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
    sandbox: ProposerSandboxConfig | None = None,
) -> ClaudeResult:
    """Run `claude -p` non-interactively and persist logs."""

    cwd = cwd.resolve(strict=False)
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
    prepared = _prepare_agent_command(command, cwd=cwd, sandbox=sandbox)

    if prepared.error:
        result = ClaudeResult(
            returncode=None,
            timed_out=False,
            stdout="",
            stderr=prepared.error,
            raw_stdout="",
            command=prepared.command,
            usage=None,
            tool_access=_empty_tool_access(),
            duration_s=0.0,
            metrics={},
        )
        _write_logs(result, log_dir=log_dir, name=name, prompt=prompt)
        return result

    if not _uses_docker_sandbox(sandbox) and not has_claude_cli():
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
            prepared.command,
            input=prompt,
            cwd=str(prepared.run_cwd),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        raw_stdout = completed.stdout or ""
        stdout, usage = _extract_stream_result(raw_stdout)
        tool_access = _extract_tool_access(raw_stdout, cwd=prepared.extract_cwd)
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
            command=prepared.command,
            usage=usage,
            tool_access=tool_access,
            duration_s=duration_s,
            metrics=metrics,
        )
    except subprocess.TimeoutExpired as exc:
        raw_stdout = _coerce(exc.stdout)
        tool_access = _extract_tool_access(raw_stdout, cwd=prepared.extract_cwd)
        duration_s = time.time() - started
        result = ClaudeResult(
            returncode=None,
            timed_out=True,
            stdout=raw_stdout,
            stderr=_coerce(exc.stderr),
            raw_stdout=raw_stdout,
            command=prepared.command,
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


def run_codex_prompt(
    prompt: str,
    *,
    cwd: Path,
    log_dir: Path,
    name: str,
    model: str = DEFAULT_CODEX_MODEL,
    timeout_s: int = 2400,
    sandbox: ProposerSandboxConfig | None = None,
) -> ClaudeResult:
    """Run `codex exec` non-interactively and persist logs."""

    cwd = cwd.resolve(strict=False)
    agent_cwd = _agent_visible_cwd(cwd, sandbox=sandbox)
    command = (
        "codex",
        "exec",
        "--model",
        model,
        "--dangerously-bypass-approvals-and-sandbox",
        "--cd",
        str(agent_cwd),
        "--json",
        "-",
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    prepared = _prepare_agent_command(command, cwd=cwd, sandbox=sandbox)

    if prepared.error:
        result = ClaudeResult(
            returncode=None,
            timed_out=False,
            stdout="",
            stderr=prepared.error,
            raw_stdout="",
            command=prepared.command,
            usage=None,
            tool_access=_empty_tool_access(),
            duration_s=0.0,
            metrics={},
        )
        _write_logs(result, log_dir=log_dir, name=name, prompt=prompt)
        return result

    if not _uses_docker_sandbox(sandbox) and not has_codex_cli():
        result = ClaudeResult(
            returncode=None,
            timed_out=False,
            stdout="",
            stderr="codex CLI not found on PATH",
            raw_stdout="",
            command=command,
            usage=None,
            tool_access=_empty_tool_access(),
            duration_s=0.0,
            metrics={},
        )
        _write_logs(result, log_dir=log_dir, name=name, prompt=prompt)
        return result

    try:
        completed = subprocess.run(
            prepared.command,
            input=prompt,
            cwd=str(prepared.run_cwd),
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        raw_stdout = completed.stdout or ""
        stdout, usage = _extract_codex_result(raw_stdout)
        tool_access = _extract_codex_tool_access(raw_stdout, cwd=prepared.extract_cwd)
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
            command=prepared.command,
            usage=usage,
            tool_access=tool_access,
            duration_s=duration_s,
            metrics=metrics,
        )
    except subprocess.TimeoutExpired as exc:
        raw_stdout = _coerce(exc.stdout)
        tool_access = _extract_codex_tool_access(raw_stdout, cwd=prepared.extract_cwd)
        duration_s = time.time() - started
        result = ClaudeResult(
            returncode=None,
            timed_out=True,
            stdout=raw_stdout,
            stderr=_coerce(exc.stderr),
            raw_stdout=raw_stdout,
            command=prepared.command,
            usage=None,
            tool_access=tool_access,
            duration_s=duration_s,
            metrics=_extract_session_metrics(
                usage=None,
                tool_access=tool_access,
                duration_s=duration_s,
            ),
        )

    _write_logs(result, log_dir=log_dir, name=name, prompt=prompt)
    return result


def run_kimi_prompt(
    prompt: str,
    *,
    cwd: Path,
    log_dir: Path,
    name: str,
    model: str = DEFAULT_KIMI_MODEL,
    timeout_s: int = 2400,
    sandbox: ProposerSandboxConfig | None = None,
) -> ClaudeResult:
    """Run a Kimi proposer CLI non-interactively and persist logs."""

    cwd = cwd.resolve(strict=False)
    cli_kind = (
        _docker_kimi_cli_kind(sandbox)
        if _uses_docker_sandbox(sandbox)
        else _kimi_cli_kind()
    )
    command = _kimi_command(
        cwd=_agent_visible_cwd(cwd, sandbox=sandbox),
        model=model,
        cli_kind=cli_kind or "claude",
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    prepared = _prepare_agent_command(command, cwd=cwd, sandbox=sandbox)

    if prepared.error:
        result = ClaudeResult(
            returncode=None,
            timed_out=False,
            stdout="",
            stderr=prepared.error,
            raw_stdout="",
            command=prepared.command,
            usage=None,
            tool_access=_empty_tool_access(),
            duration_s=0.0,
            metrics={},
        )
        _write_logs(result, log_dir=log_dir, name=name, prompt=prompt)
        return result

    if cli_kind is None:
        result = ClaudeResult(
            returncode=None,
            timed_out=False,
            stdout="",
            stderr="claude-kimi or kimi CLI not found on PATH",
            raw_stdout="",
            command=command,
            usage=None,
            tool_access=_empty_tool_access(),
            duration_s=0.0,
            metrics={},
        )
        _write_logs(result, log_dir=log_dir, name=name, prompt=prompt)
        return result

    try:
        completed = subprocess.run(
            prepared.command,
            input=prompt,
            cwd=str(prepared.run_cwd),
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        raw_stdout = completed.stdout or ""
        if cli_kind == "legacy":
            stdout, usage = _extract_kimi_result(raw_stdout)
            usage = _merge_usage_payloads(
                usage,
                _export_kimi_session_usage(
                    completed.stderr or "",
                    log_dir=log_dir,
                    name=name,
                    timeout_s=min(timeout_s, 120),
                ),
            )
        else:
            stdout, usage = _extract_stream_result(raw_stdout)
        tool_access = _extract_kimi_tool_access(raw_stdout, cwd=prepared.extract_cwd)
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
            command=prepared.command,
            usage=usage,
            tool_access=tool_access,
            duration_s=duration_s,
            metrics=metrics,
        )
    except subprocess.TimeoutExpired as exc:
        raw_stdout = _coerce(exc.stdout)
        tool_access = _extract_kimi_tool_access(raw_stdout, cwd=prepared.extract_cwd)
        duration_s = time.time() - started
        result = ClaudeResult(
            returncode=None,
            timed_out=True,
            stdout=raw_stdout,
            stderr=_coerce(exc.stderr),
            raw_stdout=raw_stdout,
            command=prepared.command,
            usage=None,
            tool_access=tool_access,
            duration_s=duration_s,
            metrics=_extract_session_metrics(
                usage=None,
                tool_access=tool_access,
                duration_s=duration_s,
            ),
        )

    _write_logs(result, log_dir=log_dir, name=name, prompt=prompt)
    return result


def _kimi_command(*, cwd: Path, model: str, cli_kind: str) -> tuple[str, ...]:
    if cli_kind == "legacy":
        command_parts = [
            LEGACY_KIMI_EXECUTABLE,
            "--work-dir",
            str(cwd),
            "--yolo",
            "--print",
            "--input-format",
            "text",
            "--output-format",
            "stream-json",
        ]
        if model:
            command_parts.extend(["--model", model])
        return tuple(command_parts)

    command_parts = [
        KIMI_CLAUDE_EXECUTABLE,
        "-p",
        "--tools",
        ",".join(DEFAULT_ALLOWED_TOOLS),
        "--allowedTools",
        ",".join(DEFAULT_ALLOWED_TOOLS),
        "--output-format",
        "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
        "--input-format",
        "text",
    ]
    if model:
        command_parts.extend(["--model", model])
    return tuple(command_parts)


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


def _extract_codex_result(raw_stdout: str) -> tuple[str, dict[str, Any] | None]:
    events = _jsonl_events(raw_stdout)
    text_chunks: list[str] = []
    usage: dict[str, Any] = {}
    for event in events:
        event_type = str(event.get("type") or event.get("event") or "")
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "agent_message":
            value = item.get("text")
            if isinstance(value, str) and value:
                text_chunks.append(value)
        if event_type in {"result", "final", "agent_message", "message"}:
            for key in ("result", "message", "text", "content", "last_message"):
                value = event.get(key)
                if isinstance(value, str) and value:
                    text_chunks.append(value)
                    break
        event_usage = event.get("usage")
        if isinstance(event_usage, dict):
            usage["usage"] = _merge_usage_dicts(
                usage.get("usage") if isinstance(usage.get("usage"), dict) else {},
                event_usage,
            )
        for key in ("total_cost_usd", "duration_ms", "duration_api_ms", "num_turns", "session_id"):
            if key in event:
                usage[key] = event[key]
    return "\n".join(text_chunks) or raw_stdout, usage or None


def _extract_kimi_result(raw_stdout: str) -> tuple[str, dict[str, Any] | None]:
    events = _jsonl_events(raw_stdout)
    text_chunks: list[str] = []
    usage: dict[str, Any] = {}
    for event in events:
        if event.get("role") == "assistant":
            content = event.get("content")
            if isinstance(content, str) and content:
                text_chunks.append(content)
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        text_chunks.append(item["text"])
        event_usage = event.get("usage")
        if isinstance(event_usage, dict):
            usage["usage"] = _merge_usage_dicts(
                usage.get("usage") if isinstance(usage.get("usage"), dict) else {},
                event_usage,
            )
    if text_chunks or usage:
        return "\n".join(text_chunks) or raw_stdout, usage or None

    text, usage = _extract_stream_result(raw_stdout)
    if text != raw_stdout or usage:
        return text, usage
    return _extract_codex_result(raw_stdout)


def _merge_usage_payloads(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not left:
        return right
    if not right:
        return left
    merged = dict(left)
    left_usage = merged.get("usage") if isinstance(merged.get("usage"), dict) else {}
    right_usage = right.get("usage") if isinstance(right.get("usage"), dict) else {}
    if left_usage or right_usage:
        merged["usage"] = _merge_usage_dicts(left_usage, right_usage)
    for key, value in right.items():
        if key != "usage":
            merged[key] = value
    return merged


def _export_kimi_session_usage(
    stderr: str,
    *,
    log_dir: Path,
    name: str,
    timeout_s: int,
) -> dict[str, Any] | None:
    session_id = _extract_kimi_session_id(stderr)
    if not session_id:
        return None
    prefix = log_dir / name
    prefix.mkdir(parents=True, exist_ok=True)
    archive_path = prefix / "kimi_session.zip"
    try:
        completed = subprocess.run(
            (
                "kimi",
                "export",
                session_id,
                "--output",
                str(archive_path),
                "--yes",
            ),
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"session_id": session_id}
    if completed.returncode != 0 or not archive_path.exists():
        return {"session_id": session_id}
    usage = _extract_kimi_usage_from_export(archive_path)
    usage["session_id"] = session_id
    return usage


def _extract_kimi_session_id(stderr: str) -> str:
    marker = "To resume this session: kimi -r "
    for line in stderr.splitlines():
        if marker not in line:
            continue
        return line.split(marker, 1)[1].strip()
    return ""


def _extract_kimi_usage_from_export(archive_path: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            with archive.open("wire.jsonl") as file:
                wire_jsonl = file.read().decode("utf-8", errors="replace")
    except (OSError, KeyError, zipfile.BadZipFile):
        return {}
    return _extract_kimi_wire_usage(wire_jsonl)


def _extract_kimi_wire_usage(wire_jsonl: str) -> dict[str, Any]:
    usage: dict[str, int] = {}
    context_tokens = 0
    max_context_tokens = 0
    for event in _jsonl_events(wire_jsonl):
        message = event.get("message")
        if not isinstance(message, dict) or message.get("type") != "StatusUpdate":
            continue
        payload = message.get("payload")
        if not isinstance(payload, dict):
            continue
        token_usage = payload.get("token_usage")
        if isinstance(token_usage, dict):
            usage["input_tokens"] = usage.get("input_tokens", 0) + _int_metric(
                token_usage.get("input_other")
            )
            usage["output_tokens"] = usage.get("output_tokens", 0) + _int_metric(
                token_usage.get("output")
            )
            usage["cache_read_input_tokens"] = usage.get(
                "cache_read_input_tokens", 0
            ) + _int_metric(token_usage.get("input_cache_read"))
            usage["cache_creation_input_tokens"] = usage.get(
                "cache_creation_input_tokens", 0
            ) + _int_metric(token_usage.get("input_cache_creation"))
        context_tokens = max(context_tokens, _int_metric(payload.get("context_tokens")))
        max_context_tokens = max(
            max_context_tokens,
            _int_metric(payload.get("max_context_tokens")),
        )
    out: dict[str, Any] = {}
    if usage:
        out["usage"] = usage
    if context_tokens:
        out["context_tokens"] = context_tokens
    if max_context_tokens:
        out["max_context_tokens"] = max_context_tokens
    return out


def _jsonl_events(raw_stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in raw_stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _merge_usage_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, int]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, (int, float)):
            merged[str(key)] = _int_metric(merged.get(str(key), 0)) + _int_metric(value)
    return merged


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
        elif _is_shell_tool_name(name):
            _add_shell_command_access(
                record,
                files_read=files_read,
                files_written=files_written,
                grep_requests=grep_requests,
                cwd=cwd,
            )

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


def _extract_codex_tool_access(raw_stdout: str, *, cwd: Path | str | None = None) -> dict[str, Any]:
    tool_uses: list[dict[str, Any]] = []
    files_read: dict[str, dict[str, int]] = {}
    files_written: dict[str, dict[str, int]] = {}
    grep_requests: list[dict[str, Any]] = []

    for event in _jsonl_events(raw_stdout):
        item = event.get("item")
        if (
            isinstance(item, dict)
            and item.get("type") == "command_execution"
            and event.get("type") != "item.completed"
        ):
            continue
        name = _codex_tool_name(event)
        if not name:
            continue
        tool_input = _codex_tool_input(event)
        record = {
            "id": event.get("id") or event.get("call_id") or event.get("item_id"),
            "name": name,
            "input": tool_input,
        }
        output = _codex_tool_output(event)
        if output:
            record["_output"] = output
        tool_uses.append(record)
        path = _tool_path(tool_input)
        if path and name in {"Read", "read_file"}:
            rel = _make_relative(path, cwd)
            current = files_read.setdefault(rel, {"reads": 0, "lines": 0})
            current["reads"] += 1
        elif path and name in {"Write", "Edit", "apply_patch", "write_file"}:
            _add_written_lines(
                files_written,
                _make_relative(path, cwd),
                _count_text_lines(tool_input.get("content") or tool_input.get("new_string")),
            )
        elif name in {"Grep", "rg", "search"}:
            grep_requests.append(
                {
                    "pattern": tool_input.get("pattern") or tool_input.get("query"),
                    "path": tool_input.get("path"),
                    "glob": tool_input.get("glob"),
                }
            )
        elif _is_shell_tool_name(name):
            _add_shell_command_access(
                record,
                files_read=files_read,
                files_written=files_written,
                grep_requests=grep_requests,
                cwd=cwd,
            )

    for record in tool_uses:
        record.pop("_output", None)

    return {
        "read_files": sorted(files_read),
        "grep_requests": _dedupe_dicts(grep_requests),
        "tool_uses": tool_uses,
        "tool_counts": dict(
            sorted(Counter(str(item.get("name") or "") for item in tool_uses).items())
        ),
        "files_read": dict(sorted(files_read.items())),
        "files_written": dict(sorted(files_written.items())),
    }


def _extract_kimi_tool_access(raw_stdout: str, *, cwd: Path | str | None = None) -> dict[str, Any]:
    access = _extract_tool_access(raw_stdout, cwd=cwd)
    if access["tool_uses"]:
        return access
    access = _extract_kimi_role_tool_access(raw_stdout, cwd=cwd)
    if access["tool_uses"]:
        return access
    return _extract_codex_tool_access(raw_stdout, cwd=cwd)


def _extract_kimi_role_tool_access(
    raw_stdout: str,
    *,
    cwd: Path | str | None = None,
) -> dict[str, Any]:
    tool_uses: list[dict[str, Any]] = []
    tool_by_id: dict[str, dict[str, Any]] = {}
    grep_requests: list[dict[str, Any]] = []

    for event in _jsonl_events(raw_stdout):
        if event.get("role") == "assistant":
            for call in event.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if not isinstance(function, dict):
                    function = {}
                name = str(function.get("name") or call.get("name") or "")
                tool_input = _parse_tool_arguments(function.get("arguments"))
                record = {
                    "id": call.get("id"),
                    "name": name,
                    "input": tool_input,
                }
                tool_uses.append(record)
                if record["id"]:
                    tool_by_id[str(record["id"])] = record
        elif event.get("role") == "tool":
            tool_id = str(event.get("tool_call_id") or "")
            record = tool_by_id.get(tool_id)
            if record is None:
                continue
            record["_output"] = _stringify_tool_result_content(event.get("content"))

    files_read: dict[str, dict[str, int]] = {}
    files_written: dict[str, dict[str, int]] = {}
    for record in tool_uses:
        name = str(record.get("name") or "")
        normalized = name.lower()
        tool_input = record.get("input") if isinstance(record.get("input"), dict) else {}
        path = _tool_path(tool_input)
        if path and normalized in {"readfile", "read", "read_file"}:
            current = files_read.setdefault(
                _make_relative(path, cwd),
                {"reads": 0, "lines": 0},
            )
            current["reads"] += 1
            lines = _count_read_lines(str(record.get("_output") or ""))
            current["lines"] += lines
            if lines:
                record["output_lines"] = lines
        elif path and normalized in {
            "writefile",
            "write",
            "write_file",
            "editfile",
            "edit",
            "apply_patch",
        }:
            _add_written_lines(
                files_written,
                _make_relative(path, cwd),
                _count_text_lines(
                    tool_input.get("content")
                    or tool_input.get("new_string")
                    or tool_input.get("text")
                ),
            )
        elif normalized in {"grep", "search", "rg"}:
            grep_requests.append(
                {
                    "pattern": tool_input.get("pattern") or tool_input.get("query"),
                    "path": tool_input.get("path"),
                    "glob": tool_input.get("glob"),
                }
            )
        elif _is_shell_tool_name(name):
            _add_shell_command_access(
                record,
                files_read=files_read,
                files_written=files_written,
                grep_requests=grep_requests,
                cwd=cwd,
            )

    for record in tool_uses:
        record.pop("_output", None)

    return {
        "read_files": sorted(files_read),
        "grep_requests": _dedupe_dicts(grep_requests),
        "tool_uses": tool_uses,
        "tool_counts": dict(
            sorted(Counter(str(item.get("name") or "") for item in tool_uses).items())
        ),
        "files_read": dict(sorted(files_read.items())),
        "files_written": dict(sorted(files_written.items())),
    }


def _parse_tool_arguments(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _codex_tool_name(event: dict[str, Any]) -> str:
    item = event.get("item")
    if isinstance(item, dict) and item.get("type") == "command_execution":
        return "Shell"
    for key in ("tool_name", "name", "tool", "command"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    if isinstance(item, dict):
        for key in ("tool_name", "name", "tool", "command"):
            value = item.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _codex_tool_input(event: dict[str, Any]) -> dict[str, Any]:
    for key in ("input", "arguments", "args"):
        value = event.get(key)
        if isinstance(value, dict):
            return value
    item = event.get("item")
    if isinstance(item, dict):
        if item.get("type") == "command_execution" and isinstance(item.get("command"), str):
            return {"command": item["command"]}
        for key in ("input", "arguments", "args"):
            value = item.get(key)
            if isinstance(value, dict):
                return value
    return {}


def _tool_path(tool_input: dict[str, Any]) -> str:
    for key in ("file_path", "path", "filename"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _codex_tool_output(event: dict[str, Any]) -> str:
    item = event.get("item")
    if isinstance(item, dict):
        for key in ("aggregated_output", "output", "stdout"):
            value = item.get(key)
            if isinstance(value, str) and value:
                return value
    for key in ("aggregated_output", "output", "stdout"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _empty_tool_access() -> dict[str, Any]:
    return {
        "read_files": [],
        "grep_requests": [],
        "tool_uses": [],
        "tool_counts": {},
        "files_read": {},
        "files_written": {},
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
    cache_read_tokens = _int_metric(
        token_usage.get("cache_read_input_tokens", token_usage.get("cached_input_tokens", 0))
    )
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


def _is_shell_tool_name(name: object) -> bool:
    normalized = str(name or "").strip().lower()
    return normalized in {"bash", "shell", "execute_commands", "bash_command"}


def _add_shell_command_access(
    record: dict[str, Any],
    *,
    files_read: dict[str, dict[str, int]],
    files_written: dict[str, dict[str, int]],
    grep_requests: list[dict[str, Any]],
    cwd: Path | str | None,
) -> None:
    tool_input = record.get("input") if isinstance(record.get("input"), dict) else {}
    commands = _shell_commands_from_input(tool_input)
    if not commands:
        return

    output = str(record.get("_output") or "")
    shell_read_paths: list[str] = []
    shell_written_paths: list[str] = []
    for command in commands:
        parsed = _parse_shell_command_access(command)
        shell_read_paths.extend(parsed["read_paths"])
        shell_written_paths.extend(parsed["written_paths"])
        grep_requests.extend(parsed["grep_requests"])

    read_paths = _dedupe_strings_preserve_order(shell_read_paths)
    written_paths = _dedupe_strings_preserve_order(shell_written_paths)
    read_lines = _count_shell_output_lines(output) if len(read_paths) == 1 else 0
    for path in read_paths:
        current = files_read.setdefault(_make_relative(path, cwd), {"reads": 0, "lines": 0})
        current["reads"] += 1
        current["lines"] += read_lines
    for path in written_paths:
        _add_written_lines(files_written, _make_relative(path, cwd), 0)
    if read_paths:
        record["shell_files_read"] = [_make_relative(path, cwd) for path in read_paths]
    if written_paths:
        record["shell_files_written"] = [_make_relative(path, cwd) for path in written_paths]


def _shell_commands_from_input(tool_input: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    command = tool_input.get("command")
    if isinstance(command, str) and command.strip():
        commands.append(command)
    raw_commands = tool_input.get("commands")
    if isinstance(raw_commands, list):
        for item in raw_commands:
            if isinstance(item, str) and item.strip():
                commands.append(item)
            elif isinstance(item, dict):
                for key in ("command", "keystrokes", "cmd"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        commands.append(value)
                        break
    return commands


def _parse_shell_command_access(command: str) -> dict[str, Any]:
    unwrapped = _unwrap_shell_command(command)
    read_paths: list[str] = []
    written_paths: list[str] = []
    grep_requests: list[dict[str, Any]] = []

    read_paths.extend(_extract_python_read_paths(unwrapped))
    written_paths.extend(_extract_python_write_paths(unwrapped))

    for segment in _shell_command_segments(unwrapped):
        if not segment:
            continue
        tokens = _strip_env_assignments(segment)
        if not tokens:
            continue
        cmd = Path(tokens[0]).name
        if cmd in {"sudo", "env", "timeout", "time", "command"} and len(tokens) > 1:
            tokens = _strip_env_assignments(tokens[1:])
            if not tokens:
                continue
            cmd = Path(tokens[0]).name

        if cmd in {"cat", "sed", "head", "tail", "nl", "wc"}:
            read_paths.extend(_path_args(tokens[1:]))
        elif cmd == "jq":
            read_paths.extend(_jq_path_args(tokens[1:]))
        elif cmd in {"grep", "egrep", "fgrep", "rg"}:
            grep_requests.append(_grep_request_from_tokens(cmd, tokens[1:]))
        elif cmd in {"tee"}:
            written_paths.extend(_path_args(tokens[1:]))

        written_paths.extend(_redirect_paths(tokens))

    return {
        "read_paths": _dedupe_strings_preserve_order(read_paths),
        "written_paths": _dedupe_strings_preserve_order(written_paths),
        "grep_requests": _dedupe_dicts(grep_requests),
    }


def _unwrap_shell_command(command: str) -> str:
    current = command.strip()
    for _ in range(3):
        try:
            tokens = shlex.split(current)
        except ValueError:
            return current
        if len(tokens) < 3:
            return current
        exe = Path(tokens[0]).name
        if exe not in {"bash", "sh", "zsh"}:
            return current
        for idx, token in enumerate(tokens[1:], start=1):
            if token.startswith("-") and "c" in token and idx + 1 < len(tokens):
                current = tokens[idx + 1]
                break
        else:
            return current
    return current


def _shell_command_segments(command: str) -> list[list[str]]:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;()<>")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except (TypeError, ValueError):
        try:
            tokens = shlex.split(command)
        except ValueError:
            return []

    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in {"|", "||", "&&", ";", "(", ")"}:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(token)
    if current:
        segments.append(current)
    return segments


def _strip_env_assignments(tokens: list[str]) -> list[str]:
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == "env":
            idx += 1
            continue
        if "=" in token and not token.startswith("-") and token.split("=", 1)[0].isidentifier():
            idx += 1
            continue
        break
    return tokens[idx:]


def _path_args(tokens: list[str]) -> list[str]:
    paths: list[str] = []
    skip_next = False
    options_with_values = {
        "-e",
        "-f",
        "-m",
        "-n",
        "-C",
        "-A",
        "-B",
        "--context",
        "--after-context",
        "--before-context",
        "--max-count",
        "--lines",
        "--bytes",
    }
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in {">", ">>", "2>", "2>>", "<", "<<"}:
            skip_next = token in {">", ">>", "2>", "2>>", "<", "<<"}
            continue
        if token in options_with_values:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        if _looks_like_path(token):
            paths.append(_clean_shell_path_token(token))
    return paths


def _jq_path_args(tokens: list[str]) -> list[str]:
    paths: list[str] = []
    filter_seen = False
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in {"-f", "--from-file", "-L"}:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        if _looks_like_path(token):
            paths.append(_clean_shell_path_token(token))
            continue
        if not filter_seen:
            filter_seen = True
    return paths


def _grep_request_from_tokens(command_name: str, tokens: list[str]) -> dict[str, Any]:
    pattern: str | None = None
    paths: list[str] = []
    skip_next = False
    expect_pattern = False
    files_only = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in {"-e", "--regexp"}:
            expect_pattern = True
            continue
        if token in {"-f", "--file", "-C", "-A", "-B", "--context", "--after-context", "--before-context"}:
            skip_next = True
            continue
        if token == "--files" and command_name == "rg":
            files_only = True
            continue
        if token.startswith("-"):
            continue
        if expect_pattern:
            pattern = token
            expect_pattern = False
            continue
        if pattern is None and not files_only:
            pattern = token
            continue
        if _looks_like_path(token):
            paths.append(_clean_shell_path_token(token))
    return {
        "pattern": pattern,
        "path": ", ".join(paths) if paths else None,
        "glob": None,
    }


def _redirect_paths(tokens: list[str]) -> list[str]:
    paths: list[str] = []
    for idx, token in enumerate(tokens[:-1]):
        if token in {">", ">>", "1>", "1>>", "2>", "2>>"} and _looks_like_path(tokens[idx + 1]):
            paths.append(_clean_shell_path_token(tokens[idx + 1]))
    for token in tokens:
        match = re.match(r"^(?:[12])?>>(.+)$", token)
        if match and _looks_like_path(match.group(1)):
            paths.append(_clean_shell_path_token(match.group(1)))
    return paths


def _extract_python_read_paths(command: str) -> list[str]:
    paths: list[str] = []
    for pattern in (
        r"(?:Path|pathlib\.Path)\(\s*['\"]([^'\"]+)['\"]\s*\)\.read_text\s*\(",
        r"\bopen\(\s*['\"]([^'\"]+)['\"]\s*(?:,\s*['\"]([^'\"]*)['\"])?",
    ):
        for match in re.finditer(pattern, command):
            path = match.group(1)
            mode = match.group(2) if len(match.groups()) > 1 else None
            if mode and any(flag in mode for flag in ("w", "a", "+")):
                continue
            if _looks_like_path(path):
                paths.append(_clean_shell_path_token(path))
    return paths


def _extract_python_write_paths(command: str) -> list[str]:
    paths: list[str] = []
    for pattern in (
        r"(?:Path|pathlib\.Path)\(\s*['\"]([^'\"]+)['\"]\s*\)\.write_text\s*\(",
        r"\bopen\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]*[wa][^'\"]*)['\"]",
    ):
        for match in re.finditer(pattern, command):
            path = match.group(1)
            if _looks_like_path(path):
                paths.append(_clean_shell_path_token(path))
    return paths


def _looks_like_path(token: str) -> bool:
    value = _clean_shell_path_token(token)
    if not value or value.startswith("$") or value in {"-", "/dev/null"}:
        return False
    if value.startswith(("/", "./", "../", "~")):
        return True
    if "/" in value or "*" in value or "?" in value:
        return True
    suffix = Path(value).suffix.lower()
    return suffix in {
        ".py",
        ".json",
        ".jsonl",
        ".md",
        ".txt",
        ".yaml",
        ".yml",
        ".toml",
        ".lock",
        ".patch",
        ".log",
        ".csv",
        ".tsv",
        ".db",
        ".pkl",
        ".npy",
    }


def _clean_shell_path_token(token: str) -> str:
    return token.strip().strip("'\"").rstrip(",:")


def _dedupe_strings_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _clean_shell_path_token(value)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _count_shell_output_lines(output: str) -> int:
    return len([line for line in output.splitlines() if line.strip()])


def _count_read_lines(output: str) -> int:
    return sum(1 for line in output.splitlines() if _is_numbered_read_line(line))


def _is_numbered_read_line(line: str) -> bool:
    stripped = line.lstrip()
    idx = 0
    while idx < len(stripped) and stripped[idx].isdigit():
        idx += 1
    return idx > 0 and (
        stripped[idx:].startswith("\u2192") or stripped[idx:].startswith("\t")
    )


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
