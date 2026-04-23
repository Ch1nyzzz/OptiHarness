import json
from types import SimpleNamespace

from memomemo.claude_runner import (
    DEFAULT_CODEX_MODEL,
    ProposerSandboxConfig,
    _extract_codex_result,
    _extract_codex_tool_access,
    _extract_kimi_result,
    _extract_kimi_tool_access,
    _extract_kimi_wire_usage,
    _extract_session_metrics,
    _extract_stream_result,
    _extract_tool_access,
    run_codex_prompt,
    run_claude_prompt,
    run_kimi_prompt,
)


def test_extract_tool_access_tracks_read_and_grep_requests():
    raw_stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_read_1",
                                "name": "Read",
                                "input": {"file_path": "/repo/src/memomemo/optimizer.py"},
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_grep_1",
                                "name": "Grep",
                                "input": {
                                    "pattern": "pending_eval",
                                    "path": "/repo/src",
                                    "glob": "*.py",
                                },
                            },
                        ]
                    },
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_read_2",
                                "name": "Read",
                                "input": {"file_path": "/repo/src/memomemo/optimizer.py"},
                            }
                        ]
                    },
                }
            ),
        ]
    )

    access = _extract_tool_access(raw_stdout)

    assert access["read_files"] == ["/repo/src/memomemo/optimizer.py"]
    assert access["grep_requests"] == [
        {"pattern": "pending_eval", "path": "/repo/src", "glob": "*.py"}
    ]
    assert [item["name"] for item in access["tool_uses"]] == ["Read", "Grep", "Read"]


def test_extract_tool_access_counts_read_and_write_file_stats():
    raw_stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_read_1",
                                "name": "Read",
                                "input": {"file_path": "/repo/src/a.py"},
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_write_1",
                                "name": "Write",
                                "input": {
                                    "file_path": "/repo/src/b.py",
                                    "content": "one\ntwo",
                                },
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_edit_1",
                                "name": "Edit",
                                "input": {
                                    "file_path": "/repo/src/a.py",
                                    "old_string": "old",
                                    "new_string": "new",
                                },
                            },
                        ]
                    },
                }
            ),
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_read_1",
                                "content": "     1\u2192first\n     2\u2192second",
                            },
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_write_1",
                                "content": "ok",
                            },
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_edit_1",
                                "content": "ok",
                            },
                        ]
                    },
                }
            ),
        ]
    )

    access = _extract_tool_access(raw_stdout, cwd="/repo")

    assert access["files_read"] == {"src/a.py": {"reads": 1, "lines": 2}}
    assert access["files_written"] == {
        "src/a.py": {"writes": 1, "lines_written": 1},
        "src/b.py": {"writes": 1, "lines_written": 2},
    }
    assert access["tool_counts"] == {"Edit": 1, "Read": 1, "Write": 1}


def test_extract_session_metrics_summarizes_tokens_cost_and_tools():
    tool_access = {
        "tool_uses": [{"name": "Read"}, {"name": "Write"}],
        "tool_counts": {"Read": 1, "Write": 1},
        "files_read": {"src/a.py": {"reads": 2, "lines": 12}},
        "files_written": {"src/b.py": {"writes": 1, "lines_written": 3}},
    }
    usage = {
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 2,
            "cache_read_input_tokens": 7,
        },
        "total_cost_usd": 0.012345,
    }

    metrics = _extract_session_metrics(
        usage=usage,
        tool_access=tool_access,
        duration_s=1.23456,
    )

    assert metrics["total_tokens"] == 15
    assert metrics["total_reported_tokens"] == 24
    assert metrics["estimated_cost_usd"] == 0.012345
    assert metrics["duration_s"] == 1.235
    assert metrics["tool_calls"] == 2
    assert metrics["read_file_calls"] == 2
    assert metrics["unique_files_read"] == 1
    assert metrics["read_lines"] == 12
    assert metrics["write_file_calls"] == 1
    assert metrics["written_lines"] == 3


def test_extract_codex_result_and_tool_access_from_jsonl():
    raw_stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "tool_name": "Read",
                    "input": {"file_path": "/repo/src/a.py"},
                }
            ),
            json.dumps(
                {
                    "type": "tool_call",
                    "tool_name": "apply_patch",
                    "input": {"path": "/repo/src/b.py", "content": "one\ntwo"},
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "result": "done",
                    "usage": {"input_tokens": 9, "output_tokens": 4},
                }
            ),
        ]
    )

    text, usage = _extract_codex_result(raw_stdout)
    access = _extract_codex_tool_access(raw_stdout, cwd="/repo")

    assert text == "done"
    assert usage == {"usage": {"input_tokens": 9, "output_tokens": 4}}
    assert access["files_read"] == {"src/a.py": {"reads": 1, "lines": 0}}
    assert access["files_written"] == {"src/b.py": {"writes": 1, "lines_written": 2}}


def test_extract_codex_result_and_tool_access_from_item_events():
    raw_stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "thread_1"}),
            json.dumps({"type": "turn.started"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "id": "item_0",
                        "type": "agent_message",
                        "text": "I will inspect the repo.",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "item.started",
                    "item": {
                        "id": "item_1",
                        "type": "command_execution",
                        "command": "/bin/bash -lc pwd",
                        "status": "in_progress",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "id": "item_1",
                        "type": "command_execution",
                        "command": "/bin/bash -lc pwd",
                        "aggregated_output": "/repo\n",
                        "exit_code": 0,
                        "status": "completed",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "item_2", "type": "agent_message", "text": "DONE"},
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 50,
                        "cached_input_tokens": 40,
                        "output_tokens": 7,
                    },
                }
            ),
        ]
    )

    text, usage = _extract_codex_result(raw_stdout)
    access = _extract_codex_tool_access(raw_stdout, cwd="/repo")
    metrics = _extract_session_metrics(usage=usage, tool_access=access, duration_s=1.2)

    assert text == "I will inspect the repo.\nDONE"
    assert usage == {
        "usage": {"input_tokens": 50, "cached_input_tokens": 40, "output_tokens": 7}
    }
    assert access["tool_counts"] == {"Shell": 1}
    assert access["tool_uses"] == [
        {"id": None, "name": "Shell", "input": {"command": "/bin/bash -lc pwd"}}
    ]
    assert metrics["input_tokens"] == 50
    assert metrics["output_tokens"] == 7
    assert metrics["cache_read_input_tokens"] == 40
    assert metrics["tool_calls"] == 1


def test_extract_codex_tool_access_counts_shell_file_reads_and_grep_requests():
    raw_stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "id": "item_1",
                        "type": "command_execution",
                        "command": "/bin/bash -lc \"sed -n '1,20p' /repo/src/a.py && jq '.' /repo/runs/out.json\"",
                        "aggregated_output": "line one\nline two\n",
                        "exit_code": 0,
                    },
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "id": "item_2",
                        "type": "command_execution",
                        "command": "rg -n 'needle' /repo/src /repo/tests",
                        "aggregated_output": "/repo/src/a.py:1:needle\n",
                        "exit_code": 0,
                    },
                }
            ),
        ]
    )

    access = _extract_codex_tool_access(raw_stdout, cwd="/repo")

    assert access["tool_counts"] == {"Shell": 2}
    assert access["files_read"] == {
        "runs/out.json": {"reads": 1, "lines": 0},
        "src/a.py": {"reads": 1, "lines": 0},
    }
    assert access["grep_requests"] == [
        {"pattern": "needle", "path": "/repo/src, /repo/tests", "glob": None}
    ]
    assert access["tool_uses"][0]["shell_files_read"] == ["src/a.py", "runs/out.json"]


def test_extract_tool_access_counts_claude_bash_shell_file_reads():
    raw_stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_bash_1",
                                "name": "Bash",
                                "input": {
                                    "command": "cat /repo/README.md > /repo/tmp/readme.copy"
                                },
                            }
                        ]
                    },
                }
            ),
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_bash_1",
                                "content": "# Title\nbody\n",
                            }
                        ]
                    },
                }
            ),
        ]
    )

    access = _extract_tool_access(raw_stdout, cwd="/repo")

    assert access["tool_counts"] == {"Bash": 1}
    assert access["files_read"] == {"README.md": {"reads": 1, "lines": 2}}
    assert access["files_written"] == {"tmp/readme.copy": {"writes": 1, "lines_written": 0}}


def test_run_codex_prompt_uses_absolute_cd_and_records_usage(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    raw_stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "thread_1"}),
            json.dumps({"type": "turn.started"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "item_0", "type": "agent_message", "text": "done"},
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 11,
                        "cached_input_tokens": 7,
                        "output_tokens": 3,
                    },
                }
            ),
        ]
    )
    calls = []

    def fake_which(name):
        return f"/bin/{name}" if name == "codex" else None

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=raw_stdout, stderr="")

    monkeypatch.setattr("memomemo.claude_runner.shutil.which", fake_which)
    monkeypatch.setattr("memomemo.claude_runner.subprocess.run", fake_run)

    result = run_codex_prompt(
        "prompt",
        cwd=repo_dir,
        log_dir=tmp_path / "logs",
        name="iter_001",
    )

    assert result.command[:4] == ("codex", "exec", "--model", DEFAULT_CODEX_MODEL)
    assert result.command[result.command.index("--cd") + 1] == str(repo_dir.resolve())
    assert result.command[-3:] == ("--ephemeral", "--json", "-")
    assert calls == [
        (
            result.command,
            {
                "input": "prompt",
                "cwd": str(repo_dir.resolve()),
                "text": True,
                "capture_output": True,
                "timeout": 2400,
            },
        )
    ]
    assert result.stdout == "done"
    assert result.usage == {
        "usage": {"input_tokens": 11, "cached_input_tokens": 7, "output_tokens": 3}
    }
    assert result.metrics["cache_read_input_tokens"] == 7


def test_run_codex_prompt_can_run_inside_docker_sandbox(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    raw_stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "tool_name": "Read",
                    "input": {"file_path": "/workspace/src/a.py"},
                }
            ),
            json.dumps({"type": "result", "result": "done"}),
        ]
    )
    calls = []

    def fake_which(name):
        return "/bin/docker" if name == "docker" else None

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=raw_stdout, stderr="")

    monkeypatch.setattr("memomemo.claude_runner.shutil.which", fake_which)
    monkeypatch.setattr("memomemo.claude_runner.subprocess.run", fake_run)

    result = run_codex_prompt(
        "prompt",
        cwd=repo_dir,
        log_dir=tmp_path / "logs",
        name="iter_001",
        sandbox=ProposerSandboxConfig(
            kind="docker",
            docker_image="memo-proposer:test",
            docker_env_vars=("OPENAI_API_KEY",),
            docker_mounts=("~/.codex:/root/.codex:ro",),
            docker_user="1000:1000",
            docker_home="/tmp/proposer-home",
        ),
    )

    assert result.command[:4] == ("docker", "run", "--rm", "-i")
    assert f"{repo_dir.resolve()}:/workspace:rw" in result.command
    assert result.command[result.command.index("-w") + 1] == "/workspace"
    assert result.command[result.command.index("--user") + 1] == "1000:1000"
    assert "HOME=/tmp/proposer-home" in result.command
    assert "memo-proposer:test" in result.command
    assert result.command[result.command.index("--cd") + 1] == "/workspace"
    assert result.command[-3:] == ("--ephemeral", "--json", "-")
    assert calls[0][1]["cwd"] == str(repo_dir.resolve())
    assert result.tool_access["files_read"] == {"src/a.py": {"reads": 1, "lines": 0}}


def test_run_claude_prompt_can_run_inside_docker_sandbox(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    raw_stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_read_1",
                                "name": "Read",
                                "input": {"file_path": "/workspace/src/a.py"},
                            }
                        ]
                    },
                }
            ),
            json.dumps({"type": "result", "result": "done"}),
        ]
    )

    def fake_which(name):
        return "/bin/docker" if name == "docker" else None

    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout=raw_stdout, stderr="")

    monkeypatch.setattr("memomemo.claude_runner.shutil.which", fake_which)
    monkeypatch.setattr("memomemo.claude_runner.subprocess.run", fake_run)

    result = run_claude_prompt(
        "prompt",
        cwd=repo_dir,
        log_dir=tmp_path / "logs",
        name="iter_001",
        sandbox=ProposerSandboxConfig(kind="docker", docker_image="memo-proposer:test"),
    )

    assert result.command[:4] == ("docker", "run", "--rm", "-i")
    assert "memo-proposer:test" in result.command
    assert result.command[result.command.index("memo-proposer:test") + 1] == "claude"
    assert result.tool_access["files_read"] == {"src/a.py": {"reads": 1, "lines": 0}}


def test_extract_kimi_result_and_tool_access_from_stream_json():
    raw_stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_read_1",
                                "name": "Read",
                                "input": {"file_path": "/repo/src/a.py"},
                            }
                        ],
                        "usage": {"input_tokens": 7, "output_tokens": 2},
                    },
                }
            ),
            json.dumps({"type": "result", "result": "done"}),
        ]
    )

    text, usage = _extract_kimi_result(raw_stdout)
    access = _extract_kimi_tool_access(raw_stdout, cwd="/repo")

    assert text == "done"
    assert usage == {"usage": {"input_tokens": 7, "output_tokens": 2}}
    assert access["files_read"] == {"src/a.py": {"reads": 1, "lines": 0}}


def test_run_kimi_prompt_prefers_claude_kimi_and_records_cost_and_tool_access(
    tmp_path, monkeypatch
):
    repo_dir = tmp_path / "repo"
    raw_stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_read_1",
                                "name": "Read",
                                "input": {"file_path": str(repo_dir / "src/a.py")},
                            }
                        ],
                    },
                }
            ),
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_read_1",
                                "content": "     1\u2192alpha\n     2\u2192beta",
                            }
                        ]
                    },
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "result": "done",
                    "total_cost_usd": 0.123,
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 3,
                    },
                }
            ),
        ]
    )
    calls = []

    def fake_which(name):
        return f"/bin/{name}" if name == "claude-kimi" else None

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=raw_stdout, stderr="")

    monkeypatch.setattr("memomemo.claude_runner.shutil.which", fake_which)
    monkeypatch.setattr("memomemo.claude_runner.subprocess.run", fake_run)

    result = run_kimi_prompt(
        "prompt",
        cwd=repo_dir,
        log_dir=tmp_path / "logs",
        name="iter_001",
        model="kimi-test",
    )

    assert result.command[0] == "claude-kimi"
    assert result.command[:2] == ("claude-kimi", "-p")
    assert "--tools" in result.command
    assert "--allowedTools" in result.command
    assert "--dangerously-skip-permissions" in result.command
    assert result.command[-2:] == ("--model", "kimi-test")
    assert calls == [
        (
            result.command,
            {
                "input": "prompt",
                "cwd": str(repo_dir),
                "text": True,
                "capture_output": True,
                "timeout": 2400,
            },
        )
    ]
    assert result.stdout == "done"
    assert result.usage == {
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 3,
        },
        "total_cost_usd": 0.123,
    }
    assert result.tool_access["tool_counts"] == {"Read": 1}
    assert result.tool_access["files_read"] == {"src/a.py": {"reads": 1, "lines": 2}}
    assert result.metrics["estimated_cost_usd"] == 0.123
    assert result.metrics["read_file_calls"] == 1
    meta = json.loads((tmp_path / "logs" / "iter_001" / "meta.json").read_text())
    assert meta["usage"] == result.usage
    assert meta["tool_access"]["files_read"] == {"src/a.py": {"reads": 1, "lines": 2}}
    assert meta["metrics"]["estimated_cost_usd"] == 0.123


def test_run_kimi_prompt_can_run_inside_docker_sandbox(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    raw_stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_read_1",
                                "name": "Read",
                                "input": {"file_path": "/workspace/src/a.py"},
                            }
                        ]
                    },
                }
            ),
            json.dumps({"type": "result", "result": "done"}),
        ]
    )

    def fake_which(name):
        return "/bin/docker" if name == "docker" else None

    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout=raw_stdout, stderr="")

    monkeypatch.setattr("memomemo.claude_runner.shutil.which", fake_which)
    monkeypatch.setattr("memomemo.claude_runner.subprocess.run", fake_run)

    result = run_kimi_prompt(
        "prompt",
        cwd=repo_dir,
        log_dir=tmp_path / "logs",
        name="iter_001",
        sandbox=ProposerSandboxConfig(kind="docker", docker_image="memo-proposer:test"),
    )

    assert result.command[:4] == ("docker", "run", "--rm", "-i")
    assert "memo-proposer:test" in result.command
    assert result.command[result.command.index("memo-proposer:test") + 1] == "claude-kimi"
    assert result.tool_access["files_read"] == {"src/a.py": {"reads": 1, "lines": 0}}


def test_extract_kimi_result_from_role_content_json():
    raw_stdout = "\n".join(
        [
            json.dumps(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "think", "think": "hidden"},
                        {"type": "text", "text": "OK"},
                    ],
                }
            ),
            "To resume this session: kimi -r session-id",
        ]
    )

    text, usage = _extract_kimi_result(raw_stdout)

    assert text == "OK"
    assert usage is None


def test_extract_kimi_tool_access_from_role_tool_calls():
    raw_stdout = "\n".join(
        [
            json.dumps(
                {
                    "role": "assistant",
                    "content": [{"type": "think", "think": "hidden"}],
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "tool_read_1",
                            "function": {
                                "name": "ReadFile",
                                "arguments": json.dumps({"path": "note.txt"}),
                            },
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "role": "tool",
                    "tool_call_id": "tool_read_1",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "<system>1 lines read from file starting from line 1.</system>"
                            ),
                        },
                        {"type": "text", "text": "     1\talpha beta gamma\n"},
                    ],
                }
            ),
        ]
    )

    access = _extract_kimi_tool_access(raw_stdout, cwd="/repo")

    assert access["tool_counts"] == {"ReadFile": 1}
    assert access["files_read"] == {"note.txt": {"reads": 1, "lines": 1}}


def test_extract_kimi_wire_usage_from_status_updates():
    wire_jsonl = "\n".join(
        [
            json.dumps(
                {
                    "message": {
                        "type": "StatusUpdate",
                        "payload": {
                            "context_tokens": 10512,
                            "max_context_tokens": 262144,
                            "token_usage": {
                                "input_other": 1296,
                                "output": 52,
                                "input_cache_read": 9216,
                                "input_cache_creation": 0,
                            },
                        },
                    }
                }
            ),
            json.dumps(
                {
                    "message": {
                        "type": "StatusUpdate",
                        "payload": {
                            "context_tokens": 10618,
                            "max_context_tokens": 262144,
                            "token_usage": {
                                "input_other": 122,
                                "output": 43,
                                "input_cache_read": 10496,
                                "input_cache_creation": 0,
                            },
                        },
                    }
                }
            ),
        ]
    )

    usage = _extract_kimi_wire_usage(wire_jsonl)

    assert usage == {
        "usage": {
            "input_tokens": 1418,
            "output_tokens": 95,
            "cache_read_input_tokens": 19712,
            "cache_creation_input_tokens": 0,
        },
        "context_tokens": 10618,
        "max_context_tokens": 262144,
    }


def test_extract_stream_result_uses_assistant_usage_when_result_event_missing():
    raw_stdout = json.dumps(
        {
            "type": "assistant",
            "message": {
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 3,
                    "cache_read_input_tokens": 5,
                },
                "content": [{"type": "text", "text": "partial"}],
            },
        }
    )

    text, usage = _extract_stream_result(raw_stdout)

    assert text == "partial"
    assert usage == {
        "usage": {
            "input_tokens": 11,
            "output_tokens": 3,
            "cache_read_input_tokens": 5,
        }
    }
