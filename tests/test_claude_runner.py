import json

from memomemo.claude_runner import (
    _extract_session_metrics,
    _extract_stream_result,
    _extract_tool_access,
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
