import json

from memomemo.claude_runner import _extract_tool_access


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
