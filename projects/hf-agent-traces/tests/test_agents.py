"""Tests for the local-discovery layer.

These exercise the discovery code against a fake $HOME-like tree so we don't
depend on the developer's real ~/.claude or ~/.codex directories.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hf_agent_traces import agents


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_AGENT_TRACES_HOME", str(tmp_path))
    return tmp_path


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_discover_claude_code(fake_home):
    sess_file = fake_home / ".claude/projects/-home-user-foo/abc-123.jsonl"
    _write_jsonl(sess_file, [{"type": "user", "content": "hi"}])

    out = agents.discover("claude-code")
    assert len(out) == 1
    s = out[0]
    assert s.agent == "claude-code"
    assert s.session_id == "abc-123"
    assert s.project == "-home-user-foo"
    assert s.path == sess_file
    assert s.size_bytes > 0
    assert s.relative_upload_path() == "claude-code/-home-user-foo/abc-123.jsonl"


def test_discover_codex_both_buckets(fake_home):
    a = fake_home / ".codex/sessions/2026/05/01.jsonl"
    b = fake_home / ".codex/archived_sessions/old.jsonl"
    _write_jsonl(a, [{"role": "user"}])
    _write_jsonl(b, [{"role": "user"}])

    out = agents.discover("codex")
    projects = {s.project for s in out}
    assert projects == {"sessions", "archived_sessions"}
    assert all(s.agent == "codex" for s in out)


def test_discover_pi(fake_home):
    p = fake_home / ".pi/sessions/run-1.jsonl"
    _write_jsonl(p, [{"x": 1}])

    out = agents.discover("pi")
    assert len(out) == 1
    assert out[0].agent == "pi"
    assert out[0].relative_upload_path() == "pi/run-1.jsonl"


def test_discover_all_groups_by_agent(fake_home):
    _write_jsonl(fake_home / ".claude/projects/p/x.jsonl", [{}])
    _write_jsonl(fake_home / ".codex/sessions/y.jsonl", [{}])

    grouped = agents.discover_all()
    assert set(grouped) == set(agents.SUPPORTED_AGENTS)
    assert len(grouped["claude-code"]) == 1
    assert len(grouped["codex"]) == 1
    assert grouped["pi"] == []


def test_discover_unknown_agent_raises():
    with pytest.raises(ValueError):
        agents.discover("nope")  # type: ignore[arg-type]


def test_peek_returns_first_records(fake_home):
    p = fake_home / ".claude/projects/proj/s.jsonl"
    _write_jsonl(p, [{"i": 0}, {"i": 1}, {"i": 2}])
    sessions = agents.discover("claude-code")
    rows = agents.peek(sessions[0], n=2)
    assert rows == [{"i": 0}, {"i": 1}]


def test_line_count_optional(fake_home):
    p = fake_home / ".claude/projects/proj/s.jsonl"
    _write_jsonl(p, [{"i": 0}, {"i": 1}, {"i": 2}])

    without = agents.discover("claude-code")
    with_counts = agents.discover("claude-code", include_line_count=True)
    assert without[0].line_count is None
    assert with_counts[0].line_count == 3
