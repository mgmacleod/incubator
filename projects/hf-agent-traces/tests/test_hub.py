"""Tests for the HF hub wrapper.

We don't hit the network. We import ``huggingface_hub`` symbols and monkeypatch
the bits we use (``HfApi.create_commit`` etc.) so we can verify our wrapper
constructs the right operations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hf_agent_traces import agents, hub


pytest.importorskip("huggingface_hub")


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_AGENT_TRACES_HOME", str(tmp_path))
    return tmp_path


def _make_session(fake_home: Path) -> agents.Session:
    p = fake_home / ".claude/projects/myproj/sess-1.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"hi": 1}) + "\n")
    return agents.discover("claude-code")[0]


def test_upload_sessions_builds_commit(fake_home, monkeypatch):
    session = _make_session(fake_home)

    captured = {}

    class FakeApi:
        def __init__(self, token=None):
            captured["token"] = token

        def create_repo(self, **kw):
            captured["create_repo"] = kw

        def create_commit(self, **kw):
            captured["commit"] = kw

    monkeypatch.setattr(hub, "_api", lambda token=None: FakeApi(token=token))

    result = hub.upload_sessions(
        [session],
        "me/agent-traces",
        token="hf_xxx",
        private=True,
    )
    assert result.uploaded == ["claude-code/myproj/sess-1.jsonl"]
    assert captured["create_repo"]["repo_id"] == "me/agent-traces"
    assert captured["create_repo"]["repo_type"] == "dataset"
    assert captured["create_repo"]["private"] is True

    ops = captured["commit"]["operations"]
    assert len(ops) == 1
    assert ops[0].path_in_repo == "claude-code/myproj/sess-1.jsonl"


def test_upload_sessions_with_prefix(fake_home, monkeypatch):
    session = _make_session(fake_home)

    class FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kw):
            pass

        def create_commit(self, **kw):
            self.last = kw

    fake = FakeApi()
    monkeypatch.setattr(hub, "_api", lambda token=None: fake)
    hub.upload_sessions([session], "me/x", path_prefix="batch-2026-05/")
    assert fake.last["operations"][0].path_in_repo == (
        "batch-2026-05/claude-code/myproj/sess-1.jsonl"
    )


def test_upload_empty_is_noop(monkeypatch):
    called = {"n": 0}

    class FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kw):
            called["n"] += 1

        def create_commit(self, **kw):
            called["n"] += 1

    monkeypatch.setattr(hub, "_api", lambda token=None: FakeApi())
    out = hub.upload_sessions([], "me/x")
    assert out.uploaded == []
    assert called["n"] == 0


def test_download_traces_passes_agent_filter(monkeypatch, tmp_path):
    captured = {}

    def fake_snapshot_download(**kw):
        captured.update(kw)
        return str(tmp_path / "snap")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    hub.download_traces("me/x", tmp_path, agent="codex")
    assert captured["allow_patterns"] == ["codex/**"]
    assert captured["repo_type"] == "dataset"
