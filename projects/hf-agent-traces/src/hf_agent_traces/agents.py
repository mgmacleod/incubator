"""Discovery of local agent transcript files.

Hugging Face's Agent Trace Viewer auto-detects formats from supported agents,
so we leave the raw JSONL files untouched. This module just locates them on
disk and exposes some lightweight metadata.

Reference: https://huggingface.co/changelog/agent-trace-viewer
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Literal

AgentKind = Literal["claude-code", "codex", "pi"]

SUPPORTED_AGENTS: tuple[AgentKind, ...] = ("claude-code", "codex", "pi")


@dataclass(frozen=True)
class Session:
    """A single agent session transcript on disk."""

    agent: AgentKind
    path: Path
    session_id: str
    project: str | None = None
    size_bytes: int = 0
    mtime: float = 0.0
    line_count: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def modified(self) -> datetime:
        return datetime.fromtimestamp(self.mtime, tz=timezone.utc)

    def relative_upload_path(self) -> str:
        """Path used inside the HF dataset repo, namespaced by agent."""
        if self.project:
            return f"{self.agent}/{self.project}/{self.path.name}"
        return f"{self.agent}/{self.path.name}"


# ---------------------------------------------------------------------------
# Per-agent root locations
# ---------------------------------------------------------------------------

def _home() -> Path:
    return Path(os.environ.get("HF_AGENT_TRACES_HOME") or Path.home())


def _claude_root() -> Path:
    return _home() / ".claude" / "projects"


def _codex_roots() -> list[Path]:
    base = _home() / ".codex"
    return [base / "sessions", base / "archived_sessions"]


def _pi_root() -> Path:
    return _home() / ".pi" / "sessions"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _stat(path: Path) -> tuple[int, float]:
    st = path.stat()
    return st.st_size, st.st_mtime


def _count_lines(path: Path, limit: int | None = None) -> int:
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
            if limit is not None and n >= limit:
                break
    return n


def _claude_sessions(include_line_count: bool) -> Iterator[Session]:
    root = _claude_root()
    if not root.is_dir():
        return
    for jsonl in root.rglob("*.jsonl"):
        try:
            size, mtime = _stat(jsonl)
        except OSError:
            continue
        # Claude Code lays out as ~/.claude/projects/<encoded-cwd>/<session-id>.jsonl
        project = jsonl.parent.name
        yield Session(
            agent="claude-code",
            path=jsonl,
            session_id=jsonl.stem,
            project=project,
            size_bytes=size,
            mtime=mtime,
            line_count=_count_lines(jsonl) if include_line_count else None,
        )


def _codex_sessions(include_line_count: bool) -> Iterator[Session]:
    for root in _codex_roots():
        if not root.is_dir():
            continue
        bucket = root.name  # sessions | archived_sessions
        for jsonl in root.rglob("*.jsonl"):
            try:
                size, mtime = _stat(jsonl)
            except OSError:
                continue
            yield Session(
                agent="codex",
                path=jsonl,
                session_id=jsonl.stem,
                project=bucket,
                size_bytes=size,
                mtime=mtime,
                line_count=_count_lines(jsonl) if include_line_count else None,
            )


def _pi_sessions(include_line_count: bool) -> Iterator[Session]:
    root = _pi_root()
    if not root.is_dir():
        return
    for jsonl in root.rglob("*.jsonl"):
        try:
            size, mtime = _stat(jsonl)
        except OSError:
            continue
        yield Session(
            agent="pi",
            path=jsonl,
            session_id=jsonl.stem,
            size_bytes=size,
            mtime=mtime,
            line_count=_count_lines(jsonl) if include_line_count else None,
        )


_DISCOVERERS = {
    "claude-code": _claude_sessions,
    "codex": _codex_sessions,
    "pi": _pi_sessions,
}


def discover(
    agent: AgentKind,
    *,
    include_line_count: bool = False,
) -> list[Session]:
    """Return all sessions found locally for the given agent kind."""
    if agent not in _DISCOVERERS:
        raise ValueError(f"unsupported agent: {agent!r}. supported={SUPPORTED_AGENTS}")
    return sorted(
        _DISCOVERERS[agent](include_line_count),
        key=lambda s: s.mtime,
        reverse=True,
    )


def discover_all(
    *,
    agents: Iterable[AgentKind] | None = None,
    include_line_count: bool = False,
) -> dict[AgentKind, list[Session]]:
    """Discover sessions for every supported (or selected) agent."""
    selected = tuple(agents) if agents is not None else SUPPORTED_AGENTS
    return {a: discover(a, include_line_count=include_line_count) for a in selected}


def peek(session: Session, n: int = 1) -> list[dict]:
    """Read the first ``n`` JSONL records from a session for quick inspection."""
    out: list[dict] = []
    with session.path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
