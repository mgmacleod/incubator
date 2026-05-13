"""Upload to and download from Hugging Face dataset repos.

Wraps ``huggingface_hub`` so callers can move agent transcripts between a
local machine and a (typically private) HF dataset that the Agent Trace
Viewer renders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from hf_agent_traces.agents import AgentKind, Session

REPO_TYPE = "dataset"


@dataclass
class UploadResult:
    repo_id: str
    uploaded: list[str]
    skipped: list[str]


def _require_hub():
    try:
        from huggingface_hub import HfApi  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from e


def _api(token: str | None):
    from huggingface_hub import HfApi

    return HfApi(token=token)


def ensure_dataset_repo(
    repo_id: str,
    *,
    private: bool = True,
    token: str | None = None,
    exist_ok: bool = True,
) -> str:
    """Create the dataset repo if it doesn't exist; return its repo_id."""
    _require_hub()
    api = _api(token)
    api.create_repo(
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        private=private,
        exist_ok=exist_ok,
    )
    return repo_id


def upload_sessions(
    sessions: Sequence[Session],
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = True,
    create_if_missing: bool = True,
    path_prefix: str = "",
    commit_message: str | None = None,
) -> UploadResult:
    """Upload a batch of session files to a HF dataset repo.

    Files are placed at ``<path_prefix>/<agent>/<project?>/<session>.jsonl``,
    matching the layout the Agent Trace Viewer expects.
    """
    _require_hub()
    from huggingface_hub import CommitOperationAdd

    if not sessions:
        return UploadResult(repo_id=repo_id, uploaded=[], skipped=[])

    api = _api(token)
    if create_if_missing:
        ensure_dataset_repo(repo_id, private=private, token=token)

    operations: list = []
    uploaded: list[str] = []
    for s in sessions:
        if not s.path.is_file():
            continue
        remote = s.relative_upload_path()
        if path_prefix:
            remote = f"{path_prefix.strip('/')}/{remote}"
        operations.append(
            CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(s.path))
        )
        uploaded.append(remote)

    if not operations:
        return UploadResult(repo_id=repo_id, uploaded=[], skipped=[])

    api.create_commit(
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        operations=operations,
        commit_message=commit_message
        or f"Upload {len(operations)} agent session trace(s)",
    )
    return UploadResult(repo_id=repo_id, uploaded=uploaded, skipped=[])


def download_traces(
    repo_id: str,
    dest: str | Path,
    *,
    token: str | None = None,
    agent: AgentKind | None = None,
    revision: str | None = None,
    allow_patterns: Iterable[str] | None = None,
) -> Path:
    """Download trace files from a HF dataset repo into ``dest``.

    If ``agent`` is given, only files under that agent's prefix are pulled.
    """
    _require_hub()
    from huggingface_hub import snapshot_download

    patterns: list[str] | None
    if allow_patterns is not None:
        patterns = list(allow_patterns)
    elif agent is not None:
        patterns = [f"{agent}/**"]
    else:
        patterns = None

    local = snapshot_download(
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        revision=revision,
        local_dir=str(Path(dest)),
        token=token,
        allow_patterns=patterns,
    )
    return Path(local)
