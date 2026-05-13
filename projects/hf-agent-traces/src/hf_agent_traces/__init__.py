"""hf_agent_traces - discover local agent transcripts and sync with Hugging Face."""

from hf_agent_traces.agents import (
    AgentKind,
    SUPPORTED_AGENTS,
    Session,
    discover,
    discover_all,
)
from hf_agent_traces.hub import (
    download_traces,
    upload_sessions,
)

__all__ = [
    "AgentKind",
    "SUPPORTED_AGENTS",
    "Session",
    "discover",
    "discover_all",
    "download_traces",
    "upload_sessions",
]

__version__ = "0.1.0"
