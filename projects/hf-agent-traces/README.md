# hf-agent-traces

A small Python library + CLI for working with [Hugging Face's Agent Trace Viewer](https://huggingface.co/changelog/agent-trace-viewer). It discovers transcript files written locally by supported coding agents and syncs them to and from a Hugging Face dataset repo. The Hub auto-detects the formats, so we upload the raw JSONL files untouched.

## Status
Active

## Description
The HF Agent Trace Viewer reads JSONL session files produced by agents like Claude Code, Codex, and Pi from a (typically private) HF dataset and renders them as browsable sessions / turns / tool calls. This project is the glue between your laptop and that dataset:

- `discover` — find session files written by supported agents
- `upload_sessions` — push them to a HF dataset repo
- `download_traces` — pull them back down
- a `hf-agent-traces` CLI for the same operations

## Supported agents
| Agent | Location |
| --- | --- |
| Claude Code | `~/.claude/projects/**/*.jsonl` |
| Codex | `~/.codex/{sessions,archived_sessions}/**/*.jsonl` |
| Pi | `~/.pi/sessions/**/*.jsonl` |

## Install
```bash
cd projects/hf-agent-traces
pip install -e .
```

Set `HF_TOKEN` (or pass `--token`) to authenticate against the Hub.

## Library usage
```python
from hf_agent_traces import discover, discover_all, upload_sessions, download_traces

# Find everything locally
groups = discover_all()
print({k: len(v) for k, v in groups.items()})

# Upload just your Claude Code sessions to a private dataset
sessions = discover("claude-code")
upload_sessions(sessions, repo_id="me/my-agent-traces", private=True)

# Pull a teammate's traces back down
download_traces("teammate/agent-traces", dest="./traces", agent="codex")
```

## CLI usage
```bash
# List local sessions (newest first)
hf-agent-traces list
hf-agent-traces list --agent codex --limit 20 --json

# Upload to a private dataset (creates it if missing)
hf-agent-traces upload me/my-agent-traces --agent claude-code

# Preview before uploading
hf-agent-traces upload me/my-agent-traces --agent claude-code --dry-run

# Download just one agent's slice of a dataset
hf-agent-traces download me/my-agent-traces ./traces --agent claude-code
```

## Repo layout produced
```
<repo>/
├── claude-code/<project>/<session-id>.jsonl
├── codex/sessions/<session-id>.jsonl
├── codex/archived_sessions/<session-id>.jsonl
└── pi/<session-id>.jsonl
```
This namespacing keeps multiple agents' transcripts cleanly separated inside a single dataset.

## Tests
```bash
pip install -e ".[test]"
pytest
```
Tests use a fake home directory (`HF_AGENT_TRACES_HOME`) and monkeypatch `huggingface_hub`, so they run offline.

## Notes
- File contents are uploaded as-is; the viewer handles parsing.
- Use `--since-mtime <unix-ts>` to do incremental uploads from cron / a stop hook.
- The library only depends on `huggingface_hub`.
