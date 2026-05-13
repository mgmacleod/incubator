"""Command-line interface: ``python -m hf_agent_traces ...``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from hf_agent_traces.agents import (
    SUPPORTED_AGENTS,
    AgentKind,
    discover,
    discover_all,
)
from hf_agent_traces.hub import download_traces, upload_sessions


def _print_session_table(sessions, json_out: bool) -> None:
    if json_out:
        json.dump(
            [
                {
                    "agent": s.agent,
                    "session_id": s.session_id,
                    "project": s.project,
                    "path": str(s.path),
                    "size_bytes": s.size_bytes,
                    "modified": s.modified.isoformat(),
                }
                for s in sessions
            ],
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return

    if not sessions:
        print("(no sessions found)")
        return
    print(f"{'AGENT':<13} {'PROJECT':<32} {'SESSION':<40} {'SIZE':>10}  MODIFIED")
    for s in sessions:
        proj = (s.project or "")[:32]
        sid = s.session_id[:40]
        print(
            f"{s.agent:<13} {proj:<32} {sid:<40} {s.size_bytes:>10}  "
            f"{s.modified.isoformat(timespec='seconds')}"
        )


def cmd_list(args: argparse.Namespace) -> int:
    if args.agent:
        sessions = discover(args.agent)
    else:
        grouped = discover_all()
        sessions = [s for items in grouped.values() for s in items]
    sessions.sort(key=lambda s: s.mtime, reverse=True)
    if args.limit:
        sessions = sessions[: args.limit]
    _print_session_table(sessions, args.json)
    return 0


def _filter_sessions(sessions, since_mtime: float | None, project: str | None):
    out = sessions
    if since_mtime is not None:
        out = [s for s in out if s.mtime >= since_mtime]
    if project:
        out = [s for s in out if (s.project or "") == project]
    return out


def cmd_upload(args: argparse.Namespace) -> int:
    if args.agent:
        sessions = discover(args.agent)
    else:
        sessions = [s for items in discover_all().values() for s in items]

    sessions = _filter_sessions(sessions, args.since_mtime, args.project)
    if args.limit:
        sessions = sessions[: args.limit]

    if not sessions:
        print("No sessions matched - nothing to upload.")
        return 0

    if args.dry_run:
        print(f"Would upload {len(sessions)} session(s) to {args.repo}:")
        for s in sessions:
            print(f"  {s.relative_upload_path()}  ({s.size_bytes} bytes)")
        return 0

    token = args.token or os.environ.get("HF_TOKEN")
    result = upload_sessions(
        sessions,
        args.repo,
        token=token,
        private=not args.public,
        create_if_missing=not args.no_create,
        path_prefix=args.prefix or "",
        commit_message=args.message,
    )
    print(f"Uploaded {len(result.uploaded)} file(s) to {result.repo_id}")
    for r in result.uploaded:
        print(f"  + {r}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    token = args.token or os.environ.get("HF_TOKEN")
    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    local = download_traces(
        args.repo,
        dest=dest,
        token=token,
        agent=args.agent,
        revision=args.revision,
    )
    print(f"Downloaded to {local}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hf-agent-traces",
        description="Discover and sync local agent transcripts with Hugging Face.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List local sessions")
    p_list.add_argument("--agent", choices=SUPPORTED_AGENTS)
    p_list.add_argument("--limit", type=int, default=0)
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_up = sub.add_parser("upload", help="Upload sessions to a HF dataset repo")
    p_up.add_argument("repo", help="Target repo id, e.g. user/my-agent-traces")
    p_up.add_argument("--agent", choices=SUPPORTED_AGENTS)
    p_up.add_argument("--project", help="Filter to sessions whose project matches")
    p_up.add_argument(
        "--since-mtime",
        type=float,
        dest="since_mtime",
        help="Only include sessions modified at/after this unix timestamp",
    )
    p_up.add_argument("--limit", type=int, default=0)
    p_up.add_argument("--prefix", help="Path prefix inside the dataset repo")
    p_up.add_argument("--message", "-m", help="Commit message")
    p_up.add_argument("--public", action="store_true", help="Create repo as public")
    p_up.add_argument(
        "--no-create",
        action="store_true",
        help="Do not create the repo if it is missing",
    )
    p_up.add_argument("--token", help="HF token (or set HF_TOKEN env var)")
    p_up.add_argument("--dry-run", action="store_true")
    p_up.set_defaults(func=cmd_upload)

    p_dl = sub.add_parser("download", help="Download a trace dataset")
    p_dl.add_argument("repo")
    p_dl.add_argument("dest")
    p_dl.add_argument("--agent", choices=SUPPORTED_AGENTS)
    p_dl.add_argument("--revision")
    p_dl.add_argument("--token")
    p_dl.set_defaults(func=cmd_download)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
