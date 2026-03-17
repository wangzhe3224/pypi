from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from pi.cli.args import CLIArgs
from pi.cli.modes.print_mode import run_print_mode
from pi.cli.tools import bash, edit, glob, grep, read, write
from pi.session.agent_session import AgentSession
from pi.session.config import get_session_path, get_sessions_dir
from pi.tui import run_interactive


def parse_args(argv: list[str] | None = None) -> CLIArgs:
    parser = argparse.ArgumentParser(
        prog="pi",
        description="AI coding agent with unified LLM API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        help="The prompt to send to the agent",
    )

    parser.add_argument(
        "-s",
        "--session",
        metavar="ID",
        help="Session ID to load or create",
    )

    parser.add_argument(
        "-m",
        "--model",
        metavar="NAME",
        help="Model to use (e.g., claude-sonnet-4, gpt-4o)",
    )

    parser.add_argument(
        "--mode",
        choices=["print", "interactive"],
        default="print",
        help="Run mode (default: print)",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )

    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List available sessions and exit",
    )

    parser.add_argument(
        "-C",
        "--cwd",
        type=Path,
        metavar="DIR",
        help="Working directory (default: current directory)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args(argv)

    return CLIArgs(
        prompt=args.prompt,
        session=args.session,
        model=args.model,
        mode=args.mode,
        list_models=args.list_models,
        list_sessions=args.list_sessions,
        config_dir=None,
        cwd=args.cwd,
        verbose=args.verbose,
        help=False,
    )


def get_default_tools() -> list[Any]:
    return [bash, edit, glob, grep, read, write]


async def list_models() -> None:
    print("Available models:")
    print("  dummy              Dummy (for testing)")
    print("  claude-sonnet-4    Anthropic Claude Sonnet 4")
    print("  claude-3-5-sonnet  Anthropic Claude 3.5 Sonnet")
    print("  gpt-4o             OpenAI GPT-4o")
    print("  gpt-4o-mini        OpenAI GPT-4o Mini")
    print("  gemini-2.0-flash   Google Gemini 2.0 Flash")


async def list_sessions() -> None:
    sessions_dir = get_sessions_dir()
    if not sessions_dir.exists():
        print("No sessions found.")
        return

    sessions = list(sessions_dir.glob("*.jsonl"))
    if not sessions:
        print("No sessions found.")
        return

    print("Available sessions:")
    for session_file in sorted(sessions):
        from datetime import datetime

        stat = session_file.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  {session_file.stem:<30} {mtime}")


def get_session_path_from_args(args: CLIArgs) -> Path | None:
    if args.session:
        return get_session_path(args.session)
    return None


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.help:
        parse_args(["--help"])
        return 0

    if args.list_models:
        await list_models()
        return 0

    if args.list_sessions:
        await list_sessions()
        return 0

    cwd = args.cwd or Path.cwd()

    session_path = get_session_path_from_args(args)
    tools = get_default_tools()

    session = AgentSession(
        session_path=session_path,
        project_path=cwd,
        model=args.model,
        tools=tools,
    )

    if args.mode == "print":
        if not args.prompt:
            print("Error: --prompt required for print mode", file=sys.stderr)
            return 1
        return await run_print_mode(session, args.prompt)

    return await run_interactive(session)


def run() -> None:
    sys.exit(asyncio.run(main()))


if __name__ == "__main__":
    run()
