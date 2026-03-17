from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pi.agent.tools import tool

DEFAULT_LIMIT = 1000
DEFAULT_MAX_BYTES = 50 * 1024


@tool
def find(
    pattern: str,
    path: str = ".",
    limit: int = DEFAULT_LIMIT,
) -> str:
    """Find files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g., "*.py", "**/*.json", "src/**/*.ts")
        path: Directory to search (default: current directory)
        limit: Maximum number of results (default: 1000)
    """
    search_path = Path(path).expanduser().resolve()

    if not search_path.exists():
        return f"Error: Path not found: {path}"

    if not search_path.is_dir():
        return f"Error: Not a directory: {path}"

    fd_available = _check_fd_available()

    if fd_available:
        return _run_fd(pattern, search_path, limit)
    return _run_glob(pattern, search_path, limit)


def _check_fd_available() -> bool:
    try:
        result = subprocess.run(
            ["fd", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run_fd(pattern: str, search_path: Path, limit: int) -> str:
    args = [
        "--glob",
        "--color=never",
        "--hidden",
        "--max-results",
        str(limit),
        pattern,
        str(search_path),
    ]

    gitignore_files = _find_gitignores(search_path)
    for gi in gitignore_files[:10]:
        args.extend(["--ignore-file", str(gi)])

    try:
        result = subprocess.run(
            ["fd"] + args,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "Error: Search timed out"
    except FileNotFoundError:
        return _run_glob(pattern, search_path, limit)

    if not output:
        return "No files found matching pattern"

    lines = output.split("\n")
    relativized = _relativize_paths(lines, search_path)

    return _format_output(relativized, limit)


def _run_glob(pattern: str, search_path: Path, limit: int) -> str:
    try:
        matches = list(search_path.rglob(pattern))[:limit]
    except OSError as e:
        return f"Error: {e}"

    if not matches:
        return "No files found matching pattern"

    relativized = [str(m.relative_to(search_path)) for m in matches]
    return _format_output(relativized, limit)


def _find_gitignores(search_path: Path) -> list[Path]:
    gitignore_files = []
    skip_dirs = {"node_modules", ".git", "__pycache__", "venv", ".venv", ".tox", "dist", "build"}

    for root, dirs, files in os.walk(search_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        if ".gitignore" in files:
            gitignore_files.append(Path(root) / ".gitignore")

    return gitignore_files


def _relativize_paths(lines: list[str], search_path: Path) -> list[str]:
    relativized = []
    search_str = str(search_path)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(search_str):
            relativized.append(line[len(search_str) + 1 :])
        else:
            try:
                relativized.append(str(Path(line).relative_to(search_path)))
            except ValueError:
                relativized.append(line)
    return relativized


def _format_output(lines: list[str], limit: int) -> str:
    output = "\n".join(lines)
    notices = []

    if len(lines) >= limit:
        notices.append(f"{limit} results limit reached")

    if len(output.encode()) > DEFAULT_MAX_BYTES:
        output = output[:DEFAULT_MAX_BYTES]
        notices.append(f"{DEFAULT_MAX_BYTES // 1024}KB limit reached")

    if notices:
        output += f"\n\n[{'. '.join(notices)}]"

    return output
