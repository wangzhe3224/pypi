"""Glob tool for finding files matching a pattern."""

from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
from pathlib import Path

from pi.agent.tools import tool

DEFAULT_LIMIT = 1000


def _run_fd(
    pattern: str,
    search_path: Path,
    limit: int,
) -> list[str] | None:
    """Run fd command and return matching paths, or None if fd not available."""
    fd_path = shutil.which("fd")
    if not fd_path:
        return None

    args: list[str] = [
        "--glob",
        "--color=never",
        "--hidden",
        "--max-results",
        str(limit),
    ]

    gitignore_files: set[str] = set()
    root_gitignore = search_path / ".gitignore"
    if root_gitignore.exists():
        gitignore_files.add(str(root_gitignore))

    try:
        for root, dirs, files in os.walk(search_path):
            dirs[:] = [
                d for d in dirs if d not in ("node_modules", ".git", "__pycache__", "venv", ".venv")
            ]
            if ".gitignore" in files:
                gitignore_files.add(str(Path(root) / ".gitignore"))
    except OSError:
        pass

    for gitignore_path in gitignore_files:
        args.extend(["--ignore-file", gitignore_path])

    args.extend([pattern, str(search_path)])

    try:
        result = subprocess.run(
            [fd_path, *args],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None

    if result.returncode not in (0, 1):
        return None

    output = result.stdout.strip()
    if not output:
        return []

    lines = output.split("\n")
    return [line.rstrip("\r").strip() for line in lines if line.strip()]


def _run_pathlib_glob(
    pattern: str,
    search_path: Path,
    limit: int,
) -> list[str]:
    """Fallback glob implementation using pathlib."""
    matches: list[tuple[str, float]] = []  # (relative_path, mtime)

    try:
        if "**" in pattern:
            glob_pattern = pattern.replace("**/", "").lstrip("*")
            for match in search_path.rglob(glob_pattern):
                try:
                    mtime = match.stat().st_mtime
                    rel_path = str(match.relative_to(search_path)).replace("\\", "/")
                    if match.is_dir():
                        rel_path += "/"
                    matches.append((rel_path, mtime))
                except OSError:
                    continue
        else:
            for root, dirs, files in os.walk(search_path):
                dirs[:] = [
                    d
                    for d in dirs
                    if not d.startswith(".")
                    and d not in ("node_modules", "__pycache__", ".git", "venv", ".venv")
                ]
                for filename in files:
                    if fnmatch.fnmatch(filename, pattern):
                        file_path = Path(root) / filename
                        try:
                            mtime = file_path.stat().st_mtime
                            rel_path = str(file_path.relative_to(search_path)).replace("\\", "/")
                            matches.append((rel_path, mtime))
                        except OSError:
                            continue

        matches.sort(key=lambda x: x[1], reverse=True)
        return [path for path, _ in matches[:limit]]

    except Exception:
        return []


def _format_relative_path(absolute_path: str, search_path: Path, had_trailing_slash: bool) -> str:
    """Format an absolute path as relative to search_path."""
    try:
        p = Path(absolute_path)
        relative = str(p.relative_to(search_path)).replace("\\", "/")
        if had_trailing_slash and not relative.endswith("/"):
            relative += "/"
        return relative
    except ValueError:
        return absolute_path


@tool
def glob(
    pattern: str,
    path: str = ".",
    limit: int = 1000,
) -> str:
    """Find files matching a pattern.

    Args:
        pattern: Glob pattern to match files (e.g., "**/*.py")
        path: Directory to search in (default: current directory)
        limit: Maximum number of results
    """
    try:
        search_path = Path(path).resolve()

        if not search_path.exists():
            return f"Path not found: {path}"

        if not search_path.is_dir():
            return f"Not a directory: {path}"

        effective_limit = min(limit, 10000)  # Cap at 10k to prevent runaway queries

        fd_results = _run_fd(pattern, search_path, effective_limit)

        if fd_results is not None:
            if not fd_results:
                return "No files found matching pattern"

            paths_with_mtime: list[tuple[str, float]] = []
            for abs_path in fd_results:
                try:
                    had_trailing_slash = abs_path.endswith("/") or abs_path.endswith("\\")
                    mtime = Path(abs_path).stat().st_mtime
                    rel_path = _format_relative_path(abs_path, search_path, had_trailing_slash)
                    paths_with_mtime.append((rel_path, mtime))
                except OSError:
                    had_trailing_slash = abs_path.endswith("/") or abs_path.endswith("\\")
                    rel_path = _format_relative_path(abs_path, search_path, had_trailing_slash)
                    paths_with_mtime.append((rel_path, 0))

            paths_with_mtime.sort(key=lambda x: x[1], reverse=True)
            results = [p for p, _ in paths_with_mtime[:effective_limit]]
        else:
            results = _run_pathlib_glob(pattern, search_path, effective_limit)
            if not results:
                return "No files found matching pattern"

        result_limit_reached = len(results) >= effective_limit
        output = "\n".join(results)

        notices: list[str] = []
        if result_limit_reached:
            notices.append(
                f"{effective_limit} results limit reached. Use limit={effective_limit * 2} for more, or refine pattern"
            )

        if notices:
            output += f"\n\n[{'. '.join(notices)}]"

        return output

    except Exception as e:
        return f"Error: {e}"
