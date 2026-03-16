"""Grep tool for searching file contents."""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pi.agent.tools import tool
from pi.cli.tools.truncate import GREP_MAX_LINE_LENGTH, truncate_line


def _is_binary_file(file_path: Path) -> bool:
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(8192)
            if b"\x00" in chunk:
                return True
            text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7F})
            non_text = sum(1 for byte in chunk if byte not in text_chars)
            if len(chunk) > 0 and non_text / len(chunk) > 0.3:
                return True
    except OSError:
        return True
    return False


def _format_relative_path(file_path: Path, base_path: Path, is_directory: bool) -> str:
    if is_directory:
        try:
            relative = file_path.relative_to(base_path)
            return str(relative).replace("\\", "/")
        except ValueError:
            pass
    return file_path.name


def _run_ripgrep(
    pattern: str,
    search_path: Path,
    glob_pattern: str | None,
    ignore_case: bool,
    literal: bool,
    context: int,
    limit: int,
) -> str | None:
    rg_path = shutil.which("rg")
    if not rg_path:
        return None

    is_directory = search_path.is_dir()

    args = [
        "--json",
        "--line-number",
        "--color=never",
        "--hidden",
    ]

    if ignore_case:
        args.append("--ignore-case")

    if literal:
        args.append("--fixed-strings")

    if glob_pattern:
        args.extend(["--glob", glob_pattern])

    args.append(pattern)
    args.append(str(search_path))

    try:
        result = subprocess.run(
            [rg_path, *args],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return f"Error running ripgrep: {e}"

    if result.returncode not in (0, 1):
        return result.stderr.strip() or f"ripgrep exited with code {result.returncode}"

    matches: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if event.get("type") == "match":
            data = event.get("data", {})
            file_path = data.get("path", {}).get("text", "")
            line_number = data.get("line_number")
            if file_path and isinstance(line_number, int):
                matches.append({"file_path": file_path, "line_number": line_number})

    if not matches:
        return "No matches found"

    match_limit_reached = len(matches) >= limit
    matches = matches[:limit]

    file_cache: dict[str, list[str]] = {}
    output_lines: list[str] = []
    lines_truncated = False

    def get_file_lines(fp: str) -> list[str]:
        if fp not in file_cache:
            try:
                with open(fp, encoding="utf-8", errors="replace") as f:
                    content = f.read()
                file_cache[fp] = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            except OSError:
                file_cache[fp] = []
        return file_cache[fp]

    for match in matches:
        file_path = Path(match["file_path"])
        line_number = match["line_number"]
        relative_path = _format_relative_path(file_path, search_path, is_directory)

        lines = get_file_lines(str(file_path))
        if not lines:
            output_lines.append(f"{relative_path}:{line_number}: (unable to read file)")
            continue

        context_start = max(1, line_number - context) if context > 0 else line_number
        context_end = min(len(lines), line_number + context) if context > 0 else line_number

        for current_line in range(context_start, context_end + 1):
            line_text = lines[current_line - 1] if current_line <= len(lines) else ""
            sanitized = line_text.replace("\r", "")
            truncated = truncate_line(sanitized, GREP_MAX_LINE_LENGTH)
            if len(truncated) != len(sanitized):
                lines_truncated = True

            is_match_line = current_line == line_number
            if is_match_line:
                output_lines.append(f"{relative_path}:{current_line}: {truncated}")
            else:
                output_lines.append(f"{relative_path}-{current_line}- {truncated}")

    output = "\n".join(output_lines)
    notices: list[str] = []

    if match_limit_reached:
        notices.append(
            f"{limit} matches limit reached. Use limit={limit * 2} for more, or refine pattern"
        )

    if lines_truncated:
        notices.append(
            f"Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. Use read tool to see full lines"
        )

    if notices:
        output += f"\n\n[{'. '.join(notices)}]"

    return output


def _run_python_grep(
    pattern: str,
    search_path: Path,
    glob_pattern: str | None,
    ignore_case: bool,
    literal: bool,
    context: int,
    limit: int,
) -> str:
    is_directory = search_path.is_dir()

    flags = re.IGNORECASE if ignore_case else 0
    search_pattern = re.escape(pattern) if literal else pattern

    try:
        regex = re.compile(search_pattern, flags)
    except re.error as e:
        return f"Invalid regex pattern: {e}"

    files_to_search: list[Path] = []
    if is_directory:
        for root, dirs, files in os.walk(search_path):
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d not in ("node_modules", "__pycache__", ".git", "venv", ".venv")
            ]
            for filename in files:
                file_path = Path(root) / filename
                if glob_pattern and not fnmatch.fnmatch(filename, glob_pattern):
                    continue
                if _is_binary_file(file_path):
                    continue
                files_to_search.append(file_path)
    else:
        if search_path.exists() and not _is_binary_file(search_path):
            files_to_search.append(search_path)

    files_to_search.sort()

    matches: list[tuple[Path, int, list[str]]] = []

    for file_path in files_to_search:
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                lines = f.read().replace("\r\n", "\n").replace("\r", "\n").split("\n")
        except OSError:
            continue

        for i, line in enumerate(lines, start=1):
            if regex.search(line):
                matches.append((file_path, i, lines))
                if len(matches) >= limit:
                    break
        if len(matches) >= limit:
            break

    if not matches:
        return "No matches found"

    match_limit_reached = len(matches) >= limit
    output_lines: list[str] = []
    lines_truncated = False

    for file_path, line_number, all_lines in matches:
        relative_path = _format_relative_path(file_path, search_path, is_directory)

        context_start = max(1, line_number - context) if context > 0 else line_number
        context_end = min(len(all_lines), line_number + context) if context > 0 else line_number

        for current_line in range(context_start, context_end + 1):
            line_text = all_lines[current_line - 1] if current_line <= len(all_lines) else ""
            sanitized = line_text.replace("\r", "")
            truncated = truncate_line(sanitized, GREP_MAX_LINE_LENGTH)
            if len(truncated) != len(sanitized):
                lines_truncated = True

            is_match_line = current_line == line_number
            if is_match_line:
                output_lines.append(f"{relative_path}:{current_line}: {truncated}")
            else:
                output_lines.append(f"{relative_path}-{current_line}- {truncated}")

    output = "\n".join(output_lines)
    notices: list[str] = []

    if match_limit_reached:
        notices.append(
            f"{limit} matches limit reached. Use limit={limit * 2} for more, or refine pattern"
        )

    if lines_truncated:
        notices.append(
            f"Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. Use read tool to see full lines"
        )

    if notices:
        output += f"\n\n[{'. '.join(notices)}]"

    return output


@tool
def grep(
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    ignore_case: bool = False,
    literal: bool = False,
    context: int = 0,
    limit: int = 100,
) -> str:
    """Search for pattern in files.

    Args:
        pattern: The regex pattern to search for
        path: Directory or file to search in (default: current directory)
        glob: File pattern to filter (e.g., "*.py")
        ignore_case: Case-insensitive search
        literal: Treat pattern as literal string, not regex
        context: Number of context lines before and after matches
        limit: Maximum number of results
    """
    try:
        search_path = Path(path).resolve()

        if not search_path.exists():
            return f"Path not found: {path}"

        result = _run_ripgrep(
            pattern=pattern,
            search_path=search_path,
            glob_pattern=glob,
            ignore_case=ignore_case,
            literal=literal,
            context=context,
            limit=limit,
        )

        if result is None:
            return _run_python_grep(
                pattern=pattern,
                search_path=search_path,
                glob_pattern=glob,
                ignore_case=ignore_case,
                literal=literal,
                context=context,
                limit=limit,
            )

        return result

    except Exception as e:
        return f"Error: {e}"
