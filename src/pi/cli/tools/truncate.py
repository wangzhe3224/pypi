"""Shared truncation utilities for CLI tool outputs.

This module provides simple, stdlib-based truncation helpers that mirror
the behavior of the reference TypeScript utilities used in the project.
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50KB
GREP_MAX_LINE_LENGTH = 500  # Max chars per grep match line


@dataclass
class TruncationResult:
    truncated: bool
    original_lines: int
    original_bytes: int
    truncated_lines: int | None = None
    truncated_bytes: int | None = None


def _bytes_of(text: str) -> int:
    return len(text.encode("utf-8"))


def _truncate_string_to_bytes_from_end(s: str, max_bytes: int) -> str:
    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return s
    # Start from the end and find a valid UTF-8 boundary
    start = len(b) - max_bytes
    # Move start forward to the first byte that starts a UTF-8 char
    while start < len(b) and (b[start] & 0xC0) == 0x80:
        start += 1
    return b[start:].decode("utf-8")


def truncate_head(
    text: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES
) -> tuple[str, TruncationResult | None]:
    """Truncate content from the head (keep the beginning of the text).

    Returns a tuple of (truncated_text, truncation_info). If no truncation
    occurred, truncation_info is None.
    """
    total_bytes = _bytes_of(text)
    lines = text.split("\n")
    total_lines = len(lines)

    # Early exit: nothing to truncate
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return text, None

    # If the first line alone exceeds the byte limit, return empty content
    if lines:
        first_line_bytes = _bytes_of(lines[0])
        if first_line_bytes > max_bytes:
            return "", TruncationResult(
                truncated=True,
                original_lines=total_lines,
                original_bytes=total_bytes,
                truncated_lines=0,
                truncated_bytes=0,
            )

    output_lines: list[str] = []
    output_bytes = 0
    for i, line in enumerate(lines):
        if i >= max_lines:
            break
        line_bytes = len(line.encode("utf-8")) + (1 if i > 0 else 0)  # newline after previous line
        if output_bytes + line_bytes > max_bytes:
            break
        output_lines.append(line)
        output_bytes += line_bytes

    # If we stopped due to hitting the line limit, it's considered truncated by lines
    if len(output_lines) >= max_lines:
        truncated_lines = len(output_lines)
        truncated = True
    else:
        truncated_lines = len(output_lines)
        truncated = bool(truncated_lines < total_lines or output_bytes < total_bytes)

    truncated_text = "\n".join(output_lines)
    return truncated_text, TruncationResult(
        truncated=truncated,
        original_lines=total_lines,
        original_bytes=total_bytes,
        truncated_lines=truncated_lines,
        truncated_bytes=output_bytes,
    )


def truncate_tail(
    text: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES
) -> tuple[str, TruncationResult | None]:
    """Truncate content from the tail (keep the ending of the text).

    Returns a tuple of (truncated_text, truncation_info). If no truncation
    occurred, truncation_info is None.
    """
    total_bytes = _bytes_of(text)
    lines = text.split("\n")
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return text, None

    output_lines: list[str] = []
    output_bytes = 0
    # Walk from the end backwards
    for i in range(len(lines) - 1, -1, -1):
        if len(output_lines) >= max_lines:
            break
        line = lines[i]
        line_bytes = len(line.encode("utf-8")) + (
            1 if len(output_lines) > 0 else 0
        )  # newline before this line (since we build backwards)
        if output_bytes + line_bytes > max_bytes:
            if len(output_lines) == 0:
                # Need to partially truncate this line from its end
                truncated_line = _truncate_string_to_bytes_from_end(line, max_bytes)
                output_lines.insert(0, truncated_line)
                output_bytes = len(truncated_line.encode("utf-8"))
            break
        output_lines.insert(0, line)
        output_bytes += line_bytes

    truncated_text = "\n".join(output_lines)
    # Compute final bytes for the output
    final_output_bytes = len(truncated_text.encode("utf-8"))
    # Determine number of lines in the output
    output_lines_count = len(output_lines)
    return truncated_text, TruncationResult(
        truncated=True,
        original_lines=total_lines,
        original_bytes=total_bytes,
        truncated_lines=output_lines_count,
        truncated_bytes=final_output_bytes,
    )


def truncate_line(line: str, max_chars: int = GREP_MAX_LINE_LENGTH) -> str:
    """Truncate a single line to at most max_chars characters.

    Appends a suffix to indicate truncation when needed.
    """
    if len(line) <= max_chars:
        return line
    return f"{line[:max_chars]}... [truncated]"
