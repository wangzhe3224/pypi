"""Read tool for reading file contents."""
from __future__ import annotations

import base64
import os
from pathlib import Path

from pi.agent.tools import tool
from pi.cli.tools.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, truncate_head

IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def _detect_image_mime_type(file_path: str) -> str | None:
    ext = Path(file_path).suffix.lower()
    return IMAGE_MIME_TYPES.get(ext)


def _format_size(bytes_count: int) -> str:
    if bytes_count < 1024:
        return f"{bytes_count}B"
    if bytes_count < 1024 * 1024:
        return f"{bytes_count // 1024}KB"
    return f"{bytes_count // (1024 * 1024)}MB"


@tool
def read(file_path: str, offset: int = 0, limit: int = 2000) -> str:
    """Read file contents.

    Args:
        file_path: The absolute path to the file to read
        offset: The line number to start reading from (1-indexed)
        limit: Maximum number of lines to read
    """
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        return f"Error: File not found: {file_path}"

    if not path.is_file():
        return f"Error: Not a file: {file_path}"

    if not os.access(path, os.R_OK):
        return f"Error: File not readable: {file_path}"

    mime_type = _detect_image_mime_type(str(path))

    if mime_type:
        try:
            with open(path, "rb") as f:
                content = f.read()

            base64_data = base64.b64encode(content).decode("utf-8")
            file_size = len(content)

            return (
                f"Image file [{mime_type}]\n"
                f"Size: {_format_size(file_size)}\n"
                f"Base64 data:\n{base64_data}"
            )
        except Exception as e:
            return f"Error reading image file: {e}"

    try:
        with open(path, "rb") as f:
            raw_content = f.read()

        try:
            text = raw_content.decode("utf-8")
            if text.startswith("\ufeff"):
                text = text[1:]
        except UnicodeDecodeError:
            return f"Error: File is not valid UTF-8 text: {file_path}"

        all_lines = text.split("\n")
        total_lines = len(all_lines)

        start_line = max(0, offset - 1) if offset > 0 else 0
        start_line_display = start_line + 1

        if start_line >= total_lines:
            return f"Error: Offset {offset} is beyond end of file ({total_lines} lines total)"

        if limit is not None and limit > 0:
            end_line = min(start_line + limit, total_lines)
            selected_lines = all_lines[start_line:end_line]
            user_limited_lines = end_line - start_line
        else:
            selected_lines = all_lines[start_line:]
            user_limited_lines = None

        numbered_lines = []
        for i, line in enumerate(selected_lines):
            line_num = start_line + i + 1
            numbered_lines.append(f"{line_num}: {line}")

        selected_content = "\n".join(numbered_lines)

        truncated_content, truncation_result = truncate_head(
            selected_content, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES
        )

        if truncation_result and truncation_result.truncated:
            end_line_display = start_line_display + (truncation_result.truncated_lines or 0) - 1
            next_offset = end_line_display + 1

            output = truncated_content
            output += f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_lines} ({_format_size(DEFAULT_MAX_BYTES)} limit). Use offset={next_offset} to continue.]"
        elif user_limited_lines is not None and start_line + user_limited_lines < total_lines:
            remaining = total_lines - (start_line + user_limited_lines)
            next_offset = start_line + user_limited_lines + 1

            output = truncated_content
            output += f"\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
        else:
            output = truncated_content

        return output

    except Exception as e:
        return f"Error reading file: {e}"
