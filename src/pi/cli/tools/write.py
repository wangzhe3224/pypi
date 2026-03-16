"""Write tool for creating and overwriting files."""

from __future__ import annotations

from pi.agent.tools import tool
from pi.cli.tools.path_utils import ensure_directory


@tool
def write(file_path: str, content: str) -> str:
    """Write content to a file.

    Creates the file if it doesn't exist, overwrites if it does.
    Automatically creates parent directories.

    Args:
        file_path: The absolute path to the file to write
        content: The content to write to the file
    """
    try:
        ensure_directory(file_path)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Successfully wrote {len(content.encode('utf-8'))} bytes to {file_path}"
    except Exception as e:
        return f"Error writing to {file_path}: {e}"
