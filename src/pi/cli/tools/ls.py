from __future__ import annotations

from pathlib import Path

from pi.agent.tools import tool

DEFAULT_LS_LIMIT = 500
DEFAULT_MAX_BYTES = 50 * 1024


@tool
def ls(
    path: str = ".",
    limit: int = DEFAULT_LS_LIMIT,
) -> str:
    """List directory contents.

    Args:
        path: Directory to list (default: current directory)
        limit: Maximum number of entries (default: 500)
    """
    dir_path = Path(path).expanduser().resolve()

    if not dir_path.exists():
        return f"Error: Path not found: {path}"

    if not dir_path.is_dir():
        return f"Error: Not a directory: {path}"

    try:
        entries = list(dir_path.iterdir())
    except PermissionError:
        return f"Error: Permission denied: {path}"

    entries.sort(key=lambda x: x.name.lower())

    results = []
    entry_limit_reached = False

    for entry in entries:
        if len(results) >= limit:
            entry_limit_reached = True
            break

        try:
            suffix = "/" if entry.is_dir() else ""
            results.append(entry.name + suffix)
        except OSError:
            results.append(entry.name + " (unreadable)")

    if not results:
        return "(empty directory)"

    output = "\n".join(results)
    notices = []

    if entry_limit_reached:
        notices.append(f"{limit} entries limit reached. Use limit={limit * 2} for more")

    if len(output.encode()) > DEFAULT_MAX_BYTES:
        output = output[:DEFAULT_MAX_BYTES]
        notices.append(f"{DEFAULT_MAX_BYTES // 1024}KB limit reached")

    if notices:
        output += f"\n\n[{'. '.join(notices)}]"

    return output
