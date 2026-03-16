"""Path resolution utilities for CLI tools."""
from __future__ import annotations

from pathlib import Path


def resolve_to_cwd(path: str, cwd: str) -> str:
    """Resolve a path relative to a given cwd, with '~' expansion.

    - If path is absolute, returns its resolved form.
    - If path is relative, it's interpreted relative to cwd.
    - Returns a string path (POSIX-like on unix-like systems).
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path(cwd) / p
    return str(p.resolve())


def ensure_directory(path: str) -> None:
    """Create parent directories for the given path if they don't exist."""
    p = Path(path).expanduser()
    # If the path has a suffix, treat it as a file path and create its parent dir.
    target_dir = p.parent if p.suffix else p
    target_dir.mkdir(parents=True, exist_ok=True)
