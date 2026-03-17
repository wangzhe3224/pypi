"""Path configuration for the coding agent.

Based on pi-mono/packages/coding-agent/src/config.ts
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# App info
APP_NAME: Final = "pi"
VERSION: Final = "0.1.0"


def get_agent_dir() -> Path:
    """Get the main agent directory (~/.pi/agent/).

    Returns:
        Path to the agent directory.
    """
    return Path.home() / ".pi" / "agent"


def get_sessions_dir() -> Path:
    """Get the sessions directory (~/.pi/agent/sessions/).

    Returns:
        Path to the sessions directory.
    """
    return get_agent_dir() / "sessions"


def get_settings_path(project_path: Path | None = None, *, global_: bool = False) -> Path:
    """Get the settings file path.

    Args:
        project_path: Optional project directory path.
        global_: If True, return global settings path regardless of project.

    Returns:
        Path to the settings file.
    """
    if global_ or project_path is None:
        return get_agent_dir() / "settings.json"
    return project_path / ".pi" / "settings.json"


def get_session_path(session_id: str | None = None) -> Path:
    """Get the path to a session file.

    Args:
        session_id: Optional session ID. If None, returns default session path.

    Returns:
        Path to the session file.
    """
    sessions_dir = get_sessions_dir()
    sessions_dir.mkdir(parents=True, exist_ok=True)

    if session_id is None:
        return sessions_dir / "default.jsonl"
    return sessions_dir / f"{session_id}.jsonl"


def get_auth_path() -> Path:
    """Get the OAuth credentials storage path.

    Returns:
        Path to the auth file.
    """
    return get_agent_dir() / "auth.json"


def get_debug_log_path() -> Path:
    """Get the debug log file path.

    Returns:
        Path to the debug log.
    """
    return get_agent_dir() / "debug.log"


def ensure_agent_dir() -> None:
    """Ensure the agent directory structure exists."""
    get_agent_dir().mkdir(parents=True, exist_ok=True)
    get_sessions_dir().mkdir(parents=True, exist_ok=True)


def get_cwd() -> Path:
    """Get the current working directory.

    Returns:
        Current working directory as Path.
    """
    return Path.cwd()
