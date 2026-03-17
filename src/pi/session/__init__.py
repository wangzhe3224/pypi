"""Session management and configuration.

This module provides:
- SessionManager: JSONL-based session persistence with tree structure
- SettingsManager: Global and project-level configuration
- AgentSession: High-level agent with session persistence
- Config: Path configuration for the agent

Example:
    from pi.session import AgentSession, SessionManager, SettingsManager

    # Create agent session with persistence
    session = AgentSession(
        session_path=Path("~/.pi/agent/sessions/default.jsonl"),
        project_path=Path.cwd(),
    )

    async for event in session.prompt("Hello"):
        print(event)
"""

from __future__ import annotations

from pi.session.config import (
    get_agent_dir,
    get_sessions_dir,
    get_settings_path,
    get_session_path,
)
from pi.session.manager import (
    SessionEntry,
    SessionEntryType,
    SessionHeader,
    SessionManager,
)
from pi.session.settings import AgentSettings, SettingsManager

__all__ = [
    # Config
    "get_agent_dir",
    "get_sessions_dir",
    "get_settings_path",
    "get_session_path",
    # Session Manager
    "SessionManager",
    "SessionEntry",
    "SessionEntryType",
    "SessionHeader",
    # Settings
    "SettingsManager",
    "AgentSettings",
]
