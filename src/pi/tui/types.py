"""Core TUI types for pi.tui."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol


class Component(Protocol):
    def render(self, width: int) -> list[str]:
        ...

    def invalidate(self) -> None:
        ...


class EventType(StrEnum):
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    MESSAGE_START = "message_start"
    MESSAGE_UPDATE = "message_update"
    MESSAGE_END = "message_end"
    TOOL_EXECUTION_START = "tool_execution_start"
    TOOL_EXECUTION_UPDATE = "tool_execution_update"
    TOOL_EXECUTION_END = "tool_execution_end"
