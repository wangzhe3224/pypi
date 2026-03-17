from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class SessionEntryType(str, Enum):
    MESSAGE = "message"
    THINKING_LEVEL_CHANGE = "thinking_level_change"
    MODEL_CHANGE = "model_change"
    COMPACTION = "compaction"
    BRANCH_SUMMARY = "branch_summary"
    CUSTOM = "custom"
    CUSTOM_MESSAGE = "custom_message"
    LABEL = "label"
    SESSION_INFO = "session_info"


class SessionHeader(BaseModel):
    version: int = 1
    cwd: str
    parent_session: str | None = None
    created_at: int = Field(default_factory=lambda: int(1000 * __import__("time").time()))


class SessionEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: SessionEntryType
    timestamp: int = Field(default_factory=lambda: int(1000 * __import__("time").time()))
    parent_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": True}
