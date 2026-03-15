"""Agent types for pi.agent runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from pi.ai.types import (
        AssistantMessage,
        Context,
        ImageContent,
        Message,
        Model,
        StreamEvent,
        TextContent,
        Tool,
        ToolResultMessage,
    )

# Import message types at runtime for AgentMessage union
from pi.ai.types import AssistantMessage as _AssistantMessage
from pi.ai.types import ImageContent as _ImageContent
from pi.ai.types import TextContent as _TextContent
from pi.ai.types import ToolResultMessage as _ToolResultMessage
from pi.ai.types import UserMessage as _UserMessage

# =============================================================================
# Base Model
# =============================================================================


class _BaseModel(BaseModel):
    """Base model with alias support."""

    model_config = ConfigDict(populate_by_name=True)


# =============================================================================
# Thinking Level
# =============================================================================

ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]

# =============================================================================
# Agent Tool
# =============================================================================


class AgentToolResult(_BaseModel):
    """Result from tool execution."""

    content: list[_TextContent | _ImageContent]
    details: Any = None


# Callback for streaming tool execution updates
AgentToolUpdateCallback = Callable[[AgentToolResult], None]


class AgentTool(_BaseModel):
    """A tool that can be called by the agent.

    Extends Tool with execution capability.
    """

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    label: str = ""  # Human-readable label for UI

    # Execute function signature:
    # async def execute(
    #     tool_call_id: str,
    #     params: dict[str, Any],
    #     signal: asyncio.Event | None = None,
    #     on_update: AgentToolUpdateCallback | None = None,
    # ) -> AgentToolResult
    execute: Callable[..., Coroutine[Any, Any, AgentToolResult]] | None = Field(
        default=None, exclude=True
    )

    def to_llm_tool(self) -> Tool:
        """Convert to LLM-compatible Tool."""
        from pi.ai.types import Tool, ToolParameter

        return Tool(
            name=self.name,
            description=self.description,
            parameters=ToolParameter(**self.parameters) if self.parameters else ToolParameter(),
        )


# =============================================================================
# Agent Message
# =============================================================================

# AgentMessage is a union of standard LLM messages.
# Apps can extend this via custom message types.
AgentMessage = _UserMessage | _AssistantMessage | _ToolResultMessage
# =============================================================================
# Agent Context & State
# =============================================================================


@dataclass
class AgentContext:
    """Agent context for conversation."""

    system_prompt: str = ""
    messages: list[AgentMessage] = field(default_factory=list)
    tools: list[AgentTool] | None = None

    def to_llm_context(self) -> Context:
        """Convert to LLM-compatible Context."""
        from pi.ai.types import Context

        tools = [t.to_llm_tool() for t in self.tools] if self.tools else None
        return Context(
            system_prompt=self.system_prompt,
            messages=self.messages,
            tools=tools,
        )


@dataclass
class AgentState:
    """Full agent state including streaming status."""

    system_prompt: str = ""
    model: Model | None = None
    thinking_level: ThinkingLevel = "off"
    tools: list[AgentTool] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)
    is_streaming: bool = False
    stream_message: AgentMessage | None = None
    pending_tool_calls: set[str] = field(default_factory=set)
    error: str | None = None


# =============================================================================
# Agent Events
# =============================================================================


class AgentEventAgentStart(_BaseModel):
    """Agent loop started."""

    type: Literal["agent_start"] = "agent_start"


class AgentEventAgentEnd(_BaseModel):
    """Agent loop ended."""

    type: Literal["agent_end"] = "agent_end"
    messages: list[AgentMessage]


class AgentEventTurnStart(_BaseModel):
    """Turn started (one assistant response + tool calls)."""

    type: Literal["turn_start"] = "turn_start"


class AgentEventTurnEnd(_BaseModel):
    """Turn ended."""

    type: Literal["turn_end"] = "turn_end"
    message: _AssistantMessage
    tool_results: list[_ToolResultMessage] = Field(default_factory=list)

class AgentEventMessageStart(_BaseModel):
    """Message started."""

    type: Literal["message_start"] = "message_start"
    message: AgentMessage


class AgentEventMessageUpdate(_BaseModel):
    """Message updated (streaming)."""

    type: Literal["message_update"] = "message_update"
    message: AgentMessage
    stream_event: StreamEvent

class AgentEventMessageEnd(_BaseModel):
    """Message ended."""

    type: Literal["message_end"] = "message_end"
    message: AgentMessage


class AgentEventToolExecutionStart(_BaseModel):
    """Tool execution started."""

    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    args: dict[str, Any]


class AgentEventToolExecutionUpdate(_BaseModel):
    """Tool execution update (streaming)."""

    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    args: dict[str, Any]
    partial_result: Any = Field(alias="partialResult")


class AgentEventToolExecutionEnd(_BaseModel):
    """Tool execution ended."""

    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    result: AgentToolResult
    is_error: bool = Field(alias="isError")


AgentEvent = Annotated[
    AgentEventAgentStart
    | AgentEventAgentEnd
    | AgentEventTurnStart
    | AgentEventTurnEnd
    | AgentEventMessageStart
    | AgentEventMessageUpdate
    | AgentEventMessageEnd
    | AgentEventToolExecutionStart
    | AgentEventToolExecutionUpdate
    | AgentEventToolExecutionEnd,
    Field(discriminator="type"),
]

# =============================================================================
# Agent Loop Config
# =============================================================================

# Type aliases for config callbacks
# Using quotes for forward references
ConvertToLlmFn = Callable[[list[AgentMessage]], "list[Message]"] | Callable[
    [list[AgentMessage]], Coroutine[Any, Any, "list[Message]"]
]
TransformContextFn = Callable[
    [list[AgentMessage], asyncio.Event | None],
    Coroutine[Any, Any, list[AgentMessage]],
]
GetApiKeyFn = Callable[[str], str | None | Coroutine[Any, Any, str | None]]
GetMessagesFn = Callable[[], Coroutine[Any, Any, list[AgentMessage]]]
@dataclass
class AgentLoopConfig:
    """Configuration for agent loop.

    Extends SimpleStreamOptions with agent-specific callbacks.
    """

    # Required
    model: Model
    convert_to_llm: ConvertToLlmFn

    # Optional streaming options
    temperature: float | None = None
    max_tokens: int | None = None
    api_key: str | None = None
    transport: Literal["sse", "websocket", "auto"] | None = None
    cache_retention: Literal["none", "short", "long"] = "short"
    session_id: str | None = None
    headers: dict[str, str] | None = None
    max_retry_delay_ms: int = 60000
    metadata: dict[str, Any] | None = None
    reasoning: ThinkingLevel | None = None

    # Agent-specific callbacks
    transform_context: TransformContextFn | None = None
    get_api_key: GetApiKeyFn | None = None
    get_steering_messages: GetMessagesFn | None = None
    get_follow_up_messages: GetMessagesFn | None = None


# =============================================================================
# Agent Options
# =============================================================================


@dataclass
class AgentOptions:
    """Options for creating an Agent."""

    # Initial state
    initial_state: AgentState | None = None

    # Message conversion
    convert_to_llm: ConvertToLlmFn | None = None
    transform_context: TransformContextFn | None = None

    # Queue modes
    steering_mode: Literal["all", "one-at-a-time"] = "one-at-a-time"
    follow_up_mode: Literal["all", "one-at-a-time"] = "one-at-a-time"

    # Session
    session_id: str | None = None

    # Dynamic API key
    get_api_key: GetApiKeyFn | None = None


# =============================================================================
# Default Convert Functions
# =============================================================================


def default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    """Default convert_to_llm: filter to LLM-compatible messages only."""
    return [
        m
        for m in messages
        if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")
    ]
