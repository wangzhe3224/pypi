"""Core types for pi-ai LLM abstraction layer."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Provider & API Types
# =============================================================================

KnownApi = Literal[
    "openai-completions",
    "mistral-conversations",
    "openai-responses",
    "azure-openai-responses",
    "openai-codex-responses",
    "anthropic-messages",
    "bedrock-converse-stream",
    "google-generative-ai",
    "google-gemini-cli",
    "google-vertex",
]

Api = str  # KnownApi or any string

KnownProvider = Literal[
    "amazon-bedrock",
    "anthropic",
    "google",
    "google-gemini-cli",
    "google-antigravity",
    "google-vertex",
    "openai",
    "azure-openai-responses",
    "openai-codex",
    "github-copilot",
    "xai",
    "groq",
    "cerebras",
    "openrouter",
    "vercel-ai-gateway",
    "zai",
    "mistral",
    "minimax",
    "minimax-cn",
    "huggingface",
    "opencode",
    "opencode-go",
    "kimi-coding",
]

Provider = str  # KnownProvider or any string

ThinkingLevel = Literal["minimal", "low", "medium", "high", "xhigh"]
CacheRetention = Literal["none", "short", "long"]
Transport = Literal["sse", "websocket", "auto"]
StopReason = Literal["stop", "length", "toolUse", "error", "aborted"]


# =============================================================================
# Content Types
# =============================================================================


class _BaseModel(BaseModel):
    """Base model with alias support."""

    model_config = ConfigDict(populate_by_name=True)


class TextContent(_BaseModel):
    """Text content in a message."""

    type: Literal["text"] = "text"
    text: str
    text_signature: str | None = Field(default=None, alias="textSignature")


class ThinkingContent(_BaseModel):
    """Thinking/reasoning content in a message."""

    type: Literal["thinking"] = "thinking"
    thinking: str
    thinking_signature: str | None = Field(default=None, alias="thinkingSignature")
    redacted: bool = False


class ImageContent(_BaseModel):
    """Image content in a message."""

    type: Literal["image"] = "image"
    data: str  # base64 encoded
    mime_type: str = Field(alias="mimeType")


class ToolCall(_BaseModel):
    """A tool call from the assistant."""

    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: dict[str, Any]
    thought_signature: str | None = Field(default=None, alias="thoughtSignature")


Content = TextContent | ThinkingContent | ImageContent
AssistantContent = TextContent | ThinkingContent | ToolCall
UserContent = TextContent | ImageContent


# =============================================================================
# Usage Tracking
# =============================================================================


class Cost(_BaseModel):
    """Token costs in USD."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = Field(default=0.0, alias="cacheRead")
    cache_write: float = Field(default=0.0, alias="cacheWrite")
    total: float = 0.0


class Usage(_BaseModel):
    """Token usage statistics."""

    input: int = 0
    output: int = 0
    cache_read: int = Field(default=0, alias="cacheRead")
    cache_write: int = Field(default=0, alias="cacheWrite")
    total_tokens: int = Field(default=0, alias="totalTokens")
    cost: Cost = Field(default_factory=Cost)


# =============================================================================
# Messages
# =============================================================================


class UserMessage(_BaseModel):
    """A message from the user."""

    role: Literal["user"] = "user"
    content: str | list[UserContent]
    timestamp: int  # Unix timestamp in milliseconds


class AssistantMessage(_BaseModel):
    """A message from the assistant."""

    role: Literal["assistant"] = "assistant"
    content: list[AssistantContent]
    api: Api
    provider: Provider
    model: str
    usage: Usage
    stop_reason: StopReason = Field(alias="stopReason")
    error_message: str | None = Field(default=None, alias="errorMessage")
    timestamp: int  # Unix timestamp in milliseconds


class ToolResultMessage(_BaseModel):
    """A tool result message."""

    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    content: list[UserContent]
    details: Any = None
    is_error: bool = Field(alias="isError")
    timestamp: int  # Unix timestamp in milliseconds


Message = UserMessage | AssistantMessage | ToolResultMessage


# =============================================================================
# Tool Definition
# =============================================================================


class ToolParameter(_BaseModel):
    """JSON Schema for a tool parameter."""

    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class Tool(_BaseModel):
    """A tool that can be called by the assistant."""

    name: str
    description: str
    parameters: ToolParameter = Field(default_factory=ToolParameter)


# =============================================================================
# Context
# =============================================================================


class Context(_BaseModel):
    """Conversation context for LLM calls."""

    system_prompt: str | None = Field(default=None, alias="systemPrompt")
    messages: list[Message] = Field(default_factory=list)
    tools: list[Tool] | None = None


# =============================================================================
# Stream Options
# =============================================================================


class ThinkingBudgets(_BaseModel):
    """Token budgets for each thinking level."""

    minimal: int | None = None
    low: int | None = None
    medium: int | None = None
    high: int | None = None


class StreamOptions(_BaseModel):
    """Base options for streaming LLM responses."""

    temperature: float | None = None
    max_tokens: int | None = Field(default=None, alias="maxTokens")
    api_key: str | None = Field(default=None, alias="apiKey")
    transport: Transport | None = None
    cache_retention: CacheRetention = Field(default="short", alias="cacheRetention")
    session_id: str | None = Field(default=None, alias="sessionId")
    headers: dict[str, str] | None = None
    max_retry_delay_ms: int = Field(default=60000, alias="maxRetryDelayMs")
    metadata: dict[str, Any] | None = None


class SimpleStreamOptions(StreamOptions):
    """Stream options with reasoning support."""

    reasoning: ThinkingLevel | None = None
    thinking_budgets: ThinkingBudgets | None = Field(default=None, alias="thinkingBudgets")


# =============================================================================
# Model Definition
# =============================================================================


InputType = Literal["text", "image"]


def _default_input() -> list[InputType]:
    return ["text"]


class ModelCost(_BaseModel):
    """Cost per million tokens."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = Field(default=0.0, alias="cacheRead")
    cache_write: float = Field(default=0.0, alias="cacheWrite")


class Model(_BaseModel):
    """Model configuration."""

    id: str
    name: str
    api: Api
    provider: Provider
    base_url: str = Field(alias="baseUrl")
    reasoning: bool = False
    input: list[InputType] = Field(default_factory=_default_input)

    cost: ModelCost = Field(default_factory=ModelCost)
    context_window: int = Field(default=4096, alias="contextWindow")
    max_tokens: int = Field(default=4096, alias="maxTokens")
    headers: dict[str, str] | None = None


# =============================================================================
# Stream Events
# =============================================================================


class StreamEventStart(_BaseModel):
    """Stream start event."""

    type: Literal["start"] = "start"
    partial: AssistantMessage


class StreamEventTextStart(_BaseModel):
    """Text content start event."""

    type: Literal["text_start"] = "text_start"
    content_index: int = Field(alias="contentIndex")
    partial: AssistantMessage


class StreamEventTextDelta(_BaseModel):
    """Text delta event."""

    type: Literal["text_delta"] = "text_delta"
    content_index: int = Field(alias="contentIndex")
    delta: str
    partial: AssistantMessage


class StreamEventTextEnd(_BaseModel):
    """Text content end event."""

    type: Literal["text_end"] = "text_end"
    content_index: int = Field(alias="contentIndex")
    content: str
    partial: AssistantMessage


class StreamEventThinkingStart(_BaseModel):
    """Thinking content start event."""

    type: Literal["thinking_start"] = "thinking_start"
    content_index: int = Field(alias="contentIndex")
    partial: AssistantMessage


class StreamEventThinkingDelta(_BaseModel):
    """Thinking delta event."""

    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int = Field(alias="contentIndex")
    delta: str
    partial: AssistantMessage


class StreamEventThinkingEnd(_BaseModel):
    """Thinking content end event."""

    type: Literal["thinking_end"] = "thinking_end"
    content_index: int = Field(alias="contentIndex")
    content: str
    partial: AssistantMessage


class StreamEventToolCallStart(_BaseModel):
    """Tool call start event."""

    type: Literal["toolcall_start"] = "toolcall_start"
    content_index: int = Field(alias="contentIndex")
    partial: AssistantMessage


class StreamEventToolCallDelta(_BaseModel):
    """Tool call delta event."""

    type: Literal["toolcall_delta"] = "toolcall_delta"
    content_index: int = Field(alias="contentIndex")
    delta: str
    partial: AssistantMessage


class StreamEventToolCallEnd(_BaseModel):
    """Tool call end event."""

    type: Literal["toolcall_end"] = "toolcall_end"
    content_index: int = Field(alias="contentIndex")
    tool_call: ToolCall = Field(alias="toolCall")
    partial: AssistantMessage


class StreamEventDone(_BaseModel):
    """Stream done event."""

    type: Literal["done"] = "done"
    reason: Literal["stop", "length", "toolUse"]
    message: AssistantMessage


class StreamEventError(_BaseModel):
    """Stream error event."""

    type: Literal["error"] = "error"
    reason: Literal["aborted", "error"]
    error: AssistantMessage


StreamEvent = Annotated[
    StreamEventStart
    | StreamEventTextStart
    | StreamEventTextDelta
    | StreamEventTextEnd
    | StreamEventThinkingStart
    | StreamEventThinkingDelta
    | StreamEventThinkingEnd
    | StreamEventToolCallStart
    | StreamEventToolCallDelta
    | StreamEventToolCallEnd
    | StreamEventDone
    | StreamEventError,
    Field(discriminator="type"),
]
