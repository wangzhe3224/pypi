"""Message conversion utilities for OpenAI API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pi.ai.types import (
        AssistantMessage,
        Context,
        Model,
        Tool,
        ToolResultMessage,
        UserMessage,
    )


def convert_messages(
    model: Model,
    context: Context,
) -> list[dict[str, Any]]:
    """Convert pi messages to OpenAI API format.

    Args:
        model: The model configuration.
        context: The conversation context.

    Returns:
        List of OpenAI-formatted message dictionaries.
    """
    params: list[dict[str, Any]] = []

    # Add system prompt if present
    if context.system_prompt:
        params.append({"role": "system", "content": context.system_prompt})

    for msg in context.messages:
        if msg.role == "user":
            params.extend(_convert_user_message(msg, model))
        elif msg.role == "assistant":
            converted = _convert_assistant_message(msg, model)
            if converted:
                params.append(converted)
        elif msg.role == "toolResult":
            params.extend(_convert_tool_result(msg, model))

    return params


def _convert_user_message(msg: UserMessage, model: Model) -> list[dict[str, Any]]:
    """Convert a user message to OpenAI format."""
    if isinstance(msg.content, str):
        return [{"role": "user", "content": msg.content}]

    # Handle content list (text + images)
    parts: list[dict[str, Any]] = []
    for item in msg.content:
        if item.type == "text":
            parts.append({"type": "text", "text": item.text})
        elif item.type == "image" and "image" in model.input:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{item.mime_type};base64,{item.data}"},
                }
            )

    if not parts:
        return []

    return [{"role": "user", "content": parts}]


def _convert_assistant_message(
    msg: AssistantMessage,
    model: Model,  # noqa: ARG001
) -> dict[str, Any] | None:
    """Convert an assistant message to OpenAI format."""
    result: dict[str, Any] = {"role": "assistant"}

    # Extract text content
    text_parts: list[str] = []
    for block in msg.content:
        if block.type == "text":
            text_parts.append(block.text)

    if text_parts:
        result["content"] = "\n".join(text_parts)
    else:
        result["content"] = None

    # Extract tool calls
    tool_calls: list[dict[str, Any]] = []
    for block in msg.content:
        if block.type == "toolCall":
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": _json_dumps(block.arguments),
                    },
                }
            )

    if tool_calls:
        result["tool_calls"] = tool_calls

    # Skip messages with no content and no tool calls
    if not result.get("content") and not result.get("tool_calls"):
        return None

    return result


def _convert_tool_result(msg: ToolResultMessage, model: Model) -> list[dict[str, Any]]:
    """Convert a tool result message to OpenAI format."""
    # Extract text content
    text_parts: list[str] = []
    for block in msg.content:
        if block.type == "text":
            text_parts.append(block.text)

    text_content = "\n".join(text_parts) or "(no output)"

    result: list[dict[str, Any]] = [
        {
            "role": "tool",
            "tool_call_id": msg.tool_call_id,
            "content": text_content,
        }
    ]

    # Handle images in tool results (send as separate user message)
    image_parts: list[dict[str, Any]] = []
    for block in msg.content:
        if block.type == "image" and "image" in model.input:
            image_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"},
                }
            )

    if image_parts:
        result.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Attached image(s) from tool result:"},
                    *image_parts,
                ],
            }
        )

    return result


def convert_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """Convert pi tools to OpenAI API format.

    Args:
        tools: List of Tool definitions.

    Returns:
        List of OpenAI-formatted tool dictionaries.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters.model_dump(by_alias=True, exclude_none=True),
            },
        }
        for tool in tools
    ]


def _json_dumps(obj: Any) -> str:
    """JSON serialize with consistent formatting."""
    import json

    return json.dumps(obj, separators=(",", ":"))


def map_stop_reason(reason: str | None) -> Literal["stop", "length", "toolUse", "error"]:
    """Map OpenAI stop reason to pi stop reason.

    Args:
        reason: OpenAI finish_reason value.

    Returns:
        Pi stop reason string.
    """
    if reason is None:
        return "stop"

    mapping: dict[str, Literal["stop", "length", "toolUse", "error"]] = {
        "stop": "stop",
        "length": "length",
        "function_call": "toolUse",
        "tool_calls": "toolUse",
        "content_filter": "error",
    }
    return mapping.get(reason, "error")
