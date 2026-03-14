"""Message conversion utilities for Anthropic API."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal, cast

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
    """Convert pi messages to Anthropic API format.

    Args:
        model: The model configuration.
        context: The conversation context.

    Returns:
        List of Anthropic-formatted message dictionaries.
    """
    params: list[dict[str, Any]] = []

    for msg in context.messages:
        if msg.role == "user":
            converted = _convert_user_message(msg, model)
            if converted:
                params.append(converted)
        elif msg.role == "assistant":
            converted = _convert_assistant_message(msg, model)
            if converted:
                params.append(converted)
        elif msg.role == "toolResult":
            # Anthropic groups tool results into a single user message
            # Check if last message is already a user message with tool_result content
            if params and params[-1].get("role") == "user":
                last_content = params[-1].get("content")
                if isinstance(last_content, list):
                    # Append tool result to existing user message
                    last_content.append(_convert_tool_result_block(msg, model))
                    continue
            # Create new user message with tool result
            params.append({
                "role": "user",
                "content": [_convert_tool_result_block(msg, model)],
            })

    return params


def _convert_user_message(msg: UserMessage, model: Model) -> dict[str, Any] | None:
    """Convert a user message to Anthropic format."""
    if isinstance(msg.content, str):
        if not msg.content.strip():
            return None
        return {"role": "user", "content": msg.content}

    # Handle content list (text + images)
    blocks: list[dict[str, Any]] = []
    for item in msg.content:
        if item.type == "text":
            if item.text.strip():
                blocks.append({"type": "text", "text": item.text})
        elif item.type == "image" and "image" in model.input:
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": item.mime_type,
                    "data": item.data,
                },
            })

    if not blocks:
        return None

    return {"role": "user", "content": blocks}


def _convert_assistant_message(
    msg: AssistantMessage,
    model: Model,  # noqa: ARG001
) -> dict[str, Any] | None:
    """Convert an assistant message to Anthropic format."""
    blocks: list[dict[str, Any]] = []

    for block in msg.content:
        if block.type == "text":
            if block.text.strip():
                blocks.append({"type": "text", "text": block.text})
        elif block.type == "thinking":
            if not block.thinking.strip():
                continue
            # Handle redacted thinking
            if block.redacted:
                blocks.append({
                    "type": "redacted_thinking",
                    "data": block.thinking_signature or "",
                })
                continue
            # Regular thinking block with signature
            if block.thinking_signature:
                blocks.append({
                    "type": "thinking",
                    "thinking": block.thinking,
                    "signature": block.thinking_signature,
                })
            else:
                # Without signature, convert to text to avoid API rejection
                blocks.append({"type": "text", "text": block.thinking})
        elif block.type == "toolCall":
            blocks.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.arguments or {},
            })

    if not blocks:
        return None

    return {"role": "assistant", "content": blocks}


def _convert_tool_result_block(
    msg: ToolResultMessage,
    model: Model,
) -> dict[str, Any]:
    """Convert a tool result message to Anthropic tool_result block."""
    # Extract text content
    text_parts: list[str] = []
    image_blocks: list[dict[str, Any]] = []

    for block in msg.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "image" and "image" in model.input:
            image_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": block.mime_type,
                    "data": block.data,
                },
            })

    text_content = "\n".join(text_parts) or "(no output)"

    result: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": msg.tool_call_id,
        "content": text_content,
        "is_error": msg.is_error,
    }

    # Anthropic supports images in tool_result content as a list
    if image_blocks:
        result["content"] = [{"type": "text", "text": text_content}, *image_blocks]

    return result


def convert_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """Convert pi tools to Anthropic API format.

    Args:
        tools: List of Tool definitions.

    Returns:
        List of Anthropic-formatted tool dictionaries.
    """
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": {
                "type": "object",
                "properties": tool.parameters.properties,
                "required": tool.parameters.required,
            },
        }
        for tool in tools
    ]


def map_stop_reason(reason: str | None) -> Literal["stop", "length", "toolUse", "error"]:
    """Map Anthropic stop reason to pi stop reason.

    Args:
        reason: Anthropic stop_reason value.

    Returns:
        Pi stop reason string.
    """
    if reason is None:
        return "stop"

    mapping: dict[str, Literal["stop", "length", "toolUse", "error"]] = {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "toolUse",
        "stop_sequence": "stop",
        "pause_turn": "stop",
        "refusal": "error",
    }
    return mapping.get(reason, "error")


def parse_streaming_json(partial: str) -> dict[str, Any]:
    """Parse potentially incomplete JSON from streaming.

    Anthropic streams tool arguments as partial JSON chunks.
    This attempts to parse what we have so far.

    Args:
        partial: The accumulated partial JSON string.

    Returns:
        Parsed dictionary, or empty dict if parsing fails.
    """
    if not partial:
        return {}

    try:
        return cast(dict[str, Any], json.loads(partial))
    except json.JSONDecodeError:
        pass

    # Try to complete partial JSON
    # Count brackets to see what's missing
    open_braces = partial.count("{") - partial.count("}")
    open_brackets = partial.count("[") - partial.count("]")
    close_braces = partial.count("}") - partial.count("{")
    close_brackets = partial.count("]") - partial.count("[")

    try:
        # Add missing closing characters
        completed = partial + "]" * open_brackets + "}" * open_braces
        return cast(dict[str, Any], json.loads(completed))
    except json.JSONDecodeError:
        pass

    try:
        # Try adding closing characters to original
        completed = partial + "}" * close_braces + "]" * close_brackets
        return cast(dict[str, Any], json.loads(completed))
    except json.JSONDecodeError:
        return {}
