"""Message conversion utilities for Google GenAI API."""

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
    """Convert pi messages to Google GenAI format.

    Args:
        model: The model configuration.
        context: The conversation context.

    Returns:
        List of Google-formatted content dictionaries.
    """
    contents: list[dict[str, Any]] = []

    for msg in context.messages:
        if msg.role == "user":
            converted = _convert_user_message(msg, model)
            if converted:
                contents.append(converted)
        elif msg.role == "assistant":
            converted = _convert_assistant_message(msg, model)
            if converted:
                contents.append(converted)
        elif msg.role == "toolResult":
            converted = _convert_tool_result(msg, model)
            if converted:
                # Check if last content is a user message with function responses
                if contents and contents[-1].get("role") == "user":
                    last_parts = contents[-1].get("parts", [])
                    if any(p.get("functionResponse") for p in last_parts):
                        # Append to existing user message
                        last_parts.append(converted["parts"][0])
                        continue
                contents.append(converted)

    return contents


def _convert_user_message(msg: UserMessage, model: Model) -> dict[str, Any] | None:
    """Convert a user message to Google format."""
    if isinstance(msg.content, str):
        if not msg.content.strip():
            return None
        return {
            "role": "user",
            "parts": [{"text": msg.content}],
        }

    # Handle content list (text + images)
    parts: list[dict[str, Any]] = []
    for item in msg.content:
        if item.type == "text":
            if item.text.strip():
                parts.append({"text": item.text})
        elif item.type == "image" and "image" in model.input:
            parts.append({
                "inlineData": {
                    "mimeType": item.mime_type,
                    "data": item.data,
                },
            })

    if not parts:
        return None

    return {"role": "user", "parts": parts}


def _convert_assistant_message(
    msg: AssistantMessage,
    model: Model,
) -> dict[str, Any] | None:
    """Convert an assistant message to Google format."""
    parts: list[dict[str, Any]] = []

    # Check if message is from same provider and model
    is_same_provider = msg.provider == model.provider and msg.model == model.id

    for block in msg.content:
        if block.type == "text":
            if block.text.strip():
                part: dict[str, Any] = {"text": block.text}
                # Preserve text signature if available
                if is_same_provider and hasattr(block, "text_signature") and block.text_signature:
                    part["thoughtSignature"] = block.text_signature
                parts.append(part)

        elif block.type == "thinking":
            if not block.thinking.strip():
                continue
            # Only keep as thinking block if same provider AND same model
            if is_same_provider:
                think_part: dict[str, Any] = {
                    "thought": True,
                    "text": block.thinking,
                }
                if block.thinking_signature:
                    think_part["thoughtSignature"] = block.thinking_signature
                parts.append(think_part)
            else:
                # Convert to plain text (no tags to avoid model mimicking them)
                parts.append({"text": block.thinking})

        elif block.type == "toolCall":
            tool_part: dict[str, Any] = {
                "functionCall": {
                    "name": block.name,
                    "args": block.arguments or {},
                },
            }
            # Preserve thought signature if available
            if is_same_provider and hasattr(block, "thought_signature") and block.thought_signature:
                tool_part["thoughtSignature"] = block.thought_signature
            parts.append(tool_part)

    if not parts:
        return None

    return {"role": "model", "parts": parts}


def _convert_tool_result(
    msg: ToolResultMessage,
    model: Model,
) -> dict[str, Any] | None:
    """Convert a tool result message to Google format."""
    # Extract text and image content
    text_parts: list[str] = []
    image_parts: list[dict[str, Any]] = []

    for block in msg.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "image" and "image" in model.input:
            image_parts.append({
                "inlineData": {
                    "mimeType": block.mime_type,
                    "data": block.data,
                },
            })

    has_text = bool(text_parts)
    has_images = bool(image_parts)

    # Build response value
    response_value = "\n".join(text_parts) if has_text else "(see attached image)" if has_images else ""

    # Check if model supports multimodal function responses (Gemini 3)
    supports_multimodal = "gemini-3" in model.id.lower()

    function_response_part: dict[str, Any] = {
        "functionResponse": {
            "name": msg.tool_name,
            "response": {"error": response_value} if msg.is_error else {"output": response_value},
        },
    }

    # Nest images inside functionResponse.parts for Gemini 3
    if has_images and supports_multimodal:
        function_response_part["functionResponse"]["parts"] = image_parts

    parts: list[dict[str, Any]] = [function_response_part]

    # For older models, add images in a separate part
    if has_images and not supports_multimodal:
        parts.append({"text": "Tool result image:"})
        parts.extend(image_parts)

    return {"role": "user", "parts": parts}


def convert_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """Convert pi tools to Google GenAI format.

    Args:
        tools: List of Tool definitions.

    Returns:
        List of Google-formatted tool dictionaries.
    """
    if not tools:
        return []

    return [
        {
            "functionDeclarations": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parametersJsonSchema": tool.parameters.model_dump(
                        by_alias=True, exclude_none=True
                    ),
                }
                for tool in tools
            ],
        }
    ]


def map_stop_reason(reason: str | None) -> Literal["stop", "length", "toolUse", "error"]:
    """Map Google finish reason to pi stop reason.

    Args:
        reason: Google finish_reason value.

    Returns:
        Pi stop reason string.
    """
    if reason is None:
        return "stop"

    mapping: dict[str, Literal["stop", "length", "toolUse", "error"]] = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "error",
        "RECITATION": "error",
        "BLOCKLIST": "error",
        "PROHIBITED_CONTENT": "error",
        "SPII": "error",
        "OTHER": "error",
        "FINISH_REASON_UNSPECIFIED": "error",
        "LANGUAGE": "error",
        "MALFORMED_FUNCTION_CALL": "error",
        "UNEXPECTED_TOOL_CALL": "error",
        "NO_IMAGE": "error",
    }
    return mapping.get(reason, "error")


def is_thinking_part(part: dict[str, Any]) -> bool:
    """Determine if a part is thinking content.

    Args:
        part: A Google GenAI part dictionary.

    Returns:
        True if the part contains thinking content.
    """
    return part.get("thought") is True
