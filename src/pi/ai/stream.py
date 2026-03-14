"""Streaming API for LLM interactions."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from pi.ai.providers.base import ProviderRegistry

if TYPE_CHECKING:
    from pi.ai.types import (
        AssistantMessage,
        Context,
        Model,
        StreamEvent,
        StreamOptions,
    )


async def stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """Stream responses from an LLM.

    Args:
        model: The model configuration.
        context: The conversation context.
        options: Optional streaming options.

    Yields:
        StreamEvent: Events as they occur during generation.

    Example:
        async for event in stream(model, context):
            if event.type == "text_delta":
                print(event.delta, end="", flush=True)
            elif event.type == "done":
                print(f"\\nCompleted: {event.message}")
    """
    provider = ProviderRegistry.get(model.provider)
    if not provider:
        raise ValueError(f"No provider registered for: {model.provider}")

    async for event in provider.stream(model, context, options):
        yield event


async def complete(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessage:
    """Complete a conversation and return the full response.

    Args:
        model: The model configuration.
        context: The conversation context.
        options: Optional streaming options.

    Returns:
        AssistantMessage: The complete assistant response.

    Example:
        response = await complete(model, context)
        print(response.content[0].text)
    """

    result: AssistantMessage | None = None

    async for event in stream(model, context, options):
        if event.type == "done":
            result = event.message
        elif event.type == "error":
            result = event.error

    if result is None:
        raise RuntimeError("Stream completed without result")

    return result
