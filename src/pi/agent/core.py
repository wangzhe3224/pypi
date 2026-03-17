"""Agent loop implementing the ReAct (Reason-Act-Observe) pattern.

This module provides the core agent loop functionality:
- agent_loop: Start a new conversation with prompts
- agent_loop_continue: Continue from existing context

The loop implements the ReAct pattern:
1. Stream assistant response from LLM
2. Execute any tool calls
3. Feed tool results back to LLM
4. Repeat until no more tool calls
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from pi.agent.types import (
    AgentContext,
    AgentEvent,
    AgentEventAgentEnd,
    AgentEventAgentStart,
    AgentEventMessageEnd,
    AgentEventMessageStart,
    AgentEventMessageUpdate,
    AgentEventTurnEnd,
    AgentEventTurnStart,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    AgentToolResult,
)

if TYPE_CHECKING:
    from pi.ai.types import (
        AssistantMessage,
        ToolResultMessage,
    )


async def agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None = None,
) -> AsyncGenerator[AgentEvent, None]:
    """Start an agent loop with new prompt messages.

    Args:
        prompts: Messages to add to the conversation.
        context: Agent context (system prompt, messages, tools).
        config: Loop configuration.
        signal: Cancellation signal.

    Yields:
        AgentEvent for UI updates.
    """
    # Build new context with prompts
    new_messages: list[AgentMessage] = list(prompts)
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=[*context.messages, *prompts],
        tools=context.tools,
    )

    yield AgentEventAgentStart()
    yield AgentEventTurnStart()

    for prompt in prompts:
        yield AgentEventMessageStart(message=prompt)
        yield AgentEventMessageEnd(message=prompt)

    async for event in _run_loop(current_context, new_messages, config, signal):
        yield event

    context.messages = current_context.messages


async def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None = None,
) -> AsyncGenerator[AgentEvent, None]:
    """Continue agent loop from existing context.

    Args:
        context: Current context.
        config: Loop configuration.
        signal: Cancellation signal.

    Yields:
        AgentEvent for UI updates.

    Raises:
        ValueError: If no messages in context.
    """
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")

    new_messages: list[AgentMessage] = []
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages),
        tools=context.tools,
    )

    yield AgentEventAgentStart()
    yield AgentEventTurnStart()

    async for event in _run_loop(current_context, new_messages, config, signal):
        yield event


async def _run_loop(
    context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
) -> AsyncGenerator[AgentEvent, None]:
    """Main loop implementing ReAct pattern."""

    while True:
        # Stream assistant response
        assistant_message: AssistantMessage | None = None

        async for event in _stream_assistant_response(context, config, signal):
            if event.type == "message_end":
                assistant_message = event.message  # type: ignore[assignment]
            yield event

        if assistant_message is None:
            return

        # Extract and execute tool calls
        tool_calls = _extract_tool_calls(assistant_message)
        tool_results: list[ToolResultMessage] = []

        if tool_calls:
            tool_results = await _execute_tool_calls(
                context.tools or [],
                tool_calls,
                signal,
            )
            for result in tool_results:
                context.messages.append(result)
                new_messages.append(result)

        yield AgentEventTurnEnd(message=assistant_message, tool_results=tool_results)
        new_messages.append(assistant_message)

        if not tool_calls:
            next_messages: list[AgentMessage] | None = None
            if config.get_steering_messages:
                next_messages = await config.get_steering_messages()
            if not next_messages and config.get_follow_up_messages:
                next_messages = await config.get_follow_up_messages()

            if next_messages:
                context.messages.extend(next_messages)
                new_messages.extend(next_messages)
                for msg in next_messages:
                    yield AgentEventMessageStart(message=msg)
                    yield AgentEventMessageEnd(message=msg)
                yield AgentEventTurnStart()
                continue
            break

    yield AgentEventAgentEnd(messages=new_messages)


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
) -> AsyncGenerator[AgentEvent, None]:
    """Stream an assistant response from the LLM."""
    from pi.ai.stream import stream
    from pi.ai.types import Context as LlmContext

    # Convert messages to LLM format
    llm_messages = config.convert_to_llm(context.messages)
    if asyncio.iscoroutine(llm_messages):
        llm_messages = await llm_messages

    # Build LLM context
    llm_context = LlmContext(
        system_prompt=context.system_prompt,
        messages=llm_messages,
        tools=[t.to_llm_tool() for t in context.tools] if context.tools else None,
    )

    # Build stream options
    from pi.ai.types import SimpleStreamOptions

    stream_opts = SimpleStreamOptions(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    partial_message: AssistantMessage | None = None

    async for event in stream(config.model, llm_context, stream_opts):
        if signal and signal.is_set():
            return

        if event.type == "start":
            partial_message = event.partial
            context.messages.append(partial_message)
            yield AgentEventMessageStart(message=partial_message)

        elif event.type in (
            "text_start",
            "text_delta",
            "text_end",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        ):
            if partial_message:
                # These events have partial attribute
                partial_message = getattr(event, "partial", partial_message)
                context.messages[-1] = partial_message
                yield AgentEventMessageUpdate(message=partial_message, stream_event=event)

        elif event.type == "done":
            final = event.message
            if partial_message:
                context.messages[-1] = final
            else:
                context.messages.append(final)
            yield AgentEventMessageEnd(message=final)
            return

        elif event.type == "error":
            err = event.error
            if partial_message:
                context.messages[-1] = err
            else:
                context.messages.append(err)
            yield AgentEventMessageEnd(message=err)
            return

    raise RuntimeError("Stream completed without result")


def _extract_tool_calls(message: AssistantMessage) -> list[dict[str, Any]]:
    """Extract tool calls from assistant message content."""
    calls: list[dict[str, Any]] = []
    for item in message.content:
        if hasattr(item, "type") and getattr(item, "type", None) == "toolCall":
            calls.append(
                {
                    "id": getattr(item, "id", ""),
                    "name": getattr(item, "name", ""),
                    "arguments": getattr(item, "arguments", {}),
                }
            )
    return calls


async def _execute_tool_calls(
    tools: list[AgentTool],
    tool_calls: list[dict[str, Any]],
    signal: asyncio.Event | None,
) -> list[ToolResultMessage]:
    """Execute tool calls and return results."""
    from pi.ai.types import ToolResultMessage

    results: list[ToolResultMessage] = []

    for call in tool_calls:
        call_id = call.get("id", "")
        name = call.get("name", "")
        args = call.get("arguments", {})

        tool = next((t for t in tools if t.name == name), None)

        result: AgentToolResult
        is_error = False

        try:
            if not tool:
                raise ValueError(f"Tool '{name}' not found")
            if not tool.execute:
                raise ValueError(f"Tool '{name}' has no execute function")

            result = await tool.execute(
                tool_call_id=call_id,
                params=args,
                signal=signal,
            )

        except Exception as e:
            result = AgentToolResult(
                content=[{"type": "text", "text": str(e)}],
                details={"error": str(e)},
            )
            is_error = True

        tr = ToolResultMessage(
            role="toolResult",
            tool_call_id=call_id,
            tool_name=name,
            content=result.content,
            details=result.details,
            is_error=is_error,
            timestamp=int(time.time() * 1000),
        )
        results.append(tr)

    return results
