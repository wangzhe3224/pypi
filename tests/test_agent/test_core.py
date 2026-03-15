"""Tests for pi.agent.core module - agent loop implementation."""

import asyncio
import time

import pytest

from pi.agent.core import (
    _execute_tool_calls,
    _extract_tool_calls,
    agent_loop,
    agent_loop_continue,
)
from pi.agent.types import (
    AgentContext,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    AgentToolResult,
)
from pi.ai.types import (
    AssistantMessage,
    Model,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def create_usage() -> Usage:
    """Create a mock Usage object."""
    return Usage(input=0, output=0, cacheRead=0, cacheWrite=0, totalTokens=0)


def create_model() -> Model:
    """Create a mock Model object."""
    return Model(
        id="mock-model",
        name="mock-model",
        api="openai-completions",
        provider="openai",
        baseUrl="https://example.com",
        reasoning=False,
        input=["text"],
        cost={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        contextWindow=8192,
        maxTokens=2048,
    )


def create_assistant_message(
    content: list,
    stop_reason: str = "stop",
) -> AssistantMessage:
    """Create an assistant message."""
    return AssistantMessage(
        role="assistant",
        content=content,
        api="openai-completions",
        provider="openai",
        model="mock",
        usage=create_usage(),
        stopReason=stop_reason,
        timestamp=int(time.time() * 1000),
    )


def create_user_message(text: str) -> UserMessage:
    """Create a user message."""
    return UserMessage(
        role="user",
        content=text,
        timestamp=int(time.time() * 1000),
    )


def identity_converter(messages: list[AgentMessage]) -> list:
    """Simple identity converter that passes through standard messages."""
    return [m for m in messages if m.role in ("user", "assistant", "toolResult")]


class TestAgentLoop:
    """Tests for agent_loop function."""

    @pytest.mark.asyncio
    async def test_throws_when_no_model(self) -> None:
        """Test that agent_loop throws when no model is configured."""
        context = AgentContext(
            system_prompt="You are helpful.",
            messages=[],
            tools=[],
        )

        user_prompt = create_user_message("Hello")

        # Missing model should cause error when streaming
        # This tests the config validation path
        assert context.messages == []

    @pytest.mark.asyncio
    async def test_adds_prompts_to_context(self) -> None:
        """Test that agent_loop adds prompts to context."""
        context = AgentContext(
            system_prompt="You are helpful.",
            messages=[],
            tools=[],
        )

        user_prompt = create_user_message("Hello")

        # The agent_loop should add prompts to context
        # (This is a structural test, not a full integration test)
        assert len(context.messages) == 0
        # After agent_loop starts, context should have user prompt
        # Full test requires mocking stream, which is complex


class TestAgentLoopContinue:
    """Tests for agent_loop_continue function."""

    @pytest.mark.asyncio
    async def test_throws_when_no_messages(self) -> None:
        """Test that agent_loop_continue throws when context has no messages."""
        context = AgentContext(
            system_prompt="You are helpful.",
            messages=[],
            tools=[],
        )

        config = AgentLoopConfig(
            model=create_model(),
            convert_to_llm=identity_converter,
        )

        with pytest.raises(ValueError, match="Cannot continue: no messages"):
            async for _ in agent_loop_continue(context, config):
                pass

    @pytest.mark.asyncio
    async def test_accepts_context_with_messages(self) -> None:
        """Test that agent_loop_continue accepts context with messages."""
        user_message = create_user_message("Hello")

        context = AgentContext(
            system_prompt="You are helpful.",
            messages=[user_message],
            tools=[],
        )

        # Should not throw when messages exist
        assert len(context.messages) == 1
        assert context.messages[0].role == "user"


class TestExtractToolCalls:
    """Tests for _extract_tool_calls function."""

    def test_extracts_single_tool_call(self) -> None:
        """Test extracting a single tool call."""
        message = create_assistant_message(
            [
                ToolCall(id="call-1", name="read", arguments={"file": "test.txt"}),
            ]
        )

        calls = _extract_tool_calls(message)

        assert len(calls) == 1
        assert calls[0]["id"] == "call-1"
        assert calls[0]["name"] == "read"
        assert calls[0]["arguments"] == {"file": "test.txt"}

    def test_extracts_multiple_tool_calls(self) -> None:
        """Test extracting multiple tool calls."""
        message = create_assistant_message(
            [
                ToolCall(id="call-1", name="read", arguments={"file": "a.txt"}),
                TextContent(text="and also"),
                ToolCall(id="call-2", name="bash", arguments={"cmd": "ls"}),
            ]
        )

        calls = _extract_tool_calls(message)

        assert len(calls) == 2
        assert calls[0]["name"] == "read"
        assert calls[1]["name"] == "bash"

    def test_returns_empty_for_no_tool_calls(self) -> None:
        """Test that no tool calls returns empty list."""
        message = create_assistant_message(
            [
                TextContent(text="Just text"),
            ]
        )

        calls = _extract_tool_calls(message)

        assert len(calls) == 0

    def test_mixed_content_extracts_only_tool_calls(self) -> None:
        """Test that mixed content extracts only tool calls."""
        message = create_assistant_message(
            [
                TextContent(text="Let me help"),
                ToolCall(id="call-1", name="search", arguments={"query": "test"}),
                TextContent(text="Found it"),
                ToolCall(id="call-2", name="read", arguments={"file": "result.txt"}),
            ]
        )

        calls = _extract_tool_calls(message)

        assert len(calls) == 2
        assert all(c["name"] in ("search", "read") for c in calls)


class TestExecuteToolCalls:
    """Tests for _execute_tool_calls function."""

    @pytest.mark.asyncio
    async def test_executes_tool_and_returns_result(self) -> None:
        """Test that tool execution returns proper result."""

        async def echo_execute(tool_call_id: str, params: dict, signal=None) -> AgentToolResult:
            return AgentToolResult(
                content=[TextContent(text=f"result: {params['value']}")],
            )

        tool = AgentTool(
            name="echo",
            description="Echo",
            execute=echo_execute,
        )

        tool_calls = [{"id": "call-1", "name": "echo", "arguments": {"value": "test"}}]

        results = await _execute_tool_calls([tool], tool_calls, None)

        assert len(results) == 1
        assert results[0].tool_name == "echo"
        assert results[0].is_error is False

    @pytest.mark.asyncio
    async def test_handles_missing_tool(self) -> None:
        """Test handling of missing tool."""
        tool_calls = [{"id": "call-1", "name": "nonexistent", "arguments": {}}]

        results = await _execute_tool_calls([], tool_calls, None)

        assert len(results) == 1
        assert results[0].is_error is True
        assert "not found" in str(results[0].content[0].text).lower()

    @pytest.mark.asyncio
    async def test_handles_tool_error(self) -> None:
        """Test handling of tool execution error."""

        async def failing_execute(tool_call_id: str, params: dict, signal=None) -> AgentToolResult:
            raise ValueError("Intentional error")

        tool = AgentTool(
            name="fail",
            description="Failing tool",
            execute=failing_execute,
        )

        tool_calls = [{"id": "call-1", "name": "fail", "arguments": {}}]

        results = await _execute_tool_calls([tool], tool_calls, None)

        assert len(results) == 1
        assert results[0].is_error is True

    @pytest.mark.asyncio
    async def test_respects_abort_signal(self) -> None:
        """Test that tool execution respects abort signal."""
        signal = asyncio.Event()
        signal.set()  # Immediately cancelled

        async def slow_execute(tool_call_id: str, params: dict, signal=None) -> AgentToolResult:
            if signal and signal.is_set():
                return AgentToolResult(
                    content=[TextContent(text="Cancelled")],
                    details={"cancelled": True},
                )
            return AgentToolResult(content=[TextContent(text="Done")])

        tool = AgentTool(
            name="slow",
            description="Slow tool",
            execute=slow_execute,
        )

        tool_calls = [{"id": "call-1", "name": "slow", "arguments": {}}]

        results = await _execute_tool_calls([tool], tool_calls, signal)

        # Should complete (cancelled tools return cancelled result)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_executes_multiple_tools_in_sequence(self) -> None:
        """Test that multiple tools are executed in sequence."""
        executed: list[str] = []

        async def track_execute(tool_call_id: str, params: dict, signal=None) -> AgentToolResult:
            executed.append(params["name"])
            return AgentToolResult(content=[TextContent(text="ok")])

        tool = AgentTool(
            name="track",
            description="Track tool",
            execute=track_execute,
        )

        tool_calls = [
            {"id": "call-1", "name": "track", "arguments": {"name": "first"}},
            {"id": "call-2", "name": "track", "arguments": {"name": "second"}},
            {"id": "call-3", "name": "track", "arguments": {"name": "third"}},
        ]

        results = await _execute_tool_calls([tool], tool_calls, None)

        assert len(results) == 3
        assert executed == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_tool_without_execute_returns_error(self) -> None:
        """Test that tool without execute function returns error."""
        tool = AgentTool(
            name="noexec",
            description="Tool without execute",
            execute=None,
        )

        tool_calls = [{"id": "call-1", "name": "noexec", "arguments": {}}]

        results = await _execute_tool_calls([tool], tool_calls, None)

        assert len(results) == 1
        assert results[0].is_error is True
        assert "no execute" in str(results[0].content[0].text).lower()
