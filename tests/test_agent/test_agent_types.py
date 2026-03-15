"""Tests for pi.agent.types module."""

import pytest

from pi.agent.types import (
    AgentContext,
    AgentEventAgentEnd,
    AgentEventAgentStart,
    AgentEventMessageEnd,
    AgentEventMessageStart,
    AgentEventTurnEnd,
    AgentEventTurnStart,
    AgentOptions,
    AgentState,
    AgentTool,
    AgentToolResult,
    ThinkingLevel,
    default_convert_to_llm,
)
from pi.ai.types import (
    AssistantMessage,
    TextContent,
    ToolResultMessage,
    Usage,
    UserMessage,
)


class TestAgentToolResult:
    """Tests for AgentToolResult."""

    def test_basic_result(self) -> None:
        """Test basic AgentToolResult creation."""
        result = AgentToolResult(
            content=[TextContent(text="Success")],
        )
        assert len(result.content) == 1

    def test_result_with_details(self) -> None:
        """Test AgentToolResult with details."""
        result = AgentToolResult(
            content=[TextContent(text="Done")],
            details={"key": "value"},
        )
        assert result.details == {"key": "value"}

    def test_result_with_none_details(self) -> None:
        """Test AgentToolResult with None details."""
        result = AgentToolResult(content=[TextContent(text="Done")])
        assert result.details is None


class TestAgentTool:
    """Tests for AgentTool."""

    def test_basic_tool(self) -> None:
        """Test basic AgentTool creation."""
        tool = AgentTool(
            name="read",
            description="Read a file",
        )
        assert tool.name == "read"
        assert tool.description == "Read a file"
        assert tool.label == ""

    def test_tool_with_parameters(self) -> None:
        """Test AgentTool with parameters."""
        tool = AgentTool(
            name="bash",
            description="Run bash command",
            parameters={
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
            label="Run Bash",
        )
        assert tool.name == "bash"
        assert tool.label == "Run Bash"
        assert "properties" in tool.parameters

    def test_to_llm_tool(self) -> None:
        """Test conversion to LLM Tool."""
        tool = AgentTool(
            name="bash",
            description="Run bash command",
            parameters={
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        )
        llm_tool = tool.to_llm_tool()
        assert llm_tool.name == "bash"
        assert llm_tool.description == "Run bash command"
        assert llm_tool.parameters.type == "object"

    def test_to_llm_tool_empty_params(self) -> None:
        """Test conversion with empty parameters."""
        tool = AgentTool(name="noop", description="No operation")
        llm_tool = tool.to_llm_tool()
        assert llm_tool.name == "noop"
        assert llm_tool.parameters.type == "object"


class TestAgentContext:
    """Tests for AgentContext."""

    def test_default_context(self) -> None:
        """Test default AgentContext."""
        context = AgentContext()
        assert context.system_prompt == ""
        assert len(context.messages) == 0
        assert context.tools is None

    def test_context_with_values(self) -> None:
        """Test AgentContext with values."""
        msg = UserMessage(content="Hello", timestamp=1000)
        tool = AgentTool(name="test", description="Test tool")
        context = AgentContext(
            system_prompt="You are helpful",
            messages=[msg],
            tools=[tool],
        )
        assert context.system_prompt == "You are helpful"
        assert len(context.messages) == 1
        assert len(context.tools or []) == 1

    def test_to_llm_context(self) -> None:
        """Test conversion to LLM Context."""
        msg = UserMessage(content="Hi", timestamp=1000)
        context = AgentContext(
            system_prompt="You are an assistant",
            messages=[msg],
        )
        llm_ctx = context.to_llm_context()
        assert llm_ctx.system_prompt == "You are an assistant"
        assert len(llm_ctx.messages) == 1


class TestAgentState:
    """Tests for AgentState."""

    def test_default_state(self) -> None:
        """Test default AgentState."""
        state = AgentState()
        assert state.system_prompt == ""
        assert state.model is None
        assert state.thinking_level == "off"
        assert len(state.tools) == 0
        assert len(state.messages) == 0
        assert not state.is_streaming
        assert state.stream_message is None
        assert len(state.pending_tool_calls) == 0
        assert state.error is None

    def test_state_with_values(self) -> None:
        """Test AgentState with values."""
        state = AgentState(
            system_prompt="Test prompt",
            thinking_level="high",
            is_streaming=True,
            error="Test error",
        )
        assert state.system_prompt == "Test prompt"
        assert state.thinking_level == "high"
        assert state.is_streaming
        assert state.error == "Test error"


class TestAgentEvents:
    """Tests for AgentEvent types."""

    def test_agent_start_event(self) -> None:
        """Test AgentEventAgentStart."""
        event = AgentEventAgentStart()
        assert event.type == "agent_start"

    def test_agent_end_event(self) -> None:
        """Test AgentEventAgentEnd."""
        msg = UserMessage(content="Test", timestamp=1000)
        event = AgentEventAgentEnd(messages=[msg])
        assert event.type == "agent_end"
        assert len(event.messages) == 1

    def test_turn_start_event(self) -> None:
        """Test AgentEventTurnStart."""
        event = AgentEventTurnStart()
        assert event.type == "turn_start"

    def test_turn_end_event(self) -> None:
        """Test AgentEventTurnEnd."""
        msg = AssistantMessage(
            content=[TextContent(text="Done")],
            api="openai-completions",
            provider="openai",
            model="gpt-4",
            usage=Usage(),
            stopReason="stop",
            timestamp=1000,
        )
        event = AgentEventTurnEnd(message=msg, tool_results=[])
        assert event.type == "turn_end"

    def test_message_start_event(self) -> None:
        """Test AgentEventMessageStart."""
        msg = UserMessage(content="Hello", timestamp=1000)
        event = AgentEventMessageStart(message=msg)
        assert event.type == "message_start"
        assert event.message.role == "user"

    def test_message_end_event(self) -> None:
        """Test AgentEventMessageEnd."""
        msg = UserMessage(content="Bye", timestamp=1000)
        event = AgentEventMessageEnd(message=msg)
        assert event.type == "message_end"


class TestAgentOptions:
    """Tests for AgentOptions."""

    def test_default_options(self) -> None:
        """Test default AgentOptions."""
        options = AgentOptions()
        assert options.initial_state is None
        assert options.convert_to_llm is None
        assert options.steering_mode == "one-at-a-time"
        assert options.follow_up_mode == "one-at-a-time"
        assert options.session_id is None

    def test_options_with_values(self) -> None:
        """Test AgentOptions with values."""
        state = AgentState(system_prompt="Test")
        options = AgentOptions(
            initial_state=state,
            steering_mode="all",
            follow_up_mode="all",
            session_id="test-session",
        )
        assert options.initial_state is not None
        assert options.steering_mode == "all"
        assert options.session_id == "test-session"


class TestDefaultConvertToLlm:
    """Tests for default_convert_to_llm."""

    def test_filters_valid_messages(self) -> None:
        """Test that only valid messages are kept."""
        user_msg = UserMessage(content="Hi", timestamp=1000)
        assistant_msg = AssistantMessage(
            content=[TextContent(text="Hello")],
            api="openai-completions",
            provider="openai",
            model="gpt-4",
            usage=Usage(),
            stopReason="stop",
            timestamp=2000,
        )
        tool_result = ToolResultMessage(
            toolCallId="call_1",
            toolName="test",
            content=[TextContent(text="result")],
            isError=False,
            timestamp=3000,
        )

        result = default_convert_to_llm([user_msg, assistant_msg, tool_result])
        assert len(result) == 3

    def test_handles_empty_list(self) -> None:
        """Test with empty list."""
        result = default_convert_to_llm([])
        assert len(result) == 0


class TestThinkingLevel:
    """Tests for ThinkingLevel."""

    def test_valid_levels(self) -> None:
        """Test valid thinking levels."""
        valid_levels: list[ThinkingLevel] = ["off", "minimal", "low", "medium", "high", "xhigh"]
        for level in valid_levels:
            # Just verify these are valid type-wise
            assert level in ["off", "minimal", "low", "medium", "high", "xhigh"]
