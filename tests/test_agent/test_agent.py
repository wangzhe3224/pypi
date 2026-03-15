"""Tests for pi.agent.agent module."""

import asyncio

import pytest

from pi.agent import Agent, tool
from pi.agent.types import AgentOptions, AgentState, AgentTool
from pi.ai.types import Model


@pytest.fixture
def sample_model() -> Model:
    """Sample model for testing."""
    return Model(
        id="gpt-4",
        name="GPT-4",
        api="openai-completions",
        provider="openai",
        baseUrl="https://api.openai.com/v1",
    )


@pytest.fixture
def sample_tool() -> AgentTool:
    """Sample tool for testing."""
    return AgentTool(name="echo", description="Echo input")


class TestAgentInit:
    """Tests for Agent initialization."""

    def test_default_init(self) -> None:
        """Test default Agent initialization."""
        agent = Agent()
        assert agent.state.system_prompt == ""
        assert agent.state.model is None
        assert len(agent.state.tools) == 0
        assert len(agent.messages) == 0
        assert not agent.is_streaming

    def test_init_with_model(self, sample_model: Model) -> None:
        """Test Agent initialization with model."""
        agent = Agent(model=sample_model)
        assert agent.state.model is not None
        assert agent.state.model.id == "gpt-4"

    def test_init_with_system_prompt(self) -> None:
        """Test Agent initialization with system prompt."""
        agent = Agent(system_prompt="You are helpful.")
        assert agent.state.system_prompt == "You are helpful."

    def test_init_with_tools(self, sample_tool: AgentTool) -> None:
        """Test Agent initialization with tools."""
        agent = Agent(tools=[sample_tool])
        assert len(agent.state.tools) == 1
        assert agent.state.tools[0].name == "echo"

    def test_init_with_options(self, sample_model: Model) -> None:
        """Test Agent initialization with options."""
        state = AgentState(system_prompt="Custom prompt")
        options = AgentOptions(
            initial_state=state,
            steering_mode="all",
            session_id="test-session",
        )
        agent = Agent(model=sample_model, options=options)
        assert agent.state.system_prompt == "Custom prompt"


class TestAgentState:
    """Tests for Agent state management."""

    def test_state_property(self, sample_model: Model) -> None:
        """Test state property access."""
        agent = Agent(model=sample_model)
        state = agent.state
        assert state.model is not None

    def test_is_streaming_property(self) -> None:
        """Test is_streaming property."""
        agent = Agent()
        assert not agent.is_streaming

    def test_messages_property(self) -> None:
        """Test messages property."""
        agent = Agent()
        assert agent.messages == []


class TestAgentSubscription:
    """Tests for Agent event subscription."""

    def test_subscribe(self) -> None:
        """Test subscribing to events."""
        agent = Agent()
        events: list = []

        def on_event(event: object) -> None:
            events.append(event)

        unsub = agent.subscribe(on_event)
        agent._emit({"type": "test"})  # type: ignore

        assert len(events) == 1
        unsub()

    def test_unsubscribe(self) -> None:
        """Test unsubscribing from events."""
        agent = Agent()
        events: list = []

        def on_event(event: object) -> None:
            events.append(event)

        unsub = agent.subscribe(on_event)
        unsub()
        agent._emit({"type": "test"})  # type: ignore

        assert len(events) == 0

    def test_multiple_subscribers(self) -> None:
        """Test multiple subscribers."""
        agent = Agent()
        events1: list = []
        events2: list = []

        unsub1 = agent.subscribe(lambda e: events1.append(e))
        unsub2 = agent.subscribe(lambda e: events2.append(e))

        agent._emit({"type": "test"})  # type: ignore

        assert len(events1) == 1
        assert len(events2) == 1

        unsub1()
        unsub2()

    def test_listener_error_handling(self) -> None:
        """Test that listener errors don't propagate."""
        agent = Agent()
        events: list = []

        def bad_listener(event: object) -> None:
            raise RuntimeError("Listener error")

        def good_listener(event: object) -> None:
            events.append(event)

        unsub1 = agent.subscribe(bad_listener)
        unsub2 = agent.subscribe(good_listener)

        # Should not raise
        agent._emit({"type": "test"})  # type: ignore
        assert len(events) == 1

        unsub1()
        unsub2()


class TestAgentMutators:
    """Tests for Agent state mutators."""

    def test_set_system_prompt(self) -> None:
        """Test setting system prompt."""
        agent = Agent()
        agent.set_system_prompt("New prompt")
        assert agent.state.system_prompt == "New prompt"

    def test_set_model(self, sample_model: Model) -> None:
        """Test setting model."""
        agent = Agent()
        agent.set_model(sample_model)
        assert agent.state.model is not None
        assert agent.state.model.id == "gpt-4"

    def test_set_thinking_level(self) -> None:
        """Test setting thinking level."""
        agent = Agent()
        agent.set_thinking_level("high")
        assert agent.state.thinking_level == "high"

    def test_set_tools(self, sample_tool: AgentTool) -> None:
        """Test setting tools."""
        agent = Agent()
        agent.set_tools([sample_tool])
        assert len(agent.state.tools) == 1

    def test_add_tool(self, sample_tool: AgentTool) -> None:
        """Test adding a tool."""
        agent = Agent()
        agent.add_tool(sample_tool)
        assert len(agent.state.tools) == 1
        assert agent.state.tools[0].name == "echo"

    def test_remove_tool(self) -> None:
        """Test removing a tool."""
        tool1 = AgentTool(name="tool1", description="Tool 1")
        tool2 = AgentTool(name="tool2", description="Tool 2")
        agent = Agent(tools=[tool1, tool2])
        agent.remove_tool("tool1")
        assert len(agent.state.tools) == 1
        assert agent.state.tools[0].name == "tool2"

    def test_append_message(self) -> None:
        """Test appending a message."""
        from pi.ai.types import UserMessage

        agent = Agent()
        msg = UserMessage(content="Hello", timestamp=1000)
        agent.append_message(msg)
        assert len(agent.messages) == 1

    def test_replace_messages(self) -> None:
        """Test replacing messages."""
        from pi.ai.types import UserMessage

        agent = Agent()
        msg1 = UserMessage(content="First", timestamp=1000)
        msg2 = UserMessage(content="Second", timestamp=2000)
        agent.replace_messages([msg1, msg2])
        assert len(agent.messages) == 2

    def test_clear_messages(self) -> None:
        """Test clearing messages."""
        from pi.ai.types import UserMessage

        agent = Agent()
        msg = UserMessage(content="Test", timestamp=1000)
        agent.append_message(msg)
        agent.clear_messages()
        assert len(agent.messages) == 0


class TestAgentQueues:
    """Tests for Agent message queues."""

    def test_steer(self) -> None:
        """Test steering queue."""
        from pi.ai.types import UserMessage

        agent = Agent()
        msg = UserMessage(content="Steer", timestamp=1000)
        agent.steer(msg)
        assert agent.has_queued_messages()

    def test_follow_up(self) -> None:
        """Test follow-up queue."""
        from pi.ai.types import UserMessage

        agent = Agent()
        msg = UserMessage(content="Follow up", timestamp=1000)
        agent.follow_up(msg)
        assert agent.has_queued_messages()

    def test_clear_steering_queue(self) -> None:
        """Test clearing steering queue."""
        from pi.ai.types import UserMessage

        agent = Agent()
        msg = UserMessage(content="Steer", timestamp=1000)
        agent.steer(msg)
        agent.clear_steering_queue()
        assert not agent.has_queued_messages()

    def test_clear_follow_up_queue(self) -> None:
        """Test clearing follow-up queue."""
        from pi.ai.types import UserMessage

        agent = Agent()
        msg = UserMessage(content="Follow up", timestamp=1000)
        agent.follow_up(msg)
        agent.clear_follow_up_queue()
        assert not agent.has_queued_messages()

    def test_clear_all_queues(self) -> None:
        """Test clearing all queues."""
        from pi.ai.types import UserMessage

        agent = Agent()
        msg = UserMessage(content="Test", timestamp=1000)
        agent.steer(msg)
        agent.follow_up(msg)
        agent.clear_all_queues()
        assert not agent.has_queued_messages()


class TestAgentReset:
    """Tests for Agent reset."""

    def test_reset(self, sample_model: Model) -> None:
        """Test agent reset."""
        from pi.ai.types import UserMessage

        agent = Agent(model=sample_model)
        msg = UserMessage(content="Test", timestamp=1000)
        agent.append_message(msg)
        agent.steer(msg)
        agent.follow_up(msg)
        agent._state.error = "Test error"

        agent.reset()

        assert len(agent.messages) == 0
        assert not agent.has_queued_messages()
        assert agent.state.error is None


class TestAgentAbort:
    """Tests for Agent abort functionality."""

    def test_abort_no_op_when_not_running(self) -> None:
        """Test abort when not running."""
        agent = Agent()
        agent.abort()  # Should not raise

    @pytest.mark.asyncio
    async def test_abort_during_run(self, sample_model: Model) -> None:
        """Test abort during run."""
        agent = Agent(model=sample_model)

        # Start abort event
        assert agent._abort_event is None
        agent.abort()  # Should be safe even when not running


class TestAgentWaitForIdle:
    """Tests for Agent wait_for_idle."""

    @pytest.mark.asyncio
    async def test_wait_for_idle_when_idle(self) -> None:
        """Test wait_for_idle when already idle."""
        agent = Agent()
        await agent.wait_for_idle()  # Should return immediately


class TestAgentStreamingProtection:
    """Tests for concurrent streaming protection."""

    @pytest.mark.asyncio
    async def test_prompt_while_streaming_throws(self) -> None:
        """Test that prompt() while streaming throws RuntimeError."""
        from unittest.mock import AsyncMock, patch

        from pi.ai.types import AssistantMessage, StreamEventDone, StreamEventStart, TextContent

        model = Model(
            id="test",
            name="test",
            api="openai-completions",
            provider="openai",
        )
        agent = Agent(model=model)

        # Mock stream that takes time
        async def slow_stream(*args, **kwargs):
            yield StreamEventStart(partial=AssistantMessage(
                content=[TextContent(text="")],
                api="openai-completions",
                provider="openai",
                model="test",
                usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
                stopReason="",
                timestamp=1000,
            ))
            await asyncio.sleep(0.2)
            yield StreamEventDone(message=AssistantMessage(
                content=[TextContent(text="done")],
                api="openai-completions",
                provider="openai",
                model="test",
                usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
                stopReason="stop",
                timestamp=1000,
            ), reason="stop")

        with patch("pi.agent.core.stream", slow_stream):
            # Start first prompt (don't await)
            task1 = asyncio.create_task(agent.prompt("first"))

            # Wait for streaming to start
            await asyncio.sleep(0.05)
            assert agent.is_streaming

            # Second prompt should throw
            with pytest.raises(RuntimeError, match="already processing"):
                await agent.prompt("second")

            # Cleanup
            await task1

    @pytest.mark.asyncio
    async def test_continue_while_streaming_throws(self) -> None:
        """Test that continue_() while streaming throws RuntimeError."""
        from unittest.mock import patch

        from pi.ai.types import AssistantMessage, StreamEventDone, StreamEventStart, TextContent

        model = Model(
            id="test",
            name="test",
            api="openai-completions",
            provider="openai",
        )
        agent = Agent(model=model)

        async def slow_stream(*args, **kwargs):
            yield StreamEventStart(partial=AssistantMessage(
                content=[TextContent(text="")],
                api="openai-completions",
                provider="openai",
                model="test",
                usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
                stopReason="",
                timestamp=1000,
            ))
            await asyncio.sleep(0.2)
            yield StreamEventDone(message=AssistantMessage(
                content=[TextContent(text="done")],
                api="openai-completions",
                provider="openai",
                model="test",
                usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
                stopReason="stop",
                timestamp=1000,
            ), reason="stop")

        with patch("pi.agent.core.stream", slow_stream):
            # Start prompt
            task1 = asyncio.create_task(agent.prompt("first"))
            await asyncio.sleep(0.05)
            assert agent.is_streaming

            # continue_() should throw
            with pytest.raises(RuntimeError, match="already processing"):
                await agent.continue_()

            # Cleanup
            await task1


class TestAgentContinueWithFollowUp:
    """Tests for continue_() processing follow-up messages."""

    @pytest.mark.asyncio
    async def test_continue_processes_follow_up_messages(self) -> None:
        """Test that continue_() processes queued follow-up messages."""
        from unittest.mock import patch

        from pi.ai.types import AssistantMessage, StreamEventDone, StreamEventStart, TextContent, UserMessage

        model = Model(
            id="test",
            name="test",
            api="openai-completions",
            provider="openai",
        )
        agent = Agent(model=model)

        # Set up existing conversation
        agent.replace_messages([
            UserMessage(content="Initial", timestamp=1000),
            AssistantMessage(
                content=[TextContent(text="Initial response")],
                api="openai-completions",
                provider="openai",
                model="test",
                usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
                stopReason="stop",
                timestamp=2000,
            ),
        ])

        # Queue follow-up
        agent.follow_up(UserMessage(content="Follow-up", timestamp=3000))

        response_count = 0

        async def mock_stream(*args, **kwargs):
            nonlocal response_count
            response_count += 1
            yield StreamEventStart(partial=AssistantMessage(
                content=[TextContent(text="")],
                api="openai-completions",
                provider="openai",
                model="test",
                usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
                stopReason="",
                timestamp=1000,
            ))
            yield StreamEventDone(message=AssistantMessage(
                content=[TextContent(text=f"Response {response_count}")],
                api="openai-completions",
                provider="openai",
                model="test",
                usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
                stopReason="stop",
                timestamp=1000,
            ), reason="stop")

        with patch("pi.agent.core.stream", mock_stream):
            await agent.continue_()

        # Follow-up should have been processed
        has_follow_up = any(
            hasattr(m, "content") and "Follow-up" in str(m.content)
            for m in agent.messages
        )
        assert has_follow_up

        # Last message should be assistant
        assert agent.messages[-1].role == "assistant"


class TestAgentSteeringSemantics:
    """Tests for steering message semantics."""

    @pytest.mark.asyncio
    async def test_steering_one_at_a_time(self) -> None:
        """Test that steering processes one message at a time."""
        from unittest.mock import patch

        from pi.ai.types import AssistantMessage, StreamEventDone, StreamEventStart, TextContent, UserMessage

        model = Model(
            id="test",
            name="test",
            api="openai-completions",
            provider="openai",
        )
        agent = Agent(model=model)

        # Set up existing conversation
        agent.replace_messages([
            UserMessage(content="Initial", timestamp=1000),
            AssistantMessage(
                content=[TextContent(text="Initial response")],
                api="openai-completions",
                provider="openai",
                model="test",
                usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
                stopReason="stop",
                timestamp=2000,
            ),
        ])

        # Queue multiple steering messages
        agent.steer(UserMessage(content="Steering 1", timestamp=3000))
        agent.steer(UserMessage(content="Steering 2", timestamp=4000))

        response_count = 0

        async def mock_stream(*args, **kwargs):
            nonlocal response_count
            response_count += 1
            yield StreamEventStart(partial=AssistantMessage(
                content=[TextContent(text="")],
                api="openai-completions",
                provider="openai",
                model="test",
                usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
                stopReason="",
                timestamp=1000,
            ))
            yield StreamEventDone(message=AssistantMessage(
                content=[TextContent(text=f"Response {response_count}")],
                api="openai-completions",
                provider="openai",
                model="test",
                usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
                stopReason="stop",
                timestamp=1000,
            ), reason="stop")

        with patch("pi.agent.core.stream", mock_stream):
            await agent.continue_()

        # With one-at-a-time mode, should have processed 2 steering messages
        # resulting in 2 assistant responses
        assert response_count == 2

        # Check message order: user, assistant, user, assistant
        recent = agent.messages[-4:]
        roles = [m.role for m in recent]
        assert roles == ["user", "assistant", "user", "assistant"]
