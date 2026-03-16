"""High-level Agent class with state management and event subscription."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pi.agent.core import agent_loop, agent_loop_continue
from pi.agent.types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentOptions,
    AgentState,
    AgentTool,
    ConvertToLlmFn,
    ThinkingLevel,
    default_convert_to_llm,
)

if TYPE_CHECKING:
    from pi.ai.types import Model


class Agent:
    """High-level agent with state management and event subscription.

    This class provides a convenient API for:
    - Managing agent state (system prompt, model, tools, messages)
    - Subscribing to agent events
    - Running prompts and continuing conversations
    - Handling steering and follow-up messages

    Example:
        agent = Agent(
            model=gpt_4o,
            system_prompt="You are a helpful assistant.",
            tools=[read_tool, bash_tool],
        )

        # Subscribe to events
        def on_event(event: AgentEvent) -> None:
            print(f"Event: {event.type}")

        unsubscribe = agent.subscribe(on_event)

        # Run a prompt
        async for event in agent.prompt("What is 2+2?"):
            if event.type == "message_update":
                print(event.message.content, end="")

        # Clean up
        unsubscribe()
    """

    def __init__(
        self,
        model: Model | None = None,
        *,
        system_prompt: str = "",
        tools: list[AgentTool] | None = None,
        options: AgentOptions | None = None,
    ) -> None:
        """Initialize an Agent.

        Args:
            model: The LLM model to use.
            system_prompt: System prompt for the agent.
            tools: List of tools available to the agent.
            options: Additional agent options.
        """
        # Initialize state
        self._state = AgentState(
            system_prompt=system_prompt,
            model=model,
            thinking_level="off",
            tools=tools or [],
            messages=[],
            is_streaming=False,
            stream_message=None,
            pending_tool_calls=set(),
            error=None,
        )

        # Apply options
        if options and options.initial_state:
            self._state.system_prompt = options.initial_state.system_prompt
            self._state.model = options.initial_state.model
            self._state.thinking_level = options.initial_state.thinking_level
            self._state.tools = options.initial_state.tools

        # Event listeners
        self._listeners: set[Callable[[AgentEvent], None]] = set()

        # Control
        self._abort_event: asyncio.Event | None = None
        self._running_task: asyncio.Task[Any] | None = None

        # Message queues
        self._steering_queue: list[AgentMessage] = []
        self._follow_up_queue: list[AgentMessage] = []

        # Configuration
        self._convert_to_llm: ConvertToLlmFn = (
            options.convert_to_llm if options and options.convert_to_llm else default_convert_to_llm
        )
        self._steering_mode = options.steering_mode if options else "one-at-a-time"
        self._follow_up_mode = options.follow_up_mode if options else "one-at-a-time"
        self._session_id = options.session_id if options else None
        self._get_api_key = options.get_api_key if options else None

    # =========================================================================
    # State Accessors
    # =========================================================================

    @property
    def state(self) -> AgentState:
        """Get current agent state."""
        return self._state

    @property
    def is_streaming(self) -> bool:
        """Check if agent is currently streaming."""
        return self._state.is_streaming

    @property
    def messages(self) -> list[AgentMessage]:
        """Get all messages in the conversation."""
        return list(self._state.messages)

    # =========================================================================
    # Event Subscription
    # =========================================================================

    def subscribe(self, listener: Callable[[AgentEvent], None]) -> Callable[[], None]:
        """Subscribe to agent events.

        Args:
            listener: Callback function for events.

        Returns:
            Unsubscribe function.

        Example:
            def on_event(event: AgentEvent) -> None:
                if event.type == "message_update":
                    print(event.message.content)

            unsub = agent.subscribe(on_event)
            # Later...
            unsub()
        """
        self._listeners.add(listener)

        def unsubscribe() -> None:
            self._listeners.discard(listener)

        return unsubscribe

    def _emit(self, event: AgentEvent) -> None:
        """Emit an event to all listeners."""
        import contextlib

        for listener in self._listeners:
            with contextlib.suppress(Exception):
                listener(event)

    # =========================================================================
    # State Mutators
    # =========================================================================

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt."""
        self._state.system_prompt = prompt

    def set_model(self, model: Model) -> None:
        """Set the LLM model."""
        self._state.model = model

    def set_thinking_level(self, level: ThinkingLevel) -> None:
        """Set the thinking/reasoning level."""
        self._state.thinking_level = level

    def set_tools(self, tools: list[AgentTool]) -> None:
        """Set available tools."""
        self._state.tools = tools

    def add_tool(self, tool: AgentTool) -> None:
        """Add a tool to the available tools."""
        self._state.tools.append(tool)

    def remove_tool(self, name: str) -> None:
        """Remove a tool by name."""
        self._state.tools = [t for t in self._state.tools if t.name != name]

    def append_message(self, message: AgentMessage) -> None:
        """Append a message to the conversation."""
        self._state.messages.append(message)

    def replace_messages(self, messages: list[AgentMessage]) -> None:
        """Replace all messages in the conversation."""
        self._state.messages = list(messages)

    def clear_messages(self) -> None:
        """Clear all messages."""
        self._state.messages = []

    # =========================================================================
    # Message Queuing
    # =========================================================================

    def steer(self, message: AgentMessage) -> None:
        """Queue a steering message (interrupts mid-run).

        Steering messages are injected after the current tool execution
        completes, skipping any remaining tool calls.

        Args:
            message: The message to queue.
        """
        self._steering_queue.append(message)

    def follow_up(self, message: AgentMessage) -> None:
        """Queue a follow-up message (continues after stop).

        Follow-up messages are processed after the agent would normally stop.

        Args:
            message: The message to queue.
        """
        self._follow_up_queue.append(message)

    def clear_steering_queue(self) -> None:
        """Clear all steering messages."""
        self._steering_queue.clear()

    def clear_follow_up_queue(self) -> None:
        """Clear all follow-up messages."""
        self._follow_up_queue.clear()

    def clear_all_queues(self) -> None:
        """Clear both steering and follow-up queues."""
        self._steering_queue.clear()
        self._follow_up_queue.clear()

    def has_queued_messages(self) -> bool:
        """Check if there are any queued messages."""
        return len(self._steering_queue) > 0 or len(self._follow_up_queue) > 0

    async def _get_steering_messages(self) -> list[AgentMessage]:
        """Get and clear steering messages based on mode."""
        if self._steering_mode == "one-at-a-time" and self._steering_queue:
            msg = self._steering_queue[0]
            self._steering_queue = self._steering_queue[1:]
            return [msg]

        messages = list(self._steering_queue)
        self._steering_queue = []
        return messages

    async def _get_follow_up_messages(self) -> list[AgentMessage]:
        """Get and clear follow-up messages based on mode."""
        if self._follow_up_mode == "one-at-a-time" and self._follow_up_queue:
            msg = self._follow_up_queue[0]
            self._follow_up_queue = self._follow_up_queue[1:]
            return [msg]

        messages = list(self._follow_up_queue)
        self._follow_up_queue = []
        return messages

    # =========================================================================
    # Control Methods
    # =========================================================================

    async def prompt(
        self,
        input: str | AgentMessage | list[AgentMessage],
    ) -> list[AgentMessage]:
        """Send a prompt and run the agent loop.

        Args:
            input: The prompt - string, message, or list of messages.

        Returns:
            List of new messages generated.

        Raises:
            RuntimeError: If agent is already streaming.
        """
        if self._state.is_streaming:
            raise RuntimeError(
                "Agent is already processing. Use steer() or follow_up() to queue messages."
            )

        # Convert input to messages
        if isinstance(input, str):
            from time import time

            from pi.ai.types import UserMessage

            prompts: list[AgentMessage] = [
                UserMessage(
                    role="user",
                    content=input,
                    timestamp=int(time() * 1000),
                )
            ]
        elif isinstance(input, list):
            prompts = input
        else:
            prompts = [input]

        # Build context
        context = AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=self._state.tools,
        )

        # Build config
        config = self._build_config()

        # Run loop
        new_messages: list[AgentMessage] = []
        self._abort_event = asyncio.Event()
        self._state.is_streaming = True

        try:
            async for event in agent_loop(prompts, context, config, self._abort_event):
                self._emit(event)
                if event.type == "agent_end":
                    new_messages = event.messages
        finally:
            self._state.is_streaming = False
            self._abort_event = None
            self._state.messages = context.messages

        return new_messages

    async def continue_(self) -> list[AgentMessage]:
        """Continue from current context (for retries/queued messages).

        Returns:
            List of new messages generated.

        Raises:
            RuntimeError: If agent is already streaming or no messages to continue from.
        """
        if self._state.is_streaming:
            raise RuntimeError("Agent is already processing. Wait for completion.")

        if not self._state.messages:
            raise RuntimeError("No messages to continue from.")

        # Check for queued messages first
        if self._steering_queue:
            steering_messages = await self._get_steering_messages()
            return await self.prompt(steering_messages)

        if self._follow_up_queue:
            follow_up_messages = await self._get_follow_up_messages()
            return await self.prompt(follow_up_messages)

        # Build context
        context = AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=self._state.tools,
        )

        # Build config
        config = self._build_config()

        # Run loop
        new_messages: list[AgentMessage] = []
        self._abort_event = asyncio.Event()
        self._state.is_streaming = True

        try:
            async for event in agent_loop_continue(context, config, self._abort_event):
                self._emit(event)
                if event.type == "agent_end":
                    new_messages = event.messages
        finally:
            self._state.is_streaming = False
            self._abort_event = None
            self._state.messages = context.messages

        return new_messages

    def abort(self) -> None:
        """Abort current execution."""
        if self._abort_event:
            self._abort_event.set()

    async def wait_for_idle(self) -> None:
        """Wait for agent to finish current operation."""
        while self._state.is_streaming:
            await asyncio.sleep(0.1)

    def reset(self) -> None:
        """Reset agent state."""
        self._state.messages = []
        self._state.is_streaming = False
        self._state.stream_message = None
        self._state.pending_tool_calls = set()
        self._state.error = None
        self._steering_queue = []
        self._follow_up_queue = []

    def _build_config(self) -> AgentLoopConfig:
        """Build AgentLoopConfig from current state."""

        model = self._state.model
        if model is None:
            raise RuntimeError("No model configured")

        return AgentLoopConfig(
            model=model,
            convert_to_llm=self._convert_to_llm,
            temperature=0.7,
            max_tokens=4096,
            get_steering_messages=self._get_steering_messages,
            get_follow_up_messages=self._get_follow_up_messages,
        )
