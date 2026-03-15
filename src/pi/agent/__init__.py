"""Agent runtime with tool calling and state management.

This module provides:
- Agent: High-level agent class with state management
- agent_loop / agent_loop_continue: Low-level async generators
- @tool decorator: Register functions as tools
- Types: AgentMessage, AgentTool, AgentEvent, etc.

Example:
    from pi.agent import Agent, tool

    @tool
    def read(file_path: str) -> str:
        '''Read file contents.'''
        with open(file_path) as f:
            return f.read()

    agent = Agent(model=gpt_4o, tools=[read])

    async for event in agent.prompt("Read the README.md file"):
        if event.type == "message_update":
            print(event.message.content, end="")
"""

from __future__ import annotations

from pi.agent.agent import Agent
from pi.agent.core import agent_loop, agent_loop_continue
from pi.agent.registry import ToolRegistry
from pi.agent.tools import create_tool, tool
from pi.agent.types import (
    AgentContext,
    AgentEvent,
    AgentEventAgentEnd,
    AgentEventAgentStart,
    AgentEventMessageEnd,
    AgentEventMessageStart,
    AgentEventMessageUpdate,
    AgentEventToolExecutionEnd,
    AgentEventToolExecutionStart,
    AgentEventTurnEnd,
    AgentEventTurnStart,
    AgentLoopConfig,
    AgentMessage,
    AgentOptions,
    AgentState,
    AgentTool,
    AgentToolResult,
    ThinkingLevel,
    default_convert_to_llm,
)

__all__ = [
    # Agent class
    "Agent",
    # Loop functions
    "agent_loop",
    "agent_loop_continue",
    # Tool utilities
    "tool",
    "create_tool",
    "ToolRegistry",
    # Types
    "AgentMessage",
    "AgentContext",
    "AgentState",
    "AgentTool",
    "AgentToolResult",
    "AgentEvent",
    "AgentEventAgentStart",
    "AgentEventAgentEnd",
    "AgentEventTurnStart",
    "AgentEventTurnEnd",
    "AgentEventMessageStart",
    "AgentEventMessageUpdate",
    "AgentEventMessageEnd",
    "AgentEventToolExecutionStart",
    "AgentEventToolExecutionEnd",
    "AgentLoopConfig",
    "AgentOptions",
    "ThinkingLevel",
    # Utilities
    "default_convert_to_llm",
]
