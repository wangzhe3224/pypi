"""Tool registry for agent tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pi.agent.types import AgentTool
    from pi.ai.types import Tool


class ToolRegistry:
    """Registry for agent tools.

    Tools can be registered and retrieved by name. The registry can also
    convert all registered tools to LLM-compatible Tool definitions.

    Example:
        @tool
        def read(file_path: str) -> str:
            '''Read file contents.'''
            return open(file_path).read()

        # Later...
        all_tools = ToolRegistry.to_llm_tools()
    """

    _tools: dict[str, AgentTool] = {}

    @classmethod
    def register(cls, tool: AgentTool) -> None:
        """Register a tool.

        Args:
            tool: The AgentTool to register.
        """
        cls._tools[tool.name] = tool

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a tool by name.

        Args:
            name: The name of the tool to unregister.
        """
        cls._tools.pop(name, None)

    @classmethod
    def get(cls, name: str) -> AgentTool | None:
        """Get a tool by name.

        Args:
            name: The name of the tool.

        Returns:
            The AgentTool if found, None otherwise.
        """
        return cls._tools.get(name)

    @classmethod
    def list_tools(cls) -> list[AgentTool]:
        """List all registered tools.

        Returns:
            List of all registered AgentTools.
        """
        return list(cls._tools.values())

    @classmethod
    def list_names(cls) -> list[str]:
        """List all registered tool names.

        Returns:
            List of tool names.
        """
        return list(cls._tools.keys())

    @classmethod
    def to_llm_tools(cls) -> list[Tool]:
        """Convert all registered tools to LLM-compatible Tool definitions.

        Returns:
            List of Tool objects for LLM API calls.
        """
        return [tool.to_llm_tool() for tool in cls._tools.values()]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tools."""
        cls._tools.clear()

    @classmethod
    def contains(cls, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: The name of the tool.

        Returns:
            True if the tool is registered, False otherwise.
        """
        return name in cls._tools
