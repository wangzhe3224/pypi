"""Tests for pi.agent.registry module."""

import pytest

from pi.agent.registry import ToolRegistry
from pi.agent.types import AgentTool


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        ToolRegistry.clear()

    def test_register_tool(self) -> None:
        """Test registering a tool."""
        tool = AgentTool(name="test", description="Test tool")
        ToolRegistry.register(tool)
        assert ToolRegistry.contains("test")

    def test_unregister_tool(self) -> None:
        """Test unregistering a tool."""
        tool = AgentTool(name="test", description="Test tool")
        ToolRegistry.register(tool)
        ToolRegistry.unregister("test")
        assert not ToolRegistry.contains("test")

    def test_unregister_nonexistent(self) -> None:
        """Test unregistering a nonexistent tool doesn't error."""
        ToolRegistry.unregister("nonexistent")  # Should not raise

    def test_get_tool(self) -> None:
        """Test getting a tool by name."""
        tool = AgentTool(name="read", description="Read a file")
        ToolRegistry.register(tool)
        retrieved = ToolRegistry.get("read")
        assert retrieved is not None
        assert retrieved.name == "read"

    def test_get_nonexistent_tool(self) -> None:
        """Test getting a nonexistent tool returns None."""
        assert ToolRegistry.get("nonexistent") is None

    def test_list_tools(self) -> None:
        """Test listing all tools."""
        tool1 = AgentTool(name="read", description="Read")
        tool2 = AgentTool(name="write", description="Write")
        ToolRegistry.register(tool1)
        ToolRegistry.register(tool2)
        tools = ToolRegistry.list_tools()
        assert len(tools) == 2
        names = [t.name for t in tools]
        assert "read" in names
        assert "write" in names

    def test_list_names(self) -> None:
        """Test listing tool names."""
        tool1 = AgentTool(name="read", description="Read")
        tool2 = AgentTool(name="write", description="Write")
        ToolRegistry.register(tool1)
        ToolRegistry.register(tool2)
        names = ToolRegistry.list_names()
        assert len(names) == 2
        assert "read" in names
        assert "write" in names

    def test_to_llm_tools(self) -> None:
        """Test converting to LLM tools."""
        tool = AgentTool(
            name="bash",
            description="Run command",
            parameters={"type": "object", "properties": {"cmd": {"type": "string"}}},
        )
        ToolRegistry.register(tool)
        llm_tools = ToolRegistry.to_llm_tools()
        assert len(llm_tools) == 1
        assert llm_tools[0].name == "bash"

    def test_to_llm_tools_empty(self) -> None:
        """Test converting empty registry."""
        ToolRegistry.clear()
        llm_tools = ToolRegistry.to_llm_tools()
        assert len(llm_tools) == 0

    def test_clear(self) -> None:
        """Test clearing the registry."""
        tool = AgentTool(name="test", description="Test")
        ToolRegistry.register(tool)
        ToolRegistry.clear()
        assert not ToolRegistry.contains("test")
        assert len(ToolRegistry.list_tools()) == 0

    def test_contains(self) -> None:
        """Test checking if tool exists."""
        tool = AgentTool(name="test", description="Test")
        ToolRegistry.register(tool)
        assert ToolRegistry.contains("test")
        assert not ToolRegistry.contains("other")

    def test_overwrite_tool(self) -> None:
        """Test overwriting an existing tool."""
        tool1 = AgentTool(name="test", description="Version 1")
        ToolRegistry.register(tool1)
        tool2 = AgentTool(name="test", description="Version 2")
        ToolRegistry.register(tool2)
        retrieved = ToolRegistry.get("test")
        assert retrieved is not None
        assert retrieved.description == "Version 2"
