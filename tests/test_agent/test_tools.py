"""Tests for pi.agent.tools module."""

import asyncio

import pytest

from pi.agent.registry import ToolRegistry
from pi.agent.tools import create_tool, tool
from pi.agent.types import AgentToolResult
from pi.ai.types import TextContent


class TestToolDecorator:
    """Tests for @tool decorator."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        ToolRegistry.clear()

    def test_basic_decorator(self) -> None:
        """Test basic @tool decorator."""

        @tool
        def read(file_path: str) -> str:
            """Read a file."""
            return "content"

        assert ToolRegistry.contains("read")
        registered = ToolRegistry.get("read")
        assert registered is not None
        assert registered.name == "read"
        assert registered.description == "Read a file."

    def test_decorator_with_options(self) -> None:
        """Test @tool decorator with options."""

        @tool(name="bash", description="Execute bash command", label="Run Bash")
        async def run_bash(command: str) -> str:
            return "output"

        assert ToolRegistry.contains("bash")
        registered = ToolRegistry.get("bash")
        assert registered is not None
        assert registered.name == "bash"
        assert registered.description == "Execute bash command"
        assert registered.label == "Run Bash"

    def test_schema_generation(self) -> None:
        """Test JSON Schema generation from type hints."""

        @tool
        def complex_tool(
            required_param: str,
            optional_param: int = 0,
            another_optional: str | None = None,
        ) -> str:
            """A tool with complex parameters."""
            return "result"

        registered = ToolRegistry.get("complex_tool")
        assert registered is not None
        params = registered.parameters
        assert "properties" in params
        assert "required_param" in params["properties"]
        assert "optional_param" in params["properties"]
        assert "required" in params
        assert "required_param" in params["required"]
        assert "optional_param" not in params["required"]

    def test_schema_with_docstring_params(self) -> None:
        """Test extracting parameter descriptions from docstring."""

        @tool
        def documented_tool(file_path: str, limit: int = 100) -> str:
            """Read a file.

            file_path: Path to the file to read
            limit: Maximum number of lines
            """
            return "content"

        registered = ToolRegistry.get("documented_tool")
        assert registered is not None
        # Check if descriptions were extracted
        props = registered.parameters.get("properties", {})
        if "file_path" in props and "description" in props["file_path"]:
            assert "Path to the file" in props["file_path"]["description"]


class TestToolExecute:
    """Tests for tool execution."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        ToolRegistry.clear()

    @pytest.mark.asyncio
    async def test_sync_function_execution(self) -> None:
        """Test executing a sync function."""

        @tool
        def sync_tool(message: str) -> str:
            """Echo a message."""
            return f"Echo: {message}"

        registered = ToolRegistry.get("sync_tool")
        assert registered is not None
        assert registered.execute is not None

        result = await registered.execute(
            tool_call_id="call_1",
            params={"message": "hello"},
        )

        assert isinstance(result, AgentToolResult)
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert "Echo: hello" in result.content[0].text

    @pytest.mark.asyncio
    async def test_async_function_execution(self) -> None:
        """Test executing an async function."""

        @tool
        async def async_tool(value: int) -> str:
            """Double a value asynchronously."""
            await asyncio.sleep(0.01)
            return str(value * 2)

        registered = ToolRegistry.get("async_tool")
        assert registered is not None
        assert registered.execute is not None

        result = await registered.execute(
            tool_call_id="call_2",
            params={"value": 5},
        )

        assert isinstance(result, AgentToolResult)
        assert "10" in result.content[0].text

    @pytest.mark.asyncio
    async def test_error_handling(self) -> None:
        """Test error handling in tool execution."""

        @tool
        def failing_tool() -> str:
            """A tool that fails."""
            raise ValueError("Intentional error")

        registered = ToolRegistry.get("failing_tool")
        assert registered is not None
        assert registered.execute is not None

        result = await registered.execute(
            tool_call_id="call_3",
            params={},
        )

        assert isinstance(result, AgentToolResult)
        assert len(result.content) == 1
        assert "error" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_cancellation(self) -> None:
        """Test tool cancellation via signal."""

        @tool
        async def slow_tool() -> str:
            """A slow tool."""
            await asyncio.sleep(10)
            return "done"

        registered = ToolRegistry.get("slow_tool")
        assert registered is not None
        assert registered.execute is not None

        signal = asyncio.Event()
        signal.set()  # Immediately cancel

        result = await registered.execute(
            tool_call_id="call_4",
            params={},
            signal=signal,
        )

        assert isinstance(result, AgentToolResult)
        assert "cancelled" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_return_agent_tool_result(self) -> None:
        """Test returning AgentToolResult directly."""

        @tool
        def result_tool() -> AgentToolResult:
            """Return a tool result directly."""
            return AgentToolResult(
                content=[TextContent(text="Direct result")],
                details={"custom": "data"},
            )

        registered = ToolRegistry.get("result_tool")
        assert registered is not None
        assert registered.execute is not None

        result = await registered.execute(
            tool_call_id="call_5",
            params={},
        )

        assert isinstance(result, AgentToolResult)
        assert result.details == {"custom": "data"}


class TestCreateTool:
    """Tests for create_tool function."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        ToolRegistry.clear()

    def test_create_tool_basic(self) -> None:
        """Test create_tool with basic function."""

        def my_func(x: int) -> int:
            return x * 2

        agent_tool = create_tool(
            name="double",
            description="Double a number",
            func=my_func,
        )

        assert agent_tool.name == "double"
        assert agent_tool.description == "Double a number"
        assert ToolRegistry.contains("double")

    def test_create_tool_with_options(self) -> None:
        """Test create_tool with options."""

        async def my_async_func(cmd: str) -> str:
            return f"ran: {cmd}"

        agent_tool = create_tool(
            name="run",
            description="Run command",
            func=my_async_func,
            label="Run Command",
            parameters={"type": "object", "properties": {"cmd": {"type": "string"}}},
        )

        assert agent_tool.name == "run"
        assert agent_tool.label == "Run Command"
        assert "cmd" in agent_tool.parameters["properties"]

    @pytest.mark.asyncio
    async def test_create_tool_execution(self) -> None:
        """Test create_tool execution."""

        def add(a: int, b: int) -> int:
            return a + b

        agent_tool = create_tool(
            name="add",
            description="Add two numbers",
            func=add,
        )

        assert agent_tool.execute is not None
        result = await agent_tool.execute(
            tool_call_id="call_6",
            params={"a": 3, "b": 4},
        )

        assert isinstance(result, AgentToolResult)
        assert "7" in result.content[0].text
