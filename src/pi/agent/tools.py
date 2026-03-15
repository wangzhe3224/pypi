"""Tool decorator and utilities for creating agent tools."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, ParamSpec, TypeVar, get_type_hints, overload

from pydantic import TypeAdapter

from pi.agent.registry import ToolRegistry
from pi.agent.types import AgentTool, AgentToolResult

P = ParamSpec("P")
T = TypeVar("T")


def _generate_json_schema(func: Callable[P, T]) -> dict[str, Any]:
    """Generate JSON Schema from function signature.

    Uses pydantic's TypeAdapter to generate proper JSON Schema
    from type hints.

    Args:
        func: The function to generate schema for.

    Returns:
        JSON Schema dict with 'type', 'properties', and 'required'.
    """
    hints = get_type_hints(func, include_extras=True)
    sig = inspect.signature(func)

    # Remove return type from hints
    hints.pop("return", None)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        param_type = hints.get(param_name, Any)

        # Use pydantic to generate schema for this type
        try:
            adapter = TypeAdapter(param_type)
            schema = adapter.json_schema()
            properties[param_name] = schema
        except Exception:
            # Fallback for complex types
            properties[param_name] = {
                "type": "string",
                "description": f"Parameter {param_name}",
            }

        # Add description from docstring if available
        if func.__doc__:
            # Parse parameter descriptions from docstring
            # Format: "param_name: description" or Args: section
            for line in func.__doc__.split("\n"):
                line = line.strip()
                if line.lower().startswith(f"{param_name}:"):
                    desc = line.split(":", 1)[1].strip()
                    properties[param_name]["description"] = desc
                    break

        # Check if parameter is required (no default value)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _extract_description(func: Callable[P, T]) -> str:
    """Extract description from function docstring.

    Args:
        func: The function to extract description from.

    Returns:
        The first line of the docstring, or function name if no docstring.
    """
    if func.__doc__:
        # Get first non-empty line
        for line in func.__doc__.strip().split("\n"):
            line = line.strip()
            if line:
                return line
    return f"Tool: {func.__name__}"


def _wrap_execute(
    func: Callable[P, T] | Callable[P, Coroutine[Any, Any, T]],
) -> Callable[..., Coroutine[Any, Any, AgentToolResult]]:
    """Wrap a function to be used as a tool execute method.

    Handles:
    - Sync/async function detection
    - Wrapping sync functions with asyncio.to_thread
    - Converting return value to AgentToolResult

    Args:
        func: The function to wrap.

    Returns:
        Async function compatible with AgentTool.execute signature.
    """
    from pi.ai.types import TextContent

    is_async = asyncio.iscoroutinefunction(func)

    @wraps(func)
    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Callable[[AgentToolResult], None] | None = None,
    ) -> AgentToolResult:
        _ = tool_call_id  # Part of API signature, not used in basic execution
        _ = on_update  # Part of API signature, not used in basic execution
        # Check for cancellation
        if signal and signal.is_set():
            return AgentToolResult(
                content=[TextContent(text="Tool execution was cancelled")],
                details={"cancelled": True},
            )

        try:
            # Execute the function
            if is_async:
                result = await func(**params)  # type: ignore[call-arg,misc]
            else:
                # Run sync function in thread pool
                result = await asyncio.to_thread(func, **params)
            # Convert result to AgentToolResult
            if isinstance(result, AgentToolResult):
                return result

            # Auto-convert string/other types to text content
            if isinstance(result, str):
                content = [TextContent(text=result)]
            elif isinstance(result, dict) and "content" in result:
                # Already in content format - convert to TextContent if needed
                raw_content = result["content"]
                if isinstance(raw_content, list) and all(isinstance(c, dict) for c in raw_content):
                    content = [TextContent(**c) if isinstance(c, dict) else c for c in raw_content]
                else:
                    content = raw_content
            elif isinstance(result, list):
                # Assume list of content blocks - convert dicts to TextContent
                content = [
                    TextContent(**item) if isinstance(item, dict) else item
                    for item in result
                ]
            else:
                # Convert to string representation
                content = [TextContent(text=str(result))]

            return AgentToolResult(content=content, details={"raw_result": result})

        except Exception as e:
            # Return error as tool result
            error_msg = f"Error executing tool: {e}"
            return AgentToolResult(
                content=[TextContent(text=error_msg)],
                details={"error": str(e), "error_type": type(e).__name__},
            )

    return execute
@overload
def tool(func: Callable[P, T]) -> AgentTool:
    ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    label: str | None = None,
) -> Callable[[Callable[P, T]], AgentTool]:
    ...


def tool(
    func: Callable[P, T] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    label: str | None = None,
) -> AgentTool | Callable[[Callable[P, T]], AgentTool]:
    """Decorator to register a function as an agent tool.

    Automatically:
    - Generates JSON Schema from type hints
    - Wraps sync functions with asyncio.to_thread
    - Registers to ToolRegistry

    Usage:
        @tool
        def read(file_path: str, offset: int = 0, limit: int = 2000) -> str:
            '''Read file contents.

            Args:
                file_path: Path to the file to read
                offset: Line number to start from
                limit: Maximum number of lines to read
            '''
            with open(file_path) as f:
                lines = f.readlines()
                return ''.join(lines[offset:offset + limit])

        @tool(name="bash", description="Execute shell commands")
        async def run_bash(command: str) -> str:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            return stdout.decode() or stderr.decode()

    Args:
        func: The function to wrap (when used without arguments).
        name: Optional custom name for the tool (defaults to function name).
        description: Optional custom description (defaults to docstring).
        label: Optional human-readable label for UI.

    Returns:
        AgentTool instance, or decorator function if called with arguments.
    """

    def decorator(fn: Callable[P, T]) -> AgentTool:
        # Generate tool properties
        tool_name = name or fn.__name__
        tool_description = description or _extract_description(fn)
        tool_label = label or tool_name.replace("_", " ").title()
        parameters = _generate_json_schema(fn)

        # Wrap the function for execution
        execute_fn = _wrap_execute(fn)

        # Create the AgentTool
        agent_tool = AgentTool(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            label=tool_label,
        )
        agent_tool.execute = execute_fn

        # Register to global registry
        ToolRegistry.register(agent_tool)

        return agent_tool

    # Handle both @tool and @tool(...) syntax
    if func is not None:
        return decorator(func)
    return decorator


def create_tool(
    name: str,
    description: str,
    func: Callable[P, T],
    *,
    label: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> AgentTool:
    """Create an AgentTool from a function with explicit configuration.

    Use this when you need more control over tool creation than @tool provides.

    Args:
        name: Tool name.
        description: Tool description.
        func: The function to wrap.
        label: Optional human-readable label.
        parameters: Optional custom JSON Schema parameters.

    Returns:
        Configured AgentTool instance.
    """
    tool_label = label or name.replace("_", " ").title()
    tool_params = parameters or _generate_json_schema(func)
    execute_fn = _wrap_execute(func)

    agent_tool = AgentTool(
        name=name,
        description=description,
        parameters=tool_params,
        label=tool_label,
    )
    agent_tool.execute = execute_fn

    ToolRegistry.register(agent_tool)

    return agent_tool
