"""Tests for Google message converters."""


from pi.ai.providers.converters_google import (
    convert_messages,
    convert_tools,
    map_stop_reason,
)
from pi.ai.types import (
    Context,
    Model,
    ModelCost,
    TextContent,
    Tool,
    ToolParameter,
    ToolResultMessage,
    UserMessage,
)


def make_model() -> Model:
    """Create a test model."""
    return Model(
        id="gemini-2.5-pro-preview-06-05",
        name="Gemini 2.5 Pro",
        api="google-generative-ai",
        provider="google",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        cost=ModelCost(input=1.25, output=10.0),
        context_window=1000000,
        max_tokens=65536,
    )


class TestConvertMessages:
    """Tests for convert_messages function."""

    def test_simple_user_message(self) -> None:
        """Test converting a simple user message."""
        model = make_model()
        context = Context(messages=[UserMessage(content="Hello", timestamp=1000)])
        result = convert_messages(model, context)
        assert len(result) == 1
        assert result[0] == {"role": "user", "parts": [{"text": "Hello"}]}

    def test_empty_user_message_skipped(self) -> None:
        """Test that empty user messages are skipped."""
        model = make_model()
        context = Context(messages=[UserMessage(content="   ", timestamp=1000)])
        result = convert_messages(model, context)
        assert len(result) == 0

    def test_user_message_with_content_list(self) -> None:
        """Test converting user message with content list."""
        model = make_model()
        context = Context(
            messages=[
                UserMessage(
                    content=[TextContent(text="Hello world")],
                    timestamp=1000,
                )
            ]
        )
        result = convert_messages(model, context)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["parts"] == [{"text": "Hello world"}]

    def test_tool_result_message(self) -> None:
        """Test converting tool result message."""
        model = make_model()
        context = Context(
            messages=[
                ToolResultMessage(
                    tool_call_id="call_123",
                    tool_name="get_weather",
                    content=[TextContent(text="Sunny, 25°C")],
                    is_error=False,
                    timestamp=1001,
                ),
            ]
        )
        result = convert_messages(model, context)
        assert len(result) == 1
        tool_result = result[0]
        assert tool_result["role"] == "user"
        # Find function response part
        func_resp_part = None
        for part in tool_result["parts"]:
            if "functionResponse" in part:
                func_resp_part = part
                break
        assert func_resp_part is not None
        assert func_resp_part["functionResponse"]["name"] == "get_weather"

    def test_tool_result_message_with_error(self) -> None:
        """Test converting tool result message with error."""
        model = make_model()
        context = Context(
            messages=[
                ToolResultMessage(
                    tool_call_id="call_123",
                    tool_name="get_weather",
                    content=[TextContent(text="Error occurred")],
                    is_error=True,
                    timestamp=1001,
                ),
            ]
        )
        result = convert_messages(model, context)
        # Should still have the function response
        assert len(result) == 1


class TestConvertTools:
    """Tests for convert_tools function."""

    def test_simple_tool(self) -> None:
        """Test converting a simple tool."""
        tools = [
            Tool(
                name="get_weather",
                description="Get weather info",
                parameters=ToolParameter(
                    properties={"city": {"type": "string"}},
                    required=["city"],
                ),
            )
        ]
        result = convert_tools(tools)
        assert len(result) == 1
        func_decl = result[0]["functionDeclarations"][0]
        assert func_decl["name"] == "get_weather"
        assert func_decl["description"] == "Get weather info"

    def test_multiple_tools(self) -> None:
        """Test converting multiple tools."""
        tools = [
            Tool(
                name="tool1",
                description="First tool",
                parameters=ToolParameter(properties={}, required=[]),
            ),
            Tool(
                name="tool2",
                description="Second tool",
                parameters=ToolParameter(properties={}, required=[]),
            ),
        ]
        result = convert_tools(tools)
        assert len(result) == 1
        assert len(result[0]["functionDeclarations"]) == 2

    def test_tool_with_no_parameters(self) -> None:
        """Test tool with default parameters."""
        tools = [
            Tool(
                name="simple_tool",
                description="A simple tool",
            )
        ]
        result = convert_tools(tools)
        assert len(result) == 1
        func_decl = result[0]["functionDeclarations"][0]
        assert func_decl["name"] == "simple_tool"


class TestMapStopReason:
    """Tests for map_stop_reason function."""

    def test_stop(self) -> None:
        """Test STOP maps to stop."""
        assert map_stop_reason("STOP") == "stop"

    def test_max_tokens(self) -> None:
        """Test MAX_TOKENS maps to length."""
        assert map_stop_reason("MAX_TOKENS") == "length"

    def test_tool_use(self) -> None:
        """Test that TOOL_USE is not a valid Google finish reason."""
        # Google doesn't use TOOL_USE, it uses other mechanisms
        assert map_stop_reason("TOOL_USE") == "error"

    def test_safety(self) -> None:
        """Test SAFETY maps to error."""
        assert map_stop_reason("SAFETY") == "error"

    def test_recitation(self) -> None:
        """Test RECITATION maps to error."""
        assert map_stop_reason("RECITATION") == "error"

    def test_unknown(self) -> None:
        """Test unknown reason maps to error."""
        assert map_stop_reason("UNKNOWN") == "error"

    def test_none(self) -> None:
        """Test None maps to stop."""
        assert map_stop_reason(None) == "stop"

    def test_case_insensitive(self) -> None:
        """Test that matching requires uppercase."""
        # Google finish reasons are uppercase
        assert map_stop_reason("stop") == "error"  # lowercase not recognized
        assert map_stop_reason("STOP") == "stop"  # uppercase works
