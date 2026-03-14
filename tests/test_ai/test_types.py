"""Tests for pi.ai.types module."""

from pi.ai.types import (
    AssistantMessage,
    Context,
    Cost,
    ImageContent,
    Model,
    SimpleStreamOptions,
    StreamEventTextDelta,
    StreamOptions,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolParameter,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def test_text_content() -> None:
    """Test TextContent model."""
    content = TextContent(text="Hello, world!")
    assert content.type == "text"
    assert content.text == "Hello, world!"


def test_thinking_content() -> None:
    """Test ThinkingContent model."""
    content = ThinkingContent(thinking="Let me think...")
    assert content.type == "thinking"
    assert content.thinking == "Let me think..."


def test_image_content() -> None:
    """Test ImageContent model."""
    content = ImageContent(data="base64data", mimeType="image/png")
    assert content.type == "image"
    assert content.data == "base64data"
    assert content.mime_type == "image/png"


def test_tool_call() -> None:
    """Test ToolCall model."""
    tool_call = ToolCall(id="call_123", name="read", arguments={"file": "test.py"})
    assert tool_call.type == "toolCall"
    assert tool_call.id == "call_123"
    assert tool_call.name == "read"
    assert tool_call.arguments == {"file": "test.py"}


def test_user_message() -> None:
    """Test UserMessage model."""
    msg = UserMessage(content="Hello", timestamp=1000)
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_user_message_with_content_list() -> None:
    """Test UserMessage with content list."""
    msg = UserMessage(
        content=[TextContent(text="What's in this image?"), ImageContent(data="abc", mimeType="image/png")],
        timestamp=1000,
    )
    assert msg.role == "user"
    assert len(msg.content) == 2  # type: ignore[arg-type]


def test_assistant_message() -> None:
    """Test AssistantMessage model."""
    msg = AssistantMessage(
        content=[TextContent(text="Hello!")],
        api="openai-completions",
        provider="openai",
        model="gpt-4",
        usage=Usage(),
        stopReason="stop",
        timestamp=1000,
    )
    assert msg.role == "assistant"
    assert len(msg.content) == 1


def test_tool_result_message() -> None:
    """Test ToolResultMessage model."""
    msg = ToolResultMessage(
        toolCallId="call_123",
        toolName="read",
        content=[TextContent(text="file contents")],
        isError=False,
        timestamp=1000,
    )
    assert msg.role == "toolResult"
    assert msg.tool_call_id == "call_123"


def test_usage() -> None:
    """Test Usage model."""
    usage = Usage(input=100, output=50, totalTokens=150)
    assert usage.input == 100
    assert usage.output == 50
    assert usage.total_tokens == 150


def test_cost() -> None:
    """Test Cost model."""
    cost = Cost(input=0.01, output=0.03, total=0.04)
    assert cost.input == 0.01
    assert cost.output == 0.03
    assert cost.total == 0.04


def test_tool() -> None:
    """Test Tool model."""
    tool = Tool(name="read", description="Read a file", parameters=ToolParameter())
    assert tool.name == "read"
    assert tool.description == "Read a file"


def test_context() -> None:
    """Test Context model."""
    ctx = Context(systemPrompt="You are helpful.", messages=[])
    assert ctx.system_prompt == "You are helpful."
    assert len(ctx.messages) == 0


def test_model() -> None:
    """Test Model model."""
    model = Model(
        id="gpt-4",
        name="GPT-4",
        api="openai-completions",
        provider="openai",
        baseUrl="https://api.openai.com/v1",
    )
    assert model.id == "gpt-4"
    assert model.api == "openai-completions"


def test_stream_options() -> None:
    """Test StreamOptions model."""
    options = StreamOptions(temperature=0.7, maxTokens=1000)
    assert options.temperature == 0.7
    assert options.max_tokens == 1000


def test_simple_stream_options() -> None:
    """Test SimpleStreamOptions model."""
    options = SimpleStreamOptions(temperature=0.7, reasoning="high")
    assert options.temperature == 0.7
    assert options.reasoning == "high"


def test_stream_event_text_delta() -> None:
    """Test StreamEventTextDelta model."""
    event = StreamEventTextDelta(
        contentIndex=0,
        delta="Hello",
        partial=AssistantMessage(
            content=[],
            api="openai-completions",
            provider="openai",
            model="gpt-4",
            usage=Usage(),
            stopReason="stop",
            timestamp=1000,
        ),
    )
    assert event.type == "text_delta"
    assert event.delta == "Hello"
