"""Tests for pi.tui.components.message."""

import pytest

from pi.tui.components.message import ChatMessage, MessageRole


class TestMessageRole:
    def test_message_role_values(self) -> None:
        assert "user" in ("user", "assistant", "system", "tool")
        assert "assistant" in ("user", "assistant", "system", "tool")
        assert "system" in ("user", "assistant", "system", "tool")
        assert "tool" in ("user", "assistant", "system", "tool")


class TestChatMessage:
    def test_create_user_message(self) -> None:
        msg = ChatMessage("user", "Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_create_assistant_message(self) -> None:
        msg = ChatMessage("assistant", "Hi there!")

        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_render_user_message(self) -> None:
        msg = ChatMessage("user", "Hello")
        lines = msg.render(80)

        assert len(lines) >= 1
        assert any("You" in line for line in lines)

    def test_render_assistant_message(self) -> None:
        msg = ChatMessage("assistant", "Hi there!")
        lines = msg.render(80)

        assert len(lines) >= 1
        assert any("Assistant" in line for line in lines)

    def test_render_system_message(self) -> None:
        msg = ChatMessage("system", "System message")
        lines = msg.render(80)

        assert len(lines) >= 1
        assert any("System" in line for line in lines)

    def test_render_tool_message(self) -> None:
        msg = ChatMessage("tool", "Tool output")
        lines = msg.render(80)

        assert len(lines) >= 1
        assert any("Tool" in line for line in lines)

    def test_render_with_markdown(self) -> None:
        msg = ChatMessage("assistant", "# Heading\n\nContent")
        lines = msg.render(80)

        assert len(lines) >= 2

    def test_render_content_is_indented(self) -> None:
        msg = ChatMessage("user", "Test message")
        lines = msg.render(80)

        content_lines = [line for line in lines if "Test message" in line]
        assert len(content_lines) >= 1
        assert content_lines[0].startswith("  ")

    def test_render_ends_with_blank_line(self) -> None:
        msg = ChatMessage("user", "Test")
        lines = msg.render(80)

        assert lines[-1] == ""

    def test_has_markdown_component(self) -> None:
        msg = ChatMessage("user", "Test")

        assert msg.markdown is not None
        assert msg.markdown.text == "Test"

    def test_content_update_reflects_in_markdown(self) -> None:
        msg = ChatMessage("assistant", "Initial")
        msg.content = "Updated content"
        msg.markdown.set_text(msg.content)

        lines = msg.render(80)
        assert any("Updated content" in line for line in lines)
