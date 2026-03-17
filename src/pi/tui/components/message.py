"""Chat message component for TUI."""

from __future__ import annotations

from typing import Literal

from .markdown import MarkdownComponent

MessageRole = Literal["user", "assistant", "system", "tool"]


class ChatMessage:
    """A single chat message with role indicator and markdown content."""

    def __init__(self, role: MessageRole, content: str):
        """Initialize chat message.

        Args:
            role: Message role (user, assistant, system, tool).
            content: Message content as markdown text.
        """
        self.role: MessageRole = role
        self.content: str = content
        self.markdown: MarkdownComponent = MarkdownComponent(content)

    def render(self, width: int) -> list[str]:
        """Render message with role prefix and indented content.

        Args:
            width: Terminal width for rendering.

        Returns:
            List of ANSI-styled strings, one per line.
        """
        lines: list[str] = []

        # Role indicator with styling
        role_styles: dict[MessageRole, str] = {
            "user": "[bold blue]You[/]",
            "assistant": "[bold green]Assistant[/]",
            "system": "[bold yellow]System[/]",
            "tool": "[bold magenta]Tool[/]",
        }
        lines.append(f"{role_styles[self.role]}:")

        # Render markdown content with reduced width for indentation
        content_width = max(1, width - 2)
        content_lines = self.markdown.render(content_width)

        # Indent content lines by 2 spaces
        for line in content_lines:
            lines.append(f"  {line}")

        # Add blank line after message
        lines.append("")

        return lines
