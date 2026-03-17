"""Command registry for slash commands in the TUI."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pi.tui.app import InteractiveApp

CommandHandler = Callable[[str, "InteractiveApp"], Awaitable[str | None]]


@dataclass
class SlashCommand:
    """A slash command definition."""

    name: str
    description: str
    handler: CommandHandler
    aliases: list[str] | None = None


class CommandRegistry:
    """Registry for slash commands."""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}

    def register(self, command: SlashCommand) -> None:
        """Register a command and its aliases."""
        self._commands[command.name] = command
        if command.aliases:
            for alias in command.aliases:
                self._commands[alias] = command

    def get(self, name: str) -> SlashCommand | None:
        """Get command by name or alias."""
        return self._commands.get(name)

    def list_commands(self) -> list[SlashCommand]:
        """List all unique registered commands (no aliases)."""
        seen: set[str] = set()
        commands: list[SlashCommand] = []
        for cmd in self._commands.values():
            if cmd.name not in seen:
                seen.add(cmd.name)
                commands.append(cmd)
        return commands
