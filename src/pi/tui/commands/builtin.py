"""Built-in slash commands for the TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .registry import CommandRegistry, SlashCommand

if TYPE_CHECKING:
    from pi.tui.app import InteractiveApp


async def _help_handler(_args: str, app: InteractiveApp) -> str:
    lines = ["Available commands:"]
    for cmd in app.command_registry.list_commands():
        aliases_str = ""
        if cmd.aliases:
            aliases_str = f" (aliases: {', '.join(cmd.aliases)})"
        lines.append(f"  /{cmd.name} - {cmd.description}{aliases_str}")
    return "\n".join(lines)


async def _model_handler(args: str, app: InteractiveApp) -> str:
    if not args.strip():
        current = app.session.model or "default"
        return f"Current model: {current}"
    model_name = args.strip()
    app.session.set_model(model_name)
    return f"Model set to: {model_name}"


async def _clear_handler(_args: str, app: InteractiveApp) -> str:
    app.clear_messages()
    return "Chat cleared."


async def _quit_handler(_args: str, _app: InteractiveApp) -> None:
    raise SystemExit(0)


def create_builtin_commands() -> CommandRegistry:
    registry = CommandRegistry()

    registry.register(
        SlashCommand(
            name="help",
            description="Show available commands",
            handler=_help_handler,
            aliases=["?", "h"],
        )
    )

    registry.register(
        SlashCommand(
            name="model",
            description="Get or set current model",
            handler=_model_handler,
            aliases=["m"],
        )
    )

    registry.register(
        SlashCommand(
            name="clear",
            description="Clear chat history",
            handler=_clear_handler,
            aliases=["c"],
        )
    )

    registry.register(
        SlashCommand(
            name="quit",
            description="Exit the application",
            handler=_quit_handler,
            aliases=["q", "exit"],
        )
    )

    return registry
