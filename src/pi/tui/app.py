"""Main TUI application for pi-ai interactive mode."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console

from pi.agent.types import AgentEvent
from pi.session.agent_session import AgentSession
from pi.tui.commands.builtin import create_builtin_commands
from pi.tui.components.message import ChatMessage, MessageRole

if TYPE_CHECKING:
    from pi.tui.commands.registry import CommandRegistry


class InteractiveApp:
    _ROLE_STYLES: dict[MessageRole, str] = {
        "user": "[bold blue]You[/]:",
        "assistant": "[bold green]Assistant[/]:",
        "system": "[bold yellow]System[/]:",
        "tool": "[bold magenta]Tool[/]:",
    }

    def __init__(self, session: AgentSession) -> None:
        self.session = session
        self.messages: list[ChatMessage] = []
        self.streaming = False
        self.command_registry: CommandRegistry = create_builtin_commands()
        self._console = Console()

        commands = [f"/{cmd.name}" for cmd in self.command_registry.list_commands()]
        for cmd in self.command_registry.list_commands():
            if cmd.aliases:
                commands.extend(f"/{alias}" for alias in cmd.aliases)
        self._completer = WordCompleter(commands, ignore_case=True)

        self._prompt_session: PromptSession[str] = PromptSession(
            completer=self._completer,
            multiline=False,
        )
        self._last_line_count = 0

    def add_message(self, role: MessageRole, content: str) -> None:
        self.messages.append(ChatMessage(role, content))

    def clear_messages(self) -> None:
        self.messages.clear()

    async def run(self) -> int:
        self._show_welcome()

        while True:
            try:
                with patch_stdout():
                    user_input = await self._prompt_session.prompt_async(">>> ")

                if not user_input.strip():
                    continue

                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                    continue

                await self._send_to_agent(user_input)

            except KeyboardInterrupt:
                if self.streaming:
                    self.streaming = False
                    self._console.print("\n[Interrupted]")
                continue
            except EOFError:
                break
            except SystemExit as e:
                code = e.code
                if isinstance(code, int):
                    return code
                return 0

        self._console.print("Goodbye!")
        return 0

    async def _handle_command(self, input_text: str) -> None:
        parts = input_text[1:].split(maxsplit=1)
        cmd_name = parts[0]
        args = parts[1] if len(parts) > 1 else ""

        command = self.command_registry.get(cmd_name)
        if command:
            try:
                result = await command.handler(args, self)
                if result:
                    self._console.print(result)
            except SystemExit:
                raise
        else:
            self._console.print(f"[red]Unknown command: {cmd_name}[/]")

    async def _send_to_agent(self, text: str) -> None:
        self._console.print(f"[bold blue]You[/]: {text}")
        self.streaming = True
        self._last_line_count = 0

        streaming_msg = ChatMessage("assistant", "")
        self.messages.append(streaming_msg)

        try:
            async for event in self.session.prompt(text):
                await self._handle_agent_event(event, streaming_msg)
        finally:
            self.streaming = False
            self._console.print()

    async def _handle_agent_event(self, event: AgentEvent, streaming_msg: ChatMessage) -> None:
        if event.type == "message_update":
            delta = getattr(event.stream_event, "delta", None)
            if delta:
                streaming_msg.content += delta
                streaming_msg.markdown.set_text(streaming_msg.content)
                self._render_streaming_message(streaming_msg)

        elif event.type == "tool_execution_start":
            self._console.print(f"[dim][Running: {event.tool_name}][/]")
            self._last_line_count = 0

        elif event.type == "tool_execution_end" and event.is_error:
            self._console.print(f"[red][Error: {event.tool_name}][/]")
            self._last_line_count = 0

    def _render_streaming_message(self, msg: ChatMessage) -> None:
        if self._last_line_count > 0:
            sys.stdout.write(f"\033[{self._last_line_count}F")
        lines = msg.render(80)
        self._last_line_count = len(lines)
        for line in lines:
            self._console.print(line, end="\n")

    def _show_welcome(self) -> None:
        self._console.print("[bold]pi-ai Interactive Mode[/]")
        self._console.print("Type /help for available commands.\n")


async def run_interactive(session: AgentSession) -> int:
    app = InteractiveApp(session)
    return await app.run()
