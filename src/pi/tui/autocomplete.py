"""Autocomplete providers for pi TUI."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from prompt_toolkit.completion import Completer, Completion, WordCompleter
from prompt_toolkit.document import Document

if TYPE_CHECKING:
    from prompt_toolkit.completion import CompleteEvent

    from .commands.registry import CommandRegistry


class CommandCompleter(Completer):
    def __init__(self, registry: CommandRegistry) -> None:
        self._registry = registry

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,  # noqa: ARG002
    ) -> Iterable[Completion]:
        word_before_cursor = document.get_word_before_cursor()
        if not word_before_cursor.startswith("/"):
            return

        for cmd in self._registry.list_commands():
            cmd_text = f"/{cmd.name}"
            if cmd_text.startswith(word_before_cursor):
                yield Completion(
                    cmd_text,
                    start_position=-len(word_before_cursor),
                    display=cmd_text,
                    display_meta=cmd.description,
                )

            if cmd.aliases:
                for alias in cmd.aliases:
                    alias_text = f"/{alias}"
                    if alias_text.startswith(word_before_cursor):
                        yield Completion(
                            cmd_text,
                            start_position=-len(word_before_cursor),
                            display=alias_text,
                            display_meta=f"alias: {cmd.description}",
                        )


def create_completer(registry: CommandRegistry | None = None) -> Completer:
    completers: list[Completer] = []

    if registry:
        completers.append(CommandCompleter(registry))

    return WordCompleter([], ignore_case=True) if not completers else completers[0]
