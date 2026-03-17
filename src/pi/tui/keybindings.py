"""Key bindings for pi TUI."""

from __future__ import annotations

from enum import StrEnum

from prompt_toolkit.key_binding import KeyBindings


class EditorAction(StrEnum):
    SUBMIT = "submit"
    INTERRUPT = "interrupt"
    EXIT = "exit"
    CLEAR = "clear"
    NEWLINE = "newline"
    HISTORY_UP = "history_up"
    HISTORY_DOWN = "history_down"


def create_keybindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add("c-c")
    def _interrupt(_event: object) -> None:
        raise KeyboardInterrupt()

    @kb.add("c-d")
    def _exit(_event: object) -> None:
        raise EOFError()

    return kb


def create_editor_keybindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add("escape", "enter")
    def _newline(event: object) -> None:
        from prompt_toolkit.key_binding import KeyPressEvent

        key_event = event if isinstance(event, KeyPressEvent) else None
        if key_event:
            key_event.app.current_buffer.newline(copy_margin=False)

    return kb
