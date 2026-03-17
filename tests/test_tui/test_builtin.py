"""Tests for pi.tui.commands.builtin."""

import pytest

from pi.tui.commands.builtin import create_builtin_commands


@pytest.fixture
def registry():
    return create_builtin_commands()


class TestBuiltinCommands:
    def test_help_command_exists(self, registry) -> None:
        cmd = registry.get("help")
        assert cmd is not None
        assert cmd.name == "help"
        assert cmd.aliases is not None
        assert "?" in cmd.aliases
        assert "h" in cmd.aliases

    def test_model_command_exists(self, registry) -> None:
        cmd = registry.get("model")
        assert cmd is not None
        assert cmd.name == "model"
        assert cmd.aliases is not None
        assert "m" in cmd.aliases

    def test_clear_command_exists(self, registry) -> None:
        cmd = registry.get("clear")
        assert cmd is not None
        assert cmd.name == "clear"
        assert cmd.aliases is not None
        assert "c" in cmd.aliases

    def test_quit_command_exists(self, registry) -> None:
        cmd = registry.get("quit")
        assert cmd is not None
        assert cmd.name == "quit"
        assert cmd.aliases is not None
        assert "q" in cmd.aliases
        assert "exit" in cmd.aliases

    def test_all_commands_have_descriptions(self, registry) -> None:
        for cmd in registry.list_commands():
            assert cmd.description
            assert len(cmd.description) > 0

    def test_command_count(self, registry) -> None:
        commands = registry.list_commands()
        assert len(commands) == 4
