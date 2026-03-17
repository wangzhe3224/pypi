"""Tests for pi.tui.commands.registry."""

import pytest

from pi.tui.commands.registry import CommandRegistry, SlashCommand


@pytest.fixture
def registry() -> CommandRegistry:
    return CommandRegistry()


@pytest.fixture
def sample_command() -> SlashCommand:
    async def handler(_args: str, _app: object) -> str:
        return "test result"

    return SlashCommand(
        name="test",
        description="Test command",
        handler=handler,
    )


class TestSlashCommand:
    def test_create_command(self, sample_command: SlashCommand) -> None:
        assert sample_command.name == "test"
        assert sample_command.description == "Test command"
        assert sample_command.aliases is None

    def test_create_command_with_aliases(self) -> None:
        async def handler(_args: str, _app: object) -> str:
            return "result"

        cmd = SlashCommand(
            name="help",
            description="Show help",
            handler=handler,
            aliases=["?", "h"],
        )

        assert cmd.name == "help"
        assert cmd.aliases == ["?", "h"]


class TestCommandRegistry:
    def test_empty_registry(self, registry: CommandRegistry) -> None:
        assert registry.get("nonexistent") is None
        assert registry.list_commands() == []

    def test_register_command(
        self, registry: CommandRegistry, sample_command: SlashCommand
    ) -> None:
        registry.register(sample_command)

        found = registry.get("test")
        assert found is not None
        assert found.name == "test"

    def test_register_command_with_aliases(self, registry: CommandRegistry) -> None:
        async def handler(_args: str, _app: object) -> str:
            return "result"

        cmd = SlashCommand(
            name="help",
            description="Show help",
            handler=handler,
            aliases=["?", "h"],
        )

        registry.register(cmd)

        assert registry.get("help") is not None
        assert registry.get("?") is not None
        assert registry.get("h") is not None

        assert registry.get("help") == registry.get("?")
        assert registry.get("help") == registry.get("h")

    def test_list_commands_no_duplicates(
        self, registry: CommandRegistry, sample_command: SlashCommand
    ) -> None:
        registry.register(sample_command)

        commands = registry.list_commands()
        assert len(commands) == 1
        assert commands[0].name == "test"

    def test_list_commands_with_aliases(self, registry: CommandRegistry) -> None:
        async def handler(_args: str, _app: object) -> str:
            return "result"

        cmd1 = SlashCommand(
            name="help",
            description="Show help",
            handler=handler,
            aliases=["?", "h"],
        )
        cmd2 = SlashCommand(
            name="quit",
            description="Exit",
            handler=handler,
            aliases=["q"],
        )

        registry.register(cmd1)
        registry.register(cmd2)

        commands = registry.list_commands()
        assert len(commands) == 2

        names = {cmd.name for cmd in commands}
        assert names == {"help", "quit"}

    def test_get_nonexistent(self, registry: CommandRegistry) -> None:
        assert registry.get("nonexistent") is None

    def test_overwrite_command(self, registry: CommandRegistry) -> None:
        async def handler1(_args: str, _app: object) -> str:
            return "result1"

        async def handler2(_args: str, _app: object) -> str:
            return "result2"

        cmd1 = SlashCommand(name="test", description="First", handler=handler1)
        cmd2 = SlashCommand(name="test", description="Second", handler=handler2)

        registry.register(cmd1)
        registry.register(cmd2)

        found = registry.get("test")
        assert found is not None
        assert found.description == "Second"
