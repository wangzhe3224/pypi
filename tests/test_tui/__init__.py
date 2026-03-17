"""Tests for pi.tui components."""

from pi.tui.types import EventType
from pi.tui.keybindings import EditorAction


class TestEventType:
    def test_event_types_match_agent_events(self) -> None:
        assert EventType.AGENT_START == "agent_start"
        assert EventType.AGENT_END == "agent_end"
        assert EventType.TURN_START == "turn_start"
        assert EventType.TURN_END == "turn_end"
        assert EventType.MESSAGE_START == "message_start"
        assert EventType.MESSAGE_UPDATE == "message_update"
        assert EventType.MESSAGE_END == "message_end"
        assert EventType.TOOL_EXECUTION_START == "tool_execution_start"
        assert EventType.TOOL_EXECUTION_UPDATE == "tool_execution_update"
        assert EventType.TOOL_EXECUTION_END == "tool_execution_end"

    def test_event_type_is_string(self) -> None:
        assert EventType.AGENT_START.value == "agent_start"
        assert isinstance(EventType.AGENT_START.value, str)


class TestEditorAction:
    def test_editor_actions_exist(self) -> None:
        assert EditorAction.SUBMIT == "submit"
        assert EditorAction.INTERRUPT == "interrupt"
        assert EditorAction.EXIT == "exit"
        assert EditorAction.CLEAR == "clear"
        assert EditorAction.NEWLINE == "newline"
        assert EditorAction.HISTORY_UP == "history_up"
        assert EditorAction.HISTORY_DOWN == "history_down"

    def test_editor_action_is_string(self) -> None:
        assert EditorAction.SUBMIT.value == "submit"
        assert isinstance(EditorAction.SUBMIT.value, str)
