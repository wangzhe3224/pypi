from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field

from pi.session.types import SessionEntry, SessionEntryType, SessionHeader
from pi.ai.types import (
    AssistantMessage,
    Context,
    Message,
    TextContent,
    ToolResultMessage,
    UserMessage,
)


class SessionManager:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.header: SessionHeader | None = None
        self.entries: list[SessionEntry] = []
        self._children: dict[str, list[SessionEntry]] = {}

    def load(self) -> None:
        if not self.path.exists():
            return

        self.header = None
        self.entries = []

        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("# header"):
                    header_data = json.loads(line[8:].strip())
                    self.header = SessionHeader(**header_data)
                    continue

                entry_data = json.loads(line)
                try:
                    entry = SessionEntry(**entry_data)
                    self.entries.append(entry)
                    self._add_to_children(entry)
                except Exception:
                    continue

    def _add_to_children(self, entry: SessionEntry) -> None:
        if entry.parent_id:
            if entry.parent_id not in self._children:
                self._children[entry.parent_id] = []
            self._children[entry.parent_id].append(entry)

    async def append(self, entry: SessionEntry) -> None:
        self.entries.append(entry)
        self._add_to_children(entry)

        with self.path.open("a") as f:
            f.write(entry.model_dump_json() + "\n")

    def get_children(self, entry_id: str) -> list[SessionEntry]:
        return self._children.get(entry_id, [])

    def get_entry(self, entry_id: str) -> SessionEntry | None:
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None

    async def fork(self, entry_id: str, new_path: Path | None = None) -> SessionManager:
        entry = self.get_entry(entry_id)
        if not entry:
            raise ValueError(f"Entry {entry_id} not found")

        new_session = SessionManager(new_path or self.path.with_suffix(".forked.jsonl"))
        new_session.header = SessionHeader(
            version=1,
            cwd=self.header.cwd if self.header else str(Path.cwd()),
            parent_session=str(self.path.stem),
        )

        for e in self.entries:
            if e.id == entry_id:
                break
            new_session.entries.append(e)

        await new_session.save()
        return new_session

    async def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open("w") as f:
            if self.header:
                f.write(f"# {self.header.model_dump_json()}\n")
            for entry in self.entries:
                f.write(entry.model_dump_json() + "\n")

    def to_messages(self) -> list[Message]:
        messages: list[Message] = []
        for entry in self.entries:
            if entry.type == SessionEntryType.MESSAGE:
                msg_data = entry.data
                role = msg_data.get("role")
                if role == "user":
                    messages.append(
                        UserMessage(
                            content=msg_data.get("content", ""),
                            timestamp=entry.timestamp,
                        )
                    )
                elif role == "assistant":
                    content = []
                    for c in msg_data.get("content", []):
                        if isinstance(c, dict) and c.get("type") == "text":
                            content.append(TextContent(text=c.get("text", "")))
                        elif isinstance(c, dict) and c.get("type") == "tool_result":
                            content.append(
                                ToolResultMessage(
                                    tool_call_id=c.get("tool_call_id", ""),
                                    content=c.get("content", ""),
                                    timestamp=entry.timestamp,
                                )
                            )
                    messages.append(
                        AssistantMessage(
                            content=content,
                            timestamp=entry.timestamp,
                        )
                    )
                elif role == "tool_result":
                    messages.append(
                        ToolResultMessage(
                            tool_call_id=msg_data.get("tool_call_id", ""),
                            content=msg_data.get("content", ""),
                            timestamp=entry.timestamp,
                        )
                    )
        return messages

    def create_user_entry(self, content: str) -> SessionEntry:
        return SessionEntry(
            type=SessionEntryType.MESSAGE,
            data={
                "role": "user",
                "content": content,
            },
        )

    def create_assistant_entry(self, message: AssistantMessage) -> SessionEntry:
        content_data = []
        for c in message.content:
            if isinstance(c, TextContent):
                content_data.append({"type": "text", "text": c.text})
            elif isinstance(c, ToolResultMessage):
                content_data.append(
                    {
                        "type": "tool_result",
                        "tool_call_id": c.tool_call_id,
                        "content": c.content
                        if isinstance(c.content, str)
                        else c.content[0].text
                        if isinstance(c.content, list)
                        else c.content,
                    }
                )

        return SessionEntry(
            type=SessionEntryType.MESSAGE,
            data={
                "role": "assistant",
                "content": content_data,
            },
        )

    def create_tool_result_entry(
        self, tool_call_id: str, result: ToolResultMessage
    ) -> SessionEntry:
        return SessionEntry(
            type=SessionEntryType.MESSAGE,
            data={
                "role": "tool_result",
                "tool_call_id": tool_call_id,
                "content": result.content,
            },
        )
