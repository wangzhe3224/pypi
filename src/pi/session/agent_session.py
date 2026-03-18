from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

from pi.agent import Agent, AgentEvent, AgentTool
from pi.ai.models import resolve_model
from pi.session.manager import SessionManager
from pi.session.settings import AgentSettings, SettingsManager


class AgentSession:
    def __init__(
        self,
        session_path: Path | None = None,
        project_path: Path | None = None,
        model: str | None = None,
        tools: list[AgentTool] | None = None,
    ):
        self.session = SessionManager(session_path) if session_path else None
        self.settings = SettingsManager(project_path)
        self.agent = Agent()

        loaded_settings = self.settings.load()
        self._model_name = model or loaded_settings.default_model or "glm-5"
        self._tools = tools or []

        if self._model_name:
            resolved = resolve_model(self._model_name)
            if resolved:
                self.agent.set_model(resolved)

        if self.session:
            self.session.load()
            self._load_messages_into_agent()

    def _load_messages_into_agent(self) -> None:
        if not self.session:
            return
        messages = self.session.to_messages()
        for msg in messages:
            self.agent.append_message(msg)

    @property
    def model(self) -> str | None:
        return self._model_name

    @model.setter
    def model(self, value: str) -> None:
        self._model_name = value

    @property
    def settings_obj(self) -> AgentSettings:
        return self.settings.load()

    async def prompt(self, text: str) -> AsyncGenerator[AgentEvent, None]:
        if self.session:
            entry = self.session.create_user_entry(text)
            await self.session.append(entry)

        events: list[AgentEvent] = []
        async for event in self.agent.prompt(text):
            events.append(event)
            yield event

        if self.session and events:
            await self._persist_events(events)

    async def _persist_events(self, events: list[AgentEvent]) -> None:
        for event in events:
            if event.type == "turn_end":
                entry = self.session.create_assistant_entry(event.message)
                await self.session.append(entry)
                for result in event.tool_results:
                    tool_entry = self.session.create_tool_result_entry(result.tool_call_id, result)
                    await self.session.append(tool_entry)

    async def load_session(self, path: Path) -> None:
        self.session = SessionManager(path)
        self.session.load()
        self._load_messages_into_agent()

    async def fork_session(self, entry_id: str) -> AgentSession:
        if not self.session:
            raise ValueError("No session to fork")

        new_session = await self.session.fork(entry_id)
        return AgentSession(
            session_path=new_session.path,
            project_path=Path(self.settings.project_path).parent
            if self.settings.project_path
            else None,
            model=self._model_name,
            tools=self._tools,
        )

    def set_model(self, model: str) -> None:
        self._model_name = model

    def get_context_summary(self) -> str:
        msg_count = len(self.agent.messages)
        return f"{msg_count} messages"
