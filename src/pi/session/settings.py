from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from pi.session.config import get_agent_dir


class AgentSettings(BaseModel):
    default_model: str | None = "glm-5"
    thinking_level: str = "medium"
    compaction_threshold: int = 100000
    compaction_style: str = "summarize"
    max_retries: int = 3
    terminal_shell: str = "auto"

    model_config = {"extra": "allow"}


class SettingsManager:
    def __init__(self, project_path: Path | None = None):
        self.global_path = get_agent_dir() / "settings.json"
        self.project_path = project_path / ".pi" / "settings.json" if project_path else None
        self._settings: AgentSettings | None = None

    def load(self) -> AgentSettings:
        settings = AgentSettings()
        if self.global_path.exists():
            with self.global_path.open() as f:
                data = __import__("json").load(f)
                settings = AgentSettings(**{**settings.model_dump(), **data})
        if self.project_path and self.project_path.exists():
            with self.project_path.open() as f:
                data = __import__("json").load(f)
                settings = AgentSettings(**{**settings.model_dump(), **data})
        return settings

    def save(self, settings: AgentSettings, *, global_: bool = False) -> None:
        path = self.global_path if global_ else self.project_path
        if path is None:
            raise ValueError("No path specified for settings")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            __import__("json").dump(settings.model_dump(), f, indent=2)

    def get(self, key: str) -> Any:
        if self._settings is None:
            self._settings = self.load()
        return getattr(self._settings, key, None)
