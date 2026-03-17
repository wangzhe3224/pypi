from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CLIArgs:
    prompt: str | None = None
    session: str | None = None
    model: str | None = None
    mode: str = "print"
    list_models: bool = False
    list_sessions: bool = False
    config_dir: Path | None = None
    cwd: Path | None = None
    verbose: bool = False
    help: bool = False
