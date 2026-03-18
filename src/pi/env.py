"""Environment variable management with .env file support."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

_LOADED: bool = False

ENV_FILE_NAMES: Final[tuple[str, ...]] = (
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
)


def _find_env_file() -> Path | None:
    """Find the first existing .env file in priority order."""
    cwd = Path.cwd()

    search_paths = [
        cwd,
        cwd.parent,
        Path.home(),
    ]

    for search_path in search_paths:
        if search_path is None:
            continue
        for env_name in ENV_FILE_NAMES:
            env_file = search_path / env_name
            if env_file.exists():
                return env_file

    return None


def load_env(force: bool = False) -> None:
    """Load environment variables from .env file.

    Searches for .env files in this order:
    1. Current working directory
    2. Parent directory (project root)
    3. Home directory

    File priority: .env.local > .env.development > .env.production > .env

    Args:
        force: Reload even if already loaded.
    """
    global _LOADED

    if _LOADED and not force:
        return

    env_file = _find_env_file()
    if env_file is None:
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=True)
        _LOADED = True
    except ImportError:
        pass


def get_env(key: str, default: str | None = None) -> str | None:
    """Get environment variable, loading .env first if needed.

    Args:
        key: Environment variable name.
        default: Default value if not found.

    Returns:
        Environment variable value or default.
    """
    load_env()
    return os.environ.get(key, default)


def require_env(key: str) -> str:
    """Get required environment variable, raising error if missing.

    Args:
        key: Environment variable name.

    Returns:
        Environment variable value.

    Raises:
        ValueError: If environment variable is not set.
    """
    value = get_env(key)
    if value is None:
        raise ValueError(f"Required environment variable '{key}' is not set")
    return value


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable.

    Args:
        key: Environment variable name.
        default: Default value if not found.

    Returns:
        Boolean value.
    """
    value = get_env(key)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def get_env_int(key: str, default: int | None = None) -> int | None:
    """Get integer environment variable.

    Args:
        key: Environment variable name.
        default: Default value if not found.

    Returns:
        Integer value or default.
    """
    value = get_env(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_env_float(key: str, default: float | None = None) -> float | None:
    """Get float environment variable.

    Args:
        key: Environment variable name.
        default: Default value if not found.

    Returns:
        Float value or default.
    """
    value = get_env(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


class _EnvConfig:
    """Typed access to common environment configuration."""

    @property
    def openai_api_key(self) -> str | None:
        return get_env("OPENAI_API_KEY")

    @property
    def anthropic_api_key(self) -> str | None:
        return get_env("ANTHROPIC_API_KEY")

    @property
    def google_api_key(self) -> str | None:
        return get_env("GOOGLE_API_KEY")

    @property
    def zhipuai_api_key(self) -> str | None:
        return get_env("ZHIPUAI_API_KEY") or get_env("ZAI_API_KEY")

    @property
    def zhipuai_region(self) -> str:
        return get_env("ZHIPUAI_REGION", "china") or "china"

    @property
    def zhipuai_base_url(self) -> str | None:
        return get_env("ZHIPUAI_BASE_URL")

    @property
    def debug(self) -> bool:
        return get_env_bool("PI_DEBUG", False)

    @property
    def log_level(self) -> str:
        return get_env("PI_LOG_LEVEL", "INFO") or "INFO"


env_config = _EnvConfig()
env = env_config
