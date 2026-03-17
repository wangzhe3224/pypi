"""Base component class for TUI components."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseComponent(ABC):
    """Base class for all TUI components.

    Provides common functionality for invalidation tracking.
    Subclasses must implement the render method.
    """

    def __init__(self) -> None:
        self._invalidated: bool = True

    @abstractmethod
    def render(self, width: int) -> list[str]:
        """Render component to ANSI-styled lines.

        Args:
            width: Maximum width in characters for rendering.

        Returns:
            List of ANSI-styled strings, one per line.
        """
        ...

    def invalidate(self) -> None:
        """Mark component as needing re-render."""
        self._invalidated = True

    def _rendered(self) -> None:
        """Mark component as rendered."""
        self._invalidated = False
