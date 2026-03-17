from io import StringIO

from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme


class MarkdownComponent:
    """Renders markdown text to ANSI-styled terminal lines with caching."""

    _THEME = Theme(
        {
            "markdown.code": "cyan",
            "markdown.code_block": "dim cyan",
            "markdown.heading": "bold yellow",
            "markdown.link": "underline blue",
            "markdown.bold": "bold",
            "markdown.italic": "italic",
        }
    )

    def __init__(self, text: str = ""):
        self.text = text
        self._cache: list[str] | None = None
        self._cached_width: int | None = None

    def set_text(self, text: str) -> None:
        if text != self.text:
            self.text = text
            self._cache = None

    def render(self, width: int) -> list[str]:
        if not self.text:
            return []

        if self._cache is not None and self._cached_width == width:
            return self._cache

        string_io = StringIO()
        console = Console(
            file=string_io,
            width=width,
            theme=self._THEME,
            legacy_windows=False,
            force_terminal=True,
        )

        console.print(Markdown(self.text))

        output = string_io.getvalue()
        lines = output.rstrip("\n").split("\n") if output else []

        self._cache = lines
        self._cached_width = width

        return lines
