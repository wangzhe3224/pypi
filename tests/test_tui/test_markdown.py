"""Tests for pi.tui.components.markdown."""

import pytest

from pi.tui.components.markdown import MarkdownComponent


class TestMarkdownComponent:
    def test_render_simple_text(self) -> None:
        md = MarkdownComponent("Hello, world!")
        lines = md.render(80)

        assert len(lines) == 1
        assert "Hello, world!" in lines[0]

    def test_render_empty_text(self) -> None:
        md = MarkdownComponent("")
        lines = md.render(80)

        assert lines == []

    def test_render_heading(self) -> None:
        md = MarkdownComponent("# Title")
        lines = md.render(80)

        assert len(lines) >= 1
        assert "Title" in "".join(lines)

    def test_render_code_block(self) -> None:
        md = MarkdownComponent("```python\nprint('hello')\n```")
        lines = md.render(80)

        assert len(lines) >= 2
        assert "print" in "".join(lines)

    def test_render_list(self) -> None:
        md = MarkdownComponent("- Item 1\n- Item 2\n- Item 3")
        lines = md.render(80)

        assert len(lines) >= 3
        combined = "\n".join(lines)
        assert "Item 1" in combined
        assert "Item 2" in combined
        assert "Item 3" in combined

    def test_render_bold_text(self) -> None:
        md = MarkdownComponent("**bold text**")
        lines = md.render(80)

        assert len(lines) >= 1
        assert "bold text" in "".join(lines)

    def test_caching(self) -> None:
        md = MarkdownComponent("Test")

        lines1 = md.render(80)
        lines2 = md.render(80)

        assert lines1 is lines2

    def test_cache_invalidation_on_text_change(self) -> None:
        md = MarkdownComponent("Test")
        lines1 = md.render(80)

        md.set_text("New test")
        lines2 = md.render(80)

        assert lines1 is not lines2

    def test_cache_invalidation_on_width_change(self) -> None:
        md = MarkdownComponent("Test content here")
        lines1 = md.render(80)
        lines2 = md.render(60)

        assert lines1 is not lines2

    def test_set_text_updates_content(self) -> None:
        md = MarkdownComponent("Original")
        md.set_text("Updated")

        lines = md.render(80)
        assert "Updated" in "".join(lines)
        assert "Original" not in "".join(lines)

    def test_render_with_narrow_width(self) -> None:
        md = MarkdownComponent("This is a long line of text that should wrap")
        lines = md.render(20)

        assert len(lines) >= 1
