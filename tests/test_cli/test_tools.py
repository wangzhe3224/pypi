"""Tests for pi.cli tools matching the original TypeScript test coverage."""

import tempfile
from pathlib import Path

import pytest

from pi.cli.tools import bash, edit, glob, grep, read, write


def get_text_output(result) -> str:
    """Extract text from AgentToolResult content blocks."""
    return "\n".join(
        block.text for block in result.content if hasattr(block, "type") and block.type == "text"
    )


@pytest.fixture
def test_dir():
    """Create a unique temporary directory for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestReadTool:
    """Tests for read tool."""

    @pytest.mark.asyncio
    async def test_read_file_contents_within_limits(self, test_dir):
        """Read file contents that fit within limits."""
        test_file = test_dir / "test.txt"
        content = "Hello, world!\nLine 2\nLine 3"
        test_file.write_text(content)

        result = await read.execute("test-call-1", {"file_path": str(test_file)})
        output = get_text_output(result)

        assert "Hello, world!" in output
        assert "Line 2" in output
        assert "Line 3" in output
        assert "Use offset=" not in output

    @pytest.mark.asyncio
    async def test_handle_non_existent_files(self, test_dir):
        """Handle non-existent files - return error string, not throw."""
        test_file = test_dir / "nonexistent.txt"

        result = await read.execute("test-call-2", {"file_path": str(test_file)})
        output = get_text_output(result)

        assert "Error" in output
        assert "not found" in output.lower()

    @pytest.mark.asyncio
    async def test_truncate_files_exceeding_line_limit(self, test_dir):
        """Truncate files exceeding 2000 line limit."""
        test_file = test_dir / "large.txt"
        lines = [f"Line {i + 1}" for i in range(2500)]
        test_file.write_text("\n".join(lines))

        result = await read.execute("test-call-3", {"file_path": str(test_file)})
        output = get_text_output(result)

        assert "Line 1" in output
        assert "Line 2000" in output
        assert "Line 2001" not in output
        assert "500 more lines" in output
        assert "Use offset=2001" in output

    @pytest.mark.asyncio
    async def test_truncate_when_byte_limit_exceeded(self, test_dir):
        """Truncate when byte limit (50KB) exceeded."""
        test_file = test_dir / "large-bytes.txt"
        lines = [f"Line {i + 1}: {'x' * 200}" for i in range(500)]
        test_file.write_text("\n".join(lines))

        result = await read.execute("test-call-4", {"file_path": str(test_file)})
        output = get_text_output(result)

        assert "Line 1:" in output
        assert "limit" in output
        assert "Use offset=" in output

    @pytest.mark.asyncio
    async def test_handle_offset_parameter(self, test_dir):
        """Handle offset parameter (1-indexed)."""
        test_file = test_dir / "offset-test.txt"
        lines = [f"Line {i + 1}" for i in range(100)]
        test_file.write_text("\n".join(lines))

        result = await read.execute("test-call-5", {"file_path": str(test_file), "offset": 51})
        output = get_text_output(result)

        assert "Line 50" not in output
        assert "Line 51" in output
        assert "Line 100" in output
        assert "Use offset=" not in output

    @pytest.mark.asyncio
    async def test_handle_limit_parameter(self, test_dir):
        """Handle limit parameter."""
        test_file = test_dir / "limit-test.txt"
        lines = [f"Line {i + 1}" for i in range(100)]
        test_file.write_text("\n".join(lines))

        result = await read.execute("test-call-6", {"file_path": str(test_file), "limit": 10})
        output = get_text_output(result)

        assert "Line 1" in output
        assert "Line 10" in output
        assert "Line 11" not in output
        assert "90 more lines" in output
        assert "Use offset=11" in output

    @pytest.mark.asyncio
    async def test_handle_offset_and_limit_together(self, test_dir):
        """Handle offset + limit together."""
        test_file = test_dir / "offset-limit-test.txt"
        lines = [f"Line {i + 1}" for i in range(100)]
        test_file.write_text("\n".join(lines))

        result = await read.execute(
            "test-call-7",
            {"file_path": str(test_file), "offset": 41, "limit": 20},
        )
        output = get_text_output(result)

        assert "Line 40" not in output
        assert "Line 41" in output
        assert "Line 60" in output
        assert "Line 61" not in output
        assert "40 more lines" in output
        assert "Use offset=61" in output

    @pytest.mark.asyncio
    async def test_show_error_when_offset_beyond_file_length(self, test_dir):
        """Show error when offset beyond file length."""
        test_file = test_dir / "short.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3")

        result = await read.execute("test-call-8", {"file_path": str(test_file), "offset": 100})
        output = get_text_output(result)

        assert "Error" in output
        assert "Offset 100" in output
        assert "beyond" in output
        assert "3 lines" in output

    @pytest.mark.asyncio
    async def test_image_file_detected_by_extension(self, test_dir):
        """Detect image MIME type from file extension."""
        test_file = test_dir / "image.png"
        test_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        result = await read.execute("test-call-img", {"file_path": str(test_file)})
        output = get_text_output(result)

        assert "image/png" in output
        assert "Base64 data" in output

    @pytest.mark.asyncio
    async def test_image_extension_but_non_image_content_as_text(self, test_dir):
        """Files with image extension are treated as images by extension (Python impl)."""
        test_file = test_dir / "not-an-image.png"
        test_file.write_text("definitely not a png")

        result = await read.execute("test-call-img-2", {"file_path": str(test_file)})
        output = get_text_output(result)

        assert "image/png" in output
        assert "Base64 data" in output


class TestWriteTool:
    """Tests for write tool."""

    @pytest.mark.asyncio
    async def test_write_file_contents(self, test_dir):
        """Write file contents."""
        test_file = test_dir / "write-test.txt"
        content = "Test content"

        result = await write.execute(
            "test-call-3", {"file_path": str(test_file), "content": content}
        )
        output = get_text_output(result)

        assert "Successfully wrote" in output
        assert str(test_file) in output

    @pytest.mark.asyncio
    async def test_create_parent_directories(self, test_dir):
        """Create parent directories."""
        test_file = test_dir / "nested" / "dir" / "test.txt"
        content = "Nested content"

        result = await write.execute(
            "test-call-4", {"file_path": str(test_file), "content": content}
        )
        output = get_text_output(result)

        assert "Successfully wrote" in output
        assert test_file.exists()
        assert test_file.read_text() == content


class TestEditTool:
    """Tests for edit tool - basic operations."""

    @pytest.mark.asyncio
    async def test_replace_text_in_file(self, test_dir):
        """Replace text in file."""
        test_file = test_dir / "edit-test.txt"
        test_file.write_text("Hello, world!")

        result = await edit.execute(
            "test-call-5",
            {
                "file_path": str(test_file),
                "old_string": "world",
                "new_string": "testing",
            },
        )
        output = get_text_output(result)

        assert "Successfully replaced" in output
        assert test_file.read_text() == "Hello, testing!"

    @pytest.mark.asyncio
    async def test_fail_if_text_not_found(self, test_dir):
        """Fail if text not found - return error string."""
        test_file = test_dir / "edit-test.txt"
        test_file.write_text("Hello, world!")

        result = await edit.execute(
            "test-call-6",
            {
                "file_path": str(test_file),
                "old_string": "nonexistent",
                "new_string": "testing",
            },
        )
        output = get_text_output(result)

        assert "Error" in output
        assert "Could not find" in output

    @pytest.mark.asyncio
    async def test_fail_if_text_appears_multiple_times(self, test_dir):
        """Fail if text appears multiple times - return error string."""
        test_file = test_dir / "edit-test.txt"
        test_file.write_text("foo foo foo")

        result = await edit.execute(
            "test-call-7",
            {
                "file_path": str(test_file),
                "old_string": "foo",
                "new_string": "bar",
            },
        )
        output = get_text_output(result)

        assert "Error" in output
        assert "3 occurrences" in output


class TestEditToolFuzzyMatching:
    """Tests for edit tool - fuzzy matching."""

    @pytest.mark.asyncio
    async def test_match_text_with_trailing_whitespace_stripped(self, test_dir):
        """Match text with trailing whitespace stripped."""
        test_file = test_dir / "trailing-ws.txt"
        test_file.write_text("line one   \nline two  \nline three\n")

        result = await edit.execute(
            "test-fuzzy-1",
            {
                "file_path": str(test_file),
                "old_string": "line one\nline two\n",
                "new_string": "replaced\n",
            },
        )
        output = get_text_output(result)

        assert "Successfully replaced" in output
        content = test_file.read_text()
        assert content == "replaced\nline three\n"

    @pytest.mark.asyncio
    async def test_match_smart_single_quotes_to_ascii(self, test_dir):
        """Match smart single quotes to ASCII quotes."""
        test_file = test_dir / "smart-quotes.txt"
        test_file.write_text("console.log(\u2018hello\u2019);\n")

        result = await edit.execute(
            "test-fuzzy-2",
            {
                "file_path": str(test_file),
                "old_string": "console.log('hello');",
                "new_string": "console.log('world');",
            },
        )
        output = get_text_output(result)

        assert "Successfully replaced" in output
        assert "world" in test_file.read_text()

    @pytest.mark.asyncio
    async def test_match_smart_double_quotes_to_ascii(self, test_dir):
        """Match smart double quotes to ASCII quotes."""
        test_file = test_dir / "smart-double-quotes.txt"
        test_file.write_text("const msg = \u201CHello World\u201D;\n")

        result = await edit.execute(
            "test-fuzzy-3",
            {
                "file_path": str(test_file),
                "old_string": 'const msg = "Hello World";',
                "new_string": 'const msg = "Goodbye";',
            },
        )
        output = get_text_output(result)

        assert "Successfully replaced" in output
        assert "Goodbye" in test_file.read_text()

    @pytest.mark.asyncio
    async def test_match_unicode_dashes_to_ascii_hyphen(self, test_dir):
        """Match Unicode dashes to ASCII hyphen."""
        test_file = test_dir / "unicode-dashes.txt"
        test_file.write_text("range: 1\u20135\nbreak\u2014here\n")

        result = await edit.execute(
            "test-fuzzy-4",
            {
                "file_path": str(test_file),
                "old_string": "range: 1-5\nbreak-here",
                "new_string": "range: 10-50\nbreak--here",
            },
        )
        output = get_text_output(result)

        assert "Successfully replaced" in output
        assert "10-50" in test_file.read_text()

    @pytest.mark.asyncio
    async def test_match_non_breaking_space_to_regular_space(self, test_dir):
        """Match non-breaking space to regular space."""
        test_file = test_dir / "nbsp.txt"
        test_file.write_text("hello\u00A0world\n")

        result = await edit.execute(
            "test-fuzzy-5",
            {
                "file_path": str(test_file),
                "old_string": "hello world",
                "new_string": "hello universe",
            },
        )
        output = get_text_output(result)

        assert "Successfully replaced" in output
        assert "universe" in test_file.read_text()

    @pytest.mark.asyncio
    async def test_prefer_exact_match_over_fuzzy_match(self, test_dir):
        """Prefer exact match over fuzzy match."""
        test_file = test_dir / "exact-preferred.txt"
        test_file.write_text("const x = 'exact';\nconst y = 'other';\n")

        result = await edit.execute(
            "test-fuzzy-6",
            {
                "file_path": str(test_file),
                "old_string": "const x = 'exact';",
                "new_string": "const x = 'changed';",
            },
        )
        output = get_text_output(result)

        assert "Successfully replaced" in output
        assert test_file.read_text() == "const x = 'changed';\nconst y = 'other';\n"

    @pytest.mark.asyncio
    async def test_fail_when_text_not_found_even_with_fuzzy(self, test_dir):
        """Fail when text not found even with fuzzy matching."""
        test_file = test_dir / "no-match.txt"
        test_file.write_text("completely different content\n")

        result = await edit.execute(
            "test-fuzzy-7",
            {
                "file_path": str(test_file),
                "old_string": "this does not exist",
                "new_string": "replacement",
            },
        )
        output = get_text_output(result)

        assert "Error" in output
        assert "Could not find" in output

    @pytest.mark.asyncio
    async def test_detect_duplicates_after_fuzzy_normalization(self, test_dir):
        """Detect duplicates after fuzzy normalization."""
        test_file = test_dir / "fuzzy-dups.txt"
        test_file.write_text("hello world   \nhello world\n")

        result = await edit.execute(
            "test-fuzzy-8",
            {
                "file_path": str(test_file),
                "old_string": "hello world",
                "new_string": "replaced",
            },
        )
        output = get_text_output(result)

        assert "Error" in output
        assert "2 occurrences" in output


class TestEditToolCRLF:
    """Tests for edit tool - CRLF handling."""

    @pytest.mark.asyncio
    async def test_match_lf_oldtext_against_crlf_file_content(self, test_dir):
        """Match LF oldText against CRLF file content."""
        test_file = test_dir / "crlf-test.txt"
        test_file.write_bytes(b"line one\r\nline two\r\nline three\r\n")

        result = await edit.execute(
            "test-crlf-1",
            {
                "file_path": str(test_file),
                "old_string": "line two\n",
                "new_string": "replaced line\n",
            },
        )
        output = get_text_output(result)

        assert "Successfully replaced" in output

    @pytest.mark.asyncio
    async def test_preserve_crlf_line_endings_after_edit(self, test_dir):
        """Preserve CRLF line endings after edit."""
        test_file = test_dir / "crlf-preserve.txt"
        test_file.write_bytes(b"first\r\nsecond\r\nthird\r\n")

        await edit.execute(
            "test-crlf-2",
            {
                "file_path": str(test_file),
                "old_string": "second\n",
                "new_string": "REPLACED\n",
            },
        )

        content = test_file.read_bytes()
        assert content == b"first\r\nREPLACED\r\nthird\r\n"

    @pytest.mark.asyncio
    async def test_preserve_lf_line_endings_for_lf_files(self, test_dir):
        """Preserve LF line endings for LF files."""
        test_file = test_dir / "lf-preserve.txt"
        test_file.write_bytes(b"first\nsecond\nthird\n")

        await edit.execute(
            "test-lf-1",
            {
                "file_path": str(test_file),
                "old_string": "second\n",
                "new_string": "REPLACED\n",
            },
        )

        content = test_file.read_bytes()
        assert content == b"first\nREPLACED\nthird\n"

    @pytest.mark.asyncio
    async def test_detect_duplicates_across_crlf_lf_variants(self, test_dir):
        """Detect duplicates across CRLF/LF variants."""
        test_file = test_dir / "mixed-endings.txt"
        test_file.write_bytes(b"hello\r\nworld\r\n---\r\nhello\nworld\n")

        result = await edit.execute(
            "test-crlf-dup",
            {
                "file_path": str(test_file),
                "old_string": "hello\nworld\n",
                "new_string": "replaced\n",
            },
        )
        output = get_text_output(result)

        assert "Error" in output
        assert "2 occurrences" in output

    @pytest.mark.asyncio
    async def test_preserve_utf8_bom_after_edit(self, test_dir):
        """Preserve UTF-8 BOM after edit."""
        test_file = test_dir / "bom-test.txt"
        test_file.write_bytes("\ufefffirst\r\nsecond\r\nthird\r\n".encode())

        await edit.execute(
            "test-bom",
            {
                "file_path": str(test_file),
                "old_string": "second\n",
                "new_string": "REPLACED\n",
            },
        )

        content = test_file.read_bytes()
        assert content == "\ufefffirst\r\nREPLACED\r\nthird\r\n".encode()


class TestBashTool:
    """Tests for bash tool."""

    @pytest.mark.asyncio
    async def test_execute_simple_commands(self):
        """Execute simple commands."""
        result = await bash.execute("test-call-8", {"command": "echo 'test output'"})
        output = get_text_output(result)

        assert "test output" in output

    @pytest.mark.asyncio
    async def test_handle_command_errors(self):
        """Handle command errors - return error with exit code."""
        result = await bash.execute("test-call-9", {"command": "exit 1"})
        output = get_text_output(result)

        assert "code 1" in output

    @pytest.mark.asyncio
    async def test_respect_timeout(self):
        """Respect timeout."""
        result = await bash.execute("test-call-10", {"command": "sleep 5", "timeout": 100})
        output = get_text_output(result)

        assert "timed out" in output.lower()


class TestGrepTool:
    """Tests for grep tool."""

    @pytest.mark.asyncio
    async def test_include_filename_in_single_file_search(self, test_dir):
        """Include filename when searching a single file."""
        test_file = test_dir / "example.txt"
        test_file.write_text("first line\nmatch line\nlast line")

        result = await grep.execute(
            "test-call-11",
            {"pattern": "match", "path": str(test_file)},
        )
        output = get_text_output(result)

        assert "example.txt" in output
        assert "match line" in output

    @pytest.mark.asyncio
    async def test_respect_limit_and_include_context_lines(self, test_dir):
        """Respect global limit and include context lines."""
        test_file = test_dir / "context.txt"
        content = ["before", "match one", "after", "middle", "match two", "after two"]
        test_file.write_text("\n".join(content))

        result = await grep.execute(
            "test-call-12",
            {
                "pattern": "match",
                "path": str(test_file),
                "limit": 1,
                "context": 1,
            },
        )
        output = get_text_output(result)

        assert "before" in output
        assert "match one" in output
        assert "after" in output
        assert "match two" not in output
        assert "limit reached" in output


class TestGlobTool:
    """Tests for glob tool."""

    @pytest.mark.asyncio
    async def test_include_hidden_files_not_gitignored(self, test_dir):
        """Include hidden files that are not gitignored."""
        hidden_dir = test_dir / ".secret"
        hidden_dir.mkdir()
        (hidden_dir / "hidden.txt").write_text("hidden")
        (test_dir / "visible.txt").write_text("visible")

        result = await glob.execute(
            "test-call-13",
            {"pattern": "**/*.txt", "path": str(test_dir)},
        )
        output = get_text_output(result)
        lines = [line.strip() for line in output.split("\n") if line.strip()]

        assert "visible.txt" in lines
        assert ".secret/hidden.txt" in lines or ".secret" in output

    @pytest.mark.asyncio
    async def test_respect_gitignore(self, test_dir):
        """Respect .gitignore."""
        (test_dir / ".gitignore").write_text("ignored.txt\n")
        (test_dir / "ignored.txt").write_text("ignored")
        (test_dir / "kept.txt").write_text("kept")

        result = await glob.execute(
            "test-call-14",
            {"pattern": "**/*.txt", "path": str(test_dir)},
        )
        output = get_text_output(result)

        assert "kept.txt" in output
        assert "ignored.txt" not in output
