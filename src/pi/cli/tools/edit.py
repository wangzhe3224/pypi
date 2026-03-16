"""Edit tool for replacing text in files with fuzzy matching support."""
from __future__ import annotations

import os
from dataclasses import dataclass
from difflib import unified_diff

from pi.agent.tools import tool
from pi.cli.tools.path_utils import resolve_to_cwd


def _detect_line_ending(content: str) -> str:
    crlf_idx = content.find("\r\n")
    lf_idx = content.find("\n")
    if lf_idx == -1 or crlf_idx == -1:
        return "\n"
    return "\r\n" if crlf_idx < lf_idx else "\n"


def _normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _restore_line_endings(text: str, ending: str) -> str:
    return text.replace("\n", ending) if ending == "\r\n" else text


def _normalize_for_fuzzy_match(text: str) -> str:
    """Normalize text for fuzzy matching with Unicode character handling."""
    lines = [line.rstrip() for line in text.split("\n")]
    result = "\n".join(lines)

    # Smart single quotes -> '
    result = result.replace("\u2018", "'")  # LEFT SINGLE QUOTATION MARK
    result = result.replace("\u2019", "'")  # RIGHT SINGLE QUOTATION MARK
    result = result.replace("\u201a", "'")  # SINGLE LOW-9 QUOTATION MARK
    result = result.replace("\u201b", "'")  # SINGLE HIGH-REVERSED-9 QUOTATION MARK

    # Smart double quotes -> "
    result = result.replace("\u201c", '"')  # LEFT DOUBLE QUOTATION MARK
    result = result.replace("\u201d", '"')  # RIGHT DOUBLE QUOTATION MARK
    result = result.replace("\u201e", '"')  # DOUBLE LOW-9 QUOTATION MARK
    result = result.replace("\u201f", '"')  # DOUBLE HIGH-REVERSED-9 QUOTATION MARK

    # Various dashes/hyphens -> -
    result = result.replace("\u2010", "-")  # HYPHEN
    result = result.replace("\u2011", "-")  # NON-BREAKING HYPHEN
    result = result.replace("\u2012", "-")  # FIGURE DASH
    result = result.replace("\u2013", "-")  # EN DASH
    result = result.replace("\u2014", "-")  # EM DASH
    result = result.replace("\u2015", "-")  # HORIZONTAL BAR
    result = result.replace("\u2212", "-")  # MINUS SIGN

    # Special spaces -> regular space
    result = result.replace("\u00a0", " ")  # NO-BREAK SPACE
    result = result.replace("\u2002", " ")  # EN SPACE
    result = result.replace("\u2003", " ")  # EM SPACE
    result = result.replace("\u2004", " ")  # THREE-PER-EM SPACE
    result = result.replace("\u2005", " ")  # FOUR-PER-EM SPACE
    result = result.replace("\u2006", " ")  # SIX-PER-EM SPACE
    result = result.replace("\u2007", " ")  # FIGURE SPACE
    result = result.replace("\u2008", " ")  # PUNCTUATION SPACE
    result = result.replace("\u2009", " ")  # THIN SPACE
    result = result.replace("\u200a", " ")  # HAIR SPACE
    result = result.replace("\u202f", " ")  # NARROW NO-BREAK SPACE
    result = result.replace("\u205f", " ")  # MEDIUM MATHEMATICAL SPACE
    result = result.replace("\u3000", " ")  # IDEOGRAPHIC SPACE

    return result


def _strip_bom(content: str) -> tuple[str, str]:
    if content.startswith("\ufeff"):
        return "\ufeff", content[1:]
    return "", content


@dataclass
class _FuzzyMatchResult:
    found: bool
    index: int
    match_length: int
    used_fuzzy_match: bool
    content_for_replacement: str


def _fuzzy_find_text(content: str, old_text: str) -> _FuzzyMatchResult:
    exact_index = content.find(old_text)
    if exact_index != -1:
        return _FuzzyMatchResult(
            found=True,
            index=exact_index,
            match_length=len(old_text),
            used_fuzzy_match=False,
            content_for_replacement=content,
        )

    fuzzy_content = _normalize_for_fuzzy_match(content)
    fuzzy_old_text = _normalize_for_fuzzy_match(old_text)
    fuzzy_index = fuzzy_content.find(fuzzy_old_text)

    if fuzzy_index == -1:
        return _FuzzyMatchResult(
            found=False,
            index=-1,
            match_length=0,
            used_fuzzy_match=False,
            content_for_replacement=content,
        )

    return _FuzzyMatchResult(
        found=True,
        index=fuzzy_index,
        match_length=len(fuzzy_old_text),
        used_fuzzy_match=True,
        content_for_replacement=fuzzy_content,
    )


def _generate_diff_string(
    old_content: str,
    new_content: str,
    filepath: str = "",
    context_lines: int = 4,
) -> str:
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    if old_lines and not old_lines[-1].endswith("\n"):
        old_lines[-1] += "\n"
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"

    diff_lines = list(
        unified_diff(
            old_lines,
            new_lines,
            fromfile=filepath,
            tofile=filepath,
            n=context_lines,
        )
    )

    return "".join(diff_lines)


@tool
def edit(file_path: str, old_string: str, new_string: str) -> str:
    """Edit a file by replacing exact text.

    Args:
        file_path: The absolute path to the file to edit
        old_string: The text to replace (must be unique in the file)
        new_string: The replacement text
    """
    absolute_path = resolve_to_cwd(file_path, os.getcwd())

    if not os.path.exists(absolute_path):
        return f"Error: File not found: {file_path}"

    try:
        with open(absolute_path, "rb") as f:
            raw_bytes = f.read()
        raw_content = raw_bytes.decode("utf-8")
    except Exception as e:
        return f"Error: Could not read file: {e}"

    bom, content = _strip_bom(raw_content)
    original_ending = _detect_line_ending(content)
    normalized_content = _normalize_to_lf(content)
    normalized_old_text = _normalize_to_lf(old_string)
    normalized_new_text = _normalize_to_lf(new_string)

    match_result = _fuzzy_find_text(normalized_content, normalized_old_text)

    if not match_result.found:
        return (
            f"Error: Could not find the exact text in {file_path}. "
            "The old text must match exactly including all whitespace and newlines."
        )

    fuzzy_content = _normalize_for_fuzzy_match(normalized_content)
    fuzzy_old_text = _normalize_for_fuzzy_match(normalized_old_text)
    occurrences = fuzzy_content.count(fuzzy_old_text)

    if occurrences > 1:
        return (
            f"Error: Found {occurrences} occurrences of the text in {file_path}. "
            "The text must be unique. Please provide more context to make it unique."
        )

    base_content = match_result.content_for_replacement
    new_content = (
        base_content[: match_result.index]
        + normalized_new_text
        + base_content[match_result.index + match_result.match_length :]
    )

    if base_content == new_content:
        return (
            f"Error: No changes made to {file_path}. " "The replacement produced identical content."
        )

    final_content = bom + _restore_line_endings(new_content, original_ending)

    try:
        with open(absolute_path, "w", encoding="utf-8", newline="") as f:
            f.write(final_content)
    except Exception as e:
        return f"Error: Could not write file: {e}"

    diff = _generate_diff_string(base_content, new_content, file_path)

    return f"Successfully replaced text in {file_path}.\n\n{diff}"
