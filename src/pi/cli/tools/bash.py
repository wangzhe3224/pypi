"""Bash tool for executing shell commands asynchronously."""

from __future__ import annotations

import asyncio

from pi.agent.tools import tool
from pi.cli.tools.truncate import truncate_tail


@tool
async def bash(command: str, timeout: int = 120000) -> str:
    """Execute a bash command.

    Args:
        command: The command to execute
        timeout: Maximum execution time in milliseconds
    """
    timeout_seconds = timeout / 1000.0
    process: asyncio.subprocess.Process | None = None

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
                await asyncio.sleep(0.1)
            except ProcessLookupError:
                pass

            output = ""
            if process.stdout:
                try:
                    remaining = await process.stdout.read()
                    output = remaining.decode("utf-8", errors="replace")
                except Exception:
                    pass

            truncated_output, _ = truncate_tail(output) if output else ("", None)
            return f"{truncated_output}\n\nCommand timed out after {timeout_seconds:.1f} seconds".strip()

        output = stdout.decode("utf-8", errors="replace") if stdout else ""
        truncated_output, truncation_info = truncate_tail(output)

        result = truncated_output if truncated_output else "(no output)"

        if truncation_info and truncation_info.truncated:
            result += (
                f"\n\n[Output truncated: showing last {truncation_info.truncated_lines} "
                f"of {truncation_info.original_lines} lines]"
            )

        if process.returncode is not None and process.returncode != 0:
            result += f"\n\nCommand exited with code {process.returncode}"

        return result

    except asyncio.CancelledError:
        if process is not None:
            try:
                process.kill()
                await asyncio.sleep(0.1)
            except (ProcessLookupError, Exception):
                pass
        return "Command aborted"

    except Exception as e:
        return f"Error executing command: {e}"
