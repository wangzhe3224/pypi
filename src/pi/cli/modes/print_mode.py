from __future__ import annotations

from pi.session.agent_session import AgentSession
from pi.agent.types import AgentEvent


async def run_print_mode(session: AgentSession, prompt: str) -> int:
    async for event in session.prompt(prompt):
        _handle_event(event)
    return 0


def _handle_event(event: AgentEvent) -> None:
    if event.type == "message_update":
        if hasattr(event, "stream_event") and hasattr(event.stream_event, "delta"):
            print(event.stream_event.delta, end="", flush=True)
    elif event.type == "tool_execution_start":
        print(f"\n[Running: {event.tool_name}]", flush=True)
    elif event.type == "tool_execution_end":
        if event.is_error:
            print(f"\n[Error: {event.tool_name}]", flush=True)
    elif event.type == "agent_end":
        print()
