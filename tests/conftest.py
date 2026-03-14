"""Test configuration."""

import pytest


@pytest.fixture
def sample_context():
    """Sample context fixture."""
    from pi.ai.types import Context, UserMessage

    return Context(
        system_prompt="You are helpful.",
        messages=[UserMessage(content="Hello", timestamp=1000)],
    )
