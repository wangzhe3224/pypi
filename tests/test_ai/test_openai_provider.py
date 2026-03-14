"""Tests for OpenAI provider."""

import pytest

from pi.ai.providers.openai import OpenAIProvider
from pi.ai.types import Context, Model, ModelCost, UserMessage


def test_openai_provider_creation() -> None:
    """Test OpenAI provider can be created."""
    provider = OpenAIProvider()
    assert provider.name == "openai"
    assert provider.api_type == "openai-completions"


def test_openai_provider_with_api_key() -> None:
    """Test OpenAI provider with API key."""
    provider = OpenAIProvider(api_key="test-key")
    assert provider._api_key == "test-key"


def test_openai_provider_env_key_name() -> None:
    """Test OpenAI provider env key name."""
    assert OpenAIProvider.get_env_api_key_name() == "OPENAI_API_KEY"


def test_openai_provider_default_base_url() -> None:
    """Test OpenAI provider default base URL."""
    assert OpenAIProvider.get_default_base_url() == "https://api.openai.com/v1"


def test_openai_provider_model_creation() -> None:
    """Test model creation with correct field names."""
    model = Model(
        id="gpt-4o",
        name="GPT-4o",
        api="openai-completions",
        provider="openai",
        base_url="https://api.openai.com/v1",
        cost=ModelCost(input=2.5, output=10.0),
        context_window=128000,
        max_tokens=16384,
    )
    assert model.id == "gpt-4o"
    assert model.base_url == "https://api.openai.com/v1"


def test_openai_provider_list_models() -> None:
    """Test list_models returns models."""
    provider = OpenAIProvider()
    import asyncio
    models = asyncio.run(provider.list_models())
    assert len(models) == 4
    assert models[0].id == "gpt-4o"


def test_openai_provider_get_model() -> None:
    """Test get_model returns correct model."""
    provider = OpenAIProvider()
    model = provider.get_model("gpt-4o")
    assert model is not None
    assert model.id == "gpt-4o"


def test_openai_provider_get_model_not_found() -> None:
    """Test get_model returns None for unknown model."""
    provider = OpenAIProvider()
    model = provider.get_model("unknown-model")
    assert model is None


@pytest.mark.asyncio
async def test_openai_stream_without_api_key() -> None:
    """Test streaming without API key raises error."""
    provider = OpenAIProvider()
    model = Model(
        id="gpt-4o",
        name="GPT-4o",
        api="openai-completions",
        provider="openai",
        base_url="https://api.openai.com/v1",
    )
    context = Context(
        messages=[UserMessage(content="Hello", timestamp=1000)]
    )

    with pytest.raises(ValueError, match="OpenAI API key is required"):
        async for _ in provider.stream(model, context):
            pass
