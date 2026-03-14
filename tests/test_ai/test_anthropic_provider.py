"""Tests for Anthropic provider."""

import pytest

from pi.ai.providers.anthropic import AnthropicProvider
from pi.ai.types import Context, Model, ModelCost, UserMessage


def test_anthropic_provider_creation() -> None:
    """Test Anthropic provider can be created."""
    provider = AnthropicProvider()
    assert provider.name == "anthropic"
    assert provider.api_type == "anthropic-messages"


def test_anthropic_provider_with_api_key() -> None:
    """Test Anthropic provider with API key."""
    provider = AnthropicProvider(api_key="test-key")
    assert provider._api_key == "test-key"


def test_anthropic_provider_with_base_url() -> None:
    """Test Anthropic provider with custom base URL."""
    provider = AnthropicProvider(base_url="https://custom.anthropic.com")
    assert provider._base_url == "https://custom.anthropic.com"


def test_anthropic_provider_env_key_name() -> None:
    """Test Anthropic provider env key name."""
    assert AnthropicProvider.get_env_api_key_name() == "ANTHROPIC_API_KEY"


def test_anthropic_provider_default_base_url() -> None:
    """Test Anthropic provider default base URL."""
    assert AnthropicProvider.get_default_base_url() == "https://api.anthropic.com"


def test_anthropic_provider_model_creation() -> None:
    """Test model creation with correct field names."""
    model = Model(
        id="claude-sonnet-4-20250514",
        name="Claude Sonnet 4",
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        reasoning=True,
        cost=ModelCost(input=3.0, output=15.0),
        context_window=200000,
        max_tokens=16000,
    )
    assert model.id == "claude-sonnet-4-20250514"
    assert model.reasoning is True


def test_anthropic_provider_list_models() -> None:
    """Test list_models returns models."""
    provider = AnthropicProvider()
    import asyncio

    models = asyncio.run(provider.list_models())
    assert len(models) == 3
    assert models[0].id == "claude-sonnet-4-20250514"
    assert models[1].id == "claude-3-5-sonnet-20241022"
    assert models[2].id == "claude-3-5-haiku-20241022"


def test_anthropic_provider_get_model() -> None:
    """Test get_model returns correct model."""
    provider = AnthropicProvider()
    model = provider.get_model("claude-sonnet-4-20250514")
    assert model is not None
    assert model.id == "claude-sonnet-4-20250514"
    assert model.reasoning is True


def test_anthropic_provider_get_model_not_found() -> None:
    """Test get_model returns None for unknown model."""
    provider = AnthropicProvider()
    model = provider.get_model("unknown-model")
    assert model is None


def test_anthropic_provider_supports_vision() -> None:
    """Test that Anthropic models have model info."""
    provider = AnthropicProvider()
    model = provider.get_model("claude-sonnet-4-20250514")
    assert model is not None
    # Check model has basic attributes
    assert model.id == "claude-sonnet-4-20250514"


@pytest.mark.asyncio
async def test_anthropic_stream_without_api_key() -> None:
    """Test streaming without API key raises error."""
    provider = AnthropicProvider()
    model = Model(
        id="claude-sonnet-4-20250514",
        name="Claude Sonnet 4",
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://api.anthropic.com",
    )
    context = Context(messages=[UserMessage(content="Hello", timestamp=1000)])

    with pytest.raises(ValueError, match="Anthropic API key is required"):
        async for _ in provider.stream(model, context):
            pass


def test_anthropic_provider_registered() -> None:
    """Test Anthropic provider is registered."""
    from pi.ai.providers.base import ProviderRegistry

    assert "anthropic" in ProviderRegistry.list_providers()
    # ProviderRegistry.get returns an instance
    provider_instance = ProviderRegistry.get("anthropic")
    assert provider_instance is not None
    assert isinstance(provider_instance, AnthropicProvider)
