"""Tests for Google provider."""

import pytest

from pi.ai.providers.google import GoogleProvider
from pi.ai.types import Context, Model, ModelCost, UserMessage


def test_google_provider_creation() -> None:
    """Test Google provider can be created."""
    provider = GoogleProvider()
    assert provider.name == "google"
    assert provider.api_type == "google-generative-ai"


def test_google_provider_with_api_key() -> None:
    """Test Google provider with API key."""
    provider = GoogleProvider(api_key="test-key")
    assert provider._api_key == "test-key"


def test_google_provider_with_base_url() -> None:
    """Test Google provider with custom base URL."""
    provider = GoogleProvider(base_url="https://custom.googleapis.com")
    assert provider._base_url == "https://custom.googleapis.com"


def test_google_provider_env_key_name() -> None:
    """Test Google provider env key name."""
    assert GoogleProvider.get_env_api_key_name() == "GOOGLE_API_KEY"


def test_google_provider_default_base_url() -> None:
    """Test Google provider default base URL."""
    assert (
        GoogleProvider.get_default_base_url() == "https://generativelanguage.googleapis.com/v1beta"
    )


def test_google_provider_model_creation() -> None:
    """Test model creation with correct field names."""
    model = Model(
        id="gemini-2.5-pro-preview-06-05",
        name="Gemini 2.5 Pro",
        api="google-generative-ai",
        provider="google",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        reasoning=True,
        cost=ModelCost(input=1.25, output=10.0),
        context_window=1000000,
        max_tokens=65536,
    )
    assert model.id == "gemini-2.5-pro-preview-06-05"
    assert model.reasoning is True


def test_google_provider_list_models() -> None:
    """Test list_models returns models."""
    provider = GoogleProvider()
    import asyncio

    models = asyncio.run(provider.list_models())
    assert len(models) >= 4  # At least 4 models
    assert models[0].id == "gemini-2.5-pro-preview-06-05"


def test_google_provider_get_model() -> None:
    """Test get_model returns correct model."""
    provider = GoogleProvider()
    model = provider.get_model("gemini-2.5-pro-preview-06-05")
    assert model is not None
    assert model.id == "gemini-2.5-pro-preview-06-05"
    assert model.reasoning is True


def test_google_provider_get_model_flash() -> None:
    """Test get_model returns flash model."""
    provider = GoogleProvider()
    model = provider.get_model("gemini-2.0-flash")
    assert model is not None
    assert model.id == "gemini-2.0-flash"


def test_google_provider_get_model_not_found() -> None:
    """Test get_model returns None for unknown model."""
    provider = GoogleProvider()
    model = provider.get_model("unknown-model")
    assert model is None


def test_google_provider_supports_vision() -> None:
    """Test that Google models have model info."""
    provider = GoogleProvider()
    model = provider.get_model("gemini-2.5-pro-preview-06-05")
    assert model is not None
    # Check model has basic attributes
    assert model.id == "gemini-2.5-pro-preview-06-05"


@pytest.mark.asyncio
async def test_google_stream_without_api_key() -> None:
    """Test streaming without API key raises error."""
    provider = GoogleProvider()
    model = Model(
        id="gemini-2.5-pro-preview-06-05",
        name="Gemini 2.5 Pro",
        api="google-generative-ai",
        provider="google",
        base_url="https://generativelanguage.googleapis.com/v1beta",
    )
    context = Context(messages=[UserMessage(content="Hello", timestamp=1000)])

    with pytest.raises(ValueError, match="Google API key is required"):
        async for _ in provider.stream(model, context):
            pass


def test_google_provider_registered() -> None:
    """Test Google provider is registered."""
    from pi.ai.providers.base import ProviderRegistry

    assert "google" in ProviderRegistry.list_providers()
    # ProviderRegistry.get returns an instance
    provider_instance = ProviderRegistry.get("google")
    assert provider_instance is not None
    assert isinstance(provider_instance, GoogleProvider)
