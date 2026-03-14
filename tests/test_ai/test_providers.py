"""Tests for pi.ai.providers.base module."""

from pi.ai.providers.base import ProviderRegistry, provider


def test_provider_registry_register() -> None:
    """Test ProviderRegistry.register decorator."""

    @ProviderRegistry.register
    class TestProvider:
        @property
        def name(self) -> str:
            return "test-provider"

    assert "test-provider" in ProviderRegistry.list_providers()


def test_provider_decorator() -> None:
    """Test provider decorator."""

    @provider("custom-provider")
    class CustomProvider:
        pass

    assert "custom-provider" in ProviderRegistry.list_providers()


def test_provider_registry_get() -> None:
    """Test ProviderRegistry.get method."""

    @provider("get-test-provider")
    class GetTestProvider:
        pass

    retrieved = ProviderRegistry.get("get-test-provider")
    assert retrieved is not None


def test_provider_registry_get_nonexistent() -> None:
    """Test ProviderRegistry.get with nonexistent provider."""
    retrieved = ProviderRegistry.get("nonexistent-provider")
    assert retrieved is None
