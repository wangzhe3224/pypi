"""Provider Protocol for LLM backends."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pi.ai.types import Context, Model, StreamEvent, StreamOptions


@runtime_checkable
class Provider(Protocol):
    """Protocol for LLM providers.

    All providers must implement this interface to be used with pi-ai.
    """

    @property
    def name(self) -> str:
        """Provider name (e.g., 'openai', 'anthropic')."""
        ...

    @property
    def api_type(self) -> str:
        """API type identifier (e.g., 'openai-completions', 'anthropic-messages')."""
        ...

    @abstractmethod
    def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream responses from the LLM.

        Args:
            model: The model configuration to use.
            context: The conversation context (system prompt, messages, tools).
            options: Optional streaming options (temperature, max_tokens, etc.).

        Yields:
            StreamEvent: Events as they occur during generation.
        """
        ...

    @abstractmethod
    async def list_models(self) -> list[Model]:
        """List available models for this provider.

        Returns:
            List of Model configurations.
        """
        ...

    @abstractmethod
    def get_model(self, model_id: str) -> Model | None:
        """Get a specific model by ID.

        Args:
            model_id: The model identifier.

        Returns:
            Model configuration if found, None otherwise.
        """
        ...

    @staticmethod
    def get_env_api_key_name() -> str | None:
        """Get the environment variable name for the API key.

        Returns:
            Environment variable name (e.g., 'OPENAI_API_KEY'), or None if not applicable.
        """
        ...

    @staticmethod
    def get_default_base_url() -> str:
        """Get the default base URL for this provider.

        Returns:
            Default API base URL.
        """
        ...


class ProviderRegistry:
    """Registry for LLM providers."""

    _providers: dict[str, type[Provider]] = {}

    @classmethod
    def register(cls, provider_class: type[Provider]) -> type[Provider]:
        """Register a provider class.

        Can be used as a decorator:

            @ProviderRegistry.register
            class OpenAIProvider(Provider):
                ...
        """
        instance = provider_class()
        cls._providers[instance.name] = provider_class
        return provider_class

    @classmethod
    def get(cls, name: str) -> Provider | None:
        """Get a provider instance by name.

        Args:
            name: Provider name (e.g., 'openai', 'anthropic').

        Returns:
            Provider instance if found, None otherwise.
        """
        provider_class = cls._providers.get(name)
        if provider_class:
            return provider_class()
        return None

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names."""
        return list(cls._providers.keys())


def provider(name: str) -> Callable[[type[Provider]], type[Provider]]:
    """Decorator to register a provider.

    Usage:
        @provider("openai")
        class OpenAIProvider(Provider):
            ...
    """

    def decorator(cls: type[Provider]) -> type[Provider]:
        ProviderRegistry._providers[name] = cls
        return cls

    return decorator
