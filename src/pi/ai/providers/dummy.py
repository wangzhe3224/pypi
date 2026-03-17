from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncGenerator

from pi.ai.providers.base import Provider, provider
from pi.ai.types import (
    AssistantMessage,
    Context,
    Model,
    StreamEvent,
    StreamEventDone,
    StreamEventStart,
    StreamEventTextDelta,
    StreamEventTextEnd,
    StreamEventTextStart,
    StreamOptions,
    TextContent,
    Usage,
)


@provider("dummy")
class DummyProvider(Provider):
    """Dummy provider for testing without API calls."""

    RESPONSES = [
        "I understand. Let me help you with that.",
        "That's an interesting question! Here's what I think...",
        "I've analyzed the request and here's my response.",
        "Great point! Let me elaborate on this.",
        "Based on my analysis, I would suggest the following approach.",
    ]

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def api_type(self) -> str:
        return "dummy"

    def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        return self._stream(model, context, options)

    async def _stream(
        self,
        model: Model,
        context: Context,
        _options: StreamOptions | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        response = random.choice(self.RESPONSES)

        if context.messages:
            last_msg = context.messages[-1]
            if hasattr(last_msg, "content") and isinstance(last_msg.content, str):
                response = f"You said: '{last_msg.content[:50]}...' Here's my response: {response}"

        output = AssistantMessage(
            content=[TextContent(text="")],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(input=10, output=len(response)),
            stop_reason="stop",
            timestamp=int(time.time() * 1000),
        )

        yield StreamEventStart(partial=output)

        yield StreamEventTextStart(content_index=0, partial=output)

        words = response.split()
        for i, word in enumerate(words):
            await asyncio.sleep(0.02)
            delta = word if i == 0 else f" {word}"
            text_content = output.content[0]
            text_content.text += delta
            yield StreamEventTextDelta(content_index=0, delta=delta, partial=output)

        text_content = output.content[0]
        yield StreamEventTextEnd(content_index=0, content=text_content.text, partial=output)

        yield StreamEventDone(message=output, reason="stop")

    async def list_models(self) -> list[Model]:
        from pi.ai.types import ModelCost

        return [
            Model(
                id="dummy",
                name="Dummy Model",
                api="dummy",
                provider="dummy",
                base_url="http://localhost",
                cost=ModelCost(input=0.0, output=0.0),
                context_window=1000000,
                max_tokens=4096,
            ),
        ]

    def get_model(self, model_id: str) -> Model | None:
        import asyncio

        models = asyncio.run(self.list_models())
        for model in models:
            if model.id == model_id:
                return model
        return None

    @staticmethod
    def get_env_api_key_name() -> str | None:
        return None

    @staticmethod
    def get_default_base_url() -> str:
        return "http://localhost"
