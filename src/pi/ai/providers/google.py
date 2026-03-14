"""Google GenAI provider implementation."""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from google import genai
from google.genai import types

from pi.ai.providers.base import Provider, provider
from pi.ai.providers.converters_google import (
    convert_messages,
    convert_tools,
    is_thinking_part,
    map_stop_reason,
)

if TYPE_CHECKING:
    from pi.ai.types import Context, Model, StreamEvent, StreamOptions, Usage


@provider("google")
class GoogleProvider(Provider):
    """Google GenAI API provider."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self._api_key = api_key
        self._base_url = base_url

    @property
    def name(self) -> str:
        return "google"

    @property
    def api_type(self) -> str:
        return "google-generative-ai"

    def _get_client(self, model: Model, options: StreamOptions | None = None) -> genai.Client:
        """Create Google GenAI client."""
        api_key = (
            options.api_key
            if options and options.api_key
            else self._api_key or os.environ.get("GOOGLE_API_KEY")
        )
        if not api_key:
            msg = "Google API key is required. Set GOOGLE_API_KEY environment variable"
            raise ValueError(msg)

        http_options: dict[str, Any] = {}
        if self._base_url or model.base_url:
            base = self._base_url or model.base_url
            http_options["baseUrl"] = base
            http_options["apiVersion"] = ""

        if http_options:
            return genai.Client(api_key=api_key, http_options=http_options)  # type: ignore[arg-type]
        return genai.Client(api_key=api_key)

    async def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream responses from Google GenAI."""
        from pi.ai.types import (
            AssistantMessage,
            StreamEventDone,
            StreamEventError,
            StreamEventStart,
            StreamEventTextDelta,
            StreamEventTextEnd,
            StreamEventTextStart,
            StreamEventThinkingDelta,
            StreamEventThinkingEnd,
            StreamEventThinkingStart,
            StreamEventToolCallDelta,
            StreamEventToolCallEnd,
            StreamEventToolCallStart,
            TextContent,
            ThinkingContent,
            ToolCall,
            Usage,
        )

        client = self._get_client(model, options)
        params = self._build_params(model, context, options)

        output = AssistantMessage(
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=int(time.time() * 1000),
        )

        current_block: dict[str, Any] | None = None

        try:
            yield StreamEventStart(partial=output)

            stream = await client.aio.models.generate_content_stream(**params)

            async for chunk in stream:
                candidate = chunk.candidates[0] if chunk.candidates else None
                if not candidate:
                    continue

                # Handle finish reason
                if candidate.finish_reason:
                    output.stop_reason = map_stop_reason(str(candidate.finish_reason))
                    if any(c.type == "toolCall" for c in output.content):
                        output.stop_reason = "toolUse"

                # Process parts
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        # Handle text/thinking
                        if hasattr(part, "text") and part.text:
                            is_thinking = is_thinking_part(
                                {"thought": getattr(part, "thought", None)}
                            )

                            # Check if we need to switch blocks
                            if (
                                not current_block
                                or (is_thinking and current_block["type"] != "thinking")
                                or (not is_thinking and current_block["type"] != "text")
                            ):
                                # End previous block
                                if current_block:
                                    if current_block["type"] == "text":
                                        yield StreamEventTextEnd(
                                            content_index=len(output.content) - 1,
                                            content=current_block["text"],
                                            partial=output,
                                        )
                                    else:
                                        yield StreamEventThinkingEnd(
                                            content_index=len(output.content) - 1,
                                            content=current_block["thinking"],
                                            partial=output,
                                        )

                                # Start new block
                                if is_thinking:
                                    current_block = {
                                        "type": "thinking",
                                        "thinking": "",
                                        "signature": getattr(part, "thought_signature", None),
                                    }
                                    output.content.append(ThinkingContent(thinking=""))
                                    yield StreamEventThinkingStart(
                                        content_index=len(output.content) - 1,
                                        partial=output,
                                    )
                                else:
                                    current_block = {"type": "text", "text": ""}
                                    output.content.append(TextContent(text=""))
                                    yield StreamEventTextStart(
                                        content_index=len(output.content) - 1,
                                        partial=output,
                                    )

                            # Update content and yield delta
                            if is_thinking and current_block["type"] == "thinking":
                                current_block["thinking"] += part.text
                                think_content = output.content[-1]
                                if isinstance(think_content, ThinkingContent):
                                    think_content.thinking = current_block["thinking"]
                                    if hasattr(part, "thought_signature"):
                                        sig = part.thought_signature
                                        think_content.thinking_signature = sig.decode() if isinstance(sig, bytes) else sig

                                yield StreamEventThinkingDelta(
                                    content_index=len(output.content) - 1,
                                    delta=part.text,
                                    partial=output,
                                )
                            elif current_block["type"] == "text":
                                current_block["text"] += part.text
                                text_content = output.content[-1]
                                if isinstance(text_content, TextContent):
                                    text_content.text = current_block["text"]

                                yield StreamEventTextDelta(
                                    content_index=len(output.content) - 1,
                                    delta=part.text,
                                    partial=output,
                                )

                        # Handle function call
                        if hasattr(part, "function_call") and part.function_call:
                            fc = part.function_call

                            # End current block
                            if current_block:
                                if current_block["type"] == "text":
                                    yield StreamEventTextEnd(
                                        content_index=len(output.content) - 1,
                                        content=current_block["text"],
                                        partial=output,
                                    )
                                else:
                                    yield StreamEventThinkingEnd(
                                        content_index=len(output.content) - 1,
                                        content=current_block["thinking"],
                                        partial=output,
                                    )
                                current_block = None

                            # Create tool call
                            tool_call = ToolCall(
                                id=getattr(fc, "id", f"tool_{int(time.time() * 1000)}"),
                                name=fc.name or "",
                                arguments=dict(fc.args) if fc.args else {},
                            )
                            output.content.append(tool_call)

                            content_idx = len(output.content) - 1
                            yield StreamEventToolCallStart(
                                content_index=content_idx,
                                partial=output,
                            )
                            yield StreamEventToolCallDelta(
                                content_index=content_idx,
                                delta=json.dumps(tool_call.arguments),
                                partial=output,
                            )
                            yield StreamEventToolCallEnd(
                                content_index=content_idx,
                                tool_call=tool_call,
                                partial=output,
                            )

                # Handle usage
                if chunk.usage_metadata:
                    meta = chunk.usage_metadata
                    input_tokens = meta.prompt_token_count or 0
                    output_tokens = (meta.candidates_token_count or 0) + (
                        meta.thoughts_token_count or 0
                    )
                    cached_tokens = meta.cached_content_token_count or 0

                    output.usage = Usage(
                        input=input_tokens,
                        output=output_tokens,
                        cacheRead=cached_tokens,
                        cacheWrite=0,
                        totalTokens=meta.total_token_count or 0,
                    )
                    self._calculate_cost(model, output.usage)

            # End final block
            if current_block:
                if current_block["type"] == "text":
                    yield StreamEventTextEnd(
                        content_index=len(output.content) - 1,
                        content=current_block["text"],
                        partial=output,
                    )
                else:
                    yield StreamEventThinkingEnd(
                        content_index=len(output.content) - 1,
                        content=current_block["thinking"],
                        partial=output,
                    )

            yield StreamEventDone(reason=output.stop_reason, message=output)

        except Exception as e:
            output.stop_reason = "error"
            output.error_message = str(e)
            yield StreamEventError(reason="error", error=output)

    def _build_params(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> dict[str, Any]:
        """Build Google GenAI API request parameters."""
        config: dict[str, Any] = {}

        # System instruction
        if context.system_prompt:
            config["system_instruction"] = context.system_prompt

        # Temperature and max tokens
        if options:
            if options.temperature is not None:
                config["temperature"] = options.temperature
            if options.max_tokens is not None:
                config["max_output_tokens"] = options.max_tokens

        # Tools
        if context.tools:
            config["tools"] = convert_tools(context.tools)

        params: dict[str, Any] = {
            "model": model.id,
            "contents": convert_messages(model, context),
        }

        if config:
            params["config"] = types.GenerateContentConfig(**config)

        return params

    def _calculate_cost(self, model: Model, usage: Usage) -> None:
        """Calculate and update cost in usage."""
        from pi.ai.types import Cost

        cost_input = (usage.input / 1_000_000) * model.cost.input
        cost_output = (usage.output / 1_000_000) * model.cost.output
        cost_cached = (usage.cache_read / 1_000_000) * model.cost.cache_read
        total_cost = cost_input + cost_output + cost_cached

        usage.cost = Cost(
            input=cost_input,
            output=cost_output,
            cache_read=cost_cached,
            cacheWrite=0,
            total=total_cost,
        )

    async def list_models(self) -> list[Model]:
        """List available Google models."""
        from pi.ai.types import Model, ModelCost

        return [
            Model(
                id="gemini-2.5-pro-preview-06-05",
                name="Gemini 2.5 Pro",
                api="google-generative-ai",
                provider="google",
                base_url="https://generativelanguage.googleapis.com/v1beta",
                reasoning=True,
                cost=ModelCost(input=1.25, output=10.0),
                context_window=2000000,
                max_tokens=65536,
            ),
            Model(
                id="gemini-2.5-flash-preview-05-20",
                name="Gemini 2.5 Flash",
                api="google-generative-ai",
                provider="google",
                base_url="https://generativelanguage.googleapis.com/v1beta",
                reasoning=True,
                cost=ModelCost(input=0.15, output=3.50),
                context_window=2000000,
                max_tokens=65536,
            ),
            Model(
                id="gemini-2.0-flash",
                name="Gemini 2.0 Flash",
                api="google-generative-ai",
                provider="google",
                base_url="https://generativelanguage.googleapis.com/v1beta",
                cost=ModelCost(input=0.10, output=0.40),
                context_window=1000000,
                max_tokens=8192,
            ),
            Model(
                id="gemini-1.5-pro",
                name="Gemini 1.5 Pro",
                api="google-generative-ai",
                provider="google",
                base_url="https://generativelanguage.googleapis.com/v1beta",
                cost=ModelCost(input=1.25, output=5.0),
                context_window=2000000,
                max_tokens=8192,
            ),
            Model(
                id="gemini-1.5-flash",
                name="Gemini 1.5 Flash",
                api="google-generative-ai",
                provider="google",
                base_url="https://generativelanguage.googleapis.com/v1beta",
                cost=ModelCost(input=0.075, output=0.30),
                context_window=1000000,
                max_tokens=8192,
            ),
        ]

    def get_model(self, model_id: str) -> Model | None:
        """Get a specific model by ID."""
        import asyncio

        models = asyncio.run(self.list_models())
        for model in models:
            if model.id == model_id:
                return model
        return None

    @staticmethod
    def get_env_api_key_name() -> str | None:
        """Get the environment variable name for the API key."""
        return "GOOGLE_API_KEY"

    @staticmethod
    def get_default_base_url() -> str:
        """Get the default base URL for this provider."""
        return "https://generativelanguage.googleapis.com/v1beta"
