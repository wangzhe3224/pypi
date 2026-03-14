"""Anthropic provider implementation."""

from __future__ import annotations

import os
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from anthropic import AsyncAnthropic

from pi.ai.providers.base import Provider, provider
from pi.ai.providers.converters_anthropic import (
    convert_messages,
    convert_tools,
    map_stop_reason,
    parse_streaming_json,
)

if TYPE_CHECKING:
    from pi.ai.types import Context, Model, StreamEvent, StreamOptions, Usage


@provider("anthropic")
class AnthropicProvider(Provider):
    """Anthropic API provider."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self._api_key = api_key
        self._base_url = base_url

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def api_type(self) -> str:
        return "anthropic-messages"

    def _get_client(self, model: Model, options: StreamOptions | None = None) -> AsyncAnthropic:
        """Create Anthropic client."""
        api_key = (
            options.api_key
            if options and options.api_key
            else self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        if not api_key:
            msg = "Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable"
            raise ValueError(msg)

        base_url = self._base_url or model.base_url
        return AsyncAnthropic(api_key=api_key, base_url=base_url)

    async def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream responses from Anthropic."""
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

        blocks: dict[int, dict[str, Any]] = {}
        partial_jsons: dict[int, str] = {}

        try:
            yield StreamEventStart(partial=output)

            async with client.messages.stream(**params) as stream:
                async for event in stream:
                    if event.type == "message_start":
                        msg = event.message
                        usage = msg.usage
                        output.usage = Usage(
                            input=usage.input_tokens,
                            output=usage.output_tokens,
                            cache_read=getattr(usage, "cache_read_input_tokens", 0) or 0,
                            cache_write=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                            total_tokens=(
                                usage.input_tokens
                                + usage.output_tokens
                                + (getattr(usage, "cache_read_input_tokens", 0) or 0)
                                + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
                            ),
                        )
                        self._calculate_cost(model, output.usage)

                    elif event.type == "content_block_start":
                        idx = event.index
                        block = event.content_block

                        if block.type == "text":
                            blocks[idx] = {"type": "text", "text": ""}
                            output.content.append(TextContent(text=""))
                            yield StreamEventTextStart(
                                content_index=len(output.content) - 1,
                                partial=output,
                            )

                        elif block.type == "thinking":
                            blocks[idx] = {"type": "thinking", "thinking": "", "signature": ""}
                            output.content.append(ThinkingContent(thinking=""))
                            yield StreamEventThinkingStart(
                                content_index=len(output.content) - 1,
                                partial=output,
                            )

                        elif block.type == "tool_use":
                            blocks[idx] = {
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                            }
                            partial_jsons[idx] = ""
                            output.content.append(
                                ToolCall(id=block.id, name=block.name, arguments={})
                            )
                            yield StreamEventToolCallStart(
                                content_index=len(output.content) - 1,
                                partial=output,
                            )

                    elif event.type == "content_block_delta":
                        idx = event.index
                        delta = event.delta
                        block_info = blocks.get(idx)

                        if delta.type == "text_delta" and block_info:
                            block_info["text"] += delta.text
                            text_content = output.content[-1]
                            if isinstance(text_content, TextContent):
                                text_content.text = block_info["text"]
                            yield StreamEventTextDelta(
                                content_index=self._block_to_content_idx(blocks, idx, output),
                                delta=delta.text,
                                partial=output,
                            )

                        elif delta.type == "thinking_delta" and block_info:
                            block_info["thinking"] += delta.thinking
                            think_content = output.content[-1]
                            if isinstance(think_content, ThinkingContent):
                                think_content.thinking = block_info["thinking"]
                            yield StreamEventThinkingDelta(
                                content_index=self._block_to_content_idx(blocks, idx, output),
                                delta=delta.thinking,
                                partial=output,
                            )

                        elif delta.type == "input_json_delta" and block_info:
                            partial_jsons[idx] += delta.partial_json
                            args = parse_streaming_json(partial_jsons[idx])
                            tool_content = output.content[-1]
                            if isinstance(tool_content, ToolCall):
                                tool_content.arguments = args
                            yield StreamEventToolCallDelta(
                                content_index=self._block_to_content_idx(blocks, idx, output),
                                delta=delta.partial_json,
                                partial=output,
                            )

                    elif event.type == "content_block_stop":
                        idx = event.index
                        block_info = blocks.get(idx)
                        content_idx = self._block_to_content_idx(blocks, idx, output)

                        if block_info:
                            if block_info["type"] == "text":
                                yield StreamEventTextEnd(
                                    content_index=content_idx,
                                    content=block_info["text"],
                                    partial=output,
                                )
                            elif block_info["type"] == "thinking":
                                yield StreamEventThinkingEnd(
                                    content_index=content_idx,
                                    content=block_info["thinking"],
                                    partial=output,
                                )
                            elif block_info["type"] == "tool_use":
                                args = parse_streaming_json(partial_jsons.get(idx, ""))
                                tool_content = output.content[content_idx]
                                if isinstance(tool_content, ToolCall):
                                    tool_content.arguments = args
                                yield StreamEventToolCallEnd(
                                    content_index=content_idx,
                                    tool_call=tool_content,
                                    partial=output,
                                )

                    elif event.type == "message_delta":
                        msg_delta = event.delta
                        msg_usage = event.usage

                        if msg_delta.stop_reason:
                            output.stop_reason = map_stop_reason(msg_delta.stop_reason)

                        if msg_usage.output_tokens is not None:
                            output.usage.output = msg_usage.output_tokens
                        if hasattr(msg_usage, "cache_read_input_tokens") and msg_usage.cache_read_input_tokens:
                            output.usage.cache_read = msg_usage.cache_read_input_tokens
                        if hasattr(msg_usage, "cache_creation_input_tokens") and msg_usage.cache_creation_input_tokens:
                            output.usage.cache_write = msg_usage.cache_creation_input_tokens


                        output.usage.total_tokens = (
                            output.usage.input
                            + output.usage.output
                            + output.usage.cache_read
                            + output.usage.cache_write
                        )
                        self._calculate_cost(model, output.usage)

            yield StreamEventDone(reason=output.stop_reason, message=output)

        except Exception as e:
            output.stop_reason = "error"
            output.error_message = str(e)
            yield StreamEventError(reason="error", error=output)

    def _block_to_content_idx(
        self, blocks: dict[int, dict[str, Any]], block_idx: int, output: Any
    ) -> int:
        """Map block index to content index."""
        sorted_indices = sorted(blocks.keys())
        try:
            return sorted_indices.index(block_idx)
        except ValueError:
            return len(output.content) - 1

    def _build_params(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> dict[str, Any]:
        """Build Anthropic API request parameters."""
        params: dict[str, Any] = {
            "model": model.id,
            "messages": convert_messages(model, context),
            "max_tokens": options.max_tokens if options and options.max_tokens else 4096,
        }

        if context.system_prompt:
            params["system"] = context.system_prompt

        if options and options.temperature is not None:
            params["temperature"] = options.temperature

        if context.tools:
            params["tools"] = convert_tools(context.tools)

        return params

    def _calculate_cost(self, model: Model, usage: Usage) -> None:
        """Calculate and update cost in usage."""
        from pi.ai.types import Cost

        cost_input = (usage.input / 1_000_000) * model.cost.input
        cost_output = (usage.output / 1_000_000) * model.cost.output
        cost_cached = (usage.cache_read / 1_000_000) * model.cost.cache_read
        cost_write = (usage.cache_write / 1_000_000) * model.cost.cache_write
        total_cost = cost_input + cost_output + cost_cached + cost_write

        usage.cost = Cost(
            input=cost_input,
            output=cost_output,
            cache_read=cost_cached,
            cache_write=cost_write,
            total=total_cost,
        )

    async def list_models(self) -> list[Model]:
        """List available Anthropic models."""
        from pi.ai.types import Model, ModelCost

        return [
            Model(
                id="claude-sonnet-4-20250514",
                name="Claude Sonnet 4",
                api="anthropic-messages",
                provider="anthropic",
                base_url="https://api.anthropic.com",
                reasoning=True,
                cost=ModelCost(input=3.0, output=15.0),
                context_window=200000,
                max_tokens=16000,
            ),
            Model(
                id="claude-3-5-sonnet-20241022",
                name="Claude 3.5 Sonnet",
                api="anthropic-messages",
                provider="anthropic",
                base_url="https://api.anthropic.com",
                cost=ModelCost(input=3.0, output=15.0),
                context_window=200000,
                max_tokens=8192,
            ),
            Model(
                id="claude-3-5-haiku-20241022",
                name="Claude 3.5 Haiku",
                api="anthropic-messages",
                provider="anthropic",
                base_url="https://api.anthropic.com",
                cost=ModelCost(input=0.80, output=4.0),
                context_window=200000,
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
        return "ANTHROPIC_API_KEY"

    @staticmethod
    def get_default_base_url() -> str:
        """Get the default base URL for this provider."""
        return "https://api.anthropic.com"
