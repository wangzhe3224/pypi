"""OpenAI provider implementation."""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI

from pi.ai.providers.base import Provider, provider
from pi.ai.providers.converters import (
    convert_messages,
    convert_tools,
    map_stop_reason,
)

if TYPE_CHECKING:
    from pi.ai.types import (
        AssistantMessage,
        Context,
        Model,
        StreamEvent,
        StreamOptions,
        Usage,
    )


@provider("openai")
class OpenAIProvider(Provider):
    """OpenAI API provider."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self._api_key = api_key
        self._base_url = base_url

    @property
    def name(self) -> str:
        return "openai"

    @property
    def api_type(self) -> str:
        return "openai-completions"

    def _get_client(self, model: Model, options: StreamOptions | None = None) -> AsyncOpenAI:
        """Create OpenAI client."""
        api_key = (
            options.api_key
            if options and options.api_key
            else self._api_key or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass it as an argument."
            )

        base_url = self._base_url or model.base_url

        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    async def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream responses from OpenAI."""
        from pi.ai.types import (
            AssistantMessage,
            StreamEventDone,
            StreamEventError,
            StreamEventStart,
            StreamEventTextDelta,
            StreamEventTextStart,
            StreamEventToolCallDelta,
            StreamEventToolCallStart,
            TextContent,
            ToolCall,
            Usage,
        )

        client = self._get_client(model, options)

        # Build request params
        params = self._build_params(model, context, options)

        # Initialize output message
        output = AssistantMessage(
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=int(time.time() * 1000),
        )

        try:
            # Yield start event
            yield StreamEventStart(partial=output)

            stream = await client.chat.completions.create(**params)

            current_block: dict[str, Any] | None = None
            partial_args = ""

            async for chunk in stream:
                # Handle usage
                if chunk.usage:
                    output.usage = self._extract_usage(chunk, model)

                # Process choices
                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue

                # Handle finish reason
                if choice.finish_reason:
                    output.stop_reason = map_stop_reason(choice.finish_reason)

                delta = choice.delta
                if not delta:
                    continue

                # Handle text content
                if delta.content:
                    if not current_block or current_block.get("type") != "text":
                        if current_block:
                            async for event in self._finish_block(
                                current_block, output, partial_args
                            ):
                                yield event
                        current_block = {"type": "text", "text": ""}
                        output.content.append(TextContent(text=""))
                        yield StreamEventTextStart(
                            content_index=len(output.content) - 1,
                            partial=output,
                        )

                    if current_block.get("type") == "text":
                        current_block["text"] += delta.content
                        # Update the actual content
                        text_content = output.content[-1]
                        if hasattr(text_content, "text"):
                            text_content.text = current_block["text"]

                        yield StreamEventTextDelta(
                            content_index=len(output.content) - 1,
                            delta=delta.content,
                            partial=output,
                        )

                # Handle tool calls
                if delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        tc_id = tool_call_delta.id
                        tc_function = tool_call_delta.function

                        # Start new tool call if needed
                        if tc_id or (not current_block or current_block.get("type") != "toolCall"):
                            if current_block:
                                async for event in self._finish_block(
                                    current_block, output, partial_args
                                ):
                                    yield event
                            current_block = {
                                "type": "toolCall",
                                "id": tc_id or "",
                                "name": "",
                                "arguments": {},
                            }
                            partial_args = ""
                            output.content.append(
                                ToolCall(
                                    id=current_block["id"],
                                    name="",
                                    arguments={},
                                )
                            )
                            yield StreamEventToolCallStart(
                                content_index=len(output.content) - 1,
                                partial=output,
                            )

                        # Update tool call
                        if current_block.get("type") == "toolCall":
                            if tc_id:
                                current_block["id"] = tc_id
                            if tc_function:
                                if tc_function.name:
                                    current_block["name"] = tc_function.name
                                if tc_function.arguments:
                                    partial_args += tc_function.arguments

                            # Update the actual content
                            tool_content = output.content[-1]
                            if isinstance(tool_content, ToolCall):
                                tool_content.id = current_block["id"]
                                tool_content.name = current_block["name"]
                                with contextlib.suppress(json.JSONDecodeError):
                                    tool_content.arguments = json.loads(partial_args)

                            yield StreamEventToolCallDelta(
                                content_index=len(output.content) - 1,
                                delta=tc_function.arguments if tc_function else "",
                                partial=output,
                            )

            # Finish current block
            if current_block:
                async for event in self._finish_block(current_block, output, partial_args):
                    yield event

            # Yield done event
            yield StreamEventDone(reason=output.stop_reason, message=output)

        except Exception as e:
            output.stop_reason = "error"
            output.error_message = str(e)
            yield StreamEventError(reason="error", error=output)

    async def _finish_block(
        self,
        block: dict[str, Any],
        output: AssistantMessage,
        partial_args: str = "",
    ) -> AsyncGenerator[StreamEvent, None]:
        """Finish and yield end event for current block."""
        from pi.ai.types import StreamEventTextEnd, StreamEventToolCallEnd, ToolCall

        block_type = block.get("type")
        index = len(output.content) - 1

        if block_type == "text":
            yield StreamEventTextEnd(
                content_index=index,
                content=block.get("text", ""),
                partial=output,
            )
        elif block_type == "toolCall":
            # Finalize arguments
            with contextlib.suppress(json.JSONDecodeError):
                arguments = json.loads(partial_args)
            tool_content = output.content[-1]
            if isinstance(tool_content, ToolCall):
                tool_content.arguments = arguments

            yield StreamEventToolCallEnd(
                content_index=index,
                tool_call=tool_content,
                partial=output,
            )

    def _build_params(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> dict[str, Any]:
        """Build OpenAI API request parameters."""
        params: dict[str, Any] = {
            "model": model.id,
            "messages": convert_messages(model, context),
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if options:
            if options.max_tokens is not None:
                params["max_completion_tokens"] = options.max_tokens
            if options.temperature is not None:
                params["temperature"] = options.temperature

        if context.tools:
            params["tools"] = convert_tools(context.tools)

        return params

    def _extract_usage(self, chunk: Any, model: Model) -> Usage:
        """Extract usage information from chunk."""
        from pi.ai.types import Cost, Usage

        usage = chunk.usage
        if not usage:
            return Usage()

        cached_tokens = (
            usage.prompt_tokens_details.cached_tokens if usage.prompt_tokens_details else 0
        ) or 0

        reasoning_tokens = (
            usage.completion_tokens_details.reasoning_tokens
            if usage.completion_tokens_details
            else 0
        ) or 0

        input_tokens = (usage.prompt_tokens or 0) - cached_tokens
        output_tokens = (usage.completion_tokens or 0) + reasoning_tokens

        # Calculate cost
        cost_input = (input_tokens / 1_000_000) * model.cost.input
        cost_output = (output_tokens / 1_000_000) * model.cost.output
        cost_cached = (cached_tokens / 1_000_000) * model.cost.cache_read
        total_cost = cost_input + cost_output + cost_cached

        return Usage(
            input=input_tokens,
            output=output_tokens,
            cacheRead=cached_tokens,
            cacheWrite=0,
            totalTokens=input_tokens + output_tokens + cached_tokens,
            cost=Cost(
                input=cost_input,
                output=cost_output,
                cacheRead=cost_cached,
                cacheWrite=0,
                total=total_cost,
            ),
        )

    async def list_models(self) -> list[Model]:
        """List available OpenAI models."""
        # Return common models
        from pi.ai.types import Model, ModelCost

        return [
            Model(
                id="gpt-4o",
                name="GPT-4o",
                api="openai-completions",
                provider="openai",
                base_url="https://api.openai.com/v1",
                cost=ModelCost(input=2.5, output=10.0),
                context_window=128000,
                max_tokens=16384,
            ),
            Model(
                id="gpt-4o-mini",
                name="GPT-4o Mini",
                api="openai-completions",
                provider="openai",
                base_url="https://api.openai.com/v1",
                cost=ModelCost(input=0.15, output=0.6),
                context_window=128000,
                max_tokens=16384,
            ),
            Model(
                id="gpt-4-turbo",
                name="GPT-4 Turbo",
                api="openai-completions",
                provider="openai",
                base_url="https://api.openai.com/v1",
                cost=ModelCost(input=10.0, output=30.0),
                context_window=128000,
                max_tokens=4096,
            ),
            Model(
                id="gpt-3.5-turbo",
                name="GPT-3.5 Turbo",
                api="openai-completions",
                provider="openai",
                base_url="https://api.openai.com/v1",
                cost=ModelCost(input=0.5, output=1.5),
                context_window=16385,
                max_tokens=4096,
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
        return "OPENAI_API_KEY"

    @staticmethod
    def get_default_base_url() -> str:
        """Get the default base URL for this provider."""
        return "https://api.openai.com/v1"
