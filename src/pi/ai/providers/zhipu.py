"""Zhipu AI (GLM) provider implementation.

Supports GLM-5, GLM-4.7, GLM-4.6, and other Zhipu AI models.
Provides OpenAI-compatible API with extended thinking mode support.

API Documentation:
- International: https://docs.z.ai
- China: https://docs.bigmodel.cn
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Literal

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

# API endpoints
ZHIPU_API_URL_CHINA = "https://open.bigmodel.cn/api/paas/v4/"
ZHIPU_API_URL_INTERNATIONAL = "https://api.z.ai/api/paas/v4/"
ZHIPU_API_URL_CODING = "https://open.bigmodel.cn/api/coding/paas/v4/"


@provider("zhipu")
class ZhipuProvider(Provider):
    """Zhipu AI (GLM) API provider.

    Supports both China and International endpoints.
    Provides extended thinking mode for reasoning tasks.

    Environment Variables:
        ZHIPUAI_API_KEY or ZAI_API_KEY: API key for authentication

    Usage:
        provider = ZhipuProvider()
        async for event in provider.stream(model, context, options):
            print(event)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        region: Literal["china", "international", "coding"] = "china",
        thinking_enabled: bool = True,
    ):
        """Initialize Zhipu AI provider.

        Args:
            api_key: Zhipu AI API key. Falls back to ZHIPUAI_API_KEY or ZAI_API_KEY env vars.
            base_url: Custom base URL. If not set, uses region-specific default.
            region: API region - "china", "international", or "coding" (for GLM Coding Plan).
            thinking_enabled: Whether to enable thinking mode by default for supported models.
        """
        self._api_key = api_key
        self._base_url = base_url
        self._region = region
        self._thinking_enabled = thinking_enabled

    @property
    def name(self) -> str:
        return "zhipu"

    @property
    def api_type(self) -> str:
        return "openai-completions"

    def _get_base_url(self) -> str:
        """Get the base URL based on region."""
        if self._base_url:
            return self._base_url

        if self._region == "international":
            return ZHIPU_API_URL_INTERNATIONAL
        elif self._region == "coding":
            return ZHIPU_API_URL_CODING
        else:
            return ZHIPU_API_URL_CHINA

    def _get_client(self, _model: Model, options: StreamOptions | None = None) -> AsyncOpenAI:
        """Create OpenAI client configured for Zhipu AI."""
        from pi.env import env

        api_key = (
            options.api_key if options and options.api_key else self._api_key or env.zhipuai_api_key
        )
        if not api_key:
            raise ValueError(
                "Zhipu AI API key is required. Set ZHIPUAI_API_KEY or ZAI_API_KEY "
                "environment variable, or pass it as an argument."
            )

        base_url = self._base_url or env.zhipuai_base_url or self._get_base_url()

        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def _supports_thinking(self, model: Model) -> bool:
        """Check if model supports thinking/reasoning mode."""
        thinking_models = {"glm-5", "glm-4.7", "glm-4.6", "glm-4.5", "glm-4.5-air"}
        return model.id.lower() in thinking_models or model.reasoning

    async def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream responses from Zhipu AI.

        Supports thinking mode for GLM-5, GLM-4.7, GLM-4.6, and GLM-4.5 models.
        When thinking is enabled, the model will output reasoning_content before content.
        """
        from pi.ai.types import (
            AssistantMessage,
            StreamEventDone,
            StreamEventError,
            StreamEventStart,
            StreamEventTextDelta,
            StreamEventTextStart,
            StreamEventThinkingDelta,
            StreamEventThinkingStart,
            StreamEventToolCallDelta,
            StreamEventToolCallStart,
            TextContent,
            ThinkingContent,
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

                # Handle reasoning/thinking content (Zhipu AI specific)
                reasoning_content = getattr(delta, "reasoning_content", None)
                if reasoning_content:
                    if not current_block or current_block.get("type") != "thinking":
                        if current_block:
                            async for event in self._finish_block(
                                current_block, output, partial_args
                            ):
                                yield event
                        current_block = {"type": "thinking", "thinking": ""}
                        output.content.append(ThinkingContent(thinking=""))
                        yield StreamEventThinkingStart(
                            content_index=len(output.content) - 1,
                            partial=output,
                        )

                    if current_block.get("type") == "thinking":
                        current_block["thinking"] += reasoning_content
                        # Update the actual content
                        thinking_content = output.content[-1]
                        if hasattr(thinking_content, "thinking"):
                            thinking_content.thinking = current_block["thinking"]

                        yield StreamEventThinkingDelta(
                            content_index=len(output.content) - 1,
                            delta=reasoning_content,
                            partial=output,
                        )

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
        from pi.ai.types import (
            StreamEventTextEnd,
            StreamEventThinkingEnd,
            StreamEventToolCallEnd,
            ToolCall,
        )

        block_type = block.get("type")
        index = len(output.content) - 1

        if block_type == "thinking":
            yield StreamEventThinkingEnd(
                content_index=index,
                content=block.get("thinking", ""),
                partial=output,
            )
        elif block_type == "text":
            yield StreamEventTextEnd(
                content_index=index,
                content=block.get("text", ""),
                partial=output,
            )
        elif block_type == "toolCall":
            # Finalize arguments
            arguments = {}
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
        """Build Zhipu AI API request parameters."""
        params: dict[str, Any] = {
            "model": model.id,
            "messages": convert_messages(model, context),
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if options:
            if options.max_tokens is not None:
                params["max_tokens"] = options.max_tokens
            if options.temperature is not None:
                params["temperature"] = options.temperature

        if context.tools:
            params["tools"] = convert_tools(context.tools)

        # Enable thinking mode for supported models
        if self._supports_thinking(model) and self._thinking_enabled:
            params["extra_body"] = {
                "thinking": {
                    "type": "enabled",
                }
            }

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
        """List available Zhipu AI models."""
        return _get_zhipu_models()

    def get_model(self, model_id: str) -> Model | None:
        """Get a specific model by ID."""
        models = _get_zhipu_models()
        for model in models:
            if model.id == model_id:
                return model
        return None

    @staticmethod
    def get_env_api_key_name() -> str | None:
        """Get the environment variable name for the API key."""
        # Check both variants
        return "ZHIPUAI_API_KEY"

    @staticmethod
    def get_default_base_url() -> str:
        """Get the default base URL for this provider."""
        return ZHIPU_API_URL_CHINA


def _get_zhipu_models() -> list[Model]:
    """Get list of Zhipu AI models."""
    from pi.ai.types import Model, ModelCost

    return [
        # GLM-5 - Flagship model with 200K context
        Model(
            id="glm-5",
            name="GLM-5",
            api="openai-completions",
            provider="zhipu",
            base_url=ZHIPU_API_URL_CHINA,
            reasoning=True,
            cost=ModelCost(
                input=0.5,  # $0.5 per million input tokens
                output=2.0,  # $2.0 per million output tokens
            ),
            context_window=200000,
            max_tokens=128000,
        ),
        # GLM-4.7 - Previous flagship with thinking
        Model(
            id="glm-4.7",
            name="GLM-4.7",
            api="openai-completions",
            provider="zhipu",
            base_url=ZHIPU_API_URL_CHINA,
            reasoning=True,
            cost=ModelCost(
                input=0.3,
                output=1.2,
            ),
            context_window=128000,
            max_tokens=16000,
        ),
        # GLM-4.6 - Vision model
        Model(
            id="glm-4.6",
            name="GLM-4.6",
            api="openai-completions",
            provider="zhipu",
            base_url=ZHIPU_API_URL_CHINA,
            reasoning=True,
            input=["text", "image"],
            cost=ModelCost(
                input=0.2,
                output=0.8,
            ),
            context_window=128000,
            max_tokens=16000,
        ),
        # GLM-4.5 - Standard model with thinking
        Model(
            id="glm-4.5",
            name="GLM-4.5",
            api="openai-completions",
            provider="zhipu",
            base_url=ZHIPU_API_URL_CHINA,
            reasoning=True,
            cost=ModelCost(
                input=0.15,
                output=0.6,
            ),
            context_window=128000,
            max_tokens=16000,
        ),
        # GLM-4.5-Air - Lightweight fast model
        Model(
            id="glm-4.5-air",
            name="GLM-4.5 Air",
            api="openai-completions",
            provider="zhipu",
            base_url=ZHIPU_API_URL_CHINA,
            reasoning=True,
            cost=ModelCost(
                input=0.05,
                output=0.2,
            ),
            context_window=128000,
            max_tokens=16000,
        ),
        # GLM-4-Plus - High performance model
        Model(
            id="glm-4-plus",
            name="GLM-4 Plus",
            api="openai-completions",
            provider="zhipu",
            base_url=ZHIPU_API_URL_CHINA,
            cost=ModelCost(
                input=0.1,
                output=0.4,
            ),
            context_window=128000,
            max_tokens=4096,
        ),
        # GLM-4-Flash - Fast and cheap
        Model(
            id="glm-4-flash",
            name="GLM-4 Flash",
            api="openai-completions",
            provider="zhipu",
            base_url=ZHIPU_API_URL_CHINA,
            cost=ModelCost(
                input=0.01,
                output=0.01,
            ),
            context_window=128000,
            max_tokens=4096,
        ),
        # GLM-4V-Plus - Vision model
        Model(
            id="glm-4v-plus",
            name="GLM-4V Plus",
            api="openai-completions",
            provider="zhipu",
            base_url=ZHIPU_API_URL_CHINA,
            input=["text", "image"],
            cost=ModelCost(
                input=0.1,
                output=0.4,
            ),
            context_window=8192,
            max_tokens=1024,
        ),
    ]
