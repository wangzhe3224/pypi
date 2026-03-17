from __future__ import annotations

from pi.ai.types import Model, ModelCost

MODELS: list[Model] = [
    Model(
        id="dummy",
        name="Dummy",
        api="dummy",
        provider="dummy",
        base_url="http://localhost",
        cost=ModelCost(input=0.0, output=0.0),
        context_window=1000000,
        max_tokens=4096,
    ),
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
        id="claude-sonnet-4",
        name="Claude Sonnet 4",
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        cost=ModelCost(input=3.0, output=15.0),
        context_window=200000,
        max_tokens=8192,
    ),
    Model(
        id="claude-3-5-sonnet",
        name="Claude 3.5 Sonnet",
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        cost=ModelCost(input=3.0, output=15.0),
        context_window=200000,
        max_tokens=8192,
    ),
    Model(
        id="gemini-2.0-flash",
        name="Gemini 2.0 Flash",
        api="google-generative-ai",
        provider="google",
        base_url="https://generativelanguage.googleapis.com",
        cost=ModelCost(input=0.1, output=0.4),
        context_window=1000000,
        max_tokens=8192,
    ),
]


def resolve_model(model_id: str) -> Model | None:
    for model in MODELS:
        if model.id == model_id:
            return model
    return None
