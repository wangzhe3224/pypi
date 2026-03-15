# pi-ai

Python port of [pi-mono](https://github.com/badlogic/pi-mono): AI coding agent with unified LLM API.

## Installation

```bash
pip install pi-ai
```

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Project skeleton + pydantic types | ✅ Complete |
| Phase 2 | `pi.ai` - OpenAI provider + streaming | ✅ Complete |
| Phase 3 | `pi.ai` - Anthropic, Google providers | ✅ Complete |
| Phase 4 | `pi.agent` - Agent loop + tool registry | ✅ Complete |
| Phase 5 | `pi.cli` - Core tools (read, bash, edit, write) | 🚧 Pending |
| Phase 6 | `pi.tui` - Interactive interface | 🚧 Pending |

## Modules

### `pi.ai` - LLM API Abstraction

Unified API for multiple LLM providers with streaming support.

```python
from pi.ai import Model, stream

# Create a model
model = Model(api="openai-completions", provider="openai", name="gpt-4o")

# Stream responses
async for event in stream(model, messages=[{"role": "user", "content": "Hello"}]):
    if event.type == "text_delta":
        print(event.delta, end="")
```

**Supported Providers:**
- OpenAI (Completions API)
- Anthropic (Messages API)
- Google (Generative AI)
- Amazon Bedrock

### `pi.agent` - Agent Runtime

Agent loop implementing the ReAct (Reason-Act-Observe) pattern.

```python
from pi.agent import Agent, tool

# Define a tool
@tool
def read(file_path: str) -> str:
    """Read a file."""
    with open(file_path) as f:
        return f.read()

# Create and run agent
agent = Agent()
agent.set_model(model)
agent.set_tools([read])

async for event in agent.prompt("Read the README.md file"):
    print(event)
```

**Features:**
- `@tool` decorator with automatic JSON Schema generation
- Tool registry for managing tools
- Steering messages (interrupt mid-run)
- Follow-up messages (continue after stop)
- Event-based subscription for UI updates
- Abort handling

## Development

```bash
# Install dev dependencies
uv sync

# Run tests
uv run pytest

# Type check
uv run mypy src/

# Lint
uv run ruff check src/
```

## License

MIT
