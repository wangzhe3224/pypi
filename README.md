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
| Phase 5 | `pi.cli` - Core tools (read, bash, edit, write) | ✅ Complete |
| Phase 6 | `pi.tui` - Interactive interface | 🚧 Pending |

## CLI Usage

```bash
# Basic usage (defaults to gpt-4o)
pi "Explain this codebase"

# Test without API key using dummy model
pi -m dummy "hello world"

# Specify a model
pi -m claude-sonnet-4 "Refactor this function"

# List available models
pi --list-models

# Use a session
pi -s my-session "Continue our discussion"

# Interactive
pi --mode interactive -m dummy 
```

**Available Models:**
- `dummy` - For testing (no API key required)
- `gpt-4o`, `gpt-4o-mini` - OpenAI
- `claude-sonnet-4`, `claude-3-5-sonnet` - Anthropic
- `gemini-2.0-flash` - Google

**Environment Variables:**
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `GOOGLE_API_KEY` - Google API key

## Modules

### `pi.ai` - LLM API Abstraction

Unified API for multiple LLM providers with streaming support.

```python
from pi.ai import Model, stream, Context

# Create a model
model = Model(
    id="gpt-4o",
    name="GPT-4o",
    api="openai-completions",
    provider="openai",
    baseUrl="https://api.openai.com/v1",
)

# Stream responses
context = Context(messages=[{"role": "user", "content": "Hello"}])
async for event in stream(model, context):
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
from pi.ai.models import resolve_model

# Define a tool
@tool
def read(file_path: str) -> str:
    """Read a file."""
    with open(file_path) as f:
        return f.read()

# Create and run agent
model = resolve_model("gpt-4o")
agent = Agent(model=model, tools=[read])

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
