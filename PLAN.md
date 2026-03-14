# Pi Python Port Plan

> Port of [pi-mono](https://github.com/badlogic/pi-mono) TypeScript/Node.js monorepo to Python

## Status

- [x] **Phase 1** - Project skeleton + pydantic types ✅ COMPLETE
- [x] **Phase 2** - `pi.ai` - OpenAI provider + streaming ✅ COMPLETE
- [x] **Phase 3** - `pi.ai` - Anthropic, Google providers ✅ COMPLETE
- [ ] **Phase 4** - `pi.agent` - Agent loop + tool registry
- [ ] **Phase 5** - `pi.cli` - Core tools (read, bash, edit, write)
- [ ] **Phase 6** - `pi.tui` - Interactive interface
- [ ] **Phase 7** - Tests + documentation

Port the core packages of pi-mono (AI coding agent ecosystem) to Python with pythonic implementations.

**User Preferences:**
- Single Python package with submodules (not monorepo)
- TUI: Rich + Prompt Toolkit
- Scope: Core packages first (ai, agent, coding-agent, tui)
- Async: asyncio native

## Project Structure

```
pypi/
├── pyproject.toml           # uv + hatchling, Python 3.12+
├── src/
│   └── pi/
│       ├── __init__.py
│       ├── ai/               # LLM API abstraction layer
│       │   ├── __init__.py
│       │   ├── types.py      # pydantic: Message, Tool, StreamEvent
│       │   ├── providers/    # OpenAI, Anthropic, Google, Bedrock, Mistral
│       │   │   ├── __init__.py
│       │   │   ├── base.py   # Provider Protocol
│       │   │   ├── openai.py
│       │   │   ├── anthropic.py
│       │   │   └── ...
│       │   ├── stream.py     # Async generator streaming
│       │   └── models.py     # Model registry + discovery
│       ├── agent/            # Agent runtime
│       │   ├── __init__.py
│       │   ├── core.py       # Agent loop (ReAct pattern)
│       │   ├── tools.py      # Tool Protocol + @tool decorator
│       │   ├── state.py      # Conversation state
│       │   └── registry.py   # Tool registry
│       ├── tui/              # Terminal UI
│       │   ├── __init__.py
│       │   ├── app.py        # Prompt Toolkit app
│       │   ├── renderer.py   # Rich text generation
│       │   └── components/   # Chat, input, status panels
│       └── cli/              # Coding agent CLI
│           ├── __init__.py
│           ├── main.py       # Entry point: `pi` command
│           └── tools/        # read, bash, edit, write, glob, grep
├── tests/
│   ├── conftest.py
│   ├── test_ai/
│   ├── test_agent/
│   └── test_cli/
├── .python-version           # 3.12
├── README.md
└── LICENSE
```

## Tech Stack Mapping

| Component | Choice | Reason |
|-----------|--------|--------|
| **Package Manager** | uv | Fast, modern, lockfile |
| **Type System** | pydantic v2 | Runtime validation + JSON Schema generation (tool definitions) |
| **LLM SDKs** | openai, anthropic, google-generativeai, boto3 (Bedrock) | Official SDKs |
| **TUI** | Prompt Toolkit + Rich | PT handles input/layout, Rich only for formatting |
| **Markdown** | markdown-it-py | Standard Markdown parsing |
| **Testing** | pytest + pytest-asyncio | `asyncio_mode = "auto"` |
| **Linter** | ruff + mypy | Replaces biome |

### Original → Python Mapping

| Original (TS/Node) | Python Equivalent |
|--------------------|-------------------|
| npm workspaces | uv (single package) |
| TypeScript 5.9 | Python 3.12+ with type hints |
| openai SDK | openai (official Python SDK) |
| @anthropic-ai/sdk | anthropic (official SDK) |
| @google/genai | google-generativeai |
| chalk | rich (built-in) |
| marked | markdown-it-py |
| vitest | pytest + pytest-asyncio |
| biome | ruff + mypy |
| tsgo | uv/hatch build |

## Core Design Patterns

### 1. Stream Events (pydantic union)

```python
# pi/ai/types.py
from pydantic import BaseModel
from typing import Literal

class TextEvent(BaseModel):
    type: Literal["text"] = "text"
    content: str

class ToolCallEvent(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    tool_name: str
    arguments: dict

class UsageEvent(BaseModel):
    type: Literal["usage"] = "usage"
    input_tokens: int
    output_tokens: int

class StopEvent(BaseModel):
    type: Literal["stop"] = "stop"
    reason: str

StreamEvent = TextEvent | ToolCallEvent | UsageEvent | StopEvent
```

### 2. Provider Protocol

```python
# pi/ai/providers/base.py
from typing import Protocol, AsyncGenerator

class Provider(Protocol):
    async def stream(
        self, 
        messages: list[Message], 
        tools: list[Tool] | None = None,
        **options
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream responses from the LLM."""
        ...
```

### 3. Tool Decorator

```python
# pi/agent/tools.py
from typing import Callable

def tool(func: Callable) -> Callable:
    """Decorator to register a function as a tool."""
    # Auto-register to registry
    # Generate JSON Schema from type hints
    ...

@tool
def read(file_path: str, offset: int = 0, limit: int = 2000) -> str:
    """Read file contents."""
    ...
```

### 4. TUI Architecture

- **Prompt Toolkit**: Input handling, layout, key bindings
- **Rich**: Only for generating formatted text (`Console(file=StringIO)`)
- **Differential rendering**: PT's `invalidate()` + content caching

**Key insight**: Don't mix PT and Rich render loops. PT renders, Rich only formats.

## Entry Points

```toml
# pyproject.toml
[project.scripts]
pi = "pi.cli.main:main"           # Main coding agent CLI
pi-ai = "pi.ai.cli:main"          # Model management CLI
```

## Implementation Phases

| Phase | Content | Dependencies |
|-------|---------|--------------|
| **Phase 1** | Project skeleton + pydantic types | - |
| **Phase 2** | `pi.ai` - OpenAI provider + streaming | Phase 1 |
| **Phase 3** | `pi.ai` - Anthropic, Google providers | Phase 2 |
| **Phase 4** | `pi.agent` - Agent loop + tool registry | Phase 2 |
| **Phase 5** | `pi.cli` - Core tools (read, bash, edit, write) | Phase 4 |
| **Phase 6** | `pi.tui` - Interactive interface | Phase 5 |
| **Phase 7** | Tests + documentation | All |

## Oracle Review Notes

### Validated Decisions
1. Single package is appropriate for 4 core modules
2. Pydantic v2 for type-safe message passing + JSON Schema generation
3. Async generators for streaming (Python's natural primitive)
4. Separation of PT (input) and Rich (formatting) responsibilities

### Risks Identified
1. **Rich + PT integration**: Don't use `rich.prompt.Prompt` inside PT apps
2. **Async context mixing**: Wrap sync tools with `asyncio.to_thread()` at boundary
3. **Streaming error handling**: Wrap yielded events in try/except at consumer level

### Edge Cases
- If TUI grows complex (split panes, real-time logs): Consider `textual` instead
- If provider plugins needed later: Design `Provider` protocol with `register_provider()` entry point now

## Dependencies

```toml
[project]
dependencies = [
    "pydantic>=2.0",
    "openai>=1.0",
    "anthropic>=0.18",
    "google-generativeai>=0.3",
    "boto3>=1.34",  # Bedrock
    "rich>=13.0",
    "prompt-toolkit>=3.0",
    "markdown-it-py>=3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.1",
    "mypy>=1.8",
]
```

## Reference

- Original project: `/Users/zhewang/Projects/2026/pi-mono`
- Key files to reference:
  - `packages/ai/src/types.ts` - Core types
  - `packages/ai/src/stream.ts` - Streaming logic
  - `packages/agent/src/` - Agent implementation
  - `packages/coding-agent/src/tools/` - Tool implementations
  - `AGENTS.md` - Development rules (adapt for Python)
