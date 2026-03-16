# Pi Python Port Plan

> Port of [pi-mono](https://github.com/badlogic/pi-mono) TypeScript/Node.js monorepo to Python

## Status

- [x] **Phase 1** - Project skeleton + pydantic types ✅ COMPLETE
- [x] **Phase 2** - `pi.ai` - OpenAI provider + streaming ✅ COMPLETE
- [x] **Phase 3** - `pi.ai` - Anthropic, Google providers ✅ COMPLETE
- [x] **Phase 4** - `pi.agent` - Agent loop + tool registry ✅ COMPLETE
  - [x] **Phase 4.1** - Agent types (`types.py`) ✅
  - [x] **Phase 4.2** - Tool registry + decorator (`registry.py`, `tools.py`) ✅
  - [x] **Phase 4.3** - Agent loop (`core.py`) ✅
  - [x] **Phase 4.4** - Agent class (`agent.py`) ✅
- [x] **Phase 5** - `pi.cli` - Core tools (read, bash, edit, write, ✅ COMPLETE
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

---

## Phase 4 Detailed Breakdown

Phase 4 implements the agent runtime with the ReAct (Reason-Act-Observe) pattern.
Based on analysis of `pi-mono/packages/agent/src/`.

### Phase 4.1 - Agent Types (`src/pi/agent/types.py`)

**Goal**: Define all agent-specific types using pydantic v2.

**Types to implement**:

```python
# AgentMessage - Union of LLM messages + extensible custom messages
AgentMessage = UserMessage | AssistantMessage | ToolResultMessage

# AgentContext - Conversation context for agent
AgentContext:
    system_prompt: str
    messages: list[AgentMessage]
    tools: list[AgentTool] | None

# AgentState - Full state including streaming status
AgentState:
    system_prompt: str
    model: Model
    thinking_level: ThinkingLevel
    tools: list[AgentTool]
    messages: list[AgentMessage]
    is_streaming: bool
    stream_message: AgentMessage | None
    pending_tool_calls: set[str]
    error: str | None

# AgentTool - Tool with execute method
AgentTool(Tool):
    label: str  # Human-readable label for UI
    execute: Callable  # async (tool_call_id, params, signal, on_update) -> AgentToolResult

# AgentToolResult - Tool execution result
AgentToolResult:
    content: list[TextContent | ImageContent]
    details: Any  # Additional metadata

# AgentEvent - Events emitted during agent loop
AgentEvent =
    | {"type": "agent_start"}
    | {"type": "agent_end", "messages": list[AgentMessage]}
    | {"type": "turn_start"}
    | {"type": "turn_end", "message": AgentMessage, "tool_results": list[ToolResultMessage]}
    | {"type": "message_start", "message": AgentMessage}
    | {"type": "message_update", "message": AgentMessage, "event": StreamEvent}
    | {"type": "message_end", "message": AgentMessage}
    | {"type": "tool_execution_start", "tool_call_id": str, "tool_name": str, "args": dict}
    | {"type": "tool_execution_update", "tool_call_id": str, "tool_name": str, "args": dict, "partial_result": Any}
    | {"type": "tool_execution_end", "tool_call_id": str, "tool_name": str, "result": AgentToolResult, "is_error": bool}

# AgentLoopConfig - Configuration for agent loop
AgentLoopConfig(SimpleStreamOptions):
    model: Model
    convert_to_llm: Callable[[list[AgentMessage]], list[Message]]  # Transform messages for LLM
    transform_context: Callable | None  # Pre-transform (e.g., context pruning)
    get_api_key: Callable | None  # Dynamic API key resolution
    get_steering_messages: Callable | None  # Mid-run interruptions
    get_follow_up_messages: Callable | None  # Post-run continuations
```

**Files**: `src/pi/agent/types.py`

**Dependencies**: `pi.ai.types` (Message, Tool, StreamEvent, etc.)

---

### Phase 4.2 - Tool Registry & Decorator (`src/pi/agent/registry.py`, `src/pi/agent/tools.py`)

**Goal**: Implement tool registration and `@tool` decorator with JSON Schema generation.

**Registry (`registry.py`)**:
```python
class ToolRegistry:
    """Registry for agent tools."""
    
    _tools: dict[str, AgentTool]
    
    @classmethod
    def register(cls, tool: AgentTool) -> None: ...
    
    @classmethod
    def get(cls, name: str) -> AgentTool | None: ...
    
    @classmethod
    def list_tools(cls) -> list[AgentTool]: ...
    
    @classmethod
    def to_llm_tools(cls) -> list[Tool]: ...
```

**Decorator (`tools.py`)**:
```python
@overload
def tool(func: Callable[..., T]) -> AgentTool: ...
@overload
def tool(*, name: str, description: str) -> Callable[[Callable[..., T]], AgentTool]: ...

def tool(func=None, *, name=None, description=None):
    """
    Decorator to register a function as an agent tool.
    
    - Auto-generates JSON Schema from type hints
    - Registers to ToolRegistry
    - Wraps sync functions with asyncio.to_thread()
    
    Usage:
        @tool
        def read(file_path: str, offset: int = 0) -> str:
            """Read file contents."""
            return open(file_path).read()
        
        @tool(name="bash", description="Execute shell command")
        async def run_bash(command: str) -> str:
            proc = await asyncio.create_subprocess_shell(command)
            ...
    """
```

**Files**: 
- `src/pi/agent/registry.py`
- `src/pi/agent/tools.py`

**Dependencies**: Phase 4.1

---

### Phase 4.3 - Agent Loop (`src/pi/agent/core.py`)

**Goal**: Implement the ReAct pattern - the core agent execution loop.

**Key Components**:

```python
async def agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None = None,
) -> AsyncGenerator[AgentEvent, None]:
    """
    Main agent loop implementing ReAct pattern.
    
    Flow:
    1. Add prompts to context
    2. Emit agent_start, turn_start, message_start/end
    3. Call run_loop()
    
    Yields AgentEvent for UI updates.
    """

async def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None = None,
) -> AsyncGenerator[AgentEvent, None]:
    """
    Continue from current context (for retries/resuming).
    """

async def run_loop(context, config, signal) -> AsyncGenerator[AgentEvent, None]:
    """
    Inner loop logic:
    
    while True:
        # Inner loop: process tool calls and steering messages
        while has_more_tool_calls or pending_messages:
            # Process pending steering messages
            # Stream assistant response via LLM
            # Execute tool calls if any
            # Emit turn_end
        
        # Check for follow-up messages
        # If none, break
    
    Emit agent_end
    """

async def stream_assistant_response(context, config, signal) -> AssistantMessage:
    """
    Stream LLM response, converting AgentMessage[] to Message[] at boundary.
    
    - Apply transform_context if configured
    - Convert to LLM messages
    - Stream and emit events
    - Return final AssistantMessage
    """

async def execute_tool_calls(tools, message, signal) -> list[ToolResultMessage]:
    """
    Execute all tool calls from assistant message.
    
    - Validate arguments against tool schema
    - Execute each tool
    - Handle errors gracefully
    - Check for steering messages (skip remaining tools if user interrupted)
    - Return ToolResultMessages
    """
```

**Key Patterns**:
- Transform `AgentMessage[]` → `Message[]` only at LLM call boundary
- Steering messages: interrupt mid-run, skip remaining tools
- Follow-up messages: continue after agent would stop
- Abort handling via `asyncio.Event`

**Files**: `src/pi/agent/core.py`

**Dependencies**: Phase 4.1, Phase 4.2, `pi.ai.stream`

---

### Phase 4.4 - Agent Class (`src/pi/agent/agent.py`)

**Goal**: High-level Agent API with state management and event subscription.

```python
class Agent:
    """High-level agent with state management and event subscription."""
    
    def __init__(self, options: AgentOptions = None):
        self._state = AgentState(...)
        self._listeners: set[Callable[[AgentEvent], None]]
        self._abort_event: asyncio.Event | None
        self._steering_queue: list[AgentMessage]
        self._follow_up_queue: list[AgentMessage]
    
    # State accessors
    @property
    def state(self) -> AgentState: ...
    
    # Event subscription
    def subscribe(self, fn: Callable[[AgentEvent], None]) -> Callable[[], None]:
        """Subscribe to agent events. Returns unsubscribe function."""
    
    # State mutators
    def set_system_prompt(self, prompt: str): ...
    def set_model(self, model: Model): ...
    def set_thinking_level(self, level: ThinkingLevel): ...
    def set_tools(self, tools: list[AgentTool]): ...
    def append_message(self, message: AgentMessage): ...
    def replace_messages(self, messages: list[AgentMessage]): ...
    
    # Message queuing
    def steer(self, message: AgentMessage): 
        """Queue steering message (interrupts mid-run)."""
    
    def follow_up(self, message: AgentMessage):
        """Queue follow-up message (continues after stop)."""
    
    # Control
    async def prompt(self, input: str | AgentMessage | list[AgentMessage]): 
        """Send a prompt and run the agent loop."""
    
    async def continue_(self):
        """Continue from current context (for retries/queued messages)."""
    
    def abort(self):
        """Abort current execution."""
    
    async def wait_for_idle(self) -> None:
        """Wait for agent to finish current operation."""
    
    def reset(self):
        """Clear messages and reset state."""
```

**AgentOptions**:
```python
@dataclass
class AgentOptions:
    initial_state: Partial[AgentState] | None
    convert_to_llm: Callable | None  # Custom message conversion
    transform_context: Callable | None  # Context pruning
    steering_mode: Literal["all", "one-at-a-time"] = "one-at-a-time"
    follow_up_mode: Literal["all", "one-at-a-time"] = "one-at-a-time"
    session_id: str | None
    get_api_key: Callable | None
```

**Files**: `src/pi/agent/agent.py`

**Dependencies**: Phase 4.3

---

### Phase 4 File Structure

```
src/pi/agent/
├── __init__.py      # Export Agent, agent_loop, @tool, types
├── types.py         # AgentMessage, AgentContext, AgentState, AgentTool, AgentEvent
├── registry.py      # ToolRegistry
├── tools.py         # @tool decorator
├── core.py          # agent_loop, agent_loop_continue, run_loop
└── agent.py         # Agent class
```

### Phase 4 Testing Strategy

Each subtask should include basic tests:

- **4.1**: Type serialization, message conversion
- **4.2**: Tool registration, schema generation, sync/async execution
- **4.3**: Loop execution, tool calling, steering/follow-up handling
- **4.4**: Full agent workflow, event subscription, abort handling


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
