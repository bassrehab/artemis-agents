# Integrations API Reference

This page documents the ARTEMIS framework integrations.

## LangChain Integration

### ArtemisDebateTool

```python
from artemis.integrations import ArtemisDebateTool
```

#### Constructor

```python
ArtemisDebateTool(
    model: str,
    default_rounds: int = 3,
    config: DebateConfig | None = None,
    safety_monitors: list[SafetyMonitor] | None = None,
    name: str = "artemis_debate",
    description: str = "...",
    args_schema: Type[BaseModel] | None = None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | required | LLM model to use |
| `default_rounds` | int | 3 | Default debate rounds |
| `config` | DebateConfig | None | Debate configuration |
| `safety_monitors` | list | None | Safety monitors |
| `name` | str | "artemis_debate" | Tool name |
| `description` | str | ... | Tool description |
| `args_schema` | Type | None | Custom input schema |

#### Methods

##### invoke

```python
def invoke(
    self,
    input: dict,
    config: RunnableConfig | None = None,
) -> DebateResult
```

Runs a synchronous debate.

##### ainvoke

```python
async def ainvoke(
    self,
    input: dict,
    config: RunnableConfig | None = None,
) -> DebateResult
```

Runs an asynchronous debate.

#### Input Schema

```python
{
    "topic": str,           # Required: debate topic
    "rounds": int,          # Optional: number of rounds
    "positions": dict,      # Optional: agent positions
}
```

---

## LangGraph Integration

### ArtemisDebateNode

```python
from artemis.integrations import ArtemisDebateNode
```

#### Constructor

```python
ArtemisDebateNode(
    model: str,
    rounds: int = 3,
    config: DebateConfig | None = None,
    safety_monitor: SafetyMonitor | None = None,
)
```

#### Usage

```python
from langgraph.graph import StateGraph
from artemis.integrations import ArtemisDebateNode, DebateState

workflow = StateGraph(DebateState)
workflow.add_node("debate", ArtemisDebateNode(model="gpt-4o"))
```

### DebateState

```python
from artemis.integrations import DebateState
```

#### Class Definition

```python
class DebateState(TypedDict):
    topic: str
    positions: dict[str, str]
    rounds: int
    current_round: int
    transcript: list[Turn]
    verdict: Verdict | None
    safety_alerts: list[SafetyAlert]
    metadata: dict
```

### create_debate_workflow

```python
from artemis.integrations import create_debate_workflow
```

#### Signature

```python
def create_debate_workflow(
    model: str,
    rounds: int = 3,
    enable_safety: bool = True,
) -> CompiledStateGraph
```

Creates a complete debate workflow.

**Returns:** Compiled LangGraph workflow.

### create_decision_workflow

```python
from artemis.integrations import create_decision_workflow
```

#### Signature

```python
def create_decision_workflow(
    model: str,
    decision_threshold: float = 0.7,
) -> CompiledStateGraph
```

Creates a decision-making workflow using debate.

---

## CrewAI Integration

### ArtemisCrewTool

```python
from artemis.integrations import ArtemisCrewTool
```

#### Constructor

```python
ArtemisCrewTool(
    model: str,
    default_rounds: int = 3,
    config: DebateConfig | None = None,
    safety_monitor: SafetyMonitor | None = None,
    name: str = "Debate Tool",
    description: str = "...",
)
```

#### Methods

##### run

```python
def run(
    self,
    topic: str,
    rounds: int | None = None,
    positions: dict[str, str] | None = None,
) -> dict
```

Runs a synchronous debate.

##### arun

```python
async def arun(
    self,
    topic: str,
    rounds: int | None = None,
    positions: dict[str, str] | None = None,
) -> dict
```

Runs an asynchronous debate.

#### Output Schema

```python
{
    "verdict": str,              # "pro", "con", or "tie"
    "confidence": float,         # 0.0 to 1.0
    "reasoning": str,            # Verdict explanation
    "transcript": list[dict],    # Debate transcript
    "safety_alerts": list[dict], # Safety alerts
    "metadata": dict,            # Additional metadata
}
```

---

## MCP Integration

### ArtemisMCPServer

```python
from artemis.mcp import ArtemisMCPServer
```

#### Constructor

```python
ArtemisMCPServer(
    default_model: str = "gpt-4o",
    api_keys: dict[str, str] | None = None,
    max_sessions: int = 100,
    session_timeout: int = 3600,
    config: DebateConfig | None = None,
    safety_monitor: SafetyMonitor | None = None,
    enable_logging: bool = True,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_model` | str | "gpt-4o" | Default LLM model |
| `api_keys` | dict | None | Provider API keys |
| `max_sessions` | int | 100 | Max concurrent sessions |
| `session_timeout` | int | 3600 | Session timeout (seconds) |
| `config` | DebateConfig | None | Debate configuration |
| `safety_monitor` | SafetyMonitor | None | Safety monitor |
| `enable_logging` | bool | True | Enable logging |

#### Methods

##### run_stdio

```python
async def run_stdio(self) -> None
```

Runs server in stdio mode for MCP clients.

##### start

```python
async def start(
    self,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None
```

Starts HTTP server.

##### handle_tool_call

```python
async def handle_tool_call(
    self,
    tool_name: str,
    arguments: dict,
) -> dict
```

Handles a tool call directly.

### Available Tools

| Tool | Description |
|------|-------------|
| `artemis_debate_start` | Start a new debate |
| `artemis_add_round` | Add a debate round |
| `artemis_get_verdict` | Get jury verdict |
| `artemis_get_transcript` | Get full transcript |
| `artemis_list_debates` | List active debates |
| `artemis_get_safety_report` | Get safety report |

### CLI

```bash
# Basic usage
artemis-mcp

# HTTP mode
artemis-mcp --http --port 8080

# With options
artemis-mcp --model gpt-4-turbo --max-sessions 50 --verbose
```

---

## Common Types

### ToolResult

Base result from integration tools.

```python
class ToolResult(BaseModel):
    verdict: Verdict
    transcript: list[Turn]
    safety_alerts: list[SafetyAlert]
    metadata: dict
```

### ToolConfig

Configuration for integration tools.

```python
class ToolConfig(BaseModel):
    model: str
    rounds: int = 3
    config: DebateConfig | None = None
    safety_enabled: bool = True
```

---

## Error Handling

All integrations raise standard exceptions:

```python
from artemis.exceptions import (
    ArtemisError,      # Base exception
    DebateError,       # Debate execution failed
    IntegrationError,  # Integration-specific error
)

try:
    result = tool.invoke({"topic": "Your topic"})
except DebateError as e:
    print(f"Debate failed: {e}")
except IntegrationError as e:
    print(f"Integration error: {e}")
```

---

## Next Steps

- [Core API](core.md)
- [Models API](models.md)
- [Safety API](safety.md)
