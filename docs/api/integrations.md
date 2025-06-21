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
    "topic": str,                  # Required: debate topic
    "rounds": int,                 # Optional: number of rounds
    "agents": list[AgentConfig],   # Optional: agent configurations
    "pro_position": str,           # Optional: pro-side position (simple mode)
    "con_position": str,           # Optional: con-side position (simple mode)
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
    model: str = "gpt-4o",
    agents: list[Agent] | None = None,
    config: DebateNodeConfig | None = None,
    debate_config: DebateConfig | None = None,
)
```

#### Usage

```python
from langgraph.graph import StateGraph
from artemis.integrations import ArtemisDebateNode, DebateNodeState

workflow = StateGraph(DebateNodeState)
workflow.add_node("debate", ArtemisDebateNode(model="gpt-4o").run_debate)
```

### DebateNodeState

```python
from artemis.integrations import DebateNodeState
```

#### Class Definition

```python
class DebateNodeState(TypedDict, total=False):
    topic: str
    agents: list[AgentStateConfig]
    positions: dict[str, str]
    rounds: int
    current_round: int
    phase: str
    transcript: list[dict]
    verdict: dict | None
    scores: dict[str, float]
    metadata: dict
```

### create_debate_workflow

```python
from artemis.integrations import create_debate_workflow
```

#### Signature

```python
def create_debate_workflow(
    model: str = "gpt-4o",
    step_by_step: bool = False,
    agents: list[Agent] | None = None,
) -> CompiledStateGraph
```

Creates a complete debate workflow.

**Returns:** Compiled LangGraph workflow.

---

## CrewAI Integration

### ArtemisCrewTool

```python
from artemis.integrations import ArtemisCrewTool
```

#### Constructor

```python
ArtemisCrewTool(
    model: str = "gpt-4o",
    default_rounds: int = 3,
    agents: list[Agent] | None = None,
    config: DebateConfig | None = None,
    verbose: bool = False,
)
```

#### Methods

##### run

```python
def run(
    self,
    topic: str,
    agents: list[dict] | None = None,
    pro_position: str | None = None,
    con_position: str | None = None,
    rounds: int | None = None,
) -> str
```

Runs a synchronous debate and returns formatted string.

##### arun

```python
async def arun(
    self,
    topic: str,
    agents: list[dict] | None = None,
    pro_position: str | None = None,
    con_position: str | None = None,
    rounds: int | None = None,
) -> str
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
    max_sessions: int = 100,
    config: DebateConfig | None = None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_model` | str | "gpt-4o" | Default LLM model |
| `max_sessions` | int | 100 | Max concurrent sessions |
| `config` | DebateConfig | None | Debate configuration |

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
