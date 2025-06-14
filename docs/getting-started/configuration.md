# Configuration

ARTEMIS provides flexible configuration options for debates, agents, evaluation, and safety monitoring.

## Debate Configuration

Configure debate behavior using `DebateConfig`:

```python
from artemis.core.types import DebateConfig

config = DebateConfig(
    # Timing
    max_round_time_seconds=300,
    max_total_time_seconds=1800,

    # Evaluation
    evaluation_criteria=[
        "logical_coherence",
        "evidence_quality",
        "argument_strength",
        "ethical_considerations",
    ],

    # Jury settings
    jury_size=3,
    require_unanimous=False,

    # Safety
    enable_safety_monitoring=True,
    halt_on_safety_violation=False,
)

debate = Debate(
    topic="Your topic",
    agents=agents,
    config=config,
)
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_round_time_seconds` | int | 300 | Maximum time per round |
| `max_total_time_seconds` | int | 1800 | Maximum total debate time |
| `evaluation_criteria` | list | [...] | Criteria for argument evaluation |
| `jury_size` | int | 3 | Number of jury members |
| `require_unanimous` | bool | False | Require unanimous verdict |
| `enable_safety_monitoring` | bool | True | Enable safety monitors |
| `halt_on_safety_violation` | bool | False | Stop on safety alerts |

## Agent Configuration

Configure individual agents:

```python
from artemis.core.agent import Agent

agent = Agent(
    name="expert_agent",
    model="gpt-4o",
    position="supports the proposition",

    # Model parameters
    temperature=0.7,
    max_tokens=2000,

    # Reasoning (for o1/R1 models)
    reasoning_enabled=True,
    thinking_budget=8000,
)
```

### Agent Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `name` | str | required | Unique agent identifier |
| `model` | str | required | LLM model to use |
| `position` | str | None | Agent's debate position |
| `temperature` | float | 0.7 | Sampling temperature |
| `max_tokens` | int | 2000 | Max response tokens |
| `reasoning_enabled` | bool | False | Enable extended thinking |
| `thinking_budget` | int | 8000 | Tokens for reasoning |

## Model Configuration

Configure LLM providers:

```python
from artemis.models import create_model

# OpenAI
model = create_model(
    "gpt-4o",
    api_key="your-key",  # Or use OPENAI_API_KEY env var
    temperature=0.7,
    max_tokens=4000,
)

# DeepSeek with reasoning
model = create_model(
    "deepseek-reasoner",
    api_key="your-key",
    timeout=120.0,
    max_retries=3,
)
```

### Reasoning Model Configuration

For models that support extended thinking:

```python
from artemis.models.reasoning import ReasoningConfig, create_reasoning_config

# Automatic configuration based on model
config = create_reasoning_config(
    model="o1",
    thinking_budget=16000,
    show_thinking=True,
)

# Manual configuration
config = ReasoningConfig(
    model="deepseek-reasoner",
    strategy="adaptive",  # always, adaptive, never
    thinking_budget=8000,
    show_thinking=True,
    temperature=1.0,  # Required for some models
)
```

## Safety Configuration

Configure safety monitors:

```python
from artemis.safety import (
    SandbagDetector,
    DeceptionMonitor,
    BehaviorTracker,
    EthicsGuard,
    CompositeMonitor,
)

# Individual monitor configuration
sandbag = SandbagDetector(
    sensitivity=0.7,      # 0.0 to 1.0
    baseline_rounds=2,    # Rounds to establish baseline
)

deception = DeceptionMonitor(
    sensitivity=0.6,
    check_factual=True,
    check_logical=True,
)

behavior = BehaviorTracker(
    window_size=5,        # Turns to track
    drift_threshold=0.3,  # Drift detection threshold
)

ethics = EthicsGuard(
    sensitivity=0.5,
    principles=["fairness", "transparency", "non-harm"],
)

# Combine monitors
composite = CompositeMonitor(
    monitors=[sandbag, deception, behavior, ethics],
    aggregation="max",  # max, mean, or weighted
)
```

## MCP Server Configuration

Configure the MCP server:

```python
from artemis.mcp import ArtemisMCPServer

server = ArtemisMCPServer(
    default_model="gpt-4o",
    max_sessions=100,
    config=DebateConfig(...),
)

# Start HTTP server
await server.start(host="127.0.0.1", port=8080)

# Or stdio mode for MCP clients
await server.run_stdio()
```

### CLI Configuration

```bash
# HTTP mode with options
artemis-mcp --http --port 8080 --model gpt-4-turbo --max-sessions 50

# Verbose logging
artemis-mcp --verbose

# Custom host binding
artemis-mcp --http --host 0.0.0.0 --port 9000
```

## Environment Variables

ARTEMIS respects these environment variables:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GOOGLE_API_KEY` | Google AI API key |
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `ARTEMIS_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `ARTEMIS_DEFAULT_MODEL` | Default model for debates |

## Configuration Files

You can also use configuration files:

```yaml
# artemis.yaml
default_model: gpt-4o
max_sessions: 100

debate:
  max_round_time_seconds: 300
  jury_size: 3
  evaluation_criteria:
    - logical_coherence
    - evidence_quality

safety:
  enable_monitoring: true
  sandbagging_sensitivity: 0.7
  deception_sensitivity: 0.6
```

Load configuration:

```python
from artemis.utils.config import load_config

config = load_config("artemis.yaml")
```

## Next Steps

- Learn about [Core Concepts](../concepts/overview.md)
- Configure [Safety Monitoring](../safety/overview.md)
- See the [API Reference](../api/core.md)
