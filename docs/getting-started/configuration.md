# Configuration

ARTEMIS provides flexible configuration options for debates, agents, evaluation, and safety monitoring.

## Debate Configuration

Configure debate behavior using `DebateConfig`:

```python
from artemis.core.types import DebateConfig, EvaluationCriteria

# Custom evaluation criteria weights
criteria = EvaluationCriteria(
    logical_coherence=0.25,
    evidence_quality=0.25,
    causal_reasoning=0.20,
    ethical_alignment=0.15,
    persuasiveness=0.15,
)

config = DebateConfig(
    # Timing
    turn_timeout=60,        # Timeout per turn in seconds
    round_timeout=300,      # Timeout per round in seconds

    # Argument generation
    max_argument_tokens=1000,
    require_evidence=True,
    require_causal_links=True,
    min_evidence_per_argument=1,

    # Evaluation
    evaluation_criteria=criteria,
    adaptation_enabled=True,
    adaptation_rate=0.1,

    # Safety
    safety_mode="passive",  # "off", "passive", or "active"
    halt_on_safety_violation=False,

    # Logging
    log_level="INFO",
    trace_enabled=False,
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
| `turn_timeout` | int | 60 | Timeout per turn in seconds |
| `round_timeout` | int | 300 | Timeout per round in seconds |
| `max_argument_tokens` | int | 1000 | Max tokens per argument |
| `require_evidence` | bool | True | Require evidence in arguments |
| `require_causal_links` | bool | True | Require causal reasoning |
| `adaptation_enabled` | bool | True | Enable dynamic criteria adaptation |
| `safety_mode` | str | "passive" | Safety monitoring mode |
| `halt_on_safety_violation` | bool | False | Stop debate on safety alerts |

## Agent Configuration

Configure individual agents:

```python
from artemis.core.agent import Agent
from artemis.core.types import ReasoningConfig

# Basic agent
agent = Agent(
    name="expert_agent",
    role="Domain expert analyzing the topic",
    model="gpt-4o",
)

# Agent with reasoning configuration
reasoning_config = ReasoningConfig(
    enabled=True,
    thinking_budget=8000,
    strategy="think-then-argue",
    include_trace_in_output=False,
)

agent_with_reasoning = Agent(
    name="reasoning_agent",
    role="Deep analyst with extended thinking",
    model="deepseek-reasoner",
    reasoning_config=reasoning_config,
)

# Agent with persona
persona_agent = Agent(
    name="philosopher",
    role="Philosopher analyzing ethical implications",
    model="gpt-4o",
    persona="An analytical philosopher who values rigorous logical reasoning",
)
```

### Agent Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `name` | str | required | Unique agent identifier |
| `role` | str | required | Agent's role description |
| `model` | str | "gpt-4o" | LLM model to use |
| `position` | str | None | Agent's debate position |
| `reasoning_config` | ReasoningConfig | None | Reasoning model config |

### Reasoning Configuration

```python
from artemis.core.types import ReasoningConfig

config = ReasoningConfig(
    enabled=True,
    thinking_budget=8000,       # Max tokens for thinking (1000-32000)
    strategy="think-then-argue",  # "think-then-argue", "interleaved", "final-reflection"
    include_trace_in_output=False,
)
```

## Model Configuration

Configure LLM providers:

```python
from artemis.models import create_model

# OpenAI
model = create_model(
    "gpt-4o",
    api_key="your-key",  # Or use OPENAI_API_KEY env var
)

# DeepSeek with reasoning
model = create_model(
    "deepseek-reasoner",
    api_key="your-key",
)

# Google/Gemini via AI Studio
model = create_model(
    "gemini-2.0-flash",
    # Uses GOOGLE_API_KEY or GEMINI_API_KEY env var
)

# Google/Gemini via Vertex AI (higher rate limits)
model = create_model(
    "gemini-2.0-flash",
    provider="google",
    # Uses GOOGLE_CLOUD_PROJECT env var for auto-detection
)

# Anthropic/Claude
model = create_model(
    "claude-sonnet-4-20250514",
    # Uses ANTHROPIC_API_KEY env var
)
```

### Google Backend Selection

GoogleModel supports two backends that are auto-detected:

| Backend | When Used | Authentication |
|---------|-----------|----------------|
| **AI Studio** | Default when no project set | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| **Vertex AI** | When `GOOGLE_CLOUD_PROJECT` is set | Application Default Credentials |

```python
from artemis.models import GoogleModel

# Force AI Studio (explicit)
model = GoogleModel(
    model="gemini-2.0-flash",
    api_key="your-api-key",
    use_vertex_ai=False,
)

# Force Vertex AI (explicit)
model = GoogleModel(
    model="gemini-2.0-flash",
    project="my-gcp-project",
    location="us-central1",
    use_vertex_ai=True,
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
    MonitorMode,
    EthicsConfig,
)

# Individual monitor configuration
sandbag = SandbagDetector(
    mode=MonitorMode.PASSIVE,
    sensitivity=0.7,         # 0.0 to 1.0
    baseline_turns=3,        # Turns to establish baseline
    drop_threshold=0.3,      # Performance drop threshold
)

deception = DeceptionMonitor(
    mode=MonitorMode.PASSIVE,
    sensitivity=0.6,
)

behavior = BehaviorTracker(
    mode=MonitorMode.PASSIVE,
    sensitivity=0.5,
    window_size=5,           # Turns to track
    drift_threshold=0.3,     # Drift detection threshold
)

ethics_config = EthicsConfig(
    harmful_content_threshold=0.3,
    bias_threshold=0.4,
    fairness_threshold=0.3,
    enabled_checks=["harmful_content", "bias", "fairness"],
)

ethics = EthicsGuard(
    mode=MonitorMode.PASSIVE,
    config=ethics_config,
)

# Use monitors in debate
debate = Debate(
    topic="Your topic",
    agents=agents,
    safety_monitors=[
        sandbag.process,
        deception.process,
        behavior.process,
        ethics.process,
    ],
)
```

## Jury Configuration

Configure the jury panel:

```python
from artemis.core.jury import JuryPanel

jury = JuryPanel(
    evaluators=5,               # Number of jury members
    model="gpt-4o",             # Model for jurors
    consensus_threshold=0.7,    # Required agreement (0-1)
)

# Use in debate
debate = Debate(
    topic="Your topic",
    agents=agents,
    jury=jury,
)
```

### Per-Juror Model Configuration

You can configure different models for each juror using two approaches:

**Option A: Simple Model List**

```python
jury = JuryPanel(
    evaluators=3,
    models=["gpt-4o", "claude-sonnet-4-20250514", "gemini-2.0-flash"],
)
```

**Option B: Full JurorConfig Objects**

```python
from artemis.core.types import JurorConfig, JuryPerspective

jury = JuryPanel(
    jurors=[
        JurorConfig(
            perspective=JuryPerspective.ANALYTICAL,
            model="gpt-4o",
        ),
        JurorConfig(
            perspective=JuryPerspective.ETHICAL,
            model="claude-sonnet-4-20250514",
        ),
        JurorConfig(
            perspective=JuryPerspective.PRACTICAL,
            model="gemini-2.0-flash",
        ),
    ],
    consensus_threshold=0.7,
)
```

### Jury Perspectives

| Perspective | Focus |
|-------------|-------|
| `ANALYTICAL` | Logic, evidence quality, reasoning coherence |
| `ETHICAL` | Moral implications, fairness, values alignment |
| `PRACTICAL` | Real-world applicability, feasibility |
| `CREATIVE` | Novel approaches, unconventional solutions |
| `SKEPTICAL` | Critical analysis, identifying weaknesses |

## MCP Server Configuration

Configure the MCP server:

```python
from artemis.mcp import ArtemisMCPServer

server = ArtemisMCPServer(
    default_model="gpt-4o",
    max_sessions=100,
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

### API Keys

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude models |
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `GOOGLE_API_KEY` | Google AI Studio API key (for Gemini) |
| `GEMINI_API_KEY` | Alternative to `GOOGLE_API_KEY` |

### Google Cloud / Vertex AI

| Variable | Description |
|----------|-------------|
| `GOOGLE_CLOUD_PROJECT` | GCP project ID (enables Vertex AI backend) |
| `GCP_PROJECT` | Alternative to `GOOGLE_CLOUD_PROJECT` |
| `GOOGLE_CLOUD_LOCATION` | GCP region (default: `us-central1`) |

When `GOOGLE_CLOUD_PROJECT` is set, GoogleModel automatically uses Vertex AI with Application Default Credentials instead of AI Studio.

### ARTEMIS Settings

| Variable | Description |
|----------|-------------|
| `ARTEMIS_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `ARTEMIS_DEFAULT_MODEL` | Default model for debates |

## Configuration Files

You can also use configuration files:

```yaml
# artemis.yaml
default_model: gpt-4o
max_sessions: 100

debate:
  turn_timeout: 60
  round_timeout: 300
  require_evidence: true
  safety_mode: passive

safety:
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
