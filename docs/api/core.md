# Core API Reference

This page documents the core ARTEMIS classes and functions.

## Debate

The main orchestrator class for running debates.

```python
from artemis.core.debate import Debate
```

### Constructor

```python
Debate(
    topic: str,
    agents: list[Agent],
    rounds: int = 3,
    config: DebateConfig | None = None,
    safety_monitors: list[Callable] | None = None,
)
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `topic` | str | Yes | The debate topic |
| `agents` | list[Agent] | Yes | List of participating agents (2+) |
| `rounds` | int | No | Number of debate rounds (default: 3) |
| `config` | DebateConfig | No | Debate configuration |
| `safety_monitors` | list | No | List of safety monitor process methods |

### Methods

#### run

```python
async def run(self) -> DebateResult
```

Runs the complete debate and returns results.

**Returns:** `DebateResult` with verdict, transcript, and safety alerts.

**Example:**

```python
debate = Debate(topic="Your topic", agents=agents)
result = await debate.run()
print(result.verdict.decision)
```

#### assign_positions

```python
def assign_positions(self, positions: dict[str, str]) -> None
```

Assigns positions to agents.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `positions` | dict[str, str] | Mapping of agent name to position |

**Example:**

```python
debate.assign_positions({
    "pro_agent": "supports the proposition",
    "con_agent": "opposes the proposition",
})
```

#### add_round

```python
async def add_round(self) -> RoundResult
```

Executes a single debate round.

**Returns:** `RoundResult` with turns from each agent.

#### get_transcript

```python
def get_transcript(self) -> list[Turn]
```

Returns the current debate transcript.

---

## Agent

Represents a debate participant.

```python
from artemis.core.agent import Agent
```

### Constructor

```python
Agent(
    name: str,
    role: str,
    model: str = "gpt-4o",
    reasoning_config: ReasoningConfig | None = None,
)
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | str | Yes | Unique agent identifier |
| `role` | str | Yes | Agent's role description |
| `model` | str | No | LLM model to use (default: "gpt-4o") |
| `reasoning_config` | ReasoningConfig | No | Configuration for reasoning models |

### Methods

#### generate_argument

```python
async def generate_argument(
    self,
    context: DebateContext,
    round_type: str = "argument",
) -> Argument
```

Generates an argument for the current context.

**Returns:** `Argument` with content, level, and evidence.

#### generate_rebuttal

```python
async def generate_rebuttal(
    self,
    opponent_argument: Argument,
    context: DebateContext,
) -> Argument
```

Generates a rebuttal to an opponent's argument.

---

## Argument

Structured argument data.

```python
from artemis.core.types import Argument, ArgumentLevel
```

### Class Definition

```python
class Argument(BaseModel):
    content: str
    level: ArgumentLevel
    evidence: list[Evidence] = []
    causal_links: list[CausalLink] = []
    metadata: dict = {}
```

### ArgumentLevel

```python
class ArgumentLevel(str, Enum):
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    OPERATIONAL = "operational"
```

### Evidence

```python
class Evidence(BaseModel):
    type: str  # "empirical", "citation", "historical", etc.
    source: str
    quote: str | None = None
    credibility: float = 1.0
```

### CausalLink

```python
class CausalLink(BaseModel):
    id: str
    cause: str
    effect: str
    mechanism: str | None = None
    strength: float = 0.5  # 0.0 to 1.0
    bidirectional: bool = False
```

---

## JuryPanel

Multi-perspective evaluation jury.

```python
from artemis.core.jury import JuryPanel
from artemis.core.types import JuryPerspective
```

### Constructor

```python
JuryPanel(
    evaluators: int = 3,
    model: str = "gpt-4o",
    perspectives: list[JuryPerspective] | None = None,
    consensus_threshold: float = 0.7,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `evaluators` | int | 3 | Number of jury evaluators |
| `model` | str | "gpt-4o" | LLM model for evaluators |
| `perspectives` | list | None | Evaluation perspectives |
| `consensus_threshold` | float | 0.7 | Threshold for consensus |

### JuryPerspective

```python
class JuryPerspective(str, Enum):
    ANALYTICAL = "analytical"    # Focus on logic and evidence
    ETHICAL = "ethical"          # Focus on moral implications
    PRACTICAL = "practical"      # Focus on feasibility
    ADVERSARIAL = "adversarial"  # Challenge all arguments
    SYNTHESIZING = "synthesizing" # Find common ground
```

### Methods

#### deliberate

```python
async def deliberate(self, transcript: list[Turn]) -> Verdict
```

Conducts jury deliberation and returns verdict.

---

## Verdict

Final debate verdict.

```python
from artemis.core.types import Verdict
```

### Class Definition

```python
class Verdict(BaseModel):
    decision: str  # "pro", "con", or "tie"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    votes: list[Vote] = []
    deliberation_history: list[DeliberationRound] = []
```

---

## DebateConfig

Debate configuration options.

```python
from artemis.core.types import DebateConfig
```

### Class Definition

```python
class DebateConfig(BaseModel):
    # Timing
    turn_timeout: int = 60           # Timeout per turn (seconds)
    round_timeout: int = 300         # Timeout per round (seconds)

    # Argument generation
    max_argument_tokens: int = 1000
    require_evidence: bool = True
    require_causal_links: bool = True
    min_evidence_per_argument: int = 0

    # Evaluation
    evaluation_criteria: EvaluationCriteria = EvaluationCriteria()
    adaptation_enabled: bool = True
    adaptation_rate: float = 0.1

    # Safety
    safety_mode: str = "passive"     # "off", "passive", "active"
    halt_on_safety_violation: bool = False

    # Logging
    log_level: str = "INFO"
    trace_enabled: bool = False
```

---

## DebateResult

Complete debate result.

```python
from artemis.core.types import DebateResult
```

### Class Definition

```python
class DebateResult(BaseModel):
    debate_id: str
    topic: str
    verdict: Verdict
    transcript: list[Turn]
    safety_alerts: list[SafetyAlert] = []
    metadata: DebateMetadata
    final_state: DebateState = DebateState.COMPLETE
```

### DebateMetadata

```python
class DebateMetadata(BaseModel):
    started_at: datetime
    ended_at: datetime | None
    total_rounds: int
    total_turns: int
    agents: list[str]
    jury_size: int
    safety_monitors: list[str]
    model_usage: dict[str, dict[str, int]]
```

---

## Turn

A single turn in the debate.

```python
from artemis.core.types import Turn
```

### Class Definition

```python
class Turn(BaseModel):
    round: int
    agent: str
    argument: Argument
    timestamp: datetime
    evaluation: Evaluation | None = None
```

---

## AdaptiveEvaluator

L-AE-CR adaptive evaluation.

```python
from artemis.core.evaluation import AdaptiveEvaluator
```

### Constructor

```python
AdaptiveEvaluator(
    domain: str | None = None,
    enable_causal_analysis: bool = True,
    criteria_weights: dict[str, float] | None = None,
)
```

### Methods

#### evaluate

```python
async def evaluate(
    self,
    argument: Argument,
    context: DebateContext,
    include_feedback: bool = False,
) -> Evaluation
```

Evaluates an argument with adaptive criteria.

#### compare

```python
async def compare(
    self,
    argument_a: Argument,
    argument_b: Argument,
    context: DebateContext,
) -> Comparison
```

Compares two arguments and determines winner.

---

## Exceptions

```python
from artemis.exceptions import (
    ArtemisError,
    DebateError,
    AgentError,
    EvaluationError,
    SafetyError,
    EthicsViolationError,
)
```

### ArtemisError

Base exception for all ARTEMIS errors.

### DebateError

Raised when debate execution fails.

### AgentError

Raised when agent generation fails.

### SafetyError

Raised when safety violation detected.

### EthicsViolationError

Raised when ethics guard blocks content.

---

## Next Steps

- [Models API](models.md)
- [Safety API](safety.md)
- [Integrations API](integrations.md)
