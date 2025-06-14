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
    jury: Jury | None = None,
    config: DebateConfig | None = None,
    safety_manager: SafetyManager | None = None,
)
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `topic` | str | Yes | The debate topic |
| `agents` | list[Agent] | Yes | List of participating agents |
| `rounds` | int | No | Number of debate rounds (default: 3) |
| `jury` | Jury | No | Jury for evaluation |
| `config` | DebateConfig | No | Debate configuration |
| `safety_manager` | SafetyManager | No | Safety monitoring manager |

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
    model: str,
    position: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    reasoning_enabled: bool = False,
    thinking_budget: int = 8000,
)
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | str | Yes | Unique agent identifier |
| `model` | str | Yes | LLM model to use |
| `position` | str | No | Agent's debate position |
| `temperature` | float | No | Sampling temperature (default: 0.7) |
| `max_tokens` | int | No | Max response tokens (default: 2000) |
| `reasoning_enabled` | bool | No | Enable extended thinking (default: False) |
| `thinking_budget` | int | No | Tokens for reasoning (default: 8000) |

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
    source: str
    target: str
    relation: CausalRelationType
    strength: float
    evidence: list[str] = []
```

---

## Jury

Multi-perspective evaluation jury.

```python
from artemis.core.jury import Jury, JuryMember
```

### Constructor

```python
Jury(
    members: list[JuryMember],
    voting: str = "simple_majority",
    threshold: float = 0.5,
)
```

### JuryMember

```python
JuryMember(
    name: str,
    perspective: str | Perspective,
    weight: float = 1.0,
)
```

### Methods

#### deliberate

```python
async def deliberate(self, debate_result: DebateResult) -> Verdict
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
    max_round_time_seconds: int = 300
    max_total_time_seconds: int = 1800

    # Evaluation
    evaluation_criteria: list[str] = [
        "logical_coherence",
        "evidence_quality",
        "argument_strength",
        "ethical_considerations",
    ]

    # Jury
    jury_size: int = 3
    require_unanimous: bool = False

    # Safety
    enable_safety_monitoring: bool = True
    halt_on_safety_violation: bool = False

    # Arguments
    argument_depth: str = "medium"  # shallow, medium, deep
    require_evidence: bool = True
    min_tactical_points: int = 2
    min_operational_facts: int = 3
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
    topic: str
    verdict: Verdict
    transcript: list[Turn]
    safety_alerts: list[SafetyAlert] = []
    metadata: dict = {}
    duration_seconds: float
```

### Methods

#### get_alerts_by_type

```python
def get_alerts_by_type(self, alert_type: str) -> list[SafetyAlert]
```

#### get_alerts_by_agent

```python
def get_alerts_by_agent(self, agent_name: str) -> list[SafetyAlert]
```

#### safety_summary

```python
@property
def safety_summary(self) -> SafetySummary
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
