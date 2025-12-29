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
    rounds: int = 5,
    config: DebateConfig | None = None,
    jury: JuryPanel | None = None,
    evaluator: AdaptiveEvaluator | None = None,
    safety_monitors: list[Callable] | None = None,
)
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `topic` | str | Yes | The debate topic |
| `agents` | list[Agent] | Yes | List of participating agents (2+) |
| `rounds` | int | No | Number of debate rounds (default: 5) |
| `config` | DebateConfig | No | Debate configuration |
| `jury` | JuryPanel | No | Custom jury panel |
| `evaluator` | AdaptiveEvaluator | No | Custom argument evaluator |
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
    id: str  # Auto-generated UUID
    agent: str  # Required: agent name
    level: ArgumentLevel
    content: str
    evidence: list[Evidence] = []
    causal_links: list[CausalLink] = []
    rebuts: str | None = None  # ID of argument this rebuts
    supports: str | None = None  # ID of argument this supports
    ethical_score: float | None = None
    thinking_trace: str | None = None  # For reasoning models
    timestamp: datetime
```

### ArgumentLevel

```python
class ArgumentLevel(str, Enum):
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    OPERATIONAL = "operational"
```

### EvaluationMode

```python
class EvaluationMode(str, Enum):
    QUALITY = "quality"    # LLM-native evaluation for maximum accuracy
    BALANCED = "balanced"  # Selective LLM use (default)
    FAST = "fast"          # Heuristic-only for minimum cost
```

| Mode | Description |
|------|-------------|
| `QUALITY` | Uses LLM to evaluate all criteria. Highest accuracy, highest cost. |
| `BALANCED` | Selective LLM use for jury and key decisions. Good accuracy, moderate cost. |
| `FAST` | Heuristic-only evaluation. Lowest cost, backwards compatible. |

### Evidence

```python
class Evidence(BaseModel):
    id: str  # Auto-generated UUID
    type: Literal["fact", "statistic", "quote", "example", "study", "expert_opinion"]
    content: str  # The evidence content
    source: str | None = None
    url: str | None = None
    confidence: float  # 0.0 to 1.0
    verified: bool = False
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
    criteria: list[str] | None = None,
    model: str = "gpt-4o",
    consensus_threshold: float = 0.7,
    api_key: str | None = None,
    **model_kwargs,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `evaluators` | int | 3 | Number of jury evaluators |
| `criteria` | list[str] | None | Custom evaluation criteria |
| `model` | str | "gpt-4o" | LLM model for evaluators |
| `consensus_threshold` | float | 0.7 | Threshold for consensus |
| `api_key` | str | None | API key (uses env var if not provided) |

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

## LLM Evaluation

LLM-based evaluation for maximum accuracy.

```python
from artemis.core.llm_evaluation import LLMCriterionEvaluator, EvaluatorFactory
```

### LLMCriterionEvaluator

```python
class LLMCriterionEvaluator:
    def __init__(
        self,
        model: str | BaseModel = "gpt-4o-mini",
        api_key: str | None = None,
        cache_enabled: bool = True,
        **model_kwargs,
    ) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str \| BaseModel | "gpt-4o-mini" | Model name or instance |
| `api_key` | str | None | API key (uses env var if not provided) |
| `cache_enabled` | bool | True | Cache evaluations by content hash |

#### evaluate_argument

```python
async def evaluate_argument(
    self,
    argument: Argument,
    context: DebateContext,
    weights: dict[str, float] | None = None,
) -> ArgumentEvaluation
```

Evaluates an argument using LLM judgment.

**Returns:** `ArgumentEvaluation` with scores, weights, and reasoning.

### EvaluatorFactory

Factory for creating evaluators based on evaluation mode.

```python
class EvaluatorFactory:
    @staticmethod
    def create(
        mode: EvaluationMode,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        **kwargs,
    ) -> LLMCriterionEvaluator | AdaptiveEvaluator
```

| Mode | Returns |
|------|---------|
| `QUALITY` | `LLMCriterionEvaluator` |
| `BALANCED` | `AdaptiveEvaluator` |
| `FAST` | `AdaptiveEvaluator` |

---

## LLM Extraction

LLM-based extraction for evidence and causal links.

```python
from artemis.core.llm_extraction import (
    LLMCausalExtractor,
    LLMEvidenceExtractor,
    HybridCausalExtractor,
    HybridEvidenceExtractor,
    clear_extraction_cache,
)
```

### LLMCausalExtractor

Extracts causal relationships using LLM analysis.

```python
class LLMCausalExtractor:
    def __init__(
        self,
        model: BaseModel | None = None,
        model_name: str = "gpt-4o-mini",
        use_cache: bool = True,
    )
```

#### extract

```python
async def extract(self, content: str) -> list[CausalLink]
```

Extracts causal links from content.

### LLMEvidenceExtractor

Extracts evidence using LLM analysis.

```python
class LLMEvidenceExtractor:
    def __init__(
        self,
        model: BaseModel | None = None,
        model_name: str = "gpt-4o-mini",
        use_cache: bool = True,
    )
```

#### extract

```python
async def extract(self, content: str) -> list[Evidence]
```

Extracts evidence from content.

### Hybrid Extractors

Try regex first, fall back to LLM:

```python
class HybridCausalExtractor:
    async def extract(self, content: str) -> list[CausalLink]

class HybridEvidenceExtractor:
    async def extract(self, content: str) -> list[Evidence]
```

### clear_extraction_cache

```python
def clear_extraction_cache() -> None
```

Clears the global extraction cache.

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
