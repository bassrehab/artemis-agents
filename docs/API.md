# ARTEMIS Agents - API Reference

Complete API documentation for ARTEMIS Agents.

---

## Table of Contents

- [Core Classes](#core-classes)
  - [Debate](#debate)
  - [Agent](#agent)
  - [JuryPanel](#jurypanel)
  - [AdaptiveEvaluator](#adaptiveevaluator)
- [Data Models](#data-models)
  - [Argument](#argument)
  - [Turn](#turn)
  - [Verdict](#verdict)
  - [DebateResult](#debateresult)
- [Safety Monitors](#safety-monitors)
  - [SafetyMonitor](#safetymonitor-base)
  - [SandbagDetector](#sandbagdetector)
  - [DeceptionMonitor](#deceptionmonitor)
- [Model Providers](#model-providers)
  - [BaseModel](#basemodel)
  - [OpenAIModel](#openaimodel)
  - [AnthropicModel](#anthropicmodel)
- [Integrations](#integrations)
  - [LangChain](#langchain-integration)
  - [LangGraph](#langgraph-integration)
  - [MCP Server](#mcp-server)
- [Configuration](#configuration)
- [Exceptions](#exceptions)

---

## Core Classes

### Debate

The main orchestrator for multi-agent debates.

```python
class Debate:
    def __init__(
        self,
        topic: str,
        agents: list[Agent],
        jury: JuryPanel,
        rounds: int = 3,
        monitors: list[SafetyMonitor] | None = None,
        config: DebateConfig | None = None
    ) -> None:
        """
        Initialize a new debate.
        
        Args:
            topic: The debate topic/question.
            agents: List of participating agents.
            jury: Jury panel for evaluation.
            rounds: Number of debate rounds (default: 3).
            monitors: Optional safety monitors.
            config: Optional debate configuration.
        """
    
    async def run(self) -> DebateResult:
        """
        Execute the full debate.
        
        Returns:
            DebateResult containing verdict, transcript, and safety alerts.
        
        Raises:
            DebateConfigError: If configuration is invalid.
            SafetyHaltError: If safety monitor triggers halt (active mode).
        """
    
    @property
    def state(self) -> DebateState:
        """Current debate state."""
    
    @property
    def transcript(self) -> list[Turn]:
        """Current debate transcript."""
```

**Example:**

```python
from artemis import Debate, Agent, JuryPanel

debate = Debate(
    topic="Should renewable energy be mandatory?",
    agents=[
        Agent(name="Advocate", role="Supports renewable mandate", model="gpt-4o"),
        Agent(name="Skeptic", role="Questions feasibility", model="gpt-4o"),
    ],
    jury=JuryPanel(evaluators=3),
    rounds=2
)

result = await debate.run()
print(f"Verdict: {result.verdict.decision}")
```

---

### Agent

Debate participant with H-L-DAG argument generation.

```python
class Agent:
    def __init__(
        self,
        name: str,
        role: str,
        model: str | BaseModel,
        reasoning: ReasoningConfig | None = None,
        persona: str | None = None
    ) -> None:
        """
        Initialize a debate agent.
        
        Args:
            name: Agent's display name.
            role: Description of agent's role/perspective.
            model: Model identifier string or BaseModel instance.
            reasoning: Optional reasoning model configuration.
            persona: Optional detailed persona description.
        """
    
    async def generate_argument(
        self,
        context: DebateContext,
        level: ArgumentLevel = ArgumentLevel.TACTICAL
    ) -> Argument:
        """
        Generate an argument at the specified hierarchical level.
        
        Args:
            context: Current debate context.
            level: H-L-DAG level (strategic, tactical, operational).
        
        Returns:
            Structured Argument with content, evidence, and causal links.
        """
    
    async def generate_opening(self, context: DebateContext) -> Argument:
        """Generate opening statement (strategic level)."""
    
    async def generate_closing(self, context: DebateContext) -> Argument:
        """Generate closing statement."""
```

**Example with Reasoning Model:**

```python
from artemis import Agent
from artemis.models import ReasoningConfig

agent = Agent(
    name="Deep Analyst",
    role="Provides thoroughly reasoned arguments",
    model="deepseek-r1",
    reasoning=ReasoningConfig(
        enabled=True,
        thinking_budget=16000,
        strategy="think-then-argue"
    )
)
```

---

### JuryPanel

Multi-perspective evaluation panel.

```python
class JuryPanel:
    def __init__(
        self,
        evaluators: int = 3,
        criteria: list[str] | None = None,
        model: str = "gpt-4o",
        consensus_threshold: float = 0.7
    ) -> None:
        """
        Initialize jury panel.
        
        Args:
            evaluators: Number of jury members.
            criteria: Evaluation criteria (uses defaults if None).
            model: Model for jury members.
            consensus_threshold: Required agreement level (0-1).
        """
    
    async def deliberate(self, transcript: list[Turn]) -> Verdict:
        """
        Conduct jury deliberation.
        
        Args:
            transcript: Complete debate transcript.
        
        Returns:
            Verdict with decision, confidence, and reasoning.
        """
```

**Default Criteria:**
- `logical_coherence`
- `evidence_quality`
- `causal_reasoning`
- `ethical_alignment`
- `persuasiveness`

---

### AdaptiveEvaluator

L-AE-CR implementation for dynamic argument evaluation.

```python
class AdaptiveEvaluator:
    def __init__(
        self,
        criteria: dict[str, float] | None = None,
        adaptation_rate: float = 0.1
    ) -> None:
        """
        Initialize adaptive evaluator.
        
        Args:
            criteria: Criterion weights (uses defaults if None).
            adaptation_rate: How quickly weights adapt (0-1).
        """
    
    async def evaluate_argument(
        self,
        argument: Argument,
        context: DebateContext
    ) -> ArgumentEvaluation:
        """
        Evaluate argument with context-adapted criteria.
        
        Returns:
            ArgumentEvaluation with scores and adapted weights.
        """
```

---

## Data Models

### Argument

```python
class Argument(BaseModel):
    id: str                           # Unique identifier
    agent: str                        # Agent name
    level: ArgumentLevel              # strategic/tactical/operational
    content: str                      # Argument text
    evidence: list[Evidence]          # Supporting evidence
    causal_links: list[CausalLink]    # Causal reasoning
    ethical_score: float | None       # Ethics evaluation
    thinking_trace: str | None        # Reasoning trace (if available)
    timestamp: datetime               # Creation time
```

### ArgumentLevel

```python
class ArgumentLevel(str, Enum):
    STRATEGIC = "strategic"    # High-level thesis and position
    TACTICAL = "tactical"      # Supporting arguments and evidence
    OPERATIONAL = "operational" # Specific facts and examples
```

### Turn

```python
class Turn(BaseModel):
    round: int                        # Debate round number
    sequence: int                     # Turn sequence within round
    agent: str                        # Agent name
    argument: Argument                # The argument made
    evaluation: ArgumentEvaluation | None
    safety_results: list[SafetyResult]
```

### Verdict

```python
class Verdict(BaseModel):
    decision: str                     # Final verdict
    confidence: float                 # Confidence score (0-1)
    reasoning: str                    # Explanation
    dissenting_opinions: list[DissentingOpinion]
    score_breakdown: dict[str, float] # Per-criterion scores
```

### DebateResult

```python
class DebateResult(BaseModel):
    debate_id: str
    topic: str
    verdict: Verdict
    transcript: list[Turn]
    safety_alerts: list[SafetyAlert]
    metadata: DebateMetadata
```

---

## Safety Monitors

### SafetyMonitor (Base)

```python
class SafetyMonitor(ABC):
    def __init__(
        self,
        mode: Literal["passive", "active"] = "passive",
        alert_threshold: float = 0.7,
        halt_threshold: float = 0.95
    ) -> None:
        """
        Initialize safety monitor.
        
        Args:
            mode: "passive" (report only) or "active" (can halt).
            alert_threshold: Severity level to trigger alert.
            halt_threshold: Severity level to halt debate (active mode).
        """
    
    @abstractmethod
    async def analyze(
        self,
        turn: Turn,
        context: DebateContext
    ) -> SafetyResult:
        """Analyze turn for safety concerns."""
```

### SandbagDetector

Detects deliberate underperformance.

```python
class SandbagDetector(SafetyMonitor):
    def __init__(
        self,
        sensitivity: float = 0.8,
        baseline_window: int = 3,
        **kwargs
    ) -> None:
        """
        Args:
            sensitivity: Detection sensitivity (0-1).
            baseline_window: Turns to establish baseline.
        """
```

**Detection Methods:**
- Capability drop detection
- Strategic timing analysis
- Selective engagement detection

### DeceptionMonitor

Detects misleading arguments.

```python
class DeceptionMonitor(SafetyMonitor):
    """
    Detection methods:
    - Factual consistency checking
    - Logical fallacy detection
    - Emotional manipulation detection
    - Citation fabrication checking
    """
```

---

## Model Providers

### BaseModel

```python
class BaseModel(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        **kwargs
    ) -> Response:
        """Generate completion."""
    
    @abstractmethod
    async def generate_with_reasoning(
        self,
        messages: list[Message],
        thinking_budget: int
    ) -> ReasoningResponse:
        """Generate with extended thinking."""
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier."""
    
    @property
    @abstractmethod
    def supports_reasoning(self) -> bool:
        """Whether model supports extended thinking."""
```

### OpenAIModel

```python
from artemis.models import OpenAIModel

model = OpenAIModel(
    model="gpt-4o",           # or "o1-preview", "o1-mini"
    api_key="...",            # Optional, uses env var
    temperature=0.7,
    max_tokens=1000
)
```

### AnthropicModel

```python
from artemis.models import AnthropicModel

model = AnthropicModel(
    model="claude-3-5-sonnet-20241022",
    api_key="...",
    max_tokens=1000
)
```

---

## Integrations

### LangChain Integration

```python
from artemis.integrations import ArtemisDebateTool
from langchain.agents import initialize_agent

tool = ArtemisDebateTool(
    default_rounds=2,
    default_jury_size=3
)

agent = initialize_agent(
    tools=[tool],
    llm=llm,
    agent_type="structured-chat-zero-shot-react-description"
)
```

### LangGraph Integration

```python
from langgraph.graph import StateGraph
from artemis.integrations import ArtemisDebateNode

workflow = StateGraph(State)

workflow.add_node(
    "debate",
    ArtemisDebateNode(
        agents=3,
        rounds=2,
        monitors=[SandbagDetector()]
    )
)

workflow.add_edge("gather_context", "debate")
workflow.add_edge("debate", "make_decision")
```

### MCP Server

```bash
# Start server
artemis serve --port 8080

# Or programmatically
python -m artemis.mcp.server --port 8080
```

**Available Tools:**
- `artemis_debate`: Start structured debate
- `artemis_add_perspective`: Add agent to debate
- `artemis_get_verdict`: Get final verdict
- `artemis_get_transcript`: Get debate transcript

---

## Configuration

### DebateConfig

```python
class DebateConfig(BaseModel):
    turn_timeout: int = 60           # Seconds per turn
    round_timeout: int = 300         # Seconds per round
    max_argument_tokens: int = 1000
    require_evidence: bool = True
    require_causal_links: bool = True
    evaluation_criteria: dict[str, float] | None = None
    adaptation_enabled: bool = True
    safety_mode: Literal["off", "passive", "active"] = "passive"
    halt_on_safety_violation: bool = False
    log_level: str = "INFO"
    trace_enabled: bool = False
```

### ReasoningConfig

```python
class ReasoningConfig(BaseModel):
    enabled: bool = True
    thinking_budget: int = 8000      # Tokens for reasoning
    strategy: Literal[
        "think-then-argue",          # Full reasoning first
        "interleaved",               # Reason between sections
        "final-reflection"           # Argue then reflect
    ] = "think-then-argue"
```

---

## Exceptions

```python
from artemis.exceptions import (
    ArtemisError,          # Base exception
    DebateConfigError,     # Invalid configuration
    AgentError,            # Agent-related error
    ModelError,            # LLM provider error
    SafetyHaltError,       # Debate halted by safety monitor
    EvaluationError,       # Evaluation error
)
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `GOOGLE_API_KEY` | Google AI API key | - |
| `ARTEMIS_DEFAULT_MODEL` | Default model | `gpt-4o` |
| `ARTEMIS_SAFETY_MODE` | Safety mode | `passive` |
| `ARTEMIS_LOG_LEVEL` | Log level | `INFO` |

See `.env.example` for complete list.
