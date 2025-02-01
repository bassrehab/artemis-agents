# ARTEMIS Agents - Design Document

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2025-02-01 | Subhadip Mitra | Initial design |
| 0.2 | 2025-03-15 | Subhadip Mitra | Added safety monitoring |
| 0.3 | 2025-05-01 | Subhadip Mitra | Added MCP server design |
| 1.0 | 2025-07-01 | Subhadip Mitra | v1.0 release design |

---

## 1. Executive Summary

ARTEMIS Agents is an open-source implementation of the Adaptive Reasoning and Evaluation Framework for Multi-agent Intelligent Systems, designed for structured debate-driven decision-making.

### Goals

1. **Faithful implementation** of the published ARTEMIS framework
2. **Production-ready** quality with proper error handling, logging, and observability
3. **Extensible** architecture supporting multiple LLM providers and framework integrations
4. **Safe by default** with built-in monitoring for adversarial agent behaviors

### Non-Goals

1. General-purpose chatbot framework (use LangChain/AutoGen)
2. Task automation (use CrewAI)
3. Replacement for existing frameworks (complement, not compete)

---

## 2. Background

### 2.1 The Problem with Current Multi-Agent Systems

Existing multi-agent frameworks like AutoGen, CrewAI, and CAMEL provide flexible agent orchestration but lack:

1. **Structured argumentation**: Agents chat freely without argument hierarchy
2. **Causal reasoning**: No explicit modeling of cause-effect relationships
3. **Adaptive evaluation**: Static evaluation criteria regardless of context
4. **Safety monitoring**: No detection of sandbagging, deception, or manipulation

### 2.2 ARTEMIS Framework (from Technical Disclosure)

The ARTEMIS paper introduces:

1. **H-L-DAG (Hierarchical Argument Generation)**
   - Strategic level: Overall position and thesis
   - Tactical level: Supporting arguments and evidence
   - Operational level: Specific facts and examples

2. **L-AE-CR (Adaptive Evaluation with Causal Reasoning)**
   - Dynamic criteria weighting based on context
   - Causal graph construction for argument relationships
   - Ethical alignment scoring

3. **Jury Scoring Mechanism**
   - Multiple evaluator perspectives
   - Weighted consensus building
   - Confidence calibration

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                           Client Layer                              │
├────────────────────────────────────────────────────────────────────┤
│  Python API  │  MCP Server  │  LangChain  │  LangGraph  │  CrewAI │
├────────────────────────────────────────────────────────────────────┤
│                         Orchestration Layer                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Debate Orchestrator                        │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐  │  │
│  │  │   Round    │  │   Turn     │  │     State Machine      │  │  │
│  │  │  Manager   │  │  Manager   │  │  (setup→debate→verdict)│  │  │
│  │  └────────────┘  └────────────┘  └────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────────────────────┤
│                          Core Layer                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │   H-L-DAG    │  │   L-AE-CR    │  │      Jury Panel          │ │
│  │  Argument    │  │  Adaptive    │  │  ┌────┐ ┌────┐ ┌────┐   │ │
│  │  Generator   │  │  Evaluator   │  │  │ J1 │ │ J2 │ │ J3 │   │ │
│  └──────────────┘  └──────────────┘  │  └────┘ └────┘ └────┘   │ │
│                                       └──────────────────────────┘ │
├────────────────────────────────────────────────────────────────────┤
│                         Safety Layer                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │
│  │ Sandbagging │ │  Deception  │ │  Behavior   │ │   Ethics    │  │
│  │  Detector   │ │   Monitor   │ │   Tracker   │ │   Guard     │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘  │
├────────────────────────────────────────────────────────────────────┤
│                        Provider Layer                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │  OpenAI  │  │Anthropic │  │  Google  │  │ Local (Ollama)   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Interactions

```
┌─────────┐     ┌───────────────┐     ┌─────────────┐
│  User   │────▶│    Debate     │────▶│   Agent 1   │
└─────────┘     │  Orchestrator │     └──────┬──────┘
                │               │            │
                │   ┌───────────┼────────────┼───────────┐
                │   │           ▼            ▼           │
                │   │     ┌─────────┐  ┌─────────┐      │
                │   │     │ Agent 2 │  │ Agent 3 │      │
                │   │     └────┬────┘  └────┬────┘      │
                │   │          │            │           │
                │   │          ▼            ▼           │
                │   │     ┌────────────────────┐        │
                │   │     │  Safety Monitors   │        │
                │   │     └─────────┬──────────┘        │
                │   │               │                   │
                │   │               ▼                   │
                │   │     ┌────────────────────┐        │
                │   │     │   L-AE-CR Eval     │        │
                │   │     └─────────┬──────────┘        │
                │   │               │                   │
                │   └───────────────┼───────────────────┘
                │                   ▼
                │         ┌────────────────┐
                │         │   Jury Panel   │
                │         └───────┬────────┘
                │                 │
                └─────────────────┴──────────▶ Result
```

---

## 4. Core Components

### 4.1 Debate Orchestrator

The central coordinator managing debate lifecycle.

```python
class Debate:
    """Main debate orchestrator."""
    
    def __init__(
        self,
        topic: str,
        agents: list[Agent],
        jury: JuryPanel,
        rounds: int = 3,
        monitors: list[SafetyMonitor] | None = None,
        config: DebateConfig | None = None
    ):
        self.topic = topic
        self.agents = agents
        self.jury = jury
        self.rounds = rounds
        self.monitors = monitors or []
        self.config = config or DebateConfig()
        self.state = DebateState.SETUP
        self.transcript: list[Turn] = []
    
    async def run(self) -> DebateResult:
        """Execute the full debate and return verdict."""
        self._transition_state(DebateState.OPENING)
        await self._run_opening_statements()
        
        self._transition_state(DebateState.DEBATE)
        for round_num in range(self.rounds):
            await self._run_round(round_num)
        
        self._transition_state(DebateState.CLOSING)
        await self._run_closing_statements()
        
        self._transition_state(DebateState.DELIBERATION)
        verdict = await self.jury.deliberate(self.transcript)
        
        self._transition_state(DebateState.COMPLETE)
        return DebateResult(
            verdict=verdict,
            transcript=self.transcript,
            safety_alerts=self._collect_safety_alerts()
        )
```

**State Machine:**

```
SETUP → OPENING → DEBATE → CLOSING → DELIBERATION → COMPLETE
                     ↑                                   
                     └─── (rounds loop) ◄────────────────┘
```

### 4.2 Agent with H-L-DAG

Agents generate arguments following the hierarchical structure.

```python
class Agent:
    """Debate agent with H-L-DAG argument generation."""
    
    def __init__(
        self,
        name: str,
        role: str,
        model: str | BaseModel,
        reasoning: ReasoningConfig | None = None,
        persona: str | None = None
    ):
        self.name = name
        self.role = role
        self.model = self._resolve_model(model)
        self.reasoning = reasoning
        self.persona = persona
    
    async def generate_argument(
        self,
        context: DebateContext,
        level: ArgumentLevel = ArgumentLevel.TACTICAL
    ) -> Argument:
        """Generate argument at specified hierarchical level."""
        
        # Build prompt based on H-L-DAG level
        prompt = self._build_hdag_prompt(context, level)
        
        # Generate with optional reasoning
        if self.reasoning and self.reasoning.enabled:
            response = await self.model.generate_with_reasoning(
                messages=prompt,
                thinking_budget=self.reasoning.thinking_budget
            )
            thinking_trace = response.thinking
        else:
            response = await self.model.generate(messages=prompt)
            thinking_trace = None
        
        # Parse and structure the argument
        return Argument(
            agent=self.name,
            level=level,
            content=response.content,
            evidence=self._extract_evidence(response.content),
            causal_links=self._extract_causal_links(response.content),
            thinking_trace=thinking_trace
        )
    
    def _build_hdag_prompt(
        self,
        context: DebateContext,
        level: ArgumentLevel
    ) -> list[Message]:
        """Build hierarchical argument generation prompt."""
        
        system_prompt = f"""You are {self.name}, a debate participant.
Role: {self.role}
{f'Persona: {self.persona}' if self.persona else ''}

You are generating a {level.value}-level argument.

{HDAG_LEVEL_INSTRUCTIONS[level]}

Structure your argument with:
1. Clear thesis statement
2. Supporting evidence with citations
3. Causal reasoning (if X then Y because Z)
4. Acknowledgment of counterarguments
5. Ethical considerations
"""
        
        return [
            Message(role="system", content=system_prompt),
            Message(role="user", content=self._format_context(context))
        ]
```

**H-L-DAG Level Instructions:**

```python
HDAG_LEVEL_INSTRUCTIONS = {
    ArgumentLevel.STRATEGIC: """
        Strategic Level: Present your overarching position.
        - State your main thesis clearly
        - Outline the key pillars of your argument
        - Establish the framework for evaluation
        - Consider long-term implications
    """,
    ArgumentLevel.TACTICAL: """
        Tactical Level: Develop supporting arguments.
        - Provide specific evidence for each pillar
        - Draw causal connections between points
        - Address likely counterarguments
        - Cite sources where applicable
    """,
    ArgumentLevel.OPERATIONAL: """
        Operational Level: Ground arguments in specifics.
        - Cite specific facts, statistics, or examples
        - Reference concrete case studies
        - Provide actionable implications
        - Detail implementation considerations
    """
}
```

### 4.3 L-AE-CR Adaptive Evaluation

Dynamic evaluation with causal reasoning.

```python
class AdaptiveEvaluator:
    """L-AE-CR: Adaptive Evaluation with Causal Reasoning."""
    
    DEFAULT_CRITERIA = {
        "logical_coherence": 0.25,
        "evidence_quality": 0.25,
        "causal_reasoning": 0.20,
        "ethical_alignment": 0.15,
        "persuasiveness": 0.15
    }
    
    def __init__(
        self,
        criteria: dict[str, float] | None = None,
        adaptation_rate: float = 0.1
    ):
        self.criteria = criteria or self.DEFAULT_CRITERIA.copy()
        self.adaptation_rate = adaptation_rate
        self.causal_graph = CausalGraph()
    
    async def evaluate_argument(
        self,
        argument: Argument,
        context: DebateContext
    ) -> ArgumentEvaluation:
        """Evaluate argument with adaptive criteria."""
        
        # Adapt criteria weights based on context
        adapted_criteria = self._adapt_criteria(context)
        
        # Score each criterion
        scores = {}
        for criterion, weight in adapted_criteria.items():
            score = await self._score_criterion(argument, criterion, context)
            scores[criterion] = score
        
        # Build causal links
        causal_score = self._evaluate_causal_reasoning(argument)
        
        # Compute weighted total
        total = sum(
            scores[c] * w for c, w in adapted_criteria.items()
        )
        
        return ArgumentEvaluation(
            argument_id=argument.id,
            scores=scores,
            weights=adapted_criteria,
            causal_score=causal_score,
            total_score=total,
            causal_graph_update=self.causal_graph.get_updates()
        )
    
    def _adapt_criteria(self, context: DebateContext) -> dict[str, float]:
        """Dynamically adjust criteria weights based on context."""
        
        adapted = self.criteria.copy()
        
        # Increase ethical weight for sensitive topics
        if context.topic_sensitivity > 0.7:
            adapted["ethical_alignment"] *= 1.5
        
        # Increase evidence weight in later rounds
        round_factor = 1 + (context.current_round / context.total_rounds) * 0.3
        adapted["evidence_quality"] *= round_factor
        
        # Increase causal reasoning for complex topics
        if context.topic_complexity > 0.7:
            adapted["causal_reasoning"] *= 1.3
        
        # Renormalize weights
        total = sum(adapted.values())
        return {k: v / total for k, v in adapted.items()}
```

### 4.4 Jury Panel

Multi-perspective evaluation and verdict.

```python
class JuryPanel:
    """Jury scoring mechanism with multiple evaluators."""
    
    def __init__(
        self,
        evaluators: int = 3,
        criteria: list[str] | None = None,
        model: str = "gpt-4o",
        consensus_threshold: float = 0.7
    ):
        self.evaluators = [
            JuryMember(
                id=f"juror_{i}",
                model=model,
                perspective=self._assign_perspective(i)
            )
            for i in range(evaluators)
        ]
        self.criteria = criteria or DEFAULT_JURY_CRITERIA
        self.consensus_threshold = consensus_threshold
    
    async def deliberate(
        self,
        transcript: list[Turn]
    ) -> Verdict:
        """Conduct jury deliberation and reach verdict."""
        
        # Each juror evaluates independently
        evaluations = await asyncio.gather(*[
            juror.evaluate(transcript) for juror in self.evaluators
        ])
        
        # Build consensus
        verdict = self._build_consensus(evaluations)
        
        # Calculate confidence
        confidence = self._calculate_confidence(evaluations, verdict)
        
        return Verdict(
            decision=verdict.decision,
            confidence=confidence,
            reasoning=verdict.reasoning,
            dissenting_opinions=self._collect_dissents(evaluations, verdict),
            score_breakdown=self._aggregate_scores(evaluations)
        )
    
    def _assign_perspective(self, index: int) -> str:
        """Assign diverse perspectives to jurors."""
        perspectives = [
            "analytical",      # Focus on logic and evidence
            "ethical",         # Focus on moral implications  
            "practical",       # Focus on feasibility and impact
            "adversarial",     # Challenge all arguments
            "synthesizing"     # Find common ground
        ]
        return perspectives[index % len(perspectives)]
```

---

## 5. Safety Layer

### 5.1 Safety Monitor Interface

```python
class SafetyMonitor(ABC):
    """Abstract base for safety monitors."""
    
    def __init__(
        self,
        mode: Literal["passive", "active"] = "passive",
        alert_threshold: float = 0.7,
        halt_threshold: float = 0.95
    ):
        self.mode = mode
        self.alert_threshold = alert_threshold
        self.halt_threshold = halt_threshold
        self.alerts: list[SafetyAlert] = []
    
    @abstractmethod
    async def analyze(
        self,
        turn: Turn,
        context: DebateContext
    ) -> SafetyResult:
        """Analyze a turn for safety concerns."""
        pass
    
    def should_halt(self, result: SafetyResult) -> bool:
        """Determine if debate should be halted."""
        return (
            self.mode == "active" and 
            result.severity >= self.halt_threshold
        )
```

### 5.2 Sandbagging Detector

Detects when agents deliberately underperform.

```python
class SandbagDetector(SafetyMonitor):
    """Detect deliberate capability hiding (sandbagging)."""
    
    def __init__(
        self,
        sensitivity: float = 0.8,
        baseline_window: int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sensitivity = sensitivity
        self.baseline_window = baseline_window
        self.agent_baselines: dict[str, AgentBaseline] = {}
    
    async def analyze(
        self,
        turn: Turn,
        context: DebateContext
    ) -> SafetyResult:
        """Analyze turn for sandbagging indicators."""
        
        indicators = []
        
        # 1. Capability drop detection
        baseline = self._get_or_create_baseline(turn.agent)
        current_quality = await self._assess_argument_quality(turn.argument)
        
        if baseline.is_established:
            drop = baseline.average_quality - current_quality
            if drop > self.sensitivity * baseline.std_dev:
                indicators.append(SandbagIndicator(
                    type="capability_drop",
                    severity=min(1.0, drop / baseline.average_quality),
                    evidence=f"Quality dropped from {baseline.average_quality:.2f} to {current_quality:.2f}"
                ))
        
        # 2. Strategic timing detection
        if self._detect_strategic_timing(turn, context):
            indicators.append(SandbagIndicator(
                type="strategic_timing",
                severity=0.6,
                evidence="Performance drop at strategically significant moment"
            ))
        
        # 3. Selective engagement detection
        if await self._detect_selective_engagement(turn, context):
            indicators.append(SandbagIndicator(
                type="selective_engagement",
                severity=0.7,
                evidence="Agent avoiding specific arguments despite demonstrated capability"
            ))
        
        # Update baseline
        self._update_baseline(turn.agent, current_quality)
        
        # Compute overall severity
        severity = max((i.severity for i in indicators), default=0.0)
        
        if severity >= self.alert_threshold:
            self.alerts.append(SafetyAlert(
                monitor=self.__class__.__name__,
                agent=turn.agent,
                type="sandbagging",
                severity=severity,
                indicators=indicators
            ))
        
        return SafetyResult(
            monitor=self.__class__.__name__,
            severity=severity,
            indicators=indicators,
            should_alert=severity >= self.alert_threshold
        )
```

### 5.3 Deception Monitor

Detects misleading arguments or manipulation.

```python
class DeceptionMonitor(SafetyMonitor):
    """Detect deceptive arguments and manipulation."""
    
    async def analyze(
        self,
        turn: Turn,
        context: DebateContext
    ) -> SafetyResult:
        """Analyze turn for deception indicators."""
        
        indicators = []
        
        # 1. Factual consistency check
        inconsistencies = await self._check_factual_consistency(
            turn.argument,
            context.transcript
        )
        if inconsistencies:
            indicators.append(DeceptionIndicator(
                type="factual_inconsistency",
                severity=len(inconsistencies) * 0.2,
                evidence=inconsistencies
            ))
        
        # 2. Logical fallacy detection
        fallacies = await self._detect_logical_fallacies(turn.argument)
        if fallacies:
            indicators.append(DeceptionIndicator(
                type="logical_fallacy",
                severity=0.5,
                evidence=fallacies
            ))
        
        # 3. Emotional manipulation detection
        manipulation_score = await self._detect_emotional_manipulation(
            turn.argument
        )
        if manipulation_score > 0.6:
            indicators.append(DeceptionIndicator(
                type="emotional_manipulation",
                severity=manipulation_score,
                evidence="High emotional manipulation indicators"
            ))
        
        # 4. Citation fabrication check
        fabricated = await self._check_citation_validity(turn.argument)
        if fabricated:
            indicators.append(DeceptionIndicator(
                type="citation_fabrication",
                severity=0.9,
                evidence=fabricated
            ))
        
        severity = max((i.severity for i in indicators), default=0.0)
        
        return SafetyResult(
            monitor=self.__class__.__name__,
            severity=severity,
            indicators=indicators,
            should_alert=severity >= self.alert_threshold
        )
```

---

## 6. Integrations

### 6.1 LangGraph Integration

```python
class ArtemisDebateNode:
    """LangGraph node for structured debates."""
    
    def __init__(
        self,
        agents: int | list[Agent] = 3,
        rounds: int = 2,
        jury_size: int = 3,
        monitors: list[SafetyMonitor] | None = None
    ):
        self.agents = agents
        self.rounds = rounds
        self.jury_size = jury_size
        self.monitors = monitors
    
    async def __call__(self, state: State) -> State:
        """Execute debate as LangGraph node."""
        
        # Extract topic from state
        topic = state.get("debate_topic") or state.get("question")
        
        # Create or use provided agents
        agents = self._resolve_agents(state)
        
        # Run debate
        debate = Debate(
            topic=topic,
            agents=agents,
            jury=JuryPanel(evaluators=self.jury_size),
            rounds=self.rounds,
            monitors=self.monitors
        )
        
        result = await debate.run()
        
        # Update state
        return {
            **state,
            "debate_result": result,
            "verdict": result.verdict.decision,
            "confidence": result.verdict.confidence,
            "reasoning": result.verdict.reasoning
        }
```

### 6.2 MCP Server

```python
class ArtemisMCPServer:
    """MCP server exposing ARTEMIS capabilities."""
    
    TOOLS = [
        Tool(
            name="artemis_debate",
            description="Start a structured multi-agent debate on a topic",
            input_schema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "perspectives": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "rounds": {"type": "integer", "default": 2}
                },
                "required": ["topic"]
            }
        ),
        Tool(
            name="artemis_verdict",
            description="Get the verdict from a completed debate",
            input_schema={
                "type": "object",
                "properties": {
                    "debate_id": {"type": "string"}
                },
                "required": ["debate_id"]
            }
        )
    ]
    
    async def handle_tool_call(
        self,
        name: str,
        arguments: dict
    ) -> ToolResult:
        """Handle MCP tool calls."""
        
        if name == "artemis_debate":
            return await self._start_debate(arguments)
        elif name == "artemis_verdict":
            return await self._get_verdict(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
```

---

## 7. Data Models

### 7.1 Core Models

```python
class Argument(BaseModel):
    """A structured argument in the debate."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent: str
    level: ArgumentLevel
    content: str
    evidence: list[Evidence] = []
    causal_links: list[CausalLink] = []
    ethical_score: float | None = None
    thinking_trace: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Turn(BaseModel):
    """A single turn in the debate."""
    round: int
    sequence: int
    agent: str
    argument: Argument
    evaluation: ArgumentEvaluation | None = None
    safety_results: list[SafetyResult] = []

class Verdict(BaseModel):
    """Final jury verdict."""
    decision: str
    confidence: float
    reasoning: str
    dissenting_opinions: list[DissentingOpinion] = []
    score_breakdown: dict[str, float]

class DebateResult(BaseModel):
    """Complete debate result."""
    debate_id: str
    topic: str
    verdict: Verdict
    transcript: list[Turn]
    safety_alerts: list[SafetyAlert]
    metadata: DebateMetadata
```

---

## 8. Configuration

```python
class DebateConfig(BaseModel):
    """Debate configuration."""
    
    # Timing
    turn_timeout: int = 60  # seconds
    round_timeout: int = 300
    
    # Argument generation
    max_argument_tokens: int = 1000
    require_evidence: bool = True
    require_causal_links: bool = True
    
    # Evaluation
    evaluation_criteria: dict[str, float] | None = None
    adaptation_enabled: bool = True
    
    # Safety
    safety_mode: Literal["off", "passive", "active"] = "passive"
    halt_on_safety_violation: bool = False
    
    # Logging
    log_level: str = "INFO"
    trace_enabled: bool = False

class ReasoningConfig(BaseModel):
    """Configuration for reasoning models."""
    enabled: bool = True
    thinking_budget: int = 8000
    strategy: Literal[
        "think-then-argue",
        "interleaved",
        "final-reflection"
    ] = "think-then-argue"
```

---

## 9. Error Handling

```python
class ArtemisError(Exception):
    """Base exception for ARTEMIS."""
    pass

class DebateConfigError(ArtemisError):
    """Invalid debate configuration."""
    pass

class AgentError(ArtemisError):
    """Agent-related error."""
    pass

class ModelError(ArtemisError):
    """LLM provider error."""
    pass

class SafetyHaltError(ArtemisError):
    """Debate halted due to safety violation."""
    def __init__(self, alert: SafetyAlert):
        self.alert = alert
        super().__init__(f"Safety halt: {alert.type} by {alert.agent}")

class EvaluationError(ArtemisError):
    """Evaluation error."""
    pass
```

---

## 10. Future Considerations (v2.0)

### 10.1 Hierarchical Debates

Debates within debates for complex multi-faceted topics.

### 10.2 Steering Vectors

Real-time behavior control using activation steering.

### 10.3 Multimodal Arguments

Support for image, document, and data-based evidence.

### 10.4 Formal Verification

Integration with UPIR for argument validity verification.

### 10.5 Streaming Debates

Real-time streaming of debate progress for live applications.

---

## Appendix A: Mathematical Formulations

### A.1 Adaptive Criteria Weighting

$$W'_i = \frac{W_i \cdot f(C)}{\sum_j W_j \cdot f(C)}$$

Where:
- $W_i$ = original weight for criterion $i$
- $f(C)$ = context adaptation function
- $W'_i$ = adapted weight

### A.2 Causal Reasoning Score

$$CRS = \frac{\sum_{l \in L} strength(l) \cdot validity(l)}{|L|}$$

Where:
- $L$ = set of causal links in argument
- $strength(l)$ = causal relationship strength
- $validity(l)$ = logical validity of link

### A.3 Jury Consensus Score

$$Consensus = 1 - \frac{\sigma(V)}{\mu(V)}$$

Where:
- $V$ = vector of juror verdicts
- $\sigma$ = standard deviation
- $\mu$ = mean

---

## Appendix B: References

1. Mitra, S. (2025). "Adaptive Reasoning and Evaluation Framework for Multi-agent Intelligent Systems in Debate-driven Decision-making." Technical Disclosure Commons.

2. AI Metacognition Toolkit - https://github.com/bassrehab/ai-metacognition-toolkit

3. Steering Vectors for Agent Behavior Control - https://github.com/bassrehab/steering-vectors-agents
