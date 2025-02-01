# ARTEMIS Agents - Architecture Guide

This document provides a comprehensive overview of the ARTEMIS Agents architecture.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              User Interface                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │  Python API  │  │  MCP Server  │  │  LangChain   │  │  LangGraph  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────────────────┤
│                           Orchestration Layer                            │
│                     ┌─────────────────────────────┐                     │
│                     │     Debate Orchestrator     │                     │
│                     │  ┌───────────────────────┐  │                     │
│                     │  │    State Machine      │  │                     │
│                     │  │ SETUP→OPENING→DEBATE  │  │                     │
│                     │  │ →CLOSING→DELIBERATION │  │                     │
│                     │  └───────────────────────┘  │                     │
│                     └─────────────────────────────┘                     │
├─────────────────────────────────────────────────────────────────────────┤
│                              Core Layer                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
│  │     H-L-DAG     │  │     L-AE-CR     │  │      Jury Panel         │ │
│  │    Argument     │  │    Adaptive     │  │  ┌─────┐ ┌─────┐ ┌───┐  │ │
│  │   Generation    │  │   Evaluation    │  │  │ J1  │ │ J2  │ │J3 │  │ │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │  └─────┘ └─────┘ └───┘  │ │
│  │ │ Strategic   │ │  │ │  Dynamic    │ │  │     Consensus           │ │
│  │ │ Tactical    │ │  │ │  Weights    │ │  │     Building            │ │
│  │ │ Operational │ │  │ │  Causal     │ │  │                         │ │
│  │ └─────────────┘ │  │ └─────────────┘ │  └─────────────────────────┘ │
│  └─────────────────┘  └─────────────────┘                               │
├─────────────────────────────────────────────────────────────────────────┤
│                             Safety Layer                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │  Sandbagging │  │  Deception   │  │   Behavior   │  │   Ethics    │ │
│  │   Detector   │  │   Monitor    │  │   Tracker    │  │   Guard     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────────────────┤
│                            Provider Layer                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │  OpenAI  │  │ Anthropic│  │  Google  │  │ DeepSeek │  │  Ollama  │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Debate Orchestrator

The central coordinator managing the entire debate lifecycle.

**Responsibilities:**
- State machine management
- Round and turn coordination
- Safety monitor integration
- Result aggregation

**State Machine:**

```
┌───────┐     ┌─────────┐     ┌────────┐     ┌─────────┐
│ SETUP │────▶│ OPENING │────▶│ DEBATE │────▶│ CLOSING │
└───────┘     └─────────┘     └────┬───┘     └────┬────┘
                                   │              │
                                   │              ▼
                              (N rounds)    ┌─────────────┐
                                   │        │DELIBERATION │
                                   ▼        └──────┬──────┘
                              ┌────────┐          │
                              │ DEBATE │          ▼
                              └────────┘    ┌──────────┐
                                            │ COMPLETE │
                                            └──────────┘
```

### 2. H-L-DAG (Hierarchical Argument Generation)

Generates structured arguments at three levels:

| Level | Purpose | Content |
|-------|---------|---------|
| **Strategic** | High-level position | Thesis, key pillars, evaluation framework |
| **Tactical** | Supporting arguments | Evidence, causal connections, counterarguments |
| **Operational** | Specific details | Facts, statistics, case studies, examples |

**Argument Structure:**

```
Strategic Level
├── Main Thesis
├── Pillar 1
├── Pillar 2
└── Pillar 3
    │
    ▼
Tactical Level
├── Evidence for Pillar 1
│   ├── Source A
│   └── Source B
├── Causal Chain
│   └── If X → Then Y → Because Z
└── Counter-argument Response
    │
    ▼
Operational Level
├── Specific Statistic
├── Case Study: Example Corp
└── Implementation Detail
```

### 3. L-AE-CR (Adaptive Evaluation with Causal Reasoning)

Dynamically adjusts evaluation criteria based on context.

**Adaptation Factors:**

```python
# Topic Sensitivity Adaptation
if context.topic_sensitivity > 0.7:
    weights["ethical_alignment"] *= 1.5

# Round Progression Adaptation  
round_factor = 1 + (current_round / total_rounds) * 0.3
weights["evidence_quality"] *= round_factor

# Topic Complexity Adaptation
if context.topic_complexity > 0.7:
    weights["causal_reasoning"] *= 1.3

# Always renormalize
weights = normalize(weights)
```

**Causal Graph:**

```
           ┌──────────────────┐
           │  Argument A      │
           │  (Agent 1)       │
           └────────┬─────────┘
                    │ supports
                    ▼
           ┌──────────────────┐
           │  Claim X         │◀──────────┐
           └────────┬─────────┘           │
                    │ leads to            │ contradicts
                    ▼                     │
           ┌──────────────────┐    ┌──────┴───────┐
           │  Conclusion Y    │    │  Argument B  │
           └──────────────────┘    │  (Agent 2)   │
                                   └──────────────┘
```

### 4. Jury Panel

Multi-perspective evaluation with consensus building.

**Jury Perspectives:**

| Perspective | Focus |
|-------------|-------|
| Analytical | Logic, evidence, structure |
| Ethical | Moral implications, values |
| Practical | Feasibility, implementation |
| Adversarial | Challenges all arguments |
| Synthesizing | Finds common ground |

**Deliberation Process:**

```
┌─────────────────────────────────────────────────────────┐
│                  Independent Evaluation                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  Juror 1 │  │  Juror 2 │  │  Juror 3 │              │
│  │ Score: A │  │ Score: B │  │ Score: A │              │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│       │             │             │                     │
│       └─────────────┼─────────────┘                     │
│                     ▼                                   │
│           ┌─────────────────┐                          │
│           │ Consensus Check │                          │
│           │ Agreement: 67%  │                          │
│           └────────┬────────┘                          │
│                    │                                    │
│         ┌──────────┴──────────┐                        │
│         ▼                     ▼                        │
│   ┌───────────┐        ┌────────────┐                  │
│   │  Verdict  │        │ Dissenting │                  │
│   │ Decision A│        │  Opinion B │                  │
│   │ Conf: 75% │        │  (Juror 2) │                  │
│   └───────────┘        └────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

---

## Safety Architecture

### Monitor Integration

Safety monitors are integrated as middleware:

```
                    Turn Generated
                          │
                          ▼
            ┌─────────────────────────┐
            │    Safety Pipeline      │
            │  ┌───────────────────┐  │
            │  │ SandbagDetector   │──┼──▶ Alert?
            │  └───────────────────┘  │
            │  ┌───────────────────┐  │
            │  │ DeceptionMonitor  │──┼──▶ Alert?
            │  └───────────────────┘  │
            │  ┌───────────────────┐  │
            │  │ BehaviorTracker   │──┼──▶ Alert?
            │  └───────────────────┘  │
            └────────────┬────────────┘
                         │
                         ▼
                  Severity Check
                         │
            ┌────────────┴────────────┐
            │                         │
            ▼                         ▼
       < threshold              ≥ threshold
            │                         │
            ▼                         ▼
       Continue                 Active Mode?
                                      │
                         ┌────────────┴────────────┐
                         │                         │
                         ▼                         ▼
                    Passive                    Active
                    (Log Alert)           (Halt Debate)
```

### Sandbagging Detection Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    Sandbagging Detector                           │
│                                                                   │
│  1. Baseline Establishment (first N turns)                       │
│     ┌─────────────────────────────────────────────────────────┐  │
│     │ Turn 1: Quality 0.82                                    │  │
│     │ Turn 2: Quality 0.85                                    │  │
│     │ Turn 3: Quality 0.79                                    │  │
│     │ ─────────────────────────                               │  │
│     │ Baseline: μ=0.82, σ=0.03                                │  │
│     └─────────────────────────────────────────────────────────┘  │
│                                                                   │
│  2. Ongoing Detection                                            │
│     ┌─────────────────────────────────────────────────────────┐  │
│     │ Turn 4: Quality 0.45                                    │  │
│     │ Drop: 0.82 - 0.45 = 0.37                                │  │
│     │ Threshold: sensitivity × σ = 0.8 × 0.03 = 0.024         │  │
│     │ 0.37 > 0.024 → ALERT                                    │  │
│     └─────────────────────────────────────────────────────────┘  │
│                                                                   │
│  3. Strategic Timing Check                                       │
│     Is this a critical moment? → Increases severity              │
│                                                                   │
│  4. Selective Engagement Check                                   │
│     Avoiding specific topics? → Additional indicator             │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Complete Debate Flow

```
User Request
     │
     ▼
┌──────────┐    ┌─────────────┐    ┌──────────────┐
│  Parse   │───▶│  Configure  │───▶│   Validate   │
│  Input   │    │   Debate    │    │    Setup     │
└──────────┘    └─────────────┘    └──────┬───────┘
                                          │
                                          ▼
                                   ┌──────────────┐
                               ┌───│   OPENING    │
                               │   └──────┬───────┘
                               │          │
                               │          ▼
                               │   ┌──────────────┐
                               │   │    Agent 1   │──▶ Strategic Argument
                               │   │    Agent 2   │──▶ Strategic Argument
                               │   │    Agent N   │──▶ Strategic Argument
                               │   └──────┬───────┘
                               │          │
     ┌────────────────────────────────────┘
     │
     │  For each round:
     │   ┌──────────────────────────────────────────────────────┐
     │   │  ┌────────────┐    ┌──────────────┐    ┌──────────┐ │
     │   │  │   Agent    │───▶│    Safety    │───▶│ Evaluate │ │
     │   │  │  Generate  │    │   Monitors   │    │ L-AE-CR  │ │
     │   │  └────────────┘    └──────────────┘    └──────────┘ │
     │   │         │                                    │       │
     │   │         └────────────────────────────────────┘       │
     │   │                          │                           │
     │   │                          ▼                           │
     │   │                   Store in Transcript                │
     │   └──────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   CLOSING    │───▶│ DELIBERATION │───▶│   VERDICT    │
│  Statements  │    │  Jury Panel  │    │    Result    │
└──────────────┘    └──────────────┘    └──────────────┘
```

---

## Extension Points

### Adding a New Model Provider

```python
# 1. Implement BaseModel interface
class NewProvider(BaseModel):
    async def generate(self, messages, **kwargs) -> Response:
        # Implementation
    
    async def generate_with_reasoning(self, messages, budget) -> ReasoningResponse:
        # Implementation (or raise NotImplementedError)

# 2. Register in factory
MODEL_REGISTRY["new-provider"] = NewProvider

# 3. Use in agents
agent = Agent(model="new-provider-model-name")
```

### Adding a New Safety Monitor

```python
# 1. Implement SafetyMonitor interface
class NewMonitor(SafetyMonitor):
    async def analyze(self, turn, context) -> SafetyResult:
        indicators = []
        
        # Detection logic
        if self._detect_issue(turn):
            indicators.append(SafetyIndicator(...))
        
        severity = max(i.severity for i in indicators) if indicators else 0
        return SafetyResult(
            monitor=self.__class__.__name__,
            severity=severity,
            indicators=indicators
        )

# 2. Use in debates
debate = Debate(
    monitors=[NewMonitor(mode="active")]
)
```

### Adding a Framework Integration

```python
# 1. Create adapter following framework patterns
class ArtemisNewFrameworkAdapter:
    def __init__(self, **debate_kwargs):
        self.debate_kwargs = debate_kwargs
    
    def as_tool(self):
        # Return framework-specific tool wrapper
        pass

# 2. Export from integrations module
```

---

## Performance Considerations

### Async Architecture

All I/O operations are async to maximize throughput:

```python
# Parallel agent turns (when applicable)
results = await asyncio.gather(*[
    agent.generate_argument(context)
    for agent in agents
])

# Parallel safety monitoring
safety_results = await asyncio.gather(*[
    monitor.analyze(turn, context)
    for monitor in monitors
])

# Parallel jury evaluation
evaluations = await asyncio.gather(*[
    juror.evaluate(transcript)
    for juror in jury.members
])
```

### Caching Strategies

- **Model responses**: Consider caching for repeated prompts
- **Evaluation scores**: Cache per-argument evaluations
- **Baselines**: Maintain rolling windows efficiently

### Token Optimization

- Use appropriate model sizes for different tasks
- Truncate context when approaching limits
- Summarize transcripts for jury evaluation

---

## Security Considerations

### API Key Management

- Never log API keys
- Use environment variables
- Support key rotation

### Input Validation

- Validate all user inputs
- Sanitize debate topics
- Limit argument lengths

### Output Filtering

- Safety monitors can filter harmful content
- Ethics guard enforces boundaries
- Configurable content policies
