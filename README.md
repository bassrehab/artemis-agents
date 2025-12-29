<p align="center">
  <img src="https://raw.githubusercontent.com/bassrehab/artemis-agents/main/assets/logo.svg" alt="ARTEMIS Logo" width="80" height="80">
</p>

<h1 align="center">ARTEMIS Agents</h1>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License: Apache 2.0"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

<p align="center">
  <strong>Adaptive Reasoning Through Evaluation of Multi-agent Intelligent Systems</strong>
</p>

<p align="center">
  <em>A production-ready framework for structured multi-agent debates with adaptive evaluation, causal reasoning, and built-in safety monitoring.</em>
</p>

---

##  What is ARTEMIS?

ARTEMIS is an open-source implementation of the [Adaptive Reasoning and Evaluation Framework for Multi-agent Intelligent Systems](https://www.tdcommons.org/dpubs_series/7729/) — a framework designed to improve complex decision-making through structured debates between AI agents.

Unlike general-purpose multi-agent frameworks, ARTEMIS is **purpose-built for debate-driven decision-making** with:

- **Hierarchical Argument Generation (H-L-DAG)**: Structured, context-aware argument synthesis
- **Adaptive Evaluation with Causal Reasoning (L-AE-CR)**: Dynamic criteria weighting with causal analysis
- **Jury Scoring Mechanism**: Fair, multi-perspective evaluation of arguments
- **Ethical Alignment**: Built-in ethical considerations in both generation and evaluation
- **Safety Monitoring**: Real-time detection of sandbagging, deception, and manipulation

##  Why ARTEMIS?

| Feature | AutoGen | CrewAI | CAMEL | **ARTEMIS** |
|---------|---------|--------|-------|-------------|
| Multi-agent debates | ⚠️ Basic | ⚠️ Basic | ⚠️ 2-3 agents | ✅ N agents |
| Structured argument generation | ❌ | ❌ | ❌ | ✅ H-L-DAG |
| Causal reasoning | ❌ | ❌ | ❌ | ✅ L-AE-CR |
| Adaptive evaluation | ❌ | ❌ | ❌ | ✅ Dynamic weights |
| Ethical alignment | ❌ | ❌ | ❌ | ✅ Built-in |
| Sandbagging detection | ❌ | ❌ | ❌ | ✅ Metacognition |
| Reasoning model support | ⚠️ | ⚠️ | ❌ | ✅ o1/R1 native |
| MCP server mode | ❌ | ❌ | ❌ | ✅ |
| Real-time streaming | ⚠️ | ❌ | ❌ | ✅ v2 |
| Hierarchical debates | ❌ | ❌ | ❌ | ✅ v2 |
| Multimodal evidence | ⚠️ | ⚠️ | ❌ | ✅ v2 |
| Steering vectors | ❌ | ❌ | ❌ | ✅ v2 |
| Argument verification | ❌ | ❌ | ❌ | ✅ v2 |

##  Installation

```bash
pip install artemis-agents
```

Or install from source:

```bash
git clone https://github.com/bassrehab/artemis-agents.git
cd artemis-agents
pip install -e ".[dev]"
```

##  Quick Start

### Basic Debate

```python
from artemis import Debate, Agent, JuryPanel

# Create debate agents with different perspectives
agents = [
    Agent(
        name="Proponent",
        role="Argues in favor of the proposition",
        model="gpt-4o"
    ),
    Agent(
        name="Opponent",
        role="Argues against the proposition",
        model="gpt-4o"
    ),
    Agent(
        name="Moderator",
        role="Ensures balanced discussion and identifies logical fallacies",
        model="gpt-4o"
    ),
]

# Create jury panel for evaluation
jury = JuryPanel(
    evaluators=3,
    criteria=["logical_coherence", "evidence_quality", "ethical_considerations"]
)

# Run the debate
debate = Debate(
    topic="Should AI systems be given legal personhood?",
    agents=agents,
    jury=jury,
    rounds=3
)

result = debate.run()

print(f"Verdict: {result.verdict}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Key arguments: {result.summary}")
```

### With Reasoning Models (o1/R1)

```python
from artemis import Debate, Agent
from artemis.models import ReasoningConfig

# Enable extended thinking for deeper analysis
agent = Agent(
    name="Deep Analyst",
    role="Provides thoroughly reasoned arguments",
    model="deepseek-r1",
    reasoning=ReasoningConfig(
        enabled=True,
        thinking_budget=16000,  # tokens for internal reasoning
        strategy="think-then-argue"
    )
)
```

### With Safety Monitoring

```python
from artemis import Debate
from artemis.safety import SandbagDetector, DeceptionMonitor

debate = Debate(
    topic="Complex ethical scenario",
    agents=[...],
    monitors=[
        SandbagDetector(sensitivity=0.8),    # Detect capability hiding
        DeceptionMonitor(alert_threshold=0.7) # Detect misleading arguments
    ]
)

result = debate.run()

# Check for safety flags
for alert in result.safety_alerts:
    print(f"⚠️ {alert.agent}: {alert.type} - {alert.description}")
```

### LangGraph Integration

```python
from langgraph.graph import StateGraph
from artemis.integrations import ArtemisDebateNode

# Use ARTEMIS as a node in your LangGraph workflow
workflow = StateGraph(State)

workflow.add_node(
    "structured_debate",
    ArtemisDebateNode(
        agents=3,
        rounds=2,
        jury_size=3
    )
)

workflow.add_edge("gather_info", "structured_debate")
workflow.add_edge("structured_debate", "final_decision")
```

### MCP Server Mode

```bash
# Start ARTEMIS as an MCP server
artemis serve --port 8080
```

Any MCP-compatible client can now invoke structured debates:

```json
{
  "method": "tools/call",
  "params": {
    "name": "artemis_debate",
    "arguments": {
      "topic": "Should we proceed with this investment?",
      "perspectives": ["risk", "opportunity", "ethics"],
      "rounds": 2
    }
  }
}
```

### Streaming Debates (v2)

```python
from artemis import StreamingDebate

debate = StreamingDebate(
    topic="Should we adopt microservices?",
    agents=[...],
)

# Stream events in real-time
async for event in debate.run_streaming():
    if event.event_type == "chunk":
        print(event.content, end="", flush=True)
    elif event.event_type == "argument_complete":
        print(f"\n[{event.agent}] argument complete")
```

### Hierarchical Debates (v2)

```python
from artemis import HierarchicalDebate
from artemis.core.decomposition import LLMTopicDecomposer

# Complex topics are automatically decomposed into sub-debates
debate = HierarchicalDebate(
    topic="Should we rewrite the monolith in microservices?",
    agents=[...],
    decomposer=LLMTopicDecomposer(),
    max_depth=2,  # Allow sub-sub-debates
)

result = await debate.run()
print(f"Final verdict: {result.final_decision}")
print(f"Sub-verdicts: {len(result.sub_verdicts)}")
```

### Steering Vectors (v2)

```python
from artemis.steering import SteeringController, SteeringVector

# Control agent behavior in real-time
controller = SteeringController(
    vector=SteeringVector(
        formality=0.9,      # Very formal
        aggression=0.2,     # Cooperative
        evidence_emphasis=0.8,  # Data-driven
    )
)

agent = Agent(
    name="Analyst",
    steering=controller,
)
```

### Multimodal Evidence (v2)

```python
from artemis.core.multimodal_evidence import MultimodalEvidenceExtractor
from artemis.core.types import ContentPart, ContentType

# Extract evidence from images and documents
extractor = MultimodalEvidenceExtractor(model="gpt-4o")

chart = ContentPart(
    type=ContentType.IMAGE,
    url="https://example.com/revenue-chart.png"
)

evidence = await extractor.extract(chart)
print(f"Extracted: {evidence.text}")
```

### Argument Verification (v2)

```python
from artemis.core.verification import ArgumentVerifier, VerificationSpec

# Verify argument validity
verifier = ArgumentVerifier(
    spec=VerificationSpec(
        rules=["causal_chain", "citation", "logical_consistency"],
        strict_mode=True,
    )
)

report = await verifier.verify(argument, context)
if not report.overall_passed:
    print(f"Violations: {report.violations}")
```

##  Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        ARTEMIS Core                            │
├────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   H-L-DAG   │  │   L-AE-CR   │  │    Jury     │             │
│  │  Argument   │──│  Adaptive   │──│   Scoring   │             │
│  │ Generation  │  │ Evaluation  │  │  Mechanism  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│         │                │                │                    │
│         └────────────────┴────────────────┘                    │
│                          │                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Safety Layer                          │   │
│  │  ┌───────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │   │
│  │  │Sandbagging│  │Deception │  │ Behavior │  │ Ethics  │ │   │
│  │  │ Detector  │  │ Monitor  │  │ Tracker  │  │ Guard   │ │   │
│  │  └───────────┘  └──────────┘  └──────────┘  └─────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
├────────────────────────────────────────────────────────────────┤
│                       Integrations                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │LangChain │  │LangGraph │  │ CrewAI   │  │   MCP    │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
├────────────────────────────────────────────────────────────────┤
│                      Model Providers                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  OpenAI  │  │Anthropic │  │  Google  │  │ DeepSeek │        │
│  │ (GPT-4o) │  │ (Claude) │  │ (Gemini) │  │  (R1)    │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└────────────────────────────────────────────────────────────────┘
```

##  Documentation

- [**Full Documentation**](https://artemis.subhadipmitra.com) - Guides, API reference, examples
- [Examples](examples/) - Real-world usage examples
- [Contributing](CONTRIBUTING.md) - How to contribute

##  Research Foundation

ARTEMIS is based on peer-reviewed research:

> **Adaptive Reasoning and Evaluation Framework for Multi-agent Intelligent Systems in Debate-driven Decision-making**  
> Mitra, S. (2025). Technical Disclosure Commons.  
> [Read the paper](https://www.tdcommons.org/dpubs_series/7729/)

Key innovations from the paper:
- **Hierarchical Argument Generation (H-L-DAG)**: Multi-level argument synthesis with strategic, tactical, and operational layers
- **Adaptive Evaluation with Causal Reasoning (L-AE-CR)**: Dynamic criteria weighting based on debate context
- **Ethical Alignment Integration**: Built-in ethical considerations at every stage

##  Safety Features

ARTEMIS includes novel safety monitoring capabilities:

| Feature | Description |
|---------|-------------|
| **Sandbagging Detection** | Identifies when agents deliberately underperform or withhold capabilities |
| **Deception Monitoring** | Detects misleading arguments or manipulation attempts |
| **Behavioral Drift Tracking** | Monitors for unexpected changes in agent behavior |
| **Ethical Boundary Enforcement** | Ensures debates stay within defined ethical bounds |

These features leverage activation-level analysis and are based on research in AI metacognition.

##  Framework Integrations

ARTEMIS is designed to complement, not replace, existing frameworks:

```python
# LangChain Tool
from artemis.integrations import ArtemisDebateTool
tools = [ArtemisDebateTool()]

# CrewAI Integration  
from artemis.integrations import ArtemisCrewTool
crew = Crew(agents=[...], tools=[ArtemisCrewTool()])

# LangGraph Node
from artemis.integrations import ArtemisDebateNode
graph.add_node("debate", ArtemisDebateNode())
```

##  Benchmarks

We ran 27 debates across three frameworks using GPT-4o. Here's what we found:

| Framework | Argument Quality | Decision Accuracy | Reasoning Depth | Consistency (σ) |
|-----------|------------------|-------------------|-----------------|-----------------|
| **ARTEMIS** | 77.9% | **86.0%** | **75.3%** | **±1.6** |
| AutoGen | **77.3%** | 55.0% | 74.7% | ±0.5 |
| CrewAI | 75.1% | 42.8% | 57.0% | ±16.0 |

**Key findings:**
- ARTEMIS leads in **decision accuracy** (86% vs next best 55%) - the jury deliberation mechanism works
- Lowest variance across runs (±1.6 vs CrewAI's ±16.0) - most consistent and predictable
- Structured H-L-DAG arguments produce reliable reasoning depth

**Trade-off:** ARTEMIS averages 102s per debate vs AutoGen's 36s. The jury deliberation adds latency but improves verdict quality.

*See [benchmarks/ANALYSIS.md](benchmarks/ANALYSIS.md) for methodology and detailed breakdown.*

##  Roadmap

### v1.0
- [x] Core ARTEMIS implementation (H-L-DAG, L-AE-CR, Jury)
- [x] Multi-provider support (OpenAI, Anthropic, Google, DeepSeek)
- [x] Reasoning model support (o1, R1, Gemini 2.5)
- [x] Safety monitoring (sandbagging, deception detection)
- [x] Framework integrations (LangChain, LangGraph, CrewAI)
- [x] MCP server mode

### v2.0 (Current)
- [x] Hierarchical debates (sub-debates for complex topics)
- [x] Steering vectors for real-time behavior control
- [x] Multimodal debates (images, documents, charts)
- [x] Formal verification of argument validity
- [x] Real-time streaming debates

### v3.0 (Planned)
- [ ] Distributed debate execution
- [ ] Custom evaluation plugins
- [ ] Debate templates and presets
- [ ] Advanced causal graph visualization

##  License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

##  Acknowledgments

- Original ARTEMIS framework design published via Google Technical Disclosure Commons
- Safety monitoring capabilities inspired by research in AI metacognition
- Built with support from the open-source AI community

##  Contact

- **Author**: [Subhadip Mitra](https://subhadipmitra.com)
- **GitHub**: [@bassrehab](https://github.com/bassrehab)
- **Twitter/X**: [@bassrehab](https://twitter.com/bassrehab)

---

<p align="center">
  <i>Making AI decision-making more transparent, reasoned, and safe.</i>
</p>
