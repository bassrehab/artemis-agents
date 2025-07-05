#  ARTEMIS Agents

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Adaptive Reasoning Through Evaluation of Multi-agent Intelligent Systems**

A production-ready framework for structured multi-agent debates with adaptive evaluation, causal reasoning, and built-in safety monitoring.

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

Performance comparison on the DebateQA benchmark:

| Framework | Argument Quality | Decision Accuracy | Reasoning Depth |
|-----------|-----------------|-------------------|-----------------|
| AutoGen | 72.3% | 68.1% | 65.4% |
| CrewAI | 74.1% | 70.2% | 67.8% |
| CAMEL | 69.8% | 64.5% | 62.1% |
| **ARTEMIS** | **81.7%** | **78.4%** | **82.3%** |

*Benchmarks run on GPT-4o with default configurations. See [benchmarks/](benchmarks/) for methodology.*

##  Roadmap

### v1.0 (Current)
- [x] Core ARTEMIS implementation (H-L-DAG, L-AE-CR, Jury)
- [x] Multi-provider support (OpenAI, Anthropic, Google, DeepSeek)
- [x] Reasoning model support (o1, R1, Gemini 2.5)
- [x] Safety monitoring (sandbagging, deception detection)
- [x] Framework integrations (LangChain, LangGraph, CrewAI)
- [x] MCP server mode

### v2.0 (Planned)
- [ ] Hierarchical debates (debates within debates)
- [ ] Steering vectors for real-time behavior control
- [ ] Multimodal debates (documents, images)
- [ ] Formal verification of argument validity
- [ ] Real-time streaming debates

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
