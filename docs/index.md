<p align="center">
  <img src="assets/logo.svg" alt="ARTEMIS Logo" width="120" height="120">
</p>

# ARTEMIS Agents

**Adaptive Reasoning Through Evaluation of Multi-agent Intelligent Systems**

A production-ready framework for structured multi-agent debates with adaptive evaluation, causal reasoning, and built-in safety monitoring.

---

## What is ARTEMIS?

ARTEMIS is an open-source implementation of the [Adaptive Reasoning and Evaluation Framework for Multi-agent Intelligent Systems](https://www.tdcommons.org/dpubs_series/7729/) — a framework designed to improve complex decision-making through structured debates between AI agents.

Unlike general-purpose multi-agent frameworks, ARTEMIS is **purpose-built for debate-driven decision-making** with:

- **Hierarchical Argument Generation (H-L-DAG)**: Structured, context-aware argument synthesis at strategic, tactical, and operational levels
- **Adaptive Evaluation with Causal Reasoning (L-AE-CR)**: Dynamic criteria weighting with causal analysis
- **Jury Scoring Mechanism**: Fair, multi-perspective evaluation of arguments
- **Ethical Alignment**: Built-in ethical considerations in both generation and evaluation
- **Safety Monitoring**: Real-time detection of sandbagging, deception, and manipulation

## Why ARTEMIS?

| Feature | AutoGen | CrewAI | CAMEL | **ARTEMIS** |
|---------|---------|--------|-------|-------------|
| Multi-agent debates | Basic | Basic | 2-3 agents | **N agents** |
| Structured argument generation | No | No | No | **H-L-DAG** |
| Causal reasoning | No | No | No | **L-AE-CR** |
| Adaptive evaluation | No | No | No | **Dynamic weights** |
| Ethical alignment | No | No | No | **Built-in** |
| Sandbagging detection | No | No | No | **Metacognition** |
| Reasoning model support | Limited | Limited | No | **o1/R1 native** |
| MCP server mode | No | No | No | **Yes** |
| Real-time streaming | Limited | No | No | **v2** |
| Hierarchical debates | No | No | No | **v2** |
| Multimodal evidence | Limited | Limited | No | **v2** |
| Steering vectors | No | No | No | **v2** |
| Argument verification | No | No | No | **v2** |

## Quick Example

```python
from artemis import Debate, Agent

# Create debate agents
agents = [
    Agent(name="Proponent", role="Argues in favor", model="gpt-4o"),
    Agent(name="Opponent", role="Argues against", model="gpt-4o"),
]

# Run the debate
debate = Debate(
    topic="Should AI systems be given legal personhood?",
    agents=agents,
    rounds=3
)

result = await debate.run()

print(f"Verdict: {result.verdict.decision}")
print(f"Confidence: {result.verdict.confidence:.0%}")
```

## What's New in v2.0

ARTEMIS v2.0 introduces five major features:

- **Hierarchical Debates**: Automatically decompose complex topics into sub-debates
- **Real-Time Streaming**: Stream argument generation with async iterators
- **Steering Vectors**: Control agent behavior (formality, aggression, evidence focus)
- **Multimodal Evidence**: Analyze images, charts, and documents as evidence
- **Formal Verification**: Validate argument logic, citations, and causal chains

See the [v2 Examples](examples/streaming.md) and [Changelog](https://github.com/bassrehab/artemis-agents/blob/main/CHANGELOG.md) for details.

## Key Features

### Structured Debates

ARTEMIS implements a rigorous debate structure with:

- **Opening statements** from each agent
- **Multiple argumentation rounds** with rebuttals
- **Evidence-based reasoning** with causal links
- **Jury deliberation** for fair verdicts

### Safety First

Built-in monitors detect problematic AI behavior:

- **Sandbagging Detection**: Identifies when agents deliberately underperform
- **Deception Monitoring**: Catches misleading arguments or manipulation
- **Behavior Tracking**: Monitors for unexpected behavioral drift
- **Ethics Guard**: Ensures debates stay within ethical bounds

### Framework Integrations

Use ARTEMIS with your existing tools:

- **LangChain**: As a structured tool
- **LangGraph**: As a workflow node
- **CrewAI**: As a crew tool
- **MCP**: As a universal server

## Research Foundation

ARTEMIS is based on peer-reviewed research:

> **Adaptive Reasoning and Evaluation Framework for Multi-agent Intelligent Systems in Debate-driven Decision-making**
> Mitra, S. (2025). Technical Disclosure Commons.
> [Read the paper](https://www.tdcommons.org/dpubs_series/7729/)

## Benchmarks

We've run ARTEMIS against AutoGen, CrewAI, and CAMEL across 60 structured debates. See the [benchmark results and analysis](https://github.com/bassrehab/artemis-agents#benchmarks) in the README.

## Get Started

Ready to dive in? Check out the [Installation Guide](getting-started/installation.md) or jump straight to the [Quick Start](getting-started/quickstart.md).

## License

ARTEMIS is released under the Apache License 2.0. See [LICENSE](https://github.com/bassrehab/artemis-agents/blob/main/LICENSE) for details.
