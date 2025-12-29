# Benchmark Experiment Setup

## Overview

Fair 3-way comparison of multi-agent debate frameworks using identical LLM configurations.

## Frameworks Tested

### 1. ARTEMIS (Ours)
- **Version**: Current development branch (`feat/artemis-improvements`)
- **Architecture**: H-L-DAG (Hierarchical Argument Generation)
- **Evaluation Mode**: BALANCED (regex extraction + LLM evaluation)
- **Jury Configuration**:
  - Jurors: 1 (optimized for benchmarks, default is 3)
  - Jury Model: `gpt-4o-mini`
  - Perspectives: analytical
- **Features Used**:
  - H-L-DAG argument levels (strategic/tactical/operational)
  - L-AE-CR adaptive evaluation
  - Jury verdict with confidence scoring
  - Closed-loop feedback between rounds
  - Adaptive level selection based on disagreement
  - Regex-based evidence/causal extraction

### 2. CrewAI
- **Version**: Latest stable (v0.80+)
- **Architecture**: Task-based sequential execution
- **Agent Configuration**:
  - Process: Sequential
  - Delegation: Disabled
  - Verbose: False
- **Model**: Explicitly set to match ARTEMIS (was defaulting to `gpt-4.1-mini`)
- **Features Used**:
  - AssistantAgent with role/goal/backstory
  - Task-based debate rounds
  - No built-in verdict mechanism

### 3. AutoGen
- **Version**: v0.7+ (Microsoft)
- **Architecture**: RoundRobinGroupChat
- **Agent Configuration**:
  - Agent Type: AssistantAgent
  - Termination: MaxMessageTermination
- **Model**: OpenAIChatCompletionClient with explicit model
- **Features Used**:
  - RoundRobinGroupChat for turn-taking
  - System message for role definition
  - No built-in verdict mechanism

## Common Configuration

| Parameter | Value |
|-----------|-------|
| LLM Model | `gpt-4o` |
| Debate Rounds | 2 |
| Trials per Topic | 3 |
| Topics | 3 |
| Total Runs per Framework | 9 |

## Topics Used

1. "Should governments mandate AI safety testing?"
2. "Is remote work better than office work?"
3. "Should social media verify user ages?"

## Evaluation Methodology

### LLM-as-Judge Evaluation
All frameworks evaluated using the same `DebateEvaluator` with `gpt-4o`:

1. **Argument Quality (0-100)**
   - Thesis clarity, evidence citations, structured reasoning
   - Explicit structure (headers, pillars) rewarded

2. **Decision Accuracy (0-100)**
   - Explicit verdict required for score > 55
   - Multi-perspective evaluation, confidence scores valued
   - ARTEMIS advantage: Has jury verdict

3. **Reasoning Depth (0-100)**
   - Causal chains, hierarchical reasoning
   - Synthesis across rounds, counter-argument engagement

### Scoring Rules
- No verdict = Maximum 55 on Decision Accuracy
- Verdict + jury + confidence = 86-100 possible

## Key Differences

| Feature | ARTEMIS | CrewAI | AutoGen |
|---------|---------|--------|---------|
| Verdict Generation | ✅ Jury | ❌ None | ❌ None |
| Structured Arguments | ✅ H-L-DAG | ❌ Prose | ❌ Prose |
| Adaptive Evaluation | ✅ L-AE-CR | ❌ None | ❌ None |
| Causal Graph | ✅ Built | ❌ None | ❌ None |
| Multi-perspective | ✅ Jury Panel | ❌ None | ❌ None |

## File Locations

- Adapters: `benchmarks/adapters/`
- Evaluator: `benchmarks/evaluator.py`
- Results: `benchmarks/results/benchmark_*.json`
- Raw Logs: Captured in JSON output file

## Notes

1. CrewAI was previously using a different default model (`gpt-4.1-mini`), fixed to use explicit `gpt-4o`
2. ARTEMIS uses 1 juror for benchmarks (speed optimization), production default is 3
3. All frameworks use the same system prompt style for fair comparison
