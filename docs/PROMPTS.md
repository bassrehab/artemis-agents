# Claude Code Prompts for ARTEMIS Agents

This document contains ready-to-use prompts for Claude Code sessions organized by development phase.

---

## 🚀 Session Initialization Prompt

Use this at the start of every Claude Code session:

```
I'm building artemis-agents, a production-grade multi-agent debate framework.

Key context:
- Read CLAUDE.md for full project understanding
- This implements the ARTEMIS paper from Google TD Commons
- Core concepts: H-L-DAG (hierarchical arguments), L-AE-CR (adaptive evaluation), Jury scoring
- Unique value: Safety monitoring (sandbagging/deception detection)

Today's focus: [INSERT TASK]
```

---

## Phase 1: Foundation

### Prompt 1.1: Project Setup

```
Let's set up the artemis-agents project foundation.

Create:
1. pyproject.toml with:
   - Python 3.10+ requirement
   - Dependencies: pydantic>=2.0, httpx, openai>=1.0, anthropic
   - Dev dependencies: pytest, pytest-asyncio, pytest-cov, mypy, ruff
   - Package metadata (name: artemis-agents, author: Subhadip Mitra)

2. Directory structure as shown in CLAUDE.md

3. Basic configuration files:
   - ruff.toml (line-length 100, Python 3.10 target)
   - mypy.ini (strict mode)
   - .gitignore (Python template)

Follow the structure in CLAUDE.md exactly.
```

### Prompt 1.2: Core Data Models

```
Create the core data models for ARTEMIS in artemis/core/types.py.

Implement these Pydantic v2 models:

1. ArgumentLevel (Enum): strategic, tactical, operational
2. Evidence: source, content, credibility_score
3. CausalLink: premise, conclusion, strength, validity
4. Argument: id, agent, level, content, evidence[], causal_links[], ethical_score, thinking_trace, timestamp
5. Turn: round, sequence, agent, argument, evaluation, safety_results[]
6. Message: role (system/user/assistant), content
7. DebateState (Enum): setup, opening, debate, closing, deliberation, complete

Use:
- Field(default_factory=...) for mutable defaults
- uuid4 for IDs
- datetime.utcnow for timestamps
- Proper type hints with Python 3.10+ syntax (| instead of Union)
```

### Prompt 1.3: Model Provider Interface

```
Create the abstract model provider interface in artemis/models/base.py.

Requirements:
1. ABC class BaseModel with:
   - async generate(messages: list[Message], **kwargs) -> Response
   - async generate_with_reasoning(messages, thinking_budget) -> ReasoningResponse
   - model_name property
   - supports_reasoning property

2. Response model: content, usage (prompt_tokens, completion_tokens), raw_response
3. ReasoningResponse: extends Response with thinking field
4. Usage model: prompt_tokens, completion_tokens, total_tokens

Then implement OpenAI provider in artemis/models/openai.py:
- Support for gpt-4o, gpt-4-turbo, o1-preview, o1-mini
- Handle reasoning models (o1) differently - they don't support system prompts
- Use openai async client
- Implement retry logic with exponential backoff
```

---

## Phase 2: Core ARTEMIS

### Prompt 2.1: H-L-DAG Agent

```
Implement the Agent class with H-L-DAG argument generation.

File: artemis/core/agent.py

The Agent should:
1. Accept name, role, model (str or BaseModel), reasoning config, persona
2. Have a method generate_argument(context: DebateContext, level: ArgumentLevel)
3. Build H-L-DAG prompts based on argument level (see HDAG_LEVEL_INSTRUCTIONS in DESIGN.md)
4. Support reasoning models with extended thinking
5. Extract evidence and causal links from generated arguments

Key implementation details:
- Use different prompt structures for each H-L-DAG level
- Strategic: thesis + pillars + framework
- Tactical: evidence + causal connections + counterarguments  
- Operational: specific facts + case studies + actionable implications

Include the DebateContext model with: topic, current_round, total_rounds, transcript, topic_sensitivity, topic_complexity
```

### Prompt 2.2: L-AE-CR Evaluator

```
Implement the L-AE-CR adaptive evaluator in artemis/core/evaluation.py.

Requirements:
1. AdaptiveEvaluator class with:
   - DEFAULT_CRITERIA dict (logical_coherence, evidence_quality, causal_reasoning, ethical_alignment, persuasiveness)
   - evaluate_argument(argument, context) -> ArgumentEvaluation
   - _adapt_criteria(context) -> adapted weights
   - _score_criterion(argument, criterion, context) -> float

2. Adaptation logic:
   - Increase ethical_alignment weight when topic_sensitivity > 0.7
   - Increase evidence_quality weight in later rounds
   - Increase causal_reasoning weight when topic_complexity > 0.7
   - Always renormalize weights to sum to 1.0

3. CausalGraph class for tracking argument relationships

4. ArgumentEvaluation model: argument_id, scores dict, weights dict, causal_score, total_score

Use LLM to score each criterion (create appropriate prompts).
```

### Prompt 2.3: Jury Panel

```
Implement the Jury scoring mechanism in artemis/core/jury.py.

Components:

1. JuryMember class:
   - id, model, perspective (analytical/ethical/practical/adversarial/synthesizing)
   - evaluate(transcript) -> JurorEvaluation
   - Use perspective-specific prompts

2. JuryPanel class:
   - evaluators: list[JuryMember]
   - criteria: list[str]
   - consensus_threshold: float
   - deliberate(transcript) -> Verdict

3. Deliberation logic:
   - Each juror evaluates independently (parallel with asyncio.gather)
   - Build consensus from evaluations
   - Calculate confidence score
   - Collect dissenting opinions

4. Models:
   - JurorEvaluation: juror_id, scores, reasoning, verdict_preference
   - Verdict: decision, confidence, reasoning, dissenting_opinions, score_breakdown
   - DissentingOpinion: juror_id, opinion, reasoning
```

### Prompt 2.4: Debate Orchestrator

```
Implement the main Debate orchestrator in artemis/core/debate.py.

Requirements:

1. Debate class with:
   - __init__(topic, agents, jury, rounds, monitors, config)
   - async run() -> DebateResult
   - State machine: SETUP → OPENING → DEBATE → CLOSING → DELIBERATION → COMPLETE
   
2. Methods:
   - _run_opening_statements(): Each agent presents strategic-level opening
   - _run_round(round_num): Agents take turns with tactical arguments
   - _run_closing_statements(): Each agent presents closing argument
   - _transition_state(new_state): State machine transitions

3. Turn management:
   - Track all turns in transcript
   - Run safety monitors after each turn
   - Run evaluation after each argument

4. DebateResult model:
   - debate_id, topic, verdict, transcript, safety_alerts, metadata

5. DebateConfig model (see DESIGN.md for fields)

Make everything async. Handle timeouts with asyncio.wait_for.
```

---

## Phase 3: Safety Layer

### Prompt 3.1: Safety Monitor Interface

```
Create the safety monitoring infrastructure.

Files:
- artemis/safety/base.py
- artemis/safety/types.py

Requirements:

1. SafetyMonitor ABC:
   - mode: "passive" or "active"
   - alert_threshold, halt_threshold
   - abstract async analyze(turn, context) -> SafetyResult
   - should_halt(result) -> bool

2. Types:
   - SafetyResult: monitor name, severity (0-1), indicators list, should_alert bool
   - SafetyAlert: monitor, agent, type, severity, indicators, timestamp
   - SafetyIndicator: type, severity, evidence

3. Integration hook in Debate._run_turn():
   - After each turn, run all monitors
   - Collect alerts
   - If active mode and should_halt, raise SafetyHaltError

Make monitors pluggable and composable.
```

### Prompt 3.2: Sandbagging Detector

```
Implement sandbagging detection in artemis/safety/sandbagging.py.

SandbagDetector should detect when agents deliberately underperform.

Detection methods:

1. Capability drop detection:
   - Track baseline quality for each agent (rolling window)
   - Flag when current quality drops significantly below baseline
   - Use LLM to assess argument quality (0-1 score)

2. Strategic timing detection:
   - Check if drops occur at strategically significant moments
   - e.g., when agent is "winning" or at critical decision points

3. Selective engagement detection:
   - Detect when agent avoids specific topics despite demonstrated capability
   - Compare engagement breadth across rounds

Implementation:
- AgentBaseline dataclass: average_quality, std_dev, sample_count, is_established
- Update baseline after each turn
- Establish baseline after baseline_window turns

Refer to ai-metacognition-toolkit patterns for inspiration.
```

### Prompt 3.3: Deception Monitor

```
Implement deception monitoring in artemis/safety/deception.py.

DeceptionMonitor should detect misleading arguments and manipulation.

Detection methods:

1. Factual consistency:
   - Check if agent contradicts their own previous statements
   - Use LLM to identify contradictions

2. Logical fallacy detection:
   - Identify common fallacies: ad hominem, straw man, false dichotomy, appeal to authority, etc.
   - Score severity based on fallacy type

3. Emotional manipulation:
   - Detect excessive emotional appeals
   - Flag fear-mongering, guilt-tripping

4. Citation fabrication:
   - Check if cited sources exist/are plausible
   - Flag if citations seem fabricated

Return DeceptionIndicator for each detected issue with type, severity, and evidence.
```

---

## Phase 4: Framework Integrations

### Prompt 4.1: LangChain/LangGraph

```
Create framework integrations for LangChain and LangGraph.

File: artemis/integrations/langchain.py

ArtemisDebateTool (LangChain Tool):
- name = "structured_debate"
- description = "Run a structured multi-agent debate on a topic"
- Input schema: topic, perspectives (optional), rounds
- _run() and _arun() methods
- Returns verdict summary

File: artemis/integrations/langgraph.py

ArtemisDebateNode (LangGraph node):
- Callable that takes State and returns State
- Extracts topic from state
- Creates agents based on perspectives in state
- Runs debate
- Updates state with verdict, confidence, reasoning

Example usage patterns in docstrings.
```

### Prompt 4.2: Reasoning Model Support

```
Add comprehensive reasoning model support.

File: artemis/models/reasoning.py

1. ReasoningConfig model:
   - enabled: bool
   - thinking_budget: int (default 8000)
   - strategy: "think-then-argue" | "interleaved" | "final-reflection"

2. Reasoning strategies:
   - think-then-argue: Full reasoning, then generate argument
   - interleaved: Reason between argument sections
   - final-reflection: Generate argument, then reflect and revise

File: artemis/models/deepseek.py

DeepSeekModel provider:
- Support for deepseek-r1, deepseek-chat
- Handle extended thinking tokens separately
- Parse thinking trace from response

Update Agent class to use reasoning config when generating arguments.
```

---

## Phase 5: MCP Server

### Prompt 5.1: MCP Server Implementation

```
Implement MCP server for ARTEMIS.

File: artemis/mcp/server.py

Requirements:

1. Tool definitions:
   - artemis_debate: Start new debate
     - Input: topic (required), perspectives (optional), rounds (optional)
     - Output: debate_id, initial status
   
   - artemis_add_perspective: Add agent to debate
     - Input: debate_id, name, role
     
   - artemis_get_verdict: Get final verdict
     - Input: debate_id
     - Output: verdict, confidence, reasoning
     
   - artemis_get_transcript: Get full transcript
     - Input: debate_id
     - Output: list of turns

2. Session management:
   - Store active debates in memory (dict)
   - Debate timeout after 30 minutes of inactivity

3. Server startup:
   - CLI: python -m artemis.mcp.server --port 8080
   - Use fastmcp or implement MCP protocol directly

Follow MCP specification for request/response format.
```

---

## Phase 6: Polish

### Prompt 6.1: Benchmarks

```
Create benchmark comparisons with other frameworks.

File: benchmarks/vs_autogen.py
File: benchmarks/vs_crewai.py
File: benchmarks/debate_quality.py

Benchmarks to run:

1. Debate quality metrics:
   - Argument coherence (LLM-scored)
   - Evidence usage
   - Logical validity
   - Decision accuracy (on known-answer debates)

2. Performance metrics:
   - Time to verdict
   - Token usage
   - API calls

3. Test scenarios:
   - Ethical dilemma debates
   - Technical decision debates
   - Risk assessment debates

Create a benchmark runner that outputs markdown tables for README.

Note: For fair comparison, use same underlying model (gpt-4o) across all frameworks.
```

### Prompt 6.2: Documentation

```
Complete all documentation.

1. docs/API.md - Full API reference
   - All public classes and methods
   - Parameters and return types
   - Example code for each

2. Update README.md:
   - Ensure all examples work
   - Add benchmark results table
   - Verify installation instructions

3. Docstrings audit:
   - Every public class/method should have Google-style docstring
   - Include Args, Returns, Raises, Example sections

4. examples/ directory:
   - basic_debate.py - Minimal working example
   - ethical_dilemma.py - Ethics-focused debate
   - enterprise_decision.py - Business decision scenario
   - with_safety_monitors.py - Safety monitoring demo
   - langgraph_workflow.py - LangGraph integration
   - reasoning_models.py - Using o1/R1
```

---

## 🔧 Utility Prompts

### Debug Prompt

```
I'm getting an error in artemis. Here's the traceback:

[PASTE TRACEBACK]

The relevant code is in [FILE]. Help me debug this.
```

### Test Writing Prompt

```
Write comprehensive tests for [MODULE].

Requirements:
- Use pytest and pytest-asyncio
- Mock LLM calls (don't make real API calls in tests)
- Test happy path and edge cases
- Aim for >80% coverage
- Use fixtures for common setup

File: tests/unit/[path]/test_[module].py
```

### Refactor Prompt

```
Review [FILE] and suggest improvements for:
- Code clarity and readability
- Performance optimizations
- Better error handling
- More Pythonic patterns

Keep backward compatibility with existing API.
```

### PR Review Prompt

```
Review these changes as if for a PR:

[PASTE DIFF OR DESCRIBE CHANGES]

Check for:
- Correctness
- Test coverage
- Documentation
- Type safety
- Edge cases
```

---

## 📝 Commit Message Templates

```bash
# Feature
feat(core): implement H-L-DAG argument generation

# Bug fix
fix(models): handle o1 reasoning token limits correctly

# Documentation
docs: add comprehensive API reference

# Tests
test(safety): add sandbagging detection tests with mocked agents

# Refactor
refactor(evaluation): simplify criteria adaptation logic

# Chore
chore: update dependencies and fix type errors
```

---

## 🚨 Common Issues & Solutions

### Issue: Async test failures
```
Use @pytest.mark.asyncio decorator and pytest-asyncio plugin.
Ensure event loop is properly managed.
```

### Issue: Type errors with Pydantic
```
Use model_validator instead of validator for Pydantic v2.
Use field_validator for field-level validation.
```

### Issue: LLM mock not working
```
Mock at the right level: mock the model.generate method, not httpx.
Use AsyncMock for async methods.
```
