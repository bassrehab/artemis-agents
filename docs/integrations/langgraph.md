# LangGraph Integration

ARTEMIS provides deep integration with LangGraph, allowing you to incorporate structured debates into complex agent workflows.

## Installation

```bash
pip install artemis-agents[langgraph]
```

## ArtemisDebateNode

The primary integration is the `ArtemisDebateNode`:

```python
from artemis.integrations import ArtemisDebateNode

# Create a debate node
debate_node = ArtemisDebateNode(
    model="gpt-4o",
    rounds=3,
)
```

## Basic Workflow

### Simple Debate Workflow

```python
from langgraph.graph import StateGraph, START, END
from artemis.integrations import ArtemisDebateNode, DebateState

# Define the workflow
workflow = StateGraph(DebateState)

# Add nodes
workflow.add_node("debate", ArtemisDebateNode(model="gpt-4o"))

# Add edges
workflow.add_edge(START, "debate")
workflow.add_edge("debate", END)

# Compile
app = workflow.compile()

# Run
result = await app.ainvoke({
    "topic": "Should we use GraphQL or REST?",
})

print(result["verdict"])
```

## Debate State

The `DebateNodeState` tracks debate progress:

```python
from artemis.integrations import DebateNodeState

# State structure
class DebateNodeState(TypedDict, total=False):
    topic: str                       # Debate topic
    agents: list[AgentStateConfig]   # Agent configurations
    positions: dict[str, str]        # Agent positions
    rounds: int                      # Number of rounds
    current_round: int               # Current round
    phase: str                       # Current debate phase
    transcript: list[dict]           # Debate transcript
    verdict: dict | None             # Final verdict
    scores: dict[str, float]         # Agent scores
    metadata: dict                   # Additional metadata
```

## Pre-built Workflows

### create_debate_workflow

A complete debate workflow:

```python
from artemis.integrations import create_debate_workflow

# Create the workflow (single-node by default)
workflow = create_debate_workflow(model="gpt-4o")

# Run
result = await workflow.ainvoke({
    "topic": "Is functional programming better than OOP?",
    "rounds": 3,
})

print(f"Verdict: {result['verdict']['decision']}")
```

### Step-by-Step Workflow

For more control, use step-by-step mode:

```python
from artemis.integrations import create_debate_workflow

# Create step-by-step workflow
workflow = create_debate_workflow(
    model="gpt-4o",
    step_by_step=True,
)

# Run with multi-agent support
result = await workflow.ainvoke({
    "topic": "Should we migrate to cloud?",
    "agents": [
        {"name": "tech_lead", "role": "Technical Lead", "position": "supports migration"},
        {"name": "finance", "role": "Finance Director", "position": "concerned about costs"},
    ],
    "rounds": 3,
})

print(f"Decision: {result['verdict']['decision']}")
print(f"Confidence: {result['verdict']['confidence']}")
```

## Custom Workflows

### With Pre-processing

```python
from langgraph.graph import StateGraph, START, END
from artemis.integrations import ArtemisDebateNode

class WorkflowState(TypedDict):
    input: str
    processed_topic: str
    debate_result: dict

def preprocess(state: WorkflowState) -> WorkflowState:
    # Extract debate topic from input
    topic = extract_topic(state["input"])
    return {"processed_topic": topic}

def postprocess(state: WorkflowState) -> WorkflowState:
    # Format the result
    verdict = state["debate_result"]["verdict"]
    return {"output": f"Analysis: {verdict.reasoning}"}

# Build workflow
workflow = StateGraph(WorkflowState)
workflow.add_node("preprocess", preprocess)
workflow.add_node("debate", ArtemisDebateNode(model="gpt-4o"))
workflow.add_node("postprocess", postprocess)

workflow.add_edge(START, "preprocess")
workflow.add_edge("preprocess", "debate")
workflow.add_edge("debate", "postprocess")
workflow.add_edge("postprocess", END)

app = workflow.compile()
```

### With Conditional Routing

```python
from langgraph.graph import StateGraph, START, END

def should_debate(state: WorkflowState) -> str:
    # Only debate complex topics
    if state["complexity"] > 0.7:
        return "debate"
    return "quick_answer"

workflow = StateGraph(WorkflowState)
workflow.add_node("analyze", analyze_complexity)
workflow.add_node("debate", ArtemisDebateNode(model="gpt-4o"))
workflow.add_node("quick_answer", quick_response)

workflow.add_edge(START, "analyze")
workflow.add_conditional_edges("analyze", should_debate)
workflow.add_edge("debate", END)
workflow.add_edge("quick_answer", END)
```

### With Human-in-the-Loop

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint import MemorySaver

def review_needed(state: WorkflowState) -> str:
    # Route to human review if controversial
    if state["verdict"].confidence < 0.6:
        return "human_review"
    return "finalize"

workflow = StateGraph(WorkflowState)
workflow.add_node("debate", ArtemisDebateNode(model="gpt-4o"))
workflow.add_node("human_review", human_review_node)
workflow.add_node("finalize", finalize_result)

workflow.add_edge(START, "debate")
workflow.add_conditional_edges("debate", review_needed)
workflow.add_edge("human_review", "finalize")
workflow.add_edge("finalize", END)

# Add checkpointing for interruption
app = workflow.compile(checkpointer=MemorySaver())

# Run with possible interrupt
result = await app.ainvoke(
    {"topic": "Controversial topic"},
    config={"configurable": {"thread_id": "1"}},
)

# If interrupted, resume after human review
if result.get("needs_review"):
    result = await app.ainvoke(
        {"human_decision": "approved"},
        config={"configurable": {"thread_id": "1"}},
    )
```

## Multi-Agent Patterns

### Parallel Debates

```python
from langgraph.graph import StateGraph, START, END

def split_topics(state):
    # Split into multiple topics
    return {
        "topics": [
            "Technical feasibility",
            "Cost analysis",
            "Risk assessment",
        ]
    }

def merge_results(state):
    # Combine debate results
    results = state["debate_results"]
    overall = synthesize(results)
    return {"final_analysis": overall}

# Create parallel branches
workflow = StateGraph(WorkflowState)
workflow.add_node("split", split_topics)

for i in range(3):
    workflow.add_node(f"debate_{i}", ArtemisDebateNode(model="gpt-4o"))

workflow.add_node("merge", merge_results)

workflow.add_edge(START, "split")
for i in range(3):
    workflow.add_edge("split", f"debate_{i}")
    workflow.add_edge(f"debate_{i}", "merge")
workflow.add_edge("merge", END)
```

### Sequential Analysis

```python
def extract_key_points(state):
    # Extract points from debate for next stage
    points = extract(state["debate_result"])
    return {"key_points": points}

workflow = StateGraph(WorkflowState)
workflow.add_node("initial_debate", ArtemisDebateNode(model="gpt-4o"))
workflow.add_node("extract", extract_key_points)
workflow.add_node("deep_dive_debate", ArtemisDebateNode(model="gpt-4o"))

workflow.add_edge(START, "initial_debate")
workflow.add_edge("initial_debate", "extract")
workflow.add_edge("extract", "deep_dive_debate")
workflow.add_edge("deep_dive_debate", END)
```

## State Persistence

### Saving Debate State

```python
from langgraph.checkpoint import SqliteSaver

# Use SQLite for persistence
checkpointer = SqliteSaver.from_conn_string("debates.db")

workflow = create_debate_workflow(model="gpt-4o")
app = workflow.compile(checkpointer=checkpointer)

# Run with thread ID for persistence
result = await app.ainvoke(
    {"topic": "Your topic"},
    config={"configurable": {"thread_id": "debate-123"}},
)

# Later: resume or replay
history = app.get_state_history(
    config={"configurable": {"thread_id": "debate-123"}}
)
```

## Integration with Safety

Safety monitoring is configured via the `DebateConfig`:

```python
from artemis.integrations import ArtemisDebateNode
from artemis.core.types import DebateConfig

# Create node with safety monitoring enabled
config = DebateConfig(
    safety_mode="active",
    halt_on_safety_violation=True,
)

debate_node = ArtemisDebateNode(
    model="gpt-4o",
    debate_config=config,
)

def check_safety(state):
    # Check for safety alerts in the debate result
    if state.get("phase") == "error":
        return "halt"
    return "continue"

workflow.add_conditional_edges("debate", check_safety)
```

## Streaming

Stream debate progress:

```python
workflow = create_debate_workflow(model="gpt-4o")
app = workflow.compile()

async for event in app.astream({"topic": "Your topic"}):
    if "debate" in event:
        turn = event["debate"]["current_turn"]
        print(f"Turn: {turn.agent} - {turn.argument.content[:100]}...")
```

## Error Handling

```python
from langgraph.graph import StateGraph

def handle_debate_error(state):
    if state.get("error"):
        # Log error and provide fallback
        return {"verdict": {"decision": "inconclusive", "reasoning": str(state["error"])}}
    return state

workflow.add_node("error_handler", handle_debate_error)
workflow.add_edge("debate", "error_handler")
```

## Next Steps

- Learn about [LangChain Integration](langchain.md)
- Explore [CrewAI Integration](crewai.md)
- Configure [MCP Server](mcp.md)
