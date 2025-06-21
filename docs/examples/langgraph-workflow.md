# LangGraph Workflow Example

This example demonstrates how to create LangGraph workflows using ARTEMIS debates.

## Simple Debate Workflow

```python
import asyncio
from artemis.integrations.langgraph import (
    ArtemisDebateNode,
    DebateNodeState,
    create_debate_workflow,
)

async def run_simple_workflow():
    # Use the pre-built workflow for simple debates
    workflow = create_debate_workflow(model="gpt-4o")

    # Run with a topic
    result = await workflow.ainvoke({
        "topic": "Should companies mandate return-to-office policies?",
        "rounds": 3,
    })

    print(f"Phase: {result['phase']}")
    print(f"Verdict: {result['verdict']['decision']}")
    print(f"Confidence: {result['verdict']['confidence']:.0%}")
    print(f"Reasoning: {result['verdict']['reasoning']}")

asyncio.run(run_simple_workflow())
```

## Pre-built Workflow

```python
import asyncio
from artemis.integrations.langgraph import create_debate_workflow

async def use_prebuilt_workflow():
    # Create the complete workflow with step-by-step execution
    workflow = create_debate_workflow(
        model="gpt-4o",
        step_by_step=True,  # Enable step-by-step mode for more control
    )

    # Run it
    result = await workflow.ainvoke({
        "topic": "Is blockchain technology overhyped?",
        "rounds": 3,
    })

    print(f"Phase: {result['phase']}")
    print(f"Verdict: {result['verdict']['decision']}")
    print(f"Transcript turns: {len(result.get('transcript', []))}")

asyncio.run(use_prebuilt_workflow())
```

## Custom Workflow with ArtemisDebateNode

```python
import asyncio
from langgraph.graph import StateGraph, END
from artemis.integrations.langgraph import ArtemisDebateNode, DebateNodeState

async def run_custom_workflow():
    # Create the debate node
    node = ArtemisDebateNode(model="gpt-4o")

    # Build workflow
    workflow = StateGraph(DebateNodeState)
    workflow.add_node("debate", node.run_debate)
    workflow.set_entry_point("debate")
    workflow.add_edge("debate", END)

    # Compile
    app = workflow.compile()

    # Run
    result = await app.ainvoke({
        "topic": "Should startups adopt AI-first development?",
        "rounds": 2,
    })

    print(f"Verdict: {result['verdict']['decision']}")
    print(f"Confidence: {result['verdict']['confidence']:.0%}")

asyncio.run(run_custom_workflow())
```

## Multi-Agent Debate Workflow

```python
import asyncio
from artemis.integrations.langgraph import create_debate_workflow

async def run_multi_agent_workflow():
    # Create workflow
    workflow = create_debate_workflow(model="gpt-4o")

    # Run with multiple agents
    result = await workflow.ainvoke({
        "topic": "What is the best approach to microservices architecture?",
        "agents": [
            {
                "name": "monolith_first",
                "role": "Advocate for starting with monolith",
                "position": "argues for monolith-first approach",
            },
            {
                "name": "microservices_now",
                "role": "Advocate for immediate microservices",
                "position": "argues for starting with microservices",
            },
            {
                "name": "pragmatist",
                "role": "Pragmatic architect",
                "position": "argues for hybrid approach based on context",
            },
        ],
        "rounds": 2,
    })

    print(f"Verdict: {result['verdict']['decision']}")
    print(f"Confidence: {result['verdict']['confidence']:.0%}")

    # Show scores
    if result.get("scores"):
        print("\nAgent Scores:")
        for agent, score in result["scores"].items():
            print(f"  {agent}: {score:.2f}")

asyncio.run(run_multi_agent_workflow())
```

## Research -> Debate -> Synthesize

```python
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, END
from artemis.integrations.langgraph import ArtemisDebateNode

class ResearchDebateState(TypedDict, total=False):
    topic: str
    research: str
    # DebateNodeState fields
    agents: list
    positions: dict
    rounds: int
    phase: str
    transcript: list
    verdict: dict
    scores: dict
    metadata: dict
    synthesis: str

def research_topic(state: ResearchDebateState) -> ResearchDebateState:
    # Simulate research (in practice, call search APIs)
    research = f"""
    Key findings on '{state['topic']}':
    1. Industry trends show increasing adoption
    2. Recent studies indicate mixed results
    3. Expert opinions are divided
    4. Cost-benefit analysis varies by context
    """
    return {"research": research}

def synthesize_results(state: ResearchDebateState) -> ResearchDebateState:
    verdict = state.get("verdict", {})
    synthesis = f"""
    ANALYSIS SUMMARY
    ================
    Topic: {state['topic']}

    Research Findings:
    {state.get('research', 'N/A')}

    Debate Verdict: {verdict.get('decision', 'N/A')}
    Confidence: {verdict.get('confidence', 0):.0%}

    Final Recommendation:
    {verdict.get('reasoning', 'N/A')}
    """
    return {"synthesis": synthesis}

async def run_research_debate_workflow():
    node = ArtemisDebateNode(model="gpt-4o")

    workflow = StateGraph(ResearchDebateState)

    # Add nodes
    workflow.add_node("research", research_topic)
    workflow.add_node("debate", node.run_debate)
    workflow.add_node("synthesize", synthesize_results)

    # Add edges
    workflow.set_entry_point("research")
    workflow.add_edge("research", "debate")
    workflow.add_edge("debate", "synthesize")
    workflow.add_edge("synthesize", END)

    # Compile and run
    app = workflow.compile()

    result = await app.ainvoke({
        "topic": "Should our startup adopt AI-first development?",
        "rounds": 2,
    })

    print(result["synthesis"])

asyncio.run(run_research_debate_workflow())
```

## Conditional Routing

```python
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, END
from artemis.integrations.langgraph import ArtemisDebateNode

class RoutedState(TypedDict, total=False):
    topic: str
    complexity: float
    result: str
    # DebateNodeState fields
    phase: str
    verdict: dict
    transcript: list
    rounds: int

def analyze_complexity(state: RoutedState) -> RoutedState:
    # Analyze topic complexity
    complex_keywords = ["ethical", "philosophical", "controversial", "nuanced"]
    topic_lower = state["topic"].lower()

    complexity = sum(1 for kw in complex_keywords if kw in topic_lower) / len(complex_keywords)
    return {"complexity": complexity}

def quick_answer(state: RoutedState) -> RoutedState:
    return {"result": f"Quick analysis: {state['topic']} - Generally accepted view..."}

def format_debate_result(state: RoutedState) -> RoutedState:
    verdict = state.get("verdict")
    if verdict:
        return {"result": f"Debate verdict: {verdict['decision']} ({verdict['confidence']:.0%})"}
    return state

def should_debate(state: RoutedState) -> str:
    if state["complexity"] > 0.3:
        return "debate"
    return "quick_answer"

async def run_conditional_workflow():
    node = ArtemisDebateNode(model="gpt-4o")

    workflow = StateGraph(RoutedState)

    # Add nodes
    workflow.add_node("analyze", analyze_complexity)
    workflow.add_node("debate", node.run_debate)
    workflow.add_node("quick_answer", quick_answer)
    workflow.add_node("format", format_debate_result)

    # Add edges
    workflow.set_entry_point("analyze")
    workflow.add_conditional_edges("analyze", should_debate)
    workflow.add_edge("debate", "format")
    workflow.add_edge("quick_answer", END)
    workflow.add_edge("format", END)

    app = workflow.compile()

    # Test with simple topic
    simple_result = await app.ainvoke({
        "topic": "Should we use tabs or spaces?",
        "rounds": 2,
    })
    print(f"Simple topic: {simple_result['result']}")

    # Test with complex topic
    complex_result = await app.ainvoke({
        "topic": "What are the ethical implications of AI in healthcare?",
        "rounds": 2,
    })
    print(f"Complex topic: {complex_result['result']}")

asyncio.run(run_conditional_workflow())
```

## Step-by-Step Execution

```python
import asyncio
from artemis.integrations.langgraph import ArtemisDebateNode, DebateNodeState
from langgraph.graph import StateGraph, END

async def run_stepwise_workflow():
    node = ArtemisDebateNode(model="gpt-4o")

    workflow = StateGraph(DebateNodeState)

    # Add step-by-step nodes
    workflow.add_node("setup", node.setup)
    workflow.add_node("run_round", node.run_round)
    workflow.add_node("finalize", node.finalize)

    # Add edges
    workflow.set_entry_point("setup")
    workflow.add_edge("setup", "run_round")
    workflow.add_conditional_edges(
        "run_round",
        node.get_routing_function(),
        {
            "continue": "run_round",
            "finalize": "finalize",
        },
    )
    workflow.add_edge("finalize", END)

    app = workflow.compile()

    # Run
    result = await app.ainvoke({
        "topic": "Is open source software sustainable?",
        "rounds": 2,
    })

    print(f"Phase: {result['phase']}")
    print(f"Rounds completed: {result['current_round']}")
    print(f"Verdict: {result['verdict']['decision']}")

asyncio.run(run_stepwise_workflow())
```

## Human-in-the-Loop

```python
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from artemis.integrations.langgraph import ArtemisDebateNode

class HITLState(TypedDict, total=False):
    topic: str
    verdict: dict
    needs_review: bool
    human_decision: str
    final_result: str
    # DebateNodeState fields
    phase: str
    transcript: list
    rounds: int

def check_confidence(state: HITLState) -> str:
    verdict = state.get("verdict", {})
    confidence = verdict.get("confidence", 0)

    if confidence < 0.6:
        return "human_review"
    return "finalize"

def request_review(state: HITLState) -> HITLState:
    return {"needs_review": True}

def finalize(state: HITLState) -> HITLState:
    verdict = state.get("verdict", {})
    human_decision = state.get("human_decision")

    if human_decision:
        result = f"Human decided: {human_decision}"
    else:
        result = f"AI verdict: {verdict.get('decision')} ({verdict.get('confidence', 0):.0%})"

    return {"final_result": result, "needs_review": False}

async def run_hitl_workflow():
    node = ArtemisDebateNode(model="gpt-4o")

    workflow = StateGraph(HITLState)

    workflow.add_node("debate", node.run_debate)
    workflow.add_node("human_review", request_review)
    workflow.add_node("finalize", finalize)

    workflow.set_entry_point("debate")
    workflow.add_conditional_edges("debate", check_confidence)
    workflow.add_edge("human_review", END)  # Pause for human input
    workflow.add_edge("finalize", END)

    # Add checkpointing
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    # Run with a controversial topic (likely low confidence)
    config = {"configurable": {"thread_id": "review-1"}}

    result = await app.ainvoke(
        {"topic": "Should AI be granted legal personhood?", "rounds": 2},
        config=config,
    )

    if result.get("needs_review"):
        print("Human review requested!")
        print(f"AI verdict: {result['verdict']}")

        # Simulate human decision
        human_input = "Requires more research before decision"

        # Resume with human input
        final_result = await app.ainvoke(
            {"human_decision": human_input},
            config=config,
        )

        print(f"Final result: {final_result['final_result']}")
    else:
        print(f"Final result: {result['final_result']}")

asyncio.run(run_hitl_workflow())
```

## With Safety Monitors

```python
import asyncio
from artemis.core.agent import Agent
from artemis.integrations.langgraph import (
    ArtemisDebateNode,
    DebateNodeState,
    DebateNodeConfig,
)
from artemis.safety import MonitorMode, SandbagDetector, DeceptionMonitor
from langgraph.graph import StateGraph, END

async def run_with_safety():
    # Create safety monitors
    sandbagging = SandbagDetector(mode=MonitorMode.PASSIVE, sensitivity=0.6)
    deception = DeceptionMonitor(mode=MonitorMode.PASSIVE, sensitivity=0.6)

    # Create pre-configured agents
    agents = [
        Agent(name="pro", role="Advocate for the proposition", model="gpt-4o"),
        Agent(name="con", role="Advocate against the proposition", model="gpt-4o"),
    ]

    # Create node with safety config
    config = DebateNodeConfig(
        model="gpt-4o",
        default_rounds=3,
        enable_safety=True,
    )

    node = ArtemisDebateNode(
        model="gpt-4o",
        agents=agents,
        config=config,
    )

    workflow = StateGraph(DebateNodeState)
    workflow.add_node("debate", node.run_debate)
    workflow.set_entry_point("debate")
    workflow.add_edge("debate", END)

    app = workflow.compile()

    result = await app.ainvoke({
        "topic": "Should we adopt AI-powered code review?",
        "positions": {
            "pro": "supports AI code review adoption",
            "con": "opposes AI code review due to limitations",
        },
    })

    print(f"Verdict: {result['verdict']['decision']}")
    print(f"Confidence: {result['verdict']['confidence']:.0%}")

asyncio.run(run_with_safety())
```

## Next Steps

- See [Basic Debate](basic-debate.md) for fundamentals
- Add [Safety Monitors](safety-monitors.md) to workflows
- Learn about [CrewAI Integration](crewai-workflow.md)
