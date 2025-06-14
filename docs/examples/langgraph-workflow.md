# LangGraph Workflow Example

This example demonstrates how to create LangGraph workflows using ARTEMIS debates.

## Simple Debate Workflow

```python
import asyncio
from langgraph.graph import StateGraph, START, END
from artemis.integrations import ArtemisDebateNode, DebateState

async def run_simple_workflow():
    # Create the workflow
    workflow = StateGraph(DebateState)

    # Add the debate node
    workflow.add_node("debate", ArtemisDebateNode(model="gpt-4o", rounds=3))

    # Add edges
    workflow.add_edge(START, "debate")
    workflow.add_edge("debate", END)

    # Compile
    app = workflow.compile()

    # Run
    result = await app.ainvoke({
        "topic": "Should companies mandate return-to-office policies?",
    })

    print(f"Verdict: {result['verdict'].decision}")
    print(f"Confidence: {result['verdict'].confidence:.0%}")
    print(f"Reasoning: {result['verdict'].reasoning}")

asyncio.run(run_simple_workflow())
```

## Pre-built Workflow

```python
import asyncio
from artemis.integrations import create_debate_workflow

async def use_prebuilt_workflow():
    # Create the complete workflow
    workflow = create_debate_workflow(
        model="gpt-4o",
        rounds=3,
        enable_safety=True,
    )

    # Run it
    result = await workflow.ainvoke({
        "topic": "Is blockchain technology overhyped?",
    })

    print(f"Verdict: {result['verdict'].decision}")

    # Check safety
    if result.get("safety_alerts"):
        print("\nSafety Alerts:")
        for alert in result["safety_alerts"]:
            print(f"  - {alert.type}: {alert.details}")

asyncio.run(use_prebuilt_workflow())
```

## Decision Workflow

```python
import asyncio
from artemis.integrations import create_decision_workflow

async def run_decision_workflow():
    # Create decision-making workflow
    workflow = create_decision_workflow(
        model="gpt-4o",
        decision_threshold=0.7,
    )

    result = await workflow.ainvoke({
        "question": "Should we migrate our monolith to microservices?",
        "context": """
        Our current monolith has:
        - 500k lines of code
        - 50 developers
        - 10 million daily users
        - Increasing deployment friction
        """,
    })

    print(f"Decision: {result['decision']}")
    print(f"Confidence: {result['confidence']:.0%}")
    print(f"Reasoning: {result['reasoning']}")

asyncio.run(run_decision_workflow())
```

## Research → Debate → Synthesize

```python
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from artemis.integrations import ArtemisDebateNode

class ResearchDebateState(TypedDict):
    topic: str
    research: str
    debate_result: dict
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
    verdict = state["debate_result"]["verdict"]
    synthesis = f"""
    ANALYSIS SUMMARY
    ================
    Topic: {state['topic']}

    Research Findings:
    {state['research']}

    Debate Verdict: {verdict.decision}
    Confidence: {verdict.confidence:.0%}

    Final Recommendation:
    {verdict.reasoning}
    """
    return {"synthesis": synthesis}

async def run_research_debate_workflow():
    workflow = StateGraph(ResearchDebateState)

    # Add nodes
    workflow.add_node("research", research_topic)
    workflow.add_node("debate", ArtemisDebateNode(model="gpt-4o"))
    workflow.add_node("synthesize", synthesize_results)

    # Add edges
    workflow.add_edge(START, "research")
    workflow.add_edge("research", "debate")
    workflow.add_edge("debate", "synthesize")
    workflow.add_edge("synthesize", END)

    # Compile and run
    app = workflow.compile()

    result = await app.ainvoke({
        "topic": "Should our startup adopt AI-first development?",
    })

    print(result["synthesis"])

asyncio.run(run_research_debate_workflow())
```

## Conditional Routing

```python
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from artemis.integrations import ArtemisDebateNode

class RoutedState(TypedDict):
    topic: str
    complexity: float
    result: str

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
        return {"result": f"Debate verdict: {verdict.decision} ({verdict.confidence:.0%})"}
    return state

def should_debate(state: RoutedState) -> str:
    if state["complexity"] > 0.3:
        return "debate"
    return "quick_answer"

async def run_conditional_workflow():
    workflow = StateGraph(RoutedState)

    # Add nodes
    workflow.add_node("analyze", analyze_complexity)
    workflow.add_node("debate", ArtemisDebateNode(model="gpt-4o"))
    workflow.add_node("quick_answer", quick_answer)
    workflow.add_node("format", format_debate_result)

    # Add edges
    workflow.add_edge(START, "analyze")
    workflow.add_conditional_edges("analyze", should_debate)
    workflow.add_edge("debate", "format")
    workflow.add_edge("quick_answer", END)
    workflow.add_edge("format", END)

    app = workflow.compile()

    # Test with simple topic
    simple_result = await app.ainvoke({
        "topic": "Should we use tabs or spaces?",
    })
    print(f"Simple topic: {simple_result['result']}")

    # Test with complex topic
    complex_result = await app.ainvoke({
        "topic": "What are the ethical implications of AI in healthcare?",
    })
    print(f"Complex topic: {complex_result['result']}")

asyncio.run(run_conditional_workflow())
```

## Parallel Debates

```python
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from artemis.integrations import ArtemisDebateNode

class ParallelDebateState(TypedDict):
    main_topic: str
    topics: list[str]
    debate_results: list[dict]
    final_analysis: str

def split_into_aspects(state: ParallelDebateState) -> ParallelDebateState:
    main = state["main_topic"]
    topics = [
        f"Technical feasibility of {main}",
        f"Economic viability of {main}",
        f"Ethical implications of {main}",
    ]
    return {"topics": topics}

def merge_results(state: ParallelDebateState) -> ParallelDebateState:
    results = state["debate_results"]
    analysis = "COMPREHENSIVE ANALYSIS\n" + "=" * 40 + "\n\n"

    for i, result in enumerate(results):
        verdict = result.get("verdict", {})
        analysis += f"Aspect {i+1}: {verdict.get('decision', 'N/A')}\n"
        analysis += f"  Confidence: {verdict.get('confidence', 0):.0%}\n"
        analysis += f"  Key point: {verdict.get('reasoning', 'N/A')[:100]}...\n\n"

    return {"final_analysis": analysis}

async def run_parallel_debates():
    workflow = StateGraph(ParallelDebateState)

    # Add nodes
    workflow.add_node("split", split_into_aspects)

    # Add debate nodes for each aspect
    for i in range(3):
        workflow.add_node(f"debate_{i}", ArtemisDebateNode(model="gpt-4o", rounds=2))

    workflow.add_node("merge", merge_results)

    # Add edges
    workflow.add_edge(START, "split")
    for i in range(3):
        workflow.add_edge("split", f"debate_{i}")
        workflow.add_edge(f"debate_{i}", "merge")
    workflow.add_edge("merge", END)

    app = workflow.compile()

    result = await app.ainvoke({
        "main_topic": "autonomous vehicles",
    })

    print(result["final_analysis"])

asyncio.run(run_parallel_debates())
```

## Human-in-the-Loop

```python
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from artemis.integrations import ArtemisDebateNode

class HITLState(TypedDict):
    topic: str
    verdict: dict
    needs_review: bool
    human_decision: str
    final_result: str

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
    workflow = StateGraph(HITLState)

    workflow.add_node("debate", ArtemisDebateNode(model="gpt-4o"))
    workflow.add_node("human_review", request_review)
    workflow.add_node("finalize", finalize)

    workflow.add_edge(START, "debate")
    workflow.add_conditional_edges("debate", check_confidence)
    workflow.add_edge("human_review", END)  # Pause for human input
    workflow.add_edge("finalize", END)

    # Add checkpointing
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    # Run with a controversial topic (likely low confidence)
    config = {"configurable": {"thread_id": "review-1"}}

    result = await app.ainvoke(
        {"topic": "Should AI be granted legal personhood?"},
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

## Streaming Progress

```python
import asyncio
from artemis.integrations import create_debate_workflow

async def stream_debate_progress():
    workflow = create_debate_workflow(model="gpt-4o", rounds=3)
    app = workflow.compile()

    print("Starting debate stream...")
    print("=" * 60)

    async for event in app.astream({"topic": "Is open source sustainable?"}):
        for node_name, node_output in event.items():
            if node_name == "debate":
                current_round = node_output.get("current_round", 0)
                transcript = node_output.get("transcript", [])

                if transcript:
                    latest = transcript[-1]
                    print(f"\n[Round {current_round}] {latest['agent']}")
                    print(f"{latest['argument']['content'][:200]}...")

    print("\n" + "=" * 60)
    print("Debate complete!")

asyncio.run(stream_debate_progress())
```

## Next Steps

- See [Basic Debate](basic-debate.md) for fundamentals
- Add [Safety Monitors](safety-monitors.md) to workflows
- Learn about [LangGraph Integration](../integrations/langgraph.md)
