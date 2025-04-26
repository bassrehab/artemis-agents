#!/usr/bin/env python3
"""
ARTEMIS LangGraph Integration Example

This example demonstrates using ARTEMIS debates within LangGraph workflows.
Run with: python examples/langgraph_workflow.py

Requirements:
    pip install langgraph
"""

import asyncio
import os

from artemis.integrations.langgraph import (
    ArtemisDebateNode,
    DebateNodeState,
    DebatePhase,
    create_debate_workflow,
)


async def simple_workflow():
    """Run a simple single-node debate workflow."""
    print("\n" + "=" * 60)
    print("Simple Debate Workflow")
    print("=" * 60)

    try:
        # Create the workflow
        workflow = create_debate_workflow(
            model="gpt-4o",
            step_by_step=False,  # Single node execution
        )

        # Define initial state
        initial_state: DebateNodeState = {
            "topic": "Should companies be required to implement AI ethics boards?",
            "pro_position": "supports mandatory AI ethics boards",
            "con_position": "opposes mandatory requirements",
            "rounds": 3,
        }

        print(f"\nTopic: {initial_state['topic']}")
        print(f"Rounds: {initial_state['rounds']}")
        print("\nRunning debate workflow...")

        # Run the workflow
        result = await workflow.ainvoke(initial_state)

        print(f"\nPhase: {result['phase']}")
        print(f"Verdict: {result.get('verdict', {}).get('decision', 'N/A')}")
        print(f"Confidence: {result.get('verdict', {}).get('confidence', 0):.1%}")
        print(f"Total Turns: {len(result.get('transcript', []))}")

        if result.get("scores"):
            print("\nScores:")
            for agent, score in result["scores"].items():
                print(f"  {agent}: {score:.2f}")

    except ImportError:
        print("\nNote: Install langgraph for LangGraph integration")
        print("  pip install langgraph")


async def step_by_step_workflow():
    """Run a step-by-step debate workflow."""
    print("\n" + "=" * 60)
    print("Step-by-Step Debate Workflow")
    print("=" * 60)

    try:
        # Create step-by-step workflow
        workflow = create_debate_workflow(
            model="gpt-4o",
            step_by_step=True,  # Multi-node execution
        )

        # Define initial state
        initial_state: DebateNodeState = {
            "topic": "Is a 4-day work week more productive than a 5-day week?",
            "pro_position": "advocates for 4-day work week",
            "con_position": "prefers traditional 5-day schedule",
            "rounds": 2,
        }

        print(f"\nTopic: {initial_state['topic']}")
        print("\nThis workflow will:")
        print("  1. Setup: Initialize agents and debate")
        print("  2. Run Round: Execute each round sequentially")
        print("  3. Finalize: Get jury verdict")
        print("\nRunning step-by-step workflow...")

        # Run the workflow
        result = await workflow.ainvoke(initial_state)

        print(f"\nFinal Phase: {result['phase']}")
        if result.get("verdict"):
            print(f"Verdict: {result['verdict']['decision']}")
            print(f"Reasoning: {result['verdict'].get('reasoning', 'N/A')[:200]}...")

    except ImportError:
        print("\nNote: Install langgraph for LangGraph integration")
        print("  pip install langgraph")


async def custom_node_usage():
    """Use ArtemisDebateNode directly for custom workflows."""
    print("\n" + "=" * 60)
    print("Custom Node Usage")
    print("=" * 60)

    # Create debate node
    node = ArtemisDebateNode(model="gpt-4o")

    print(f"\nNode: {node}")
    print("\nAvailable methods:")
    print("  - node.run_debate(state): Run complete debate")
    print("  - node.setup(state): Initialize debate")
    print("  - node.run_round(state): Run single round")
    print("  - node.finalize(state): Get final verdict")
    print("  - node.get_routing_function(): Get router for conditional edges")

    # Example state
    example_state: DebateNodeState = {
        "topic": "Example topic",
        "rounds": 3,
        "phase": DebatePhase.SETUP.value,
    }

    print(f"\nExample initial state: {example_state}")

    # In a real workflow:
    """
    from langgraph.graph import StateGraph, END

    workflow = StateGraph(DebateNodeState)
    workflow.add_node("setup", node.setup)
    workflow.add_node("debate_round", node.run_round)
    workflow.add_node("get_verdict", node.finalize)

    workflow.set_entry_point("setup")
    workflow.add_edge("setup", "debate_round")
    workflow.add_conditional_edges(
        "debate_round",
        node.get_routing_function(),
        {
            "continue": "debate_round",
            "finalize": "get_verdict",
        }
    )
    workflow.add_edge("get_verdict", END)

    app = workflow.compile()
    result = await app.ainvoke(initial_state)
    """


async def integrated_workflow_example():
    """Example of integrating debate into a larger workflow."""
    print("\n" + "=" * 60)
    print("Integrated Workflow Example")
    print("=" * 60)

    print("\nExample: Decision-making pipeline with debate analysis")
    print("""
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   Gather    │────▶│   ARTEMIS   │────▶│   Decide    │
    │   Context   │     │   Debate    │     │   Action    │
    └─────────────┘     └─────────────┘     └─────────────┘

    1. Gather Context: Collect information about the decision
    2. ARTEMIS Debate: Run structured debate on options
    3. Decide Action: Use verdict to inform decision
    """)

    # Pseudo-code for integrated workflow
    print("Code structure:")
    print("""
    from langgraph.graph import StateGraph, END

    class PipelineState(TypedDict):
        question: str
        context: dict
        debate_result: dict
        final_decision: str

    def gather_context(state):
        # Collect relevant information
        return {"context": {...}}

    workflow = StateGraph(PipelineState)
    workflow.add_node("gather", gather_context)
    workflow.add_node("debate", ArtemisDebateNode().run_debate)
    workflow.add_node("decide", make_decision)

    workflow.set_entry_point("gather")
    workflow.add_edge("gather", "debate")
    workflow.add_edge("debate", "decide")
    workflow.add_edge("decide", END)
    """)


async def main():
    """Run all LangGraph examples."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        print("\nExample: export OPENAI_API_KEY='your-api-key'")
        return

    print("=" * 60)
    print("ARTEMIS Agents - LangGraph Integration Examples")
    print("=" * 60)

    # Show custom node usage (doesn't require API calls)
    await custom_node_usage()

    # Show integrated workflow concept
    await integrated_workflow_example()

    # Note: The following examples require API calls and langgraph
    print("\n" + "-" * 60)
    print("The following examples require API calls and langgraph.")
    print("Uncomment them to run actual workflows.")
    print("-" * 60)

    # Uncomment to run actual workflows:
    # await simple_workflow()
    # await step_by_step_workflow()

    print("\n[Examples configured - uncomment sections to run full workflows]")


if __name__ == "__main__":
    asyncio.run(main())
