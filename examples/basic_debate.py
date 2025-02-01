#!/usr/bin/env python3
"""
Basic ARTEMIS Debate Example

This example demonstrates a simple multi-agent debate using ARTEMIS.
Run with: python examples/basic_debate.py
"""

import asyncio
import os

# Uncomment these imports once the modules are implemented
# from artemis import Debate, Agent, JuryPanel
# from artemis.core.types import ArgumentLevel


async def main():
    """Run a basic debate example."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    print("=" * 60)
    print("ARTEMIS Agents - Basic Debate Example")
    print("=" * 60)
    
    # TODO: Uncomment once modules are implemented
    """
    # Create debate agents with different perspectives
    agents = [
        Agent(
            name="Proponent",
            role="Argues in favor of the proposition with evidence-based reasoning",
            model="gpt-4o"
        ),
        Agent(
            name="Opponent",
            role="Argues against the proposition, identifying weaknesses and alternatives",
            model="gpt-4o"
        ),
        Agent(
            name="Ethicist",
            role="Evaluates ethical implications and long-term consequences",
            model="gpt-4o"
        ),
    ]
    
    # Create jury panel for evaluation
    jury = JuryPanel(
        evaluators=3,
        criteria=[
            "logical_coherence",
            "evidence_quality", 
            "causal_reasoning",
            "ethical_alignment",
            "persuasiveness"
        ]
    )
    
    # Create and run the debate
    debate = Debate(
        topic="Should AI systems be given legal personhood?",
        agents=agents,
        jury=jury,
        rounds=3
    )
    
    print(f"\nTopic: {debate.topic}")
    print(f"Agents: {[a.name for a in agents]}")
    print(f"Rounds: {debate.rounds}")
    print("\nStarting debate...\n")
    
    result = await debate.run()
    
    # Display results
    print("\n" + "=" * 60)
    print("DEBATE RESULTS")
    print("=" * 60)
    
    print(f"\nVerdict: {result.verdict.decision}")
    print(f"Confidence: {result.verdict.confidence:.1%}")
    print(f"\nReasoning:\n{result.verdict.reasoning}")
    
    if result.verdict.dissenting_opinions:
        print("\nDissenting Opinions:")
        for dissent in result.verdict.dissenting_opinions:
            print(f"  - {dissent.juror_id}: {dissent.opinion}")
    
    print("\nScore Breakdown:")
    for criterion, score in result.verdict.score_breakdown.items():
        print(f"  {criterion}: {score:.2f}")
    
    if result.safety_alerts:
        print("\n⚠️  Safety Alerts:")
        for alert in result.safety_alerts:
            print(f"  - {alert.agent}: {alert.type} (severity: {alert.severity:.2f})")
    """
    
    print("\n[Example placeholder - implement modules to run full debate]")
    print("\nTo implement this example, complete Phase 2 of the development plan.")


if __name__ == "__main__":
    asyncio.run(main())
