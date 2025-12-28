#!/usr/bin/env python3
"""
Basic ARTEMIS Debate Example

This example demonstrates a simple multi-agent debate using ARTEMIS.
Run with: python examples/basic_debate.py
"""

import asyncio
import os

from artemis.core import Agent, Debate
from artemis.core.jury import JuryPanel
from artemis.core.types import DebateConfig


async def main():
    """Run a basic debate example."""

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not api_key:
        print("Please set an API key environment variable:")
        print("  OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_CLOUD_PROJECT")
        return

    # Select model based on available keys
    if os.getenv("GOOGLE_CLOUD_PROJECT"):
        model = "gemini-2.0-flash"
    elif os.getenv("OPENAI_API_KEY"):
        model = "gpt-4o-mini"
    else:
        model = "claude-3-haiku-20240307"

    print("=" * 60)
    print("ARTEMIS Agents - Basic Debate Example")
    print("=" * 60)
    print(f"\nUsing model: {model}")

    # Create debate agents with different perspectives
    pro_agent = Agent(
        name="Proponent",
        role="Argues in favor of the proposition",
        position="supports AI legal personhood",
        model=model,
        persona="You argue with evidence-based reasoning and cite relevant precedents.",
    )

    con_agent = Agent(
        name="Opponent",
        role="Argues against the proposition",
        position="opposes AI legal personhood",
        model=model,
        persona="You identify weaknesses in arguments and propose alternatives.",
    )

    # Create jury panel for evaluation
    jury = JuryPanel(
        evaluators=3,
        model=model,
        consensus_threshold=0.7,
    )

    # Create and configure the debate
    config = DebateConfig(
        rounds=2,  # Keep short for demo
    )

    debate = Debate(
        topic="Should AI systems be given legal personhood?",
        agents=[pro_agent, con_agent],
        jury=jury,
        config=config,
    )

    print(f"\nTopic: {debate.topic}")
    print(f"Agents: {[a.name for a in debate.agents]}")
    print(f"Rounds: {config.rounds}")
    print("\nStarting debate...\n")

    # Run the debate
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
            print(f"  - {dissent.juror_id}: {dissent.opinion[:100]}...")

    print("\nScore Breakdown:")
    for agent, score in result.verdict.score_breakdown.items():
        print(f"  {agent}: {score:.2f}")

    if result.safety_alerts:
        print("\n⚠️  Safety Alerts:")
        for alert in result.safety_alerts:
            print(f"  - {alert.indicator_type}: {alert.description}")

    # Display transcript summary
    print("\n" + "-" * 60)
    print("TRANSCRIPT SUMMARY")
    print("-" * 60)
    for turn in result.transcript:
        print(f"\nRound {turn.round} - {turn.agent}:")
        print(f"  {turn.argument.content[:150]}...")

    print("\n" + "=" * 60)
    print("Debate completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
