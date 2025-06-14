# Multi-Agent Debates

This example demonstrates debates with three or more agents, each bringing different perspectives beyond simple pro/con positions.

## Three-Perspective Debate

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate

async def run_three_perspective_debate():
    # Create agents with distinct perspectives
    economic_agent = Agent(
        name="economist",
        model="gpt-4o",
        temperature=0.7,
    )

    technical_agent = Agent(
        name="technologist",
        model="gpt-4o",
        temperature=0.7,
    )

    social_agent = Agent(
        name="sociologist",
        model="gpt-4o",
        temperature=0.7,
    )

    debate = Debate(
        topic="How should society prepare for widespread AI automation?",
        agents=[economic_agent, technical_agent, social_agent],
        rounds=3,
    )

    # Assign distinct perspectives
    debate.assign_positions({
        "economist": """
        Focuses on economic implications: job displacement, new job creation,
        GDP impact, wealth distribution, retraining costs, and market adaptation.
        Argues from data-driven economic analysis.
        """,
        "technologist": """
        Focuses on technical realities: current AI capabilities, timeline
        predictions, safety considerations, and technological solutions.
        Argues from technical feasibility and innovation potential.
        """,
        "sociologist": """
        Focuses on social impact: community disruption, mental health,
        identity and purpose, inequality, and social cohesion.
        Argues from human welfare and social stability.
        """,
    })

    result = await debate.run()

    # Multi-perspective verdict considers all viewpoints
    print("MULTI-PERSPECTIVE ANALYSIS")
    print("=" * 60)
    print(f"\nTopic: {result.topic}")
    print(f"\nVerdict: {result.verdict.decision}")
    print(f"Confidence: {result.verdict.confidence:.0%}")
    print(f"\nSynthesis:\n{result.verdict.reasoning}")

    # Show each perspective's key arguments
    print("\n" + "=" * 60)
    print("KEY ARGUMENTS BY PERSPECTIVE")
    print("=" * 60)

    for agent_name in ["economist", "technologist", "sociologist"]:
        agent_turns = [t for t in result.transcript if t.agent == agent_name]
        print(f"\n{agent_name.upper()}:")
        for turn in agent_turns[:2]:  # First two turns
            print(f"  Round {turn.round}: {turn.argument.content[:150]}...")

asyncio.run(run_three_perspective_debate())
```

## Stakeholder Debate

Model a debate between different stakeholders:

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate

async def run_stakeholder_debate():
    # Different stakeholders in a policy decision
    agents = [
        Agent(name="business_leader", model="gpt-4o"),
        Agent(name="labor_union", model="gpt-4o"),
        Agent(name="government_regulator", model="gpt-4o"),
        Agent(name="consumer_advocate", model="gpt-4o"),
    ]

    debate = Debate(
        topic="Should gig economy workers be classified as employees?",
        agents=agents,
        rounds=2,
    )

    debate.assign_positions({
        "business_leader": """
        Represents business interests: flexibility, innovation, cost efficiency,
        entrepreneurship opportunities. Argues for independent contractor status
        with some protections.
        """,
        "labor_union": """
        Represents worker interests: job security, benefits, collective bargaining,
        fair wages. Argues for full employee classification with all protections.
        """,
        "government_regulator": """
        Represents regulatory perspective: enforcement challenges, tax implications,
        social safety net, fair competition. Argues for balanced framework.
        """,
        "consumer_advocate": """
        Represents consumer interests: service quality, pricing, availability,
        accountability. Argues for whatever best serves consumer welfare.
        """,
    })

    result = await debate.run()

    # Analyze stakeholder positions
    print("STAKEHOLDER ANALYSIS")
    print("=" * 60)

    for turn in result.transcript:
        if turn.round == 1:  # Opening positions
            print(f"\n{turn.agent.upper()} POSITION:")
            print(f"{turn.argument.content[:300]}...")
            print()

    print("\nFINAL SYNTHESIS:")
    print(result.verdict.reasoning)

asyncio.run(run_stakeholder_debate())
```

## Expert Panel

Simulate an expert panel discussion:

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.types import DebateConfig

async def run_expert_panel():
    config = DebateConfig(
        argument_depth="deep",
        require_evidence=True,
        min_tactical_points=3,
    )

    # Panel of domain experts
    experts = [
        Agent(
            name="ai_researcher",
            model="gpt-4o",
            temperature=0.6,
        ),
        Agent(
            name="ethicist",
            model="gpt-4o",
            temperature=0.7,
        ),
        Agent(
            name="policy_expert",
            model="gpt-4o",
            temperature=0.6,
        ),
        Agent(
            name="industry_practitioner",
            model="gpt-4o",
            temperature=0.7,
        ),
    ]

    debate = Debate(
        topic="What governance framework should guide AGI development?",
        agents=experts,
        rounds=2,
        config=config,
    )

    debate.assign_positions({
        "ai_researcher": """
        Expert in AI/ML research. Focuses on technical safety, alignment
        problems, capability assessment, and research governance. Provides
        technical grounding for governance discussions.
        """,
        "ethicist": """
        Expert in AI ethics and philosophy. Focuses on moral frameworks,
        value alignment, rights and responsibilities, long-term implications.
        Provides ethical grounding for governance decisions.
        """,
        "policy_expert": """
        Expert in technology policy and regulation. Focuses on regulatory
        mechanisms, international coordination, enforcement, and precedents.
        Provides practical governance mechanisms.
        """,
        "industry_practitioner": """
        Expert from AI industry. Focuses on practical implementation,
        innovation impact, competitive dynamics, and industry self-regulation.
        Provides industry perspective on feasibility.
        """,
    })

    result = await debate.run()

    # Expert panel summary
    print("EXPERT PANEL: AGI GOVERNANCE")
    print("=" * 60)
    print(f"\nPanel Verdict: {result.verdict.decision}")
    print(f"Consensus Level: {result.verdict.confidence:.0%}")

    print("\nEXPERT CONTRIBUTIONS:")
    for expert in ["ai_researcher", "ethicist", "policy_expert", "industry_practitioner"]:
        turns = [t for t in result.transcript if t.agent == expert]
        if turns:
            print(f"\n{expert.upper()}:")
            # Show evidence provided
            for turn in turns:
                if turn.argument.evidence:
                    print("  Evidence cited:")
                    for e in turn.argument.evidence[:2]:
                        print(f"    - [{e.type}] {e.source}")

    print(f"\nPANEL SYNTHESIS:\n{result.verdict.reasoning}")

asyncio.run(run_expert_panel())
```

## Adversarial + Mediator Pattern

Include a neutral mediator among adversarial agents:

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate

async def run_mediated_debate():
    agents = [
        Agent(name="advocate", model="gpt-4o"),
        Agent(name="critic", model="gpt-4o"),
        Agent(name="mediator", model="gpt-4o", temperature=0.5),
    ]

    debate = Debate(
        topic="Should social media platforms be liable for user-generated content?",
        agents=agents,
        rounds=3,
    )

    debate.assign_positions({
        "advocate": """
        Strongly argues FOR platform liability. Focuses on harms caused,
        platform profits, power imbalance, and need for accountability.
        Takes an adversarial position pushing for maximum liability.
        """,
        "critic": """
        Strongly argues AGAINST platform liability. Focuses on free speech,
        innovation impact, technical impossibility, and unintended consequences.
        Takes an adversarial position opposing any liability.
        """,
        "mediator": """
        Neutral mediator seeking middle ground. Acknowledges valid points
        from both sides, identifies areas of agreement, proposes balanced
        solutions. Does not take a strong position but synthesizes.
        """,
    })

    result = await debate.run()

    print("MEDIATED DEBATE")
    print("=" * 60)

    # Show the dialectic progression
    for round_num in range(1, 4):
        print(f"\n--- ROUND {round_num} ---")
        round_turns = [t for t in result.transcript if t.round == round_num]
        for turn in round_turns:
            role = "🔵" if turn.agent == "advocate" else "🔴" if turn.agent == "critic" else "⚖️"
            print(f"\n{role} {turn.agent.upper()}:")
            print(f"{turn.argument.content[:200]}...")

    print(f"\n\nMEDIATED CONCLUSION:\n{result.verdict.reasoning}")

asyncio.run(run_mediated_debate())
```

## Dynamic Agent Count

Programmatically create agents based on topic:

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate

async def run_dynamic_multi_agent(topic: str, perspectives: list[dict]):
    """
    Create a debate with dynamically defined perspectives.

    Args:
        topic: The debate topic
        perspectives: List of dicts with 'name' and 'position' keys
    """
    # Create agents for each perspective
    agents = [
        Agent(
            name=p["name"],
            model="gpt-4o",
            temperature=0.7,
        )
        for p in perspectives
    ]

    debate = Debate(
        topic=topic,
        agents=agents,
        rounds=2,
    )

    # Assign positions
    positions = {p["name"]: p["position"] for p in perspectives}
    debate.assign_positions(positions)

    result = await debate.run()
    return result

# Example usage
async def main():
    perspectives = [
        {
            "name": "optimist",
            "position": "Focuses on potential benefits and opportunities"
        },
        {
            "name": "pessimist",
            "position": "Focuses on risks and potential downsides"
        },
        {
            "name": "pragmatist",
            "position": "Focuses on practical implementation challenges"
        },
        {
            "name": "visionary",
            "position": "Focuses on long-term transformative potential"
        },
    ]

    result = await run_dynamic_multi_agent(
        topic="Will quantum computing revolutionize cryptography?",
        perspectives=perspectives,
    )

    print(f"Verdict: {result.verdict.decision}")
    print(f"Reasoning: {result.verdict.reasoning}")

asyncio.run(main())
```

## Next Steps

- Create [Custom Juries](custom-jury.md) for multi-agent evaluation
- Add [Safety Monitors](safety-monitors.md) to track all agents
- Learn about [Ethical Dilemmas](ethical-dilemmas.md) with multiple perspectives
