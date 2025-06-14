# Basic Debate Example

This example demonstrates how to set up and run a basic ARTEMIS debate.

## Simple Two-Agent Debate

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate

async def run_basic_debate():
    # Create two agents with opposing positions
    pro_agent = Agent(
        name="proponent",
        model="gpt-4o",
        position="supports the proposition",
    )

    con_agent = Agent(
        name="opponent",
        model="gpt-4o",
        position="opposes the proposition",
    )

    # Create the debate
    debate = Debate(
        topic="Should AI development be regulated by governments?",
        agents=[pro_agent, con_agent],
        rounds=3,
    )

    # Assign specific positions
    debate.assign_positions({
        "proponent": "supports government regulation of AI development",
        "opponent": "opposes government regulation of AI development",
    })

    # Run the debate
    result = await debate.run()

    # Print results
    print("=" * 60)
    print(f"Topic: {result.topic}")
    print("=" * 60)
    print()

    # Print transcript
    print("DEBATE TRANSCRIPT")
    print("-" * 60)
    for turn in result.transcript:
        print(f"\n[Round {turn.round}] {turn.agent.upper()}")
        print(f"Level: {turn.argument.level}")
        print(f"\n{turn.argument.content}")
        if turn.argument.evidence:
            print("\nEvidence:")
            for e in turn.argument.evidence:
                print(f"  - [{e.type}] {e.source}")
        print("-" * 40)

    # Print verdict
    print("\nVERDICT")
    print("=" * 60)
    print(f"Decision: {result.verdict.decision}")
    print(f"Confidence: {result.verdict.confidence:.0%}")
    print(f"\nReasoning:\n{result.verdict.reasoning}")

if __name__ == "__main__":
    asyncio.run(run_basic_debate())
```

## With Custom Configuration

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.types import DebateConfig

async def run_configured_debate():
    # Create configuration
    config = DebateConfig(
        max_round_time_seconds=300,
        max_total_time_seconds=1800,
        jury_size=5,
        require_unanimous=False,
        evaluation_criteria=[
            "logical_coherence",
            "evidence_quality",
            "argument_strength",
            "ethical_considerations",
        ],
        argument_depth="deep",
        require_evidence=True,
        min_tactical_points=3,
        min_operational_facts=5,
    )

    # Create agents
    agents = [
        Agent(name="advocate", model="gpt-4o", temperature=0.7),
        Agent(name="critic", model="gpt-4o", temperature=0.7),
    ]

    # Create debate with config
    debate = Debate(
        topic="Should remote work become the default for knowledge workers?",
        agents=agents,
        rounds=4,
        config=config,
    )

    debate.assign_positions({
        "advocate": "argues that remote work should be the default",
        "critic": "argues for traditional office work",
    })

    result = await debate.run()

    print(f"Verdict: {result.verdict.decision}")
    print(f"Confidence: {result.verdict.confidence:.0%}")
    print(f"Duration: {result.duration_seconds:.1f}s")

asyncio.run(run_configured_debate())
```

## With Custom Jury

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import Jury, JuryMember, PERSPECTIVES

async def run_debate_with_jury():
    # Create a custom jury
    jury = Jury(
        members=[
            JuryMember(name="logician", perspective=PERSPECTIVES["logical"]),
            JuryMember(name="ethicist", perspective=PERSPECTIVES["ethical"]),
            JuryMember(name="pragmatist", perspective=PERSPECTIVES["practical"]),
            JuryMember(name="skeptic", perspective=PERSPECTIVES["skeptical"]),
            JuryMember(name="analyst", perspective=PERSPECTIVES["analytical"]),
        ],
        voting="simple_majority",
    )

    agents = [
        Agent(name="pro", model="gpt-4o"),
        Agent(name="con", model="gpt-4o"),
    ]

    debate = Debate(
        topic="Is universal basic income a viable economic policy?",
        agents=agents,
        rounds=3,
        jury=jury,
    )

    debate.assign_positions({
        "pro": "supports universal basic income",
        "con": "opposes universal basic income",
    })

    result = await debate.run()

    # Show individual jury votes
    print("JURY VOTES")
    print("-" * 40)
    for vote in result.verdict.votes:
        print(f"{vote.juror}: {vote.decision} (confidence: {vote.confidence:.0%})")
        print(f"  Reasoning: {vote.reasoning[:100]}...")
    print()
    print(f"Final Verdict: {result.verdict.decision}")

asyncio.run(run_debate_with_jury())
```

## Multi-Agent Debate

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate

async def run_multi_agent_debate():
    # Three agents with different perspectives
    agents = [
        Agent(
            name="economist",
            model="gpt-4o",
            temperature=0.6,
        ),
        Agent(
            name="technologist",
            model="gpt-4o",
            temperature=0.7,
        ),
        Agent(
            name="humanist",
            model="gpt-4o",
            temperature=0.8,
        ),
    ]

    debate = Debate(
        topic="How should society prepare for AGI?",
        agents=agents,
        rounds=2,
    )

    debate.assign_positions({
        "economist": "focuses on economic implications and market adaptation",
        "technologist": "focuses on technical safety and alignment",
        "humanist": "focuses on social impact and human values",
    })

    result = await debate.run()

    print("MULTI-PERSPECTIVE ANALYSIS")
    print("=" * 60)

    for turn in result.transcript:
        print(f"\n[{turn.agent.upper()}]")
        print(turn.argument.content[:500] + "...")

    print(f"\n\nSynthesis: {result.verdict.reasoning}")

asyncio.run(run_multi_agent_debate())
```

## Accessing Argument Structure

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate

async def examine_argument_structure():
    agents = [
        Agent(name="pro", model="gpt-4o"),
        Agent(name="con", model="gpt-4o"),
    ]

    debate = Debate(
        topic="Should programming be taught in elementary schools?",
        agents=agents,
        rounds=2,
    )

    debate.assign_positions({
        "pro": "supports early programming education",
        "con": "opposes mandatory programming in elementary schools",
    })

    result = await debate.run()

    # Examine H-L-DAG structure
    for turn in result.transcript:
        arg = turn.argument

        print(f"\n{'='*60}")
        print(f"Agent: {turn.agent}")
        print(f"Round: {turn.round}")
        print(f"Level: {arg.level}")
        print(f"{'='*60}")

        print(f"\nContent:\n{arg.content}")

        if arg.evidence:
            print("\nEvidence:")
            for e in arg.evidence:
                print(f"  Type: {e.type}")
                print(f"  Source: {e.source}")
                if e.quote:
                    print(f"  Quote: \"{e.quote}\"")
                print()

        if arg.causal_links:
            print("\nCausal Links:")
            for link in arg.causal_links:
                print(f"  {link.source} --[{link.relation}]--> {link.target}")
                print(f"  Strength: {link.strength:.2f}")

asyncio.run(examine_argument_structure())
```

## Next Steps

- Add [Safety Monitors](safety-monitors.md) to your debate
- Create a [LangGraph Workflow](langgraph-workflow.md)
- Learn about [Core Concepts](../concepts/overview.md)
