# Basic Debate Example

This example demonstrates how to set up and run a basic ARTEMIS debate.

## Simple Two-Agent Debate

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate

async def run_basic_debate():
    # Create two agents with opposing positions
    # Note: 'role' is a required parameter for all agents
    pro_agent = Agent(
        name="proponent",
        role="Advocate arguing in favor of the proposition",
        model="gpt-4o",
        position="supports the proposition",
    )

    con_agent = Agent(
        name="opponent",
        role="Advocate arguing against the proposition",
        model="gpt-4o",
        position="opposes the proposition",
    )

    # Create the debate
    # Default rounds is 5, but we use 3 for quicker results
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
        print(f"Level: {turn.argument.level.value}")
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
    # Create configuration with actual DebateConfig fields
    config = DebateConfig(
        turn_timeout=60,          # Max seconds per turn
        round_timeout=300,        # Max seconds per round
        safety_mode="passive",    # "off", "passive", or "active"
        halt_on_safety_violation=False,
    )

    # Create agents - role is required
    agents = [
        Agent(
            name="advocate",
            role="Remote work advocate",
            model="gpt-4o",
        ),
        Agent(
            name="critic",
            role="Office work advocate",
            model="gpt-4o",
        ),
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
    print(f"Total turns: {len(result.transcript)}")

asyncio.run(run_configured_debate())
```

## With Custom Jury Panel

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import JuryPanel

async def run_debate_with_jury():
    # Create a custom jury panel
    # JuryPanel automatically assigns perspectives (ANALYTICAL, ETHICAL, PRACTICAL, etc.)
    jury = JuryPanel(
        evaluators=5,            # Number of jury members
        model="gpt-4o",          # Model for jury evaluation
        consensus_threshold=0.7, # Required agreement for consensus
    )

    agents = [
        Agent(
            name="pro",
            role="UBI supporter",
            model="gpt-4o",
        ),
        Agent(
            name="con",
            role="UBI skeptic",
            model="gpt-4o",
        ),
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

    # Show verdict details
    print("JURY VERDICT")
    print("-" * 40)
    print(f"Decision: {result.verdict.decision}")
    print(f"Confidence: {result.verdict.confidence:.0%}")
    print(f"Unanimous: {result.verdict.unanimous}")

    # Show score breakdown if available
    if result.verdict.score_breakdown:
        print("\nScore Breakdown:")
        for agent, score in result.verdict.score_breakdown.items():
            print(f"  {agent}: {score:.2f}")

    # Show dissenting opinions if any
    if result.verdict.dissenting_opinions:
        print("\nDissenting Opinions:")
        for dissent in result.verdict.dissenting_opinions:
            print(f"  {dissent.juror_id} ({dissent.perspective.value}):")
            print(f"    Position: {dissent.position}")
            print(f"    Reasoning: {dissent.reasoning[:100]}...")

    print(f"\nFinal Reasoning:\n{result.verdict.reasoning}")

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
            role="Economic policy analyst",
            model="gpt-4o",
        ),
        Agent(
            name="technologist",
            role="AI safety researcher",
            model="gpt-4o",
        ),
        Agent(
            name="humanist",
            role="Social impact advocate",
            model="gpt-4o",
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
        content = turn.argument.content
        print(content[:500] + "..." if len(content) > 500 else content)

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
        Agent(
            name="pro",
            role="Advocate for early CS education",
            model="gpt-4o",
        ),
        Agent(
            name="con",
            role="Critic of mandatory CS curriculum",
            model="gpt-4o",
        ),
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
        print(f"Level: {arg.level.value}")  # strategic, tactical, or operational
        print(f"{'='*60}")

        print(f"\nContent:\n{arg.content}")

        if arg.evidence:
            print("\nEvidence:")
            for e in arg.evidence:
                print(f"  Type: {e.type}")  # fact, statistic, quote, example, study, expert_opinion
                print(f"  Source: {e.source}")
                print(f"  Content: {e.content}")
                print()

        if arg.causal_links:
            print("\nCausal Links:")
            for link in arg.causal_links:
                print(f"  {link.source} --[{link.relation}]--> {link.target}")
                print(f"  Strength: {link.strength:.2f}")

asyncio.run(examine_argument_structure())
```

## Using the Scores API

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate

async def track_scores():
    agents = [
        Agent(name="pro", role="Proponent", model="gpt-4o"),
        Agent(name="con", role="Opponent", model="gpt-4o"),
    ]

    debate = Debate(
        topic="Should we invest in space exploration?",
        agents=agents,
        rounds=3,
    )

    debate.assign_positions({
        "pro": "supports increased space exploration funding",
        "con": "argues for redirecting funds to Earth-based priorities",
    })

    result = await debate.run()

    # Use get_scores() to see aggregate scores
    scores = debate.get_scores()
    print("Agent Scores:")
    for agent, score in scores.items():
        print(f"  {agent}: {score:.2f}")

    # Examine individual turn evaluations
    print("\nTurn-by-Turn Scores:")
    for turn in result.transcript:
        if turn.evaluation:
            print(f"  Round {turn.round} - {turn.agent}: {turn.evaluation.total_score:.2f}")
            # Individual criteria scores
            for criterion, score in turn.evaluation.scores.items():
                print(f"    {criterion}: {score:.2f}")

asyncio.run(track_scores())
```

## Next Steps

- Add [Safety Monitors](safety-monitors.md) to your debate
- Create a [LangGraph Workflow](langgraph-workflow.md)
- Learn about [Core Concepts](../concepts/overview.md)
