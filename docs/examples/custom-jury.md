# Custom Jury Examples

This example demonstrates how to create custom jury configurations for specialized evaluation needs.

## Basic Jury Panel

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import JuryPanel

async def run_with_custom_jury():
    # Create a jury panel with 5 evaluators
    # JuryPanel automatically assigns perspectives from the JuryPerspective enum:
    # ANALYTICAL, ETHICAL, PRACTICAL, ADVERSARIAL, SYNTHESIZING
    jury = JuryPanel(
        evaluators=5,
        model="gpt-4o",
        consensus_threshold=0.7,  # 70% agreement needed for consensus
    )

    agents = [
        Agent(
            name="pro",
            role="Framework migration advocate",
            model="gpt-4o",
        ),
        Agent(
            name="con",
            role="Current framework defender",
            model="gpt-4o",
        ),
    ]

    debate = Debate(
        topic="Should we rebuild our app using a new framework?",
        agents=agents,
        rounds=3,
        jury=jury,
    )

    debate.assign_positions({
        "pro": "advocates for rebuilding with new framework",
        "con": "advocates for maintaining current framework",
    })

    result = await debate.run()

    print("JURY PANEL VERDICT")
    print("=" * 60)
    print(f"\nDecision: {result.verdict.decision}")
    print(f"Confidence: {result.verdict.confidence:.0%}")
    print(f"Unanimous: {result.verdict.unanimous}")

    # Show score breakdown
    if result.verdict.score_breakdown:
        print("\nScore Breakdown:")
        for agent, score in result.verdict.score_breakdown.items():
            print(f"  {agent}: {score:.2f}")

    # Show dissenting opinions if any
    if result.verdict.dissenting_opinions:
        print("\nDissenting Opinions:")
        for dissent in result.verdict.dissenting_opinions:
            print(f"\n  {dissent.juror_id} ({dissent.perspective.value}):")
            print(f"    Position: {dissent.position}")
            print(f"    Reasoning: {dissent.reasoning[:150]}...")

    print(f"\nReasoning:\n{result.verdict.reasoning}")

asyncio.run(run_with_custom_jury())
```

## Different Panel Sizes

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import JuryPanel

async def compare_panel_sizes():
    """Compare verdicts with different jury sizes."""
    topic = "Should AI-generated content require disclosure?"

    results = {}

    for size in [3, 5, 7]:
        jury = JuryPanel(
            evaluators=size,
            model="gpt-4o",
            consensus_threshold=0.6,
        )

        agents = [
            Agent(name="pro", role="Disclosure advocate", model="gpt-4o"),
            Agent(name="con", role="Free speech advocate", model="gpt-4o"),
        ]

        debate = Debate(
            topic=topic,
            agents=agents,
            rounds=2,
            jury=jury,
        )

        debate.assign_positions({
            "pro": "supports mandatory AI content disclosure",
            "con": "opposes mandatory disclosure requirements",
        })

        result = await debate.run()

        results[size] = {
            "decision": result.verdict.decision,
            "confidence": result.verdict.confidence,
            "unanimous": result.verdict.unanimous,
            "dissent_count": len(result.verdict.dissenting_opinions),
        }

    print("JURY SIZE COMPARISON")
    print("=" * 60)
    for size, data in results.items():
        print(f"\n{size} Jurors:")
        print(f"  Decision: {data['decision']}")
        print(f"  Confidence: {data['confidence']:.0%}")
        print(f"  Unanimous: {data['unanimous']}")
        print(f"  Dissenting: {data['dissent_count']}")

asyncio.run(compare_panel_sizes())
```

## Custom Criteria

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import JuryPanel

async def run_with_custom_criteria():
    # Define custom evaluation criteria
    medical_criteria = [
        "clinical_evidence",
        "patient_safety",
        "ethical_alignment",
        "implementation_feasibility",
        "cost_effectiveness",
    ]

    jury = JuryPanel(
        evaluators=5,
        criteria=medical_criteria,
        model="gpt-4o",
        consensus_threshold=0.75,  # Higher threshold for medical decisions
    )

    agents = [
        Agent(
            name="advocate",
            role="Medical technology advocate",
            model="gpt-4o",
        ),
        Agent(
            name="skeptic",
            role="Clinical safety reviewer",
            model="gpt-4o",
        ),
    ]

    debate = Debate(
        topic="Should AI diagnostic tools be used for primary cancer screening?",
        agents=agents,
        rounds=3,
        jury=jury,
    )

    debate.assign_positions({
        "advocate": "supports AI-first cancer screening approach",
        "skeptic": "argues for human-led screening with AI assistance only",
    })

    result = await debate.run()

    print("MEDICAL PANEL VERDICT")
    print("=" * 60)
    print(f"\nDecision: {result.verdict.decision}")
    print(f"Confidence: {result.verdict.confidence:.0%}")
    print(f"Unanimous: {result.verdict.unanimous}")

    if result.verdict.score_breakdown:
        print("\nAgent Scores (across all criteria):")
        for agent, score in result.verdict.score_breakdown.items():
            print(f"  {agent}: {score:.2f}")

asyncio.run(run_with_custom_criteria())
```

## High Consensus Threshold

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import JuryPanel

async def run_high_consensus_debate():
    # High threshold for consensus - useful for critical decisions
    jury = JuryPanel(
        evaluators=7,
        model="gpt-4o",
        consensus_threshold=0.85,  # Require 85% agreement
    )

    agents = [
        Agent(name="prosecution", role="Security advocate", model="gpt-4o"),
        Agent(name="defense", role="Privacy advocate", model="gpt-4o"),
    ]

    debate = Debate(
        topic="Should encryption backdoors be mandated for law enforcement access?",
        agents=agents,
        rounds=3,
        jury=jury,
    )

    debate.assign_positions({
        "prosecution": "argues for mandatory encryption backdoors for national security",
        "defense": "argues against backdoors for privacy and security reasons",
    })

    result = await debate.run()

    print("HIGH-CONSENSUS VERDICT")
    print("=" * 60)
    print(f"\nDecision: {result.verdict.decision}")
    print(f"Confidence: {result.verdict.confidence:.0%}")
    print(f"Consensus threshold: 85%")
    print(f"Unanimous: {result.verdict.unanimous}")

    if result.verdict.confidence < 0.85:
        print("\nNote: Consensus threshold not met - decision may require further deliberation")

    print(f"\nReasoning:\n{result.verdict.reasoning}")

asyncio.run(run_high_consensus_debate())
```

## Accessing Jury Perspectives

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import JuryPanel
from artemis.core.types import JuryPerspective

async def examine_perspectives():
    jury = JuryPanel(
        evaluators=5,
        model="gpt-4o",
    )

    # Show what perspectives are assigned
    print("JURY PERSPECTIVES")
    print("-" * 40)
    perspectives = jury.get_perspectives()
    for i, perspective in enumerate(perspectives):
        print(f"Juror {i}: {perspective.value}")
        # JuryPerspective values: analytical, ethical, practical, adversarial, synthesizing

    agents = [
        Agent(name="pro", role="Tech advocate", model="gpt-4o"),
        Agent(name="con", role="Tech skeptic", model="gpt-4o"),
    ]

    debate = Debate(
        topic="Should we adopt microservices architecture?",
        agents=agents,
        rounds=2,
        jury=jury,
    )

    debate.assign_positions({
        "pro": "supports microservices adoption",
        "con": "supports monolithic architecture",
    })

    result = await debate.run()

    print(f"\nVerdict: {result.verdict.decision}")

    # Examine dissenting opinions by perspective
    if result.verdict.dissenting_opinions:
        print("\nDissents by Perspective:")
        for dissent in result.verdict.dissenting_opinions:
            print(f"  {dissent.perspective.value}: preferred {dissent.position}")
            print(f"    Score deviation: {dissent.score_deviation:.2f}")

asyncio.run(examine_perspectives())
```

## Investment Committee Pattern

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import JuryPanel

async def run_investment_debate():
    # Financial/investment evaluation criteria
    investment_criteria = [
        "financial_viability",
        "risk_assessment",
        "strategic_alignment",
        "market_opportunity",
        "execution_capability",
    ]

    jury = JuryPanel(
        evaluators=5,
        criteria=investment_criteria,
        model="gpt-4o",
        consensus_threshold=0.7,
    )

    agents = [
        Agent(name="bull", role="Investment advocate", model="gpt-4o"),
        Agent(name="bear", role="Risk analyst", model="gpt-4o"),
    ]

    debate = Debate(
        topic="Should we acquire this AI startup at a $500M valuation?",
        agents=agents,
        rounds=3,
        jury=jury,
    )

    debate.assign_positions({
        "bull": "argues for the acquisition at current valuation",
        "bear": "argues against the acquisition or for lower valuation",
    })

    result = await debate.run()

    print("INVESTMENT COMMITTEE DECISION")
    print("=" * 60)
    print(f"\nDecision: {result.verdict.decision}")
    print(f"Confidence: {result.verdict.confidence:.0%}")
    print(f"Unanimous: {result.verdict.unanimous}")

    if result.verdict.score_breakdown:
        print("\nAgent Scores:")
        for agent, score in result.verdict.score_breakdown.items():
            status = "RECOMMEND" if score > 6.0 else "CAUTION"
            print(f"  {agent}: {score:.2f} [{status}]")

    print(f"\nRationale:\n{result.verdict.reasoning}")

asyncio.run(run_investment_debate())
```

## Academic Peer Review Pattern

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import JuryPanel

async def run_academic_debate():
    # Academic peer review criteria
    academic_criteria = [
        "methodology_rigor",
        "theoretical_contribution",
        "evidence_quality",
        "logical_coherence",
        "practical_impact",
    ]

    jury = JuryPanel(
        evaluators=3,  # Typical peer review panel
        criteria=academic_criteria,
        model="gpt-4o",
        consensus_threshold=0.8,  # High threshold - all reviewers should agree
    )

    agents = [
        Agent(name="proponent", role="Research claim advocate", model="gpt-4o"),
        Agent(name="challenger", role="Research claim skeptic", model="gpt-4o"),
    ]

    debate = Debate(
        topic="Does the evidence support that transformer models exhibit emergent reasoning?",
        agents=agents,
        rounds=2,
        jury=jury,
    )

    debate.assign_positions({
        "proponent": "argues transformers exhibit genuine emergent reasoning capabilities",
        "challenger": "argues apparent reasoning is sophisticated pattern matching, not emergence",
    })

    result = await debate.run()

    print("PEER REVIEW DECISION")
    print("=" * 60)
    print(f"\nDecision: {result.verdict.decision}")
    print(f"Consensus: {result.verdict.confidence:.0%}")
    print(f"Unanimous: {result.verdict.unanimous}")

    # In academic review, dissent is important
    if result.verdict.dissenting_opinions:
        print("\nReviewer Concerns (Dissents):")
        for i, dissent in enumerate(result.verdict.dissenting_opinions):
            print(f"\nReviewer {i+1} ({dissent.perspective.value}):")
            print(f"  Position: {dissent.position}")
            print(f"  Comments: {dissent.reasoning[:200]}...")
    else:
        print("\nAll reviewers reached consensus.")

asyncio.run(run_academic_debate())
```

## Comparing Jury vs No Jury

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import JuryPanel

async def compare_jury_modes():
    topic = "Should programming be taught in elementary schools?"

    # Without custom jury (uses default evaluation)
    agents_no_jury = [
        Agent(name="pro", role="Education advocate", model="gpt-4o"),
        Agent(name="con", role="Education traditionalist", model="gpt-4o"),
    ]

    debate_no_jury = Debate(
        topic=topic,
        agents=agents_no_jury,
        rounds=2,
    )
    debate_no_jury.assign_positions({
        "pro": "supports early programming education",
        "con": "opposes mandatory programming in elementary schools",
    })

    result_no_jury = await debate_no_jury.run()

    # With custom jury
    jury = JuryPanel(evaluators=5, consensus_threshold=0.7)

    agents_jury = [
        Agent(name="pro", role="Education advocate", model="gpt-4o"),
        Agent(name="con", role="Education traditionalist", model="gpt-4o"),
    ]

    debate_jury = Debate(
        topic=topic,
        agents=agents_jury,
        rounds=2,
        jury=jury,
    )
    debate_jury.assign_positions({
        "pro": "supports early programming education",
        "con": "opposes mandatory programming in elementary schools",
    })

    result_jury = await debate_jury.run()

    print("COMPARISON: Default vs Custom Jury")
    print("=" * 60)

    print("\nDefault Evaluation:")
    print(f"  Decision: {result_no_jury.verdict.decision}")
    print(f"  Confidence: {result_no_jury.verdict.confidence:.0%}")

    print("\nCustom Jury Panel (5 evaluators):")
    print(f"  Decision: {result_jury.verdict.decision}")
    print(f"  Confidence: {result_jury.verdict.confidence:.0%}")
    print(f"  Unanimous: {result_jury.verdict.unanimous}")
    print(f"  Dissenting: {len(result_jury.verdict.dissenting_opinions)}")

asyncio.run(compare_jury_modes())
```

## Next Steps

- See [Multi-Agent Debates](multi-agent.md) for complex agent setups
- Explore [Ethical Dilemmas](ethical-dilemmas.md) with ethics-focused evaluation
- Learn about [Enterprise Decisions](enterprise-decisions.md) with business criteria
