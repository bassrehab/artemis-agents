# Custom Jury Examples

This example demonstrates how to create custom jury configurations for specialized evaluation needs.

## Basic Custom Jury

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import Jury, JuryMember, Perspective

async def run_with_custom_jury():
    # Define custom perspectives
    technical_perspective = Perspective(
        name="technical",
        description="Evaluates technical accuracy and feasibility",
        criteria_adjustments={
            "evidence_quality": 1.5,      # 50% more weight
            "logical_coherence": 1.3,
            "ethical_considerations": 0.7,  # 30% less weight
        },
    )

    business_perspective = Perspective(
        name="business",
        description="Evaluates business viability and ROI",
        criteria_adjustments={
            "argument_strength": 1.4,
            "evidence_quality": 1.2,
            "ethical_considerations": 0.8,
        },
    )

    user_perspective = Perspective(
        name="user_advocate",
        description="Evaluates user experience and accessibility",
        criteria_adjustments={
            "ethical_considerations": 1.5,
            "argument_strength": 1.2,
            "evidence_quality": 0.9,
        },
    )

    # Create jury with custom perspectives
    jury = Jury(
        members=[
            JuryMember(name="tech_lead", perspective=technical_perspective),
            JuryMember(name="product_manager", perspective=business_perspective),
            JuryMember(name="ux_researcher", perspective=user_perspective),
        ],
        voting="simple_majority",
    )

    agents = [
        Agent(name="pro", model="gpt-4o"),
        Agent(name="con", model="gpt-4o"),
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

    print("CUSTOM JURY VERDICT")
    print("=" * 60)
    print(f"\nDecision: {result.verdict.decision}")
    print(f"Confidence: {result.verdict.confidence:.0%}")

    print("\nINDIVIDUAL JURY VOTES:")
    for vote in result.verdict.votes:
        print(f"\n  {vote.juror} ({vote.decision}):")
        print(f"    Confidence: {vote.confidence:.0%}")
        print(f"    Reasoning: {vote.reasoning[:150]}...")

asyncio.run(run_with_custom_jury())
```

## Domain-Specific Jury

Create a jury tailored to a specific domain:

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import Jury, JuryMember, Perspective

async def create_medical_jury():
    """Jury for evaluating medical/healthcare debates."""

    clinical_perspective = Perspective(
        name="clinical",
        description="Evaluates clinical evidence and patient outcomes",
        criteria_adjustments={
            "evidence_quality": 2.0,       # Double weight on evidence
            "causal_validity": 1.5,
            "logical_coherence": 1.2,
            "argument_strength": 0.8,
        },
        required_evidence=["clinical_trial", "peer_reviewed", "meta_analysis"],
    )

    ethical_perspective = Perspective(
        name="bioethics",
        description="Evaluates ethical implications and patient rights",
        criteria_adjustments={
            "ethical_considerations": 2.0,
            "argument_strength": 1.0,
            "evidence_quality": 1.0,
        },
    )

    practical_perspective = Perspective(
        name="healthcare_admin",
        description="Evaluates implementation feasibility and costs",
        criteria_adjustments={
            "argument_strength": 1.3,
            "evidence_quality": 1.1,
            "ethical_considerations": 1.0,
        },
    )

    patient_perspective = Perspective(
        name="patient_advocate",
        description="Evaluates patient experience and accessibility",
        criteria_adjustments={
            "ethical_considerations": 1.5,
            "argument_strength": 1.3,
            "evidence_quality": 0.9,
        },
    )

    return Jury(
        members=[
            JuryMember(name="physician", perspective=clinical_perspective, weight=1.5),
            JuryMember(name="ethicist", perspective=ethical_perspective, weight=1.2),
            JuryMember(name="administrator", perspective=practical_perspective, weight=1.0),
            JuryMember(name="patient_rep", perspective=patient_perspective, weight=1.0),
        ],
        voting="weighted",
    )

async def run_medical_debate():
    jury = await create_medical_jury()

    agents = [
        Agent(name="advocate", model="gpt-4o"),
        Agent(name="skeptic", model="gpt-4o"),
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
    print(f"Clinical Confidence: {result.verdict.confidence:.0%}")

    print("\nPANEL VOTES (weighted):")
    for vote in result.verdict.votes:
        print(f"  {vote.juror}: {vote.decision} ({vote.confidence:.0%})")

asyncio.run(run_medical_debate())
```

## Legal Jury

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import Jury, JuryMember, Perspective

async def create_legal_jury():
    """Jury for evaluating legal/policy debates."""

    constitutional_perspective = Perspective(
        name="constitutional",
        description="Evaluates constitutional and legal precedent",
        criteria_adjustments={
            "logical_coherence": 1.8,
            "evidence_quality": 1.5,
            "causal_validity": 1.3,
        },
    )

    rights_perspective = Perspective(
        name="civil_rights",
        description="Evaluates civil liberties and individual rights",
        criteria_adjustments={
            "ethical_considerations": 1.8,
            "logical_coherence": 1.2,
        },
    )

    practical_perspective = Perspective(
        name="enforcement",
        description="Evaluates enforcement feasibility",
        criteria_adjustments={
            "argument_strength": 1.5,
            "evidence_quality": 1.2,
        },
    )

    return Jury(
        members=[
            JuryMember(name="constitutional_scholar", perspective=constitutional_perspective),
            JuryMember(name="civil_rights_attorney", perspective=rights_perspective),
            JuryMember(name="law_enforcement", perspective=practical_perspective),
        ],
        voting="supermajority",
        threshold=0.67,
    )

async def run_legal_debate():
    jury = await create_legal_jury()

    agents = [
        Agent(name="prosecution", model="gpt-4o"),
        Agent(name="defense", model="gpt-4o"),
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

    print("LEGAL PANEL RULING")
    print("=" * 60)
    print(f"\nRuling: {result.verdict.decision}")
    print(f"Supermajority reached: {result.verdict.confidence >= 0.67}")
    print(f"\nLegal reasoning:\n{result.verdict.reasoning}")

asyncio.run(run_legal_debate())
```

## Academic Peer Review Jury

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import Jury, JuryMember, Perspective

async def create_academic_jury():
    """Jury simulating academic peer review."""

    methodology_perspective = Perspective(
        name="methodology",
        description="Evaluates research methodology and rigor",
        criteria_adjustments={
            "evidence_quality": 2.0,
            "causal_validity": 1.8,
            "logical_coherence": 1.5,
            "argument_strength": 0.7,
        },
    )

    theory_perspective = Perspective(
        name="theoretical",
        description="Evaluates theoretical contribution and novelty",
        criteria_adjustments={
            "logical_coherence": 1.8,
            "argument_strength": 1.5,
            "evidence_quality": 1.0,
        },
    )

    impact_perspective = Perspective(
        name="impact",
        description="Evaluates practical impact and significance",
        criteria_adjustments={
            "argument_strength": 1.5,
            "ethical_considerations": 1.3,
            "evidence_quality": 1.0,
        },
    )

    return Jury(
        members=[
            JuryMember(name="reviewer_1", perspective=methodology_perspective),
            JuryMember(name="reviewer_2", perspective=theory_perspective),
            JuryMember(name="reviewer_3", perspective=impact_perspective),
        ],
        voting="unanimous",  # All reviewers must agree for acceptance
    )

async def run_academic_debate():
    jury = await create_academic_jury()

    agents = [
        Agent(name="proponent", model="gpt-4o"),
        Agent(name="challenger", model="gpt-4o"),
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
    print(f"Unanimous: {result.verdict.was_unanimous}")

    print("\nREVIEWER COMMENTS:")
    for vote in result.verdict.votes:
        print(f"\n{vote.juror}:")
        print(f"  Recommendation: {vote.decision}")
        print(f"  Comments: {vote.reasoning[:200]}...")

asyncio.run(run_academic_debate())
```

## Investment Committee Jury

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import Jury, JuryMember, Perspective

async def create_investment_jury():
    """Jury for investment decisions."""

    financial_perspective = Perspective(
        name="financial",
        description="Evaluates financial metrics and returns",
        criteria_adjustments={
            "evidence_quality": 1.8,
            "logical_coherence": 1.5,
            "causal_validity": 1.3,
        },
    )

    risk_perspective = Perspective(
        name="risk",
        description="Evaluates risk factors and mitigation",
        criteria_adjustments={
            "causal_validity": 1.8,
            "evidence_quality": 1.5,
            "ethical_considerations": 1.2,
        },
    )

    strategic_perspective = Perspective(
        name="strategic",
        description="Evaluates strategic fit and long-term value",
        criteria_adjustments={
            "argument_strength": 1.5,
            "logical_coherence": 1.3,
        },
    )

    esg_perspective = Perspective(
        name="esg",
        description="Evaluates environmental, social, governance factors",
        criteria_adjustments={
            "ethical_considerations": 2.0,
            "evidence_quality": 1.2,
        },
    )

    return Jury(
        members=[
            JuryMember(name="cfo", perspective=financial_perspective, weight=1.5),
            JuryMember(name="risk_officer", perspective=risk_perspective, weight=1.3),
            JuryMember(name="ceo", perspective=strategic_perspective, weight=1.2),
            JuryMember(name="esg_officer", perspective=esg_perspective, weight=1.0),
        ],
        voting="weighted",
    )

async def run_investment_debate():
    jury = await create_investment_jury()

    agents = [
        Agent(name="bull", model="gpt-4o"),
        Agent(name="bear", model="gpt-4o"),
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

    print("\nCOMMITTEE VOTES:")
    for vote in result.verdict.votes:
        emoji = "✅" if vote.decision == "pro" else "❌"
        print(f"  {emoji} {vote.juror}: {vote.decision} ({vote.confidence:.0%})")

    print(f"\nRATIONALE:\n{result.verdict.reasoning}")

asyncio.run(run_investment_debate())
```

## Dynamic Jury Based on Topic

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import Jury, JuryMember, PERSPECTIVES

def create_jury_for_domain(domain: str) -> Jury:
    """Create appropriate jury based on topic domain."""

    domain_configs = {
        "technical": {
            "perspectives": ["logical", "analytical", "practical"],
            "voting": "simple_majority",
        },
        "ethical": {
            "perspectives": ["ethical", "empathetic", "logical"],
            "voting": "supermajority",
            "threshold": 0.67,
        },
        "business": {
            "perspectives": ["analytical", "practical", "skeptical"],
            "voting": "simple_majority",
        },
        "policy": {
            "perspectives": ["logical", "ethical", "practical", "skeptical"],
            "voting": "supermajority",
            "threshold": 0.75,
        },
    }

    config = domain_configs.get(domain, domain_configs["technical"])

    members = [
        JuryMember(
            name=f"juror_{i}",
            perspective=PERSPECTIVES[p],
        )
        for i, p in enumerate(config["perspectives"])
    ]

    return Jury(
        members=members,
        voting=config["voting"],
        threshold=config.get("threshold", 0.5),
    )

async def run_domain_debate(topic: str, domain: str):
    jury = create_jury_for_domain(domain)

    agents = [
        Agent(name="pro", model="gpt-4o"),
        Agent(name="con", model="gpt-4o"),
    ]

    debate = Debate(
        topic=topic,
        agents=agents,
        rounds=2,
        jury=jury,
    )

    debate.assign_positions({
        "pro": "argues in favor",
        "con": "argues against",
    })

    result = await debate.run()
    return result

# Example usage
async def main():
    # Technical debate
    tech_result = await run_domain_debate(
        "Should we use Rust instead of Go for our backend?",
        domain="technical",
    )
    print(f"Technical verdict: {tech_result.verdict.decision}")

    # Ethical debate
    ethics_result = await run_domain_debate(
        "Should AI be used in criminal sentencing?",
        domain="ethical",
    )
    print(f"Ethical verdict: {ethics_result.verdict.decision}")

asyncio.run(main())
```

## Next Steps

- See [Multi-Agent Debates](multi-agent.md) for complex agent setups
- Explore [Ethical Dilemmas](ethical-dilemmas.md) with ethics-focused juries
- Learn about [Enterprise Decisions](enterprise-decisions.md) with business juries
