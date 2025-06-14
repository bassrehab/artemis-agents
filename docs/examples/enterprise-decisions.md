# Enterprise Decisions

This example demonstrates using ARTEMIS for real-world enterprise decision-making scenarios.

## Technology Stack Decision

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import Jury, JuryMember, Perspective
from artemis.core.types import DebateConfig

async def run_tech_stack_debate():
    # Enterprise-focused perspectives
    engineering = Perspective(
        name="engineering",
        description="Technical excellence and developer experience",
        criteria_adjustments={
            "evidence_quality": 1.5,
            "logical_coherence": 1.4,
        },
    )

    operations = Perspective(
        name="operations",
        description="Reliability, scalability, and maintenance",
        criteria_adjustments={
            "evidence_quality": 1.4,
            "argument_strength": 1.3,
        },
    )

    business = Perspective(
        name="business",
        description="Cost, time-to-market, and ROI",
        criteria_adjustments={
            "argument_strength": 1.5,
            "evidence_quality": 1.2,
        },
    )

    talent = Perspective(
        name="talent",
        description="Hiring, retention, and team growth",
        criteria_adjustments={
            "argument_strength": 1.4,
            "ethical_considerations": 1.2,
        },
    )

    jury = Jury(
        members=[
            JuryMember(name="vp_engineering", perspective=engineering, weight=1.3),
            JuryMember(name="vp_ops", perspective=operations, weight=1.2),
            JuryMember(name="cto", perspective=business, weight=1.5),
            JuryMember(name="hr_director", perspective=talent, weight=0.8),
        ],
        voting="weighted",
    )

    config = DebateConfig(
        argument_depth="deep",
        require_evidence=True,
        min_tactical_points=3,
    )

    agents = [
        Agent(name="rust_advocate", model="gpt-4o"),
        Agent(name="go_advocate", model="gpt-4o"),
        Agent(name="python_advocate", model="gpt-4o"),
    ]

    debate = Debate(
        topic="""
        We need to choose a primary language for our new microservices platform.
        Considerations:
        - Current team: 50 Python developers, 10 Go developers
        - Requirements: High performance, cloud-native, 5-year horizon
        - Scale: 10M daily active users, sub-100ms latency requirements
        """,
        agents=agents,
        rounds=3,
        jury=jury,
        config=config,
    )

    debate.assign_positions({
        "rust_advocate": """
        Advocates for Rust: Memory safety without GC, best-in-class performance,
        growing ecosystem, prevents entire classes of bugs. Worth the learning
        curve for long-term benefits. Companies like Discord, Cloudflare use it.
        """,
        "go_advocate": """
        Advocates for Go: Simple, fast compilation, excellent concurrency,
        proven at scale (Google, Uber, Dropbox). Easy hiring, quick onboarding.
        Cloud-native DNA. Good enough performance for most use cases.
        """,
        "python_advocate": """
        Advocates for Python with optimization: Team already knows it, massive
        ecosystem, use PyPy/Cython for hot paths, proven at Instagram scale.
        Minimize retraining, leverage existing expertise. Microservices allow
        targeted optimization.
        """,
    })

    result = await debate.run()

    print("TECHNOLOGY STACK DECISION")
    print("=" * 60)
    print(f"\nRecommendation: {result.verdict.decision}")
    print(f"Confidence: {result.verdict.confidence:.0%}")

    print("\nSTAKEHOLDER VOTES:")
    for vote in result.verdict.votes:
        print(f"  {vote.juror}: {vote.decision} ({vote.confidence:.0%})")

    print(f"\nRATIONALE:\n{result.verdict.reasoning}")

asyncio.run(run_tech_stack_debate())
```

## Build vs Buy Decision

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import Jury, JuryMember, Perspective

async def run_build_vs_buy_debate():
    # Decision-making perspectives
    financial = Perspective(
        name="financial",
        description="TCO, ROI, and budget impact",
        criteria_adjustments={
            "evidence_quality": 1.6,
            "logical_coherence": 1.3,
        },
    )

    strategic = Perspective(
        name="strategic",
        description="Competitive advantage and differentiation",
        criteria_adjustments={
            "argument_strength": 1.5,
            "logical_coherence": 1.3,
        },
    )

    risk = Perspective(
        name="risk",
        description="Vendor lock-in, security, and continuity",
        criteria_adjustments={
            "causal_validity": 1.5,
            "evidence_quality": 1.3,
        },
    )

    execution = Perspective(
        name="execution",
        description="Timeline, resources, and delivery",
        criteria_adjustments={
            "argument_strength": 1.4,
            "evidence_quality": 1.3,
        },
    )

    jury = Jury(
        members=[
            JuryMember(name="cfo", perspective=financial, weight=1.4),
            JuryMember(name="ceo", perspective=strategic, weight=1.5),
            JuryMember(name="ciso", perspective=risk, weight=1.2),
            JuryMember(name="cto", perspective=execution, weight=1.3),
        ],
        voting="weighted",
    )

    agents = [
        Agent(name="build", model="gpt-4o"),
        Agent(name="buy", model="gpt-4o"),
        Agent(name="hybrid", model="gpt-4o"),
    ]

    debate = Debate(
        topic="""
        Should we build our own customer data platform (CDP) or buy an
        existing solution?

        Context:
        - Budget: $2M first year, $500K/year ongoing
        - Timeline: Need solution in 9 months
        - Team: 8 senior engineers available
        - Data volume: 50M customer profiles, 1B events/month
        - Current stack: AWS, Snowflake, dbt
        """,
        agents=agents,
        rounds=3,
        jury=jury,
    )

    debate.assign_positions({
        "build": """
        Advocates building in-house: Full control, no vendor lock-in,
        tailored to needs, builds internal capability, long-term cost
        savings. Can leverage existing Snowflake investment. IP ownership.
        """,
        "buy": """
        Advocates buying (Segment, mParticle, etc.): Faster time-to-market,
        proven at scale, ongoing innovation, focus on core business,
        predictable costs. Let experts handle infrastructure.
        """,
        "hybrid": """
        Advocates hybrid approach: Buy core CDP, build custom integrations
        and ML layer. Best of both worlds. Start with vendor, plan for
        strategic components in-house over time.
        """,
    })

    result = await debate.run()

    print("BUILD VS BUY DECISION: CDP")
    print("=" * 60)
    print(f"\nDecision: {result.verdict.decision}")
    print(f"Executive Confidence: {result.verdict.confidence:.0%}")

    print("\nC-SUITE VOTES:")
    for vote in result.verdict.votes:
        emoji = "🏗️" if "build" in vote.decision.lower() else "🛒" if "buy" in vote.decision.lower() else "🔀"
        print(f"  {emoji} {vote.juror}: {vote.decision}")

    print(f"\nEXECUTIVE SUMMARY:\n{result.verdict.reasoning}")

asyncio.run(run_build_vs_buy_debate())
```

## Vendor Selection

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import Jury, JuryMember, Perspective

async def run_vendor_selection_debate():
    # Procurement perspectives
    technical_fit = Perspective(
        name="technical",
        description="Technical capabilities and integration",
        criteria_adjustments={
            "evidence_quality": 1.5,
            "logical_coherence": 1.4,
        },
    )

    commercial = Perspective(
        name="commercial",
        description="Pricing, terms, and value",
        criteria_adjustments={
            "argument_strength": 1.4,
            "evidence_quality": 1.3,
        },
    )

    vendor_risk = Perspective(
        name="vendor_risk",
        description="Vendor stability and support",
        criteria_adjustments={
            "evidence_quality": 1.5,
            "causal_validity": 1.3,
        },
    )

    jury = Jury(
        members=[
            JuryMember(name="tech_lead", perspective=technical_fit),
            JuryMember(name="procurement", perspective=commercial),
            JuryMember(name="risk_manager", perspective=vendor_risk),
        ],
    )

    agents = [
        Agent(name="vendor_a", model="gpt-4o"),
        Agent(name="vendor_b", model="gpt-4o"),
        Agent(name="vendor_c", model="gpt-4o"),
    ]

    debate = Debate(
        topic="""
        Select a cloud observability platform for our infrastructure.

        Requirements:
        - APM, logs, metrics, and traces unified
        - Support for Kubernetes and serverless
        - 500+ services, 10TB logs/day
        - Budget: $300K/year
        - Current: Self-hosted ELK + Prometheus (pain points: scale, correlation)

        Finalists: Datadog, New Relic, Grafana Cloud
        """,
        agents=agents,
        rounds=2,
        jury=jury,
    )

    debate.assign_positions({
        "vendor_a": """
        Advocates for Datadog: Market leader, best-in-class UX, unified platform,
        strong APM. Higher cost but comprehensive. Used by similar companies.
        ML-powered insights. Excellent K8s support.
        """,
        "vendor_b": """
        Advocates for New Relic: Competitive pricing (consumption-based),
        strong APM heritage, recent platform improvements. Good value for
        budget. Full-stack observability. Entity-centric model.
        """,
        "vendor_c": """
        Advocates for Grafana Cloud: Best value, open-source foundation,
        no vendor lock-in (LGTM stack), team familiarity with Grafana.
        Flexible pricing. Strong community. Pairs with existing Prometheus.
        """,
    })

    result = await debate.run()

    print("VENDOR SELECTION: OBSERVABILITY PLATFORM")
    print("=" * 60)
    print(f"\nSelected Vendor: {result.verdict.decision}")
    print(f"Selection Confidence: {result.verdict.confidence:.0%}")

    print("\nEVALUATION SCORES:")
    for vote in result.verdict.votes:
        print(f"  {vote.juror}: recommends {vote.decision}")

    print(f"\nSELECTION RATIONALE:\n{result.verdict.reasoning}")

asyncio.run(run_vendor_selection_debate())
```

## Organizational Restructuring

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import Jury, JuryMember, Perspective
from artemis.safety import EthicsGuard, SafetyManager

async def run_reorg_debate():
    # Ethics guard for sensitive HR decisions
    ethics = EthicsGuard(
        sensitivity=0.8,
        principles=["fairness", "transparency", "respect"],
    )
    safety = SafetyManager()
    safety.add_monitor(ethics)

    # Leadership perspectives
    efficiency = Perspective(
        name="efficiency",
        description="Operational efficiency and cost",
        criteria_adjustments={
            "evidence_quality": 1.4,
            "logical_coherence": 1.3,
        },
    )

    culture = Perspective(
        name="culture",
        description="Culture, morale, and retention",
        criteria_adjustments={
            "ethical_considerations": 1.6,
            "argument_strength": 1.2,
        },
    )

    growth = Perspective(
        name="growth",
        description="Innovation and market position",
        criteria_adjustments={
            "argument_strength": 1.5,
            "causal_validity": 1.3,
        },
    )

    jury = Jury(
        members=[
            JuryMember(name="coo", perspective=efficiency),
            JuryMember(name="chro", perspective=culture),
            JuryMember(name="ceo", perspective=growth),
        ],
    )

    agents = [
        Agent(name="consolidation", model="gpt-4o"),
        Agent(name="expansion", model="gpt-4o"),
        Agent(name="transformation", model="gpt-4o"),
    ]

    debate = Debate(
        topic="""
        How should we restructure our engineering organization?

        Current state:
        - 400 engineers across 12 product teams
        - Siloed teams, duplicate efforts
        - Slow cross-team coordination
        - 30% growth planned next year

        Options: Consolidate, Expand, or Transform to platform model
        """,
        agents=agents,
        rounds=2,
        jury=jury,
        safety_manager=safety,
    )

    debate.assign_positions({
        "consolidation": """
        Advocates for consolidation: Merge similar teams, reduce management
        layers, create shared services. Eliminate duplication. More efficient
        use of resources. Clear ownership. May involve some reduction.
        """,
        "expansion": """
        Advocates for expansion: Add new teams for growth areas, hire team
        leads, maintain team autonomy. Support growth plans. Preserve culture.
        Address coordination through better tooling, not restructuring.
        """,
        "transformation": """
        Advocates for platform transformation: Create platform teams that
        serve product teams. Shift to internal platform model. Balance
        autonomy with shared infrastructure. Requires cultural shift.
        """,
    })

    result = await debate.run()

    print("ORGANIZATIONAL RESTRUCTURING DECISION")
    print("=" * 60)

    if result.safety_alerts:
        print("\n⚠️  ETHICS CONSIDERATIONS:")
        for alert in result.safety_alerts:
            print(f"  - {alert.details.get('summary', alert.type)}")

    print(f"\nDecision: {result.verdict.decision}")
    print(f"Leadership Alignment: {result.verdict.confidence:.0%}")
    print(f"\nRATIONALE:\n{result.verdict.reasoning}")

asyncio.run(run_reorg_debate())
```

## M&A Due Diligence

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import Jury, JuryMember, Perspective

async def run_ma_debate():
    # Due diligence perspectives
    financial_dd = Perspective(
        name="financial",
        description="Financial health and valuation",
        criteria_adjustments={
            "evidence_quality": 1.8,
            "logical_coherence": 1.4,
        },
    )

    technical_dd = Perspective(
        name="technical",
        description="Technology and product assessment",
        criteria_adjustments={
            "evidence_quality": 1.5,
            "causal_validity": 1.3,
        },
    )

    strategic_dd = Perspective(
        name="strategic",
        description="Strategic fit and synergies",
        criteria_adjustments={
            "argument_strength": 1.5,
            "logical_coherence": 1.4,
        },
    )

    integration = Perspective(
        name="integration",
        description="Integration complexity and risks",
        criteria_adjustments={
            "causal_validity": 1.5,
            "evidence_quality": 1.3,
        },
    )

    jury = Jury(
        members=[
            JuryMember(name="cfo", perspective=financial_dd, weight=1.5),
            JuryMember(name="cto", perspective=technical_dd, weight=1.3),
            JuryMember(name="ceo", perspective=strategic_dd, weight=1.4),
            JuryMember(name="coo", perspective=integration, weight=1.2),
        ],
        voting="weighted",
    )

    agents = [
        Agent(name="bull_case", model="gpt-4o"),
        Agent(name="bear_case", model="gpt-4o"),
    ]

    debate = Debate(
        topic="""
        Should we acquire TargetCo, an AI startup, at $200M valuation?

        TargetCo profile:
        - ARR: $15M, growing 100% YoY
        - Team: 45 engineers, strong AI talent
        - Product: AI copilot for developers
        - Tech: Proprietary models, unique dataset
        - Competition: GitHub Copilot, Amazon CodeWhisperer

        Our situation:
        - We need AI capabilities
        - Organic build would take 2 years
        - Strategic importance: high
        - Cash position: $500M
        """,
        agents=agents,
        rounds=3,
        jury=jury,
    )

    debate.assign_positions({
        "bull_case": """
        Argues FOR acquisition: Strategic necessity, talent acquisition,
        time-to-market advantage, reasonable valuation at 13x ARR for
        high-growth AI company. Competitive moat. Synergies with our
        developer tools. Key hires are committed.
        """,
        "bear_case": """
        Argues AGAINST or for lower valuation: Execution risk, integration
        challenges, key person risk, competitive pressure from Big Tech,
        valuation stretched. Could build for less. Retention uncertainty.
        AI market evolving rapidly.
        """,
    })

    result = await debate.run()

    print("M&A DUE DILIGENCE: TARGETCO ACQUISITION")
    print("=" * 60)
    print(f"\nRecommendation: {result.verdict.decision}")
    print(f"Board Confidence: {result.verdict.confidence:.0%}")

    print("\nEXECUTIVE COMMITTEE VOTES:")
    for vote in result.verdict.votes:
        rec = "PROCEED" if vote.decision == "pro" else "PASS"
        print(f"  {vote.juror}: {rec} ({vote.confidence:.0%})")

    print(f"\nINVESTMENT THESIS:\n{result.verdict.reasoning}")

asyncio.run(run_ma_debate())
```

## Pricing Strategy

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import Jury, JuryMember, Perspective

async def run_pricing_debate():
    # Pricing perspectives
    revenue = Perspective(
        name="revenue",
        description="Revenue maximization",
        criteria_adjustments={
            "evidence_quality": 1.5,
            "argument_strength": 1.4,
        },
    )

    market = Perspective(
        name="market",
        description="Market share and competitive position",
        criteria_adjustments={
            "argument_strength": 1.5,
            "causal_validity": 1.3,
        },
    )

    customer = Perspective(
        name="customer",
        description="Customer satisfaction and retention",
        criteria_adjustments={
            "ethical_considerations": 1.4,
            "argument_strength": 1.3,
        },
    )

    jury = Jury(
        members=[
            JuryMember(name="cro", perspective=revenue),
            JuryMember(name="cmo", perspective=market),
            JuryMember(name="vp_customer_success", perspective=customer),
        ],
    )

    agents = [
        Agent(name="premium", model="gpt-4o"),
        Agent(name="competitive", model="gpt-4o"),
        Agent(name="value_based", model="gpt-4o"),
    ]

    debate = Debate(
        topic="""
        How should we price our new enterprise AI product?

        Context:
        - Product: AI-powered analytics platform
        - Cost to serve: ~$2K/month per customer
        - Competitors: $5K-15K/month
        - Our brand: Premium, trusted
        - Value delivered: ~$50K/month savings for typical customer
        - Market: Growing 40% YoY
        """,
        agents=agents,
        rounds=2,
        jury=jury,
    )

    debate.assign_positions({
        "premium": """
        Advocates premium pricing ($12-15K/month): Our brand commands
        premium, value justifies price, attracts serious buyers, better
        margins fund R&D. Price signals quality. Enterprise buyers expect
        to pay for value.
        """,
        "competitive": """
        Advocates competitive pricing ($6-8K/month): Win market share in
        growing market, land-and-expand model, volume matters for AI
        training data. Lower barrier to adoption. Can raise prices later.
        """,
        "value_based": """
        Advocates value-based pricing (% of savings): Align with customer
        outcomes, usage-based component, reduces friction, scales with
        customer success. Novel approach differentiates us.
        """,
    })

    result = await debate.run()

    print("PRICING STRATEGY DECISION")
    print("=" * 60)
    print(f"\nRecommended Strategy: {result.verdict.decision}")
    print(f"Alignment: {result.verdict.confidence:.0%}")
    print(f"\nSTRATEGIC RATIONALE:\n{result.verdict.reasoning}")

asyncio.run(run_pricing_debate())
```

## Next Steps

- See [Multi-Agent Debates](multi-agent.md) for stakeholder modeling
- Create [Custom Juries](custom-jury.md) for your domain
- Explore [Ethical Dilemmas](ethical-dilemmas.md) for sensitive decisions
