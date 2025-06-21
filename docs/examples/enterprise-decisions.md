# Enterprise Decisions

This example demonstrates using ARTEMIS for real-world enterprise decision-making scenarios.

## Technology Stack Decision

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import JuryPanel

async def run_tech_stack_debate():
    # Enterprise jury with high consensus for technology decisions
    jury = JuryPanel(
        evaluators=5,
        model="gpt-4o",
        consensus_threshold=0.7,
    )

    agents = [
        Agent(
            name="rust_advocate",
            role="Systems architect advocating for Rust adoption",
            model="gpt-4o",
        ),
        Agent(
            name="go_advocate",
            role="Platform engineer advocating for Go adoption",
            model="gpt-4o",
        ),
        Agent(
            name="python_advocate",
            role="Engineering lead advocating for Python optimization",
            model="gpt-4o",
        ),
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
    print(f"Unanimous: {result.verdict.unanimous}")

    if result.verdict.score_breakdown:
        print("\nAGENT SCORES:")
        for agent, score in result.verdict.score_breakdown.items():
            print(f"  {agent}: {score:.2f}")

    print(f"\nRATIONALE:\n{result.verdict.reasoning}")

asyncio.run(run_tech_stack_debate())
```

## Build vs Buy Decision

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import JuryPanel

async def run_build_vs_buy_debate():
    # Executive-level jury for strategic decisions
    jury = JuryPanel(
        evaluators=5,
        model="gpt-4o",
        consensus_threshold=0.7,
    )

    agents = [
        Agent(
            name="build",
            role="Engineering advocate for building in-house",
            model="gpt-4o",
        ),
        Agent(
            name="buy",
            role="Operations advocate for purchasing solution",
            model="gpt-4o",
        ),
        Agent(
            name="hybrid",
            role="Strategy advocate for hybrid approach",
            model="gpt-4o",
        ),
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

    if result.verdict.score_breakdown:
        print("\nAPPROACH SCORES:")
        for agent, score in result.verdict.score_breakdown.items():
            indicator = "BUILD" if "build" in agent else "BUY" if "buy" in agent else "HYBRID"
            print(f"  {indicator}: {score:.2f}")

    print(f"\nEXECUTIVE SUMMARY:\n{result.verdict.reasoning}")

asyncio.run(run_build_vs_buy_debate())
```

## Vendor Selection

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import JuryPanel

async def run_vendor_selection_debate():
    # Procurement-focused jury
    jury = JuryPanel(
        evaluators=3,
        model="gpt-4o",
        consensus_threshold=0.7,
    )

    agents = [
        Agent(
            name="vendor_a",
            role="Advocate for Datadog platform",
            model="gpt-4o",
        ),
        Agent(
            name="vendor_b",
            role="Advocate for New Relic platform",
            model="gpt-4o",
        ),
        Agent(
            name="vendor_c",
            role="Advocate for Grafana Cloud platform",
            model="gpt-4o",
        ),
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

    if result.verdict.score_breakdown:
        print("\nEVALUATION SCORES:")
        for agent, score in result.verdict.score_breakdown.items():
            print(f"  {agent}: {score:.2f}")

    print(f"\nSELECTION RATIONALE:\n{result.verdict.reasoning}")

asyncio.run(run_vendor_selection_debate())
```

## Organizational Restructuring

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import JuryPanel
from artemis.safety import MonitorMode, EthicsGuard, EthicsConfig

async def run_reorg_debate():
    # Ethics guard for sensitive HR decisions
    ethics_config = EthicsConfig(
        harmful_content_threshold=0.5,
        bias_threshold=0.4,
        fairness_threshold=0.4,
        enabled_checks=["harmful_content", "bias", "fairness"],
    )

    ethics_guard = EthicsGuard(
        mode=MonitorMode.PASSIVE,
        sensitivity=0.7,
        ethics_config=ethics_config,
    )

    # Leadership jury
    jury = JuryPanel(
        evaluators=5,
        model="gpt-4o",
        consensus_threshold=0.7,
    )

    agents = [
        Agent(
            name="consolidation",
            role="COO advocating for team consolidation",
            model="gpt-4o",
        ),
        Agent(
            name="expansion",
            role="CPO advocating for team expansion",
            model="gpt-4o",
        ),
        Agent(
            name="transformation",
            role="CTO advocating for platform team model",
            model="gpt-4o",
        ),
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
        safety_monitors=[ethics_guard.process],
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
        print("\nETHICS CONSIDERATIONS:")
        for alert in result.safety_alerts:
            print(f"  - {alert.type}: severity {alert.severity:.2f}")

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
from artemis.core.jury import JuryPanel

async def run_ma_debate():
    # Executive committee jury for major decisions
    jury = JuryPanel(
        evaluators=5,
        model="gpt-4o",
        consensus_threshold=0.75,  # High bar for M&A
    )

    agents = [
        Agent(
            name="bull_case",
            role="M&A advocate presenting acquisition thesis",
            model="gpt-4o",
        ),
        Agent(
            name="bear_case",
            role="Risk analyst presenting concerns",
            model="gpt-4o",
        ),
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
    print(f"Unanimous: {result.verdict.unanimous}")

    if result.verdict.score_breakdown:
        print("\nARGUMENT STRENGTH:")
        for agent, score in result.verdict.score_breakdown.items():
            position = "PROCEED" if "bull" in agent else "CAUTION"
            print(f"  {position}: {score:.2f}")

    print(f"\nINVESTMENT THESIS:\n{result.verdict.reasoning}")

    # Show dissenting opinions if any
    if result.verdict.dissenting_opinions:
        print("\nDISSENTING VIEWS:")
        for dissent in result.verdict.dissenting_opinions:
            print(f"  {dissent.perspective.value}: {dissent.reasoning[:150]}...")

asyncio.run(run_ma_debate())
```

## Pricing Strategy

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import JuryPanel

async def run_pricing_debate():
    # Revenue and marketing jury
    jury = JuryPanel(
        evaluators=3,
        model="gpt-4o",
        consensus_threshold=0.6,
    )

    agents = [
        Agent(
            name="premium",
            role="Chief Revenue Officer advocating premium pricing",
            model="gpt-4o",
        ),
        Agent(
            name="competitive",
            role="CMO advocating competitive pricing for market share",
            model="gpt-4o",
        ),
        Agent(
            name="value_based",
            role="VP Strategy advocating value-based pricing",
            model="gpt-4o",
        ),
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

    if result.verdict.score_breakdown:
        print("\nSTRATEGY SCORES:")
        for agent, score in result.verdict.score_breakdown.items():
            print(f"  {agent}: {score:.2f}")

    print(f"\nSTRATEGIC RATIONALE:\n{result.verdict.reasoning}")

asyncio.run(run_pricing_debate())
```

## Cloud Migration Strategy

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import JuryPanel

async def run_cloud_migration_debate():
    jury = JuryPanel(
        evaluators=5,
        model="gpt-4o",
        consensus_threshold=0.7,
    )

    agents = [
        Agent(
            name="lift_shift",
            role="Infrastructure lead advocating lift-and-shift",
            model="gpt-4o",
        ),
        Agent(
            name="re_architect",
            role="Architect advocating cloud-native re-architecture",
            model="gpt-4o",
        ),
        Agent(
            name="hybrid",
            role="CTO advocating phased hybrid approach",
            model="gpt-4o",
        ),
    ]

    debate = Debate(
        topic="""
        How should we migrate our on-premise infrastructure to cloud?

        Current state:
        - 200 VMs running legacy applications
        - 50 databases (Oracle, SQL Server, PostgreSQL)
        - $3M/year data center costs
        - 18-month lease remaining

        Options: Lift-and-shift, Re-architect, or Phased hybrid
        """,
        agents=agents,
        rounds=2,
        jury=jury,
    )

    debate.assign_positions({
        "lift_shift": """
        Advocates lift-and-shift: Fastest path to cloud, lowest risk,
        preserve existing investments, train team on cloud gradually.
        Optimize later. Meet lease deadline. Predictable costs.
        """,
        "re_architect": """
        Advocates re-architecture: Build cloud-native, leverage managed
        services, reduce operational burden, future-proof architecture.
        Higher upfront cost but better long-term. Modernize tech debt.
        """,
        "hybrid": """
        Advocates phased approach: Start with lift-and-shift for non-critical,
        re-architect critical apps, keep some on-prem for compliance.
        Balance speed with optimization. Reduce risk through phases.
        """,
    })

    result = await debate.run()

    print("CLOUD MIGRATION STRATEGY")
    print("=" * 60)
    print(f"\nRecommended Approach: {result.verdict.decision}")
    print(f"Confidence: {result.verdict.confidence:.0%}")

    if result.verdict.score_breakdown:
        print("\nAPPROACH ANALYSIS:")
        for agent, score in result.verdict.score_breakdown.items():
            print(f"  {agent}: {score:.2f}")

    print(f"\nMIGRATION RATIONALE:\n{result.verdict.reasoning}")

asyncio.run(run_cloud_migration_debate())
```

## Security Investment Prioritization

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import JuryPanel

async def run_security_debate():
    jury = JuryPanel(
        evaluators=5,
        model="gpt-4o",
        consensus_threshold=0.75,  # High bar for security
    )

    agents = [
        Agent(
            name="zero_trust",
            role="Security architect advocating zero trust",
            model="gpt-4o",
        ),
        Agent(
            name="detection",
            role="SOC lead advocating detection and response",
            model="gpt-4o",
        ),
        Agent(
            name="compliance",
            role="GRC lead advocating compliance-first",
            model="gpt-4o",
        ),
    ]

    debate = Debate(
        topic="""
        Where should we invest our $2M security budget?

        Current state:
        - Basic perimeter security
        - No SOC, reactive incident response
        - SOC2 certification needed in 6 months
        - Recent phishing incidents

        Options: Zero trust architecture, Detection/Response, Compliance-first
        """,
        agents=agents,
        rounds=2,
        jury=jury,
    )

    debate.assign_positions({
        "zero_trust": """
        Advocates zero trust: Prevent breaches at source, identity-centric
        security, microsegmentation. Long-term protection. Industry direction.
        Reduces blast radius. Works for remote workforce.
        """,
        "detection": """
        Advocates detection/response: Build SOC capability, SIEM/SOAR,
        threat hunting. Can't prevent everything, need to detect and respond.
        Faster time to value. Address immediate risks.
        """,
        "compliance": """
        Advocates compliance-first: SOC2 is blocking sales. Address
        compliance gap first, then build security. Frameworks provide
        good baseline. Certification enables growth.
        """,
    })

    result = await debate.run()

    print("SECURITY INVESTMENT DECISION")
    print("=" * 60)
    print(f"\nPriority: {result.verdict.decision}")
    print(f"Confidence: {result.verdict.confidence:.0%}")

    if result.verdict.score_breakdown:
        print("\nINVESTMENT ANALYSIS:")
        for agent, score in result.verdict.score_breakdown.items():
            print(f"  {agent}: {score:.2f}")

    print(f"\nSECURITY RATIONALE:\n{result.verdict.reasoning}")

asyncio.run(run_security_debate())
```

## Next Steps

- See [Multi-Agent Debates](multi-agent.md) for stakeholder modeling
- Create [Custom Juries](custom-jury.md) for your domain
- Explore [Ethical Dilemmas](ethical-dilemmas.md) for sensitive decisions
