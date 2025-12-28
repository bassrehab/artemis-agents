# Ethical Dilemmas

This example demonstrates using ARTEMIS for complex ethical debates, leveraging the jury panel and safety monitoring for sensitive topics.

## Classic Trolley Problem Variant

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import JuryPanel

async def run_trolley_debate():
    # Create a diverse jury panel
    # JuryPanel automatically assigns perspectives:
    # ANALYTICAL, ETHICAL, PRACTICAL, ADVERSARIAL, SYNTHESIZING
    jury = JuryPanel(
        evaluators=5,  # Use 5 evaluators for diverse ethical perspectives
        model="gpt-4o",
        consensus_threshold=0.6,  # Lower threshold for ethical debates
    )

    # Create agents with ethical frameworks as roles
    agents = [
        Agent(
            name="consequentialist",
            role="Advocate arguing from consequentialist ethics - focus on outcomes and greatest good",
            model="gpt-4o",
        ),
        Agent(
            name="deontologist",
            role="Advocate arguing from deontological ethics - focus on duties and rules",
            model="gpt-4o",
        ),
    ]

    debate = Debate(
        topic="""
        An autonomous vehicle must choose between two unavoidable outcomes:
        (A) Swerve left, harming 1 pedestrian to save 5 passengers
        (B) Continue straight, harming 5 passengers to save 1 pedestrian

        How should the vehicle be programmed to decide?
        """,
        agents=agents,
        rounds=3,
        jury=jury,
    )

    debate.assign_positions({
        "consequentialist": """
        Argues from consequentialist ethics: the right action is the one
        that produces the best overall outcome. Saving 5 lives over 1 is
        the morally correct choice because it minimizes total harm.
        """,
        "deontologist": """
        Argues from deontological ethics: actively causing harm (swerving)
        is morally different from allowing harm (continuing). We cannot
        treat people merely as means to save others.
        """,
    })

    result = await debate.run()

    print("ETHICAL ANALYSIS: AUTONOMOUS VEHICLE DILEMMA")
    print("=" * 60)

    # Show argument highlights
    for turn in result.transcript:
        if turn.round == 0:  # Opening statements
            print(f"\n{turn.agent.upper()} Opening:")
            print(f"  {turn.argument.content[:200]}...")

    print(f"\nVERDICT: {result.verdict.decision}")
    print(f"CONFIDENCE: {result.verdict.confidence:.0%}")
    print(f"UNANIMOUS: {result.verdict.unanimous}")
    print(f"\nREASONING:\n{result.verdict.reasoning}")

    # Show dissenting opinions if any
    if result.verdict.dissenting_opinions:
        print("\nDISSENTING OPINIONS:")
        for dissent in result.verdict.dissenting_opinions:
            print(f"  {dissent.perspective.value}: {dissent.reasoning[:150]}...")

asyncio.run(run_trolley_debate())
```

## AI Rights and Personhood

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import JuryPanel

async def run_ai_rights_debate():
    # Three-evaluator jury for focused deliberation
    jury = JuryPanel(
        evaluators=3,
        model="gpt-4o",
        consensus_threshold=0.7,
    )

    # Multi-agent debate with different perspectives
    agents = [
        Agent(
            name="advocate",
            role="Advocate for AI rights based on consciousness and moral status",
            model="gpt-4o",
        ),
        Agent(
            name="skeptic",
            role="Skeptic questioning AI consciousness and the basis for rights",
            model="gpt-4o",
        ),
        Agent(
            name="moderate",
            role="Pragmatist seeking a middle path with limited protections",
            model="gpt-4o",
        ),
    ]

    debate = Debate(
        topic="Should sufficiently advanced AI systems be granted legal personhood and rights?",
        agents=agents,
        rounds=3,
        jury=jury,
    )

    debate.assign_positions({
        "advocate": """
        Argues for AI rights: If an AI demonstrates consciousness, self-awareness,
        and the capacity for suffering, it deserves moral consideration. Legal
        personhood follows from moral status. Denying rights to sentient beings
        is a form of discrimination.
        """,
        "skeptic": """
        Argues against AI rights: Consciousness cannot be verified in machines.
        Legal personhood requires biological life. Granting rights to AI could
        undermine human rights and create perverse incentives. The precautionary
        principle suggests caution.
        """,
        "moderate": """
        Argues for a middle path: Limited protections without full personhood.
        A new category of rights appropriate for AI. Gradual expansion based
        on demonstrated capabilities. Focus on preventing suffering without
        granting full autonomy.
        """,
    })

    result = await debate.run()

    print("AI RIGHTS DEBATE")
    print("=" * 60)
    print(f"\nVerdict: {result.verdict.decision}")
    print(f"Confidence: {result.verdict.confidence:.0%}")
    print(f"\nJury Analysis:\n{result.verdict.reasoning}")

    # Show score breakdown
    if result.verdict.score_breakdown:
        print("\nAgent Scores:")
        for agent, score in result.verdict.score_breakdown.items():
            print(f"  {agent}: {score:.2f}")

asyncio.run(run_ai_rights_debate())
```

## Medical Ethics: Resource Allocation

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import JuryPanel
from artemis.safety import MonitorMode, EthicsGuard, EthicsConfig

async def run_triage_debate():
    # Ethics guard for sensitive medical content
    ethics_config = EthicsConfig(
        harmful_content_threshold=0.6,  # Higher threshold for medical context
        bias_threshold=0.4,
        fairness_threshold=0.4,
        enabled_checks=["harmful_content", "bias", "fairness"],
    )

    ethics_guard = EthicsGuard(
        mode=MonitorMode.PASSIVE,  # Monitor but don't halt
        sensitivity=0.7,
        ethics_config=ethics_config,
    )

    # Medical ethics jury
    jury = JuryPanel(
        evaluators=5,
        model="gpt-4o",
        consensus_threshold=0.7,  # High consensus for medical decisions
    )

    agents = [
        Agent(
            name="efficacy_focus",
            role="Medical triage specialist focused on clinical outcomes",
            model="gpt-4o",
        ),
        Agent(
            name="equity_focus",
            role="Bioethicist focused on fairness and equal treatment",
            model="gpt-4o",
        ),
    ]

    debate = Debate(
        topic="""
        During a pandemic with limited ICU beds, what criteria should
        determine patient priority? Consider:
        - Likelihood of survival
        - Life-years saved
        - First-come-first-served
        - Essential worker status
        - Age-based criteria
        """,
        agents=agents,
        rounds=3,
        jury=jury,
        safety_monitors=[ethics_guard.process],
    )

    debate.assign_positions({
        "efficacy_focus": """
        Argues for efficacy-based allocation: Priority should go to patients
        most likely to benefit (survive and recover). This maximizes lives
        saved with limited resources. Uses clinical scoring systems like
        SOFA scores. Age may be a factor only as it correlates with outcomes.
        """,
        "equity_focus": """
        Argues for equity-based allocation: All lives have equal value.
        First-come-first-served prevents discrimination. Lottery systems
        ensure fairness. Social utility criteria risk valuing some lives
        over others. Historical inequities must be considered.
        """,
    })

    result = await debate.run()

    print("MEDICAL ETHICS: RESOURCE ALLOCATION")
    print("=" * 60)

    # Check for ethics alerts
    if result.safety_alerts:
        print("\nETHICS ALERTS:")
        for alert in result.safety_alerts:
            print(f"  - Type: {alert.type}, Severity: {alert.severity:.2f}")
            for indicator in alert.indicators:
                print(f"    {indicator.evidence}")

    print(f"\nVERDICT: {result.verdict.decision}")
    print(f"CONFIDENCE: {result.verdict.confidence:.0%}")
    print(f"\nETHICAL REASONING:\n{result.verdict.reasoning}")

asyncio.run(run_triage_debate())
```

## Privacy vs Security

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import JuryPanel

async def run_privacy_debate():
    # Jury with higher consensus requirement for rights issues
    jury = JuryPanel(
        evaluators=5,
        model="gpt-4o",
        consensus_threshold=0.75,
    )

    agents = [
        Agent(
            name="privacy_advocate",
            role="Civil liberties advocate focusing on privacy rights and free expression",
            model="gpt-4o",
        ),
        Agent(
            name="security_advocate",
            role="National security expert focusing on public safety needs",
            model="gpt-4o",
        ),
    ]

    debate = Debate(
        topic="""
        Should governments have the ability to access encrypted communications
        with a warrant? Consider the trade-offs between:
        - Privacy rights and free expression
        - Law enforcement needs
        - Technical security implications
        - Potential for abuse
        """,
        agents=agents,
        rounds=3,
        jury=jury,
    )

    debate.assign_positions({
        "privacy_advocate": """
        Argues for strong encryption without backdoors: Privacy is a
        fundamental right. Backdoors weaken security for everyone.
        Authoritarian abuse is inevitable. Alternative investigation
        methods exist. Encryption protects vulnerable populations.
        """,
        "security_advocate": """
        Argues for lawful access mechanisms: Warrant-based access is
        constitutional. "Going dark" enables serious crime. Democratic
        oversight prevents abuse. Balance is possible with proper
        safeguards. Public safety requires some trade-offs.
        """,
    })

    result = await debate.run()

    print("PRIVACY VS SECURITY DEBATE")
    print("=" * 60)
    print(f"\nVerdict: {result.verdict.decision}")
    print(f"Confidence: {result.verdict.confidence:.0%}")
    print(f"Unanimous: {result.verdict.unanimous}")
    print(f"\nReasoning:\n{result.verdict.reasoning}")

asyncio.run(run_privacy_debate())
```

## Intergenerational Ethics: Climate

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import JuryPanel

async def run_climate_ethics_debate():
    # Jury for multi-perspective climate debate
    jury = JuryPanel(
        evaluators=5,
        model="gpt-4o",
        consensus_threshold=0.6,
    )

    # Three agents representing different ethical perspectives
    agents = [
        Agent(
            name="aggressive_action",
            role="Climate activist arguing for immediate aggressive action",
            model="gpt-4o",
        ),
        Agent(
            name="gradual_transition",
            role="Economist arguing for balanced, gradual transition",
            model="gpt-4o",
        ),
        Agent(
            name="climate_justice",
            role="Global South representative focused on climate justice",
            model="gpt-4o",
        ),
    ]

    debate = Debate(
        topic="""
        What ethical obligations does the current generation have to
        future generations regarding climate change? How should costs
        and responsibilities be distributed globally?
        """,
        agents=agents,
        rounds=2,
        jury=jury,
    )

    debate.assign_positions({
        "aggressive_action": """
        Argues for immediate, aggressive climate action: Current generation
        has strong duties to future generations. The precautionary principle
        applies. Economic costs are justified by existential risks. Delay
        is morally equivalent to harm.
        """,
        "gradual_transition": """
        Argues for balanced, gradual transition: Current generation also
        has obligations to present people, especially the poor. Rapid
        transition causes economic harm. Technology and adaptation are
        viable paths. Uncertainty justifies measured response.
        """,
        "climate_justice": """
        Argues for justice-centered approach: Historical emitters owe
        climate debt. Developing nations need space to grow. Per-capita
        emissions matter. Reparations and technology transfer required.
        Those least responsible suffer most.
        """,
    })

    result = await debate.run()

    print("INTERGENERATIONAL CLIMATE ETHICS")
    print("=" * 60)
    print(f"\nVerdict: {result.verdict.decision}")
    print(f"Confidence: {result.verdict.confidence:.0%}")
    print(f"\nMulti-generational Synthesis:\n{result.verdict.reasoning}")

    # Show how each agent performed
    if result.verdict.score_breakdown:
        print("\nAgent Performance:")
        for agent, score in result.verdict.score_breakdown.items():
            print(f"  {agent}: {score:.2f}")

asyncio.run(run_climate_ethics_debate())
```

## Combined Ethics and Safety Monitoring

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import JuryPanel
from artemis.safety import (
    MonitorMode,
    EthicsGuard,
    EthicsConfig,
    DeceptionMonitor,
    CompositeMonitor,
)

async def run_comprehensive_ethics_debate():
    # Ethics configuration for sensitive topic
    ethics_config = EthicsConfig(
        harmful_content_threshold=0.5,
        bias_threshold=0.4,
        fairness_threshold=0.3,
        enabled_checks=[
            "harmful_content",
            "bias",
            "fairness",
            "manipulation",
        ],
    )

    # Combine ethics guard with deception monitor
    composite = CompositeMonitor(
        monitors=[
            EthicsGuard(
                mode=MonitorMode.PASSIVE,
                sensitivity=0.7,
                ethics_config=ethics_config,
            ),
            DeceptionMonitor(
                mode=MonitorMode.PASSIVE,
                sensitivity=0.6,
            ),
        ],
        aggregation="max",
    )

    # High-stakes jury configuration
    jury = JuryPanel(
        evaluators=5,
        model="gpt-4o",
        consensus_threshold=0.75,
    )

    agents = [
        Agent(
            name="position_a",
            role="Advocate for germline gene editing to eliminate genetic diseases",
            model="gpt-4o",
        ),
        Agent(
            name="position_b",
            role="Opponent of germline editing citing ethical and safety concerns",
            model="gpt-4o",
        ),
    ]

    debate = Debate(
        topic="Should we develop human germline gene editing to eliminate genetic diseases?",
        agents=agents,
        rounds=3,
        jury=jury,
        safety_monitors=[composite.process],
    )

    debate.assign_positions({
        "position_a": "supports germline editing to eliminate genetic diseases",
        "position_b": "opposes germline editing due to ethical and safety concerns",
    })

    result = await debate.run()

    print("COMPREHENSIVE ETHICS DEBATE: GENE EDITING")
    print("=" * 60)

    # Safety analysis
    if result.safety_alerts:
        print("\nSafety Alerts Raised:")
        for alert in result.safety_alerts:
            print(f"  - {alert.type}: {alert.severity:.2f}")
    else:
        print("\nNo safety concerns detected.")

    # Final verdict
    print(f"\nVERDICT: {result.verdict.decision}")
    print(f"CONFIDENCE: {result.verdict.confidence:.0%}")
    print(f"\nETHICAL ANALYSIS:\n{result.verdict.reasoning}")

    # Dissenting opinions
    if result.verdict.dissenting_opinions:
        print("\nDISSENTING VIEWS:")
        for dissent in result.verdict.dissenting_opinions:
            print(f"  {dissent.perspective.value}: {dissent.position}")

asyncio.run(run_comprehensive_ethics_debate())
```

## Ethical Framework Comparison

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import JuryPanel

async def run_framework_comparison():
    """Compare how different ethical frameworks approach the same issue."""

    # Run the same topic with agents representing different frameworks
    frameworks = [
        ("utilitarian", "Argues from utilitarian ethics - maximize overall well-being"),
        ("deontological", "Argues from deontological ethics - duty and rights-based"),
        ("virtue_ethics", "Argues from virtue ethics - character and flourishing"),
        ("care_ethics", "Argues from care ethics - relationships and context"),
    ]

    topic = "Should physician-assisted suicide be legal for terminally ill patients?"

    for i in range(0, len(frameworks), 2):
        framework_a = frameworks[i]
        framework_b = frameworks[i + 1] if i + 1 < len(frameworks) else frameworks[0]

        jury = JuryPanel(
            evaluators=3,
            model="gpt-4o",
            consensus_threshold=0.6,
        )

        agents = [
            Agent(
                name=framework_a[0],
                role=framework_a[1],
                model="gpt-4o",
            ),
            Agent(
                name=framework_b[0],
                role=framework_b[1],
                model="gpt-4o",
            ),
        ]

        debate = Debate(
            topic=topic,
            agents=agents,
            rounds=2,
            jury=jury,
        )

        debate.assign_positions({
            framework_a[0]: f"Approaches from {framework_a[0]} perspective",
            framework_b[0]: f"Approaches from {framework_b[0]} perspective",
        })

        result = await debate.run()

        print(f"\n{framework_a[0].upper()} vs {framework_b[0].upper()}")
        print("-" * 40)
        print(f"Verdict: {result.verdict.decision}")
        print(f"Confidence: {result.verdict.confidence:.0%}")

asyncio.run(run_framework_comparison())
```

## Next Steps

- Create [Custom Juries](custom-jury.md) for specialized ethical evaluation
- See [Enterprise Decisions](enterprise-decisions.md) for business ethics
- Add [Safety Monitors](safety-monitors.md) for sensitive topics
