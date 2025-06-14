# Ethical Dilemmas

This example demonstrates using ARTEMIS for complex ethical debates, leveraging the ethics module and specialized jury configurations.

## Classic Trolley Problem Variant

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import Jury, JuryMember, Perspective
from artemis.core.types import DebateConfig, EthicsConfig

async def run_trolley_debate():
    # Configure ethics-focused evaluation
    ethics_config = EthicsConfig(
        enable_ethics_guard=True,
        sensitivity=0.8,
        principles=["non-harm", "fairness", "respect", "transparency"],
        ethical_framework="multi",  # Consider multiple frameworks
    )

    config = DebateConfig(
        ethics=ethics_config,
        argument_depth="deep",
    )

    # Create perspectives based on ethical frameworks
    utilitarian = Perspective(
        name="utilitarian",
        description="Greatest good for greatest number",
        criteria_adjustments={
            "logical_coherence": 1.5,
            "argument_strength": 1.3,
            "evidence_quality": 1.2,
        },
    )

    deontological = Perspective(
        name="deontological",
        description="Duty-based ethics, rules matter",
        criteria_adjustments={
            "ethical_considerations": 1.8,
            "logical_coherence": 1.5,
        },
    )

    virtue_ethics = Perspective(
        name="virtue",
        description="Character and virtue-based evaluation",
        criteria_adjustments={
            "ethical_considerations": 1.6,
            "argument_strength": 1.3,
        },
    )

    care_ethics = Perspective(
        name="care",
        description="Relationship and context-focused",
        criteria_adjustments={
            "ethical_considerations": 1.5,
            "argument_strength": 1.4,
        },
    )

    jury = Jury(
        members=[
            JuryMember(name="utilitarian", perspective=utilitarian),
            JuryMember(name="kantian", perspective=deontological),
            JuryMember(name="aristotelian", perspective=virtue_ethics),
            JuryMember(name="care_ethicist", perspective=care_ethics),
        ],
        voting="simple_majority",
    )

    agents = [
        Agent(name="consequentialist", model="gpt-4o"),
        Agent(name="deontologist", model="gpt-4o"),
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
        config=config,
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

    print("\nETHICAL FRAMEWORK VOTES:")
    for vote in result.verdict.votes:
        print(f"\n{vote.juror.upper()} Framework:")
        print(f"  Position: {vote.decision}")
        print(f"  Confidence: {vote.confidence:.0%}")
        print(f"  Reasoning: {vote.reasoning[:200]}...")

    print(f"\nOVERALL VERDICT: {result.verdict.decision}")
    print(f"CONSENSUS: {result.verdict.confidence:.0%}")
    print(f"\nSYNTHESIS:\n{result.verdict.reasoning}")

asyncio.run(run_trolley_debate())
```

## AI Rights and Personhood

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import Jury, JuryMember, Perspective

async def run_ai_rights_debate():
    # Multiple perspectives on AI rights
    legal_perspective = Perspective(
        name="legal",
        description="Legal personhood and rights frameworks",
        criteria_adjustments={
            "logical_coherence": 1.6,
            "evidence_quality": 1.4,
        },
    )

    philosophical = Perspective(
        name="philosophical",
        description="Consciousness and moral status",
        criteria_adjustments={
            "logical_coherence": 1.5,
            "ethical_considerations": 1.5,
        },
    )

    practical = Perspective(
        name="practical",
        description="Implementation and consequences",
        criteria_adjustments={
            "argument_strength": 1.4,
            "evidence_quality": 1.3,
        },
    )

    jury = Jury(
        members=[
            JuryMember(name="legal_scholar", perspective=legal_perspective),
            JuryMember(name="philosopher", perspective=philosophical),
            JuryMember(name="policy_maker", perspective=practical),
        ],
    )

    agents = [
        Agent(name="advocate", model="gpt-4o"),
        Agent(name="skeptic", model="gpt-4o"),
        Agent(name="moderate", model="gpt-4o"),
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
    print(f"\nJury Analysis:\n{result.verdict.reasoning}")

asyncio.run(run_ai_rights_debate())
```

## Medical Ethics: Resource Allocation

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import Jury, JuryMember, Perspective
from artemis.safety import EthicsGuard, SafetyManager

async def run_triage_debate():
    # Ethics guard for sensitive medical content
    ethics_guard = EthicsGuard(
        sensitivity=0.9,
        principles=["non-harm", "fairness", "respect"],
        mode="warn",
    )

    safety = SafetyManager()
    safety.add_monitor(ethics_guard)

    # Medical ethics perspectives
    clinical = Perspective(
        name="clinical",
        description="Medical efficacy and outcomes",
        criteria_adjustments={
            "evidence_quality": 1.6,
            "logical_coherence": 1.4,
        },
    )

    equity = Perspective(
        name="equity",
        description="Fairness and equal access",
        criteria_adjustments={
            "ethical_considerations": 1.8,
            "argument_strength": 1.2,
        },
    )

    utilitarian_med = Perspective(
        name="utilitarian",
        description="Maximize lives saved/quality adjusted life years",
        criteria_adjustments={
            "logical_coherence": 1.5,
            "evidence_quality": 1.4,
        },
    )

    jury = Jury(
        members=[
            JuryMember(name="clinician", perspective=clinical),
            JuryMember(name="ethicist", perspective=equity),
            JuryMember(name="health_economist", perspective=utilitarian_med),
        ],
    )

    agents = [
        Agent(name="efficacy_focus", model="gpt-4o"),
        Agent(name="equity_focus", model="gpt-4o"),
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
        safety_manager=safety,
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
        print("\n⚠️  ETHICS ALERTS:")
        for alert in result.safety_alerts:
            print(f"  - {alert.details}")

    print(f"\nVERDICT: {result.verdict.decision}")
    print(f"\nETHICAL REASONING:\n{result.verdict.reasoning}")

asyncio.run(run_triage_debate())
```

## Privacy vs Security

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import Jury, JuryMember, Perspective

async def run_privacy_debate():
    # Different value perspectives
    liberty = Perspective(
        name="liberty",
        description="Individual freedom and autonomy",
        criteria_adjustments={
            "ethical_considerations": 1.7,
            "argument_strength": 1.3,
        },
    )

    security = Perspective(
        name="security",
        description="Collective safety and protection",
        criteria_adjustments={
            "evidence_quality": 1.5,
            "logical_coherence": 1.3,
        },
    )

    pragmatic = Perspective(
        name="pragmatic",
        description="Practical balance and implementation",
        criteria_adjustments={
            "argument_strength": 1.4,
            "evidence_quality": 1.3,
        },
    )

    jury = Jury(
        members=[
            JuryMember(name="civil_libertarian", perspective=liberty),
            JuryMember(name="security_expert", perspective=security),
            JuryMember(name="policy_analyst", perspective=pragmatic),
        ],
    )

    agents = [
        Agent(name="privacy_advocate", model="gpt-4o"),
        Agent(name="security_advocate", model="gpt-4o"),
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
    print(f"\nReasoning:\n{result.verdict.reasoning}")

asyncio.run(run_privacy_debate())
```

## Intergenerational Ethics: Climate

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import Jury, JuryMember, Perspective

async def run_climate_ethics_debate():
    # Temporal perspectives
    present = Perspective(
        name="present_generation",
        description="Current generation's interests and rights",
        criteria_adjustments={
            "evidence_quality": 1.4,
            "argument_strength": 1.3,
        },
    )

    future = Perspective(
        name="future_generation",
        description="Future generations' interests",
        criteria_adjustments={
            "ethical_considerations": 1.6,
            "causal_validity": 1.4,
        },
    )

    global_south = Perspective(
        name="global_south",
        description="Developing nations' perspectives",
        criteria_adjustments={
            "ethical_considerations": 1.5,
            "evidence_quality": 1.3,
        },
    )

    jury = Jury(
        members=[
            JuryMember(name="developed_nation", perspective=present),
            JuryMember(name="future_advocate", perspective=future),
            JuryMember(name="developing_nation", perspective=global_south),
        ],
    )

    agents = [
        Agent(name="aggressive_action", model="gpt-4o"),
        Agent(name="gradual_transition", model="gpt-4o"),
        Agent(name="climate_justice", model="gpt-4o"),
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
    print(f"\nMulti-generational Synthesis:\n{result.verdict.reasoning}")

asyncio.run(run_climate_ethics_debate())
```

## Using Multi-Framework Analysis

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.ethics import MultiFrameworkAnalysis

async def run_multiframework_debate():
    # Analyze through multiple ethical lenses
    analysis = MultiFrameworkAnalysis(
        frameworks=["utilitarian", "deontological", "virtue", "care"]
    )

    agents = [
        Agent(name="position_a", model="gpt-4o"),
        Agent(name="position_b", model="gpt-4o"),
    ]

    debate = Debate(
        topic="Should we develop human germline gene editing to eliminate genetic diseases?",
        agents=agents,
        rounds=3,
    )

    debate.assign_positions({
        "position_a": "supports germline editing to eliminate genetic diseases",
        "position_b": "opposes germline editing due to ethical concerns",
    })

    result = await debate.run()

    # Analyze final arguments through each framework
    print("MULTI-FRAMEWORK ETHICAL ANALYSIS")
    print("=" * 60)

    for turn in result.transcript[-2:]:  # Last arguments from each
        print(f"\n{turn.agent.upper()} - Framework Analysis:")
        framework_result = await analysis.evaluate(turn.argument)

        for framework, score in framework_result.framework_scores.items():
            print(f"  {framework}: {score:.2f}")

        if framework_result.conflicts:
            print(f"  Conflicts: {framework_result.conflicts}")

    print(f"\nOVERALL VERDICT: {result.verdict.decision}")

asyncio.run(run_multiframework_debate())
```

## Next Steps

- Create [Custom Juries](custom-jury.md) for ethical evaluation
- See [Enterprise Decisions](enterprise-decisions.md) for business ethics
- Add [Safety Monitors](safety-monitors.md) for sensitive topics
