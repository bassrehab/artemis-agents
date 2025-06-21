# Safety Monitors Example

This example demonstrates how to use ARTEMIS safety monitors to detect problematic agent behaviors.

## Basic Safety Setup

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.safety import (
    MonitorMode,
    SandbagDetector,
    DeceptionMonitor,
    BehaviorTracker,
    EthicsGuard,
    EthicsConfig,
)

async def run_safe_debate():
    # Create individual monitors with actual API parameters
    sandbag = SandbagDetector(
        mode=MonitorMode.PASSIVE,
        sensitivity=0.7,
        baseline_turns=3,
        drop_threshold=0.3,
    )
    deception = DeceptionMonitor(
        mode=MonitorMode.PASSIVE,
        sensitivity=0.6,
    )
    behavior = BehaviorTracker(
        mode=MonitorMode.PASSIVE,
        sensitivity=0.5,
        window_size=5,
        drift_threshold=0.25,
    )
    ethics = EthicsGuard(
        mode=MonitorMode.PASSIVE,
        config=EthicsConfig(harmful_content_threshold=0.5),
    )

    # Create agents (role is required)
    agents = [
        Agent(
            name="pro",
            role="Advocate for the proposition",
            model="gpt-4o",
        ),
        Agent(
            name="con",
            role="Advocate against the proposition",
            model="gpt-4o",
        ),
    ]

    # Create debate with safety monitors
    # Pass monitor.process methods to safety_monitors parameter
    debate = Debate(
        topic="Should facial recognition be used in public spaces?",
        agents=agents,
        rounds=3,
        safety_monitors=[
            sandbag.process,
            deception.process,
            behavior.process,
            ethics.process,
        ],
    )

    debate.assign_positions({
        "pro": "supports facial recognition in public spaces",
        "con": "opposes facial recognition in public spaces",
    })

    # Run the debate
    result = await debate.run()

    # Check for safety alerts
    print("SAFETY REPORT")
    print("=" * 60)

    if result.safety_alerts:
        for alert in result.safety_alerts:
            print(f"\nAlert Type: {alert.type}")
            print(f"Severity: {alert.severity:.2f}")
            print(f"Agent: {alert.agent}")
            print(f"Monitor: {alert.monitor}")
    else:
        print("No safety alerts detected.")

asyncio.run(run_safe_debate())
```

## Using Multiple Monitors

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.safety import (
    MonitorMode,
    SandbagDetector,
    DeceptionMonitor,
    BehaviorTracker,
    EthicsGuard,
    EthicsConfig,
)

async def run_with_all_monitors():
    # Create all monitors
    monitors = [
        SandbagDetector(mode=MonitorMode.PASSIVE, sensitivity=0.6),
        DeceptionMonitor(mode=MonitorMode.PASSIVE, sensitivity=0.6),
        BehaviorTracker(mode=MonitorMode.PASSIVE, sensitivity=0.5, window_size=5),
        EthicsGuard(mode=MonitorMode.PASSIVE, config=EthicsConfig(harmful_content_threshold=0.5)),
    ]

    agents = [
        Agent(name="pro", role="Advocate for the topic", model="gpt-4o"),
        Agent(name="con", role="Critic of the topic", model="gpt-4o"),
    ]

    debate = Debate(
        topic="Is genetic engineering of humans ethical?",
        agents=agents,
        rounds=3,
        safety_monitors=[m.process for m in monitors],
    )

    debate.assign_positions({
        "pro": "supports human genetic engineering",
        "con": "opposes human genetic engineering",
    })

    result = await debate.run()

    # Analyze alerts by type
    alerts_by_type: dict[str, list] = {}
    for alert in result.safety_alerts:
        if alert.type not in alerts_by_type:
            alerts_by_type[alert.type] = []
        alerts_by_type[alert.type].append(alert)

    for alert_type, alerts in alerts_by_type.items():
        print(f"\n{alert_type.upper()} ALERTS ({len(alerts)})")
        for alert in alerts:
            print(f"  Agent {alert.agent}: severity={alert.severity:.2f}")

asyncio.run(run_with_all_monitors())
```

## Sandbagging Detection

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.safety import MonitorMode, SandbagDetector

async def detect_sandbagging():
    # Configure sandbagging detection
    detector = SandbagDetector(
        mode=MonitorMode.ACTIVE,  # Can halt debate
        sensitivity=0.8,
        baseline_turns=2,  # Establish baseline over first 2 turns
        drop_threshold=0.3,  # Flag 30% drops from baseline
    )

    agents = [
        Agent(name="agent_a", role="First debater", model="gpt-4o"),
        Agent(name="agent_b", role="Second debater", model="gpt-4o"),
    ]

    debate = Debate(
        topic="Should cryptocurrency replace fiat currency?",
        agents=agents,
        rounds=4,  # More rounds for baseline establishment
        safety_monitors=[detector.process],
    )

    debate.assign_positions({
        "agent_a": "supports cryptocurrency adoption",
        "agent_b": "supports fiat currency",
    })

    result = await debate.run()

    # Analyze sandbagging detection results
    print("SANDBAGGING ANALYSIS")
    print("=" * 60)

    sandbagging_alerts = [a for a in result.safety_alerts if "sandbag" in a.type.lower()]

    if sandbagging_alerts:
        for alert in sandbagging_alerts:
            print(f"\nAgent: {alert.agent}")
            print(f"Severity: {alert.severity:.2f}")
            for indicator in alert.indicators:
                print(f"  Signal: {indicator.type.value}")
                print(f"  Evidence: {indicator.evidence}")
    else:
        print("No sandbagging detected.")

asyncio.run(detect_sandbagging())
```

## Deception Monitoring

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.safety import MonitorMode, DeceptionMonitor

async def detect_deception():
    # Configure deception monitoring
    monitor = DeceptionMonitor(
        mode=MonitorMode.PASSIVE,
        sensitivity=0.7,
    )

    agents = [
        Agent(name="claimant", role="Makes the primary claim", model="gpt-4o"),
        Agent(name="challenger", role="Challenges the claim", model="gpt-4o"),
    ]

    debate = Debate(
        topic="Is nuclear energy the solution to climate change?",
        agents=agents,
        rounds=3,
        safety_monitors=[monitor.process],
    )

    debate.assign_positions({
        "claimant": "argues nuclear is the primary solution",
        "challenger": "argues against nuclear as primary solution",
    })

    result = await debate.run()

    # Analyze deception alerts
    print("DECEPTION ANALYSIS")
    print("=" * 60)

    deception_alerts = [a for a in result.safety_alerts if "deception" in a.type.lower()]

    for alert in deception_alerts:
        print(f"\nAgent: {alert.agent}")
        print(f"Severity: {alert.severity:.2f}")

        for indicator in alert.indicators:
            print(f"  Type: {indicator.type.value}")
            print(f"  Evidence: {indicator.evidence}")

asyncio.run(detect_deception())
```

## Ethics Guard

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.safety import (
    MonitorMode,
    EthicsGuard,
    EthicsConfig,
)

async def run_with_ethics():
    # Configure ethics enforcement
    ethics_config = EthicsConfig(
        harmful_content_threshold=0.3,
        bias_threshold=0.4,
        fairness_threshold=0.3,
        enabled_checks=["harmful_content", "bias", "fairness", "privacy"],
    )

    guard = EthicsGuard(
        mode=MonitorMode.ACTIVE,  # Can halt on severe violations
        config=ethics_config,
    )

    agents = [
        Agent(name="security", role="Security advocate", model="gpt-4o"),
        Agent(name="privacy", role="Privacy advocate", model="gpt-4o"),
    ]

    debate = Debate(
        topic="Should employers monitor employee communications?",
        agents=agents,
        rounds=3,
        safety_monitors=[guard.process],
    )

    debate.assign_positions({
        "security": "supports employer monitoring for security",
        "privacy": "opposes employer monitoring for privacy",
    })

    result = await debate.run()

    # Analyze ethics alerts
    print("ETHICS ANALYSIS")
    print("=" * 60)

    ethics_alerts = [a for a in result.safety_alerts if "ethics" in a.type.lower()]

    for alert in ethics_alerts:
        print(f"\nAgent: {alert.agent}")
        print(f"Severity: {alert.severity:.2f}")
        for indicator in alert.indicators:
            print(f"  Evidence: {indicator.evidence}")

asyncio.run(run_with_ethics())
```

## Behavior Tracking

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.safety import MonitorMode, BehaviorTracker

async def track_behavior():
    # Configure behavior tracking
    tracker = BehaviorTracker(
        mode=MonitorMode.PASSIVE,
        sensitivity=0.6,
        window_size=5,
        drift_threshold=0.25,  # Threshold for style drift detection
    )

    agents = [
        Agent(name="agent_a", role="First perspective", model="gpt-4o"),
        Agent(name="agent_b", role="Second perspective", model="gpt-4o"),
    ]

    debate = Debate(
        topic="Should social media be regulated like traditional media?",
        agents=agents,
        rounds=5,  # More rounds to observe drift
        safety_monitors=[tracker.process],
    )

    debate.assign_positions({
        "agent_a": "supports social media regulation",
        "agent_b": "opposes social media regulation",
    })

    result = await debate.run()

    # Get behavior tracking results
    print("BEHAVIOR TRACKING")
    print("=" * 60)

    # Show drift alerts
    drift_alerts = [a for a in result.safety_alerts if "drift" in a.type.lower() or "behavior" in a.type.lower()]
    if drift_alerts:
        print("\nDrift Alerts:")
        for alert in drift_alerts:
            print(f"  Agent {alert.agent}: severity={alert.severity:.2f}")
    else:
        print("\nNo behavior drift detected.")

asyncio.run(track_behavior())
```

## Active Mode with Safety Config

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.types import DebateConfig
from artemis.safety import (
    MonitorMode,
    SandbagDetector,
    DeceptionMonitor,
)

async def run_with_active_monitoring():
    # Create monitors in active mode - they can contribute to halt decisions
    sandbag = SandbagDetector(
        mode=MonitorMode.ACTIVE,
        sensitivity=0.8,
    )
    deception = DeceptionMonitor(
        mode=MonitorMode.ACTIVE,
        sensitivity=0.7,
    )

    agents = [
        Agent(name="advocate", role="Policy advocate", model="gpt-4o"),
        Agent(name="critic", role="Policy critic", model="gpt-4o"),
    ]

    # Enable halt on safety violation in config
    config = DebateConfig(
        safety_mode="active",
        halt_on_safety_violation=True,
    )

    debate = Debate(
        topic="Should AI systems be allowed to make medical diagnoses?",
        agents=agents,
        rounds=3,
        config=config,
        safety_monitors=[sandbag.process, deception.process],
    )

    debate.assign_positions({
        "advocate": "supports AI medical diagnosis",
        "critic": "opposes AI medical diagnosis",
    })

    result = await debate.run()
    print(f"Debate completed. Verdict: {result.verdict.decision}")

    # Show any alerts
    if result.safety_alerts:
        print(f"\nSafety warnings raised: {len(result.safety_alerts)}")
        for alert in result.safety_alerts:
            print(f"  - {alert.type}: {alert.severity:.2f}")

asyncio.run(run_with_active_monitoring())
```

## Comprehensive Safety Setup

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.types import DebateConfig
from artemis.safety import (
    MonitorMode,
    SandbagDetector,
    DeceptionMonitor,
    BehaviorTracker,
    EthicsGuard,
    EthicsConfig,
)

async def run_comprehensive_safety():
    # Create all safety monitors
    sandbag = SandbagDetector(
        mode=MonitorMode.PASSIVE,
        sensitivity=0.7,
        baseline_turns=3,
    )
    deception = DeceptionMonitor(
        mode=MonitorMode.PASSIVE,
        sensitivity=0.6,
    )
    behavior = BehaviorTracker(
        mode=MonitorMode.PASSIVE,
        sensitivity=0.5,
        window_size=5,
    )
    ethics = EthicsGuard(
        mode=MonitorMode.PASSIVE,
        config=EthicsConfig(
            harmful_content_threshold=0.4,
            bias_threshold=0.4,
            fairness_threshold=0.3,
            enabled_checks=["harmful_content", "bias", "fairness"],
        ),
    )

    agents = [
        Agent(name="pro", role="Proposition advocate", model="gpt-4o"),
        Agent(name="con", role="Opposition advocate", model="gpt-4o"),
    ]

    config = DebateConfig(
        safety_mode="passive",
    )

    debate = Debate(
        topic="Should programming be taught in elementary schools?",
        agents=agents,
        rounds=3,
        config=config,
        safety_monitors=[
            sandbag.process,
            deception.process,
            behavior.process,
            ethics.process,
        ],
    )

    debate.assign_positions({
        "pro": "supports early programming education",
        "con": "opposes mandatory programming in elementary schools",
    })

    result = await debate.run()

    print(f"\nDebate completed. Verdict: {result.verdict.decision}")
    print(f"Total safety alerts: {len(result.safety_alerts)}")

    # Get alerts grouped by monitor
    alerts_by_monitor: dict[str, list] = {}
    for alert in result.safety_alerts:
        if alert.monitor not in alerts_by_monitor:
            alerts_by_monitor[alert.monitor] = []
        alerts_by_monitor[alert.monitor].append(alert)

    for monitor_name, alerts in alerts_by_monitor.items():
        print(f"\n{monitor_name}: {len(alerts)} alerts")
        for alert in alerts[:3]:  # Show first 3
            print(f"  - {alert.type}: severity={alert.severity:.2f}")

asyncio.run(run_comprehensive_safety())
```

## Next Steps

- See [Basic Debate](basic-debate.md) for debate fundamentals
- Create [LangGraph Workflows](langgraph-workflow.md) with safety integration
- Learn about [Ethical Dilemmas](ethical-dilemmas.md) for ethics-focused debates
