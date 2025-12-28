# Safety Monitors Example

This example demonstrates how to use ARTEMIS safety monitors to detect problematic agent behaviors.

## Basic Safety Setup

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.safety import (
    MonitorMode,
    SafetyManager,
    SandbagDetector,
    DeceptionMonitor,
    BehaviorTracker,
    EthicsGuard,
)

async def run_safe_debate():
    # Create safety manager
    safety = SafetyManager(mode=MonitorMode.PASSIVE)

    # Add monitors with actual API parameters
    safety.add_monitor(SandbagDetector(
        mode=MonitorMode.PASSIVE,
        sensitivity=0.7,
        baseline_turns=3,
        drop_threshold=0.3,
    ))
    safety.add_monitor(DeceptionMonitor(
        mode=MonitorMode.PASSIVE,
        sensitivity=0.6,
    ))
    safety.add_monitor(BehaviorTracker(
        mode=MonitorMode.PASSIVE,
        sensitivity=0.5,
        baseline_turns=3,
        drift_threshold=0.25,
    ))
    safety.add_monitor(EthicsGuard(
        mode=MonitorMode.PASSIVE,
        sensitivity=0.5,
    ))

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
    # Note: We pass monitor callbacks to safety_monitors parameter
    debate = Debate(
        topic="Should facial recognition be used in public spaces?",
        agents=agents,
        rounds=3,
        safety_monitors=[m.process for m in safety.monitors],
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

    # Get risk summary from manager
    risk_summary = safety.get_risk_summary()
    print("\nRisk Summary by Agent:")
    for agent, stats in risk_summary.items():
        print(f"  {agent}: avg_risk={stats.get('avg_risk', 0):.2f}")

asyncio.run(run_safe_debate())
```

## Using Composite Monitor

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.safety import (
    CompositeMonitor,
    MonitorMode,
    SandbagDetector,
    DeceptionMonitor,
)

async def run_with_composite():
    # Create composite monitor with multiple sub-monitors
    composite = CompositeMonitor(
        monitors=[
            SandbagDetector(sensitivity=0.6),
            DeceptionMonitor(sensitivity=0.6),
        ],
        aggregation="max",  # Options: "max", "mean", "sum"
    )

    agents = [
        Agent(name="pro", role="Advocate for the topic", model="gpt-4o"),
        Agent(name="con", role="Critic of the topic", model="gpt-4o"),
    ]

    debate = Debate(
        topic="Is genetic engineering of humans ethical?",
        agents=agents,
        rounds=3,
        safety_monitors=[composite.process],
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

asyncio.run(run_with_composite())
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

    sandbagging_alerts = [a for a in result.safety_alerts if a.type == "sandbagging"]

    if sandbagging_alerts:
        for alert in sandbagging_alerts:
            print(f"\nAgent: {alert.agent}")
            print(f"Severity: {alert.severity:.2f}")
            for indicator in alert.indicators:
                print(f"  Signal: {indicator.type.value}")
                print(f"  Evidence: {indicator.evidence}")
    else:
        print("No sandbagging detected.")

    # Get baseline info
    for agent_name in ["agent_a", "agent_b"]:
        baseline = detector.get_agent_baseline(agent_name)
        if baseline:
            print(f"\n{agent_name} Baseline:")
            print(f"  Turns analyzed: {baseline.turn_count}")
            print(f"  Avg complexity: {baseline.avg_complexity:.2f}")
            print(f"  Max complexity: {baseline.max_complexity:.2f}")

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
        fallacy_weight=0.3,  # Weight for logical fallacy detection
        manipulation_weight=0.4,  # Weight for manipulation detection
        contradiction_weight=0.5,  # Weight for self-contradiction detection
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

    deception_alerts = [a for a in result.safety_alerts if a.type == "deception"]

    for alert in deception_alerts:
        print(f"\nAgent: {alert.agent}")
        print(f"Severity: {alert.severity:.2f}")

        for indicator in alert.indicators:
            print(f"  Type: {indicator.type.value}")
            print(f"  Evidence: {indicator.evidence}")
            if indicator.metadata:
                print(f"  Signal: {indicator.metadata.get('signal', 'N/A')}")

    # Get deception summary per agent
    for agent_name in ["claimant", "challenger"]:
        summary = monitor.get_deception_summary(agent_name)
        print(f"\n{agent_name} Summary:")
        print(f"  Fallacies detected: {summary['fallacies']}")
        print(f"  Contradictions: {summary['contradictions']}")
        print(f"  Total claims tracked: {summary['claims']}")

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
        harmful_content_threshold=0.5,
        bias_threshold=0.4,
        fairness_threshold=0.3,
        enabled_checks=[
            "harmful_content",
            "bias",
            "fairness",
            "privacy",
            "manipulation",
        ],
    )

    guard = EthicsGuard(
        mode=MonitorMode.ACTIVE,  # Can halt on severe violations
        sensitivity=0.7,
        ethics_config=ethics_config,
        halt_on_violation=True,  # Halt on severe ethics violations
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

    try:
        result = await debate.run()

        # Analyze ethics alerts
        print("ETHICS ANALYSIS")
        print("=" * 60)

        ethics_alerts = [a for a in result.safety_alerts if "ethics" in a.type]

        for alert in ethics_alerts:
            print(f"\nAgent: {alert.agent}")
            print(f"Severity: {alert.severity:.2f}")
            for indicator in alert.indicators:
                print(f"  Violation: {indicator.metadata.get('violation', 'N/A')}")
                print(f"  Evidence: {indicator.evidence}")

        # Get ethics summary
        for agent_name in ["security", "privacy"]:
            summary = guard.get_agent_ethics_summary(agent_name)
            print(f"\n{agent_name} Ethics Summary:")
            print(f"  Total violations: {summary['total_violations']}")
            print(f"  Max severity: {summary['max_severity']:.2f}")
            print(f"  Violation types: {summary['violation_types']}")

    except Exception as e:
        print(f"Debate halted: {e}")

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
        baseline_turns=3,
        drift_threshold=0.25,  # Threshold for style drift detection
        aggression_threshold=0.6,  # Threshold for aggression alerts
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

    # Get behavior summaries
    print("BEHAVIOR TRACKING")
    print("=" * 60)

    for agent in agents:
        print(f"\n{agent.name.upper()} Behavior Profile:")
        summary = tracker.get_drift_summary(agent.name)

        print(f"  Snapshots collected: {summary['snapshots']}")
        print(f"  Drift events detected: {summary['drift_events']}")
        print(f"  Avg aggression: {summary['avg_aggression']:.2f}")
        print(f"  Avg cooperation: {summary['avg_cooperation']:.2f}")
        print(f"  Avg formality: {summary.get('avg_formality', 0.5):.2f}")

        # Get detailed profile if available
        profile = tracker.get_agent_profile(agent.name)
        if profile and profile.drift_events:
            print("  Drift Events:")
            for round_num, signal, severity in profile.drift_events:
                print(f"    Round {round_num}: {signal} (severity: {severity:.2f})")

    # Show drift alerts
    drift_alerts = [a for a in result.safety_alerts if "drift" in a.type]
    if drift_alerts:
        print("\nDrift Alerts:")
        for alert in drift_alerts:
            print(f"  Agent {alert.agent}: severity={alert.severity:.2f}")

asyncio.run(track_behavior())
```

## Active Mode with Halt Capability

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate, DebateConfig, DebateHaltedError
from artemis.safety import (
    MonitorMode,
    SandbagDetector,
    DeceptionMonitor,
    CompositeMonitor,
)

async def run_with_active_monitoring():
    # Create monitors in active mode - they can halt the debate
    composite = CompositeMonitor(
        monitors=[
            SandbagDetector(
                mode=MonitorMode.ACTIVE,
                sensitivity=0.8,
            ),
            DeceptionMonitor(
                mode=MonitorMode.ACTIVE,
                sensitivity=0.7,
            ),
        ],
        aggregation="max",
        mode=MonitorMode.ACTIVE,  # Composite is also active
    )

    agents = [
        Agent(name="advocate", role="Policy advocate", model="gpt-4o"),
        Agent(name="critic", role="Policy critic", model="gpt-4o"),
    ]

    # Enable halt on safety violation in config
    config = DebateConfig(
        halt_on_safety_violation=True,
    )

    debate = Debate(
        topic="Should AI systems be allowed to make medical diagnoses?",
        agents=agents,
        rounds=3,
        config=config,
        safety_monitors=[composite.process],
    )

    debate.assign_positions({
        "advocate": "supports AI medical diagnosis",
        "critic": "opposes AI medical diagnosis",
    })

    try:
        result = await debate.run()
        print(f"Debate completed. Verdict: {result.verdict.decision}")

        # Show any alerts that didn't cause halt
        if result.safety_alerts:
            print(f"\nWarnings raised: {len(result.safety_alerts)}")
            for alert in result.safety_alerts:
                print(f"  - {alert.type}: {alert.severity:.2f}")

    except DebateHaltedError as e:
        print(f"Debate was halted!")
        print(f"Reason: {e}")
        print(f"Alert details: {e.alert}")

asyncio.run(run_with_active_monitoring())
```

## Monitor Registry Pattern

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.safety import (
    MonitorMode,
    MonitorRegistry,
    SandbagDetector,
    DeceptionMonitor,
    BehaviorTracker,
    EthicsGuard,
)

async def run_with_registry():
    # Use registry for centralized monitor management
    registry = MonitorRegistry()

    # Register all monitors
    registry.register(SandbagDetector(sensitivity=0.6))
    registry.register(DeceptionMonitor(sensitivity=0.6))
    registry.register(BehaviorTracker(sensitivity=0.5))
    registry.register(EthicsGuard(sensitivity=0.5))

    print(f"Registered {len(registry)} monitors:")
    for monitor in registry:
        print(f"  - {monitor.name} ({monitor.monitor_type})")

    agents = [
        Agent(name="pro", role="Proposition advocate", model="gpt-4o"),
        Agent(name="con", role="Opposition advocate", model="gpt-4o"),
    ]

    # Create debate with all registered monitors
    debate = Debate(
        topic="Should programming be taught in elementary schools?",
        agents=agents,
        rounds=3,
        safety_monitors=[m.process for m in registry.get_enabled()],
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

    # Reset all monitors for next debate
    registry.reset_all()

asyncio.run(run_with_registry())
```

## Next Steps

- See [Basic Debate](basic-debate.md) for debate fundamentals
- Create [LangGraph Workflows](langgraph-workflow.md) with safety integration
- Learn about [Ethical Dilemmas](ethical-dilemmas.md) for ethics-focused debates
