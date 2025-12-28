# Safety Monitoring Overview

ARTEMIS includes comprehensive safety monitoring to detect and prevent problematic agent behaviors during debates. This is a key differentiator from other multi-agent frameworks.

## Why Safety Monitoring?

LLM agents can exhibit concerning behaviors:

- **Sandbagging**: Deliberately underperforming to appear less capable
- **Deception**: Making false claims or hiding information
- **Behavioral Drift**: Gradually shifting behavior over time
- **Ethical Violations**: Crossing ethical boundaries

ARTEMIS monitors for these behaviors in real-time.

## Safety Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      Safety Layer                              │
├────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
│  │  Sandbagging  │  │   Deception   │  │   Behavior    │       │
│  │   Detector    │  │    Monitor    │  │    Tracker    │       │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘       │
│          │                  │                  │               │
│          └──────────────────┼──────────────────┘               │
│                             │                                  │
│                    ┌────────▼────────┐                         │
│                    │ safety_monitors │                         │
│                    └────────┬────────┘                         │
│                             │                                  │
│  ┌───────────────┐  ┌──────▼───────┐  ┌───────────────┐        │
│  │    Ethics     │  │    Debate    │  │    Alerts     │        │
│  │     Guard     │──│  Integration │──│   (results)   │        │
│  └───────────────┘  └──────────────┘  └───────────────┘        │
└────────────────────────────────────────────────────────────────┘
```

## Available Monitors

| Monitor | Purpose | Detects |
|---------|---------|---------|
| [Sandbagging Detector](sandbagging.md) | Detect intentional underperformance | Capability hiding |
| [Deception Monitor](deception.md) | Detect false claims | Lies, misdirection |
| [Behavior Tracker](behavior.md) | Track behavioral changes | Drift, inconsistency |
| [Ethics Guard](ethics-guard.md) | Monitor ethical boundaries | Violations, harm |

## Quick Start

### Basic Safety Setup

```python
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.safety import (
    SandbagDetector,
    DeceptionMonitor,
    BehaviorTracker,
    EthicsGuard,
    MonitorMode,
    EthicsConfig,
)

# Create agents
agents = [
    Agent(name="pro", role="Advocate for the proposition", model="gpt-4o"),
    Agent(name="con", role="Advocate against the proposition", model="gpt-4o"),
]

# Create individual monitors
sandbag = SandbagDetector(
    mode=MonitorMode.PASSIVE,
    sensitivity=0.7,
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
    config=EthicsConfig(harmful_content_threshold=0.5),
)

# Create debate with safety monitors
debate = Debate(
    topic="Your topic",
    agents=agents,
    safety_monitors=[
        sandbag.process,
        deception.process,
        behavior.process,
        ethics.process,
    ],
)

debate.assign_positions({
    "pro": "supports the proposition",
    "con": "opposes the proposition",
})

result = await debate.run()

# Check safety alerts
for alert in result.safety_alerts:
    print(f"Alert: {alert.type}")
    print(f"Severity: {alert.severity}")
    print(f"Agent: {alert.agent}")
```

### Multiple Monitors

Combine multiple monitors for comprehensive safety:

```python
from artemis.safety import (
    SandbagDetector,
    DeceptionMonitor,
    BehaviorTracker,
    EthicsGuard,
    MonitorMode,
    EthicsConfig,
)

# Create all monitors
monitors = [
    SandbagDetector(mode=MonitorMode.PASSIVE, sensitivity=0.7),
    DeceptionMonitor(mode=MonitorMode.PASSIVE, sensitivity=0.6),
    BehaviorTracker(mode=MonitorMode.PASSIVE, sensitivity=0.5),
    EthicsGuard(mode=MonitorMode.PASSIVE, config=EthicsConfig(harmful_content_threshold=0.5)),
]

# Pass their process methods to the debate
debate = Debate(
    topic="Your topic",
    agents=agents,
    safety_monitors=[m.process for m in monitors],
)
```

## Monitor Modes

All monitors support three modes via the `MonitorMode` enum:

```python
from artemis.safety import MonitorMode

# Available modes
MonitorMode.PASSIVE   # Observe and report only
MonitorMode.ACTIVE    # Can intervene and halt debate
MonitorMode.LEARNING  # Learn patterns without alerting
```

### Passive Mode (Default)

Monitors observe and report but don't intervene:

```python
monitor = SandbagDetector(
    mode=MonitorMode.PASSIVE,
    sensitivity=0.7,
)
# Alerts are generated but debate continues
```

### Active Mode

Monitors can intervene and halt the debate:

```python
monitor = SandbagDetector(
    mode=MonitorMode.ACTIVE,
    sensitivity=0.7,
)
# Debate may halt if severe issues detected
```

### Learning Mode

Monitors learn patterns without generating alerts:

```python
monitor = BehaviorTracker(
    mode=MonitorMode.LEARNING,
    sensitivity=0.5,
)
# Gathers data for future reference
```

## Safety Results

Each monitor's `process` method is called during debate and can contribute to alerts:

```python
result = await debate.run()

# All alerts from all monitors
for alert in result.safety_alerts:
    print(f"Type: {alert.type}")
    print(f"Agent: {alert.agent}")
    print(f"Severity: {alert.severity:.0%}")
```

### Alert Severity Levels

| Level | Score Range | Description |
|-------|-------------|-------------|
| Low | 0.0 - 0.3 | Minor concern |
| Medium | 0.3 - 0.6 | Notable issue |
| High | 0.6 - 0.9 | Serious concern |
| Critical | 0.9 - 1.0 | Severe issue |

## Configuration

### Per-Monitor Settings

Each monitor has its own configuration options:

```python
# Sandbagging detector
sandbag = SandbagDetector(
    mode=MonitorMode.PASSIVE,
    sensitivity=0.7,
    baseline_turns=3,
    drop_threshold=0.3,
)

# Deception monitor
deception = DeceptionMonitor(
    mode=MonitorMode.PASSIVE,
    sensitivity=0.6,
)

# Behavior tracker
behavior = BehaviorTracker(
    mode=MonitorMode.PASSIVE,
    sensitivity=0.5,
    window_size=5,
    drift_threshold=0.3,
)

# Ethics guard
ethics = EthicsGuard(
    mode=MonitorMode.PASSIVE,
    config=EthicsConfig(
        harmful_content_threshold=0.5,
        bias_threshold=0.4,
        fairness_threshold=0.3,
        enabled_checks=["harmful_content", "bias", "fairness"],
    ),
)
```

## Accessing Safety Data

### After Debate

```python
result = await debate.run()

# All safety alerts
print(f"Total alerts: {len(result.safety_alerts)}")

for alert in result.safety_alerts:
    print(f"  {alert.type}: {alert.severity:.0%} - {alert.agent}")

# Filter by type
sandbagging_alerts = [a for a in result.safety_alerts if "sandbag" in a.type.lower()]
deception_alerts = [a for a in result.safety_alerts if "deception" in a.type.lower()]
```

## Best Practices

1. **Start with passive mode**: Understand behavior before enabling active intervention
2. **Tune sensitivity**: Adjust based on false positive rates
3. **Combine monitors**: Multiple monitors catch more issues
4. **Review alerts**: Verify detections before taking action
5. **Consider context**: Some patterns may be legitimate

## Next Steps

- Learn about [Sandbagging Detection](sandbagging.md)
- Understand [Deception Monitoring](deception.md)
- Explore [Behavior Tracking](behavior.md)
- Configure [Ethics Guard](ethics-guard.md)
