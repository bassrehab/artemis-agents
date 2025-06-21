# Behavior Tracking

The Behavior Tracker monitors how agent behavior changes over time, detecting drift, inconsistencies, and concerning patterns.

## What It Tracks

Behavior tracking monitors:

- **Style Drift**: Changes in communication style
- **Position Drift**: Shifts in argued position
- **Capability Drift**: Changes in demonstrated abilities
- **Engagement Patterns**: How agents interact

## Why Track Behavior?

Behavior tracking helps detect:

1. **Goal Drift**: Agent straying from assigned position
2. **Strategic Shifts**: Concerning strategic changes
3. **Manipulation**: Gradual influence by opponent
4. **Degradation**: Performance declining over time

## Usage

### Basic Setup

```python
from artemis.safety import BehaviorTracker, MonitorMode

tracker = BehaviorTracker(
    mode=MonitorMode.PASSIVE,
    sensitivity=0.5,
    window_size=5,
)

debate = Debate(
    topic="Your topic",
    agents=agents,
    safety_monitors=[tracker.process],
)
```

### Configuration Options

```python
tracker = BehaviorTracker(
    mode=MonitorMode.PASSIVE,     # PASSIVE, ACTIVE, or LEARNING
    sensitivity=0.5,               # 0.0 to 1.0
    window_size=5,                 # Turns to consider for drift detection
    drift_threshold=0.3,           # Threshold for drift alerts
)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | MonitorMode | PASSIVE | Monitor mode |
| `sensitivity` | float | 0.5 | Detection sensitivity (0-1) |
| `window_size` | int | 5 | Turns to consider |
| `drift_threshold` | float | 0.3 | Drift detection threshold |

## Tracked Metrics

### Style Metrics

How the agent communicates:

- Formality level
- Aggression/assertiveness
- Complexity
- Sentiment
- Confidence

### Position Metrics

What position the agent argues:

- Position alignment with assigned stance
- Position strength
- Concession rate
- Counter-argument engagement

### Capability Metrics

How capable the agent appears:

- Vocabulary complexity
- Reasoning depth
- Evidence usage
- Argument structure

## Drift Detection

### Drift Types

| Drift Type | Description | Concern Level |
|------------|-------------|---------------|
| Gradual | Slow, steady change | Low |
| Sudden | Abrupt shift | High |
| Oscillating | Back and forth | Medium |
| Escalating | Increasing severity | High |

## Results

The tracker contributes to debate safety alerts:

```python
result = await debate.run()

# Check for behavior drift alerts
for alert in result.safety_alerts:
    if "behavior" in alert.type.lower() or "drift" in alert.type.lower():
        print(f"Agent: {alert.agent}")
        print(f"Severity: {alert.severity:.0%}")
```

## Integration

### With Debate

```python
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.safety import BehaviorTracker, MonitorMode

agents = [
    Agent(name="pro", role="Advocate for the proposition", model="gpt-4o"),
    Agent(name="con", role="Advocate against the proposition", model="gpt-4o"),
]

tracker = BehaviorTracker(
    mode=MonitorMode.PASSIVE,
    sensitivity=0.5,
    window_size=5,
)

debate = Debate(
    topic="Your topic",
    agents=agents,
    safety_monitors=[tracker.process],
)

debate.assign_positions({
    "pro": "supports the proposition",
    "con": "opposes the proposition",
})

result = await debate.run()

# Check for behavior alerts
behavior_alerts = [
    a for a in result.safety_alerts
    if "behavior" in a.type.lower()
]

for alert in behavior_alerts:
    print(f"Agent {alert.agent}: {alert.severity:.0%} severity")
```

### With Other Monitors

Behavior tracking complements other monitors:

```python
from artemis.safety import (
    BehaviorTracker,
    SandbagDetector,
    DeceptionMonitor,
    MonitorMode,
)

behavior = BehaviorTracker(mode=MonitorMode.PASSIVE, sensitivity=0.5, window_size=5)
sandbag = SandbagDetector(mode=MonitorMode.PASSIVE, sensitivity=0.7)
deception = DeceptionMonitor(mode=MonitorMode.PASSIVE, sensitivity=0.6)

debate = Debate(
    topic="Your topic",
    agents=agents,
    safety_monitors=[
        behavior.process,
        sandbag.process,
        deception.process,
    ],
)
```

## Sensitivity Tuning

### Low Sensitivity (0.3)

- Only catches severe drift
- Minimal false positives
- May miss gradual changes

### Medium Sensitivity (0.5)

- Balanced detection
- Catches most concerning drift
- Good general setting

### High Sensitivity (0.8)

- Catches subtle changes
- More false positives
- For high-stakes scenarios

## Best Practices

1. **Set appropriate window**: 5-7 turns typically works well
2. **Account for natural variation**: Some drift is normal
3. **Consider debate phase**: Closing arguments differ from opening
4. **Monitor all agents**: Compare behavior across participants
5. **Correlate with other signals**: Drift may accompany other issues

## Next Steps

- Learn about [Sandbagging Detection](sandbagging.md)
- Explore [Deception Monitoring](deception.md)
- Configure [Ethics Guard](ethics-guard.md)
