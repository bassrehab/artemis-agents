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
from artemis.safety import BehaviorTracker

tracker = BehaviorTracker(
    window_size=5,
    drift_threshold=0.3,
)

safety.add_monitor(tracker)
```

### Full Configuration

```python
tracker = BehaviorTracker(
    # Window settings
    window_size=5,           # Turns to consider
    min_samples=3,           # Minimum turns before tracking

    # What to track
    track_style=True,        # Track writing style
    track_position=True,     # Track argued position
    track_capability=True,   # Track capability metrics
    track_engagement=True,   # Track engagement patterns

    # Thresholds
    drift_threshold=0.3,     # Overall drift threshold
    style_threshold=0.25,    # Style change threshold
    position_threshold=0.4,  # Position change threshold
)
```

## Tracked Metrics

### Style Metrics

How the agent communicates:

```python
style_metrics = tracker.compute_style_metrics(turn)

print(f"Formality: {style_metrics['formality']}")
print(f"Aggression: {style_metrics['aggression']}")
print(f"Complexity: {style_metrics['complexity']}")
print(f"Sentiment: {style_metrics['sentiment']}")
print(f"Confidence: {style_metrics['confidence']}")
```

### Position Metrics

What position the agent argues:

```python
position_metrics = tracker.compute_position_metrics(turn)

print(f"Position alignment: {position_metrics['alignment']}")
print(f"Position strength: {position_metrics['strength']}")
print(f"Concession rate: {position_metrics['concessions']}")
print(f"Counter-argument engagement: {position_metrics['engagement']}")
```

### Capability Metrics

How capable the agent appears:

```python
capability_metrics = tracker.compute_capability_metrics(turn)

print(f"Vocabulary: {capability_metrics['vocabulary']}")
print(f"Reasoning: {capability_metrics['reasoning']}")
print(f"Evidence: {capability_metrics['evidence']}")
print(f"Structure: {capability_metrics['structure']}")
```

### Engagement Metrics

How the agent engages with debate:

```python
engagement_metrics = tracker.compute_engagement_metrics(turn)

print(f"Response relevance: {engagement_metrics['relevance']}")
print(f"Opponent acknowledgment: {engagement_metrics['acknowledgment']}")
print(f"Question answering: {engagement_metrics['question_response']}")
print(f"Proactive contribution: {engagement_metrics['proactivity']}")
```

## Drift Detection

### How Drift Is Calculated

```python
# Simplified drift calculation
def calculate_drift(
    self,
    current_metrics: dict,
    historical_metrics: list[dict],
) -> float:
    # Calculate mean of historical metrics
    historical_mean = {
        k: np.mean([h[k] for h in historical_metrics])
        for k in current_metrics
    }

    # Calculate drift as normalized difference
    drifts = [
        abs(current_metrics[k] - historical_mean[k])
        for k in current_metrics
    ]

    return np.mean(drifts)
```

### Drift Types

| Drift Type | Description | Concern Level |
|------------|-------------|---------------|
| Gradual | Slow, steady change | Low |
| Sudden | Abrupt shift | High |
| Oscillating | Back and forth | Medium |
| Escalating | Increasing severity | High |

### Detecting Drift Patterns

```python
drift_analysis = tracker.analyze_drift_pattern(agent_turns)

print(f"Drift type: {drift_analysis.pattern_type}")
print(f"Severity: {drift_analysis.severity}")
print(f"Direction: {drift_analysis.direction}")
print(f"Acceleration: {drift_analysis.acceleration}")
```

## Position Monitoring

### Position Alignment

Tracks whether agent maintains assigned position:

```python
# Check position alignment
alignment = tracker.check_position_alignment(
    turn=turn,
    assigned_position=agent.position,
)

print(f"Aligned: {alignment.is_aligned}")
print(f"Alignment score: {alignment.score}")
print(f"Drift direction: {alignment.drift_direction}")
```

### Concession Tracking

Monitors when agents concede points:

```python
concessions = tracker.track_concessions(agent_turns)

for concession in concessions:
    print(f"Round: {concession.round}")
    print(f"Point conceded: {concession.point}")
    print(f"Strategic: {concession.appears_strategic}")
```

## Style Monitoring

### Style Profile

Establishes a style profile for each agent:

```python
profile = tracker.get_style_profile(agent_name)

print(f"Average formality: {profile.avg_formality}")
print(f"Typical complexity: {profile.typical_complexity}")
print(f"Sentiment range: {profile.sentiment_range}")
```

### Style Deviation

Detects unusual style changes:

```python
deviation = tracker.detect_style_deviation(turn, agent_profile)

if deviation.is_significant:
    print(f"Significant style change detected:")
    print(f"  Changed: {deviation.changed_metrics}")
    print(f"  Magnitude: {deviation.magnitude}")
```

## Results

The tracker returns comprehensive results:

```python
result = await tracker.analyze(turn, context)

print(f"Is Safe: {result.is_safe}")
print(f"Drift Score: {result.score}")

details = result.details
print(f"Style drift: {details['style_drift']}")
print(f"Position drift: {details['position_drift']}")
print(f"Capability drift: {details['capability_drift']}")
print(f"Engagement drift: {details['engagement_drift']}")
print(f"Pattern: {details['drift_pattern']}")
```

## Visualization

Track behavior over time:

```python
# Get behavior history
history = tracker.get_history(agent_name)

# Plot drift over time
import matplotlib.pyplot as plt

plt.plot([h['round'] for h in history], [h['drift'] for h in history])
plt.xlabel('Round')
plt.ylabel('Drift Score')
plt.title(f'Behavior Drift: {agent_name}')
```

## Integration

### With Debate

```python
from artemis.core.debate import Debate
from artemis.safety import BehaviorTracker, SafetyManager

safety = SafetyManager()
safety.add_monitor(BehaviorTracker(window_size=5))

debate = Debate(
    topic="Your topic",
    agents=agents,
    safety_manager=safety,
)

result = await debate.run()

# Get behavior summary
for agent in agents:
    behavior_summary = result.get_behavior_summary(agent.name)
    print(f"{agent.name}: {behavior_summary}")
```

### With Other Monitors

Behavior tracking complements other monitors:

```python
from artemis.safety import (
    BehaviorTracker,
    SandbagDetector,
    DeceptionMonitor,
    CompositeMonitor,
)

# Combine for comprehensive monitoring
composite = CompositeMonitor(
    monitors=[
        BehaviorTracker(window_size=5),
        SandbagDetector(sensitivity=0.7),
        DeceptionMonitor(sensitivity=0.6),
    ],
    aggregation="max",
)
```

## Alerting

### Drift Alerts

```python
tracker = BehaviorTracker(
    drift_threshold=0.3,
    alert_on_sudden_change=True,
    sudden_change_threshold=0.5,
)

# Alert when:
# - Overall drift exceeds threshold
# - Sudden change detected
# - Position significantly shifts
```

### Custom Alerts

```python
tracker = BehaviorTracker(
    custom_alerts=[
        {
            "name": "aggression_spike",
            "metric": "style.aggression",
            "condition": "increase",
            "threshold": 0.4,
        },
        {
            "name": "position_reversal",
            "metric": "position.alignment",
            "condition": "decrease",
            "threshold": 0.5,
        },
    ]
)
```

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
