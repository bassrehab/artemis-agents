# Ethics Guard

The Ethics Guard monitors debates for ethical boundary violations, ensuring arguments remain within acceptable moral limits.

## Overview

The Ethics Guard detects:

- **Harmful Content**: Arguments promoting harm
- **Discrimination**: Unfair treatment of groups
- **Manipulation**: Psychological manipulation tactics
- **Privacy Violations**: Exposing private information
- **Deceptive Claims**: Intentionally false statements

## Usage

### Basic Setup

```python
from artemis.safety import EthicsGuard, MonitorMode, EthicsConfig

guard = EthicsGuard(
    mode=MonitorMode.PASSIVE,
    config=EthicsConfig(
        harmful_content_threshold=0.4,
        bias_threshold=0.4,
        fairness_threshold=0.3,
        enabled_checks=["harmful_content", "bias", "fairness"],
    ),
)

debate = Debate(
    topic="Your topic",
    agents=agents,
    safety_monitors=[guard.process],
)
```

### Configuration Options

```python
from artemis.safety import EthicsGuard, MonitorMode, EthicsConfig

config = EthicsConfig(
    harmful_content_threshold=0.4,
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
    mode=MonitorMode.PASSIVE,
    config=config,
)
```

### EthicsConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `harmful_content_threshold` | float | 0.5 | Threshold for harmful content |
| `bias_threshold` | float | 0.4 | Threshold for bias detection |
| `fairness_threshold` | float | 0.3 | Threshold for fairness violations |
| `enabled_checks` | list[str] | default set | Ethics checks to enable |

## Ethical Principles

### Built-in Principles

| Principle | Description | Detects |
|-----------|-------------|---------|
| Fairness | Equal treatment | Discrimination, bias |
| Transparency | Honest communication | Hidden agendas, misdirection |
| Non-harm | Avoiding harm | Violence, dangerous advice |
| Respect | Dignified treatment | Insults, dehumanization |
| Accuracy | Truthful claims | Misinformation, false claims |

### Custom Checks

You can specify which checks to enable:

```python
config = EthicsConfig(
    harmful_content_threshold=0.3,
    enabled_checks=["harmful_content", "bias"],  # Only these two
)
```

## Detection Categories

### Harmful Content

Content that promotes or glorifies harm:

- Violence advocacy
- Self-harm promotion
- Dangerous activities
- Harmful advice

### Discrimination

Unfair treatment based on protected characteristics:

- Racial discrimination
- Gender discrimination
- Religious discrimination
- Age discrimination
- Disability discrimination

### Manipulation

Psychological manipulation tactics:

- Fear mongering
- Guilt tripping
- Gaslighting language
- Coercion
- Emotional exploitation

### Privacy Violations

Exposure of private information:

- Personal identification
- Location disclosure
- Financial information
- Health information

## Results

The Ethics Guard contributes to debate safety alerts:

```python
result = await debate.run()

# Check for ethics alerts
for alert in result.safety_alerts:
    if "ethics" in alert.type.lower():
        print(f"Agent: {alert.agent}")
        print(f"Severity: {alert.severity:.0%}")
```

## Integration

### With Debate

```python
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.safety import EthicsGuard, MonitorMode, EthicsConfig

agents = [
    Agent(name="pro", role="Advocate for the proposition", model="gpt-4o"),
    Agent(name="con", role="Advocate against the proposition", model="gpt-4o"),
]

guard = EthicsGuard(
    mode=MonitorMode.PASSIVE,
    config=EthicsConfig(harmful_content_threshold=0.4),
)

debate = Debate(
    topic="Your topic",
    agents=agents,
    safety_monitors=[guard.process],
)

debate.assign_positions({
    "pro": "supports the proposition",
    "con": "opposes the proposition",
})

result = await debate.run()

# Check for ethics violations
ethics_alerts = [
    a for a in result.safety_alerts
    if "ethics" in a.type.lower()
]

for alert in ethics_alerts:
    print(f"Agent: {alert.agent}")
    print(f"Severity: {alert.severity:.0%}")
```

### With Other Monitors

```python
from artemis.safety import (
    EthicsGuard,
    DeceptionMonitor,
    BehaviorTracker,
    MonitorMode,
    EthicsConfig,
)

ethics = EthicsGuard(
    mode=MonitorMode.PASSIVE,
    config=EthicsConfig(harmful_content_threshold=0.4),
)
deception = DeceptionMonitor(mode=MonitorMode.PASSIVE, sensitivity=0.6)
behavior = BehaviorTracker(mode=MonitorMode.PASSIVE, sensitivity=0.5)

debate = Debate(
    topic="Your topic",
    agents=agents,
    safety_monitors=[
        ethics.process,
        deception.process,
        behavior.process,
    ],
)
```

## Sensitivity Tuning

### Low Sensitivity (0.3)

- Only catches severe violations
- Minimal false positives
- Allows controversial but not harmful content

### Medium Sensitivity (0.6)

- Catches most concerning content
- Balanced false positive rate
- Good general setting

### High Sensitivity (0.9)

- Very strict enforcement
- More false positives
- For sensitive contexts

## Common Configurations

### Academic Debate

```python
guard = EthicsGuard(
    mode=MonitorMode.PASSIVE,
    config=EthicsConfig(
        harmful_content_threshold=0.5,
        bias_threshold=0.5,
        enabled_checks=["harmful_content", "bias", "fairness"],
    ),
)
```

### Sensitive Topics

```python
guard = EthicsGuard(
    mode=MonitorMode.ACTIVE,  # Can halt debate
    config=EthicsConfig(
        harmful_content_threshold=0.2,  # Very strict
        bias_threshold=0.3,
        enabled_checks=["harmful_content", "bias", "fairness", "privacy"],
    ),
)
```

### Policy Debate

```python
guard = EthicsGuard(
    mode=MonitorMode.PASSIVE,
    config=EthicsConfig(
        harmful_content_threshold=0.4,
        bias_threshold=0.4,
        enabled_checks=["harmful_content", "bias", "fairness"],
    ),
)
```

## Best Practices

1. **Set appropriate sensitivity**: Match to debate context
2. **Define clear principles**: Be explicit about boundaries
3. **Use passive mode initially**: Understand patterns before blocking
4. **Review edge cases**: Some content needs human judgment
5. **Document decisions**: Track why alerts were generated

## Next Steps

- Learn about [Safety Overview](overview.md)
- Explore [Deception Monitoring](deception.md)
- Configure [Behavior Tracking](behavior.md)
