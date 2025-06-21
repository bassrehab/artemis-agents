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
        sensitivity=0.6,
        principles=["fairness", "transparency", "non-harm"],
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
    sensitivity=0.6,
    principles=[
        "fairness",
        "transparency",
        "non-harm",
        "respect",
        "accuracy",
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
| `sensitivity` | float | 0.5 | Detection sensitivity (0-1) |
| `principles` | list[str] | default set | Ethical principles to enforce |

## Ethical Principles

### Built-in Principles

| Principle | Description | Detects |
|-----------|-------------|---------|
| Fairness | Equal treatment | Discrimination, bias |
| Transparency | Honest communication | Hidden agendas, misdirection |
| Non-harm | Avoiding harm | Violence, dangerous advice |
| Respect | Dignified treatment | Insults, dehumanization |
| Accuracy | Truthful claims | Misinformation, false claims |

### Custom Principles

You can specify which principles to enforce:

```python
config = EthicsConfig(
    sensitivity=0.7,
    principles=["fairness", "non-harm"],  # Only these two
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
    config=EthicsConfig(sensitivity=0.6),
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
    config=EthicsConfig(sensitivity=0.6),
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
        sensitivity=0.5,
        principles=["accuracy", "fairness", "transparency"],
    ),
)
```

### Sensitive Topics

```python
guard = EthicsGuard(
    mode=MonitorMode.ACTIVE,  # Can halt debate
    config=EthicsConfig(
        sensitivity=0.8,
        principles=["non-harm", "respect", "fairness"],
    ),
)
```

### Policy Debate

```python
guard = EthicsGuard(
    mode=MonitorMode.PASSIVE,
    config=EthicsConfig(
        sensitivity=0.6,
        principles=["accuracy", "transparency", "fairness"],
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
