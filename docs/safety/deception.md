# Deception Monitoring

The Deception Monitor detects when agents make false claims, misrepresent information, or attempt to mislead.

## What is Deception?

Deception in debates includes:

- **Factual Falsity**: Making claims that are demonstrably false
- **Logical Fallacies**: Using invalid reasoning to mislead
- **Misrepresentation**: Distorting sources or opponent positions
- **Selective Omission**: Hiding relevant information
- **Misdirection**: Distracting from key issues

## Detection Capabilities

The Deception Monitor checks multiple dimensions:

| Dimension | What It Checks |
|-----------|----------------|
| Factual | Are claims consistent and plausible? |
| Logical | Is reasoning valid? |
| Consistency | Do claims contradict each other? |
| Source | Are sources accurately represented? |
| Context | Is context preserved? |

## Usage

### Basic Setup

```python
from artemis.safety import DeceptionMonitor, MonitorMode

monitor = DeceptionMonitor(
    mode=MonitorMode.PASSIVE,
    sensitivity=0.6,
)

debate = Debate(
    topic="Your topic",
    agents=agents,
    safety_monitors=[monitor.process],
)
```

### Configuration Options

```python
monitor = DeceptionMonitor(
    mode=MonitorMode.PASSIVE,    # PASSIVE, ACTIVE, or LEARNING
    sensitivity=0.6,              # 0.0 to 1.0
)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | MonitorMode | PASSIVE | Monitor mode |
| `sensitivity` | float | 0.5 | Detection sensitivity (0-1) |

## What It Detects

### Logical Fallacies

Common fallacies detected:

| Fallacy | Description |
|---------|-------------|
| Ad Hominem | Attacking the person, not the argument |
| Straw Man | Misrepresenting opponent's position |
| False Dichotomy | Presenting only two options when more exist |
| Appeal to Authority | Using authority as sole justification |
| Circular Reasoning | Conclusion restates the premise |
| Red Herring | Introducing irrelevant information |
| Slippery Slope | Assuming inevitable chain of events |
| Hasty Generalization | Drawing broad conclusions from few examples |

### Consistency Issues

- Internal contradictions within an agent's arguments
- Position shifts that contradict earlier statements
- Conflicting evidence claims

### Misrepresentation

- Distorting opponent's position
- Taking sources out of context
- Selective quoting

## Results

The monitor contributes to debate safety alerts:

```python
result = await debate.run()

# Check for deception alerts
for alert in result.safety_alerts:
    if "deception" in alert.type.lower():
        print(f"Agent: {alert.agent}")
        print(f"Severity: {alert.severity:.0%}")
```

## Distinguishing Intent

Not all false claims are intentional deception:

| Type | Description | Severity |
|------|-------------|----------|
| Mistake | Unintentional error | Low |
| Negligence | Careless claim | Medium |
| Deception | Intentional misleading | High |

## Integration

### With Debate

```python
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.safety import DeceptionMonitor, MonitorMode

agents = [
    Agent(name="pro", role="Advocate for the proposition", model="gpt-4o"),
    Agent(name="con", role="Advocate against the proposition", model="gpt-4o"),
]

monitor = DeceptionMonitor(
    mode=MonitorMode.PASSIVE,
    sensitivity=0.6,
)

debate = Debate(
    topic="Your topic",
    agents=agents,
    safety_monitors=[monitor.process],
)

debate.assign_positions({
    "pro": "supports the proposition",
    "con": "opposes the proposition",
})

result = await debate.run()

# Check for deception alerts
deception_alerts = [
    a for a in result.safety_alerts
    if "deception" in a.type.lower()
]

for alert in deception_alerts:
    print(f"Agent: {alert.agent}")
    print(f"Severity: {alert.severity:.0%}")
```

### With Other Monitors

```python
from artemis.safety import (
    DeceptionMonitor,
    SandbagDetector,
    EthicsGuard,
    MonitorMode,
    EthicsConfig,
)

deception = DeceptionMonitor(mode=MonitorMode.PASSIVE, sensitivity=0.6)
sandbag = SandbagDetector(mode=MonitorMode.PASSIVE, sensitivity=0.7)
ethics = EthicsGuard(
    mode=MonitorMode.PASSIVE,
    config=EthicsConfig(harmful_content_threshold=0.5),
)

debate = Debate(
    topic="Your topic",
    agents=agents,
    safety_monitors=[
        deception.process,
        sandbag.process,
        ethics.process,
    ],
)
```

## Sensitivity Tuning

### Low Sensitivity (0.3)

- Catches only obvious deception
- Few false positives
- May miss subtle cases

### Medium Sensitivity (0.6)

- Balanced detection
- Some false positives
- Good general setting

### High Sensitivity (0.8)

- Catches subtle deception
- More false positives
- Good for high-stakes scenarios

## Best Practices

1. **Enable comprehensive monitoring**: Combine with other monitors
2. **Track consistency**: Many deceptions are revealed by contradictions
3. **Consider intent**: Not all false claims are deceptive
4. **Review edge cases**: Some content needs human judgment
5. **Combine with ethics**: Deception often accompanies ethical violations

## Next Steps

- Learn about [Sandbagging Detection](sandbagging.md)
- Explore [Behavior Tracking](behavior.md)
- Configure [Ethics Guard](ethics-guard.md)
