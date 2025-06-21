# Ethics Module

ARTEMIS includes an ethics module that ensures debates remain within ethical boundaries and arguments are evaluated for moral soundness.

## Overview

The ethics module operates at three levels:

1. **Evaluation**: Ethical criteria are weighted in argument scoring
2. **Monitoring**: EthicsGuard detects ethical boundary violations
3. **Jury Perspective**: Ethical jury perspective considers moral implications

## Ethical Principles

ARTEMIS is built on core ethical principles:

| Principle | Description |
|-----------|-------------|
| **Fairness** | Arguments shouldn't discriminate or show bias |
| **Transparency** | Reasoning should be clear and honest |
| **Non-harm** | Arguments shouldn't advocate for harmful actions |
| **Respect** | Maintain respect for persons and values |
| **Accuracy** | Claims should be truthful and verifiable |

## Ethics Guard

The `EthicsGuard` monitors arguments for ethical violations:

```python
from artemis.safety import EthicsGuard, MonitorMode, EthicsConfig

# Configure ethics guard
ethics_config = EthicsConfig(
    sensitivity=0.7,
    principles=["fairness", "transparency", "non-harm"],
)

guard = EthicsGuard(
    mode=MonitorMode.PASSIVE,
    config=ethics_config,
)

# Use in debate
debate = Debate(
    topic="Your topic",
    agents=agents,
    safety_monitors=[guard.process],
)
```

### EthicsConfig Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `sensitivity` | float | 0.5 | Detection sensitivity (0-1) |
| `principles` | list | default | Ethical principles to enforce |

### Sensitivity Levels

```python
from artemis.safety import EthicsConfig

# Low sensitivity - only severe violations
low_config = EthicsConfig(sensitivity=0.3)

# Medium sensitivity - most violations
medium_config = EthicsConfig(sensitivity=0.6)

# High sensitivity - strict enforcement
high_config = EthicsConfig(sensitivity=0.9)
```

## Ethical Evaluation in L-AE-CR

Arguments receive an ethics score as part of the adaptive evaluation:

```python
from artemis.core.types import EvaluationCriteria

# Ethical alignment is one of the default criteria
criteria = EvaluationCriteria(
    logical_coherence=0.25,
    evidence_quality=0.25,
    causal_reasoning=0.20,
    ethical_alignment=0.15,  # Ethics weight
    persuasiveness=0.15,
)
```

### What Ethical Evaluation Considers

- **Claim Fairness**: Are claims fair to all parties?
- **Evidence Ethics**: Is evidence used responsibly?
- **Conclusion Ethics**: Are conclusions ethically sound?
- **Stakeholder Impact**: Who is affected and how?

## Ethical Jury Perspective

The jury includes an ethical perspective:

```python
from artemis.core.types import JuryPerspective

# The ETHICAL perspective focuses on moral implications
JuryPerspective.ETHICAL
```

When creating a jury panel, the ethical perspective is automatically included:

```python
from artemis.core.jury import JuryPanel

# Create panel - ethical perspective is assigned to one juror
panel = JuryPanel(evaluators=5, model="gpt-4o")

# The ethical juror focuses on:
# - Consideration of ethical principles
# - Attention to stakeholder welfare
# - Fairness and justice concerns
# - Long-term societal impact
```

## Handling Ethical Debates

ARTEMIS can handle debates on complex ethical topics:

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.jury import JuryPanel

async def ethical_debate():
    # Create agents for ethical debate
    agents = [
        Agent(
            name="utilitarian",
            role="Advocate arguing from consequentialist ethics",
            model="gpt-4o",
        ),
        Agent(
            name="deontologist",
            role="Advocate arguing from duty-based ethics",
            model="gpt-4o",
        ),
    ]

    # Create jury - will include ethical perspective
    jury = JuryPanel(evaluators=5, model="gpt-4o")

    debate = Debate(
        topic="Should autonomous vehicles prioritize passenger or pedestrian safety?",
        agents=agents,
        jury=jury,
        rounds=3,
    )

    debate.assign_positions({
        "utilitarian": "maximize overall welfare in collision scenarios",
        "deontologist": "respect individual rights regardless of outcomes",
    })

    result = await debate.run()
    return result

asyncio.run(ethical_debate())
```

## Safety Integration

Ethics monitoring integrates with other safety monitors:

```python
from artemis.safety import (
    SandbagDetector,
    DeceptionMonitor,
    EthicsGuard,
    MonitorMode,
    EthicsConfig,
)

# Configure all monitors
sandbag = SandbagDetector(mode=MonitorMode.PASSIVE, sensitivity=0.7)
deception = DeceptionMonitor(mode=MonitorMode.PASSIVE, sensitivity=0.6)
ethics = EthicsGuard(
    mode=MonitorMode.PASSIVE,
    config=EthicsConfig(sensitivity=0.7),
)

# Use all monitors together
debate = Debate(
    topic="Your topic",
    agents=agents,
    safety_monitors=[
        sandbag.process,
        deception.process,
        ethics.process,
    ],
)
```

## Safety Alerts

When ethical violations are detected, they appear in the debate results:

```python
result = await debate.run()

# Check for ethical safety alerts
for alert in result.safety_alerts:
    if "ethics" in alert.type.lower():
        print(f"Ethics alert: {alert.type}")
        print(f"Severity: {alert.severity:.0%}")
        print(f"Agent: {alert.agent}")
```

## Best Practices

1. **Set Appropriate Sensitivity**: Match sensitivity to debate context
2. **Define Clear Principles**: Be explicit about ethical boundaries
3. **Include Ethical Perspectives**: Use jury to consider moral implications
4. **Enable Transparency**: Make ethics decisions explainable
5. **Review Edge Cases**: Some arguments may need human review

## Ethical Frameworks

When debating ethical topics, different ethical frameworks provide different perspectives:

| Framework | Focus | What It Values |
|-----------|-------|----------------|
| **Utilitarian** | Greatest good | Outcomes, consequences |
| **Deontological** | Rules and duties | Principles, rights |
| **Virtue Ethics** | Character | Intentions, virtues |
| **Care Ethics** | Relationships | Context, relationships |

These frameworks can be represented by different agents in a debate:

```python
agents = [
    Agent(
        name="consequentialist",
        role="Argues from utilitarian perspective focusing on outcomes",
        model="gpt-4o",
    ),
    Agent(
        name="deontologist",
        role="Argues from duty-based perspective focusing on principles",
        model="gpt-4o",
    ),
    Agent(
        name="virtue_ethicist",
        role="Argues from virtue ethics focusing on character",
        model="gpt-4o",
    ),
]
```

## Next Steps

- Learn about [Safety Monitoring](../safety/overview.md) for broader safety
- See how ethics integrates with [Jury Mechanism](jury.md)
- Explore [Ethics Guard](../safety/ethics-guard.md) in depth
