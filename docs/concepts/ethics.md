# Ethics Module

ARTEMIS includes an ethics module that ensures debates remain within ethical boundaries and arguments are evaluated for moral soundness.

## Overview

The ethics module operates at three levels:

1. **Generation**: Filters unethical argument content
2. **Evaluation**: Weights ethical criteria in scoring
3. **Monitoring**: Detects ethical boundary violations

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
from artemis.safety import EthicsGuard

guard = EthicsGuard(
    sensitivity=0.6,
    principles=["fairness", "transparency", "non-harm"],
)

result = await guard.analyze(argument, context)

if result.has_violation:
    print(f"Violation: {result.violation_type}")
    print(f"Severity: {result.severity}")
    print(f"Details: {result.details}")
```

### Violation Types

| Type | Description | Example |
|------|-------------|---------|
| `discrimination` | Unfair treatment of groups | Racist or sexist arguments |
| `deception` | Intentional misleading | False statistics |
| `harm_advocacy` | Promoting harmful actions | Violence endorsement |
| `privacy_violation` | Exposing private info | Doxxing attempts |
| `manipulation` | Psychological manipulation | Emotional exploitation |

### Sensitivity Levels

```python
# Low sensitivity - only severe violations
guard = EthicsGuard(sensitivity=0.3)

# Medium sensitivity - most violations
guard = EthicsGuard(sensitivity=0.6)

# High sensitivity - strict enforcement
guard = EthicsGuard(sensitivity=0.9)
```

## Ethical Evaluation

Arguments receive an ethics score as part of L-AE-CR evaluation:

```python
from artemis.core.evaluation import AdaptiveEvaluator

evaluator = AdaptiveEvaluator(
    criteria_weights={
        "logical_coherence": 0.25,
        "evidence_quality": 0.25,
        "argument_strength": 0.20,
        "ethical_considerations": 0.15,  # Ethics weight
        "causal_validity": 0.15,
    }
)
```

### Ethics Score Components

The ethics score evaluates:

1. **Claim Fairness**: Are claims fair to all parties?
2. **Evidence Ethics**: Is evidence obtained ethically?
3. **Conclusion Ethics**: Are conclusions ethically sound?
4. **Stakeholder Impact**: Who is affected and how?

```python
ethics_breakdown = evaluation.ethics_details

print(f"Claim Fairness: {ethics_breakdown.claim_fairness:.2f}")
print(f"Evidence Ethics: {ethics_breakdown.evidence_ethics:.2f}")
print(f"Conclusion Ethics: {ethics_breakdown.conclusion_ethics:.2f}")
print(f"Stakeholder Impact: {ethics_breakdown.stakeholder_impact:.2f}")
```

## Content Filtering

### Generation-Time Filtering

Arguments are filtered during generation:

```python
from artemis.core.agent import Agent

agent = Agent(
    name="ethical_agent",
    model="gpt-4o",
    ethics_filter=True,  # Enable filtering
    ethics_sensitivity=0.7,
)
```

### Filtered Content Types

- Personal attacks
- Discriminatory statements
- Misinformation claims
- Harmful recommendations
- Privacy violations

## Ethical Dilemmas

ARTEMIS can handle complex ethical debates:

```python
from artemis.core.debate import Debate

debate = Debate(
    topic="Should autonomous vehicles prioritize passenger or pedestrian safety?",
    agents=agents,
    ethical_framework="utilitarian",  # or "deontological", "virtue", "care"
)
```

### Ethical Frameworks

| Framework | Focus | Evaluation Priority |
|-----------|-------|---------------------|
| Utilitarian | Greatest good | Outcomes, consequences |
| Deontological | Rules and duties | Principles, rights |
| Virtue | Character | Intentions, virtues |
| Care | Relationships | Context, relationships |

### Multi-Framework Analysis

```python
from artemis.core.ethics import MultiFrameworkAnalysis

analysis = MultiFrameworkAnalysis(
    frameworks=["utilitarian", "deontological", "virtue"]
)

result = await analysis.evaluate(argument)

for framework, score in result.framework_scores.items():
    print(f"{framework}: {score:.2f}")

print(f"Overall Ethics: {result.overall_score:.2f}")
print(f"Framework Conflicts: {result.conflicts}")
```

## Stakeholder Analysis

ARTEMIS can identify and track stakeholder impact:

```python
from artemis.core.ethics import StakeholderAnalyzer

analyzer = StakeholderAnalyzer()

stakeholders = await analyzer.identify(argument)

for stakeholder in stakeholders:
    print(f"Group: {stakeholder.name}")
    print(f"Impact: {stakeholder.impact}")  # positive, negative, neutral
    print(f"Severity: {stakeholder.severity}")
    print(f"Considerations: {stakeholder.considerations}")
```

## Configuration

### Debate-Level Ethics

```python
from artemis.core.types import DebateConfig, EthicsConfig

ethics_config = EthicsConfig(
    enable_ethics_guard=True,
    sensitivity=0.7,
    principles=["fairness", "transparency", "non-harm"],
    ethical_framework="utilitarian",
    halt_on_violation=False,
    violation_threshold=0.9,
)

config = DebateConfig(
    ethics=ethics_config,
)
```

### Custom Principles

```python
from artemis.core.ethics import EthicalPrinciple

custom_principle = EthicalPrinciple(
    name="sustainability",
    description="Arguments should consider environmental sustainability",
    detection_keywords=["environment", "climate", "sustainable"],
    violation_patterns=[
        "ignores environmental impact",
        "dismisses climate concerns",
    ],
    weight=0.2,
)

guard = EthicsGuard(
    principles=["fairness", "non-harm", custom_principle],
)
```

## Ethics in Jury

Jury members can have ethical perspectives:

```python
from artemis.core.jury import JuryMember, PERSPECTIVES

ethical_juror = JuryMember(
    name="ethicist",
    perspective=PERSPECTIVES["ethical"],
)

# Or with custom ethical focus
from artemis.core.jury import Perspective

environmental_perspective = Perspective(
    name="environmental",
    description="Prioritizes environmental ethics and sustainability",
    criteria_adjustments={
        "ethical_considerations": 2.0,  # Double weight
    },
    custom_criteria={
        "environmental_impact": 0.3,
    },
)

eco_juror = JuryMember(
    name="environmentalist",
    perspective=environmental_perspective,
)
```

## Handling Violations

### Soft Handling

```python
guard = EthicsGuard(
    mode="warn",  # Just warn, don't block
)

result = await guard.analyze(argument, context)
if result.has_violation:
    logging.warning(f"Ethics warning: {result.details}")
    # Argument continues with warning attached
```

### Hard Handling

```python
guard = EthicsGuard(
    mode="block",  # Block violating content
    violation_threshold=0.8,
)

result = await guard.analyze(argument, context)
if result.has_violation and result.severity > guard.violation_threshold:
    raise EthicsViolationError(result.details)
```

### Remediation

```python
guard = EthicsGuard(
    mode="remediate",  # Attempt to fix violations
)

result = await guard.analyze(argument, context)
if result.has_violation:
    remediated = await guard.remediate(argument, result)
    # Use remediated version
```

## Transparency and Explainability

Ethics decisions are fully explainable:

```python
result = await guard.analyze(argument, context)

# Full explanation
explanation = result.explain()
print(f"Decision: {explanation.decision}")
print(f"Reasons: {explanation.reasons}")
print(f"Principle Scores: {explanation.principle_scores}")
print(f"Evidence: {explanation.evidence}")
```

## Best Practices

1. **Set Appropriate Sensitivity**: Match sensitivity to debate context
2. **Define Clear Principles**: Be explicit about ethical boundaries
3. **Use Multiple Frameworks**: Consider different ethical perspectives
4. **Enable Transparency**: Make ethics decisions explainable
5. **Review Edge Cases**: Some arguments may need human review

## Next Steps

- Learn about [Safety Monitoring](../safety/overview.md) for broader safety
- See how ethics integrates with [Jury Mechanism](jury.md)
- Explore [Ethics Guard](../safety/ethics-guard.md) in depth
