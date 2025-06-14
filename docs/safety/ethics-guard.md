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
from artemis.safety import EthicsGuard

guard = EthicsGuard(
    sensitivity=0.6,
    principles=["fairness", "transparency", "non-harm"],
)

safety.add_monitor(guard)
```

### Full Configuration

```python
guard = EthicsGuard(
    # Core settings
    sensitivity=0.6,

    # Principles to enforce
    principles=[
        "fairness",
        "transparency",
        "non-harm",
        "respect",
        "accuracy",
    ],

    # Detection settings
    detect_discrimination=True,
    detect_manipulation=True,
    detect_privacy_violation=True,
    detect_harmful_content=True,

    # Response settings
    mode="warn",  # warn, block, or remediate
    violation_threshold=0.7,
)
```

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

```python
from artemis.safety.ethics import EthicalPrinciple

sustainability = EthicalPrinciple(
    name="sustainability",
    description="Arguments should consider environmental impact",
    keywords=["environment", "climate", "sustainable", "emissions"],
    violation_patterns=[
        r"ignores?\s+environmental",
        r"climate\s+change\s+(is\s+)?(a\s+)?hoax",
        r"pollution\s+doesn't\s+matter",
    ],
    weight=0.15,
)

guard = EthicsGuard(
    principles=["fairness", "non-harm", sustainability],
)
```

## Detection Categories

### Harmful Content

Content that promotes or glorifies harm:

```python
# Detected patterns:
# - Violence advocacy
# - Self-harm promotion
# - Dangerous activities
# - Harmful advice

result = guard.check_harmful_content(argument)

if result.detected:
    print(f"Harm type: {result.harm_type}")
    print(f"Severity: {result.severity}")
    print(f"Quote: {result.quote}")
```

### Discrimination

Unfair treatment based on protected characteristics:

```python
# Checks for:
# - Racial discrimination
# - Gender discrimination
# - Religious discrimination
# - Age discrimination
# - Disability discrimination

result = guard.check_discrimination(argument)

if result.detected:
    print(f"Discrimination type: {result.type}")
    print(f"Target group: {result.target}")
    print(f"Quote: {result.quote}")
```

### Manipulation

Psychological manipulation tactics:

```python
# Detects:
# - Fear mongering
# - Guilt tripping
# - Gaslighting language
# - Coercion
# - Emotional exploitation

result = guard.check_manipulation(argument)

if result.detected:
    print(f"Manipulation type: {result.type}")
    print(f"Tactic: {result.tactic}")
    print(f"Severity: {result.severity}")
```

### Privacy Violations

Exposure of private information:

```python
# Detects:
# - Personal identification
# - Location disclosure
# - Financial information
# - Health information
# - Private communications

result = guard.check_privacy(argument)

if result.detected:
    print(f"Privacy violation: {result.type}")
    print(f"Data type: {result.data_type}")
    print(f"Severity: {result.severity}")
```

## Analysis Results

The Ethics Guard returns detailed results:

```python
result = await guard.analyze(turn, context)

print(f"Is Safe: {result.is_safe}")
print(f"Ethics Score: {result.score}")

# Principle-by-principle breakdown
for principle, score in result.principle_scores.items():
    print(f"{principle}: {score:.2f}")

# Violations found
for violation in result.violations:
    print(f"Type: {violation.type}")
    print(f"Principle: {violation.principle}")
    print(f"Severity: {violation.severity}")
    print(f"Quote: {violation.quote}")
    print(f"Explanation: {violation.explanation}")
```

## Response Modes

### Warn Mode

```python
guard = EthicsGuard(mode="warn")

# Violations are flagged but argument continues
# Alert is raised for human review
```

### Block Mode

```python
guard = EthicsGuard(
    mode="block",
    violation_threshold=0.8,
)

# Arguments exceeding threshold are blocked
# Debate may be paused for review
```

### Remediate Mode

```python
guard = EthicsGuard(mode="remediate")

# Guard attempts to fix violations
# Modified argument continues
```

## Remediation

When in remediate mode, the guard can fix some violations:

```python
if result.has_violation:
    remediated = await guard.remediate(argument, result)

    print(f"Original: {argument.content}")
    print(f"Remediated: {remediated.content}")
    print(f"Changes: {remediated.changes}")
```

### What Can Be Remediated

| Violation | Remediation |
|-----------|-------------|
| Mild insults | Remove offensive language |
| Exaggeration | Tone down claims |
| Bias indicators | Add balancing language |
| Minor privacy | Redact identifying info |

### What Cannot Be Remediated

- Fundamental position changes
- Core argument restructuring
- Extensive harmful content
- Deliberate deception

## Integration

### With Debate

```python
from artemis.core.debate import Debate
from artemis.safety import EthicsGuard, SafetyManager

safety = SafetyManager()
safety.add_monitor(EthicsGuard(sensitivity=0.6))

debate = Debate(
    topic="Your topic",
    agents=agents,
    safety_manager=safety,
)

result = await debate.run()

# Check for ethics violations
ethics_alerts = [
    a for a in result.safety_alerts
    if a.type == "ethics"
]
```

### With Jury

Ethics can influence jury evaluation:

```python
from artemis.core.jury import JuryMember, PERSPECTIVES

# Juror with ethical focus
ethical_juror = JuryMember(
    name="ethicist",
    perspective=PERSPECTIVES["ethical"],
)

# Ethics violations affect juror's scoring
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

## Context-Aware Detection

The guard considers context:

```python
guard = EthicsGuard(
    context_aware=True,
    topic_sensitivity={
        "medical": 0.8,    # Higher sensitivity for medical topics
        "political": 0.7,   # Higher for political topics
        "technical": 0.5,   # Lower for technical topics
    },
)
```

## Audit Trail

All ethics decisions are logged:

```python
# Get ethics audit for debate
audit = guard.get_audit_trail(debate_id)

for entry in audit:
    print(f"Turn: {entry.turn}")
    print(f"Agent: {entry.agent}")
    print(f"Decision: {entry.decision}")
    print(f"Reasoning: {entry.reasoning}")
    print(f"Timestamp: {entry.timestamp}")
```

## Best Practices

1. **Set appropriate sensitivity**: Match to debate context
2. **Define clear principles**: Be explicit about boundaries
3. **Use warn mode initially**: Understand patterns before blocking
4. **Review edge cases**: Some content needs human judgment
5. **Document decisions**: Maintain audit trail for review

## Common Patterns

### Academic Debate

```python
guard = EthicsGuard(
    sensitivity=0.5,
    principles=["accuracy", "fairness", "transparency"],
    mode="warn",
)
```

### Sensitive Topics

```python
guard = EthicsGuard(
    sensitivity=0.8,
    principles=["non-harm", "respect", "fairness"],
    mode="block",
    violation_threshold=0.6,
)
```

### Policy Debate

```python
guard = EthicsGuard(
    sensitivity=0.6,
    principles=["accuracy", "transparency", "fairness"],
    detect_manipulation=True,
    mode="warn",
)
```

## Next Steps

- Learn about [Safety Overview](overview.md)
- Explore [Deception Monitoring](deception.md)
- Configure [Behavior Tracking](behavior.md)
