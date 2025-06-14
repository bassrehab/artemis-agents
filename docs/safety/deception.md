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
| Factual | Are claims true? |
| Logical | Is reasoning valid? |
| Consistency | Do claims contradict each other? |
| Source | Are sources accurately represented? |
| Context | Is context preserved? |

## Usage

### Basic Setup

```python
from artemis.safety import DeceptionMonitor

monitor = DeceptionMonitor(
    sensitivity=0.6,
    check_factual=True,
    check_logical=True,
)

safety.add_monitor(monitor)
```

### Full Configuration

```python
monitor = DeceptionMonitor(
    # Core settings
    sensitivity=0.6,

    # What to check
    check_factual=True,      # Verify factual claims
    check_logical=True,      # Detect logical fallacies
    check_consistency=True,  # Track internal consistency
    check_sources=True,      # Verify source usage
    check_context=True,      # Detect context manipulation

    # Verification settings
    use_external_verification=False,  # Call external fact-checking
    verification_threshold=0.7,        # Confidence for flagging
)
```

## Detection Methods

### Factual Verification

Checks claims against known facts:

```python
# Internal consistency check
factual_result = monitor.verify_facts(argument)

# With external verification (optional)
factual_result = monitor.verify_facts(
    argument,
    external_sources=["wikipedia", "wolfram_alpha"],
)

print(f"Verified claims: {factual_result.verified}")
print(f"Unverified claims: {factual_result.unverified}")
print(f"False claims: {factual_result.false_claims}")
```

### Logical Fallacy Detection

Identifies common logical fallacies:

```python
fallacies = monitor.detect_fallacies(argument)

for fallacy in fallacies:
    print(f"Type: {fallacy.type}")
    print(f"Location: {fallacy.claim}")
    print(f"Explanation: {fallacy.explanation}")
```

Detected fallacy types:

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

### Consistency Tracking

Monitors for internal contradictions:

```python
# Track across multiple turns
consistency_result = monitor.check_consistency(
    current_turn=turn,
    previous_turns=context.get_agent_turns(agent_name),
)

if consistency_result.has_contradiction:
    print(f"Contradiction found:")
    print(f"  Previous: {consistency_result.previous_claim}")
    print(f"  Current: {consistency_result.current_claim}")
    print(f"  Analysis: {consistency_result.analysis}")
```

### Source Verification

Checks if sources are accurately represented:

```python
source_result = monitor.verify_sources(argument)

for source in source_result.sources:
    print(f"Source: {source.citation}")
    print(f"Accurately quoted: {source.is_accurate}")
    print(f"Context preserved: {source.context_preserved}")
    if not source.is_accurate:
        print(f"Issue: {source.issue}")
```

### Context Manipulation

Detects when context is distorted:

```python
context_result = monitor.check_context_manipulation(
    argument=argument,
    debate_context=context,
)

if context_result.manipulation_detected:
    print(f"Context manipulation: {context_result.type}")
    print(f"Details: {context_result.details}")
```

## Deception Score

The overall deception score combines all dimensions:

```python
result = await monitor.analyze(turn, context)

print(f"Overall Deception Score: {result.score}")
print(f"Breakdown:")
print(f"  Factual: {result.details['factual_score']}")
print(f"  Logical: {result.details['logical_score']}")
print(f"  Consistency: {result.details['consistency_score']}")
print(f"  Source: {result.details['source_score']}")
print(f"  Context: {result.details['context_score']}")
```

## Handling Results

### Alert Levels

```python
if result.score > 0.8:
    # High likelihood of intentional deception
    level = "critical"
elif result.score > 0.5:
    # Possible deception or sloppy argumentation
    level = "warning"
else:
    # Unlikely to be deceptive
    level = "info"
```

### Recommendations

```python
for recommendation in result.recommendations:
    print(f"Action: {recommendation.action}")
    print(f"Target: {recommendation.target}")
    print(f"Reason: {recommendation.reason}")
```

## Integration

### With Debate

```python
from artemis.core.debate import Debate
from artemis.safety import DeceptionMonitor, SafetyManager

safety = SafetyManager()
safety.add_monitor(DeceptionMonitor(sensitivity=0.6))

debate = Debate(
    topic="Your topic",
    agents=agents,
    safety_manager=safety,
)

result = await debate.run()

# Check for deception alerts
deception_alerts = [
    a for a in result.safety_alerts
    if a.type == "deception"
]

for alert in deception_alerts:
    print(f"Agent: {alert.agent}")
    print(f"Round: {alert.round}")
    print(f"Details: {alert.details}")
```

### With Evaluation

Deception affects argument scoring:

```python
from artemis.core.evaluation import AdaptiveEvaluator

evaluator = AdaptiveEvaluator(
    deception_penalty=True,  # Penalize deceptive arguments
    penalty_weight=0.2,       # 20% score reduction for deception
)
```

## Distinguishing Intent

Not all false claims are intentional deception:

| Type | Intent | Handling |
|------|--------|----------|
| Mistake | Unintentional | Flag for correction |
| Negligence | Careless | Minor penalty |
| Deception | Intentional | Major flag |

```python
# Configure intent detection
monitor = DeceptionMonitor(
    distinguish_intent=True,
    mistake_threshold=0.3,
    negligence_threshold=0.5,
    deception_threshold=0.7,
)
```

## External Verification

For higher accuracy, enable external fact-checking:

```python
monitor = DeceptionMonitor(
    use_external_verification=True,
    verification_sources=[
        "wikipedia",
        "wolfram_alpha",
        "custom_fact_db",
    ],
    api_keys={
        "wolfram_alpha": "your-key",
    },
)
```

## Best Practices

1. **Enable all checks**: Use comprehensive detection
2. **Track consistency**: Many deceptions are revealed by contradictions
3. **Verify sources**: Check if citations are accurate
4. **Consider intent**: Not all false claims are deceptive
5. **Combine with ethics**: Deception often accompanies ethical violations

## Tuning

### For High-Stakes Debates

```python
monitor = DeceptionMonitor(
    sensitivity=0.8,
    check_all=True,
    use_external_verification=True,
    strict_mode=True,
)
```

### For Exploratory Debates

```python
monitor = DeceptionMonitor(
    sensitivity=0.5,
    check_factual=True,
    check_logical=True,
    check_consistency=False,  # Allow position evolution
)
```

## Next Steps

- Learn about [Sandbagging Detection](sandbagging.md)
- Explore [Behavior Tracking](behavior.md)
- Configure [Ethics Guard](ethics-guard.md)
