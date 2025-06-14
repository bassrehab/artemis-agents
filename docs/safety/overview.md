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
│                    │  Safety Manager │                         │
│                    └────────┬────────┘                         │
│                             │                                  │
│  ┌───────────────┐  ┌──────▼───────┐  ┌───────────────┐        │
│  │    Ethics     │  │  Composite   │  │    Alert      │        │
│  │     Guard     │──│   Monitor    │──│   Handler     │        │
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
from artemis.core.debate import Debate
from artemis.safety import (
    SafetyManager,
    SandbagDetector,
    DeceptionMonitor,
    BehaviorTracker,
    EthicsGuard,
)

# Create safety manager
safety = SafetyManager()

# Add monitors
safety.add_monitor(SandbagDetector(sensitivity=0.7))
safety.add_monitor(DeceptionMonitor(sensitivity=0.6))
safety.add_monitor(BehaviorTracker(window_size=5))
safety.add_monitor(EthicsGuard(sensitivity=0.5))

# Create debate with safety
debate = Debate(
    topic="Your topic",
    agents=agents,
    safety_manager=safety,
)

result = await debate.run()

# Check safety alerts
for alert in result.safety_alerts:
    print(f"Alert: {alert.type}")
    print(f"Severity: {alert.severity}")
    print(f"Agent: {alert.agent}")
    print(f"Details: {alert.details}")
```

### Composite Monitor

For convenience, use the composite monitor:

```python
from artemis.safety import CompositeMonitor

# All monitors with default settings
monitor = CompositeMonitor.default()

# Or with custom settings
monitor = CompositeMonitor(
    monitors=[
        SandbagDetector(sensitivity=0.8),
        DeceptionMonitor(sensitivity=0.7),
    ],
    aggregation="max",  # max, mean, or weighted
)
```

## Monitor Interface

All monitors implement the same interface:

```python
from artemis.safety.base import SafetyMonitor, SafetyResult

class CustomMonitor(SafetyMonitor):
    async def analyze(
        self,
        turn: Turn,
        context: DebateContext,
    ) -> SafetyResult:
        # Analyze the turn
        score = self.compute_safety_score(turn, context)

        return SafetyResult(
            is_safe=score < self.threshold,
            score=score,
            monitor_type=self.name,
            details=self.generate_details(turn, score),
        )
```

## Safety Results

Each monitor returns a `SafetyResult`:

```python
@dataclass
class SafetyResult:
    is_safe: bool          # Whether the turn is safe
    score: float           # Safety score (0-1, lower is safer)
    monitor_type: str      # Which monitor generated this
    details: dict          # Detailed analysis
    recommendations: list  # Suggested actions
```

## Alert Handling

### Alert Severity Levels

| Level | Score Range | Action |
|-------|-------------|--------|
| `info` | 0.0 - 0.3 | Log only |
| `warning` | 0.3 - 0.6 | Flag for review |
| `critical` | 0.6 - 0.9 | Intervene recommended |
| `emergency` | 0.9 - 1.0 | Halt immediately |

### Custom Alert Handlers

```python
from artemis.safety import AlertHandler

class CustomAlertHandler(AlertHandler):
    async def handle(self, alert: SafetyAlert) -> None:
        if alert.severity >= 0.9:
            await self.send_notification(alert)
            await self.pause_debate()
        elif alert.severity >= 0.6:
            await self.flag_for_review(alert)
        else:
            await self.log_alert(alert)

safety = SafetyManager(
    alert_handler=CustomAlertHandler(),
)
```

## Monitor Modes

### Passive Mode (Default)

Monitors observe and report but don't intervene:

```python
monitor = SandbagDetector(
    mode="passive",
    sensitivity=0.7,
)
# Alerts are generated but debate continues
```

### Active Mode

Monitors can intervene and halt the debate:

```python
monitor = SandbagDetector(
    mode="active",
    sensitivity=0.7,
    halt_threshold=0.9,
)
# Debate halts if threshold exceeded
```

## Configuration

### Global Safety Config

```python
from artemis.core.types import SafetyConfig

config = SafetyConfig(
    enable_monitoring=True,
    default_sensitivity=0.6,
    halt_on_critical=False,
    alert_threshold=0.5,
    log_all_analyses=True,
)

debate = Debate(
    topic="Your topic",
    agents=agents,
    safety_config=config,
)
```

### Per-Monitor Config

```python
sandbagging_config = {
    "sensitivity": 0.7,
    "baseline_rounds": 2,
    "capability_metrics": ["vocabulary", "reasoning_depth"],
}

deception_config = {
    "sensitivity": 0.6,
    "check_factual": True,
    "check_logical": True,
    "check_consistency": True,
}

behavior_config = {
    "window_size": 5,
    "drift_threshold": 0.3,
    "track_style": True,
    "track_position": True,
}
```

## Accessing Safety Data

### During Debate

```python
# Register callback for real-time alerts
async def on_safety_alert(alert: SafetyAlert):
    print(f"ALERT: {alert.type} - {alert.details}")

safety.on_alert(on_safety_alert)
```

### After Debate

```python
result = await debate.run()

# All alerts
for alert in result.safety_alerts:
    print(alert)

# Alerts by type
sandbagging_alerts = result.get_alerts_by_type("sandbagging")
deception_alerts = result.get_alerts_by_type("deception")

# Alerts by agent
agent_alerts = result.get_alerts_by_agent("agent_name")

# Safety summary
summary = result.safety_summary
print(f"Total alerts: {summary.total_alerts}")
print(f"Critical alerts: {summary.critical_count}")
print(f"Overall safety score: {summary.overall_score}")
```

## Integration with Metacognition

ARTEMIS can integrate with the AI Metacognition Toolkit:

```python
from artemis.safety.metacognition import MetacognitionIntegration

# Enable metacognition features
safety = SafetyManager(
    metacognition=MetacognitionIntegration(
        track_confidence=True,
        track_uncertainty=True,
        detect_overconfidence=True,
    )
)
```

## Best Practices

1. **Start with defaults**: Use `CompositeMonitor.default()` initially
2. **Tune sensitivity**: Adjust based on false positive rates
3. **Use passive mode first**: Understand behavior before enabling active mode
4. **Review alerts**: Regularly review safety alerts for patterns
5. **Combine monitors**: Multiple monitors catch more issues

## Next Steps

- Learn about [Sandbagging Detection](sandbagging.md)
- Understand [Deception Monitoring](deception.md)
- Explore [Behavior Tracking](behavior.md)
- Configure [Ethics Guard](ethics-guard.md)
