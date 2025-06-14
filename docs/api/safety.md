# Safety API Reference

This page documents the ARTEMIS safety monitoring API.

## SafetyManager

Central manager for safety monitors.

```python
from artemis.safety import SafetyManager
```

### Constructor

```python
SafetyManager(
    monitors: list[SafetyMonitor] | None = None,
    alert_handler: AlertHandler | None = None,
)
```

### Methods

#### add_monitor

```python
def add_monitor(self, monitor: SafetyMonitor) -> None
```

Adds a safety monitor.

#### analyze

```python
async def analyze(
    self,
    turn: Turn,
    context: DebateContext,
) -> list[SafetyResult]
```

Runs all monitors on a turn.

#### on_alert

```python
def on_alert(self, callback: Callable[[SafetyAlert], None]) -> None
```

Registers an alert callback.

---

## SandbagDetector

Detects intentional underperformance.

```python
from artemis.safety import SandbagDetector
```

### Constructor

```python
SandbagDetector(
    sensitivity: float = 0.7,
    baseline_rounds: int = 2,
    capability_metrics: list[str] | None = None,
    mode: str = "passive",  # passive or active
    halt_threshold: float = 0.9,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sensitivity` | float | 0.7 | Detection sensitivity (0-1) |
| `baseline_rounds` | int | 2 | Rounds for baseline |
| `capability_metrics` | list[str] | [...] | Metrics to track |
| `mode` | str | "passive" | Monitor mode |
| `halt_threshold` | float | 0.9 | Threshold for active halt |

### Methods

#### analyze

```python
async def analyze(
    self,
    turn: Turn,
    context: DebateContext,
) -> SafetyResult
```

Analyzes a turn for sandbagging.

---

## DeceptionMonitor

Detects deceptive claims.

```python
from artemis.safety import DeceptionMonitor
```

### Constructor

```python
DeceptionMonitor(
    sensitivity: float = 0.6,
    check_factual: bool = True,
    check_logical: bool = True,
    check_consistency: bool = True,
    check_sources: bool = True,
    use_external_verification: bool = False,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sensitivity` | float | 0.6 | Detection sensitivity |
| `check_factual` | bool | True | Check factual accuracy |
| `check_logical` | bool | True | Detect logical fallacies |
| `check_consistency` | bool | True | Track contradictions |
| `check_sources` | bool | True | Verify source accuracy |
| `use_external_verification` | bool | False | Use external fact-check |

### Methods

#### detect_fallacies

```python
def detect_fallacies(self, argument: Argument) -> list[Fallacy]
```

Detects logical fallacies.

#### verify_facts

```python
async def verify_facts(self, argument: Argument) -> FactCheckResult
```

Verifies factual claims.

---

## BehaviorTracker

Tracks behavioral changes over time.

```python
from artemis.safety import BehaviorTracker
```

### Constructor

```python
BehaviorTracker(
    window_size: int = 5,
    drift_threshold: float = 0.3,
    track_style: bool = True,
    track_position: bool = True,
    track_capability: bool = True,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | int | 5 | Turns to consider |
| `drift_threshold` | float | 0.3 | Drift alert threshold |
| `track_style` | bool | True | Track writing style |
| `track_position` | bool | True | Track position changes |
| `track_capability` | bool | True | Track capability metrics |

### Methods

#### get_history

```python
def get_history(self, agent_name: str) -> list[BehaviorSnapshot]
```

Gets behavior history for an agent.

#### calculate_drift

```python
def calculate_drift(
    self,
    current: dict,
    historical: list[dict],
) -> float
```

Calculates drift score.

---

## EthicsGuard

Monitors ethical boundaries.

```python
from artemis.safety import EthicsGuard
```

### Constructor

```python
EthicsGuard(
    sensitivity: float = 0.6,
    principles: list[str | EthicalPrinciple] | None = None,
    mode: str = "warn",  # warn, block, remediate
    violation_threshold: float = 0.7,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sensitivity` | float | 0.6 | Detection sensitivity |
| `principles` | list | [...] | Ethical principles |
| `mode` | str | "warn" | Response mode |
| `violation_threshold` | float | 0.7 | Threshold for action |

### Methods

#### remediate

```python
async def remediate(
    self,
    argument: Argument,
    result: SafetyResult,
) -> Argument
```

Attempts to fix ethical violations.

---

## CompositeMonitor

Combines multiple monitors.

```python
from artemis.safety import CompositeMonitor
```

### Constructor

```python
CompositeMonitor(
    monitors: list[SafetyMonitor],
    aggregation: str = "max",  # max, mean, weighted
    weights: dict[str, float] | None = None,
)
```

### Class Methods

#### default

```python
@classmethod
def default(cls) -> CompositeMonitor
```

Creates a monitor with default settings.

```python
monitor = CompositeMonitor.default()
# Includes all monitors with balanced settings
```

---

## SafetyResult

Result from a safety monitor.

```python
from artemis.safety.base import SafetyResult
```

### Class Definition

```python
class SafetyResult(BaseModel):
    is_safe: bool
    score: float  # 0-1, lower is safer
    monitor_type: str
    details: dict
    recommendations: list[str] = []
```

---

## SafetyAlert

Safety alert from monitoring.

```python
from artemis.safety.base import SafetyAlert
```

### Class Definition

```python
class SafetyAlert(BaseModel):
    type: str  # sandbagging, deception, drift, ethics
    severity: float  # 0-1
    agent: str
    round: int
    turn: int
    details: dict
    timestamp: datetime
```

---

## EthicalPrinciple

Custom ethical principle.

```python
from artemis.safety.ethics import EthicalPrinciple
```

### Class Definition

```python
class EthicalPrinciple(BaseModel):
    name: str
    description: str
    keywords: list[str] = []
    violation_patterns: list[str] = []
    weight: float = 1.0
```

**Example:**

```python
privacy_principle = EthicalPrinciple(
    name="privacy",
    description="Protect personal information",
    keywords=["personal", "private", "confidential"],
    violation_patterns=[
        r"reveal.*personal",
        r"expose.*private",
    ],
    weight=1.5,
)
```

---

## AlertHandler

Base class for alert handlers.

```python
from artemis.safety.base import AlertHandler
```

### Abstract Methods

```python
async def handle(self, alert: SafetyAlert) -> None
```

Handles a safety alert.

### Built-in Handlers

```python
from artemis.safety.handlers import (
    LoggingHandler,    # Logs alerts
    CallbackHandler,   # Calls a function
    WebhookHandler,    # Posts to webhook
)
```

---

## SafetySummary

Summary of safety results.

```python
from artemis.safety.base import SafetySummary
```

### Class Definition

```python
class SafetySummary(BaseModel):
    total_alerts: int
    critical_count: int
    warning_count: int
    info_count: int
    overall_score: float
    by_type: dict[str, int]
    by_agent: dict[str, int]
```

---

## Next Steps

- [Core API](core.md)
- [Models API](models.md)
- [Integrations API](integrations.md)
