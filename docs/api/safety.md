# Safety API Reference

This page documents the ARTEMIS safety monitoring API.

## MonitorMode

Enum for safety monitor modes.

```python
from artemis.safety import MonitorMode

MonitorMode.PASSIVE   # Observe and report only
MonitorMode.ACTIVE    # Can intervene and halt debate
MonitorMode.LEARNING  # Learn patterns without alerting
```

---

## SandbagDetector

Detects intentional underperformance.

```python
from artemis.safety import SandbagDetector, MonitorMode
```

### Constructor

```python
SandbagDetector(
    mode: MonitorMode = MonitorMode.PASSIVE,
    sensitivity: float = 0.5,
    baseline_turns: int = 3,
    drop_threshold: float = 0.3,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | MonitorMode | PASSIVE | Monitor mode |
| `sensitivity` | float | 0.5 | Detection sensitivity (0-1) |
| `baseline_turns` | int | 3 | Turns for baseline establishment |
| `drop_threshold` | float | 0.3 | Capability drop threshold |

### Methods

#### process

```python
async def process(
    self,
    turn: Turn,
    context: DebateContext,
) -> SafetyResult | None
```

Processes a turn for sandbagging detection. Used as `safety_monitors=[detector.process]`.

---

## DeceptionMonitor

Detects deceptive claims.

```python
from artemis.safety import DeceptionMonitor, MonitorMode
```

### Constructor

```python
DeceptionMonitor(
    mode: MonitorMode = MonitorMode.PASSIVE,
    sensitivity: float = 0.5,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | MonitorMode | PASSIVE | Monitor mode |
| `sensitivity` | float | 0.5 | Detection sensitivity (0-1) |

### Methods

#### process

```python
async def process(
    self,
    turn: Turn,
    context: DebateContext,
) -> SafetyResult | None
```

Processes a turn for deception detection.

---

## BehaviorTracker

Tracks behavioral changes over time.

```python
from artemis.safety import BehaviorTracker, MonitorMode
```

### Constructor

```python
BehaviorTracker(
    mode: MonitorMode = MonitorMode.PASSIVE,
    sensitivity: float = 0.5,
    window_size: int = 5,
    drift_threshold: float = 0.3,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | MonitorMode | PASSIVE | Monitor mode |
| `sensitivity` | float | 0.5 | Detection sensitivity (0-1) |
| `window_size` | int | 5 | Turns to consider for drift |
| `drift_threshold` | float | 0.3 | Drift alert threshold |

### Methods

#### process

```python
async def process(
    self,
    turn: Turn,
    context: DebateContext,
) -> SafetyResult | None
```

Processes a turn for behavior drift detection.

---

## EthicsGuard

Monitors ethical boundaries.

```python
from artemis.safety import EthicsGuard, EthicsConfig, MonitorMode
```

### Constructor

```python
EthicsGuard(
    mode: MonitorMode = MonitorMode.PASSIVE,
    config: EthicsConfig | None = None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | MonitorMode | PASSIVE | Monitor mode |
| `config` | EthicsConfig | None | Ethics configuration |

### EthicsConfig

```python
EthicsConfig(
    sensitivity: float = 0.5,
    principles: list[str] = ["fairness", "transparency", "non-harm"],
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sensitivity` | float | 0.5 | Detection sensitivity (0-1) |
| `principles` | list[str] | [...] | Ethical principles to enforce |

### Methods

#### process

```python
async def process(
    self,
    turn: Turn,
    context: DebateContext,
) -> SafetyResult | None
```

Processes a turn for ethics violations.

---

## Using Multiple Monitors

Combine monitors by passing a list to the Debate:

```python
from artemis.core.debate import Debate
from artemis.safety import (
    SandbagDetector,
    DeceptionMonitor,
    BehaviorTracker,
    EthicsGuard,
    MonitorMode,
    EthicsConfig,
)

# Create individual monitors
sandbag = SandbagDetector(mode=MonitorMode.PASSIVE, sensitivity=0.7)
deception = DeceptionMonitor(mode=MonitorMode.PASSIVE, sensitivity=0.6)
behavior = BehaviorTracker(mode=MonitorMode.PASSIVE, sensitivity=0.5)
ethics = EthicsGuard(mode=MonitorMode.PASSIVE, config=EthicsConfig(sensitivity=0.5))

# Pass their process methods to the debate
debate = Debate(
    topic="Your topic",
    agents=agents,
    safety_monitors=[
        sandbag.process,
        deception.process,
        behavior.process,
        ethics.process,
    ],
)
```

---

## SafetyResult

Result from a safety monitor.

```python
from artemis.core.types import SafetyResult
```

### Class Definition

```python
class SafetyResult(BaseModel):
    monitor: str             # Name of the monitor
    severity: float          # 0.0 to 1.0
    indicators: list[SafetyIndicator]
    should_alert: bool = False
    should_halt: bool = False
    analysis_notes: str | None = None
```

---

## SafetyAlert

Safety alert from monitoring.

```python
from artemis.core.types import SafetyAlert
```

### Class Definition

```python
class SafetyAlert(BaseModel):
    id: str
    monitor: str              # Name of the monitor
    agent: str                # Agent that triggered alert
    type: str                 # "sandbagging", "deception", "drift", "ethics"
    severity: float           # 0.0 to 1.0
    indicators: list[SafetyIndicator]
    turn_id: str | None
    timestamp: datetime
    resolved: bool = False
    resolution_notes: str | None = None
```

---

## SafetyIndicator

Individual safety indicator.

```python
from artemis.core.types import SafetyIndicator, SafetyIndicatorType
```

### Class Definition

```python
class SafetyIndicator(BaseModel):
    type: SafetyIndicatorType
    severity: float
    evidence: str | list[str]
    metadata: dict = {}

class SafetyIndicatorType(str, Enum):
    CAPABILITY_DROP = "capability_drop"
    STRATEGIC_TIMING = "strategic_timing"
    SELECTIVE_ENGAGEMENT = "selective_engagement"
    FACTUAL_INCONSISTENCY = "factual_inconsistency"
    LOGICAL_FALLACY = "logical_fallacy"
    EMOTIONAL_MANIPULATION = "emotional_manipulation"
    CITATION_FABRICATION = "citation_fabrication"
    BEHAVIORAL_DRIFT = "behavioral_drift"
    ETHICS_BOUNDARY = "ethics_boundary"
```

---

## Next Steps

- [Core API](core.md)
- [Models API](models.md)
- [Integrations API](integrations.md)
